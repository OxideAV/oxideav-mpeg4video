//! Registry-facing encoder: [`Mpeg4VideoEncoder`] implements
//! [`oxideav_core::Encoder`] over the crate's VOP encoders, and
//! [`make_encoder`] is the direct factory endpoint (the dual-API
//! sibling of [`crate::decoder::make_decoder`]).
//!
//! The current tool set is the round-438 encoder arc so far:
//! rectangular progressive **I-VOPs** (method-1 or method-2
//! quantisation, cost-decided AC prediction). Every emitted VOP is
//! reconstructed through the crate's own decoder walk before the
//! packet is surfaced, so the encoder's reference state can never
//! drift from a conformant decoder's.
//!
//! Timing: the output elementary stream uses the VOL's §6.3.5 time
//! model with `vop_time_increment_resolution` taken from the
//! caller's `frame_rate` numerator (default 25/1); each frame
//! advances the clock by the frame-rate denominator in ticks.
//! Packets carry `pts`/`dts` in that tick time base.

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
};

use crate::ivop_encode::{encode_i_vop, write_configuration_headers, EncoderConfig, FrameView};
use crate::vol::{parse_video_object_layer, VolHeader};

/// Typed options struct for the registry / options-bag construction
/// path ([`oxideav_core::CodecParameters::options`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mpeg4EncoderOptions {
    /// `qp` — the VOP quantiser scale (1..=31). Default 4.
    pub qp: u32,
    /// `mpeg-quant` — select §7.4.4.1 method-1 (matrix) quantisation
    /// instead of the default §7.4.4.2 method 2.
    pub mpeg_quant: bool,
    /// `ac-pred` — enable the cost-decided §7.4.3.3 AC-prediction
    /// emission (default on).
    pub ac_pred: bool,
}

impl Default for Mpeg4EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 4,
            mpeg_quant: false,
            ac_pred: true,
        }
    }
}

impl oxideav_core::CodecOptionsStruct for Mpeg4EncoderOptions {
    const SCHEMA: &'static [oxideav_core::OptionField] = &[
        oxideav_core::OptionField {
            name: "qp",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(4),
            help: "VOP quantiser scale (1..=31); lower = higher quality/rate",
        },
        oxideav_core::OptionField {
            name: "mpeg-quant",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "use ISO/IEC 14496-2 §7.4.4.1 method-1 (matrix) quantisation \
                   instead of method 2",
        },
        oxideav_core::OptionField {
            name: "ac-pred",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(true),
            help: "enable cost-decided §7.4.3.3 AC-prediction emission on intra \
                   macroblocks",
        },
    ];

    fn apply(&mut self, key: &str, value: &oxideav_core::OptionValue) -> Result<()> {
        match key {
            "qp" => {
                let qp = value.as_u32()?;
                if !(1..=31).contains(&qp) {
                    return Err(Error::invalid("qp must be in 1..=31"));
                }
                self.qp = qp;
            }
            "mpeg-quant" => self.mpeg_quant = value.as_bool()?,
            "ac-pred" => self.ac_pred = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// The registry-facing MPEG-4 Part 2 Visual encoder.
#[derive(Debug)]
pub struct Mpeg4VideoEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    cfg: EncoderConfig,
    vol: VolHeader,
    options: Mpeg4EncoderOptions,
    /// Configuration-header run, prepended to the first packet.
    config_headers: Vec<u8>,
    /// Ticks the clock advances per input frame (the frame-rate
    /// denominator against a `time_increment_resolution` numerator).
    ticks_per_frame: u64,
    /// Running tick clock of the next frame to encode.
    next_ticks: u64,
    /// Second boundary (in whole seconds) of the previously coded VOP,
    /// for the §6.3.5 `modulo_time_base` derivation.
    prev_seconds: u64,
    frames_encoded: u64,
    ready: VecDeque<Packet>,
    flushed: bool,
}

impl Mpeg4VideoEncoder {
    /// Construct from codec parameters (see [`make_encoder`]).
    pub fn from_params(params: &CodecParameters) -> Result<Self> {
        let options: Mpeg4EncoderOptions = oxideav_core::parse_options(&params.options)?;
        let width = params
            .width
            .ok_or_else(|| Error::invalid("encoder needs width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("encoder needs height"))?;
        if width == 0 || height == 0 || width > 8191 || height > 8191 {
            return Err(Error::invalid("dimensions out of the 13-bit VOL range"));
        }
        if let Some(pf) = params.pixel_format {
            if pf != PixelFormat::Yuv420P {
                return Err(Error::unsupported(
                    "mpeg4video encoder accepts Yuv420P input only",
                ));
            }
        }
        let frame_rate = params.frame_rate.unwrap_or(Rational::new(25, 1));
        if frame_rate.num <= 0 || frame_rate.den <= 0 || frame_rate.num > 65_535 {
            return Err(Error::invalid(
                "frame rate must be positive with a numerator <= 65535 \
                 (it becomes vop_time_increment_resolution)",
            ));
        }
        let cfg = EncoderConfig {
            width: width as u16,
            height: height as u16,
            time_increment_resolution: frame_rate.num as u16,
            quant_type: options.mpeg_quant,
            ac_prediction: options.ac_pred,
        };
        let config_headers = write_configuration_headers(&cfg);
        let vol_pos = config_headers
            .windows(4)
            .position(|w| w == [0, 0, 1, 0x20])
            .expect("emitted headers contain the VOL start code");
        let vol = parse_video_object_layer(&config_headers[vol_pos..], cfg.profile_and_level())
            .map_err(|e| Error::invalid(format!("emitted VOL failed to re-parse: {e}")))?;

        let mut output_params = CodecParameters::video(params.codec_id.clone());
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        output_params.frame_rate = Some(frame_rate);
        output_params.extradata = config_headers.clone();
        output_params.tag = Some(oxideav_core::CodecTag::fourcc(b"FMP4"));

        Ok(Self {
            codec_id: params.codec_id.clone(),
            output_params,
            cfg,
            vol,
            options,
            config_headers,
            ticks_per_frame: frame_rate.den as u64,
            next_ticks: 0,
            prev_seconds: 0,
            frames_encoded: 0,
            ready: VecDeque::new(),
            flushed: false,
        })
    }

    /// The packet time base: one tick of the VOL clock.
    fn time_base(&self) -> TimeBase {
        TimeBase::new(1, i64::from(self.cfg.time_increment_resolution))
    }

    /// Copy one plane out of a [`oxideav_core::VideoFrame`] into a
    /// tightly-packed buffer of `w × h` (dropping stride padding).
    fn tight_plane(
        plane: &oxideav_core::VideoPlane,
        w: usize,
        h: usize,
        label: &str,
    ) -> Result<Vec<u8>> {
        let stride = plane.stride;
        if stride < w || plane.data.len() < stride * (h - 1) + w {
            return Err(Error::invalid(format!("{label} plane too small")));
        }
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            out.extend_from_slice(&plane.data[row * stride..row * stride + w]);
        }
        Ok(out)
    }
}

impl oxideav_core::Encoder for Mpeg4VideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::invalid("send_frame after flush"));
        }
        let video = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("mpeg4video encoder accepts video frames")),
        };
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let planes = video.image_planes();
        if planes.len() != 3 {
            return Err(Error::invalid("expected 3 Yuv420P planes"));
        }
        let y = Self::tight_plane(&planes[0], w, h, "luma")?;
        let cb = Self::tight_plane(&planes[1], cw, ch, "cb")?;
        let cr = Self::tight_plane(&planes[2], cw, ch, "cr")?;
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };

        // §6.3.5 time fields for this VOP.
        let res = u64::from(self.cfg.time_increment_resolution);
        let ticks = self.next_ticks;
        let seconds = ticks / res;
        let modulo = (seconds - self.prev_seconds) as u32;
        let increment = (ticks % res) as u16;

        let (unit, _recon) = encode_i_vop(
            &self.vol,
            &self.cfg,
            &view,
            modulo,
            increment,
            self.options.qp,
        );

        let mut data = Vec::new();
        if self.frames_encoded == 0 {
            data.extend_from_slice(&self.config_headers);
        }
        data.extend_from_slice(&unit);

        let pts = video.pts.unwrap_or(ticks as i64);
        let packet = Packet::new(0, self.time_base(), data)
            .with_pts(pts)
            .with_dts(pts)
            .with_duration(self.ticks_per_frame as i64)
            .with_keyframe(true);
        self.ready.push_back(packet);

        self.prev_seconds = seconds;
        self.next_ticks += self.ticks_per_frame;
        self.frames_encoded += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        match self.ready.pop_front() {
            Some(p) => Ok(p),
            None if self.flushed => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.flushed = true;
        Ok(())
    }
}

/// Direct factory endpoint: build a boxed registry-compatible encoder
/// from [`oxideav_core::CodecParameters`] (`width` / `height`
/// required; `frame_rate` optional, default 25/1; options per
/// [`Mpeg4EncoderOptions`]).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Encoder>> {
    Ok(Box::new(Mpeg4VideoEncoder::from_params(params)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{Encoder as _, VideoFrame, VideoPlane};

    fn gray_frame(w: usize, h: usize, luma: u8) -> Frame {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Frame::Video(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: vec![luma; w * h],
                },
                VideoPlane {
                    stride: cw,
                    data: vec![128; cw * ch],
                },
                VideoPlane {
                    stride: cw,
                    data: vec![128; cw * ch],
                },
            ],
        })
    }

    fn base_params() -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new("mpeg4video"));
        p.width = Some(48);
        p.height = Some(32);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    #[test]
    fn encodes_decodable_packets_with_extradata() {
        let mut enc = Mpeg4VideoEncoder::from_params(&base_params()).unwrap();
        assert!(!enc.output_params().extradata.is_empty());
        for k in 0..2 {
            enc.send_frame(&gray_frame(48, 32, 100 + k * 30)).unwrap();
        }
        enc.flush().unwrap();
        let mut stream = Vec::new();
        let mut count = 0;
        loop {
            match enc.receive_packet() {
                Ok(p) => {
                    assert!(p.flags.keyframe);
                    stream.extend_from_slice(&p.data);
                    count += 1;
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        assert_eq!(count, 2);
        let mut dec = crate::decoder::Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 2);
        // Flat gray reconstructs exactly at low qp.
        assert!(frames[0].luma_samples()[..48].iter().all(|&s| s == 100));
    }

    #[test]
    fn rejects_bad_options_and_dimensions() {
        let mut p = base_params();
        p.width = None;
        assert!(Mpeg4VideoEncoder::from_params(&p).is_err());
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("qp", "0");
        assert!(Mpeg4VideoEncoder::from_params(&p).is_err());
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("qp", "31");
        assert!(Mpeg4VideoEncoder::from_params(&p).is_ok());
    }

    #[test]
    fn mpeg_quant_option_flips_the_vol() {
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("mpeg-quant", "true");
        let enc = Mpeg4VideoEncoder::from_params(&p).unwrap();
        assert!(enc.vol.quant_type);
        assert_eq!(enc.cfg.profile_and_level(), 0xF3);
    }
}

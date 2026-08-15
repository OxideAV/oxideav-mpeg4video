//! Registry-facing encoder: [`Mpeg4VideoEncoder`] implements
//! [`oxideav_core::Encoder`] over the crate's VOP encoders, and
//! [`make_encoder`] is the direct factory endpoint (the dual-API
//! sibling of [`crate::decoder::make_decoder`]).
//!
//! The current tool set is the round-438 encoder arc: rectangular
//! progressive **I- and P-VOPs** (method-1 or method-2 quantisation,
//! cost-decided AC prediction, §7.6 motion estimation with half-pel
//! refinement, `not_coded` skips, `gop-size`-driven keyframe cadence).
//! Every emitted VOP is reconstructed through the crate's own decoder
//! walk before the packet is surfaced, so the encoder's reference
//! state can never drift from a conformant decoder's.
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

use crate::bvop_encode::encode_b_vop;
use crate::framestore::FrameStore;
use crate::ivop_encode::{encode_i_vop, write_configuration_headers, EncoderConfig, FrameView};
use crate::pvop_encode::{encode_p_vop, reconstruct_own_p_vop_with_motion};
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
    /// `four-mv` — enable cost-decided §6.3.7 four-motion-vector
    /// (`inter4v`) P-VOP macroblocks (default off).
    pub four_mv: bool,
    /// `qpel` — emit quarter-sample motion (§7.6.2.2; sets the VOL's
    /// `quarter_sample` flag and the ASP profile). Default off.
    pub qpel: bool,
    /// `bf` — number of B-VOPs between anchors (0..=8; selects the
    /// ASP profile when non-zero). Default 0.
    pub bf: u32,
    /// `gop-size` — I-VOP cadence: one keyframe every `gop_size`
    /// frames (`1` = intra-only). Default 12.
    pub gop_size: u32,
}

impl Default for Mpeg4EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 4,
            mpeg_quant: false,
            ac_pred: true,
            four_mv: false,
            qpel: false,
            bf: 0,
            gop_size: 12,
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
        oxideav_core::OptionField {
            name: "four-mv",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "enable cost-decided four-motion-vector (inter4v) P-VOP \
                   macroblocks",
        },
        oxideav_core::OptionField {
            name: "qpel",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "emit quarter-sample motion vectors (ISO/IEC 14496-2 \
                   §7.6.2.2; selects the ASP profile)",
        },
        oxideav_core::OptionField {
            name: "bf",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "number of B-VOPs between anchors (0..=8; non-zero \
                   selects the ASP profile and adds one anchor of \
                   encoder latency)",
        },
        oxideav_core::OptionField {
            name: "gop-size",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(12),
            help: "keyframe cadence: one I-VOP every gop-size frames (1 = intra-only)",
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
            "four-mv" => self.four_mv = value.as_bool()?,
            "qpel" => self.qpel = value.as_bool()?,
            "bf" => {
                let bf = value.as_u32()?;
                if bf > 8 {
                    return Err(Error::invalid("bf must be in 0..=8"));
                }
                self.bf = bf;
            }
            "gop-size" => {
                let g = value.as_u32()?;
                if g == 0 {
                    return Err(Error::invalid("gop-size must be >= 1"));
                }
                self.gop_size = g;
            }
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// One buffered input picture awaiting its bracketing future anchor
/// (the B-VOP reorder queue).
#[derive(Debug)]
struct PendingFrame {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    /// §6.3.5 composed VOP time in ticks.
    ticks: u64,
    /// Caller-supplied presentation stamp.
    pts: Option<i64>,
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
    /// §6.3.5 seconds accumulated at the most recently coded anchor's
    /// sync point (mirrors the decoder's `sync_sec`).
    sync_sec: u64,
    /// The sync value one anchor earlier — the base a B-VOP's
    /// `modulo_time_base` counts from.
    b_base_sec: u64,
    /// Absolute tick time of the past (forward) anchor.
    prev_anchor_ticks: Option<u64>,
    /// Absolute tick time of the most recent (backward) anchor.
    last_anchor_ticks: Option<u64>,
    /// Display-order input frame counter (keyframe cadence).
    frames_seen: u64,
    /// Coded (bitstream-order) packet counter.
    frames_coded: u64,
    /// The B-VOP reorder queue (display order).
    pending_bs: Vec<PendingFrame>,
    /// The future anchor's per-macroblock decoded motion — the
    /// §7.6.9.5.1 / §6.2.6 co-located source for the B-VOPs it
    /// brackets (`None` after an intra anchor).
    anchor_motion: Option<Vec<crate::pvop_mv::PvopMbMotion>>,
    /// §7.6.1 anchor chain — the closed-loop references produced by
    /// decoding our own emitted units.
    store: FrameStore,
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
            four_mv: options.four_mv,
            quarter_sample: options.qpel,
            b_vops: options.bf > 0,
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
            sync_sec: 0,
            b_base_sec: 0,
            prev_anchor_ticks: None,
            last_anchor_ticks: None,
            frames_seen: 0,
            frames_coded: 0,
            pending_bs: Vec::new(),
            anchor_motion: None,
            store: FrameStore::new(),
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
        let pending = PendingFrame {
            y: Self::tight_plane(&planes[0], w, h, "luma")?,
            cb: Self::tight_plane(&planes[1], cw, ch, "cb")?,
            cr: Self::tight_plane(&planes[2], cw, ch, "cr")?,
            ticks: self.next_ticks,
            pts: video.pts,
        };
        self.next_ticks += self.ticks_per_frame;
        let display_index = self.frames_seen;
        self.frames_seen += 1;

        // Anchor selection: the first frame, the keyframe cadence, and
        // every (bf + 1)-th frame terminate a B run.
        let keyframe_due = display_index % u64::from(self.options.gop_size) == 0;
        let no_anchor = self.store.backward().is_none();
        let is_anchor =
            no_anchor || keyframe_due || self.pending_bs.len() as u32 >= self.options.bf;
        if !is_anchor {
            self.pending_bs.push(pending);
            return Ok(());
        }
        self.encode_anchor(&pending, no_anchor || keyframe_due);
        self.drain_pending_bs();
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
        if self.flushed {
            return Ok(());
        }
        // A tail of buffered pictures with no future anchor yet: code
        // the last one as the terminating anchor, the rest as the
        // B-VOPs it brackets.
        if let Some(last) = self.pending_bs.pop() {
            let keyframe_due = self.store.backward().is_none() || {
                // The anchor's display index is frames_seen - 1.
                (self.frames_seen - 1) % u64::from(self.options.gop_size) == 0
            };
            self.encode_anchor(&last, keyframe_due);
            self.drain_pending_bs();
        }
        self.flushed = true;
        Ok(())
    }
}

impl Mpeg4VideoEncoder {
    /// Bitstream-order packet emission: wrap `unit` (prepending the
    /// configuration run on the very first packet) with the Annex D
    /// item-7 timing — an anchor's decode time is the previous
    /// anchor's composition time once B-VOPs are in play (`bf > 0`);
    /// with `low_delay`-style `bf == 0` streams `dts == pts`.
    fn push_packet(
        &mut self,
        unit: Vec<u8>,
        ticks: u64,
        pts: Option<i64>,
        keyframe: bool,
        dts: i64,
    ) {
        let mut data = Vec::new();
        if self.frames_coded == 0 {
            data.extend_from_slice(&self.config_headers);
        }
        data.extend_from_slice(&unit);
        let pts = pts.unwrap_or(ticks as i64);
        let packet = Packet::new(0, self.time_base(), data)
            .with_pts(pts)
            .with_dts(dts)
            .with_duration(self.ticks_per_frame as i64)
            .with_keyframe(keyframe);
        self.ready.push_back(packet);
        self.frames_coded += 1;
    }

    /// Encode one anchor VOP (I when `force_i`, else P), advance the
    /// §7.6.1 chain + §6.3.5 anchor time bases, and emit its packet.
    fn encode_anchor(&mut self, frame: &PendingFrame, force_i: bool) {
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let view = FrameView {
            y: &frame.y,
            cb: &frame.cb,
            cr: &frame.cr,
            width: w,
            height: h,
        };
        let res = u64::from(self.cfg.time_increment_resolution);
        let seconds = frame.ticks / res;
        let modulo = (seconds - self.sync_sec) as u32;
        let increment = (frame.ticks % res) as u16;

        let unit = if force_i {
            let (unit, recon) = encode_i_vop(
                &self.vol,
                &self.cfg,
                &view,
                modulo,
                increment,
                self.options.qp,
            );
            self.store.push_anchor(recon);
            self.anchor_motion = None;
            unit
        } else {
            let reference = self
                .store
                .backward()
                .expect("anchor present on the P path")
                .clone();
            let (unit, _stats) = encode_p_vop(
                &self.vol,
                &self.cfg,
                &view,
                &reference,
                modulo,
                increment,
                self.options.qp,
            );
            let (_recon, motion) =
                reconstruct_own_p_vop_with_motion(&self.vol, &unit, &mut self.store);
            self.anchor_motion = Some(motion);
            unit
        };

        // §6.3.5 anchor time bookkeeping (mirrors the decoder).
        self.b_base_sec = self.sync_sec;
        self.sync_sec = seconds;
        self.prev_anchor_ticks = self.last_anchor_ticks;
        self.last_anchor_ticks = Some(frame.ticks);

        // Annex D item 7: with B-VOPs in play the anchor's decode time
        // is the *previous* anchor's composition time (t0 = τ0 − Δ for
        // the first anchor); without B-VOPs, dts == pts.
        let dts = if self.options.bf == 0 {
            frame.pts.unwrap_or(frame.ticks as i64)
        } else {
            match self.prev_anchor_ticks {
                Some(prev) => prev as i64,
                None => frame.ticks as i64 - self.ticks_per_frame as i64,
            }
        };
        self.push_packet(unit, frame.ticks, frame.pts, force_i, dts);
    }

    /// Encode every buffered B-VOP (display order) between the two
    /// anchors now in the chain and emit their packets.
    fn drain_pending_bs(&mut self) {
        if self.pending_bs.is_empty() {
            return;
        }
        let (prev, last) = (
            self.prev_anchor_ticks.expect("B-VOPs need a past anchor"),
            self.last_anchor_ticks.expect("B-VOPs need a future anchor"),
        );
        let trd = (last - prev) as i32;
        let res = u64::from(self.cfg.time_increment_resolution);
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let pending = std::mem::take(&mut self.pending_bs);
        for b in &pending {
            let view = FrameView {
                y: &b.y,
                cb: &b.cb,
                cr: &b.cr,
                width: w,
                height: h,
            };
            let trb = (b.ticks - prev) as i32;
            let seconds = b.ticks / res;
            let modulo = (seconds - self.b_base_sec) as u32;
            let increment = (b.ticks % res) as u16;
            let (unit, _recon, _stats) = encode_b_vop(
                &self.vol,
                &self.cfg,
                &view,
                &self.store,
                self.anchor_motion.as_deref(),
                trb,
                trd,
                modulo,
                increment,
                self.options.qp,
            );
            // Annex D item 7: a B-VOP's decode time is its own
            // composition time.
            self.push_packet(unit, b.ticks, b.pts, false, b.ticks as i64);
        }
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
    fn encodes_decodable_ip_packets_with_extradata() {
        let mut enc = Mpeg4VideoEncoder::from_params(&base_params()).unwrap();
        assert!(!enc.output_params().extradata.is_empty());
        for k in 0..3 {
            enc.send_frame(&gray_frame(48, 32, 100 + k * 30)).unwrap();
        }
        enc.flush().unwrap();
        let mut stream = Vec::new();
        let mut keyframes = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => {
                    keyframes.push(p.flags.keyframe);
                    stream.extend_from_slice(&p.data);
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        // gop-size 12 → frame 0 is the only keyframe of the three.
        assert_eq!(keyframes, vec![true, false, false]);
        let mut dec = crate::decoder::Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 3);
        // Flat gray reconstructs exactly at low qp, including through
        // the two P-VOPs' luminance step changes.
        assert!(frames[0].luma_samples()[..48].iter().all(|&s| s == 100));
        assert!(frames[2].luma_samples()[..48].iter().all(|&s| s == 160));
    }

    #[test]
    fn gop_size_one_is_intra_only() {
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("gop-size", "1");
        let mut enc = Mpeg4VideoEncoder::from_params(&p).unwrap();
        for _ in 0..3 {
            enc.send_frame(&gray_frame(48, 32, 90)).unwrap();
        }
        enc.flush().unwrap();
        let mut count = 0;
        loop {
            match enc.receive_packet() {
                Ok(pk) => {
                    assert!(pk.flags.keyframe, "every packet must be an I-VOP");
                    count += 1;
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        assert_eq!(count, 3);
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

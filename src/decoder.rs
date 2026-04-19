//! MPEG-4 Part 2 video decoder.
//!
//! Scope:
//! * Parses Visual Object Sequence, Visual Object, Video Object Layer and
//!   Video Object Plane headers from a stream of annexed start codes.
//! * Populates `CodecParameters` from the VOL.
//! * **Decodes I-VOPs** — full intra path (DC+AC VLCs, AC/DC prediction,
//!   H.263 + MPEG-4 dequantisation, IDCT).
//! * **Decodes P-VOPs** — half-pel motion compensation, 1MV / 4MV modes,
//!   inter texture decode, MV-median prediction, and skipped MBs.
//! * Holds one reference picture (`prev_ref`) — refreshed by each I-VOP and
//!   each newly-reconstructed P-VOP.
//!
//! Out of scope (returns `Error::Unsupported`):
//! * B-VOPs, S-VOPs (sprites), GMC.
//! * Quarter-pel motion (`quarter_sample` rejected at VOL parse time).
//! * Interlaced field coding, scalability, data partitioning.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame,
};

use crate::headers::vol::{parse_vol, VideoObjectLayer};
use crate::headers::vop::{parse_vop, VideoObjectPlane, VopCodingType};
use crate::headers::vos::{parse_visual_object, parse_vos, VisualObject, VisualObjectSequence};
use crate::inter::{decode_p_mb, MvGrid};
use crate::mb::{decode_intra_mb, IVopPicture, PredGrid};
use crate::resync::{try_consume_resync_marker_after, ResyncResult};
use crate::start_codes::{
    self, is_video_object, is_video_object_layer, GOV_START_CODE, USER_DATA_START_CODE,
    VIDEO_SESSION_ERROR_CODE, VISUAL_OBJECT_START_CODE, VOP_START_CODE, VOS_END_CODE,
    VOS_START_CODE,
};
use oxideav_core::bits::BitReader;

/// Factory for the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mpeg4VideoDecoder::new(params.codec_id.clone())))
}

pub struct Mpeg4VideoDecoder {
    codec_id: CodecId,
    buffer: Vec<u8>,
    vos: Option<VisualObjectSequence>,
    vo: Option<VisualObject>,
    vol: Option<VideoObjectLayer>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
    /// Last decoded reference picture — used as `prev_ref` for the next
    /// P-VOP. Refreshed by each I-VOP and each P-VOP.
    prev_ref: Option<IVopPicture>,
    /// First-packet sniff: catches the mislabel case where a container
    /// tagged the stream as MPEG-4 Part 2 (XVID / DX50 / …) but the
    /// bytes are actually an MS-MPEG4 bitstream. `None` until the
    /// first `send_packet`, then cached so we only pay the scan once
    /// per stream.
    format_verified: bool,
}

impl Mpeg4VideoDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            vos: None,
            vo: None,
            vol: None,
            ready_frames: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 90_000),
            eof: false,
            prev_ref: None,
            format_verified: false,
        }
    }

    pub fn vol(&self) -> Option<&VideoObjectLayer> {
        self.vol.as_ref()
    }

    /// Walk start codes in the buffer, updating header state and dispatching
    /// VOPs. I-VOPs and P-VOPs are decoded; B/S VOPs return `Unsupported`.
    fn process(&mut self) -> Result<()> {
        let data = std::mem::take(&mut self.buffer);
        let markers: Vec<(usize, u8)> = start_codes::iter_start_codes(&data).collect();
        for (idx, (pos, code)) in markers.iter().enumerate() {
            let payload_end = markers.get(idx + 1).map(|(p, _)| *p).unwrap_or(data.len());
            let payload_start = *pos + 4;
            if payload_start > data.len() {
                break;
            }
            let payload = &data[payload_start..payload_end];

            match *code {
                VOS_START_CODE => {
                    let mut br = BitReader::new(payload);
                    self.vos = Some(parse_vos(&mut br)?);
                }
                VISUAL_OBJECT_START_CODE => {
                    let mut br = BitReader::new(payload);
                    self.vo = Some(parse_visual_object(&mut br)?);
                }
                c if is_video_object(c) => {
                    // Video object start code — no payload of interest.
                }
                c if is_video_object_layer(c) => {
                    let mut br = BitReader::new(payload);
                    self.vol = Some(parse_vol(&mut br)?);
                }
                GOV_START_CODE | USER_DATA_START_CODE | VIDEO_SESSION_ERROR_CODE | VOS_END_CODE => {
                    // Not yet used by this decoder — skip.
                }
                VOP_START_CODE => {
                    let Some(vol) = self.vol.clone() else {
                        return Err(Error::invalid("mpeg4: VOP before VOL"));
                    };
                    if vol.data_partitioned {
                        return Err(Error::unsupported("mpeg4 data-partitioned VOL: follow-up"));
                    }
                    if vol.interlaced {
                        return Err(Error::unsupported(
                            "mpeg4 interlaced field coding: follow-up",
                        ));
                    }
                    let mut br = BitReader::new(payload);
                    let vop = parse_vop(&mut br, &vol)?;
                    self.handle_vop(&vol, &vop, &mut br)?;
                }
                _ => {
                    // Unknown marker — skip.
                }
            }
        }
        Ok(())
    }

    fn handle_vop(
        &mut self,
        vol: &VideoObjectLayer,
        vop: &VideoObjectPlane,
        br: &mut BitReader<'_>,
    ) -> Result<()> {
        if !vop.vop_coded {
            // "Not coded" VOP (§6.2.5): the decoder must re-emit the previous
            // reference frame at the new pts. The reference itself is not
            // modified — the next coded VOP still predicts from the last
            // coded picture.
            if let Some(reference) = self.prev_ref.as_ref() {
                let frame = pic_to_video_frame(vol, reference, self.pending_pts, self.pending_tb);
                self.ready_frames.push_back(frame);
            }
            return Ok(());
        }
        match vop.vop_coding_type {
            VopCodingType::I => {
                let pic = decode_ivop_pic(vol, vop, br)?;
                let frame = pic_to_video_frame(vol, &pic, self.pending_pts, self.pending_tb);
                self.prev_ref = Some(pic);
                self.ready_frames.push_back(frame);
                Ok(())
            }
            VopCodingType::P => {
                let Some(reference) = self.prev_ref.as_ref() else {
                    return Err(Error::invalid("mpeg4 P-VOP: no reference frame yet"));
                };
                let pic = decode_pvop_pic(vol, vop, br, reference)?;
                let frame = pic_to_video_frame(vol, &pic, self.pending_pts, self.pending_tb);
                self.prev_ref = Some(pic);
                self.ready_frames.push_back(frame);
                Ok(())
            }
            VopCodingType::B => Err(Error::unsupported(
                "mpeg4 B frames: follow-up (bidirectional MC)",
            )),
            VopCodingType::S => Err(Error::unsupported("mpeg4 S-VOP (sprite): out of scope")),
        }
    }
}

/// Decode an I-VOP and return the reconstructed `IVopPicture`.
pub fn decode_ivop_pic(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
) -> Result<IVopPicture> {
    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut grid = PredGrid::new(mb_w, mb_h);

    let mb_total = (mb_w * mb_h) as u32;
    let mut quant = vop.vop_quant;
    let mut mb_idx: u32 = 0;
    while (mb_idx as usize) < mb_w * mb_h {
        let mb_x = (mb_idx as usize) % mb_w;
        let mb_y = (mb_idx as usize) / mb_w;
        quant =
            decode_intra_mb(br, mb_x, mb_y, quant, vol, vop, &mut pic, &mut grid).map_err(|e| {
                oxideav_core::Error::invalid(format!("mpeg4 I-VOP MB ({mb_x},{mb_y}): {e}"))
            })?;
        mb_idx += 1;
        if (mb_idx as usize) >= mb_w * mb_h {
            break;
        }
        match try_consume_resync_marker_after(br, vol, vop, mb_total, mb_idx)? {
            ResyncResult::None => {}
            ResyncResult::Resync { mb_num, new_quant } => {
                if mb_num < mb_idx || mb_num >= mb_total {
                    return Err(Error::invalid(format!(
                        "mpeg4 I-VOP: resync mb_num={mb_num} not at or after current={mb_idx}"
                    )));
                }
                grid = PredGrid::new(mb_w, mb_h);
                if new_quant != 0 {
                    quant = new_quant;
                }
                mb_idx = mb_num;
            }
        }
    }
    Ok(pic)
}

/// Decode a P-VOP relative to `reference` and return the reconstructed
/// `IVopPicture`.
pub fn decode_pvop_pic(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
    reference: &IVopPicture,
) -> Result<IVopPicture> {
    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut pred_grid = PredGrid::new(mb_w, mb_h);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);

    let mb_total = (mb_w * mb_h) as u32;
    let mut quant = vop.vop_quant;
    let mut mb_idx: u32 = 0;
    let mut slice_first_mb = (0usize, 0usize);
    while (mb_idx as usize) < mb_w * mb_h {
        let mb_x = (mb_idx as usize) % mb_w;
        let mb_y = (mb_idx as usize) / mb_w;
        quant = decode_p_mb(
            br,
            mb_x,
            mb_y,
            quant,
            vol,
            vop,
            &mut pic,
            &mut pred_grid,
            &mut mv_grid,
            reference,
            slice_first_mb,
        )
        .map_err(|e| {
            oxideav_core::Error::invalid(format!("mpeg4 P-VOP MB ({mb_x},{mb_y}): {e}"))
        })?;
        mb_idx += 1;
        if (mb_idx as usize) >= mb_w * mb_h {
            break;
        }
        match try_consume_resync_marker_after(br, vol, vop, mb_total, mb_idx)? {
            ResyncResult::None => {}
            ResyncResult::Resync { mb_num, new_quant } => {
                if mb_num < mb_idx || mb_num >= mb_total {
                    return Err(Error::invalid(format!(
                        "mpeg4 P-VOP: resync mb_num={mb_num} not at or after current={mb_idx}"
                    )));
                }
                // Reset prediction state across packet boundaries.
                pred_grid = PredGrid::new(mb_w, mb_h);
                mv_grid = MvGrid::new(mb_w, mb_h);
                if new_quant != 0 {
                    quant = new_quant;
                }
                mb_idx = mb_num;
                slice_first_mb = ((mb_idx as usize) % mb_w, (mb_idx as usize) / mb_w);
            }
        }
    }
    Ok(pic)
}

/// Build a stride-packed YUV420P `VideoFrame` from an `IVopPicture`.
pub fn pic_to_video_frame(
    vol: &VideoObjectLayer,
    pic: &IVopPicture,
    pts: Option<i64>,
    tb: TimeBase,
) -> VideoFrame {
    let w = vol.width as usize;
    let h = vol.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        y[row * w..row * w + w].copy_from_slice(&pic.y[row * pic.y_stride..row * pic.y_stride + w]);
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        cb[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cb[row * pic.c_stride..row * pic.c_stride + cw]);
        cr[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cr[row * pic.c_stride..row * pic.c_stride + cw]);
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w as u32,
        height: h as u32,
        pts,
        time_base: tb,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

/// Decode a single I-VOP body and return a `VideoFrame`. Kept for backwards
/// compatibility with existing tests.
pub fn decode_ivop(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
    pts: Option<i64>,
    tb: TimeBase,
) -> Result<VideoFrame> {
    let pic = decode_ivop_pic(vol, vop, br)?;
    Ok(pic_to_video_frame(vol, &pic, pts, tb))
}

impl Decoder for Mpeg4VideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Sniff the first packet to catch the mislabel case — a
        // container tagged the stream as MPEG-4 Part 2 (XVID / DX50 /
        // DIVX / …) but the bytes are actually MS-MPEG4 (DIV3-style).
        // An ISO stream MUST start with a `0x000001` start code prefix
        // somewhere in its first couple of KB (VOS / VOL / VOP / GOV);
        // an MS stream has no such prefix anywhere. If we don't find
        // one, refuse the packet with a clear dispatch hint.
        if !self.format_verified {
            if !crate::probe_is_mpeg4_part2(&packet.data) {
                return Err(Error::unsupported(
                    "mpeg4video: packet has no MPEG-4 Part 2 start code — \
                     bitstream is likely MS-MPEG4 (DIV3 / MP43 / …). \
                     Dispatch to oxideav-msmpeg4 instead.",
                ));
            }
            self.format_verified = true;
        }
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.process()
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Drop the NAL-accumulator buffer, the ready-frame queue, and the
        // stored reference picture (used as `prev_ref` for the next P-VOP).
        // VOS / VO / VOL are stream-level config extracted once from the
        // bitstream — kept so we can decode subsequent VOPs without
        // waiting for the headers to reappear.
        self.buffer.clear();
        self.ready_frames.clear();
        self.pending_pts = None;
        self.eof = false;
        self.prev_ref = None;
        Ok(())
    }
}

/// Build a `CodecParameters` from a VOL.
pub fn codec_parameters_from_vol(vol: &VideoObjectLayer) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(vol.width);
    params.height = Some(vol.height);
    let (num, den) = vol.frame_rate();
    if num > 0 && den > 0 {
        params.frame_rate = Some(Rational::new(num, den));
    }
    params
}

#[cfg(test)]
mod decoder_tests {
    use super::*;
    use oxideav_core::{CodecId, Packet, TimeBase};

    /// The send_packet sniff rejects a stream whose first packet has no
    /// 0x000001 start-code prefix — i.e. an MS-MPEG4 bitstream that was
    /// mislabelled as MPEG-4 Part 2 by the container (typical AVI
    /// FourCC mismatch: file stamped `XVID` but payload is DIV3-style).
    #[test]
    fn send_packet_rejects_msmpeg4_bitstream() {
        let mut dec = Mpeg4VideoDecoder::new(CodecId::new(crate::CODEC_ID_STR));
        // MS-MPEG4v3 picture header opens with `1` bit + quant — no
        // 0x000001 start-code sequence anywhere.
        let pkt = Packet::new(
            0,
            TimeBase::new(1, 90_000),
            vec![0x85, 0x3F, 0xD4, 0x80, 0x00, 0xA2, 0x10, 0xFF],
        );
        let err = dec.send_packet(&pkt).expect_err("expected Unsupported");
        let msg = err.to_string();
        assert!(
            msg.contains("no MPEG-4 Part 2 start code"),
            "error should mention missing start code, got: {msg}",
        );
        assert!(
            msg.to_lowercase().contains("msmpeg4"),
            "error should dispatch to msmpeg4, got: {msg}",
        );
    }

    /// A genuine MPEG-4 Part 2 elementary stream (starts with VOS start
    /// code 0x000001B0) passes the sniff and reaches the normal parse
    /// path — which will likely error further in (we're not feeding a
    /// valid VOL here) but crucially NOT with the mislabel message.
    #[test]
    fn send_packet_accepts_iso_bitstream() {
        let mut dec = Mpeg4VideoDecoder::new(CodecId::new(crate::CODEC_ID_STR));
        // Visual Object Sequence start code with trailing junk.
        let pkt = Packet::new(
            0,
            TimeBase::new(1, 90_000),
            vec![0x00, 0x00, 0x01, 0xB0, 0x01, 0x00, 0x00, 0x01, 0xB5],
        );
        // The sniff must pass — further parsing may or may not
        // succeed, but the mislabel-error must not fire.
        let result = dec.send_packet(&pkt);
        if let Err(e) = &result {
            assert!(
                !e.to_string().contains("no MPEG-4 Part 2 start code"),
                "ISO bitstream wrongly rejected as mislabelled: {e}",
            );
        }
    }

    /// After the first successful sniff, format_verified stays `true`
    /// and subsequent packets (even fragmented ones that don't
    /// contain start codes in isolation) are accepted.
    #[test]
    fn sniff_only_runs_on_first_packet() {
        let mut dec = Mpeg4VideoDecoder::new(CodecId::new(crate::CODEC_ID_STR));
        let first = Packet::new(
            0,
            TimeBase::new(1, 90_000),
            vec![0x00, 0x00, 0x01, 0xB0, 0x01],
        );
        let _ = dec.send_packet(&first);
        // Second packet is coded-data continuation — no start code. Must
        // not be flagged as mislabelled.
        let second = Packet::new(0, TimeBase::new(1, 90_000), vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let result = dec.send_packet(&second);
        if let Err(e) = &result {
            assert!(
                !e.to_string().contains("no MPEG-4 Part 2 start code"),
                "sniff wrongly re-ran on second packet: {e}",
            );
        }
    }
}

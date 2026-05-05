//! MPEG-4 Part 2 video encoder — I-VOP + P-VOP.
//!
//! Scope:
//! * Visual Object Sequence (VOS), Visual Object (VO), Video Object Layer
//!   (VOL) and Video Object Plane (VOP) headers — §6.2.
//! * I-VOP body: per-MB MCBPC + ac_pred + CBPY (no dquant), then six 8×8
//!   intra blocks (Y0..Y3, Cb, Cr) with intra DC VLC + signed residual and
//!   intra AC tcoef VLC walk (Table B-16).
//! * P-VOP body: half-pel motion estimation (integer diamond + half-pel
//!   refinement), 1MV / 4MV / Intra-in-P mode decision (§7.5.7 / §7.6.7
//!   / §6.3.7), median-predicted MVD with Table B-12, inter texture
//!   coding (H.263 inter quant + Table B-17 tcoef walk), `not_coded`
//!   skip MBs. Embedded intra MBs use `encode_intra_mb_in_p` which
//!   shares the I-VOP intra encode path but emits Table B-13 Intra
//!   MCBPC rows. See `pvop.rs`.
//! * H.263 quantisation (`mpeg_quant = 0`) — chosen to avoid mismatch
//!   control. `vop_quant` is configurable (default 5) and stays constant
//!   across the picture (no dquant).
//! * AC prediction strategy: **disabled** for every intra MB. The decoder
//!   still accepts `ac_pred_flag = 0`; emitting AC predictions only saves
//!   bits and is not required for correctness.
//! * DC prediction: gradient-direction predictor matching the decoder
//!   (§7.4.3.1) — only the differential is written.
//! * Resync markers: not emitted (`resync_marker_disable = 1` in the VOL).
//!   The encoder is correct without them; ffmpeg accepts streams with the
//!   flag set.
//! * GOP structure: I-VOP every `DEFAULT_GOP_SIZE` frames, P-VOPs in between.
//!   Reference frame is the most recent reconstructed picture.
//!
//! Out of scope (returns `Error::Unsupported` from the encoder factory):
//! * S VOPs (§6.2.5).
//! * Sprites / GMC (§6.2.4 sprite_enable).
//! * Interlace, scalability, data partitioning, reversible VLCs.
//!
//! AC tcoef encoding uses Table B-16 (intra) / B-17 (inter) directly when
//! `(last, run, level)` has a short codeword and falls back to the **third
//! escape mode** (§6.3.8) for any combination that isn't in the short table.
//! Third-escape encodes `(last, run, level)` literally — 1+6+12 bits framed
//! by markers — so any signed 12-bit non-zero level survives.

use std::collections::{HashMap, VecDeque};

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::block::{choose_dc_predictor, BlockNeighbour};
use crate::bvop::trb_trd;
use crate::bvop_enc::encode_b_vop_body;
use crate::headers::vol::ZIGZAG;
use crate::inter::MvGrid;
use crate::iq::{dc_scaler, Y_DC_SCALE_TABLE};
use crate::mb::{IVopPicture, PredGrid};
use crate::pvop::encode_p_vop_body_with_grid;
use crate::start_codes::{VISUAL_OBJECT_START_CODE, VOP_START_CODE, VOS_END_CODE, VOS_START_CODE};
use crate::tables::tcoef;
use oxideav_core::bits::BitWriter;

// -------------------------------------------------------------------------
// Public factory + Encoder impl
// -------------------------------------------------------------------------

/// Default vop_quant for the encoder. The acceptance bar specifies
/// `vop_quant = 5`.
pub const DEFAULT_VOP_QUANT: u32 = 5;

/// Minimum and maximum legal `vop_quant` values per the VOP-header 5-bit
/// field (§6.2.5). Zero is reserved and 32+ doesn't fit in 5 bits, so the
/// encoder rejects anything outside `[1, 31]`.
pub const MIN_VOP_QUANT: u32 = 1;
pub const MAX_VOP_QUANT: u32 = 31;

/// Default GOP size (I-VOP cadence). Emit an I-VOP every `DEFAULT_GOP_SIZE`
/// frames; all other frames are P-VOPs. The P-VOP test in
/// `tests/p_vop.rs` exercises this with `GOP_SIZE = 16` (1 I + 15 P).
pub const DEFAULT_GOP_SIZE: u32 = 16;

/// Conservative upper bound on the `g` (GOP-size) option. Larger
/// values are accepted at the spec level but break our reference-frame
/// drift tests; 300 is a sensible ceiling for any short-form clip.
pub const MAX_GOP_SIZE: u32 = 300;

/// Forward motion-vector range code for P-VOPs. `f_code = 1` gives the
/// smallest range `[-32, 31]` half-pels which is plenty for the encoder's
/// tiny diamond search (bounded at ±7 integer pels). The decoder accepts 1-7.
pub const DEFAULT_F_CODE_FWD: u8 = 1;

/// Default number of B-frames between reference pictures. 0 disables the
/// B-VOP path entirely (I + P only, matching pre-round-8 behaviour).
pub const DEFAULT_MAX_B_FRAMES: u32 = 0;

/// Default `quarter_sample` flag. When `true` the encoder advertises QPel
/// in the VOL header (verid=2 + `quarter_sample = 1`) and uses the
/// quarter-pel motion-estimation + 8-tap-filter prediction path
/// (§7.6.2.2). When `false` (default) the encoder uses half-pel motion
/// (the round-1..14 path).
pub const DEFAULT_QUARTER_SAMPLE: bool = false;

/// Default GMC mode. When `true` the encoder enables single-warp-point
/// Global Motion Compensation (ASP §7.6.7 / §7.7): the VOL advertises
/// `sprite_enable = 2` + `no_of_sprite_warping_points = 1`, every P-VOP
/// header carries one `(du, dv)` `sprite_trajectory()` pair, and each
/// Inter-MB carries an `mcsel` bit picking between translational MC
/// and warp-predicted MC. When `false` (default) the encoder behaves
/// exactly as in round-19.
pub const DEFAULT_GMC: bool = false;

/// Default number of GMC warp points. `1` = pure global translation
/// (round-20 behaviour). `2` = conformal (rotation + scale + translate).
/// `3` = affine (general 2D linear + translate). `4` = perspective
/// (full 8-DOF projective transform). Values 2..=4 require the multi-
/// warp encoder path (`build_multi_warp_trajectory`). Override via
/// the `gmc_warp_points` codec option (1..=4); ignored when `gmc=0`.
pub const DEFAULT_GMC_WARP_POINTS: u8 = 1;

/// Default `data_partitioned` mode. When `true` the encoder advertises
/// `data_partitioned = 1` in the VOL and emits each VOP's body using
/// the `data_partitioned_motion_shape_texture()` layout (§6.2.6 /
/// §6.3.7) — per-MB header/MV bits in part 1, then a 19-/17-bit marker,
/// then per-MB texture (CBPY + AC) in part 2. Used by error-resilient
/// transmission profiles. When `false` (default) the encoder uses the
/// combined-mode layout (every MB's bits inline), preserving round-20
/// behaviour. Mutually compatible with QPel and B-frames at the
/// bitstream level — DP is purely a per-VOP body re-ordering.
pub const DEFAULT_DATA_PARTITIONED: bool = false;

/// `sprite_warping_accuracy` advertised when GMC is on. Code `0` = 1/2-pel
/// (s = 2), which matches the encoder's half-pel ME unit and keeps the
/// trajectory `(du, dv)` representation small. Codes 1/2/3 = 1/4, 1/8,
/// 1/16-pel are valid but require correspondingly finer global ME.
pub const GMC_SPRITE_WARPING_ACCURACY: u8 = 0;

/// Default `reversible_vlc` mode. When `true` (and `data_partitioned`
/// is also on, per §6.2.5), every DCT-coefficient AC walk in the
/// emitted bitstream goes through the Table B.23 RVLC writer
/// (`crate::rvlc`) instead of the standard Table B.16/B.17 tcoef
/// writer. Costs slightly more bits but lets a decoder recover from a
/// mid-block corruption by walking back from the end-of-block marker
/// (Annex E.1.4.4). When `false` (default), AC walks use the standard
/// forward-only tcoef tables — preserving round-21 behaviour.
pub const DEFAULT_REVERSIBLE_VLC: bool = false;

/// Encoder factory used by `register()`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("mpeg4 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("mpeg4 encoder: missing height"))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid("mpeg4 encoder: zero-sized frame"));
    }
    if width > 8191 || height > 8191 {
        // VOL `video_object_layer_width` / `_height` are 13-bit fields.
        return Err(Error::invalid(
            "mpeg4 encoder: dimensions exceed 13-bit VOL field",
        ));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "mpeg4 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(24, 1));

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);

    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    // Options — `bf` (max B-frames), `qpel` (quarter-sample motion),
    // `qp` (per-VOP-type quantiser; aliases `qp_i` / `qp_p` / `qp_b`
    // for I, P and B VOPs separately) and `g` (GOP size: I-VOP cadence
    // in frames). Defaults preserve the round-14..18 behaviour
    // (no B-frames, half-pel motion, vop_quant=5, GOP=16).
    let max_b_frames = params
        .options
        .get("bf")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MAX_B_FRAMES);

    let quarter_sample = params
        .options
        .get("qpel")
        .map(|s| !matches!(s, "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(DEFAULT_QUARTER_SAMPLE);

    // `gmc` enables single-warp-point Global Motion Compensation
    // (§7.6.7 / §7.7). Mutually compatible with QPel and B-frames at the
    // bitstream level — the VOL advertises ASP + verid=2 anyway when
    // either of those is on.
    let gmc_enabled = params
        .options
        .get("gmc")
        .map(|s| !matches!(s, "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(DEFAULT_GMC);

    // `gmc_warp_points` (1..=4) — only meaningful when `gmc=1`. 1 keeps the
    // round-20 single-warp behaviour; 2/3/4 unlock conformal / affine /
    // perspective warps via per-corner pel-domain ME (see
    // `build_multi_warp_trajectory`). Out-of-range values are rejected at
    // factory time so callers get a clear error.
    let gmc_warp_points = match params.options.get("gmc_warp_points") {
        Some(s) => {
            let v: u8 = s.parse().map_err(|_| {
                Error::invalid(format!(
                    "mpeg4 encoder: option gmc_warp_points={s} not an integer"
                ))
            })?;
            if !(1..=4).contains(&v) {
                return Err(Error::invalid(format!(
                    "mpeg4 encoder: option gmc_warp_points={v} outside [1, 4]"
                )));
            }
            v
        }
        None => DEFAULT_GMC_WARP_POINTS,
    };
    if gmc_warp_points > 1 && !gmc_enabled {
        return Err(Error::invalid(
            "mpeg4 encoder: gmc_warp_points>1 requires gmc=1",
        ));
    }

    // `dp` enables data partitioning (§6.2.6 / §6.3.7). When on, every
    // VOP body is emitted using `data_partitioned_motion_shape_texture()`
    // layout: per-MB part 1 (mcbpc + DC for I-VOPs; not_coded + mcbpc +
    // mcsel + MV for P-VOPs), DC marker (19 bits) or motion marker
    // (17 bits), per-MB part 2 (cbpy + texture), per-MB AC walks. The
    // VOL advertises `data_partitioned = 1` + `reversible_vlc = 0`
    // (RVLC is a follow-up). DP at the spec level requires
    // `resync_marker_disable = 0` — the encoder still emits one packet
    // per picture (no mid-VOP video_packet_header() splits yet) but
    // the VOL bit is set so future encoder passes can introduce splits
    // without re-flipping it.
    let data_partitioned = params
        .options
        .get("dp")
        .map(|s| !matches!(s, "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(DEFAULT_DATA_PARTITIONED);
    if data_partitioned && (gmc_enabled || quarter_sample || max_b_frames > 0) {
        // The DP encoder body covers I + P at half-pel, including
        // 1MV-Inter / Inter4MV / Intra-in-P / not_coded MBs. GMC's
        // mcsel bit interaction with the motion partition, QPel's MV
        // unit doubling, and B-VOPs (which fall back to combined-mode
        // per spec NOTE in §6.2.5.3) still need additional plumbing
        // before we can advertise them under DP. Reject the combo
        // explicitly so callers get a clear error.
        return Err(Error::unsupported(
            "mpeg4 encoder: data_partitioned=1 currently requires gmc=0, qpel=0, bf=0",
        ));
    }

    // `rvlc` enables Reversible VLC for DCT-coefficient AC walks
    // (Table B.23, §7.4.1.2). The spec only allows `reversible_vlc =
    // 1` together with `data_partitioned = 1`, so we reject `rvlc=1
    // dp=0` at the factory rather than silently disabling. RVLC plus
    // any feature already disabled under DP (qpel / gmc / bf) is
    // already rejected by the DP combo check above.
    let reversible_vlc = params
        .options
        .get("rvlc")
        .map(|s| !matches!(s, "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(DEFAULT_REVERSIBLE_VLC);
    if reversible_vlc && !data_partitioned {
        return Err(Error::unsupported(
            "mpeg4 encoder: rvlc=1 requires dp=1 (per ISO/IEC 14496-2 §6.2.5)",
        ));
    }

    // `qp` sets the default quant for all VOP types; `qp_i`, `qp_p`
    // and `qp_b` override per VOP type. All values are clamped to
    // [MIN_VOP_QUANT, MAX_VOP_QUANT] — out-of-range strings are
    // rejected with `Error::invalid` so callers get a clear signal
    // rather than silent clamping.
    let parse_qp = |key: &str| -> Result<Option<u32>> {
        let Some(s) = params.options.get(key) else {
            return Ok(None);
        };
        let v: u32 = s.parse().map_err(|_| {
            Error::invalid(format!("mpeg4 encoder: option {key}={s} not an integer"))
        })?;
        if !(MIN_VOP_QUANT..=MAX_VOP_QUANT).contains(&v) {
            return Err(Error::invalid(format!(
                "mpeg4 encoder: option {key}={v} outside [{MIN_VOP_QUANT}, {MAX_VOP_QUANT}]"
            )));
        }
        Ok(Some(v))
    };
    let qp_default = parse_qp("qp")?.unwrap_or(DEFAULT_VOP_QUANT);
    let vop_quant_i = parse_qp("qp_i")?.unwrap_or(qp_default);
    let vop_quant_p = parse_qp("qp_p")?.unwrap_or(qp_default);
    let vop_quant_b = parse_qp("qp_b")?.unwrap_or(qp_default);

    let gop_size = match params.options.get("g") {
        Some(s) => {
            let v: u32 = s.parse().map_err(|_| {
                Error::invalid(format!("mpeg4 encoder: option g={s} not an integer"))
            })?;
            if !(1..=MAX_GOP_SIZE).contains(&v) {
                return Err(Error::invalid(format!(
                    "mpeg4 encoder: option g={v} outside [1, {MAX_GOP_SIZE}]"
                )));
            }
            v
        }
        None => DEFAULT_GOP_SIZE,
    };

    // Round-16: QPel + B-frames is now implemented. The B-VOP encoder
    // (`bvop_enc.rs`) accepts a `quarter_sample` flag and switches its
    // forward/backward ME, MC, chroma reduction, and direct-mode MV
    // scaling to the quarter-pel paths (§7.6.2.2). The bitstream stays
    // conformant: the VOL `quarter_sample = 1` flag tells the decoder
    // that all forward/backward MVDs in B-VOPs are in QPel units.

    Ok(Box::new(Mpeg4VideoEncoder {
        output_params,
        width,
        height,
        frame_rate,
        time_base,
        vop_quant_i,
        vop_quant_p,
        vop_quant_b,
        gop_size,
        f_code_fwd: DEFAULT_F_CODE_FWD,
        max_b_frames,
        quarter_sample,
        gmc_enabled,
        gmc_warp_points,
        data_partitioned,
        reversible_vlc,
        pending: VecDeque::new(),
        b_queue: VecDeque::new(),
        eof: false,
        finalised: false,
        headers_emitted: false,
        display_index: 0,
        vop_count: 0,
        reference: None,
        reference_grid: None,
        reference_time: 0,
        rounding_type: false,
    }))
}

struct Mpeg4VideoEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    frame_rate: Rational,
    time_base: TimeBase,
    /// Per-VOP-type quantisers. Each VOP is encoded with the matching
    /// `vop_quant_{i,p,b}`; the value is written verbatim to the
    /// 5-bit VOP-header `vop_quant` field (§6.2.5). Defaults to the
    /// shared `qp` knob (= `DEFAULT_VOP_QUANT` when unset). All three
    /// stay constant within a picture (no dquant emission).
    vop_quant_i: u32,
    vop_quant_p: u32,
    vop_quant_b: u32,
    gop_size: u32,
    f_code_fwd: u8,
    /// Max consecutive B-VOPs between two reference pictures. 0 = no B-VOPs.
    max_b_frames: u32,
    /// VOL `quarter_sample` flag. When `true` the encoder operates in
    /// QPel mode (verid=2 + quarter-pel motion vectors + 8-tap MC
    /// filter). See `pvop::encode_p_vop_body_with_grid` for the QPel
    /// motion-estimation path.
    quarter_sample: bool,
    /// VOL `sprite_enable == 2` (single-warp-point GMC) flag. When `true`
    /// the encoder advertises GMC in the VOL (verid=2 +
    /// `no_of_sprite_warping_points = 1` + `sprite_warping_accuracy = 0`),
    /// emits one `(du, dv)` `sprite_trajectory()` per P-VOP, and adds an
    /// `mcsel` bit to each Inter MB. See `pvop::encode_p_vop_body_with_grid`
    /// for the per-MB warp-vs-translational decision and global-motion
    /// estimation.
    gmc_enabled: bool,
    /// Number of GMC warp points (1..=4). Controls the warp's degrees of
    /// freedom: 1 = pure translation, 2 = conformal (rotation/scale +
    /// translation), 3 = affine (general 2D linear + translation),
    /// 4 = perspective (full 8-DOF projective). The encoder advertises
    /// this value in the VOL `no_of_sprite_warping_points` field and emits
    /// `n` `(du, dv)` pairs per `sprite_trajectory()`. See
    /// [`build_multi_warp_trajectory`] for the per-corner pel-domain
    /// estimator. Ignored when `gmc_enabled == false`.
    gmc_warp_points: u8,
    /// VOL `data_partitioned == 1` flag — see [`crate::dp`] for the layout.
    /// When `true`, every I/P-VOP body is emitted via
    /// `dp::encode_i_vop_body_dp_and_reconstruct` /
    /// `dp::encode_p_vop_body_dp_with_grid` instead of the combined-mode
    /// emit path. The VOL also flips `resync_marker_disable = 0` so the
    /// stream is conformant for future per-packet splitting.
    data_partitioned: bool,
    /// VOL `reversible_vlc == 1` flag — see [`crate::rvlc`]. Only valid
    /// together with `data_partitioned == 1`; the encoder factory
    /// rejects any other combination. When `true`, every DCT-coefficient
    /// AC walk inside the DP body is emitted through the Table B.23
    /// RVLC writer (`rvlc::write_intra_ac` / `rvlc::write_inter_ac`)
    /// instead of `encoder::write_intra_ac` / `pvop::write_inter_ac`.
    reversible_vlc: bool,
    pending: VecDeque<Packet>,
    /// Display-order B-frame queue — flushed on the next I/P encode.
    b_queue: VecDeque<VideoFrame>,
    eof: bool,
    finalised: bool,
    headers_emitted: bool,
    /// Display-order index of the next incoming frame. Used to decide
    /// I/P vs B for each frame given the GOP cadence + `max_b_frames`.
    display_index: u32,
    /// Decode-order VOP count — used as the `vop_time_increment` field.
    vop_count: u32,
    /// Reconstructed previous I/P picture — used as the forward reference
    /// for subsequent P-VOPs and B-VOPs. Refreshed by every I/P encode.
    reference: Option<IVopPicture>,
    /// MV grid of `reference` — used by B-VOPs for co-located inheritance.
    reference_grid: Option<MvGrid>,
    /// Absolute display time (in display-index units) of `reference`.
    reference_time: i64,
    /// `vop_rounding_type` to emit on the next P-VOP. Per FFmpeg convention
    /// we toggle this between P-VOPs (starts at 0 after an I-VOP, alternates
    /// afterwards) — it matches the half-pel rounding inside `mc.rs`.
    rounding_type: bool,
}

impl Encoder for Mpeg4VideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("mpeg4 encoder: video frames only")),
        };
        if v.planes.len() != 3 {
            return Err(Error::invalid("mpeg4 encoder: expected 3 planes"));
        }

        if self.max_b_frames == 0 {
            // Legacy path: I + P only. Emit each incoming frame directly.
            self.emit_i_or_p(v)?;
        } else {
            // B-frame path. We classify each incoming display-order frame
            // as a reference (I/P) or a B. Reference positions are at
            // `display_index % (bf + 1) == 0` (and every `gop_size`-th is
            // an I). B frames are buffered until the next reference
            // arrives — at that point we encode the reference (which
            // becomes the new backward ref), then encode each buffered
            // B using (forward_ref, backward_ref).
            let bf = self.max_b_frames;
            let is_reference_position = self.display_index % (bf + 1) == 0;
            if is_reference_position {
                self.flush_as_reference(v)?;
            } else {
                // Buffer for later.
                self.b_queue.push_back(v.clone());
            }
        }
        self.display_index += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof && !self.b_queue.is_empty() {
            // EOF flush: any trailing B-frames lose their backward
            // reference — re-encode them as P-VOPs against the current
            // forward reference. This keeps the stream parseable and
            // loses only a small amount of compression on the tail.
            let queued: Vec<VideoFrame> = self.b_queue.drain(..).collect();
            for v in queued {
                self.emit_p_using_forward_ref(&v)?;
            }
            if let Some(p) = self.pending.pop_front() {
                return Ok(p);
            }
        }
        if self.eof && !self.finalised {
            self.finalised = true;
            // Emit a VOS end marker so downstream tools see a clean trailer.
            let mut bw = BitWriter::new();
            write_start_code(&mut bw, VOS_END_CODE);
            let bytes = bw.finish();
            let mut pkt = Packet::new(0, self.time_base, bytes);
            pkt.flags.header = true;
            return Ok(pkt);
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

impl Mpeg4VideoEncoder {
    /// Legacy I + P emit path (no B-frames).
    fn emit_i_or_p(&mut self, v: &VideoFrame) -> Result<()> {
        let is_keyframe = self.vop_count % self.gop_size == 0 || self.reference.is_none();
        let mut bw = BitWriter::with_capacity(8192);
        if !self.headers_emitted {
            write_vos_vo_vol(
                &mut bw,
                self.width,
                self.height,
                self.frame_rate,
                self.vop_quant_i,
                self.max_b_frames > 0,
                self.quarter_sample,
                self.gmc_enabled,
                self.gmc_warp_points,
                self.data_partitioned,
                self.reversible_vlc,
            );
            self.headers_emitted = true;
        }
        // Display time in VOL time-base ticks = original frame pts.
        let time_inc = display_time_inc(v, self.display_index);
        let vti_resolution = (self.frame_rate.num as u32).max(1);
        if is_keyframe {
            write_i_vop_header(&mut bw, time_inc, self.vop_quant_i, vti_resolution);
            let pic = if self.data_partitioned {
                crate::dp::encode_i_vop_body_dp_and_reconstruct(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    self.vop_quant_i,
                    self.reversible_vlc,
                )?
            } else {
                encode_i_vop_body_and_reconstruct(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    self.vop_quant_i,
                )?
            };
            self.reference = Some(pic);
            self.reference_grid = None;
            self.reference_time = time_inc as i64;
            self.rounding_type = false;
        } else {
            let reference = self
                .reference
                .as_ref()
                .expect("P-VOP path requires a reference picture");
            // GMC trajectory + warp derivation. Built once per P-VOP from
            // a coarse global-translation search; written after fcode in
            // the P-VOP header and consumed by every Inter MB during the
            // body encode.
            let (trajectory, warp) = if self.gmc_enabled {
                let (t, w) = build_gmc_trajectory(
                    v,
                    self.width,
                    self.height,
                    reference,
                    self.gmc_warp_points,
                );
                (Some(t), Some(w))
            } else {
                (None, None)
            };
            write_p_vop_header(
                &mut bw,
                time_inc,
                self.vop_quant_p,
                self.rounding_type,
                self.f_code_fwd,
                vti_resolution,
                trajectory.as_ref(),
            );
            let (pic, grid) = if self.data_partitioned {
                crate::dp::encode_p_vop_body_dp_with_grid(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    reference,
                    self.vop_quant_p,
                    self.f_code_fwd,
                    self.rounding_type,
                    self.reversible_vlc,
                )?
            } else {
                encode_p_vop_body_with_grid(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    reference,
                    self.vop_quant_p,
                    self.f_code_fwd,
                    self.rounding_type,
                    self.quarter_sample,
                    warp.as_ref(),
                )?
            };
            self.reference = Some(pic);
            self.reference_grid = Some(grid);
            self.reference_time = time_inc as i64;
        }
        // MPEG-4 §5.2.4 next_start_code() stuffing — `0` then `1`'s to
        // byte boundary; `0x7F` if already aligned. DP-conformant decoders
        // (ffmpeg) require this exact pattern at the end of a video
        // packet so they don't keep parsing into the trailing zeros and
        // misinterpret them as more MB data.
        if self.data_partitioned {
            align_with_one_zero_then_ones(&mut bw);
        } else {
            bw.align_to_byte_zero();
        }
        let bytes = bw.finish();
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        self.vop_count += 1;
        Ok(())
    }

    /// B-frame emit path — the current frame is a reference (I or P). We
    /// encode and push it first (decode order places the reference before
    /// the B frames that point at it), then drain the `b_queue` and emit
    /// each buffered B against (prev_forward_ref, this-new-ref).
    fn flush_as_reference(&mut self, v: &VideoFrame) -> Result<()> {
        // Save the previous forward reference before overwriting it.
        let prev_forward_ref = self.reference.clone();
        let prev_forward_time = self.reference_time;

        let is_keyframe = self.vop_count % self.gop_size == 0 || self.reference.is_none();

        let mut bw = BitWriter::with_capacity(8192);
        if !self.headers_emitted {
            write_vos_vo_vol(
                &mut bw,
                self.width,
                self.height,
                self.frame_rate,
                self.vop_quant_i,
                self.max_b_frames > 0,
                self.quarter_sample,
                self.gmc_enabled,
                self.gmc_warp_points,
                self.data_partitioned,
                self.reversible_vlc,
            );
            self.headers_emitted = true;
        }

        let time_inc = display_time_inc(v, self.display_index);
        let vti_resolution = (self.frame_rate.num as u32).max(1);
        if is_keyframe {
            write_i_vop_header(&mut bw, time_inc, self.vop_quant_i, vti_resolution);
            let pic = if self.data_partitioned {
                crate::dp::encode_i_vop_body_dp_and_reconstruct(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    self.vop_quant_i,
                    self.reversible_vlc,
                )?
            } else {
                encode_i_vop_body_and_reconstruct(
                    &mut bw,
                    v,
                    self.width,
                    self.height,
                    self.vop_quant_i,
                )?
            };
            self.reference = Some(pic);
            self.reference_grid = None; // I-VOPs have no MV grid.
            self.reference_time = time_inc as i64;
            self.rounding_type = false;
        } else {
            let reference = prev_forward_ref
                .as_ref()
                .expect("P-VOP path requires a reference picture");
            let (trajectory, warp) = if self.gmc_enabled {
                let (t, w) = build_gmc_trajectory(
                    v,
                    self.width,
                    self.height,
                    reference,
                    self.gmc_warp_points,
                );
                (Some(t), Some(w))
            } else {
                (None, None)
            };
            write_p_vop_header(
                &mut bw,
                time_inc,
                self.vop_quant_p,
                self.rounding_type,
                self.f_code_fwd,
                vti_resolution,
                trajectory.as_ref(),
            );
            let (pic, grid) = encode_p_vop_body_with_grid(
                &mut bw,
                v,
                self.width,
                self.height,
                reference,
                self.vop_quant_p,
                self.f_code_fwd,
                self.rounding_type,
                self.quarter_sample,
                warp.as_ref(),
            )?;
            self.reference = Some(pic);
            self.reference_grid = Some(grid);
            self.reference_time = time_inc as i64;
        }
        // Spec-conformant `next_start_code()` stuffing — see `emit_i_or_p`
        // for the full rationale (DP-conformant decoders need this exact
        // pattern at the end of a video packet).
        if self.data_partitioned {
            align_with_one_zero_then_ones(&mut bw);
        } else {
            bw.align_to_byte_zero();
        }
        let bytes = bw.finish();
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        self.vop_count += 1;

        // Now emit each buffered B-VOP against (prev_forward_ref,
        // new_backward_ref).
        let bs: Vec<VideoFrame> = self.b_queue.drain(..).collect();
        if !bs.is_empty() {
            // For B-VOPs, we need both a forward (past) and backward
            // (future) reference. At this point `self.reference` is the
            // NEW backward ref; `prev_forward_ref` is the past ref.
            // If there's no prev_forward_ref (e.g. this is the first
            // reference in the stream), fall back to treating each
            // buffered B as a stand-alone P — they can't B-code.
            if let Some(prev_forward) = prev_forward_ref {
                let next_backward = self
                    .reference
                    .as_ref()
                    .expect("backward reference set by this function")
                    .clone();
                let next_backward_grid = self.reference_grid.clone();
                let cur_ref_time = self.reference_time;
                for b_frame in &bs {
                    self.emit_b_vop(
                        b_frame,
                        &prev_forward,
                        &next_backward,
                        next_backward_grid.as_ref(),
                        prev_forward_time,
                        cur_ref_time,
                    )?;
                }
            } else {
                for b_frame in &bs {
                    self.emit_p_using_forward_ref(b_frame)?;
                }
            }
        }
        Ok(())
    }

    /// Emit one B-VOP packet. Uses `prev_forward` as the past reference
    /// and `next_backward` (with optional MV grid) as the future
    /// reference. `trb / trd` are derived from the display times.
    fn emit_b_vop(
        &mut self,
        v: &VideoFrame,
        prev_forward: &IVopPicture,
        next_backward: &IVopPicture,
        next_backward_grid: Option<&MvGrid>,
        prev_time: i64,
        next_time: i64,
    ) -> Result<()> {
        // Fallback display index = current display_index - 1 - queue size.
        let fallback = (self.display_index as i64).saturating_sub(1);
        let time_inc_u = v
            .pts
            .map(|p| p.max(0) as u32)
            .unwrap_or(fallback.max(0) as u32);
        let cur_time = time_inc_u as i64;
        let (trb, trd) = trb_trd(prev_time, cur_time, next_time);
        let mut bw = BitWriter::with_capacity(8192);
        let vti_resolution = (self.frame_rate.num as u32).max(1);
        write_b_vop_header(
            &mut bw,
            time_inc_u,
            self.vop_quant_b,
            self.f_code_fwd,
            self.f_code_fwd,
            vti_resolution,
        );

        // Co-located grid: `next_backward`'s MV grid. An I-VOP has no
        // MV grid (`next_backward_grid = None`) — `encode_b_vop_body`
        // treats every MB as implicitly not-coded in that case, but we
        // have none right now (B-VOPs only appear after the first P).
        // Make a zero-MV grid if missing so our encoder has a grid to
        // query; that still emits a full body (no implicit skips).
        let mb_w = (self.width as usize).div_ceil(16);
        let mb_h = (self.height as usize).div_ceil(16);
        let fallback_grid = MvGrid::new(mb_w, mb_h);
        let grid = next_backward_grid.unwrap_or(&fallback_grid);

        encode_b_vop_body(
            &mut bw,
            v,
            self.width,
            self.height,
            prev_forward,
            next_backward,
            grid,
            self.vop_quant_b,
            self.f_code_fwd,
            self.f_code_fwd,
            trb,
            trd,
            self.quarter_sample,
        )?;
        bw.align_to_byte_zero();
        let bytes = bw.finish();
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = false;
        self.pending.push_back(pkt);
        self.vop_count += 1;
        Ok(())
    }

    /// Fallback: emit a single frame as a P-VOP against the current
    /// forward reference. Used when a B-frame can't be coded (no
    /// backward ref) — e.g. trailing B-frames at EOF or when the first
    /// reference hasn't been emitted yet.
    fn emit_p_using_forward_ref(&mut self, v: &VideoFrame) -> Result<()> {
        self.emit_i_or_p(v)
    }
}

// -------------------------------------------------------------------------
// Start-code + header emission
// -------------------------------------------------------------------------

fn write_start_code(bw: &mut BitWriter, code: u8) {
    bw.align_to_byte_zero();
    bw.write_bytes(&[0x00, 0x00, 0x01, code]);
}

/// Compute the display-order time-increment to stamp into a VOP header.
/// Uses the frame's own PTS when available, else the monotonic
/// `display_index` counter. Wrapping to a u32 is safe for our tests —
/// the VTI field masks against 24 in the header writer.
fn display_time_inc(v: &VideoFrame, fallback: u32) -> u32 {
    v.pts.map(|p| p.max(0) as u32).unwrap_or(fallback)
}

/// Compute the smallest number of bits required to hold `max_value`. Mirrors
/// the decoder's `bits_needed` helper.
fn bits_needed(max_value: u32) -> u32 {
    if max_value == 0 {
        1
    } else {
        32 - max_value.leading_zeros()
    }
}

/// Emit the canonical Visual Object Sequence + Visual Object + Video Object +
/// Video Object Layer headers. Profile is **Simple Profile @ Level 1**
/// (`profile_and_level_indication = 0x01`) — the most-compatible PLI for
/// typical XVID/DivX-style elementary streams that ffmpeg's `mpeg4` decoder
/// happily consumes. Layer geometry is encoded at the picture's natural
/// resolution; the `frame_rate` is encoded as
/// `vop_time_increment_resolution = num`, `fixed_vop_time_increment = den`.
#[allow(clippy::too_many_arguments)]
fn write_vos_vo_vol(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    frame_rate: Rational,
    _q: u32,
    enable_b_vops: bool,
    quarter_sample: bool,
    gmc_enabled: bool,
    gmc_warp_points: u8,
    data_partitioned: bool,
    reversible_vlc: bool,
) {
    // VOS.
    write_start_code(bw, VOS_START_CODE);
    // profile_and_level_indication — pick the smallest PLI that admits
    // every feature we actually emit:
    //   * ASP Level 1 (`0xF1`) — B-VOPs / QPel / GMC (Annex N), AND DP
    //     (ARTS profile lacks Inter4MV; ASP is the smallest profile
    //     that admits DP + 4MV simultaneously per Table G.1 / Annex G).
    //   * Simple Profile Level 1 (`0x01`) — most-compatible default
    //     for plain I+P half-pel.
    let pli = if enable_b_vops || quarter_sample || gmc_enabled || data_partitioned {
        0xF1
    } else {
        0x01
    };
    bw.write_bits(pli, 8);

    // Visual Object.
    write_start_code(bw, VISUAL_OBJECT_START_CODE);
    bw.write_bits(0, 1); // is_visual_object_identifier = 0
    bw.write_bits(1, 4); // visual_object_type = 1 (Video)
    bw.write_bits(0, 1); // video_signal_type = 0
                         // next_start_code() — pad to byte boundary with `0_111_1111` style stuffing.
    align_with_one_zero_then_ones(bw);

    // Video Object — id 0, no payload of interest.
    write_start_code(bw, 0x00);

    // Video Object Layer — id 0x20.
    write_start_code(bw, 0x20);
    bw.write_bits(0, 1); // random_accessible_vol = 0
                         // video_object_type_indication — ASP (4) covers
                         // every "advanced" feature we ship: B-VOPs,
                         // QPel, GMC and DP+Inter4MV. The bare Simple
                         // (1) profile is reserved for plain I+P
                         // half-pel without DP.
    let vot_indication = if enable_b_vops || quarter_sample || gmc_enabled || data_partitioned {
        4
    } else {
        1
    };
    bw.write_bits(vot_indication, 8);
    // `is_object_layer_identifier` is required when we need verid=2 to
    // unlock QPel / GMC / DP-with-4MV syntax. For the half-pel + no-B +
    // no-GMC + no-DP path keep it 0 so the bitstream is byte-for-byte
    // identical to round-14.
    let needs_verid2 = quarter_sample || gmc_enabled || data_partitioned;
    if needs_verid2 {
        bw.write_bits(1, 1); // is_object_layer_identifier = 1
        bw.write_bits(2, 4); // verid = 2 (QPel + GMC + newpred syntax)
        bw.write_bits(0, 3); // priority = 0
    } else {
        bw.write_bits(0, 1); // is_object_layer_identifier = 0 (verid implicitly 1)
    }
    bw.write_bits(1, 4); // aspect_ratio_info = 1 (square)
    bw.write_bits(1, 1); // vol_control_parameters = 1
    bw.write_bits(1, 2); // chroma_format = 1 (4:2:0)
                         // low_delay — must be 0 when B-VOPs are in the stream, so the decoder
                         // knows to enable its reorder queue. 1 otherwise (no reorder needed).
    let low_delay = if enable_b_vops { 0 } else { 1 };
    bw.write_bits(low_delay, 1);
    bw.write_bits(0, 1); // vbv_parameters = 0
    bw.write_bits(0, 2); // video_object_layer_shape = 0 (Rectangular)
    bw.write_bits(1, 1); // marker

    let resolution = (frame_rate.num as u32).clamp(1, 0xFFFF);
    bw.write_bits(resolution, 16); // vop_time_increment_resolution
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // fixed_vop_rate = 1
    let vti_bits = bits_needed(resolution.saturating_sub(1)).max(1);
    let fixed_vti = (frame_rate.den as u32).max(1);
    bw.write_bits(fixed_vti, vti_bits);

    bw.write_bits(1, 1); // marker
    bw.write_bits(width, 13);
    bw.write_bits(1, 1); // marker
    bw.write_bits(height, 13);
    bw.write_bits(1, 1); // marker

    bw.write_bits(0, 1); // interlaced = 0
    bw.write_bits(1, 1); // obmc_disable = 1
                         // sprite_enable — 1 bit when verid==1, 2 bits when verid>=2
                         // (per the round-2000 corrigendum that introduced GMC).
                         // GMC sets this to 2; static-sprite mode (1) is not
                         // emitted by the encoder.
    if needs_verid2 {
        let sprite_enable = if gmc_enabled { 2 } else { 0 };
        bw.write_bits(sprite_enable, 2);
    } else {
        bw.write_bits(0, 1); // sprite_enable = 0 (verid==1 → 1 bit)
    }
    // GMC sprite-trajectory descriptors (§6.2.3 + amendment 1). For
    // `sprite_enable == 2` we follow the GMC branch: warping-points + accuracy
    // + brightness-change. The 1-warp-point encoder uses the canonical
    // 1/2-pel accuracy quantiser (`s = 2`) which matches our half-pel ME.
    if gmc_enabled {
        let n_points = gmc_warp_points.clamp(1, 4) as u32;
        bw.write_bits(n_points, 6); // no_of_sprite_warping_points
        bw.write_bits(GMC_SPRITE_WARPING_ACCURACY as u32, 2); // 0 → s=2 (1/2-pel)
        bw.write_bits(0, 1); // sprite_brightness_change = 0
                             // (no `low_latency_sprite_enable` for sprite_enable=2)
    }
    bw.write_bits(0, 1); // not_8_bit = 0
    bw.write_bits(0, 1); // mpeg_quant = 0 (use H.263 quant)
                         // quarter_sample — verid>=2 only. `1` selects QPel motion
                         // (§7.6.2.2 8-tap filter); `0` keeps half-pel.
    if needs_verid2 {
        bw.write_bits(if quarter_sample { 1 } else { 0 }, 1);
    }
    bw.write_bits(1, 1); // complexity_estimation_disable = 1
                         // resync_marker_disable — must be 0 when DP is on
                         // (DP is only legal inside a video packet — the
                         // first packet starts at the VOP header). 1
                         // otherwise.
    bw.write_bits(if data_partitioned { 0 } else { 1 }, 1);
    bw.write_bits(if data_partitioned { 1 } else { 0 }, 1); // data_partitioned
    if data_partitioned {
        // reversible_vlc — only emitted when DP is on (§6.2.3). When
        // set, DCT-coefficient AC walks go through the Table B.23 RVLC
        // tables (`crate::rvlc`) instead of the standard B.16/B.17
        // tcoef tables.
        bw.write_bits(if reversible_vlc { 1 } else { 0 }, 1);
    }
    // verid>=2 adds newpred_enable + reduced_resolution_vop_enable.
    // Both 0 — we don't emit those features.
    if needs_verid2 {
        bw.write_bits(0, 1); // newpred_enable = 0
        bw.write_bits(0, 1); // reduced_resolution_vop_enable = 0
    }
    bw.write_bits(0, 1); // scalability = 0

    align_with_one_zero_then_ones(bw);
}

/// MPEG-4 spec stuffing rule for `next_start_code()` (§6.3.4): write a `0`
/// bit followed by `n` `1` bits where `n` is just enough to byte-align the
/// stream. If already aligned, write a full `01111111` byte.
fn align_with_one_zero_then_ones(bw: &mut BitWriter) {
    if bw.is_byte_aligned() {
        bw.write_byte(0x7F);
        return;
    }
    bw.write_bits(0, 1);
    while !bw.is_byte_aligned() {
        bw.write_bits(1, 1);
    }
}

/// Emit the VOP header for an I-VOP. `time_inc` is the VOP's index in the
/// stream — encoded as the `vop_time_increment` (with `modulo_time_base = 0`
/// since we keep time_inc < resolution). `vop_quant` is constant across the
/// picture.
///
/// `vti_resolution` is the `vop_time_increment_resolution` written into the
/// VOL header (== frame_rate.num) — used to derive `vop_time_increment`'s
/// width per §6.3.5: `bits_needed(resolution - 1)`. Round-12 fix: was
/// previously hardcoded to `bits_needed(23) = 5`, which broke any non-24fps
/// stream because the decoder reads the bits-width from the VOL header.
fn write_i_vop_header(bw: &mut BitWriter, time_inc: u32, vop_quant: u32, vti_resolution: u32) {
    write_start_code(bw, VOP_START_CODE);
    bw.write_bits(0, 2); // vop_coding_type = 00 (I)
    bw.write_bits(0, 1); // modulo_time_base = `0` (terminator)
    bw.write_bits(1, 1); // marker
    let vti_bits = bits_needed(vti_resolution.saturating_sub(1)).max(1);
    bw.write_bits(time_inc % vti_resolution.max(1), vti_bits);
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // vop_coded = 1
                         // (No vop_rounding_type for I.)
                         // intra_dc_vlc_thr = 0 → always use intra DC VLC.
    bw.write_bits(0, 3);
    bw.write_bits(vop_quant, 5);
    // No fcode for I-VOP.
}

/// Emit the VOP header for a B-VOP (§6.2.5, vop_coding_type == 2). Mirrors
/// `write_p_vop_header` plus the extra `vop_fcode_backward` field. No
/// `vop_rounding_type` for B-VOPs (inherited from the enclosing P context).
///
/// `vti_resolution` — see `write_i_vop_header`; round-12 fix to honour the
/// VOL's actual `vop_time_increment_resolution`.
fn write_b_vop_header(
    bw: &mut BitWriter,
    time_inc: u32,
    vop_quant: u32,
    f_code_fwd: u8,
    f_code_bwd: u8,
    vti_resolution: u32,
) {
    write_start_code(bw, VOP_START_CODE);
    bw.write_bits(0b10, 2); // vop_coding_type = 10 (B)
    bw.write_bits(0, 1); // modulo_time_base = `0` terminator
    bw.write_bits(1, 1); // marker
    let vti_bits = bits_needed(vti_resolution.saturating_sub(1)).max(1);
    bw.write_bits(time_inc % vti_resolution.max(1), vti_bits);
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // vop_coded = 1
                         // (No vop_rounding_type for B — see §6.2.5.)
    bw.write_bits(0, 3); // intra_dc_vlc_thr = 0
    bw.write_bits(vop_quant, 5);
    bw.write_bits(f_code_fwd as u32, 3); // vop_fcode_forward
    bw.write_bits(f_code_bwd as u32, 3); // vop_fcode_backward
}

/// Emit the VOP header for a P-VOP. Field layout mirrors `write_i_vop_header`
/// plus the P-VOP-specific `vop_rounding_type` and `vop_fcode_forward` fields
/// (§6.2.5). `time_inc` is per-picture. `rounding_type` is the half-pel
/// rounding flag; `f_code_fwd` is the forward motion range code (1..=7).
///
/// When `sprite_trajectory` is `Some`, the encoder emits an **S(GMC)-VOP**
/// (`vop_coding_type = "S"`) rather than a plain P-VOP, with the
/// trajectory placed BEFORE `vop_quant` per §6.2.5 spec order. Per the
/// spec definition (§6.2.5: "An S(GMC)-VOP can be regarded as a
/// P-VOP"), the body decode is identical to a P-VOP — the per-MB
/// `mcsel` bit picks between translational MC and the warp predictor.
///
/// `vti_resolution` — see `write_i_vop_header`; round-12 fix to honour the
/// VOL's actual `vop_time_increment_resolution`.
fn write_p_vop_header(
    bw: &mut BitWriter,
    time_inc: u32,
    vop_quant: u32,
    rounding_type: bool,
    f_code_fwd: u8,
    vti_resolution: u32,
    sprite_trajectory: Option<&crate::gmc::SpriteTrajectory>,
) {
    write_start_code(bw, VOP_START_CODE);
    // GMC P-substitute: vop_coding_type = "S" (binary 11) when a
    // trajectory is present; plain P (binary 01) otherwise.
    let coding_type = if sprite_trajectory.is_some() {
        0b11
    } else {
        0b01
    };
    bw.write_bits(coding_type, 2);
    bw.write_bits(0, 1); // modulo_time_base = `0` terminator
    bw.write_bits(1, 1); // marker
    let vti_bits = bits_needed(vti_resolution.saturating_sub(1)).max(1);
    bw.write_bits(time_inc % vti_resolution.max(1), vti_bits);
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // vop_coded = 1
                         // vop_rounding_type — emitted for P AND for S+GMC per §6.2.5.
    bw.write_bits(if rounding_type { 1 } else { 0 }, 1);
    bw.write_bits(0, 3); // intra_dc_vlc_thr = 0
                         // GMC sprite_trajectory comes BEFORE vop_quant in the
                         // §6.2.5 order. (Pre-r20 the encoder placed it after
                         // vop_fcode_forward — that emitted byte-decoded
                         // bitstream which our own decoder accepted, but ffmpeg
                         // and any spec-conformant decoder rejected.)
    if let Some(t) = sprite_trajectory {
        crate::gmc::encode_sprite_trajectory(bw, t);
        // (sprite_brightness_change is 0 in the VOL — no
        // brightness_change_factor() bits to emit.)
    }
    bw.write_bits(vop_quant, 5);
    bw.write_bits(f_code_fwd as u32, 3); // vop_fcode_forward
                                         // (No fcode_backward for P / S(GMC).)
}

// -------------------------------------------------------------------------
// GMC: per-VOP global-translation estimation
// -------------------------------------------------------------------------

/// Maximum global-translation search range (integer pels) on each axis.
/// The single-warp-point GMC encoder picks `(du, dv)` = `2 * (dx, dy)` in
/// 1/2-pel units (s=2), so the bitstream MV magnitude is `±2 *
/// MAX_GMC_GLOBAL_SEARCH`. The Table 11-32 `warping_mv_code()` accepts
/// up to ±16383, but a smaller search keeps the coarse pel-domain SAD
/// fast and matches the per-MB f_code=1 motion range.
pub(crate) const MAX_GMC_GLOBAL_SEARCH: i32 = 16;

/// Sampling stride (in pels) for the coarse global-translation SAD scan.
/// We sample every `GMC_SAMPLE_STRIDE`-th pel on each axis to keep the
/// search O(N²/stride²) rather than O(N²·search_range²). The result is
/// not bit-exact relative to a full search but typically agrees within
/// ±1 pel on stationary / panning content — adequate for the
/// per-MB `mcsel` decision pass that follows.
const GMC_SAMPLE_STRIDE: usize = 4;

/// Run a coarse per-VOP global-translation estimator: for each candidate
/// `(dx, dy)` in `[-MAX_GMC_GLOBAL_SEARCH, +MAX_GMC_GLOBAL_SEARCH]`,
/// compute the strided luma SAD between the source frame and the
/// reference frame translated by `(dx, dy)`. Return the best
/// translation as a `SpriteTrajectory` (one `(du, dv)` pair in 1/2-pel
/// units, i.e. `du = 2 * dx`, `dv = 2 * dy`) and the matching
/// `WarpParams`.
///
/// `width` / `height` are the picture dimensions in luma pels. `s = 2`
/// (1/2-pel accuracy) is hard-wired to match the encoder's half-pel
/// motion-estimation grid.
///
/// `n_points` selects the warp shape (1 = pure translation, 2 =
/// conformal, 3 = affine, 4 = perspective). For n>=2 we delegate to
/// [`build_multi_warp_trajectory`] which estimates per-corner motion
/// vectors and inverts the §7.7.4 cumulative-delta encoding.
pub(crate) fn build_gmc_trajectory(
    v: &VideoFrame,
    width: u32,
    height: u32,
    reference: &IVopPicture,
    n_points: u8,
) -> (crate::gmc::SpriteTrajectory, crate::gmc::WarpParams) {
    let n_points = n_points.clamp(1, 4);
    if n_points >= 2 {
        return build_multi_warp_trajectory(v, width, height, reference, n_points);
    }
    let width_us = width as usize;
    let height_us = height as usize;
    let src = &v.planes[0];
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = u64::MAX;
    for dy in -MAX_GMC_GLOBAL_SEARCH..=MAX_GMC_GLOBAL_SEARCH {
        for dx in -MAX_GMC_GLOBAL_SEARCH..=MAX_GMC_GLOBAL_SEARCH {
            let sad = global_translation_sad(src, width_us, height_us, reference, dx, dy);
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    // Half-pel accuracy: du/dv = 2 * pel-shift. (s = 2.)
    let mut t = crate::gmc::SpriteTrajectory {
        points: 1,
        ..Default::default()
    };
    t.du[0] = 2 * best_dx;
    t.dv[0] = 2 * best_dy;
    let vol = synthesise_gmc_vol(width, height, 1);
    let warp = crate::gmc::WarpParams::from_trajectory(&t, &vol);
    (t, warp)
}

/// Multi-warp-point GMC estimator (n=2/3/4). Per §7.7.4 the spec
/// reference points are the picture corners:
/// * point 0 → (0, 0)
/// * point 1 → (W, 0)
/// * point 2 → (0, H)
/// * point 3 → (W, H)
///
/// We estimate per-corner pel-domain motion vectors `D_k = (dx_k, dy_k)`
/// by running a coarse SAD search over a small window centred on each
/// corner of the source frame against the reference (with edge
/// replication), then convert to the spec's cumulative-delta encoding:
/// * du[0] = 2 * dx_0                  (s = 2)
/// * du[1] = 2 * (dx_1 - dx_0)         (delta from point 0 to point 1)
/// * du[2] = 2 * (dx_2 - dx_0)         (delta from point 0 to point 2)
/// * du[3] = 2 * (dx_3 - dx_1 - dx_2 + dx_0)
///   (delta from sum to reach point 3)
///
/// (Same for `dv`.)
///
/// The corner SAD windows use luma blocks of size `CORNER_BLOCK` taken
/// from the source frame; the warp sampler later replicates edge pels
/// for out-of-bounds reads, so corner blocks that extend past the
/// reference rectangle still produce a meaningful SAD.
pub(crate) fn build_multi_warp_trajectory(
    v: &VideoFrame,
    width: u32,
    height: u32,
    reference: &IVopPicture,
    n_points: u8,
) -> (crate::gmc::SpriteTrajectory, crate::gmc::WarpParams) {
    let n = n_points.clamp(2, 4);
    let w = width as i32;
    let h = height as i32;
    let block = CORNER_BLOCK as i32;
    // Corner anchor points — top-left of each corner SAD window in
    // SOURCE-FRAME coordinates. We bias the windows inward by `block/2`
    // so each window straddles the corner pixel rather than running off
    // the picture entirely.
    //
    // Corner indexing matches §7.7.4: 0 = TL, 1 = TR, 2 = BL, 3 = BR.
    let half = block / 2;
    let corners: [(i32, i32); 4] = [
        (-half, -half),       // (0, 0)
        (w - half, -half),    // (W, 0)
        (-half, h - half),    // (0, H)
        (w - half, h - half), // (W, H)
    ];
    // Per-corner pel-domain MV (before scaling to du/dv). For n=2/3 we
    // only use the first n corners; corner 3's MV is left at (0,0) and
    // ignored downstream.
    let mut corner_mv = [(0i32, 0i32); 4];
    for (k, &(cx, cy)) in corners.iter().enumerate() {
        if k as u8 >= n {
            break;
        }
        corner_mv[k] = corner_motion_search(v, reference, cx, cy);
    }

    // Encode per the §7.7.4 cumulative-delta scheme. Half-pel accuracy
    // (s = 2) means du/dv = 2 * pel-shift.
    let mut t = crate::gmc::SpriteTrajectory {
        points: n as usize,
        ..Default::default()
    };
    t.du[0] = 2 * corner_mv[0].0;
    t.dv[0] = 2 * corner_mv[0].1;
    if n >= 2 {
        t.du[1] = 2 * (corner_mv[1].0 - corner_mv[0].0);
        t.dv[1] = 2 * (corner_mv[1].1 - corner_mv[0].1);
    }
    if n >= 3 {
        t.du[2] = 2 * (corner_mv[2].0 - corner_mv[0].0);
        t.dv[2] = 2 * (corner_mv[2].1 - corner_mv[0].1);
    }
    if n == 4 {
        t.du[3] = 2 * (corner_mv[3].0 - corner_mv[1].0 - corner_mv[2].0 + corner_mv[0].0);
        t.dv[3] = 2 * (corner_mv[3].1 - corner_mv[1].1 - corner_mv[2].1 + corner_mv[0].1);
    }
    let vol = synthesise_gmc_vol(width, height, n);
    let warp = crate::gmc::WarpParams::from_trajectory(&t, &vol);
    (t, warp)
}

/// Per-corner pel-domain ME — runs a brute-force `±MAX_GMC_GLOBAL_SEARCH`
/// integer-pel SAD over a `CORNER_BLOCK × CORNER_BLOCK` source window
/// against the reference (with edge replication). Returns the best
/// `(dx, dy)` pel translation. Used by [`build_multi_warp_trajectory`].
fn corner_motion_search(
    v: &VideoFrame,
    reference: &IVopPicture,
    src_x0: i32,
    src_y0: i32,
) -> (i32, i32) {
    let block = CORNER_BLOCK as i32;
    let src = &v.planes[0];
    let src_h = (src.data.len() / src.stride) as i32;
    let src_w = src.stride as i32;
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let read_src = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(0, src_w - 1) as usize;
        let yc = y.clamp(0, src_h - 1) as usize;
        src.data[yc * src.stride + xc] as i32
    };
    let read_ref = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(0, ref_w - 1) as usize;
        let yc = y.clamp(0, ref_h - 1) as usize;
        reference.y[yc * reference.y_stride + xc] as i32
    };
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = u64::MAX;
    for dy in -MAX_GMC_GLOBAL_SEARCH..=MAX_GMC_GLOBAL_SEARCH {
        for dx in -MAX_GMC_GLOBAL_SEARCH..=MAX_GMC_GLOBAL_SEARCH {
            let mut sad: u64 = 0;
            let mut j = 0i32;
            while j < block {
                let mut i = 0i32;
                while i < block {
                    let s = read_src(src_x0 + i, src_y0 + j);
                    let r = read_ref(src_x0 + i + dx, src_y0 + j + dy);
                    sad += (s - r).unsigned_abs() as u64;
                    i += CORNER_SAMPLE_STRIDE as i32;
                }
                j += CORNER_SAMPLE_STRIDE as i32;
            }
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    (best_dx, best_dy)
}

/// Side length (in pels) of the per-corner SAD window used by
/// [`corner_motion_search`]. 32 pels gives enough texture to lock onto
/// the corner motion while keeping the search cost modest
/// (32×32×33×33 ≈ 1.1M ops per corner, dominated by the 33×33 search
/// grid). Sampled every `CORNER_SAMPLE_STRIDE` pels in each axis.
const CORNER_BLOCK: usize = 32;

/// Sampling stride inside each corner window. Mirrors `GMC_SAMPLE_STRIDE`'s
/// use in the global-translation path.
const CORNER_SAMPLE_STRIDE: usize = 4;

/// Strided luma SAD of source vs reference translated by `(dx, dy)`. Each
/// out-of-bounds reference pel is replaced by the nearest edge pel
/// (matches the warp sampler's edge-replication behaviour).
fn global_translation_sad(
    src: &oxideav_core::VideoPlane,
    width: usize,
    height: usize,
    reference: &IVopPicture,
    dx: i32,
    dy: i32,
) -> u64 {
    let ref_h = height as i32;
    let ref_w = width as i32;
    let mut total: u64 = 0;
    let mut y = 0usize;
    while y < height {
        let mut x = 0usize;
        while x < width {
            let s = src.data[y * src.stride + x] as i32;
            let rx = (x as i32 + dx).clamp(0, ref_w - 1) as usize;
            let ry = (y as i32 + dy).clamp(0, ref_h - 1) as usize;
            let r = reference.y[ry * reference.y_stride + rx] as i32;
            total += (s - r).unsigned_abs() as u64;
            x += GMC_SAMPLE_STRIDE;
        }
        y += GMC_SAMPLE_STRIDE;
    }
    total
}

/// Build a minimal `VideoObjectLayer` describing the GMC settings the
/// encoder advertises (verid=2, sprite_enable=2, `n_points` warp points,
/// half-pel accuracy). Used only as context for `WarpParams::from_trajectory`.
fn synthesise_gmc_vol(
    width: u32,
    height: u32,
    n_points: u8,
) -> crate::headers::vol::VideoObjectLayer {
    use crate::headers::vol::{AspectRatioInfo, ChromaFormat, ShapeType, VideoObjectLayer};
    VideoObjectLayer {
        random_accessible_vol: false,
        video_object_type_indication: 4,
        is_object_layer_identifier: true,
        verid: 2,
        priority: 0,
        aspect_ratio_info: AspectRatioInfo::Square,
        vol_control_parameters: true,
        chroma_format: ChromaFormat::Yuv420,
        low_delay: true,
        vbv_parameters_present: false,
        shape: ShapeType::Rectangular,
        vop_time_increment_resolution: 24,
        vop_time_increment_bits: 5,
        fixed_vop_rate: true,
        fixed_vop_time_increment: 1,
        width,
        height,
        interlaced: false,
        obmc_disable: true,
        sprite_enable: 2,
        no_of_sprite_warping_points: n_points.clamp(1, 4),
        sprite_warping_accuracy: GMC_SPRITE_WARPING_ACCURACY,
        sprite_brightness_change: false,
        low_latency_sprite_enable: false,
        sprite_rect: None,
        not_8_bit: false,
        quant_precision: 5,
        bits_per_pixel: 8,
        mpeg_quant: false,
        intra_quant_matrix: None,
        non_intra_quant_matrix: None,
        quarter_sample: false,
        complexity_estimation_disable: true,
        resync_marker_disable: true,
        data_partitioned: false,
        reversible_vlc: false,
        newpred_enable: false,
        reduced_resolution_vop_enable: false,
        scalability: false,
    }
}

// -------------------------------------------------------------------------
// I-VOP body: per-MB encoding
// -------------------------------------------------------------------------

/// Encode an I-VOP body AND return the reconstructed picture so it can be
/// used as the MC reference for subsequent P-VOPs. Uses the shared decoder
/// path so the reconstruction is bit-exact relative to what the decoder
/// would produce from the same bitstream.
pub(crate) fn encode_i_vop_body_and_reconstruct(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: u32,
    height: u32,
    vop_quant: u32,
) -> Result<IVopPicture> {
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut grid = PredGrid::new(mb_w, mb_h);
    // We reconstruct by re-reading our emitted bitstream. That's the same
    // recipe the P-VOP path will need at decode time. To avoid a full
    // second-pass re-decode here, we stash the reconstructed 8×8 samples
    // per block directly as we quantise + IDCT inside `encode_intra_mb`
    // (done below via the `out` parameter).
    let mut pic = IVopPicture::new(width, height);

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            encode_intra_mb_reconstruct(
                bw, v, width, height, mb_x, mb_y, vop_quant, &mut grid, &mut pic,
            )?;
        }
    }
    Ok(pic)
}

/// Selects which MCBPC table to use when emitting an intra MB. I-VOPs use
/// the small Table B-10 (4 entries: 0..=3 for the four `cbpc` values).
/// P-VOPs use Table B-13's "Intra" rows 4..=7 — the call site also
/// emits a leading `not_coded = 0` bit and the encoded MCBPC bit-pattern
/// is different. See §6.3.5 / §6.3.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntraMcbpcKind {
    /// I-VOP: Table B-10.
    IVop,
    /// P-VOP: Table B-13 rows 4..=7 (Intra group). The caller is expected
    /// to also emit `not_coded = 0` BEFORE invoking the intra encoder.
    PVop,
}

/// Encode one intra macroblock AND reconstruct it into `pic`. The
/// reconstructed samples mirror what the decoder would produce from the
/// emitted bitstream, so the resulting picture is bit-exact w.r.t. downstream
/// P-VOP motion compensation references.
fn encode_intra_mb_reconstruct(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
    pic: &mut IVopPicture,
) -> Result<()> {
    encode_intra_mb_inner(
        bw,
        v,
        width,
        height,
        mb_x,
        mb_y,
        quant,
        grid,
        Some(pic),
        IntraMcbpcKind::IVop,
    )
}

/// Encode one intra macroblock embedded inside a P-VOP. Identical to the
/// I-VOP intra encoder except the MCBPC table is Table B-13's Intra rows
/// (caller emits `not_coded = 0` BEFORE calling this). The reconstructed
/// MB is written into `pic` and the DC/AC prediction `grid` is updated so
/// downstream MBs predict from it.
///
/// `pic` is a 16×16 luma + 2×8×8 chroma reconstruction destination
/// (stamped via `write_recon_to_picture`). It must be the same picture
/// that the P-VOP's MC reconstruction writes into so that subsequent
/// reference-frame access is consistent.
pub(crate) fn encode_intra_mb_in_p(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
    pic: &mut IVopPicture,
) -> Result<()> {
    encode_intra_mb_inner(
        bw,
        v,
        width,
        height,
        mb_x,
        mb_y,
        quant,
        grid,
        Some(pic),
        IntraMcbpcKind::PVop,
    )
}

fn encode_intra_mb_inner(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
    mut pic: Option<&mut IVopPicture>,
    mcbpc_kind: IntraMcbpcKind,
) -> Result<()> {
    // Read all six 8×8 sample blocks from the source frame (with edge
    // replication for the bottom-right partial macroblocks if any).
    let mut blocks = [[0i32; 64]; 6];
    for blk in 0..6 {
        load_block_samples(v, width, height, mb_x, mb_y, blk, &mut blocks[blk]);
    }

    // Forward DCT each block (no level shift — MPEG-4 stores DC directly in
    // pel domain).
    let mut dct = [[0i32; 64]; 6];
    for blk in 0..6 {
        let mut f = [0.0f32; 64];
        for i in 0..64 {
            f[i] = blocks[blk][i] as f32;
        }
        fdct8x8(&mut f);
        for i in 0..64 {
            dct[blk][i] = f[i].round() as i32;
        }
    }

    // Quantise each block to (DC_units, AC_levels) and reconstruct in parallel
    // so the decoder side will see the same DC predictor neighbour values.
    // For each block:
    //   * DC: dc_units = (dct[0] + scale/2) / scale (matching decoder
    //     formula `recon_pel = recon_units * scale`).
    //   * AC: level[i] = round(coef / (2*Q)). H.263 dequant is
    //     `2*Q*|l| + Q_plus * sign`, so encoding by halving by `2Q` and
    //     rounding gives the closest level.
    let mut dc_units = [0i32; 6];
    let mut ac_levels = [[0i32; 64]; 6];
    for blk in 0..6 {
        let scale = dc_scaler(blk, quant) as i32;
        // Quantise DC. Round-to-nearest.
        let dc_pel = dct[blk][0];
        let dc_q = round_div(dc_pel, scale).clamp(-2048, 2047);
        dc_units[blk] = dc_q;
        // Quantise ACs.  H.263 dequant for intra is
        //   recon(l != 0) = (2*Q*|l| + Q_plus) * sign(l)
        //   recon(0)      = 0
        // where Q_plus = Q if Q is odd, Q-1 if Q is even (§7.4.4.2). The
        // forward step picks the level whose reconstruction is closest to
        // the input coefficient.
        for i in 1..64 {
            let l = quantise_ac_intra_h263(dct[blk][i], quant as i32).clamp(-2047, 2047);
            ac_levels[blk][i] = l;
        }
    }

    // Compute MCBPC + CBPY from the AC-coded flags.
    let mut luma_coded = [false; 4];
    let mut chroma_coded = [false; 2];
    for blk in 0..4 {
        luma_coded[blk] = ac_levels[blk][1..64].iter().any(|&v| v != 0);
    }
    chroma_coded[0] = ac_levels[4][1..64].iter().any(|&v| v != 0);
    chroma_coded[1] = ac_levels[5][1..64].iter().any(|&v| v != 0);
    // cbpc bits: bit1 = Cb, bit0 = Cr.
    let cbpc = ((chroma_coded[0] as u8) << 1) | (chroma_coded[1] as u8);
    // cbpy bits: bit3..bit0 = Y0..Y3.
    let mut cbpy: u8 = 0;
    for (i, &c) in luma_coded.iter().enumerate() {
        if c {
            cbpy |= 1 << (3 - i);
        }
    }

    // MCBPC: I-VOP uses Table B-10 (mcbpc value = cbpc, no IntraQ — quant
    // is constant across the picture). P-VOP intra uses Table B-13 rows
    // 4..=7 (Intra group); the caller already emitted the `not_coded = 0`
    // bit so the next field on the wire is the MCBPC VLC.
    match mcbpc_kind {
        IntraMcbpcKind::IVop => write_mcbpc_intra(bw, cbpc),
        IntraMcbpcKind::PVop => write_mcbpc_p_intra(bw, cbpc),
    }
    // ac_pred_flag = 0 (we never emit AC predictions).
    bw.write_bits(0, 1);
    // CBPY (decoder uses the raw value directly for intra MBs — see mb.rs
    // for I-VOP, inter.rs `decode_intra_blocks_in_p` for P-VOP).
    write_cbpy(bw, cbpy);

    // For each block: emit DC VLC + sign + residual + (if AC coded) AC walk.
    for blk in 0..6 {
        // Predicted DC: gradient direction over the neighbour grid.
        let (left, top_left, top) = lookup_neighbour_dcs(blk, mb_x, mb_y, grid);
        let (predicted_dc_pel, _dir) = choose_dc_predictor(left, top_left, top);
        let scale = dc_scaler(blk, quant) as i32;
        let pred_units = (predicted_dc_pel + scale / 2) / scale;
        let dc_diff = dc_units[blk] - pred_units;

        write_intra_dc_diff(bw, blk, dc_diff);

        let coded = if blk < 4 {
            luma_coded[blk]
        } else {
            chroma_coded[blk - 4]
        };
        if coded {
            // Emit AC tcoef walk (zigzag scan; ac_pred=0 → default scan).
            write_intra_ac(bw, &ac_levels[blk])?;
        }

        // Update neighbour grid with the *reconstructed* DC (pel domain) so
        // future MBs predict from the same DC the decoder will see.
        let recon_dc = (dc_units[blk] * scale).clamp(0, 2047);
        update_neighbour(grid, blk, mb_x, mb_y, recon_dc, quant as u8);

        // Optionally reconstruct the 8×8 block into `pic`. Mirrors the
        // decoder's reconstruct_intra_block path: dequantise the ACs under
        // the H.263 rule, install the reconstructed pel-domain DC, IDCT,
        // and clip to u8.
        if let Some(pic_mut) = pic.as_deref_mut() {
            let mut coeffs = ac_levels[blk];
            // H.263 intra dequant matches `iq::dequantise_intra_h263`.
            let q = quant as i32;
            let q_plus = if q & 1 == 1 { q } else { q - 1 };
            for i in 1..64 {
                let l = coeffs[i];
                if l == 0 {
                    continue;
                }
                let abs = l.abs();
                let mut val = 2 * q * abs + q_plus;
                if l < 0 {
                    val = -val;
                }
                coeffs[i] = val.clamp(-2048, 2047);
            }
            coeffs[0] = recon_dc.clamp(-2048, 2047);
            let mut f = [0.0f32; 64];
            for i in 0..64 {
                f[i] = coeffs[i] as f32;
            }
            crate::block::idct8x8(&mut f);
            write_recon_to_picture(pic_mut, blk, mb_x, mb_y, &f);
        }
    }

    let _ = Y_DC_SCALE_TABLE;
    Ok(())
}

pub(crate) fn write_recon_to_picture(
    pic: &mut IVopPicture,
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    samples: &[f32; 64],
) {
    let (plane, stride, px, py) = match blk {
        0 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16, mb_y * 16),
        1 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16 + 8, mb_y * 16),
        2 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16, mb_y * 16 + 8),
        3 => (
            pic.y.as_mut_slice(),
            pic.y_stride,
            mb_x * 16 + 8,
            mb_y * 16 + 8,
        ),
        4 => (pic.cb.as_mut_slice(), pic.c_stride, mb_x * 8, mb_y * 8),
        5 => (pic.cr.as_mut_slice(), pic.c_stride, mb_x * 8, mb_y * 8),
        _ => unreachable!(),
    };
    for j in 0..8 {
        for i in 0..8 {
            let v = samples[j * 8 + i].round() as i32;
            plane[(py + j) * stride + (px + i)] = v.clamp(0, 255) as u8;
        }
    }
}

// -------------------------------------------------------------------------
// Sample fetch + neighbour-grid bookkeeping
// -------------------------------------------------------------------------

pub(crate) fn load_block_samples(
    v: &VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    blk: usize,
    out: &mut [i32; 64],
) {
    let (plane_idx, x0, y0, pw, ph) = block_pel_position(width, height, mb_x, mb_y, blk);
    let p = &v.planes[plane_idx];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            out[j * 8 + i] = p.data[yy * p.stride + xx] as i32;
        }
    }
}

pub(crate) fn block_pel_position(
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    blk: usize,
) -> (usize, usize, usize, usize, usize) {
    let w = width;
    let h = height;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    match blk {
        0 => (0, mb_x * 16, mb_y * 16, w, h),
        1 => (0, mb_x * 16 + 8, mb_y * 16, w, h),
        2 => (0, mb_x * 16, mb_y * 16 + 8, w, h),
        3 => (0, mb_x * 16 + 8, mb_y * 16 + 8, w, h),
        4 => (1, mb_x * 8, mb_y * 8, cw, ch),
        5 => (2, mb_x * 8, mb_y * 8, cw, ch),
        _ => unreachable!(),
    }
}

pub(crate) fn lookup_neighbour_dcs(
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    grid: &PredGrid,
) -> (i32, i32, i32) {
    let (plane, bx, by, stride) = block_grid_position(blk, mb_x, mb_y, grid);
    let read = |px: isize, py: isize| -> i32 {
        if px < 0 || py < 0 {
            return 1024;
        }
        let rows = plane.len() / stride;
        if (px as usize) >= stride || (py as usize) >= rows {
            return 1024;
        }
        let nbr = &plane[(py as usize) * stride + (px as usize)];
        if nbr.is_intra {
            nbr.dc
        } else {
            1024
        }
    };
    let left = read(bx as isize - 1, by as isize);
    let top = read(bx as isize, by as isize - 1);
    let top_left = read(bx as isize - 1, by as isize - 1);
    (left, top_left, top)
}

fn block_grid_position(
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    grid: &PredGrid,
) -> (&[BlockNeighbour], usize, usize, usize) {
    match blk {
        0 => (&grid.y, mb_x * 2, mb_y * 2, grid.y_stride),
        1 => (&grid.y, mb_x * 2 + 1, mb_y * 2, grid.y_stride),
        2 => (&grid.y, mb_x * 2, mb_y * 2 + 1, grid.y_stride),
        3 => (&grid.y, mb_x * 2 + 1, mb_y * 2 + 1, grid.y_stride),
        4 => (&grid.cb, mb_x, mb_y, grid.c_stride),
        5 => (&grid.cr, mb_x, mb_y, grid.c_stride),
        _ => unreachable!(),
    }
}

pub(crate) fn update_neighbour(
    grid: &mut PredGrid,
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    dc_pel: i32,
    quant: u8,
) {
    let (bx, by, stride) = match blk {
        0 => (mb_x * 2, mb_y * 2, grid.y_stride),
        1 => (mb_x * 2 + 1, mb_y * 2, grid.y_stride),
        2 => (mb_x * 2, mb_y * 2 + 1, grid.y_stride),
        3 => (mb_x * 2 + 1, mb_y * 2 + 1, grid.y_stride),
        4 => (mb_x, mb_y, grid.c_stride),
        5 => (mb_x, mb_y, grid.c_stride),
        _ => unreachable!(),
    };
    let plane: &mut [BlockNeighbour] = match blk {
        0..=3 => &mut grid.y,
        4 => &mut grid.cb,
        5 => &mut grid.cr,
        _ => unreachable!(),
    };
    let n = &mut plane[by * stride + bx];
    n.dc = dc_pel;
    n.quant = quant;
    n.is_intra = true;
    // ACs left at zero (we set ac_pred=0 so the decoder won't read them; but
    // keep zero so any future code path is benign).
    for i in 0..7 {
        n.ac_top_row[i] = 0;
        n.ac_left_col[i] = 0;
    }
}

// -------------------------------------------------------------------------
// MCBPC + CBPY emit (Tables B-10, B-9)
// -------------------------------------------------------------------------

pub(crate) fn write_mcbpc_intra(bw: &mut BitWriter, cbpc: u8) {
    // Table B-10 — Intra MCBPC values 0..=3 (cbpc).
    let (bits, code) = match cbpc {
        0 => (1, 0b1),
        1 => (3, 0b001),
        2 => (3, 0b010),
        3 => (3, 0b011),
        _ => unreachable!("cbpc out of range: {cbpc}"),
    };
    bw.write_bits(code, bits);
}

/// Table B-13 rows 4..=7 — Intra MB inside a P-VOP, indexed by `cbpc`.
/// The decoder's `decompose_inter` maps this to `(PMbType::Intra, cbpc)`.
/// Codewords mirror the table in `tables/mcbpc.rs::P_ROWS`.
pub(crate) fn write_mcbpc_p_intra(bw: &mut BitWriter, cbpc: u8) {
    let (bits, code) = match cbpc {
        0 => (5, 0b00011),
        1 => (8, 0b00000100),
        2 => (8, 0b00000011),
        3 => (7, 0b0000011),
        _ => unreachable!("cbpc out of range: {cbpc}"),
    };
    bw.write_bits(code, bits);
}

pub(crate) fn write_cbpy(bw: &mut BitWriter, cbpy: u8) {
    // Table B-9 raw values (mirrors the decoder table in tables/cbpy.rs).
    let (bits, code) = match cbpy {
        0 => (4, 0b0011),
        1 => (5, 0b00101),
        2 => (5, 0b00100),
        3 => (4, 0b1001),
        4 => (5, 0b00011),
        5 => (4, 0b0111),
        6 => (6, 0b000010),
        7 => (4, 0b1011),
        8 => (5, 0b00010),
        9 => (6, 0b000011),
        10 => (4, 0b0101),
        11 => (4, 0b1010),
        12 => (4, 0b0100),
        13 => (4, 0b1000),
        14 => (4, 0b0110),
        15 => (2, 0b11),
        _ => unreachable!("cbpy out of range: {cbpy}"),
    };
    bw.write_bits(code, bits);
}

// -------------------------------------------------------------------------
// Intra DC VLC encode (Tables B-12 / B-13)
// -------------------------------------------------------------------------

pub(crate) fn write_intra_dc_diff(bw: &mut BitWriter, block_idx: usize, diff: i32) {
    let (size_codes, size_bits) = if block_idx < 4 {
        // Luma
        (
            [3u32, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_slice(),
            [3u8, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11].as_slice(),
        )
    } else {
        (
            [3u32, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_slice(),
            [2u8, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_slice(),
        )
    };
    let abs = diff.unsigned_abs();
    let size = if abs == 0 {
        0
    } else {
        // size = number of bits needed to hold |diff| in unsigned form.
        32 - abs.leading_zeros()
    };
    let size = size.min(12) as usize;
    bw.write_bits(size_codes[size], size_bits[size] as u32);
    if size == 0 {
        return;
    }
    // Bits encoding: positive → raw `diff` in `size` low bits with the MSB
    // (sign bit) set. Negative → `(2^size - 1) - |diff|` packed in `size`
    // bits — equivalently, bitwise NOT of (|diff|) within `size` bits.
    let raw: u32 = if diff > 0 {
        diff as u32
    } else {
        let mask = (1u32 << size) - 1;
        (!(abs)) & mask
    };
    bw.write_bits(raw, size as u32);
    if size > 8 {
        bw.write_bits(1, 1); // marker bit per §6.3.8
    }
}

// -------------------------------------------------------------------------
// Intra AC tcoef encode (Table B-16 + 3rd escape)
// -------------------------------------------------------------------------

/// Walk `block` in zigzag order, emitting one VLC per non-zero coefficient.
/// `block` is in natural order (block[ZIGZAG[i]] is scan position i).
///
/// Exposed publicly so the round-24 RVLC error-recovery test can build
/// matched RVLC and standard-Tcoef AC partitions for the same coefficient
/// stream and compare each path's resilience to bit-level damage.
pub fn write_intra_ac(bw: &mut BitWriter, block: &[i32; 64]) -> Result<()> {
    // Find the last non-zero AC scan index (we encode AC starting at scan 1).
    let mut last_nz: Option<usize> = None;
    for i in 1..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        // CBPY claimed this block was coded but no ACs remain — defensive
        // path. Encode a single (last=1, run=0, level=±1) using third escape
        // with level=1 to keep the bitstream parseable. In practice we skip
        // this branch because the caller checks `coded` from the level
        // array.
        return Err(Error::other(
            "mpeg4 encoder: AC walk requested but block is all zero",
        ));
    };
    let mut run = 0u8;
    let mut i = 1;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        write_intra_tcoef_symbol(bw, last, run, lv);
        run = 0;
        i += 1;
    }
    Ok(())
}

/// Encode one (last, run, level) intra tcoef symbol — short VLC where
/// possible, otherwise third escape.
fn write_intra_tcoef_symbol(bw: &mut BitWriter, last: bool, run: u8, level: i32) {
    let abs = level.unsigned_abs() as u8;
    if let Some((bits, code)) = lookup_intra_short_vlc(last, run, abs) {
        bw.write_bits(code, bits as u32);
        // Sign: 0 = positive, 1 = negative.
        bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
        return;
    }
    // Third escape (§6.3.8 escape mode 3):
    //   `0000011` (escape prefix, 7 bits) +
    //   `1` (marker for "not 1st mode") +
    //   `1` (marker for "not 2nd mode") +
    //   last (1 bit) + run (6 bits) + marker(1) + level (12 signed) + marker(1).
    bw.write_bits(0b0000011, 7);
    bw.write_bits(1, 1);
    bw.write_bits(1, 1);
    bw.write_bits(if last { 1 } else { 0 }, 1);
    bw.write_bits(run as u32 & 0x3F, 6);
    bw.write_bits(1, 1); // marker
    let lvl12 = (level & 0x0FFF) as u32;
    bw.write_bits(lvl12, 12);
    bw.write_bits(1, 1); // marker
}

/// Cached `(last, run, level_abs)` → `(bits, code)` for the intra tcoef
/// short-VLC reverse lookup.
type IntraShortVlcMap = HashMap<(bool, u8, u8), (u8, u32)>;

/// Reverse-lookup of the short VLC in Table B-16 keyed by `(last, run, abs)`.
fn lookup_intra_short_vlc(last: bool, run: u8, level_abs: u8) -> Option<(u8, u32)> {
    use std::sync::OnceLock;
    static MAP: OnceLock<IntraShortVlcMap> = OnceLock::new();
    let m = MAP.get_or_init(build_intra_short_vlc_map);
    m.get(&(last, run, level_abs)).copied()
}

fn build_intra_short_vlc_map() -> IntraShortVlcMap {
    let mut m = HashMap::new();
    for entry in tcoef::intra_table() {
        if let tcoef::TcoefSym::RunLevel {
            last,
            run,
            level_abs,
        } = entry.value
        {
            m.insert((last, run, level_abs), (entry.bits, entry.code));
        }
    }
    m
}

// -------------------------------------------------------------------------
// FDCT + small numeric helpers
// -------------------------------------------------------------------------

/// Round-to-nearest division of `a` by `b` (with sign).
pub(crate) fn round_div(a: i32, b: i32) -> i32 {
    debug_assert!(b > 0);
    if a >= 0 {
        (a + b / 2) / b
    } else {
        -(((-a) + b / 2) / b)
    }
}

/// Quantise one intra-AC coefficient under the H.263 dequant rule.
///
/// Reconstructions are at `0` and `±(2*Q*l + Q_plus)` for `l ≥ 1`, where
/// `Q_plus = Q | 1` (i.e. `Q` if odd, `Q-1` if even). This routine picks the
/// integer level whose reconstruction is closest to `coef`. Ties prefer the
/// lower-magnitude level (cheaper to code).
pub(crate) fn quantise_ac_intra_h263(coef: i32, q: i32) -> i32 {
    if coef == 0 || q <= 0 {
        return 0;
    }
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let two_q = 2 * q;
    let abs = coef.unsigned_abs() as i32;
    // Coarse seed: floor(|coef| / (2*Q)). Compare candidates `l` and `l+1`
    // against the input. `l = 0` is the deadzone choice (recon = 0).
    let l_low = abs / two_q;
    // Compare three candidates: l_low, l_low + 1, and (if l_low > 0) l_low - 1.
    let mut best_l = 0i32;
    let mut best_err = abs;
    let consider = |l: i32| {
        if l < 0 {
            return None;
        }
        let recon = if l == 0 { 0 } else { two_q * l + q_plus };
        Some((l, (abs - recon).abs()))
    };
    for cand in [l_low.saturating_sub(1), l_low, l_low + 1] {
        if let Some((l, e)) = consider(cand) {
            if e < best_err {
                best_err = e;
                best_l = l;
            }
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

/// Forward 8×8 DCT — built by inverting the IDCT used by the decoder so the
/// two are bit-exact inverses (within float rounding). The DCT matrix is
/// orthonormal with our normalisation, so `FDCT == IDCT^T == IDCT` when
/// applied as a matrix on the left (rows). Using the same `idct8x8` for the
/// forward transform via an explicit transpose path gives a self-inverse
/// transform under the same f32 cosine table.
///
/// Dispatches to the compile-time selected kernel in [`crate::simd`].
#[inline]
pub fn fdct8x8(block: &mut [f32; 64]) {
    crate::simd::fdct8x8(block);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdct_idct_round_trip() {
        use crate::block::idct8x8;
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = ((i * 7) % 255) as f32;
        }
        let original = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < 1e-2,
                "round-trip mismatch at {i}: got {} want {}",
                block[i],
                original[i]
            );
        }
    }

    #[test]
    fn dc_size_round_trip_luma() {
        // For a few representative diff values, encode+decode and compare.
        use crate::block::decode_intra_dc_diff;
        use oxideav_core::bits::BitReader;
        for &diff in &[0i32, 1, -1, 5, -5, 127, -127, 2047, -2047] {
            let mut bw = BitWriter::new();
            write_intra_dc_diff(&mut bw, 0, diff);
            // The reader peeks 16 bits so we need at least 2 bytes worth of input.
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let got = decode_intra_dc_diff(&mut br, 0).unwrap();
            assert_eq!(got, diff, "luma DC round-trip failed for {diff}");
        }
    }

    #[test]
    fn dc_size_round_trip_chroma() {
        use crate::block::decode_intra_dc_diff;
        use oxideav_core::bits::BitReader;
        for &diff in &[0i32, 3, -3, 200, -200, 1000, -1000] {
            let mut bw = BitWriter::new();
            write_intra_dc_diff(&mut bw, 4, diff);
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let got = decode_intra_dc_diff(&mut br, 4).unwrap();
            assert_eq!(got, diff, "chroma DC round-trip failed for {diff}");
        }
    }

    /// Verify the P-VOP intra MCBPC writer (Table B-13 rows 4..=7) round-trips
    /// through the decoder's `decompose_inter` to `(PMbType::Intra, cbpc)`.
    /// Catches any codeword swap with the I-VOP table or the Intra4MV rows.
    #[test]
    fn mcbpc_p_intra_roundtrip() {
        use crate::tables::mcbpc::{decompose_inter, p_table, PMbType};
        use crate::tables::vlc;
        use oxideav_core::bits::BitReader;
        for cbpc in 0..4u8 {
            let mut bw = BitWriter::new();
            write_mcbpc_p_intra(&mut bw, cbpc);
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let v = vlc::decode(&mut br, p_table()).unwrap();
            let (mb_type, dec_cbpc) = decompose_inter(v);
            assert_eq!(
                mb_type,
                PMbType::Intra,
                "cbpc={cbpc} decoded as {mb_type:?} not Intra"
            );
            assert_eq!(dec_cbpc, cbpc, "cbpc round-trip mismatch for {cbpc}");
        }
    }
}

//! Rectangular progressive **S(GMC)-VOP encoder** — global motion
//! compensation with one §7.8.4 warping point at half-pel accuracy.
//!
//! An S(GMC)-VOP is the P-VOP macroblock layer plus the §6.3.6 `mcsel`
//! flag: each inter macroblock predicts either from the §7.8.7.1
//! GMC-warped reference (`mcsel == 1`, no `motion_vector()` bodies) or
//! through the plain §7.6 local-MC path (`mcsel == 0`; inter4v never
//! codes `mcsel`). A `not_coded` S macroblock is a GMC copy (implied
//! `mcsel == 1`, zero residual) — *not* the P-VOP zero-MV copy.
//!
//! The encoder:
//!
//! * estimates the **global translation** as the most common per-MB
//!   §7.6 motion vector (the mode of the estimate field), emitted as
//!   the §6.2.5 `sprite_trajectory()` `(du[0], dv[0])` pair in
//!   half-sample units (`i0' = (s/2)·du[0]` — one warping point is a
//!   pure translation of `du/2` pels whatever `s` is);
//! * scores, per macroblock, the GMC prediction (built by the
//!   decoder's own [`gmc_prediction_macroblock`]) against the local
//!   §7.6 candidates and the intra activity, with a small bias toward
//!   GMC (an `mcsel == 1` macroblock spends no motion bits);
//! * mirrors the decoder's predictor bookkeeping: a GMC macroblock
//!   records the §7.8.7.3 **averaged motion vector** into the shared
//!   [`MvGrid`] exactly as `MvDriver::record_gmc_macroblock` does.
//!
//! The emitted unit is decoded back through
//! [`crate::vop_decode::decode_s_gmc_vop_macroblocks`] +
//! [`assemble_s_gmc_vop_frame`] — the same closed loop as the I/P/B
//! encoders.
//!
//! Provenance: §6.2.5 (`sprite_trajectory()`, `warping_mv_code()`),
//! §6.2.6/§6.3.6 (`mcsel`, S(GMC) `not_coded`), §7.8.4/§7.8.7 (warp
//! geometry, GMC prediction, averaged MV) of ISO/IEC 14496-2:2004
//! (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! via the crate's decoder transcriptions.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::nonintra_quant_matrix;
use crate::data_partition::use_intra_dc_vlc;
use crate::framestore::{DecodedFrame, FrameStore};
use crate::ivop_encode::{EncoderConfig, FrameView};
use crate::motion::{predict_motion_vector, MotionVector};
use crate::mv_predictor_grid::MvGrid;
use crate::neighbour::IntraBlockGrid;
use crate::packet_encode::{Layout, MbFields, PacketVopInfo, PacketWriter};
use crate::pvop_encode::{
    estimate_motion, intra_activity, intra_mb_in_p_fields, quantise_inter_block, sample_mode_of,
    source_luma_mb, PVopEncodeStats,
};
use crate::pvop_mv::PvopMbMotion;
use crate::s_gmc_recon::{gmc_prediction_macroblock, GmcReferencePlanes};
use crate::sprite::SpriteTrajectory;
use crate::texture::AcEvent;
use crate::vol::SpriteWarpingAccuracy;
use crate::vop::{parse_vop_header_body, vop_time_increment_bits, VopCodingType, VopContext};
use crate::vop_decode::gmc_averaged_mv;
use crate::warp::WarpGeometry;

const VOP_START_CODE: u32 = 0x0000_01B6;

/// Emit one §6.2.5 `warping_mv_code()`: the Table B.34 `dmv_length`
/// VLC (`00` → 0, `010`..`110` → 1..=5, then `SSS − 3` one-bits and a
/// `0` for 6..=14), the `SSS`-bit `dmv_code` when `SSS != 0`, and the
/// trailing `marker_bit`. Exact inverse of
/// [`crate::sprite::decode_warping_mv_code`].
///
/// # Panics
///
/// Panics when `|dmv|` exceeds the Table B.34 range (`2^14 - 1`).
pub fn put_warping_mv_code(bw: &mut BitWriter, dmv: i32) {
    let magnitude = dmv.unsigned_abs();
    assert!(magnitude < (1 << 14), "warping dmv {dmv} out of range");
    if dmv == 0 {
        bw.write_bits(0b00, 2); // dmv_length = 0
        bw.write_marker();
        return;
    }
    let sss = 32 - magnitude.leading_zeros(); // |dmv| in [2^(SSS-1), 2^SSS - 1]
    match sss {
        1..=5 => bw.write_bits(sss + 1, 3), // 010, 011, 100, 101, 110
        _ => {
            for _ in 0..(sss - 3) {
                bw.write_bit(true);
            }
            bw.write_bit(false);
        }
    }
    let span = (1i64 << sss) - 1;
    let code = if dmv > 0 {
        i64::from(dmv)
    } else {
        i64::from(dmv) + span
    };
    bw.write_bits(code as u32, sss as usize);
    bw.write_marker();
}

/// Emit a §6.2.5 S(GMC)-VOP header through `vop_fcode_forward`,
/// including the one-point `sprite_trajectory()`. The writer is left
/// mid-unit — the macroblock walk follows.
#[allow(clippy::too_many_arguments)]
pub fn write_s_vop_header(
    bw: &mut BitWriter,
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    quant: u32,
    fcode: u8,
    du: i32,
    dv: i32,
) {
    bw.write_start_code(VOP_START_CODE);
    bw.write_bits(0b11, 2); // vop_coding_type = S
    for _ in 0..modulo_time_base {
        bw.write_bit(true);
    }
    bw.write_bit(false);
    bw.write_marker();
    bw.write_bits(
        u32::from(time_increment),
        usize::from(vop_time_increment_bits(resolution)),
    );
    bw.write_marker();
    bw.write_bit(true); // vop_coded = 1
    bw.write_bit(false); // vop_rounding_type = 0 (S(GMC) carries it like P)
    bw.write_bits(0, 3); // intra_dc_vlc_thr = 0
                         // sprite_trajectory(): one warping point.
    put_warping_mv_code(bw, du);
    put_warping_mv_code(bw, dv);
    assert!((1..=31).contains(&quant), "vop_quant {quant} out of range");
    bw.write_bits(quant, 5);
    assert!((1..=7).contains(&fcode), "vop_fcode_forward out of range");
    bw.write_bits(u32::from(fcode), 3);
}

/// Per-VOP S(GMC) encode statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SVopEncodeStats {
    /// `not_coded` GMC copies (implied `mcsel == 1`, zero residual).
    pub gmc_skipped: usize,
    /// Coded `mcsel == 1` macroblocks.
    pub gmc: usize,
    /// `mcsel == 0` local-MC macroblocks (1-MV).
    pub local: usize,
    /// inter4v macroblocks (no `mcsel` in the syntax).
    pub inter4v: usize,
    /// Intra macroblocks.
    pub intra: usize,
    /// Macroblocks that carried a `dquant`.
    pub dquant: usize,
    /// Video packets cut inside the VOP.
    pub packets: usize,
    /// The emitted trajectory `(du[0], dv[0])` in half-sample units.
    pub trajectory: (i32, i32),
}

/// The most common estimated motion vector across the macroblock grid
/// — the global translation candidate, in the VOL's MV units.
fn dominant_motion(mvs: &[(MotionVector, u32)]) -> MotionVector {
    let mut best = MotionVector { x: 0, y: 0 };
    let mut best_count = 0usize;
    for (i, &(mv, _)) in mvs.iter().enumerate() {
        let count = mvs.iter().filter(|&&(m, _)| m == mv).count();
        if count > best_count || (count == best_count && i == 0) {
            best_count = count;
            best = mv;
        }
    }
    best
}

/// SAD of the source macroblock against an arbitrary prediction.
fn sad_against(src: &[[i32; 16]; 16], pred: &crate::reconstruct::InterPredictionMacroblock) -> u32 {
    let mut sad = 0u32;
    for (j, row) in src.iter().enumerate() {
        for (i, &s) in row.iter().enumerate() {
            sad += (s - pred.luma[j][i]).unsigned_abs();
        }
    }
    sad
}

/// Encode one rectangular progressive S(GMC)-VOP against `reference`
/// (the closed-loop reconstruction of the previous anchor). Returns
/// the emitted unit and the mode statistics.
pub fn encode_s_vop(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    reference: &DecodedFrame,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
) -> (Vec<u8>, SVopEncodeStats) {
    assert!((1..=31).contains(&qp), "vop_quant {qp} out of range");
    assert!(cfg.gmc, "encode_s_vop needs a GMC VOL");
    assert!(!cfg.interlaced, "S(GMC)-VOPs are progressive-only");
    let (mb_width, mb_height) = cfg.mb_dimensions();
    let w_intra = crate::block::intra_quant_matrix(vol);
    let w_inter = nonintra_quant_matrix(vol);
    let use_dc_vlc = use_intra_dc_vlc(0, qp);
    let mode = sample_mode_of(vol);
    let fcode = cfg.fcode;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();
    let gmc_planes = GmcReferencePlanes {
        luma: reference.luma_reference(),
        cb: reference.cb_reference(),
        cr: reference.cr_reference(),
    };

    // ---- Pass 1: per-MB local motion estimates + the global mode ----
    let mut local_mvs: Vec<(MotionVector, u32)> = Vec::with_capacity(mb_width * mb_height);
    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);
            local_mvs.push(estimate_motion(&src, &luma_ref, mb_x, mb_y, mode, fcode));
        }
    }
    let global = dominant_motion(&local_mvs);
    // §7.8.4: du[0]/dv[0] are half-sample units for any s (i0' =
    // (s/2)·du). Quarter-sample local MVs quantise to the half grid.
    // The trajectory is kept inside the Table 7-9 range for the VOP's
    // fcode so the §7.8.7.3 averaged-MV clip can never fire: for a
    // one-point warp the AMV *is* the translation, and a clipped AMV
    // would make every neighbour's §7.6.5 median depend on the exact
    // clip behaviour — the encoder simply never emits that corner.
    let (low, high) = crate::pvop_encode::mv_range(fcode);
    let (amv_low, amv_high) = if vol.quarter_sample {
        // AMV is derived in quarter-sample units (2·du).
        (low / 2, high / 2)
    } else {
        (low, high)
    };
    let (du, dv) = if vol.quarter_sample {
        (global.x / 2, global.y / 2)
    } else {
        (global.x, global.y)
    };
    let (du, dv) = (du.clamp(amv_low, amv_high), dv.clamp(amv_low, amv_high));
    let trajectory = SpriteTrajectory {
        count: 1,
        points: [[du, dv], [0, 0], [0, 0]],
    };
    let geometry = WarpGeometry::decode(
        &trajectory,
        u32::from(cfg.width),
        u32::from(cfg.height),
        SpriteWarpingAccuracy::HalfPel,
    );

    let mut bw = BitWriter::new();
    write_s_vop_header(
        &mut bw,
        cfg.time_increment_resolution,
        modulo_time_base,
        time_increment,
        qp,
        fcode,
        du,
        dv,
    );
    let mut pw = PacketWriter::new(
        bw,
        cfg.resilience,
        PacketVopInfo {
            coding_type: VopCodingType::S,
            fcode_fwd: fcode,
            fcode_bwd: 0,
            modulo_time_base,
            time_increment,
            time_increment_bits: vop_time_increment_bits(cfg.time_increment_resolution),
            intra_dc_vlc_thr: 0,
            total_macroblocks: (mb_width * mb_height) as u32,
            interlaced: false,
        },
        Layout::Combined,
    );

    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut mv_grid = MvGrid::new(mb_height, mb_width);
    let mut stats = SVopEncodeStats {
        trajectory: (du, dv),
        ..Default::default()
    };
    let vop_qp = qp;
    let mut running_qp = vop_qp;

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let idx = mb_row * mb_width + mb_col;
            if pw.maybe_cut(idx, running_qp) {
                intra_grid = IntraBlockGrid::new(mb_height, mb_width);
                mv_grid = MvGrid::new(mb_height, mb_width);
            }
            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);
            let activity = intra_activity(&src);
            let plan_quant = |running: u32| -> (u32, Option<i8>) {
                if cfg.adaptive_quant {
                    let class = crate::mb_quant::activity_class(activity);
                    crate::mb_quant::plan_dquant(running, crate::mb_quant::target_qp(vop_qp, class))
                } else {
                    (running, None)
                }
            };

            // GMC candidate: the decoder's own §7.8.7.1 prediction.
            let gmc_pred = gmc_prediction_macroblock(
                &geometry,
                &gmc_planes,
                i64::from(mb_x),
                i64::from(mb_y),
                0,
                8,
            );
            let gmc_sad = sad_against(&src, &gmc_pred);
            let (local_mv, local_sad) = local_mvs[idx];

            // Mode decision: GMC saves the motion_vector() body, so it
            // gets a small preference; intra wins on flat-vs-motion
            // activity exactly as in the P walk.
            let choose_gmc = gmc_sad <= local_sad.saturating_add(64);
            let inter_sad = gmc_sad.min(local_sad);
            let choose_intra = activity + 512 < inter_sad;

            if choose_intra {
                stats.intra += 1;
                let (qp, dquant) = plan_quant(running_qp);
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                let fields = intra_mb_in_p_fields(
                    &pw,
                    frame,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    qp,
                    cfg,
                    &w_intra,
                    use_dc_vlc,
                    dquant,
                );
                pw.push(&fields);
                continue;
            }

            let (qp, dquant) = plan_quant(running_qp);

            // Build the chosen prediction + quantise the residual.
            let prediction = if choose_gmc {
                gmc_pred
            } else {
                crate::pvop_mv::predict_inter_macroblock(
                    PvopMbMotion::OneMv(local_mv),
                    &luma_ref,
                    &cb_ref,
                    &cr_ref,
                    mb_x,
                    mb_y,
                    0,
                    mode,
                )
                .expect("inter motion always yields a prediction")
            };
            let mut events: Vec<Vec<AcEvent>> = Vec::with_capacity(6);
            for i in 0..6 {
                let src_block = frame.block(mb_row, mb_col, i);
                let mut residual = [[0i32; 8]; 8];
                for y in 0..8 {
                    for x in 0..8 {
                        let p = match i {
                            0..=3 => prediction.luma[y + 8 * (i / 2)][x + 8 * (i % 2)],
                            4 => prediction.cb[y][x],
                            _ => prediction.cr[y][x],
                        };
                        residual[y][x] = src_block[y][x] - p;
                    }
                }
                let (ev, _qf) = quantise_inter_block(
                    &residual,
                    qp,
                    cfg.quant_type,
                    &w_inter,
                    crate::scan::ScanType::Zigzag,
                );
                events.push(ev);
            }
            let all_zero = events.iter().all(|e| e.is_empty());

            if choose_gmc {
                // The decoder records the §7.8.7.3 averaged MV for the
                // neighbours' §7.6.5 medians.
                let amv = gmc_averaged_mv(
                    &geometry,
                    i64::from(mb_x),
                    i64::from(mb_y),
                    vol.quarter_sample,
                    fcode,
                    false, // the encoder mirrors the spec-literal decode
                )
                .expect("AMV derivation cannot fail on a valid fcode");
                mv_grid
                    .record_one_mv(mb_row, mb_col, amv)
                    .expect("grid coordinates in range");
                if all_zero {
                    // §6.3.6 S(GMC) not_coded: a GMC copy.
                    stats.gmc_skipped += 1;
                    pw.push(&MbFields {
                        not_coded: true,
                        mb_type: 0,
                        cbpc: 0,
                        cbpy: 0,
                        ac_pred_flag: false,
                        dquant: None,
                        mcsel: None,
                        mvds: Vec::new(),
                        fcode,
                        intra_dc: None,
                        blocks: Default::default(),
                        interlaced: None,
                    });
                    continue;
                }
                stats.gmc += 1;
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
            } else {
                stats.local += 1;
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
            }

            let cbpy = (u8::from(!events[0].is_empty()) << 3)
                | (u8::from(!events[1].is_empty()) << 2)
                | (u8::from(!events[2].is_empty()) << 1)
                | u8::from(!events[3].is_empty());
            let cbpc = (u8::from(!events[4].is_empty()) << 1) | u8::from(!events[5].is_empty());
            let mvds = if choose_gmc {
                Vec::new()
            } else {
                let candidates = mv_grid
                    .predictor_candidates(mb_row, mb_col, 0)
                    .expect("grid coordinates in range");
                let predictor = predict_motion_vector(candidates);
                mv_grid
                    .record_one_mv(mb_row, mb_col, local_mv)
                    .expect("grid coordinates in range");
                vec![(local_mv.x - predictor.x, local_mv.y - predictor.y)]
            };
            let blocks: [Vec<AcEvent>; 6] = events
                .try_into()
                .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
            pw.push(&MbFields {
                not_coded: false,
                mb_type: if dquant.is_some() { 1 } else { 0 },
                cbpc,
                cbpy,
                ac_pred_flag: false,
                dquant,
                mcsel: Some(choose_gmc),
                mvds,
                fcode,
                intra_dc: None,
                blocks,
                interlaced: None,
            });
        }
    }
    stats.packets = pw.packets_cut();
    (pw.finish(), stats)
}

/// Decode an emitted S(GMC)-VOP unit through the crate's decoder walk,
/// advance `store`'s anchor chain, and return the reconstruction plus
/// the per-macroblock motion the following B-VOPs' §7.6.9.5.1 /
/// §7.6.9.6 co-located substitution consumes (the same mapping the
/// stream decoder applies: a skipped GMC macroblock contributes its
/// averaged MV; a coded GMC or intra macroblock the zero-vector
/// fallback).
pub fn reconstruct_own_s_vop_with_motion(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &mut FrameStore,
) -> (DecodedFrame, Vec<PvopMbMotion>) {
    let (mb_width, mb_height) = (
        usize::from(vol.width).div_ceil(16),
        usize::from(vol.height).div_ceil(16),
    );
    let mut br = BitReader::new(unit);
    let sc = br.read_bits(32).expect("unit starts with a start code");
    assert_eq!(sc, VOP_START_CODE, "encoder emitted a malformed unit");
    let vop = parse_vop_header_body(
        &mut br,
        vol.time_increment_resolution,
        VopContext::from_vol(vol),
    )
    .expect("own S-VOP header must parse");
    assert!(matches!(vop.coding_type, VopCodingType::S));
    let (entries, geometry) = crate::vop_decode::decode_s_gmc_vop_macroblocks(
        &mut br,
        vol,
        &vop,
        crate::compat::DecodeOptions::spec(),
    )
    .expect("own S-VOP payload must decode");
    let motion = entries
        .iter()
        .map(|e| match e {
            crate::frame_decode::SGmcMbContent::Local { motion, .. } => *motion,
            crate::frame_decode::SGmcMbContent::Gmc {
                amv,
                not_coded: true,
                ..
            } => PvopMbMotion::OneMv(*amv),
            crate::frame_decode::SGmcMbContent::Gmc {
                not_coded: false, ..
            }
            | crate::frame_decode::SGmcMbContent::Intra(_) => PvopMbMotion::Intra,
        })
        .collect();
    let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
        store,
        mb_width,
        mb_height,
        &entries,
        &geometry,
        vop.rounding_type,
        sample_mode_of(vol),
        8,
    )
    .expect("own S-VOP must assemble")
    .clone();
    store.push_anchor(frame.clone());
    (frame, motion)
}

/// Statistics bridge so callers tracking [`PVopEncodeStats`]-shaped
/// numbers can fold an S-VOP in (skips = GMC copies, inter = GMC +
/// local).
pub fn as_p_stats(stats: &SVopEncodeStats) -> PVopEncodeStats {
    PVopEncodeStats {
        skipped: stats.gmc_skipped,
        inter: stats.gmc + stats.local,
        inter4v: stats.inter4v,
        field: 0,
        field_dct: 0,
        intra: stats.intra,
        dquant: stats.dquant,
        packets: stats.packets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::decode_warping_mv_code;

    #[test]
    fn warping_mv_codes_round_trip() {
        for v in (-16383..=16383).step_by(7).chain([-16383, -1, 0, 1, 16383]) {
            let mut bw = BitWriter::new();
            put_warping_mv_code(&mut bw, v);
            bw.next_start_code();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(decode_warping_mv_code(&mut br).unwrap(), v, "dmv {v}");
        }
    }

    #[test]
    fn s_vop_header_round_trips_with_trajectory() {
        let cfg = EncoderConfig {
            width: 64,
            height: 48,
            gmc: true,
            ..EncoderConfig::default()
        };
        let headers = crate::ivop_encode::write_configuration_headers(&cfg);
        let pos = headers
            .windows(4)
            .position(|w| w == [0, 0, 1, 0x20])
            .unwrap();
        let vol =
            crate::vol::parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
        assert!(matches!(vol.sprite_enable, crate::vol::SpriteEnable::Gmc));
        assert_eq!(vol.no_of_sprite_warping_points, Some(1));

        let mut bw = BitWriter::new();
        write_s_vop_header(&mut bw, 25, 1, 7, 9, 2, -6, 3);
        bw.next_start_code();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(32).unwrap(), VOP_START_CODE);
        let vop = parse_vop_header_body(&mut br, 25, VopContext::from_vol(&vol)).unwrap();
        assert!(matches!(vop.coding_type, VopCodingType::S));
        assert_eq!(vop.quant, 9);
        assert_eq!(vop.fcode_fwd, 2);
        let traj = vop.sprite_trajectory.expect("trajectory present");
        assert_eq!(traj.count, 1);
        assert_eq!(traj.points[0], [-6, 3]);
    }
}

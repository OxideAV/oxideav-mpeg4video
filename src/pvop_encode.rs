//! Rectangular progressive **P-VOP encoder** — §7.6 motion estimation
//! + MC emission over the I-VOP encoder's block machinery.
//!
//! Per macroblock the walk runs:
//!
//! * **motion estimation** — full-pel SAD search (±[`SEARCH_RANGE`]
//!   pels, §7.6.4 edge-clamped reference fetches) around the zero
//!   vector, then a §7.6.2.1 half-sample refinement over the eight
//!   half-pel neighbours using the decoder's own interpolator, all
//!   against the **closed-loop reference** (the decoder-walk
//!   reconstruction of the previous anchor);
//! * **mode decision** — intra when the macroblock's mean-removed
//!   activity undercuts the best inter SAD by the classic margin;
//!   `not_coded` (skip) when the zero-vector prediction leaves no
//!   quantised residual;
//! * **emission** — §6.2.6 P-VOP syntax: `not_coded` / `mcbpc`
//!   (Table B.7) / `cbpy` / `motion_vector()` (Table B.12 against the
//!   §7.6.5 median predictor over the same [`MvGrid`] state the
//!   decoder's `MvDriver` maintains) / §6.2.7 `block()` bodies
//!   (Table B.17 inter events; intra macroblocks reuse the I-VOP
//!   plan/emission path including cost-decided AC prediction).
//!
//! The §7.6.5 predictor, §7.6.3 differential wrap, quantisers and
//! VLCs are all the crate's own decode-side transcriptions (or their
//! exact inverses), so the emitted stream decodes bit-identically
//! through [`crate::vop_decode::decode_p_vop_macroblocks`] — which is
//! also how [`reconstruct_own_p_vop`] produces the next reference.
//!
//! Provenance: §6.2.5/§6.2.6/§6.2.7 syntax, §7.6.2–§7.6.5 motion
//! semantics from ISO/IEC 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! via the crate's decoder transcriptions. No third-party source was
//! consulted.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::nonintra_quant_matrix;
use crate::bvop_prediction::BVopSampleMode;
use crate::data_partition::use_intra_dc_vlc;
use crate::fdct::forward_dct_8x8;
use crate::framestore::{DecodedFrame, FrameStore};
use crate::ivop_encode::{
    emit_intra_blocks, forward_scan, intra_mb_cbp, plan_block, qfs_to_events, quantise_intra_block,
    BlockPlan, EncoderConfig, FrameView, PreparedBlock,
};
use crate::motion::{predict_motion_vector, MotionVector};
use crate::mv_predictor_grid::MvGrid;
use crate::neighbour::{BlockNeighbour, IntraBlockGrid};
use crate::predictor::select_dc_direction;
use crate::pvop_mv::{predict_inter_macroblock, PvopMbMotion};
use crate::quantise::{quantise_method1_inter, quantise_method2_inter};
use crate::scan::ScanType;
use crate::texture::{AcEvent, DcComponent, TcoefTable};
use crate::vlc_encode::{put_cbpy, put_mcbpc_p, put_motion_vector};
use crate::vop::{parse_vop_header_body, vop_time_increment_bits, VopCodingType, VopContext};
use crate::vop_decode::decode_p_vop_macroblocks;

/// Dense full-pel motion search radius in pels (each direction) —
/// the `fcode == 1` window, whose refined half-pel vector stays inside
/// the Table 7-9 range `[-32, 31]` (half-sample units). Wider `fcode`
/// windows extend this with a coarse-to-fine search
/// ([`search_window_pels`]).
pub const SEARCH_RANGE: i32 = 8;

/// The Table 7-9 motion-vector range `[low, high]` for `fcode`, in
/// the unitless MV integers of `mode` (half- or quarter-sample).
pub(crate) fn mv_range(fcode: u8) -> (i32, i32) {
    assert!((1..=7).contains(&fcode), "fcode {fcode} out of range");
    let f = 1i32 << (fcode - 1);
    (-32 * f, 32 * f - 1)
}

/// Full-pel search window radius for `fcode` under `mode`: the
/// largest full-pel displacement whose sub-pel refinement still lands
/// inside the Table 7-9 range — `16 << (fcode - 1)` pels in
/// half-sample mode, `8 << (fcode - 1)` in quarter-sample mode
/// (minus one so the positive side's sub-pel neighbours stay
/// representable).
pub(crate) fn search_window_pels(fcode: u8, mode: BVopSampleMode) -> i32 {
    let (_, high) = mv_range(fcode);
    (high + 1) / units_per_pel(mode) - 1
}

const VOP_START_CODE: u32 = 0x0000_01B6;

/// Emit a §6.2.5 P-VOP header (through `vop_fcode_forward`). The
/// writer is left mid-unit — the macroblock walk follows.
pub fn write_p_vop_header(
    bw: &mut BitWriter,
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    quant: u32,
    fcode: u8,
) {
    bw.write_start_code(VOP_START_CODE);
    bw.write_bits(0b01, 2); // vop_coding_type = P
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
    bw.write_bit(false); // vop_rounding_type = 0
    bw.write_bits(0, 3); // intra_dc_vlc_thr = 0
    assert!((1..=31).contains(&quant), "vop_quant {quant} out of range");
    bw.write_bits(quant, 5);
    assert!(
        (1..=7).contains(&fcode),
        "vop_fcode_forward {fcode} out of range"
    );
    bw.write_bits(u32::from(fcode), 3); // vop_fcode_forward
}

/// Per-VOP encode statistics (mode-decision observability for tests).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PVopEncodeStats {
    /// Macroblocks emitted as `not_coded` skips.
    pub skipped: usize,
    /// Inter (1-MV) macroblocks.
    pub inter: usize,
    /// inter4v (4-MV) macroblocks.
    pub inter4v: usize,
    /// Intra macroblocks.
    pub intra: usize,
    /// Macroblocks that carried a `dquant` (`inter+q` / `intra+q`).
    pub dquant: usize,
}

/// 16×16 source luma of one macroblock (edge-replicated), as rows.
pub(crate) fn source_luma_mb(
    frame: &FrameView<'_>,
    mb_row: usize,
    mb_col: usize,
) -> [[i32; 16]; 16] {
    let mut out = [[0i32; 16]; 16];
    for (i, &(row_off, col_off)) in [(0usize, 0usize), (0, 8), (8, 0), (8, 8)]
        .iter()
        .enumerate()
    {
        let block = frame.block(mb_row, mb_col, i);
        for y in 0..8 {
            for x in 0..8 {
                out[row_off + y][col_off + x] = block[y][x];
            }
        }
    }
    out
}

/// SAD of the source macroblock against a full-pel reference position.
fn sad_full_pel(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    dx: i32,
    dy: i32,
) -> u32 {
    let mut sad = 0u32;
    for (j, row) in src.iter().enumerate() {
        for (i, &s) in row.iter().enumerate() {
            let r = reference.fetch_clamped(mb_x + i as i32 + dx, mb_y + j as i32 + dy);
            sad += (s - i32::from(r)).unsigned_abs();
        }
    }
    sad
}

/// The number of MV units per pel under a sub-pel `mode` (§7.6.3:
/// half-sample units when `quarter_sample == 0`, quarter-sample units
/// otherwise).
fn units_per_pel(mode: BVopSampleMode) -> i32 {
    match mode {
        BVopSampleMode::HalfPel => 2,
        BVopSampleMode::QuarterPel { .. } => 4,
    }
}

/// SAD of the source macroblock against a sub-pel motion vector (in
/// `mode`'s units), through the decoder's own §7.6.2.1 / §7.6.2.2
/// interpolator.
fn sad_subpel(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    mv: MotionVector,
    mode: BVopSampleMode,
) -> u32 {
    let pred = crate::pvop_mv::predict_luma_macroblock(
        PvopMbMotion::OneMv(mv),
        reference,
        mb_x,
        mb_y,
        0,
        mode,
    )
    .expect("OneMv always yields a prediction");
    let mut sad = 0u32;
    for (j, row) in src.iter().enumerate() {
        for (i, &s) in row.iter().enumerate() {
            sad += (s - i32::from(pred[j * 16 + i])).unsigned_abs();
        }
    }
    sad
}

/// The full-pel window the estimators actually walk: the dense
/// ±[`SEARCH_RANGE`] under `fcode == 1`, the Table 7-9 window
/// ([`search_window_pels`]) beyond it.
fn full_pel_window(fcode: u8, mode: BVopSampleMode) -> i32 {
    if fcode == 1 {
        SEARCH_RANGE
    } else {
        search_window_pels(fcode, mode).max(SEARCH_RANGE)
    }
}

/// Sub-pel refinement around a full-pel winner: a half-pel ring pass,
/// then (in quarter mode) a quarter-pel ring pass, each keeping the
/// running best. The `[low, high]` clamp is the §7.6.3 / Table 7-9
/// range for `fcode` in `mode`'s units.
fn refine_subpel<F: FnMut(MotionVector) -> u32>(
    full: (i32, i32),
    mode: BVopSampleMode,
    fcode: u8,
    mut sad_of: F,
) -> (MotionVector, u32) {
    let unit = units_per_pel(mode);
    let (low, high) = mv_range(fcode);
    let mut best_mv = MotionVector {
        x: (full.0 * unit).clamp(low, high),
        y: (full.1 * unit).clamp(low, high),
    };
    let mut best_sad = sad_of(best_mv);
    let mut steps = vec![unit / 2];
    if matches!(mode, BVopSampleMode::QuarterPel { .. }) {
        steps.push(1);
    }
    for step in steps {
        let centre = best_mv;
        for hy in -1..=1 {
            for hx in -1..=1 {
                if (hx, hy) == (0, 0) {
                    continue;
                }
                let cand = MotionVector {
                    x: (centre.x + hx * step).clamp(low, high),
                    y: (centre.y + hy * step).clamp(low, high),
                };
                let sad = sad_of(cand);
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = cand;
                }
            }
        }
    }
    (best_mv, best_sad)
}

/// Full-pel search + sub-pel refinement. Returns the best MV (in
/// `mode`'s units) and its SAD. The zero vector gets the classic small
/// favouring bias so flat areas stay skippable.
///
/// The search is the dense ±[`SEARCH_RANGE`] window around the zero
/// vector (the whole search under `fcode == 1`); for `fcode > 1` a
/// coarse 4-pel lattice over the whole
/// Table 7-9 window ([`search_window_pels`]) is scanned first and its
/// winner densely refined (±3 pels), so long displacements are found
/// without an exhaustive walk of the (up to ±1024-pel) window. Lattice
/// points whose 16×16 block lies entirely beyond the §7.6.4 padded
/// reference edge are skipped (they only duplicate the edge sample).
pub(crate) fn estimate_motion(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    mode: BVopSampleMode,
    fcode: u8,
) -> (MotionVector, u32) {
    let mut best = (0i32, 0i32);
    let mut best_sad = sad_full_pel(src, reference, mb_x, mb_y, 0, 0).saturating_sub(128);
    let consider = |dx: i32, dy: i32, best: &mut (i32, i32), best_sad: &mut u32| {
        if (dx, dy) == (0, 0) {
            return;
        }
        let sad = sad_full_pel(src, reference, mb_x, mb_y, dx, dy);
        if sad < *best_sad {
            *best_sad = sad;
            *best = (dx, dy);
        }
    };
    for dy in -SEARCH_RANGE..=SEARCH_RANGE {
        for dx in -SEARCH_RANGE..=SEARCH_RANGE {
            consider(dx, dy, &mut best, &mut best_sad);
        }
    }
    let window = full_pel_window(fcode, mode);
    if window > SEARCH_RANGE {
        // Coarse lattice over the wide window, clipped so the block
        // keeps at least one sample inside the padded reference.
        let (ref_w, ref_h) = (reference.width() as i32, reference.height() as i32);
        let dx_lo = (-window).max(-mb_x - 15);
        let dx_hi = window.min(ref_w - 1 - mb_x);
        let dy_lo = (-window).max(-mb_y - 15);
        let dy_hi = window.min(ref_h - 1 - mb_y);
        let mut coarse = best;
        let mut coarse_sad = best_sad;
        let mut dy = dy_lo;
        while dy <= dy_hi {
            let mut dx = dx_lo;
            while dx <= dx_hi {
                if dx.abs() > SEARCH_RANGE || dy.abs() > SEARCH_RANGE {
                    consider(dx, dy, &mut coarse, &mut coarse_sad);
                }
                dx += 4;
            }
            dy += 4;
        }
        if coarse != best {
            // Dense refinement around the lattice winner.
            for dy in (coarse.1 - 3)..=(coarse.1 + 3) {
                for dx in (coarse.0 - 3)..=(coarse.0 + 3) {
                    if dx.abs() <= window && dy.abs() <= window {
                        consider(dx, dy, &mut coarse, &mut coarse_sad);
                    }
                }
            }
            best = coarse;
        }
    }
    refine_subpel(best, mode, fcode, |cand| {
        sad_subpel(src, reference, mb_x, mb_y, cand, mode)
    })
}

/// SAD of one 8×8 luminance sub-block (Figure 6-8 index `i`, luma
/// blocks 0..=3) against a full-pel reference position.
fn sad_full_pel_block8(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    block: usize,
    dx: i32,
    dy: i32,
) -> u32 {
    let (row0, col0) = (8 * (block / 2), 8 * (block % 2));
    let (bx, by) = (mb_x + col0 as i32, mb_y + row0 as i32);
    let mut sad = 0u32;
    for j in 0..8 {
        for i in 0..8 {
            let s = src[row0 + j][col0 + i];
            let r = reference.fetch_clamped(bx + i as i32 + dx, by + j as i32 + dy);
            sad += (s - i32::from(r)).unsigned_abs();
        }
    }
    sad
}

/// SAD of one 8×8 sub-block against a sub-pel motion vector (in
/// `mode`'s units), through the decoder's own §7.6.2.1 / §7.6.2.2
/// per-block interpolator (the 8×8 §7.6.2.2 path carries its own
/// Figure 7-30 boundary mirroring, exactly as a decoded inter4v block
/// does).
fn sad_subpel_block8(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    block: usize,
    mv: MotionVector,
    mode: BVopSampleMode,
) -> u32 {
    let (row0, col0) = (8 * (block / 2), 8 * (block % 2));
    let (bx, by) = (mb_x + col0 as i32, mb_y + row0 as i32);
    let mut pred = [0u8; 64];
    match mode {
        BVopSampleMode::HalfPel => {
            crate::half_sample::interpolate_block_into(
                reference, mv.x, mv.y, bx, by, 8, 8, 0, &mut pred,
            );
        }
        BVopSampleMode::QuarterPel { bits_per_pixel } => {
            crate::quarter_sample::interpolate_block_qpel_into(
                reference,
                mv.x,
                mv.y,
                bx,
                by,
                8,
                8,
                0,
                bits_per_pixel,
                &mut pred,
            );
        }
    }
    let mut sad = 0u32;
    for j in 0..8 {
        for i in 0..8 {
            sad += (src[row0 + j][col0 + i] - i32::from(pred[j * 8 + i])).unsigned_abs();
        }
    }
    sad
}

/// §6.3.7 inter4v motion estimation: per 8×8 luminance block, a
/// full-pel search over a window around the 1-MV winner (plus the zero
/// vector), then the sub-pel refinement. Returns the four Figure
/// 6-8-ordered block MVs (in `mode`'s units) and the summed SAD.
fn estimate_motion_4mv(
    src: &[[i32; 16]; 16],
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    seed: MotionVector,
    mode: BVopSampleMode,
    fcode: u8,
) -> ([MotionVector; 4], u32) {
    let window = full_pel_window(fcode, mode);
    // Full-pel part of the 1-MV winner (truncating — the sub-pel
    // refinement recovers the fraction).
    let unit = units_per_pel(mode);
    let (sx, sy) = (seed.x / unit, seed.y / unit);
    let mut mvs = [MotionVector { x: 0, y: 0 }; 4];
    let mut total = 0u32;
    for (block, out) in mvs.iter_mut().enumerate() {
        let mut best = (0i32, 0i32);
        let mut best_sad = sad_full_pel_block8(src, reference, mb_x, mb_y, block, 0, 0);
        for dy in (sy - 2)..=(sy + 2) {
            for dx in (sx - 2)..=(sx + 2) {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                if !(-window..=window).contains(&dx) || !(-window..=window).contains(&dy) {
                    continue;
                }
                let sad = sad_full_pel_block8(src, reference, mb_x, mb_y, block, dx, dy);
                if sad < best_sad {
                    best_sad = sad;
                    best = (dx, dy);
                }
            }
        }
        let (best_mv, best_sub) = refine_subpel(best, mode, fcode, |cand| {
            sad_subpel_block8(src, reference, mb_x, mb_y, block, cand, mode)
        });
        *out = best_mv;
        total += best_sub;
    }
    (mvs, total)
}

/// Mean-removed activity of the source macroblock — the classic
/// intra/inter decision statistic (also the `crate::mb_quant`
/// activity-class input).
pub(crate) fn intra_activity(src: &[[i32; 16]; 16]) -> u32 {
    let sum: i32 = src.iter().flatten().sum();
    let mean = sum / 256;
    src.iter()
        .flatten()
        .map(|&s| (s - mean).unsigned_abs())
        .sum()
}

/// Quantise one inter residual block; returns the EVENT list (empty
/// when every level is zero).
pub(crate) fn quantise_inter_block(
    residual: &[[i32; 8]; 8],
    qp: u32,
    quant_type: bool,
    w_inter: &[[u8; 8]; 8],
) -> (Vec<AcEvent>, [[i32; 8]; 8]) {
    let f = forward_dct_8x8(residual, 8);
    let mut qf = [[0i32; 8]; 8];
    for v in 0..8 {
        for u in 0..8 {
            qf[v][u] = if quant_type {
                quantise_method1_inter(f[v][u], w_inter[v][u], qp)
            } else {
                quantise_method2_inter(f[v][u], qp)
            };
        }
    }
    let qfs = forward_scan(&qf, ScanType::Zigzag);
    (qfs_to_events(&qfs, 0), qf)
}

/// Encode one rectangular progressive P-VOP against `reference` (the
/// closed-loop reconstruction of the previous anchor). Returns the
/// emitted unit (start-code delimited, stuffed) and the mode
/// statistics.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_vop(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    reference: &DecodedFrame,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
) -> (Vec<u8>, PVopEncodeStats) {
    assert!((1..=31).contains(&qp), "vop_quant {qp} out of range");
    let (mb_width, mb_height) = cfg.mb_dimensions();
    let w_intra = crate::block::intra_quant_matrix(vol);
    let w_inter = nonintra_quant_matrix(vol);
    let use_dc_vlc = use_intra_dc_vlc(0, qp);
    let mode = sample_mode_of(vol);
    let fcode = cfg.fcode;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();

    let mut bw = BitWriter::new();
    write_p_vop_header(
        &mut bw,
        cfg.time_increment_resolution,
        modulo_time_base,
        time_increment,
        qp,
        fcode,
    );

    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut mv_grid = MvGrid::new(mb_height, mb_width);
    let mut stats = PVopEncodeStats::default();
    // §6.3.7 running quantiser (seeded by vop_quant, moved by dquant;
    // a skipped macroblock leaves it untouched).
    let vop_qp = qp;
    let mut running_qp = vop_qp;

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
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

            // Motion estimation + mode decision.
            let (mv, one_mv_sad) = estimate_motion(&src, &luma_ref, mb_x, mb_y, mode, fcode);
            // §6.3.7 inter4v: per-block refinement around the 1-MV
            // winner; chosen only when the summed block SADs undercut
            // the 1-MV SAD by a margin covering the three extra
            // motion_vector() bodies. Four equal vectors collapse to
            // the (cheaper, prediction-identical on the half-pel grid)
            // 1-MV form.
            let four = if cfg.four_mv {
                let (mvs, sad4) = estimate_motion_4mv(&src, &luma_ref, mb_x, mb_y, mv, mode, fcode);
                let distinct = mvs.iter().any(|&m| m != mvs[0]);
                if distinct && sad4 + 256 < one_mv_sad {
                    Some((mvs, sad4))
                } else {
                    None
                }
            } else {
                None
            };
            let inter_sad = four.map_or(one_mv_sad, |(_, s)| s);
            let choose_intra = activity + 512 < inter_sad;

            if choose_intra {
                stats.intra += 1;
                let (qp, dquant) = plan_quant(running_qp);
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
                // §7.6.5: an intra macroblock contributes a zero-vector
                // candidate (mirrors MvDriver::decode_macroblock).
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                encode_intra_mb_in_p(
                    &mut bw,
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
                continue;
            }

            // Table B.7 has no `inter4v+q` type, so a four-vector
            // macroblock keeps the running quantiser.
            let (qp, dquant) = if four.is_some() {
                (running_qp, None)
            } else {
                plan_quant(running_qp)
            };

            // Inter: build the prediction, quantise the residual.
            let motion = match four {
                Some((mvs, _)) => PvopMbMotion::FourMv(mvs),
                None => PvopMbMotion::OneMv(mv),
            };
            let prediction =
                predict_inter_macroblock(motion, &luma_ref, &cb_ref, &cr_ref, mb_x, mb_y, 0, mode)
                    .expect("inter motion always yields a prediction");

            let mut events: Vec<Vec<AcEvent>> = Vec::with_capacity(6);
            for i in 0..6 {
                let src_block = frame.block(mb_row, mb_col, i);
                let mut residual = [[0i32; 8]; 8];
                for y in 0..8 {
                    for x in 0..8 {
                        let p = match i {
                            0..=3 => {
                                let ry = y + 8 * (i / 2);
                                let rx = x + 8 * (i % 2);
                                prediction.luma[ry][rx]
                            }
                            4 => prediction.cb[y][x],
                            _ => prediction.cr[y][x],
                        };
                        residual[y][x] = src_block[y][x] - p;
                    }
                }
                let (ev, _qf) = quantise_inter_block(&residual, qp, cfg.quant_type, &w_inter);
                events.push(ev);
            }
            let all_zero = events.iter().all(|e| e.is_empty());

            if all_zero
                && matches!(motion, PvopMbMotion::OneMv(m) if m == (MotionVector { x: 0, y: 0 }))
            {
                // §6.3.6 skip: not_coded = 1, zero MV, no residual (the
                // running quantiser is untouched — no dquant is sent).
                stats.skipped += 1;
                bw.write_bit(true);
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                continue;
            }
            running_qp = qp;
            if dquant.is_some() {
                stats.dquant += 1;
            }

            let cbpy = (u8::from(!events[0].is_empty()) << 3)
                | (u8::from(!events[1].is_empty()) << 2)
                | (u8::from(!events[2].is_empty()) << 1)
                | u8::from(!events[3].is_empty());
            let cbpc = (u8::from(!events[4].is_empty()) << 1) | u8::from(!events[5].is_empty());

            bw.write_bit(false); // not_coded = 0
            match motion {
                PvopMbMotion::FourMv(mvs) => {
                    stats.inter4v += 1;
                    put_mcbpc_p(&mut bw, 2, cbpc); // derived_mb_type 2 = inter4v
                    put_cbpy(&mut bw, cbpy, false);
                    // Four §6.2.6.2 motion_vector() bodies, each against
                    // the §7.6.5 median for *that* block index, with the
                    // in-MB candidates made visible incrementally —
                    // mirroring the decoder's MvDriver::decode_four_mv.
                    let mut recorded = [MotionVector { x: 0, y: 0 }; 4];
                    mv_grid
                        .record_four_mv(mb_row, mb_col, recorded)
                        .expect("grid coordinates in range");
                    for (i, &block_mv) in mvs.iter().enumerate() {
                        let candidates = mv_grid
                            .predictor_candidates(mb_row, mb_col, i)
                            .expect("grid coordinates in range");
                        let predictor = predict_motion_vector(candidates);
                        put_motion_vector(
                            &mut bw,
                            block_mv.x - predictor.x,
                            block_mv.y - predictor.y,
                            fcode,
                        );
                        recorded[i] = block_mv;
                        mv_grid
                            .record_four_mv(mb_row, mb_col, recorded)
                            .expect("grid coordinates in range");
                    }
                }
                _ => {
                    stats.inter += 1;
                    // derived_mb_type 0 = inter (1 MV), 1 = inter+q.
                    put_mcbpc_p(&mut bw, if dquant.is_some() { 1 } else { 0 }, cbpc);
                    put_cbpy(&mut bw, cbpy, false);
                    if let Some(d) = dquant {
                        crate::vlc_encode::put_dquant(&mut bw, d);
                    }

                    // §7.6.5 median predictor over the shared grid state,
                    // then the §7.6.3 differential under the VOP's fcode.
                    let candidates = mv_grid
                        .predictor_candidates(mb_row, mb_col, 0)
                        .expect("grid coordinates in range");
                    let predictor = predict_motion_vector(candidates);
                    put_motion_vector(&mut bw, mv.x - predictor.x, mv.y - predictor.y, fcode);
                    mv_grid
                        .record_one_mv(mb_row, mb_col, mv)
                        .expect("grid coordinates in range");
                }
            }

            for ev in &events {
                if !ev.is_empty() {
                    crate::vlc_encode::put_ac_events(&mut bw, TcoefTable::Inter, ev);
                }
            }
        }
    }
    bw.next_start_code();
    (bw.into_bytes(), stats)
}

/// Encode one intra macroblock inside a P-VOP (Table B.7 `mcbpc`,
/// otherwise the I-VOP plan/emission path with cost-decided AC
/// prediction).
#[allow(clippy::too_many_arguments)]
fn encode_intra_mb_in_p(
    bw: &mut BitWriter,
    frame: &FrameView<'_>,
    grid: &mut IntraBlockGrid,
    mb_row: usize,
    mb_col: usize,
    qp: u32,
    cfg: &EncoderConfig,
    w_intra: &[[u8; 8]; 8],
    use_dc_vlc: bool,
    dquant: Option<i8>,
) {
    let mut plans_off: Vec<BlockPlan> = Vec::with_capacity(6);
    let mut plans_on: Option<Vec<BlockPlan>> = if cfg.ac_prediction {
        Some(Vec::with_capacity(6))
    } else {
        None
    };
    for i in 0..6 {
        let component = DcComponent::from_block_index(i);
        let samples = frame.block(mb_row, mb_col, i);
        let f = forward_dct_8x8(&samples, 8);
        let prep: PreparedBlock = quantise_intra_block(&f, component, qp, cfg.quant_type, w_intra);
        let predictors = grid.predictors_for(mb_row, mb_col, i, 8, qp);
        let direction = select_dc_direction(predictors.fa_dc, predictors.fb_dc, predictors.fc_dc);
        let off = plan_block(&prep, &predictors, direction, component, qp, false)
            .expect("no-prediction differentials are always codable");
        plans_off.push(off);
        if let Some(on) = plans_on.as_mut() {
            match plan_block(&prep, &predictors, direction, component, qp, true) {
                Some(p) => on.push(p),
                None => plans_on = None,
            }
        }
        grid.record(
            mb_row,
            mb_col,
            i,
            Some(BlockNeighbour::from_qf(&prep.qf, prep.dc_f, qp)),
        );
    }
    let plans_off: [BlockPlan; 6] = plans_off
        .try_into()
        .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));

    let emit = |bw: &mut BitWriter, plans: &[BlockPlan; 6], flag: bool| {
        let (cbpy, cbpc) = intra_mb_cbp(plans);
        bw.write_bit(false); // not_coded = 0
                             // derived_mb_type 3 = intra, 4 = intra+q.
        put_mcbpc_p(bw, if dquant.is_some() { 4 } else { 3 }, cbpc);
        bw.write_bit(flag); // ac_pred_flag
        put_cbpy(bw, cbpy, true);
        if let Some(d) = dquant {
            crate::vlc_encode::put_dquant(bw, d);
        }
        emit_intra_blocks(bw, plans, use_dc_vlc);
    };

    let chosen_on = plans_on.and_then(|on| {
        let on: [BlockPlan; 6] = on
            .try_into()
            .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
        let mut probe_off = BitWriter::new();
        emit(&mut probe_off, &plans_off, false);
        let mut probe_on = BitWriter::new();
        emit(&mut probe_on, &on, true);
        if probe_on.bit_position() < probe_off.bit_position() {
            Some(on)
        } else {
            None
        }
    });
    match &chosen_on {
        Some(on) => emit(bw, on, true),
        None => emit(bw, &plans_off, false),
    }
}

/// Decode an emitted P-VOP unit through the crate's decoder walk and
/// advance `store`'s anchor chain, returning a clone of the freshly
/// reconstructed frame (the closed-loop reference for the next VOP).
pub fn reconstruct_own_p_vop(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &mut FrameStore,
) -> DecodedFrame {
    reconstruct_own_p_vop_with_motion(vol, unit, store).0
}

/// [`reconstruct_own_p_vop`] plus the decoded per-macroblock motion in
/// raster order — the §7.6.9.5.1 / §7.6.9.6 co-located source the
/// following B-VOPs consume (`crate::bvop_encode`).
pub fn reconstruct_own_p_vop_with_motion(
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
    .expect("own VOP header must parse");
    assert!(matches!(vop.coding_type, VopCodingType::P));
    let entries =
        decode_p_vop_macroblocks(&mut br, vol, &vop, crate::compat::DecodeOptions::spec())
            .expect("own P-VOP payload must decode");
    let motion = entries
        .iter()
        .map(|e| match e {
            crate::frame_decode::PVopMbContent::Inter { motion, .. } => *motion,
            crate::frame_decode::PVopMbContent::Intra(_) => PvopMbMotion::Intra,
            // The progressive encoder never emits field-predicted MBs.
            crate::frame_decode::PVopMbContent::FieldInter { .. } => {
                unreachable!("progressive encoder emitted a field-predicted macroblock")
            }
        })
        .collect();
    let frame = crate::frame_decode::decode_p_vop(
        store,
        mb_width,
        mb_height,
        &entries,
        vop.rounding_type,
        sample_mode_of(vol),
        8,
    )
    .expect("own P-VOP must assemble")
    .clone();
    (frame, motion)
}

/// The §7.6.2 sub-pel interpolation mode a VOL selects
/// (`quarter_sample == 1` → §7.6.2.2 quarter-sample, else §7.6.2.1
/// half-sample).
pub(crate) fn sample_mode_of(vol: &crate::vol::VolHeader) -> BVopSampleMode {
    if vol.quarter_sample {
        BVopSampleMode::QuarterPel {
            bits_per_pixel: u32::from(vol.bits_per_pixel),
        }
    } else {
        BVopSampleMode::HalfPel
    }
}

/// The I-VOP sibling of [`reconstruct_own_p_vop`]: install an
/// I-VOP reconstruction as the new anchor.
pub fn push_i_vop_anchor(store: &mut FrameStore, recon: DecodedFrame) {
    store.push_anchor(recon);
}

/// One inter residual block for spot checks in tests.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_vop_header_round_trips() {
        let mut bw = BitWriter::new();
        write_p_vop_header(&mut bw, 25, 1, 3, 7, 1);
        bw.next_start_code();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(32).unwrap(), VOP_START_CODE);
        let vop = parse_vop_header_body(&mut br, 25, VopContext::default()).unwrap();
        assert!(matches!(vop.coding_type, VopCodingType::P));
        assert_eq!(vop.modulo_time_base, 1);
        assert_eq!(vop.time_increment, 3);
        assert_eq!(vop.quant, 7);
        assert_eq!(vop.fcode_fwd, 1);
        assert_eq!(vop.rounding_type, 0);
    }

    #[test]
    fn intra_activity_is_zero_on_flat_blocks() {
        let flat = [[100i32; 16]; 16];
        assert_eq!(intra_activity(&flat), 0);
    }
}

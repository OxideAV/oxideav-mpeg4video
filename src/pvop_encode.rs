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
use crate::field_encode::{estimate_field_motion, field_mv_differential, FieldEstimate};
use crate::framestore::{DecodedFrame, FrameStore};
use crate::ivop_encode::{
    elect_field_dct, field_dct_luma, forward_scan, intra_mb_fields, plan_block, qfs_to_events,
    quantise_intra_block, BlockPlan, EncoderConfig, FrameView, PreparedBlock, VopInterlaceFlags,
};
use crate::motion::{predict_motion_vector, MotionVector};
use crate::mv_predictor_grid::MvGrid;
use crate::neighbour::{BlockNeighbour, IntraBlockGrid};
use crate::packet_encode::{InterlacedMbInfo, Layout, MbFields, PacketVopInfo, PacketWriter};
use crate::predictor::select_dc_direction;
use crate::pvop_mv::{predict_inter_macroblock, PvopMbMotion};
use crate::quantise::{quantise_method1_inter, quantise_method2_inter};
use crate::scan::ScanType;
use crate::texture::{AcEvent, DcComponent};
use crate::vop::{parse_vop_header_body, vop_time_increment_bits, VopCodingType, VopContext};
use crate::vop_decode::{decode_p_vop_macroblocks, AnchorMbMotion};

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

/// SAD margin a §7.7.2.1 field-predicted macroblock must beat the
/// frame prediction by (the second motion body + the two
/// reference-field bits it costs).
pub const FIELD_MODE_BIAS: u32 = 96;

/// Emit a §6.2.5 P-VOP header (through `vop_fcode_forward`). The
/// writer is left mid-unit — the macroblock walk follows.
pub fn write_p_vop_header(
    bw: &mut BitWriter,
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    quant: u32,
    fcode: u8,
    interlace: Option<VopInterlaceFlags>,
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
    if let Some(flags) = interlace {
        flags.write(bw); // top_field_first + alternate_vertical_scan_flag
    }
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
    /// §7.7.2.1 field-predicted macroblocks (interlaced VOLs).
    pub field: usize,
    /// Macroblocks coded with the §7.7.1 field DCT (`dct_type == 1`).
    pub field_dct: usize,
    /// Intra macroblocks.
    pub intra: usize,
    /// Macroblocks that carried a `dquant` (`inter+q` / `intra+q`).
    pub dquant: usize,
    /// Video packets cut inside the VOP (resync markers emitted).
    pub packets: usize,
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
pub(crate) fn estimate_motion_4mv(
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
    scan: ScanType,
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
    let qfs = forward_scan(&qf, scan);
    (qfs_to_events(&qfs, 0), qf)
}

/// The inter-block scan of a VOP: zigzag, or the §6.3.5 alternate
/// vertical scan when the interlaced VOP header forces it.
pub(crate) fn inter_scan(cfg: &EncoderConfig) -> ScanType {
    cfg.forced_scan().unwrap_or(ScanType::Zigzag)
}

/// The six quantised residual blocks of an inter macroblock from a
/// 16×16 luminance prediction residual (already field-permuted when
/// the macroblock codes `dct_type == 1`) plus the two chroma
/// residuals.
pub(crate) fn quantise_inter_residual(
    luma: &[[i32; 16]; 16],
    cb: &[[i32; 8]; 8],
    cr: &[[i32; 8]; 8],
    qp: u32,
    quant_type: bool,
    w_inter: &[[u8; 8]; 8],
    scan: ScanType,
) -> Vec<Vec<AcEvent>> {
    let mut events: Vec<Vec<AcEvent>> = Vec::with_capacity(6);
    for i in 0..4 {
        let (row0, col0) = (8 * (i / 2), 8 * (i % 2));
        let mut block = [[0i32; 8]; 8];
        for (y, row) in block.iter_mut().enumerate() {
            row.copy_from_slice(&luma[row0 + y][col0..col0 + 8]);
        }
        events.push(quantise_inter_block(&block, qp, quant_type, w_inter, scan).0);
    }
    events.push(quantise_inter_block(cb, qp, quant_type, w_inter, scan).0);
    events.push(quantise_inter_block(cr, qp, quant_type, w_inter, scan).0);
    events
}

/// One macroblock's prediction residual: the 16×16 luminance plus the
/// two 8×8 chroma residuals.
pub(crate) type MacroblockResidual = ([[i32; 16]; 16], [[i32; 8]; 8], [[i32; 8]; 8]);

/// Source-minus-prediction residual of one macroblock: the 16×16
/// luminance plus the two 8×8 chroma residuals.
pub(crate) fn macroblock_residual(
    frame: &FrameView<'_>,
    mb_row: usize,
    mb_col: usize,
    prediction: &crate::reconstruct::InterPredictionMacroblock,
) -> MacroblockResidual {
    let src = source_luma_mb(frame, mb_row, mb_col);
    let mut luma = [[0i32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            luma[y][x] = src[y][x] - prediction.luma[y][x];
        }
    }
    let src_cb = frame.block(mb_row, mb_col, 4);
    let src_cr = frame.block(mb_row, mb_col, 5);
    let mut cb = [[0i32; 8]; 8];
    let mut cr = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            cb[y][x] = src_cb[y][x] - prediction.cb[y][x];
            cr[y][x] = src_cr[y][x] - prediction.cr[y][x];
        }
    }
    (luma, cb, cr)
}

/// The `(cbpy, cbpc)` pattern of six inter EVENT lists (§6.3.7
/// "1 = coded", Figure 6-8 order).
pub(crate) fn inter_cbp(events: &[Vec<AcEvent>]) -> (u8, u8) {
    let cbpy = (u8::from(!events[0].is_empty()) << 3)
        | (u8::from(!events[1].is_empty()) << 2)
        | (u8::from(!events[2].is_empty()) << 1)
        | u8::from(!events[3].is_empty());
    let cbpc = (u8::from(!events[4].is_empty()) << 1) | u8::from(!events[5].is_empty());
    (cbpy, cbpc)
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
        cfg.vop_interlace(),
    );
    let scan = inter_scan(cfg);

    let layout = if cfg.resilience.data_partitioned {
        Layout::PartitionedP
    } else {
        Layout::Combined
    };
    let mut pw = PacketWriter::new(
        bw,
        cfg.resilience,
        PacketVopInfo {
            coding_type: VopCodingType::P,
            fcode_fwd: fcode,
            fcode_bwd: 0,
            modulo_time_base,
            time_increment,
            time_increment_bits: vop_time_increment_bits(cfg.time_increment_resolution),
            intra_dc_vlc_thr: 0,
            total_macroblocks: (mb_width * mb_height) as u32,
            interlaced: cfg.interlaced,
        },
        layout,
    );

    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut mv_grid = MvGrid::new(mb_height, mb_width);
    let mut stats = PVopEncodeStats::default();
    // §6.3.7 running quantiser (seeded by vop_quant, moved by dquant,
    // re-seeded by each video packet's quant_scale; a skipped
    // macroblock leaves it untouched).
    let vop_qp = qp;
    let mut running_qp = vop_qp;

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            if pw.maybe_cut(mb_row * mb_width + mb_col, running_qp) {
                // §E.1.2: no prediction crosses a packet boundary —
                // the decoder rebuilds both grids at the header.
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
            let frame_sad = four.map_or(one_mv_sad, |(_, s)| s);
            // §7.7.2.1 field prediction (interlaced VOL): one vector per
            // output field against the better reference field parity,
            // coded against the shared CASE 1/2/3 predictor the grid
            // resolves for this macroblock. Chosen when the two field
            // SADs undercut the frame prediction by a margin covering
            // the extra motion body + reference bits.
            let field: Option<(MotionVector, FieldEstimate, FieldEstimate)> = if cfg.interlaced {
                let candidates = mv_grid
                    .field_predictor_candidates(mb_row, mb_col)
                    .expect("grid coordinates in range");
                let predictor = crate::motion::predict_field_motion_vector(candidates);
                let top = estimate_field_motion(
                    &src, &luma_ref, mb_x, mb_y, false, predictor, mode, fcode,
                );
                let bottom = estimate_field_motion(
                    &src, &luma_ref, mb_x, mb_y, true, predictor, mode, fcode,
                );
                (top.sad + bottom.sad + FIELD_MODE_BIAS < frame_sad)
                    .then_some((predictor, top, bottom))
            } else {
                None
            };
            let inter_sad = field.map_or(frame_sad, |(_, t, b)| t.sad + b.sad);
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
                if fields.interlaced.is_some_and(|i| i.field_dct) {
                    stats.field_dct += 1;
                }
                pw.push(&fields);
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
            let prediction = match field {
                Some((_, top, bottom)) => {
                    let mvs = crate::field_motion::FieldMotionVectors {
                        top: top.mv,
                        bottom: bottom.mv,
                    };
                    match mode {
                        BVopSampleMode::HalfPel => {
                            crate::field_motion::field_motion_compensate_one_reference(
                                &luma_ref,
                                &cb_ref,
                                &cr_ref,
                                mvs,
                                top.ref_field,
                                bottom.ref_field,
                                mb_x,
                                mb_y,
                                0,
                            )
                        }
                        BVopSampleMode::QuarterPel { bits_per_pixel } => {
                            crate::field_motion::field_motion_compensate_one_reference_qpel(
                                &luma_ref,
                                &cb_ref,
                                &cr_ref,
                                mvs,
                                top.ref_field,
                                bottom.ref_field,
                                mb_x,
                                mb_y,
                                0,
                                bits_per_pixel,
                            )
                        }
                    }
                }
                None => predict_inter_macroblock(
                    motion, &luma_ref, &cb_ref, &cr_ref, mb_x, mb_y, 0, mode,
                )
                .expect("inter motion always yields a prediction"),
            };

            let (res_luma, res_cb, res_cr) =
                macroblock_residual(frame, mb_row, mb_col, &prediction);
            // §7.7.1 dct_type election on the residual (interlaced VOL).
            let field_dct = cfg.interlaced && elect_field_dct(&res_luma);
            let res_luma = if field_dct {
                field_dct_luma(&res_luma)
            } else {
                res_luma
            };
            let events = quantise_inter_residual(
                &res_luma,
                &res_cb,
                &res_cr,
                qp,
                cfg.quant_type,
                &w_inter,
                scan,
            );
            let all_zero = events.iter().all(|e| e.is_empty());

            if all_zero
                && field.is_none()
                && matches!(motion, PvopMbMotion::OneMv(m) if m == (MotionVector { x: 0, y: 0 }))
            {
                // §6.3.6 skip: not_coded = 1, zero MV, no residual (the
                // running quantiser is untouched — no dquant is sent).
                stats.skipped += 1;
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
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                continue;
            }
            running_qp = qp;
            if dquant.is_some() {
                stats.dquant += 1;
            }

            let (cbpy, cbpc) = inter_cbp(&events);
            // dct_type is only carried by a macroblock with a coded
            // block; an all-zero residual has no luminance to permute.
            let field_dct = field_dct && !all_zero;
            if field_dct {
                stats.field_dct += 1;
            }

            let (mb_type, mvds) = match (field, motion) {
                (Some((predictor, top, bottom)), _) => {
                    stats.field += 1;
                    // Two §6.2.6.2 bodies (top then bottom field),
                    // each against the shared §7.7.2.1 predictor —
                    // mirroring MvDriver::decode_field_macroblock.
                    let mvds = vec![
                        field_mv_differential(top.mv, predictor),
                        field_mv_differential(bottom.mv, predictor),
                    ];
                    mv_grid
                        .record_field(mb_row, mb_col, top.mv, bottom.mv)
                        .expect("grid coordinates in range");
                    (if dquant.is_some() { 1u8 } else { 0u8 }, mvds)
                }
                (None, PvopMbMotion::FourMv(mvs)) => {
                    stats.inter4v += 1;
                    // Four §6.2.6.2 motion_vector() bodies, each against
                    // the §7.6.5 median for *that* block index, with the
                    // in-MB candidates made visible incrementally —
                    // mirroring the decoder's MvDriver::decode_four_mv.
                    let mut recorded = [MotionVector { x: 0, y: 0 }; 4];
                    mv_grid
                        .record_four_mv(mb_row, mb_col, recorded)
                        .expect("grid coordinates in range");
                    let mut mvds = Vec::with_capacity(4);
                    for (i, &block_mv) in mvs.iter().enumerate() {
                        let candidates = mv_grid
                            .predictor_candidates(mb_row, mb_col, i)
                            .expect("grid coordinates in range");
                        let predictor = predict_motion_vector(candidates);
                        mvds.push((block_mv.x - predictor.x, block_mv.y - predictor.y));
                        recorded[i] = block_mv;
                        mv_grid
                            .record_four_mv(mb_row, mb_col, recorded)
                            .expect("grid coordinates in range");
                    }
                    (2u8, mvds) // derived_mb_type 2 = inter4v
                }
                (None, _) => {
                    stats.inter += 1;
                    // §7.6.5 median predictor over the shared grid state,
                    // then the §7.6.3 differential under the VOP's fcode.
                    let candidates = mv_grid
                        .predictor_candidates(mb_row, mb_col, 0)
                        .expect("grid coordinates in range");
                    let predictor = predict_motion_vector(candidates);
                    mv_grid
                        .record_one_mv(mb_row, mb_col, mv)
                        .expect("grid coordinates in range");
                    // derived_mb_type 0 = inter (1 MV), 1 = inter+q.
                    (
                        if dquant.is_some() { 1 } else { 0 },
                        vec![(mv.x - predictor.x, mv.y - predictor.y)],
                    )
                }
            };
            let blocks: [Vec<AcEvent>; 6] = events
                .try_into()
                .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
            pw.push(&MbFields {
                not_coded: false,
                mb_type,
                cbpc,
                cbpy,
                ac_pred_flag: false,
                dquant,
                mcsel: None,
                mvds,
                fcode,
                intra_dc: None,
                blocks,
                interlaced: cfg.interlaced.then_some(InterlacedMbInfo {
                    field_dct,
                    field_refs: field.map(|(_, t, b)| (t.ref_field, b.ref_field)),
                }),
            });
        }
    }
    stats.packets = pw.packets_cut();
    (pw.finish(), stats)
}

/// Build the [`MbFields`] of one intra macroblock inside a P-VOP
/// (Table B.7 `intra` / `intra+q`, otherwise the I-VOP plan path with
/// the cost-decided `ac_pred_flag` measured under the writer's
/// layout).
#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_mb_in_p_fields(
    pw: &PacketWriter,
    frame: &FrameView<'_>,
    grid: &mut IntraBlockGrid,
    mb_row: usize,
    mb_col: usize,
    qp: u32,
    cfg: &EncoderConfig,
    w_intra: &[[u8; 8]; 8],
    use_dc_vlc: bool,
    dquant: Option<i8>,
) -> MbFields {
    let mut plans_off: Vec<BlockPlan> = Vec::with_capacity(6);
    let mut plans_on: Option<Vec<BlockPlan>> = if cfg.ac_prediction {
        Some(Vec::with_capacity(6))
    } else {
        None
    };
    let forced_scan = cfg.forced_scan();
    // §7.7.1 dct_type election (interlaced VOL only).
    let field_dct = cfg.interlaced && elect_field_dct(&source_luma_mb(frame, mb_row, mb_col));
    let interlaced = cfg.interlaced.then_some(InterlacedMbInfo {
        field_dct,
        field_refs: None,
    });
    for i in 0..6 {
        let component = DcComponent::from_block_index(i);
        let samples = frame.block_with_field_dct(mb_row, mb_col, i, field_dct);
        let f = forward_dct_8x8(&samples, 8);
        let prep: PreparedBlock = quantise_intra_block(&f, component, qp, cfg.quant_type, w_intra);
        let predictors = grid.predictors_for(mb_row, mb_col, i, 8, qp);
        let direction = select_dc_direction(predictors.fa_dc, predictors.fb_dc, predictors.fc_dc);
        let off = plan_block(
            &prep,
            &predictors,
            direction,
            component,
            qp,
            false,
            forced_scan,
        )
        .expect("no-prediction differentials are always codable");
        plans_off.push(off);
        if let Some(on) = plans_on.as_mut() {
            match plan_block(
                &prep,
                &predictors,
                direction,
                component,
                qp,
                true,
                forced_scan,
            ) {
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
    let fields_off = intra_mb_fields(&plans_off, false, use_dc_vlc, dquant, interlaced);
    plans_on
        .and_then(|on| {
            let on: [BlockPlan; 6] = on
                .try_into()
                .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
            let fields_on = intra_mb_fields(&on, true, use_dc_vlc, dquant, interlaced);
            (pw.cost_of(&fields_on) < pw.cost_of(&fields_off)).then_some(fields_on)
        })
        .unwrap_or(fields_off)
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
    let (frame, motion) = reconstruct_own_p_vop_with_anchor_motion(vol, unit, store);
    (frame, motion.iter().map(|m| m.progressive()).collect())
}

/// [`reconstruct_own_p_vop_with_motion`] keeping the interlaced shape
/// of the decoded motion: a §7.7.2.1 field-predicted macroblock
/// surfaces as [`AnchorMbMotion::Field`] (its field MV pair +
/// reference selections — the §7.7.2.2 interlaced-direct source an
/// interlaced B-VOP needs), everything else as
/// [`AnchorMbMotion::Frame`].
pub fn reconstruct_own_p_vop_with_anchor_motion(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &mut FrameStore,
) -> (DecodedFrame, Vec<AnchorMbMotion>) {
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
    let entries = if vol.data_partitioned {
        crate::vop_decode::decode_p_vop_macroblocks_dp(
            &mut br,
            vol,
            &vop,
            crate::compat::DecodeOptions::spec(),
        )
    } else {
        decode_p_vop_macroblocks(&mut br, vol, &vop, crate::compat::DecodeOptions::spec())
    }
    .expect("own P-VOP payload must decode");
    let motion = entries
        .iter()
        .map(|e| match e {
            crate::frame_decode::PVopMbContent::Inter { motion, .. } => {
                AnchorMbMotion::Frame(*motion)
            }
            crate::frame_decode::PVopMbContent::Intra(_) => {
                AnchorMbMotion::Frame(PvopMbMotion::Intra)
            }
            crate::frame_decode::PVopMbContent::FieldInter {
                mvs,
                top_field_ref,
                bottom_field_ref,
                ..
            } => AnchorMbMotion::Field {
                mvs: *mvs,
                top_ref: *top_field_ref,
                bottom_ref: *bottom_field_ref,
            },
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
        write_p_vop_header(&mut bw, 25, 1, 3, 7, 1, None);
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

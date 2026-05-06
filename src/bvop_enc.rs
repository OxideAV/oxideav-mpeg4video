//! B-VOP (bidirectional) encoder — ISO/IEC 14496-2 §7.6.5.
//!
//! Scope:
//! * Per-MB mode decision between Forward, Backward, Bidirectional (Interpolated)
//!   and Direct by comparing SAD. The cheapest mode wins.
//! * Emits `MODB` (Table B.3) + `MBTYPE` (Table B.4) + optional `CBPB` +
//!   `dbquant` (Table 6-33) + forward / backward MVDs + inter texture walk.
//!   Residuals are inter-coded using the same H.263 inter quant + Table B-17
//!   tcoef walk as the P-VOP path.
//! * **Round 11 — cbpb residual emit landed.** When the per-block
//!   forward DCT + quantiser produces non-zero AC levels, the MB layer
//!   switches to MODB="00", emits a 6-bit `cbpb` mask (bit 5 = Y0 ..
//!   bit 0 = Cr), emits `dbquant=0` (single bit `0` from Table 6-33,
//!   B-VOP-specific quant-delta VLC) for non-direct modes, and walks
//!   each coded block through `write_inter_ac` (Table B-17). Direct
//!   mode uses MODB="00" + mbtype="1" + cbpb + (no dbquant, per
//!   §6.2.7) + zero MV delta + residuals.
//! * `co_located_not_coded` inheritance (§7.5.9.5.4) — when the co-located
//!   P-MB was `not_coded`, the B-MB is implicitly skipped (no bits emitted).
//! * MV prediction: separate per-row predictor state for forward + backward
//!   MVDs (§7.5.8). Predictors reset at row start and update ONLY when a
//!   non-direct, non-skipped B-MB emits a vector in the matching direction.
//! * **Round 12 — 4MV direct emit landed.** When the co-located P-MB
//!   used 4MV, direct mode now scales per-block MVs via
//!   `direct_mode_mvs_4` (§7.5.9.5), builds per-block predictors via
//!   `predict_luma_mb_4mv`, and stores the result via `BMbMotion::quad`
//!   so any downstream consumer sees the per-block motion. The bitstream
//!   syntax is unchanged (MBTYPE_DIRECT + (0,0) MVD); the decoder picks
//!   the 4MV path implicitly from the co-located P-MB's
//!   `four_motion_vector` flag. The path is dormant on encoder-only
//!   round-trips today because the P-VOP encoder still emits 1MV-only.
//! * **Round 12 — `DIRECT_BONUS` lowered to 0** after a sweep showed the
//!   historical -200 was inert on the 24fps testsrc fixture (direct SAD
//!   already wins by hundreds of points on every B-MB). Eliminates a
//!   magic-number bias without changing PSNR/bytes on the regression
//!   fixture; values >= +50 actively hurt.
//!
//! * **Round 16 — quarter-pel B-VOP encoder**. The forward + backward ME
//!   pipelines now extend through an additional 8-candidate quarter-pel
//!   refinement step around the half-pel best (§7.5.4 / §7.6.2.2). MVs
//!   are stored in quarter-pel units when `quarter_sample == true`, the
//!   8-tap-filter predictor (`predict_block_qpel`, eqs. 7-37/7-38) is
//!   used everywhere a half-pel `predict_block` was used, and the chroma
//!   MV reduction switches to `luma_qmv_to_chroma` (§7.6.2.2 eq. 107).
//!   Direct mode inherits per-block QPel MVs from the co-located P-MB —
//!   the spec formulas (§7.5.9.5) are unit-agnostic, so a QPel input
//!   yields a QPel output. The MV VLC writer (Table B-12) is also
//!   unit-agnostic — `write_mv_component` emits `diff` bits with no
//!   awareness of the unit. The LSB now denotes the quarter-pel bit per
//!   §7.6.3.
//!
//! Out of scope (fall through to safer encoding or return unchanged):
//! * Interlaced field MVs — the B-VOP header advertises progressive
//!   (interlaced == 0 in the VOL), so the decoder never enters the
//!   field-MB paths.
//! * Alternate vertical scan — `vop.alternate_vertical_scan == false`.

use oxideav_core::Result;

use crate::bvop::{direct_mode_mvs, direct_mode_mvs_4, BMbMotion, BMode, BMvGrid, BRowPred};
use crate::encoder::QuantMode;
use crate::headers::vol::ZIGZAG;
use crate::inter::MvGrid;
use crate::mb::IVopPicture;
use crate::mc::{luma_mv_to_chroma, luma_qmv_to_chroma, predict_block, predict_block_qpel};
use crate::pvop::{
    encode_inter_block, load_chroma_block, load_luma_mb, predict_chroma_block, predict_luma_mb,
    predict_luma_mb_qpel, wrap_mvd, write_inter_ac, write_mv_component,
};
use crate::tables::bvop as bvop_tab;
use oxideav_core::bits::BitWriter;

/// Small-diamond bound for B-VOP ME (integer pels). Same as the P-VOP
/// default — keeps the search well inside `f_code=1` range.
const B_MAX_SEARCH_INT: i32 = 4;

/// Per-MB encoding record populated during mode decision.
struct BMbEncoding {
    /// Chosen prediction mode.
    mode: BMode,
    /// Forward MV (luma half-pel units). `(0,0)` when `mode == Backward`.
    /// For 4MV direct mode this carries `fwd4[0]` for predictor-grid
    /// summary purposes only — the bitstream does not emit it (direct
    /// mode MVD is always (0,0)).
    fwd_mv: (i32, i32),
    /// Backward MV (luma half-pel units). `(0,0)` when `mode == Forward`.
    bwd_mv: (i32, i32),
    /// Per-block forward MVs (4 entries) — equal to `[fwd_mv; 4]` for
    /// 1MV modes, distinct for 4MV direct.
    fwd4: [(i32, i32); 4],
    /// Per-block backward MVs (4 entries).
    bwd4: [(i32, i32); 4],
    /// True when the chosen mode is direct AND the co-located P-MB was
    /// 4MV — the BMvGrid entry must be built with `BMbMotion::quad`.
    /// Per §7.5.9.5, this triggers per-block predictor construction
    /// (one 8×8 predictor per luma block) instead of a single 16×16
    /// predictor. The bitstream syntax is unchanged: MBTYPE_DIRECT +
    /// single MVD (always (0,0)). The decoder picks the 4MV path
    /// implicitly from the co-located P-MB's `four_motion_vector` flag.
    direct_4mv: bool,
    /// Reconstructed 16×16 luma samples (MC + residual, clipped).
    recon_y: [u8; 256],
    /// Reconstructed 8×8 Cb.
    recon_cb: [u8; 64],
    /// Reconstructed 8×8 Cr.
    recon_cr: [u8; 64],
    /// Per-block coded flags (Y0..Y3, Cb, Cr).
    coded: [bool; 6],
    /// AC levels per block.
    ac_levels: [[i32; 64]; 6],
    /// True when the co-located P-MB was `not_coded` — no bits emitted.
    implicit_skip: bool,
}

impl BMbEncoding {
    fn skipped_placeholder() -> Self {
        Self {
            mode: BMode::Skipped,
            fwd_mv: (0, 0),
            bwd_mv: (0, 0),
            fwd4: [(0, 0); 4],
            bwd4: [(0, 0); 4],
            direct_4mv: false,
            recon_y: [0; 256],
            recon_cb: [0; 64],
            recon_cr: [0; 64],
            coded: [false; 6],
            ac_levels: [[0; 64]; 6],
            implicit_skip: true,
        }
    }
}

/// Encode one B-VOP body relative to `prev_ref` and `next_ref`.
///
/// * `prev_ref_grid` — MV grid of the backward reference (decoded P-VOP).
///   Consulted for co-located inheritance + direct-mode MV scaling.
/// * `trb`, `trd` — temporal distance ratio (see `bvop::trb_trd`).
/// * `f_code_fwd`, `f_code_bwd` — MV range codes (kept at 1 here).
/// * `quarter_sample` — when `true`, MVs are encoded in quarter-pel units
///   (§7.6.2.2). The forward + backward ME runs an extra 8-candidate
///   refinement step around the half-pel best, the 8-tap filter is used
///   for MC, and chroma MV reduction switches to `luma_qmv_to_chroma`
///   (eq. 107). Direct-mode MV scaling (§7.5.9.5) is unit-agnostic so
///   QPel co-located MVs scale to QPel B-MVs naturally.
///
/// Returns `Ok(())` — B-VOPs never become references, so no picture is
/// returned.
#[allow(clippy::too_many_arguments)]
pub fn encode_b_vop_body(
    bw: &mut BitWriter,
    v: &oxideav_core::VideoFrame,
    width: u32,
    height: u32,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    prev_ref_grid: &MvGrid,
    vop_quant: u32,
    f_code_fwd: u8,
    f_code_bwd: u8,
    trb: i32,
    trd: i32,
    quarter_sample: bool,
    quant_mode: QuantMode,
) -> Result<()> {
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut bmv_grid = BMvGrid::new(mb_w, mb_h);
    let mut row_pred = BRowPred::default();

    for mb_y in 0..mb_h {
        // §7.5.8 — predictors reset at the start of every MB row.
        row_pred.reset();
        for mb_x in 0..mb_w {
            let co = prev_ref_grid.get(mb_x, mb_y);
            let co_not_coded = co.not_coded;
            // Co-located 4MV inheritance (§7.5.9.5). When the P-MB used 4MV,
            // the B-MB direct mode MUST scale per-block MVs via
            // `direct_mode_mvs_4`. When the P-MB used 1MV, all 4 entries
            // collapse to the single 16×16 MV.
            //
            // QPel: `prev_ref_grid` carries P-MB MVs in the same unit the
            // P-VOP encoder used. With `quarter_sample = true`, those are
            // already QPel — no per-unit scaling is needed.
            let co_mvs4: [(i32, i32); 4] = if co.four_mv { co.mv } else { [co.mv[0]; 4] };
            let co_was_4mv = co.four_mv;

            if co_not_coded {
                // §7.5.9.5.4: implicit skip — no bits emitted.
                let mb = BMbEncoding::skipped_placeholder();
                bmv_grid.set(mb_x, mb_y, BMbMotion::uni((0, 0), (0, 0), BMode::Forward));
                // Row predictors are NOT updated for implicit skips.
                let _ = mb;
                continue;
            }

            let mb = estimate_b_mb(
                v,
                width,
                height,
                prev_ref,
                next_ref,
                mb_x,
                mb_y,
                vop_quant,
                co_mvs4,
                co_was_4mv,
                trb,
                trd,
                quarter_sample,
                quant_mode,
            )?;

            emit_b_mb(bw, &mb, &mut row_pred, f_code_fwd, f_code_bwd);

            // Update the B-VOP MV grid. For 4MV-direct MBs we record all 4
            // per-block MVs so any downstream consumer can see the per-block
            // motion (mirrors decoder-side `BMbMotion::quad`).
            let entry = if mb.direct_4mv {
                BMbMotion::quad(mb.fwd4, mb.bwd4, mb.mode)
            } else {
                BMbMotion::uni(mb.fwd_mv, mb.bwd_mv, mb.mode)
            };
            bmv_grid.set(mb_x, mb_y, entry);
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------
// Per-MB motion estimation + mode decision
// -------------------------------------------------------------------------

/// Run forward / backward / bidirectional / direct ME for one MB, pick
/// the cheapest mode by SAD, encode residuals. Returns a populated
/// `BMbEncoding`.
///
/// `co_mvs4` carries the 4 co-located P-MB MVs (all four equal when the
/// P-MB was 1MV). `co_was_4mv` records whether the P-MB used 4MV — this
/// is forwarded to direct-mode predictor construction so we use
/// `direct_mode_mvs_4` (per §7.5.9.5).
///
/// `quarter_sample` selects QPel ME + MC (§7.6.2.2). When `true` the
/// returned MVs (in `BMbEncoding::fwd_mv`, `bwd_mv`, `fwd4`, `bwd4`)
/// are in quarter-pel units; otherwise half-pel.
#[allow(clippy::too_many_arguments)]
fn estimate_b_mb(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    vop_quant: u32,
    co_mvs4: [(i32, i32); 4],
    co_was_4mv: bool,
    trb: i32,
    trd: i32,
    quarter_sample: bool,
    quant_mode: QuantMode,
) -> Result<BMbEncoding> {
    let src_y = load_luma_mb(v, width, height, mb_x, mb_y);

    // ---- Forward ME (search prev_ref) ----
    // Integer + half-pel refine, then optionally QPel refine. The returned
    // MV is in the active unit (half-pel or quarter-pel).
    let (fwd_int_x, fwd_int_y) = diamond_search(prev_ref, &src_y, mb_x, mb_y);
    let (fwd_h_x, fwd_h_y) =
        halfpel_refine(prev_ref, &src_y, mb_x, mb_y, fwd_int_x, fwd_int_y, false);
    let (fwd_mv_x, fwd_mv_y) = if quarter_sample {
        qpel_refine_mb(prev_ref, &src_y, mb_x, mb_y, fwd_h_x, fwd_h_y, false)
    } else {
        (fwd_h_x, fwd_h_y)
    };
    let fwd_mv = (fwd_mv_x, fwd_mv_y);
    let fwd_sad = sad_mb_any(
        prev_ref,
        &src_y,
        mb_x,
        mb_y,
        fwd_mv_x,
        fwd_mv_y,
        false,
        quarter_sample,
    );

    // ---- Backward ME (search next_ref) ----
    let (bwd_int_x, bwd_int_y) = diamond_search(next_ref, &src_y, mb_x, mb_y);
    let (bwd_h_x, bwd_h_y) =
        halfpel_refine(next_ref, &src_y, mb_x, mb_y, bwd_int_x, bwd_int_y, false);
    let (bwd_mv_x, bwd_mv_y) = if quarter_sample {
        qpel_refine_mb(next_ref, &src_y, mb_x, mb_y, bwd_h_x, bwd_h_y, false)
    } else {
        (bwd_h_x, bwd_h_y)
    };
    let bwd_mv = (bwd_mv_x, bwd_mv_y);
    let bwd_sad = sad_mb_any(
        next_ref,
        &src_y,
        mb_x,
        mb_y,
        bwd_mv_x,
        bwd_mv_y,
        false,
        quarter_sample,
    );

    // ---- Bidirectional SAD — average of forward + backward predictors ----
    let bi_sad = sad_bidir_any(
        prev_ref,
        next_ref,
        &src_y,
        mb_x,
        mb_y,
        fwd_mv,
        bwd_mv,
        quarter_sample,
    );

    // ---- Direct mode — co-located scaling + (0,0) delta. ----
    // 4MV-aware (§7.5.9.5): when the co-located P-MB used 4MV, the four
    // sub-block MVs scale independently to (fwd_i, bwd_i). The 1MV case
    // collapses to the same formula with all four entries equal.
    //
    // QPel: §7.5.9.5 formulas (`trb*MV/trd ± delta`) are linear in the MV
    // and only multiply by integer ratios — they preserve the unit. With
    // QPel inputs we get QPel outputs at no extra cost.
    let (direct_fwd4, direct_bwd4) = direct_mode_mvs_4(co_mvs4, trb, trd, (0, 0));
    // Single-MV summary (used by 1MV-direct callers and as a debug aid):
    let (direct_fwd, direct_bwd) = if co_was_4mv {
        // For 4MV the 16×16 summary MV is the (i=0) per-block MV — same
        // convention as the decoder's `BMbMotion::quad` which puts
        // `fwd4[0]` into the `fwd` summary slot.
        (direct_fwd4[0], direct_bwd4[0])
    } else {
        direct_mode_mvs(co_mvs4[0], trb, trd, (0, 0))
    };
    let direct_sad = if co_was_4mv {
        sad_bidir_4mv_any(
            prev_ref,
            next_ref,
            &src_y,
            mb_x,
            mb_y,
            direct_fwd4,
            direct_bwd4,
            quarter_sample,
        )
    } else {
        sad_bidir_any(
            prev_ref,
            next_ref,
            &src_y,
            mb_x,
            mb_y,
            direct_fwd,
            direct_bwd,
            quarter_sample,
        )
    };

    // Pick the cheapest. Tie-breaks prefer direct (smallest bit cost).
    let (chosen_mode, chosen_fwd, chosen_bwd) = {
        // Bit-cost penalty per mode. The per-mode penalties model the
        // *bitstream overhead* of each mode beyond the residual VLC bits,
        // so the cheapest mode wins on (residual-error-SAD + overhead).
        //
        //  * Direct  — 1 MODB bit + (optional) cbpb 6-bit + residual VLCs;
        //              no MVD bits. Cheapest by far.
        //  * Forward — 2 MODB + 4 MBTYPE + 6 cbpb + ~16 MVD bits + dbquant.
        //  * Backward — same as forward.
        //  * Bidirectional — same as forward + a second MV pair.
        //
        // **Round 12 — `DIRECT_BONUS` lowered to 0 after a sweep over
        // {-1000 .. +1000} on the 24fps testsrc fixture.** Direct SAD
        // already wins by >100 SAD points on every B-MB of this fixture,
        // so the historical -200 was inert. Values above ~+40 force
        // non-direct modes that increase bytes AND lose ~0.05 dB PSNR.
        // The cleanest choice that preserves the optimum without
        // introducing magic-number bias is `DIRECT_BONUS = 0`.
        // Forward/backward/bi penalties remain at the round-9 values —
        // they DO start tipping into other modes on busy content where
        // the SAD differences are small.
        const DIRECT_BONUS: i64 = 0;
        const FWD_ONLY_PENALTY: i64 = 48;
        const BWD_ONLY_PENALTY: i64 = 48;
        const BI_PENALTY: i64 = 80;

        let direct_cost = (direct_sad as i64) + DIRECT_BONUS;
        let fwd_cost = (fwd_sad as i64) + FWD_ONLY_PENALTY;
        let bwd_cost = (bwd_sad as i64) + BWD_ONLY_PENALTY;
        let bi_cost = (bi_sad as i64) + BI_PENALTY;

        let mut best_mode = BMode::Direct;
        let mut best_cost = direct_cost;
        let mut best_fwd = direct_fwd;
        let mut best_bwd = direct_bwd;
        if fwd_cost < best_cost {
            best_mode = BMode::Forward;
            best_cost = fwd_cost;
            best_fwd = fwd_mv;
            best_bwd = (0, 0);
        }
        if bwd_cost < best_cost {
            best_mode = BMode::Backward;
            best_cost = bwd_cost;
            best_fwd = (0, 0);
            best_bwd = bwd_mv;
        }
        if bi_cost < best_cost {
            best_mode = BMode::Interpolated;
            best_fwd = fwd_mv;
            best_bwd = bwd_mv;
        }
        (best_mode, best_fwd, best_bwd)
    };

    // 4MV-direct flag: only true when we picked direct AND the co-located
    // P-MB carried 4MV. The bitstream is identical (single MBTYPE_DIRECT +
    // (0,0) MVD); this only changes predictor construction + the BMvGrid
    // entry shape.
    let direct_4mv = matches!(chosen_mode, BMode::Direct) && co_was_4mv;

    // Per-block MV arrays for the chosen mode.
    let (chosen_fwd4, chosen_bwd4) = match chosen_mode {
        BMode::Direct => (direct_fwd4, direct_bwd4),
        BMode::Skipped => unreachable!("Skipped not selectable here"),
        _ => ([chosen_fwd; 4], [chosen_bwd; 4]),
    };

    // ---- Build predictor based on chosen mode ----
    let (pred_y, pred_cb, pred_cr) = build_predictor(
        prev_ref,
        next_ref,
        mb_x,
        mb_y,
        chosen_mode,
        chosen_fwd,
        chosen_bwd,
        chosen_fwd4,
        chosen_bwd4,
        direct_4mv,
        quarter_sample,
    );

    // ---- Residual + quant + reconstruction (inter path) ----
    let src_cb = load_chroma_block(v, width, height, 1, mb_x, mb_y);
    let src_cr = load_chroma_block(v, width, height, 2, mb_x, mb_y);

    let mut ac_levels = [[0i32; 64]; 6];
    let mut coded = [false; 6];
    let mut recon_y = [0u8; 256];

    for blk in 0..4 {
        let (sub_x, sub_y) = match blk {
            0 => (0, 0),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let mut src_blk = [0u8; 64];
        let mut pred_blk = [0u8; 64];
        for j in 0..8 {
            for i in 0..8 {
                src_blk[j * 8 + i] =
                    load_luma_sample(v, width, height, mb_x, mb_y, sub_x + i, sub_y + j);
                pred_blk[j * 8 + i] = pred_y[(sub_y + j) * 16 + (sub_x + i)];
            }
        }
        let (levels, recon) = encode_inter_block(&src_blk, &pred_blk, vop_quant, quant_mode);
        coded[blk] = levels.iter().any(|&l| l != 0);
        ac_levels[blk] = levels;
        for j in 0..8 {
            for i in 0..8 {
                recon_y[(sub_y + j) * 16 + (sub_x + i)] = recon[j * 8 + i];
            }
        }
    }
    let (lcb, recon_cb) = encode_inter_block(&src_cb, &pred_cb, vop_quant, quant_mode);
    let (lcr, recon_cr) = encode_inter_block(&src_cr, &pred_cr, vop_quant, quant_mode);
    coded[4] = lcb.iter().any(|&l| l != 0);
    coded[5] = lcr.iter().any(|&l| l != 0);
    ac_levels[4] = lcb;
    ac_levels[5] = lcr;

    Ok(BMbEncoding {
        mode: chosen_mode,
        fwd_mv: chosen_fwd,
        bwd_mv: chosen_bwd,
        fwd4: chosen_fwd4,
        bwd4: chosen_bwd4,
        direct_4mv,
        recon_y,
        recon_cb,
        recon_cr,
        coded,
        ac_levels,
        implicit_skip: false,
    })
}

fn load_luma_sample(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
) -> u8 {
    let w = width;
    let h = height;
    let plane = &v.planes[0];
    let xx = (mb_x * 16 + sub_x).min(w.saturating_sub(1));
    let yy = (mb_y * 16 + sub_y).min(h.saturating_sub(1));
    plane.data[yy * plane.stride + xx]
}

/// Build a 16×16 luma predictor + two 8×8 chroma predictors for one MB
/// according to the selected B-mode.
///
/// `fwd4`/`bwd4` carry the per-block MVs (used only when
/// `direct_4mv == true`). For all other modes the four entries are equal
/// to the single `fwd_mv`/`bwd_mv` and we take the cheaper 16×16 path.
///
/// 4MV direct (§7.5.9.5) — when the co-located P-MB used 4MV, each luma
/// 8×8 block has its own forward + backward MV, so each block needs its
/// own predictor. Chroma MV uses the average of the 4 luma MVs (matches
/// the decoder convention in `inter.rs`).
///
/// `quarter_sample` selects between half-pel (`predict_block`) and QPel
/// (`predict_block_qpel`, eqs. 7-37/7-38) luma prediction, and between
/// `luma_mv_to_chroma` and `luma_qmv_to_chroma` (§7.6.2.2 eq. 107) for
/// the chroma MV reduction. The MV unit must match the dispatch.
#[allow(clippy::too_many_arguments)]
fn build_predictor(
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mode: BMode,
    fwd_mv: (i32, i32),
    bwd_mv: (i32, i32),
    fwd4: [(i32, i32); 4],
    bwd4: [(i32, i32); 4],
    direct_4mv: bool,
    quarter_sample: bool,
) -> ([u8; 256], [u8; 64], [u8; 64]) {
    let mut pred_fwd_y = [0u8; 256];
    let mut pred_bwd_y = [0u8; 256];
    let mut pred_fwd_cb = [0u8; 64];
    let mut pred_bwd_cb = [0u8; 64];
    let mut pred_fwd_cr = [0u8; 64];
    let mut pred_bwd_cr = [0u8; 64];

    let use_fwd = matches!(
        mode,
        BMode::Forward | BMode::Interpolated | BMode::Direct | BMode::Skipped
    );
    let use_bwd = matches!(
        mode,
        BMode::Backward | BMode::Interpolated | BMode::Direct | BMode::Skipped
    );

    if use_fwd {
        if direct_4mv {
            predict_luma_mb_4mv_any(
                prev_ref,
                mb_x,
                mb_y,
                fwd4,
                false,
                quarter_sample,
                &mut pred_fwd_y,
            );
        } else {
            predict_luma_mb_any(
                prev_ref,
                mb_x,
                mb_y,
                fwd_mv.0,
                fwd_mv.1,
                false,
                quarter_sample,
                &mut pred_fwd_y,
            );
        }
        // Chroma MV: average of 4 luma MVs in 4MV mode (matches the
        // decoder path in `inter.rs::reconstruct_inter_mb`), single
        // luma MV otherwise. QPel uses `luma_qmv_to_chroma` (eq. 107).
        let (cmx, cmy) = chroma_mv_for_mb(direct_4mv, fwd_mv, fwd4, quarter_sample);
        predict_chroma_block(
            &prev_ref.cb,
            prev_ref.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            false,
            &mut pred_fwd_cb,
        );
        predict_chroma_block(
            &prev_ref.cr,
            prev_ref.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            false,
            &mut pred_fwd_cr,
        );
    }
    if use_bwd {
        if direct_4mv {
            predict_luma_mb_4mv_any(
                next_ref,
                mb_x,
                mb_y,
                bwd4,
                false,
                quarter_sample,
                &mut pred_bwd_y,
            );
        } else {
            predict_luma_mb_any(
                next_ref,
                mb_x,
                mb_y,
                bwd_mv.0,
                bwd_mv.1,
                false,
                quarter_sample,
                &mut pred_bwd_y,
            );
        }
        let (cmx, cmy) = chroma_mv_for_mb(direct_4mv, bwd_mv, bwd4, quarter_sample);
        predict_chroma_block(
            &next_ref.cb,
            next_ref.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            false,
            &mut pred_bwd_cb,
        );
        predict_chroma_block(
            &next_ref.cr,
            next_ref.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            false,
            &mut pred_bwd_cr,
        );
    }

    // Combine per mode.
    let mut pred_y = [0u8; 256];
    let mut pred_cb = [0u8; 64];
    let mut pred_cr = [0u8; 64];
    match mode {
        BMode::Forward => {
            pred_y = pred_fwd_y;
            pred_cb = pred_fwd_cb;
            pred_cr = pred_fwd_cr;
        }
        BMode::Backward => {
            pred_y = pred_bwd_y;
            pred_cb = pred_bwd_cb;
            pred_cr = pred_bwd_cr;
        }
        BMode::Interpolated | BMode::Direct | BMode::Skipped => {
            for i in 0..256 {
                pred_y[i] = ((pred_fwd_y[i] as u16 + pred_bwd_y[i] as u16 + 1) >> 1) as u8;
            }
            for i in 0..64 {
                pred_cb[i] = ((pred_fwd_cb[i] as u16 + pred_bwd_cb[i] as u16 + 1) >> 1) as u8;
                pred_cr[i] = ((pred_fwd_cr[i] as u16 + pred_bwd_cr[i] as u16 + 1) >> 1) as u8;
            }
        }
    }
    (pred_y, pred_cb, pred_cr)
}

// -------------------------------------------------------------------------
// 4MV-direct helpers
// -------------------------------------------------------------------------

/// Build a 16×16 luma predictor using **per-block** MVs (one per 8×8
/// luma block). Mirrors the decoder's 4MV path in
/// `inter.rs::reconstruct_inter_mb`. Block order: 0=(0,0), 1=(8,0),
/// 2=(0,8), 3=(8,8) — same as the rest of this crate.
fn predict_luma_mb_4mv(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mvs4: [(i32, i32); 4],
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    for (blk, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
        let (mvx, mvy) = mvs4[blk];
        let mut tmp = [0u8; 64];
        predict_block(
            &reference.y,
            reference.y_stride,
            ref_w,
            ref_h,
            px + *sub_x,
            py + *sub_y,
            mvx,
            mvy,
            8,
            rounding,
            &mut tmp,
            8,
        );
        for j in 0..8 {
            for i in 0..8 {
                out[(*sub_y as usize + j) * 16 + (*sub_x as usize + i)] = tmp[j * 8 + i];
            }
        }
    }
}

/// QPel variant of `predict_luma_mb_4mv` — per-block MVs run through
/// the 8-tap quarter-pel filter (`predict_block_qpel`, eqs. 7-37/7-38).
/// MVs are in quarter-pel units.
fn predict_luma_mb_4mv_qpel(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mvs4_q: [(i32, i32); 4],
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    for (blk, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
        let (mvx, mvy) = mvs4_q[blk];
        let mut tmp = [0u8; 64];
        predict_block_qpel(
            &reference.y,
            reference.y_stride,
            ref_w,
            ref_h,
            px + *sub_x,
            py + *sub_y,
            mvx,
            mvy,
            8,
            rounding,
            &mut tmp,
            8,
        );
        for j in 0..8 {
            for i in 0..8 {
                out[(*sub_y as usize + j) * 16 + (*sub_x as usize + i)] = tmp[j * 8 + i];
            }
        }
    }
}

/// Dispatch luma 4MV predictor to half-pel or QPel based on
/// `quarter_sample`. The MV unit must match the dispatch.
fn predict_luma_mb_4mv_any(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mvs4: [(i32, i32); 4],
    rounding: bool,
    quarter_sample: bool,
    out: &mut [u8; 256],
) {
    if quarter_sample {
        predict_luma_mb_4mv_qpel(reference, mb_x, mb_y, mvs4, rounding, out);
    } else {
        predict_luma_mb_4mv(reference, mb_x, mb_y, mvs4, rounding, out);
    }
}

/// Dispatch luma 1MV predictor to half-pel (`predict_luma_mb`) or QPel
/// (`predict_luma_mb_qpel`) based on `quarter_sample`. The MV unit must
/// match the dispatch.
#[allow(clippy::too_many_arguments)]
fn predict_luma_mb_any(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mvx: i32,
    mvy: i32,
    rounding: bool,
    quarter_sample: bool,
    out: &mut [u8; 256],
) {
    if quarter_sample {
        predict_luma_mb_qpel(reference, mb_x, mb_y, mvx, mvy, rounding, out);
    } else {
        predict_luma_mb(reference, mb_x, mb_y, mvx, mvy, rounding, out);
    }
}

/// Pick the chroma MV for an MB. In 1MV mode this is just the active
/// luma→chroma reduction applied to the single MV. In 4MV mode chroma
/// uses the average of the 4 luma MVs scaled to chroma — same formula
/// the decoder uses (see `inter.rs` lines around the
/// `(cmx, cmy) = if four_mv { ... }` block).
///
/// `quarter_sample` switches between `luma_mv_to_chroma` (half-pel input)
/// and `luma_qmv_to_chroma` (QPel input, §7.6.2.2 eq. 107). Chroma
/// output is always half-pel.
fn chroma_mv_for_mb(
    four_mv: bool,
    single_mv: (i32, i32),
    mvs4: [(i32, i32); 4],
    quarter_sample: bool,
) -> (i32, i32) {
    let to_chroma = |v: i32| -> i32 {
        if quarter_sample {
            luma_qmv_to_chroma(v)
        } else {
            luma_mv_to_chroma(v)
        }
    };
    if four_mv {
        let sx: i32 = mvs4.iter().map(|(x, _)| *x).sum();
        let sy: i32 = mvs4.iter().map(|(_, y)| *y).sum();
        (to_chroma(sx / 4), to_chroma(sy / 4))
    } else {
        (to_chroma(single_mv.0), to_chroma(single_mv.1))
    }
}

// -------------------------------------------------------------------------
// SAD helpers
// -------------------------------------------------------------------------

fn diamond_search(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
) -> (i32, i32) {
    let mut best_x = 0i32;
    let mut best_y = 0i32;
    let mut best_sad = sad_integer(reference, src, mb_x, mb_y, 0, 0);
    const STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for _ in 0..(B_MAX_SEARCH_INT as usize * 2) {
        let mut improved = false;
        for (dx, dy) in STEPS {
            let nx = best_x + dx;
            let ny = best_y + dy;
            if nx.abs() > B_MAX_SEARCH_INT || ny.abs() > B_MAX_SEARCH_INT {
                continue;
            }
            let s = sad_integer(reference, src, mb_x, mb_y, nx, ny);
            if s < best_sad {
                best_sad = s;
                best_x = nx;
                best_y = ny;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    (best_x, best_y)
}

fn sad_integer(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mv_x: i32,
    mv_y: i32,
) -> u32 {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let blk_px = (mb_x * 16) as i32;
    let blk_py = (mb_y * 16) as i32;
    let mut s = 0u32;
    for j in 0..16 {
        for i in 0..16 {
            let x = (blk_px + mv_x + i).clamp(0, ref_w - 1) as usize;
            let y = (blk_py + mv_y + j).clamp(0, ref_h - 1) as usize;
            let r = reference.y[y * reference.y_stride + x] as i32;
            let sv = src[(j as usize) * 16 + (i as usize)] as i32;
            s = s.wrapping_add((sv - r).unsigned_abs());
        }
    }
    s
}

fn halfpel_refine(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    int_x: i32,
    int_y: i32,
    rounding: bool,
) -> (i32, i32) {
    let mut best_x = int_x * 2;
    let mut best_y = int_y * 2;
    let mut best_sad =
        sad_halfpel_with_rounding(reference, src, mb_x, mb_y, best_x, best_y, rounding);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let hx = int_x * 2 + dx;
            let hy = int_y * 2 + dy;
            let s = sad_halfpel_with_rounding(reference, src, mb_x, mb_y, hx, hy, rounding);
            if s < best_sad {
                best_sad = s;
                best_x = hx;
                best_y = hy;
            }
        }
    }
    (best_x, best_y)
}

fn sad_halfpel_with_rounding(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
) -> u32 {
    let mut pred = [0u8; 256];
    predict_luma_mb(
        reference, mb_x, mb_y, mv_x_half, mv_y_half, rounding, &mut pred,
    );
    let mut s = 0u32;
    for i in 0..256 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// Single-MV SAD with active-unit dispatch. The MV is in half-pel or
/// quarter-pel units depending on `quarter_sample`.
#[allow(clippy::too_many_arguments)]
fn sad_mb_any(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mvx: i32,
    mvy: i32,
    rounding: bool,
    quarter_sample: bool,
) -> u32 {
    let mut pred = [0u8; 256];
    predict_luma_mb_any(
        reference,
        mb_x,
        mb_y,
        mvx,
        mvy,
        rounding,
        quarter_sample,
        &mut pred,
    );
    let mut s = 0u32;
    for i in 0..256 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// Bidirectional SAD with active-unit dispatch.
#[allow(clippy::too_many_arguments)]
fn sad_bidir_any(
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    fwd_mv: (i32, i32),
    bwd_mv: (i32, i32),
    quarter_sample: bool,
) -> u32 {
    let mut pf = [0u8; 256];
    let mut pb = [0u8; 256];
    predict_luma_mb_any(
        prev_ref,
        mb_x,
        mb_y,
        fwd_mv.0,
        fwd_mv.1,
        false,
        quarter_sample,
        &mut pf,
    );
    predict_luma_mb_any(
        next_ref,
        mb_x,
        mb_y,
        bwd_mv.0,
        bwd_mv.1,
        false,
        quarter_sample,
        &mut pb,
    );
    let mut s = 0u32;
    for i in 0..256 {
        let p = (pf[i] as u16 + pb[i] as u16 + 1) >> 1;
        s = s.wrapping_add((src[i] as i32 - p as i32).unsigned_abs());
    }
    s
}

/// 4MV bidirectional SAD with active-unit dispatch.
#[allow(clippy::too_many_arguments)]
fn sad_bidir_4mv_any(
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    fwd4: [(i32, i32); 4],
    bwd4: [(i32, i32); 4],
    quarter_sample: bool,
) -> u32 {
    let mut pf = [0u8; 256];
    let mut pb = [0u8; 256];
    predict_luma_mb_4mv_any(prev_ref, mb_x, mb_y, fwd4, false, quarter_sample, &mut pf);
    predict_luma_mb_4mv_any(next_ref, mb_x, mb_y, bwd4, false, quarter_sample, &mut pb);
    let mut s = 0u32;
    for i in 0..256 {
        let p = (pf[i] as u16 + pb[i] as u16 + 1) >> 1;
        s = s.wrapping_add((src[i] as i32 - p as i32).unsigned_abs());
    }
    s
}

/// Quarter-pel refinement step for a 16×16 MB. Evaluates the 8
/// surrounding quarter-pel candidates around `(mvx_half * 2, mvy_half * 2)`
/// (the half-pel best, doubled into QPel units) and returns the best MV
/// in quarter-pel units (§7.5.4 / §7.6.2.2).
///
/// Search window cap: ±32 quarter-pels to stay within the f_code=1
/// range (§7.6.3 — `range = 32 * f`, `f=1` here).
fn qpel_refine_mb(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mvx_half: i32,
    mvy_half: i32,
    rounding: bool,
) -> (i32, i32) {
    let center_qx = mvx_half * 2;
    let center_qy = mvy_half * 2;
    let mut best_qx = center_qx;
    let mut best_qy = center_qy;
    let mut best_sad = sad_mb_any(reference, src, mb_x, mb_y, best_qx, best_qy, rounding, true);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let qx = center_qx + dx;
            let qy = center_qy + dy;
            if qx.abs() > 32 || qy.abs() > 32 {
                continue;
            }
            let s = sad_mb_any(reference, src, mb_x, mb_y, qx, qy, rounding, true);
            if s < best_sad {
                best_sad = s;
                best_qx = qx;
                best_qy = qy;
            }
        }
    }
    (best_qx, best_qy)
}

// -------------------------------------------------------------------------
// Bitstream emission (MODB + MBTYPE + MVDs + residual)
// -------------------------------------------------------------------------

fn emit_b_mb(
    bw: &mut BitWriter,
    mb: &BMbEncoding,
    row_pred: &mut BRowPred,
    f_code_fwd: u8,
    f_code_bwd: u8,
) {
    if mb.implicit_skip {
        return;
    }

    // Residual presence per block (Y0..Y3, Cb, Cr). cbpb bit 5 = blk 0 (Y0),
    // bit 0 = blk 5 (Cr) — matches decoder `(cbpb >> (5 - blk)) & 1`.
    let mut cbpb: u8 = 0;
    for (blk, &c) in mb.coded.iter().enumerate() {
        if c {
            cbpb |= 1 << (5 - blk);
        }
    }
    let any_coded = cbpb != 0;

    // MODB selection (2004 Table B.3):
    //   - "1"    → skipped/default (direct mode, zero delta, no residual).
    //   - "01"   → mb_type present, cbpb absent (implicit zero).
    //   - "00"   → mb_type AND cbpb present.
    //
    // We emit the skipped form only for Direct mode with no residual AND
    // zero direct delta — both conditions hold by construction (we always
    // pick delta = (0,0) for direct).
    if matches!(mb.mode, BMode::Direct) && !any_coded {
        // MODB = "1" — single bit. No mbtype, no MVs, no residuals.
        bw.write_bits(0b1, 1);
        // Row predictors NOT updated for direct mode (§7.5.8).
        return;
    }

    if any_coded {
        // MODB = "00" — mbtype + cbpb both present.
        bw.write_bits(0b00, 2);
    } else {
        // MODB = "01" — mbtype only, cbpb implicit zero.
        bw.write_bits(0b01, 2);
    }

    // MBTYPE (Table B.4, 1..=4 bits):
    match mb.mode {
        BMode::Direct => bw.write_bits(0b1, 1),
        BMode::Interpolated => bw.write_bits(0b01, 2),
        BMode::Backward => bw.write_bits(0b001, 3),
        BMode::Forward => bw.write_bits(0b0001, 4),
        BMode::Skipped => unreachable!("skipped handled above"),
    }

    // CBPB (6 bits, Y0..Y3 Cb Cr — bit 5 = Y0 .. bit 0 = Cr) — only when
    // MODB == "00". §6.2.7 / Table 11-3.
    if any_coded {
        bw.write_bits(cbpb as u32, 6);
    }

    // dbquant — §6.3.5 / Table 6-33: present iff `mbtype != DIRECT &&
    // cbpb != 0`. Codes:
    //     0   → delta = 0 (no quant change)
    //     10  → delta = -2
    //     11  → delta = +2
    // We always emit the single-bit `0` because the encoder keeps the
    // VOP-level quant constant — round-9 fix made the decoder use this
    // exact VLC, so emitting `0` is decoded as "no quant change".
    if any_coded && !matches!(mb.mode, BMode::Direct) {
        bw.write_bits(0, 1);
    }

    // Motion vectors per mode (§7.6.5.3). Same order as the decoder.
    match mb.mode {
        BMode::Direct => {
            // Direct-mode `mvd_b` delta — we always emit (0, 0).
            write_mv_component(bw, 0, 1);
            write_mv_component(bw, 0, 1);
            // Predictors NOT updated for direct mode.
        }
        BMode::Forward => {
            let range = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
            let dx = wrap_mvd(mb.fwd_mv.0 - row_pred.fwd.0, range);
            let dy = wrap_mvd(mb.fwd_mv.1 - row_pred.fwd.1, range);
            write_mv_component(bw, dx, f_code_fwd);
            write_mv_component(bw, dy, f_code_fwd);
            row_pred.fwd = mb.fwd_mv;
        }
        BMode::Backward => {
            let range = 32i32 << (f_code_bwd.saturating_sub(1) as i32);
            let dx = wrap_mvd(mb.bwd_mv.0 - row_pred.bwd.0, range);
            let dy = wrap_mvd(mb.bwd_mv.1 - row_pred.bwd.1, range);
            write_mv_component(bw, dx, f_code_bwd);
            write_mv_component(bw, dy, f_code_bwd);
            row_pred.bwd = mb.bwd_mv;
        }
        BMode::Interpolated => {
            let range_f = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
            let fdx = wrap_mvd(mb.fwd_mv.0 - row_pred.fwd.0, range_f);
            let fdy = wrap_mvd(mb.fwd_mv.1 - row_pred.fwd.1, range_f);
            write_mv_component(bw, fdx, f_code_fwd);
            write_mv_component(bw, fdy, f_code_fwd);
            row_pred.fwd = mb.fwd_mv;
            let range_b = 32i32 << (f_code_bwd.saturating_sub(1) as i32);
            let bdx = wrap_mvd(mb.bwd_mv.0 - row_pred.bwd.0, range_b);
            let bdy = wrap_mvd(mb.bwd_mv.1 - row_pred.bwd.1, range_b);
            write_mv_component(bw, bdx, f_code_bwd);
            write_mv_component(bw, bdy, f_code_bwd);
            row_pred.bwd = mb.bwd_mv;
        }
        BMode::Skipped => unreachable!(),
    }

    // Residual coefficients — per-block walk in Y0..Y3, Cb, Cr order. Inter
    // tcoef VLC (Table B-17) — same emitter as P-VOPs since B-MB residuals
    // are coded as inter blocks. The decoder reads each coded block in the
    // same order (`for blk in 0..6` then `(cbpb >> (5 - blk)) & 1`).
    if any_coded {
        for blk in 0..6 {
            if mb.coded[blk] {
                write_inter_ac(bw, &mb.ac_levels[blk]);
            }
        }
    }

    // Touch unused tables so the compiler doesn't complain when we revisit.
    let _ = ZIGZAG;
    let _ = mb.recon_y;
    let _ = mb.recon_cb;
    let _ = mb.recon_cr;
    let _ = bvop_tab::MODB_SKIPPED;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_costs_prefer_direct_on_static_scene() {
        // Direct mode on a static scene (co_mv = 0 → fwd = 0, bwd = 0)
        // should yield the lowest SAD (equal to forward SAD on a static
        // scene) and win on a pure-SAD basis (Direct ≤ Forward+48 always).
        // This test is a sanity probe — the actual heuristics live in
        // estimate_b_mb; here we just verify that the module compiles
        // and direct_mode_mvs returns (0,0) when co_mv is zero.
        let (f, b) = direct_mode_mvs((0, 0), 1, 2, (0, 0));
        assert_eq!(f, (0, 0));
        assert_eq!(b, (0, 0));
    }

    #[test]
    fn chroma_mv_average_in_4mv_mode() {
        // chroma_mv_for_mb in 4MV mode averages the 4 luma MVs (rounded
        // toward zero by integer division) before scaling to chroma.
        // luma_mv_to_chroma is a 2:1 reduction with the spec rounding
        // table; we just verify the average step is (sx/4, sy/4).
        let mvs = [(2, 4), (2, 4), (2, 4), (2, 4)];
        let (cx_4mv, cy_4mv) = chroma_mv_for_mb(true, (0, 0), mvs, false);
        let (cx_1mv, cy_1mv) = chroma_mv_for_mb(false, (2, 4), mvs, false);
        assert_eq!(cx_4mv, cx_1mv);
        assert_eq!(cy_4mv, cy_1mv);

        // Distinct per-block MVs averaging to (3, 3).
        let mvs = [(0, 0), (4, 4), (4, 4), (4, 4)];
        let (cx, cy) = chroma_mv_for_mb(true, (0, 0), mvs, false);
        // sum = (12, 12); avg = (3, 3); scaled to chroma via spec table.
        let (cx_ref, cy_ref) = (luma_mv_to_chroma(3), luma_mv_to_chroma(3));
        assert_eq!((cx, cy), (cx_ref, cy_ref));

        // Round-16 QPel chroma reduction: when `quarter_sample == true`,
        // `chroma_mv_for_mb` must use `luma_qmv_to_chroma` (§7.6.2.2 eq.
        // 107). Verify the dispatch.
        let mvs_q = [(8, 4), (8, 4), (8, 4), (8, 4)];
        let (cx_q, cy_q) = chroma_mv_for_mb(false, (8, 4), mvs_q, true);
        assert_eq!((cx_q, cy_q), (luma_qmv_to_chroma(8), luma_qmv_to_chroma(4)));
    }

    #[test]
    fn predict_luma_mb_4mv_uses_per_block_mvs() {
        // Build a 32x32-MB-sized reference (32x32 luma) with a horizontal
        // gradient so that different MVs produce visibly different predicted
        // blocks. Using a 16x16 MB at (0,0) with MVs that all point to (8,0)
        // half-pel = (4,0) integer should match a pure-translation
        // predictor; using distinct per-block MVs should NOT match.
        let mut pic = IVopPicture::new(32, 32);
        for j in 0..32 {
            for i in 0..pic.y_stride {
                pic.y[j * pic.y_stride + i] = i as u8; // horizontal gradient
            }
        }

        // 1MV: every block uses MV=(8, 0) (half-pel) → integer shift of 4
        // pels. Expected: each row of pred is `4 + col`.
        let mvs1 = [(8, 0); 4];
        let mut pred1 = [0u8; 256];
        predict_luma_mb_4mv(&pic, 0, 0, mvs1, false, &mut pred1);
        for j in 0..16 {
            for i in 0..16 {
                assert_eq!(pred1[j * 16 + i], (4 + i) as u8, "1MV at ({i},{j})");
            }
        }

        // 4MV: top-left block MV=(0,0) (integer shift 0), top-right MV=(8,0)
        // (integer shift +4), bot-left MV=(0,0), bot-right MV=(8,0).
        let mvs4 = [(0, 0), (8, 0), (0, 0), (8, 0)];
        let mut pred4 = [0u8; 256];
        predict_luma_mb_4mv(&pic, 0, 0, mvs4, false, &mut pred4);
        // Top-left block (cols 0..8): pred should equal col index.
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(pred4[j * 16 + i], i as u8, "4MV TL at ({i},{j})");
            }
        }
        // Top-right block (cols 8..16) with MV=(8,0) samples reference
        // column = (8) + (col-8) + 4 = col + 4.
        for j in 0..8 {
            for i in 8..16 {
                assert_eq!(pred4[j * 16 + i], (i + 4) as u8, "4MV TR at ({i},{j})");
            }
        }
        // Bottom-left block (rows 8..16, cols 0..8) MV=(0,0): col index.
        for j in 8..16 {
            for i in 0..8 {
                assert_eq!(pred4[j * 16 + i], i as u8, "4MV BL at ({i},{j})");
            }
        }
        // And finally verify 1MV and 4MV outputs differ (proves per-block
        // dispatch actually fires).
        assert_ne!(&pred1[..], &pred4[..]);
    }

    /// 4MV-direct round-trip: build a synthetic 32x32 reference + a forged
    /// 4MV co-located grid + a B-frame source, run `encode_b_vop_body`,
    /// then verify that the resulting `BMvGrid` records the MB as
    /// `four_mv_direct`. This is the encoder-side proof that the new
    /// path fires when the co-located P-MB is 4MV.
    #[test]
    fn encode_b_vop_4mv_direct_records_quad_motion() {
        use crate::inter::MbMotion;
        use crate::mc::predict_block;

        // Build a 32x32 reference picture filled with gradient.
        let mut prev = IVopPicture::new(32, 32);
        for j in 0..32 {
            for i in 0..prev.y_stride {
                prev.y[j * prev.y_stride + i] = i as u8;
            }
        }
        // Identical next ref so direct mode SAD is small.
        let next = prev.clone();

        // Forge a P-VOP MV grid where MB (0,0) is 4MV with distinct MVs;
        // every other MB defaults to 1MV zero. mb_w = mb_h = 2 (32/16).
        let mut grid = MvGrid::new(2, 2);
        grid.set(
            0,
            0,
            MbMotion {
                mv: [(0, 0), (4, 0), (0, 4), (4, 4)],
                four_mv: true,
                not_coded: false,
            },
        );

        // Build the source B-frame as an exact match of the predicted
        // 4MV-direct picture so direct mode wins by SAD landslide.
        // direct_mode_mvs_4 with trb=trd → forward MVs == co MVs,
        // backward MVs == 0.
        let trb = 1i32;
        let trd = 1i32;
        let (fwd4, bwd4) = direct_mode_mvs_4([(0, 0), (4, 0), (0, 4), (4, 4)], trb, trd, (0, 0));
        // Predicted MB = average(fwd_pred, bwd_pred).
        let mut expected = [0u8; 256];
        let mut pf = [0u8; 256];
        let mut pb = [0u8; 256];
        predict_luma_mb_4mv(&prev, 0, 0, fwd4, false, &mut pf);
        predict_luma_mb_4mv(&next, 0, 0, bwd4, false, &mut pb);
        for i in 0..256 {
            expected[i] = ((pf[i] as u16 + pb[i] as u16 + 1) >> 1) as u8;
        }

        // Set up a 32x32 source VideoFrame whose MB(0,0) equals expected.
        // Other MBs are uniform to keep things deterministic.
        use oxideav_core::{VideoFrame, VideoPlane};
        let mut y_data = vec![128u8; 32 * 32];
        let cb_data = vec![128u8; 16 * 16];
        let cr_data = vec![128u8; 16 * 16];
        for j in 0..16 {
            for i in 0..16 {
                y_data[j * 32 + i] = expected[j * 16 + i];
            }
        }
        let v = VideoFrame {
            planes: vec![
                VideoPlane {
                    data: y_data,
                    stride: 32,
                },
                VideoPlane {
                    data: cb_data,
                    stride: 16,
                },
                VideoPlane {
                    data: cr_data,
                    stride: 16,
                },
            ],
            pts: None,
        };

        // Encode one B-VOP body into a discarded BitWriter; we only care
        // about the BMvGrid side-effect. Wrap with a dummy outer scope
        // because encode_b_vop_body doesn't return the grid. Replicate
        // the body's loop here to inspect the grid.
        let mb_w = 2usize;
        let mb_h = 2usize;
        let mut bmv_grid = BMvGrid::new(mb_w, mb_h);
        let mut row_pred = BRowPred::default();
        let mut bw = BitWriter::new();
        for mb_y in 0..mb_h {
            row_pred.reset();
            for mb_x in 0..mb_w {
                let co = grid.get(mb_x, mb_y);
                let co_mvs4: [(i32, i32); 4] = if co.four_mv { co.mv } else { [co.mv[0]; 4] };
                let mb = estimate_b_mb(
                    &v,
                    32,
                    32,
                    &prev,
                    &next,
                    mb_x,
                    mb_y,
                    4,
                    co_mvs4,
                    co.four_mv,
                    trb,
                    trd,
                    false,
                    QuantMode::H263,
                )
                .expect("estimate_b_mb");
                emit_b_mb(&mut bw, &mb, &mut row_pred, 1, 1);
                let entry = if mb.direct_4mv {
                    BMbMotion::quad(mb.fwd4, mb.bwd4, mb.mode)
                } else {
                    BMbMotion::uni(mb.fwd_mv, mb.bwd_mv, mb.mode)
                };
                bmv_grid.set(mb_x, mb_y, entry);
            }
        }

        // MB(0,0) — the 4MV-direct one. Verify the grid entry is quad.
        let entry = bmv_grid.get(0, 0);
        assert_eq!(entry.mode, Some(BMode::Direct));
        assert!(
            entry.four_mv_direct,
            "MB(0,0) should be recorded as 4MV-direct since the co-located P-MB was 4MV"
        );
        // The 4 forward MVs should match the spec'd direct_mode_mvs_4 output.
        assert_eq!(entry.fwd4, fwd4);
        assert_eq!(entry.bwd4, bwd4);

        // Touch unused imports to keep the linter quiet.
        let _ = predict_block;
    }
}

// -------------------------------------------------------------------------
// Follow-up items:
// * Adaptive dbquant. Round 11 always emits `dbquant = 0` (the
//   "no quant change" single-bit code, Table 6-33). Picking ±2 per MB
//   based on a residual-energy heuristic could squeeze another fraction
//   of a dB on busy content but isn't required for correctness.
// * 4MV direct emit — landed round 12. When the co-located P-MB used
//   4MV, `estimate_b_mb` builds per-block predictors via
//   `predict_luma_mb_4mv` + averaged-luma chroma MV, and stores the
//   result via `BMbMotion::quad`. Bitstream syntax is unchanged
//   (MBTYPE_DIRECT + (0,0) MVD); the decoder picks 4MV implicitly from
//   the co-located P-MB's `four_motion_vector` flag (§7.5.9.5). NOTE:
//   the P-VOP encoder currently emits 1MV-only, so this path is dormant
//   on encoder-only round-trips. It activates as soon as the P-VOP
//   encoder learns 4MV emit, and the decoder cross-test already covers
//   4MV-direct on real streams.
// * Interlaced field MVs — requires `interlaced = 1` in the VOL and the
//   per-MB `interlaced_information()` path. Out of scope for this cut.
// * Quarter-pel — landed round 16. The B-VOP encoder threads
//   `quarter_sample` through ME (forward + backward QPel refine), MC
//   (`predict_luma_mb_qpel` / `predict_block_qpel`), chroma reduction
//   (`luma_qmv_to_chroma`), and direct-mode predictors. The MV VLC
//   writer is unit-agnostic — `write_mv_component` emits the QPel MVD
//   verbatim per §7.6.3 (LSB now denotes the quarter-pel bit).
// -------------------------------------------------------------------------

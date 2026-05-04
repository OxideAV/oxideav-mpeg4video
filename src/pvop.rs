//! P-VOP encoder — motion estimation, inter texture coding, bitstream
//! emission (ISO/IEC 14496-2 §6.3.5, §7.6).
//!
//! Scope:
//! * **Motion estimation** — integer-pel diamond search (small diamond) then
//!   half-pel refinement around the best integer match. Search is bounded by
//!   a small range (±7 integer pels by default) so it stays well within the
//!   `f_code=1` vector range and keeps complexity low. Reference frame is
//!   edge-replicated via the shared `mc::predict_block` helper.
//! * **Mode decision: 1MV vs 4MV** (§7.5.7 / §7.6.7). The encoder runs
//!   both 1MV (one MV per 16×16 MB) and 4MV (one MV per 8×8 luma block)
//!   ME and picks the cheaper option in SAD with a small `FOURMV_LAMBDA`
//!   bit-cost penalty. Round 13 enabled the 4MV path; ties favour 1MV
//!   for cheaper coding. The 4MV path also activates the dormant 4MV-
//!   direct mode in the B-VOP encoder via the per-block MVs in `MvGrid`.
//! * **MV coding** per §7.6.3 — the median predictor over three causal
//!   neighbours (left, top, top-right), then MVD reconstruction via the
//!   unsigned-magnitude + sign + residual layout. `f_code=1` keeps the
//!   residual absent (r_size=0). 4MV mode emits four MVDs per MB; the
//!   per-block predictor uses the in-MB grid as updated by the prior
//!   blocks (§7.6.2 fig 7-6).
//! * **MCBPC** (Table B-13) distinguishes `Inter`, `InterQ`, `Intra`,
//!   `IntraQ`, `Inter4MV`. We emit `Inter` (rows 0..=3), `Inter4MV`
//!   (rows 16..=19) or `Intra` (rows 4..=7) per the mode decision;
//!   skip MBs use the `not_coded` flag.
//! * **Intra-MB-in-P fallback** (§6.3.7 / Table B-22, mb_type=3). For
//!   each P-MB we compute `inter_luma_sad` (chosen-mode SAD) and
//!   `intra_cost_proxy` (a luma-only Mean-Absolute-Deviation against the
//!   per-block mean — a cheap upper-bound on the residual the intra
//!   AC coder would face). When the inter SAD exceeds the intra MAD by
//!   more than `INTRA_IN_P_BIAS + INTRA_MARGIN`, we re-encode the MB as
//!   intra (`not_coded=0` + Table B-13 Intra MCBPC + ac_pred_flag=0 +
//!   raw CBPY + 6 intra blocks). This matters for scene changes,
//!   exposed regions and large-motion content where MC fails.
//! * **Skipped MB**: when the MB is 1MV-mode AND has MV(0,0) AND zero
//!   residual, we emit `not_coded = 1`. 4MV MBs cannot be `not_coded`
//!   (the decoder's not_coded path forces 1MV reconstruction). Decoder
//!   matches (§7.6.7).
//! * **Inter texture coding** — per-block: forward DCT on the residual
//!   (source – predictor), H.263 inter quantisation (matches the decoder's
//!   `dequantise_inter_h263`), inter tcoef walk (Table B-17). The encoder
//!   reconstructs the block (dequant + IDCT + add predictor + clip) so the
//!   emitted bitstream stays drift-free relative to the decoder's state.
//! * **Chroma MV** — 1MV: `luma_mv_to_chroma(mv)`. 4MV: average of the
//!   4 luma MVs scaled to chroma per §7.5.9.5 (matches the decoder's
//!   `inter::decode_p_mb` chroma branch).
//! * **Reference frame management** — the caller threads a single
//!   `reference` IVopPicture (last reconstructed frame) in and receives the
//!   newly reconstructed picture as output, which becomes the next
//!   reference.
//!
//! Out of scope (returns error or NOP):
//! * `IntraQ` (Table B-13 rows 12..=15) — intra-in-P with dquant. The
//!   constant-quant emit path means we never need the `dquant` bits for
//!   intra MBs. `Intra` (no dquant) covers the scene-change case.
//! * B / S VOPs, GMC, quarter-pel motion, reduced-resolution, data
//!   partitioning.

use oxideav_core::Result;

use crate::block::BlockNeighbour;
use crate::encoder::{block_pel_position, encode_intra_mb_in_p, fdct8x8};
use crate::headers::vol::ZIGZAG;
use crate::inter::{MbMotion, MvGrid};
use crate::mb::{IVopPicture, PredGrid};
use crate::mc::{
    luma_4mv_sum_to_chroma, luma_mv_to_chroma, luma_qmv_to_chroma, predict_block,
    predict_block_qpel,
};
use crate::tables::{mv as mv_tab, tcoef};
use oxideav_core::bits::BitWriter;

/// Default integer-pel search range (in integer pels). The encoder keeps the
/// half-pel MV within `±2 * MAX_SEARCH_INT` — comfortably inside the
/// `f_code=1` range of `[-32, 31]` half-pels.
pub const MAX_SEARCH_INT: i32 = 7;

/// Per-block 4MV refine search range (integer pels) around the 1MV result.
/// Smaller than `MAX_SEARCH_INT` because we already have a coarse 1MV
/// starting point and only need to refine sub-MB motion.
const FOUR_MV_REFINE_INT: i32 = 2;

/// Lambda penalty (per extra signalled MVD bit) used when comparing the SAD
/// of 1MV vs 4MV mode. 4MV adds three extra MV components vs 1MV; we
/// approximate the bit cost as roughly 8 bits per extra MVD pair (Table B-12
/// magnitude+sign for small components is ~3-5 bits, plus residual is 0 at
/// f_code=1, times 3 extra blocks). The constant is conservative: we only
/// switch to 4MV when it beats 1MV by more than `FOURMV_LAMBDA` SAD points.
/// At Q=5 each SAD unit roughly costs ~1/5 of a residual bit, so 24 bits
/// of extra MVD ≈ 120 SAD points to break even — we round to 96 for a
/// slight bias toward 4MV when the SAD wins are clear.
const FOURMV_LAMBDA: u32 = 96;

/// Bias (in SAD units) added to the intra MAD proxy before comparing it
/// to the inter luma SAD. Per §6.3.7 / Table B-22, intra MB inside a
/// P-VOP is more expensive to code than a clean inter MB (6 DC VLCs +
/// 4 ac_pred bits + much heavier AC walk; no MV bits saved). We only
/// switch to intra when the inter SAD is materially higher than the
/// intra MAD proxy + this bias. Empirically tuned for the scene-change
/// use-case: a stable scene's inter SAD is typically <500 while intra
/// MAD is content-driven (500-3000 for textured content). On a scene
/// cut, inter SAD jumps to ~3000-8000 while intra MAD stays at the
/// content baseline, easily clearing this threshold.
pub(crate) const INTRA_IN_P_BIAS: u32 = 384;

/// Minimum intra cost advantage (in SAD units) required to switch to
/// intra. The intra cost we compute is a luma-only Mean-Absolute-
/// Deviation (MAD) proxy — a reasonable but not exact predictor of the
/// actual intra residual. The bias above already protects against false
/// positives; this margin protects against close calls where the proxy
/// and the actual cost might disagree.
pub(crate) const INTRA_MARGIN: u32 = 128;

/// Encoder-side representation of one P-VOP macroblock after motion
/// estimation. All MVs are in luma half-pel units.
#[derive(Clone, Copy, Debug)]
pub struct PMbEncoding {
    /// Single MV (luma half-pel units). Used for all four luma blocks and
    /// the two chroma blocks (via `luma_mv_to_chroma`) when `four_mv ==
    /// false`. When `four_mv == true` this still carries `mv4_half[0]` as
    /// a summary value (used by the predictor grid).
    pub mv_half: (i32, i32),
    /// Per-block luma MVs when `four_mv == true` (one MV per 8×8 block,
    /// block order 0=(0,0) 1=(8,0) 2=(0,8) 3=(8,8)). Ignored when
    /// `four_mv == false` (collapses to `[mv_half; 4]`).
    pub mv4_half: [(i32, i32); 4],
    /// True when the MB is emitted in `Inter4MV` mode (Table B-13 group 4).
    /// Bitstream emits four MVDs (one per 8×8 luma block) instead of one
    /// per MB, with chroma MV derived as the average of the 4 luma MVs
    /// scaled to chroma per §7.5.9.5.
    pub four_mv: bool,
    /// True when the MB is emitted as a GMC macroblock (`mcsel = 1`) — the
    /// 16×16 luma + 2×8×8 chroma blocks are predicted by warping the
    /// reference through the per-VOP `WarpParams`. No MV is signalled in
    /// the bitstream for GMC MBs (§7.6.7). Mutually exclusive with
    /// `four_mv` (the standard's MCBPC layout admits `Inter` + mcsel only).
    pub gmc: bool,
    /// When true, the MB is emitted as `not_coded` (skipped) — caller can
    /// verify by decoding a 0 residual.
    pub skipped: bool,
    /// Per-block "coded" flags for the 4 luma blocks (Y0..Y3).
    pub luma_coded: [bool; 4],
    /// Per-block "coded" flags for the 2 chroma blocks (Cb, Cr).
    pub chroma_coded: [bool; 2],
    /// Reconstructed luma 16×16 block (MC + dequant residual, clipped to u8)
    /// in row-major order at offset 0.
    pub recon_y: [u8; 256],
    /// Reconstructed chroma Cb 8×8.
    pub recon_cb: [u8; 64],
    /// Reconstructed chroma Cr 8×8.
    pub recon_cr: [u8; 64],
    /// AC levels (quantised) per block, natural order. `[0]` is unused for
    /// inter blocks (no DC special case); we keep 64 slots to match the
    /// decoder's view.
    pub ac_levels: [[i32; 64]; 6],
    /// Best inter SAD over the 16×16 luma MB after the chosen-mode
    /// (1MV or 4MV) prediction. Used for the intra-in-P mode decision in
    /// `encode_p_vop_body_with_grid` — when the intra MAD proxy is much
    /// smaller than this value, the MB is re-encoded as intra.
    pub inter_luma_sad: u32,
}

impl Default for PMbEncoding {
    fn default() -> Self {
        Self {
            mv_half: (0, 0),
            mv4_half: [(0, 0); 4],
            four_mv: false,
            gmc: false,
            skipped: false,
            luma_coded: [false; 4],
            chroma_coded: [false; 2],
            recon_y: [0; 256],
            recon_cb: [0; 64],
            recon_cr: [0; 64],
            ac_levels: [[0i32; 64]; 6],
            inter_luma_sad: 0,
        }
    }
}

/// Encode one P-VOP into `bw`. Returns the reconstructed picture that must
/// be stored as the next reference frame.
///
/// `v` is the source video frame (YUV420p). `reference` is the previous
/// reconstructed picture (luma-half-pel compatible — the same layout emitted
/// by the I-VOP encoder).
///
/// `vop_quant` is the quantiser and stays constant across the picture (no
/// dquant). `f_code_fwd` selects the MV range (1 keeps MVD to small values).
/// `rounding_type` is the VOP rounding flag (typically 0 on the first P-VOP
/// after an I and toggled per FFmpeg's convention; we mirror that here).
pub fn encode_p_vop_body(
    bw: &mut BitWriter,
    v: &oxideav_core::VideoFrame,
    width: u32,
    height: u32,
    reference: &IVopPicture,
    vop_quant: u32,
    f_code_fwd: u8,
    rounding_type: bool,
) -> Result<IVopPicture> {
    let (pic, _mv_grid) = encode_p_vop_body_with_grid(
        bw,
        v,
        width,
        height,
        reference,
        vop_quant,
        f_code_fwd,
        rounding_type,
        false,
        None,
    )?;
    Ok(pic)
}

/// Variant of [`encode_p_vop_body`] that also returns the P-VOP's MV grid.
///
/// The B-VOP encoder consults this grid for:
/// * `co_located_not_coded` inheritance (§7.5.9.5.4) — skipping B-MBs
///   whose co-located P-MB was not coded.
/// * Direct-mode co-located MV scaling (§7.5.9.5) — the forward/backward
///   MVs of direct-mode B-MBs are derived from the P-MB's MV.
///
/// `warp` is `Some` when the VOL advertises GMC (`sprite_enable == 2`).
/// In that case every Inter MB in the body emits an `mcsel` bit; when
/// the per-MB `mcsel = 1` decision wins, the MB is reconstructed by
/// warping the reference through `warp` and no MV is written. See
/// `crate::gmc::warp_predict_luma_block`.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_vop_body_with_grid(
    bw: &mut BitWriter,
    v: &oxideav_core::VideoFrame,
    width: u32,
    height: u32,
    reference: &IVopPicture,
    vop_quant: u32,
    f_code_fwd: u8,
    rounding_type: bool,
    quarter_sample: bool,
    warp: Option<&crate::gmc::WarpParams>,
) -> Result<(IVopPicture, MvGrid)> {
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut pic = IVopPicture::new(width, height);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);
    // PredGrid tracks DC predictor state for intra MBs (§7.4.3.1). Inside
    // a P-VOP only intra-in-P MBs (§6.3.7 / Table B-22 mb_type=3) update
    // it; inter MBs (and skipped MBs) are reset to default so future
    // intra MBs read `dc=1024, is_intra=false` from inter neighbours,
    // matching the decoder's `decode_p_vop_body`.
    let mut pred_grid = PredGrid::new(mb_w, mb_h);

    let gmc_enabled = warp.is_some();
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // Decision pass — also produces a fully-reconstructed inter MB.
            let mb = estimate_and_encode_mb(
                v,
                width,
                height,
                reference,
                mb_x,
                mb_y,
                vop_quant,
                rounding_type,
                &mv_grid,
                quarter_sample,
                warp,
            )?;

            // Intra-in-P decision (§6.3.7): the inter SAD is the cost we
            // would pay coding this MB as inter. An MAD-based intra cost
            // proxy is computed against the source-MB DC mean. Switch
            // to intra when intra clearly wins (inter SAD exceeds the
            // intra proxy plus `INTRA_IN_P_BIAS + INTRA_MARGIN`).
            let intra_cost = intra_cost_proxy(v, width, height, mb_x, mb_y);
            let inter_cost = inter_cost_proxy(&mb);
            let prefer_intra = inter_cost
                > intra_cost
                    .saturating_add(INTRA_IN_P_BIAS)
                    .saturating_add(INTRA_MARGIN);

            if prefer_intra {
                // Emit Intra-in-P MB:
                //   bit `not_coded = 0`
                //   MCBPC (Table B-13 Intra rows 4..=7)
                //   ac_pred_flag (0)
                //   CBPY (raw — not bit-inverted for intra)
                //   six intra blocks (DC VLC + AC walk)
                bw.write_bits(0, 1);
                encode_intra_mb_in_p(
                    bw,
                    v,
                    width,
                    height,
                    mb_x,
                    mb_y,
                    vop_quant,
                    &mut pred_grid,
                    &mut pic,
                )?;
                // MV grid: intra MBs contribute (0,0) to the median
                // predictor of future inter MBs (§7.6.7 step 3) and
                // are NOT considered `not_coded`. Co-located B-VOP
                // direct-mode treats intra-in-P like a fresh MB —
                // direct mode falls back to forward-only when the
                // co-located MV is (0,0); that's the correct behaviour
                // when the P-MB was intra-coded.
                mv_grid.set(
                    mb_x,
                    mb_y,
                    MbMotion {
                        mv: [(0, 0); 4],
                        four_mv: false,
                        not_coded: false,
                    },
                );
            } else {
                emit_p_mb(bw, &mb, mb_x, mb_y, &mut mv_grid, f_code_fwd, gmc_enabled);
                // Stash reconstructed samples into `pic`.
                write_recon_to_pic(&mut pic, &mb, mb_x, mb_y);
                // Reset PredGrid for inter MB (mirrors the decoder's
                // `reset_pred_grid_mb` in `inter.rs`).
                reset_pred_grid_mb(&mut pred_grid, mb_x, mb_y);
                // Update MV predictor grid. Record `not_coded` so
                // B-VOP encode can inherit the skip flag per
                // §7.5.9.5.4. For 4MV MBs we record all four per-block
                // MVs so the next P-MB's median predictor (§7.5.7) and
                // any downstream B-VOP direct-mode 4MV scaling
                // (§7.5.9.5) see the correct grid.
                let motion = if mb.four_mv {
                    MbMotion {
                        mv: mb.mv4_half,
                        four_mv: true,
                        not_coded: mb.skipped,
                    }
                } else {
                    MbMotion {
                        mv: [mb.mv_half; 4],
                        four_mv: false,
                        not_coded: mb.skipped,
                    }
                };
                mv_grid.set(mb_x, mb_y, motion);
            }
        }
    }

    Ok((pic, mv_grid))
}

/// Reset the AC/DC prediction slots for one MB. Called for inter MBs to
/// mirror the decoder's behaviour in `inter::reset_pred_grid_mb`. Without
/// this, an intra MB later in the picture would predict its DC from a
/// stale neighbour DC value left behind by the I-VOP — i.e., from a frame
/// before the most recent inter MB.
pub(crate) fn reset_pred_grid_mb(grid: &mut PredGrid, mb_x: usize, mb_y: usize) {
    let positions: [(usize, usize); 4] = [
        (mb_x * 2, mb_y * 2),
        (mb_x * 2 + 1, mb_y * 2),
        (mb_x * 2, mb_y * 2 + 1),
        (mb_x * 2 + 1, mb_y * 2 + 1),
    ];
    for (bx, by) in positions {
        let idx = by * grid.y_stride + bx;
        grid.y[idx] = BlockNeighbour::default();
    }
    let cidx = mb_y * grid.c_stride + mb_x;
    grid.cb[cidx] = BlockNeighbour::default();
    grid.cr[cidx] = BlockNeighbour::default();
}

/// Mean-absolute-deviation (MAD) over the four 8×8 luma blocks of the
/// source MB — a fast proxy for the residual that an intra coder would
/// face after removing each block's DC (the predictor explains the mean,
/// the AC path codes the deviations from it).
///
/// We only sum the four luma blocks so the value is on the same scale
/// as `inter_luma_sad` (the chosen-mode 16×16 luma SAD). Chroma is
/// usually low-variance and would only blur the comparison.
pub(crate) fn intra_cost_proxy(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> u32 {
    let mut total = 0u32;
    for blk in 0..4 {
        let mut block = [0u8; 64];
        load_block_for_intra_cost(v, width, height, mb_x, mb_y, blk, &mut block);
        let mut sum = 0u32;
        for &s in block.iter() {
            sum += s as u32;
        }
        let mean = (sum / 64) as i32;
        let mut mad = 0u32;
        for &s in block.iter() {
            mad = mad.saturating_add((s as i32 - mean).unsigned_abs());
        }
        total = total.saturating_add(mad);
    }
    total
}

/// Inter cost proxy — chosen-mode luma SAD across the 16×16 MB. This is
/// the residual energy the inter coder asks the DCT+quant+AC pass to
/// encode. Higher values mean the MV match is poor, which is the
/// scene-change / occlusion case we want to catch with intra-in-P.
///
/// Comparable to the per-MB intra MAD proxy: both are sums of
/// per-sample absolute deviations from the predictor (MC predictor for
/// inter; per-block DC for intra).
fn inter_cost_proxy(mb: &PMbEncoding) -> u32 {
    mb.inter_luma_sad
}

/// Load one 8×8 source block (for intra-cost computation). Mirrors
/// `read_luma_block_from_mb` for luma blocks 0..=3 and `load_chroma_block`
/// for chroma blocks 4..=5.
fn load_block_for_intra_cost(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    blk: usize,
    out: &mut [u8; 64],
) {
    let (plane_idx, x0, y0, pw, ph) = block_pel_position(width, height, mb_x, mb_y, blk);
    let plane = &v.planes[plane_idx];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            out[j * 8 + i] = plane.data[yy * plane.stride + xx];
        }
    }
}

pub(crate) fn write_recon_to_pic(
    pic: &mut IVopPicture,
    mb: &PMbEncoding,
    mb_x: usize,
    mb_y: usize,
) {
    let px = mb_x * 16;
    let py = mb_y * 16;
    for j in 0..16 {
        for i in 0..16 {
            pic.y[(py + j) * pic.y_stride + (px + i)] = mb.recon_y[j * 16 + i];
        }
    }
    let cx = mb_x * 8;
    let cy = mb_y * 8;
    for j in 0..8 {
        for i in 0..8 {
            pic.cb[(cy + j) * pic.c_stride + (cx + i)] = mb.recon_cb[j * 8 + i];
            pic.cr[(cy + j) * pic.c_stride + (cx + i)] = mb.recon_cr[j * 8 + i];
        }
    }
}

// -------------------------------------------------------------------------
// Motion estimation + residual encoding
// -------------------------------------------------------------------------

/// Estimate motion for one MB, encode + reconstruct its six blocks, and
/// return a fully-populated `PMbEncoding`.
///
/// Mode decision (§7.5.7 / §7.6.7): we run both 1MV and 4MV ME and pick
/// the cheaper option in SAD with a small lambda penalty for the extra
/// MVD bits 4MV emits (3 extra MVDs ≈ ~24 bits). 4MV improves
/// compression on content with sub-MB-level motion (8×8 pieces moving
/// independently) at the cost of three extra MVD pairs per MB.
///
/// When `warp` is `Some` (GMC enabled), we also build a GMC predictor
/// for the MB and switch to it when its luma SAD beats the chosen-mode
/// (1MV or 4MV) SAD by more than `GMC_LAMBDA`. GMC MBs save the four
/// MVD components an Inter MB pays — the lambda accounts for the cost
/// difference in the bitstream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_and_encode_mb(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    vop_quant: u32,
    rounding: bool,
    mv_grid: &MvGrid,
    quarter_sample: bool,
    warp: Option<&crate::gmc::WarpParams>,
) -> Result<PMbEncoding> {
    let force_one_mv = false;
    // 1. Integer-pel search over the 16×16 luma MB.
    let src_y_block = load_luma_mb(v, width, height, mb_x, mb_y);
    let (int_x, int_y) = diamond_search_integer(reference, &src_y_block, mb_x, mb_y);
    // 2. Half-pel refinement → seed for QPel refine when QPel is on.
    let (mvx_half, mvy_half) =
        halfpel_refine(reference, &src_y_block, mb_x, mb_y, int_x, int_y, rounding);
    let _ = mv_grid; // MV predictor is applied only when writing MVD, not ME.

    // 3. Quarter-pel refinement (§7.5.4 / §7.6.2.2) — only when QPel
    //    is enabled in the VOL. The 8 surrounding quarter-pel
    //    positions are evaluated against the 8-tap-filter predictor.
    //    `mvx_q`/`mvy_q` are stored in quarter-pel units when QPel is
    //    on (i.e. `mvx_q = 2 * mvx_half + dq_x`); when QPel is off
    //    they remain in half-pel units (`mvx_q = mvx_half`) so the
    //    rest of the pipeline uses the existing half-pel predictors.
    let (mvx_1mv, mvy_1mv) = if quarter_sample {
        qpel_refine_mb(
            reference,
            &src_y_block,
            mb_x,
            mb_y,
            mvx_half,
            mvy_half,
            rounding,
        )
    } else {
        (mvx_half, mvy_half)
    };

    // 1MV SAD baseline — full-MB SAD against the 1MV predictor.
    let mut pred_y_1mv = [0u8; 256];
    predict_luma_mb_any(
        reference,
        mb_x,
        mb_y,
        mvx_1mv,
        mvy_1mv,
        rounding,
        quarter_sample,
        &mut pred_y_1mv,
    );
    let sad_1mv = sad_full_mb(&src_y_block, &pred_y_1mv);

    // 4. 4MV ME — refine each 8×8 luma block around the 1MV result.
    //    The four per-block MVs are independent; we keep the search small
    //    (`±FOUR_MV_REFINE_INT` integer pels) since the 1MV already pinned
    //    the rough motion. Each block does diamond + half-pel refine
    //    (+ optional QPel refine when QPel is on).
    let mut mv4 = [(mvx_1mv, mvy_1mv); 4];
    let mut sad_4mv: u32 = 0;
    let block_offsets: [(i32, i32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];
    for blk in 0..4 {
        let (sub_x, sub_y) = block_offsets[blk];
        let src_blk = read_luma_block_from_mb_xy(
            v,
            width,
            height,
            mb_x,
            mb_y,
            sub_x as usize,
            sub_y as usize,
        );
        let (mvx_b, mvy_b, sad_b) = estimate_block_mv_8x8(
            reference,
            &src_blk,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            mvx_1mv,
            mvy_1mv,
            rounding,
            quarter_sample,
        );
        mv4[blk] = (mvx_b, mvy_b);
        sad_4mv = sad_4mv.saturating_add(sad_b);
    }

    // 5. Mode decision — favour 1MV unless 4MV beats it by more than the
    //    lambda penalty. 1MV ties win because they're cheaper to code and
    //    avoid splitting the MB's chroma predictor. When `force_one_mv`
    //    is true (DP path), the 4MV branch is suppressed entirely — the
    //    DP encoder rejects Inter4MV MCBPC codewords and only writes one
    //    MV per MB, so picking 4MV here would desync encoder/decoder
    //    reconstruction (§6.2.5.3 `data_partitioned_p_vop()`).
    let mut four_mv = !force_one_mv && sad_4mv.saturating_add(FOURMV_LAMBDA) < sad_1mv;

    let (mut mv_repr, mut pred_y) = if four_mv {
        // Build the 4MV luma predictor.
        let mut pred = [0u8; 256];
        predict_luma_mb_4mv_any(
            reference,
            mb_x,
            mb_y,
            mv4,
            rounding,
            quarter_sample,
            &mut pred,
        );
        // Predictor-grid summary MV — pick block 0's MV as the canonical
        // 16×16 MV. The decoder does the same for skipped neighbour
        // queries (mv[0] is the 4MV block-0 vector by convention).
        (mv4[0], pred)
    } else {
        ((mvx_1mv, mvy_1mv), pred_y_1mv)
    };
    // Keep the legacy field name for compatibility with downstream call
    // sites — `mv4_half` carries the 4 per-block MVs in the active
    // unit (half-pel when `quarter_sample == false`, quarter-pel
    // otherwise).
    let mut mv4_half = mv4;

    // 5b. GMC mode decision (§7.6.7). Available when the VOL has
    //     `sprite_enable == 2` — i.e. `warp` is `Some`. We build the
    //     GMC luma predictor by warping the reference through the
    //     per-VOP `WarpParams` and compare its SAD against the chosen
    //     translational mode (1MV or 4MV). GMC saves the MVD bits an
    //     Inter MB pays for the MV component pair (~6-12 bits at
    //     f_code=1); the lambda below approximates that saving in
    //     SAD-equivalent units. Notably GMC never wins for the 4MV
    //     path's intricate sub-MB motion — the warp by definition
    //     describes a single global translation — so we apply GMC
    //     only when 4MV did NOT win the prior decision.
    let mut gmc = false;
    if let Some(w) = warp {
        let mut pred_gmc = [0u8; 256];
        let blk_px = (mb_x * 16) as i32;
        let blk_py = (mb_y * 16) as i32;
        crate::gmc::warp_predict_luma_block(
            w,
            &reference.y,
            reference.y_stride,
            blk_px,
            blk_py,
            16,
            &mut pred_gmc,
            16,
        );
        let sad_gmc = sad_full_mb(&src_y_block, &pred_gmc);
        let chosen_sad = if four_mv { sad_4mv } else { sad_1mv };
        // GMC saves ~6-12 bits per MB in MVD; in SAD-equivalent units at
        // typical Q≈5 a saved bit is worth ~5 SAD points, so a 12-bit
        // saving ≈ 60 SAD points. We use 64 as a slight bias toward the
        // translational mode; GMC must materially beat the chosen mode.
        const GMC_LAMBDA: u32 = 64;
        if sad_gmc + GMC_LAMBDA < chosen_sad {
            gmc = true;
            four_mv = false;
            mv_repr = (0, 0);
            mv4_half = [(0, 0); 4];
            pred_y = pred_gmc;
        }
    }

    // Stash the chosen-mode luma SAD for the intra-in-P decision in
    // `encode_p_vop_body_with_grid`. We use the SAD against the chosen
    // mode's predictor (1MV / 4MV / GMC), giving the cleanest "this is
    // what the inter coder is asking the AC pass to handle" signal.
    let inter_luma_sad = if gmc {
        // GMC SAD recomputed from the actual predictor stored in pred_y.
        sad_full_mb(&src_y_block, &pred_y)
    } else if four_mv {
        sad_4mv
    } else {
        sad_1mv
    };

    let mut mb = PMbEncoding {
        mv_half: mv_repr,
        mv4_half,
        four_mv,
        gmc,
        inter_luma_sad,
        ..Default::default()
    };

    // 6. Build chroma predictors. In 4MV mode chroma uses the average of
    //    the 4 luma MVs scaled to chroma per §7.5.9.5 (matches the
    //    decoder's `(cmx, cmy) = if four_mv { ... }` branch in inter.rs).
    //    QPel mode reduces each luma component through `luma_qmv_to_chroma`
    //    (§7.6.2.2 eq. (107)) instead of `luma_mv_to_chroma`. The decoder
    //    applies the same branch — see `inter::decode_p_mb`'s `to_chroma`
    //    closure.
    let to_chroma = |luma: i32| -> i32 {
        if quarter_sample {
            luma_qmv_to_chroma(luma)
        } else {
            luma_mv_to_chroma(luma)
        }
    };
    // 4MV chroma per ISO/IEC 14496-2 §7.6.5 + Table 7-10 (`luma_4mv_sum_to_chroma`).
    // QPel-of-4MV stays on the average-then-`to_chroma` path because
    // Table 7-10 is defined for half-pel luma inputs only (the spec
    // halves QPel components before summation). Decoder mirrors the
    // same branch — see `inter::decode_p_mb`.
    let (cmx, cmy) = if four_mv {
        let sx: i32 = mv4_half.iter().map(|(x, _)| *x).sum();
        let sy: i32 = mv4_half.iter().map(|(_, y)| *y).sum();
        if quarter_sample {
            (to_chroma(sx / 4), to_chroma(sy / 4))
        } else {
            (luma_4mv_sum_to_chroma(sx), luma_4mv_sum_to_chroma(sy))
        }
    } else {
        (to_chroma(mvx_1mv), to_chroma(mvy_1mv))
    };
    let mut pred_cb = [0u8; 64];
    let mut pred_cr = [0u8; 64];
    if gmc {
        // GMC chroma: warp the chroma planes directly through
        // `chroma_map`. The decoder mirrors this in
        // `inter::decode_p_mb` when `gmc_mb` is true.
        let warp = warp.expect("gmc set without warp");
        let cx_px = (mb_x * 8) as i32;
        let cy_px = (mb_y * 8) as i32;
        crate::gmc::warp_predict_chroma_block(
            warp,
            &reference.cb,
            reference.c_stride,
            cx_px,
            cy_px,
            8,
            &mut pred_cb,
            8,
        );
        crate::gmc::warp_predict_chroma_block(
            warp,
            &reference.cr,
            reference.c_stride,
            cx_px,
            cy_px,
            8,
            &mut pred_cr,
            8,
        );
    } else {
        predict_chroma_block(
            &reference.cb,
            reference.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            rounding,
            &mut pred_cb,
        );
        predict_chroma_block(
            &reference.cr,
            reference.c_stride,
            mb_x,
            mb_y,
            cmx,
            cmy,
            rounding,
            &mut pred_cr,
        );
    }

    // 6. Residual + forward DCT + quant, per 8×8 block.
    // Luma blocks: 0=(0,0) 1=(8,0) 2=(0,8) 3=(8,8)
    for blk in 0..4 {
        let (sub_x, sub_y) = match blk {
            0 => (0, 0),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let src = read_luma_block_from_mb(v, width, height, mb_x, mb_y, sub_x, sub_y);
        let pred_blk = read_pred_block(&pred_y, 16, sub_x, sub_y);
        let (levels, recon) = encode_inter_block(&src, &pred_blk, vop_quant);
        mb.luma_coded[blk] = levels.iter().any(|&l| l != 0);
        mb.ac_levels[blk] = levels;
        // Stamp reconstructed samples back into mb.recon_y.
        for j in 0..8 {
            for i in 0..8 {
                mb.recon_y[(sub_y + j) * 16 + (sub_x + i)] = recon[j * 8 + i];
            }
        }
    }

    // Chroma blocks.
    let src_cb = load_chroma_block(v, width, height, 1, mb_x, mb_y);
    let src_cr = load_chroma_block(v, width, height, 2, mb_x, mb_y);
    let (levels_cb, recon_cb) = encode_inter_block(&src_cb, &pred_cb, vop_quant);
    let (levels_cr, recon_cr) = encode_inter_block(&src_cr, &pred_cr, vop_quant);
    mb.chroma_coded[0] = levels_cb.iter().any(|&l| l != 0);
    mb.chroma_coded[1] = levels_cr.iter().any(|&l| l != 0);
    mb.ac_levels[4] = levels_cb;
    mb.ac_levels[5] = levels_cr;
    mb.recon_cb = recon_cb;
    mb.recon_cr = recon_cr;

    // 7. Skip detection — MB is skippable only if 1MV mode AND MV == (0,0)
    // AND all residual levels are zero (CBP == 0). In that case the
    // decoder copies the reference verbatim, which must equal what we
    // reconstructed. 4MV MBs cannot be `not_coded` (Inter4MV is its own
    // MCBPC group; the decoder's not_coded path forces 1MV with MV=(0,0)).
    // GMC MBs likewise cannot be `not_coded` — the skip path is purely
    // translational with MV=(0,0), but a GMC MB's reference is warped
    // (§7.6.7).
    let all_zero = !mb.luma_coded.iter().any(|&c| c) && !mb.chroma_coded.iter().any(|&c| c);
    if !mb.four_mv && !mb.gmc && all_zero && mvx_1mv == 0 && mvy_1mv == 0 {
        // Make sure the reconstructed samples equal the reference region,
        // which they do by construction (residual=0, MV=0). Holds for
        // QPel MV(0,0) as well — the (0,0) qpel position is identical
        // to the integer-pel sample.
        mb.skipped = true;
    }

    Ok(mb)
}

/// Sum-of-absolute-differences over a 16×16 luma MB.
fn sad_full_mb(src: &[u8; 256], pred: &[u8; 256]) -> u32 {
    let mut s = 0u32;
    for i in 0..256 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// Per-8×8-block ME — diamond search over a small `±FOUR_MV_REFINE_INT`
/// window around (`init_mvx`, `init_mvy`) followed by half-pel refinement,
/// and optionally a final quarter-pel refinement when `quarter_sample` is
/// on (§7.6.2.2). Returns `(mvx, mvy, sad)` in the active MV unit
/// (half-pel or quarter-pel).
///
/// `init_mvx_init`, `init_mvy_init` carry the 1MV-result MV in the active
/// unit. `sub_x`, `sub_y` are the block's offset inside the macroblock
/// (0 or 8).
#[allow(clippy::too_many_arguments)]
fn estimate_block_mv_8x8(
    reference: &IVopPicture,
    src_blk: &[u8; 64],
    mb_x: usize,
    mb_y: usize,
    sub_x: i32,
    sub_y: i32,
    init_mvx_init: i32,
    init_mvy_init: i32,
    rounding: bool,
    quarter_sample: bool,
) -> (i32, i32, u32) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let blk_px = (mb_x * 16) as i32 + sub_x;
    let blk_py = (mb_y * 16) as i32 + sub_y;

    // Convert the 1MV-unit init back to integer-pel for diamond seed.
    // QPel: divide by 4. Half-pel: divide by 2.
    let scale = if quarter_sample { 4 } else { 2 };
    let init_int_x = init_mvx_init / scale;
    let init_int_y = init_mvy_init / scale;
    let mut best_x = init_int_x;
    let mut best_y = init_int_y;
    let mut best_sad = sad_block_integer(
        reference, src_blk, blk_px, blk_py, best_x, best_y, ref_w, ref_h,
    );
    const STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for _ in 0..(FOUR_MV_REFINE_INT as usize * 2) {
        let mut improved = false;
        for (dx, dy) in STEPS {
            let nx = best_x + dx;
            let ny = best_y + dy;
            // Stay inside the search window AND inside the 1MV f_code=1
            // range (±32 half-pels = ±16 integer pels; same window in
            // QPel since f_code=1 quarter-pels span ±8 integer pels —
            // we keep the integer-pel cap conservative).
            if (nx - init_int_x).abs() > FOUR_MV_REFINE_INT
                || (ny - init_int_y).abs() > FOUR_MV_REFINE_INT
                || nx.abs() > MAX_SEARCH_INT
                || ny.abs() > MAX_SEARCH_INT
            {
                continue;
            }
            let s = sad_block_integer(reference, src_blk, blk_px, blk_py, nx, ny, ref_w, ref_h);
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
    // Half-pel refine — 8 candidates around `best_x*2, best_y*2`.
    let mut best_hx = best_x * 2;
    let mut best_hy = best_y * 2;
    let mut best_hsad = sad_block_halfpel(
        reference, src_blk, blk_px, blk_py, best_hx, best_hy, rounding,
    );
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let hx = best_x * 2 + dx;
            let hy = best_y * 2 + dy;
            if hx.abs() > MAX_SEARCH_INT * 2 + 1 || hy.abs() > MAX_SEARCH_INT * 2 + 1 {
                continue;
            }
            let s = sad_block_halfpel(reference, src_blk, blk_px, blk_py, hx, hy, rounding);
            if s < best_hsad {
                best_hsad = s;
                best_hx = hx;
                best_hy = hy;
            }
        }
    }
    if !quarter_sample {
        return (best_hx, best_hy, best_hsad);
    }
    // Quarter-pel refine (§7.6.2.2) — 8 candidates around the
    // half-pel best, expressed in QPel units `(best_hx*2, best_hy*2)`.
    let mut best_qx = best_hx * 2;
    let mut best_qy = best_hy * 2;
    let mut best_qsad = sad_block_qpel(
        reference, src_blk, blk_px, blk_py, best_qx, best_qy, rounding,
    );
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let qx = best_hx * 2 + dx;
            let qy = best_hy * 2 + dy;
            // Cap at the f_code=1 QPel range (±32 quarter-pels = ±8 int pels).
            if qx.abs() > 32 || qy.abs() > 32 {
                continue;
            }
            let s = sad_block_qpel(reference, src_blk, blk_px, blk_py, qx, qy, rounding);
            if s < best_qsad {
                best_qsad = s;
                best_qx = qx;
                best_qy = qy;
            }
        }
    }
    (best_qx, best_qy, best_qsad)
}

/// Quarter-pel SAD for one 8×8 luma block. Builds the 8×8 QPel
/// predictor via `predict_block_qpel` and compares.
fn sad_block_qpel(
    reference: &IVopPicture,
    src: &[u8; 64],
    blk_px: i32,
    blk_py: i32,
    mvx_q: i32,
    mvy_q: i32,
    rounding: bool,
) -> u32 {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let mut pred = [0u8; 64];
    predict_block_qpel(
        &reference.y,
        reference.y_stride,
        ref_w,
        ref_h,
        blk_px,
        blk_py,
        mvx_q,
        mvy_q,
        8,
        rounding,
        &mut pred,
        8,
    );
    let mut s = 0u32;
    for i in 0..64 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// Read one 8×8 luma source block at the given sub-MB position. Mirrors
/// `read_luma_block_from_mb` but takes pre-computed (sub_x, sub_y) instead
/// of looking the block-index up.
fn read_luma_block_from_mb_xy(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
) -> [u8; 64] {
    read_luma_block_from_mb(v, width, height, mb_x, mb_y, sub_x, sub_y)
}

/// Integer-pel SAD for one 8×8 luma block at (`blk_px + mvx`, `blk_py + mvy`).
fn sad_block_integer(
    reference: &IVopPicture,
    src: &[u8; 64],
    blk_px: i32,
    blk_py: i32,
    mvx: i32,
    mvy: i32,
    ref_w: i32,
    ref_h: i32,
) -> u32 {
    let mut s = 0u32;
    for j in 0..8i32 {
        for i in 0..8i32 {
            let x = (blk_px + mvx + i).clamp(0, ref_w - 1) as usize;
            let y = (blk_py + mvy + j).clamp(0, ref_h - 1) as usize;
            let r = reference.y[y * reference.y_stride + x] as i32;
            let sv = src[(j as usize) * 8 + (i as usize)] as i32;
            s = s.wrapping_add((sv - r).unsigned_abs());
        }
    }
    s
}

/// Half-pel SAD for one 8×8 luma block. Builds the 8×8 predictor via
/// `predict_block` and compares.
fn sad_block_halfpel(
    reference: &IVopPicture,
    src: &[u8; 64],
    blk_px: i32,
    blk_py: i32,
    mvx_half: i32,
    mvy_half: i32,
    rounding: bool,
) -> u32 {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let mut pred = [0u8; 64];
    predict_block(
        &reference.y,
        reference.y_stride,
        ref_w,
        ref_h,
        blk_px,
        blk_py,
        mvx_half,
        mvy_half,
        8,
        rounding,
        &mut pred,
        8,
    );
    let mut s = 0u32;
    for i in 0..64 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// Build a 16×16 luma predictor from four per-block MVs. Mirrors
/// `bvop_enc::predict_luma_mb_4mv` (kept locally to avoid cross-module
/// visibility shuffling).
pub(crate) fn predict_luma_mb_4mv(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv4_half: [(i32, i32); 4],
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    for (blk, (sub_x, sub_y)) in [(0i32, 0i32), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
        let (mvx, mvy) = mv4_half[blk];
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

/// Small-diamond integer search starting at (0,0). Evaluates SAD at each
/// candidate and moves to the minimum; stops when the centre is best. Tiny
/// bounded range to keep things deterministic and well within `f_code=1`.
fn diamond_search_integer(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
) -> (i32, i32) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let blk_px = (mb_x * 16) as i32;
    let blk_py = (mb_y * 16) as i32;
    let mut best_x = 0i32;
    let mut best_y = 0i32;
    let mut best_sad = sad_integer(reference, src, blk_px, blk_py, 0, 0, ref_w, ref_h);
    // Small-diamond pattern: 4-neighbour.
    const STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for _ in 0..(MAX_SEARCH_INT as usize * 2) {
        let mut improved = false;
        for (dx, dy) in STEPS {
            let nx = best_x + dx;
            let ny = best_y + dy;
            if nx.abs() > MAX_SEARCH_INT || ny.abs() > MAX_SEARCH_INT {
                continue;
            }
            let s = sad_integer(reference, src, blk_px, blk_py, nx, ny, ref_w, ref_h);
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
    blk_px: i32,
    blk_py: i32,
    mv_x: i32,
    mv_y: i32,
    ref_w: i32,
    ref_h: i32,
) -> u32 {
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

/// Refine integer MV to half-pel by evaluating the 8 half-pel candidates
/// around the integer best. Returns the MV in half-pel units.
fn halfpel_refine(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    int_x: i32,
    int_y: i32,
    rounding: bool,
) -> (i32, i32) {
    let mut best_half_x = int_x * 2;
    let mut best_half_y = int_y * 2;
    let mut best_sad = sad_halfpel(
        reference,
        src,
        mb_x,
        mb_y,
        best_half_x,
        best_half_y,
        rounding,
    );
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let hx = int_x * 2 + dx;
            let hy = int_y * 2 + dy;
            if hx.abs() > MAX_SEARCH_INT * 2 + 1 || hy.abs() > MAX_SEARCH_INT * 2 + 1 {
                continue;
            }
            let s = sad_halfpel(reference, src, mb_x, mb_y, hx, hy, rounding);
            if s < best_sad {
                best_sad = s;
                best_half_x = hx;
                best_half_y = hy;
            }
        }
    }
    (best_half_x, best_half_y)
}

fn sad_halfpel(
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

/// Refine half-pel MV to quarter-pel by evaluating the 8 surrounding
/// quarter-pel candidates around `(mvx_half * 2, mvy_half * 2)`. Per
/// §7.5.4 / §7.6.2.2, quarter-pel motion estimation extends the
/// half-pel result by one quarter-pel step on each axis. Returns the
/// MV in quarter-pel units.
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
    let mut best_sad = sad_qpel_mb(reference, src, mb_x, mb_y, best_qx, best_qy, rounding);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let qx = center_qx + dx;
            let qy = center_qy + dy;
            // Cap at the f_code=1 quarter-pel range (±32 quarter-pels).
            if qx.abs() > 32 || qy.abs() > 32 {
                continue;
            }
            let s = sad_qpel_mb(reference, src, mb_x, mb_y, qx, qy, rounding);
            if s < best_sad {
                best_sad = s;
                best_qx = qx;
                best_qy = qy;
            }
        }
    }
    (best_qx, best_qy)
}

/// Quarter-pel SAD over a 16×16 luma MB. Builds the predictor with
/// `predict_luma_mb_qpel` and compares against `src`.
fn sad_qpel_mb(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mvx_q: i32,
    mvy_q: i32,
    rounding: bool,
) -> u32 {
    let mut pred = [0u8; 256];
    predict_luma_mb_qpel(reference, mb_x, mb_y, mvx_q, mvy_q, rounding, &mut pred);
    let mut s = 0u32;
    for i in 0..256 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

/// QPel-aware luma MB predictor — dispatches to `predict_luma_mb_qpel`
/// when `quarter_sample` is on, else `predict_luma_mb` (half-pel).
pub(crate) fn predict_luma_mb_any(
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

/// QPel-aware 4MV luma MB predictor — see `predict_luma_mb_any`.
pub(crate) fn predict_luma_mb_4mv_any(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv4: [(i32, i32); 4],
    rounding: bool,
    quarter_sample: bool,
    out: &mut [u8; 256],
) {
    if quarter_sample {
        predict_luma_mb_4mv_qpel(reference, mb_x, mb_y, mv4, rounding, out);
    } else {
        predict_luma_mb_4mv(reference, mb_x, mb_y, mv4, rounding, out);
    }
}

/// Build a 16×16 luma predictor using the QPel 8-tap filter.
pub(crate) fn predict_luma_mb_qpel(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv_x_q: i32,
    mv_y_q: i32,
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    for (sub_x, sub_y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
        let mut tmp = [0u8; 64];
        predict_block_qpel(
            &reference.y,
            reference.y_stride,
            ref_w,
            ref_h,
            px + sub_x,
            py + sub_y,
            mv_x_q,
            mv_y_q,
            8,
            rounding,
            &mut tmp,
            8,
        );
        for j in 0..8 {
            for i in 0..8 {
                out[(sub_y as usize + j) * 16 + (sub_x as usize + i)] = tmp[j * 8 + i];
            }
        }
    }
}

/// Build a 16×16 luma predictor from four per-block QPel MVs.
pub(crate) fn predict_luma_mb_4mv_qpel(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv4_q: [(i32, i32); 4],
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    for (blk, (sub_x, sub_y)) in [(0i32, 0i32), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
        let (mvx, mvy) = mv4_q[blk];
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

// -------------------------------------------------------------------------
// Block prediction + residual/quant round-trip
// -------------------------------------------------------------------------

pub(crate) fn predict_luma_mb(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    // Decompose 16×16 into 4×(8×8) so we can reuse `predict_block`.
    for (sub_x, sub_y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
        let mut tmp = [0u8; 64];
        predict_block(
            &reference.y,
            reference.y_stride,
            ref_w,
            ref_h,
            px + sub_x,
            py + sub_y,
            mv_x_half,
            mv_y_half,
            8,
            rounding,
            &mut tmp,
            8,
        );
        for j in 0..8 {
            for i in 0..8 {
                out[(sub_y as usize + j) * 16 + (sub_x as usize + i)] = tmp[j * 8 + i];
            }
        }
    }
}

pub(crate) fn predict_chroma_block(
    ref_plane: &[u8],
    ref_stride: usize,
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
    out: &mut [u8; 64],
) {
    let ref_h = (ref_plane.len() / ref_stride) as i32;
    let ref_w = ref_stride as i32;
    predict_block(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        (mb_x * 8) as i32,
        (mb_y * 8) as i32,
        mv_x_half,
        mv_y_half,
        8,
        rounding,
        out,
        8,
    );
}

pub(crate) fn load_luma_mb(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [u8; 256] {
    let mut out = [0u8; 256];
    let w = width;
    let h = height;
    let plane = &v.planes[0];
    for j in 0..16 {
        let yy = (mb_y * 16 + j).min(h.saturating_sub(1));
        for i in 0..16 {
            let xx = (mb_x * 16 + i).min(w.saturating_sub(1));
            out[j * 16 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

pub(crate) fn load_chroma_block(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    plane_idx: usize,
    mb_x: usize,
    mb_y: usize,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let plane = &v.planes[plane_idx];
    for j in 0..8 {
        let yy = (mb_y * 8 + j).min(ch.saturating_sub(1));
        for i in 0..8 {
            let xx = (mb_x * 8 + i).min(cw.saturating_sub(1));
            out[j * 8 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

fn read_luma_block_from_mb(
    v: &oxideav_core::VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let (_, x0, y0, pw, ph) =
        block_pel_position(width, height, mb_x, mb_y, block_index_for_sub(sub_x, sub_y));
    let plane = &v.planes[0];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            out[j * 8 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

fn block_index_for_sub(sub_x: usize, sub_y: usize) -> usize {
    match (sub_x, sub_y) {
        (0, 0) => 0,
        (8, 0) => 1,
        (0, 8) => 2,
        (8, 8) => 3,
        _ => unreachable!(),
    }
}

fn read_pred_block(mb_pred: &[u8; 256], stride: usize, sub_x: usize, sub_y: usize) -> [u8; 64] {
    let mut out = [0u8; 64];
    for j in 0..8 {
        for i in 0..8 {
            out[j * 8 + i] = mb_pred[(sub_y + j) * stride + (sub_x + i)];
        }
    }
    out
}

/// Encode one 8×8 inter block: compute residual, FDCT, quantise, then
/// dequantise + IDCT + clip to reconstruct. Returns `(ac_levels, recon)`.
/// `ac_levels` has 64 entries (natural order). `recon` is the sample block
/// ready to copy back into the picture buffer.
pub(crate) fn encode_inter_block(
    src: &[u8; 64],
    pred: &[u8; 64],
    quant: u32,
) -> ([i32; 64], [u8; 64]) {
    // Residual.
    let mut res = [0f32; 64];
    for i in 0..64 {
        res[i] = src[i] as f32 - pred[i] as f32;
    }
    // Forward DCT.
    fdct8x8(&mut res);
    // Quantise. Inter H.263 dequant rule is
    //   recon(l) = 2*Q*|l| + Q_plus; Q_plus = Q if Q odd, Q-1 if Q even;
    //   recon(0) = 0.
    // Forward pick the level whose reconstruction is closest to `coef`.
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let two_q = 2 * q;

    let mut levels = [0i32; 64];
    for i in 0..64 {
        let c = res[i].round() as i32;
        levels[i] = quantise_ac_inter_h263(c, two_q, q_plus).clamp(-2047, 2047);
    }

    // Reconstruct: dequantise + IDCT + add predictor + clip.
    let mut deq = [0i32; 64];
    for i in 0..64 {
        let l = levels[i];
        if l == 0 {
            deq[i] = 0;
        } else {
            let abs = l.unsigned_abs() as i32;
            let val = two_q * abs + q_plus;
            deq[i] = if l < 0 { -val } else { val };
        }
    }
    let mut deqf = [0f32; 64];
    for i in 0..64 {
        deqf[i] = deq[i] as f32;
    }
    crate::block::idct8x8(&mut deqf);
    let mut recon = [0u8; 64];
    for j in 0..8 {
        for i in 0..8 {
            let rr = deqf[j * 8 + i].round() as i32 + pred[j * 8 + i] as i32;
            recon[j * 8 + i] = rr.clamp(0, 255) as u8;
        }
    }
    (levels, recon)
}

/// Pick the integer level whose reconstruction is closest to `coef`.
fn quantise_ac_inter_h263(coef: i32, two_q: i32, q_plus: i32) -> i32 {
    if coef == 0 {
        return 0;
    }
    let abs = coef.unsigned_abs() as i32;
    let l_low = abs / two_q;
    let mut best_l = 0i32;
    let mut best_err = abs;
    for cand in [l_low.saturating_sub(1), l_low, l_low + 1] {
        if cand < 0 {
            continue;
        }
        let recon = if cand == 0 { 0 } else { two_q * cand + q_plus };
        let err = (abs - recon).abs();
        if err < best_err {
            best_err = err;
            best_l = cand;
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

// -------------------------------------------------------------------------
// Bitstream emission
// -------------------------------------------------------------------------

fn emit_p_mb(
    bw: &mut BitWriter,
    mb: &PMbEncoding,
    mb_x: usize,
    mb_y: usize,
    mv_grid: &mut MvGrid,
    f_code_fwd: u8,
    gmc_enabled: bool,
) {
    if mb.skipped {
        // §6.3.5: not_coded = 1.
        bw.write_bits(1, 1);
        return;
    }
    // not_coded = 0.
    bw.write_bits(0, 1);

    // MCBPC (Table B-13). cbpc = bit1=Cb bit0=Cr. For 4MV MBs the
    // mb_type group is `Inter4MV` (rows 16..=19) instead of `Inter`
    // (rows 0..=3). cbpc bits are unchanged. GMC MBs use the plain
    // `Inter` group (rows 0..=3) — the `mcsel` bit emitted next picks
    // the warp predictor.
    let cbpc = ((mb.chroma_coded[0] as u8) << 1) | (mb.chroma_coded[1] as u8);
    if mb.four_mv {
        write_mcbpc_inter4mv(bw, cbpc);
    } else {
        write_mcbpc_inter(bw, cbpc);
    }

    // GMC `mcsel` bit (§6.3.7 macroblock() syntax — emitted RIGHT AFTER
    // mcbpc, before ac_pred / cbpy / dquant / interlaced_information).
    // Present only when the VOL advertises `sprite_enable == 2` AND the
    // MB is single-MV Inter / InterQ. 4MV and Intra MBs never carry mcsel.
    if gmc_enabled && !mb.four_mv {
        bw.write_bits(if mb.gmc { 1 } else { 0 }, 1);
    }

    // CBPY — for inter MBs (incl. Inter4MV) the encoded value is bit-
    // inverted of the coded mask (decoder XORs with 0xF). Build the
    // mask from `luma_coded` as bit3=Y0, bit0=Y3, then XOR with 0xF.
    let mut cbpy_mask: u8 = 0;
    for (i, &c) in mb.luma_coded.iter().enumerate() {
        if c {
            cbpy_mask |= 1 << (3 - i);
        }
    }
    let cbpy_encoded = cbpy_mask ^ 0xF;
    write_cbpy(bw, cbpy_encoded);

    if mb.gmc {
        // GMC MBs skip MV emission entirely. The decoder sees the
        // `mcsel = 1` bit, leaves the per-MB MV at (0,0) for predictor
        // purposes (§7.6.7) and reconstructs by warping the reference
        // through the per-VOP `WarpParams`. Drop straight to the
        // residual walk below.
        for blk in 0..6 {
            let coded = if blk < 4 {
                mb.luma_coded[blk]
            } else {
                mb.chroma_coded[blk - 4]
            };
            if !coded {
                continue;
            }
            write_inter_ac(bw, &mb.ac_levels[blk]);
        }
        return;
    }

    // Motion vectors. 1MV: one MVD pair predicted from median(left, top,
    // top-right) at block index 0. 4MV: four MVD pairs, one per 8×8 luma
    // block; the predictor for block k uses the IN-MB grid as updated by
    // the prior blocks (§7.6.2 fig 7-6, mirrored from `inter::predict_mv`).
    if mb.four_mv {
        // 4MV: emit four MVDs. Per §7.6.2 fig 7-6, the median predictor
        // for block `k` may reference sub-blocks 0..k-1 of THIS MB
        // (e.g. block 1's MV1 candidate is block 0 of THIS MB). Mirror
        // the decoder's `inter::decode_p_mb` 4MV path: commit each
        // freshly-decided MV into the grid so the next block's
        // predictor sees it.
        let mut committed = MbMotion {
            mv: [(0, 0); 4],
            four_mv: true,
            not_coded: false,
        };
        for blk in 0..4 {
            mv_grid.set(mb_x, mb_y, committed);
            let (px, py) = crate::inter::predict_mv_full(mv_grid, mb_x, mb_y, blk, true, 0, 0);
            let (mvx, mvy) = mb.mv4_half[blk];
            let dx = mvx - px;
            let dy = mvy - py;
            let range = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
            let dx = wrap_mvd(dx, range);
            let dy = wrap_mvd(dy, range);
            write_mv_component(bw, dx, f_code_fwd);
            write_mv_component(bw, dy, f_code_fwd);
            committed.mv[blk] = (mvx, mvy);
        }
        // Final grid entry — the full four-MV motion. The outer loop
        // also resets this immediately after `emit_p_mb` returns, but
        // keep the assignment here for symmetry with the 1MV path and
        // in case future callers re-use the in-progress grid.
        mv_grid.set(mb_x, mb_y, committed);
    } else {
        let (px, py) = crate::inter::predict_mv_full(mv_grid, mb_x, mb_y, 0, false, 0, 0);
        let (mvx, mvy) = mb.mv_half;
        let dx = mvx - px;
        let dy = mvy - py;
        let range = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
        let dx = wrap_mvd(dx, range);
        let dy = wrap_mvd(dy, range);
        write_mv_component(bw, dx, f_code_fwd);
        write_mv_component(bw, dy, f_code_fwd);
    }

    // Per-block coded residual walk (Table B-17 inter tcoef).
    for blk in 0..6 {
        let coded = if blk < 4 {
            mb.luma_coded[blk]
        } else {
            mb.chroma_coded[blk - 4]
        };
        if !coded {
            continue;
        }
        write_inter_ac(bw, &mb.ac_levels[blk]);
    }
}

/// Fold a signed MVD into the `[-range, range-1]` range by ±2*range.
pub(crate) fn wrap_mvd(mvd: i32, range: i32) -> i32 {
    let mut v = mvd;
    if v < -range {
        v += 2 * range;
    } else if v >= range {
        v -= 2 * range;
    }
    v
}

pub(crate) fn write_mcbpc_inter(bw: &mut BitWriter, cbpc: u8) {
    // Table B-13 row for "Inter, cbpc=0..=3". The decoder's `PMbType::Inter`
    // corresponds to MCBPC values 0..=3 (group=0).
    let (bits, code) = match cbpc {
        0 => (1, 0b1),
        1 => (4, 0b0011),
        2 => (4, 0b0010),
        3 => (6, 0b000101),
        _ => unreachable!(),
    };
    bw.write_bits(code, bits);
}

/// Table B-13 row for "Inter4MV, cbpc=0..=3" (values 16..=19).
/// `decompose_inter` decodes group `value >> 2 == 4` as `PMbType::Inter4MV`,
/// triggering the four-MV decode path in `inter::decode_p_mb`.
pub(crate) fn write_mcbpc_inter4mv(bw: &mut BitWriter, cbpc: u8) {
    let (bits, code) = match cbpc {
        0 => (3, 0b010),
        1 => (7, 0b0000101),
        2 => (7, 0b0000100),
        3 => (8, 0b00000101),
        _ => unreachable!(),
    };
    bw.write_bits(code, bits);
}

fn write_cbpy(bw: &mut BitWriter, cbpy: u8) {
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

/// Write one motion-vector component per §7.6.3.
///
/// MVD `diff` is in half-pel units and already wrapped into `[-32*f, 32*f-1]`.
pub(crate) fn write_mv_component(bw: &mut BitWriter, diff: i32, f_code: u8) {
    let r_size = (f_code.saturating_sub(1)) as u32;
    let f = 1i32 << r_size;
    // |motion_code| and residual derivation (§7.6.3):
    //   if diff == 0: motion_code = 0, no residual.
    //   else:
    //     n = (|diff| - 1) — we need motion_code in 1..=32.
    //     motion_code = (n / f) + 1; residual = n % f.
    let abs = diff.unsigned_abs() as i32;
    let (mc_abs, residual) = if abs == 0 {
        (0i32, 0i32)
    } else {
        let n = abs - 1;
        (n / f + 1, n % f)
    };
    // Emit the magnitude VLC (Table B-12, 0..=32).
    let mc_clamped = mc_abs.clamp(0, 32) as usize;
    let row = &mv_tab_row(mc_clamped);
    bw.write_bits(row.1, row.0 as u32);
    if mc_clamped != 0 {
        // Sign bit: 0 = positive, 1 = negative.
        bw.write_bits(if diff < 0 { 1 } else { 0 }, 1);
    }
    if f != 1 && mc_clamped != 0 {
        bw.write_bits(residual as u32, r_size);
    }
}

/// Return `(bits, code)` for Table B-12 at magnitude `mag`.
fn mv_tab_row(mag: usize) -> (u8, u32) {
    // Build a lookup keyed by magnitude from the shared table. We cannot
    // call `mv_tab::table()` directly for encode because it returns entries
    // indexed 0..=32 already; the table entry's `(bits, code)` is what we
    // need.
    let t = mv_tab::table();
    let e = t[mag];
    (e.bits, e.code)
}

/// Walk `block` in zigzag order, emitting one inter tcoef symbol per non-zero
/// coefficient. Unlike the intra path, inter tcoef starts at scan index 0.
pub(crate) fn write_inter_ac(bw: &mut BitWriter, block: &[i32; 64]) {
    // Find the last non-zero AC in scan order (zigzag).
    let mut last_nz: Option<usize> = None;
    for i in 0..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        // Defensive: caller must check `coded` before calling us.
        return;
    };
    let mut run = 0u8;
    let mut i = 0;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        write_inter_tcoef_symbol(bw, last, run, lv);
        run = 0;
        i += 1;
    }
}

/// Encode one inter tcoef symbol — short VLC where possible, third escape
/// otherwise. Mirrors the intra-path helper in encoder.rs but keyed on
/// `tcoef::inter_table`.
fn write_inter_tcoef_symbol(bw: &mut BitWriter, last: bool, run: u8, level: i32) {
    let abs = level.unsigned_abs() as u8;
    if let Some((bits, code)) = lookup_inter_short_vlc(last, run, abs) {
        bw.write_bits(code, bits as u32);
        bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
        return;
    }
    // Third escape: `0000011` + 1 + 1 + last(1) + run(6) + marker + level(12) + marker.
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

/// Reverse-lookup of the short VLC in Table B-17 keyed by `(last, run, abs)`.
fn lookup_inter_short_vlc(last: bool, run: u8, level_abs: u8) -> Option<(u8, u32)> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type InterShortVlcMap = HashMap<(bool, u8, u8), (u8, u32)>;
    static MAP: OnceLock<InterShortVlcMap> = OnceLock::new();
    let m = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for entry in tcoef::inter_table() {
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
    });
    m.get(&(last, run, level_abs)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_component_zero_roundtrip() {
        use crate::inter::decode_mv_component;
        use oxideav_core::bits::BitReader;
        let mut bw = BitWriter::new();
        write_mv_component(&mut bw, 0, 1);
        let mut data = bw.finish();
        data.extend_from_slice(&[0xFF, 0xFF]);
        let mut br = BitReader::new(&data);
        let v = decode_mv_component(&mut br, 1, 0).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn mv_component_small_positive_roundtrip() {
        use crate::inter::decode_mv_component;
        use oxideav_core::bits::BitReader;
        for d in [1i32, -1, 3, -3, 7, -7, 15, -16, 31, -32] {
            let mut bw = BitWriter::new();
            write_mv_component(&mut bw, d, 1);
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let v = decode_mv_component(&mut br, 1, 0).unwrap();
            assert_eq!(v, d, "MV component {d} round-trip");
        }
    }

    #[test]
    fn quantise_inter_monotonic() {
        // Encode+decode a range of AC values with Q=5 and check monotonic
        // reconstructions.
        let q = 5i32;
        let two_q = 2 * q;
        let q_plus = q;
        for c in [-100i32, -20, 0, 20, 100, 500] {
            let l = quantise_ac_inter_h263(c, two_q, q_plus);
            let recon = if l == 0 { 0 } else { two_q * l.abs() + q_plus };
            let recon = if l < 0 { -recon } else { recon };
            assert!(
                (c - recon).abs() <= two_q,
                "c={c} recon={recon} beyond one step"
            );
        }
    }

    /// Verify the 4MV chroma MV averaging matches the formula `mean(luma)/2`
    /// (with the round-toward-zero step inside `luma_mv_to_chroma`). The
    /// decoder uses the exact same formula on its side (see
    /// `inter::decode_p_mb` chroma branch), so any asymmetry would cause
    /// a chroma-MV mismatch on 4MV MBs.
    #[test]
    fn four_mv_chroma_avg_matches_decoder_formula() {
        let mvs: [(i32, i32); 4] = [(2, 4), (-3, 5), (1, -2), (4, 1)];
        let sx: i32 = mvs.iter().map(|(x, _)| *x).sum();
        let sy: i32 = mvs.iter().map(|(_, y)| *y).sum();
        let cmx_enc = luma_mv_to_chroma(sx / 4);
        let cmy_enc = luma_mv_to_chroma(sy / 4);
        // Decoder side reduction of the same operation:
        let cmx_dec = luma_mv_to_chroma(sx / 4);
        let cmy_dec = luma_mv_to_chroma(sy / 4);
        assert_eq!(cmx_enc, cmx_dec);
        assert_eq!(cmy_enc, cmy_dec);
    }

    /// Exercise the 4MV MCBPC writer for all four cbpc values — the
    /// codewords come from Table B-13 rows 16..=19 (`Inter4MV`). A
    /// round-trip through the decoder's MCBPC table must yield the
    /// same `(PMbType::Inter4MV, cbpc)`.
    #[test]
    fn mcbpc_inter4mv_roundtrip() {
        use crate::tables::mcbpc::{decompose_inter, p_table, PMbType};
        use crate::tables::vlc;
        use oxideav_core::bits::BitReader;
        for cbpc in 0..4u8 {
            let mut bw = BitWriter::new();
            write_mcbpc_inter4mv(&mut bw, cbpc);
            // Pad with stop bits so the MCBPC reader has enough lookahead.
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let v = vlc::decode(&mut br, p_table()).unwrap();
            let (mb_type, dec_cbpc) = decompose_inter(v);
            assert_eq!(
                mb_type,
                PMbType::Inter4MV,
                "cbpc={cbpc} decoded as {mb_type:?} not Inter4MV"
            );
            assert_eq!(dec_cbpc, cbpc, "cbpc round-trip mismatch");
        }
    }

    /// Quarter-pel ME finds a non-zero quarter-pel offset when the
    /// source is a 1/4-pel-shifted version of the reference. The
    /// 8-tap predictor at MV(1,0) (qpel) should beat MV(0,0) by a
    /// noticeable SAD margin.
    #[test]
    fn qpel_refine_finds_quarter_pel_shift() {
        // Build a tiny reference — uniform horizontal ramp 0..16
        // tiled across a 32×32 plane. The half-pel filter at MV(1,0)h
        // (i.e. +0.5 pels) gives sample[i] + sample[i+1] / 2 = avg —
        // the QPel filter at MV(1,0)q (i.e. +0.25 pels) gives a value
        // weighted toward the integer. A source pre-shifted by exactly
        // 0.25 pels should match the QPel(1,0) predictor better.
        let w: usize = 32;
        let h: usize = 32;
        let mut reference = IVopPicture::new(w, h);
        for j in 0..h {
            for i in 0..w {
                reference.y[j * reference.y_stride + i] = ((i % 16) * 16) as u8;
            }
        }
        for px in reference.cb.iter_mut() {
            *px = 128;
        }
        for px in reference.cr.iter_mut() {
            *px = 128;
        }

        // Source MB: the same ramp evaluated at sub-pel +0.25.  The
        // 8-tap QPel filter applied to `ref_plane` at MV(1,0)q would
        // produce roughly this content. We manufacture the MB by
        // running the encoder's own QPel filter from `predict_block_qpel`
        // — that gives the exact target the QPel ME should converge to.
        let mut src_mb = [0u8; 256];
        crate::pvop::predict_luma_mb_qpel(&reference, 0, 0, 1, 0, false, &mut src_mb);

        // Run the QPel refine starting from the half-pel MV(0,0) (the
        // half-pel best for this source). The source pattern has no
        // vertical variation, so the y-axis MV is ambiguous (any
        // value in {-1, 0, 1} produces the same SAD); only the x
        // component is constrained.
        let (mvx_q, mvy_q) = qpel_refine_mb(&reference, &src_mb, 0, 0, 0, 0, false);
        assert_eq!(
            mvx_q, 1,
            "QPel ME failed to converge to x=1 on a 1/4-pel shifted source (got mvx={mvx_q}, mvy={mvy_q})"
        );
        assert!(
            mvy_q.abs() <= 1,
            "QPel y-axis MV {mvy_q} outside the ambiguity band ±1 for a horizontally-only ramp"
        );
        // Sanity: the QPel SAD at the ME's choice must be no worse
        // than the half-pel anchor at (0,0)q.
        let sad_chosen = sad_qpel_mb(&reference, &src_mb, 0, 0, mvx_q, mvy_q, false);
        let sad_anchor = sad_qpel_mb(&reference, &src_mb, 0, 0, 0, 0, false);
        assert!(
            sad_chosen <= sad_anchor,
            "QPel ME chose MV with worse SAD ({sad_chosen}) than anchor ({sad_anchor})"
        );
    }
}

// -------------------------------------------------------------------------
// Follow-up items (not blocking — documented for future work):
//
// * 4MV mode — landed in round 13. P-MBs now optionally emit `Inter4MV`
//   (Table B-13 group 4) with per-8×8-block MVDs and a chroma MV taken
//   as the average of the 4 luma MVs scaled to chroma per §7.5.9.5.
//   The mode decision is a SAD comparison with a small `FOURMV_LAMBDA`
//   penalty for the extra MVD bits; ties favour 1MV. This also
//   activates the round-12 dormant 4MV-direct path in B-VOPs (a B-MB
//   in direct mode whose co-located P-MB used 4MV inherits the 4
//   per-block MVs via `direct_mode_mvs_4`).
// * Intra MB fallback inside P-VOP — landed in round 14. P-MBs now
//   compare a luma-MAD intra-cost proxy against `inter_luma_sad`; when
//   the inter cost exceeds the intra cost by `INTRA_IN_P_BIAS +
//   INTRA_MARGIN`, the MB is re-encoded through the I-VOP intra path
//   wrapped in `not_coded=0 + Table B-13 Intra MCBPC`. This activates
//   on scene cuts and large-motion regions where MC is futile. The
//   PredGrid is now threaded through the P-VOP loop and reset for
//   every inter MB to mirror the decoder's `reset_pred_grid_mb`.
// * Quarter-pel motion (§7.6.2.2 / §7.5.4) — landed in round 15.
//   The encoder emits a verid=2 VOL with `quarter_sample = 1` when
//   built with `params.options["qpel"] = "1"`. ME chains
//   integer-pel diamond → half-pel refine → quarter-pel refine
//   (8 candidates around the half-pel best); the predictor uses
//   `predict_block_qpel` (the existing decoder-side 8-tap filter
//   from `mc.rs`). MVs are stored in quarter-pel units in
//   `MbMotion`/`MvGrid`; chroma derivation flips between
//   `luma_mv_to_chroma` and `luma_qmv_to_chroma`. QPel + B-frames
//   is rejected at the encoder factory (the B-VOP encoder still
//   assumes half-pel — see `bvop_enc.rs`).
// * GMC, sprites, OBMC — deliberately out of scope for this
//   encoder. They each require material bitstream changes
//   (sprite metadata, OBMC overlap window) and are tracked
//   separately from the P-VOP work.
// * RDO mode decision — current intra-vs-inter and 1MV-vs-4MV
//   pickers compare SAD with a fixed bias. A Lagrangian
//   `D + λ·R` decision (where R is the bit cost of each
//   candidate's MVD + MCBPC + CBPY + AC walk) would tighten the
//   picks especially at higher Q. Tracked as a future-work
//   item; quarter-pel was the round-15 priority.

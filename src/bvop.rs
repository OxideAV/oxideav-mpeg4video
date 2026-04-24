//! B-VOP (bidirectional) decode — ISO/IEC 14496-2 §7.6.5.
//!
//! Per-MB flow:
//! 1. `modb` VLC (Table B-16). If skipped, the MB inherits direct-mode
//!    prediction with zero MVD — emit co-located P-VOP reference region.
//! 2. `mbtype` VLC (Table B-18) — Direct / Forward / Backward / Interpolated.
//! 3. If MODB=="00", `cbpb` (6 bits, one per block — Y0..Y3, Cb, Cr).
//! 4. MVs:
//!    * Direct mode — scales the co-located P-VOP MV by (TRB/TRD) and adds
//!      an optional small delta `mvd_b` (single MV pair).
//!    * Forward mode — decodes `mvd_forward` via the standard MV VLC.
//!    * Backward mode — decodes `mvd_backward`.
//!    * Interpolated — decodes both `mvd_forward` and `mvd_backward`.
//! 5. Texture residual per coded block.
//! 6. Motion compensation: forward-only, backward-only, or average of the
//!    two predictions, plus optional residual add.
//!
//! Frame types in this decoder's two-reference model:
//!   * `prev_ref` — the most recent past I/P VOP (forward reference).
//!   * `next_ref` — the most recent future I/P VOP (backward reference).
//!
//! In decode order, the future P-VOP is decoded BEFORE the B-VOPs that
//! reference it, so by the time we hit a B-VOP both references are ready.

use oxideav_core::{Error, Result};

use crate::block::{decode_inter_ac, reconstruct_inter_block};
use crate::headers::vol::{VideoObjectLayer, ZIGZAG};
use crate::headers::vop::VideoObjectPlane;
use crate::inter::{decode_mv_component, MbMotion, MvGrid};
use crate::mb::{IVopPicture, PredGrid};
use crate::mc::{luma_mv_to_chroma, luma_qmv_to_chroma, predict_block, predict_block_qpel};
use crate::tables::{bvop as bvop_tab, vlc};
use oxideav_core::bits::BitReader;

/// B-VOP MB prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BMode {
    Direct,
    Forward,
    Backward,
    Interpolated,
    /// `not_coded == 1` skip path — spec treats it as Direct mode with zero
    /// delta.
    Skipped,
}

/// Per-MB B-VOP state recorded for MV prediction of later MBs.
///
/// B-VOPs use independent MV prediction streams for forward and backward
/// MVs (§7.6.5) — the forward MV of the current MB is predicted only from
/// forward MVs of previous MBs. The grid is threaded separately for each.
#[derive(Clone, Copy, Debug, Default)]
pub struct BMbMotion {
    pub fwd: (i32, i32),
    pub bwd: (i32, i32),
    pub mode: Option<BMode>,
}

/// Compute TRB and TRD for direct-mode scaling.
///
/// `TRD` is the temporal distance (in VOP time-increment ticks) between the
/// two reference frames. `TRB` is the distance from the previous reference
/// to the current B-VOP. Both are positive; if the two references happen to
/// carry the same timestamp we fall back to `TRD = 1, TRB = 1` to avoid
/// division by zero — the resulting 1:1 ratio reduces direct mode to a
/// plain averaging of the co-located vector.
pub fn trb_trd(prev_time: i64, cur_time: i64, next_time: i64) -> (i32, i32) {
    let trd = (next_time - prev_time).max(1) as i32;
    let trb = (cur_time - prev_time).max(1) as i32;
    (trb, trd)
}

/// Compute direct-mode forward and backward MVs from a co-located P-VOP
/// MV (§7.5.9.5.2 equations in the committee draft).
///
/// Per the ISO spec:
/// ```text
/// MVFx = (TRB * MVx) / TRD + MVDx
/// MVBx = (MVDx == 0) ? ((TRB - TRD) * MVx) / TRD : MVFx - MVx
/// MVFy = (TRB * MVy) / TRD + MVDy
/// MVBy = (MVDy == 0) ? ((TRB - TRD) * MVy) / TRD : MVFy - MVy
/// ```
/// The decision on which backward formula to use is **per-component**.
///
/// `co_mv` is the MV of the corresponding MB in the most recently decoded
/// I- or P-VOP (the backward reference). `trb, trd` come from `trb_trd()`.
/// `delta` is the optional small `mvd_b` vector (read only for direct mode).
///
/// Returns `(fwd_mv, bwd_mv)` in luma half-pel units.
pub fn direct_mode_mvs(
    co_mv: (i32, i32),
    trb: i32,
    trd: i32,
    delta: (i32, i32),
) -> ((i32, i32), (i32, i32)) {
    let mv_f_x = trb * co_mv.0 / trd + delta.0;
    let mv_f_y = trb * co_mv.1 / trd + delta.1;
    let mv_b_x = if delta.0 == 0 {
        (trb - trd) * co_mv.0 / trd
    } else {
        mv_f_x - co_mv.0
    };
    let mv_b_y = if delta.1 == 0 {
        (trb - trd) * co_mv.1 / trd
    } else {
        mv_f_y - co_mv.1
    };
    ((mv_f_x, mv_f_y), (mv_b_x, mv_b_y))
}

/// Read the optional small `mvd_b` delta used by direct mode
/// (§7.6.5.3 — "motion vector encoding for direct mode").
///
/// The delta is present only when `modb != SKIPPED` and `mbtype == Direct`.
/// The encoder is free to omit it by sending the default pair (0,0) — we
/// decode one MV-component pair with the special `fcode=1` range.
pub fn decode_direct_delta(br: &mut BitReader<'_>) -> Result<(i32, i32)> {
    // Per spec, the direct-mode delta uses the standard MV VLC with the
    // implicit `f_code=1` (no residual bits, direct magnitude).
    let dx = decode_mv_component(br, 1, 0)?;
    let dy = decode_mv_component(br, 1, 0)?;
    Ok((dx, dy))
}

/// Row-local predictor state for B-VOP MV decode (§7.5.8).
///
/// Forward and backward MV predictors run as independent streams; each
/// resets to `(0, 0)` at the start of every macroblock row and updates
/// only when a macroblock of the matching mode decodes a vector. Skipped
/// and direct-mode MBs do NOT update the predictors.
#[derive(Clone, Copy, Debug, Default)]
pub struct BRowPred {
    pub fwd: (i32, i32),
    pub bwd: (i32, i32),
}

impl BRowPred {
    pub fn reset(&mut self) {
        self.fwd = (0, 0);
        self.bwd = (0, 0);
    }
}

/// Decode one B-VOP macroblock. Writes reconstructed pels to `pic`, updates
/// `bmv_grid`, returns the new quant.
///
/// `co_mv_grid` is the MV grid from the future reference P-VOP — used for
/// direct-mode co-located MV lookup.
///
/// `row_pred` holds the running forward/backward predictors for the current
/// MB row (§7.5.8 — "reset to zero only at the beginning of each macroblock
/// row"). It is updated in place when a forward/backward/interpolated MB
/// decodes its vector.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_mb(
    br: &mut BitReader<'_>,
    mb_x: usize,
    mb_y: usize,
    quant_in: u32,
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    pic: &mut IVopPicture,
    bmv_grid: &mut BMvGrid,
    row_pred: &mut BRowPred,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    co_mv_grid: Option<&MvGrid>,
    trb: i32,
    trd: i32,
    _slice_first_mb: (usize, usize),
) -> Result<u32> {
    let mut quant = quant_in;

    // §6.2.7 / §7.5.9.5.4 — if the co-located P-VOP MB has
    // `not_coded == 1`, the B-VOP MB is NOT coded in the bitstream (no
    // MODB read). It is reconstructed as forward mode with the zero MV.
    let co_located_not_coded = co_mv_grid
        .map(|g| g.get(mb_x, mb_y).not_coded)
        .unwrap_or(false);
    if co_located_not_coded {
        bmv_grid.set(
            mb_x,
            mb_y,
            BMbMotion {
                fwd: (0, 0),
                bwd: (0, 0),
                mode: Some(BMode::Forward),
            },
        );
        reconstruct_b_mb(
            pic,
            prev_ref,
            next_ref,
            mb_x,
            mb_y,
            (0, 0),
            (0, 0),
            BMode::Forward,
            vol,
            vop,
            None,
        );
        return Ok(quant);
    }

    // 1. MODB (§6.2.7 / Table 11-3).
    let modb = vlc::decode(br, bvop_tab::modb_table())?;
    if modb == bvop_tab::MODB_SKIPPED {
        // §7.5.9.5.4 — MODB=='0' reconstructs via direct mode with zero
        // delta. Predictors are NOT updated.
        let co_mv = co_mv_grid
            .map(|g| g.get(mb_x, mb_y).mv[0])
            .unwrap_or((0, 0));
        let (fwd, bwd) = direct_mode_mvs(co_mv, trb, trd, (0, 0));
        bmv_grid.set(
            mb_x,
            mb_y,
            BMbMotion {
                fwd,
                bwd,
                mode: Some(BMode::Skipped),
            },
        );
        reconstruct_b_mb(
            pic,
            prev_ref,
            next_ref,
            mb_x,
            mb_y,
            fwd,
            bwd,
            BMode::Skipped,
            vol,
            vop,
            None,
        );
        return Ok(quant);
    }

    // 2. MBTYPE (§6.2.7 / Table 11-4).
    let mbtype = vlc::decode(br, bvop_tab::mbtype_table())?;

    // 3. Optional CBPB (6 bits) when modb == MODB_MBTYPE_CBPB. Only present
    //    for non-direct modes per the syntax diagram.
    let cbpb = if modb == bvop_tab::MODB_MBTYPE_CBPB {
        br.read_u32(6)? as u8
    } else {
        0
    };

    // 4. dquant — per spec: "if (mb_type != '1' && cbpb != 0) dquant". That
    //    is, dquant is present for non-direct modes whenever any residual
    //    block is coded. The 2-bit dquant table maps 00→-1, 01→-2, 10→+1,
    //    11→+2 (Table 6-17).
    if mbtype != bvop_tab::MBTYPE_DIRECT && cbpb != 0 {
        let dquant_code = br.read_u32(2)? as i32;
        let delta = match dquant_code {
            0b00 => -1,
            0b01 => -2,
            0b10 => 1,
            0b11 => 2,
            _ => unreachable!(),
        };
        let q = (quant as i32 + delta).clamp(1, 31) as u32;
        quant = q;
    }

    // 5. MVs per mode (§7.5.8). Predictors from `row_pred`; direct-mode
    //    delta has its own f_code=1 path.
    let f_code_f = vop.vop_fcode_forward.max(1);
    let f_code_b = vop.vop_fcode_backward.max(1);
    let (fwd_mv, bwd_mv, mode) = match mbtype {
        bvop_tab::MBTYPE_DIRECT => {
            let delta = decode_direct_delta(br)?;
            let co_mv = co_mv_grid
                .map(|g| g.get(mb_x, mb_y).mv[0])
                .unwrap_or((0, 0));
            let (fwd, bwd) = direct_mode_mvs(co_mv, trb, trd, delta);
            // Direct-mode: predictors NOT updated (spec says row predictors
            // track only forward/backward MBs that explicitly decoded an MV).
            (fwd, bwd, BMode::Direct)
        }
        bvop_tab::MBTYPE_FORWARD => {
            let mvx = decode_mv_component(br, f_code_f, row_pred.fwd.0)?;
            let mvy = decode_mv_component(br, f_code_f, row_pred.fwd.1)?;
            row_pred.fwd = (mvx, mvy);
            ((mvx, mvy), (0, 0), BMode::Forward)
        }
        bvop_tab::MBTYPE_BACKWARD => {
            let mvx = decode_mv_component(br, f_code_b, row_pred.bwd.0)?;
            let mvy = decode_mv_component(br, f_code_b, row_pred.bwd.1)?;
            row_pred.bwd = (mvx, mvy);
            ((0, 0), (mvx, mvy), BMode::Backward)
        }
        bvop_tab::MBTYPE_INTERPOLATED => {
            let fx = decode_mv_component(br, f_code_f, row_pred.fwd.0)?;
            let fy = decode_mv_component(br, f_code_f, row_pred.fwd.1)?;
            row_pred.fwd = (fx, fy);
            let bx = decode_mv_component(br, f_code_b, row_pred.bwd.0)?;
            let by = decode_mv_component(br, f_code_b, row_pred.bwd.1)?;
            row_pred.bwd = (bx, by);
            ((fx, fy), (bx, by), BMode::Interpolated)
        }
        _ => {
            return Err(Error::invalid(format!(
                "mpeg4 B-VOP: unknown mbtype {mbtype}"
            )))
        }
    };

    bmv_grid.set(
        mb_x,
        mb_y,
        BMbMotion {
            fwd: fwd_mv,
            bwd: bwd_mv,
            mode: Some(mode),
        },
    );

    // 6. Residual + MC.
    let mut residual_blocks = [[0i32; 64]; 6];
    for blk in 0..6 {
        let coded = (cbpb >> (5 - blk)) & 1 != 0;
        if coded {
            decode_inter_ac(br, &mut residual_blocks[blk], &ZIGZAG)?;
            let mut tmp = [0i32; 64];
            reconstruct_inter_block(&mut residual_blocks[blk], vol, quant, &mut tmp)?;
            residual_blocks[blk] = tmp;
        }
    }

    reconstruct_b_mb(
        pic,
        prev_ref,
        next_ref,
        mb_x,
        mb_y,
        fwd_mv,
        bwd_mv,
        mode,
        vol,
        vop,
        Some(&residual_blocks),
    );

    Ok(quant)
}

/// Public wrapper over `reconstruct_b_mb` used by the decoder when it
/// needs to reconstruct an implicitly-skipped B-MB (e.g. jumping over a
/// range covered by a video-packet resync marker).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_b_mb_public(
    pic: &mut IVopPicture,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    fwd_mv: (i32, i32),
    bwd_mv: (i32, i32),
    mode: BMode,
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    residual_blocks: Option<&[[i32; 64]; 6]>,
) {
    reconstruct_b_mb(
        pic,
        prev_ref,
        next_ref,
        mb_x,
        mb_y,
        fwd_mv,
        bwd_mv,
        mode,
        vol,
        vop,
        residual_blocks,
    );
}

/// Combine forward + backward predictions according to `mode`, add
/// residuals, write into `pic`. Dispatch is driven by the explicit
/// `mode` argument — relying on MV magnitudes alone is ambiguous because
/// a genuine (0,0) forward MV is valid.
#[allow(clippy::too_many_arguments)]
fn reconstruct_b_mb(
    pic: &mut IVopPicture,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    fwd_mv: (i32, i32),
    bwd_mv: (i32, i32),
    mode: BMode,
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    residual_blocks: Option<&[[i32; 64]; 6]>,
) {
    // Per §7.5.9: luminance MC runs on 8x8 blocks in direct mode and on
    // the full 16x16 MB in all other modes. Our MV is 16x16 (non-direct)
    // or scaled from the co-located 16x16 (direct w/o 4MV on the P-VOP);
    // we apply the same 8x8-block sampling for both cases — the MV is
    // identical across the 4 luma blocks when the co-located MV is 16x16.
    for blk in 0..4 {
        let (sub_x, sub_y) = match blk {
            0 => (0, 0),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let blk_px = (mb_x * 16 + sub_x) as i32;
        let blk_py = (mb_y * 16 + sub_y) as i32;

        let use_fwd = matches!(
            mode,
            BMode::Forward | BMode::Interpolated | BMode::Direct | BMode::Skipped
        );
        let use_bwd = matches!(
            mode,
            BMode::Backward | BMode::Interpolated | BMode::Direct | BMode::Skipped
        );

        let mut pred_fwd = [0u8; 64];
        let mut pred_bwd = [0u8; 64];
        if use_fwd {
            predict_luma_block(&mut pred_fwd, prev_ref, blk_px, blk_py, fwd_mv, vol, vop);
        }
        if use_bwd {
            predict_luma_block(&mut pred_bwd, next_ref, blk_px, blk_py, bwd_mv, vol, vop);
        }

        let combined: [u8; 64] = match mode {
            BMode::Forward => pred_fwd,
            BMode::Backward => pred_bwd,
            BMode::Interpolated | BMode::Direct | BMode::Skipped => {
                // Average — rounding to nearest (§7.5.9 — bi-directional MC
                // emits `(fwd + bwd + 1) >> 1`).
                let mut out = [0u8; 64];
                for i in 0..64 {
                    out[i] = ((pred_fwd[i] as u16 + pred_bwd[i] as u16 + 1) >> 1) as u8;
                }
                out
            }
        };

        let dst_off = (blk_py as usize) * pic.y_stride + (blk_px as usize);
        if let Some(residuals) = residual_blocks {
            crate::simd::add_residual_clip_block(
                &combined,
                &residuals[blk],
                &mut pic.y,
                dst_off,
                pic.y_stride,
            );
        } else {
            crate::simd::copy_block_u8(&combined, &mut pic.y, dst_off, pic.y_stride);
        }
    }

    // Chroma — derive half-pel chroma MVs.
    let to_chroma = |v: i32| -> i32 {
        if vol.quarter_sample {
            luma_qmv_to_chroma(v)
        } else {
            luma_mv_to_chroma(v)
        }
    };
    let fwd_c = (to_chroma(fwd_mv.0), to_chroma(fwd_mv.1));
    let bwd_c = (to_chroma(bwd_mv.0), to_chroma(bwd_mv.1));
    let use_fwd = matches!(
        mode,
        BMode::Forward | BMode::Interpolated | BMode::Direct | BMode::Skipped
    );
    let use_bwd = matches!(
        mode,
        BMode::Backward | BMode::Interpolated | BMode::Direct | BMode::Skipped
    );
    for plane_idx in 0..2 {
        let (ref_prev_plane, ref_next_plane, ref_stride) = if plane_idx == 0 {
            (&prev_ref.cb, &next_ref.cb, prev_ref.c_stride)
        } else {
            (&prev_ref.cr, &next_ref.cr, prev_ref.c_stride)
        };
        let blk_px = (mb_x * 8) as i32;
        let blk_py = (mb_y * 8) as i32;
        let mut pred_fwd = [0u8; 64];
        let mut pred_bwd = [0u8; 64];
        if use_fwd {
            predict_block(
                ref_prev_plane,
                ref_stride,
                ref_stride as i32,
                (ref_prev_plane.len() / ref_stride) as i32,
                blk_px,
                blk_py,
                fwd_c.0,
                fwd_c.1,
                8,
                vop.rounding_type,
                &mut pred_fwd,
                8,
            );
        }
        if use_bwd {
            predict_block(
                ref_next_plane,
                ref_stride,
                ref_stride as i32,
                (ref_next_plane.len() / ref_stride) as i32,
                blk_px,
                blk_py,
                bwd_c.0,
                bwd_c.1,
                8,
                vop.rounding_type,
                &mut pred_bwd,
                8,
            );
        }
        let combined: [u8; 64] = match mode {
            BMode::Forward => pred_fwd,
            BMode::Backward => pred_bwd,
            BMode::Interpolated | BMode::Direct | BMode::Skipped => {
                let mut out = [0u8; 64];
                for i in 0..64 {
                    out[i] = ((pred_fwd[i] as u16 + pred_bwd[i] as u16 + 1) >> 1) as u8;
                }
                out
            }
        };
        let dst_plane = if plane_idx == 0 {
            &mut pic.cb
        } else {
            &mut pic.cr
        };
        let dst_off = (blk_py as usize) * pic.c_stride + (blk_px as usize);
        if let Some(residuals) = residual_blocks {
            let res_idx = 4 + plane_idx;
            crate::simd::add_residual_clip_block(
                &combined,
                &residuals[res_idx],
                dst_plane,
                dst_off,
                pic.c_stride,
            );
        } else {
            crate::simd::copy_block_u8(&combined, dst_plane, dst_off, pic.c_stride);
        }
    }
}

/// Luma prediction with VOL-directed half-pel or quarter-pel filter.
fn predict_luma_block(
    dst: &mut [u8; 64],
    reference: &IVopPicture,
    blk_px: i32,
    blk_py: i32,
    mv: (i32, i32),
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
) {
    let rows = (reference.y.len() / reference.y_stride) as i32;
    if vol.quarter_sample {
        predict_block_qpel(
            &reference.y,
            reference.y_stride,
            reference.y_stride as i32,
            rows,
            blk_px,
            blk_py,
            mv.0,
            mv.1,
            8,
            vop.rounding_type,
            dst,
            8,
        );
    } else {
        predict_block(
            &reference.y,
            reference.y_stride,
            reference.y_stride as i32,
            rows,
            blk_px,
            blk_py,
            mv.0,
            mv.1,
            8,
            vop.rounding_type,
            dst,
            8,
        );
    }
}

/// Grid mirror of `MvGrid` for B-VOPs — holds separate fwd/bwd MVs per MB.
#[derive(Clone)]
pub struct BMvGrid {
    pub mb_w: usize,
    pub mb_h: usize,
    pub mvs: Vec<BMbMotion>,
}

impl BMvGrid {
    pub fn new(mb_w: usize, mb_h: usize) -> Self {
        Self {
            mb_w,
            mb_h,
            mvs: vec![BMbMotion::default(); mb_w * mb_h],
        }
    }
    pub fn get(&self, mb_x: usize, mb_y: usize) -> &BMbMotion {
        &self.mvs[mb_y * self.mb_w + mb_x]
    }
    pub fn set(&mut self, mb_x: usize, mb_y: usize, m: BMbMotion) {
        self.mvs[mb_y * self.mb_w + mb_x] = m;
    }
}

/// Suppress unused-warning for PredGrid — B-VOPs don't use AC/DC intra
/// prediction (all MBs are bidirectional inter). Kept for API symmetry
/// if future work adds embedded intra MBs inside B-VOPs.
#[allow(dead_code)]
pub(crate) fn _pred_grid_placeholder(_g: &mut PredGrid) {}

/// Used to compute MB bit-match between MV grids via inter module.
///
/// This is a no-op module-local helper; re-exported symbols above
/// (`MbMotion`, `MvGrid`) remain in `crate::inter` and are accessed
/// through the re-exports there.
#[allow(dead_code)]
pub(crate) fn _mb_motion_placeholder(_m: MbMotion) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mode_scales_linearly() {
        // TRB = TRD → direct = co_mv for forward, (0,0) for backward.
        let ((fx, fy), (bx, by)) = direct_mode_mvs((8, -4), 4, 4, (0, 0));
        assert_eq!((fx, fy), (8, -4));
        assert_eq!((bx, by), (0, 0));

        // TRB = TRD/2 → fwd = co_mv/2, bwd = -co_mv/2.
        let ((fx, fy), (bx, by)) = direct_mode_mvs((8, -4), 2, 4, (0, 0));
        assert_eq!((fx, fy), (4, -2));
        assert_eq!((bx, by), (-4, 2));
    }

    #[test]
    fn direct_mode_with_delta_uses_fwd_diff() {
        // Per-component: dx != 0 → bwdx = fwdx - co_mvx;  dy == 0 → bwdy uses
        // the linear formula.
        let ((fx, fy), (bx, by)) = direct_mode_mvs((8, 0), 2, 4, (1, 2));
        assert_eq!((fx, fy), (4 + 1, 0 + 2));
        // dx = 1 (nonzero) → bwdx = fwdx - co_mvx = 5 - 8 = -3
        // dy = 2 (nonzero) → bwdy = fwdy - co_mvy = 2 - 0 = 2
        assert_eq!((bx, by), (fx - 8, fy - 0));
    }

    #[test]
    fn direct_mode_per_component_delta() {
        // Only x has a delta → y uses the linear backward formula.
        let ((fx, fy), (bx, by)) = direct_mode_mvs((8, -4), 2, 4, (1, 0));
        // fwdx = 2*8/4 + 1 = 5;  fwdy = 2*(-4)/4 + 0 = -2
        assert_eq!((fx, fy), (5, -2));
        // bwdx (dx=1, nonzero): fwdx - co_mvx = 5 - 8 = -3
        // bwdy (dy=0):          (trb-trd)*co_mvy/trd = -2*(-4)/4 = 2
        assert_eq!((bx, by), (-3, 2));
    }

    #[test]
    fn trb_trd_basic() {
        assert_eq!(trb_trd(0, 2, 5), (2, 5));
        // When prev == next, clamp to 1 to avoid divide-by-zero.
        assert_eq!(trb_trd(10, 10, 10), (1, 1));
    }

    #[test]
    fn bmv_grid_round_trip() {
        let mut g = BMvGrid::new(4, 3);
        g.set(
            2,
            1,
            BMbMotion {
                fwd: (3, -7),
                bwd: (-1, 4),
                mode: Some(BMode::Interpolated),
            },
        );
        let m = g.get(2, 1);
        assert_eq!(m.fwd, (3, -7));
        assert_eq!(m.bwd, (-1, 4));
        assert_eq!(m.mode, Some(BMode::Interpolated));
    }
}

//! B-VOP (bidirectional) decode — ISO/IEC 14496-2:2004 §7.6.8–§7.6.9.
//!
//! Per-MB flow (spec syntax §6.2.7 / semantics §6.3.5):
//! 1. `modb` VLC (Table B.3). If skipped ('1'), the MB inherits direct-mode
//!    prediction with zero MVD — emit co-located P-VOP reference region.
//! 2. `mbtype` VLC (Table B.4) — Direct / Forward / Backward / Interpolated.
//!    (present only when `modb != '1'`)
//! 3. If MODB=="00", `cbpb` (6 bits for 4:2:0 — one per block, Y0..Y3 Cb Cr).
//! 4. `dbquant` VLC (Table 6-33) — present only when `mb_type != '1'` AND
//!    `cbpb != 0`. Codes: `0`→0, `10`→-2, `11`→+2. This is a B-VOP-specific
//!    quantiser-delta VLC — **not** the P-VOP 2-bit `dquant` (Table 6-32).
//! 5. MVs:
//!    * Direct mode — scales the co-located P-VOP MV by (TRB/TRD) and adds
//!      an optional small delta `mvd_b` (single MV pair).
//!    * Forward mode — decodes `mvd_forward` via the standard MV VLC.
//!    * Backward mode — decodes `mvd_backward`.
//!    * Interpolated — decodes both `mvd_forward` and `mvd_backward`.
//! 6. Texture residual per coded block.
//! 7. Motion compensation: forward-only, backward-only, or average of the
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
use crate::headers::vol::{VideoObjectLayer, ALTERNATE_VERTICAL_SCAN, ZIGZAG};
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
///
/// For 4MV direct mode (§7.5.9.5) the four per-block forward/backward
/// vectors are stored in `fwd4`/`bwd4`; `fwd`/`bwd` keep the block-0
/// vectors so callers that only need a single 16x16 summary stay valid.
#[derive(Clone, Copy, Debug, Default)]
pub struct BMbMotion {
    pub fwd: (i32, i32),
    pub bwd: (i32, i32),
    /// Per-block forward MVs (index 0..3 in raster order inside the MB).
    /// All four entries equal `fwd` when not 4MV direct.
    pub fwd4: [(i32, i32); 4],
    /// Per-block backward MVs — same layout as `fwd4`.
    pub bwd4: [(i32, i32); 4],
    /// True when the MB used 4MV direct mode (per-block MVs differ).
    pub four_mv_direct: bool,
    pub mode: Option<BMode>,
}

impl BMbMotion {
    /// Construct from a single 16x16 forward/backward pair.
    pub fn uni(fwd: (i32, i32), bwd: (i32, i32), mode: BMode) -> Self {
        Self {
            fwd,
            bwd,
            fwd4: [fwd; 4],
            bwd4: [bwd; 4],
            four_mv_direct: false,
            mode: Some(mode),
        }
    }

    /// Construct from 4 per-block forward/backward pairs (4MV direct mode).
    pub fn quad(fwd4: [(i32, i32); 4], bwd4: [(i32, i32); 4], mode: BMode) -> Self {
        Self {
            fwd: fwd4[0],
            bwd: bwd4[0],
            fwd4,
            bwd4,
            four_mv_direct: true,
            mode: Some(mode),
        }
    }
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

/// 4MV variant of `direct_mode_mvs` — one MVD delta applies to all four
/// sub-block vectors (§7.5.9.5.2, per-i formulas with `MVDx/MVDy` shared).
///
/// `co_mvs4` are the 4 per-block MVs of the co-located P-MB. Returns arrays
/// of forward and backward MVs, one per luma 8x8 block in raster order.
#[allow(clippy::type_complexity)]
pub fn direct_mode_mvs_4(
    co_mvs4: [(i32, i32); 4],
    trb: i32,
    trd: i32,
    delta: (i32, i32),
) -> ([(i32, i32); 4], [(i32, i32); 4]) {
    let mut fwd = [(0i32, 0i32); 4];
    let mut bwd = [(0i32, 0i32); 4];
    for i in 0..4 {
        let (f, b) = direct_mode_mvs(co_mvs4[i], trb, trd, delta);
        fwd[i] = f;
        bwd[i] = b;
    }
    (fwd, bwd)
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
    //
    // Per 2004 spec §6.3.5: "co_located_not_coded ... is set to 1 when
    // the current VOP is a B-VOP, the future reference VOP is a P-VOP,
    // and the co-located macroblock in the future reference VOP is
    // skipped". When the backward reference is an I-VOP the flag is
    // unset (0) — every MB carries MODB bits in the bitstream.
    let co_located_not_coded = co_mv_grid
        .map(|g| g.get(mb_x, mb_y).not_coded)
        .unwrap_or(false);
    if co_located_not_coded {
        bmv_grid.set(mb_x, mb_y, BMbMotion::uni((0, 0), (0, 0), BMode::Forward));
        reconstruct_b_mb(
            pic,
            prev_ref,
            next_ref,
            mb_x,
            mb_y,
            [(0, 0); 4],
            [(0, 0); 4],
            BMode::Forward,
            vol,
            vop,
            None,
            None,
        );
        return Ok(quant);
    }

    // 1. MODB (§6.2.7 / Table 11-3).
    let modb = vlc::decode(br, bvop_tab::modb_table())?;
    if modb == bvop_tab::MODB_SKIPPED {
        // §7.5.9.5.4 — MODB=='0' reconstructs via direct mode with zero
        // delta. Predictors are NOT updated.
        let (co_mvs4, was_4mv) = colocated_mvs4(co_mv_grid, mb_x, mb_y);
        let (fwd4, bwd4) = direct_mode_mvs_4(co_mvs4, trb, trd, (0, 0));
        let entry = if was_4mv {
            BMbMotion::quad(fwd4, bwd4, BMode::Skipped)
        } else {
            BMbMotion::uni(fwd4[0], bwd4[0], BMode::Skipped)
        };
        bmv_grid.set(mb_x, mb_y, entry);
        reconstruct_b_mb(
            pic,
            prev_ref,
            next_ref,
            mb_x,
            mb_y,
            fwd4,
            bwd4,
            BMode::Skipped,
            vol,
            vop,
            None,
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

    // 4. dbquant — per 2004 spec §6.3.5 / Table 6-33: "if (mb_type != '1'
    //    && cbpb != 0) dbquant". B-VOPs use `dbquant` (a 1-or-2-bit VLC),
    //    NOT the P-VOP 2-bit `dquant`:
    //
    //        dbquant code  value
    //             0          0
    //            10         -2
    //            11         +2
    //
    //    The quantiser is then clipped to [1, 2^quant_precision - 1]. An
    //    earlier version of this decoder used the P-VOP 2-bit dquant table
    //    here, which desynced every non-direct B-MB whenever the encoder
    //    emitted `dbquant = 0` (the "no change" single-bit code). The
    //    `Round 9` fix switches to the proper VLC (see Table 6-33 in the
    //    2004 final edition).
    if mbtype != bvop_tab::MBTYPE_DIRECT && cbpb != 0 {
        let first = br.read_u1()?;
        let delta = if first == 0 {
            0
        } else {
            let second = br.read_u1()?;
            if second == 0 {
                -2
            } else {
                2
            }
        };
        let q_limit = (1u32 << vol.quant_precision).saturating_sub(1).max(1);
        quant = (quant as i32 + delta).clamp(1, q_limit as i32) as u32;
    }

    // 4b. Interlaced information (§6.2.7 / §6.2.7.3). Present only when
    //     the VOP advertises `interlaced == 1`. For B-VOP non-direct MBs
    //     the parser consumes `dct_type` (iff cbpb != 0), `field_prediction`,
    //     and optional `forward_*` / `backward_*` field-reference bits.
    //     For direct-mode MBs only `dct_type` is read (iff cbpb != 0).
    //
    //     Field-predicted MBs decode TWO MVs per direction (top + bottom
    //     field) instead of one — see §7.6.2.2. Field MC applies the two
    //     MVs to the two fields of the reference VOP. Residual field-DCT
    //     reordering (§7.6.1) maps blocks 0,1 → even rows and 2,3 → odd
    //     rows when `dct_type == 1`.
    let il_info = if vop.interlaced {
        let class = if mbtype == bvop_tab::MBTYPE_DIRECT {
            crate::interlaced::MbClass::BVopDirect
        } else {
            crate::interlaced::MbClass::BVopNonDirect
        };
        crate::interlaced::parse_interlaced_information(
            br,
            class,
            crate::headers::vop::VopCodingType::B,
            cbpb != 0,
        )?
    } else {
        crate::interlaced::InterlacedInfo::default()
    };

    // 5. MVs per mode (§7.5.8). Predictors from `row_pred`; direct-mode
    //    delta has its own f_code=1 path.
    //
    //    Field-predicted non-direct MBs decode TWO MVs per direction
    //    (top + bottom field) per §7.6.2.2. Field predictors are shared
    //    with the frame row_pred (coarse approximation — proper §7.6.2.2
    //    maintains 4 independent PMVs; this path follows FFmpeg's heuristic
    //    of using the shared row predictor with Py/2 for the field MVD.y).
    let f_code_f = vop.vop_fcode_forward.max(1);
    let f_code_b = vop.vop_fcode_backward.max(1);
    let field_pred = il_info.field_prediction;
    let mut field_mvs_fwd: Option<[(i32, i32); 2]> = None;
    let mut field_mvs_bwd: Option<[(i32, i32); 2]> = None;
    let (fwd4, bwd4, mode, is_4mv_direct) = match mbtype {
        bvop_tab::MBTYPE_DIRECT => {
            let delta = decode_direct_delta(br)?;
            let (co_mvs4, was_4mv) = colocated_mvs4(co_mv_grid, mb_x, mb_y);
            let (fwd4, bwd4) = direct_mode_mvs_4(co_mvs4, trb, trd, delta);
            // Direct-mode: predictors NOT updated (spec says row predictors
            // track only forward/backward MBs that explicitly decoded an MV).
            (fwd4, bwd4, BMode::Direct, was_4mv)
        }
        bvop_tab::MBTYPE_FORWARD => {
            if field_pred {
                // Decode 2 MVs for top + bottom field (§7.6.2.2).
                let (px, py) = row_pred.fwd;
                let mut top = (0, 0);
                let mut bot = (0, 0);
                for (slot, out) in [(&mut top, 0), (&mut bot, 1)] {
                    let mvx = decode_mv_component(br, f_code_f, px)?;
                    let mvy = decode_mv_component(br, f_code_f, py / 2)?;
                    *slot = (mvx, 2 * mvy);
                    let _ = out;
                }
                let avg = ((top.0 + bot.0) / 2, (top.1 + bot.1) / 2);
                row_pred.fwd = avg;
                field_mvs_fwd = Some([top, bot]);
                ([avg; 4], [(0, 0); 4], BMode::Forward, false)
            } else {
                let mvx = decode_mv_component(br, f_code_f, row_pred.fwd.0)?;
                let mvy = decode_mv_component(br, f_code_f, row_pred.fwd.1)?;
                row_pred.fwd = (mvx, mvy);
                ([(mvx, mvy); 4], [(0, 0); 4], BMode::Forward, false)
            }
        }
        bvop_tab::MBTYPE_BACKWARD => {
            if field_pred {
                let (px, py) = row_pred.bwd;
                let mut top = (0, 0);
                let mut bot = (0, 0);
                for slot in [&mut top, &mut bot] {
                    let mvx = decode_mv_component(br, f_code_b, px)?;
                    let mvy = decode_mv_component(br, f_code_b, py / 2)?;
                    *slot = (mvx, 2 * mvy);
                }
                let avg = ((top.0 + bot.0) / 2, (top.1 + bot.1) / 2);
                row_pred.bwd = avg;
                field_mvs_bwd = Some([top, bot]);
                ([(0, 0); 4], [avg; 4], BMode::Backward, false)
            } else {
                let mvx = decode_mv_component(br, f_code_b, row_pred.bwd.0)?;
                let mvy = decode_mv_component(br, f_code_b, row_pred.bwd.1)?;
                row_pred.bwd = (mvx, mvy);
                ([(0, 0); 4], [(mvx, mvy); 4], BMode::Backward, false)
            }
        }
        bvop_tab::MBTYPE_INTERPOLATED => {
            if field_pred {
                // Field-bidirectional: 2 forward MVs + 2 backward MVs
                // (§7.6.2.2 — PMVs 0/1/2/3 in the field predictor table).
                let (px_f, py_f) = row_pred.fwd;
                let (px_b, py_b) = row_pred.bwd;
                let mut ftop = (0, 0);
                let mut fbot = (0, 0);
                for slot in [&mut ftop, &mut fbot] {
                    let mvx = decode_mv_component(br, f_code_f, px_f)?;
                    let mvy = decode_mv_component(br, f_code_f, py_f / 2)?;
                    *slot = (mvx, 2 * mvy);
                }
                let mut btop = (0, 0);
                let mut bbot = (0, 0);
                for slot in [&mut btop, &mut bbot] {
                    let mvx = decode_mv_component(br, f_code_b, px_b)?;
                    let mvy = decode_mv_component(br, f_code_b, py_b / 2)?;
                    *slot = (mvx, 2 * mvy);
                }
                let fwd_avg = ((ftop.0 + fbot.0) / 2, (ftop.1 + fbot.1) / 2);
                let bwd_avg = ((btop.0 + bbot.0) / 2, (btop.1 + bbot.1) / 2);
                row_pred.fwd = fwd_avg;
                row_pred.bwd = bwd_avg;
                field_mvs_fwd = Some([ftop, fbot]);
                field_mvs_bwd = Some([btop, bbot]);
                ([fwd_avg; 4], [bwd_avg; 4], BMode::Interpolated, false)
            } else {
                let fx = decode_mv_component(br, f_code_f, row_pred.fwd.0)?;
                let fy = decode_mv_component(br, f_code_f, row_pred.fwd.1)?;
                row_pred.fwd = (fx, fy);
                let bx = decode_mv_component(br, f_code_b, row_pred.bwd.0)?;
                let by = decode_mv_component(br, f_code_b, row_pred.bwd.1)?;
                row_pred.bwd = (bx, by);
                ([(fx, fy); 4], [(bx, by); 4], BMode::Interpolated, false)
            }
        }
        _ => {
            return Err(Error::invalid(format!(
                "mpeg4 B-VOP: unknown mbtype {mbtype}"
            )))
        }
    };
    // Field-prediction MC — build the per-MB `FieldMc` descriptor with
    // the just-captured top/bottom field MVs + the reference-field flags
    // carried in `il_info`. `reconstruct_b_mb` dispatches into the
    // `field_predict_luma_mb` / `field_predict_chroma_mb` helpers when
    // this is `Some`. For the progressive path it stays `None`.
    let field_mc = if field_pred {
        Some(FieldMc {
            fwd: field_mvs_fwd,
            bwd: field_mvs_bwd,
            fwd_top_ref: il_info.forward_top_field_ref,
            fwd_bot_ref: il_info.forward_bottom_field_ref,
            bwd_top_ref: il_info.backward_top_field_ref,
            bwd_bot_ref: il_info.backward_bottom_field_ref,
        })
    } else {
        None
    };

    let entry = if is_4mv_direct {
        BMbMotion::quad(fwd4, bwd4, mode)
    } else {
        BMbMotion::uni(fwd4[0], bwd4[0], mode)
    };
    bmv_grid.set(mb_x, mb_y, entry);

    // 6. Residual + MC.
    let mut residual_blocks = [[0i32; 64]; 6];
    for blk in 0..6 {
        let coded = (cbpb >> (5 - blk)) & 1 != 0;
        if coded {
            // Per §7.4.3.3 / interlaced scan selection: inter blocks use
            // zigzag unless alternate_vertical_scan is on at the VOP level.
            let scan = if vop.alternate_vertical_scan {
                &ALTERNATE_VERTICAL_SCAN
            } else {
                &ZIGZAG
            };
            decode_inter_ac(br, &mut residual_blocks[blk], scan)?;
            let mut tmp = [0i32; 64];
            reconstruct_inter_block(&mut residual_blocks[blk], vol, quant, &mut tmp)?;
            residual_blocks[blk] = tmp;
        }
    }

    // Field-DCT reorder for luma residuals (§7.6.1). When dct_type=1, the
    // four 8x8 luma residual blocks are laid out as field strips:
    // blocks 0,1 = top field (even MB rows 0,2,...,14), blocks 2,3 =
    // bottom field (odd MB rows 1,3,...,15). We rebuild a 16x16 frame
    // layout and then re-split into 8x8 blocks in raster order so the
    // downstream reconstruction can apply each residual as if frame DCT.
    if il_info.dct_type {
        let mut frame_mb = [0i32; 256];
        crate::interlaced::field_dct_reorder_mb(
            &residual_blocks[0],
            &residual_blocks[1],
            &residual_blocks[2],
            &residual_blocks[3],
            &mut frame_mb,
        );
        for blk in 0..4 {
            let (sub_x, sub_y) = match blk {
                0 => (0, 0),
                1 => (8, 0),
                2 => (0, 8),
                3 => (8, 8),
                _ => unreachable!(),
            };
            for r in 0..8 {
                for c in 0..8 {
                    residual_blocks[blk][r * 8 + c] = frame_mb[(sub_y + r) * 16 + (sub_x + c)];
                }
            }
        }
    }

    reconstruct_b_mb(
        pic,
        prev_ref,
        next_ref,
        mb_x,
        mb_y,
        fwd4,
        bwd4,
        mode,
        vol,
        vop,
        Some(&residual_blocks),
        field_mc,
    );

    Ok(quant)
}

/// Pull the 4 co-located per-block MVs from the reference P-VOP grid.
/// Returns `(mvs4, was_4mv)` — when `was_4mv` is false, all four entries
/// are equal to the single 16x16 MV.
fn colocated_mvs4(
    co_mv_grid: Option<&MvGrid>,
    mb_x: usize,
    mb_y: usize,
) -> ([(i32, i32); 4], bool) {
    match co_mv_grid {
        Some(g) => {
            let co = g.get(mb_x, mb_y);
            if co.four_mv {
                (co.mv, true)
            } else {
                ([co.mv[0]; 4], false)
            }
        }
        None => ([(0, 0); 4], false),
    }
}

/// 4MV-aware variant of `reconstruct_b_mb_public`: accepts per-block
/// forward and backward MVs directly. Used by the decoder for implicit
/// direct-mode skips when the co-located P-MB was coded in 4MV mode.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_b_mb_public_4mv(
    pic: &mut IVopPicture,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    fwd4: [(i32, i32); 4],
    bwd4: [(i32, i32); 4],
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
        fwd4,
        bwd4,
        mode,
        vol,
        vop,
        residual_blocks,
        None,
    );
}

/// Public wrapper over `reconstruct_b_mb` used by the decoder when it
/// needs to reconstruct an implicitly-skipped B-MB (e.g. jumping over a
/// range covered by a video-packet resync marker). Accepts a single
/// 16x16 fwd/bwd pair — internal per-block storage is populated by
/// replicating it across the four blocks.
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
        [fwd_mv; 4],
        [bwd_mv; 4],
        mode,
        vol,
        vop,
        residual_blocks,
        None,
    );
}

/// Optional field MVs for a field-predicted B-MB. When `Some`, the
/// luma + chroma MC fall through to the `field_predict_luma_mb` /
/// `field_predict_chroma_mb` helpers instead of the frame-level
/// `predict_block` path. `fwd` carries `[top, bottom]` field MVs for
/// the forward reference; `bwd` is the backward counterpart.
///
/// `top_field_ref` / `bot_field_ref` flags come from the MB's
/// `interlaced_information()` — they pick which field of the reference
/// each half of the MB samples from.
#[derive(Clone, Copy, Debug, Default)]
struct FieldMc {
    fwd: Option<[(i32, i32); 2]>,
    bwd: Option<[(i32, i32); 2]>,
    fwd_top_ref: bool,
    fwd_bot_ref: bool,
    bwd_top_ref: bool,
    bwd_bot_ref: bool,
}

/// Combine forward + backward predictions according to `mode`, add
/// residuals, write into `pic`. Dispatch is driven by the explicit
/// `mode` argument — relying on MV magnitudes alone is ambiguous because
/// a genuine (0,0) forward MV is valid.
///
/// `fwd4`/`bwd4` carry one MV per luma 8x8 block in raster order. In
/// all modes except 4MV direct the four entries are equal (the caller
/// fills them by replication).
///
/// `field_mc` drives the optional field-sample path (§7.6.2.2). When
/// provided, the MB is sampled strip-by-strip from alternating fields
/// in the reference instead of the frame-level predictor.
#[allow(clippy::too_many_arguments)]
fn reconstruct_b_mb(
    pic: &mut IVopPicture,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    fwd4: [(i32, i32); 4],
    bwd4: [(i32, i32); 4],
    mode: BMode,
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    residual_blocks: Option<&[[i32; 64]; 6]>,
    field_mc: Option<FieldMc>,
) {
    // Field-predicted path: sample 16x16 luma + 8x8 chroma MB buffers
    // from alternating reference fields, then add residual + clip.
    if let Some(fm) = field_mc.filter(|f| f.fwd.is_some() || f.bwd.is_some()) {
        return reconstruct_b_mb_field(
            pic,
            prev_ref,
            next_ref,
            mb_x,
            mb_y,
            mode,
            vol,
            vop,
            residual_blocks,
            fm,
        );
    }
    // Per §7.5.9.5.3: for direct mode, luma MC runs on 8x8 blocks using
    // the per-block MVs (fwd4/bwd4). For non-direct modes each of fwd4
    // holds the same 16x16 MV across all 4 entries, so the same 8x8
    // loop yields the correct 16x16 prediction.
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
        let fwd_mv = fwd4[blk];
        let bwd_mv = bwd4[blk];

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

    // Chroma — derive half-pel chroma MVs. §7.5.9.5.3: the chroma forward
    // MV is the sum of K luma forward MVs divided by 2K with rounding;
    // K = 4 when all four luma vectors differ (4MV direct), K = 1 when
    // they are equal. For non-direct modes fwd4[i] is the same vector
    // for all i, so the average equals the 16x16 MV and the formula
    // degenerates to the usual chroma half-pel reduction.
    let avg4 = |v: &[(i32, i32); 4]| -> (i32, i32) {
        let sx: i32 = v.iter().map(|(x, _)| *x).sum();
        let sy: i32 = v.iter().map(|(_, y)| *y).sum();
        (sx / 4, sy / 4)
    };
    let to_chroma = |v: i32| -> i32 {
        if vol.quarter_sample {
            luma_qmv_to_chroma(v)
        } else {
            luma_mv_to_chroma(v)
        }
    };
    let fwd_luma = avg4(&fwd4);
    let bwd_luma = avg4(&bwd4);
    let fwd_c = (to_chroma(fwd_luma.0), to_chroma(fwd_luma.1));
    let bwd_c = (to_chroma(bwd_luma.0), to_chroma(bwd_luma.1));
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

/// Field-sample reconstruction path for B-MBs (§7.6.2.2). Samples the
/// 16x16 luma + 8x8 chroma MBs from alternating reference fields using
/// the top + bottom field MVs captured during parse, applies residuals,
/// and writes into `pic`.
///
/// Chroma field MVs are derived by `Div2Round` (`luma_mv_to_chroma`)
/// from the luma field MVs — §7.6.2.2 specifies the same reduction
/// table as for frame MC.
#[allow(clippy::too_many_arguments)]
fn reconstruct_b_mb_field(
    pic: &mut IVopPicture,
    prev_ref: &IVopPicture,
    next_ref: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mode: BMode,
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    residual_blocks: Option<&[[i32; 64]; 6]>,
    fm: FieldMc,
) {
    let use_fwd = matches!(mode, BMode::Forward | BMode::Interpolated);
    let use_bwd = matches!(mode, BMode::Backward | BMode::Interpolated);
    let mb_px = (mb_x * 16) as i32;
    let mb_py = (mb_y * 16) as i32;

    // --- Luma 16x16 predictor ---
    let ref_h = prev_ref.y.len() / prev_ref.y_stride;
    let ref_h_next = next_ref.y.len() / next_ref.y_stride;
    let mut pred_fwd = [0u8; 256];
    let mut pred_bwd = [0u8; 256];
    if use_fwd {
        if let Some([t, b]) = fm.fwd {
            crate::interlaced::field_predict_luma_mb(
                &prev_ref.y,
                prev_ref.y_stride,
                ref_h,
                mb_px,
                mb_py,
                t,
                b,
                fm.fwd_top_ref,
                fm.fwd_bot_ref,
                vop.rounding_type,
                &mut pred_fwd,
            );
        }
    }
    if use_bwd {
        if let Some([t, b]) = fm.bwd {
            crate::interlaced::field_predict_luma_mb(
                &next_ref.y,
                next_ref.y_stride,
                ref_h_next,
                mb_px,
                mb_py,
                t,
                b,
                fm.bwd_top_ref,
                fm.bwd_bot_ref,
                vop.rounding_type,
                &mut pred_bwd,
            );
        }
    }
    let mut combined_luma = [0u8; 256];
    match mode {
        BMode::Forward => combined_luma.copy_from_slice(&pred_fwd),
        BMode::Backward => combined_luma.copy_from_slice(&pred_bwd),
        BMode::Interpolated | BMode::Direct | BMode::Skipped => {
            for i in 0..256 {
                combined_luma[i] = ((pred_fwd[i] as u16 + pred_bwd[i] as u16 + 1) >> 1) as u8;
            }
        }
    }
    // Write luma into pic.y, per-block with residual.
    for blk in 0..4 {
        let (sub_x, sub_y) = match blk {
            0 => (0usize, 0usize),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let dst_off = (mb_py as usize + sub_y) * pic.y_stride + (mb_px as usize + sub_x);
        // Gather the 8x8 predictor block from the 16x16 combined buffer.
        let mut pblk = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                pblk[r * 8 + c] = combined_luma[(sub_y + r) * 16 + (sub_x + c)];
            }
        }
        if let Some(residuals) = residual_blocks {
            crate::simd::add_residual_clip_block(
                &pblk,
                &residuals[blk],
                &mut pic.y,
                dst_off,
                pic.y_stride,
            );
        } else {
            crate::simd::copy_block_u8(&pblk, &mut pic.y, dst_off, pic.y_stride);
        }
    }

    // --- Chroma 8x8 predictor ---
    let to_chroma = |v: i32| -> i32 {
        if vol.quarter_sample {
            luma_qmv_to_chroma(v)
        } else {
            luma_mv_to_chroma(v)
        }
    };
    let fwd_c = fm.fwd.map(|[t, b]| {
        (
            (to_chroma(t.0), to_chroma(t.1)),
            (to_chroma(b.0), to_chroma(b.1)),
        )
    });
    let bwd_c = fm.bwd.map(|[t, b]| {
        (
            (to_chroma(t.0), to_chroma(t.1)),
            (to_chroma(b.0), to_chroma(b.1)),
        )
    });
    let c_px = (mb_x * 8) as i32;
    let c_py = (mb_y * 8) as i32;
    for plane_idx in 0..2 {
        let (ref_prev_plane, ref_next_plane, ref_stride) = if plane_idx == 0 {
            (&prev_ref.cb, &next_ref.cb, prev_ref.c_stride)
        } else {
            (&prev_ref.cr, &next_ref.cr, prev_ref.c_stride)
        };
        let ref_h = ref_prev_plane.len() / ref_stride;
        let ref_h_next = ref_next_plane.len() / ref_stride;
        let mut pred_fwd_c = [0u8; 64];
        let mut pred_bwd_c = [0u8; 64];
        if use_fwd {
            if let Some((t, b)) = fwd_c {
                crate::interlaced::field_predict_chroma_mb(
                    ref_prev_plane,
                    ref_stride,
                    ref_h,
                    c_px,
                    c_py,
                    t,
                    b,
                    fm.fwd_top_ref,
                    fm.fwd_bot_ref,
                    vop.rounding_type,
                    &mut pred_fwd_c,
                );
            }
        }
        if use_bwd {
            if let Some((t, b)) = bwd_c {
                crate::interlaced::field_predict_chroma_mb(
                    ref_next_plane,
                    ref_stride,
                    ref_h_next,
                    c_px,
                    c_py,
                    t,
                    b,
                    fm.bwd_top_ref,
                    fm.bwd_bot_ref,
                    vop.rounding_type,
                    &mut pred_bwd_c,
                );
            }
        }
        let combined_c: [u8; 64] = match mode {
            BMode::Forward => pred_fwd_c,
            BMode::Backward => pred_bwd_c,
            BMode::Interpolated | BMode::Direct | BMode::Skipped => {
                let mut out = [0u8; 64];
                for i in 0..64 {
                    out[i] = ((pred_fwd_c[i] as u16 + pred_bwd_c[i] as u16 + 1) >> 1) as u8;
                }
                out
            }
        };
        let dst_plane = if plane_idx == 0 {
            &mut pic.cb
        } else {
            &mut pic.cr
        };
        let dst_off = (c_py as usize) * pic.c_stride + (c_px as usize);
        if let Some(residuals) = residual_blocks {
            let res_idx = 4 + plane_idx;
            crate::simd::add_residual_clip_block(
                &combined_c,
                &residuals[res_idx],
                dst_plane,
                dst_off,
                pic.c_stride,
            );
        } else {
            crate::simd::copy_block_u8(&combined_c, dst_plane, dst_off, pic.c_stride);
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
        assert_eq!((fx, fy), (5, 2));
        // dx = 1 (nonzero) → bwdx = fwdx - co_mvx = 5 - 8 = -3
        // dy = 2 (nonzero) → bwdy = fwdy - co_mvy = 2 - 0 = 2
        assert_eq!((bx, by), (fx - 8, fy));
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
        g.set(2, 1, BMbMotion::uni((3, -7), (-1, 4), BMode::Interpolated));
        let m = g.get(2, 1);
        assert_eq!(m.fwd, (3, -7));
        assert_eq!(m.bwd, (-1, 4));
        assert_eq!(m.fwd4, [(3, -7); 4]);
        assert_eq!(m.bwd4, [(-1, 4); 4]);
        assert!(!m.four_mv_direct);
        assert_eq!(m.mode, Some(BMode::Interpolated));
    }

    #[test]
    fn direct_mode_4mv_per_block() {
        // Four distinct co-located MVs → four distinct direct-mode MVs.
        let co4 = [(8, 0), (-8, 0), (0, 8), (0, -8)];
        let (fwd4, bwd4) = direct_mode_mvs_4(co4, 2, 4, (0, 0));
        // TRB/TRD = 1/2 → fwd_i = co_i / 2.
        assert_eq!(fwd4, [(4, 0), (-4, 0), (0, 4), (0, -4)]);
        // (TRB - TRD)/TRD = -1/2 → bwd_i = -co_i / 2.
        assert_eq!(bwd4, [(-4, 0), (4, 0), (0, -4), (0, 4)]);
    }

    #[test]
    fn direct_mode_4mv_delta_shared() {
        // Single delta applies to all four sub-block MVs (§7.5.9.5.2).
        let co4 = [(8, 0), (8, 0), (0, 8), (0, 8)];
        let (fwd4, _) = direct_mode_mvs_4(co4, 4, 4, (1, -1));
        // TRB/TRD = 1 → fwd_i = co_i + delta.
        assert_eq!(fwd4, [(9, -1), (9, -1), (1, 7), (1, 7)]);
    }

    #[test]
    fn quad_builder_preserves_flag() {
        let f = [(1, 2), (3, 4), (5, 6), (7, 8)];
        let b = [(-1, -2), (-3, -4), (-5, -6), (-7, -8)];
        let m = BMbMotion::quad(f, b, BMode::Direct);
        assert!(m.four_mv_direct);
        assert_eq!(m.fwd, f[0]);
        assert_eq!(m.bwd, b[0]);
        assert_eq!(m.fwd4, f);
        assert_eq!(m.bwd4, b);
    }
}

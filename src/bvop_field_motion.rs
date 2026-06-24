//! §7.7.2.2 field motion-compensated reconstruction for interlaced
//! B-VOPs.
//!
//! When an interlaced B-VOP macroblock is field-predicted
//! (`field_prediction == 1`, §6.2.6.3) and **not** direct mode, its
//! 16×16 luminance area is reconstructed by the §7.7.2.2
//! `field_motion_compensate_one_reference()` pseudo code — the same
//! per-reference field fetch used by interlaced P-VOPs
//! ([`crate::field_motion::field_motion_compensate_one_reference`]) —
//! once per active prediction direction, with the bidirectional case
//! averaging the forward and backward field predictions.
//!
//! ## §7.7.2.2 mode map (Table 7-15)
//!
//! | `mb_type` | Mode                | PMVs read | PMVs updated | References  |
//! |-----------|---------------------|-----------|--------------|-------------|
//! | `"0001"`  | Field forward       | 0,1       | 0,1          | forward     |
//! | `"001"`   | Field backward      | 2,3       | 2,3          | backward    |
//! | `"01"`    | Field bidirectional | 0,1,2,3   | 0,1,2,3      | both        |
//!
//! The four predictor slots (Table 7-14: top-fwd, bot-fwd, top-bwd,
//! bot-bwd) live in [`crate::bvop_field_predictor::FieldPmvBank`], which
//! reconstructs each field component per the §7.7.2.2 rule
//! `PMV[k].y = 2 * (PMV[k].y / 2 + MVD[i].y)` and writes the result back
//! into the bank for the next macroblock. This module consumes the
//! reconstructed `(top, bottom)` field-MV pairs that bank produces and
//! assembles the prediction macroblock.
//!
//! ## §7.7.2.2 field MC + bidirectional average
//!
//! Each direction calls `field_motion_compensate_one_reference` with the
//! direction's top/bottom field MVs and its top/bottom
//! `*_field_reference` flags (§6.3.6.3), producing one
//! [`InterPredictionMacroblock`]. The forward / backward case copies that
//! block straight through; the bidirectional case averages the two with
//! the §7.7.2.2 rounding
//! `pred[y][x] = (pred_fwd[y][x] + pred_bak[y][x] + 1) >> 1` across all
//! three planes.
//!
//! ## Quarter-sample mode
//!
//! When `quarter_sample == 1` the per-reference luma blocks use the
//! §7.6.2.2 quarter-pel cascade on the field reference grid
//! ([`crate::field_motion::field_motion_compensate_one_reference_qpel`]);
//! the field MVs are then in quarter-pel frame coordinates. The chroma
//! path inside that routine applies the §7.7.2.2 `Div2Round`-of-half rule
//! (the luma quarter-pel value divided by 2, then `Div2Round`).
//!
//! ## Scope
//!
//! Interlaced **direct** mode (§7.7.2.2, last pseudo-code block) is *not*
//! handled here — it needs the field-period temporal references
//! `TRB[i]` / `TRD[i]` and the Table 7-16 `δ` parity selection, a
//! distinct sub-subsystem. This module owns the three non-direct field
//! modes (forward / backward / bidirectional), whose pseudo code is fully
//! self-contained in §7.7.2.2.
//!
//! ## Clean-room provenance
//!
//! Truth taken from `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`
//! §7.7.2.2 (motion vector decoding in B-VOP) and §7.7.2.1
//! (`field_motion_compensate_one_reference` pseudo code). No external
//! MPEG-4 implementation consulted.

use crate::bvop_field_predictor::{FieldMvDirection, FieldPmvBank};
use crate::field_motion::{
    field_motion_compensate_one_reference, field_motion_compensate_one_reference_qpel,
    FieldMotionVectors,
};
use crate::half_sample::ReferenceVop;
use crate::interlaced_information::FieldReference;
use crate::motion::MotionVector;
use crate::reconstruct::{InterPredictionMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

/// The four §7.7.2.2 field motion-vector differentials of one macroblock,
/// in the bitstream order `MVD[0..4]` (top-fwd, bot-fwd, top-bwd,
/// bot-bwd). A forward-only macroblock populates `MVD[0..2]`, a
/// backward-only `MVD[0..2]` (top-bwd, bot-bwd), a bidirectional all
/// four.
///
/// These are the decoded `(MVDx, MVDy)` deltas — the predictor add and
/// the §7.7.2.2 vertical doubling are performed inside
/// [`FieldPmvBank`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMvDeltas {
    /// Top-field differential of the active forward / backward direction.
    pub top: MotionVector,
    /// Bottom-field differential of the active forward / backward
    /// direction.
    pub bottom: MotionVector,
}

/// The four §6.3.6.3 field-reference flags of an interlaced B-VOP
/// macroblock — which reference field (top / bottom) each output field
/// draws from, per prediction direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldReferenceFlags {
    /// `forward_top_field_reference`.
    pub forward_top: FieldReference,
    /// `forward_bottom_field_reference`.
    pub forward_bottom: FieldReference,
    /// `backward_top_field_reference`.
    pub backward_top: FieldReference,
    /// `backward_bottom_field_reference`.
    pub backward_bottom: FieldReference,
}

impl FieldReferenceFlags {
    /// All four references point at the top field (the all-`0` default).
    pub const ALL_TOP: FieldReferenceFlags = FieldReferenceFlags {
        forward_top: FieldReference::Top,
        forward_bottom: FieldReference::Top,
        backward_top: FieldReference::Top,
        backward_bottom: FieldReference::Top,
    };
}

/// The six reference planes (forward + backward luma / Cb / Cr) a
/// field-predicted B-VOP macroblock draws from.
///
/// Each plane is a progressive VOP plane in which even lines are the top
/// field and odd lines the bottom field (§7.7.2.1). `forward_*` are the
/// past anchor's planes, `backward_*` the future anchor's.
#[derive(Debug, Clone, Copy)]
pub struct BVopFieldReferences<'a> {
    /// Forward (past) luminance reference plane.
    pub forward_luma: &'a ReferenceVop<'a>,
    /// Forward (past) Cb reference plane.
    pub forward_cb: &'a ReferenceVop<'a>,
    /// Forward (past) Cr reference plane.
    pub forward_cr: &'a ReferenceVop<'a>,
    /// Backward (future) luminance reference plane.
    pub backward_luma: &'a ReferenceVop<'a>,
    /// Backward (future) Cb reference plane.
    pub backward_cb: &'a ReferenceVop<'a>,
    /// Backward (future) Cr reference plane.
    pub backward_cr: &'a ReferenceVop<'a>,
}

/// Sample-accuracy selector for the §7.7.2.2 luma field interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSampleMode {
    /// Half-sample luma interpolation (`quarter_sample == 0`).
    HalfSample,
    /// Quarter-sample luma interpolation (`quarter_sample == 1`).
    /// `bits_per_pixel` drives the §7.6.2.2.1 FIR clip.
    QuarterSample {
        /// VOL `bits_per_pixel` (8 for `not_8_bit == 0`).
        bits_per_pixel: u32,
    },
}

/// §7.7.2.2 average two prediction macroblocks
/// `(fwd + bak + 1) >> 1` across all three planes.
fn average_into(
    fwd: &InterPredictionMacroblock,
    bak: &InterPredictionMacroblock,
) -> InterPredictionMacroblock {
    let mut out = InterPredictionMacroblock::zero();
    for y in 0..MACROBLOCK_LUMA_SIDE {
        for x in 0..MACROBLOCK_LUMA_SIDE {
            out.luma[y][x] = (fwd.luma[y][x] + bak.luma[y][x] + 1) >> 1;
        }
    }
    for y in 0..MACROBLOCK_CHROMA_SIDE {
        for x in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[y][x] = (fwd.cb[y][x] + bak.cb[y][x] + 1) >> 1;
            out.cr[y][x] = (fwd.cr[y][x] + bak.cr[y][x] + 1) >> 1;
        }
    }
    out
}

/// Compensate one prediction direction from a single reference, choosing
/// the half- or quarter-sample luma interpolator.
#[allow(clippy::too_many_arguments)]
fn compensate_one(
    luma_ref: &ReferenceVop<'_>,
    cb_ref: &ReferenceVop<'_>,
    cr_ref: &ReferenceVop<'_>,
    mvs: FieldMotionVectors,
    top_field_ref: FieldReference,
    bottom_field_ref: FieldReference,
    x: i32,
    y: i32,
    rounding_type: u8,
    sample_mode: FieldSampleMode,
) -> InterPredictionMacroblock {
    match sample_mode {
        FieldSampleMode::HalfSample => field_motion_compensate_one_reference(
            luma_ref,
            cb_ref,
            cr_ref,
            mvs,
            top_field_ref.as_bit(),
            bottom_field_ref.as_bit(),
            x,
            y,
            rounding_type,
        ),
        FieldSampleMode::QuarterSample { bits_per_pixel } => {
            field_motion_compensate_one_reference_qpel(
                luma_ref,
                cb_ref,
                cr_ref,
                mvs,
                top_field_ref.as_bit(),
                bottom_field_ref.as_bit(),
                x,
                y,
                rounding_type,
                bits_per_pixel,
            )
        }
    }
}

/// §7.7.2.2 field **forward** mode (`mb_type == "0001"`,
/// `field_prediction == 1`).
///
/// Reconstructs the top/bottom forward field MVs through `bank`
/// (updating PMV slots 0,1) and compensates from the forward reference.
/// `deltas` carries `MVD[0]` (top) / `MVD[1]` (bottom).
#[allow(clippy::too_many_arguments)]
pub fn field_forward_prediction(
    bank: &mut FieldPmvBank,
    deltas: FieldMvDeltas,
    references: &BVopFieldReferences<'_>,
    refs: FieldReferenceFlags,
    mb_x: i32,
    mb_y: i32,
    rounding_type: u8,
    sample_mode: FieldSampleMode,
) -> InterPredictionMacroblock {
    let FieldMvDirection { top, bottom } = bank.field_forward(deltas.top, deltas.bottom);
    compensate_one(
        references.forward_luma,
        references.forward_cb,
        references.forward_cr,
        FieldMotionVectors { top, bottom },
        refs.forward_top,
        refs.forward_bottom,
        mb_x,
        mb_y,
        rounding_type,
        sample_mode,
    )
}

/// §7.7.2.2 field **backward** mode (`mb_type == "001"`,
/// `field_prediction == 1`).
///
/// Reconstructs the top/bottom backward field MVs through `bank`
/// (updating PMV slots 2,3 — with the §7.7.2.2 bottom-x quirk that reads
/// PMV[1].x) and compensates from the backward reference.
#[allow(clippy::too_many_arguments)]
pub fn field_backward_prediction(
    bank: &mut FieldPmvBank,
    deltas: FieldMvDeltas,
    references: &BVopFieldReferences<'_>,
    refs: FieldReferenceFlags,
    mb_x: i32,
    mb_y: i32,
    rounding_type: u8,
    sample_mode: FieldSampleMode,
) -> InterPredictionMacroblock {
    let FieldMvDirection { top, bottom } = bank.field_backward(deltas.top, deltas.bottom);
    compensate_one(
        references.backward_luma,
        references.backward_cb,
        references.backward_cr,
        FieldMotionVectors { top, bottom },
        refs.backward_top,
        refs.backward_bottom,
        mb_x,
        mb_y,
        rounding_type,
        sample_mode,
    )
}

/// §7.7.2.2 field **bidirectional** mode (`mb_type == "01"`,
/// `field_prediction == 1`).
///
/// Reconstructs all four field MVs through `bank` (updating all four PMV
/// slots), compensates the forward and backward predictions
/// independently, and averages them with the §7.7.2.2 rounding
/// `(fwd + bak + 1) >> 1`. `mvd` is the four differentials in bitstream
/// order `MVD[0..4]` (top-fwd, bot-fwd, top-bwd, bot-bwd).
#[allow(clippy::too_many_arguments)]
pub fn field_bidirectional_prediction(
    bank: &mut FieldPmvBank,
    mvd: [MotionVector; 4],
    references: &BVopFieldReferences<'_>,
    refs: FieldReferenceFlags,
    mb_x: i32,
    mb_y: i32,
    rounding_type: u8,
    sample_mode: FieldSampleMode,
) -> InterPredictionMacroblock {
    let (forward, backward) = bank.field_bidirectional(mvd);
    let fwd = compensate_one(
        references.forward_luma,
        references.forward_cb,
        references.forward_cr,
        FieldMotionVectors {
            top: forward.top,
            bottom: forward.bottom,
        },
        refs.forward_top,
        refs.forward_bottom,
        mb_x,
        mb_y,
        rounding_type,
        sample_mode,
    );
    let bak = compensate_one(
        references.backward_luma,
        references.backward_cb,
        references.backward_cr,
        FieldMotionVectors {
            top: backward.top,
            bottom: backward.bottom,
        },
        refs.backward_top,
        refs.backward_bottom,
        mb_x,
        mb_y,
        rounding_type,
        sample_mode,
    );
    average_into(&fwd, &bak)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    /// Build a `WIDTH × HEIGHT` reference plane from a closure.
    fn make_plane(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut v = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                v[y * width + x] = f(x, y);
            }
        }
        v
    }

    const W: usize = 48;
    const H: usize = 48;

    #[test]
    fn forward_zero_mv_copies_top_and_bottom_fields() {
        // A reference whose value encodes (x, y) so we can verify the
        // exact field copy. Zero MV ⇒ each output field copies its own
        // reference field at the macroblock origin.
        let plane = make_plane(W, H, |x, y| ((x + y) % 251) as u8);
        let luma = ReferenceVop::new(&plane, W, H).unwrap();
        // 8×8 chroma planes; reuse the same closure scaled.
        let cplane = make_plane(W / 2, H / 2, |x, y| ((x + y) % 251) as u8);
        let cb = ReferenceVop::new(&cplane, W / 2, H / 2).unwrap();
        let cr = ReferenceVop::new(&cplane, W / 2, H / 2).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &luma,
            forward_cb: &cb,
            forward_cr: &cr,
            backward_luma: &luma,
            backward_cb: &cb,
            backward_cr: &cr,
        };
        let mut bank = FieldPmvBank::new();
        // Zero deltas ⇒ field MVs (0,0).
        let out = field_forward_prediction(
            &mut bank,
            FieldMvDeltas {
                top: mv(0, 0),
                bottom: mv(0, 0),
            },
            &references,
            FieldReferenceFlags::ALL_TOP,
            16,
            16,
            0,
            FieldSampleMode::HalfSample,
        );
        // Top output field (even rows) draws the top reference field
        // (even rows) starting at (16,16); bottom output field (odd
        // rows) draws the top reference field too (ALL_TOP), stepping by
        // 2 in field space.
        for col in 0..MACROBLOCK_LUMA_SIDE {
            // Row 0 (top field) reads ref row 16.
            assert_eq!(out.luma[0][col], plane[16 * W + (16 + col)] as i32);
            // Row 1 (bottom field) reads ref row 16 too (top ref field).
            assert_eq!(out.luma[1][col], plane[16 * W + (16 + col)] as i32);
            // Row 2 (top field) reads ref row 18.
            assert_eq!(out.luma[2][col], plane[18 * W + (16 + col)] as i32);
        }
    }

    #[test]
    fn bidirectional_averages_forward_and_backward() {
        // Forward reference all 100, backward reference all 200 ⇒
        // average (100 + 200 + 1) >> 1 = 150.
        let fwd_plane = make_plane(W, H, |_, _| 100);
        let bak_plane = make_plane(W, H, |_, _| 200);
        let fwd_c = make_plane(W / 2, H / 2, |_, _| 100);
        let bak_c = make_plane(W / 2, H / 2, |_, _| 200);
        let fl = ReferenceVop::new(&fwd_plane, W, H).unwrap();
        let bl = ReferenceVop::new(&bak_plane, W, H).unwrap();
        let fc = ReferenceVop::new(&fwd_c, W / 2, H / 2).unwrap();
        let bc = ReferenceVop::new(&bak_c, W / 2, H / 2).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let mut bank = FieldPmvBank::new();
        let out = field_bidirectional_prediction(
            &mut bank,
            [mv(0, 0), mv(0, 0), mv(0, 0), mv(0, 0)],
            &references,
            FieldReferenceFlags::ALL_TOP,
            16,
            16,
            0,
            FieldSampleMode::HalfSample,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(out.luma[row][col], 150, "luma[{row}][{col}]");
            }
        }
        for row in 0..MACROBLOCK_CHROMA_SIDE {
            for col in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(out.cb[row][col], 150);
                assert_eq!(out.cr[row][col], 150);
            }
        }
    }

    #[test]
    fn forward_updates_only_forward_pmv_slots() {
        let plane = make_plane(W, H, |_, _| 50);
        let cplane = make_plane(W / 2, H / 2, |_, _| 50);
        let luma = ReferenceVop::new(&plane, W, H).unwrap();
        let cb = ReferenceVop::new(&cplane, W / 2, H / 2).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &luma,
            forward_cb: &cb,
            forward_cr: &cb,
            backward_luma: &luma,
            backward_cb: &cb,
            backward_cr: &cb,
        };
        let mut bank = FieldPmvBank::new();
        field_forward_prediction(
            &mut bank,
            FieldMvDeltas {
                top: mv(2, 1),
                bottom: mv(3, 2),
            },
            &references,
            FieldReferenceFlags::ALL_TOP,
            16,
            16,
            0,
            FieldSampleMode::HalfSample,
        );
        // Forward slots updated (field rule: y = 2*(0/2 + d)).
        use crate::bvop_field_predictor::{PMV_BOT_BWD, PMV_BOT_FWD, PMV_TOP_BWD, PMV_TOP_FWD};
        assert_eq!(bank.get(PMV_TOP_FWD), mv(2, 2));
        assert_eq!(bank.get(PMV_BOT_FWD), mv(3, 4));
        // Backward slots untouched.
        assert_eq!(bank.get(PMV_TOP_BWD), mv(0, 0));
        assert_eq!(bank.get(PMV_BOT_BWD), mv(0, 0));
    }

    #[test]
    fn backward_draws_from_backward_reference() {
        // Forward all 10, backward all 90; backward-only must read 90.
        let fwd_plane = make_plane(W, H, |_, _| 10);
        let bak_plane = make_plane(W, H, |_, _| 90);
        let fc = make_plane(W / 2, H / 2, |_, _| 10);
        let bc = make_plane(W / 2, H / 2, |_, _| 90);
        let fl = ReferenceVop::new(&fwd_plane, W, H).unwrap();
        let bl = ReferenceVop::new(&bak_plane, W, H).unwrap();
        let fcv = ReferenceVop::new(&fc, W / 2, H / 2).unwrap();
        let bcv = ReferenceVop::new(&bc, W / 2, H / 2).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fcv,
            forward_cr: &fcv,
            backward_luma: &bl,
            backward_cb: &bcv,
            backward_cr: &bcv,
        };
        let mut bank = FieldPmvBank::new();
        let out = field_backward_prediction(
            &mut bank,
            FieldMvDeltas {
                top: mv(0, 0),
                bottom: mv(0, 0),
            },
            &references,
            FieldReferenceFlags::ALL_TOP,
            16,
            16,
            0,
            FieldSampleMode::HalfSample,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(out.luma[row][col], 90);
            }
        }
    }
}

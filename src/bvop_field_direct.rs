//! §7.7.2.2 interlaced **direct** mode motion-vector derivation for
//! B-VOPs.
//!
//! Interlaced direct mode is used when the co-located macroblock (same
//! coordinates) of the *future* reference VOP is itself field-predicted
//! (`field_prediction == 1`) — and not skipped / GMC / intra (those force
//! progressive direct mode, §7.6.9.5). Four derived field motion vectors
//! are computed from the future macroblock's two forward **field** motion
//! vectors `MV[i]` (`i == 0` top, `i == 1` bottom), a single transmitted
//! differential `MVD[0]`, and the §7.7.2.2 field-period temporal
//! references `TRB[i]` / `TRD[i]`:
//!
//! ```text
//!   for (i = 0; i < 2; i++) {
//!     mvf[i].x = (TRB[i] * MV[i].x) / TRD[i] + MVD[0].x;
//!     mvf[i].y = (TRB[i] * MV[i].y) / TRD[i] + MVD[0].y;
//!     mvb[i].x = (MVD[0].x == 0) ? (((TRB[i]-TRD[i]) * MV[i].x) / TRD[i])
//!                                : (mvf[i].x - MV[i].x);
//!     mvb[i].y = (MVD[0].y == 0) ? (((TRB[i]-TRD[i]) * MV[i].y) / TRD[i])
//!                                : (mvf[i].y - MV[i].y);
//!   }
//! ```
//!
//! `/` is integer division truncating toward zero (§3.4). `mvf[0]` /
//! `mvf[1]` are the top / bottom forward field MVs; `mvb[0]` / `mvb[1]`
//! the top / bottom backward field MVs. (Note the spec's pseudo code
//! lists the single bitstream delta as `MVD[0]` for both fields; the
//! `MVD[i]` index in the `?:` guard refers to that same `MVD[0]`.)
//! The *vertical* arithmetic runs on the field grid (the co-located
//! `MV[i].y` halved from its even frame-coordinate form, the sum
//! doubled back for the frame-coordinate `mc` call) so the §7.7.2
//! evenness invariant of field motion vectors is preserved — see
//! [`derive_field`] for the clause-level derivation.
//!
//! ## §7.7.2.2 field-period temporal references
//!
//! `TRB[i]` / `TRD[i]` are distances in time expressed in **field**
//! periods (Figure 7-48):
//!
//! ```text
//!   TRD[i] = 2*(T(future)//Tframe - T(past)//Tframe) + δ[i]
//!   TRB[i] = 2*(T(current)//Tframe - T(past)//Tframe) + δ[i]
//! ```
//!
//! where `T(future)` / `T(current)` / `T(past)` are the cumulative VOP
//! display times and `Tframe` is the frame period. `//` is integer
//! division. `δ[i]` (Table 7-16) is selected by the current field parity
//! (top `i==0` / bottom `i==1`), the reference field of the co-located
//! future macroblock for that field, and `top_field_first`. This module
//! takes the already-computed frame distances `trd_frame =
//! T(future)//Tframe - T(past)//Tframe` and `trb_frame =
//! T(current)//Tframe - T(past)//Tframe` (the same frame-period distances
//! the progressive §7.6.7 direct mode uses) and applies the `2*… + δ`
//! field-period conversion.
//!
//! ## Clean-room provenance
//!
//! Truth taken from `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`
//! §7.7.2.2 (interlaced direct mode pseudo code, Figure 7-48, Table
//! 7-16). No external MPEG-4 implementation consulted.

use crate::interlaced_information::FieldReference;
use crate::motion::MotionVector;

/// One field of the §7.7.2.2 interlaced-direct derivation: the future
/// macroblock's forward field motion vector `MV[i]` and the reference
/// field that macroblock selected for this field (drives the Table 7-16
/// `δ` lookup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColocatedFutureField {
    /// `MV[i]` — the co-located future P-VOP macroblock's forward field
    /// motion vector for this field (`i == 0` top, `i == 1` bottom).
    pub mv: MotionVector,
    /// The reference field the co-located future macroblock selected for
    /// this field (Table 7-16 row selector).
    pub reference_field: FieldReference,
}

/// The four derived field motion vectors of an interlaced-direct B-VOP
/// macroblock (§7.7.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterlacedDirectMvs {
    /// `mvf[0]` — top-field forward.
    pub forward_top: MotionVector,
    /// `mvf[1]` — bottom-field forward.
    pub forward_bottom: MotionVector,
    /// `mvb[0]` — top-field backward.
    pub backward_top: MotionVector,
    /// `mvb[1]` — bottom-field backward.
    pub backward_bottom: MotionVector,
}

/// Table 7-16 — selection of the parameter `δ`.
///
/// Indexed by the co-located future macroblock's reference fields for the
/// top (`i == 0`) and bottom (`i == 1`) field, and by `top_field_first`.
/// Returns `(δ[0], δ[1])` — the top-field and bottom-field offsets.
///
/// | top ref | bottom ref | tff==0 (δ0, δ1) | tff==1 (δ0, δ1) |
/// |---------|------------|-----------------|-----------------|
/// | Top     | Top        | (0, -1)         | (0, 1)          |
/// | Top     | Bottom     | (0, 0)          | (0, 0)          |
/// | Bottom  | Top        | (1, -1)         | (-1, 1)         |
/// | Bottom  | Bottom     | (1, 0)          | (-1, 0)         |
pub fn delta(
    top_reference_field: FieldReference,
    bottom_reference_field: FieldReference,
    top_field_first: bool,
) -> (i32, i32) {
    use FieldReference::{Bottom, Top};
    match (top_reference_field, bottom_reference_field, top_field_first) {
        (Top, Top, false) => (0, -1),
        (Top, Top, true) => (0, 1),
        (Top, Bottom, false) => (0, 0),
        (Top, Bottom, true) => (0, 0),
        (Bottom, Top, false) => (1, -1),
        (Bottom, Top, true) => (-1, 1),
        (Bottom, Bottom, false) => (1, 0),
        (Bottom, Bottom, true) => (-1, 0),
    }
}

/// Convert the §7.6.7 frame-period temporal distances into the §7.7.2.2
/// field-period `TRB[i]` / `TRD[i]` for one field:
/// `TRD = 2*trd_frame + δ`, `TRB = 2*trb_frame + δ`.
#[inline]
fn field_temporal(trb_frame: i32, trd_frame: i32, delta_i: i32) -> (i32, i32) {
    (2 * trb_frame + delta_i, 2 * trd_frame + delta_i)
}

/// Derive one field's forward / backward direct MVs from the future
/// macroblock's field MV, the transmitted delta, and that field's
/// `TRB` / `TRD`.
///
/// Returns `(mvf, mvb)` in **frame** half-sample coordinates (the form
/// [`crate::field_motion::mc`] consumes). `/` truncates toward zero
/// (§3.4, Rust `/`).
///
/// ## Vertical components run on the field grid
///
/// §7.7.2 fixes an invariant for every field motion vector: "the
/// vertical component of field motion vectors [is] always even (in
/// half- or quarter-pel frame coordinates)" — one field line spans two
/// frame lines, so a field MV's frame-coordinate vertical is always a
/// doubled field-grid value, and §7.7.2.1 recovers coded field MVs as
/// `MVy = 2 * (MVDy + Py / 2)` (the wire delta lives on the field
/// grid). The four derived vectors here are field motion vectors fed
/// straight into `field_motion_compensate_one_reference`, and `MVD[0]`
/// is the §7.7.2.2 "field direct mode" delta (decoded with
/// `f_code == 1`); were the sum `(TRB*MV)/TRD + MVD[0]` evaluated on
/// frame coordinates, an odd `MVD[0].y` would produce an odd `mvf.y`,
/// violating the invariant (and silently losing its low bit in the
/// §7.6.9.1 `mc` routine, whose field path masks `dy_halfpel` with
/// `y_incr == 2`). The §7.7.2.2 vertical arithmetic therefore runs in
/// field-grid units: the co-located `MV[i].y` (frame coordinates, even
/// by the invariant) is halved exactly, scaled and summed with
/// `MVD[0].y` on the field grid, and the result is doubled back to the
/// frame coordinates. Horizontal components have no field/frame split.
/// This reading is black-box-confirmed: a conformant reference decode
/// reconstructs a direct macroblock with `MVD[0].y == -21` at exactly
/// 21 frame lines (= 42 half-sample frame units) of vertical
/// displacement in both derived fields (see `tests/conformance.rs`).
fn derive_field(
    mv: MotionVector,
    delta_mv: MotionVector,
    trb: i32,
    trd: i32,
) -> (MotionVector, MotionVector) {
    // §7.7.2 invariant: field MVs are vertically even in frame coords.
    debug_assert!(mv.y % 2 == 0, "co-located field MV.y must be even");
    let mv_y_field = mv.y / 2;
    // mvf = TRB*MV/TRD + MVD[0]  (y on the field grid, then doubled).
    let mvf_y_field = (trb * mv_y_field) / trd + delta_mv.y;
    let mvf = MotionVector {
        x: (trb * mv.x) / trd + delta_mv.x,
        y: 2 * mvf_y_field,
    };
    // mvb component: when the delta component is 0, the scaled
    // (TRB-TRD)*MV/TRD form; otherwise mvf - MV (both identities on the
    // field grid for y; doubling distributes over the subtraction).
    let mvb = MotionVector {
        x: if delta_mv.x == 0 {
            ((trb - trd) * mv.x) / trd
        } else {
            mvf.x - mv.x
        },
        y: if delta_mv.y == 0 {
            2 * (((trb - trd) * mv_y_field) / trd)
        } else {
            2 * (mvf_y_field - mv_y_field)
        },
    };
    (mvf, mvb)
}

/// §7.7.2.2 interlaced-direct field motion-vector derivation.
///
/// * `top` / `bottom` — the co-located future macroblock's two forward
///   field MVs + their reference fields.
/// * `delta_mv` — the single transmitted differential `MVD[0]` (decoded
///   with `f_code == 1`, §7.7.2.2; shared by both fields).
/// * `trb_frame` / `trd_frame` — the §7.6.7 *frame-period* distances
///   (`T(current)//Tframe - T(past)//Tframe` and
///   `T(future)//Tframe - T(past)//Tframe`).
/// * `top_field_first` — the B-VOP's `top_field_first` flag.
///
/// Returns the four derived field MVs (`mvf[0..2]`, `mvb[0..2]`). The δ
/// lookup uses the *future macroblock's* per-field reference selections
/// (`top.reference_field`, `bottom.reference_field`).
pub fn interlaced_direct_mvs(
    top: ColocatedFutureField,
    bottom: ColocatedFutureField,
    delta_mv: MotionVector,
    trb_frame: i32,
    trd_frame: i32,
    top_field_first: bool,
) -> InterlacedDirectMvs {
    let (delta_top, delta_bottom) =
        delta(top.reference_field, bottom.reference_field, top_field_first);
    let (trb_top, trd_top) = field_temporal(trb_frame, trd_frame, delta_top);
    let (trb_bot, trd_bot) = field_temporal(trb_frame, trd_frame, delta_bottom);

    let (mvf0, mvb0) = derive_field(top.mv, delta_mv, trb_top, trd_top);
    let (mvf1, mvb1) = derive_field(bottom.mv, delta_mv, trb_bot, trd_bot);

    InterlacedDirectMvs {
        forward_top: mvf0,
        forward_bottom: mvf1,
        backward_top: mvb0,
        backward_bottom: mvb1,
    }
}

/// §7.7.2.2 interlaced-direct prediction macroblock assembly.
///
/// Runs the two §7.7.2.2 `field_motion_compensate_one_reference` calls of
/// the interlaced-direct pseudo code and averages them
/// `(fwd + bak + 1) >> 1`:
///
/// * **forward** uses `mvf[0]` (top) / `mvf[1]` (bottom) against the
///   *forward* (past) reference, with the co-located future macroblock's
///   `top` / `bottom` reference fields;
/// * **backward** uses `mvb[0]` (top) / `mvb[1]` (bottom) against the
///   *backward* (future) reference, with reference fields fixed at
///   top → Top, bottom → Bottom (the spec's `0, 1` literal).
///
/// The *printed* backward call passes `mvb[1]` in all four MV slots —
/// which contradicts the callee signature (§7.7.2 fixes slots 7–10 as
/// `top_x, top_y, bot_x, bot_y`), leaves the prose-defined `mvb[0]`
/// ("top field backward") unused, and breaks symmetry with the sibling
/// forward call (`mvf[0], mvf[0], mvf[1], mvf[1]`). The resolved reading
/// (top slot = `mvb[0]`) is confirmed against a conformant black-box
/// stream — see `tests/conformance.rs`
/// `interlaced_direct_bframes_stream_matches_reference_decode` and the
/// in-repo erratum `docs/video/mpeg4-visual/mpeg4-visual-errata.md` (E1).
///
/// `mb_x` / `mb_y` are the top-left luma pixel coordinates. Half-sample
/// luma interpolation (the pseudo code's `mc` routine).
#[allow(clippy::too_many_arguments)]
pub fn interlaced_direct_prediction(
    mvs: InterlacedDirectMvs,
    forward_top_ref: FieldReference,
    forward_bottom_ref: FieldReference,
    references: &crate::bvop_field_motion::BVopFieldReferences<'_>,
    mb_x: i32,
    mb_y: i32,
    rounding_type: u8,
) -> crate::reconstruct::InterPredictionMacroblock {
    use crate::field_motion::{field_motion_compensate_one_reference, FieldMotionVectors};
    use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

    let fwd = field_motion_compensate_one_reference(
        references.forward_luma,
        references.forward_cb,
        references.forward_cr,
        FieldMotionVectors {
            top: mvs.forward_top,
            bottom: mvs.forward_bottom,
        },
        forward_top_ref.as_bit(),
        forward_bottom_ref.as_bit(),
        mb_x,
        mb_y,
        rounding_type,
    );
    // §7.7.2.2 backward call, erratum-corrected (E1): mvb[0] drives the
    // top field, mvb[1] the bottom field — matching the callee signature
    // and the forward call's ordering. Reference fields are fixed at
    // (top → Top, bottom → Bottom), the pseudo code's `0, 1` literals.
    let bak = field_motion_compensate_one_reference(
        references.backward_luma,
        references.backward_cb,
        references.backward_cr,
        FieldMotionVectors {
            top: mvs.backward_top,
            bottom: mvs.backward_bottom,
        },
        false,
        true,
        mb_x,
        mb_y,
        rounding_type,
    );

    let mut out = crate::reconstruct::InterPredictionMacroblock::zero();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    fn field(x: i32, y: i32, r: FieldReference) -> ColocatedFutureField {
        ColocatedFutureField {
            mv: mv(x, y),
            reference_field: r,
        }
    }

    #[test]
    fn delta_table_all_rows() {
        use FieldReference::{Bottom, Top};
        // tff == 0
        assert_eq!(delta(Top, Top, false), (0, -1));
        assert_eq!(delta(Top, Bottom, false), (0, 0));
        assert_eq!(delta(Bottom, Top, false), (1, -1));
        assert_eq!(delta(Bottom, Bottom, false), (1, 0));
        // tff == 1
        assert_eq!(delta(Top, Top, true), (0, 1));
        assert_eq!(delta(Top, Bottom, true), (0, 0));
        assert_eq!(delta(Bottom, Top, true), (-1, 1));
        assert_eq!(delta(Bottom, Bottom, true), (-1, 0));
    }

    #[test]
    fn field_temporal_doubles_frame_distance_plus_delta() {
        // trb_frame=3, trd_frame=5, δ=-1 → TRB=5, TRD=9.
        assert_eq!(field_temporal(3, 5, -1), (5, 9));
        // δ=0 → exactly 2×.
        assert_eq!(field_temporal(3, 5, 0), (6, 10));
        // δ=1 → +1.
        assert_eq!(field_temporal(3, 5, 1), (7, 11));
    }

    #[test]
    fn zero_delta_uses_scaled_backward_form() {
        // Top/Bottom refs Top/Bottom → δ = (0, 0); TRB=2*trb, TRD=2*trd.
        // trb_frame=1, trd_frame=2 → TRB=2, TRD=4 for both fields.
        // MV[0]=(8,4). MVD=(0,0).
        // mvf = 2*8/4 = 4, 2*4/4 = 2 → (4,2).
        // mvb = (2-4)*8/4 = -4, (2-4)*4/4 = -2 → (-4,-2).
        let out = interlaced_direct_mvs(
            field(8, 4, FieldReference::Top),
            field(8, 4, FieldReference::Bottom),
            mv(0, 0),
            1,
            2,
            false,
        );
        assert_eq!(out.forward_top, mv(4, 2));
        assert_eq!(out.backward_top, mv(-4, -2));
        assert_eq!(out.forward_bottom, mv(4, 2));
        assert_eq!(out.backward_bottom, mv(-4, -2));
    }

    #[test]
    fn nonzero_delta_uses_mvf_minus_mv_backward_form() {
        // δ = (0,0) again. MV[0]=(8,4), MVD=(1,0).
        // mvf.x = 2*8/4 + 1 = 5; mvf.y = 2*4/4 + 0 = 2.
        // mvb.x (delta.x != 0) = mvf.x - MV.x = 5 - 8 = -3.
        // mvb.y (delta.y == 0) = (2-4)*4/4 = -2.
        let out = interlaced_direct_mvs(
            field(8, 4, FieldReference::Top),
            field(8, 4, FieldReference::Bottom),
            mv(1, 0),
            1,
            2,
            false,
        );
        assert_eq!(out.forward_top, mv(5, 2));
        assert_eq!(out.backward_top, mv(-3, -2));
    }

    #[test]
    fn delta_shifts_field_temporal_per_field() {
        // Top ref Bottom, bottom ref Top, tff=0 → δ = (1, -1). So the
        // top field gets TRB=2*trb+1 / TRD=2*trd+1, the bottom field
        // TRB=2*trb-1 / TRD=2*trd-1. With trb_frame=2, trd_frame=4:
        // top: TRB=5, TRD=9; bottom: TRB=3, TRD=7.
        // MV top=(9,0), MV bottom=(7,0), MVD=(0,0).
        // mvf top.x = 5*9/9 = 5; mvf bottom.x = 3*7/7 = 3.
        let out = interlaced_direct_mvs(
            field(9, 0, FieldReference::Bottom),
            field(7, 0, FieldReference::Top),
            mv(0, 0),
            2,
            4,
            false,
        );
        assert_eq!(out.forward_top.x, 5);
        assert_eq!(out.forward_bottom.x, 3);
    }

    #[test]
    fn delta_y_lands_on_field_grid_doubled_to_frame() {
        // §7.7.2 evenness invariant: MVD[0].y = -21 (an odd field-grid
        // value) must displace both derived fields by 21 *frame* lines
        // (-42 half-sample frame units), not 10.5. Co-located MVs zero
        // isolate the delta term; MVD.x stays on the shared
        // half-sample grid (no doubling).
        let out = interlaced_direct_mvs(
            field(0, 0, FieldReference::Top),
            field(0, 0, FieldReference::Bottom),
            mv(-1, -21),
            1,
            3,
            true,
        );
        assert_eq!(out.forward_top, mv(-1, -42));
        assert_eq!(out.forward_bottom, mv(-1, -42));
        // MVD != 0 → mvb = mvf - MV = mvf (co-located MV zero).
        assert_eq!(out.backward_top, mv(-1, -42));
        assert_eq!(out.backward_bottom, mv(-1, -42));
    }

    #[test]
    fn scaled_vertical_term_truncates_on_the_field_grid() {
        // δ = (0,0); TRB = 2, TRD = 4. MV.y = 6 (frame, even) → 3 on
        // the field grid. Field-grid scaling: (2*3)/4 = 1 (truncated),
        // doubled → 2 frame units. Frame-grid scaling would give
        // (2*6)/4 = 3 — an odd frame vertical, violating the §7.7.2
        // invariant. Backward: ((2-4)*3)/4 = -1 → -2 frame units.
        let out = interlaced_direct_mvs(
            field(0, 6, FieldReference::Top),
            field(0, 6, FieldReference::Bottom),
            mv(0, 0),
            1,
            2,
            false,
        );
        assert_eq!(out.forward_top, mv(0, 2));
        assert_eq!(out.backward_top, mv(0, -2));
    }

    #[test]
    fn derived_vertical_components_are_always_even() {
        // Exhaustive spot-check of the §7.7.2 invariant across
        // co-located MVs, deltas, and TRB/TRD combinations.
        for mvy in (-64..=64).step_by(2) {
            for dy in [-21, -3, 0, 1, 17] {
                for (trb, trd) in [(1, 3), (2, 3), (1, 2), (3, 4)] {
                    let out = interlaced_direct_mvs(
                        field(0, mvy, FieldReference::Top),
                        field(0, mvy, FieldReference::Bottom),
                        mv(0, dy),
                        trb,
                        trd,
                        false,
                    );
                    for v in [
                        out.forward_top,
                        out.forward_bottom,
                        out.backward_top,
                        out.backward_bottom,
                    ] {
                        assert_eq!(v.y % 2, 0, "mvy={mvy} dy={dy} trb={trb} trd={trd}");
                    }
                }
            }
        }
    }

    #[test]
    fn truncation_toward_zero_for_negative_products() {
        // δ = (0,0); TRB=2, TRD=4. MV=(-3,0), MVD=(0,0).
        // mvf.x = 2*-3/4 = -6/4 = -1 (truncates toward 0).
        let out = interlaced_direct_mvs(
            field(-3, 0, FieldReference::Top),
            field(-3, 0, FieldReference::Bottom),
            mv(0, 0),
            1,
            2,
            false,
        );
        assert_eq!(out.forward_top.x, -1);
    }

    #[test]
    fn backward_prediction_uses_mvb0_for_top_field() {
        // Erratum E1 regression: the backward field MC must feed mvb[0]
        // to the top field and mvb[1] to the bottom field (the printed
        // pseudo code passes mvb[1] for both). Derive mvb[0] = (0,0) and
        // mvb[1] = (-4,0): top MV=(0,0), bottom MV=(8,0), MVD=(0,0),
        // δ=(0,0) (refs Top/Bottom), trb_frame=1, trd_frame=2 →
        // mvb[1].x = (2-4)*8/4 = -4 (-2 full pel), mvb[0].x = 0.
        use crate::bvop_field_motion::BVopFieldReferences;
        use crate::half_sample::ReferenceVop;

        let mvs = interlaced_direct_mvs(
            field(0, 0, FieldReference::Top),
            field(8, 0, FieldReference::Bottom),
            mv(0, 0),
            1,
            2,
            false,
        );
        assert_eq!(mvs.backward_top, mv(0, 0));
        assert_eq!(mvs.backward_bottom, mv(-4, 0));

        // Forward reference: constant 0. Backward reference: horizontal
        // gradient value == x. Output rows of the top field (even y)
        // must sample the backward gradient unshifted; bottom rows (odd
        // y) shifted by -2. Under the printed (defective) call both
        // parities would be shifted.
        let fwd = vec![0u8; 48 * 48];
        let mut bak = vec![0u8; 48 * 48];
        for y in 0..48 {
            for x in 0..48 {
                bak[y * 48 + x] = x as u8;
            }
        }
        let fwd_c = vec![0u8; 24 * 24];
        let bak_c = vec![0u8; 24 * 24];
        let fl = ReferenceVop::new(&fwd, 48, 48).unwrap();
        let bl = ReferenceVop::new(&bak, 48, 48).unwrap();
        let fc = ReferenceVop::new(&fwd_c, 24, 24).unwrap();
        let bc = ReferenceVop::new(&bak_c, 24, 24).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let pred = interlaced_direct_prediction(
            mvs,
            FieldReference::Top,
            FieldReference::Bottom,
            &references,
            16,
            16,
            0,
        );
        // Top field row 0: (0 + (16+x) + 1) >> 1; bottom field row 1:
        // (0 + (14+x) + 1) >> 1 — one gray level below the top row.
        for x in 0..16 {
            let top = (16 + x as i32 + 1) >> 1;
            let bot = (14 + x as i32 + 1) >> 1;
            assert_eq!(pred.luma[0][x], top, "top row x={x}");
            assert_eq!(pred.luma[1][x], bot, "bottom row x={x}");
        }
    }

    #[test]
    fn prediction_averages_forward_and_backward_zero_mv() {
        use crate::bvop_field_motion::BVopFieldReferences;
        use crate::half_sample::ReferenceVop;

        // All-zero derived MVs (zero future MV + zero delta) ⇒ each field
        // copies its reference field. Forward ref 80, backward 160 ⇒
        // average (80 + 160 + 1) >> 1 = 120.
        let mvs = interlaced_direct_mvs(
            field(0, 0, FieldReference::Top),
            field(0, 0, FieldReference::Bottom),
            mv(0, 0),
            1,
            2,
            false,
        );
        assert_eq!(mvs.forward_top, mv(0, 0));
        assert_eq!(mvs.backward_bottom, mv(0, 0));

        let fwd = vec![80u8; 48 * 48];
        let bak = vec![160u8; 48 * 48];
        let fwd_c = vec![80u8; 24 * 24];
        let bak_c = vec![160u8; 24 * 24];
        let fl = ReferenceVop::new(&fwd, 48, 48).unwrap();
        let bl = ReferenceVop::new(&bak, 48, 48).unwrap();
        let fc = ReferenceVop::new(&fwd_c, 24, 24).unwrap();
        let bc = ReferenceVop::new(&bak_c, 24, 24).unwrap();
        let references = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let pred = interlaced_direct_prediction(
            mvs,
            FieldReference::Top,
            FieldReference::Bottom,
            &references,
            16,
            16,
            0,
        );
        for row in pred.luma {
            for px in row {
                assert_eq!(px, 120);
            }
        }
        for row in pred.cb {
            for px in row {
                assert_eq!(px, 120);
            }
        }
    }
}

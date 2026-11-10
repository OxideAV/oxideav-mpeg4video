//! §7.6.8 four-PMV motion-vector predictor bank for interlaced B-VOPs.
//!
//! In an interlaced B-VOP a macroblock may be field-predicted; the
//! motion-vector predictor is no longer the single progressive PMV but a
//! bank of **four** prediction motion vectors (Table 7-14):
//!
//! | Index | Function              |
//! |-------|-----------------------|
//! | 0     | Top field forward     |
//! | 1     | Bottom field forward  |
//! | 2     | Top field backward    |
//! | 3     | Bottom field backward |
//!
//! Each macroblock mode reads a subset of the bank, reconstructs its
//! motion vectors by adding the transmitted `MVD[]`, and writes the
//! reconstructed vectors back into the bank for the next macroblock
//! (Table 7-15):
//!
//! | Mode                | PMVs used | PMVs updated |
//! |---------------------|-----------|--------------|
//! | Direct              | none      | none         |
//! | Frame forward       | 0         | 0,1          |
//! | Frame backward      | 2         | 2,3          |
//! | Frame bidirectional | 0,2       | 0,1,2,3      |
//! | Field forward       | 0,1       | 0,1          |
//! | Field backward      | 2,3       | 2,3          |
//! | Field bidirectional | 0,1,2,3   | 0,1,2,3      |
//!
//! Reconstruction of a *field* component follows the §7.6.8 pseudo-code
//! `PMV[k].y = 2 * (PMV[k].y / 2 + MVD[i].y)` so the vertical component
//! is always even in half-pel frame coordinates; the horizontal is the
//! plain `PMV[k].x = PMV[k].x + MVD[i].x`. A *frame* macroblock sets the
//! two field PMVs (top + bottom) of each used direction to the same
//! frame value (§7.6.8). The bank is reset to zero at the start of each
//! macroblock row; skipped and direct-mode macroblocks do **not** zero
//! it.
//!
//! This module owns the predictor *bank state machine* — the PMV
//! add/update bookkeeping. The motion-compensated pixel assembly is the
//! `field_motion_compensate_one_reference()` routine in
//! [`crate::field_motion`]; this module produces the `(top, bottom)`
//! field motion-vector pairs that routine consumes.

use crate::motion::MotionVector;

/// The four interlaced-B-VOP prediction motion vectors (Table 7-14),
/// indexed `[0]`=top-fwd, `[1]`=bot-fwd, `[2]`=top-bwd, `[3]`=bot-bwd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldPmvBank {
    pmv: [MotionVector; 4],
    /// `vop_fcode_forward` / `vop_fcode_backward` when the bank applies
    /// the §7.6.3 `[low:high]` reduction to the field vectors it
    /// reconstructs (`0` = no reduction, the plain §7.6.8 add).
    fcodes: [u8; 2],
}

impl Default for FieldPmvBank {
    fn default() -> Self {
        FieldPmvBank {
            pmv: [MotionVector { x: 0, y: 0 }; 4],
            fcodes: [0; 2],
        }
    }
}

/// Table 7-14 PMV slot identifiers.
pub const PMV_TOP_FWD: usize = 0;
/// Bottom-field forward predictor slot.
pub const PMV_BOT_FWD: usize = 1;
/// Top-field backward predictor slot.
pub const PMV_TOP_BWD: usize = 2;
/// Bottom-field backward predictor slot.
pub const PMV_BOT_BWD: usize = 3;

/// One direction's reconstructed field motion vectors (top + bottom),
/// e.g. the forward pair or the backward pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMvDirection {
    /// Top-field motion vector.
    pub top: MotionVector,
    /// Bottom-field motion vector.
    pub bottom: MotionVector,
}

/// Reconstruct one field component from a predictor slot value and its
/// differential per the §7.6.8 pseudo-code:
/// `x = PMVx + MVDx`, `y = 2 * (PMVy / 2 + MVDy)`. `/` truncates toward
/// zero (Rust `/` matches the spec).
///
/// With a non-zero `fcode` each component is reduced into the Table
/// 7-9 `[low:high]` range on its own grid — the horizontal in frame
/// units, the vertical in **field** units before the doubling
/// (§7.6.3: the general decoding process, range reduction included,
/// is "valid for the motion vector decoding in interlaced/progressive
/// P-, S(GMC)- and B-VOPs except that the generation of the predictor
/// (Px, Py) may be different"; the units of the vertical reduction
/// are those of erratum E4 in `mpeg4-visual-errata.md`). For a stream
/// whose field vectors already lie inside the range the reduction is
/// the identity.
#[inline]
fn reconstruct_field_component(pmv: MotionVector, mvd: MotionVector, fcode: u8) -> MotionVector {
    let (x, y_field) = (pmv.x + mvd.x, pmv.y / 2 + mvd.y);
    if fcode == 0 {
        return MotionVector { x, y: 2 * y_field };
    }
    let f = 1i32 << (fcode.clamp(1, 7) - 1);
    let (low, high, range) = (-32 * f, 32 * f - 1, 64 * f);
    let reduce = |mut v: i32| {
        if v < low {
            v += range;
        }
        if v > high {
            v -= range;
        }
        v
    };
    MotionVector {
        x: reduce(x),
        y: 2 * reduce(y_field),
    }
}

impl FieldPmvBank {
    /// A zeroed bank — the row-start state (§7.6.8).
    pub fn new() -> Self {
        Self::default()
    }

    /// A zeroed bank whose **field** reconstructions apply the §7.6.3
    /// `[low:high]` reduction of `vop_fcode_forward` (slots 0, 1) and
    /// `vop_fcode_backward` (slots 2, 3): horizontal components in
    /// frame units, vertical components in field units before the
    /// doubling (erratum E4). This is the bank a B-VOP decode uses;
    /// [`FieldPmvBank::new`] keeps the plain §7.6.8 add.
    pub fn with_fcodes(vop_fcode_forward: u8, vop_fcode_backward: u8) -> Self {
        FieldPmvBank {
            pmv: [MotionVector { x: 0, y: 0 }; 4],
            fcodes: [vop_fcode_forward, vop_fcode_backward],
        }
    }

    /// Reset the bank to zero at the start of a macroblock row (§7.6.8).
    pub fn reset_row(&mut self) {
        self.pmv = [MotionVector { x: 0, y: 0 }; 4];
    }

    /// Read-only access to one PMV slot.
    pub fn get(&self, slot: usize) -> MotionVector {
        self.pmv[slot]
    }

    /// Direct mode (Table 7-15): no PMVs used, none updated. The bank is
    /// left untouched.
    pub fn direct(&self) {}

    /// Frame forward mode (Table 7-15): uses PMV[0], updates PMV[0] and
    /// PMV[1] to the same reconstructed frame vector. Returns the
    /// reconstructed forward MV.
    ///
    /// A frame vector reconstructs with the progressive rule
    /// `MV = PMV + MVD` (no field doubling); both forward field PMVs are
    /// then set to that value per §7.6.8.
    pub fn frame_forward(&mut self, mvd: MotionVector) -> MotionVector {
        let mv = MotionVector {
            x: self.pmv[PMV_TOP_FWD].x + mvd.x,
            y: self.pmv[PMV_TOP_FWD].y + mvd.y,
        };
        self.pmv[PMV_TOP_FWD] = mv;
        self.pmv[PMV_BOT_FWD] = mv;
        mv
    }

    /// Frame backward mode (Table 7-15): uses PMV[2], updates PMV[2] and
    /// PMV[3]. Returns the reconstructed backward MV.
    pub fn frame_backward(&mut self, mvd: MotionVector) -> MotionVector {
        let mv = MotionVector {
            x: self.pmv[PMV_TOP_BWD].x + mvd.x,
            y: self.pmv[PMV_TOP_BWD].y + mvd.y,
        };
        self.pmv[PMV_TOP_BWD] = mv;
        self.pmv[PMV_BOT_BWD] = mv;
        mv
    }

    /// Frame forward-direction update without reconstruction: set both
    /// forward field PMVs (slots 0, 1) to an already-reconstructed
    /// frame vector (Table 7-15 "Frame forward → updates 0,1"; §7.7.2.2
    /// "when a frame macroblock is decoded, the two field PMVs for each
    /// prediction direction are set to the same frame value"). Used by
    /// the driver when the frame vector reconstructs through the
    /// §7.6.3 wrap rather than the bank's plain add.
    pub fn set_frame_forward(&mut self, mv: MotionVector) {
        self.pmv[PMV_TOP_FWD] = mv;
        self.pmv[PMV_BOT_FWD] = mv;
    }

    /// Frame backward-direction update without reconstruction: set both
    /// backward field PMVs (slots 2, 3) to an already-reconstructed
    /// frame vector (Table 7-15 "Frame backward → updates 2,3").
    pub fn set_frame_backward(&mut self, mv: MotionVector) {
        self.pmv[PMV_TOP_BWD] = mv;
        self.pmv[PMV_BOT_BWD] = mv;
    }

    /// Frame bidirectional mode (Table 7-15): uses PMV[0] + PMV[2],
    /// updates all four. `mvd_fwd` / `mvd_bwd` are the two transmitted
    /// differentials. Returns `(forward, backward)` reconstructed MVs.
    pub fn frame_bidirectional(
        &mut self,
        mvd_fwd: MotionVector,
        mvd_bwd: MotionVector,
    ) -> (MotionVector, MotionVector) {
        let fwd = self.frame_forward(mvd_fwd);
        let bwd = self.frame_backward(mvd_bwd);
        (fwd, bwd)
    }

    /// Field forward mode (`mb_type == "0001"`, `field_prediction == 1`):
    /// uses PMV[0] + PMV[1], updates them. `mvd_top` / `mvd_bot` are the
    /// two transmitted differentials (top field, bottom field).
    /// Reconstructs with the §7.6.8 field rule and returns the pair.
    pub fn field_forward(
        &mut self,
        mvd_top: MotionVector,
        mvd_bot: MotionVector,
    ) -> FieldMvDirection {
        let top = reconstruct_field_component(self.pmv[PMV_TOP_FWD], mvd_top, self.fcodes[0]);
        let bottom = reconstruct_field_component(self.pmv[PMV_BOT_FWD], mvd_bot, self.fcodes[0]);
        self.pmv[PMV_TOP_FWD] = top;
        self.pmv[PMV_BOT_FWD] = bottom;
        FieldMvDirection { top, bottom }
    }

    /// Field backward mode (`mb_type == "001"`, `field_prediction == 1`):
    /// uses PMV[2] + PMV[3], updates them.
    ///
    /// The §7.7.2.2 backward pseudo-code as printed reads
    /// `PMV[3].x = PMV[1].x + MVD[1].x` — the *forward* bottom slot —
    /// for the backward bottom-field x while taking `PMV[3].y` for its
    /// y. That contradicts Table 7-15 ("Field backward: PMVs used 2,3"),
    /// the sibling forward listing (`PMV[1].x = PMV[1].x + MVD[1].x`,
    /// same-slot), and its own y component; it is the same class of
    /// printed defect as the mvb[] slot swap documented in
    /// `docs/video/mpeg4-visual/mpeg4-visual-errata.md`. The corrected
    /// same-slot reading (`PMV[3].x = PMV[3].x + MVD[1].x`) is
    /// arbitrated against a conformant black-box stream
    /// (tests/conformance.rs, interlaced-direct fixture: field-backward
    /// macroblocks only reproduce the reference decode with the
    /// same-slot read).
    pub fn field_backward(
        &mut self,
        mvd_top: MotionVector,
        mvd_bot: MotionVector,
    ) -> FieldMvDirection {
        let top = reconstruct_field_component(self.pmv[PMV_TOP_BWD], mvd_top, self.fcodes[1]);
        let bottom = reconstruct_field_component(self.pmv[PMV_BOT_BWD], mvd_bot, self.fcodes[1]);
        self.pmv[PMV_TOP_BWD] = top;
        self.pmv[PMV_BOT_BWD] = bottom;
        FieldMvDirection { top, bottom }
    }

    /// Field bidirectional mode (`mb_type == "01"`, `field_prediction ==
    /// 1`): uses all four PMVs, updates all four. `mvd` is the four
    /// transmitted differentials in bitstream order (PMV[0..4]). Returns
    /// `(forward, backward)` field pairs.
    ///
    /// Per §7.6.8 every slot is reconstructed in place with the field
    /// rule `PMV[k].y = 2 * (PMV[k].y / 2 + MVD[k].y)`.
    pub fn field_bidirectional(
        &mut self,
        mvd: [MotionVector; 4],
    ) -> (FieldMvDirection, FieldMvDirection) {
        for (k, (slot, mvd_k)) in self.pmv.iter_mut().zip(mvd.iter()).enumerate() {
            *slot = reconstruct_field_component(*slot, *mvd_k, self.fcodes[k / 2]);
        }
        (
            FieldMvDirection {
                top: self.pmv[PMV_TOP_FWD],
                bottom: self.pmv[PMV_BOT_FWD],
            },
            FieldMvDirection {
                top: self.pmv[PMV_TOP_BWD],
                bottom: self.pmv[PMV_BOT_BWD],
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    #[test]
    fn field_reconstruction_reduces_in_field_units_under_fcodes() {
        // fcode 1: [low:high] = [-32:31]. Forward top PMV (0, 30):
        // Py/2 = 15; MVDy = 17 gives 32 -> reduced to -32 field units,
        // doubled to -64 frame units; MVDy = 16 gives 31 -> 62.
        let mut bank = FieldPmvBank::with_fcodes(1, 2);
        bank.frame_forward(mv(30, 30));
        let d = bank.field_forward(mv(5, 16), mv(-70 + 64, 17));
        assert_eq!(d.top, mv(-29, 62));
        assert_eq!(d.bottom, mv(24, -64));
        // The backward pair reduces under vop_fcode_backward = 2:
        // [-64:63]; 0 + 63 stays, 0 - 64 stays, both doubled.
        let d = bank.field_backward(mv(0, 63), mv(0, -64));
        assert_eq!(d.top, mv(0, 126));
        assert_eq!(d.bottom, mv(0, -128));
        // 126 / 2 + 1 = 64 > 63 -> -64 -> -128.
        let d = bank.field_backward(mv(0, 1), mv(0, 0));
        assert_eq!(d.top, mv(0, -128));
        // The plain bank keeps the unreduced §7.6.8 add.
        let mut plain = FieldPmvBank::new();
        plain.frame_forward(mv(30, 30));
        assert_eq!(plain.field_forward(mv(5, 16), mv(0, 17)).bottom, mv(30, 64));
        // The row reset keeps the range configuration.
        bank.reset_row();
        assert_eq!(bank.field_forward(mv(0, 32 - 64), mv(0, 0)).top, mv(0, -64));
    }

    #[test]
    fn new_and_reset_are_zero() {
        let mut bank = FieldPmvBank::new();
        for s in 0..4 {
            assert_eq!(bank.get(s), mv(0, 0));
        }
        bank.frame_forward(mv(5, 6));
        bank.reset_row();
        for s in 0..4 {
            assert_eq!(bank.get(s), mv(0, 0));
        }
    }

    #[test]
    fn direct_mode_leaves_bank_untouched() {
        let mut bank = FieldPmvBank::new();
        bank.frame_forward(mv(3, 4));
        let before = bank;
        bank.direct();
        assert_eq!(bank, before, "direct must not touch the bank");
    }

    #[test]
    fn frame_forward_sets_both_forward_pmvs_equal() {
        let mut bank = FieldPmvBank::new();
        let out = bank.frame_forward(mv(7, -3));
        assert_eq!(out, mv(7, -3));
        assert_eq!(bank.get(PMV_TOP_FWD), mv(7, -3));
        assert_eq!(bank.get(PMV_BOT_FWD), mv(7, -3));
        // Backward slots untouched.
        assert_eq!(bank.get(PMV_TOP_BWD), mv(0, 0));
        assert_eq!(bank.get(PMV_BOT_BWD), mv(0, 0));
        // Second MB adds to the predictor.
        let out2 = bank.frame_forward(mv(2, 1));
        assert_eq!(out2, mv(9, -2));
    }

    #[test]
    fn frame_backward_uses_and_updates_backward_slots() {
        let mut bank = FieldPmvBank::new();
        let out = bank.frame_backward(mv(-4, 8));
        assert_eq!(out, mv(-4, 8));
        assert_eq!(bank.get(PMV_TOP_BWD), mv(-4, 8));
        assert_eq!(bank.get(PMV_BOT_BWD), mv(-4, 8));
        assert_eq!(bank.get(PMV_TOP_FWD), mv(0, 0));
    }

    #[test]
    fn frame_bidirectional_updates_all_four() {
        let mut bank = FieldPmvBank::new();
        let (f, b) = bank.frame_bidirectional(mv(1, 2), mv(3, 4));
        assert_eq!(f, mv(1, 2));
        assert_eq!(b, mv(3, 4));
        assert_eq!(bank.get(PMV_TOP_FWD), mv(1, 2));
        assert_eq!(bank.get(PMV_BOT_FWD), mv(1, 2));
        assert_eq!(bank.get(PMV_TOP_BWD), mv(3, 4));
        assert_eq!(bank.get(PMV_BOT_BWD), mv(3, 4));
    }

    #[test]
    fn field_forward_vertical_is_even_via_field_rule() {
        // PMV starts at 0; MVDy = 3 ⇒ y = 2 * (0/2 + 3) = 6 (even).
        let mut bank = FieldPmvBank::new();
        let out = bank.field_forward(mv(2, 3), mv(-1, 5));
        assert_eq!(out.top, mv(2, 6));
        assert_eq!(out.bottom, mv(-1, 10));
        // Both vertical components even.
        assert_eq!(out.top.y % 2, 0);
        assert_eq!(out.bottom.y % 2, 0);
        assert_eq!(bank.get(PMV_TOP_FWD), mv(2, 6));
        assert_eq!(bank.get(PMV_BOT_FWD), mv(-1, 10));
    }

    #[test]
    fn field_rule_truncates_predictor_half_toward_zero() {
        // Seed PMV[0] with an odd-ish even value to exercise PMVy/2.
        let mut bank = FieldPmvBank::new();
        // First field-forward: PMV starts 0 ⇒ top.y = 2*(0+2)=4.
        bank.field_forward(mv(0, 2), mv(0, 2));
        assert_eq!(bank.get(PMV_TOP_FWD).y, 4);
        // Second: PMVy=4 ⇒ 4/2=2; MVDy=-1 ⇒ y = 2*(2 + -1) = 2.
        let out = bank.field_forward(mv(0, -1), mv(0, 0));
        assert_eq!(out.top.y, 2);
    }

    #[test]
    fn field_backward_bottom_x_reads_backward_bottom_slot() {
        // The printed §7.7.2.2 backward listing reads PMV[1].x (a
        // *forward* slot) for the backward bottom-field x — a printed
        // defect (see the `field_backward` doc comment); the corrected
        // same-slot reading uses PMV[3]. A forward decode must
        // therefore leave the backward reconstruction untouched.
        let mut bank = FieldPmvBank::new();
        bank.field_forward(mv(11, 0), mv(22, 0)); // PMV[1].x = 22
        let out = bank.field_backward(mv(1, 1), mv(2, 1));
        // top.x = PMV[2].x(0) + 1 = 1.
        assert_eq!(out.top.x, 1);
        // bottom.x = PMV[3].x(0) + 2 = 2 — NOT the forward slot's 22.
        assert_eq!(out.bottom.x, 2);
    }

    #[test]
    fn field_bidirectional_updates_all_four_with_field_rule() {
        let mut bank = FieldPmvBank::new();
        let (f, b) = bank.field_bidirectional([mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4)]);
        assert_eq!(f.top, mv(1, 2));
        assert_eq!(f.bottom, mv(2, 4));
        assert_eq!(b.top, mv(3, 6));
        assert_eq!(b.bottom, mv(4, 8));
        // All bank slots match.
        assert_eq!(bank.get(PMV_TOP_FWD), mv(1, 2));
        assert_eq!(bank.get(PMV_BOT_BWD), mv(4, 8));
    }

    #[test]
    fn predictors_persist_across_macroblocks_until_row_reset() {
        let mut bank = FieldPmvBank::new();
        bank.frame_forward(mv(10, 4));
        // Direct MB must not zero the predictor.
        bank.direct();
        let out = bank.frame_forward(mv(0, 0));
        assert_eq!(out, mv(10, 4), "direct MB must not zero predictors");
        bank.reset_row();
        let out2 = bank.frame_forward(mv(0, 0));
        assert_eq!(out2, mv(0, 0), "row reset zeroes predictors");
    }
}

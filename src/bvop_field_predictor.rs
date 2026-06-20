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
}

impl Default for FieldPmvBank {
    fn default() -> Self {
        FieldPmvBank {
            pmv: [MotionVector { x: 0, y: 0 }; 4],
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
#[inline]
fn reconstruct_field_component(pmv: MotionVector, mvd: MotionVector) -> MotionVector {
    MotionVector {
        x: pmv.x + mvd.x,
        y: 2 * (pmv.y / 2 + mvd.y),
    }
}

impl FieldPmvBank {
    /// A zeroed bank — the row-start state (§7.6.8).
    pub fn new() -> Self {
        Self::default()
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
        let top = reconstruct_field_component(self.pmv[PMV_TOP_FWD], mvd_top);
        let bottom = reconstruct_field_component(self.pmv[PMV_BOT_FWD], mvd_bot);
        self.pmv[PMV_TOP_FWD] = top;
        self.pmv[PMV_BOT_FWD] = bottom;
        FieldMvDirection { top, bottom }
    }

    /// Field backward mode (`mb_type == "001"`, `field_prediction == 1`):
    /// uses PMV[2] + PMV[3], updates them.
    ///
    /// Mirrors the §7.6.8 backward pseudo-code, whose top-field update
    /// reads PMV[2] but whose bottom-field update reads PMV[1] for its
    /// `x` (the spec's `PMV[3].x = PMV[1].x + MVD[1].x`) while taking
    /// PMV[3].y for its `y`. This asymmetry is reproduced faithfully.
    pub fn field_backward(
        &mut self,
        mvd_top: MotionVector,
        mvd_bot: MotionVector,
    ) -> FieldMvDirection {
        let top = reconstruct_field_component(self.pmv[PMV_TOP_BWD], mvd_top);
        // §7.6.8: PMV[3].x = PMV[1].x + MVD[1].x (note PMV[1], not [3]);
        //          PMV[3].y = 2 * (PMV[3].y / 2 + MVD[1].y).
        let bottom = MotionVector {
            x: self.pmv[PMV_BOT_FWD].x + mvd_bot.x,
            y: 2 * (self.pmv[PMV_BOT_BWD].y / 2 + mvd_bot.y),
        };
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
        for (slot, mvd_k) in self.pmv.iter_mut().zip(mvd.iter()) {
            *slot = reconstruct_field_component(*slot, *mvd_k);
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
    fn field_backward_bottom_x_reads_forward_bottom_slot() {
        // §7.6.8 quirk: field-backward bottom-field x reads PMV[1]
        // (bottom forward), not PMV[3].
        let mut bank = FieldPmvBank::new();
        bank.field_forward(mv(11, 0), mv(22, 0)); // PMV[1].x = 22
        let out = bank.field_backward(mv(1, 1), mv(2, 1));
        // top.x = PMV[2].x(0) + 1 = 1.
        assert_eq!(out.top.x, 1);
        // bottom.x = PMV[1].x(22) + 2 = 24.
        assert_eq!(out.bottom.x, 24);
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

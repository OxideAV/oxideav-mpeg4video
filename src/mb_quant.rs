//! Per-macroblock quantiser modulation for the encoder — the
//! `dquant` (§6.3.7 Table 6-32, I-/P-VOPs) and `dbquant` (Table 6-33,
//! B-VOPs) differentials.
//!
//! The syntax makes the quantiser a *running* value: `vop_quant`
//! seeds it, every `dquant` / `dbquant` moves it, a `video_packet_header`
//! re-seeds it with `quant_scale`, and the result is clipped to
//! `[1, 2^quant_precision − 1]` after each step (§6.3.6 / §6.3.7).
//! The encoder mirrors the decoder's running value exactly and plans
//! each macroblock's step against it:
//!
//! * **activity classes** — the classic perceptual rule: flat
//!   macroblocks (low mean-absolute-deviation luma) get a finer
//!   quantiser, busy ones a coarser one, in a ±2 band around the VOP
//!   quantiser ([`activity_class`]);
//! * **`dquant` planning** ([`plan_dquant`]) — the step towards the
//!   target is limited to the Table 6-32 alphabet `{−2, −1, +1, +2}`
//!   (zero is "no dquant": the macroblock is coded as the plain
//!   `inter` / `intra` type);
//! * **`dbquant` planning** ([`plan_dbquant`]) — the Table 6-33
//!   alphabet is `{−2, 0, +2}` and the field is only present on
//!   non-direct macroblocks with `cbpb != 0`, so the planner also
//!   tells the caller when the step could not be carried;
//! * **budget regulation** ([`MbRegulator`]) — when the VOP carries a
//!   bit budget ([`MbBudget`], set by the budget-driven mode of
//!   `crate::rate_control`), the target quantiser of every macroblock
//!   is derived from the bits spent so far against the share of the
//!   budget the macroblock layer was expected to have consumed by that
//!   point (an activity-weighted allocation), so that the `dquant` /
//!   `dbquant` steps steer the VOP onto its budget instead of only
//!   following the activity classes.
//!
//! Provenance: ISO/IEC 14496-2:2004 (3rd edition) §6.3.6 / §6.3.7
//! (`dquant`, `dbquant`, Tables 6-32 / 6-33) read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.

/// Maximum quantiser scale under the default `quant_precision == 5`.
pub const MAX_QP: u32 = 31;

/// The activity class of a macroblock: a signed quantiser offset in
/// `-2..=2` from its mean-removed luma activity (the sum over the 256
/// samples of `|s − mean|`, i.e. `256 × MAD`). Flat blocks (MAD < 2)
/// go two steps finer, smooth ones (MAD < 5) one step finer; busy
/// blocks (MAD > 15) one step coarser, very busy ones (MAD > 30) two.
pub fn activity_class(activity: u32) -> i32 {
    match activity {
        a if a < 512 => -2,
        a if a < 1280 => -1,
        a if a > 7680 => 2,
        a if a > 3840 => 1,
        _ => 0,
    }
}

/// The macroblock's target quantiser: `vop_qp + class`, clipped to
/// `[1, MAX_QP]`.
pub fn target_qp(vop_qp: u32, class: i32) -> u32 {
    (vop_qp as i64 + i64::from(class)).clamp(1, i64::from(MAX_QP)) as u32
}

/// Plan an I-/P-VOP macroblock's `dquant` towards `target` from the
/// `running` quantiser: returns the quantiser the macroblock is coded
/// with and the Table 6-32 delta to emit (`None` = no `dquant`, plain
/// macroblock type). Steps are limited to ±2 per macroblock.
pub fn plan_dquant(running: u32, target: u32) -> (u32, Option<i8>) {
    let delta = (target as i64 - running as i64).clamp(-2, 2);
    if delta == 0 {
        (running, None)
    } else {
        let qp = (running as i64 + delta).clamp(1, i64::from(MAX_QP)) as u32;
        (qp, Some(delta as i8))
    }
}

/// Plan a B-VOP macroblock's `dbquant` towards `target` from the
/// `running` quantiser: Table 6-33 only offers ±2, so a one-step
/// distance is left for a later macroblock (returns `(running,
/// None)`); the caller emits the delta only when the syntax carries
/// `dbquant` (non-direct, `cbpb != 0`) and commits the new running
/// value in that case alone.
pub fn plan_dbquant(running: u32, target: u32) -> (u32, Option<i8>) {
    let distance = target as i64 - running as i64;
    if distance >= 2 && running + 2 <= MAX_QP {
        (running + 2, Some(2))
    } else if distance <= -2 && running >= 3 {
        (running - 2, Some(-2))
    } else {
        (running, None)
    }
}

/// A VOP-level bit budget handed to the macroblock loop (the
/// budget-driven mode of `crate::rate_control`): the per-macroblock
/// `dquant` / `dbquant` steps then regulate the spend against it
/// through an [`MbRegulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbBudget {
    /// Target size of the whole VOP unit in bits (VOP header and
    /// packet headers included — the regulator measures the writer's
    /// total position, so the header bits count as spend).
    pub target_bits: u32,
    /// Half-width of the quantiser band the regulator may roam around
    /// `vop_quant` (a wide band tracks the budget tightly, a narrow
    /// one keeps the quantiser — and the quality — more uniform).
    pub band: u8,
}

/// The per-macroblock budget regulator: maps the bits spent before a
/// macroblock against the expected spend at that point onto a target
/// quantiser (the caller then steps towards it with [`plan_dquant`] /
/// [`plan_dbquant`]).
///
/// Expected spend is the budget scaled by the cumulative activity
/// weight of the macroblocks already coded (`activity + 256` per
/// macroblock, so a flat macroblock still owns the bits of its
/// header), i.e. busy regions are *allotted* proportionally more bits
/// before any correction applies. The correction is the bits model
/// `bits ∝ 1 / qp`: the remaining budget is compared with the spend
/// the remaining macroblocks were expected to need, and the VOP
/// quantiser is scaled by that ratio (bounded to `[1/4, 4]` and then
/// to the configured band).
#[derive(Debug, Clone)]
pub struct MbRegulator {
    target: f64,
    vop_qp: u32,
    band: i64,
    /// `prefix[j]` = fraction of the total weight held by macroblocks
    /// `0..j` (so `prefix[0] == 0`, `prefix[n] == 1`).
    prefix: Vec<f64>,
    activity_classes: bool,
    qp_sum: u64,
    qp_count: u64,
}

impl MbRegulator {
    /// Build the regulator for a VOP of `activities.len()` macroblocks
    /// (each entry the macroblock's mean-removed luma activity). With
    /// `activity_classes` the [`activity_class`] offset is added on top
    /// of the budget-derived target (the `mb-aq` behaviour inside the
    /// budget mode).
    pub fn new(budget: MbBudget, vop_qp: u32, activities: &[u32], activity_classes: bool) -> Self {
        assert!((1..=MAX_QP).contains(&vop_qp));
        let mut prefix = Vec::with_capacity(activities.len() + 1);
        let mut acc = 0.0f64;
        prefix.push(0.0);
        for &a in activities {
            acc += f64::from(a) + 256.0;
            prefix.push(acc);
        }
        if acc > 0.0 {
            for p in prefix.iter_mut() {
                *p /= acc;
            }
        }
        Self {
            target: f64::from(budget.target_bits.max(1)),
            vop_qp,
            band: i64::from(budget.band),
            prefix,
            activity_classes,
            qp_sum: 0,
            qp_count: 0,
        }
    }

    /// The target quantiser for macroblock `mb_index` given the
    /// `bits_spent` so far in the VOP (header included). `activity` is
    /// the macroblock's own activity (for the optional class offset).
    pub fn target_qp(&self, mb_index: usize, bits_spent: usize, activity: u32) -> u32 {
        let n = self.prefix.len() - 1;
        let done = self.prefix[mb_index.min(n)];
        let expected_remaining = (self.target * (1.0 - done)).max(self.target * 0.02);
        let remaining = (self.target - bits_spent as f64).max(self.target * 0.02);
        let scale = (expected_remaining / remaining).clamp(0.25, 4.0);
        let scaled = (f64::from(self.vop_qp) * scale).round() as i64;
        let lo = (i64::from(self.vop_qp) - self.band).max(1);
        let hi = (i64::from(self.vop_qp) + self.band).min(i64::from(MAX_QP));
        let base = scaled.clamp(lo, hi) as u32;
        if self.activity_classes {
            target_qp(base, activity_class(activity))
        } else {
            base
        }
    }

    /// Record the quantiser a coded macroblock actually used (for
    /// [`MbRegulator::mean_qp`]).
    pub fn record(&mut self, qp: u32) {
        self.qp_sum += u64::from(qp);
        self.qp_count += 1;
    }

    /// Mean quantiser over the recorded macroblocks (the VOP quantiser
    /// when nothing was recorded) — the effective quantiser the rate
    /// model pairs with the VOP's bit count.
    pub fn mean_qp(&self) -> f64 {
        if self.qp_count == 0 {
            f64::from(self.vop_qp)
        } else {
            self.qp_sum as f64 / self.qp_count as f64
        }
    }
}

/// The encoder-side quantiser decision for one I-/P-/S-VOP
/// macroblock: the budget regulator's target when a budget is set,
/// else the activity class around the VOP quantiser when `adaptive`,
/// else the running quantiser untouched. Returns the quantiser to
/// code with and the Table 6-32 `dquant` to emit.
#[allow(clippy::too_many_arguments)]
pub fn plan_mb_dquant(
    regulator: Option<&MbRegulator>,
    adaptive: bool,
    running: u32,
    vop_qp: u32,
    mb_index: usize,
    bits_spent: usize,
    activity: u32,
) -> (u32, Option<i8>) {
    match regulator {
        Some(reg) => plan_dquant(running, reg.target_qp(mb_index, bits_spent, activity)),
        None if adaptive => plan_dquant(running, target_qp(vop_qp, activity_class(activity))),
        None => (running, None),
    }
}

/// The B-VOP counterpart of [`plan_mb_dquant`] (Table 6-33
/// `dbquant`).
#[allow(clippy::too_many_arguments)]
pub fn plan_mb_dbquant(
    regulator: Option<&MbRegulator>,
    adaptive: bool,
    running: u32,
    vop_qp: u32,
    mb_index: usize,
    bits_spent: usize,
    activity: u32,
) -> (u32, Option<i8>) {
    match regulator {
        Some(reg) => plan_dbquant(running, reg.target_qp(mb_index, bits_spent, activity)),
        None if adaptive => plan_dbquant(running, target_qp(vop_qp, activity_class(activity))),
        None => (running, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_span_the_band() {
        assert_eq!(activity_class(0), -2);
        assert_eq!(activity_class(600), -1);
        assert_eq!(activity_class(2000), 0);
        assert_eq!(activity_class(4000), 1);
        assert_eq!(activity_class(9000), 2);
    }

    #[test]
    fn dquant_steps_stay_in_table_6_32() {
        assert_eq!(plan_dquant(10, 10), (10, None));
        assert_eq!(plan_dquant(10, 11), (11, Some(1)));
        assert_eq!(plan_dquant(10, 14), (12, Some(2)));
        assert_eq!(plan_dquant(10, 9), (9, Some(-1)));
        assert_eq!(plan_dquant(10, 5), (8, Some(-2)));
        assert_eq!(target_qp(31, 2), 31);
        assert_eq!(target_qp(1, -2), 1);
    }

    #[test]
    fn dbquant_steps_are_even_and_clipped() {
        assert_eq!(plan_dbquant(10, 11), (10, None));
        assert_eq!(plan_dbquant(10, 12), (12, Some(2)));
        assert_eq!(plan_dbquant(10, 7), (8, Some(-2)));
        assert_eq!(plan_dbquant(30, 31), (30, None));
        assert_eq!(plan_dbquant(2, 1), (2, None));
    }

    #[test]
    fn regulator_scales_with_the_spend() {
        let acts = vec![1000u32; 16];
        let reg = MbRegulator::new(
            MbBudget {
                target_bits: 1600,
                band: 8,
            },
            10,
            &acts,
            false,
        );
        // On budget half-way through: the VOP quantiser.
        assert_eq!(reg.target_qp(8, 800, 1000), 10);
        // Twice the expected spend half-way: remaining 0 → coarse.
        assert!(reg.target_qp(8, 1500, 1000) > 14);
        // Under-spend: finer.
        assert!(reg.target_qp(8, 200, 1000) < 10);
        // Band clamp.
        let tight = MbRegulator::new(
            MbBudget {
                target_bits: 1600,
                band: 1,
            },
            10,
            &acts,
            false,
        );
        assert_eq!(tight.target_qp(8, 1590, 1000), 11);
        assert_eq!(tight.target_qp(8, 10, 1000), 9);
    }

    #[test]
    fn regulator_mean_qp_defaults_to_vop_qp() {
        let mut reg = MbRegulator::new(
            MbBudget {
                target_bits: 100,
                band: 2,
            },
            7,
            &[0, 0],
            true,
        );
        assert_eq!(reg.mean_qp(), 7.0);
        reg.record(6);
        reg.record(8);
        assert_eq!(reg.mean_qp(), 7.0);
        // Activity classes ride on top of the budget target.
        assert_eq!(reg.target_qp(0, 0, 0), 5);
    }
}

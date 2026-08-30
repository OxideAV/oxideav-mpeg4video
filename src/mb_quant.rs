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
//!   tells the caller when the step could not be carried.
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
}

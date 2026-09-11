//! The `rate_control::FirstPassStats` text parser (the `stats-file` a
//! `pass=2` encoder reads from disk): arbitrary text must either parse
//! or be rejected, a parsed set must round-trip through `to_text`, and
//! feeding it to a `BudgetPlanner` must plan every frame without
//! panicking or leaving the quantiser range.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_mpeg4video::rate_control::{BudgetPlanner, FirstPassStats, VopClass};

fuzz_target!(|data: &[u8]| {
    if data.len() > 16384 {
        return;
    }
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(stats) = FirstPassStats::parse(text) else {
        return;
    };
    let again = FirstPassStats::parse(&stats.to_text()).expect("to_text must re-parse");
    assert_eq!(again.frames.len(), stats.frames.len());
    let classes: Vec<VopClass> = stats.frames.iter().map(|f| f.class).collect();
    let mut planner = BudgetPlanner::new(250_000, 0.04, 12, 2, 8).with_first_pass(stats);
    let b = 16384.0 * 20.0;
    for (i, class) in classes.iter().take(256).enumerate() {
        let plan = planner.plan(*class, b * 0.6, b);
        assert!((1..=31).contains(&plan.qp), "frame {i}: qp {}", plan.qp);
        assert!(plan.target_bits >= 64);
        planner.commit(*class, u64::from(plan.target_bits), f64::from(plan.qp));
    }
});

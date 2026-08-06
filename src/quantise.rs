//! Forward quantisation for the encoder — the duals of the §7.4.4
//! inverse quantisers in [`crate::inverse_quant`].
//!
//! The forward quantiser is **not normative**: ISO/IEC 14496-2 only
//! prescribes the decoder-side reconstruction (§7.4.4.1 method 1,
//! §7.4.4.2 method 2, §7.4.4.1.1 intra DC). An encoder may pick any
//! `QF[v][u]` it likes; what matters is that the *decoder's* formulas
//! map the chosen levels back to acceptable coefficients. The
//! functions here therefore choose each level by centring the
//! decoder's reconstruction on the input coefficient:
//!
//! * **Intra DC** (§7.4.4.1.1: `F'' = dc_scaler * QF`) — nearest
//!   integer: `QF = F // dc_scaler` (§4.1 `//`, half away from zero).
//! * **Method 2 intra / non-DC** (§7.4.4.2.1:
//!   `|F''| = (2|QF| + 1)·qs`, minus one when `qs` even) — the
//!   classic dead-zone quantiser `|QF| = |F| / (2·qs)` (truncating):
//!   every `|F|` in `[2·qs·k, 2·qs·(k+1))` reconstructs to the
//!   interval's centre `(2k + 1)·qs`.
//! * **Method 2 inter** — the same reconstruction but with a wider
//!   dead zone, `|QF| = (|F| − qs/2) / (2·qs)`, the standard
//!   inter-residual choice (residual noise near zero is cheaper to
//!   drop than to code).
//! * **Method 1 intra** (§7.4.4.1.2: `F'' = (QF·W·qs·2)/16`) —
//!   nearest integer: `QF = (8·F) // (W·qs)`.
//! * **Method 1 inter** (`F'' = ((2·QF + Sign(QF))·W·qs)/16`) — dead
//!   zone: `|QF| = (16·|F| − W·qs) / (2·W·qs)` truncated at zero.
//!
//! Every AC level is clamped to `[-2047, 2047]` — inside the §7.4.3.4
//! saturation domain and avoiding the Table B.18 reserved escape
//! LEVEL `-2048`.
//!
//! Provenance: the reconstruction formulas being inverted are
//! §7.4.4.1 / §7.4.4.2 of ISO/IEC 14496-2:2004 (3rd edition), read
//! from `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`
//! (see [`crate::inverse_quant`] for the decoder-side transcription).
//! No third-party source was consulted.

use crate::predictor::dc_scaler;
use crate::texture::DcComponent;

/// The encoder-side AC level clamp: the §7.4.3.4 domain minus the
/// reserved Type-3 escape LEVEL `-2048`.
const LEVEL_MIN: i32 = -2047;
const LEVEL_MAX: i32 = 2047;

/// §4.1 `//` — division rounding to nearest, halves away from zero.
#[inline]
fn div_round_half_away(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// Quantise the intra DC coefficient `F[0][0]` for the given component
/// and quantiser scale: `QF = F // dc_scaler` (Table 7-1 scaler).
///
/// # Panics
///
/// Panics on `quantiser_scale == 0` (via [`dc_scaler`]).
pub fn quantise_intra_dc(f00: i32, component: DcComponent, quantiser_scale: u32) -> i32 {
    let scaler = dc_scaler(component, quantiser_scale) as i32;
    div_round_half_away(f00, scaler)
}

/// Method-2 forward quantiser for an intra AC coefficient (or any
/// coefficient of an intra block except the DC).
///
/// # Panics
///
/// Panics on `quantiser_scale == 0`.
pub fn quantise_method2_intra(f: i32, quantiser_scale: u32) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    let qs = quantiser_scale as i32;
    let level = f.abs() / (2 * qs);
    apply_sign_clamped(level, f)
}

/// Method-2 forward quantiser for a non-intra (inter residual)
/// coefficient, with the standard half-`qs` dead-zone widening.
///
/// # Panics
///
/// Panics on `quantiser_scale == 0`.
pub fn quantise_method2_inter(f: i32, quantiser_scale: u32) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    let qs = quantiser_scale as i32;
    let level = (f.abs() - qs / 2).max(0) / (2 * qs);
    apply_sign_clamped(level, f)
}

/// Method-1 forward quantiser for an intra coefficient (non-DC):
/// nearest-integer inverse of `F'' = (QF · W · qs · 2) / 16`.
///
/// # Panics
///
/// Panics on `quantiser_scale == 0` or `w == 0`.
pub fn quantise_method1_intra(f: i32, w: u8, quantiser_scale: u32) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    assert!(w > 0, "quant matrix entry must be > 0");
    let d = w as i32 * quantiser_scale as i32;
    let level = div_round_half_away(8 * f.abs(), d);
    apply_sign_clamped(level, f)
}

/// Method-1 forward quantiser for a non-intra coefficient: dead-zone
/// inverse of `F'' = ((2·QF + Sign(QF)) · W · qs) / 16`.
///
/// # Panics
///
/// Panics on `quantiser_scale == 0` or `w == 0`.
pub fn quantise_method1_inter(f: i32, w: u8, quantiser_scale: u32) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    assert!(w > 0, "quant matrix entry must be > 0");
    let d = w as i32 * quantiser_scale as i32;
    let level = (16 * f.abs() - d).max(0) / (2 * d);
    apply_sign_clamped(level, f)
}

/// Re-apply the sign of `f` to a non-negative `level` and clamp to the
/// encoder-side AC level domain.
#[inline]
fn apply_sign_clamped(level: i32, f: i32) -> i32 {
    let signed = if f < 0 { -level } else { level };
    signed.clamp(LEVEL_MIN, LEVEL_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_quant::{
        inverse_quant_intra_dc, inverse_quant_method1_coef, inverse_quant_method2_coef,
    };

    #[test]
    fn intra_dc_round_trips_within_half_scaler() {
        for qs in 1..=31u32 {
            for &component in &[DcComponent::Luminance, DcComponent::Chrominance] {
                let scaler = dc_scaler(component, qs) as i32;
                for f in (0..=2040).step_by(7) {
                    let qf = quantise_intra_dc(f, component, qs);
                    let back = inverse_quant_intra_dc(qf, component, qs, false);
                    assert!(
                        (back - f).abs() * 2 <= scaler,
                        "qs {qs} f {f}: recon {back} off by more than scaler/2 ({scaler})"
                    );
                }
            }
        }
    }

    #[test]
    fn method2_intra_reconstruction_error_is_bounded() {
        for qs in 1..=31u32 {
            for f in -2040..=2040 {
                let qf = quantise_method2_intra(f, qs);
                let back = inverse_quant_method2_coef(qf, qs);
                let bound = 2 * qs as i32;
                assert!(
                    (back - f).abs() <= bound,
                    "qs {qs} f {f}: level {qf} recon {back} exceeds ±{bound}"
                );
                // Sign preservation (a non-zero level never flips sign).
                if qf != 0 {
                    assert_eq!(qf.signum(), f.signum());
                }
            }
        }
    }

    #[test]
    fn method2_inter_dead_zone_and_bound() {
        for qs in 1..=31u32 {
            let qs_i = qs as i32;
            for f in -2040..=2040 {
                let qf = quantise_method2_inter(f, qs);
                let back = inverse_quant_method2_coef(qf, qs);
                // Inside the widened dead zone the level is zero.
                if f.abs() < qs_i / 2 + 2 * qs_i {
                    assert_eq!(qf, 0, "qs {qs} f {f} should fall in the dead zone");
                }
                let bound = 2 * qs_i + qs_i / 2;
                assert!(
                    (back - f).abs() <= bound,
                    "qs {qs} f {f}: level {qf} recon {back} exceeds ±{bound}"
                );
            }
        }
    }

    #[test]
    fn method1_intra_round_trips_within_step() {
        // Default intra matrix corner values + a mid value, across qs.
        for &w in &[8u8, 17, 33, 46] {
            for qs in 1..=31u32 {
                let step = (w as i32 * qs as i32 * 2) / 16 + 1;
                for f in (-2040..=2040).step_by(11) {
                    let qf = quantise_method1_intra(f, w, qs);
                    let back = inverse_quant_method1_coef(qf, w, qs, true);
                    assert!(
                        (back - f).abs() <= step,
                        "w {w} qs {qs} f {f}: level {qf} recon {back} exceeds ±{step}"
                    );
                }
            }
        }
    }

    #[test]
    fn method1_inter_round_trips_within_step() {
        for &w in &[16u8, 21, 27] {
            for qs in 1..=31u32 {
                // Dead-zone width dominates: levels are zero while
                // 16·|f| < 3·d, so the worst reconstruction error is
                // just under 3·d/16; above the zone the step error is
                // d/8. Bound by the larger plus rounding slack.
                let step = (3 * w as i32 * qs as i32) / 16 + 2;
                for f in (-2040..=2040).step_by(11) {
                    let qf = quantise_method1_inter(f, w, qs);
                    let back = inverse_quant_method1_coef(qf, w, qs, false);
                    assert!(
                        (back - f).abs() <= step,
                        "w {w} qs {qs} f {f}: level {qf} recon {back} exceeds ±{step}"
                    );
                }
            }
        }
    }

    #[test]
    fn levels_clamp_inside_escape_domain() {
        // qs = 1, huge coefficient: the level must clamp to ±2047, never
        // reach the reserved −2048.
        assert_eq!(quantise_method2_intra(2047 * 4, 1), 2047);
        assert_eq!(quantise_method1_intra(2047, 1, 1), 2047);
        assert_eq!(quantise_method1_intra(-2047, 1, 1), -2047);
    }
}

//! §7.4.4 inverse quantisation for 8×8 DCT coefficient blocks.
//!
//! This module implements every numerically-distinct sub-section of
//! ISO/IEC 14496-2:2004 §7.4.4 except for the §7.4.4.5 mismatch
//! control (which is fused into [`inverse_quant_method1`] because its
//! per-coefficient sum has to be computed in the same pass as the
//! per-coefficient arithmetic + saturation):
//!
//! * §7.4.4.1.1 — intra DC reconstruction `F''[0][0] = dc_scaler *
//!   QF[0][0]`.
//! * §7.4.4.1.2 — first inverse quantisation method ("method 1",
//!   `quant_type == 1`), the matrix-using path with separate intra /
//!   non-intra formulas. The matrix-free DC coefficient of an intra
//!   block is handled by §7.4.4.1.1.
//! * §7.4.4.2.1 — second inverse quantisation method ("method 2",
//!   `quant_type == 0`), the matrix-free path with separate
//!   odd / even `quantiser_scale` formulas. The DC of an intra block
//!   is *also* handled by §7.4.4.1.1 per §7.4.4.2's "is quantised
//!   using the same method as in the first inverse quantisation
//!   method" sentence.
//! * §7.4.4.4 — saturation of `F''[v][u]` to
//!   `[-2^(bits_per_pixel + 3), 2^(bits_per_pixel + 3) - 1]`.
//! * §7.4.4.5 — mismatch control. The block sum's parity gates a
//!   one-bit toggle on `F[7][7]`; only applicable to method 1.
//! * §7.4.4.6 — summary of method-1 quantiser process. The
//!   [`inverse_quant_method1`] entry point implements this pseudo-code
//!   verbatim (with the bracketing reshuffle that the spec text
//!   itself allows under "any process numerically equivalent to").
//!
//! Notes:
//! * The §7.4.4.3 "non-linear inverse DC quantisation" sub-section is
//!   the Table 7-1 `dc_scaler` that already lives in
//!   [`crate::predictor::dc_scaler`]; this module re-uses that
//!   function for the intra DC path.
//! * Both the §7.4.4.6 pseudo-code's `/` and the §7.4.4.1.2 NOTE
//!   confirm the `/` operator is the §4.1 "integer division with
//!   truncation of the result toward zero" — Rust's `/` on signed
//!   integers matches.
//! * `Sign(x)` per §4.1 is `+1` for `x >= 0` and `-1` for `x < 0`
//!   (note that `Sign(0) == 1`). It never enters the method-1
//!   formula for `QF[v][u] == 0` because that branch short-circuits
//!   to `F''[v][u] = 0`.
//! * `bits_per_pixel` comes from §6.3.3 — default 8 (the `not_8_bit
//!   == 0` path), valid range 4..=12 (`not_8_bit == 1`). This module
//!   accepts any value up to 28 to leave headroom for the `+3`
//!   saturation exponent without overflowing `i32`.

use crate::predictor::dc_scaler;
use crate::texture::DcComponent;

/// One §7.4.4 inverse-quantisation context for a single 8×8 block.
///
/// Bundles the per-block scalars the §7.4.4.6 summary pseudo-code
/// reads to keep call-sites readable. Per-coefficient inputs are the
/// quantisation matrix `w` (`W[0]` for intra, `W[1]` for non-intra
/// per §7.4.4.1.2) and the 8×8 `QF[v][u]` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseQuantContext {
    /// `true` if the macroblock owning this block is an intra
    /// macroblock. Selects:
    ///
    /// * the §7.4.4.1.1 DC formula (only when this block's `(v, u) =
    ///   (0, 0)` cell is being processed),
    /// * the `k = 0` branch in method 1 (per §7.4.4.1.2 / §7.4.4.6),
    /// * the `W[0]` matrix in method 1 (intra) vs `W[1]` (non-intra).
    pub macroblock_intra: bool,
    /// The current block's component, used by [`dc_scaler`] when the
    /// `(0, 0)` coefficient of an intra block is reconstructed.
    /// Ignored when `macroblock_intra == false`.
    pub component: DcComponent,
    /// The §6.2.5 `quantiser_scale` after `dquant` / `dbquant`
    /// adjustment, in the §7.4.4 spec's `quantiser_scale` slot. Valid
    /// range `1..=31` when `not_8_bit == 0` (§7.4.4.1.2), or
    /// `1..=2^quant_precision - 1` when `not_8_bit == 1` (§7.4.4.2).
    pub quantiser_scale: u32,
    /// `bits_per_pixel` from §6.3.3 — default 8, valid range 4..=12.
    /// Used by §7.4.4.4 saturation.
    pub bits_per_pixel: u32,
    /// `short_video_header == 1` selects the fixed `dc_scaler = 8`
    /// path of §7.4.4.3 for the DC coefficient of intra blocks; per
    /// §7.4.4.1.1 / §7.4.4.3, otherwise [`dc_scaler`] (Table 7-1) is
    /// used.
    pub short_video_header: bool,
}

/// §7.4.4.4 saturation bounds for `F'[v][u]` given `bits_per_pixel`.
///
/// Returns `(lo, hi)` where `lo = -2^(bits_per_pixel + 3)` and
/// `hi = 2^(bits_per_pixel + 3) - 1`. Caller-side clamps use these
/// directly via [`i32::clamp`].
///
/// # Panics
///
/// Panics if `bits_per_pixel` exceeds 28; `2^31` would overflow `i32`.
#[inline]
pub fn saturation_bounds(bits_per_pixel: u32) -> (i32, i32) {
    assert!(
        bits_per_pixel < 29,
        "bits_per_pixel = {bits_per_pixel} is out of range"
    );
    let mag = 1i32 << (bits_per_pixel + 3);
    (-mag, mag - 1)
}

/// §7.4.4.4 saturation applied to a single coefficient.
#[inline]
pub fn saturate_fprime(value: i32, bits_per_pixel: u32) -> i32 {
    let (lo, hi) = saturation_bounds(bits_per_pixel);
    value.clamp(lo, hi)
}

/// §7.4.4.1.1 intra DC reconstruction: `F''[0][0] = dc_scaler * QF[0][0]`.
///
/// When `short_video_header == 1`, `dc_scaler` is fixed at 8 per
/// §7.4.4.3 + §7.4.1.1; otherwise Table 7-1 is consulted via
/// [`dc_scaler`].
///
/// # Panics
///
/// Panics on `quantiser_scale == 0` when `short_video_header == 0`.
#[inline]
pub fn inverse_quant_intra_dc(
    qf00: i32,
    component: DcComponent,
    quantiser_scale: u32,
    short_video_header: bool,
) -> i32 {
    let scaler = if short_video_header {
        8
    } else {
        dc_scaler(component, quantiser_scale) as i32
    };
    scaler * qf00
}

/// §7.4.4.2.1 method-2 reconstruction for one *non-DC* coefficient
/// (or the DC of a non-intra block).
///
/// ```text
///   if QF[v][u] == 0                       -> F''[v][u] = 0
///   else if quantiser_scale is odd         -> F''[v][u] = (2*|QF| + 1) * quantiser_scale
///   else                                   -> F''[v][u] = (2*|QF| + 1) * quantiser_scale - 1
///   F''[v][u] *= Sign(QF[v][u])            // §7.4.4.2.1 trailing sentence
/// ```
///
/// The "Sign incorporation" sentence in §7.4.4.2.1 says "The sign of
/// QF[v][u] is then incorporated to obtain F''[v][u]: F''[v][u] =
/// Sign(QF[v][u]) * |F''[v][u]|." The intermediate magnitude is
/// computed on `|QF|`.
///
/// # Panics
///
/// Panics on `quantiser_scale == 0`.
#[inline]
pub fn inverse_quant_method2_coef(qf: i32, quantiser_scale: u32) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    if qf == 0 {
        return 0;
    }
    let abs_qf = qf.unsigned_abs() as i32;
    let qs = quantiser_scale as i32;
    let magnitude = if qs % 2 == 1 {
        (2 * abs_qf + 1) * qs
    } else {
        (2 * abs_qf + 1) * qs - 1
    };
    if qf < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Apply §7.4.4.2 (method 2) end-to-end to one 8×8 block.
///
/// Per §7.4.4.2 first paragraph, the DC coefficient of an intra block
/// uses §7.4.4.1.1 (the same method 1 DC formula); every other
/// coefficient uses §7.4.4.2.1. Each `F''[v][u]` is then saturated
/// per §7.4.4.4. Mismatch control (§7.4.4.5) is method 1-only.
///
/// # Panics
///
/// Panics on `ctx.quantiser_scale == 0`.
pub fn inverse_quant_method2(qf: &[[i32; 8]; 8], ctx: InverseQuantContext) -> [[i32; 8]; 8] {
    assert!(ctx.quantiser_scale > 0, "quantiser_scale must be > 0");
    let (lo, hi) = saturation_bounds(ctx.bits_per_pixel);
    let mut out = [[0i32; 8]; 8];
    for v in 0..8 {
        for u in 0..8 {
            let f_pp = if ctx.macroblock_intra && u == 0 && v == 0 {
                inverse_quant_intra_dc(
                    qf[0][0],
                    ctx.component,
                    ctx.quantiser_scale,
                    ctx.short_video_header,
                )
            } else {
                inverse_quant_method2_coef(qf[v][u], ctx.quantiser_scale)
            };
            out[v][u] = f_pp.clamp(lo, hi);
        }
    }
    out
}

/// §7.4.4.1.2 method-1 reconstruction for one non-DC coefficient.
///
/// Intra: `F''[v][u] = (QF[v][u] * W[0][v][u] * quantiser_scale * 2) / 16`.
/// Non-intra: `F''[v][u] = ((2*QF[v][u] + Sign(QF[v][u])) * W[1][v][u]
/// * quantiser_scale) / 16`.
///
/// Both branches short-circuit to 0 when `QF[v][u] == 0` (the
/// §7.4.4.6 pseudo-code's first `if`); the caller-driven short-circuit
/// avoids invoking the Sign-of-zero branch on the non-intra side.
///
/// The `/` is §4.1 truncation-toward-zero division; Rust's `/` on
/// signed integers matches.
///
/// # Panics
///
/// Panics on `quantiser_scale == 0`.
#[inline]
pub fn inverse_quant_method1_coef(
    qf: i32,
    w: u8,
    quantiser_scale: u32,
    macroblock_intra: bool,
) -> i32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    if qf == 0 {
        return 0;
    }
    let w = w as i32;
    let qs = quantiser_scale as i32;
    if macroblock_intra {
        (qf * w * qs * 2) / 16
    } else {
        // Sign(qf): +1 for qf >= 0, -1 for qf < 0. qf == 0 already
        // short-circuited above so the >= 0 branch always returns +1
        // (consistent with §4.1).
        let sign = if qf < 0 { -1 } else { 1 };
        ((2 * qf + sign) * w * qs) / 16
    }
}

/// Apply §7.4.4 (method 1) end-to-end to one 8×8 block, including
/// §7.4.4.4 saturation and §7.4.4.5 mismatch control.
///
/// This is a faithful Rust transcription of §7.4.4.6's pseudo-code:
///
/// 1. For every `(v, u)`, compute `F''[v][u]` via the §7.4.4.1.1 DC
///    formula (intra `(0,0)`) or [`inverse_quant_method1_coef`] (all
///    other cells).
/// 2. Saturate `F''[v][u]` to `[-2^(bpp+3), 2^(bpp+3) - 1]` to obtain
///    `F'[v][u]` (§7.4.4.4).
/// 3. Sum every `F'[v][u]`; if the sum is even, toggle the least
///    significant bit of `F'[7][7]` to obtain `F[7][7]` (§7.4.4.5).
///    Else `F = F'`.
///
/// `w` is `W[0]` (intra) or `W[1]` (non-intra) per §7.4.4.1.2; the
/// `(0, 0)` cell is ignored for intra blocks (DC path uses
/// [`dc_scaler`] instead).
///
/// # Panics
///
/// Panics on `ctx.quantiser_scale == 0`.
pub fn inverse_quant_method1(
    qf: &[[i32; 8]; 8],
    w: &[[u8; 8]; 8],
    ctx: InverseQuantContext,
) -> [[i32; 8]; 8] {
    inverse_quant_method1_impl(qf, w, ctx, true)
}

/// [`inverse_quant_method1`] **without** the §7.4.4.5 mismatch toggle
/// (steps 1–2 of the §7.4.4.6 pseudo-code only, so `F = F'`).
///
/// This is the ecosystem-compat intra path (see [`crate::compat`]):
/// black-box-observed reference decodes apply the method-1 mismatch
/// control to non-intra blocks only, so a compat-mode decode
/// reconstructs intra blocks through this entry while non-intra blocks
/// keep [`inverse_quant_method1`]. The spec text itself contains no
/// intra exemption — spec-mode decodes never call this function.
///
/// # Panics
///
/// Panics on `ctx.quantiser_scale == 0`.
pub fn inverse_quant_method1_no_mismatch(
    qf: &[[i32; 8]; 8],
    w: &[[u8; 8]; 8],
    ctx: InverseQuantContext,
) -> [[i32; 8]; 8] {
    inverse_quant_method1_impl(qf, w, ctx, false)
}

/// Shared §7.4.4.6 body: per-coefficient reconstruction + §7.4.4.4
/// saturation, with the §7.4.4.5 mismatch toggle gated by
/// `mismatch_control`.
fn inverse_quant_method1_impl(
    qf: &[[i32; 8]; 8],
    w: &[[u8; 8]; 8],
    ctx: InverseQuantContext,
    mismatch_control: bool,
) -> [[i32; 8]; 8] {
    assert!(ctx.quantiser_scale > 0, "quantiser_scale must be > 0");
    let (lo, hi) = saturation_bounds(ctx.bits_per_pixel);
    let mut f_prime = [[0i32; 8]; 8];
    let mut sum: i64 = 0;
    for v in 0..8 {
        for u in 0..8 {
            let f_pp = if ctx.macroblock_intra && u == 0 && v == 0 {
                inverse_quant_intra_dc(
                    qf[0][0],
                    ctx.component,
                    ctx.quantiser_scale,
                    ctx.short_video_header,
                )
            } else {
                inverse_quant_method1_coef(
                    qf[v][u],
                    w[v][u],
                    ctx.quantiser_scale,
                    ctx.macroblock_intra,
                )
            };
            let f_p = f_pp.clamp(lo, hi);
            f_prime[v][u] = f_p;
            sum += f_p as i64;
        }
    }
    // §7.4.4.5 mismatch control: if sum is even, toggle LSB of F[7][7].
    if mismatch_control && sum & 1 == 0 {
        // The spec's "if F'[7][7] is odd -> F[7][7] = F'[7][7] - 1;
        // else F[7][7] = F'[7][7] + 1" is equivalent to flipping the
        // LSB via XOR (NOTE 1 of §7.4.4.5).
        f_prime[7][7] ^= 1;
    }
    f_prime
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(intra: bool, qs: u32, bpp: u32) -> InverseQuantContext {
        InverseQuantContext {
            macroblock_intra: intra,
            component: DcComponent::Luminance,
            quantiser_scale: qs,
            bits_per_pixel: bpp,
            short_video_header: false,
        }
    }

    // ----- saturation_bounds / saturate_fprime -----

    #[test]
    fn saturation_bounds_at_8_bpp() {
        let (lo, hi) = saturation_bounds(8);
        // 2^(8+3) = 2048.
        assert_eq!(lo, -2048);
        assert_eq!(hi, 2047);
    }

    #[test]
    fn saturation_bounds_at_minimum_bpp() {
        // bits_per_pixel = 4 -> 2^7 = 128.
        let (lo, hi) = saturation_bounds(4);
        assert_eq!(lo, -128);
        assert_eq!(hi, 127);
    }

    #[test]
    fn saturation_bounds_at_maximum_bpp() {
        // bits_per_pixel = 12 -> 2^15 = 32768.
        let (lo, hi) = saturation_bounds(12);
        assert_eq!(lo, -32768);
        assert_eq!(hi, 32767);
    }

    #[test]
    fn saturate_fprime_clamps_at_8_bpp() {
        assert_eq!(saturate_fprime(0, 8), 0);
        assert_eq!(saturate_fprime(2047, 8), 2047);
        assert_eq!(saturate_fprime(2048, 8), 2047);
        assert_eq!(saturate_fprime(-2048, 8), -2048);
        assert_eq!(saturate_fprime(-2049, 8), -2048);
        assert_eq!(saturate_fprime(i32::MAX, 8), 2047);
        assert_eq!(saturate_fprime(i32::MIN, 8), -2048);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn saturation_bounds_reject_excessive_bpp() {
        let _ = saturation_bounds(30);
    }

    // ----- inverse_quant_intra_dc -----

    #[test]
    fn intra_dc_short_video_header_uses_fixed_eight() {
        // short_video_header == 1 -> dc_scaler = 8 regardless of qs.
        // QF = 5 -> F'' = 8 * 5 = 40.
        let f00 = inverse_quant_intra_dc(5, DcComponent::Luminance, 31, true);
        assert_eq!(f00, 40);
    }

    #[test]
    fn intra_dc_table_71_luminance_at_qs_5() {
        // Luminance qs=5 -> dc_scaler = 2*qs = 10. QF=12 -> F'' = 120.
        let f00 = inverse_quant_intra_dc(12, DcComponent::Luminance, 5, false);
        assert_eq!(f00, 120);
    }

    #[test]
    fn intra_dc_table_71_chrominance_at_qs_25() {
        // Chrominance qs=25 -> dc_scaler = qs - 6 = 19. QF=-3 -> F'' = -57.
        let f00 = inverse_quant_intra_dc(-3, DcComponent::Chrominance, 25, false);
        assert_eq!(f00, -57);
    }

    // ----- inverse_quant_method2_coef -----

    #[test]
    fn method2_zero_qf_yields_zero() {
        assert_eq!(inverse_quant_method2_coef(0, 1), 0);
        assert_eq!(inverse_quant_method2_coef(0, 31), 0);
    }

    #[test]
    fn method2_odd_quantiser_scale_matches_spec() {
        // qs=5 (odd): F'' = Sign(qf) * (2*|qf| + 1) * 5.
        // qf=+3 -> (2*3+1)*5 = 35.
        assert_eq!(inverse_quant_method2_coef(3, 5), 35);
        // qf=-3 -> -((2*3+1)*5) = -35.
        assert_eq!(inverse_quant_method2_coef(-3, 5), -35);
    }

    #[test]
    fn method2_even_quantiser_scale_subtracts_one() {
        // qs=8 (even): F'' = Sign(qf) * ((2*|qf| + 1) * 8 - 1).
        // qf=+1 -> (2*1+1)*8 - 1 = 23.
        assert_eq!(inverse_quant_method2_coef(1, 8), 23);
        // qf=-1 -> -23.
        assert_eq!(inverse_quant_method2_coef(-1, 8), -23);
    }

    #[test]
    fn method2_large_magnitude() {
        // qs=31 (odd), qf=100 -> (200+1)*31 = 6231.
        assert_eq!(inverse_quant_method2_coef(100, 31), 6231);
    }

    #[test]
    #[should_panic(expected = "quantiser_scale must be > 0")]
    fn method2_rejects_zero_quantiser_scale() {
        let _ = inverse_quant_method2_coef(1, 0);
    }

    // ----- inverse_quant_method2 (whole block) -----

    #[test]
    fn method2_block_intra_dc_path() {
        // Intra block; only (0,0) non-zero. qs=5 -> dc_scaler = 10.
        // QF[0][0] = 7 -> F'' = 70 -> well within bpp=8 [-2048,2047].
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 7;
        let f = inverse_quant_method2(&qf, ctx(true, 5, 8));
        assert_eq!(f[0][0], 70);
        for (v, row) in f.iter().enumerate() {
            for (u, cell) in row.iter().enumerate() {
                if (v, u) != (0, 0) {
                    assert_eq!(*cell, 0);
                }
            }
        }
    }

    #[test]
    fn method2_block_non_intra_dc_uses_method2() {
        // Non-intra block: (0,0) goes through the same method-2 formula
        // as every other coefficient — *not* §7.4.4.1.1.
        // qs=5 (odd), QF[0][0] = 3 -> (7)*5 = 35.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 3;
        let f = inverse_quant_method2(&qf, ctx(false, 5, 8));
        assert_eq!(f[0][0], 35);
    }

    #[test]
    fn method2_block_saturates_to_bpp_range() {
        // bpp=4 -> [-128, 127]. Force a large F'' by giving qf=10, qs=15.
        // (2*10+1)*15 = 315 -> saturates to 127.
        let mut qf = [[0i32; 8]; 8];
        qf[3][4] = 10;
        let f = inverse_quant_method2(&qf, ctx(false, 15, 4));
        assert_eq!(f[3][4], 127);
        // Negative side.
        let mut qf = [[0i32; 8]; 8];
        qf[2][2] = -10;
        let f = inverse_quant_method2(&qf, ctx(false, 15, 4));
        assert_eq!(f[2][2], -128);
    }

    // ----- inverse_quant_method1_coef -----

    #[test]
    fn method1_coef_zero_qf_yields_zero() {
        // No matter what the matrix or qs say.
        assert_eq!(inverse_quant_method1_coef(0, 16, 31, true), 0);
        assert_eq!(inverse_quant_method1_coef(0, 16, 31, false), 0);
    }

    #[test]
    fn method1_coef_intra_matches_spec_arithmetic() {
        // Intra: F'' = (qf * W * qs * 2) / 16.
        // qf=3, W=16, qs=5 -> 3 * 16 * 5 * 2 / 16 = 30.
        assert_eq!(inverse_quant_method1_coef(3, 16, 5, true), 30);
        // qf=-3 -> -30 (truncation toward zero — -3 * 16 * 5 * 2 / 16 = -30).
        assert_eq!(inverse_quant_method1_coef(-3, 16, 5, true), -30);
        // Truncation: qf=1, W=8, qs=1 -> 1*8*1*2/16 = 16/16 = 1.
        assert_eq!(inverse_quant_method1_coef(1, 8, 1, true), 1);
        // qf=-1 -> -1*8*1*2/16 = -16/16 = -1 (Rust `/` is truncation
        // toward zero).
        assert_eq!(inverse_quant_method1_coef(-1, 8, 1, true), -1);
        // Truncation example: qf=1, W=3, qs=1 -> 6/16 = 0.
        assert_eq!(inverse_quant_method1_coef(1, 3, 1, true), 0);
    }

    #[test]
    fn method1_coef_non_intra_adds_sign() {
        // Non-intra: F'' = ((2*qf + Sign(qf)) * W * qs) / 16.
        // qf=+3 -> (6+1)*W*qs/16. With W=16, qs=5 -> 7*16*5/16 = 35.
        assert_eq!(inverse_quant_method1_coef(3, 16, 5, false), 35);
        // qf=-3 -> (-6-1)*16*5/16 = -7*5 = -35.
        assert_eq!(inverse_quant_method1_coef(-3, 16, 5, false), -35);
        // Truncation: qf=+1 -> (2+1)*8*1/16 = 24/16 = 1.
        assert_eq!(inverse_quant_method1_coef(1, 8, 1, false), 1);
        // qf=-1 -> (-2-1)*8*1/16 = -24/16 = -1 (truncation toward
        // zero: -24/16 = -1 in Rust).
        assert_eq!(inverse_quant_method1_coef(-1, 8, 1, false), -1);
    }

    #[test]
    #[should_panic(expected = "quantiser_scale must be > 0")]
    fn method1_coef_rejects_zero_quantiser_scale() {
        let _ = inverse_quant_method1_coef(1, 16, 0, true);
    }

    // ----- inverse_quant_method1 (whole block, w/ mismatch) -----

    #[test]
    fn method1_block_intra_only_dc_then_mismatch() {
        // Intra block, (0,0) = 7, qs=5 (luminance -> dc_scaler = 10).
        // F''[0][0] = 70, every other F''[v][u] = 0 (since W·QF = 0).
        // F'[v][u] = F''[v][u] (within bpp=8 range).
        // sum = 70 -> even -> toggle F'[7][7] LSB. F'[7][7] was 0 ->
        // becomes 1.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 7;
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(&qf, &w, ctx(true, 5, 8));
        assert_eq!(f[0][0], 70);
        for (v, row) in f.iter().enumerate() {
            for (u, cell) in row.iter().enumerate() {
                if (v, u) == (0, 0) {
                    continue;
                }
                if (v, u) == (7, 7) {
                    assert_eq!(*cell, 1, "mismatch toggle on F[7][7]");
                } else {
                    assert_eq!(*cell, 0, "other cells stay zero");
                }
            }
        }
    }

    #[test]
    fn method1_no_mismatch_variant_skips_the_even_sum_toggle() {
        // Same input as `method1_block_intra_only_dc_then_mismatch`
        // (sum = 70, even → the spec entry toggles F[7][7]), but via
        // the ecosystem-compat entry: F = F', so F[7][7] stays 0 and
        // every other cell matches the spec entry exactly.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 7;
        let w = [[16u8; 8]; 8];
        let spec = inverse_quant_method1(&qf, &w, ctx(true, 5, 8));
        let compat = inverse_quant_method1_no_mismatch(&qf, &w, ctx(true, 5, 8));
        assert_eq!(compat[0][0], 70);
        assert_eq!(compat[7][7], 0, "no toggle in the no-mismatch entry");
        assert_eq!(spec[7][7], 1);
        for v in 0..8 {
            for u in 0..8 {
                if (v, u) != (7, 7) {
                    assert_eq!(compat[v][u], spec[v][u], "({v},{u})");
                }
            }
        }
    }

    #[test]
    fn method1_no_mismatch_variant_matches_on_odd_sums() {
        // Odd post-saturation sum → the spec entry applies no toggle
        // either, so the two entries agree on every cell.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 3; // chroma qs=9 → dc_scaler 11 → F[0][0] = 33 (odd)
        let w = [[16u8; 8]; 8];
        let c = InverseQuantContext {
            macroblock_intra: true,
            component: DcComponent::Chrominance,
            quantiser_scale: 9,
            bits_per_pixel: 8,
            short_video_header: false,
        };
        assert_eq!(
            inverse_quant_method1(&qf, &w, c),
            inverse_quant_method1_no_mismatch(&qf, &w, c)
        );
    }

    #[test]
    fn method1_block_intra_odd_sum_skips_mismatch() {
        // Construct an intra block whose post-saturation sum is ODD so
        // mismatch control is a no-op. (0,0)=7 (->70) plus (0,1) with
        // qf=1, W=16, qs=5 -> 1*16*5*2/16 = 10. Sum = 70+10 = 80 (still
        // even). Bump (0,1) qf=2 -> 2*16*5*2/16 = 20. Sum = 70+20 = 90
        // (even). Try a small W odd contribution: (0,2) with qf=1, W=8,
        // qs=1 -> 1 (intra). With (0,0)=7 (dc) qs=5 will give 70; sum
        // 70+0+1+... but qs changes globally so we can't mix qs.
        //
        // Easier: choose so sum is odd by selecting an odd dc_scaler.
        // Chrominance, qs=7 -> dc_scaler = (7+13)/2 = 10.  Still even.
        // qs=9 -> chroma: (9+13)/2 = 11. dc_scaler=11. QF[0][0]=3 ->
        // F'[0][0] = 33 (odd). Other cells zero. Sum=33 (odd) -> NO
        // mismatch toggle. F[7][7] stays 0.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 3;
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(
            &qf,
            &w,
            InverseQuantContext {
                macroblock_intra: true,
                component: DcComponent::Chrominance,
                quantiser_scale: 9,
                bits_per_pixel: 8,
                short_video_header: false,
            },
        );
        assert_eq!(f[0][0], 33);
        assert_eq!(f[7][7], 0, "odd sum -> no mismatch toggle");
    }

    #[test]
    fn method1_block_mismatch_toggles_lsb_of_existing_odd_f77() {
        // Force a non-zero F'[7][7] and ensure mismatch toggle flips
        // its LSB. Use intra, qs=5, dc_scaler=10. QF[7][7]=3, W[7][7]=16
        // -> F'[7][7] = 3*16*5*2/16 = 30 (even). Sum = 30 (even) ->
        // toggle LSB -> F[7][7] = 31.
        let mut qf = [[0i32; 8]; 8];
        qf[7][7] = 3;
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(&qf, &w, ctx(true, 5, 8));
        assert_eq!(f[7][7], 31, "30 was even, toggle to 31");
    }

    #[test]
    fn method1_block_non_intra_uses_w_at_zero_zero() {
        // Non-intra: (0,0) goes through method1_coef like any other
        // cell. qf=3, W=16, qs=5 -> (6+1)*16*5/16 = 35. F[7][7] toggles
        // because sum = 35 odd? -> no, 35 is odd -> no toggle.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 3;
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(&qf, &w, ctx(false, 5, 8));
        assert_eq!(f[0][0], 35);
        assert_eq!(f[7][7], 0, "sum=35 odd -> no toggle");
    }

    #[test]
    fn method1_block_saturation_clamps_then_sum() {
        // bpp=4 (-> [-128,127]). Force F''[0][0] way out of range.
        // intra, qs=15 (luma dc_scaler = qs+8 = 23), QF[0][0]=20 ->
        // F''[0][0] = 460 -> saturates to 127. Sum = 127 (odd) -> no
        // toggle, F[7][7] stays 0.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 20;
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(&qf, &w, ctx(true, 15, 4));
        assert_eq!(f[0][0], 127);
        assert_eq!(f[7][7], 0);
    }

    #[test]
    fn method1_block_zero_block_triggers_mismatch_on_zero_f77() {
        // Every QF is zero -> every F' is zero -> sum = 0 (even) ->
        // toggle F'[7][7] LSB: 0 ^ 1 = 1.
        let qf = [[0i32; 8]; 8];
        let w = [[16u8; 8]; 8];
        let f = inverse_quant_method1(&qf, &w, ctx(false, 5, 8));
        assert_eq!(f[7][7], 1, "all-zero sum is even -> toggle to 1");
        for (v, row) in f.iter().enumerate() {
            for (u, cell) in row.iter().enumerate() {
                if (v, u) != (7, 7) {
                    assert_eq!(*cell, 0);
                }
            }
        }
    }

    #[test]
    fn method1_block_short_video_header_intra_dc_uses_eight() {
        // short_video_header == 1 -> dc_scaler=8 regardless of Table 7-1.
        // QF[0][0]=4, qs=31 (would normally give dc_scaler = 2*31-16 =
        // 46). F'[0][0] = 8*4 = 32. Sum=32 even -> toggle F[7][7] to 1.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 4;
        let w = [[16u8; 8]; 8];
        let ctx = InverseQuantContext {
            macroblock_intra: true,
            component: DcComponent::Luminance,
            quantiser_scale: 31,
            bits_per_pixel: 8,
            short_video_header: true,
        };
        let f = inverse_quant_method1(&qf, &w, ctx);
        assert_eq!(f[0][0], 32);
        assert_eq!(f[7][7], 1);
    }

    #[test]
    #[should_panic(expected = "quantiser_scale must be > 0")]
    fn method1_block_rejects_zero_quantiser_scale() {
        let qf = [[0i32; 8]; 8];
        let w = [[16u8; 8]; 8];
        let _ = inverse_quant_method1(&qf, &w, ctx(false, 0, 8));
    }

    // ----- integration: a full intra block round trip -----

    #[test]
    fn method1_integration_diagonal_intra_block_matches_handwork() {
        // Construct a worked example end-to-end:
        //   intra block, qs=5 (luminance), bpp=8, dc_scaler = 10.
        //   QF[0][0] = 6 -> F'[0][0] = 60.
        //   QF[1][1] = 2, W[1][1] = 16 -> intra: 2*16*5*2/16 = 20.
        //   QF[2][2] = 1, W[2][2] = 8  -> 1*8*5*2/16 = 80/16 = 5.
        //   All other QFs zero.
        //   Sum F' = 60 + 20 + 5 = 85 (odd) -> NO toggle. F[7][7] = 0.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 6;
        qf[1][1] = 2;
        qf[2][2] = 1;
        let mut w = [[16u8; 8]; 8];
        w[2][2] = 8;
        let f = inverse_quant_method1(&qf, &w, ctx(true, 5, 8));
        assert_eq!(f[0][0], 60);
        assert_eq!(f[1][1], 20);
        assert_eq!(f[2][2], 5);
        assert_eq!(f[7][7], 0, "sum=85 odd -> no toggle");
    }

    #[test]
    fn method2_integration_non_intra_block_matches_handwork() {
        // Non-intra, qs=6 (even), bpp=8.
        //   QF[0][0] = 2  -> (2*2+1)*6 - 1 = 5*6 - 1 = 29 (positive).
        //   QF[3][5] = -4 -> -((2*4+1)*6 - 1) = -(9*6 - 1) = -53.
        let mut qf = [[0i32; 8]; 8];
        qf[0][0] = 2;
        qf[3][5] = -4;
        let f = inverse_quant_method2(&qf, ctx(false, 6, 8));
        assert_eq!(f[0][0], 29);
        assert_eq!(f[3][5], -53);
    }
}

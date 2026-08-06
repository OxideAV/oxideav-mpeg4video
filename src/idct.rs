//! §7.4.5 + Annex A inverse discrete cosine transform (IDCT) for one
//! 8×8 DCT coefficient block.
//!
//! The spec text in §7.4.5 simply says: "Once the DCT coefficients,
//! `F[u][v]` are reconstructed, the inverse DCT transform defined in
//! Annex A shall be applied to obtain the inverse transformed values,
//! `f[y][x]`. These values shall be saturated so that:
//! `-2^bits_per_pixel ≤ f[y][x] ≤ 2^bits_per_pixel - 1`, for all
//! `x, y`."
//!
//! Annex A.1 gives the orthonormal 8×8 IDCT directly:
//!
//! ```text
//!                 2  N-1 N-1                       (2x + 1)uπ     (2y + 1)vπ
//!     f(x, y) =  --- ∑    ∑   C(u) C(v) F(u, v) cos ----------- cos -----------
//!                 N  u=0  v=0                          2N             2N
//!
//!     with  N = 8,  C(0) = 1/√2,  C(k) = 1 for k ≠ 0.
//! ```
//!
//! Annex A.1 then defers the accuracy requirements to IEEE Std
//! 1180-1990 with two normative deviations (the test-set parameters in
//! §3.2 and the per-pixel "peak error ≤ 1" tolerance in §3.3). It does
//! NOT prescribe a specific factorisation; any process whose output
//! satisfies the §3.3 tolerance against the saturated mathematical
//! integer-number IDCT is conformant. The §7.4.4.5 mismatch-control
//! step on `F[7][7]` already gives the bitstream side enough latitude
//! that the IDCT itself can be evaluated in a wider intermediate type
//! (the standard practice that 1180-1990 was written around).
//!
//! Implementation: we evaluate the 2-D IDCT as two passes of the 1-D
//! 8-point IDCT (row pass, then column pass) using `f64` intermediates
//! against a pre-computed cosine-table `COS[u][x] = cos((2x+1)uπ/16)`,
//! then round the final value to the nearest integer and saturate to
//! `[-2^bpp, 2^bpp - 1]` per §7.4.5. `f64` mantissa precision (53 bits)
//! is more than enough headroom against the §7.4.4.4 coefficient range
//! (`[-2^(bpp+3), 2^(bpp+3) - 1]`, so ≤ 20 bits at `bpp = 12`).
//!
//! The math is plain Annex A — no library or third-party
//! transcription. The cosine table is generated at runtime by
//! `f64::cos` on multiples of π/16. The output type sits in `i32` to
//! match the rest of the §7.4.x pipeline (`InverseQuantContext` outputs
//! `[[i32; 8]; 8]`).
//!
//! ## Lint allowance
//!
//! `#![allow(clippy::needless_range_loop)]` applies to the whole
//! module. Every loop in this file is the Annex A.1 sum `Σ_{u, v}`
//! whose index `(v, u)` (or `(x, y)`) tracks the spec notation
//! directly. Rewriting as `.iter().enumerate()` over the outer
//! dimension would obscure the spec correspondence with no measurable
//! benefit.

#![allow(clippy::needless_range_loop)]

const N: usize = 8;

/// Pre-computed 1-D cosine table: `COS[u][x] = cos((2x + 1) * u * π / (2 * N))`
/// for `N = 8`. Lazy-initialised on first use.
///
/// Used by both passes of the separable 2-D IDCT.
/// `cos(kπ/16)` for `k = 0..=8`, each the **correctly-rounded**
/// nearest-`f64` to the mathematical value (`k == 4` is exactly
/// `1/√2`, `k == 8` exactly `0`). Compile-time literals instead of
/// runtime `f64::cos`: libm implementations differ by an ulp across
/// platforms, and both the conformance pins and the encoder's closed
/// decode loop need the transform byte-deterministic everywhere.
pub(crate) const COS_K_PI_16: [f64; 9] = [
    1.0,
    0.980_785_280_403_230_4,
    0.923_879_532_511_286_7,
    0.831_469_612_302_545_2,
    core::f64::consts::FRAC_1_SQRT_2,
    0.555_570_233_019_602_2,
    0.382_683_432_365_089_8,
    0.195_090_322_016_128_28,
    0.0,
];

/// Build `COS[u][x] = cos((2x + 1)uπ / 16)` from [`COS_K_PI_16`] via
/// the quarter-wave symmetry of the cosine (`m = (2x+1)·u mod 32`;
/// the four quadrants map onto `±COS_K_PI_16[k]`). Sign flips and
/// copies only — every value stays correctly rounded.
pub(crate) fn cos_table() -> &'static [[f64; N]; N] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[f64; N]; N]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f64; N]; N];
        for (u, row) in table.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let m = ((2 * x + 1) * u) % 32;
                *cell = match m {
                    0..=8 => COS_K_PI_16[m],
                    9..=16 => -COS_K_PI_16[16 - m],
                    17..=24 => -COS_K_PI_16[m - 16],
                    _ => COS_K_PI_16[32 - m],
                };
            }
        }
        table
    })
}

/// `C(k)` from Annex A.1: `1/√2` when `k == 0`, `1` otherwise.
#[inline]
fn c(k: usize) -> f64 {
    if k == 0 {
        core::f64::consts::FRAC_1_SQRT_2
    } else {
        1.0
    }
}

/// §7.4.5 saturation bounds for `f[y][x]` given `bits_per_pixel`.
///
/// Returns `(lo, hi)` where `lo = -2^bits_per_pixel` and
/// `hi = 2^bits_per_pixel - 1`. With the default `bits_per_pixel = 8`
/// this gives `(-256, 255)` — note that this is the §7.4.5 *output*
/// range, which is one bit wider on the negative side than the
/// `[0, 255]` pixel-space clip eventually applied at the end of §6.3.2
/// after the prediction add.
///
/// # Panics
///
/// Panics if `bits_per_pixel` exceeds 30; `2^31` would overflow `i32`.
#[inline]
pub fn idct_saturation_bounds(bits_per_pixel: u32) -> (i32, i32) {
    assert!(
        bits_per_pixel < 31,
        "bits_per_pixel = {bits_per_pixel} is out of range"
    );
    let mag = 1i32 << bits_per_pixel;
    (-mag, mag - 1)
}

/// §7.4.5 saturation applied to a single transformed sample.
#[inline]
pub fn saturate_idct_sample(value: i32, bits_per_pixel: u32) -> i32 {
    let (lo, hi) = idct_saturation_bounds(bits_per_pixel);
    value.clamp(lo, hi)
}

/// Annex A.1 1-D 8-point orthonormal IDCT.
///
/// `input[u]` is `F(u)` (the 1-D coefficient at index `u`); `output[x]`
/// is the spatial value at index `x`. The transform is:
///
/// ```text
///                √(2/N) Σ_u C(u) F(u) cos((2x + 1)uπ / (2N))
/// ```
///
/// (`√(2/N)` is the orthonormal scaling that makes the transform its
/// own inverse; combined with `C(0) = 1/√2` the DC contribution
/// reduces to `F(0) / √N`.)
#[inline]
fn idct_1d(input: &[f64; N]) -> [f64; N] {
    let cos = cos_table();
    let scale = (2.0_f64 / N as f64).sqrt();
    let mut output = [0.0f64; N];
    for x in 0..N {
        let mut acc = 0.0f64;
        for u in 0..N {
            acc += c(u) * input[u] * cos[u][x];
        }
        output[x] = scale * acc;
    }
    output
}

/// Annex A.1 8×8 orthonormal inverse DCT followed by §7.4.5 saturation.
///
/// `coefficients[v][u]` is `F[v][u]` (the §7.4.4 reconstructed DCT
/// coefficient block). The function returns `f[y][x]` rounded to the
/// nearest integer and clamped to `[-2^bpp, 2^bpp - 1]` per the §7.4.5
/// closing sentence.
///
/// The computation is a straightforward two-pass separable IDCT (row
/// pass over `u`, column pass over `v`) in `f64`. The §4.1 rounding
/// note ("round to nearest") is followed verbatim — half-up rounding
/// via `(value + 0.5).floor()` on the non-negative branch and
/// `(value - 0.5).ceil()` on the negative branch (Rust's `f64::round`
/// rounds half-away-from-zero, which matches §4.1).
pub fn idct_8x8(coefficients: &[[i32; 8]; 8], bits_per_pixel: u32) -> [[i32; 8]; 8] {
    // Row pass: 1-D IDCT on each row of F (varying u for each fixed v).
    let mut row_out = [[0.0f64; N]; N];
    for v in 0..N {
        let mut row_in = [0.0f64; N];
        for u in 0..N {
            row_in[u] = coefficients[v][u] as f64;
        }
        row_out[v] = idct_1d(&row_in);
    }

    // Column pass: 1-D IDCT on each column of the row-pass output
    // (varying v for each fixed x).
    let mut full = [[0.0f64; N]; N];
    for x in 0..N {
        let mut col_in = [0.0f64; N];
        for v in 0..N {
            col_in[v] = row_out[v][x];
        }
        let col_out = idct_1d(&col_in);
        for y in 0..N {
            full[y][x] = col_out[y];
        }
    }

    // §7.4.5 rounding (§4.1 round-to-nearest) + saturation.
    let (lo, hi) = idct_saturation_bounds(bits_per_pixel);
    let mut out = [[0i32; 8]; 8];
    for y in 0..N {
        for x in 0..N {
            let rounded = full[y][x].round() as i32;
            out[y][x] = rounded.clamp(lo, hi);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip helper: forward 8×8 orthonormal DCT for tests.
    ///
    /// Identical structure to [`idct_8x8`] but with the cosine kernel
    /// transposed (forward DCT: sum over `x, y`; output indexed by
    /// `u, v`). Used by tests only — the decoder doesn't need a
    /// forward DCT.
    fn forward_dct_8x8(samples: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
        let cos = cos_table();
        let scale = (2.0_f64 / N as f64).sqrt();
        // Row pass: 1-D forward DCT on each row.
        let mut row_out = [[0.0f64; N]; N];
        for y in 0..N {
            for u in 0..N {
                let mut acc = 0.0f64;
                for x in 0..N {
                    acc += samples[y][x] * cos[u][x];
                }
                row_out[y][u] = scale * c(u) * acc;
            }
        }
        // Column pass.
        let mut out = [[0.0f64; N]; N];
        for u in 0..N {
            for v in 0..N {
                let mut acc = 0.0f64;
                for y in 0..N {
                    acc += row_out[y][u] * cos[v][y];
                }
                out[v][u] = scale * c(v) * acc;
            }
        }
        out
    }

    #[test]
    fn saturation_bounds_default_bpp() {
        assert_eq!(idct_saturation_bounds(8), (-256, 255));
    }

    #[test]
    fn saturation_bounds_4bpp() {
        assert_eq!(idct_saturation_bounds(4), (-16, 15));
    }

    #[test]
    fn saturation_bounds_12bpp() {
        assert_eq!(idct_saturation_bounds(12), (-4096, 4095));
    }

    #[test]
    fn saturate_sample_clamps_high() {
        assert_eq!(saturate_idct_sample(1000, 8), 255);
    }

    #[test]
    fn saturate_sample_clamps_low() {
        assert_eq!(saturate_idct_sample(-1000, 8), -256);
    }

    #[test]
    fn saturate_sample_passthrough() {
        assert_eq!(saturate_idct_sample(42, 8), 42);
    }

    /// A DC-only block (`F[0][0] = k`, all else zero) inverse-transforms
    /// to a uniform spatial block `f[y][x] = k / N` (where `N = 8`) per
    /// Annex A.1 — the `C(0) = 1/√2` factor on both passes combines
    /// with `√(2/N)` on both passes to give `k * (2/N) * (1/√2) * (1/√2)
    /// = k / N`.
    #[test]
    fn idct_dc_only_block() {
        let mut f = [[0i32; 8]; 8];
        f[0][0] = 256;
        let spatial = idct_8x8(&f, 12);
        // 256 / 8 == 32, exactly representable.
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(spatial[y][x], 32, "y={y} x={x}");
            }
        }
    }

    /// Round-trip: forward-DCT a flat sample block, then IDCT it; the
    /// reconstructed block must match within ±1 LSB (the §7.4.5 +
    /// Annex A IEEE 1180-1990 §3.3 tolerance after our two
    /// normative deviations).
    #[test]
    fn roundtrip_flat_block() {
        let mut samples = [[0.0f64; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                samples[y][x] = 128.0;
            }
        }
        let coef = forward_dct_8x8(&samples);
        let mut coef_i = [[0i32; 8]; 8];
        for v in 0..8 {
            for u in 0..8 {
                coef_i[v][u] = coef[v][u].round() as i32;
            }
        }
        let recon = idct_8x8(&coef_i, 12);
        for y in 0..8 {
            for x in 0..8 {
                let diff = (recon[y][x] - 128).abs();
                assert!(diff <= 1, "y={y} x={x} recon={} diff={diff}", recon[y][x]);
            }
        }
    }

    /// Round-trip: forward-DCT a deterministic pseudo-random integer
    /// sample block in [0, 255], then IDCT; reconstruction must match
    /// within ±1 LSB per IEEE 1180-1990 §3.3.
    #[test]
    fn roundtrip_random_block_within_1_lsb() {
        let mut samples = [[0.0f64; 8]; 8];
        // Deterministic LCG (Numerical Recipes constants) so the test
        // is reproducible without an `rand` dep.
        let mut state: u32 = 0x1234_5678;
        for y in 0..8 {
            for x in 0..8 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let pix = (state >> 24) as i32; // 0..=255
                samples[y][x] = pix as f64;
            }
        }
        let coef = forward_dct_8x8(&samples);
        let mut coef_i = [[0i32; 8]; 8];
        for v in 0..8 {
            for u in 0..8 {
                coef_i[v][u] = coef[v][u].round() as i32;
            }
        }
        let recon = idct_8x8(&coef_i, 12);
        let mut max_diff = 0i32;
        for y in 0..8 {
            for x in 0..8 {
                let original = samples[y][x] as i32;
                let diff = (recon[y][x] - original).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff <= 1,
                    "y={y} x={x} recon={} original={original} diff={diff}",
                    recon[y][x]
                );
            }
        }
        assert!(max_diff <= 1, "max_diff = {max_diff}");
    }

    /// Cross-validation against the existing §7.4.4 inverse-quant
    /// pipeline (intra DC path): a non-zero `QF[0][0]` reconstructs via
    /// §7.4.4.1.1 to `F''[0][0] = dc_scaler * QF[0][0]`, then the IDCT
    /// yields a uniform spatial block at `F[0][0] / 8`. Adding the
    /// §6.3.2 inverse prediction (mid-grey `128` for an isolated intra
    /// block) and clipping to `[0, 255]` gives a pixel in the legal
    /// range.
    #[test]
    fn idct_after_intra_dc_quant_is_in_pixel_range() {
        use crate::inverse_quant::inverse_quant_intra_dc;
        use crate::texture::DcComponent;

        // QF[0][0] = 4 at quantiser_scale = 5, luminance component.
        // Table 7-1 luminance band 1..=4: dc_scaler = 8.
        // Band 5..=8: dc_scaler = 2*qs = 10. So F''[0][0] = 10 * 4 = 40.
        let f00 = inverse_quant_intra_dc(4, DcComponent::Luminance, 5, false);
        assert_eq!(f00, 40);

        let mut block = [[0i32; 8]; 8];
        block[0][0] = f00;
        let spatial = idct_8x8(&block, 8);
        // 40 / 8 = 5: every pixel should be the same value, ≤ ±1 LSB.
        for y in 0..8 {
            for x in 0..8 {
                assert!(
                    (spatial[y][x] - 5).abs() <= 1,
                    "y={y} x={x} spatial={}",
                    spatial[y][x]
                );
            }
        }

        // Adding the §6.3.2 inverse prediction (mid-grey for an
        // isolated intra block, no neighbour available) and clipping to
        // [0, 255] yields a legal pixel.
        for y in 0..8 {
            for x in 0..8 {
                let pixel = (spatial[y][x] + 128).clamp(0, 255);
                assert!((0..=255).contains(&pixel));
            }
        }
    }

    /// An all-zero input transforms to an all-zero output.
    #[test]
    fn idct_zero_block() {
        let f = [[0i32; 8]; 8];
        let spatial = idct_8x8(&f, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(spatial[y][x], 0);
            }
        }
    }

    /// Output of an all-zero `F` block saturates to 0 cleanly at every
    /// supported `bits_per_pixel`.
    #[test]
    fn idct_zero_block_all_bpp() {
        for bpp in [4u32, 8, 10, 12] {
            let f = [[0i32; 8]; 8];
            let spatial = idct_8x8(&f, bpp);
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(spatial[y][x], 0, "bpp={bpp} y={y} x={x}");
                }
            }
        }
    }

    /// Saturation kicks in when reconstruction would exceed `2^bpp - 1`.
    /// A maxed-out DC coefficient `F[0][0] = 2^(bpp+3) - 1` reconstructs
    /// to `(2^(bpp+3) - 1) / 8` ≈ `2^bpp` in spatial domain — slightly
    /// over the §7.4.5 high bound `2^bpp - 1`, so the clamp engages.
    #[test]
    fn idct_saturation_high() {
        // bpp = 8: F''[0][0] saturation bound is 2^11 - 1 = 2047.
        // 2047 / 8 = 255.875 -> rounds to 256 -> clamps to 255.
        let mut f = [[0i32; 8]; 8];
        f[0][0] = 2047;
        let spatial = idct_8x8(&f, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(spatial[y][x], 255, "y={y} x={x} value={}", spatial[y][x]);
            }
        }
    }

    /// Saturation kicks in on the negative side too. `F[0][0] = -2^(bpp+3)`
    /// reconstructs to `-2^(bpp+3) / 8 = -2^bpp` which is exactly the
    /// §7.4.5 low bound `-2^bpp`.
    #[test]
    fn idct_saturation_low() {
        let mut f = [[0i32; 8]; 8];
        f[0][0] = -2048;
        let spatial = idct_8x8(&f, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(spatial[y][x], -256, "y={y} x={x} value={}", spatial[y][x]);
            }
        }
    }

    /// Cosine-table sanity: `cos((2*0 + 1) * 0 * π / 16) = 1` at `u = 0`,
    /// and `cos((2*0 + 1) * 4 * π / 16) = cos(π/4) = √2/2`.
    #[test]
    fn cosine_table_known_values() {
        let cos = cos_table();
        assert!((cos[0][0] - 1.0).abs() < 1e-12);
        assert!((cos[4][0] - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    /// The two-pass IDCT is separable: setting only `F[0][u]` (a single
    /// row of the input) should give an output whose rows are all
    /// identical to that row's 1-D IDCT (because the `v = 0` column
    /// pass is the only non-zero contribution and `C(0) = 1/√2` spreads
    /// it uniformly along `y`).
    #[test]
    fn idct_single_row_input_is_y_uniform() {
        let mut f = [[0i32; 8]; 8];
        for u in 0..8 {
            f[0][u] = (u as i32 + 1) * 16;
        }
        let spatial = idct_8x8(&f, 12);
        // Row 0 and row 7 should be the same scaled by 1/√8 column-pass
        // scaling — i.e. every row should be identical to row 0.
        for y in 1..8 {
            for x in 0..8 {
                assert_eq!(
                    spatial[y][x], spatial[0][x],
                    "y={y} x={x} row0={} rowy={}",
                    spatial[0][x], spatial[y][x]
                );
            }
        }
    }

    /// Highest-frequency input (`F[7][7] = k`) exercises every entry of
    /// the cosine table. The 2-D spatial output is the outer product of
    /// the two 1-D kernels `cos((2x+1)*7π/16)` × `cos((2y+1)*7π/16)`,
    /// which produces a sign-checkerboard `(-1)^(x+y)` pattern (the two
    /// 1-D cosines change sign whenever `(2k+1)*7` crosses an odd
    /// multiple of π/2, which happens on every adjacent `k`).
    #[test]
    fn idct_high_frequency_checkerboard() {
        let mut f = [[0i32; 8]; 8];
        f[7][7] = 2040; // close to the §7.4.4.4 high bound at bpp=8
        let spatial = idct_8x8(&f, 12);
        // Sign pattern: spatial[y][x] sign should match (-1)^(x+y) (or
        // the opposite — what matters is that it alternates uniformly).
        let pivot_sign = spatial[0][0].signum();
        assert!(pivot_sign != 0, "pivot sample is zero");
        for y in 0..8 {
            for x in 0..8 {
                let expected = if (x + y) % 2 == 0 {
                    pivot_sign
                } else {
                    -pivot_sign
                };
                assert_eq!(
                    spatial[y][x].signum(),
                    expected,
                    "y={y} x={x} value={}",
                    spatial[y][x]
                );
            }
        }
    }
}

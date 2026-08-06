//! Forward 8×8 discrete cosine transform for the encoder.
//!
//! Annex A.1 defines the codec's transform pair; the decoder evaluates
//! the inverse ([`crate::idct::idct_8x8`]). The encoder needs the
//! matching forward transform:
//!
//! ```text
//!                 2  N-1 N-1                        (2x + 1)uπ     (2y + 1)vπ
//!     F(u, v) =  --- C(u) C(v)  ∑    ∑   f(x, y) cos ----------- cos -----------
//!                 N             x=0  y=0                 2N             2N
//!
//!     with  N = 8,  C(0) = 1/√2,  C(k) = 1 for k ≠ 0.
//! ```
//!
//! (The orthonormal scaling `2/N = √(2/N)·√(2/N)` is split across the
//! two separable 1-D passes, mirroring the inverse.) The forward
//! transform is not normative for an encoder — any transform whose
//! coefficients, after quantisation and the decoder's normative
//! inverse pipeline, reconstruct acceptably is conformant — but using
//! the ideal Annex A.1 kernel in `f64` keeps this encoder's residual
//! model exactly dual to the crate's own decoder.
//!
//! The output is rounded to the nearest integer (§4.1 half-away
//! rounding, same discipline as the inverse side) and saturated to the
//! §7.4.4.4 coefficient range `[-2^(bpp+3), 2^(bpp+3) - 1]`, i.e. the
//! domain the quantisers consume.
//!
//! Provenance: the transform definition is Annex A.1 of ISO/IEC
//! 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
//! No third-party source was consulted.

#![allow(clippy::needless_range_loop)]

use crate::inverse_quant::saturation_bounds;

const N: usize = 8;

/// Pre-computed cosine table `COS[u][x] = cos((2x + 1)uπ / 16)` —
/// the same kernel the inverse side uses, indexed here as the forward
/// sum over `x` for each output frequency `u`.
///
/// The values are **compile-time `f64` literals** (each is the
/// nearest-`f64` to the mathematical cosine, printed to full
/// round-trip precision) rather than runtime `f64::cos` calls: libm
/// implementations legitimately differ by an ulp across platforms,
/// and an encoder must emit **byte-identical** streams everywhere for
/// the committed black-box fixtures to pin anything. With literal
/// kernel constants every operation below is IEEE-754-determined, so
/// the emitted bitstream is platform-independent.
#[rustfmt::skip]
const COS: [[f64; N]; N] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.9807852804032304, 0.8314696123025452, 0.5555702330196023, 0.19509032201612833, -0.1950903220161282, -0.555570233019602, -0.8314696123025453, -0.9807852804032304],
    [0.9238795325112867, 0.38268343236508984, -0.3826834323650897, -0.9238795325112867, -0.9238795325112868, -0.38268343236509034, 0.38268343236509, 0.9238795325112865],
    [0.8314696123025452, -0.1950903220161282, -0.9807852804032304, -0.5555702330196022, 0.5555702330196018, 0.9807852804032304, 0.19509032201612878, -0.8314696123025451],
    [0.7071067811865476, -0.7071067811865475, -0.7071067811865477, 0.7071067811865474, 0.7071067811865477, -0.7071067811865467, -0.7071067811865472, 0.7071067811865466],
    [0.5555702330196023, -0.9807852804032304, 0.1950903220161283, 0.8314696123025455, -0.8314696123025451, -0.19509032201612803, 0.9807852804032307, -0.5555702330196015],
    [0.38268343236508984, -0.9238795325112868, 0.9238795325112865, -0.3826834323650899, -0.38268343236509056, 0.9238795325112867, -0.9238795325112864, 0.38268343236508956],
    [0.19509032201612833, -0.5555702330196022, 0.8314696123025455, -0.9807852804032307, 0.9807852804032304, -0.831469612302545, 0.5555702330196015, -0.19509032201612858],
];

#[inline]
fn cos_table() -> &'static [[f64; N]; N] {
    &COS
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

/// 1-D 8-point forward DCT: `F(u) = √(2/N) C(u) Σ_x f(x) cos((2x+1)uπ/2N)`.
#[inline]
fn fdct_1d(input: &[f64; N]) -> [f64; N] {
    let cos = cos_table();
    let scale = (2.0_f64 / N as f64).sqrt();
    let mut output = [0.0f64; N];
    for u in 0..N {
        let mut acc = 0.0f64;
        for x in 0..N {
            acc += input[x] * cos[u][x];
        }
        output[u] = scale * c(u) * acc;
    }
    output
}

/// Annex A.1 8×8 orthonormal forward DCT in `f64`, rounded to the
/// nearest integer (§4.1 half-away-from-zero) and saturated to the
/// §7.4.4.4 range for `bits_per_pixel`.
///
/// `samples[y][x]` is the spatial input (for intra blocks the raw
/// pixel values `0..=2^bpp - 1`; for inter blocks the signed residual
/// after motion compensation). The returned `F[v][u]` block is laid
/// out exactly as the decoder's §7.4.4 output, ready for the
/// [`crate::quantise`] forward quantisers.
pub fn forward_dct_8x8(samples: &[[i32; 8]; 8], bits_per_pixel: u32) -> [[i32; 8]; 8] {
    // Row pass over x for each fixed y.
    let mut row_out = [[0.0f64; N]; N];
    for y in 0..N {
        let mut row_in = [0.0f64; N];
        for x in 0..N {
            row_in[x] = samples[y][x] as f64;
        }
        row_out[y] = fdct_1d(&row_in);
    }
    // Column pass over y for each fixed u.
    let (lo, hi) = saturation_bounds(bits_per_pixel);
    let mut out = [[0i32; 8]; 8];
    for u in 0..N {
        let mut col_in = [0.0f64; N];
        for y in 0..N {
            col_in[y] = row_out[y][u];
        }
        let col_out = fdct_1d(&col_in);
        for v in 0..N {
            out[v][u] = (col_out[v].round() as i32).clamp(lo, hi);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::idct_8x8;

    /// Deterministic LCG so the tests need no external inputs.
    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    #[test]
    fn flat_block_transforms_to_pure_dc() {
        let samples = [[128i32; 8]; 8];
        let f = forward_dct_8x8(&samples, 8);
        // DC of a flat block: 8 * value (orthonormal 2-D: N * value).
        assert_eq!(f[0][0], 128 * 8);
        for v in 0..8 {
            for u in 0..8 {
                if (v, u) != (0, 0) {
                    assert_eq!(f[v][u], 0, "AC ({v},{u}) must be zero");
                }
            }
        }
    }

    #[test]
    fn forward_then_inverse_is_identity_on_pixel_data() {
        // The orthonormal pair round-trips exactly (integer rounding
        // both ways stays within ±1, and for typical pixel data the
        // reconstruction is exact — assert the tight ±1 envelope).
        let mut state = 0xDEAD_BEEFu32;
        for _ in 0..50 {
            let mut samples = [[0i32; 8]; 8];
            for row in samples.iter_mut() {
                for cell in row.iter_mut() {
                    *cell = (lcg(&mut state) >> 24) as i32; // 0..=255
                }
            }
            let f = forward_dct_8x8(&samples, 8);
            let back = idct_8x8(&f, 8);
            for y in 0..8 {
                for x in 0..8 {
                    let d = (back[y][x] - samples[y][x]).abs();
                    assert!(
                        d <= 1,
                        "roundtrip drift {d} at ({y},{x}): {} -> {}",
                        samples[y][x],
                        back[y][x]
                    );
                }
            }
        }
    }

    #[test]
    fn saturates_to_coefficient_range() {
        // An all-255 block has DC 2040 — inside [−2048, 2047]. Push
        // beyond with the widest legal signed residual input.
        let samples = [[255i32; 8]; 8];
        let f = forward_dct_8x8(&samples, 8);
        assert_eq!(f[0][0], 2040);
        let big = [[300i32; 8]; 8]; // out-of-gamut input clamps at +2047
        let f = forward_dct_8x8(&big, 8);
        assert_eq!(f[0][0], 2047);
    }
}

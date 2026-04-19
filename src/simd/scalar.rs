//! Scalar reference implementations.
//!
//! Always compiled, used as the fallback on any target and as the oracle
//! for `simd::tests::*_matches_scalar`. Every other implementation in
//! this directory must match the numeric output of the scalar path
//! (aside from unavoidable FP reassociation in the DCT kernels).

use std::f32::consts::PI;
use std::sync::OnceLock;

/// Shared 8×8 cosine table — `t[k][n] = 0.5 * C_k * cos((2n+1) k π / 16)`
/// with `C_0 = 1/√2`, `C_{>0} = 1`. Both the IDCT and FDCT use this same
/// table (so `idct(fdct(x)) ≈ x` within float rounding).
pub(super) fn cos_table() -> &'static [[f32; 8]; 8] {
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 {
                (1.0_f32 / 2.0_f32).sqrt()
            } else {
                1.0
            };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    })
}

/// Inverse DCT of an 8×8 natural-order block, in-place.
#[inline]
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Forward DCT of an 8×8 natural-order block, in-place.
#[inline]
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    // Row-wise forward: tmp[y][k] = Σn t[k][n] * block[y][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    // Column-wise.
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
}

/// Dequantise 64 coefficients with the H.263 rule in-place. `start=0`
/// for inter (all 64), `start=1` for intra (DC excluded).
#[inline]
pub fn dequant_h263(coeffs: &mut [i32; 64], q: i32, q_plus: i32, start: usize) {
    let two_q = 2 * q;
    for i in start..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        let abs = l.abs();
        let mut val = two_q * abs + q_plus;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
    }
}

/// Clip + store an 8×8 signed tile to a `u8` plane at row stride `stride`.
#[inline]
pub fn clip_block_to_u8(src: &[i32; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    for j in 0..8 {
        for i in 0..8 {
            let v = src[j * 8 + i];
            let c = if v < 0 {
                0
            } else if v > 255 {
                255
            } else {
                v as u8
            };
            dst[dst_off + j * stride + i] = c;
        }
    }
}

/// Add residual + predictor, clip to `u8` and store at row stride.
#[inline]
pub fn add_residual_clip_block(
    pred: &[u8; 64],
    residual: &[i32; 64],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    for j in 0..8 {
        for i in 0..8 {
            let v = pred[j * 8 + i] as i32 + residual[j * 8 + i];
            let c = if v < 0 {
                0
            } else if v > 255 {
                255
            } else {
                v as u8
            };
            dst[dst_off + j * stride + i] = c;
        }
    }
}

/// Copy an 8×8 `u8` block into a row-strided destination.
#[inline]
pub fn copy_block_u8(src: &[u8; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    for j in 0..8 {
        for i in 0..8 {
            dst[dst_off + j * stride + i] = src[j * 8 + i];
        }
    }
}

/// Copy a 16×16 luma macroblock.
#[inline]
pub fn copy_mb_luma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for j in 0..16 {
        for i in 0..16 {
            dst[dst_off + j * dst_stride + i] = src[src_off + j * src_stride + i];
        }
    }
}

/// Copy an 8×8 chroma block.
#[inline]
pub fn copy_mb_chroma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for j in 0..8 {
        for i in 0..8 {
            dst[dst_off + j * dst_stride + i] = src[src_off + j * src_stride + i];
        }
    }
}

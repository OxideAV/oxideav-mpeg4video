//! "Variant RC" — the integer row–column 8×8 inverse transform that
//! streams of one encoder family are reconstructed with.
//!
//! ISO/IEC 14496-2 Annex A bounds the inverse DCT by accuracy only, so
//! two conformant transforms may differ by a level per sample, and
//! inside the prediction loop that difference accumulates until the
//! next intra picture. Streams signed with the user-data string the
//! [`crate::decoder`] stream-selection rule recognises were produced
//! against one specific 16-bit-oriented fixed-point transform; decoding
//! them drift-free needs the same integers. This module is that
//! transform, written from its defining equations
//! (`docs/video/mpeg4-visual/integer-idct-ecosystem-variant.md`, whose
//! section and equation numbers the comments below cite as "RC §n").
//!
//! ## Shape of the computation
//!
//! * **Row pass** (RC §4). Every output of a row is the rounding term
//!   plus a weighted sum of the eight inputs, shifted right by 11:
//!   `y_k = (ρ_v + Σ_u B_v[u][k] · x_u) >> 11`. The weight `B_v[u][k]`
//!   is the fixed-point value of `C(u) · cos((2k + 1)uπ/16)` scaled for
//!   row `v`; the eight weights that multiply one input form that
//!   input's *basis vector*. The row pass accumulates one basis vector
//!   per non-zero input over all eight outputs at once — eight
//!   independent 32-bit lanes with no cross-lane dependency. Because
//!   32-bit wrapping addition and multiplication form a ring, the
//!   order of accumulation cannot change any result, so this evaluates
//!   exactly the integers of equations R1–R10.
//! * **Column pass** (RC §5). A tangent-rotation factorisation with
//!   floor shifts; all eight columns advance together through
//!   equations C1–C20, again as independent lanes.
//!
//! Word widths follow RC §3: inputs saturate to 16 bits, 32-bit sums
//! and products wrap, the row-pass result keeps only its low 16 bits
//! (no saturation), column intermediates stay 32 bits wide.
//!
//! The two normative rules: the "empty row" rule (RC §4.3) is
//! implemented in the DC-only branch of the row pass; the row-skipping
//! rule (RC §5.3) is result-neutral by the specification's own
//! analysis, so the column pass simply reads all eight rows.
//!
//! ## Lint allowance
//!
//! `#![allow(clippy::needless_range_loop)]`: the loops index by the
//! specification's own `u` / `k` / `x` / `v` so each line can be read
//! against its equation.

#![allow(clippy::needless_range_loop)]

const N: usize = 8;

/// Right shift closing the row pass (RC §4.2, R9 / R10).
const ROW_SHIFT: u32 = 11;

/// Right shift closing the column pass (RC §5.2, C19 / C20).
const COLUMN_SHIFT: u32 = 6;

/// Row multipliers `w1 … w7` of the four sets (RC §4.1), stored so
/// that index `k` holds the weight of phase `kπ/16`: entry `k` of set
/// `m` is `2^15 · cos(kπ/16) · cos(mπ/16)` rounded, with `m = 4, 1, 2,
/// 3` for the sets used by rows 0/4, 1/7, 2/6 and 3/5. Index 0 is
/// unused by the specification's equations; it is filled with the
/// phase-4 weight because the `C(0) = 1/√2 = cos(4π/16)` factor makes
/// that the weight of the DC input. Index 8 (`cos(π/2)`) is zero.
const PHASE_WEIGHTS: [[i32; 9]; 4] = [
    [16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520, 0],
    [22725, 31521, 29692, 26722, 22725, 17855, 12299, 6270, 0],
    [21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906, 0],
    [19266, 26722, 25172, 22654, 19266, 15137, 10426, 5315, 0],
];

/// Which multiplier set each row uses (RC §4.1).
const WEIGHT_SET_OF_ROW: [usize; N] = [0, 1, 2, 3, 0, 3, 2, 1];

/// Rounding term `ρ_v` of each row (RC §4.1). Not symmetric: rows that
/// share a multiplier set do not share a rounding term. `ρ_0 = 2^16`
/// is half a unit of the combined 17-bit output scale; the others
/// dither the remaining fraction.
const ROW_ROUNDING: [i32; N] = [65536, 3597, 2260, 1203, 0, 120, 512, 512];

/// Basis vectors of one multiplier set: `basis[u][k]` is the signed
/// weight of input `x_u` in output `y_k`.
type RowBasis = [[i32; N]; N];

/// Build the basis vectors of one multiplier set from its phase
/// weights via the quarter-wave symmetry of the cosine: the weight of
/// `x_u` in `y_k` has phase `(2k + 1)·u mod 32` (in units of π/16),
/// folded onto the first quadrant with its sign.
const fn row_basis(weights: &[i32; 9]) -> RowBasis {
    let mut basis = [[0i32; N]; N];
    let mut u = 0;
    while u < N {
        let mut k = 0;
        while k < N {
            let phase = ((2 * k + 1) * u) % 32;
            basis[u][k] = if u == 0 {
                weights[0]
            } else if phase <= 8 {
                weights[phase]
            } else if phase <= 16 {
                -weights[16 - phase]
            } else if phase <= 24 {
                -weights[phase - 16]
            } else {
                weights[32 - phase]
            };
            k += 1;
        }
        u += 1;
    }
    basis
}

/// The four sets' basis vectors.
const ROW_BASES: [RowBasis; 4] = [
    row_basis(&PHASE_WEIGHTS[0]),
    row_basis(&PHASE_WEIGHTS[1]),
    row_basis(&PHASE_WEIGHTS[2]),
    row_basis(&PHASE_WEIGHTS[3]),
];

/// `tan(π/16) · 2^16` (RC §5.1).
const TAN_1_PI_16: i32 = 13036;
/// `tan(2π/16) · 2^16`.
const TAN_2_PI_16: i32 = 27146;
/// `tan(3π/16) · 2^16` — a positive value above `2^15`.
const TAN_3_PI_16: i32 = 43790;
/// `(1/√2) · 2^15`.
const INV_SQRT_2: i32 = 23170;

/// The scaled multiply of RC §5.1 (C0): the 32-bit wrapping product
/// shifted right arithmetically by 16, no rounding.
#[inline(always)]
fn scaled(constant: i32, value: i32) -> i32 {
    constant.wrapping_mul(value) >> 16
}

/// Saturate a coefficient to the 16-bit input domain (RC §3).
#[inline(always)]
fn to_input_word(value: i32) -> i32 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

/// RC §4: transform one row. `row` is the row index `v` (it selects
/// the multiplier set and the rounding term); `x` holds the eight
/// 16-bit-domain inputs.
#[inline]
fn row_pass(row: usize, x: &[i32; N]) -> [i16; N] {
    let basis = &ROW_BASES[WEIGHT_SET_OF_ROW[row]];
    let rounding = ROW_ROUNDING[row];
    if x[1..] == [0i32; N - 1] {
        // Only the DC input can contribute: every output is the same
        // shifted value.
        let flat = basis[0][0].wrapping_mul(x[0]).wrapping_add(rounding) >> ROW_SHIFT;
        if flat == 0 {
            // RC §4.3 "empty row": the row is left exactly as it was,
            // so `x_0` stays in position 0.
            let mut kept = [0i16; N];
            kept[0] = x[0] as i16;
            return kept;
        }
        return [flat as i16; N];
    }
    let mut acc = [rounding; N];
    for (u, &input) in x.iter().enumerate() {
        if input == 0 {
            continue;
        }
        let vector = &basis[u];
        for k in 0..N {
            acc[k] = acc[k].wrapping_add(vector[k].wrapping_mul(input));
        }
    }
    let mut y = [0i16; N];
    for k in 0..N {
        // Low 16 bits of the shifted sum — truncation, not saturation.
        y[k] = (acc[k] >> ROW_SHIFT) as i16;
    }
    y
}

/// RC §5: transform all eight columns of the row-pass result `g`.
#[inline]
fn column_pass(g: &[[i16; N]; N]) -> [[i16; N]; N] {
    let mut out = [[0i16; N]; N];
    for x in 0..N {
        let c: [i32; N] = [
            i32::from(g[0][x]),
            i32::from(g[1][x]),
            i32::from(g[2][x]),
            i32::from(g[3][x]),
            i32::from(g[4][x]),
            i32::from(g[5][x]),
            i32::from(g[6][x]),
            i32::from(g[7][x]),
        ];
        // Odd part (C1–C10).
        let p = c[1].wrapping_add(scaled(TAN_1_PI_16, c[7]));
        let q = scaled(TAN_1_PI_16, c[1]).wrapping_sub(c[7]);
        let r = c[3].wrapping_add(scaled(TAN_3_PI_16, c[5]));
        let s = scaled(TAN_3_PI_16, c[3]).wrapping_sub(c[5]);
        let t = p.wrapping_sub(r);
        let d = q.wrapping_add(s);
        let odd = [
            p.wrapping_add(r),
            // Doubling after the shift (always even), not a 15-bit shift.
            scaled(INV_SQRT_2, t.wrapping_add(d)).wrapping_mul(2),
            scaled(INV_SQRT_2, t.wrapping_sub(d)).wrapping_mul(2),
            q.wrapping_sub(s),
        ];
        // Even part (C11–C18).
        let g2 = c[2].wrapping_add(scaled(TAN_2_PI_16, c[6]));
        let h = scaled(TAN_2_PI_16, c[2]).wrapping_sub(c[6]);
        let m = c[0].wrapping_add(c[4]);
        let n = c[0].wrapping_sub(c[4]);
        let even = [
            m.wrapping_add(g2),
            n.wrapping_add(h),
            n.wrapping_sub(h),
            m.wrapping_sub(g2),
        ];
        // C19 / C20.
        for k in 0..4 {
            out[k][x] = (even[k].wrapping_add(odd[k]) >> COLUMN_SHIFT) as i16;
            out[7 - k][x] = (even[k].wrapping_sub(odd[k]) >> COLUMN_SHIFT) as i16;
        }
    }
    out
}

/// The row-pass result `G` of a coefficient block (RC §4), inputs
/// saturated to 16 bits first.
#[inline]
fn row_stage(coefficients: &[[i32; 8]; 8]) -> [[i16; N]; N] {
    let mut g = [[0i16; N]; N];
    for v in 0..N {
        let mut x = [0i32; N];
        for u in 0..N {
            x[u] = to_input_word(coefficients[v][u]);
        }
        g[v] = row_pass(v, &x);
    }
    g
}

/// Variant RC on one block, **without** output saturation: the raw
/// 16-bit transform result `f[y][x]` for the coefficient block
/// `coefficients[v][u]` (values outside the 16-bit range are saturated
/// to it first).
pub fn variant_rc_raw(coefficients: &[[i32; 8]; 8]) -> [[i16; 8]; 8] {
    column_pass(&row_stage(coefficients))
}

/// Variant RC followed by the §7.4.5 output saturation to
/// `[-2^bits_per_pixel, 2^bits_per_pixel - 1]` — a drop-in for
/// [`crate::idct::idct_8x8`] at the decoder's inverse-transform call.
pub fn idct_8x8_variant_rc(coefficients: &[[i32; 8]; 8], bits_per_pixel: u32) -> [[i32; 8]; 8] {
    let raw = variant_rc_raw(coefficients);
    let (lo, hi) = crate::idct::idct_saturation_bounds(bits_per_pixel);
    let mut out = [[0i32; 8]; 8];
    for y in 0..N {
        for x in 0..N {
            out[y][x] = i32::from(raw[y][x]).clamp(lo, hi);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equations R1–R10 and C1–C20 exactly as the specification prints
    /// them, one row / one column at a time, including both rules.
    fn printed_equations(coefficients: &[[i32; 8]; 8]) -> ([[i16; 8]; 8], [[i16; 8]; 8]) {
        const SETS: [[i32; 8]; 4] = [
            [0, 22725, 21407, 19266, 16384, 12873, 8867, 4520],
            [0, 31521, 29692, 26722, 22725, 17855, 12299, 6270],
            [0, 29692, 27969, 25172, 21407, 16819, 11585, 5906],
            [0, 26722, 25172, 22654, 19266, 15137, 10426, 5315],
        ];
        let mul = |a: i32, b: i32| a.wrapping_mul(b);
        let sum = |terms: &[i32]| terms.iter().fold(0i32, |a, &b| a.wrapping_add(b));
        let mut g = [[0i16; 8]; 8];
        let mut non_empty = [true; 8];
        for v in 0..8 {
            let w = &SETS[[0usize, 1, 2, 3, 0, 3, 2, 1][v]];
            let rho = [65536, 3597, 2260, 1203, 0, 120, 512, 512][v];
            let mut x = [0i32; 8];
            for u in 0..8 {
                x[u] = coefficients[v][u].clamp(-32768, 32767);
            }
            let base = mul(w[4], x[0]).wrapping_add(rho);
            if x[1..].iter().all(|&c| c == 0) && base >> 11 == 0 {
                g[v][0] = x[0] as i16;
                non_empty[v] = false;
                continue;
            }
            let e = [
                sum(&[base, mul(w[2], x[2]), mul(w[4], x[4]), mul(w[6], x[6])]),
                sum(&[base, mul(w[6], x[2]), -mul(w[4], x[4]), -mul(w[2], x[6])]),
                sum(&[base, -mul(w[6], x[2]), -mul(w[4], x[4]), mul(w[2], x[6])]),
                sum(&[base, -mul(w[2], x[2]), mul(w[4], x[4]), -mul(w[6], x[6])]),
            ];
            let o = [
                sum(&[
                    mul(w[1], x[1]),
                    mul(w[3], x[3]),
                    mul(w[5], x[5]),
                    mul(w[7], x[7]),
                ]),
                sum(&[
                    mul(w[3], x[1]),
                    -mul(w[7], x[3]),
                    -mul(w[1], x[5]),
                    -mul(w[5], x[7]),
                ]),
                sum(&[
                    mul(w[5], x[1]),
                    -mul(w[1], x[3]),
                    mul(w[7], x[5]),
                    mul(w[3], x[7]),
                ]),
                sum(&[
                    mul(w[7], x[1]),
                    -mul(w[5], x[3]),
                    mul(w[3], x[5]),
                    -mul(w[1], x[7]),
                ]),
            ];
            for k in 0..4 {
                g[v][k] = (e[k].wrapping_add(o[k]) >> 11) as i16;
                g[v][7 - k] = (e[k].wrapping_sub(o[k]) >> 11) as i16;
            }
        }
        // Rule 2: which rows the column pass reads.
        let rows_read = if non_empty[4..].iter().any(|&b| b) {
            8
        } else if non_empty[3] {
            4
        } else {
            3
        };
        let m16 = |c: i32, a: i32| c.wrapping_mul(a) >> 16;
        let mut f = [[0i16; 8]; 8];
        for x in 0..8 {
            let mut c = [0i32; 8];
            for v in 0..rows_read {
                c[v] = i32::from(g[v][x]);
            }
            let p = c[1] + m16(13036, c[7]);
            let q = m16(13036, c[1]) - c[7];
            let r = c[3] + m16(43790, c[5]);
            let s = m16(43790, c[3]) - c[5];
            let (t, d) = (p - r, q + s);
            let o = [p + r, 2 * m16(23170, t + d), 2 * m16(23170, t - d), q - s];
            let gg = c[2] + m16(27146, c[6]);
            let h = m16(27146, c[2]) - c[6];
            let (m, n) = (c[0] + c[4], c[0] - c[4]);
            let e = [m + gg, n + h, n - h, m - gg];
            for k in 0..4 {
                f[k][x] = ((e[k] + o[k]) >> 6) as i16;
                f[7 - k][x] = ((e[k] - o[k]) >> 6) as i16;
            }
        }
        (g, f)
    }

    fn sparse(entries: &[(usize, usize, i32)]) -> [[i32; 8]; 8] {
        let mut f = [[0i32; 8]; 8];
        for &(v, u, value) in entries {
            f[v][u] = value;
        }
        f
    }

    #[test]
    fn phase_weights_are_the_scaled_cosine_products() {
        // Entry k of set m is 2^15 · cos(kπ/16) · cos(mπ/16), rounded
        // to nearest, m = 4, 1, 2, 3.
        let cos = |k: usize| crate::idct::COS_K_PI_16[k];
        for (set, m) in [4usize, 1, 2, 3].into_iter().enumerate() {
            for k in 1..=7 {
                let ideal = 32768.0 * cos(k) * cos(m);
                assert_eq!(
                    PHASE_WEIGHTS[set][k],
                    ideal.round() as i32,
                    "set {set} phase {k}: ideal {ideal}"
                );
            }
            assert_eq!(PHASE_WEIGHTS[set][0], PHASE_WEIGHTS[set][4]);
            assert_eq!(PHASE_WEIGHTS[set][8], 0);
        }
    }

    #[test]
    fn column_constants_are_the_scaled_tangents() {
        let cos = |k: usize| crate::idct::COS_K_PI_16[k];
        let tan = |k: usize| cos(8 - k) / cos(k);
        assert_eq!(TAN_1_PI_16, (tan(1) * 65536.0).round() as i32);
        assert_eq!(TAN_2_PI_16, (tan(2) * 65536.0).round() as i32);
        assert_eq!(TAN_3_PI_16, (tan(3) * 65536.0).round() as i32);
        assert_eq!(INV_SQRT_2, (cos(4) * 32768.0).round() as i32);
    }

    #[test]
    fn basis_vectors_reproduce_the_printed_sign_pattern() {
        // Set A, spelled out from R1–R8 for outputs 0 and 1.
        let b = &ROW_BASES[0];
        let y0: Vec<i32> = (0..8).map(|u| b[u][0]).collect();
        let y1: Vec<i32> = (0..8).map(|u| b[u][1]).collect();
        assert_eq!(y0, [16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520]);
        assert_eq!(
            y1,
            [16384, 19266, 8867, -4520, -16384, -22725, -21407, -12873]
        );
        // Mirror outputs: even inputs keep their sign, odd inputs flip.
        for set in &ROW_BASES {
            for u in 0..8 {
                for k in 0..4 {
                    let sign = if u % 2 == 0 { 1 } else { -1 };
                    assert_eq!(set[u][7 - k], sign * set[u][k]);
                }
            }
        }
    }

    #[test]
    fn all_zero_block_is_zero_through_non_zero_rows() {
        // RC §8.4: G rows 0 / 1 / 2 are eight 32s / 1s / 1s, the rest
        // zero; the output is nevertheless all zero.
        let zero = [[0i32; 8]; 8];
        let g = row_stage(&zero);
        assert_eq!(g[0], [32i16; 8]);
        assert_eq!(g[1], [1i16; 8]);
        assert_eq!(g[2], [1i16; 8]);
        for row in &g[3..] {
            assert_eq!(*row, [0i16; 8]);
        }
        assert_eq!(variant_rc_raw(&zero), [[0i16; 8]; 8]);
        assert_eq!(idct_8x8_variant_rc(&zero, 8), [[0i32; 8]; 8]);
    }

    #[test]
    fn dc_extremes_saturate_at_the_output() {
        // RC §8.4: F[0][0] = 2047 alone → 256 everywhere → 255 after
        // the 8-bit output saturation; −2048 alone → −256 everywhere.
        let hi = sparse(&[(0, 0, 2047)]);
        assert_eq!(variant_rc_raw(&hi), [[256i16; 8]; 8]);
        assert_eq!(idct_8x8_variant_rc(&hi, 8), [[255i32; 8]; 8]);
        let lo = sparse(&[(0, 0, -2048)]);
        assert_eq!(variant_rc_raw(&lo), [[-256i16; 8]; 8]);
        assert_eq!(idct_8x8_variant_rc(&lo, 8), [[-256i32; 8]; 8]);
        // A wider sample range leaves the raw value alone.
        assert_eq!(idct_8x8_variant_rc(&hi, 10), [[256i32; 8]; 8]);
    }

    #[test]
    fn dc_only_example_dithers_the_half() {
        // RC §8.1: F[0][0] = 100 → G row 0 all 832, rows 1 / 2 all 1;
        // output rows 0, 1, 2, 5, 6, 7 all 13, rows 3 and 4 all 12.
        let f = sparse(&[(0, 0, 100)]);
        let g = row_stage(&f);
        assert_eq!(g[0], [832i16; 8]);
        assert_eq!(g[1], [1i16; 8]);
        assert_eq!(g[2], [1i16; 8]);
        let out = variant_rc_raw(&f);
        for (y, row) in out.iter().enumerate() {
            let want = if y == 3 || y == 4 { 12 } else { 13 };
            assert_eq!(*row, [want as i16; 8], "row {y}");
        }
    }

    #[test]
    fn empty_row_rule_keeps_minus_four_in_place() {
        // RC §4.3 worked example: F[0][0] = −4 alone.
        let f = sparse(&[(0, 0, -4)]);
        let g = row_stage(&f);
        assert_eq!(g[0], [-4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(g[1], [1i16; 8]);
        assert_eq!(g[2], [1i16; 8]);
        for row in &g[3..] {
            assert_eq!(*row, [0i16; 8]);
        }
        let out = variant_rc_raw(&f);
        for (y, row) in out.iter().enumerate() {
            let want: [i16; 8] = if y == 3 || y == 4 {
                [-1; 8]
            } else {
                [-1, 0, 0, 0, 0, 0, 0, 0]
            };
            assert_eq!(*row, want, "row {y}");
        }
        // What the formula without the rule would give: G row 0 all
        // zero → rows 3 and 4 all −1, everything else 0 (six samples of
        // column 0 differ).
        let mut g_formula = g;
        g_formula[0] = [0; 8];
        let formula = column_pass(&g_formula);
        let mut differing = 0;
        for y in 0..8 {
            for x in 0..8 {
                let want = if y == 3 || y == 4 { -1 } else { 0 };
                assert_eq!(formula[y][x], want);
                differing += usize::from(formula[y][x] != out[y][x]);
            }
        }
        assert_eq!(differing, 6);
    }

    #[test]
    fn empty_row_condition_holds_only_where_the_table_says() {
        // RC §4.3 enumeration over every 16-bit x0 with x1…x7 = 0:
        // row 0 is empty for −4 only, rows 1 and 2 never, rows 3–7 for
        // 0 only. Everywhere else a DC-only row is the flat shifted
        // value (low 16 bits).
        let step = if cfg!(miri) { 257 } else { 1 };
        for row in 0..8 {
            let w4 = PHASE_WEIGHTS[WEIGHT_SET_OF_ROW[row]][4];
            let mut empties = Vec::new();
            let mut x0 = -32768i32;
            while x0 <= 32767 {
                let mut x = [0i32; 8];
                x[0] = x0;
                let y = row_pass(row, &x);
                let flat = (w4 * x0 + ROW_ROUNDING[row]) >> 11;
                if flat == 0 {
                    empties.push(x0);
                    let mut kept = [0i16; 8];
                    kept[0] = x0 as i16;
                    assert_eq!(y, kept, "row {row} x0 {x0}");
                } else {
                    assert_eq!(y, [flat as i16; 8], "row {row} x0 {x0}");
                }
                x0 += step;
            }
            if step == 1 {
                let want: &[i32] = match row {
                    0 => &[-4],
                    1 | 2 => &[],
                    _ => &[0],
                };
                assert_eq!(empties, want, "row {row}");
            }
        }
        // The observable case and its neighbours, at any step.
        assert_eq!(row_pass(0, &[-4, 0, 0, 0, 0, 0, 0, 0])[0], -4);
        assert_eq!(row_pass(0, &[-3, 0, 0, 0, 0, 0, 0, 0]), [8i16; 8]);
        assert_eq!(row_pass(0, &[-5, 0, 0, 0, 0, 0, 0, 0]), [-8i16; 8]);
        assert_eq!(row_pass(1, &[0; 8]), [1i16; 8]);
        assert_eq!(row_pass(2, &[0; 8]), [1i16; 8]);
        // A row whose results cancel to zero with a non-zero AC input
        // is not empty: it becomes zeros, x0 is not kept.
        assert_eq!(row_pass(4, &[0, 0, 0, 0, 0, 0, 0, 0]), [0i16; 8]);
    }

    #[test]
    fn row_skipping_example_matches_with_all_rows_read() {
        // RC §5.3 worked example: F[0][0] = 64, F[2][1] = −20.
        let f = sparse(&[(0, 0, 64), (2, 1, -20)]);
        let g = row_stage(&f);
        assert_eq!(g[0], [544i16; 8]);
        assert_eq!(g[1], [1i16; 8]);
        assert_eq!(g[2], [-289, -245, -164, -57, 58, 165, 246, 291]);
        for row in &g[3..] {
            assert_eq!(*row, [0i16; 8]);
        }
        let want: [[i16; 8]; 8] = [
            [4, 4, 5, 7, 9, 11, 12, 13],
            [6, 6, 7, 8, 8, 9, 10, 10],
            [10, 10, 9, 8, 8, 7, 6, 6],
            [13, 12, 11, 9, 7, 5, 4, 3],
            [13, 12, 11, 9, 7, 5, 4, 3],
            [10, 10, 9, 8, 8, 7, 6, 6],
            [6, 6, 7, 8, 8, 9, 10, 10],
            [3, 4, 5, 7, 9, 11, 12, 13],
        ];
        assert_eq!(variant_rc_raw(&f), want);
        // The three-row read of the printed equations agrees.
        assert_eq!(printed_equations(&f).1, want);
    }

    #[test]
    fn sparse_example_row_and_column_stages() {
        // RC §8.2.
        let f = sparse(&[
            (0, 0, -240),
            (0, 1, 57),
            (1, 0, -33),
            (2, 3, 12),
            (5, 2, -9),
            (7, 7, 4),
        ]);
        let g_want: [[i16; 8]; 8] = [
            [-1256, -1352, -1530, -1763, -2014, -2247, -2425, -2521],
            [-365; 8],
            [148, -34, -173, -98, 99, 175, 35, -147],
            [0; 8],
            [0; 8],
            [-111, -46, 45, 110, 110, 45, -46, -111],
            [0; 8],
            [12, -35, 52, -62, 61, -52, 35, -12],
        ];
        assert_eq!(row_stage(&f), g_want);
        let f_want: [[i16; 8]; 8] = [
            [-25, -28, -32, -34, -35, -38, -44, -49],
            [-22, -26, -32, -35, -39, -40, -42, -44],
            [-25, -25, -26, -31, -35, -41, -42, -43],
            [-25, -22, -23, -25, -34, -38, -41, -40],
            [-19, -20, -20, -28, -33, -39, -37, -35],
            [-18, -18, -21, -24, -31, -33, -36, -35],
            [-16, -18, -19, -22, -24, -29, -34, -38],
            [-11, -16, -22, -25, -26, -27, -32, -35],
        ];
        assert_eq!(variant_rc_raw(&f), f_want);
        // The scaled multiply floors toward −∞ (column 0 working).
        assert_eq!(scaled(TAN_1_PI_16, 12), 2);
        assert_eq!(scaled(TAN_1_PI_16, -365), -73);
        assert_eq!(scaled(TAN_3_PI_16, -111), -75);
        assert_eq!(scaled(INV_SQRT_2, -262), -93);
        assert_eq!(scaled(INV_SQRT_2, -314), -112);
        assert_eq!(scaled(TAN_2_PI_16, 148), 61);
    }

    #[test]
    fn dense_example_row_stage() {
        // RC §8.3 (the first staged vector): the row-pass result.
        let f: [[i32; 8]; 8] = [
            [21, 9, 10, 25, 10, -2, 12, 15],
            [14, -10, -25, -6, 9, 11, 29, 20],
            [-10, -24, 21, 19, 22, 28, -19, -5],
            [-13, 9, -1, 0, 22, -9, -22, -1],
            [-2, 3, 14, -7, 15, 26, 24, 27],
            [20, 15, -13, -7, 22, -23, -24, 23],
            [-8, -4, -29, 20, 20, -8, -2, -20],
            [-12, 26, -1, 17, 10, 29, 0, -17],
        ];
        let g_want: [[i16; 8]; 8] = [
            [792, -5, 117, -200, 446, 286, 80, 80],
            [-7, -970, 927, 302, 587, 328, -58, 143],
            [406, -670, -1165, 138, -245, -259, 759, 204],
            [9, 161, -562, 146, 272, -627, -288, -88],
            [544, -741, 462, -151, -142, -354, 88, 163],
            [121, 524, 149, 213, 1140, -645, -105, 105],
            [-218, -256, -748, 548, 517, 436, -603, -347],
            [786, -261, -406, 563, -579, -70, -239, -860],
        ];
        assert_eq!(row_stage(&f), g_want);
        let f_want: [[i16; 8]; 8] = [
            [29, -32, -9, 11, 27, -14, 3, 4],
            [0, -10, 13, -14, -10, 20, 18, 13],
            [8, -2, 4, 15, 15, 29, -14, -17],
            [5, 8, 49, -18, 31, -2, -1, 19],
            [26, -8, 14, -6, -21, 1, -11, -14],
            [-13, 24, -24, -2, 22, 7, -16, 0],
            [19, 31, -17, -3, 8, -18, 9, -3],
            [22, -16, -20, -11, -20, 9, 17, 5],
        ];
        assert_eq!(variant_rc_raw(&f), f_want);
    }

    #[test]
    fn lane_evaluation_equals_the_printed_equations() {
        // Dense / sparse / DC-only / single-row / single-column blocks
        // over the legal range, the full 16-bit range and beyond it
        // (input saturation): the basis-vector row pass and the
        // all-rows column pass give the integers of the printed
        // equations with both rules applied.
        let mut state = 0x5EED_0451u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let rounds = if cfg!(miri) { 60 } else { 60_000 };
        for round in 0..rounds {
            let mut block = [[0i32; 8]; 8];
            let span: u32 = match (round / 7) % 4 {
                0 => 61,
                1 => 4097,
                2 => 65536,
                _ => 200_001,
            };
            let draw = |next: &mut dyn FnMut() -> u32| (next() % span) as i32 - (span / 2) as i32;
            match round % 7 {
                0 => {
                    for cell in block.iter_mut().flatten() {
                        *cell = draw(&mut next);
                    }
                }
                1 => {
                    for cell in block.iter_mut().flatten() {
                        if next() % 6 == 0 {
                            *cell = draw(&mut next);
                        }
                    }
                }
                2 => block[0][0] = draw(&mut next),
                3 => {
                    let v = (next() % 8) as usize;
                    for cell in block[v].iter_mut() {
                        *cell = draw(&mut next);
                    }
                }
                4 => {
                    let u = (next() % 8) as usize;
                    for row in block.iter_mut() {
                        row[u] = draw(&mut next);
                    }
                }
                5 => {
                    // Low rows only: the three- and four-row reads.
                    let rows = 3 + (next() % 2) as usize;
                    for row in block.iter_mut().take(rows) {
                        for cell in row.iter_mut() {
                            if next() % 3 == 0 {
                                *cell = draw(&mut next);
                            }
                        }
                    }
                }
                _ => {
                    // Around the empty-row case.
                    block[0][0] = -4;
                    let v = (next() % 8) as usize;
                    block[v][0] += (next() % 9) as i32 - 4;
                    if next() % 2 == 0 {
                        let (v, u) = ((next() % 8) as usize, (next() % 8) as usize);
                        block[v][u] += draw(&mut next);
                    }
                }
            }
            let (g, f) = printed_equations(&block);
            assert_eq!(row_stage(&block), g, "round {round}: row stage");
            assert_eq!(variant_rc_raw(&block), f, "round {round}: output");
        }
    }
}

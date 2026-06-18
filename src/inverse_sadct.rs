//! Inverse Shape-Adaptive DCT (SA-DCT) — Annex A §A.3.2, steps I-S2..I-S5.
//!
//! When a block belongs to a non-rectangular VOP with `sadct_disable == 0`
//! and `opaque_pels < 64`, the textbook 8×8 inverse DCT (Annex A §A.1) is
//! replaced by the **inverse SA-DCT** (§7.3.5, Table 7-2). The SA-DCT
//! reconverts the inverse-quantised coefficients `F[v][u]` — laid out in
//! the `PQF[v][u]` packing produced by the §7.4.2 modified inverse scan
//! (see [`crate::scan`]) — back into the decoded texture `f[y][x]`, using
//! only the opaque samples described by the decoded binary shape
//! `f_shape[y][x]`.
//!
//! The inverse SA-DCT is a *separable, shape-adaptive* transform built
//! from variable-length one-dimensional inverse DCTs:
//!
//! * **I-S1** derives the auxiliary shape parameters `coeff_width[v]`,
//!   `shift_shape[y][x]` and `pels_height[x]` from `f_shape[y][x]`. The
//!   `coeff_width[]` / `opaque_pels` halves already live in
//!   [`crate::scan::ShapeParams`] (they feed the modified inverse scan);
//!   this module recomputes the full set locally from the same
//!   `f_shape[y][x]` so that the transform has a single source of truth.
//! * **I-S2** runs, for every row `v` with `coeff_width[v] != 0`, an
//!   `coeff_width[v]`-point inverse DCT over the first `coeff_width[v]`
//!   coefficients of `F[v][u]`, producing `shift_intermediate[v][x]`.
//! * **I-S3** re-shifts each row of `shift_intermediate[v][·]` from the
//!   left-packed position back to the original column positions defined
//!   by `shift_shape[v][x]`, yielding `F_intermediate[v][x]`.
//! * **I-S4** runs, for every column `x` with `pels_height[x] != 0`, a
//!   `pels_height[x]`-point inverse DCT over the first `pels_height[x]`
//!   intermediate coefficients `F_intermediate[v][x]`, producing
//!   `shift_texture[y][x]`.
//! * **I-S5** re-shifts each column of `shift_texture[·][x]` from the
//!   top-packed position back to the original row positions defined by
//!   `f_shape[y][x]`, yielding the decoded texture `f[y][x]`.
//!
//! Per Annex A NOTE 3 the entire transform runs in floating point; the
//! input coefficients are 12-bit integers in `[-2048, 2047]` and the
//! output is rounded to the nearest integer (half-away-from-zero,
//! matching §4.1 and [`crate::idct::idct_8x8`]). Transparent positions
//! (`f_shape[y][x] == 0`) carry no reconstructed texture and are left at
//! the caller-supplied fill (this module returns `0` there).
//!
//! ## Spec transcription notes
//!
//! The ISO/IEC 14496-2:2004 listing of I-S3 / I-S5 contains two obvious
//! typos that this module corrects to the unambiguous intent:
//!
//! * I-S3 reads `…=shift_intermediate[v][coff_count]`; `coff_count` is
//!   the loop's `coeff_count`.
//! * I-S5 reads `f[y][x]=shift_texture[x]` (missing the `[pels_count]`
//!   row index); the value re-shifted is `shift_texture[pels_count][x]`,
//!   the next top-packed pel of column `x`, exactly mirroring the I-S1 /
//!   forward-S1 vertical shift.
//!
//! ## References
//!
//! * Annex A §A.3.2 — *Definition of Inverse SA-DCT* (steps I-S1..I-S5).
//! * §7.3.5 / Table 7-2 — the transform-selection decision rule.
//! * §7.4.2 / [`crate::scan`] — the modified inverse scan that produces
//!   the `PQF[v][u]` (= `F[v][u]`) layout consumed here.

use crate::sample_padding::SamplePresence;

/// Shape parameters of an 8×8 block needed by the inverse SA-DCT, derived
/// from the decoded binary shape `f_shape[y][x]` per Annex A §A.3.2 step
/// I-S1.
///
/// This is the *full* I-S1 output: in addition to the `coeff_width[v]` /
/// `opaque_pels` pair exposed by [`crate::scan::ShapeParams`], it carries
/// `pels_height[x]` (opaque count per column) and `shift_shape[y][x]`
/// (the vertically top-packed binary mask) that steps I-S3..I-S5 need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SadctShape {
    /// `pels_height[x]` — opaque samples in column `x`, `0..=8`.
    pels_height: [u8; 8],
    /// `coeff_width[v]` — SA-DCT coefficients in row `v`, `0..=8`.
    coeff_width: [u8; 8],
    /// `shift_shape[y][x]` — the per-column top-packed opacity mask
    /// (`true` == opaque). Column `x` has its `pels_height[x]` opaque
    /// cells packed against `y = 0`.
    shift_shape: [[bool; 8]; 8],
}

impl SadctShape {
    /// I-S1: derive `pels_height[x]`, `coeff_width[v]` and
    /// `shift_shape[y][x]` from the decoded `f_shape[y][x]`.
    fn from_shape(f_shape: &[[SamplePresence; 8]; 8]) -> Self {
        let mut pels_height = [0u8; 8];
        let mut shift_shape = [[false; 8]; 8];
        // Per-column vertical shift: pack the opaque cells of each column
        // against the top into shift_shape, counting pels_height[x].
        for x in 0..8 {
            let mut pels_count = 0usize;
            for row in f_shape.iter() {
                if row[x].is_opaque() {
                    shift_shape[pels_count][x] = true;
                    pels_count += 1;
                }
            }
            pels_height[x] = pels_count as u8;
        }
        // coeff_width[v] = number of opaque cells in row v of shift_shape.
        let mut coeff_width = [0u8; 8];
        for (v, cw) in coeff_width.iter_mut().enumerate() {
            *cw = shift_shape[v].iter().filter(|&&c| c).count() as u8;
        }
        SadctShape {
            pels_height,
            coeff_width,
            shift_shape,
        }
    }
}

/// A shape-adaptive `n`-point inverse DCT (I-S2 / I-S4 kernel).
///
/// Computes, for the first `n` inputs `input[0..n]`:
///
/// ```text
///   out[k] = √(2/n) · Σ_{u=0}^{n-1} C(u) · cos(u·(k+0.5)·π/n) · input[u]
/// ```
///
/// with `C(0) = √0.5`, `C(u>0) = 1`, exactly mirroring the Annex A
/// §A.3.2 pseudo-C inner loops. `n` must be in `1..=8`; the result rows
/// `out[n..8]` are left at zero. This is the SA-DCT 1-D kernel: the same
/// orthonormal `√(2/N)` normalisation as the standard IDCT, but with `N`
/// taken from the shape parameter rather than fixed at 8.
#[inline]
fn sadct_idct_1d(input: &[f64; 8], n: usize) -> [f64; 8] {
    debug_assert!((1..=8).contains(&n));
    let scaling = (2.0_f64 / n as f64).sqrt();
    let mut out = [0.0f64; 8];
    for (k, slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f64;
        for (u, &coef) in input.iter().enumerate().take(n) {
            let c0 = if u == 0 { 0.5_f64.sqrt() } else { 1.0 };
            let dct_n =
                c0 * (u as f64 * (k as f64 + 0.5) * (core::f64::consts::PI / n as f64)).cos();
            acc += scaling * dct_n * coef;
        }
        *slot = acc;
    }
    out
}

/// Round a reconstructed sample to the nearest integer, half-away-from-zero
/// (§4.1), matching the rounding used by [`crate::idct::idct_8x8`].
#[inline]
fn round_sample(value: f64) -> i32 {
    if value >= 0.0 {
        (value + 0.5).floor() as i32
    } else {
        (value - 0.5).ceil() as i32
    }
}

/// Apply the inverse SA-DCT (Annex A §A.3.2 steps I-S2..I-S5) to a single
/// 8×8 block.
///
/// * `pqf` is the inverse-quantised coefficient block `F[v][u]` in the
///   `PQF[v][u]` packing produced by the §7.4.2 modified inverse scan:
///   row `v` carries `coeff_width[v]` meaningful coefficients left-packed
///   at `u < coeff_width[v]`, all other positions zero.
/// * `f_shape` is the decoded binary shape `f_shape[y][x]` of the block.
///
/// The return value is the decoded texture `f[y][x]` rounded to the
/// nearest integer; positions outside the shape (`f_shape[y][x]`
/// transparent) are returned as `0`.
///
/// Caller-applied saturation/clip (e.g. §7.4.5) and the prediction add
/// are *not* performed here — this routine returns the raw inverse
/// transform output, exactly like [`crate::idct::idct_8x8`] returns
/// before any reconstruction add.
pub fn inverse_sadct(pqf: &[[i32; 8]; 8], f_shape: &[[SamplePresence; 8]; 8]) -> [[i32; 8]; 8] {
    let shape = SadctShape::from_shape(f_shape);

    // I-S2: per-row coeff_width[v]-point inverse DCT over F[v][0..cw].
    // shift_intermediate[v][x] for x in 0..coeff_width[v].
    let mut shift_intermediate = [[0.0f64; 8]; 8];
    for v in 0..8 {
        let cw = shape.coeff_width[v] as usize;
        if cw == 0 {
            // Per the spec loop `(v<8) && (coeff_width[v]!=0)` the row
            // scan stops at the first empty row; because coeff_width is
            // monotonically non-increasing in v, all later rows are 0 too.
            break;
        }
        let mut row_in = [0.0f64; 8];
        for (u, slot) in row_in.iter_mut().enumerate().take(cw) {
            *slot = pqf[v][u] as f64;
        }
        shift_intermediate[v] = sadct_idct_1d(&row_in, cw);
    }

    // I-S3: re-shift each row of shift_intermediate from the left-packed
    // position back to the original column positions of shift_shape[v][x].
    let mut f_intermediate = [[0.0f64; 8]; 8];
    for (v, fi_row) in f_intermediate.iter_mut().enumerate() {
        let mut coeff_count = 0usize;
        for (x, slot) in fi_row.iter_mut().enumerate() {
            if shape.shift_shape[v][x] {
                *slot = shift_intermediate[v][coeff_count];
                coeff_count += 1;
            }
        }
    }

    // I-S4: per-column pels_height[x]-point inverse DCT over the first
    // pels_height[x] intermediate coefficients F_intermediate[v][x].
    let mut shift_texture = [[0.0f64; 8]; 8];
    for x in 0..8 {
        let ph = shape.pels_height[x] as usize;
        if ph == 0 {
            continue;
        }
        let mut col_in = [0.0f64; 8];
        for (v, slot) in col_in.iter_mut().enumerate().take(ph) {
            *slot = f_intermediate[v][x];
        }
        let col_out = sadct_idct_1d(&col_in, ph);
        for (y, &val) in col_out.iter().enumerate().take(ph) {
            shift_texture[y][x] = val;
        }
    }

    // I-S5: re-shift each column of shift_texture from the top-packed
    // position back to the original row positions of f_shape[y][x].
    let mut f = [[0i32; 8]; 8];
    for x in 0..8 {
        let mut pels_count = 0usize;
        for y in 0..8 {
            if f_shape[y][x].is_opaque() {
                f[y][x] = round_sample(shift_texture[pels_count][x]);
                pels_count += 1;
            }
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::idct_8x8;

    /// Build an `f_shape[y][x]` from a `bool` opacity grid.
    fn shape(rows: [[bool; 8]; 8]) -> [[SamplePresence; 8]; 8] {
        let mut s = [[SamplePresence::Transparent; 8]; 8];
        for (y, row) in rows.iter().enumerate() {
            for (x, &op) in row.iter().enumerate() {
                if op {
                    s[y][x] = SamplePresence::Opaque;
                }
            }
        }
        s
    }

    const FULL: [[bool; 8]; 8] = [[true; 8]; 8];

    /// A fully-opaque block has coeff_width[v] = pels_height[x] = 8 and
    /// shift_shape == f_shape, so I-S1 degenerates to identity.
    #[test]
    fn full_shape_params_are_eight() {
        let s = SadctShape::from_shape(&shape(FULL));
        assert_eq!(s.coeff_width, [8u8; 8]);
        assert_eq!(s.pels_height, [8u8; 8]);
        for row in s.shift_shape.iter() {
            assert_eq!(*row, [true; 8]);
        }
    }

    /// For a fully-opaque block the inverse SA-DCT reduces to the standard
    /// 8×8 inverse DCT (Annex A §A.1): the shape-adaptive 1-D kernel with
    /// N = 8 is exactly the orthonormal IDCT kernel, and the shifts are
    /// identity. We compare against the saturation-free reference by using
    /// a large `bits_per_pixel` so `idct_8x8`'s clamp never triggers.
    #[test]
    fn full_block_matches_standard_idct() {
        // A varied coefficient block (DC + a few AC terms).
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 512;
        pqf[0][1] = -130;
        pqf[1][0] = 64;
        pqf[2][3] = -40;
        pqf[5][5] = 17;
        pqf[7][7] = -3;

        let sadct = inverse_sadct(&pqf, &shape(FULL));
        // bits_per_pixel large enough that the §7.4.5 clamp is inert for
        // these magnitudes (range ±2^24).
        let idct = idct_8x8(&pqf, 24);
        assert_eq!(sadct, idct, "full-shape SA-DCT must equal the 8×8 IDCT");
    }

    /// A DC-only fully-opaque block produces a constant `round(DC/8)`
    /// everywhere — the orthonormal DC term is `DC·√0.5·√0.5·(2/8)·... `
    /// which collapses to `DC/8` for the 8-point kernel applied twice.
    #[test]
    fn full_block_dc_only_is_flat() {
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 800;
        let out = inverse_sadct(&pqf, &shape(FULL));
        let expected = round_sample(800.0 / 8.0);
        for row in out.iter() {
            for &v in row.iter() {
                assert_eq!(v, expected);
            }
        }
    }

    /// A single fully-opaque column (pels_height = 8 in one column, 0
    /// elsewhere) has coeff_width[v] = 1 for all v. A DC-only coefficient
    /// then reconstructs a flat column: the 1-point row IDCT is identity
    /// (scaling √2 · √0.5 = 1), and the 8-point column IDCT of a single
    /// DC gives DC/√8 per the orthonormal kernel.
    #[test]
    fn single_opaque_column_dc() {
        let mut rows = [[false; 8]; 8];
        for r in rows.iter_mut() {
            r[3] = true; // column 3 fully opaque
        }
        let s = SadctShape::from_shape(&shape(rows));
        assert_eq!(s.pels_height, [0, 0, 0, 8, 0, 0, 0, 0]);
        assert_eq!(s.coeff_width, [1u8; 8]);

        let mut pqf = [[0i32; 8]; 8];
        // I-S3 maps the single row coefficient to column 3, so the column
        // DC lives at F[0][3] after the modified-scan packing puts the
        // row-0 coefficient at u=0 of row 0.
        pqf[0][0] = 100;
        let out = inverse_sadct(&pqf, &shape(rows));

        // Row 1-D IDCT with N=1: out = √2·√0.5·100 = 100 placed at column 3
        // of every intermediate row 0..? Only row v=0 has a coefficient.
        // Column 1-D IDCT with N=8 on a single DC at v=0:
        //   f[y] = √(2/8)·√0.5·100 = 100/√8 each y.
        let expected = round_sample(100.0 / 8.0_f64.sqrt());
        for (y, row) in out.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                if x == 3 {
                    assert_eq!(val, expected, "opaque column at y={y}");
                } else {
                    assert_eq!(val, 0, "transparent at ({y},{x})");
                }
            }
        }
    }

    /// A single fully-opaque row reconstructs a flat row from a DC term:
    /// pels_height[x] = 1 in eight columns, coeff_width[0] = 8.
    #[test]
    fn single_opaque_row_dc() {
        let mut rows = [[false; 8]; 8];
        rows[2] = [true; 8]; // row 2 fully opaque
        let s = SadctShape::from_shape(&shape(rows));
        assert_eq!(s.pels_height, [1u8; 8]);
        assert_eq!(s.coeff_width, [8, 0, 0, 0, 0, 0, 0, 0]);

        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 800; // DC of the single coded row
        let out = inverse_sadct(&pqf, &shape(rows));

        // Row 1-D IDCT N=8 of a single DC → 800/√8 per column at row v=0.
        // I-S3 leaves them at their columns (shift_shape row0 all opaque).
        // Column 1-D IDCT N=1 (identity) → 800/√8 in row 2 of each column.
        let expected = round_sample(800.0 / 8.0_f64.sqrt());
        for (y, row) in out.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                if y == 2 {
                    assert_eq!(val, expected, "opaque row at x={x}");
                } else {
                    assert_eq!(val, 0, "transparent at ({y},{x})");
                }
            }
        }
    }

    /// Transparent positions are always returned as zero regardless of the
    /// coefficient block.
    #[test]
    fn transparent_positions_are_zero() {
        // A 4×4 opaque region in the top-left corner.
        let mut rows = [[false; 8]; 8];
        for row in rows.iter_mut().take(4) {
            for cell in row.iter_mut().take(4) {
                *cell = true;
            }
        }
        let f = shape(rows);
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 333;
        pqf[1][1] = -47;
        let out = inverse_sadct(&pqf, &f);
        for (y, row) in out.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                if !(y < 4 && x < 4) {
                    assert_eq!(val, 0, "outside-shape sample at ({y},{x})");
                }
            }
        }
    }

    /// A non-rectangular (staircase) shape must still satisfy the I-S1
    /// invariants: coeff_width is the column-monotone transpose of the
    /// pels_height histogram, and Σcoeff_width == Σpels_height == opaque.
    #[test]
    fn staircase_shape_param_invariants() {
        // Column x has x+1 opaque pels (a right triangle): cell (y,x) is
        // opaque iff y <= x.
        let mut rows = [[false; 8]; 8];
        for (y, row) in rows.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                if y <= x {
                    *cell = true;
                }
            }
        }
        let s = SadctShape::from_shape(&shape(rows));
        assert_eq!(s.pels_height, [1, 2, 3, 4, 5, 6, 7, 8]);
        // coeff_width[v] = #{x : pels_height[x] > v} = 8 - v.
        assert_eq!(s.coeff_width, [8, 7, 6, 5, 4, 3, 2, 1]);
        let total_cw: u16 = s.coeff_width.iter().map(|&c| c as u16).sum();
        let total_ph: u16 = s.pels_height.iter().map(|&c| c as u16).sum();
        assert_eq!(total_cw, total_ph);
        assert_eq!(total_cw, 36);
    }

    /// The inverse SA-DCT of an all-zero coefficient block is all-zero,
    /// for any shape.
    #[test]
    fn zero_block_is_zero() {
        let mut rows = [[false; 8]; 8];
        for (y, row) in rows.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate().take(5) {
                if y < x + 2 {
                    *cell = true;
                }
            }
        }
        let out = inverse_sadct(&[[0i32; 8]; 8], &shape(rows));
        assert_eq!(out, [[0i32; 8]; 8]);
    }
}

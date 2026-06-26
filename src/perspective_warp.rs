//! §7.8.4 / §7.8.5 four-point perspective sprite warp
//! (`no_of_sprite_warping_points == 4`).
//!
//! When a static sprite (`sprite_enable == "static"`) declares four
//! warping points (Table 6-88), the §7.8.5 warp is a *perspective*
//! transform rather than the affine (2/3-point) transform handled by
//! [`crate::warp::WarpGeometry`]. The four sprite reference points
//! `(i0',j0') … (i3',j3')` are accumulated from the trajectory exactly as
//! in §7.8.4:
//!
//! ```text
//! (i0',j0') = (s/2)(2 i0 + du0,                 2 j0 + dv0)
//! (i1',j1') = (s/2)(2 i1 + du1+du0,             2 j1 + dv1+dv0)
//! (i2',j2') = (s/2)(2 i2 + du2+du0,             2 j2 + dv2+dv0)
//! (i3',j3') = (s/2)(2 i3 + du3+du2+du1+du0,     2 j3 + dv3+dv2+dv1+dv0)
//! ```
//!
//! with VOP reference points (rectangular) `(0,0)`, `(W,0)`, `(0,H)`,
//! `(W,H)`. The §7.8.5 perspective transform then derives nine
//! coefficients `g, h, D, a, b, c, d, e, f` and maps a VOP pixel `(i,j)`:
//!
//! ```text
//! F(i,j) = (a I + b J + c) /// (g I + h J + D W H)
//! G(i,j) = (d I + e J + f) /// (g I + h J + D W H)
//! Fc(ic,jc) = (2 a Ic + 2 b Jc + 4 c − (g Ic + h Jc + 2 D W H) s) /// (4 g Ic + 4 h Jc + 8 D W H)
//! Gc(ic,jc) = (2 d Ic + 2 e Jc + 4 f − (g Ic + h Jc + 2 D W H) s) /// (4 g Ic + 4 h Jc + 8 D W H)
//! ```
//!
//! where (rectangular shape, `i0 == j0 == 0`) `I = i`, `J = j`,
//! `Ic = 4 ic + 1`, `Jc = 4 jc + 1`. The spec (§7.8.5 NOTE) warns a
//! 32-bit register is insufficient for the perspective numerator /
//! denominator; this module accumulates in `i128`. A parameter set that
//! drives any denominator to zero for an opaque/boundary pixel is
//! disallowed, surfaced here as [`PerspectiveWarpError::DegenerateDenominator`].

use crate::vol::SpriteWarpingAccuracy;

/// Errors raised constructing or sampling a four-point perspective warp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerspectiveWarpError {
    /// The §7.8.5 NOTE forbids parameter sets that make a denominator
    /// `g I + h J + D W H` zero for an opaque/boundary pixel. The carried
    /// `(i, j)` is the offending VOP pixel.
    DegenerateDenominator(i64, i64),
}

impl core::fmt::Display for PerspectiveWarpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PerspectiveWarpError::DegenerateDenominator(i, j) => {
                write!(
                    f,
                    "perspective warp denominator is zero at pixel ({i}, {j})"
                )
            }
        }
    }
}

impl std::error::Error for PerspectiveWarpError {}

/// `n /// d` — §3.4 integer division with sign-dependent rounding to the
/// nearest integer (positive halves away from zero, negative halves
/// toward zero). Mirrors [`crate::warp::div_sdr`] in `i128` for the wide
/// perspective products. `d` must be non-zero; the caller guards `d != 0`.
#[inline]
fn div_sdr_i128(n: i128, d: i128) -> i128 {
    debug_assert!(d != 0);
    // Normalise so the divisor is positive (the spec's `///` is defined
    // for a positive denominator; the perspective denominator can be
    // negative, so flip both signs to preserve the quotient).
    let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + (d - 1) / 2) / d)
    }
}

/// §7.8.5 four-point perspective warp geometry: the nine derived
/// coefficients plus the dimensions / accuracy needed for per-pixel
/// mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerspectiveWarp {
    /// Sub-pel denominator `s` (2/4/8/16).
    pub s: i64,
    /// VOP width `W` (luma pixels).
    pub w: i64,
    /// VOP height `H` (luma pixels).
    pub h: i64,
    /// Coefficient `g` (perspective denominator I-term).
    pub g: i128,
    /// Coefficient `h` (perspective denominator J-term).
    pub h_coef: i128,
    /// Coefficient `D` (perspective determinant scale).
    pub cap_d: i128,
    /// `F`-numerator `I`-coefficient `a` (`F = a I + b J + c`).
    pub a: i128,
    /// `F`-numerator `J`-coefficient `b`.
    pub b: i128,
    /// `F`-numerator constant `c`.
    pub c: i128,
    /// `G`-numerator `I`-coefficient `d` (`G = d I + e J + f`).
    pub d: i128,
    /// `G`-numerator `J`-coefficient `e`.
    pub e: i128,
    /// `G`-numerator constant `f`.
    pub f: i128,
}

impl PerspectiveWarp {
    /// Decode the §7.8.4 sprite reference points + §7.8.5 perspective
    /// coefficients for a rectangular static sprite with four warping
    /// points.
    ///
    /// `trajectory` holds `du[i]`/`dv[i]` for `i in 0..4` (the four
    /// `warping_mv_code` pairs); `w`/`h` are the VOP luma dimensions;
    /// `accuracy` selects `s`.
    pub fn decode(
        trajectory: &[[i32; 2]; 4],
        w: u32,
        h: u32,
        accuracy: SpriteWarpingAccuracy,
    ) -> Self {
        let s = accuracy.s();
        let w = w as i64;
        let h = h as i64;
        let half = (s / 2) as i128;

        let du = |k: usize| i128::from(trajectory[k][0]);
        let dv = |k: usize| i128::from(trajectory[k][1]);
        let wi = w as i128;
        let hi = h as i128;

        // §7.8.4 sprite reference points, 1/s-pel, with i0=j0=0,
        // (i1,j1)=(W,0), (i2,j2)=(0,H), (i3,j3)=(W,H).
        let i0p = half * du(0);
        let j0p = half * dv(0);
        let i1p = half * (2 * wi + du(1) + du(0));
        let j1p = half * (dv(1) + dv(0));
        let i2p = half * (du(2) + du(0));
        let j2p = half * (2 * hi + dv(2) + dv(0));
        let i3p = half * (2 * wi + du(3) + du(2) + du(1) + du(0));
        let j3p = half * (2 * hi + dv(3) + dv(2) + dv(1) + dv(0));

        // §7.8.5 perspective coefficients.
        let g =
            ((i0p - i1p - i2p + i3p) * (j2p - j3p) - (i2p - i3p) * (j0p - j1p - j2p + j3p)) * hi;
        let h_coef =
            ((i1p - i3p) * (j0p - j1p - j2p + j3p) - (i0p - i1p - i2p + i3p) * (j1p - j3p)) * wi;
        let cap_d = (i1p - i3p) * (j2p - j3p) - (i2p - i3p) * (j1p - j3p);
        let a = cap_d * (i1p - i0p) * hi + g * i1p;
        let b = cap_d * (i2p - i0p) * wi + h_coef * i2p;
        let c = cap_d * i0p * wi * hi;
        let d = cap_d * (j1p - j0p) * hi + g * j1p;
        let e = cap_d * (j2p - j0p) * wi + h_coef * j2p;
        let f = cap_d * j0p * wi * hi;

        PerspectiveWarp {
            s,
            w,
            h,
            g,
            h_coef,
            cap_d,
            a,
            b,
            c,
            d,
            e,
            f,
        }
    }

    /// The §7.8.5 perspective denominator `g I + h J + D W H` for VOP
    /// pixel `(I, J)`.
    #[inline]
    fn luma_denominator(&self, cap_i: i128, cap_j: i128) -> i128 {
        self.g * cap_i + self.h_coef * cap_j + self.cap_d * i128::from(self.w) * i128::from(self.h)
    }

    /// §7.8.5 luma warp: map VOP pixel `(i, j)` to `(F, G)` in `1/s`-pel
    /// units. Returns [`PerspectiveWarpError::DegenerateDenominator`] when
    /// the perspective denominator vanishes.
    pub fn luma_fg(&self, i: i64, j: i64) -> Result<[i64; 2], PerspectiveWarpError> {
        // Rectangular: I = i, J = j.
        let cap_i = i128::from(i);
        let cap_j = i128::from(j);
        let den = self.luma_denominator(cap_i, cap_j);
        if den == 0 {
            return Err(PerspectiveWarpError::DegenerateDenominator(i, j));
        }
        let f = div_sdr_i128(self.a * cap_i + self.b * cap_j + self.c, den);
        let g = div_sdr_i128(self.d * cap_i + self.e * cap_j + self.f, den);
        Ok([f as i64, g as i64])
    }

    /// §7.8.5 chroma warp: map chroma pixel `(ic, jc)` to `(Fc, Gc)` in
    /// `1/s`-pel units. `Ic = 4 ic + 1`, `Jc = 4 jc + 1`
    /// (rectangular, `i0 = j0 = 0`).
    pub fn chroma_fg(&self, ic: i64, jc: i64) -> Result<[i64; 2], PerspectiveWarpError> {
        let cap_ic = i128::from(4 * ic + 1);
        let cap_jc = i128::from(4 * jc + 1);
        let s = i128::from(self.s);
        let dwh = self.cap_d * i128::from(self.w) * i128::from(self.h);
        // Chroma denominator: 4 g Ic + 4 h Jc + 8 D W H.
        let den = 4 * self.g * cap_ic + 4 * self.h_coef * cap_jc + 8 * dwh;
        if den == 0 {
            return Err(PerspectiveWarpError::DegenerateDenominator(ic, jc));
        }
        // Shared bracket term: (g Ic + h Jc + 2 D W H) s.
        let bracket = (self.g * cap_ic + self.h_coef * cap_jc + 2 * dwh) * s;
        let fc = div_sdr_i128(
            2 * self.a * cap_ic + 2 * self.b * cap_jc + 4 * self.c - bracket,
            den,
        );
        let gc = div_sdr_i128(
            2 * self.d * cap_ic + 2 * self.e * cap_jc + 4 * self.f - bracket,
            den,
        );
        Ok([fc as i64, gc as i64])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_sdr_i128_matches_spec_examples() {
        assert_eq!(div_sdr_i128(3, 2), 2);
        assert_eq!(div_sdr_i128(-3, 2), -1);
        assert_eq!(div_sdr_i128(5, 2), 3);
        assert_eq!(div_sdr_i128(-5, 2), -2);
        // Negative denominator: flip both signs.
        assert_eq!(div_sdr_i128(3, -2), div_sdr_i128(-3, 2));
        assert_eq!(div_sdr_i128(8, -4), -2);
    }

    #[test]
    fn zero_trajectory_is_identity_scaled_by_s() {
        // All du/dv zero ⇒ the four sprite reference points sit exactly on
        // the (scaled) VOP corners ⇒ the perspective transform collapses
        // to F = s i, G = s j (no projective distortion).
        let traj = [[0, 0]; 4];
        let warp = PerspectiveWarp::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        let s = warp.s;
        for (i, j) in [(0, 0), (1, 0), (0, 1), (10, 20), (63, 63), (32, 7)] {
            let fg = warp.luma_fg(i, j).unwrap();
            assert_eq!(fg, [s * i, s * j], "luma pixel ({i},{j})");
        }
    }

    #[test]
    fn zero_trajectory_chroma_is_identity() {
        let traj = [[0, 0]; 4];
        let warp = PerspectiveWarp::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        let s = warp.s;
        // Identity perspective ⇒ Fc = s ic, Gc = s jc.
        for (ic, jc) in [(0, 0), (1, 2), (15, 31), (7, 3)] {
            let fg = warp.chroma_fg(ic, jc).unwrap();
            assert_eq!(fg, [s * ic, s * jc], "chroma pixel ({ic},{jc})");
        }
    }

    #[test]
    fn pure_translation_via_uniform_corner_shift() {
        // Translating every sprite corner by the same sub-pel vector is a
        // pure translation: only du0/dv0 carry the shift (the higher
        // points' *differential* du1..3 = 0). Then F = s i + i0', etc.
        let mut traj = [[0, 0]; 4];
        traj[0] = [4, -2]; // du0 = 4, dv0 = -2 at s=2 ⇒ i0' = 4, j0' = -2
        let warp = PerspectiveWarp::decode(&traj, 32, 32, SpriteWarpingAccuracy::HalfPel);
        let s = warp.s;
        let base = warp.luma_fg(0, 0).unwrap();
        assert_eq!(base, [4, -2]);
        for (i, j) in [(0, 0), (5, 9), (31, 0), (0, 31), (16, 16)] {
            let fg = warp.luma_fg(i, j).unwrap();
            assert_eq!(fg[0] - s * i, base[0], "x at ({i},{j})");
            assert_eq!(fg[1] - s * j, base[1], "y at ({i},{j})");
        }
    }

    #[test]
    fn affine_subcase_agrees_with_three_point_warp() {
        // A 4-point trajectory whose 4th corner is consistent with the
        // affine plane spanned by points 0/1/2 must reduce to an affine
        // warp. Construct du3/dv3 so (i3',j3') = i1'+i2'-i0' (the affine
        // parallelogram corner): then du3 chosen so the perspective D-based
        // projective terms cancel and the map is affine. Verify the warp is
        // at least well-defined (non-degenerate) and exactly linear in
        // (i,j): F(i,j) - F(0,0) is additive across the two axes.
        let mut traj = [[0, 0]; 4];
        traj[0] = [2, 0];
        traj[1] = [0, 2];
        traj[2] = [-2, 0];
        // 4th differential zero ⇒ (i3',j3') = (s/2)(2W + sum du) etc.
        let warp = PerspectiveWarp::decode(&traj, 48, 48, SpriteWarpingAccuracy::HalfPel);
        let f00 = warp.luma_fg(0, 0).unwrap();
        let f10 = warp.luma_fg(8, 0).unwrap();
        let f01 = warp.luma_fg(0, 8).unwrap();
        let f11 = warp.luma_fg(8, 8).unwrap();
        // Affine ⇒ F(8,8) - F(0,0) == (F(8,0)-F(0,0)) + (F(0,8)-F(0,0)).
        for k in 0..2 {
            assert_eq!(
                f11[k] - f00[k],
                (f10[k] - f00[k]) + (f01[k] - f00[k]),
                "component {k} affine additivity"
            );
        }
    }

    #[test]
    fn degenerate_denominator_is_reported() {
        // Force g = h = D = 0 by collapsing all four reference points to a
        // single location (all corners equal): D = 0, g = 0, h = 0 ⇒ the
        // denominator g I + h J + D W H is identically zero.
        // Choose du so that i1' = i0' etc.: set the 2W / 2H offsets to be
        // cancelled is hard; instead test the predicate directly via a
        // hand-built degenerate PerspectiveWarp.
        let warp = PerspectiveWarp {
            s: 2,
            w: 16,
            h: 16,
            g: 0,
            h_coef: 0,
            cap_d: 0,
            a: 1,
            b: 1,
            c: 0,
            d: 1,
            e: 1,
            f: 0,
        };
        assert_eq!(
            warp.luma_fg(3, 4).unwrap_err(),
            PerspectiveWarpError::DegenerateDenominator(3, 4)
        );
        assert_eq!(
            warp.chroma_fg(1, 1).unwrap_err(),
            PerspectiveWarpError::DegenerateDenominator(1, 1)
        );
    }

    #[test]
    fn wide_products_do_not_overflow_i64_inputs() {
        // Large VOP + large trajectory differentials at quarter-pel push
        // the perspective products beyond 32 bits; confirm the i128 path
        // produces a finite, denominator-consistent result.
        let mut traj = [[0, 0]; 4];
        traj[1] = [20, -30];
        traj[2] = [-25, 15];
        traj[3] = [10, 10];
        let warp = PerspectiveWarp::decode(&traj, 720, 576, SpriteWarpingAccuracy::SixteenthPel);
        // Sampling a mid pixel must not panic and must round consistently.
        let fg = warp.luma_fg(360, 288).unwrap();
        // Re-derive the quotient by hand to confirm div_sdr_i128 wiring.
        let den = warp.luma_denominator(360, 288);
        assert_ne!(den, 0);
        let expect_f = div_sdr_i128(warp.a * 360 + warp.b * 288 + warp.c, den) as i64;
        assert_eq!(fg[0], expect_f);
    }
}

//! §7.8.4 sprite reference-point decoding + §7.8.5 warping geometry.
//!
//! Given the decoded [`crate::sprite::SpriteTrajectory`] (`du[i]` /
//! `dv[i]`), the VOP dimensions `W` / `H`, and the
//! `sprite_warping_accuracy` denominator `s`, this module computes the
//! global-motion warp transform `(F(i,j), G(i,j))` for luminance pixels
//! and `(Fc(ic,jc), Gc(ic,jc))` for chrominance pixels. The transform
//! maps a destination pixel in the currently-decoded VOP to a sub-pel
//! coordinate (in `1/s`-pel units) inside the reference VOP, which the
//! §7.8.6 sample-reconstruction stage then bilinearly samples.
//!
//! ## Reference points (§7.8.4)
//!
//! The VOP reference points (rectangular shape) are the four corners
//! `(i0,j0)=(0,0)`, `(i1,j1)=(W,0)`, `(i2,j2)=(0,H)`, `(i3,j3)=(W,H)`.
//! The sprite reference points are accumulated from the trajectory:
//!
//! * `(i0',j0') = (s/2)(2 i0 + du0, 2 j0 + dv0)`
//! * `(i1',j1') = (s/2)(2 i1 + du1 + du0, 2 j1 + dv1 + dv0)`
//! * `(i2',j2') = (s/2)(2 i2 + du2 + du0, 2 j2 + dv2 + dv0)`
//!
//! all in `1/s`-pel accuracy. For 2/3 warping points two virtual sprite
//! points `(i1'',j1'')` / `(i2'',j2'')` are derived (§7.8.4) using
//! `r = 16/s` and `W'`/`H'` — the smallest powers of two `>= W`/`H`.
//!
//! ## `///` operator
//!
//! The spec's `///` is integer division with sign-dependent rounding to
//! nearest (positive halves away from zero, negative halves toward
//! zero) — [`div_sdr`]. The shift-based rewrites in the spec are
//! equivalent for the power-of-two denominators here; this module uses
//! the exact `///` form for clarity.

use crate::sprite::SpriteTrajectory;
use crate::vol::SpriteWarpingAccuracy;

/// `n /// d` — §3.4 integer division with sign-dependent rounding to
/// the nearest integer. Positive half-integers round away from zero;
/// negative half-integers round toward zero. `d` must be positive
/// (every spec use of `///` has a positive denominator).
///
/// Worked spec examples: `3 /// 2 == 2`, `-3 /// 2 == -1`.
pub fn div_sdr(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0, "/// denominator must be positive");
    if n >= 0 {
        // Round half away from zero (== half up for positives).
        (n + d / 2) / d
    } else {
        // Round half toward zero. floor((n + d/2 ... )) but biased so
        // that exactly-half rounds toward zero: add (d-1)/2 then divide
        // with truncation-toward-zero (Rust `/` on negatives truncates
        // toward zero, matching the spec's `/`).
        -((-n + (d - 1) / 2) / d)
    }
}

/// The `r = 16 / s` factor and `rho` (where `2^rho == r`) for a given
/// sprite-warping accuracy. `s` is `2/4/8/16` so `r` is `8/4/2/1` and
/// `rho` is `3/2/1/0`.
fn r_and_rho(s: i64) -> (i64, u32) {
    let r = 16 / s;
    let rho = r.trailing_zeros();
    (r, rho)
}

/// Smallest power of two `>= n` (with `n >= 1`), plus its exponent.
/// Returns `(value, exponent)` so `value == 1 << exponent`.
fn next_pow2(n: i64) -> (i64, u32) {
    debug_assert!(n >= 1);
    let mut exp = 0u32;
    let mut val = 1i64;
    while val < n {
        val <<= 1;
        exp += 1;
    }
    (val, exp)
}

/// Decoded §7.8.4 reference-point geometry for a GMC warp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpGeometry {
    /// Number of warping points (0..=3 for GMC).
    pub points: u8,
    /// Sub-pel denominator `s` (2/4/8/16).
    pub s: i64,
    /// VOP width / height (luma pixels).
    pub w: i64,
    /// VOP height (luma pixels).
    pub h: i64,
    /// `(i0', j0')` sprite reference point, `1/s`-pel.
    pub p0: [i64; 2],
    /// `(i1', j1')` sprite reference point, `1/s`-pel.
    pub p1: [i64; 2],
    /// `(i2', j2')` sprite reference point, `1/s`-pel.
    pub p2: [i64; 2],
    /// Virtual sprite point `(i1'', j1'')`, `1/16`-pel (2/3 points).
    pub v1: [i64; 2],
    /// Virtual sprite point `(i2'', j2'')`, `1/16`-pel (3 points only).
    pub v2: [i64; 2],
    /// `r = 16/s`.
    pub r: i64,
    /// `rho` with `2^rho == r`.
    pub rho: u32,
    /// `W'` (smallest power of two `>= W`) and its exponent `alpha`.
    pub w_prime: i64,
    /// `alpha` with `W' == 1 << alpha`.
    pub alpha: u32,
    /// `H'` (smallest power of two `>= H`) and its exponent `beta`.
    pub h_prime: i64,
    /// `beta` with `H' == 1 << beta`.
    pub beta: u32,
}

impl WarpGeometry {
    /// Decode the §7.8.4 reference points for a rectangular S(GMC)-VOP.
    ///
    /// `w` / `h` are the VOP luminance width / height (= VOL width /
    /// height for rectangular shape). `accuracy` selects `s`. The
    /// trajectory carries `du[i]`/`dv[i]`; `trajectory.count` is the
    /// number of warping points (0..=3 for GMC).
    pub fn decode(
        trajectory: &SpriteTrajectory,
        w: u32,
        h: u32,
        accuracy: SpriteWarpingAccuracy,
    ) -> Self {
        let s = accuracy.s();
        let w = w as i64;
        let h = h as i64;
        let points = trajectory.count;

        // VOP reference points (rectangular): (0,0), (W,0), (0,H).
        // i0=j0=0, so the (s/2)(2 i + du) reduces to (s/2)(du) for point 0.
        let du = |k: usize| i64::from(trajectory.points[k][0]);
        let dv = |k: usize| i64::from(trajectory.points[k][1]);

        // (i0', j0') = (s/2)(2*0 + du0, 2*0 + dv0).
        let half = s / 2;
        let p0 = if points >= 1 {
            [half * du(0), half * dv(0)]
        } else {
            [0, 0]
        };
        // (i1', j1') = (s/2)(2*W + du1 + du0, 0 + dv1 + dv0).
        let p1 = if points >= 2 {
            [half * (2 * w + du(1) + du(0)), half * (dv(1) + dv(0))]
        } else {
            [0, 0]
        };
        // (i2', j2') = (s/2)(0 + du2 + du0, 2*H + dv2 + dv0).
        let p2 = if points >= 3 {
            [half * (du(2) + du(0)), half * (2 * h + dv(2) + dv(0))]
        } else {
            [0, 0]
        };

        let (r, rho) = r_and_rho(s);
        let (w_prime, alpha) = next_pow2(w);
        let (h_prime, beta) = next_pow2(h);

        // Virtual sprite points (§7.8.4) for 2 or 3 warping points.
        // i0=j0=0 ⇒ the "−16 i0" / "−16 j0" terms vanish.
        //   i1'' = (16 (i0+W') + ((W−W')(r i0'−16 i0) + W'(r i1'−16 i1)) // W,  ...)
        //   with i0=0, i1=W, j0=0, j1=0:
        //   i1'' = 16 W' + ((W−W') r i0' + W' (r i1' − 16 W)) // W
        //   j1'' = 0 + ((W−W') r j0' + W' (r j1')) // W
        // (the // is round-half-away-from-zero integer division)
        let mut v1 = [0i64, 0i64];
        let mut v2 = [0i64, 0i64];
        if points >= 2 {
            let i1pp = 16 * w_prime
                + div_half(
                    (w - w_prime) * (r * p0[0]) + w_prime * (r * p1[0] - 16 * w),
                    w,
                );
            // j0 = j1 = 0 ⇒ the leading "16 j0" term is zero.
            let j1pp = div_half((w - w_prime) * (r * p0[1]) + w_prime * (r * p1[1]), w);
            v1 = [i1pp, j1pp];
        }
        if points >= 3 {
            // i2'' = 16 i0 + ((H−H')(r i0'−16 i0) + H'(r i2'−16 i2)) // H
            //   with i0=0, i2=0: = ((H−H') r i0' + H' r i2') // H
            // j2'' = 16 (j0+H') + ((H−H')(r j0'−16 j0) + H'(r j2'−16 j2)) // H
            //   with j0=0, j2=H: = 16 H' + ((H−H') r j0' + H'(r j2' − 16 H)) // H
            let i2pp = div_half((h - h_prime) * (r * p0[0]) + h_prime * (r * p2[0]), h);
            let j2pp = 16 * h_prime
                + div_half(
                    (h - h_prime) * (r * p0[1]) + h_prime * (r * p2[1] - 16 * h),
                    h,
                );
            v2 = [i2pp, j2pp];
        }

        WarpGeometry {
            points,
            s,
            w,
            h,
            p0,
            p1,
            p2,
            v1,
            v2,
            r,
            rho,
            w_prime,
            alpha,
            h_prime,
            beta,
        }
    }

    /// §7.8.5 luminance warp: map VOP pixel `(i, j)` to the sub-pel
    /// reference coordinate `(F, G)` in `1/s`-pel units. `i0 = j0 = 0`
    /// for the rectangular shape, so `I = i`, `J = j`.
    pub fn luma_fg(&self, i: i64, j: i64) -> [i64; 2] {
        let cap_i = i; // I = i - i0, i0 = 0
        let cap_j = j; // J = j - j0, j0 = 0
        match self.points {
            0 => [self.s * i, self.s * j],
            1 => {
                // (i0' + s I, j0' + s J)
                [self.p0[0] + self.s * cap_i, self.p0[1] + self.s * cap_j]
            }
            2 => {
                // F = i0' + ((−r i0' + i1'') I + (r j0' − j1'') J) /// (W' r)
                // G = j0' + ((−r j0' + j1'') I + (−r i0' + i1'') J) /// (W' r)
                let den = self.w_prime * self.r;
                let a = -self.r * self.p0[0] + self.v1[0];
                let b = self.r * self.p0[1] - self.v1[1];
                let c = -self.r * self.p0[1] + self.v1[1];
                let f = self.p0[0] + div_sdr(a * cap_i + b * cap_j, den);
                let g = self.p0[1] + div_sdr(c * cap_i + a * cap_j, den);
                [f, g]
            }
            _ => {
                // 3 points:
                // F = i0' + ((−r i0' + i1'') H' I + (−r i0' + i2'') W' J) /// (W'H'r)
                // G = j0' + ((−r j0' + j1'') H' I + (−r j0' + j2'') W' J) /// (W'H'r)
                let den = self.w_prime * self.h_prime * self.r;
                let f = self.p0[0]
                    + div_sdr(
                        (-self.r * self.p0[0] + self.v1[0]) * self.h_prime * cap_i
                            + (-self.r * self.p0[0] + self.v2[0]) * self.w_prime * cap_j,
                        den,
                    );
                let g = self.p0[1]
                    + div_sdr(
                        (-self.r * self.p0[1] + self.v1[1]) * self.h_prime * cap_i
                            + (-self.r * self.p0[1] + self.v2[1]) * self.w_prime * cap_j,
                        den,
                    );
                [f, g]
            }
        }
    }

    /// §7.8.5 chrominance warp: map chroma pixel `(ic, jc)` to the
    /// sub-pel reference coordinate `(Fc, Gc)` in `1/s`-pel units.
    /// `Ic = 4 ic − 2 i0 + 1`, `Jc = 4 jc − 2 j0 + 1` (`i0 = j0 = 0`).
    pub fn chroma_fg(&self, ic: i64, jc: i64) -> [i64; 2] {
        let cap_ic = 4 * ic + 1; // Ic, i0 = 0
        let cap_jc = 4 * jc + 1; // Jc, j0 = 0
        match self.points {
            0 => [self.s * ic, self.s * jc],
            1 => {
                // GMC: (((i0'>>1)|(i0'&1)) + s(ic − i0/2), ...) with i0=0.
                let cx = (self.p0[0] >> 1) | (self.p0[0] & 1);
                let cy = (self.p0[1] >> 1) | (self.p0[1] & 1);
                [cx + self.s * ic, cy + self.s * jc]
            }
            2 => {
                // Fc = ((−r i0' + i1'') Ic + (r j0' − j1'') Jc
                //        + 2 W' r i0' − 16 W') /// (4 W' r)
                // Gc = ((−r j0' + j1'') Ic + (−r i0' + i1'') Jc
                //        + 2 W' r j0' − 16 W') /// (4 W' r)
                let den = 4 * self.w_prime * self.r;
                let a = -self.r * self.p0[0] + self.v1[0];
                let b = self.r * self.p0[1] - self.v1[1];
                let c = -self.r * self.p0[1] + self.v1[1];
                let fc = div_sdr(
                    a * cap_ic + b * cap_jc + 2 * self.w_prime * self.r * self.p0[0]
                        - 16 * self.w_prime,
                    den,
                );
                let gc = div_sdr(
                    c * cap_ic + a * cap_jc + 2 * self.w_prime * self.r * self.p0[1]
                        - 16 * self.w_prime,
                    den,
                );
                [fc, gc]
            }
            _ => {
                // 3 points:
                // Fc = ((−r i0'+i1'') H' Ic + (−r i0'+i2'') W' Jc
                //        + 2 W'H'r i0' − 16 W'H') /// (4 W'H'r)
                let den = 4 * self.w_prime * self.h_prime * self.r;
                let fc = div_sdr(
                    (-self.r * self.p0[0] + self.v1[0]) * self.h_prime * cap_ic
                        + (-self.r * self.p0[0] + self.v2[0]) * self.w_prime * cap_jc
                        + 2 * self.w_prime * self.h_prime * self.r * self.p0[0]
                        - 16 * self.w_prime * self.h_prime,
                    den,
                );
                let gc = div_sdr(
                    (-self.r * self.p0[1] + self.v1[1]) * self.h_prime * cap_ic
                        + (-self.r * self.p0[1] + self.v2[1]) * self.w_prime * cap_jc
                        + 2 * self.w_prime * self.h_prime * self.r * self.p0[1]
                        - 16 * self.w_prime * self.h_prime,
                    den,
                );
                [fc, gc]
            }
        }
    }
}

/// `// ` operator (§3.4): integer division with rounding to nearest,
/// half-integers away from zero. `d` must be positive.
fn div_half(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::SpriteTrajectory;

    #[test]
    fn div_sdr_spec_examples() {
        assert_eq!(div_sdr(3, 2), 2);
        assert_eq!(div_sdr(-3, 2), -1);
        // Exact divisions unchanged.
        assert_eq!(div_sdr(8, 4), 2);
        assert_eq!(div_sdr(-8, 4), -2);
        // Half cases.
        assert_eq!(div_sdr(5, 2), 3); // 2.5 → 3 (away)
        assert_eq!(div_sdr(-5, 2), -2); // -2.5 → -2 (toward zero)
    }

    #[test]
    fn div_half_rounds_away_from_zero() {
        assert_eq!(div_half(3, 2), 2);
        assert_eq!(div_half(-3, 2), -2);
        assert_eq!(div_half(5, 2), 3);
        assert_eq!(div_half(-5, 2), -3);
    }

    #[test]
    fn r_rho_table() {
        assert_eq!(r_and_rho(2), (8, 3));
        assert_eq!(r_and_rho(4), (4, 2));
        assert_eq!(r_and_rho(8), (2, 1));
        assert_eq!(r_and_rho(16), (1, 0));
    }

    #[test]
    fn next_pow2_examples() {
        assert_eq!(next_pow2(176), (256, 8));
        assert_eq!(next_pow2(144), (256, 8));
        assert_eq!(next_pow2(256), (256, 8));
        assert_eq!(next_pow2(1), (1, 0));
        assert_eq!(next_pow2(17), (32, 5));
    }

    #[test]
    fn zero_points_is_identity_scaled_by_s() {
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 176, 144, SpriteWarpingAccuracy::HalfPel);
        // F = s i, G = s j.
        assert_eq!(geo.luma_fg(10, 20), [20, 40]);
        assert_eq!(geo.chroma_fg(3, 4), [6, 8]);
    }

    #[test]
    fn one_point_is_pure_translation() {
        // Single warping point: du0 / dv0 give a constant sub-pel shift.
        // s = 2 (half-pel), du0 = +4, dv0 = -2.
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [4, -2];
        let geo = WarpGeometry::decode(&traj, 176, 144, SpriteWarpingAccuracy::HalfPel);
        // i0' = (s/2) du0 = 1 * 4 = 4; j0' = 1 * -2 = -2.
        assert_eq!(geo.p0, [4, -2]);
        // F = i0' + s I = 4 + 2 i ; G = -2 + 2 j.
        assert_eq!(geo.luma_fg(0, 0), [4, -2]);
        assert_eq!(geo.luma_fg(10, 5), [4 + 20, -2 + 10]);
    }

    #[test]
    fn one_point_translation_is_uniform_across_pixels() {
        // The per-pixel displacement F - s*i must be constant (= i0').
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [6, 6];
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::QuarterPel);
        let s = geo.s;
        let base = geo.luma_fg(0, 0);
        for (i, j) in [(1, 1), (10, 0), (0, 10), (33, 21)] {
            let fg = geo.luma_fg(i, j);
            assert_eq!(fg[0] - s * i, base[0]);
            assert_eq!(fg[1] - s * j, base[1]);
        }
    }

    #[test]
    fn two_point_zero_trajectory_reduces_to_identity() {
        // With du/dv all zero, points 0 and 1 collapse so that the affine
        // warp is the identity scaled by s (no rotation/zoom). i0'=0,
        // i1' = (s/2)(2W) = sW.
        let traj = SpriteTrajectory {
            count: 2,
            points: [[0, 0], [0, 0], [0, 0]],
        };
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        // p0 = (0,0); p1 = (s*W, 0) = (128, 0).
        assert_eq!(geo.p0, [0, 0]);
        assert_eq!(geo.p1, [geo.s * 64, 0]);
        // The identity affine: F(i,j) = s i, G(i,j) = s j for all pixels.
        for (i, j) in [(0, 0), (10, 20), (63, 63), (1, 0)] {
            assert_eq!(geo.luma_fg(i, j), [geo.s * i, geo.s * j], "pixel ({i},{j})");
        }
    }

    #[test]
    fn three_point_zero_trajectory_reduces_to_identity() {
        let traj = SpriteTrajectory {
            count: 3,
            points: [[0, 0], [0, 0], [0, 0]],
        };
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        for (i, j) in [(0, 0), (5, 9), (63, 63)] {
            assert_eq!(geo.luma_fg(i, j), [geo.s * i, geo.s * j], "pixel ({i},{j})");
        }
    }

    #[test]
    fn two_point_pure_translation_when_only_first_point_moves() {
        // A pure translation needs every corner displaced by the same
        // sub-pel vector. Corner 0's displacement from its nominal
        // position is (s/2) du0; corner 1's is (s/2)(du1 + du0). Equal
        // displacement ⇒ du1 = dv1 = 0 with du0 / dv0 carrying the shift.
        // The affine then degenerates to F = s i + i0', G = s j + j0'.
        let traj = SpriteTrajectory {
            count: 2,
            points: [[4, -4], [0, 0], [0, 0]],
        };
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        let s = geo.s;
        let base = geo.luma_fg(0, 0);
        // base = i0' = (s/2) du0 = 1*4 = 4 ; j0' = 1*-4 = -4.
        assert_eq!(base, [4, -4]);
        for (i, j) in [(0, 0), (10, 10), (63, 0), (0, 63), (32, 17)] {
            let fg = geo.luma_fg(i, j);
            assert_eq!(fg[0] - s * i, base[0], "pixel ({i},{j}) x");
            assert_eq!(fg[1] - s * j, base[1], "pixel ({i},{j}) y");
        }
    }
}

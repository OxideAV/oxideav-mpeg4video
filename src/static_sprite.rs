//! §7.8.2 / §7.8.6 static-sprite reconstruction (`sprite_enable == "static"`).
//!
//! A basic static sprite (`low_latency_sprite_enable == 0`) is a
//! two-dimensional luminance / chrominance array decoded once (the
//! initial I-VOP, stored in *sprite memory* and never displayed) and
//! then warped, for every subsequent S-VOP, onto the visible VOP using
//! the §7.8.4 reference-point geometry, §7.8.5 sub-pel warp and the
//! §7.8.6 sample reconstruction. This module performs that warp-and-
//! sample stage for the rectangular case.
//!
//! ## Sprite-memory addressing (§7.8.2)
//!
//! The sprite luminance array is `sprite_width × sprite_height`; its
//! top-left sample lives at the §7.8.2 absolute coordinate
//! `(sprite_left_coordinate, sprite_top_coordinate)`. The §7.8.5 warp
//! maps a visible-VOP pixel `(i, j)` to a sub-pel sprite coordinate
//! `(F, G)` in `1/s`-pel units; the integer reference position is
//! `(F////s, G////s)` in that same absolute space, so the array index
//! is `(F////s − sprite_left, G////s − sprite_top)`. Out-of-array
//! fetches use the §7.6.4 last-full-pel clamp ([`SpriteMemory::fetch`]).
//!
//! ## §7.8.6 static blend (vs. GMC)
//!
//! The static reconstruction differs from the GMC blend in two ways:
//!
//! * No `s²/2 − rounding_control` bias term.
//! * The final divide is the `//` operator (round half away from zero,
//!   §3.4), not the GMC `/` truncation.
//!
//! ```text
//! Y = ((s − rj)((s − ri) Y00 + ri Y01) + rj ((s − ri) Y10 + ri Y11)) // s²
//! ```
//!
//! When `sprite_brightness_change == 1`, the §7.8.6 brightness post-
//! adjustment `Y = (Y · (brightness_change_factor + 100)) // 100`
//! (clipped to `[0, 2^bpp − 1]`) is applied; see
//! [`StaticSpriteParams::brightness_change_factor`].

use crate::vol::SpriteWarpingAccuracy;
use crate::warp::WarpGeometry;

/// One 16×16 luminance macroblock side.
pub const MB_LUMA_SIDE: usize = 16;
/// One 8×8 chrominance block side (4:2:0).
pub const MB_CHROMA_SIDE: usize = 8;

/// `n //// d` (§3.4): integer division with truncation toward negative
/// infinity. `d` must be positive.
#[inline]
fn div_floor(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    let q = n / d;
    let r = n % d;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// `// ` operator (§3.4): integer division with rounding to nearest,
/// half-integers away from zero. `d` must be positive.
#[inline]
fn div_half(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// A static-sprite memory plane (luminance *or* one chrominance
/// component). Samples are stored row-major; the top-left sample is at
/// the §7.8.2 absolute coordinate `(left, top)`.
///
/// Fetches outside the stored array are clamped to the nearest edge
/// sample per §7.6.4 (the rectangular bounding-rectangle case used for
/// basic sprites).
#[derive(Debug)]
pub struct SpriteMemory<'a> {
    samples: &'a [u8],
    width: usize,
    height: usize,
    /// Absolute §7.8.2 x-coordinate of array column 0.
    left: i64,
    /// Absolute §7.8.2 y-coordinate of array row 0.
    top: i64,
}

impl<'a> SpriteMemory<'a> {
    /// Build a sprite-memory plane. `width`/`height` are the stored
    /// array dimensions; `left`/`top` are the absolute coordinate of the
    /// top-left sample. Returns `None` if `samples` is too small.
    pub fn new(
        samples: &'a [u8],
        width: usize,
        height: usize,
        left: i64,
        top: i64,
    ) -> Option<Self> {
        if width == 0 || height == 0 || samples.len() < width * height {
            return None;
        }
        Some(SpriteMemory {
            samples,
            width,
            height,
            left,
            top,
        })
    }

    /// Fetch the sprite sample at absolute coordinate `(ax, ay)` with
    /// §7.6.4 edge clamping (per component).
    #[inline]
    pub fn fetch(&self, ax: i64, ay: i64) -> u8 {
        let cx = (ax - self.left).clamp(0, self.width as i64 - 1) as usize;
        let cy = (ay - self.top).clamp(0, self.height as i64 - 1) as usize;
        self.samples[cy * self.width + cx]
    }
}

/// Per-S-VOP static-sprite parameters that are not part of the warp
/// geometry: the rounding accuracy `s` and the optional brightness
/// change.
#[derive(Debug, Clone, Copy)]
pub struct StaticSpriteParams {
    /// Sub-pel denominator `s` (== `SpriteWarpingAccuracy::s()`).
    pub s: i64,
    /// `bits_per_pixel` from the VOL header (sample clip range).
    pub bits_per_pixel: u32,
    /// `brightness_change_factor` when `sprite_brightness_change == 1`,
    /// else `None` (no post-adjustment).
    pub brightness_change_factor: Option<i32>,
}

impl StaticSpriteParams {
    /// Build params from the warping accuracy + bit depth, with no
    /// brightness change.
    pub fn new(accuracy: SpriteWarpingAccuracy, bits_per_pixel: u32) -> Self {
        StaticSpriteParams {
            s: accuracy.s(),
            bits_per_pixel,
            brightness_change_factor: None,
        }
    }

    /// Apply the §7.8.6 brightness post-adjustment `Y = (Y · (factor +
    /// 100)) // 100`, clipped to `[0, 2^bpp − 1]`. A `None` factor is the
    /// identity.
    #[inline]
    fn apply_brightness(&self, value: i64) -> i64 {
        let max = (1i64 << self.bits_per_pixel) - 1;
        match self.brightness_change_factor {
            Some(factor) => {
                let adjusted = div_half(value * (i64::from(factor) + 100), 100);
                adjusted.clamp(0, max)
            }
            None => value.clamp(0, max),
        }
    }
}

/// Reconstruct one §7.8.6 static-sprite sample at warp coordinate
/// `(cap_f, cap_g)` (in `1/s`-pel units) from a sprite-memory plane.
#[inline]
fn reconstruct_static_sample(
    sprite: &SpriteMemory<'_>,
    cap_f: i64,
    cap_g: i64,
    params: &StaticSpriteParams,
) -> u8 {
    let s = params.s;
    // Integer reference position (toward −inf) and 1/s-pel residuals.
    let fx = div_floor(cap_f, s);
    let fy = div_floor(cap_g, s);
    let ri = cap_f - fx * s;
    let rj = cap_g - fy * s;

    // The four neighbour samples (with §7.6.4 last-full-pel clamping).
    let y00 = i64::from(sprite.fetch(fx, fy));
    let y01 = i64::from(sprite.fetch(fx + 1, fy));
    let y10 = i64::from(sprite.fetch(fx, fy + 1));
    let y11 = i64::from(sprite.fetch(fx + 1, fy + 1));

    // §7.8.6 static blend: bilinear, `//`-rounded, no rounding-control.
    let numer = (s - rj) * ((s - ri) * y00 + ri * y01) + rj * ((s - ri) * y10 + ri * y11);
    let value = div_half(numer, s * s);

    params.apply_brightness(value) as u8
}

/// Warp the static-sprite luminance memory into the visible VOP.
///
/// Produces a `vop_width × vop_height` luminance plane (row-major). Each
/// output pixel `(i, j)` (`0 <= i < vop_width`, `0 <= j < vop_height`,
/// relative to the §7.8.2 top-left `(0, 0)` for rectangular VOPs) is the
/// §7.8.6 static reconstruction of warp coordinate `geometry.luma_fg(i,
/// j)`.
pub fn static_sprite_luma(
    geometry: &WarpGeometry,
    sprite: &SpriteMemory<'_>,
    vop_width: usize,
    vop_height: usize,
    params: &StaticSpriteParams,
) -> Vec<u8> {
    let mut out = vec![0u8; vop_width * vop_height];
    for j in 0..vop_height {
        for i in 0..vop_width {
            let [cap_f, cap_g] = geometry.luma_fg(i as i64, j as i64);
            out[j * vop_width + i] = reconstruct_static_sample(sprite, cap_f, cap_g, params);
        }
    }
    out
}

/// Warp one chrominance component of the static-sprite memory into the
/// visible VOP. Produces a `(vop_width / 2) × (vop_height / 2)` plane
/// (4:2:0) using the §7.8.5 chroma warp `(Fc, Gc)`.
pub fn static_sprite_chroma(
    geometry: &WarpGeometry,
    sprite: &SpriteMemory<'_>,
    vop_width: usize,
    vop_height: usize,
    params: &StaticSpriteParams,
) -> Vec<u8> {
    let cw = vop_width / 2;
    let ch = vop_height / 2;
    let mut out = vec![0u8; cw * ch];
    for jc in 0..ch {
        for ic in 0..cw {
            let [cap_fc, cap_gc] = geometry.chroma_fg(ic as i64, jc as i64);
            out[jc * cw + ic] = reconstruct_static_sample(sprite, cap_fc, cap_gc, params);
        }
    }
    out
}

/// Warp one 16×16 luminance macroblock of the static sprite, whose
/// top-left visible-VOP pixel is `(mb_x, mb_y)`. Mirrors
/// [`crate::gmc::gmc_luma_prediction`] but uses the §7.8.6 static blend.
pub fn static_sprite_luma_macroblock(
    geometry: &WarpGeometry,
    sprite: &SpriteMemory<'_>,
    mb_x: i64,
    mb_y: i64,
    params: &StaticSpriteParams,
) -> [u8; MB_LUMA_SIDE * MB_LUMA_SIDE] {
    let mut out = [0u8; MB_LUMA_SIDE * MB_LUMA_SIDE];
    for y in 0..MB_LUMA_SIDE {
        for x in 0..MB_LUMA_SIDE {
            let [cap_f, cap_g] = geometry.luma_fg(mb_x + x as i64, mb_y + y as i64);
            out[y * MB_LUMA_SIDE + x] = reconstruct_static_sample(sprite, cap_f, cap_g, params);
        }
    }
    out
}

/// Warp the static-sprite luminance memory into the visible VOP using the
/// §7.8.5 **four-point perspective** transform
/// ([`crate::perspective_warp::PerspectiveWarp`]) instead of the affine
/// [`WarpGeometry`]. Mirrors [`static_sprite_luma`]: each output pixel is
/// the §7.8.6 static blend of the perspective warp coordinate. A pixel
/// whose perspective denominator vanishes (disallowed for opaque/boundary
/// pixels, §7.8.5) is surfaced as an error.
pub fn static_sprite_luma_perspective(
    warp: &crate::perspective_warp::PerspectiveWarp,
    sprite: &SpriteMemory<'_>,
    vop_width: usize,
    vop_height: usize,
    params: &StaticSpriteParams,
) -> Result<Vec<u8>, crate::perspective_warp::PerspectiveWarpError> {
    let mut out = vec![0u8; vop_width * vop_height];
    for j in 0..vop_height {
        for i in 0..vop_width {
            let [cap_f, cap_g] = warp.luma_fg(i as i64, j as i64)?;
            out[j * vop_width + i] = reconstruct_static_sample(sprite, cap_f, cap_g, params);
        }
    }
    Ok(out)
}

/// Warp one chrominance component of the static-sprite memory into the
/// visible VOP using the §7.8.5 **four-point perspective** transform.
/// Produces a `(vop_width / 2) × (vop_height / 2)` plane (4:2:0) from the
/// perspective chroma warp `(Fc, Gc)`.
pub fn static_sprite_chroma_perspective(
    warp: &crate::perspective_warp::PerspectiveWarp,
    sprite: &SpriteMemory<'_>,
    vop_width: usize,
    vop_height: usize,
    params: &StaticSpriteParams,
) -> Result<Vec<u8>, crate::perspective_warp::PerspectiveWarpError> {
    let cw = vop_width / 2;
    let ch = vop_height / 2;
    let mut out = vec![0u8; cw * ch];
    for jc in 0..ch {
        for ic in 0..cw {
            let [cap_fc, cap_gc] = warp.chroma_fg(ic as i64, jc as i64)?;
            out[jc * cw + ic] = reconstruct_static_sample(sprite, cap_fc, cap_gc, params);
        }
    }
    Ok(out)
}

/// Warp one 16×16 luminance macroblock of the static sprite via the
/// §7.8.5 four-point perspective transform, whose top-left visible-VOP
/// pixel is `(mb_x, mb_y)`. Mirrors [`static_sprite_luma_macroblock`]
/// but uses [`crate::perspective_warp::PerspectiveWarp`].
pub fn static_sprite_luma_macroblock_perspective(
    warp: &crate::perspective_warp::PerspectiveWarp,
    sprite: &SpriteMemory<'_>,
    mb_x: i64,
    mb_y: i64,
    params: &StaticSpriteParams,
) -> Result<[u8; MB_LUMA_SIDE * MB_LUMA_SIDE], crate::perspective_warp::PerspectiveWarpError> {
    let mut out = [0u8; MB_LUMA_SIDE * MB_LUMA_SIDE];
    for y in 0..MB_LUMA_SIDE {
        for x in 0..MB_LUMA_SIDE {
            let [cap_f, cap_g] = warp.luma_fg(mb_x + x as i64, mb_y + y as i64)?;
            out[y * MB_LUMA_SIDE + x] = reconstruct_static_sample(sprite, cap_f, cap_g, params);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perspective_warp::PerspectiveWarp;
    use crate::sprite::SpriteTrajectory;

    fn flat_sprite(w: usize, h: usize, v: u8) -> Vec<u8> {
        vec![v; w * h]
    }

    #[test]
    fn div_floor_truncates_toward_negative_infinity() {
        assert_eq!(div_floor(7, 4), 1);
        assert_eq!(div_floor(-7, 4), -2);
        assert_eq!(div_floor(8, 4), 2);
        assert_eq!(div_floor(-1, 4), -1);
    }

    #[test]
    fn div_half_rounds_away_from_zero() {
        assert_eq!(div_half(5, 2), 3);
        assert_eq!(div_half(-5, 2), -3);
        assert_eq!(div_half(4, 2), 2);
    }

    #[test]
    fn sprite_memory_fetch_offset_and_clamp() {
        // 4×4 sprite, top-left at absolute (10, 20). Sample (r,c) holds
        // value r*4 + c.
        let mut data = vec![0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                data[r * 4 + c] = (r * 4 + c) as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, 4, 4, 10, 20).unwrap();
        // Absolute (10, 20) → array (0, 0) → value 0.
        assert_eq!(sprite.fetch(10, 20), 0);
        // Absolute (13, 23) → array (3, 3) → value 15.
        assert_eq!(sprite.fetch(13, 23), 15);
        // Out-of-range below clamps to (0,0).
        assert_eq!(sprite.fetch(-100, -100), 0);
        // Out-of-range above clamps to (3,3).
        assert_eq!(sprite.fetch(1000, 1000), 15);
    }

    #[test]
    fn flat_sprite_reproduces_flat_vop_zero_warp() {
        // Zero warping points + flat sprite → flat VOP everywhere, with
        // no rounding bias.
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 32, 32, SpriteWarpingAccuracy::HalfPel);
        let data = flat_sprite(64, 64, 137);
        let sprite = SpriteMemory::new(&data, 64, 64, 0, 0).unwrap();
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);
        let plane = static_sprite_luma(&geo, &sprite, 16, 16, &params);
        assert!(plane.iter().all(|&p| p == 137), "flat sprite → flat VOP");
    }

    #[test]
    fn integer_translation_samples_shifted_sprite() {
        // One warping point: du0 = 2*s (a whole-pel shift of +1 in the
        // sprite at s=2 ⇒ i0' = (s/2)*du0 = 2). F(i) = i0' + s*i = 2 + 2i.
        // F////s = 1 + i ⇒ output pixel i reads sprite column (i+1).
        // Ramp sprite: sample at (r,c) = c (column index).
        let w = 32usize;
        let h = 8usize;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = c as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, w, h, 0, 0).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [4, 0]; // du0 = 4, s = 2 ⇒ i0' = (s/2)*4 = 4 ⇒ shift +2 cols
        let geo = WarpGeometry::decode(&traj, 16, 8, SpriteWarpingAccuracy::HalfPel);
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);
        let plane = static_sprite_luma(&geo, &sprite, 8, 4, &params);
        // i0' = 4, s = 2 ⇒ F(i)=4+2i ⇒ F////s = 2+i ⇒ pixel i = column (i+2).
        for (i, &p) in plane.iter().take(8).enumerate() {
            assert_eq!(p, (i + 2) as u8, "pixel {i}");
        }
    }

    #[test]
    fn brightness_change_scales_samples() {
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 16, 16, SpriteWarpingAccuracy::HalfPel);
        let data = flat_sprite(32, 32, 100);
        let sprite = SpriteMemory::new(&data, 32, 32, 0, 0).unwrap();
        let mut params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);
        params.brightness_change_factor = Some(50); // (100 * 150) // 100 = 150
        let plane = static_sprite_luma(&geo, &sprite, 8, 8, &params);
        assert!(plane.iter().all(|&p| p == 150));
    }

    #[test]
    fn brightness_change_clips_to_bit_depth() {
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 16, 16, SpriteWarpingAccuracy::HalfPel);
        let data = flat_sprite(32, 32, 200);
        let sprite = SpriteMemory::new(&data, 32, 32, 0, 0).unwrap();
        let mut params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);
        params.brightness_change_factor = Some(100); // (200 * 200)//100 = 400 → clip 255
        let plane = static_sprite_luma(&geo, &sprite, 8, 8, &params);
        assert!(plane.iter().all(|&p| p == 255));
    }

    #[test]
    fn macroblock_matches_full_plane_subregion() {
        // The per-MB warp must agree with the full-plane warp on the
        // overlapping 16×16 region.
        let w = 64usize;
        let h = 64usize;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = ((r * 7 + c * 3) & 0xff) as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, w, h, 0, 0).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 2;
        traj.points = [[2, -2], [3, 1], [0, 0]];
        let geo = WarpGeometry::decode(&traj, 48, 48, SpriteWarpingAccuracy::QuarterPel);
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::QuarterPel, 8);
        let plane = static_sprite_luma(&geo, &sprite, 48, 48, &params);
        let mb = static_sprite_luma_macroblock(&geo, &sprite, 16, 16, &params);
        for y in 0..MB_LUMA_SIDE {
            for x in 0..MB_LUMA_SIDE {
                let pi = (16 + y) * 48 + (16 + x);
                assert_eq!(mb[y * MB_LUMA_SIDE + x], plane[pi], "({x},{y})");
            }
        }
    }

    #[test]
    fn perspective_zero_trajectory_matches_affine_identity() {
        // With a zero 4-point trajectory the perspective warp collapses to
        // the identity (F = s i, G = s j), so the reconstructed plane must
        // equal the zero-point affine static warp over the same region.
        let w = 64usize;
        let h = 64usize;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = ((r * 5 + c * 2) & 0xff) as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, w, h, 0, 0).unwrap();
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);

        let traj0 = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj0, 48, 48, SpriteWarpingAccuracy::HalfPel);
        let affine = static_sprite_luma(&geo, &sprite, 48, 48, &params);

        let pwarp = PerspectiveWarp::decode(&[[0, 0]; 4], 48, 48, SpriteWarpingAccuracy::HalfPel);
        let persp = static_sprite_luma_perspective(&pwarp, &sprite, 48, 48, &params).unwrap();

        assert_eq!(affine, persp);
    }

    #[test]
    fn perspective_chroma_zero_trajectory_matches_affine() {
        let w = 64usize;
        let h = 64usize;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = ((r * 3 + c * 7) & 0xff) as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, w, h, 0, 0).unwrap();
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::QuarterPel, 8);

        let traj0 = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj0, 32, 32, SpriteWarpingAccuracy::QuarterPel);
        let affine = static_sprite_chroma(&geo, &sprite, 32, 32, &params);

        let pwarp =
            PerspectiveWarp::decode(&[[0, 0]; 4], 32, 32, SpriteWarpingAccuracy::QuarterPel);
        let persp = static_sprite_chroma_perspective(&pwarp, &sprite, 32, 32, &params).unwrap();

        assert_eq!(affine, persp);
    }

    #[test]
    fn perspective_macroblock_matches_full_plane_subregion() {
        let w = 96usize;
        let h = 96usize;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = ((r * 11 + c * 5) & 0xff) as u8;
            }
        }
        let sprite = SpriteMemory::new(&data, w, h, 0, 0).unwrap();
        let params = StaticSpriteParams::new(SpriteWarpingAccuracy::HalfPel, 8);
        // A mild non-degenerate perspective trajectory.
        let traj = [[2, 0], [0, 2], [-2, 1], [1, -1]];
        let pwarp = PerspectiveWarp::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);
        let plane = static_sprite_luma_perspective(&pwarp, &sprite, 64, 64, &params).unwrap();
        let mb =
            static_sprite_luma_macroblock_perspective(&pwarp, &sprite, 16, 16, &params).unwrap();
        for y in 0..MB_LUMA_SIDE {
            for x in 0..MB_LUMA_SIDE {
                let pi = (16 + y) * 64 + (16 + x);
                assert_eq!(mb[y * MB_LUMA_SIDE + x], plane[pi], "({x},{y})");
            }
        }
    }

    #[test]
    fn sprite_left_top_offset_repositions_array() {
        // A sprite stored at absolute (left, top) must be sampled at the
        // same VOP output as one stored at (0,0) when the warp's
        // reference points are shifted by the same offset is non-trivial;
        // here just confirm the offset is honoured by fetch().
        let data: Vec<u8> = (0..16).map(|v| v as u8).collect();
        let s0 = SpriteMemory::new(&data, 4, 4, 0, 0).unwrap();
        let s1 = SpriteMemory::new(&data, 4, 4, -8, 5).unwrap();
        assert_eq!(s0.fetch(2, 1), s1.fetch(2 - 8, 1 + 5));
    }
}

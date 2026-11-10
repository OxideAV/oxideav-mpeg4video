//! §7.8 **basic static-sprite** encoding (`sprite_enable == "static"`,
//! `low_latency_sprite_enable == 0`).
//!
//! A static-sprite VOL codes one picture — the sprite — as its initial
//! I-VOP (§7.8.1: reconstructed into the sprite memory, never
//! displayed) and then describes every displayed VOP by a warp of that
//! sprite: an S-VOP that carries only a `sprite_trajectory()` (§6.2.5:
//! after the trajectory the VOP ends — no quantiser, no f_code, no
//! macroblock data). The decoder reconstructs each S-VOP with the
//! §7.8.4 reference points, the §7.8.5 warp (stationary / translation
//! / the affine family for 0..=3 points, perspective for 4) and the
//! §7.8.6 static sample reconstruction.
//!
//! [`StaticSpriteEncoder`] emits such a stream:
//!
//! * the configuration headers of a static-sprite VOL (the sprite
//!   geometry, warp model and accuracy of
//!   [`crate::ivop_encode::StaticSpriteVol`]) followed by the sprite
//!   I-VOP, coded on the sprite's own macroblock grid
//!   (`sprite_width × sprite_height`, the §7.8.2 sprite array) by the
//!   intra encoder;
//! * per displayed picture, an S-VOP whose trajectory is **fitted** to
//!   the picture ([`StaticSpriteEncoder::encode_vop`]) or supplied by
//!   the caller ([`StaticSpriteEncoder::encode_vop_with_trajectory`]).
//!
//! The fit minimises the luminance SAD between the picture and the
//! decoder's own warp of the closed-loop sprite reconstruction: a
//! coarse translation scan of the picture over the sprite (first VOP;
//! later VOPs scan a window around the previous trajectory), then a
//! multi-scale descent over every transmitted `du[i]` / `dv[i]`. What
//! is measured is exactly the picture a conformant decoder forms.
//!
//! Provenance: §6.2.3 (VOL sprite fields), §6.2.5 (S-VOP syntax,
//! `sprite_trajectory()`), §7.8.1–§7.8.6 of ISO/IEC 14496-2:2004 (3rd
//! edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
//! third-party source was consulted.

use crate::bitwriter::BitWriter;
use crate::framestore::DecodedFrame;
use crate::ivop_encode::{
    EncoderConfig, FrameView, StaticSpriteVol, encode_i_vop, write_configuration_headers,
};
use crate::static_sprite::{
    SpritePlanes, StaticSpriteParams, StaticWarp, static_sample, warp_static_sprite_vop,
};
use crate::svop_encode::put_warping_mv_code;
use crate::vol::{VolHeader, parse_video_object_layer};
use crate::vop::{VopCodingType, vop_time_increment_bits};

const VOP_START_CODE: u32 = 0x0000_01B6;

/// Largest `|du|` / `|dv|` Table B.34 can carry.
const TRAJECTORY_BOUND: i32 = (1 << 14) - 1;

/// Sampling pitch of the trajectory-fit cost (every fourth luminance
/// sample in both directions).
const FIT_PITCH: usize = 4;

/// Emit one §6.2.5 static-sprite S-VOP: the VOP header through
/// `intra_dc_vlc_thr`, then `sprite_trajectory()` — `points` pairs of
/// `warping_mv_code()` — and the §5.2.4 stuffing to the next start
/// code. (`sprite_brightness_change == 0` and the basic sprite's
/// `low_latency_sprite_enable == 0` leave nothing else to write.)
///
/// # Panics
///
/// Panics when `points > 4` or a trajectory component exceeds the
/// Table B.34 range.
pub fn write_static_sprite_vop(
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    intra_dc_vlc_thr: u8,
    points: u8,
    trajectory: &[[i32; 2]; 4],
) -> Vec<u8> {
    assert!(points <= 4, "Table 6-20: 0..=4 warping points");
    let mut bw = BitWriter::new();
    bw.write_start_code(VOP_START_CODE);
    bw.write_bits(0b11, 2); // vop_coding_type = S
    for _ in 0..modulo_time_base {
        bw.write_bit(true);
    }
    bw.write_bit(false);
    bw.write_marker();
    bw.write_bits(
        u32::from(time_increment),
        usize::from(vop_time_increment_bits(resolution)),
    );
    bw.write_marker();
    bw.write_bit(true); // vop_coded = 1
    // No vop_rounding_type: §6.2.5 sends it for P and S(GMC) only.
    assert!(intra_dc_vlc_thr <= 7, "intra_dc_vlc_thr is a 3-bit field");
    bw.write_bits(u32::from(intra_dc_vlc_thr), 3);
    for point in &trajectory[..usize::from(points)] {
        put_warping_mv_code(&mut bw, point[0]);
        put_warping_mv_code(&mut bw, point[1]);
    }
    bw.next_start_code();
    bw.into_bytes()
}

/// Encoder for one basic static-sprite VOL (see the module docs).
#[derive(Debug, Clone)]
pub struct StaticSpriteEncoder {
    cfg: EncoderConfig,
    sprite_vol: StaticSpriteVol,
    /// The closed-loop sprite memory — the decoder's reconstruction of
    /// the emitted sprite I-VOP, cropped to the sprite arrays.
    sprite: SpritePlanes,
    /// The trajectory of the previous S-VOP (the next fit's start).
    previous: Option<[[i32; 2]; 4]>,
}

impl StaticSpriteEncoder {
    /// Start a static-sprite stream: returns the encoder and the
    /// stream head — the §6.2 configuration headers followed by the
    /// sprite I-VOP (`sprite` at `cfg.static_sprite`'s
    /// `width × height`, quantiser `qp`, VOP time zero).
    ///
    /// # Panics
    ///
    /// Panics when `cfg.static_sprite` is `None`, the sprite view does
    /// not have the configured sprite dimensions, or the configuration
    /// combines the sprite with a tool a static-sprite VOL cannot hold
    /// (GMC, B-VOPs, interlace, data partitioning, short header).
    pub fn new(cfg: &EncoderConfig, sprite: &FrameView<'_>, qp: u32) -> (Self, Vec<u8>) {
        let sprite_vol = cfg
            .static_sprite
            .expect("StaticSpriteEncoder needs cfg.static_sprite");
        assert!(
            sprite.width == usize::from(sprite_vol.width)
                && sprite.height == usize::from(sprite_vol.height),
            "the sprite picture must be sprite_width x sprite_height"
        );
        let mut stream = write_configuration_headers(cfg);
        // The sprite I-VOP is an ordinary I-VOP on the sprite's own
        // macroblock grid: code it through the intra encoder under a
        // plain VOL of the sprite's dimensions (the tools an I-VOP
        // reads — quantiser type, AC prediction, intra_dc_vlc_thr —
        // are those of `cfg`).
        let plain = EncoderConfig {
            width: sprite_vol.width,
            height: sprite_vol.height,
            static_sprite: None,
            ..*cfg
        };
        let plain_vol = vol_of(&plain);
        let (unit, recon) = encode_i_vop(&plain_vol, &plain, sprite, 0, 0, qp);
        stream.extend_from_slice(&unit);
        let (mb_width, _) = plain.mb_dimensions();
        let planes = SpritePlanes::from_aligned(
            recon.luma_samples(),
            recon.cb_samples(),
            recon.cr_samples(),
            mb_width * 16,
            sprite.width,
            sprite.height,
            i64::from(sprite_vol.left),
            i64::from(sprite_vol.top),
        )
        .expect("the reconstruction covers the sprite");
        (
            Self {
                cfg: *cfg,
                sprite_vol,
                sprite: planes,
                previous: None,
            },
            stream,
        )
    }

    /// The closed-loop sprite memory (what a decoder holds after the
    /// sprite I-VOP).
    #[doc(hidden)]
    pub fn sprite(&self) -> &SpritePlanes {
        &self.sprite
    }

    /// The warp a decoder derives from `trajectory` for this VOL.
    fn warp(&self, trajectory: &[[i32; 2]; 4]) -> StaticWarp {
        StaticWarp::decode(
            self.sprite_vol.warping_points,
            trajectory,
            u32::from(self.cfg.width),
            u32::from(self.cfg.height),
            self.sprite_vol.accuracy,
        )
    }

    /// Whether `trajectory` defines the warp at every luminance and
    /// chrominance sample of the coded (macroblock-aligned) picture.
    /// The affine family always does; the four-point perspective
    /// transform does not where its denominator vanishes.
    fn warp_is_defined(&self, trajectory: &[[i32; 2]; 4]) -> bool {
        let warp = self.warp(trajectory);
        if matches!(warp, StaticWarp::Affine(_)) {
            return true;
        }
        let (mb_width, mb_height) = self.cfg.mb_dimensions();
        let (fw, fh) = ((mb_width * 16) as i64, (mb_height * 16) as i64);
        (0..fh).all(|j| (0..fw).all(|i| warp.luma_fg(i, j).is_some()))
            && (0..fh / 2).all(|j| (0..fw / 2).all(|i| warp.chroma_fg(i, j).is_some()))
    }

    /// Luminance SAD between `frame` and the decoder's warp of the
    /// sprite under `trajectory`, on the [`FIT_PITCH`] sampling grid.
    /// `u64::MAX` for a warp that is undefined inside the picture.
    fn fit_cost(&self, frame: &FrameView<'_>, trajectory: &[[i32; 2]; 4]) -> u64 {
        let warp = self.warp(trajectory);
        let params = StaticSpriteParams::new(self.sprite_vol.accuracy, 8);
        let Some(memory) = self.sprite.luma_memory() else {
            return u64::MAX;
        };
        let mut cost = 0u64;
        let mut j = FIT_PITCH / 2;
        while j < frame.height {
            let mut i = FIT_PITCH / 2;
            while i < frame.width {
                let Some(fg) = warp.luma_fg(i as i64, j as i64) else {
                    return u64::MAX;
                };
                let predicted = static_sample(&memory, fg, &params);
                cost += u64::from(frame.y[j * frame.width + i].abs_diff(predicted));
                i += FIT_PITCH;
            }
            j += FIT_PITCH;
        }
        cost
    }

    /// Fit the trajectory of `frame` (see the module docs).
    fn fit(&self, frame: &FrameView<'_>) -> [[i32; 2]; 4] {
        let points = usize::from(self.sprite_vol.warping_points);
        let mut best = [[0i32; 2]; 4];
        if points == 0 {
            return best;
        }
        let mut best_cost = u64::MAX;
        let consider = |cand: [[i32; 2]; 4], best: &mut [[i32; 2]; 4], best_cost: &mut u64| {
            let cost = self.fit_cost(frame, &cand);
            // The cost samples a sub-grid; a winner must define the
            // warp at every sample of the coded picture (§7.8.5
            // disallows a vanishing perspective denominator).
            if cost < *best_cost && self.warp_is_defined(&cand) {
                *best_cost = cost;
                *best = cand;
                return true;
            }
            false
        };
        // Stage 1 — where the picture sits: a 4-sample lattice of
        // picture origins (du[0] = 2 · x, dv[0] = 2 · y in half-sample
        // units; §7.8.4 with i0 = j0 = 0). The first VOP scans every
        // position that keeps the picture inside the sprite; later
        // VOPs scan a ±32-sample window around the previous
        // trajectory, keeping its other points.
        match self.previous {
            Some(previous) => {
                consider(previous, &mut best, &mut best_cost);
                for y in (-32..=32).step_by(4) {
                    for x in (-32..=32).step_by(4) {
                        let mut cand = previous;
                        cand[0][0] =
                            (cand[0][0] + 2 * x).clamp(-TRAJECTORY_BOUND, TRAJECTORY_BOUND);
                        cand[0][1] =
                            (cand[0][1] + 2 * y).clamp(-TRAJECTORY_BOUND, TRAJECTORY_BOUND);
                        consider(cand, &mut best, &mut best_cost);
                    }
                }
            }
            None => {
                let (left, top) = (
                    i32::from(self.sprite_vol.left),
                    i32::from(self.sprite_vol.top),
                );
                let span_x = (self.sprite.width as i32 - frame.width as i32).max(0);
                let span_y = (self.sprite.height as i32 - frame.height as i32).max(0);
                for y in (0..=span_y).step_by(4) {
                    for x in (0..=span_x).step_by(4) {
                        let mut cand = [[0i32; 2]; 4];
                        cand[0] = [2 * (left + x), 2 * (top + y)];
                        consider(cand, &mut best, &mut best_cost);
                    }
                }
            }
        }
        // Stage 2 — multi-scale descent. A move changes one
        // transmitted component; for the points beyond the first a
        // second move family changes the component by twice the step
        // while pulling point 0 back by the step, which reshapes the
        // warp about the picture centre instead of dragging the whole
        // picture along (du[i] of a far reference point displaces the
        // centre by half of it).
        let clamp = |v: i32| v.clamp(-TRAJECTORY_BOUND, TRAJECTORY_BOUND);
        for step in [8i32, 4, 2, 1] {
            for _pass in 0..16 {
                let mut improved = false;
                for point in 0..points {
                    for axis in 0..2 {
                        for delta in [-step, step] {
                            let mut cand = best;
                            cand[point][axis] = clamp(cand[point][axis] + delta);
                            improved |= consider(cand, &mut best, &mut best_cost);
                            if point > 0 {
                                let mut cand = best;
                                cand[point][axis] = clamp(cand[point][axis] + 2 * delta);
                                cand[0][axis] = clamp(cand[0][axis] - delta);
                                improved |= consider(cand, &mut best, &mut best_cost);
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        best
    }

    /// Encode one displayed picture as an S-VOP with a fitted
    /// trajectory. Returns the unit, the closed-loop reconstruction
    /// (what a decoder displays) and the transmitted trajectory.
    ///
    /// # Panics
    ///
    /// Panics when `frame` does not have the VOL dimensions.
    pub fn encode_vop(
        &mut self,
        frame: &FrameView<'_>,
        modulo_time_base: u32,
        time_increment: u16,
    ) -> (Vec<u8>, DecodedFrame, [[i32; 2]; 4]) {
        assert!(
            frame.width == usize::from(self.cfg.width)
                && frame.height == usize::from(self.cfg.height),
            "the picture must have the VOL dimensions"
        );
        let trajectory = self.fit(frame);
        let (unit, recon) =
            self.encode_vop_with_trajectory(&trajectory, modulo_time_base, time_increment);
        (unit, recon, trajectory)
    }

    /// Encode an S-VOP with a caller-chosen trajectory (the first
    /// `no_of_sprite_warping_points` `[du, dv]` pairs are sent).
    /// Returns the unit and the closed-loop reconstruction.
    ///
    /// # Panics
    ///
    /// Panics when a transmitted component exceeds the Table B.34
    /// range, or the warp is undefined inside the picture (a vanishing
    /// perspective denominator — §7.8.5 disallows such parameters).
    pub fn encode_vop_with_trajectory(
        &mut self,
        trajectory: &[[i32; 2]; 4],
        modulo_time_base: u32,
        time_increment: u16,
    ) -> (Vec<u8>, DecodedFrame) {
        let points = self.sprite_vol.warping_points;
        let mut sent = [[0i32; 2]; 4];
        sent[..usize::from(points)].copy_from_slice(&trajectory[..usize::from(points)]);
        let unit = write_static_sprite_vop(
            self.cfg.time_increment_resolution,
            modulo_time_base,
            time_increment,
            self.cfg.intra_dc_vlc_thr,
            points,
            &sent,
        );
        let (mb_width, mb_height) = self.cfg.mb_dimensions();
        let (fw, fh) = (mb_width * 16, mb_height * 16);
        let (luma, cb, cr) = warp_static_sprite_vop(
            &self.sprite,
            u32::from(self.cfg.width),
            u32::from(self.cfg.height),
            fw,
            fh,
            self.sprite_vol.accuracy,
            8,
            points,
            &sent,
            None,
        )
        .expect("the trajectory defines a warp over the whole picture");
        let recon = DecodedFrame::from_planes(fw, fh, VopCodingType::S, luma, cb, cr)
            .expect("macroblock-aligned planes");
        self.previous = Some(sent);
        (unit, recon)
    }
}

/// Re-parse the VOL the configuration headers of `cfg` declare.
fn vol_of(cfg: &EncoderConfig) -> VolHeader {
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("emitted headers contain the VOL start code");
    parse_video_object_layer(&headers[pos..], cfg.profile_and_level())
        .expect("emitted VOL re-parses")
}

//! §6.2.5 / §6.3.5.4 `sprite_trajectory()` decode for S(GMC)-VOPs.
//!
//! The sprite trajectory carries the per-warping-point differential
//! motion vectors `du[i]` / `dv[i]` (`0 <= i < no_of_sprite_warping_points`)
//! that, together with the §7.8.4 reference-point geometry, drive the
//! global-motion (GMC) and static-sprite warp. Each component is coded
//! with `warping_mv_code()` (§6.3.5.4): a VLC `dmv_length` giving the
//! magnitude category `SSS`, an `SSS`-bit fixed-length `dmv_code`
//! selecting the value within that category, and a trailing
//! `marker_bit`.
//!
//! The codeword table is Table B.34 ("Code table for the first
//! trajectory point"). The VLC half is a run of `SSS` leading `1` bits
//! followed by a `0` (so `SSS == 0` is the bare code `00`-prefixed two
//! bits `00`); the §7.8 GMC milestone only needs the `2 <= SSS <= 14`
//! main-profile range (`dmv_length` is `2-12` bits per §6.2.6's
//! `warping_mv_code`). The fixed-length `dmv_code` maps:
//!
//! * `SSS == 0` → `dmv == 0`, no `dmv_code` bits.
//! * `SSS == k` (`k >= 1`) → the `k`-bit code selects one of the `2^k`
//!   values in `{ -(2^k - 1) .. -2^(k-1) , 2^(k-1) .. 2^k - 1 }`. Codes
//!   whose top bit is `1` map to the positive half (the code is the
//!   value); codes whose top bit is `0` map to the negative half
//!   (`value = code - (2^k - 1)`), in ascending code order. Table B.34
//!   worked examples: `SSS == 1`: `0` → `-1`, `1` → `+1`;
//!   `SSS == 2`: `00,01` → `-3,-2`, `10,11` → `+2,+3`;
//!   `SSS == 5`: `00000` → `-31`, `11111` → `+31`.
//!
//! `du[i]` / `dv[i]` are the *differential* values as transmitted; the
//! §7.8.4 reference-point reconstruction sums them (the spec's
//! `du[1] + du[0]`, etc.) — that accumulation is performed by the
//! geometry stage, not here.

use crate::bitreader::BitReader;

/// Maximum warping points a GMC stream may carry. The perspective
/// (4-point) transform is disallowed under `sprite_enable == "GMC"`
/// (§6.3.3), so the GMC trajectory holds at most 3 points. Static
/// sprites can use 4 (the §7.8.5 perspective transform); see
/// [`decode_sprite_trajectory_static`].
pub const MAX_GMC_WARPING_POINTS: usize = 3;

/// Maximum warping points a static-sprite stream may carry: the §7.8.5
/// perspective transform uses 4 (`no_of_sprite_warping_points == 4`,
/// Table 6-88).
pub const MAX_STATIC_WARPING_POINTS: usize = 4;

/// Errors raised while decoding a `sprite_trajectory()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteTrajectoryError {
    /// The bitstream ran out mid-codeword.
    Truncated,
    /// A `marker_bit` in a `warping_mv_code()` was `0` (§6.2 marker
    /// convention requires `1`).
    MarkerBitMissing,
    /// `dmv_length` ran past the 14-bit maximum of Table B.34 without a
    /// terminating `0` — the stream is malformed.
    LengthOverflow,
    /// `no_of_sprite_warping_points` exceeded [`MAX_GMC_WARPING_POINTS`]
    /// (the caller passed a static/perspective count the GMC trajectory
    /// container cannot hold). Carries the offending count.
    TooManyPoints(u8),
}

impl core::fmt::Display for SpriteTrajectoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpriteTrajectoryError::Truncated => {
                write!(f, "sprite_trajectory truncated")
            }
            SpriteTrajectoryError::MarkerBitMissing => {
                write!(f, "warping_mv_code marker_bit was 0 (expected 1)")
            }
            SpriteTrajectoryError::LengthOverflow => {
                write!(f, "dmv_length exceeded the 14-bit Table B.34 maximum")
            }
            SpriteTrajectoryError::TooManyPoints(n) => {
                write!(
                    f,
                    "no_of_sprite_warping_points={n} exceeds the GMC maximum of {MAX_GMC_WARPING_POINTS}"
                )
            }
        }
    }
}

impl std::error::Error for SpriteTrajectoryError {}

/// The maximum `dmv_length` (SSS) value defined by Table B.34.
const MAX_SSS: u32 = 14;

/// Decode one `warping_mv_code()` (§6.3.5.4) returning the signed
/// differential `dmv` value.
///
/// Layout: VLC `dmv_length` (a unary run of `SSS` `1`-bits then a `0`),
/// then — when `SSS != 0` — an `SSS`-bit FLC `dmv_code`, then a
/// `marker_bit` (`1`).
pub fn decode_warping_mv_code(br: &mut BitReader<'_>) -> Result<i32, SpriteTrajectoryError> {
    // Read the unary `dmv_length` (SSS): count leading 1s, stop at the 0.
    let mut sss: u32 = 0;
    loop {
        let bit = br
            .read_bits(1)
            .map_err(|_| SpriteTrajectoryError::Truncated)?;
        if bit == 0 {
            break;
        }
        sss += 1;
        if sss > MAX_SSS {
            return Err(SpriteTrajectoryError::LengthOverflow);
        }
    }

    let dmv = if sss == 0 {
        0
    } else {
        // `SSS`-bit FLC. Per Table B.34, `dmv_length == SSS` covers the
        // magnitude range `[2^(SSS-1), 2^SSS - 1]`. The SSS-bit code is
        // read as an unsigned integer in `[0, 2^SSS - 1]`:
        //   * top bit set  → positive half: the code *is* the value
        //     (already in `[2^(SSS-1), 2^SSS - 1]`).
        //   * top bit clear → negative half: value = code - (2^SSS - 1),
        //     mapping code 0 → -(2^SSS - 1) (the most negative) up to
        //     code 2^(SSS-1)-1 → -2^(SSS-1) (the least negative).
        // SSS=1: 0→-1, 1→+1. SSS=2: 00→-3,01→-2,10→+2,11→+3.
        let code = br
            .read_bits(sss as usize)
            .map_err(|_| SpriteTrajectoryError::Truncated)?;
        let span = (1i64 << sss) - 1; // 2^SSS - 1
        let top_bit_set = (code >> (sss - 1)) & 1 == 1;
        let value = if top_bit_set {
            i64::from(code)
        } else {
            i64::from(code) - span
        };
        value as i32
    };

    // Trailing marker_bit.
    let marker = br
        .read_bits(1)
        .map_err(|_| SpriteTrajectoryError::Truncated)?;
    if marker != 1 {
        return Err(SpriteTrajectoryError::MarkerBitMissing);
    }
    Ok(dmv)
}

/// Decoded `sprite_trajectory()` (§6.2.5): the `du[i]` / `dv[i]`
/// differential warping vectors, plus the active point count.
///
/// `Copy` so it can ride inside the `Copy` [`crate::vop::VopHeader`].
/// Only `count` entries are valid; the rest are `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpriteTrajectory {
    /// Number of warping points actually decoded (`==
    /// no_of_sprite_warping_points`).
    pub count: u8,
    /// Per-point differential vectors `[du[i], dv[i]]`. Only the first
    /// [`Self::count`] entries are meaningful.
    pub points: [[i32; 2]; MAX_GMC_WARPING_POINTS],
}

impl SpriteTrajectory {
    /// An empty (stationary, `count == 0`) trajectory.
    pub const fn stationary() -> Self {
        Self {
            count: 0,
            points: [[0, 0]; MAX_GMC_WARPING_POINTS],
        }
    }
}

/// Decode a §6.2.5 `sprite_trajectory()` body: `no_of_sprite_warping_points`
/// pairs of `warping_mv_code()` codewords (`du[i]`, then `dv[i]`).
///
/// The caller passes `no_of_sprite_warping_points` (Table 6-20). A
/// value of `0` is the stationary case and produces an empty trajectory
/// without consuming any bits (the §6.2.5 syntax guards
/// `sprite_trajectory()` behind `no_of_sprite_warping_points > 0`).
pub fn decode_sprite_trajectory(
    br: &mut BitReader<'_>,
    no_of_sprite_warping_points: u8,
) -> Result<SpriteTrajectory, SpriteTrajectoryError> {
    if no_of_sprite_warping_points as usize > MAX_GMC_WARPING_POINTS {
        return Err(SpriteTrajectoryError::TooManyPoints(
            no_of_sprite_warping_points,
        ));
    }
    let mut traj = SpriteTrajectory::stationary();
    traj.count = no_of_sprite_warping_points;
    for i in 0..no_of_sprite_warping_points as usize {
        let du = decode_warping_mv_code(br)?;
        let dv = decode_warping_mv_code(br)?;
        traj.points[i] = [du, dv];
    }
    Ok(traj)
}

/// Decode a §6.2.5 `sprite_trajectory()` body for a **static** sprite,
/// which may carry up to [`MAX_STATIC_WARPING_POINTS`] (4) points — the
/// §7.8.5 perspective transform case the GMC-capped
/// [`decode_sprite_trajectory`] rejects.
///
/// Returns the raw `[du[i], dv[i]]` pairs (unused tail entries are `0`)
/// plus the active `count`. The 4-point array feeds
/// [`crate::perspective_warp::PerspectiveWarp::decode`]; counts 0..=3 feed
/// [`crate::warp::WarpGeometry::decode`] after taking the leading three
/// pairs.
pub fn decode_sprite_trajectory_static(
    br: &mut BitReader<'_>,
    no_of_sprite_warping_points: u8,
) -> Result<(u8, [[i32; 2]; MAX_STATIC_WARPING_POINTS]), SpriteTrajectoryError> {
    if no_of_sprite_warping_points as usize > MAX_STATIC_WARPING_POINTS {
        return Err(SpriteTrajectoryError::TooManyPoints(
            no_of_sprite_warping_points,
        ));
    }
    let mut points = [[0i32; 2]; MAX_STATIC_WARPING_POINTS];
    for p in points.iter_mut().take(no_of_sprite_warping_points as usize) {
        let du = decode_warping_mv_code(br)?;
        let dv = decode_warping_mv_code(br)?;
        *p = [du, dv];
    }
    Ok((no_of_sprite_warping_points, points))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer mirroring the spec's `bslbf` / `uimsbf`.
    struct BitWriter {
        buf: Vec<u8>,
        bit: u8,
        cur: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                bit: 0,
                cur: 0,
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for k in (0..n).rev() {
                let b = ((value >> k) & 1) as u8;
                self.cur |= b << (7 - self.bit);
                self.bit += 1;
                if self.bit == 8 {
                    self.buf.push(self.cur);
                    self.cur = 0;
                    self.bit = 0;
                }
            }
        }
        /// Emit a `warping_mv_code` for the given `dmv`: unary SSS, FLC, marker.
        fn write_warping(&mut self, sss: u32, code: u32) {
            for _ in 0..sss {
                self.write_bits(1, 1);
            }
            self.write_bits(0, 1); // terminating 0
            if sss != 0 {
                self.write_bits(code, sss as usize);
            }
            self.write_bits(1, 1); // marker
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bit != 0 {
                self.buf.push(self.cur);
            }
            self.buf
        }
    }

    #[test]
    fn sss_zero_is_dmv_zero() {
        let mut w = BitWriter::new();
        w.write_warping(0, 0);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), 0);
    }

    #[test]
    fn sss_one_maps_minus_one_and_plus_one() {
        // SSS=1, code 0 → -1.
        let mut w = BitWriter::new();
        w.write_warping(1, 0);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), -1);

        // SSS=1, code 1 → +1.
        let mut w = BitWriter::new();
        w.write_warping(1, 1);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), 1);
    }

    #[test]
    fn sss_two_maps_table_b34_examples() {
        // SSS=2: 00→-3, 01→-2, 10→+2, 11→+3.
        for (code, expected) in [(0b00, -3), (0b01, -2), (0b10, 2), (0b11, 3)] {
            let mut w = BitWriter::new();
            w.write_warping(2, code);
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(
                decode_warping_mv_code(&mut br).unwrap(),
                expected,
                "SSS=2 code={code:02b}"
            );
        }
    }

    #[test]
    fn sss_five_range_endpoints() {
        // SSS=5: range [-31..-16, 16..31]. code 00000 → -31, 11111 → 31.
        let mut w = BitWriter::new();
        w.write_warping(5, 0b00000);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), -31);

        let mut w = BitWriter::new();
        w.write_warping(5, 0b11111);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), 31);

        // code 10000 (top bit set, lowest positive) → +16.
        let mut w = BitWriter::new();
        w.write_warping(5, 0b10000);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), 16);

        // code 01111 (top bit clear, least negative) → -16.
        let mut w = BitWriter::new();
        w.write_warping(5, 0b01111);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_warping_mv_code(&mut br).unwrap(), -16);
    }

    #[test]
    fn missing_marker_is_rejected() {
        // SSS=0 followed by a 0 instead of the marker 1.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // SSS terminator (SSS=0)
        w.write_bits(0, 1); // bad marker
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_warping_mv_code(&mut br).unwrap_err(),
            SpriteTrajectoryError::MarkerBitMissing
        );
    }

    #[test]
    fn trajectory_two_points_decodes_du_dv() {
        // 2 warping points (affine): du0=+1, dv0=-1, du1=+3, dv1=-2.
        let mut w = BitWriter::new();
        w.write_warping(1, 1); // du0 = +1
        w.write_warping(1, 0); // dv0 = -1
        w.write_warping(2, 0b11); // du1 = +3
        w.write_warping(2, 0b01); // dv1 = -2
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let traj = decode_sprite_trajectory(&mut br, 2).unwrap();
        assert_eq!(traj.count, 2);
        assert_eq!(traj.points[0], [1, -1]);
        assert_eq!(traj.points[1], [3, -2]);
    }

    #[test]
    fn stationary_trajectory_consumes_nothing() {
        let buf = [0xFFu8; 2];
        let mut br = BitReader::new(&buf);
        let traj = decode_sprite_trajectory(&mut br, 0).unwrap();
        assert_eq!(traj.count, 0);
        assert_eq!(traj, SpriteTrajectory::stationary());
    }

    #[test]
    fn too_many_points_rejected() {
        let buf = [0u8; 4];
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_sprite_trajectory(&mut br, 4).unwrap_err(),
            SpriteTrajectoryError::TooManyPoints(4)
        );
    }

    #[test]
    fn static_trajectory_four_points() {
        // 4 warping points (perspective). du/dv pairs:
        // (+1,-1),(+3,-2),(-1,-3),(+2,+2).
        let mut w = BitWriter::new();
        w.write_warping(1, 1); // du0 = +1
        w.write_warping(1, 0); // dv0 = -1
        w.write_warping(2, 0b11); // du1 = +3
        w.write_warping(2, 0b01); // dv1 = -2
        w.write_warping(1, 0); // du2 = -1
        w.write_warping(2, 0b00); // dv2 = -3
        w.write_warping(2, 0b10); // du3 = +2
        w.write_warping(2, 0b10); // dv3 = +2
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let (count, points) = decode_sprite_trajectory_static(&mut br, 4).unwrap();
        assert_eq!(count, 4);
        assert_eq!(points[0], [1, -1]);
        assert_eq!(points[1], [3, -2]);
        assert_eq!(points[2], [-1, -3]);
        assert_eq!(points[3], [2, 2]);
    }

    #[test]
    fn static_trajectory_rejects_five_points() {
        let buf = [0u8; 8];
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_sprite_trajectory_static(&mut br, 5).unwrap_err(),
            SpriteTrajectoryError::TooManyPoints(5)
        );
    }

    #[test]
    fn static_trajectory_zero_points_consumes_nothing() {
        let buf = [0xFFu8; 2];
        let mut br = BitReader::new(&buf);
        let (count, points) = decode_sprite_trajectory_static(&mut br, 0).unwrap();
        assert_eq!(count, 0);
        assert_eq!(points, [[0, 0]; MAX_STATIC_WARPING_POINTS]);
    }
}

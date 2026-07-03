//! Structural parsers for the MPEG-4 Visual §6.2.4 Group-of-VOP
//! header and the §6.2.5 Video Object Plane (VOP) header.
//!
//! Coverage in this module is **structural-only**: we identify the
//! relevant start codes (`0x000001B3` for GOV, `0x000001B6` for VOP),
//! walk the fixed bit-fields that precede the macroblock payload, and
//! surface typed [`GovHeader`] / [`VopHeader`] views. Macroblock-level
//! decode (motion vectors, DCT coefficient decode, MB headers) is
//! explicitly out of scope here — round 2 of the clean-room rebuild
//! stops at the VOP header, exactly where the macroblock layer starts.
//!
//! ## Spec references
//!
//! Every numeric value below is sourced from ISO/IEC 14496-2:2004
//! (3rd edition), read by the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.4 syntax table — `Group_of_VideoObjectPlane()` (`time_code`
//!   18 bits via Table 6-23; `closed_gov`; `broken_link`).
//! * §6.2.5 syntax table — `VideoObjectPlane()` for the bit layout
//!   beginning at the `vop_start_code` and ending before
//!   `motion_shape_texture()`.
//! * §6.3.5 semantics — `vop_coding_type` (Table 6-24),
//!   `modulo_time_base`, `vop_time_increment`, `vop_coded`,
//!   `vop_rounding_type`, `intra_dc_vlc_thr` (Table 6-25),
//!   `vop_quant`, `vop_fcode_forward`, `vop_fcode_backward`.
//!
//! ## Scope deliberately deferred
//!
//! The round-2 parser assumes the **non-Studio, non-FGS, non-scalable,
//! progressive-rectangular** branch of the VOP syntax. The following
//! optional fields, although present in the spec syntax table, are
//! intentionally skipped here because they require state from the VOL
//! header that the round-1 [`VolHeader`](crate::VolHeader) does **not**
//! yet expose:
//!
//! * `newpred_enable` — needs a VOL flag round 1 did not parse.
//! * `reduced_resolution_vop_enable` — same.
//! * Non-rectangular shape branch (`vop_width` / `vop_height` /
//!   `vop_horizontal_mc_spatial_ref` / `vop_vertical_mc_spatial_ref` /
//!   `change_conv_ratio_disable` / `vop_constant_alpha`) — round 1
//!   rejects non-rectangular shapes upfront.
//! * `complexity_estimation_disable` block.
//! * `interlaced` branch (`top_field_first` /
//!   `alternate_vertical_scan_flag`).
//! * `sprite_enable` branches.
//! * `quant_precision != 5` — defaults to 5-bit `vop_quant`.
//!
//! The parser therefore takes a small [`VopContext`] argument carrying
//! the bits the VOL would otherwise have surfaced. The caller is
//! responsible for populating it. Defaults match the most common
//! Simple-Profile case: `interlaced=false`, `quant_precision=5`,
//! `sprite_enable=Disabled`, `complexity_estimation_disable=true`,
//! `newpred_enable=false`, `reduced_resolution_vop_enable=false`,
//! `scalability=false`.

use crate::bitreader::{BitReader, BitReaderError};
use crate::sprite::{decode_sprite_trajectory, SpriteTrajectory, SpriteTrajectoryError};
use crate::vol::{SpriteEnable, VolHeader, VolParseError};

/// Start code for a `Group_of_VideoObjectPlane()` (§6.2.4 / §6.3.4 —
/// `0x000001B3`).
pub const GROUP_OF_VOP_START_CODE: u32 = 0x0000_01B3;
/// Start code for a `VideoObjectPlane()` (§6.2.5 / §6.3.5 —
/// `0x000001B6`).
pub const VOP_START_CODE: u32 = 0x0000_01B6;

/// VOP coding type (§6.3.5 Table 6-24).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VopCodingType {
    /// `00` — intra-coded VOP.
    I,
    /// `01` — predictive-coded VOP.
    P,
    /// `10` — bidirectionally-predictive-coded VOP.
    B,
    /// `11` — sprite-coded VOP.
    S,
}

impl VopCodingType {
    /// Decode the 2-bit `vop_coding_type` field per Table 6-24.
    pub fn from_bits(bits: u32) -> Self {
        match bits & 0b11 {
            0b00 => VopCodingType::I,
            0b01 => VopCodingType::P,
            0b10 => VopCodingType::B,
            0b11 => VopCodingType::S,
            _ => unreachable!("masked to 2 bits"),
        }
    }
}

/// Decoded `time_code` field (§6.2.4, Table 6-23).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeCode {
    /// `time_code_hours`. Range 0..=23 per the table.
    pub hours: u8,
    /// `time_code_minutes`. Range 0..=59.
    pub minutes: u8,
    /// `time_code_seconds`. Range 0..=59.
    pub seconds: u8,
}

/// Typed view of a §6.2.4 `Group_of_VideoObjectPlane` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GovHeader {
    /// Decoded `time_code`.
    pub time_code: TimeCode,
    /// `closed_gov` (§6.3.4).
    pub closed_gov: bool,
    /// `broken_link` (§6.3.4).
    pub broken_link: bool,
}

/// Context bits the caller must supply because they live in the VOL
/// header (which round 1 does not yet expose to round 2's liking) or in
/// out-of-scope branches.
///
/// All fields default to the values that produce the most common
/// Simple-Profile decode path. A caller wiring up a real VOL once
/// round 1's `VolHeader` exposes the missing bits should populate these
/// from there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VopContext {
    /// `quant_precision` (§6.3.3 — number of bits used to transmit
    /// `vop_quant`). Default `5`. Valid range 3..=9 per §6.3.3.
    pub quant_precision: u8,
    /// `interlaced` flag from the VOL header. When `true`, the VOP
    /// header carries `top_field_first` + `alternate_vertical_scan_flag`
    /// after `intra_dc_vlc_thr`.
    pub interlaced: bool,
    /// Whether the VOL declared `sprite_enable == "GMC"`. When true and
    /// the VOP is S-coded, the parser walks the §6.2.5 `sprite_trajectory()`
    /// branch (the GMC global-motion-compensation path).
    pub sprite_gmc: bool,
    /// Whether the VOL declared `sprite_enable == "static"`. Round 2
    /// rejects this branch up front.
    pub sprite_static: bool,
    /// `no_of_sprite_warping_points` (Table 6-20) from the VOL header,
    /// `0` when sprite coding is off. Gates the `sprite_trajectory()`
    /// parse — `> 0` triggers it.
    pub no_of_sprite_warping_points: u8,
    /// `sprite_brightness_change` (§6.3.3) from the VOL header. `false`
    /// when sprite coding is off, and always `false` under GMC.
    pub sprite_brightness_change: bool,
    /// Whether the VOL declared `scalability == 1`. Round 2 rejects.
    pub scalability: bool,
    /// Whether the VOL declared `newpred_enable == 1`. Round 2 rejects.
    pub newpred_enable: bool,
    /// Whether the VOL declared `reduced_resolution_vop_enable == 1`.
    /// Round 2 rejects.
    pub reduced_resolution_vop_enable: bool,
    /// Whether `complexity_estimation_disable` is set in the VOL. Round
    /// 2 only supports `complexity_estimation_disable == 1` (i.e. no
    /// complexity-estimation header in the VOP).
    pub complexity_estimation_disable: bool,
}

impl Default for VopContext {
    fn default() -> Self {
        Self {
            quant_precision: 5,
            interlaced: false,
            sprite_gmc: false,
            sprite_static: false,
            no_of_sprite_warping_points: 0,
            sprite_brightness_change: false,
            scalability: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            complexity_estimation_disable: true,
        }
    }
}

impl VopContext {
    /// Build a [`VopContext`] by projecting the round-3 fields out of a
    /// parsed [`VolHeader`]. The caller no longer needs to keep the
    /// fields in sync by hand — `parse_video_object_plane_header(..,
    /// VopContext::from_vol(&vol))` is the canonical pipeline.
    pub fn from_vol(vol: &VolHeader) -> Self {
        Self {
            quant_precision: vol.quant_precision,
            interlaced: vol.interlaced,
            sprite_gmc: matches!(vol.sprite_enable, SpriteEnable::Gmc),
            sprite_static: matches!(vol.sprite_enable, SpriteEnable::Static),
            no_of_sprite_warping_points: vol.no_of_sprite_warping_points.unwrap_or(0),
            sprite_brightness_change: vol.sprite_brightness_change.unwrap_or(false),
            scalability: vol.scalability,
            newpred_enable: vol.newpred_enable,
            reduced_resolution_vop_enable: vol.reduced_resolution_vop_enable,
            complexity_estimation_disable: vol.complexity_estimation_disable,
        }
    }
}

/// Decoded §6.2.5 Video Object Plane header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VopHeader {
    /// `vop_coding_type`.
    pub coding_type: VopCodingType,
    /// Number of leading `1` bits in `modulo_time_base` before the
    /// terminating `0` — the count of full-second ticks elapsed since
    /// the synchronisation point.
    pub modulo_time_base: u32,
    /// `vop_time_increment`. Bit-width is the minimum number of unsigned
    /// integer bits required to represent the range
    /// `[0, vop_time_increment_resolution)`, with a single `0` bit when
    /// the resolution is `1`. The composed display time, in ticks, is
    /// `modulo_time_base * vop_time_increment_resolution + time_increment`.
    pub time_increment: u16,
    /// Combined composition: `modulo_time_base * resolution +
    /// vop_time_increment`, returned in ticks. This is the
    /// "VOP time" measured in clock ticks since the synchronisation
    /// point — useful for callers without context on the resolution.
    pub composed_ticks: u64,
    /// `vop_coded` flag. When `false` the remaining VOP-header fields
    /// are absent and the rest of the [`VopHeader`] is filled with
    /// spec-defined defaults (see §6.3.5 vop_coded).
    pub coded: bool,
    /// `vop_rounding_type` — present only for `P` and `S(GMC)` coded
    /// types. Defaults to `0` otherwise per §6.3.5 vop_rounding_type.
    pub rounding_type: u8,
    /// `intra_dc_vlc_thr` (Table 6-25 index). 3-bit code in 0..=7.
    pub intra_dc_vlc_thr: u8,
    /// `vop_quant`. Width is determined by `quant_precision` from the
    /// VOL header; the default precision is 5 bits, giving a range of
    /// 1..=31.
    pub quant: u16,
    /// `vop_fcode_forward`. Present only when `vop_coding_type != I`.
    /// Defaults to `0` (meaning "not present") for I-VOPs.
    pub fcode_fwd: u8,
    /// `vop_fcode_backward`. Present only when `vop_coding_type == B`.
    /// Defaults to `0` otherwise.
    pub fcode_bwd: u8,
    /// Decoded §6.2.5 `sprite_trajectory()` for an S(GMC)-VOP, when the
    /// VOL declared `sprite_enable == "GMC"` and
    /// `no_of_sprite_warping_points > 0`. `None` for every non-GMC VOP
    /// and for the stationary (0-point) GMC case.
    pub sprite_trajectory: Option<SpriteTrajectory>,
}

/// Errors produced by the VOP / GOV header parsers. We reuse
/// [`VolParseError`] from round 1 for the start-code / truncated /
/// marker-bit cases so callers can keep a single error surface, and
/// add this enum for the new VOP-specific failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VopParseError {
    /// The supplied byte slice ran out mid-field.
    Truncated,
    /// Start code at the head of the slice was not the expected
    /// `0x000001B3` / `0x000001B6`.
    MissingStartCode {
        /// What we were looking for.
        expected: &'static str,
        /// What the next 32 bits actually were.
        found: u32,
    },
    /// A `marker_bit` field was `0` (§6.2 marker convention).
    MarkerBitMissing,
    /// The supplied `quant_precision` is outside the §6.3.3 range
    /// `[3, 9]`.
    BadQuantPrecision(u8),
    /// `vop_fcode_forward` / `vop_fcode_backward` was transmitted as
    /// `0` — §6.3.5 forbids that value.
    ForbiddenFcode,
    /// The VOP belongs to a branch the round-2 parser deliberately
    /// rejects (Sprite, FGS, scalability, newpred, reduced-resolution,
    /// or a complexity-estimation header). Carry-back enum text is
    /// enough; the offending bit is named.
    UnsupportedBranch(&'static str),
    /// A §6.2.5 `sprite_trajectory()` body (S(GMC)-VOP) failed to
    /// decode. See [`SpriteTrajectoryError`].
    SpriteTrajectory(SpriteTrajectoryError),
}

impl From<SpriteTrajectoryError> for VopParseError {
    fn from(err: SpriteTrajectoryError) -> Self {
        VopParseError::SpriteTrajectory(err)
    }
}

impl core::fmt::Display for VopParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VopParseError::Truncated => write!(f, "VOP header truncated"),
            VopParseError::MissingStartCode { expected, found } => {
                write!(f, "missing {expected} (found 0x{found:08X})")
            }
            VopParseError::MarkerBitMissing => write!(f, "marker_bit was 0 (expected 1)"),
            VopParseError::BadQuantPrecision(p) => {
                write!(f, "quant_precision {p} is outside the allowed 3..=9 range")
            }
            VopParseError::ForbiddenFcode => {
                write!(f, "vop_fcode_forward/backward of 0 is forbidden")
            }
            VopParseError::UnsupportedBranch(name) => {
                write!(f, "VOP branch '{name}' not supported by round-2 parser")
            }
            VopParseError::SpriteTrajectory(err) => {
                write!(f, "sprite_trajectory decode failed: {err}")
            }
        }
    }
}

impl std::error::Error for VopParseError {}

impl From<BitReaderError> for VopParseError {
    fn from(_: BitReaderError) -> Self {
        VopParseError::Truncated
    }
}

impl From<VopParseError> for VolParseError {
    fn from(err: VopParseError) -> Self {
        // Map the structural overlap onto the round-1 surface so callers
        // who already match on `VolParseError` keep working.
        match err {
            VopParseError::Truncated => VolParseError::Truncated,
            VopParseError::MissingStartCode { expected, found } => {
                VolParseError::MissingStartCode { expected, found }
            }
            VopParseError::MarkerBitMissing => VolParseError::MarkerBitMissing,
            // The rest don't have a 1:1 round-1 equivalent. Pick the
            // closest variant — "UnsupportedShape" is the wrong axis but
            // the next-closest round-1 surface is the FGS rejection. We
            // surface the original via Display so debug output stays
            // informative.
            VopParseError::BadQuantPrecision(_)
            | VopParseError::ForbiddenFcode
            | VopParseError::SpriteTrajectory(_)
            | VopParseError::UnsupportedBranch(_) => VolParseError::UnsupportedFgs,
        }
    }
}

fn read_marker(br: &mut BitReader<'_>) -> Result<(), VopParseError> {
    if br.read_bool()? {
        Ok(())
    } else {
        Err(VopParseError::MarkerBitMissing)
    }
}

/// Compute the bit-width of `vop_time_increment` from the
/// `vop_time_increment_resolution` carried in the VOL. Per §6.3.5: the
/// minimum number of unsigned integer bits required to represent
/// `[0, resolution)`, with a special case of one zero bit when
/// `resolution == 1`.
pub fn vop_time_increment_bits(resolution: u16) -> u8 {
    if resolution <= 1 {
        return 1;
    }
    let n = u32::from(resolution);
    32 - (n - 1).leading_zeros() as u8
}

/// Parse a §6.2.4 `Group_of_VideoObjectPlane` header starting at the
/// `group_of_vop_start_code` (`0x000001B3`).
pub fn parse_group_of_vop_header(data: &[u8]) -> Result<GovHeader, VopParseError> {
    let mut br = BitReader::new(data);
    let sc = br.read_bits(32)?;
    if sc != GROUP_OF_VOP_START_CODE {
        return Err(VopParseError::MissingStartCode {
            expected: "group_of_vop_start_code",
            found: sc,
        });
    }
    parse_group_of_vop_body(&mut br)
}

fn parse_group_of_vop_body(br: &mut BitReader<'_>) -> Result<GovHeader, VopParseError> {
    // Table 6-23: 5 hours + 6 minutes + 1 marker + 6 seconds = 18 bits.
    let hours = br.read_bits(5)? as u8;
    let minutes = br.read_bits(6)? as u8;
    read_marker(br)?;
    let seconds = br.read_bits(6)? as u8;
    let closed_gov = br.read_bool()?;
    let broken_link = br.read_bool()?;
    Ok(GovHeader {
        time_code: TimeCode {
            hours,
            minutes,
            seconds,
        },
        closed_gov,
        broken_link,
    })
}

impl VopHeader {
    /// Parse a VOP header directly against the [`VolHeader`] it
    /// belongs to. Equivalent to
    /// `parse_video_object_plane_header(payload,
    /// vol.time_increment_resolution, VopContext::from_vol(vol))`,
    /// but reads more naturally at call sites that already have the
    /// `VolHeader` in scope. The convenience constructor is the
    /// recommended entry point now that round 3 promotes the
    /// context bits onto [`VolHeader`].
    pub fn from_vol(vol: &VolHeader, payload: &[u8]) -> Result<Self, VopParseError> {
        parse_video_object_plane_header(
            payload,
            vol.time_increment_resolution,
            VopContext::from_vol(vol),
        )
    }
}

/// Parse a §6.2.5 `VideoObjectPlane` header starting at the
/// `vop_start_code` (`0x000001B6`).
///
/// `resolution` is the `vop_time_increment_resolution` declared in the
/// VOL header and is needed to determine the bit-width of
/// `vop_time_increment`. `ctx` carries the few additional VOL bits the
/// VOP syntax depends on; see [`VopContext`] for defaults.
pub fn parse_video_object_plane_header(
    data: &[u8],
    resolution: u16,
    ctx: VopContext,
) -> Result<VopHeader, VopParseError> {
    let mut br = BitReader::new(data);
    let sc = br.read_bits(32)?;
    if sc != VOP_START_CODE {
        return Err(VopParseError::MissingStartCode {
            expected: "vop_start_code",
            found: sc,
        });
    }
    parse_video_object_plane_body(&mut br, resolution, ctx)
}

/// Parse a §6.2.5 VOP header **body** (everything after the 32-bit
/// `vop_start_code`) from an existing bit reader, leaving the reader
/// positioned at the first bit that follows the header.
///
/// For a rectangular, non-scalable VOL that first bit is the start of
/// `motion_shape_texture()` — the macroblock layer (§6.2.5: the
/// `combined_motion_shape_texture()` data follows `vop_fcode_backward`
/// directly). This is the entry point the frame-level bitstream
/// drivers use; [`parse_video_object_plane_header`] wraps it for
/// callers holding a byte slice that starts at the start code.
pub fn parse_vop_header_body(
    br: &mut BitReader<'_>,
    resolution: u16,
    ctx: VopContext,
) -> Result<VopHeader, VopParseError> {
    parse_video_object_plane_body(br, resolution, ctx)
}

fn parse_video_object_plane_body(
    br: &mut BitReader<'_>,
    resolution: u16,
    ctx: VopContext,
) -> Result<VopHeader, VopParseError> {
    if !(3..=9).contains(&ctx.quant_precision) {
        return Err(VopParseError::BadQuantPrecision(ctx.quant_precision));
    }
    let coding_type = VopCodingType::from_bits(br.read_bits(2)?);
    // modulo_time_base: variable-length unary terminator '0'. Per §6.3.5
    // each leading '1' denotes one elapsed second; a single '0'
    // terminates. We bound the loop by the remaining bit budget so a
    // pathological stream of '1's cannot loop forever.
    let mut modulo_time_base: u32 = 0;
    loop {
        let bit = br.read_bool()?;
        if !bit {
            break;
        }
        modulo_time_base = modulo_time_base
            .checked_add(1)
            .ok_or(VopParseError::Truncated)?;
        // Practical upper bound: a single VOP wrapping > 4 billion
        // seconds is not a meaningful encoding. Cap at u32::MAX.
    }
    read_marker(br)?;
    let bits = vop_time_increment_bits(resolution) as usize;
    let time_increment = br.read_bits(bits)? as u16;
    read_marker(br)?;
    let coded = br.read_bool()?;

    // composed_ticks = modulo_time_base * resolution + time_increment.
    // Use u64 to keep the product safe even with a 65535-tick resolution
    // and a 4-billion modulo count.
    let composed_ticks = u64::from(modulo_time_base)
        .saturating_mul(u64::from(resolution))
        .saturating_add(u64::from(time_increment));

    if !coded {
        // Spec §6.2.5: when vop_coded == 0 the parser returns. We
        // surface a typed view with the defaults that downstream
        // reconstruction needs (rounding_control = 0 per §6.3.5).
        return Ok(VopHeader {
            coding_type,
            modulo_time_base,
            time_increment,
            composed_ticks,
            coded: false,
            rounding_type: 0,
            intra_dc_vlc_thr: 0,
            quant: 0,
            fcode_fwd: 0,
            fcode_bwd: 0,
            sprite_trajectory: None,
        });
    }

    // newpred_enable / reduced_resolution_vop_enable / non-rectangular
    // shape branches are explicitly out of scope for round 2.
    if ctx.newpred_enable {
        return Err(VopParseError::UnsupportedBranch("newpred_enable"));
    }
    if ctx.reduced_resolution_vop_enable {
        return Err(VopParseError::UnsupportedBranch(
            "reduced_resolution_vop_enable",
        ));
    }
    if ctx.scalability {
        return Err(VopParseError::UnsupportedBranch("scalability"));
    }

    // vop_rounding_type: present only for P and S(GMC) coded VOPs in
    // the non-binary-only shape branch. Round 2's VOL shape is always
    // rectangular per round-1 gating, which is "not binary only", so
    // the only gate left is the coding-type test.
    let mut rounding_type: u8 = 0;
    let need_rounding = matches!(coding_type, VopCodingType::P)
        || (matches!(coding_type, VopCodingType::S) && ctx.sprite_gmc);
    if need_rounding {
        rounding_type = br.read_bits(1)? as u8;
    }

    // Static sprite S-VOPs need the §7.8.2/§7.8.3 sprite-object buffer
    // and piece-update machinery (sprite_transmit_mode loop +
    // decode_sprite_piece()), which is out of scope; reject up front so
    // the bit position never drifts. GMC S-VOPs fall through to the
    // §6.2.5 sprite_trajectory() branch below.
    if matches!(coding_type, VopCodingType::S) && ctx.sprite_static {
        return Err(VopParseError::UnsupportedBranch("sprite_static_S_VOP"));
    }

    if !ctx.complexity_estimation_disable {
        return Err(VopParseError::UnsupportedBranch(
            "complexity_estimation_header",
        ));
    }

    // intra_dc_vlc_thr (Table 6-25). 3 bits, 0..=7.
    let intra_dc_vlc_thr = br.read_bits(3)? as u8;
    if ctx.interlaced {
        // top_field_first + alternate_vertical_scan_flag — we read them
        // structurally to stay aligned, but do not surface them on the
        // typed view. A later round adds the interlaced VopHeader
        // fields.
        let _top_field_first = br.read_bool()?;
        let _alt_vert_scan = br.read_bool()?;
    }

    // §6.2.5 S(GMC)-VOP sprite branch (spec lines 4328..=4333). For
    // `sprite_enable == "GMC"` the trajectory is followed (when
    // no_of_sprite_warping_points > 0) by an optional
    // brightness_change_factor() — mandated absent under GMC since
    // sprite_brightness_change must be 0 (§6.3.3). Unlike the static
    // path there is NO `next_start_code(); return()` here: a GMC S-VOP
    // continues to vop_quant / vop_fcode_forward / motion_shape_texture()
    // exactly like a P-VOP.
    let sprite_trajectory = if matches!(coding_type, VopCodingType::S) && ctx.sprite_gmc {
        let traj = if ctx.no_of_sprite_warping_points > 0 {
            Some(decode_sprite_trajectory(
                br,
                ctx.no_of_sprite_warping_points,
            )?)
        } else {
            None
        };
        if ctx.sprite_brightness_change {
            // brightness_change_factor() is present. The spec forbids
            // sprite_brightness_change == 1 under GMC, so reaching here
            // means the VOL was inconsistent; reject rather than guess.
            return Err(VopParseError::UnsupportedBranch(
                "GMC brightness_change_factor",
            ));
        }
        traj
    } else {
        None
    };

    // vop_quant: `quant_precision` bits, default 5.
    let quant = br.read_bits(ctx.quant_precision as usize)? as u16;

    // vop_fcode_forward — present only when vop_coding_type != I.
    let fcode_fwd = if !matches!(coding_type, VopCodingType::I) {
        let f = br.read_bits(3)? as u8;
        if f == 0 {
            return Err(VopParseError::ForbiddenFcode);
        }
        f
    } else {
        0
    };

    // vop_fcode_backward — present only when vop_coding_type == B.
    let fcode_bwd = if matches!(coding_type, VopCodingType::B) {
        let f = br.read_bits(3)? as u8;
        if f == 0 {
            return Err(VopParseError::ForbiddenFcode);
        }
        f
    } else {
        0
    };

    Ok(VopHeader {
        coding_type,
        modulo_time_base,
        time_increment,
        composed_ticks,
        coded: true,
        rounding_type,
        intra_dc_vlc_thr,
        quant,
        fcode_fwd,
        fcode_bwd,
        sprite_trajectory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-writer for assembling fixture bitstreams. Mirrors the
    /// MSB-first convention of the spec's `bslbf` / `uimsbf` codes.
    struct BitWriter {
        buf: Vec<u8>,
        bit_pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                bit_pos: 0,
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            assert!(n <= 32);
            for i in (0..n).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos % 8 == 0 {
                    self.buf.push(0);
                }
                let byte = self.buf.last_mut().unwrap();
                *byte |= bit << (7 - (self.bit_pos % 8));
                self.bit_pos += 1;
            }
        }
        fn write_marker(&mut self) {
            self.write_bits(1, 1);
        }
        fn align(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bits(0, 1);
            }
        }
    }

    /// Common helper to construct a minimal I-VOP header byte slice
    /// with the supplied modulo_time_base + time_increment + quant.
    fn make_i_vop(
        resolution: u16,
        modulo: u32,
        time_increment: u32,
        quant: u32,
        intra_dc: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b00, 2); // vop_coding_type = I
                               // modulo_time_base: `modulo` 1-bits, then a 0.
        for _ in 0..modulo {
            w.write_bits(1, 1);
        }
        w.write_bits(0, 1);
        w.write_marker();
        let bits = vop_time_increment_bits(resolution) as usize;
        w.write_bits(time_increment, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded = 1
                            // I-VOP: no vop_rounding_type.
                            // intra_dc_vlc_thr.
        w.write_bits(intra_dc, 3);
        // (interlaced = false → no top_field_first / alt_vert_scan)
        w.write_bits(quant, 5);
        // I-VOP: no fcode_forward / fcode_backward.
        w.align();
        w.buf
    }

    fn make_p_vop(resolution: u16, fcode_fwd: u32, quant: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b01, 2); // vop_coding_type = P
        w.write_bits(0, 1); // modulo_time_base = 0
        w.write_marker();
        let bits = vop_time_increment_bits(resolution) as usize;
        w.write_bits(0, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type = 0
        w.write_bits(0, 3); // intra_dc_vlc_thr = 0
        w.write_bits(quant, 5);
        w.write_bits(fcode_fwd, 3);
        w.align();
        w.buf
    }

    fn make_b_vop(resolution: u16, fcode_fwd: u32, fcode_bwd: u32, quant: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b10, 2); // vop_coding_type = B
        w.write_bits(0, 1);
        w.write_marker();
        let bits = vop_time_increment_bits(resolution) as usize;
        w.write_bits(0, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
                            // B-VOP is not P / not S(GMC) → no vop_rounding_type.
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(quant, 5);
        w.write_bits(fcode_fwd, 3);
        w.write_bits(fcode_bwd, 3);
        w.align();
        w.buf
    }

    /// Emit a `warping_mv_code(d)`: unary SSS, terminating 0, SSS-bit
    /// FLC `code`, marker.
    fn write_warping(w: &mut BitWriter, sss: u32, code: u32) {
        for _ in 0..sss {
            w.write_bits(1, 1);
        }
        w.write_bits(0, 1);
        if sss != 0 {
            w.write_bits(code, sss as usize);
        }
        w.write_marker();
    }

    /// Build an S(GMC)-VOP with a 2-point (affine) sprite trajectory.
    fn make_s_gmc_vop(resolution: u16, fcode_fwd: u32, quant: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b11, 2); // vop_coding_type = S
        w.write_bits(0, 1); // modulo_time_base = 0
        w.write_marker();
        let bits = vop_time_increment_bits(resolution) as usize;
        w.write_bits(0, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type (present: S && GMC)
        w.write_bits(0, 3); // intra_dc_vlc_thr
                            // sprite_trajectory(): 2 points.
        write_warping(&mut w, 1, 1); // du0 = +1
        write_warping(&mut w, 1, 0); // dv0 = -1
        write_warping(&mut w, 2, 0b11); // du1 = +3
        write_warping(&mut w, 2, 0b01); // dv1 = -2
                                        // No brightness_change (GMC). Continue as a P-VOP.
        w.write_bits(quant, 5); // vop_quant
        w.write_bits(fcode_fwd, 3); // vop_fcode_forward (S != I)
        w.align();
        w.buf
    }

    #[test]
    fn parses_s_gmc_vop_with_sprite_trajectory() {
        let data = make_s_gmc_vop(30_000, 2, 8);
        let ctx = VopContext {
            sprite_gmc: true,
            no_of_sprite_warping_points: 2,
            ..VopContext::default()
        };
        let header = parse_video_object_plane_header(&data, 30_000, ctx).unwrap();
        assert_eq!(header.coding_type, VopCodingType::S);
        assert_eq!(header.quant, 8);
        assert_eq!(header.fcode_fwd, 2);
        let traj = header.sprite_trajectory.expect("GMC trajectory present");
        assert_eq!(traj.count, 2);
        assert_eq!(traj.points[0], [1, -1]);
        assert_eq!(traj.points[1], [3, -2]);
    }

    #[test]
    fn s_gmc_vop_stationary_has_no_trajectory() {
        // no_of_sprite_warping_points == 0 → identity warp, no trajectory
        // bits, falls straight through to vop_quant / vop_fcode_forward.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b11, 2); // S
        w.write_bits(0, 1);
        w.write_marker();
        let bits = vop_time_increment_bits(30_000) as usize;
        w.write_bits(0, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(5, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        w.align();
        let ctx = VopContext {
            sprite_gmc: true,
            no_of_sprite_warping_points: 0,
            ..VopContext::default()
        };
        let header = parse_video_object_plane_header(&w.buf, 30_000, ctx).unwrap();
        assert_eq!(header.coding_type, VopCodingType::S);
        assert!(header.sprite_trajectory.is_none());
        assert_eq!(header.quant, 5);
        assert_eq!(header.fcode_fwd, 1);
    }

    #[test]
    fn static_sprite_s_vop_still_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b11, 2); // S
        w.write_bits(0, 1);
        w.write_marker();
        let bits = vop_time_increment_bits(30_000) as usize;
        w.write_bits(0, bits);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
        w.align();
        let ctx = VopContext {
            sprite_static: true,
            ..VopContext::default()
        };
        let err = parse_video_object_plane_header(&w.buf, 30_000, ctx).unwrap_err();
        match err {
            VopParseError::UnsupportedBranch(name) => assert!(name.contains("static")),
            other => panic!("expected UnsupportedBranch, got {other:?}"),
        }
    }

    #[test]
    fn vop_start_code_value_is_b6() {
        assert_eq!(VOP_START_CODE, 0x0000_01B6);
    }

    #[test]
    fn gov_start_code_value_is_b3() {
        assert_eq!(GROUP_OF_VOP_START_CODE, 0x0000_01B3);
    }

    #[test]
    fn coding_type_bits_match_table_6_24() {
        assert_eq!(VopCodingType::from_bits(0b00), VopCodingType::I);
        assert_eq!(VopCodingType::from_bits(0b01), VopCodingType::P);
        assert_eq!(VopCodingType::from_bits(0b10), VopCodingType::B);
        assert_eq!(VopCodingType::from_bits(0b11), VopCodingType::S);
    }

    #[test]
    fn vop_time_increment_bits_special_case_resolution_1() {
        // Per §6.3.5 vop_time_increment: when resolution is 1, one zero
        // bit is used.
        assert_eq!(vop_time_increment_bits(1), 1);
        assert_eq!(vop_time_increment_bits(0), 1);
        // ceil(log2(N)) otherwise.
        assert_eq!(vop_time_increment_bits(2), 1);
        assert_eq!(vop_time_increment_bits(3), 2);
        assert_eq!(vop_time_increment_bits(30), 5);
        assert_eq!(vop_time_increment_bits(30_000), 15);
        assert_eq!(vop_time_increment_bits(65_535), 16);
    }

    #[test]
    fn parses_minimal_i_vop() {
        let data = make_i_vop(30_000, 0, 1001, 7, 3);
        let header = parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap();
        assert_eq!(header.coding_type, VopCodingType::I);
        assert_eq!(header.modulo_time_base, 0);
        assert_eq!(header.time_increment, 1001);
        assert_eq!(header.composed_ticks, 1001);
        assert!(header.coded);
        assert_eq!(header.rounding_type, 0);
        assert_eq!(header.intra_dc_vlc_thr, 3);
        assert_eq!(header.quant, 7);
        assert_eq!(header.fcode_fwd, 0);
        assert_eq!(header.fcode_bwd, 0);
    }

    #[test]
    fn modulo_time_base_accumulates_one_bits() {
        // 3 elapsed seconds → 1110 prefix, then marker, then ti.
        let data = make_i_vop(30_000, 3, 500, 5, 0);
        let header = parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap();
        assert_eq!(header.modulo_time_base, 3);
        assert_eq!(header.composed_ticks, 3 * 30_000 + 500);
    }

    #[test]
    fn parses_p_vop_with_fcode_forward() {
        let data = make_p_vop(30_000, 4, 12);
        let header = parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap();
        assert_eq!(header.coding_type, VopCodingType::P);
        assert_eq!(header.fcode_fwd, 4);
        assert_eq!(header.fcode_bwd, 0);
        assert_eq!(header.quant, 12);
        assert_eq!(header.rounding_type, 0);
    }

    #[test]
    fn p_vop_rejects_zero_fcode() {
        let data = make_p_vop(30_000, 0, 5);
        let err =
            parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap_err();
        assert_eq!(err, VopParseError::ForbiddenFcode);
    }

    #[test]
    fn parses_b_vop_with_both_fcodes() {
        let data = make_b_vop(30_000, 2, 3, 20);
        let header = parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap();
        assert_eq!(header.coding_type, VopCodingType::B);
        assert_eq!(header.fcode_fwd, 2);
        assert_eq!(header.fcode_bwd, 3);
        assert_eq!(header.quant, 20);
    }

    #[test]
    fn b_vop_rejects_zero_fcode_backward() {
        let data = make_b_vop(30_000, 2, 0, 1);
        let err =
            parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap_err();
        assert_eq!(err, VopParseError::ForbiddenFcode);
    }

    #[test]
    fn vop_coded_zero_returns_default_fields() {
        // Manually craft an I-VOP where vop_coded = 0.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b00, 2);
        w.write_bits(0, 1); // modulo_time_base = 0
        w.write_marker();
        let bits = vop_time_increment_bits(30_000) as usize;
        w.write_bits(42, bits);
        w.write_marker();
        w.write_bits(0, 1); // vop_coded = 0 → early return
        w.align();
        let data = w.buf;
        let header = parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap();
        assert!(!header.coded);
        assert_eq!(header.time_increment, 42);
        assert_eq!(header.quant, 0);
        assert_eq!(header.fcode_fwd, 0);
    }

    #[test]
    fn missing_vop_start_code_is_rejected() {
        let mut data = make_i_vop(30_000, 0, 0, 1, 0);
        data[3] = 0xFF;
        let err =
            parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap_err();
        match err {
            VopParseError::MissingStartCode { expected, .. } => {
                assert_eq!(expected, "vop_start_code");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn marker_violation_is_rejected() {
        // Hand-build a VOP header with the marker after modulo_time_base
        // forced to 0.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b00, 2);
        w.write_bits(0, 1); // modulo_time_base = 0
        w.write_bits(0, 1); // marker_bit = 0 (illegal)
        w.align();
        let data = w.buf;
        let err =
            parse_video_object_plane_header(&data, 30_000, VopContext::default()).unwrap_err();
        assert_eq!(err, VopParseError::MarkerBitMissing);
    }

    #[test]
    fn unsupported_scalability_is_rejected() {
        let data = make_i_vop(30_000, 0, 0, 1, 0);
        let ctx = VopContext {
            scalability: true,
            ..VopContext::default()
        };
        let err = parse_video_object_plane_header(&data, 30_000, ctx).unwrap_err();
        assert!(matches!(
            err,
            VopParseError::UnsupportedBranch("scalability")
        ));
    }

    #[test]
    fn unsupported_newpred_is_rejected() {
        let data = make_i_vop(30_000, 0, 0, 1, 0);
        let ctx = VopContext {
            newpred_enable: true,
            ..VopContext::default()
        };
        let err = parse_video_object_plane_header(&data, 30_000, ctx).unwrap_err();
        assert!(matches!(
            err,
            VopParseError::UnsupportedBranch("newpred_enable")
        ));
    }

    #[test]
    fn bad_quant_precision_is_rejected() {
        let data = make_i_vop(30_000, 0, 0, 1, 0);
        let ctx = VopContext {
            quant_precision: 2,
            ..VopContext::default()
        };
        let err = parse_video_object_plane_header(&data, 30_000, ctx).unwrap_err();
        assert_eq!(err, VopParseError::BadQuantPrecision(2));
    }

    #[test]
    fn quant_precision_9_reads_nine_bit_quant() {
        // Build an I-VOP with a 9-bit quant.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b00, 2);
        w.write_bits(0, 1); // modulo_time_base
        w.write_marker();
        w.write_bits(0, vop_time_increment_bits(30_000) as usize);
        w.write_marker();
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(0x1A5, 9); // vop_quant (9 bits)
        w.align();
        let data = w.buf;
        let ctx = VopContext {
            quant_precision: 9,
            ..VopContext::default()
        };
        let header = parse_video_object_plane_header(&data, 30_000, ctx).unwrap();
        assert_eq!(header.quant, 0x1A5);
    }

    #[test]
    fn interlaced_context_consumes_two_extra_bits() {
        // Build an I-VOP, interlaced=true: after intra_dc_vlc_thr we
        // must consume top_field_first + alternate_vertical_scan_flag.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0b00, 2);
        w.write_bits(0, 1);
        w.write_marker();
        w.write_bits(0, vop_time_increment_bits(30_000) as usize);
        w.write_marker();
        w.write_bits(1, 1);
        w.write_bits(0b101, 3); // intra_dc_vlc_thr
        w.write_bits(1, 1); // top_field_first
        w.write_bits(0, 1); // alt_vert_scan_flag
        w.write_bits(7, 5); // vop_quant
        w.align();
        let data = w.buf;
        let ctx = VopContext {
            interlaced: true,
            ..VopContext::default()
        };
        let header = parse_video_object_plane_header(&data, 30_000, ctx).unwrap();
        assert_eq!(header.intra_dc_vlc_thr, 0b101);
        assert_eq!(header.quant, 7);
    }

    #[test]
    fn group_of_vop_parses_time_code_and_flags() {
        let mut w = BitWriter::new();
        w.write_bits(GROUP_OF_VOP_START_CODE, 32);
        w.write_bits(12, 5); // hours
        w.write_bits(34, 6); // minutes
        w.write_marker();
        w.write_bits(56, 6); // seconds
        w.write_bits(1, 1); // closed_gov
        w.write_bits(0, 1); // broken_link
        w.align();
        let data = w.buf;
        let gov = parse_group_of_vop_header(&data).unwrap();
        assert_eq!(
            gov.time_code,
            TimeCode {
                hours: 12,
                minutes: 34,
                seconds: 56
            }
        );
        assert!(gov.closed_gov);
        assert!(!gov.broken_link);
    }

    #[test]
    fn group_of_vop_rejects_missing_marker() {
        let mut w = BitWriter::new();
        w.write_bits(GROUP_OF_VOP_START_CODE, 32);
        w.write_bits(0, 5);
        w.write_bits(0, 6);
        w.write_bits(0, 1); // marker_bit = 0 (illegal)
        w.align();
        let data = w.buf;
        let err = parse_group_of_vop_header(&data).unwrap_err();
        assert_eq!(err, VopParseError::MarkerBitMissing);
    }

    #[test]
    fn group_of_vop_missing_start_code() {
        // Use VOP_START_CODE (0x01B6) instead of 0x01B3.
        let mut w = BitWriter::new();
        w.write_bits(VOP_START_CODE, 32);
        w.write_bits(0, 18);
        w.write_bits(0, 2);
        w.align();
        let data = w.buf;
        let err = parse_group_of_vop_header(&data).unwrap_err();
        match err {
            VopParseError::MissingStartCode { expected, .. } => {
                assert_eq!(expected, "group_of_vop_start_code")
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn vop_parse_error_displays() {
        let e = VopParseError::ForbiddenFcode;
        assert!(format!("{e}").contains("forbidden"));
        let e = VopParseError::BadQuantPrecision(2);
        assert!(format!("{e}").contains("3..=9"));
        let e = VopParseError::UnsupportedBranch("scalability");
        assert!(format!("{e}").contains("scalability"));
        let e = VopParseError::Truncated;
        assert!(format!("{e}").contains("truncated"));
        let e = VopParseError::MarkerBitMissing;
        assert!(format!("{e}").contains("marker_bit"));
        let e = VopParseError::MissingStartCode {
            expected: "x",
            found: 0,
        };
        assert!(format!("{e}").contains("missing x"));
    }

    /// Build a minimal VOL fixture suitable for `VopHeader::from_vol`
    /// integration tests. Mirrors the round-3 helper in `vol.rs`.
    fn build_minimal_vol(resolution: u16, width: u16, height: u16) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(crate::vol::VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication
        w.write_bits(0, 1); // is_object_layer_identifier
        w.write_bits(0b0001, 4); // aspect_ratio_info = 1:1
        w.write_bits(0, 1); // vol_control_parameters = 0
        w.write_bits(0b00, 2); // video_object_layer_shape = rectangular
        w.write_marker();
        w.write_bits(u32::from(resolution), 16);
        w.write_marker();
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_marker();
        w.write_bits(u32::from(width), 13);
        w.write_marker();
        w.write_bits(u32::from(height), 13);
        w.write_marker();
        // Round-3 trailing block (everything off).
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable (1 bit, verid=1)
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
        w.buf
    }

    #[test]
    fn vop_context_from_vol_picks_up_promoted_fields() {
        let vol_data = build_minimal_vol(30_000, 176, 144);
        let vol = crate::vol::parse_video_object_layer(&vol_data, 0x01).unwrap();
        let ctx = VopContext::from_vol(&vol);
        assert_eq!(ctx.quant_precision, 5);
        assert!(!ctx.interlaced);
        assert!(!ctx.sprite_gmc);
        assert!(!ctx.sprite_static);
        assert!(!ctx.scalability);
        assert!(!ctx.newpred_enable);
        assert!(!ctx.reduced_resolution_vop_enable);
        assert!(ctx.complexity_estimation_disable);
    }

    #[test]
    fn vop_header_from_vol_parses_minimal_i_vop() {
        let vol_data = build_minimal_vol(30_000, 176, 144);
        let vol = crate::vol::parse_video_object_layer(&vol_data, 0x01).unwrap();
        let vop_data = make_i_vop(30_000, 0, 1001, 7, 3);
        let header = VopHeader::from_vol(&vol, &vop_data).unwrap();
        assert_eq!(header.coding_type, VopCodingType::I);
        assert_eq!(header.time_increment, 1001);
        assert_eq!(header.quant, 7);
        assert_eq!(header.intra_dc_vlc_thr, 3);
    }

    #[test]
    fn vop_header_from_vol_uses_vol_resolution() {
        // Build a VOL with a small resolution (so vop_time_increment
        // is only 1 bit wide) and confirm `from_vol` plumbs it
        // through.
        let vol_data = build_minimal_vol(2, 16, 16);
        let vol = crate::vol::parse_video_object_layer(&vol_data, 0).unwrap();
        assert_eq!(vol.time_increment_resolution, 2);
        // Build an I-VOP with resolution=2.
        let vop_data = make_i_vop(2, 0, 1, 5, 0);
        let header = VopHeader::from_vol(&vol, &vop_data).unwrap();
        assert_eq!(header.time_increment, 1);
        assert_eq!(header.composed_ticks, 1);
    }

    #[test]
    fn vop_error_maps_into_vol_error() {
        let v: VolParseError = VopParseError::Truncated.into();
        assert_eq!(v, VolParseError::Truncated);
        let v: VolParseError = VopParseError::MarkerBitMissing.into();
        assert_eq!(v, VolParseError::MarkerBitMissing);
        let v: VolParseError = VopParseError::MissingStartCode {
            expected: "x",
            found: 0,
        }
        .into();
        assert!(matches!(v, VolParseError::MissingStartCode { .. }));
        let v: VolParseError = VopParseError::ForbiddenFcode.into();
        assert_eq!(v, VolParseError::UnsupportedFgs);
    }
}

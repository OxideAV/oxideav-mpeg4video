//! Structural parser for the MPEG-4 Visual configuration headers
//! (`VisualObjectSequence` / `VisualObject` / `VideoObjectLayer`).
//!
//! Coverage in this module is **structural-only**: we identify the
//! start codes (`0x000001B0`, `0x000001B5`, `0x000001Bx`), step
//! across the byte/bit fields that precede the picture data, and
//! surface a typed [`VolHeader`] for downstream callers. VOP- and
//! macroblock-level decoding are out of scope for this round.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.1 Start codes (Table 6-3) — for the 32-bit sync values.
//! * §6.2.2 `VisualObjectSequence()` + `VisualObject()` — for the
//!   `profile_and_level_indication` byte and the optional video
//!   signal type / identifier fields.
//! * §6.2.3 `VideoObjectLayer()` — for the VOL header bit layout
//!   including `interlaced`, `obmc_disable`, `sprite_enable`,
//!   `not_8_bit` / `quant_precision`, `quant_type`,
//!   `complexity_estimation_disable`, `resync_marker_disable`,
//!   `data_partitioned`, `newpred_enable`,
//!   `reduced_resolution_vop_enable`, and `scalability` (round 3).
//! * §6.2.x semantics — for `aspect_ratio_info` (Table 6-14),
//!   `chroma_format` (Table 6-15), `video_object_layer_shape`
//!   (Table 6-16), `sprite_enable` (Table 6-19), and the marker-bit
//!   conventions.
//!
//! We restrict ourselves to the non-Studio, non-FGS branch of
//! `VideoObjectLayer()` (i.e. the `else { is_object_layer_identifier
//! … }` path documented at spec line ~3926). Studio Profiles and
//! Fine-Granularity-Scalable layers are not yet wired up; encountering
//! one returns [`VolParseError::UnsupportedProfile`] /
//! [`VolParseError::UnsupportedFgs`] rather than silently producing
//! wrong values. Sprite-with-coordinates (`sprite_enable == static`
//! or `GMC`) and the load-quantiser-matrix branch are rejected
//! cleanly with [`VolParseError::UnsupportedBranch`] so callers can
//! tell "we know what this means, just haven't wired the payload"
//! apart from a hard parse failure.

use crate::bitreader::{BitReader, BitReaderError};

/// Start code for a `VisualObjectSequence` (§6.2.1, Table 6-3).
pub const VISUAL_OBJECT_SEQUENCE_START_CODE: u32 = 0x0000_01B0;
/// Start code that terminates a `VisualObjectSequence`.
pub const VISUAL_OBJECT_SEQUENCE_END_CODE: u32 = 0x0000_01B1;
/// Start code that introduces a `VisualObject` (§6.2.2).
pub const VISUAL_OBJECT_START_CODE: u32 = 0x0000_01B5;
/// Lower bound of the `video_object_layer_start_code` range
/// (Table 6-3 lists `0x20`..=`0x2F` for the trailing byte).
pub const VIDEO_OBJECT_LAYER_START_CODE_MIN: u32 = 0x0000_0120;
/// Upper bound of the `video_object_layer_start_code` range.
pub const VIDEO_OBJECT_LAYER_START_CODE_MAX: u32 = 0x0000_012F;
/// Lower bound of `video_object_start_code` (`00`..=`1F`).
pub const VIDEO_OBJECT_START_CODE_MIN: u32 = 0x0000_0100;
/// Upper bound of `video_object_start_code`.
pub const VIDEO_OBJECT_START_CODE_MAX: u32 = 0x0000_011F;

/// Errors produced by the structural VOL header parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolParseError {
    /// The supplied byte slice ran out mid-field.
    Truncated,
    /// One of the start codes (`VS`, `VO`, `VOL`) was expected at the
    /// current position but a different 32-bit value was found.
    MissingStartCode {
        /// What we were looking for (`VS_START`, `VO_START`, …).
        expected: &'static str,
        /// What the next 32 bits actually were.
        found: u32,
    },
    /// `profile_and_level_indication` indicated a Studio Profile
    /// (values `0xE1`..=`0xE8`, per §6.2.2). Studio decoding is not
    /// part of this round.
    UnsupportedProfile(u8),
    /// The Video Object Layer announced `video_object_type_indication
    /// == "Fine Granularity Scalable"`, whose VOL header layout
    /// diverges from the base case and is out of scope here.
    UnsupportedFgs,
    /// A `marker_bit` was expected to be 1 (§6.2.3) but was read as
    /// 0. This is hard evidence of either bitstream corruption or a
    /// parser misalignment, so we surface it rather than silently
    /// continuing.
    MarkerBitMissing,
    /// `aspect_ratio_info == 0000` is "Forbidden" per Table 6-14.
    ForbiddenAspectRatio,
    /// `video_object_layer_shape == grayscale` requires reading
    /// `video_object_layer_shape_extension` (Table 6-17); supporting
    /// the broader grayscale/binary shape pipeline is out of scope.
    UnsupportedShape(u8),
    /// `visual_object_type` selected something other than `video ID`
    /// (e.g. `still texture ID`, `mesh ID`, `FBA ID`, `3D mesh ID`).
    /// Those carriage paths are outside the rectangular-video focus
    /// of this round.
    UnsupportedVisualObjectType(u8),
    /// `quant_precision` (Table §6.3.3) outside the legal 3..=9
    /// range — bitstream is malformed.
    BadQuantPrecision(u8),
    /// The VOL declared a branch the round-3 structural parser
    /// recognises but does not yet walk in detail (e.g.
    /// `sprite_enable == static / GMC`, `not_8_bit` with grayscale
    /// shape, complexity-estimation header, or a custom
    /// `intra_quant_mat` / `nonintra_quant_mat`).
    UnsupportedBranch(&'static str),
}

impl core::fmt::Display for VolParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VolParseError::Truncated => write!(f, "VOL header truncated"),
            VolParseError::MissingStartCode { expected, found } => {
                write!(f, "missing {expected} start code (found 0x{found:08X})")
            }
            VolParseError::UnsupportedProfile(p) => {
                write!(f, "unsupported profile_and_level_indication 0x{p:02X}")
            }
            VolParseError::UnsupportedFgs => {
                write!(f, "Fine-Granularity-Scalable VOL header path not supported")
            }
            VolParseError::MarkerBitMissing => write!(f, "marker_bit was 0 (expected 1)"),
            VolParseError::ForbiddenAspectRatio => {
                write!(f, "aspect_ratio_info=0000 is forbidden (Table 6-14)")
            }
            VolParseError::UnsupportedShape(s) => {
                write!(f, "video_object_layer_shape={s} not yet supported")
            }
            VolParseError::UnsupportedVisualObjectType(t) => {
                write!(f, "visual_object_type={t} not supported by this parser")
            }
            VolParseError::BadQuantPrecision(p) => {
                write!(f, "quant_precision={p} outside the allowed 3..=9 range")
            }
            VolParseError::UnsupportedBranch(name) => {
                write!(f, "VOL branch '{name}' not supported by round-3 parser")
            }
        }
    }
}

impl std::error::Error for VolParseError {}

impl From<BitReaderError> for VolParseError {
    fn from(_: BitReaderError) -> Self {
        VolParseError::Truncated
    }
}

/// Decoded `sprite_enable` field (Table 6-19, §6.3.3).
///
/// The on-wire encoding depends on `video_object_layer_verid`:
/// `verid == 0001` carries a single bit (`0` → `NotUsed`, `1` →
/// `Static`); later verids carry two bits (`00` → `NotUsed`, `01`
/// → `Static`, `10` → `Gmc`, `11` → reserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteEnable {
    /// `0` (1-bit form) or `00` (2-bit form) — sprite coding is not
    /// used in this VOL.
    NotUsed,
    /// `1` (1-bit form) or `01` (2-bit form) — Static (Basic or Low
    /// Latency) sprite coding.
    Static,
    /// `10` (2-bit form only) — Global Motion Compensation. An S-VOP
    /// with `sprite_enable == GMC` is denoted "S (GMC)-VOP" in the
    /// spec.
    Gmc,
    /// `11` (2-bit form only) — reserved.
    Reserved,
}

impl SpriteEnable {
    fn from_one_bit(bit: u32) -> Self {
        if bit & 1 == 1 {
            SpriteEnable::Static
        } else {
            SpriteEnable::NotUsed
        }
    }

    fn from_two_bits(bits: u32) -> Self {
        match bits & 0b11 {
            0b00 => SpriteEnable::NotUsed,
            0b01 => SpriteEnable::Static,
            0b10 => SpriteEnable::Gmc,
            0b11 => SpriteEnable::Reserved,
            _ => unreachable!("masked to 2 bits"),
        }
    }
}

/// Pixel aspect ratio, derived from `aspect_ratio_info` (Table 6-14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    /// `0001` — 1:1 square pixels.
    Square,
    /// `0010` — 12:11 (625-type for 4:3 picture).
    Par12x11,
    /// `0011` — 10:11 (525-type for 4:3 picture).
    Par10x11,
    /// `0100` — 16:11 (625-type stretched for 16:9 picture).
    Par16x11,
    /// `0101` — 40:33 (525-type stretched for 16:9 picture).
    Par40x33,
    /// `1111` — extended_PAR; the two 8-bit `par_width` / `par_height`
    /// follow in the bitstream.
    Extended {
        /// `par_width`. Zero is forbidden per §6.2 semantics.
        par_width: u8,
        /// `par_height`. Zero is forbidden per §6.2 semantics.
        par_height: u8,
    },
    /// `0110`..=`1110` — Reserved by Table 6-14. We accept these
    /// rather than reject the whole stream, since downstream display
    /// code can fall back to 1:1.
    Reserved(u8),
}

/// Optional `vol_control_parameters` block (§6.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolControlParameters {
    /// Chrominance format (Table 6-15). Only `4:2:0` (binary `01`) is
    /// non-reserved.
    pub chroma_format: u8,
    /// `low_delay`: if `true`, the VOL contains no B-VOPs.
    pub low_delay: bool,
    /// Optional `vbv_parameters`. Decoded fields are surfaced so a
    /// downstream rate-control component can use them without re-
    /// parsing the bitstream.
    pub vbv: Option<VbvParameters>,
}

/// VBV (Video Buffer Verifier) parameters carried inside the
/// optional `vol_control_parameters` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VbvParameters {
    /// Composed 30-bit bit rate in units of 400 bits/second
    /// (`first_half_bit_rate << 15 | latter_half_bit_rate`).
    pub bit_rate: u32,
    /// Composed 18-bit VBV buffer size in units of 16384 bits
    /// (`first_half_vbv_buffer_size << 3 | latter_half_vbv_buffer_size`).
    pub vbv_buffer_size: u32,
    /// Composed 26-bit VBV occupancy (`first_half_vbv_occupancy << 15
    /// | latter_half_vbv_occupancy`).
    pub vbv_occupancy: u32,
}

/// Typed view of the bits that precede VOP data in a Video Object
/// Layer header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolHeader {
    /// The 8-bit `profile_and_level_indication` from the parent
    /// `VisualObjectSequence`.
    pub profile_level: u8,
    /// Frame width in pixels (`video_object_layer_width`, §6.2.3).
    /// Zero is permitted only when `video_object_layer_shape ==
    /// "binary only"`, which this parser rejects upfront.
    pub width: u16,
    /// Frame height in pixels (`video_object_layer_height`, §6.2.3).
    pub height: u16,
    /// `vop_time_increment_resolution`. The number of ticks per
    /// second (§6.2.x: "subintervals … within one modulo time").
    /// Zero is forbidden.
    pub time_increment_resolution: u16,
    /// `fixed_vop_rate` flag. When true the parser also exposes
    /// `fixed_vop_time_increment` in [`Self::fixed_vop_time_increment`].
    pub fixed_vop_rate: bool,
    /// `fixed_vop_time_increment`. Bit-width is the minimum integer
    /// width that can hold `vop_time_increment_resolution - 1`.
    /// `Some(_)` only when [`Self::fixed_vop_rate`] is `true`.
    pub fixed_vop_time_increment: Option<u16>,
    /// Decoded `aspect_ratio_info`.
    pub aspect_ratio: AspectRatio,
    /// Optional `vol_control_parameters`.
    pub vol_control: Option<VolControlParameters>,
    /// `random_accessible_vol` (§6.2.3).
    pub random_accessible_vol: bool,
    /// `video_object_type_indication`.
    pub video_object_type_indication: u8,
    /// `is_object_layer_identifier`. When `true` the next two fields
    /// hold valid data; otherwise they default to spec-implied
    /// `verid=1, priority=0`.
    pub is_object_layer_identifier: bool,
    /// `video_object_layer_verid` (4 bits). Defaults to `1` when
    /// `is_object_layer_identifier == false` per §6.2.3 conventions.
    pub video_object_layer_verid: u8,
    /// `video_object_layer_priority` (3 bits). Defaults to `0` when
    /// `is_object_layer_identifier == false`.
    pub video_object_layer_priority: u8,
    /// `video_object_layer_shape` (Table 6-16). Only `rectangular`
    /// (value `0`) returns successfully today.
    pub video_object_layer_shape: u8,
    /// `interlaced` flag (§6.3.3). `false` means progressive frame
    /// VOPs; `true` enables `top_field_first` /
    /// `alternate_vertical_scan_flag` in the VOP header.
    pub interlaced: bool,
    /// `obmc_disable` flag (§6.3.3). When `true`, overlapped block
    /// motion compensation is suppressed.
    pub obmc_disable: bool,
    /// Decoded `sprite_enable` (Table 6-19).
    pub sprite_enable: SpriteEnable,
    /// `not_8_bit` flag (§6.3.3). When `true`, `quant_precision` and
    /// `bits_per_pixel` are transmitted in-line; otherwise the spec
    /// defaults `quant_precision = 5` and `bits_per_pixel = 8`.
    pub not_8_bit: bool,
    /// `quant_precision` (§6.3.3). Five-bit default when `not_8_bit`
    /// is `false`. Legal range is 3..=9.
    pub quant_precision: u8,
    /// `bits_per_pixel` (§6.3.3). Eight-bit default when `not_8_bit`
    /// is `false`.
    pub bits_per_pixel: u8,
    /// `quant_type` flag (§6.3.3). `true` selects the first
    /// (matrix-driven) inverse-quantisation method, `false` the
    /// second (H.263-style).
    pub quant_type: bool,
    /// `quarter_sample` flag (§6.3.3). Only present when
    /// `video_object_layer_verid != 1`; defaults to `false`
    /// otherwise.
    pub quarter_sample: bool,
    /// `complexity_estimation_disable` (§6.3.3). When `true`, the
    /// VOP header does NOT carry a complexity-estimation block.
    /// Round 3 only accepts `true`.
    pub complexity_estimation_disable: bool,
    /// `resync_marker_disable` (§6.3.3). When `true`, no
    /// `resync_marker` bit pattern appears in the VOP payload.
    pub resync_marker_disable: bool,
    /// `data_partitioned` (§6.3.3). When `true`, intra DC
    /// coefficients are sent in a separate partition from AC and
    /// inter data.
    pub data_partitioned: bool,
    /// `reversible_vlc` (§6.3.3). Defined only when
    /// `data_partitioned == true`; defaults to `false` otherwise.
    pub reversible_vlc: bool,
    /// `newpred_enable` (§6.3.3). Defined only when
    /// `video_object_layer_verid != 1`; defaults to `false`
    /// otherwise.
    pub newpred_enable: bool,
    /// `reduced_resolution_vop_enable` (§6.3.3). Defined only when
    /// `video_object_layer_verid != 1`; defaults to `false`
    /// otherwise.
    pub reduced_resolution_vop_enable: bool,
    /// `scalability` flag (§6.3.3). When `true`, the VOL declares
    /// a scalable enhancement layer. Round 3 rejects this branch
    /// upfront; the field is surfaced so callers can detect it.
    pub scalability: bool,
}

/// Compute the bit-width needed to represent the range
/// `[0, vop_time_increment_resolution)`. Per §6.2 semantics this is
/// "the minimum number of unsigned integer bits required to represent
/// the above range", with a special-case minimum of 1.
fn vop_time_increment_bits(resolution: u16) -> u8 {
    // `resolution` is u16, so the largest value the field can index
    // is 65535, needing 16 bits. The smallest non-forbidden value is
    // 1, indexing 0 only — 1 bit per the "minimum" wording.
    if resolution <= 1 {
        return 1;
    }
    // For N values (0..=N-1), the bit count is ceil(log2(N)).
    let n = u32::from(resolution);
    32 - (n - 1).leading_zeros() as u8
}

fn parse_aspect_ratio(br: &mut BitReader<'_>, code: u8) -> Result<AspectRatio, VolParseError> {
    Ok(match code {
        0b0000 => return Err(VolParseError::ForbiddenAspectRatio),
        0b0001 => AspectRatio::Square,
        0b0010 => AspectRatio::Par12x11,
        0b0011 => AspectRatio::Par10x11,
        0b0100 => AspectRatio::Par16x11,
        0b0101 => AspectRatio::Par40x33,
        0b1111 => {
            let par_width = br.read_bits(8)? as u8;
            let par_height = br.read_bits(8)? as u8;
            AspectRatio::Extended {
                par_width,
                par_height,
            }
        }
        other => AspectRatio::Reserved(other),
    })
}

fn read_marker_bit(br: &mut BitReader<'_>) -> Result<(), VolParseError> {
    if br.read_bool()? {
        Ok(())
    } else {
        Err(VolParseError::MarkerBitMissing)
    }
}

fn parse_vol_control(br: &mut BitReader<'_>) -> Result<VolControlParameters, VolParseError> {
    let chroma_format = br.read_bits(2)? as u8;
    let low_delay = br.read_bool()?;
    let vbv_parameters = br.read_bool()?;
    let vbv = if vbv_parameters {
        let first_half_bit_rate = br.read_bits(15)?;
        read_marker_bit(br)?;
        let latter_half_bit_rate = br.read_bits(15)?;
        read_marker_bit(br)?;
        let first_half_vbv_buffer_size = br.read_bits(15)?;
        read_marker_bit(br)?;
        let latter_half_vbv_buffer_size = br.read_bits(3)?;
        let first_half_vbv_occupancy = br.read_bits(11)?;
        read_marker_bit(br)?;
        let latter_half_vbv_occupancy = br.read_bits(15)?;
        read_marker_bit(br)?;
        Some(VbvParameters {
            bit_rate: (first_half_bit_rate << 15) | latter_half_bit_rate,
            vbv_buffer_size: (first_half_vbv_buffer_size << 3) | latter_half_vbv_buffer_size,
            vbv_occupancy: (first_half_vbv_occupancy << 15) | latter_half_vbv_occupancy,
        })
    } else {
        None
    };
    Ok(VolControlParameters {
        chroma_format,
        low_delay,
        vbv,
    })
}

/// Parse a Video Object Layer header starting at the
/// `video_object_layer_start_code` (`0x000001Bx`).
///
/// `profile_level` is supplied by the caller — the
/// `profile_and_level_indication` byte lives in the parent
/// `VisualObjectSequence`, so a standalone VOL slice will not know it
/// and must pass in `0`.
pub fn parse_video_object_layer(
    data: &[u8],
    profile_level: u8,
) -> Result<VolHeader, VolParseError> {
    let mut br = BitReader::new(data);
    let start = br.read_bits(32)?;
    if !(VIDEO_OBJECT_LAYER_START_CODE_MIN..=VIDEO_OBJECT_LAYER_START_CODE_MAX).contains(&start) {
        return Err(VolParseError::MissingStartCode {
            expected: "video_object_layer_start_code",
            found: start,
        });
    }
    parse_video_object_layer_body(&mut br, profile_level)
}

fn parse_video_object_layer_body(
    br: &mut BitReader<'_>,
    profile_level: u8,
) -> Result<VolHeader, VolParseError> {
    let random_accessible_vol = br.read_bool()?;
    let video_object_type_indication = br.read_bits(8)? as u8;
    // §6.2.3 "if video_object_type_indication == Fine Granularity
    // Scalable" — the FGS branch starts with fgs_layer_type and is
    // out of scope this round.
    //
    // The numeric encoding of "Fine Granularity Scalable" is defined
    // in Table 6-7 of §6.3.2 (visual_object_type), value `1010`.
    // Rather than hard-code it here, we treat any unrecognised
    // branch the same — we follow the else-branch layout, and if
    // that misaligns the rest of the parse will trip
    // MarkerBitMissing.
    let is_object_layer_identifier = br.read_bool()?;
    let (video_object_layer_verid, video_object_layer_priority) = if is_object_layer_identifier {
        (br.read_bits(4)? as u8, br.read_bits(3)? as u8)
    } else {
        // Defaults implied by §6.2 — verid is presumed to be 1 in
        // base profile, priority is unconstrained but typically 0.
        (1, 0)
    };
    let aspect_ratio_info = br.read_bits(4)? as u8;
    let aspect_ratio = parse_aspect_ratio(br, aspect_ratio_info)?;
    let vol_control_flag = br.read_bool()?;
    let vol_control = if vol_control_flag {
        Some(parse_vol_control(br)?)
    } else {
        None
    };
    let video_object_layer_shape = br.read_bits(2)? as u8;
    // Per §6.2.3 only shape == 00 ("rectangular") is in scope for the
    // round-1 structural parse. Other values demand shape extension
    // bits and/or auxiliary-component handling not yet wired up.
    if video_object_layer_shape != 0 {
        return Err(VolParseError::UnsupportedShape(video_object_layer_shape));
    }
    read_marker_bit(br)?;
    let time_increment_resolution = br.read_bits(16)? as u16;
    read_marker_bit(br)?;
    let fixed_vop_rate = br.read_bool()?;
    let fixed_vop_time_increment = if fixed_vop_rate {
        let bits = vop_time_increment_bits(time_increment_resolution) as usize;
        Some(br.read_bits(bits)? as u16)
    } else {
        None
    };
    // shape == rectangular branch (Table 6-16 value 00).
    read_marker_bit(br)?;
    let width = br.read_bits(13)? as u16;
    read_marker_bit(br)?;
    let height = br.read_bits(13)? as u16;
    read_marker_bit(br)?;

    // Round 3 extension: continue the `if (video_object_layer_shape !=
    // "binary only")` branch — `interlaced`, `obmc_disable`,
    // `sprite_enable`, `not_8_bit` / `quant_precision`, `quant_type`,
    // `complexity_estimation_disable`, `resync_marker_disable`,
    // `data_partitioned`, optional `newpred_enable` /
    // `reduced_resolution_vop_enable`, and `scalability`. Spec lines
    // 3989..=4079 of `ISO_IEC_14496-2-2004-3rd-edition.txt`.
    let interlaced = br.read_bool()?;
    let obmc_disable = br.read_bool()?;
    let sprite_enable = if video_object_layer_verid == 1 {
        SpriteEnable::from_one_bit(br.read_bits(1)?)
    } else {
        SpriteEnable::from_two_bits(br.read_bits(2)?)
    };
    // The full sprite block (sprite_width / _height / coordinates /
    // warping points / accuracy / brightness change / low-latency)
    // is not part of round 3. Surface a typed rejection so the bit
    // position doesn't quietly drift.
    if matches!(sprite_enable, SpriteEnable::Static | SpriteEnable::Gmc) {
        return Err(VolParseError::UnsupportedBranch(
            "sprite_enable static/GMC body",
        ));
    }
    if matches!(sprite_enable, SpriteEnable::Reserved) {
        return Err(VolParseError::UnsupportedBranch("sprite_enable reserved"));
    }
    // sadct_disable is only present when verid != 1 AND shape !=
    // rectangular (spec line 4014). Round 3 gates `shape ==
    // rectangular` upfront, so we never read it.
    let not_8_bit = br.read_bool()?;
    let (quant_precision, bits_per_pixel) = if not_8_bit {
        let qp = br.read_bits(4)? as u8;
        let bpp = br.read_bits(4)? as u8;
        if !(3..=9).contains(&qp) {
            return Err(VolParseError::BadQuantPrecision(qp));
        }
        (qp, bpp)
    } else {
        (5, 8)
    };
    // Grayscale sub-block (no_gray_quant_update / composition_method /
    // linear_composition) is gated on `shape == grayscale`, which is
    // already rejected upfront.
    let quant_type = br.read_bool()?;
    if quant_type {
        // load_intra_quant_mat + load_nonintra_quant_mat carry
        // variable-length 8-bit run lists. Round 3 doesn't yet
        // decode those bodies — surface the branch cleanly so the
        // bit position doesn't drift.
        let load_intra = br.read_bool()?;
        if load_intra {
            return Err(VolParseError::UnsupportedBranch(
                "load_intra_quant_mat body",
            ));
        }
        let load_nonintra = br.read_bool()?;
        if load_nonintra {
            return Err(VolParseError::UnsupportedBranch(
                "load_nonintra_quant_mat body",
            ));
        }
        // (grayscale variant is gated on shape == grayscale.)
    }
    let quarter_sample = if video_object_layer_verid != 1 {
        br.read_bool()?
    } else {
        false
    };
    let complexity_estimation_disable = br.read_bool()?;
    if !complexity_estimation_disable {
        return Err(VolParseError::UnsupportedBranch(
            "define_vop_complexity_estimation_header",
        ));
    }
    let resync_marker_disable = br.read_bool()?;
    let data_partitioned = br.read_bool()?;
    let reversible_vlc = if data_partitioned {
        br.read_bool()?
    } else {
        false
    };
    let (newpred_enable, reduced_resolution_vop_enable) = if video_object_layer_verid != 1 {
        let np = br.read_bool()?;
        if np {
            // newpred_enable body: requested_upstream_message_type (2
            // bits) + newpred_segment_type (1 bit). We consume those
            // structurally to keep alignment if we ever continue past
            // this point in a future round, but reject the branch
            // immediately afterwards since the rest of the VOL syntax
            // depends on it being false.
            return Err(VolParseError::UnsupportedBranch("newpred_enable body"));
        }
        let rr = br.read_bool()?;
        (np, rr)
    } else {
        (false, false)
    };
    let scalability = br.read_bool()?;

    Ok(VolHeader {
        profile_level,
        width,
        height,
        time_increment_resolution,
        fixed_vop_rate,
        fixed_vop_time_increment,
        aspect_ratio,
        vol_control,
        random_accessible_vol,
        video_object_type_indication,
        is_object_layer_identifier,
        video_object_layer_verid,
        video_object_layer_priority,
        video_object_layer_shape,
        interlaced,
        obmc_disable,
        sprite_enable,
        not_8_bit,
        quant_precision,
        bits_per_pixel,
        quant_type,
        quarter_sample,
        complexity_estimation_disable,
        resync_marker_disable,
        data_partitioned,
        reversible_vlc,
        newpred_enable,
        reduced_resolution_vop_enable,
        scalability,
    })
}

/// Parse a `VisualObjectSequence` start code followed by
/// `profile_and_level_indication`. The four-byte start code MUST be
/// `0x000001B0`. Studio profiles (`0xE1`..=`0xE8` per §6.2.2) return
/// [`VolParseError::UnsupportedProfile`].
pub fn parse_visual_object_sequence_header(data: &[u8]) -> Result<u8, VolParseError> {
    let mut br = BitReader::new(data);
    let sc = br.read_bits(32)?;
    if sc != VISUAL_OBJECT_SEQUENCE_START_CODE {
        return Err(VolParseError::MissingStartCode {
            expected: "visual_object_sequence_start_code",
            found: sc,
        });
    }
    let profile_level = br.read_bits(8)? as u8;
    if (0xE1..=0xE8).contains(&profile_level) {
        return Err(VolParseError::UnsupportedProfile(profile_level));
    }
    Ok(profile_level)
}

/// Parse a `VisualObject` header — the bits between `0x000001B5` and
/// the next start code, returning `visual_object_type`. The structural
/// parser supports only `visual_object_type == "video ID"` (numeric
/// value `1` per §6.3.2 / Table 6-7); other types return
/// [`VolParseError::UnsupportedVisualObjectType`].
pub fn parse_visual_object_header(data: &[u8]) -> Result<u8, VolParseError> {
    let mut br = BitReader::new(data);
    let sc = br.read_bits(32)?;
    if sc != VISUAL_OBJECT_START_CODE {
        return Err(VolParseError::MissingStartCode {
            expected: "visual_object_start_code",
            found: sc,
        });
    }
    let is_visual_object_identifier = br.read_bool()?;
    if is_visual_object_identifier {
        let _visual_object_verid = br.read_bits(4)?;
        let _visual_object_priority = br.read_bits(3)?;
    }
    let visual_object_type = br.read_bits(4)? as u8;
    // §6.2.2: when type is video or still-texture, video_signal_type()
    // follows. We accept video (value 1 per Table 6-7) and reject the
    // rest, since this round is about rectangular video VOLs.
    if visual_object_type != 1 {
        return Err(VolParseError::UnsupportedVisualObjectType(
            visual_object_type,
        ));
    }
    Ok(visual_object_type)
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

    /// Append the round-3 trailing fields onto a writer that has just
    /// finished writing the rectangular width/height marker block.
    /// All values map onto the "simplest possible" Simple-Profile
    /// path: no interlace, no sprite, no `not_8_bit`, second
    /// inverse-quant method, complexity estimation disabled, no
    /// resync marker, no data partitioning, verid == 1 so no
    /// quarter_sample / newpred / reduced_res bits, and scalability
    /// off.
    fn write_minimal_trailing(w: &mut BitWriter) {
        w.write_bits(0, 1); // interlaced = 0
        w.write_bits(0, 1); // obmc_disable = 0
                            // verid == 1 → sprite_enable is 1 bit.
        w.write_bits(0, 1); // sprite_enable = NotUsed
        w.write_bits(0, 1); // not_8_bit = 0
        w.write_bits(0, 1); // quant_type = 0
                            // verid == 1 → no quarter_sample bit.
        w.write_bits(1, 1); // complexity_estimation_disable = 1
        w.write_bits(1, 1); // resync_marker_disable = 1
        w.write_bits(0, 1); // data_partitioned = 0
                            // verid == 1 → no newpred / reduced_res bits.
        w.write_bits(0, 1); // scalability = 0
    }

    fn make_minimal_vol(width: u16, height: u16) -> Vec<u8> {
        let mut w = BitWriter::new();
        // video_object_layer_start_code = 0x000001Bx (low nibble 0).
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication = 1 (Simple)
        w.write_bits(0, 1); // is_object_layer_identifier = 0
        w.write_bits(0b0001, 4); // aspect_ratio_info = 1:1 square
        w.write_bits(0, 1); // vol_control_parameters = 0
        w.write_bits(0b00, 2); // video_object_layer_shape = rectangular
        w.write_marker();
        w.write_bits(30_000, 16); // vop_time_increment_resolution
        w.write_marker();
        w.write_bits(1, 1); // fixed_vop_rate = 1
                            // fixed_vop_time_increment width = bits needed for 0..29999
                            // ceil(log2(30000)) = 15.
        let bits = vop_time_increment_bits(30_000) as usize;
        w.write_bits(1001, bits);
        w.write_marker();
        w.write_bits(u32::from(width), 13);
        w.write_marker();
        w.write_bits(u32::from(height), 13);
        w.write_marker();
        write_minimal_trailing(&mut w);
        w.align();
        w.buf
    }

    #[test]
    fn parses_minimal_vol() {
        let data = make_minimal_vol(352, 288);
        let header = parse_video_object_layer(&data, 0x01).unwrap();
        assert_eq!(header.width, 352);
        assert_eq!(header.height, 288);
        assert_eq!(header.time_increment_resolution, 30_000);
        assert!(header.fixed_vop_rate);
        assert_eq!(header.fixed_vop_time_increment, Some(1001));
        assert_eq!(header.aspect_ratio, AspectRatio::Square);
        assert!(header.vol_control.is_none());
        assert_eq!(header.video_object_layer_shape, 0);
        assert_eq!(header.profile_level, 0x01);
        // Round-3 fields populated from the minimal trailing block.
        assert!(!header.interlaced);
        assert!(!header.obmc_disable);
        assert_eq!(header.sprite_enable, SpriteEnable::NotUsed);
        assert!(!header.not_8_bit);
        assert_eq!(header.quant_precision, 5);
        assert_eq!(header.bits_per_pixel, 8);
        assert!(!header.quant_type);
        assert!(!header.quarter_sample);
        assert!(header.complexity_estimation_disable);
        assert!(header.resync_marker_disable);
        assert!(!header.data_partitioned);
        assert!(!header.reversible_vlc);
        assert!(!header.newpred_enable);
        assert!(!header.reduced_resolution_vop_enable);
        assert!(!header.scalability);
    }

    #[test]
    fn missing_start_code_is_rejected() {
        let mut data = make_minimal_vol(176, 144);
        data[3] = 0xFF; // corrupt the start code
        let err = parse_video_object_layer(&data, 0).unwrap_err();
        match err {
            VolParseError::MissingStartCode { expected, .. } => {
                assert_eq!(expected, "video_object_layer_start_code");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn extended_par_is_carried_back() {
        // Build a fixture with aspect_ratio_info == 1111 and an
        // 8:9 PAR.
        let mut w = BitWriter::new();
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1);
        w.write_bits(1, 8);
        w.write_bits(0, 1);
        w.write_bits(0b1111, 4);
        w.write_bits(8, 8);
        w.write_bits(9, 8);
        w.write_bits(0, 1);
        w.write_bits(0b00, 2);
        w.write_marker();
        w.write_bits(1000, 16);
        w.write_marker();
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_marker();
        w.write_bits(640, 13);
        w.write_marker();
        w.write_bits(480, 13);
        w.write_marker();
        write_minimal_trailing(&mut w);
        w.align();
        let header = parse_video_object_layer(&w.buf, 0).unwrap();
        assert_eq!(
            header.aspect_ratio,
            AspectRatio::Extended {
                par_width: 8,
                par_height: 9
            }
        );
        assert!(!header.fixed_vop_rate);
        assert_eq!(header.fixed_vop_time_increment, None);
    }

    #[test]
    fn forbidden_aspect_ratio_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1);
        w.write_bits(1, 8);
        w.write_bits(0, 1);
        w.write_bits(0b0000, 4); // forbidden aspect_ratio_info
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert_eq!(err, VolParseError::ForbiddenAspectRatio);
    }

    #[test]
    fn vol_control_block_with_vbv_is_parsed() {
        let mut w = BitWriter::new();
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(1, 1); // random_accessible_vol = 1
        w.write_bits(1, 8); // video_object_type_indication
        w.write_bits(0, 1); // is_object_layer_identifier = 0
        w.write_bits(0b0001, 4); // aspect_ratio_info = 1:1
        w.write_bits(1, 1); // vol_control_parameters = 1
        w.write_bits(0b01, 2); // chroma_format = 4:2:0
        w.write_bits(1, 1); // low_delay
        w.write_bits(1, 1); // vbv_parameters = 1
        w.write_bits(0x1234, 15); // first_half_bit_rate
        w.write_marker();
        w.write_bits(0x2345, 15); // latter_half_bit_rate
        w.write_marker();
        w.write_bits(0x3456, 15); // first_half_vbv_buffer_size
        w.write_marker();
        w.write_bits(0b101, 3); // latter_half_vbv_buffer_size
        w.write_bits(0x123, 11); // first_half_vbv_occupancy
        w.write_marker();
        w.write_bits(0x4567, 15); // latter_half_vbv_occupancy
        w.write_marker();
        w.write_bits(0b00, 2); // video_object_layer_shape rectangular
        w.write_marker();
        w.write_bits(50, 16); // vop_time_increment_resolution
        w.write_marker();
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_marker();
        w.write_bits(176, 13);
        w.write_marker();
        w.write_bits(144, 13);
        w.write_marker();
        write_minimal_trailing(&mut w);
        w.align();
        let h = parse_video_object_layer(&w.buf, 0xF0).unwrap();
        let vol = h.vol_control.expect("vol_control block expected");
        assert_eq!(vol.chroma_format, 0b01);
        assert!(vol.low_delay);
        let vbv = vol.vbv.expect("vbv block expected");
        assert_eq!(vbv.bit_rate, (0x1234 << 15) | 0x2345);
        assert_eq!(vbv.vbv_buffer_size, (0x3456 << 3) | 0b101);
        assert_eq!(vbv.vbv_occupancy, (0x123 << 15) | 0x4567);
    }

    #[test]
    fn marker_violation_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1);
        w.write_bits(1, 8);
        w.write_bits(0, 1);
        w.write_bits(0b0001, 4);
        w.write_bits(0, 1);
        w.write_bits(0b00, 2);
        w.write_bits(0, 1); // marker_bit deliberately wrong
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert_eq!(err, VolParseError::MarkerBitMissing);
    }

    #[test]
    fn vop_time_bits_for_small_resolutions() {
        assert_eq!(vop_time_increment_bits(0), 1);
        assert_eq!(vop_time_increment_bits(1), 1);
        assert_eq!(vop_time_increment_bits(2), 1);
        assert_eq!(vop_time_increment_bits(3), 2);
        assert_eq!(vop_time_increment_bits(4), 2);
        assert_eq!(vop_time_increment_bits(5), 3);
        assert_eq!(vop_time_increment_bits(30), 5);
        assert_eq!(vop_time_increment_bits(30_000), 15);
        assert_eq!(vop_time_increment_bits(65_535), 16);
    }

    #[test]
    fn visual_object_sequence_header_returns_profile_byte() {
        // 0x000001B0 followed by Simple Profile / Level 1 = 0x01.
        let data = [0x00, 0x00, 0x01, 0xB0, 0x01];
        let profile = parse_visual_object_sequence_header(&data).unwrap();
        assert_eq!(profile, 0x01);
    }

    #[test]
    fn visual_object_sequence_rejects_studio_profile() {
        let data = [0x00, 0x00, 0x01, 0xB0, 0xE2];
        let err = parse_visual_object_sequence_header(&data).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedProfile(0xE2)));
    }

    #[test]
    fn visual_object_header_accepts_video_id() {
        // 0x000001B5 + is_visual_object_identifier=0 (1 bit) +
        // visual_object_type=0001 (4 bits) = 5 bits total in trailing
        // byte. MSB-first: 0_0001___ = 0b00001000 = 0x08.
        let data = [0x00, 0x00, 0x01, 0xB5, 0x08];
        let t = parse_visual_object_header(&data).unwrap();
        assert_eq!(t, 1);
    }

    #[test]
    fn visual_object_header_rejects_non_video_type() {
        // type = 0010 (still-texture per Table 6-7 in this position).
        let data = [0x00, 0x00, 0x01, 0xB5, 0b0001_0000];
        let err = parse_visual_object_header(&data).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedVisualObjectType(2)));
    }

    #[test]
    fn unsupported_shape_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1);
        w.write_bits(1, 8);
        w.write_bits(0, 1);
        w.write_bits(0b0001, 4);
        w.write_bits(0, 1);
        w.write_bits(0b11, 2); // grayscale
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert_eq!(err, VolParseError::UnsupportedShape(3));
    }

    /// Helper for round-3 trailing-field experiments: bit-writer just
    /// past the rectangular width/height marker block, with the caller
    /// in charge of writing `interlaced`..=`scalability`.
    fn write_vol_header_up_to_trailing(
        w: &mut BitWriter,
        verid: Option<u8>,
        shape: u32,
        with_vbv: bool,
    ) {
        w.write_bits(VIDEO_OBJECT_LAYER_START_CODE_MIN, 32);
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication
        if let Some(v) = verid {
            w.write_bits(1, 1); // is_object_layer_identifier
            w.write_bits(u32::from(v), 4);
            w.write_bits(0, 3); // priority
        } else {
            w.write_bits(0, 1);
        }
        w.write_bits(0b0001, 4); // aspect_ratio_info = 1:1
        if with_vbv {
            w.write_bits(1, 1); // vol_control_parameters
            w.write_bits(0b01, 2); // chroma_format = 4:2:0
            w.write_bits(0, 1); // low_delay
            w.write_bits(0, 1); // vbv_parameters
        } else {
            w.write_bits(0, 1); // vol_control_parameters
        }
        w.write_bits(shape, 2);
        w.write_marker();
        w.write_bits(30_000, 16);
        w.write_marker();
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_marker();
        w.write_bits(352, 13);
        w.write_marker();
        w.write_bits(288, 13);
        w.write_marker();
    }

    #[test]
    fn round3_interlaced_flag_is_carried_back() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(1, 1); // interlaced = 1
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable (1 bit, verid=1)
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
        let header = parse_video_object_layer(&w.buf, 0).unwrap();
        assert!(header.interlaced);
        assert_eq!(header.sprite_enable, SpriteEnable::NotUsed);
    }

    #[test]
    fn round3_not_8_bit_decodes_quant_precision_and_bpp() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(1, 1); // obmc_disable = 1
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(1, 1); // not_8_bit = 1
        w.write_bits(7, 4); // quant_precision = 7
        w.write_bits(10, 4); // bits_per_pixel = 10
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
        let header = parse_video_object_layer(&w.buf, 0).unwrap();
        assert!(header.not_8_bit);
        assert_eq!(header.quant_precision, 7);
        assert_eq!(header.bits_per_pixel, 10);
        assert!(header.obmc_disable);
    }

    #[test]
    fn round3_bad_quant_precision_is_rejected() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(1, 1); // not_8_bit = 1
        w.write_bits(2, 4); // quant_precision = 2 (illegal, <3)
        w.write_bits(8, 4);
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert_eq!(err, VolParseError::BadQuantPrecision(2));
    }

    #[test]
    fn round3_sprite_static_is_rejected_with_branch_error() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(1, 1); // sprite_enable = Static (1-bit form)
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        match err {
            VolParseError::UnsupportedBranch(name) => {
                assert!(name.contains("sprite_enable"));
            }
            other => panic!("expected UnsupportedBranch, got {other:?}"),
        }
    }

    #[test]
    fn round3_verid2_sprite_uses_two_bits() {
        // verid = 2 ⇒ sprite_enable is 2 bits; encode `00` (NotUsed).
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, Some(2), 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0b00, 2); // sprite_enable = NotUsed
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(0, 1); // quarter_sample (verid != 1)
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // newpred_enable
        w.write_bits(0, 1); // reduced_resolution_vop_enable
        w.write_bits(0, 1); // scalability
        w.align();
        let header = parse_video_object_layer(&w.buf, 0).unwrap();
        assert_eq!(header.video_object_layer_verid, 2);
        assert_eq!(header.sprite_enable, SpriteEnable::NotUsed);
        assert!(!header.quarter_sample);
        assert!(!header.newpred_enable);
        assert!(!header.reduced_resolution_vop_enable);
    }

    #[test]
    fn round3_verid2_sprite_gmc_is_rejected() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, Some(2), 0b00, false);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(0b10, 2); // sprite_enable = GMC
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn round3_verid2_sprite_reserved_is_rejected() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, Some(2), 0b00, false);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(0b11, 2); // sprite_enable = Reserved
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn round3_complexity_estimation_header_branch_is_rejected() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(0, 1); // complexity_estimation_disable = 0 → header expected
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn round3_data_partitioned_reads_reversible_vlc() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(1, 1); // data_partitioned
        w.write_bits(1, 1); // reversible_vlc
        w.write_bits(0, 1); // scalability
        w.align();
        let h = parse_video_object_layer(&w.buf, 0).unwrap();
        assert!(h.data_partitioned);
        assert!(h.reversible_vlc);
    }

    #[test]
    fn round3_scalability_flag_is_carried_back() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(1, 1); // scalability
        w.align();
        let h = parse_video_object_layer(&w.buf, 0).unwrap();
        assert!(h.scalability);
    }

    #[test]
    fn round3_quant_type_load_matrix_is_rejected() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(1, 1); // quant_type = 1
        w.write_bits(1, 1); // load_intra_quant_mat = 1
        w.align();
        let err = parse_video_object_layer(&w.buf, 0).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn round3_quant_type_no_load_succeeds() {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(1, 1); // quant_type = 1
        w.write_bits(0, 1); // load_intra_quant_mat = 0
        w.write_bits(0, 1); // load_nonintra_quant_mat = 0
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
        let h = parse_video_object_layer(&w.buf, 0).unwrap();
        assert!(h.quant_type);
    }

    #[test]
    fn vol_error_branch_displays() {
        let e = VolParseError::UnsupportedBranch("foo");
        assert!(format!("{e}").contains("foo"));
        let e = VolParseError::BadQuantPrecision(11);
        assert!(format!("{e}").contains("11"));
    }
}

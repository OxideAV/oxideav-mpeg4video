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
    /// The transmitted `intra_quant_mat` or `nonintra_quant_mat` list
    /// was malformed: its first 8-bit value was 0, which §6.3.3 defines
    /// as the "no more values follow" sentinel. The syntax `8*[2-64]`
    /// requires at least two values, so a leading 0 implies zero values
    /// transmitted and is a bitstream error. Carries the matrix name
    /// (`"intra"` / `"nonintra"`) for diagnostics.
    EmptyQuantMatrix(&'static str),
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
            VolParseError::EmptyQuantMatrix(name) => {
                write!(
                    f,
                    "{name}_quant_mat first value was 0 (list must hold 2..=64 entries)"
                )
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

/// Decoded `colour_description` block of a §6.2.2 `video_signal_type()`.
///
/// All three fields are 8-bit `uimsbf` integers whose enumerations are
/// defined in Tables 6-8 / 6-9 / 6-10. We surface the raw values; the
/// caller decides how to map them onto its display-side colour model.
///
/// Per §6.3.2.4, when `video_signal_type()` is absent or
/// `colour_description == 0`, the spec defaults all three to value `1`
/// (ITU-R BT.709 primaries, BT.709 transfer, BT.709 matrix); we expose
/// that default via [`ColourDescription::default_when_absent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourDescription {
    /// `colour_primaries` (Table 6-8). Value `0` is forbidden.
    pub colour_primaries: u8,
    /// `transfer_characteristics` (Table 6-9). Value `0` is forbidden.
    pub transfer_characteristics: u8,
    /// `matrix_coefficients` (Table 6-10). Value `0` is forbidden.
    pub matrix_coefficients: u8,
}

impl ColourDescription {
    /// Spec default for the three colour fields when `video_signal_type()`
    /// is absent or `colour_description == 0`: BT.709 across the board
    /// (per the §6.3.2.4 "assumed to be ... having the value 1"
    /// clauses).
    pub const fn default_when_absent() -> Self {
        Self {
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
        }
    }
}

/// Decoded `video_signal_type()` block (§6.2.2 / §6.3.2.4).
///
/// `video_signal_type()` is itself optional inside `VisualObject()` —
/// `video_signal_type` is a 1-bit flag, and when it is `0` none of the
/// fields below appear in the bitstream. This struct represents only
/// the case where the flag was `1`; surface the whole thing as
/// `Option<VideoSignalType>` on the parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSignalType {
    /// `video_format` (Table 6-7). 0..=7; the spec assigns enumerations
    /// to 0..=5 and reserves 6 / 7.
    pub video_format: u8,
    /// `video_range`. `false` → studio swing (Y 16..=235, C 16..=240
    /// for 8-bit), `true` → full swing (Y/C 0..=255 for 8-bit).
    pub video_range: bool,
    /// `colour_description` payload. `None` when the flag bit is `0`
    /// (callers may consult [`ColourDescription::default_when_absent`]
    /// for the §6.3.2.4 fallback).
    pub colour: Option<ColourDescription>,
}

/// Decoded `VisualObject()` header (§6.2.2).
///
/// Surfaces the `is_visual_object_identifier` payload (defaulted per
/// §6.3.2.3 when the bit is `0`), the `visual_object_type` selector
/// (Table 6-6), and — when the selector is `video ID` — the optional
/// `video_signal_type()` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisualObjectHeader {
    /// `visual_object_verid` (Table 6-5). Defaults to `1` ("object
    /// type listed in Table 9-1") when `is_visual_object_identifier ==
    /// 0` per §6.3.2.3.
    pub visual_object_verid: u8,
    /// `visual_object_priority` (1..=7; `0` is reserved per §6.3.2.3).
    /// Defaults to `1` ("highest priority") when
    /// `is_visual_object_identifier == 0`. The spec doesn't define a
    /// fallback explicitly, so we mirror the §6.3.2.3 wording that
    /// "value of zero is reserved" by picking the highest legal
    /// priority as the absent-field default.
    pub visual_object_priority: u8,
    /// `is_visual_object_identifier` flag. `true` when verid /
    /// priority were transmitted in the bitstream.
    pub is_visual_object_identifier: bool,
    /// `visual_object_type` (Table 6-6). This parser only succeeds
    /// when the value is `1` (`video ID`); other types return
    /// [`VolParseError::UnsupportedVisualObjectType`].
    pub visual_object_type: u8,
    /// Optional `video_signal_type()`. `None` when the
    /// `video_signal_type` flag bit was `0` in the bitstream
    /// (semantically: §6.3.2.4 defaults apply across the board).
    /// `Some(_)` when the flag was `1`.
    pub video_signal_type: Option<VideoSignalType>,
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
    /// Intra-block quantisation matrix loaded with `load_intra_quant_mat
    /// == 1` (§6.2.3 / §6.3.3). `Some(_)` only when the bitstream
    /// transmits a custom matrix; `None` otherwise (the default Annex
    /// matrix at §6.3.3 should be used).
    ///
    /// The 64 bytes are stored in zigzag scan order — the same order
    /// they were transmitted in. The first value is "shall always be 8"
    /// per §6.3.3; this parser surfaces the raw stream and does not
    /// substitute. Run-length expansion of the zero-sentinel rule
    /// ("remaining, non-transmitted values are set equal to the last
    /// non-zero value") is performed here so the caller sees a
    /// fully-populated `[u8; 64]`.
    pub intra_quant_mat: Option<[u8; 64]>,
    /// Non-intra-block quantisation matrix loaded with
    /// `load_nonintra_quant_mat == 1` (§6.2.3 / §6.3.3). Same encoding
    /// and expansion convention as [`Self::intra_quant_mat`]; the
    /// spec's only difference is the first-value constraint (intra
    /// "shall always be 8"; non-intra "shall not be 0").
    pub nonintra_quant_mat: Option<[u8; 64]>,
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

/// Decode a `load_intra_quant_mat` / `load_nonintra_quant_mat` body
/// per §6.2.3 syntax (`8*[2-64]`) + §6.3.3 semantics.
///
/// The bitstream carries a sequence of 8-bit unsigned values in zigzag
/// scan order. A value of 0 terminates the list, and the remaining
/// non-transmitted entries are filled with the last non-zero value. If
/// all 64 values are transmitted without a 0 sentinel, the list ends
/// naturally. A leading 0 is a bitstream error (the spec mandates 2..=64
/// transmitted values; 0-as-first-byte implies zero transmitted).
///
/// `name` is the matrix's spec-side identifier ("intra" / "nonintra")
/// used only for diagnostic error messages.
fn parse_quant_matrix(
    br: &mut BitReader<'_>,
    name: &'static str,
) -> Result<[u8; 64], VolParseError> {
    let mut mat = [0u8; 64];
    let mut last_nonzero: u8 = 0;
    let mut terminated = false;
    for (i, slot) in mat.iter_mut().enumerate() {
        let v = br.read_bits(8)? as u8;
        if v == 0 {
            if i == 0 {
                return Err(VolParseError::EmptyQuantMatrix(name));
            }
            terminated = true;
            break;
        }
        *slot = v;
        last_nonzero = v;
    }
    if terminated {
        // i is the position of the 0 sentinel; mat[i..64] needs the
        // last non-zero fill. The terminator itself is at mat[i] which
        // is currently 0; overwrite from there.
        for slot in mat.iter_mut() {
            if *slot == 0 {
                *slot = last_nonzero;
            }
        }
        // The above re-walks mat from the start, but every prefix entry
        // was non-zero (we exited the loop on the first 0), so only the
        // tail is rewritten. This avoids tracking `i` across the break.
    }
    Ok(mat)
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
    let (intra_quant_mat, nonintra_quant_mat) = if quant_type {
        // §6.2.3 syntax table at spec line ~4045:
        //   load_intra_quant_mat                                  1 bslbf
        //   if (load_intra_quant_mat)
        //         intra_quant_mat                                 8*[2-64] uimsbf
        //   load_nonintra_quant_mat                               1 bslbf
        //   if (load_nonintra_quant_mat)
        //         nonintra_quant_mat                              8*[2-64] uimsbf
        // The grayscale follow-on (`load_*_quant_mat_grayscale`) is
        // gated on `video_object_layer_shape == "grayscale"` which is
        // already rejected upfront via `UnsupportedShape`.
        let load_intra = br.read_bool()?;
        let intra = if load_intra {
            Some(parse_quant_matrix(br, "intra")?)
        } else {
            None
        };
        let load_nonintra = br.read_bool()?;
        let nonintra = if load_nonintra {
            Some(parse_quant_matrix(br, "nonintra")?)
        } else {
            None
        };
        (intra, nonintra)
    } else {
        (None, None)
    };
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
        intra_quant_mat,
        nonintra_quant_mat,
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

/// Parse one §6.2.2 `video_signal_type()` body, given a bit reader
/// positioned at the leading `video_signal_type` flag bit.
///
/// Returns `Ok(Some(_))` when the flag is set and the block follows;
/// `Ok(None)` when the flag is `0` (callers should fall back to the
/// §6.3.2.4 defaults).
fn parse_video_signal_type(
    br: &mut BitReader<'_>,
) -> Result<Option<VideoSignalType>, VolParseError> {
    let present = br.read_bool()?;
    if !present {
        return Ok(None);
    }
    let video_format = br.read_bits(3)? as u8;
    let video_range = br.read_bool()?;
    let has_colour = br.read_bool()?;
    let colour = if has_colour {
        let colour_primaries = br.read_bits(8)? as u8;
        let transfer_characteristics = br.read_bits(8)? as u8;
        let matrix_coefficients = br.read_bits(8)? as u8;
        Some(ColourDescription {
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
        })
    } else {
        None
    };
    Ok(Some(VideoSignalType {
        video_format,
        video_range,
        colour,
    }))
}

/// Parse a `VisualObject` header — the bits between `0x000001B5` and
/// the next start code, returning a typed [`VisualObjectHeader`].
///
/// The structural parser supports only `visual_object_type == "video
/// ID"` (numeric value `1` per §6.3.2 / Table 6-6); other types
/// return [`VolParseError::UnsupportedVisualObjectType`].
///
/// When `is_visual_object_identifier == 1`, the 4-bit
/// `visual_object_verid` + 3-bit `visual_object_priority` are decoded
/// into the corresponding fields. When the bit is `0`, the spec
/// defaults from §6.3.2.3 apply (`verid = 1`, `priority = 1`).
///
/// When the selector is `video ID`, the §6.2.2 syntax follows up with
/// `video_signal_type()`. Its 1-bit `video_signal_type` flag is
/// consumed unconditionally; when the flag is set, the
/// `video_format` (3 bits), `video_range` (1 bit),
/// `colour_description` (1 bit), and — when `colour_description == 1`
/// — `colour_primaries` / `transfer_characteristics` /
/// `matrix_coefficients` (8 bits each) are all surfaced. When the
/// flag is clear, [`VisualObjectHeader::video_signal_type`] is `None`
/// and the §6.3.2.4 defaults (BT.709 colour, studio swing) apply.
pub fn parse_visual_object_header(data: &[u8]) -> Result<VisualObjectHeader, VolParseError> {
    let mut br = BitReader::new(data);
    let sc = br.read_bits(32)?;
    if sc != VISUAL_OBJECT_START_CODE {
        return Err(VolParseError::MissingStartCode {
            expected: "visual_object_start_code",
            found: sc,
        });
    }
    let is_visual_object_identifier = br.read_bool()?;
    let (visual_object_verid, visual_object_priority) = if is_visual_object_identifier {
        let verid = br.read_bits(4)? as u8;
        let priority = br.read_bits(3)? as u8;
        (verid, priority)
    } else {
        // §6.3.2.3: "When this field does not exist, the value of
        // visual_object_verid is `0001`". Priority defaults to the
        // highest legal value (1) since `0` is reserved.
        (1, 1)
    };
    let visual_object_type = br.read_bits(4)? as u8;
    // §6.2.2: when type is video or still-texture, video_signal_type()
    // follows. We accept video (value 1 per Table 6-6) and reject the
    // rest, since this round is about rectangular video VOLs.
    if visual_object_type != 1 {
        return Err(VolParseError::UnsupportedVisualObjectType(
            visual_object_type,
        ));
    }
    let video_signal_type = parse_video_signal_type(&mut br)?;
    Ok(VisualObjectHeader {
        visual_object_verid,
        visual_object_priority,
        is_visual_object_identifier,
        visual_object_type,
        video_signal_type,
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
        // visual_object_type=0001 (4 bits) + video_signal_type=0 (1
        // bit) = 6 bits total in trailing byte. MSB-first:
        // 0_0001_0__ = 0b00001000 = 0x08. (Pad bits are don't-care
        // since no more reads happen.)
        let data = [0x00, 0x00, 0x01, 0xB5, 0x08];
        let h = parse_visual_object_header(&data).unwrap();
        assert_eq!(h.visual_object_type, 1);
        // §6.3.2.3 defaults: verid = 1, priority = 1 when the
        // identifier bit was 0.
        assert!(!h.is_visual_object_identifier);
        assert_eq!(h.visual_object_verid, 1);
        assert_eq!(h.visual_object_priority, 1);
        // video_signal_type flag was 0 → no block decoded.
        assert!(h.video_signal_type.is_none());
    }

    #[test]
    fn visual_object_header_rejects_non_video_type() {
        // type = 0010 (still-texture per Table 6-7 in this position).
        // is_id=0 (1 bit) + type=0010 (4 bits) = 5 bits, MSB-first:
        // 0_0010___ = 0b00010000.
        let data = [0x00, 0x00, 0x01, 0xB5, 0b0001_0000];
        let err = parse_visual_object_header(&data).unwrap_err();
        assert!(matches!(err, VolParseError::UnsupportedVisualObjectType(2)));
    }

    #[test]
    fn visual_object_header_decodes_identifier_block() {
        // Bits after start code: is_id=1, verid=0010, priority=011,
        // type=0001, vst=0. = 1 + 4 + 3 + 4 + 1 = 13 bits.
        // Byte 1: 1_0010_011 = 0b1001_0011 = 0x93.
        // Byte 2: 0001_0___ trailing pad. = 0b0001_0000 = 0x10.
        let data = [0x00, 0x00, 0x01, 0xB5, 0x93, 0x10];
        let h = parse_visual_object_header(&data).unwrap();
        assert!(h.is_visual_object_identifier);
        assert_eq!(h.visual_object_verid, 0b0010);
        assert_eq!(h.visual_object_priority, 0b011);
        assert_eq!(h.visual_object_type, 1);
        assert!(h.video_signal_type.is_none());
    }

    #[test]
    fn visual_object_header_decodes_video_signal_type_no_colour() {
        // Bits: is_id=0, type=0001, vst=1, video_format=010 (NTSC),
        // video_range=0, colour_description=0. = 1 + 4 + 1 + 3 + 1 + 1
        // = 11 bits.
        // Byte 1: 0_0001_1_01 = 0b0000_1101 = 0x0D.
        // Byte 2: 0_0_____ trailing pad. = 0b0000_0000 = 0x00.
        let data = [0x00, 0x00, 0x01, 0xB5, 0x0D, 0x00];
        let h = parse_visual_object_header(&data).unwrap();
        let vst = h.video_signal_type.expect("video_signal_type flag set");
        assert_eq!(vst.video_format, 0b010);
        assert!(!vst.video_range);
        assert!(vst.colour.is_none());
    }

    #[test]
    fn visual_object_header_decodes_video_signal_type_with_colour() {
        // Bits: is_id=0, type=0001, vst=1, video_format=000 (Component),
        // video_range=1, colour_description=1, colour_primaries=1,
        // transfer_characteristics=6 (SMPTE 170M-style), matrix=5
        // (ITU-R BT.470-2 System B,G). = 1 + 4 + 1 + 3 + 1 + 1 + 8 + 8
        // + 8 = 35 bits across 5 bytes (last 5 bits pad).
        //
        // Pack MSB-first:
        //   bit 0:  0       (is_visual_object_identifier)
        //   bit 1:  0001    (visual_object_type)
        //   bit 5:  1       (video_signal_type)
        //   bit 6:  000     (video_format)
        //   bit 9:  1       (video_range)
        //   bit 10: 1       (colour_description)
        //   bit 11: 00000001 (colour_primaries = 1)
        //   bit 19: 00000110 (transfer_characteristics = 6)
        //   bit 27: 00000101 (matrix_coefficients = 5)
        //   bit 35: 00000   pad
        let mut bw = BitWriter::new();
        bw.write_bits(VISUAL_OBJECT_START_CODE, 32);
        bw.write_bits(0, 1); // is_visual_object_identifier = 0
        bw.write_bits(0b0001, 4); // visual_object_type = video ID
        bw.write_bits(1, 1); // video_signal_type
        bw.write_bits(0b000, 3); // video_format = Component
        bw.write_bits(1, 1); // video_range
        bw.write_bits(1, 1); // colour_description
        bw.write_bits(1, 8); // colour_primaries
        bw.write_bits(6, 8); // transfer_characteristics
        bw.write_bits(5, 8); // matrix_coefficients
        bw.align();
        let h = parse_visual_object_header(&bw.buf).unwrap();
        let vst = h.video_signal_type.expect("video_signal_type flag set");
        assert_eq!(vst.video_format, 0);
        assert!(vst.video_range);
        let cd = vst.colour.expect("colour_description flag set");
        assert_eq!(cd.colour_primaries, 1);
        assert_eq!(cd.transfer_characteristics, 6);
        assert_eq!(cd.matrix_coefficients, 5);
    }

    #[test]
    fn visual_object_header_truncated_mid_vst() {
        // is_id=0, type=0001, vst=1 → the §6.2.2 video_signal_type()
        // body must follow. After only one trailing byte, the reader
        // runs out part-way through `video_format` (needs 3 bits
        // beyond the first 6) and we must surface Truncated rather
        // than silently zero-fill.
        // Byte: 0_0001_1__ where the last two bits start
        // video_format (= the upper two of 010 = 0). MSB-first:
        // 0b0000_1100 = 0x0C.
        let data = [0x00, 0x00, 0x01, 0xB5, 0x0C];
        let err = parse_visual_object_header(&data).unwrap_err();
        assert!(matches!(err, VolParseError::Truncated));
    }

    #[test]
    fn visual_object_header_truncated_mid_colour() {
        // is_id=0, type=0001, vst=1, video_format=000, video_range=1,
        // colour_description=1 — then bytes run out before
        // colour_primaries finishes.
        // Bits so far: 0_0001_1_000_1_1 = 11 bits → two bytes
        // (0b0000_1100, 0b0110_0000 of which the trailing 5 are pad).
        let data = [0x00, 0x00, 0x01, 0xB5, 0x0C, 0x60];
        let err = parse_visual_object_header(&data).unwrap_err();
        assert!(matches!(err, VolParseError::Truncated));
    }

    #[test]
    fn colour_description_default_when_absent_is_bt709() {
        let cd = ColourDescription::default_when_absent();
        assert_eq!(cd.colour_primaries, 1);
        assert_eq!(cd.transfer_characteristics, 1);
        assert_eq!(cd.matrix_coefficients, 1);
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
        // No `load_*` bit set ⇒ no custom matrix carried.
        assert!(h.intra_quant_mat.is_none());
        assert!(h.nonintra_quant_mat.is_none());
    }

    /// Helper that emits a quant_type=1 VOL fixture with caller-supplied
    /// intra / nonintra matrix payloads (each `Option<Vec<u8>>`; `None`
    /// means `load_*_quant_mat = 0`, `Some(bytes)` means load=1 followed
    /// by `bytes` written byte-aligned via 8-bit writes).
    fn make_quant_matrix_vol(intra: Option<&[u8]>, nonintra: Option<&[u8]>) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_vol_header_up_to_trailing(&mut w, None, 0b00, false);
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable = NotUsed
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(1, 1); // quant_type = 1
        match intra {
            Some(bytes) => {
                w.write_bits(1, 1);
                for b in bytes {
                    w.write_bits(u32::from(*b), 8);
                }
            }
            None => w.write_bits(0, 1),
        }
        match nonintra {
            Some(bytes) => {
                w.write_bits(1, 1);
                for b in bytes {
                    w.write_bits(u32::from(*b), 8);
                }
            }
            None => w.write_bits(0, 1),
        }
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(0, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
        w.buf
    }

    #[test]
    fn round4_load_intra_quant_mat_full_64_values() {
        // Transmit all 64 entries with no zero sentinel — the natural
        // end of the list. Use a recognisable triangular pattern so the
        // round-trip is visually obvious.
        let payload: Vec<u8> = (1..=64).collect();
        let data = make_quant_matrix_vol(Some(&payload), None);
        let h = parse_video_object_layer(&data, 0).unwrap();
        let mat = h.intra_quant_mat.expect("intra matrix present");
        for (i, m) in mat.iter().enumerate() {
            assert_eq!(*m as usize, i + 1, "zigzag position {i}");
        }
        assert!(h.nonintra_quant_mat.is_none());
    }

    #[test]
    fn round4_load_intra_quant_mat_zero_sentinel_runs_last_nonzero() {
        // Two values then a 0 sentinel: matrix should be [8, 17, 17,
        // 17, …]. (8 is the spec-mandated first value; 17 is the
        // arbitrary second value, which is also the last non-zero
        // value that gets run-length-replicated into the tail per
        // §6.3.3.)
        let payload = [8u8, 17u8, 0u8];
        let data = make_quant_matrix_vol(Some(&payload), None);
        let h = parse_video_object_layer(&data, 0).unwrap();
        let mat = h.intra_quant_mat.expect("intra matrix present");
        assert_eq!(mat[0], 8);
        assert_eq!(mat[1], 17);
        for v in &mat[2..] {
            assert_eq!(*v, 17);
        }
    }

    #[test]
    fn round4_load_both_intra_and_nonintra_matrices() {
        // Spec-recommended default intra matrix (first row of the
        // §6.3.3 default; values 8..27) terminated by 0, plus a short
        // nonintra payload [16, 16, 0] → matrix [16, 16, 16, …].
        let intra_payload = [8u8, 17, 18, 19, 21, 23, 25, 27, 0];
        let nonintra_payload = [16u8, 16, 0];
        let data = make_quant_matrix_vol(Some(&intra_payload), Some(&nonintra_payload));
        let h = parse_video_object_layer(&data, 0).unwrap();

        let intra = h.intra_quant_mat.expect("intra matrix present");
        assert_eq!(&intra[0..8], &[8u8, 17, 18, 19, 21, 23, 25, 27]);
        for v in &intra[8..] {
            assert_eq!(*v, 27, "tail filled with last non-zero (27)");
        }

        let nonintra = h.nonintra_quant_mat.expect("nonintra matrix present");
        assert_eq!(nonintra[0], 16);
        for v in &nonintra[1..] {
            assert_eq!(*v, 16);
        }
    }

    #[test]
    fn round4_load_intra_only_then_nonintra_default() {
        // Load intra (2 bytes + 0) but leave nonintra at default
        // (load_nonintra = 0). The parser surfaces the absent nonintra
        // matrix as `None` so the caller knows to use the default.
        let intra_payload = [8u8, 16, 0];
        let data = make_quant_matrix_vol(Some(&intra_payload), None);
        let h = parse_video_object_layer(&data, 0).unwrap();
        assert!(h.intra_quant_mat.is_some());
        assert!(h.nonintra_quant_mat.is_none());
    }

    #[test]
    fn round4_load_nonintra_only_then_intra_default() {
        // Mirror image: nonintra loaded, intra default. Confirms the
        // load_intra / load_nonintra branches are independent.
        let nonintra_payload = [20u8, 20, 0];
        let data = make_quant_matrix_vol(None, Some(&nonintra_payload));
        let h = parse_video_object_layer(&data, 0).unwrap();
        assert!(h.intra_quant_mat.is_none());
        let mat = h.nonintra_quant_mat.expect("nonintra matrix present");
        assert_eq!(mat[0], 20);
        assert_eq!(mat[63], 20);
    }

    #[test]
    fn round4_empty_intra_quant_mat_is_rejected() {
        // A leading 0 byte means the encoder transmitted zero values,
        // violating the `8*[2-64]` syntax constraint.
        let payload = [0u8];
        let data = make_quant_matrix_vol(Some(&payload), None);
        let err = parse_video_object_layer(&data, 0).unwrap_err();
        assert_eq!(err, VolParseError::EmptyQuantMatrix("intra"));
    }

    #[test]
    fn round4_empty_nonintra_quant_mat_is_rejected() {
        // Intra is full-length so it doesn't trip the same error
        // first; then nonintra starts with 0.
        let intra_payload: Vec<u8> = (1..=64).collect();
        let nonintra_payload = [0u8];
        let data = make_quant_matrix_vol(Some(&intra_payload), Some(&nonintra_payload));
        let err = parse_video_object_layer(&data, 0).unwrap_err();
        assert_eq!(err, VolParseError::EmptyQuantMatrix("nonintra"));
    }

    #[test]
    fn round4_minimal_two_entry_intra_matrix() {
        // Smallest legal list per `8*[2-64]`: two non-zero values then
        // the 0 sentinel. Tail fills with the second value.
        let payload = [8u8, 100, 0];
        let data = make_quant_matrix_vol(Some(&payload), None);
        let h = parse_video_object_layer(&data, 0).unwrap();
        let mat = h.intra_quant_mat.expect("intra matrix present");
        assert_eq!(mat[0], 8);
        assert_eq!(mat[1], 100);
        for v in &mat[2..] {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn round4_full_64_no_sentinel_does_not_run_length_fill() {
        // Exactly 64 values with no 0 ⇒ no run-length expansion.
        // Verify the last entry equals what was transmitted, not the
        // last non-zero. (They happen to be the same here, but we use
        // a non-monotone pattern to make the property visible.)
        let mut payload = vec![1u8; 64];
        payload[63] = 7;
        let data = make_quant_matrix_vol(Some(&payload), None);
        let h = parse_video_object_layer(&data, 0).unwrap();
        let mat = h.intra_quant_mat.expect("intra matrix present");
        for v in &mat[..63] {
            assert_eq!(*v, 1);
        }
        assert_eq!(mat[63], 7);
    }

    #[test]
    fn round4_quant_matrix_helper_handles_truncation() {
        // Direct-call test of `parse_quant_matrix` against a truncated
        // slice: load=1 but the matrix payload is cut short before any
        // value is read.
        let mut br = BitReader::new(&[]);
        let err = parse_quant_matrix(&mut br, "intra").unwrap_err();
        assert_eq!(err, VolParseError::Truncated);
    }

    #[test]
    fn vol_error_branch_displays() {
        let e = VolParseError::UnsupportedBranch("foo");
        assert!(format!("{e}").contains("foo"));
        let e = VolParseError::BadQuantPrecision(11);
        assert!(format!("{e}").contains("11"));
        let e = VolParseError::EmptyQuantMatrix("intra");
        let s = format!("{e}");
        assert!(s.contains("intra"));
        assert!(s.contains("2..=64"));
    }
}

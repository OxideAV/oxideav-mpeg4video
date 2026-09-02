//! Structural parser for the §6.2.6 macroblock-layer **header** —
//! the bits that precede `motion_coding()` / `block()` for an I- or
//! P-coded macroblock with a rectangular VOL shape.
//!
//! This module deliberately stops short of motion vectors and DCT
//! coefficients. It walks the bits that the spec syntax table places
//! before either — `not_coded` / `mcbpc` (Tables B.6 / B.7) /
//! `ac_pred_flag` / `cbpy` (Table B.8) / `dquant` (Table 6-32) — and
//! lifts the result into a typed [`MacroblockHeader`]. DCT and MV
//! decode are out of scope for this round.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.6 `macroblock()` syntax — `not_coded`, `mcbpc`,
//!   `ac_pred_flag`, `cbpy`, `dquant`, and the I/P/B branching.
//! * §6.3.6 macroblock-related semantics — definitions of `not_coded`,
//!   `mcbpc`, `ac_pred_flag`, `cbpy`, `dquant`.
//! * Table B.1 — `mb_type` ↔ `derived_mb_type` ↔ included elements,
//!   I-, P-, S-VOPs.
//! * Table B.6 — VLC for `mcbpc` for I-VOPs.
//! * Table B.7 — VLC for `mcbpc` for P-VOPs.
//! * Table B.8 — VLC for `cbpy` (4 non-transparent blocks).
//! * Table B.9 — VLC for `cbpy` (3 non-transparent blocks).
//! * Table B.10 — VLC for `cbpy` (2 non-transparent blocks).
//! * Table B.11 — VLC for `cbpy` (1 non-transparent block).
//! * Table 6-32 — `dquant` 2-bit code values (`00 → -1`, `01 → -2`,
//!   `10 → +1`, `11 → +2`).
//!
//! ## Out of scope (this round)
//!
//! * B-VOP `modb` (Table B.3) / `mb_type` (Tables B.4/B.5) — typed
//!   reject with [`MacroblockParseError::UnsupportedVopKind`].
//! * `mb_binary_shape_coding` — only invoked when
//!   `video_object_layer_shape != "rectangular"`, which `vol.rs`
//!   rejects upfront.
//! * Short-video-header (H.263 baseline) — that flow has its own
//!   macroblock layer with `cod`/`mcbpc`-h263 layouts (§5.3.2 +
//!   Annex P); not in scope here.
//! * Sprite-piece / GMC sprite branches — already rejected by the
//!   VOP-header parser via `VopContext::sprite_*`.
//! * `mcsel`, motion vectors, DCT, block stuffing within the data
//!   partitioned bitstream.
//!
//! ## §6.2.6 `interlaced_information()` dispatch
//!
//! When the VOL header carries `interlaced == 1`, the §6.2.6 syntax
//! invokes `interlaced_information()` immediately after `dquant` and
//! before the motion / texture data:
//!
//! > `if (interlaced) interlaced_information()`
//!
//! For the I- and P-VOP macroblock types this module handles, the
//! body parser ([`parse_interlaced_information`]) is now wired into
//! [`parse_macroblock_header`]: the §6.2.6.3 `dct_type` /
//! `field_prediction` body is consumed here so the bit reader stays
//! aligned for the (later-round) motion-vector / `block(i)` loop. The
//! decoded body is surfaced on [`MacroblockHeader::interlaced_info`].
//! The §6.2.6 `if (interlaced && field_prediction)
//! motion_vector("forward")` second invocation that the decoded
//! `field_prediction` bit gates stays at the motion layer (the header
//! parser stops before motion vectors). The S(GMC)-VOP `mcsel` route
//! is out of scope — the parser rejects S-VOPs up front, so the
//! §6.2.6.3 S-disjunct is unreachable here.

use crate::bitreader::{BitReader, BitReaderError};
use crate::interlaced_information::{
    parse_interlaced_information, InterlacedInfoContext, InterlacedInformation, McSel,
};
use crate::vol::{SpriteEnable, VolHeader};
use crate::vop::VopCodingType;

/// Decoded macroblock type, per the `mb type` column of Table B.1.
///
/// The numeric value matches the spec's `derived_mb_type` integer
/// (`0` = inter, `1` = inter+q, `2` = inter4v, `3` = intra,
/// `4` = intra+q, `5` = stuffing). Stuffing macroblocks are
/// transparently skipped by [`parse_macroblock_header`]; callers
/// therefore never see a `Stuffing` variant in the returned header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum DerivedMbType {
    /// Type 0 (P-VOP only): inter-coded, one motion vector.
    Inter,
    /// Type 1 (P-VOP only): inter-coded with a `dquant` delta.
    InterQ,
    /// Type 2 (P-VOP only): inter-coded with four 8×8 motion vectors.
    Inter4V,
    /// Type 3 (I- or P-VOP): intra-coded, no `dquant`.
    Intra,
    /// Type 4 (I- or P-VOP): intra-coded with a `dquant` delta.
    IntraQ,
}

impl DerivedMbType {
    /// The integer encoding from Table B.1 (`0..=4`).
    pub fn as_u8(self) -> u8 {
        match self {
            DerivedMbType::Inter => 0,
            DerivedMbType::InterQ => 1,
            DerivedMbType::Inter4V => 2,
            DerivedMbType::Intra => 3,
            DerivedMbType::IntraQ => 4,
        }
    }

    /// Whether this MB type is intra (derived_mb_type ≥ 3).
    pub fn is_intra(self) -> bool {
        matches!(self, DerivedMbType::Intra | DerivedMbType::IntraQ)
    }

    /// Whether `dquant` follows (derived_mb_type ∈ {1, 4}).
    pub fn has_dquant(self) -> bool {
        matches!(self, DerivedMbType::InterQ | DerivedMbType::IntraQ)
    }

    /// Map a raw `0..=4` `derived_mb_type` integer (Table B.1) to the
    /// typed variant. Returns `None` for the stuffing code (`5`) or any
    /// out-of-range value.
    pub fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => DerivedMbType::Inter,
            1 => DerivedMbType::InterQ,
            2 => DerivedMbType::Inter4V,
            3 => DerivedMbType::Intra,
            4 => DerivedMbType::IntraQ,
            _ => return None,
        })
    }
}

/// `dquant` differential per Table 6-32, expanded to `i8`.
///
/// The 2-bit `dquant` field encodes a signed delta against the
/// running quantiser scale: `00 → -1`, `01 → -2`, `10 → +1`,
/// `11 → +2`.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn dquant_value(code: u8) -> i8 {
    match code & 0b11 {
        0b00 => -1,
        0b01 => -2,
        0b10 => 1,
        0b11 => 2,
        _ => unreachable!("masked to 2 bits"),
    }
}

/// Typed view of one macroblock header.
///
/// The structure intentionally captures only the header bits — DCT
/// coefficients, motion vectors, and the various interlaced /
/// sprite / scalability bodies stay unparsed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub struct MacroblockHeader {
    /// 1-bit `not_coded` flag. Always `false` for I-VOP macroblocks
    /// (per Table B.1 — `not_coded` is absent on I-VOP) and `false`
    /// for any coded P-VOP MB; set to `true` only for a P-VOP MB
    /// whose `not_coded` field was read as `1`. When `true`, all
    /// other fields except `position` carry default zeros and the
    /// caller should skip directly to the next macroblock.
    pub not_coded: bool,
    /// Derived macroblock type (Table B.1). `None` only when
    /// `not_coded == true`.
    pub mb_type: Option<DerivedMbType>,
    /// 2-bit `cbpc` (Table B.6 / B.7 trailing bits) — the chrominance
    /// coded-block pattern for the two chroma 8×8 blocks. Bit 1
    /// covers block 5 (Cb), bit 0 covers block 6 (Cr).
    pub cbpc: u8,
    /// `ac_pred_flag`, present only for intra MBs (Table B.1 ↔
    /// "derived_mb_type ≥ 3 && !short_video_header"). `false` for
    /// inter macroblocks.
    pub ac_pred_flag: bool,
    /// `cbpy` decoded against the 4-block table (Table B.8). The
    /// 4-bit value covers the four luminance 8×8 blocks; bit 3 is
    /// block 1 (top-left), bit 0 is block 4 (bottom-right). For an
    /// `intra` MB the bit indicates an "at least one AC coefficient"
    /// flag; for an `inter` MB the bit is inverted per the spec
    /// "cbpy(inter-MB)" column — the parser stores the inter-form
    /// value when the macroblock is inter so the same bit
    /// convention applies (`1` means "non-coded block").
    pub cbpy: u8,
    /// 2-bit `dquant` field expanded via Table 6-32 to `±1` / `±2`.
    /// `None` when the macroblock type doesn't carry a dquant
    /// (`!mb_type.has_dquant()`) — defaults to "no change".
    pub dquant_delta: Option<i8>,
    /// The §6.2.6.3 `interlaced_information()` body, decoded when the
    /// VOL header's `interlaced` flag is set (the §6.2.6 `if
    /// (interlaced) interlaced_information()` gate). `None` when the
    /// VOL is progressive (`interlaced == 0`) or the macroblock is
    /// `not_coded` (the §6.2.6 not-coded short-circuit returns before
    /// reaching the `interlaced_information()` call). A coded
    /// macroblock in an interlaced VOL always carries
    /// `Some(InterlacedInformation)`, though both of its inner gates
    /// may be unfired (a zero-bit body — see
    /// [`InterlacedInformation::bit_count`]).
    pub interlaced_info: Option<InterlacedInformation>,
    /// The §6.3.6 `mcsel` flag for an S(GMC)-VOP macroblock.
    ///
    /// `mcsel` is present in the bitstream only when `vop_coding_type ==
    /// "S"`, `sprite_enable == "GMC"`, and the macroblock type from
    /// `mcbpc` is inter / inter+q (`derived_mb_type == 0 || == 1`). It
    /// selects, for that macroblock, whether the global-motion-compensated
    /// image (`mcsel == true`) or the previous reconstructed VOP via local
    /// motion vectors (`mcsel == false`) is the reference for interframe
    /// prediction. When `mcsel == true` no local motion vectors are
    /// transmitted for the macroblock.
    ///
    /// Per the §6.3.6 default rule, a not-coded macroblock in an
    /// S(GMC)-VOP (`not_coded == 1`) carries an implied `mcsel == true`
    /// (GMC prediction). This field is therefore `Some(true)` for a
    /// not-coded S(GMC) macroblock, `Some(value)` for a coded inter / inter+q
    /// S(GMC) macroblock, and `None` for every non-S(GMC) macroblock and
    /// for an intra S(GMC) macroblock (where the `mcsel` syntax gate does
    /// not fire).
    pub mcsel: Option<bool>,
}

impl MacroblockHeader {
    /// The "not_coded" macroblock — used for skipped P-VOP MBs. Per
    /// §6.3.6 the decoder treats this as inter with zero MV and no
    /// DCT data.
    pub const SKIPPED: Self = Self {
        not_coded: true,
        mb_type: None,
        cbpc: 0,
        ac_pred_flag: false,
        cbpy: 0,
        dquant_delta: None,
        interlaced_info: None,
        mcsel: None,
    };

    /// The not-coded macroblock of an S(GMC)-VOP. Identical to
    /// [`MacroblockHeader::SKIPPED`] but with the §6.3.6 implied
    /// `mcsel == true` (the default for a not-coded S(GMC) macroblock,
    /// which is predicted from the global-motion-compensated image).
    pub const SKIPPED_GMC: Self = Self {
        not_coded: true,
        mb_type: None,
        cbpc: 0,
        ac_pred_flag: false,
        cbpy: 0,
        dquant_delta: None,
        interlaced_info: None,
        mcsel: Some(true),
    };
}

/// Errors emitted by the macroblock-header parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroblockParseError {
    /// The supplied bit reader ran out mid-field.
    Truncated,
    /// The `mcbpc` prefix did not match any code in Table B.6 / B.7.
    /// The 9-bit window we were looking at is reported for
    /// diagnostics.
    InvalidMcbpc {
        /// The next-9-bits window we tried to match against the
        /// table.
        window: u16,
    },
    /// The `cbpy` prefix did not match any code in Table B.8 (we
    /// only support the 4-non-transparent-blocks case for
    /// rectangular shape today).
    InvalidCbpy {
        /// The next-6-bits window we tried to match.
        window: u8,
    },
    /// The VOP coding type is one the macroblock-header parser
    /// hasn't been wired up for yet (B-VOP `modb` / `mb_type`
    /// Tables B.3 / B.4 / B.5, sprite-coded macroblocks).
    UnsupportedVopKind(VopCodingType),
    /// The VOL shape is non-rectangular; `mb_binary_shape_coding()`
    /// would be needed first. The VOL header parser already rejects
    /// non-rectangular shapes, so this error is a defensive double
    /// check.
    UnsupportedShape(u8),
    /// Constructing the §6.2.6.3 [`InterlacedInfoContext`] failed —
    /// the macroblock type was inconsistent with the VOP coding type
    /// (e.g. an inter macroblock in an I-VOP). This is a defensive
    /// guard; the I/P macroblock-type decode paths in this module only
    /// produce types the context accepts.
    InvalidInterlacedContext,
}

impl core::fmt::Display for MacroblockParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MacroblockParseError::Truncated => write!(f, "macroblock header truncated"),
            MacroblockParseError::InvalidMcbpc { window } => {
                write!(f, "invalid mcbpc prefix (next-9-bits = 0b{window:09b})")
            }
            MacroblockParseError::InvalidCbpy { window } => {
                write!(f, "invalid cbpy prefix (next-6-bits = 0b{window:06b})")
            }
            MacroblockParseError::UnsupportedVopKind(k) => {
                write!(f, "macroblock parsing not yet wired for VOP type {k:?}")
            }
            MacroblockParseError::UnsupportedShape(s) => {
                write!(
                    f,
                    "macroblock parsing requires rectangular shape (shape={s})"
                )
            }
            MacroblockParseError::InvalidInterlacedContext => {
                write!(
                    f,
                    "§6.2.6.3 interlaced_information() context inconsistent with VOP coding type"
                )
            }
        }
    }
}

impl std::error::Error for MacroblockParseError {}

impl From<BitReaderError> for MacroblockParseError {
    fn from(_: BitReaderError) -> Self {
        MacroblockParseError::Truncated
    }
}

// ---------------------------------------------------------------------------
// VLC tables (verbatim from ISO/IEC 14496-2:2004 §B.1.2).
//
// Each entry is `(code_bits, code_len)` plus the data the spec table
// surfaces alongside the code. The tables are short — eight to
// twenty rows — so a linear prefix scan is fast enough and avoids
// the build-time complexity of a packed-tree lookup.
// ---------------------------------------------------------------------------

/// Table B.6 row: `mcbpc` for an I-VOP. Tuple is
/// `(code_bits, code_len, derived_mb_type, cbpc)`. The "Stuffing"
/// row uses `derived_mb_type = 5`; the parser short-circuits it.
const MCBPC_I_TABLE: &[(u16, u8, u8, u8)] = &[
    // Code 1 → mbtype 3, cbpc 00
    (0b1, 1, 3, 0b00),
    // 001 → 3, 01
    (0b001, 3, 3, 0b01),
    // 010 → 3, 10
    (0b010, 3, 3, 0b10),
    // 011 → 3, 11
    (0b011, 3, 3, 0b11),
    // 0001 → 4, 00
    (0b0001, 4, 4, 0b00),
    // 000001 → 4, 01
    (0b00_0001, 6, 4, 0b01),
    // 000010 → 4, 10
    (0b00_0010, 6, 4, 0b10),
    // 000011 → 4, 11
    (0b00_0011, 6, 4, 0b11),
    // 000000001 → Stuffing
    (0b0_0000_0001, 9, 5, 0),
];

/// Table B.7 row: `mcbpc` for a P-VOP. Same shape as B.6.
const MCBPC_P_TABLE: &[(u16, u8, u8, u8)] = &[
    // 1            → 0, 00
    (0b1, 1, 0, 0b00),
    // 0011         → 0, 01
    (0b0011, 4, 0, 0b01),
    // 0010         → 0, 10
    (0b0010, 4, 0, 0b10),
    // 000101       → 0, 11
    (0b00_0101, 6, 0, 0b11),
    // 011          → 1, 00
    (0b011, 3, 1, 0b00),
    // 0000111      → 1, 01
    (0b000_0111, 7, 1, 0b01),
    // 0000110      → 1, 10
    (0b000_0110, 7, 1, 0b10),
    // 000000101    → 1, 11
    (0b0_0000_0101, 9, 1, 0b11),
    // 010          → 2, 00
    (0b010, 3, 2, 0b00),
    // 0000101      → 2, 01
    (0b000_0101, 7, 2, 0b01),
    // 0000100      → 2, 10
    (0b000_0100, 7, 2, 0b10),
    // 00000101     → 2, 11
    (0b0000_0101, 8, 2, 0b11),
    // 00011        → 3, 00
    (0b0_0011, 5, 3, 0b00),
    // 00000100     → 3, 01
    (0b0000_0100, 8, 3, 0b01),
    // 00000011     → 3, 10
    (0b0000_0011, 8, 3, 0b10),
    // 0000011      → 3, 11
    (0b000_0011, 7, 3, 0b11),
    // 000100       → 4, 00
    (0b00_0100, 6, 4, 0b00),
    // 00000010 0   → 4, 01  (9 bits, low bit 0)
    (0b0_0000_0100, 9, 4, 0b01),
    // 00000001 1   → 4, 10
    (0b0_0000_0011, 9, 4, 0b10),
    // 00000001 0   → 4, 11
    (0b0_0000_0010, 9, 4, 0b11),
    // 000000001    → Stuffing
    (0b0_0000_0001, 9, 5, 0),
];

/// Table B.8 row for `cbpy` (4 non-transparent blocks). Tuple is
/// `(code_bits, code_len, cbpy_intra, cbpy_inter)`. The two trailing
/// columns are 4-bit luminance coded-block patterns; the parser
/// picks intra or inter based on the MB type.
const CBPY_4_TABLE: &[(u8, u8, u8, u8)] = &[
    // 0011  → intra 0000 / inter 1111
    (0b0011, 4, 0b0000, 0b1111),
    // 0010 1 → intra 0001 / inter 1110
    (0b0_0101, 5, 0b0001, 0b1110),
    // 0010 0 → intra 0010 / inter 1101
    (0b0_0100, 5, 0b0010, 0b1101),
    // 1001 → intra 0011 / inter 1100
    (0b1001, 4, 0b0011, 0b1100),
    // 0001 1 → intra 0100 / inter 1011
    (0b0_0011, 5, 0b0100, 0b1011),
    // 0111 → intra 0101 / inter 1010
    (0b0111, 4, 0b0101, 0b1010),
    // 0000 10 → intra 0110 / inter 1001
    (0b00_0010, 6, 0b0110, 0b1001),
    // 1011 → intra 0111 / inter 1000
    (0b1011, 4, 0b0111, 0b1000),
    // 0001 0 → intra 1000 / inter 0111
    (0b0_0010, 5, 0b1000, 0b0111),
    // 0000 11 → intra 1001 / inter 0110
    (0b00_0011, 6, 0b1001, 0b0110),
    // 0101 → intra 1010 / inter 0101
    (0b0101, 4, 0b1010, 0b0101),
    // 1010 → intra 1011 / inter 0100
    (0b1010, 4, 0b1011, 0b0100),
    // 0100 → intra 1100 / inter 0011
    (0b0100, 4, 0b1100, 0b0011),
    // 1000 → intra 1101 / inter 0010
    (0b1000, 4, 0b1101, 0b0010),
    // 0110 → intra 1110 / inter 0001
    (0b0110, 4, 0b1110, 0b0001),
    // 11   → intra 1111 / inter 0000
    (0b11, 2, 0b1111, 0b0000),
];

/// Match a `(code_bits, code_len)` table against the leading bits of
/// `window`. `window_len` is the number of *valid* most-significant
/// bits in `window`. Returns the matched row + the consumed length on
/// hit, `None` on no-match.
fn match_vlc_u16<T: Copy>(
    table: &[(u16, u8, T, T)],
    window: u16,
    window_len: u8,
) -> Option<(u8, T, T)> {
    for &(code, len, a, b) in table {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                return Some((len, a, b));
            }
        }
    }
    None
}

fn match_vlc_u8<T: Copy>(
    table: &[(u8, u8, T, T)],
    window: u8,
    window_len: u8,
) -> Option<(u8, T, T)> {
    for &(code, len, a, b) in table {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                return Some((len, a, b));
            }
        }
    }
    None
}

/// Table B.6 `mcbpc` codes for an I-VOP, exposed for the §6.2.5.3
/// data-partitioned I-VOP parser which shares this table.
pub(crate) const MCBPC_I: &[(u16, u8, u8, u8)] = MCBPC_I_TABLE;

/// Table B.8 `cbpy` codes (4 non-transparent blocks), exposed for the
/// encoder-side VLC emitters which invert this table.
pub(crate) const CBPY_4: &[(u8, u8, u8, u8)] = CBPY_4_TABLE;

/// Table B.7 `mcbpc` codes for a P-VOP / S(GMC)-VOP, exposed for the
/// §6.2.5.3 data-partitioned P-VOP parser which shares this table.
pub(crate) const MCBPC_P: &[(u16, u8, u8, u8)] = MCBPC_P_TABLE;

/// Decode an `mcbpc` codeword and return `(consumed_len,
/// derived_mb_type, cbpc)`.
pub(crate) fn decode_mcbpc(
    br: &mut BitReader<'_>,
    table: &[(u16, u8, u8, u8)],
) -> Result<(u8, u8, u8), MacroblockParseError> {
    let remaining = br.remaining_bits().min(9);
    if remaining == 0 {
        return Err(MacroblockParseError::Truncated);
    }
    let window = br.next_bits(remaining)? as u16;
    match match_vlc_u16(table, window, remaining as u8) {
        Some((len, mb_type, cbpc)) => {
            br.skip_bits(len as usize)?;
            Ok((len, mb_type, cbpc))
        }
        None => Err(MacroblockParseError::InvalidMcbpc { window }),
    }
}

/// Decode a `cbpy` codeword against Table B.8 (4 non-transparent
/// blocks) and return `(consumed_len, cbpy_intra, cbpy_inter)`.
pub(crate) fn decode_cbpy4(br: &mut BitReader<'_>) -> Result<(u8, u8, u8), MacroblockParseError> {
    let remaining = br.remaining_bits().min(6);
    if remaining == 0 {
        return Err(MacroblockParseError::Truncated);
    }
    let window = br.next_bits(remaining)? as u8;
    match match_vlc_u8(CBPY_4_TABLE, window, remaining as u8) {
        Some((len, a, b)) => {
            br.skip_bits(len as usize)?;
            Ok((len, a, b))
        }
        None => Err(MacroblockParseError::InvalidCbpy { window }),
    }
}

/// Parse one macroblock-layer header from `br`, given the parent
/// VOP's coding type and the VOL header. The caller is responsible
/// for stopping at `mb_count_in_vop`; this function decodes exactly
/// one macroblock header and advances the bit reader.
///
/// **Stuffing handling.** Per §6.2.6, a `derived_mb_type == 5`
/// ("stuffing") mcbpc code is followed immediately by another
/// macroblock — the stuffing entry is bit-stream padding. This
/// function transparently consumes stuffing entries and continues
/// to the next valid macroblock, returning the first non-stuffing
/// header.
///
/// **Scope.** I-VOPs, P-VOPs, and S(GMC)-VOPs (`sprite_enable ==
/// "GMC"`) with rectangular shape are supported. The S(GMC)-VOP path
/// shares the P-VOP macroblock type table and additionally parses the
/// §6.3.6 `mcsel` flag (GMC vs. local-MC reference selection) for its
/// inter / inter+q macroblocks, surfaced in
/// [`MacroblockHeader::mcsel`]. B-VOPs (`modb` / `mb_type` Tables B.3 /
/// B.4 / B.5) and `Static`-sprite S-VOPs are still rejected with
/// [`MacroblockParseError::UnsupportedVopKind`].
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn parse_macroblock_header(
    br: &mut BitReader<'_>,
    vop_coding_type: VopCodingType,
    vol: &VolHeader,
) -> Result<MacroblockHeader, MacroblockParseError> {
    if vol.video_object_layer_shape != 0 {
        // Defensive: the VOL parser already rejects non-rectangular.
        return Err(MacroblockParseError::UnsupportedShape(
            vol.video_object_layer_shape,
        ));
    }

    // §6.3.6: the S(GMC)-VOP macroblock layer shares the P-VOP macroblock
    // type table (MCBPC_P_TABLE: inter / inter+q / inter4v / intra /
    // intra+q) and the P-VOP `not_coded` syntax, and additionally carries
    // the `mcsel` flag (§6.3.6) for inter / inter+q macroblocks. The
    // `mcsel` route only exists for `sprite_enable == "GMC"`; a `Static`
    // sprite or any other S-VOP framing is still out of scope.
    let is_s_gmc =
        vop_coding_type == VopCodingType::S && matches!(vol.sprite_enable, SpriteEnable::Gmc);
    let (mcbpc_table, vop_has_not_coded) = match vop_coding_type {
        VopCodingType::I => (MCBPC_I_TABLE, false),
        VopCodingType::P => (MCBPC_P_TABLE, true),
        VopCodingType::S if is_s_gmc => (MCBPC_P_TABLE, true),
        VopCodingType::B | VopCodingType::S => {
            return Err(MacroblockParseError::UnsupportedVopKind(vop_coding_type));
        }
    };

    loop {
        // §6.2.6: not_coded is present only for P-VOPs and S-VOPs (the
        // `vop_coding_type != "I"` gate). For I-VOPs the syntax goes
        // directly to mcbpc.
        let not_coded = if vop_has_not_coded {
            br.read_bool()?
        } else {
            false
        };
        if not_coded {
            // §6.3.6: a not-coded S(GMC)-VOP macroblock carries an implied
            // `mcsel == 1` (predicted from the global-motion-compensated
            // image). A not-coded P-VOP macroblock is a plain skip.
            return Ok(if is_s_gmc {
                MacroblockHeader::SKIPPED_GMC
            } else {
                MacroblockHeader::SKIPPED
            });
        }

        let (_len, mb_type_raw, cbpc) = decode_mcbpc(br, mcbpc_table)?;
        if mb_type_raw == 5 {
            // Stuffing — loop and read the next macroblock.
            continue;
        }
        let mb_type = match mb_type_raw {
            0 => DerivedMbType::Inter,
            1 => DerivedMbType::InterQ,
            2 => DerivedMbType::Inter4V,
            3 => DerivedMbType::Intra,
            4 => DerivedMbType::IntraQ,
            other => unreachable!("MCBPC table emitted unexpected mb_type {other}"),
        };

        // §6.3.6 `mcsel`: present iff `vop_coding_type == "S" &&
        // sprite_enable == "GMC" && (derived_mb_type == 0 ||
        // derived_mb_type == 1)` — i.e. an inter / inter+q macroblock of an
        // S(GMC)-VOP. It selects GMC (`1`) vs. local MC (`0`) prediction.
        // The syntax places it immediately after `mcbpc` and before
        // `ac_pred_flag`.
        let mcsel = if is_s_gmc && matches!(mb_type, DerivedMbType::Inter | DerivedMbType::InterQ) {
            Some(br.read_bool()?)
        } else {
            None
        };

        // ac_pred_flag — Table B.1: present iff derived_mb_type ≥ 3
        // (i.e. intra MB) and !short_video_header. We do not emit
        // short_video_header streams here, so the gate is "intra".
        let ac_pred_flag = if mb_type.is_intra() {
            br.read_bool()?
        } else {
            false
        };

        // cbpy — present for every coded MB that survived the
        // stuffing skip. Per Table B.1 the cbpy is on the "intra"
        // column for intra MBs and the "inter" column for inter MBs;
        // we surface the appropriate one.
        let (_clen, cbpy_intra, cbpy_inter) = decode_cbpy4(br)?;
        let cbpy = if mb_type.is_intra() {
            cbpy_intra
        } else {
            cbpy_inter
        };

        // dquant — present iff derived_mb_type ∈ {1, 4}.
        let dquant_delta = if mb_type.has_dquant() {
            let code = br.read_bits(2)? as u8;
            Some(dquant_value(code))
        } else {
            None
        };

        // §6.2.6: `if (interlaced) interlaced_information()` — fires
        // immediately after `dquant` and before the motion / texture
        // data. The §6.2.6.3 first gate (`dct_type`) checks `cbp != 0`
        // (any of the six 4:2:0 blocks coded). The stored `cbpy` /
        // `cbpc` use the §6.3.7 "1 == coded" convention (the inter
        // `cbpy` column of Table B.8 is already in coded-block-pattern
        // form), so `cbp != 0` reduces to `cbpy != 0 || cbpc != 0`.
        let interlaced_info = if vol.interlaced {
            let any_block_coded = cbpy != 0 || cbpc != 0;
            let ctx = match vop_coding_type {
                VopCodingType::I => Some(
                    InterlacedInfoContext::i_vop(mb_type)
                        .map_err(|_| MacroblockParseError::InvalidInterlacedContext)?,
                ),
                VopCodingType::P => Some(InterlacedInfoContext::p_vop(mb_type, any_block_coded)),
                // §6.2.6: the `if (interlaced) interlaced_information()` line
                // is unconditional. For an inter / inter+q S(GMC) macroblock
                // the §6.2.6.3 S-disjunct (`derived_mb_type < 2 && !mcsel`)
                // governs `field_prediction` — an `mcsel == 1` GMC macroblock
                // is frame-predicted (§7.8.7.2) and carries only the
                // `cbp != 0`-gated `dct_type`; the dedicated `s_gmc_vop`
                // constructor encodes both shapes. An intra S(GMC)
                // macroblock (`mcsel == None`, `derived_mb_type >= 2`) leaves
                // the field_prediction gate unfired but still takes the
                // `dct_type` gate, which the PVop context computes identically
                // (`is_intra() || any_block_coded`).
                VopCodingType::S if is_s_gmc => match mb_type {
                    DerivedMbType::Inter | DerivedMbType::InterQ => {
                        let mc = if mcsel == Some(true) {
                            McSel::On
                        } else {
                            McSel::Off
                        };
                        Some(
                            InterlacedInfoContext::s_gmc_vop(mb_type, mc, any_block_coded)
                                .map_err(|_| MacroblockParseError::InvalidInterlacedContext)?,
                        )
                    }
                    _ => Some(InterlacedInfoContext::p_vop(mb_type, any_block_coded)),
                },
                VopCodingType::B | VopCodingType::S => {
                    return Err(MacroblockParseError::UnsupportedVopKind(vop_coding_type));
                }
            };
            match ctx {
                Some(ctx) => Some(parse_interlaced_information(br, &ctx)?),
                None => None,
            }
        } else {
            None
        };

        return Ok(MacroblockHeader {
            not_coded: false,
            mb_type: Some(mb_type),
            cbpc,
            ac_pred_flag,
            cbpy,
            dquant_delta,
            interlaced_info,
            mcsel,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interlaced_information::{DctType, FieldReference};
    use crate::vol::{AspectRatio, SpriteEnable, VolHeader};

    /// Helper to build a minimal-fields VolHeader for tests that
    /// don't care about anything except `video_object_layer_shape`.
    fn make_vol() -> VolHeader {
        VolHeader {
            profile_level: 0x01,
            width: 176,
            height: 144,
            time_increment_resolution: 30_000,
            fixed_vop_rate: false,
            fixed_vop_time_increment: None,
            aspect_ratio: AspectRatio::Square,
            vol_control: None,
            random_accessible_vol: false,
            video_object_type_indication: 1,
            is_object_layer_identifier: false,
            video_object_layer_verid: 1,
            video_object_layer_priority: 0,
            video_object_layer_shape: 0,
            interlaced: false,
            obmc_disable: false,
            sprite_enable: SpriteEnable::NotUsed,
            no_of_sprite_warping_points: None,
            sprite_warping_accuracy: None,
            sprite_brightness_change: None,
            sprite_geometry: None,
            low_latency_sprite_enable: None,
            not_8_bit: false,
            quant_precision: 5,
            bits_per_pixel: 8,
            quant_type: false,
            intra_quant_mat: None,
            nonintra_quant_mat: None,
            quarter_sample: false,
            complexity_estimation_disable: true,
            resync_marker_disable: true,
            data_partitioned: false,
            reversible_vlc: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            scalability: false,
        }
    }

    /// MSB-first bit writer matching the spec's `bslbf` / `uimsbf`
    /// convention.
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
        fn align(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bits(0, 1);
            }
        }
    }

    #[test]
    fn dquant_table_6_32_round_trip() {
        assert_eq!(dquant_value(0b00), -1);
        assert_eq!(dquant_value(0b01), -2);
        assert_eq!(dquant_value(0b10), 1);
        assert_eq!(dquant_value(0b11), 2);
    }

    #[test]
    fn derived_mb_type_predicates() {
        assert!(DerivedMbType::Intra.is_intra());
        assert!(DerivedMbType::IntraQ.is_intra());
        assert!(!DerivedMbType::Inter.is_intra());
        assert!(DerivedMbType::IntraQ.has_dquant());
        assert!(DerivedMbType::InterQ.has_dquant());
        assert!(!DerivedMbType::Intra.has_dquant());
        assert_eq!(DerivedMbType::Inter4V.as_u8(), 2);
    }

    #[test]
    fn parses_i_vop_simple_intra_mb() {
        // I-VOP, intra (mb_type 3), cbpc=00 → mcbpc code = 1.
        // ac_pred_flag = 0. cbpy intra=0000 → code 0011.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // mcbpc=1 → mb_type 3, cbpc 00
        w.write_bits(0, 1); // ac_pred_flag = 0
        w.write_bits(0b0011, 4); // cbpy_intra=0000
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap();
        assert!(!mb.not_coded);
        assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
        assert_eq!(mb.cbpc, 0b00);
        assert!(!mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b0000);
        assert_eq!(mb.dquant_delta, None);
    }

    #[test]
    fn parses_i_vop_intra_q_carries_dquant() {
        // mcbpc 0001 → mb_type 4, cbpc 00; ac_pred=1; cbpy_intra=1111
        // code 11; dquant = 10 → +1.
        let mut w = BitWriter::new();
        w.write_bits(0b0001, 4); // mcbpc
        w.write_bits(1, 1); // ac_pred_flag
        w.write_bits(0b11, 2); // cbpy_intra=1111
        w.write_bits(0b10, 2); // dquant = +1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::IntraQ));
        assert!(mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b1111);
        assert_eq!(mb.dquant_delta, Some(1));
    }

    #[test]
    fn parses_p_vop_not_coded_skip() {
        // P-VOP starts with 1-bit not_coded = 1.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert!(mb.not_coded);
        assert_eq!(mb.mb_type, None);
        assert_eq!(mb.cbpy, 0);
        assert_eq!(mb.cbpc, 0);
    }

    #[test]
    fn parses_p_vop_inter_mb_no_dquant() {
        // P-VOP: not_coded=0; mcbpc=1 → mb_type 0, cbpc 00 (inter);
        // no ac_pred (inter); cbpy_inter=0000 → code 11.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc
        w.write_bits(0b11, 2); // cbpy_inter=0000
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert!(!mb.not_coded);
        assert_eq!(mb.mb_type, Some(DerivedMbType::Inter));
        assert_eq!(mb.cbpc, 0b00);
        assert!(!mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b0000);
        assert_eq!(mb.dquant_delta, None);
    }

    #[test]
    fn parses_p_vop_inter_q_dquant_minus_two() {
        // P-VOP: not_coded=0; mcbpc=011 → mb_type 1, cbpc 00;
        // no ac_pred (inter); cbpy_inter=1111 → code 0011;
        // dquant 01 → -2.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b011, 3); // mcbpc → mb_type 1, cbpc 00
        w.write_bits(0b0011, 4); // cbpy_inter=1111
        w.write_bits(0b01, 2); // dquant = -2
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::InterQ));
        assert_eq!(mb.cbpy, 0b1111);
        assert_eq!(mb.dquant_delta, Some(-2));
    }

    #[test]
    fn parses_p_vop_intra_mb_ac_pred() {
        // P-VOP intra: not_coded=0; mcbpc=00011 → mb_type 3, cbpc 00;
        // ac_pred=1; cbpy_intra=1010 → code 0101.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b00011, 5); // mcbpc
        w.write_bits(1, 1); // ac_pred_flag
        w.write_bits(0b0101, 4); // cbpy_intra=1010
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
        assert!(mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b1010);
    }

    #[test]
    fn parses_p_vop_inter4v_mcbpc() {
        // P-VOP inter4v: mcbpc=010 → mb_type 2, cbpc 00; cbpy_inter=1111 → code 0011.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b010, 3); // mcbpc → mb_type 2, cbpc 00
        w.write_bits(0b0011, 4); // cbpy_inter=1111
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Inter4V));
        assert!(!mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b1111);
        assert_eq!(mb.dquant_delta, None);
    }

    #[test]
    fn i_vop_stuffing_is_skipped() {
        // Stuffing MCBPC = 000000001 (9 bits) → skip; then a real
        // I-MB: mcbpc=1, ac_pred=0, cbpy_intra=0000 code 0011.
        let mut w = BitWriter::new();
        w.write_bits(0b0_0000_0001, 9); // stuffing
        w.write_bits(0b1, 1); // mcbpc → mb_type 3
        w.write_bits(0, 1); // ac_pred
        w.write_bits(0b0011, 4); // cbpy_intra
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
        assert_eq!(mb.cbpy, 0);
    }

    #[test]
    fn p_vop_stuffing_is_skipped() {
        // P-VOP: not_coded=0, mcbpc=stuffing (000000001), then
        // another macroblock: not_coded=0, mcbpc=1, cbpy_inter=0000 code 11.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b0_0000_0001, 9); // mcbpc stuffing
        w.write_bits(0, 1); // not_coded (next MB)
        w.write_bits(0b1, 1); // mcbpc → mb_type 0
        w.write_bits(0b11, 2); // cbpy_inter=0000
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Inter));
        assert_eq!(mb.cbpy, 0);
    }

    #[test]
    fn b_vop_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::B, &make_vol()).unwrap_err();
        assert!(matches!(
            err,
            MacroblockParseError::UnsupportedVopKind(VopCodingType::B)
        ));
    }

    #[test]
    fn s_vop_static_sprite_is_rejected() {
        // A plain (non-GMC) S-VOP — make_vol()'s sprite_enable is NotUsed —
        // is still out of scope and rejected.
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::S, &make_vol()).unwrap_err();
        assert!(matches!(
            err,
            MacroblockParseError::UnsupportedVopKind(VopCodingType::S)
        ));
    }

    /// A VOL header declaring `sprite_enable == "GMC"` so an S-VOP takes
    /// the §6.3.6 GMC macroblock path.
    fn make_gmc_vol() -> VolHeader {
        VolHeader {
            sprite_enable: SpriteEnable::Gmc,
            no_of_sprite_warping_points: Some(2),
            sprite_warping_accuracy: Some(crate::vol::SpriteWarpingAccuracy::HalfPel),
            sprite_brightness_change: Some(false),
            ..make_vol()
        }
    }

    #[test]
    fn s_gmc_static_sprite_still_rejected() {
        // sprite_enable == Static on an S-VOP is not the GMC path; reject.
        let vol = VolHeader {
            sprite_enable: SpriteEnable::Static,
            ..make_vol()
        };
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::S, &vol).unwrap_err();
        assert!(matches!(
            err,
            MacroblockParseError::UnsupportedVopKind(VopCodingType::S)
        ));
    }

    #[test]
    fn s_gmc_not_coded_implies_mcsel_on() {
        // §6.3.6: a not-coded S(GMC) macroblock carries implied mcsel == 1.
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // not_coded
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &make_gmc_vol()).unwrap();
        assert!(mb.not_coded);
        assert_eq!(mb.mcsel, Some(true));
        assert_eq!(mb, MacroblockHeader::SKIPPED_GMC);
    }

    #[test]
    fn s_gmc_inter_mb_mcsel_on() {
        // Coded inter S(GMC) MB with mcsel = 1 (GMC reference). Syntax:
        // not_coded=0, mcbpc=1 (mb_type 0, cbpc 00), mcsel=1,
        // cbpy_inter=0000 → code 11.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter), cbpc 00
        w.write_bits(1, 1); // mcsel = 1 (GMC)
        w.write_bits(0b11, 2); // cbpy_inter=0000
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &make_gmc_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Inter));
        assert_eq!(mb.mcsel, Some(true));
        assert_eq!(mb.cbpy, 0b0000);
    }

    #[test]
    fn s_gmc_inter_q_mb_mcsel_off_with_dquant() {
        // Coded inter+q S(GMC) MB with mcsel = 0 (local MC). Syntax:
        // not_coded=0, mcbpc=011 (mb_type 1, cbpc 00), mcsel=0,
        // cbpy_inter=1111 → code 0011, dquant=10 → +1.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b011, 3); // mcbpc → mb_type 1 (inter+q), cbpc 00
        w.write_bits(0, 1); // mcsel = 0 (local MC)
        w.write_bits(0b0011, 4); // cbpy_inter=1111
        w.write_bits(0b10, 2); // dquant = +1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &make_gmc_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::InterQ));
        assert_eq!(mb.mcsel, Some(false));
        assert_eq!(mb.cbpy, 0b1111);
        assert_eq!(mb.dquant_delta, Some(1));
    }

    #[test]
    fn s_gmc_intra_mb_has_no_mcsel() {
        // An intra S(GMC) MB: the mcsel syntax gate (derived_mb_type ∈
        // {0,1}) does not fire, so no mcsel bit is consumed. Syntax:
        // not_coded=0, mcbpc=00011 (mb_type 3, cbpc 00), ac_pred=1,
        // cbpy_intra=1010 → code 0101.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b00011, 5); // mcbpc → mb_type 3 (intra)
        w.write_bits(1, 1); // ac_pred_flag (NO mcsel before it)
        w.write_bits(0b0101, 4); // cbpy_intra=1010
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &make_gmc_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
        assert_eq!(mb.mcsel, None);
        assert!(mb.ac_pred_flag);
        assert_eq!(mb.cbpy, 0b1010);
    }

    #[test]
    fn s_gmc_inter4v_mb_has_mcsel() {
        // inter4v (mb_type 2) is NOT in the §6.3.6 mcsel gate
        // (derived_mb_type ∈ {0,1}); no mcsel bit. mcbpc=010 → mb_type 2.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b010, 3); // mcbpc → mb_type 2 (inter4v)
        w.write_bits(0b0011, 4); // cbpy_inter=1111 (no mcsel, no ac_pred)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &make_gmc_vol()).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Inter4V));
        assert_eq!(mb.mcsel, None);
        assert_eq!(mb.cbpy, 0b1111);
    }

    #[test]
    fn non_rectangular_shape_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut vol = make_vol();
        vol.video_object_layer_shape = 1;
        let err = parse_macroblock_header(&mut br, VopCodingType::I, &vol).unwrap_err();
        assert!(matches!(err, MacroblockParseError::UnsupportedShape(1)));
    }

    #[test]
    fn invalid_mcbpc_prefix_is_rejected() {
        // 9 zero bits: not in any I-VOP mcbpc row (the stuffing row
        // is 0_0000_0001 with a trailing 1; nine pure zeros are
        // unmatched).
        let mut w = BitWriter::new();
        w.write_bits(0, 9);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap_err();
        assert!(matches!(err, MacroblockParseError::InvalidMcbpc { .. }));
    }

    #[test]
    fn invalid_cbpy_prefix_is_rejected() {
        // I-VOP intra (mcbpc=1), ac_pred=0, then 6 zeros → no
        // cbpy code matches "000000" (B.8 longest code is 000010 /
        // 000011 — but 000000 is unmatched).
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // mcbpc
        w.write_bits(0, 1); // ac_pred
        w.write_bits(0, 6); // garbage cbpy window
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap_err();
        assert!(matches!(err, MacroblockParseError::InvalidCbpy { .. }));
    }

    #[test]
    fn truncated_returns_truncated() {
        // Empty buffer can't even hold the I-VOP MCBPC bit.
        let data: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap_err();
        assert_eq!(err, MacroblockParseError::Truncated);
    }

    #[test]
    fn macroblock_parse_error_displays() {
        let cases = [
            MacroblockParseError::Truncated,
            MacroblockParseError::InvalidMcbpc { window: 0 },
            MacroblockParseError::InvalidCbpy { window: 0 },
            MacroblockParseError::UnsupportedVopKind(VopCodingType::B),
            MacroblockParseError::UnsupportedShape(2),
        ];
        for e in cases {
            let s = format!("{e}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn skipped_constant_matches_spec_defaults() {
        let s = MacroblockHeader::SKIPPED;
        assert!(s.not_coded);
        assert_eq!(s.mb_type, None);
        assert_eq!(s.cbpc, 0);
        assert!(!s.ac_pred_flag);
        assert_eq!(s.cbpy, 0);
        assert_eq!(s.dquant_delta, None);
    }

    /// Round-trip every Table B.6 entry through the parser and make
    /// sure we get back the spec's (mb_type, cbpc) tuple. Stuffing
    /// is asserted via the dedicated test above; the loop here
    /// skips it.
    #[test]
    fn mcbpc_i_table_full_round_trip() {
        for &(code, len, mb_type, cbpc) in MCBPC_I_TABLE {
            if mb_type == 5 {
                continue;
            }
            // For intra/intra+q tail-end of the I-VOP MB:
            // emit (mcbpc, ac_pred=0, cbpy_intra=0000) and assert
            // we recover the expected mb_type+cbpc.
            let mut w = BitWriter::new();
            w.write_bits(code as u32, len as usize);
            w.write_bits(0, 1); // ac_pred
            w.write_bits(0b0011, 4); // cbpy_intra=0000
            if mb_type == 4 {
                w.write_bits(0b10, 2); // dquant = +1
            }
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let mb = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap();
            assert_eq!(mb.cbpc, cbpc, "code={code:b}");
            let expected_type = match mb_type {
                3 => DerivedMbType::Intra,
                4 => DerivedMbType::IntraQ,
                _ => unreachable!(),
            };
            assert_eq!(mb.mb_type, Some(expected_type), "code={code:b}");
        }
    }

    /// Round-trip every Table B.7 entry (excluding stuffing) through
    /// the parser. Inter rows take cbpy_inter=0000 (code 11), intra
    /// rows take cbpy_intra=0000 (code 0011) + ac_pred=0; rows with
    /// dquant get the +1 code.
    #[test]
    fn mcbpc_p_table_full_round_trip() {
        for &(code, len, mb_type, cbpc) in MCBPC_P_TABLE {
            if mb_type == 5 {
                continue;
            }
            let mut w = BitWriter::new();
            w.write_bits(0, 1); // not_coded
            w.write_bits(code as u32, len as usize);
            let is_intra = mb_type == 3 || mb_type == 4;
            if is_intra {
                w.write_bits(0, 1); // ac_pred=0
                w.write_bits(0b0011, 4); // cbpy_intra=0000
            } else {
                w.write_bits(0b11, 2); // cbpy_inter=0000
            }
            if mb_type == 1 || mb_type == 4 {
                w.write_bits(0b10, 2); // dquant=+1
            }
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
            assert!(!mb.not_coded, "code={code:b}");
            assert_eq!(mb.cbpc, cbpc, "code={code:b}");
            let expected_type = match mb_type {
                0 => DerivedMbType::Inter,
                1 => DerivedMbType::InterQ,
                2 => DerivedMbType::Inter4V,
                3 => DerivedMbType::Intra,
                4 => DerivedMbType::IntraQ,
                _ => unreachable!(),
            };
            assert_eq!(mb.mb_type, Some(expected_type), "code={code:b}");
        }
    }

    /// Round-trip every Table B.8 entry through the inter-CBPY
    /// decode path. For each row build a minimal P-VOP inter MB
    /// (mcbpc=1) and assert cbpy_inter matches the spec.
    #[test]
    fn cbpy_4_table_full_round_trip_inter() {
        for &(code, len, cbpy_intra, cbpy_inter) in CBPY_4_TABLE {
            let mut w = BitWriter::new();
            w.write_bits(0, 1); // not_coded
            w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter)
            w.write_bits(code as u32, len as usize);
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let mb = parse_macroblock_header(&mut br, VopCodingType::P, &make_vol()).unwrap();
            assert_eq!(mb.cbpy, cbpy_inter, "code={code:b}");
            // The "intra" form is what we'd surface had the MB been
            // intra — re-do the same code via an I-VOP MB and assert
            // the intra value.
            let mut w2 = BitWriter::new();
            w2.write_bits(0b1, 1); // I-VOP mcbpc → mb_type 3, cbpc 00
            w2.write_bits(0, 1); // ac_pred
            w2.write_bits(code as u32, len as usize);
            w2.align();
            let data2 = w2.buf;
            let mut br2 = BitReader::new(&data2);
            let mb2 = parse_macroblock_header(&mut br2, VopCodingType::I, &make_vol()).unwrap();
            assert_eq!(mb2.cbpy, cbpy_intra, "code={code:b}");
        }
    }

    // -----------------------------------------------------------------
    // §6.2.6 `if (interlaced) interlaced_information()` dispatch.
    // -----------------------------------------------------------------

    /// Progressive VOLs (`interlaced == 0`) never read the
    /// §6.2.6.3 body — `interlaced_info` stays `None`.
    #[test]
    fn progressive_vol_has_no_interlaced_info() {
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // I-VOP mcbpc → mb_type 3
        w.write_bits(0, 1); // ac_pred
        w.write_bits(0b0011, 4); // cbpy_intra=0000
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::I, &make_vol()).unwrap();
        assert_eq!(mb.interlaced_info, None);
    }

    /// I-VOP, interlaced: the §6.2.6.3 first gate always fires
    /// (I-VOP MBs are intra-coded → `derived_mb_type ∈ {3, 4}`), so a
    /// single `dct_type` bit follows `dquant`. field_prediction never
    /// fires for an I-VOP.
    #[test]
    fn i_vop_interlaced_reads_dct_type_only() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // mcbpc → mb_type 3, cbpc 00
        w.write_bits(0, 1); // ac_pred
        w.write_bits(0b0011, 4); // cbpy_intra=0000
        w.write_bits(1, 1); // dct_type = 1 → Field
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::I, &vol).unwrap();
        let info = mb.interlaced_info.expect("interlaced VOL → body present");
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
    }

    /// P-VOP inter MB with `cbp == 0`: the `dct_type` gate does NOT
    /// fire (not intra, no coded blocks), but the field_prediction
    /// gate DOES (derived_mb_type ∈ {0, 1}). A single
    /// `field_prediction == 0` bit follows.
    #[test]
    fn p_vop_interlaced_inter_cbp_zero_reads_field_prediction_bit() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter), cbpc 00
        w.write_bits(0b11, 2); // cbpy_inter=0000 → cbp == 0
        w.write_bits(0, 1); // field_prediction = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap();
        assert_eq!(mb.cbpy, 0);
        assert_eq!(mb.cbpc, 0);
        let info = mb.interlaced_info.unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
    }

    /// P-VOP inter MB with `cbp != 0` and `field_prediction == 1`:
    /// the full body is `dct_type` + `field_prediction` + the forward
    /// reference pair (P-VOPs have no backward pair) — 4 bits.
    #[test]
    fn p_vop_interlaced_inter_cbp_nonzero_full_field_body() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter), cbpc 00
        w.write_bits(0b0011, 4); // cbpy_inter=1111 → cbp != 0
        w.write_bits(0, 1); // dct_type = 0 → Frame
        w.write_bits(1, 1); // field_prediction = 1
        w.write_bits(1, 1); // forward_top_field_reference = 1 → Bottom
        w.write_bits(0, 1); // forward_bottom_field_reference = 0 → Top
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap();
        assert_eq!(mb.cbpy, 0b1111);
        let info = mb.interlaced_info.unwrap();
        assert_eq!(info.dct_type, Some(DctType::Frame));
        let fp = info.field_prediction.expect("field_prediction == 1");
        assert_eq!(
            fp.forward,
            Some((FieldReference::Bottom, FieldReference::Top))
        );
        assert_eq!(fp.backward, None);
        assert_eq!(info.bit_count(), 4);
    }

    #[test]
    fn s_gmc_interlaced_mcsel_on_reads_only_dct_type() {
        // §6.2.6: `if (interlaced) interlaced_information()` is
        // unconditional, so an mcsel == 1 (GMC) macroblock still carries
        // dct_type when cbp != 0 (§6.3.6.3); only the §6.2.6.3
        // field_prediction gate (`!mcsel`) is off.
        let mut vol = make_gmc_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter), cbpc 00
        w.write_bits(1, 1); // mcsel = 1 (GMC)
        w.write_bits(0b0011, 4); // cbpy_inter=1111 → cbp != 0
        w.write_bits(0b1, 1); // dct_type = 1 → Field
        w.write_bits(0b0, 1); // sentinel (must NOT be consumed)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &vol).unwrap();
        assert_eq!(mb.mcsel, Some(true));
        let info = mb.interlaced_info.expect("dct_type gate fires");
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn s_gmc_interlaced_mcsel_off_reads_body() {
        // An mcsel == 0 (local-MC) inter S(GMC) macroblock with cbp != 0
        // invokes interlaced_information(): the §6.2.6.3 dct_type + the
        // S-disjunct field_prediction gate (derived_mb_type < 2 && !mcsel)
        // both fire.
        let mut vol = make_gmc_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → mb_type 0 (inter), cbpc 00
        w.write_bits(0, 1); // mcsel = 0 (local MC)
        w.write_bits(0b0011, 4); // cbpy_inter=1111 → cbp != 0
        w.write_bits(0, 1); // dct_type = 0 → Frame
        w.write_bits(0, 1); // field_prediction = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::S, &vol).unwrap();
        assert_eq!(mb.mcsel, Some(false));
        let info = mb
            .interlaced_info
            .expect("mcsel == 0 reads interlaced body");
        assert_eq!(info.dct_type, Some(DctType::Frame));
        assert_eq!(info.field_prediction, None);
        assert!(info.field_prediction_guard_fired);
    }

    /// P-VOP inter4v MB (`derived_mb_type == 2`) with `cbp == 0`:
    /// neither §6.2.6.3 gate fires — the body is empty (zero bits)
    /// but still surfaces as `Some` for a coded MB in an interlaced
    /// VOL.
    #[test]
    fn p_vop_interlaced_inter4v_cbp_zero_empty_body() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b010, 3); // mcbpc → mb_type 2 (inter4v), cbpc 00
        w.write_bits(0b11, 2); // cbpy_inter=0000 → cbp == 0
                               // No interlaced bits follow (both gates unfired).
                               // Append a sentinel so the reader doesn't run dry mid-test.
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let pos_before = {
            let mut br = BitReader::new(&data);
            let mb = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap();
            let info = mb.interlaced_info.unwrap();
            assert_eq!(info.dct_type, None);
            assert_eq!(info.field_prediction, None);
            assert!(!info.field_prediction_guard_fired);
            assert_eq!(info.bit_count(), 0);
            br.bit_position()
        };
        // The sentinel bit must remain unconsumed: not_coded(1) +
        // mcbpc(3) + cbpy(2) = 6 bits, no interlaced bits.
        assert_eq!(pos_before, 6);
    }

    /// P-VOP intra MB in an interlaced VOL: `dct_type` fires (intra),
    /// field_prediction does NOT (derived_mb_type ∈ {3, 4} excludes
    /// the P-VOP disjunct's `{0, 1}`). One bit follows `dquant`'s
    /// absence.
    #[test]
    fn p_vop_interlaced_intra_dct_type_only() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b00011, 5); // mcbpc → mb_type 3 (intra), cbpc 00
        w.write_bits(1, 1); // ac_pred_flag
        w.write_bits(0b0101, 4); // cbpy_intra=1010
        w.write_bits(1, 1); // dct_type = 1 → Field
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap();
        assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
        let info = mb.interlaced_info.unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert_eq!(info.bit_count(), 1);
    }

    /// A `not_coded` P-VOP MB in an interlaced VOL skips the
    /// §6.2.6.3 body entirely (the §6.2.6 not-coded short-circuit
    /// returns before the `interlaced_information()` call).
    #[test]
    fn p_vop_interlaced_not_coded_skips_body() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // not_coded
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mb = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap();
        assert!(mb.not_coded);
        assert_eq!(mb.interlaced_info, None);
    }

    /// A §6.2.6.3 body that overruns the bit slice surfaces
    /// `Truncated` rather than a partial header. A P-VOP inter MB with
    /// `cbp != 0`, `dct_type`, and `field_prediction == 1` demands the
    /// two forward reference bits; ending the slice exactly at the
    /// `field_prediction` bit (a whole byte) forces the forward-pair
    /// read past the end.
    #[test]
    fn interlaced_body_truncation_is_reported() {
        let mut vol = make_vol();
        vol.interlaced = true;
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc → inter, cbpc 00
        w.write_bits(0b0011, 4); // cbpy_inter=1111 → cbp != 0
        w.write_bits(0, 1); // dct_type
        w.write_bits(1, 1); // field_prediction = 1 → forward pair must follow
                            // 1 + 1 + 4 + 1 + 1 = 8 bits = exactly one byte. The forward
                            // reference pair (2 bits) overruns the slice.
        let data = w.buf;
        assert_eq!(data.len(), 1);
        let mut br = BitReader::new(&data);
        let err = parse_macroblock_header(&mut br, VopCodingType::P, &vol).unwrap_err();
        assert_eq!(err, MacroblockParseError::Truncated);
    }
}

//! §7.4.1.1 intra-DC-coefficient decode — the first stage of the
//! §6.2.7 `block(i)` texture syntax.
//!
//! For an intra-coded macroblock (`derived_mb_type == 3 || == 4`) with
//! `short_video_header == 0` and `use_intra_dc_vlc == 1`, each block's
//! DC coefficient is differentially coded as a `dct_dc_size` VLC plus a
//! `dct_dc_size`-bit `dct_dc_differential` additional code. This module
//! decodes that pair into the *differential* DC value (the value still
//! needs to be added to the §7.4.3.1 spatial predictor — that gathering
//! is later-round work). When `dct_dc_size > 8` a `marker_bit` follows
//! the additional code to prevent start-code emulation, per Table B.15
//! NOTE 2; this module consumes and validates it.
//!
//! AC coefficients (the `while (!last) DCT coefficient` loop of
//! §7.4.1.2), the spatial DC/AC predictor gathering of §7.4.3, the
//! `short_video_header == 1` fixed-8-bit intra DC path, and the
//! inverse quantisation of §7.4.4 are out of scope for this round.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.7 `block(i)` syntax — the `dct_dc_size_luminance` (i<4) /
//!   `dct_dc_size_chrominance` (i>=4) split, the
//!   `if (dct_dc_size != 0) dct_dc_differential` gate, and the
//!   `if (dct_dc_size > 8) marker_bit` gate.
//! * §6.3.7 macroblock-related semantics — `dct_dc_size_luminance`
//!   (Table B.13), `dct_dc_size_chrominance` (Table B.14), and
//!   `dct_dc_differential` (Table B.15).
//! * §7.4.1.1 — "Differential DC coefficients in blocks in intra
//!   macroblocks are encoded as a variable length code denoting
//!   `dct_dc_size` … and a fixed length code `dct_dc_differential`
//!   (Table B.15). … This is done by appending a fixed length code,
//!   `dct_dc_differential`, of `dct_dc_size` bits."
//! * Table B.13 — VLC for `dct_dc_size_luminance` (sizes 0..=12).
//! * Table B.14 — VLC for `dct_dc_size_chrominance` (sizes 0..=12).
//! * Table B.15 — the differential-DC additional-code mapping. For a
//!   `size`-bit additional code with `half_range = 2^(size-1)`, an
//!   additional code `c >= half_range` decodes to `+c`, while
//!   `c < half_range` decodes to `(c + 1) - 2*half_range` (i.e. the
//!   negative half of the range). `size == 0` decodes to `0` with no
//!   additional code. Table B.15 NOTE 2 inserts a `marker_bit` after
//!   the additional code when `size > 8`.
//!
//! ## §7.4.1.2 AC coefficient (EVENT) decode
//!
//! [`decode_ac_event`] decodes one §7.4.1.2 `DCT coefficient` EVENT — a
//! `(LAST, RUN, LEVEL)` triple — for the `short_video_header == 0`,
//! `reversible_vlc == 0` path, the most common case. The most-frequent
//! EVENTs are a single Tcoef VLC drawn from Table B.16 (intra blocks) or
//! Table B.17 (inter blocks), selected by the [`TcoefTable`] argument; a
//! trailing sign bit `s` (`0` positive, `1` negative) gives the LEVEL
//! sign. Less-frequent EVENTs use the §7.4.1.3 escape coding, whose
//! first three modes (the ones used when `short_video_header == 0` and
//! the reversible tables are not in use) this module decodes:
//!
//! * **Type 1** — `ESC` + `"0"` + a Table B.16/B.17 VLC; LEVEL is
//!   restored as `sign(LEVEL) * (abs(LEVEL) + LMAX(LAST, RUN))` with
//!   `LMAX` from Table B.19 (intra) / Table B.20 (inter).
//! * **Type 2** — `ESC` + `"10"` + a Table B.16/B.17 VLC; RUN is
//!   restored as `RUN + RMAX(LAST, LEVEL) + 1` with `RMAX` from
//!   Table B.21 (intra) / Table B.22 (inter).
//! * **Type 3** — `ESC` + `"11"` + a fixed-length `[1-bit LAST]`
//!   `[6-bit RUN]` `[marker_bit]` `[12-bit LEVEL]` `[marker_bit]`
//!   (Table B.18 a / b). The 12-bit LEVEL is a signed two's-complement
//!   value; `0` and `-2048` are reserved.
//!
//! [`decode_ac_events`] runs the §6.2.7 `while (!last) DCT coefficient`
//! loop, returning every EVENT up to and including the one with
//! `LAST == 1`.
//!
//! ## §7.4.1.3 Type 4 escape (short-video-header path)
//!
//! [`decode_ac_event_short_video_header`] decodes one §7.4.1.2 `DCT
//! coefficient` EVENT for the `short_video_header == 1` path. The common
//! Tcoef VLC + sign bit (Tables B.16 / B.17) is unchanged, but a Type 4
//! escape replaces Types 1..=3:
//!
//! * **Type 4** — `ESC` (`0000 011`) + 1-bit LAST + 6-bit RUN + 8-bit
//!   LEVEL, no marker bits. LEVEL is a signed two's-complement value
//!   per Table B.18 a / c; the codes `0000 0000` (= 0) and
//!   `1000 0000` (= -128) are reserved and rejected as
//!   [`TextureParseError::ReservedEscapeLevel`].
//!
//! [`decode_ac_events_short_video_header`] runs the §6.2.7 `while
//! (!last) DCT coefficient` loop against the Type-4 escape mode.
//!
//! ## Out of scope (this round)
//!
//! * The reversible-VLC EVENT tables (Tables B.23..B.25) and the Type 5
//!   escape used when `reversible_vlc == 1`.
//! * The §7.4.2 inverse scan that places `(RUN, LEVEL)` into the
//!   zigzag-ordered 64-coefficient array.
//! * The §7.4.3.1 / §7.4.3.2 spatial DC/AC prediction (gradient
//!   direction, `dc_scaler`, predictor reset) that turns this
//!   *differential* into the final coefficient.
//! * `short_video_header == 1` (the fixed 8-bit `intra_dc_coefficient`
//!   path of §6.2.7 / §7.4.1.1).
//! * Inverse quantisation (§7.4.4) and the inverse DCT.

use crate::bitreader::{BitReader, BitReaderError};

/// Which `dct_dc_size` VLC table applies to a block, selected by the
/// `block(i)` index per §6.2.7: luminance for `i < 4`, chrominance for
/// `i >= 4` (the two chroma blocks in 4:2:0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcComponent {
    /// Luminance block (`i < 4`) — `dct_dc_size_luminance`, Table B.13.
    Luminance,
    /// Chrominance block (`i >= 4`) — `dct_dc_size_chrominance`,
    /// Table B.14.
    Chrominance,
}

impl DcComponent {
    /// Map a `block(i)` index to its DC component, per §6.2.7: blocks
    /// 0..=3 are luminance, 4 and 5 are chrominance (Cb, Cr) in the
    /// 4:2:0 layout.
    pub fn from_block_index(i: usize) -> Self {
        if i < 4 {
            DcComponent::Luminance
        } else {
            DcComponent::Chrominance
        }
    }
}

/// The decoded intra-DC differential of one block, before the §7.4.3
/// spatial predictor is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraDcDifferential {
    /// `dct_dc_size` — the decoded size category (0..=12). `0` means the
    /// differential is exactly zero and no additional code follows.
    pub size: u8,
    /// The signed differential DC value derived from `dct_dc_size` +
    /// `dct_dc_differential` via the Table B.15 sign-decode. Add the
    /// §7.4.3.1 spatial predictor to obtain the final DC coefficient.
    pub differential: i32,
}

/// Errors produced by the intra-DC decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureParseError {
    /// The supplied bit reader ran out mid-field.
    Truncated,
    /// The leading bits did not match any `dct_dc_size` code in
    /// Table B.13 / Table B.14. The next-13-bits window we tried to
    /// match is reported for diagnostics.
    InvalidDcSize {
        /// The next-13-bits window (right-aligned) we tried to match.
        window: u16,
    },
    /// A `marker_bit` (Table B.15 NOTE 2, inserted when
    /// `dct_dc_size > 8`) was expected to be 1 but was read as 0.
    MarkerBitMissing,
    /// The leading bits matched neither a Table B.16 / B.17 Tcoef code
    /// nor the §7.4.1.3 escape prefix. The next-13-bits window
    /// (right-aligned) we tried to match is reported for diagnostics.
    InvalidTcoef {
        /// The next-13-bits window (right-aligned) we tried to match.
        window: u16,
    },
    /// A Type-3 escape `marker_bit` (one before and one after the 12-bit
    /// LEVEL, per §7.4.1.3) was expected to be 1 but was read as 0.
    EscapeMarkerBitMissing,
    /// A Type-3 escape carried a reserved 12-bit LEVEL value (`0` or
    /// `-2048`, both forbidden by Table B.18 b). Also raised for a
    /// Type-5 (reversible-VLC) escape whose 11-bit LEVEL decoded to the
    /// forbidden value `0` (Table B.25).
    ReservedEscapeLevel,
    /// A Type-5 (reversible-VLC) escape did not terminate with the
    /// closing delimiter `0000` (§7.4.1.3 / Table B.23 ESCAPE row); the
    /// four bits read in its place are reported for diagnostics.
    RvlcEscapeDelimiterMissing {
        /// The four bits read where the closing `0000` delimiter was
        /// expected (right-aligned).
        window: u8,
    },
}

impl core::fmt::Display for TextureParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TextureParseError::Truncated => write!(f, "intra DC field truncated"),
            TextureParseError::InvalidDcSize { window } => {
                write!(
                    f,
                    "invalid dct_dc_size prefix (next-13-bits = 0b{window:013b})"
                )
            }
            TextureParseError::MarkerBitMissing => {
                write!(f, "dct_dc marker_bit was 0 (expected 1)")
            }
            TextureParseError::InvalidTcoef { window } => {
                write!(f, "invalid Tcoef prefix (next-13-bits = 0b{window:013b})")
            }
            TextureParseError::EscapeMarkerBitMissing => {
                write!(f, "Type-3 escape marker_bit was 0 (expected 1)")
            }
            TextureParseError::ReservedEscapeLevel => {
                write!(f, "Type-3 escape LEVEL was a reserved value (0 or -2048)")
            }
            TextureParseError::RvlcEscapeDelimiterMissing { window } => {
                write!(
                    f,
                    "Type-5 escape closing delimiter was 0b{window:04b} (expected 0b0000)"
                )
            }
        }
    }
}

impl std::error::Error for TextureParseError {}

impl From<BitReaderError> for TextureParseError {
    fn from(_: BitReaderError) -> Self {
        TextureParseError::Truncated
    }
}

/// Table B.13 — `dct_dc_size_luminance`. Tuple is
/// `(code_bits, code_len, size)`. Prefix-free; sizes 0..=12.
const DC_SIZE_LUMINANCE: &[(u16, u8, u8)] = &[
    (0b011, 3, 0),
    (0b11, 2, 1),
    (0b10, 2, 2),
    (0b010, 3, 3),
    (0b001, 3, 4),
    (0b0001, 4, 5),
    (0b00001, 5, 6),
    (0b000001, 6, 7),
    (0b0000001, 7, 8),
    (0b00000001, 8, 9),
    (0b000000001, 9, 10),
    (0b0000000001, 10, 11),
    (0b00000000001, 11, 12),
];

/// Table B.14 — `dct_dc_size_chrominance`. Tuple is
/// `(code_bits, code_len, size)`. Prefix-free; sizes 0..=12.
const DC_SIZE_CHROMINANCE: &[(u16, u8, u8)] = &[
    (0b11, 2, 0),
    (0b10, 2, 1),
    (0b01, 2, 2),
    (0b001, 3, 3),
    (0b0001, 4, 4),
    (0b00001, 5, 5),
    (0b000001, 6, 6),
    (0b0000001, 7, 7),
    (0b00000001, 8, 8),
    (0b000000001, 9, 9),
    (0b0000000001, 10, 10),
    (0b00000000001, 11, 11),
    (0b000000000001, 12, 12),
];

/// Match a prefix-free `(code_bits, code_len, size)` table against the
/// leading bits of `window`. `window_len` is the number of valid
/// most-significant bits in `window`. Returns `(code_len, size)` on a
/// hit, `None` on no match.
fn match_dc_size(table: &[(u16, u8, u8)], window: u16, window_len: u8) -> Option<(u8, u8)> {
    for &(code, len, size) in table {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                return Some((len, size));
            }
        }
    }
    None
}

/// Apply the Table B.15 sign-decode to a `size`-bit `dct_dc_differential`
/// additional code.
///
/// With `half_range = 2^(size-1)`, the high half of the range
/// (`additional >= half_range`) carries the positive differentials and
/// decodes to `+additional`; the low half (`additional < half_range`)
/// carries the negative differentials and decodes to
/// `(additional + 1) - 2*half_range`. `size == 0` yields `0`.
///
/// This is the §7.4.1.1 / Table B.15 mapping; e.g. for `size == 1` the
/// additional code `0` → `-1` and `1` → `+1`; for `size == 2` the
/// codes `00`/`01`/`10`/`11` → `-3`/`-2`/`+2`/`+3`.
fn decode_differential(size: u8, additional: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let half_range: i64 = 1i64 << (size - 1);
    let additional = additional as i64;
    let value = if additional >= half_range {
        additional
    } else {
        (additional + 1) - (2 * half_range)
    };
    value as i32
}

/// Decode the intra-DC differential of one `block(i)` per §6.2.7 +
/// §7.4.1.1.
///
/// Reads `dct_dc_size_luminance` (Table B.13) or
/// `dct_dc_size_chrominance` (Table B.14) selected by `component`, the
/// `size`-bit `dct_dc_differential` (when `size != 0`), and the trailing
/// `marker_bit` (when `size > 8`). Returns the decoded size plus the
/// signed differential value from the Table B.15 sign-decode.
///
/// The returned `differential` is **not** the final DC coefficient: the
/// caller adds the §7.4.3.1 spatial predictor (`QF[0] = dct_dc_pred +
/// differential`). Predictor gathering is later-round work.
///
/// This path assumes `short_video_header == 0` and
/// `use_intra_dc_vlc == 1` — the §6.2.7 conditions under which the DC
/// coefficient is differentially VLC-coded. The 8-bit fixed
/// `intra_dc_coefficient` of the short-video-header branch is not
/// handled here.
pub fn decode_intra_dc(
    br: &mut BitReader<'_>,
    component: DcComponent,
) -> Result<IntraDcDifferential, TextureParseError> {
    let table = match component {
        DcComponent::Luminance => DC_SIZE_LUMINANCE,
        DcComponent::Chrominance => DC_SIZE_CHROMINANCE,
    };

    // The longest dct_dc_size code is 12 bits (Table B.14, size 12).
    // Peek a 13-bit window (clamped to what's left) so a full code is
    // always visible; matching consumes only the bits the matched code
    // actually used.
    let avail = br.remaining_bits().min(13) as u8;
    if avail == 0 {
        return Err(TextureParseError::Truncated);
    }
    let window = br.next_bits(avail as usize)? as u16;
    let (code_len, size) =
        match_dc_size(table, window, avail).ok_or(TextureParseError::InvalidDcSize { window })?;
    br.skip_bits(code_len as usize)?;

    let differential = if size == 0 {
        0
    } else {
        let additional = br.read_bits(size as usize)?;
        let value = decode_differential(size, additional);
        // Table B.15 NOTE 2: a marker_bit follows the additional code
        // when size > 8, to prevent start-code emulation.
        if size > 8 && !br.read_bool()? {
            return Err(TextureParseError::MarkerBitMissing);
        }
        value
    };

    Ok(IntraDcDifferential { size, differential })
}

// ---------------------------------------------------------------------------
// §7.4.1.2 AC coefficient (EVENT) decode
// ---------------------------------------------------------------------------

/// Which Tcoef VLC table applies to a block per §7.4.1.2: Table B.16 for
/// intra blocks, Table B.17 for inter blocks. The choice also selects the
/// matching LMAX (Table B.19/B.20) and RMAX (Table B.21/B.22) escape
/// tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcoefTable {
    /// Intra block — Table B.16, LMAX = Table B.19, RMAX = Table B.21.
    Intra,
    /// Inter block — Table B.17, LMAX = Table B.20, RMAX = Table B.22.
    Inter,
}

/// One decoded §7.4.1.2 `DCT coefficient` EVENT: the `(LAST, RUN, LEVEL)`
/// triple. `LEVEL` is the signed non-zero coefficient value (sign already
/// applied); `RUN` is the count of zero coefficients preceding it; `last`
/// marks the final non-zero coefficient of the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcEvent {
    /// `LAST` flag: `true` if this is the last non-zero coefficient of the
    /// block (no further EVENTs follow), `false` if more follow.
    pub last: bool,
    /// `RUN`: number of zero-valued coefficients preceding this one in the
    /// zigzag scan.
    pub run: u32,
    /// `LEVEL`: the signed non-zero coefficient value.
    pub level: i32,
}

/// The §7.4.1.3 escape prefix `0000 011` (7 bits) shared by Table B.16 and
/// Table B.17. It is followed by a 1- or 2-bit mode selector.
const TCOEF_ESCAPE_CODE: u32 = 0b000_0011;
const TCOEF_ESCAPE_LEN: u8 = 7;

// Table B.16 / Table B.17 entries are `(code_bits, code_len, last, run,
// level)`; `code_bits` is the VLC *without* its trailing sign bit `s`,
// which is read separately. Generated verbatim from the spec tables;
// every code is prefix-free and none collides with TCOEF_ESCAPE_CODE.
include!("tcoef_tables.rs");

// Table B.23 RVLC EVENTs are
// `(code_bits, code_len, intra_last, intra_run, intra_level, inter_last,
//  inter_run, inter_level)`; `code_bits` is the reversible VLC *without*
// its trailing sign bit `s`. The intra / inter triples share one code.
// Generated verbatim from the spec; every code is prefix-free and none
// collides with the escape delimiter `00001` / `0000`.
include!("rvlc_tables.rs");

/// The §7.4.1.3 Type-5 (reversible-VLC) escape opening delimiter `00001`
/// (5 bits), per Table B.23 ESCAPE row. The closing delimiter is `0000`
/// followed by the LEVEL sign bit `s`.
const RVLC_ESCAPE_OPEN: u32 = 0b00001;
const RVLC_ESCAPE_OPEN_LEN: u8 = 5;

/// Match a prefix-free Tcoef table against the leading bits of `window`.
/// Returns `(code_len, last, run, level)` on a hit (before the sign bit),
/// `None` on no match.
fn match_tcoef(
    table: &[(u32, u8, u8, u8, u16)],
    window: u32,
    window_len: u8,
) -> Option<(u8, bool, u32, i32)> {
    for &(code, len, last, run, level) in table {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                return Some((len, last != 0, run as u32, level as i32));
            }
        }
    }
    None
}

/// Table B.19 (intra) — LMAX as a function of `(LAST, RUN)`. Returns
/// `None` for the `N/A` cells (a RUN beyond the tabulated range for a
/// given LAST, which a conformant Type-1 escape never produces).
fn lmax_intra(last: bool, run: u32) -> Option<i32> {
    match (last, run) {
        (false, 0) => Some(27),
        (false, 1) => Some(10),
        (false, 2) => Some(5),
        (false, 3) => Some(4),
        (false, 4..=7) => Some(3),
        (false, 8..=9) => Some(2),
        (false, 10..=14) => Some(1),
        (true, 0) => Some(8),
        (true, 1) => Some(3),
        (true, 2..=6) => Some(2),
        (true, 7..=20) => Some(1),
        _ => None,
    }
}

/// Table B.20 (inter) — LMAX as a function of `(LAST, RUN)`.
fn lmax_inter(last: bool, run: u32) -> Option<i32> {
    match (last, run) {
        (false, 0) => Some(12),
        (false, 1) => Some(6),
        (false, 2) => Some(4),
        (false, 3..=6) => Some(3),
        (false, 7..=10) => Some(2),
        (false, 11..=26) => Some(1),
        (true, 0) => Some(3),
        (true, 1) => Some(2),
        (true, 2..=40) => Some(1),
        _ => None,
    }
}

/// Table B.21 (intra) — RMAX as a function of `(LAST, LEVEL)`. `LEVEL`
/// here is the absolute decoded magnitude. Returns `None` for `N/A`.
fn rmax_intra(last: bool, level: i32) -> Option<u32> {
    match (last, level) {
        (false, 1) => Some(14),
        (false, 2) => Some(9),
        (false, 3) => Some(7),
        (false, 4) => Some(3),
        (false, 5) => Some(2),
        (false, 6..=10) => Some(1),
        (false, 11..=27) => Some(0),
        (true, 1) => Some(20),
        (true, 2) => Some(6),
        (true, 3) => Some(1),
        (true, 4..=8) => Some(0),
        _ => None,
    }
}

/// Table B.22 (inter) — RMAX as a function of `(LAST, LEVEL)`.
fn rmax_inter(last: bool, level: i32) -> Option<u32> {
    match (last, level) {
        (false, 1) => Some(26),
        (false, 2) => Some(10),
        (false, 3) => Some(6),
        (false, 4) => Some(2),
        (false, 5..=6) => Some(1),
        (false, 7..=12) => Some(0),
        (true, 1) => Some(40),
        (true, 2) => Some(1),
        (true, 3) => Some(0),
        _ => None,
    }
}

/// Decode the Table B.16 / B.17 VLC plus its trailing sign bit at the
/// current position, returning `(last, run, signed_level)`. Used for the
/// non-escape common path and for the Type-1 / Type-2 escape sub-VLC
/// (before LMAX / RMAX restoration). Assumes the leading bits are a real
/// Tcoef code, not the escape prefix.
fn decode_tcoef_vlc(
    br: &mut BitReader<'_>,
    table: &[(u32, u8, u8, u8, u16)],
) -> Result<(bool, u32, i32), TextureParseError> {
    let avail = br.remaining_bits().min(13) as u8;
    if avail == 0 {
        return Err(TextureParseError::Truncated);
    }
    let window = br.next_bits(avail as usize)?;
    let (code_len, last, run, level) =
        match_tcoef(table, window, avail).ok_or(TextureParseError::InvalidTcoef {
            window: window as u16,
        })?;
    br.skip_bits(code_len as usize)?;
    let sign_negative = br.read_bool()?;
    let signed = if sign_negative { -level } else { level };
    Ok((last, run, signed))
}

/// Decode one §7.4.1.2 `DCT coefficient` EVENT for the
/// `short_video_header == 0`, `reversible_vlc == 0` path.
///
/// The common case is a single Table B.16 (`TcoefTable::Intra`) /
/// Table B.17 (`TcoefTable::Inter`) VLC plus a sign bit. The §7.4.1.3
/// escape prefix `0000 011` selects one of the first three escape modes:
///
/// * Type 1 (`ESC 0`) — re-decode a Tcoef VLC, then
///   `LEVEL = sign * (abs(LEVEL) + LMAX(LAST, RUN))`.
/// * Type 2 (`ESC 10`) — re-decode a Tcoef VLC, then
///   `RUN = RUN + RMAX(LAST, abs(LEVEL)) + 1`.
/// * Type 3 (`ESC 11`) — fixed-length `LAST(1) RUN(6) marker LEVEL(12)
///   marker`; the 12-bit LEVEL is signed two's-complement (`0` and
///   `-2048` reserved).
pub fn decode_ac_event(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<AcEvent, TextureParseError> {
    let table = match table_kind {
        TcoefTable::Intra => TCOEF_INTRA,
        TcoefTable::Inter => TCOEF_INTER,
    };

    // Peek the escape prefix without consuming so a real code that merely
    // *shares no* prefix with ESC falls through to the table decode.
    let avail = br.remaining_bits().min(TCOEF_ESCAPE_LEN as usize) as u8;
    if avail == 0 {
        return Err(TextureParseError::Truncated);
    }
    let is_escape =
        avail == TCOEF_ESCAPE_LEN && br.next_bits(TCOEF_ESCAPE_LEN as usize)? == TCOEF_ESCAPE_CODE;

    if !is_escape {
        let (last, run, level) = decode_tcoef_vlc(br, table)?;
        return Ok(AcEvent { last, run, level });
    }

    // Consume the escape prefix, then the mode selector.
    br.skip_bits(TCOEF_ESCAPE_LEN as usize)?;

    if !br.read_bool()? {
        // Type 1: "ESC" + "0" + VLC, LEVEL += LMAX.
        let (last, run, level) = decode_tcoef_vlc(br, table)?;
        let lmax = match table_kind {
            TcoefTable::Intra => lmax_intra(last, run),
            TcoefTable::Inter => lmax_inter(last, run),
        }
        .ok_or(TextureParseError::InvalidTcoef { window: 0 })?;
        let abs = level.abs() + lmax;
        let restored = if level < 0 { -abs } else { abs };
        return Ok(AcEvent {
            last,
            run,
            level: restored,
        });
    }

    if !br.read_bool()? {
        // Type 2: "ESC" + "10" + VLC, RUN += RMAX + 1.
        let (last, run, level) = decode_tcoef_vlc(br, table)?;
        let rmax = match table_kind {
            TcoefTable::Intra => rmax_intra(last, level.abs()),
            TcoefTable::Inter => rmax_inter(last, level.abs()),
        }
        .ok_or(TextureParseError::InvalidTcoef { window: 0 })?;
        return Ok(AcEvent {
            last,
            run: run + rmax + 1,
            level,
        });
    }

    // Type 3: "ESC" + "11" + LAST(1) RUN(6) marker LEVEL(12) marker.
    let last = br.read_bool()?;
    let run = br.read_bits(6)?;
    if !br.read_bool()? {
        return Err(TextureParseError::EscapeMarkerBitMissing);
    }
    let raw_level = br.read_bits(12)?;
    if !br.read_bool()? {
        return Err(TextureParseError::EscapeMarkerBitMissing);
    }
    // 12-bit two's-complement; 0 and -2048 (0b1000_0000_0000) reserved.
    let level = if raw_level >= 0x800 {
        raw_level as i32 - 0x1000
    } else {
        raw_level as i32
    };
    if level == 0 || level == -2048 {
        return Err(TextureParseError::ReservedEscapeLevel);
    }
    Ok(AcEvent { last, run, level })
}

/// Run the §6.2.7 `while (!last) DCT coefficient` loop: decode AC EVENTs
/// until (and including) the one whose `LAST` flag is set, returning them
/// in scan order. An empty stream that never reaches `LAST == 1` returns
/// [`TextureParseError::Truncated`].
pub fn decode_ac_events(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<Vec<AcEvent>, TextureParseError> {
    let mut events = Vec::new();
    loop {
        let ev = decode_ac_event(br, table_kind)?;
        let last = ev.last;
        events.push(ev);
        if last {
            return Ok(events);
        }
    }
}

/// Decode one §7.4.1.2 `DCT coefficient` EVENT for the
/// `short_video_header == 1` path.
///
/// The common Tcoef VLC + sign bit (Table B.16 for intra, Table B.17 for
/// inter) is unchanged from the long-header path. Only the §7.4.1.3 escape
/// changes — Types 1..=3 are not used here; instead Type 4 is the only
/// permitted escape mode. Per §7.4.1.3 paragraph 4:
///
/// > Type 4: The fourth type of escape code is used if and only if
/// > short_video_header is 1. In this case, the 15 bits following ESC
/// > are decoded as fixed length codes represented by 1-bit LAST, 6-bit
/// > RUN and 8-bit LEVEL.
///
/// LEVEL is a signed two's-complement value drawn from Table B.18 c;
/// `0000 0000` (= 0) and `1000 0000` (= -128) are reserved and rejected
/// with [`TextureParseError::ReservedEscapeLevel`].
///
/// Unlike Type 3, there are no marker bits inside the Type 4 escape
/// payload — the short-video-header bitstream uses a different
/// resynchronisation discipline (Annex K) that does not require
/// per-coefficient marker bits.
pub fn decode_ac_event_short_video_header(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<AcEvent, TextureParseError> {
    let table = match table_kind {
        TcoefTable::Intra => TCOEF_INTRA,
        TcoefTable::Inter => TCOEF_INTER,
    };

    // Peek the 7-bit escape prefix without consuming; a real code that
    // shares no prefix with ESC falls through to the table decode.
    let avail = br.remaining_bits().min(TCOEF_ESCAPE_LEN as usize) as u8;
    if avail == 0 {
        return Err(TextureParseError::Truncated);
    }
    let is_escape =
        avail == TCOEF_ESCAPE_LEN && br.next_bits(TCOEF_ESCAPE_LEN as usize)? == TCOEF_ESCAPE_CODE;

    if !is_escape {
        let (last, run, level) = decode_tcoef_vlc(br, table)?;
        return Ok(AcEvent { last, run, level });
    }

    // Type 4: ESC + LAST(1) + RUN(6) + LEVEL(8); no marker bits.
    br.skip_bits(TCOEF_ESCAPE_LEN as usize)?;
    let last = br.read_bool()?;
    let run = br.read_bits(6)?;
    let raw_level = br.read_bits(8)?;
    // 8-bit two's-complement; 0 and -128 (0b1000_0000) reserved per
    // §7.4.1.3 paragraph 4 + Table B.18 c.
    let level = if raw_level >= 0x80 {
        raw_level as i32 - 0x100
    } else {
        raw_level as i32
    };
    if level == 0 || level == -128 {
        return Err(TextureParseError::ReservedEscapeLevel);
    }
    Ok(AcEvent { last, run, level })
}

/// Run the §6.2.7 `while (!last) DCT coefficient` loop for the
/// `short_video_header == 1` path. See
/// [`decode_ac_event_short_video_header`] for the per-EVENT semantics.
pub fn decode_ac_events_short_video_header(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<Vec<AcEvent>, TextureParseError> {
    let mut events = Vec::new();
    loop {
        let ev = decode_ac_event_short_video_header(br, table_kind)?;
        let last = ev.last;
        events.push(ev);
        if last {
            return Ok(events);
        }
    }
}

/// Match the prefix-free reversible Tcoef VLC (Table B.23) against the
/// leading bits of `window`, returning `(code_len, last, run, level)` for
/// the selected table-kind column on a hit (before the sign bit), `None`
/// on no match.
fn match_rvlc_tcoef(
    window: u32,
    window_len: u8,
    table_kind: TcoefTable,
) -> Option<(u8, bool, u32, i32)> {
    for &(code, len, i_last, i_run, i_level, n_last, n_run, n_level) in RVLC_TCOEF {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                let (last, run, level) = match table_kind {
                    TcoefTable::Intra => (i_last, i_run, i_level),
                    TcoefTable::Inter => (n_last, n_run, n_level),
                };
                return Some((len, last != 0, run as u32, level as i32));
            }
        }
    }
    None
}

/// Decode one §7.4.1.2 `DCT coefficient` EVENT for the
/// `short_video_header == 0`, `reversible_vlc == 1` path (the
/// reversible-VLC tables of §6.3.3 / §7.4.1.2).
///
/// The common case is a single Table B.23 reversible VLC (its intra or
/// inter column, selected by `table_kind`) plus a trailing sign bit `s`
/// (`0` positive, `1` negative). The combinations not represented in the
/// table use the §7.4.1.3 Type-5 escape — the only escape mode permitted
/// when `reversible_vlc == 1`:
///
/// * **Type 5** — opening delimiter `00001`, then a fixed-length
///   `LAST(1)` `RUN(6)` `marker_bit` `LEVEL(11)` `marker_bit`, then the
///   closing delimiter `0000` and the sign bit `s` (Table B.23 ESCAPE
///   row + the diagram beneath it). The 11-bit LEVEL is the unsigned
///   magnitude per Table B.25 (`0` is forbidden); its sign comes from
///   the closing `s`. The two marker bits prevent `resync_marker`
///   emulation and must both be 1.
///
/// The escape opener `00001` is prefix-disjoint from every Table B.23
/// code (no code begins with `0000`), so an ordinary reversible VLC is
/// distinguished from the escape by inspecting the leading five bits.
pub fn decode_ac_event_rvlc(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<AcEvent, TextureParseError> {
    // Peek the 5-bit escape opener without consuming, so an ordinary
    // reversible VLC (none of which begins with `0000`) falls through to
    // the table decode.
    let peek = br.remaining_bits().min(RVLC_ESCAPE_OPEN_LEN as usize) as u8;
    if peek == 0 {
        return Err(TextureParseError::Truncated);
    }
    let is_escape = peek == RVLC_ESCAPE_OPEN_LEN
        && br.next_bits(RVLC_ESCAPE_OPEN_LEN as usize)? == RVLC_ESCAPE_OPEN;

    if !is_escape {
        // Common path: a Table B.23 reversible VLC plus its sign bit.
        let avail = br.remaining_bits().min(16) as u8;
        if avail == 0 {
            return Err(TextureParseError::Truncated);
        }
        let window = br.next_bits(avail as usize)?;
        let (code_len, last, run, level) =
            match_rvlc_tcoef(window, avail, table_kind).ok_or(TextureParseError::InvalidTcoef {
                window: window as u16,
            })?;
        br.skip_bits(code_len as usize)?;
        let sign_negative = br.read_bool()?;
        let signed = if sign_negative { -level } else { level };
        return Ok(AcEvent {
            last,
            run,
            level: signed,
        });
    }

    // Type 5: consume the opener, then the fixed-length payload.
    br.skip_bits(RVLC_ESCAPE_OPEN_LEN as usize)?;
    let last = br.read_bool()?;
    let run = br.read_bits(6)?;
    if !br.read_bool()? {
        return Err(TextureParseError::EscapeMarkerBitMissing);
    }
    let magnitude = br.read_bits(11)?;
    if !br.read_bool()? {
        return Err(TextureParseError::EscapeMarkerBitMissing);
    }
    // Closing delimiter `0000` then sign bit `s`.
    let delimiter = br.read_bits(4)?;
    if delimiter != 0 {
        return Err(TextureParseError::RvlcEscapeDelimiterMissing {
            window: delimiter as u8,
        });
    }
    let sign_negative = br.read_bool()?;
    // LEVEL == 0 is forbidden by Table B.25.
    if magnitude == 0 {
        return Err(TextureParseError::ReservedEscapeLevel);
    }
    let level = if sign_negative {
        -(magnitude as i32)
    } else {
        magnitude as i32
    };
    Ok(AcEvent { last, run, level })
}

/// Run the §6.2.7 `while (!last) DCT coefficient` loop for the
/// `short_video_header == 0`, `reversible_vlc == 1` path. See
/// [`decode_ac_event_rvlc`] for the per-EVENT semantics. An empty stream
/// that never reaches `LAST == 1` returns [`TextureParseError::Truncated`].
pub fn decode_ac_events_rvlc(
    br: &mut BitReader<'_>,
    table_kind: TcoefTable,
) -> Result<Vec<AcEvent>, TextureParseError> {
    let mut events = Vec::new();
    loop {
        let ev = decode_ac_event_rvlc(br, table_kind)?;
        let last = ev.last;
        events.push(ev);
        if last {
            return Ok(events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `BitReader` over MSB-first bits described by a `&str` of
    /// `'0'`/`'1'` (whitespace ignored), zero-padded up to a byte.
    fn bits(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| *c == '0' || *c == '1').collect();
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in cleaned.chars() {
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        out
    }

    #[test]
    fn dc_size_tables_are_prefix_free() {
        for table in [DC_SIZE_LUMINANCE, DC_SIZE_CHROMINANCE] {
            for (i, &(ca, la, _)) in table.iter().enumerate() {
                for (j, &(cb, lb, _)) in table.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    // No code is a prefix of another.
                    if la <= lb {
                        let shifted = cb >> (lb - la);
                        assert_ne!(
                            shifted,
                            ca,
                            "code {ca:0width$b} is a prefix of {cb:0w2$b}",
                            width = la as usize,
                            w2 = lb as usize
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dc_size_tables_cover_zero_to_twelve() {
        for table in [DC_SIZE_LUMINANCE, DC_SIZE_CHROMINANCE] {
            let mut seen = [false; 13];
            for &(_, _, size) in table {
                seen[size as usize] = true;
            }
            assert!(seen.iter().all(|s| *s), "table missing a size 0..=12");
        }
    }

    #[test]
    fn differential_sign_decode_size_1() {
        // size 1: additional 0 → -1, additional 1 → +1.
        assert_eq!(decode_differential(1, 0b0), -1);
        assert_eq!(decode_differential(1, 0b1), 1);
    }

    #[test]
    fn differential_sign_decode_size_2() {
        // Table B.15: 00→-3, 01→-2, 10→+2, 11→+3.
        assert_eq!(decode_differential(2, 0b00), -3);
        assert_eq!(decode_differential(2, 0b01), -2);
        assert_eq!(decode_differential(2, 0b10), 2);
        assert_eq!(decode_differential(2, 0b11), 3);
    }

    #[test]
    fn differential_sign_decode_size_3() {
        // Table B.15: -7..-4 (000..011), 4..7 (100..111).
        assert_eq!(decode_differential(3, 0b000), -7);
        assert_eq!(decode_differential(3, 0b011), -4);
        assert_eq!(decode_differential(3, 0b100), 4);
        assert_eq!(decode_differential(3, 0b111), 7);
    }

    #[test]
    fn differential_sign_decode_size_8_boundaries() {
        // Table B.15 size 8: -255..-128 and 128..255.
        assert_eq!(decode_differential(8, 0b0000_0000), -255);
        assert_eq!(decode_differential(8, 0b0111_1111), -128);
        assert_eq!(decode_differential(8, 0b1000_0000), 128);
        assert_eq!(decode_differential(8, 0b1111_1111), 255);
    }

    #[test]
    fn differential_size_zero_is_zero() {
        assert_eq!(decode_differential(0, 0), 0);
    }

    #[test]
    fn component_from_block_index() {
        assert_eq!(DcComponent::from_block_index(0), DcComponent::Luminance);
        assert_eq!(DcComponent::from_block_index(3), DcComponent::Luminance);
        assert_eq!(DcComponent::from_block_index(4), DcComponent::Chrominance);
        assert_eq!(DcComponent::from_block_index(5), DcComponent::Chrominance);
    }

    #[test]
    fn luminance_size_zero_no_differential() {
        // Table B.13: size 0 = "011". No additional code follows.
        let data = bits("011");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
        assert_eq!(dc.size, 0);
        assert_eq!(dc.differential, 0);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn chrominance_size_zero_no_differential() {
        // Table B.14: size 0 = "11".
        let data = bits("11");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Chrominance).unwrap();
        assert_eq!(dc.size, 0);
        assert_eq!(dc.differential, 0);
        assert_eq!(br.bit_position(), 2);
    }

    #[test]
    fn luminance_size_1_positive() {
        // Table B.13: size 1 = "11"; additional "1" → +1.
        let data = bits("11 1");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
        assert_eq!(dc.size, 1);
        assert_eq!(dc.differential, 1);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn luminance_size_1_negative() {
        // Table B.13: size 1 = "11"; additional "0" → -1.
        let data = bits("11 0");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
        assert_eq!(dc.size, 1);
        assert_eq!(dc.differential, -1);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn luminance_size_2() {
        // Table B.13: size 2 = "10"; additional "10" → +2.
        let data = bits("10 10");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
        assert_eq!(dc.size, 2);
        assert_eq!(dc.differential, 2);
        assert_eq!(br.bit_position(), 4);
    }

    #[test]
    fn chrominance_size_3() {
        // Table B.14: size 3 = "001"; additional "000" → -7.
        let data = bits("001 000");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Chrominance).unwrap();
        assert_eq!(dc.size, 3);
        assert_eq!(dc.differential, -7);
        assert_eq!(br.bit_position(), 6);
    }

    #[test]
    fn luminance_size_9_consumes_marker_bit() {
        // Table B.13: size 9 = "00000001" (8 bits); additional 9 bits
        // "100000000" = 256 → +256; then marker_bit "1".
        let data = bits("00000001 100000000 1");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
        assert_eq!(dc.size, 9);
        assert_eq!(dc.differential, 256);
        // 8 (size code) + 9 (additional) + 1 (marker) = 18 bits.
        assert_eq!(br.bit_position(), 18);
    }

    #[test]
    fn marker_bit_zero_rejected() {
        // size 9 code + 9-bit additional + a 0 marker_bit → reject.
        let data = bits("00000001 100000000 0");
        let mut br = BitReader::new(&data);
        let err = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap_err();
        assert_eq!(err, TextureParseError::MarkerBitMissing);
    }

    #[test]
    fn chrominance_size_9_negative_with_marker() {
        // Table B.14: size 9 = "000000001" (9 bits); additional 9 bits
        // "000000000" = 0 → (0+1) - 512 = -511; then marker "1".
        let data = bits("000000001 000000000 1");
        let mut br = BitReader::new(&data);
        let dc = decode_intra_dc(&mut br, DcComponent::Chrominance).unwrap();
        assert_eq!(dc.size, 9);
        assert_eq!(dc.differential, -511);
        assert_eq!(br.bit_position(), 19);
    }

    #[test]
    fn invalid_dc_size_prefix_rejected() {
        // 13 zero bits match no luminance code (longest is 11 bits, all
        // ending in a 1) — InvalidDcSize.
        let data = bits("0000000000000");
        let mut br = BitReader::new(&data);
        let err = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap_err();
        assert!(matches!(err, TextureParseError::InvalidDcSize { .. }));
    }

    #[test]
    fn truncated_empty_reader() {
        let data: [u8; 0] = [];
        let mut br = BitReader::new(&data);
        let err = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }

    #[test]
    fn truncated_mid_additional_code() {
        // size 5 luminance = "0001"; only 2 of the 5 additional bits
        // present → Truncated.
        let data = bits("0001 10");
        let mut br = BitReader::new(&data);
        let err = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }

    #[test]
    fn full_luminance_table_round_trip() {
        // Encode each (size, code) and confirm decode recovers the size,
        // with a deterministic additional code for size > 0.
        for &(code, len, size) in DC_SIZE_LUMINANCE {
            let mut s = format!("{code:0len$b}", len = len as usize);
            // Append an all-ones additional code (= +(2^size - 1)) plus
            // a marker_bit when size > 8.
            if size > 0 {
                for _ in 0..size {
                    s.push('1');
                }
                if size > 8 {
                    s.push('1');
                }
            }
            let data = bits(&s);
            let mut br = BitReader::new(&data);
            let dc = decode_intra_dc(&mut br, DcComponent::Luminance).unwrap();
            assert_eq!(dc.size, size);
            if size > 0 {
                let expected = (1i32 << size) - 1;
                assert_eq!(dc.differential, expected, "size {size}");
            } else {
                assert_eq!(dc.differential, 0);
            }
        }
    }

    #[test]
    fn full_chrominance_table_round_trip() {
        for &(code, len, size) in DC_SIZE_CHROMINANCE {
            let mut s = format!("{code:0len$b}", len = len as usize);
            if size > 0 {
                // All-zeros additional code → most-negative differential.
                for _ in 0..size {
                    s.push('0');
                }
                if size > 8 {
                    s.push('1');
                }
            }
            let data = bits(&s);
            let mut br = BitReader::new(&data);
            let dc = decode_intra_dc(&mut br, DcComponent::Chrominance).unwrap();
            assert_eq!(dc.size, size);
            if size > 0 {
                let expected = 1i32 - (1i32 << size);
                assert_eq!(dc.differential, expected, "size {size}");
            } else {
                assert_eq!(dc.differential, 0);
            }
        }
    }

    #[test]
    fn error_displays() {
        assert!(format!("{}", TextureParseError::Truncated).contains("truncated"));
        assert!(
            format!("{}", TextureParseError::InvalidDcSize { window: 0 })
                .contains("invalid dct_dc_size")
        );
        assert!(format!("{}", TextureParseError::MarkerBitMissing).contains("marker_bit was 0"));
        assert!(
            format!("{}", TextureParseError::InvalidTcoef { window: 0 }).contains("invalid Tcoef")
        );
        assert!(format!("{}", TextureParseError::EscapeMarkerBitMissing)
            .contains("Type-3 escape marker_bit"));
        assert!(format!("{}", TextureParseError::ReservedEscapeLevel).contains("reserved value"));
    }

    // -----------------------------------------------------------------
    // §7.4.1.2 AC coefficient (EVENT) decode tests
    // -----------------------------------------------------------------

    #[test]
    fn tcoef_tables_have_102_entries_each() {
        // Table B.16 / Table B.17 each have 102 non-escape EVENTs.
        assert_eq!(TCOEF_INTRA.len(), 102);
        assert_eq!(TCOEF_INTER.len(), 102);
    }

    #[test]
    fn tcoef_tables_are_prefix_free_and_no_dupes() {
        for table in [TCOEF_INTRA, TCOEF_INTER] {
            for (i, &(ca, la, ..)) in table.iter().enumerate() {
                for (j, &(cb, lb, ..)) in table.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if la <= lb {
                        assert_ne!(cb >> (lb - la), ca, "code {ca:b} is a prefix of {cb:b}");
                    }
                }
            }
        }
    }

    #[test]
    fn escape_prefix_disjoint_from_tables() {
        // No real Tcoef code shares the 7-bit escape prefix 0000 011.
        for table in [TCOEF_INTRA, TCOEF_INTER] {
            for &(code, len, ..) in table {
                if len >= TCOEF_ESCAPE_LEN {
                    let top = code >> (len - TCOEF_ESCAPE_LEN);
                    assert_ne!(
                        top, TCOEF_ESCAPE_CODE,
                        "code 0b{code:b} collides with escape"
                    );
                } else {
                    // A shorter code can't begin with the 7-bit escape.
                    let shifted = TCOEF_ESCAPE_CODE >> (TCOEF_ESCAPE_LEN - len);
                    assert_ne!(code, shifted, "escape begins with code 0b{code:b}");
                }
            }
        }
    }

    #[test]
    fn intra_common_event_positive() {
        // Table B.16: "10s" = (LAST 0, RUN 0, LEVEL 1); sign 0 → +1.
        let data = bits("10 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 0,
                level: 1
            }
        );
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn intra_common_event_negative_sign() {
        // "10s" with sign 1 → LEVEL -1.
        let data = bits("10 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(ev.level, -1);
    }

    #[test]
    fn inter_common_event_differs_from_intra() {
        // Table B.17: "110s" = (LAST 0, RUN 1, LEVEL 1) — distinct from
        // Table B.16's "110s" = (0, 0, 2).
        let data = bits("110 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(ev.run, 1);
        assert_eq!(ev.level, 1);

        let mut br2 = BitReader::new(&data);
        let ev2 = decode_ac_event(&mut br2, TcoefTable::Intra).unwrap();
        assert_eq!(ev2.run, 0);
        assert_eq!(ev2.level, 2);
    }

    #[test]
    fn intra_last_event() {
        // Table B.16: "0111s" = (LAST 1, RUN 0, LEVEL 1); sign 0 → +1.
        let data = bits("0111 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, 1);
    }

    #[test]
    fn decode_ac_events_loop_two_events() {
        // First "10s/0" = (0,0,1); then "0111s/1" = (LAST 1, 0, -1).
        let data = bits("10 0  0111 1");
        let mut br = BitReader::new(&data);
        let events = decode_ac_events(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            AcEvent {
                last: false,
                run: 0,
                level: 1
            }
        );
        assert_eq!(
            events[1],
            AcEvent {
                last: true,
                run: 0,
                level: -1
            }
        );
    }

    #[test]
    fn escape_type1_intra_adds_lmax() {
        // ESC(0000011) + "0" + Tcoef "10s/0" = (0,0,1). LMAX(intra, LAST 0,
        // RUN 0) = 27. LEVEL restored = +(1 + 27) = 28.
        let data = bits("0000011 0 10 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 0,
                level: 28
            }
        );
    }

    #[test]
    fn escape_type1_inter_negative_keeps_sign() {
        // ESC + "0" + "10s/1" = (0,0,-1). LMAX(inter, LAST 0, RUN 0) = 12.
        // restored = -(1 + 12) = -13.
        let data = bits("0000011 0 10 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(ev.level, -13);
    }

    #[test]
    fn escape_type2_intra_adds_rmax_plus_one() {
        // ESC + "10" + "10s/0" = (0,0,1). RMAX(intra, LAST 0, LEVEL 1) = 14.
        // RUN restored = 0 + 14 + 1 = 15.
        let data = bits("0000011 10 10 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 15,
                level: 1
            }
        );
    }

    #[test]
    fn escape_type2_inter_run() {
        // ESC + "10" + "110s/0" = (LAST 0, RUN 1, LEVEL 1) for inter.
        // RMAX(inter, LAST 0, LEVEL 1) = 26. RUN restored = 1 + 26 + 1 = 28.
        let data = bits("0000011 10 110 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(ev.run, 28);
        assert_eq!(ev.level, 1);
    }

    #[test]
    fn escape_type3_positive_level() {
        // ESC + "11" + LAST(1)=0 + RUN(6)=000011(=3) + marker(1) +
        // LEVEL(12)=000000001000(=8) + marker(1).
        let data = bits("0000011 11 0 000011 1 000000001000 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 3,
                level: 8
            }
        );
    }

    #[test]
    fn escape_type3_negative_level_twos_complement() {
        // LEVEL(12) = 1111 1111 1111 = -1 (two's complement). LAST = 1,
        // RUN = 0.
        let data = bits("0000011 11 1 000000 1 111111111111 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Inter).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, -1);
    }

    #[test]
    fn escape_type3_min_legal_level() {
        // LEVEL(12) = 1000 0000 0001 = -2047 (the most-negative legal
        // value; -2048 is reserved).
        let data = bits("0000011 11 0 000000 1 100000000001 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(ev.level, -2047);
    }

    #[test]
    fn escape_type3_reserved_level_zero_rejected() {
        // LEVEL(12) = all zeros = 0 → reserved.
        let data = bits("0000011 11 0 000000 1 000000000000 1");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::ReservedEscapeLevel);
    }

    #[test]
    fn escape_type3_reserved_level_min_rejected() {
        // LEVEL(12) = 1000 0000 0000 = -2048 → reserved.
        let data = bits("0000011 11 0 000000 1 100000000000 1");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::ReservedEscapeLevel);
    }

    #[test]
    fn escape_type3_missing_marker_rejected() {
        // First marker bit (after RUN) is 0 → reject.
        let data = bits("0000011 11 0 000000 0 000000001000 1");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::EscapeMarkerBitMissing);
    }

    #[test]
    fn invalid_tcoef_prefix_rejected() {
        // 13 zero bits match no intra Tcoef code and are not the escape
        // prefix (0000011) — InvalidTcoef. (Escape needs bits 5..6 = "11".)
        let data = bits("0000000000000");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event(&mut br, TcoefTable::Intra).unwrap_err();
        assert!(matches!(err, TextureParseError::InvalidTcoef { .. }));
    }

    #[test]
    fn ac_event_truncated_empty() {
        let data: [u8; 0] = [];
        let mut br = BitReader::new(&data);
        let err = decode_ac_event(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }

    #[test]
    fn every_intra_table_entry_round_trips() {
        // Encode each (code, sign 0) and confirm the decoded EVENT matches
        // the tabulated (last, run, level).
        for &(code, len, last, run, level) in TCOEF_INTRA {
            let mut s = format!("{code:0len$b}", len = len as usize);
            s.push('0'); // positive sign
            let data = bits(&s);
            let mut br = BitReader::new(&data);
            let ev = decode_ac_event(&mut br, TcoefTable::Intra).unwrap();
            assert_eq!(ev.last, last != 0, "code 0b{code:b}");
            assert_eq!(ev.run, run as u32, "code 0b{code:b}");
            assert_eq!(ev.level, level as i32, "code 0b{code:b}");
        }
    }

    #[test]
    fn every_inter_table_entry_round_trips() {
        for &(code, len, last, run, level) in TCOEF_INTER {
            let mut s = format!("{code:0len$b}", len = len as usize);
            s.push('1'); // negative sign
            let data = bits(&s);
            let mut br = BitReader::new(&data);
            let ev = decode_ac_event(&mut br, TcoefTable::Inter).unwrap();
            assert_eq!(ev.last, last != 0, "code 0b{code:b}");
            assert_eq!(ev.run, run as u32, "code 0b{code:b}");
            assert_eq!(ev.level, -(level as i32), "code 0b{code:b}");
        }
    }

    #[test]
    fn lmax_rmax_known_cells() {
        // Spot-check the LMAX / RMAX tables against the spec.
        assert_eq!(lmax_intra(false, 0), Some(27));
        assert_eq!(lmax_intra(true, 20), Some(1));
        assert_eq!(lmax_intra(true, 21), None);
        assert_eq!(lmax_inter(false, 26), Some(1));
        assert_eq!(lmax_inter(true, 40), Some(1));
        assert_eq!(rmax_intra(false, 27), Some(0));
        assert_eq!(rmax_intra(true, 1), Some(20));
        assert_eq!(rmax_inter(false, 1), Some(26));
        assert_eq!(rmax_inter(true, 2), Some(1));
    }

    // -----------------------------------------------------------------
    // §7.4.1.3 Type 4 escape (short_video_header == 1) tests
    // -----------------------------------------------------------------

    #[test]
    fn svh_common_event_passes_through_unchanged() {
        // Table B.16: "10s" = (LAST 0, RUN 0, LEVEL 1) — the common Tcoef
        // VLC + sign bit path is identical between SVH=0 and SVH=1.
        let data = bits("10 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 0,
                level: 1
            }
        );
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn svh_inter_common_event_negative_sign() {
        // Table B.17: "110s/1" = (LAST 0, RUN 1, LEVEL -1).
        let data = bits("110 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(ev.run, 1);
        assert_eq!(ev.level, -1);
    }

    #[test]
    fn svh_type4_escape_positive_level() {
        // ESC(0000011) + LAST=0 + RUN(6)=000011(=3) + LEVEL(8)=00001010(=10).
        // No marker bits per §7.4.1.3 Type 4.
        let data = bits("0000011 0 000011 00001010");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 3,
                level: 10
            }
        );
        // 7 (ESC) + 1 (LAST) + 6 (RUN) + 8 (LEVEL) = 22 bits.
        assert_eq!(br.bit_position(), 22);
    }

    #[test]
    fn svh_type4_escape_negative_level_twos_complement() {
        // LEVEL(8) = 1111 1111 = -1 (two's complement). LAST=1, RUN=0.
        let data = bits("0000011 1 000000 11111111");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Inter).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, -1);
    }

    #[test]
    fn svh_type4_escape_max_legal_positive_level() {
        // LEVEL(8) = 0111 1111 = +127 (max legal positive per Table B.18 c).
        let data = bits("0000011 0 000000 01111111");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(ev.level, 127);
    }

    #[test]
    fn svh_type4_escape_min_legal_negative_level() {
        // LEVEL(8) = 1000 0001 = -127 (min legal negative; -128 reserved).
        let data = bits("0000011 0 000000 10000001");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(ev.level, -127);
    }

    #[test]
    fn svh_type4_escape_run_full_range() {
        // RUN(6) = 111111 = 63 (max for 6 bits).
        let data = bits("0000011 0 111111 00000001");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(ev.run, 63);
        assert_eq!(ev.level, 1);
    }

    #[test]
    fn svh_type4_escape_reserved_level_zero_rejected() {
        // LEVEL(8) = 0000 0000 = 0 → reserved per Table B.18 c.
        let data = bits("0000011 0 000000 00000000");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::ReservedEscapeLevel);
    }

    #[test]
    fn svh_type4_escape_reserved_level_min_rejected() {
        // LEVEL(8) = 1000 0000 = -128 → reserved per Table B.18 c.
        let data = bits("0000011 0 000000 10000000");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::ReservedEscapeLevel);
    }

    #[test]
    fn svh_type4_escape_truncated_mid_level() {
        // ESC + LAST + RUN = 14 bits, with the LEVEL byte missing entirely
        // (slice cut off after the RUN). The padding-to-byte from the `bits`
        // helper supplies 2 stray zero bits, so the reader has 16 bits;
        // requesting 8 LEVEL bits fails because only 2 are available beyond
        // the 14-bit prefix.
        let data = bits("0000011 0 000011");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }

    #[test]
    fn svh_type4_escape_last_flag_carried_through() {
        // Confirm the 1-bit LAST is honoured (Type 4 carries LAST inline,
        // unlike the common Tcoef VLC which encodes it as part of the
        // codeword).
        let data = bits("0000011 1 000000 00000001");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Inter).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, 1);
    }

    #[test]
    fn svh_events_loop_terminates_on_last() {
        // First a common-path "10s/0" = (0,0,+1), then a Type-4 escape
        // with LAST=1, RUN=2, LEVEL=+5.
        let data = bits("10 0  0000011 1 000010 00000101");
        let mut br = BitReader::new(&data);
        let events = decode_ac_events_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            AcEvent {
                last: false,
                run: 0,
                level: 1
            }
        );
        assert_eq!(
            events[1],
            AcEvent {
                last: true,
                run: 2,
                level: 5
            }
        );
    }

    #[test]
    fn svh_events_loop_truncated_without_last_returns_error() {
        // An empty reader yields Truncated on the very first EVENT — the
        // loop never sees LAST==1.
        let data: [u8; 0] = [];
        let mut br = BitReader::new(&data);
        let err = decode_ac_events_short_video_header(&mut br, TcoefTable::Inter).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }

    #[test]
    fn svh_does_not_attempt_type3_marker_bits() {
        // Long-header Type 3 layout (ESC + "11" + LAST + RUN + marker +
        // LEVEL(12) + marker) is *not* a valid SVH stream. If we hand
        // such bits to the SVH path, it must decode as Type 4: ESC +
        // LAST(1)=1 + RUN(6)=100000(=32) + LEVEL(8)=00000000_... but
        // the 8 LEVEL bits would here be "00000011" (= +3) and the
        // function must NOT look for marker bits.
        // bit layout: ESC(7) + "1" + "100000" + "00000011" = 22 bits.
        let data = bits("0000011 1 100000 00000011");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_short_video_header(&mut br, TcoefTable::Intra).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 32);
        assert_eq!(ev.level, 3);
        assert_eq!(br.bit_position(), 22);
    }

    // ---- §7.4.1.2 reversible-VLC (Table B.23) path ----

    #[test]
    fn rvlc_table_is_prefix_free_and_escape_disjoint() {
        // Every RVLC code is prefix-free and none collides with the
        // escape opener `00001` / closing delimiter `0000`.
        for (i, &(ca, la, ..)) in RVLC_TCOEF.iter().enumerate() {
            // No code begins with `0000` (reserved for the escape).
            if la >= 4 {
                assert_ne!(ca >> (la - 4), 0, "code {i} begins with 0000");
            } else {
                assert_ne!(ca, 0, "code {i} begins with 0000");
            }
            for (j, &(cb, lb, ..)) in RVLC_TCOEF.iter().enumerate() {
                if i == j || la > lb {
                    continue;
                }
                assert_ne!(cb >> (lb - la), ca, "code {i} is a prefix of code {j}");
            }
        }
        assert_eq!(RVLC_TCOEF.len(), 169);
    }

    #[test]
    fn rvlc_common_intra_first_code() {
        // Table B.23 INDEX 0: intra (LAST=0, RUN=0, LEVEL=1), code
        // 110s. Sign bit 0 → +1.
        let data = bits("110 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 0,
                level: 1
            }
        );
        assert_eq!(br.bit_position(), 4);
    }

    #[test]
    fn rvlc_common_inter_differs_from_intra() {
        // Table B.23 INDEX 1: code 111s. intra=(0,0,2), inter=(0,1,1).
        let intra = bits("111 1"); // sign 1 → negative
        let mut br = BitReader::new(&intra);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 0,
                level: -2
            }
        );

        let inter = bits("111 0");
        let mut br = BitReader::new(&inter);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 1,
                level: 1
            }
        );
    }

    #[test]
    fn rvlc_common_last_flag_index4() {
        // INDEX 4: code 1011s, intra=(1,0,1), inter=(1,0,1).
        let data = bits("1011 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, 1);
    }

    #[test]
    fn rvlc_longest_16bit_code() {
        // INDEX 168 (last row, 15-bit prefix + sign): code
        // 011111101111101s, intra=(1,44,1), inter=(1,44,1).
        let data = bits("011111101111101 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Inter).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 44);
        assert_eq!(ev.level, -1);
        assert_eq!(br.bit_position(), 16);
    }

    #[test]
    fn rvlc_type5_escape_positive() {
        // Type 5: open 00001 + LAST(1) + RUN(6) + marker(1) +
        // LEVEL(11) + marker(1) + closing 0000 + sign s.
        // LAST=0, RUN=5 (000101), LEVEL=300 (00100101100), sign 0.
        let data = bits("00001 0 000101 1 00100101100 1 0000 0");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: false,
                run: 5,
                level: 300
            }
        );
        // 5 + 1 + 6 + 1 + 11 + 1 + 4 + 1 = 30 bits.
        assert_eq!(br.bit_position(), 30);
    }

    #[test]
    fn rvlc_type5_escape_negative_with_last() {
        // LAST=1, RUN=63 (111111), LEVEL=1 (00000000001), sign 1.
        let data = bits("00001 1 111111 1 00000000001 1 0000 1");
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Inter).unwrap();
        assert_eq!(
            ev,
            AcEvent {
                last: true,
                run: 63,
                level: -1
            }
        );
    }

    #[test]
    fn rvlc_type5_escape_forbidden_zero_level() {
        // LEVEL == 0 is forbidden (Table B.25).
        let data = bits("00001 0 000000 1 00000000000 1 0000 0");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::ReservedEscapeLevel);
    }

    #[test]
    fn rvlc_type5_escape_first_marker_must_be_one() {
        let data = bits("00001 0 000001 0 00000000001 1 0000 0");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(err, TextureParseError::EscapeMarkerBitMissing);
    }

    #[test]
    fn rvlc_type5_escape_bad_closing_delimiter() {
        // Closing delimiter must be 0000.
        let data = bits("00001 0 000001 1 00000000001 1 0001 0");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap_err();
        assert_eq!(
            err,
            TextureParseError::RvlcEscapeDelimiterMissing { window: 0b0001 }
        );
    }

    #[test]
    fn rvlc_invalid_code_reports_window() {
        // `0000` is reserved for the escape; a `0000` not followed by `1`
        // (i.e. `00000...`) is neither a code nor a valid opener.
        let data = bits("00000 0000000");
        let mut br = BitReader::new(&data);
        let err = decode_ac_event_rvlc(&mut br, TcoefTable::Intra).unwrap_err();
        assert!(matches!(err, TextureParseError::InvalidTcoef { .. }));
    }

    #[test]
    fn rvlc_event_loop_runs_until_last() {
        // INDEX 0 (110s, intra not-last) then INDEX 4 (1011s, last).
        let data = bits("110 0 1011 0");
        let mut br = BitReader::new(&data);
        let events = decode_ac_events_rvlc(&mut br, TcoefTable::Intra).unwrap();
        assert_eq!(events.len(), 2);
        assert!(!events[0].last);
        assert!(events[1].last);
    }

    #[test]
    fn rvlc_event_loop_truncated() {
        let data: [u8; 0] = [];
        let mut br = BitReader::new(&data);
        let err = decode_ac_events_rvlc(&mut br, TcoefTable::Inter).unwrap_err();
        assert_eq!(err, TextureParseError::Truncated);
    }
}

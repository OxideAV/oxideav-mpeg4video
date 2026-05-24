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
//! ## Out of scope (this round)
//!
//! * AC coefficient EVENTs (Tables B.16..B.25), the escape modes of
//!   §7.4.1.3, and the `last`/`run`/`level` zigzag placement.
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
    }
}

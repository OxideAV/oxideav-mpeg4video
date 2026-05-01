//! Reversible Variable-Length Codes (RVLC) for MPEG-4 Part 2 DCT
//! coefficients — ISO/IEC 14496-2 Tables B.23–B.25.
//!
//! When `reversible_vlc = 1` is advertised in the VOL header (only legal
//! together with `data_partitioned = 1` per §6.2.5), every DCT-coefficient
//! `(LAST, RUN, LEVEL)` triplet in I-, P- and S(GMC)-VOPs is encoded
//! through the RVLC table instead of the standard B.16 / B.17 tcoef
//! tables. Each codeword is constructed so that decoding it bit-by-bit
//! from either end of the codeword stream yields the same triplet, which
//! lets the decoder recover from a corruption anywhere inside the AC
//! partition by walking back from the end-of-block marker
//! (Annex E.1.4.4).
//!
//! Layout of one (LAST, RUN, |LEVEL|) short codeword:
//!
//! ```text
//!   <prefix bits per Table B.23>  <sign s>
//! ```
//!
//! Layout of an RVLC ESCAPE — used for any (LAST, RUN, |LEVEL|) combo
//! NOT listed in Table B.23 (and forbidden for combos that ARE listed,
//! per §7.4.1.3 Type 5):
//!
//! ```text
//!   00001  LAST(1)  RUN(6)  marker(1)  LEVEL(11)  marker(1)  0000  s
//!   ^opening                                                  ^closing prefix + sign
//! ```
//!
//! Total escape size = 5 + 1 + 6 + 1 + 11 + 1 + 4 + 1 = 30 bits. The
//! opening (`00001`) and closing (`0000`) prefixes share the four-bit
//! pattern `0000`, which is reserved — no short RVLC codeword starts
//! with `0000`, so a forward parser that encounters `0000` knows it is
//! looking at an escape boundary. The closing-prefix sign bit is the
//! sign of LEVEL (`0` = positive, `1` = negative); the FLC LEVEL field
//! is unsigned (1..=2047).
//!
//! Per §7.4.1.2 the **same** Table B.23 prefix codes are used for both
//! intra and inter blocks; only the (LAST, RUN, LEVEL) triplet that the
//! prefix decodes to differs between the two contexts. Tables B.24 and
//! B.25 are FLC tables for the escape RUN (6-bit binary, 0..=63) and
//! LEVEL (11-bit binary, 1..=2047) — see [`encode_escape`].
//!
//! The encoder forbids combos listed in Table B.23 from going through
//! the escape path (the spec calls this out explicitly: "Use of escape
//! sequence of the reversible VLC for encoding the combinations listed
//! in Table B.23 is prohibited"). Anything else is allowed via escape.

use std::sync::OnceLock;

use oxideav_core::{Error, Result};

use crate::headers::vol::ZIGZAG;
use oxideav_core::bits::{BitReader, BitWriter};

// -------------------------------------------------------------------------
// Table B.23 — prefix codewords
// -------------------------------------------------------------------------

/// One short-form RVLC entry. `prefix_bits` excludes the trailing sign
/// bit; the encoder appends the sign separately. Both `(intra_*)` and
/// `(inter_*)` triplets are valid decodes of the same prefix in their
/// respective contexts.
#[derive(Clone, Copy, Debug)]
struct RvlcEntry {
    /// Number of prefix bits (= total table-bits − 1 sign).
    prefix_bits: u8,
    /// Prefix value, MSB-first in the low `prefix_bits` bits.
    prefix_code: u32,
    /// `(LAST, RUN, |LEVEL|)` for intra blocks.
    intra: (bool, u8, u16),
    /// `(LAST, RUN, |LEVEL|)` for inter blocks.
    inter: (bool, u8, u16),
}

/// All 169 short RVLC codewords from Table B.23, indices 0..=168. The
/// row at index 169 (escape) is NOT in this table — it's encoded via
/// [`encode_escape`] / [`decode_escape_body`].
///
/// Each entry is `(intra_last, intra_run, intra_level, inter_last,
/// inter_run, inter_level, total_bits, vlc_code_with_sign_low_bit)`.
/// The encoder stores `prefix_bits = total_bits − 1` and `prefix_code =
/// vlc_code_with_sign_low_bit >> 1` after construction.
#[rustfmt::skip]
#[allow(clippy::type_complexity)]
const RVLC_TABLE_RAW: &[(bool, u8, u16, bool, u8, u16, u8, u32)] = &[
    // idx LAST RUN LEVEL  LAST RUN LEVEL  BITS  VLC (incl. sign as LSB)
    (false,  0,  1, false,  0,  1,  4, 0b1100),         // 0   110s
    (false,  0,  2, false,  1,  1,  4, 0b1110),         // 1   111s
    (false,  1,  1, false,  0,  2,  5, 0b00010),        // 2   0001s
    (false,  0,  3, false,  2,  1,  5, 0b10100),        // 3   1010s
    (true,   0,  1, true,   0,  1,  5, 0b10110),        // 4   1011s
    (false,  2,  1, false,  0,  3,  6, 0b001000),       // 5   00100s
    (false,  3,  1, false,  3,  1,  6, 0b001010),       // 6   00101s
    (false,  1,  2, false,  4,  1,  6, 0b010000),       // 7   01000s
    (false,  0,  4, false,  5,  1,  6, 0b010010),       // 8   01001s
    (true,   1,  1, true,   1,  1,  6, 0b100100),       // 9   10010s
    (true,   2,  1, true,   2,  1,  6, 0b100110),       // 10  10011s
    (false,  4,  1, false,  1,  2,  7, 0b0011000),      // 11  001100s
    (false,  5,  1, false,  6,  1,  7, 0b0011010),      // 12  001101s
    (false,  0,  5, false,  7,  1,  7, 0b0101000),      // 13  010100s
    (false,  0,  6, false,  8,  1,  7, 0b0101010),      // 14  010101s
    (true,   3,  1, true,   3,  1,  7, 0b0110000),      // 15  011000s
    (true,   4,  1, true,   4,  1,  7, 0b0110010),      // 16  011001s
    (true,   5,  1, true,   5,  1,  7, 0b1000100),      // 17  100010s
    (true,   6,  1, true,   6,  1,  7, 0b1000110),      // 18  100011s
    (false,  6,  1, false,  0,  4,  8, 0b00111000),     // 19  0011100s
    (false,  7,  1, false,  2,  2,  8, 0b00111010),     // 20  0011101s
    (false,  2,  2, false,  9,  1,  8, 0b01011000),     // 21  0101100s
    (false,  1,  3, false, 10,  1,  8, 0b01011010),     // 22  0101101s
    (false,  0,  7, false, 11,  1,  8, 0b01101000),     // 23  0110100s
    (true,   7,  1, true,   7,  1,  8, 0b01101010),     // 24  0110101s
    (true,   8,  1, true,   8,  1,  8, 0b01110000),     // 25  0111000s
    (true,   9,  1, true,   9,  1,  8, 0b01110010),     // 26  0111001s
    (true,  10,  1, true,  10,  1,  8, 0b10000100),     // 27  1000010s
    (true,  11,  1, true,  11,  1,  8, 0b10000110),     // 28  1000011s
    (false,  8,  1, false,  0,  5,  9, 0b001111000),    // 29  00111100s
    (false,  9,  1, false,  0,  6,  9, 0b001111010),    // 30  00111101s
    (false,  3,  2, false,  1,  3,  9, 0b010111000),    // 31  01011100s
    (false,  4,  2, false,  3,  2,  9, 0b010111010),    // 32  01011101s
    (false,  1,  4, false,  4,  2,  9, 0b011011000),    // 33  01101100s
    (false,  1,  5, false, 12,  1,  9, 0b011011010),    // 34  01101101s
    (false,  0,  8, false, 13,  1,  9, 0b011101000),    // 35  01110100s
    (false,  0,  9, false, 14,  1,  9, 0b011101010),    // 36  01110101s
    (true,   0,  2, true,   0,  2,  9, 0b011110000),    // 37  01111000s
    (true,  12,  1, true,  12,  1,  9, 0b011110010),    // 38  01111001s
    (true,  13,  1, true,  13,  1,  9, 0b100000100),    // 39  10000010s
    (true,  14,  1, true,  14,  1,  9, 0b100000110),    // 40  10000011s
    (false, 10,  1, false,  0,  7, 10, 0b0011111000),   // 41  001111100s
    (false,  5,  2, false,  1,  4, 10, 0b0011111010),   // 42  001111101s
    (false,  2,  3, false,  2,  3, 10, 0b0101111000),   // 43  010111100s
    (false,  3,  3, false,  5,  2, 10, 0b0101111010),   // 44  010111101s
    (false,  1,  6, false, 15,  1, 10, 0b0110111000),   // 45  011011100s
    (false,  0, 10, false, 16,  1, 10, 0b0110111010),   // 46  011011101s
    (false,  0, 11, false, 17,  1, 10, 0b0111011000),   // 47  011101100s
    (true,   1,  2, true,   1,  2, 10, 0b0111011010),   // 48  011101101s
    (true,  15,  1, true,  15,  1, 10, 0b0111101000),   // 49  011110100s
    (true,  16,  1, true,  16,  1, 10, 0b0111101010),   // 50  011110101s
    (true,  17,  1, true,  17,  1, 10, 0b0111110000),   // 51  011111000s
    (true,  18,  1, true,  18,  1, 10, 0b0111110010),   // 52  011111001s
    (true,  19,  1, true,  19,  1, 10, 0b1000000100),   // 53  100000010s
    (true,  20,  1, true,  20,  1, 10, 0b1000000110),   // 54  100000011s
    (false, 11,  1, false,  0,  8, 11, 0b00111111000),  // 55  0011111100s
    (false, 12,  1, false,  0,  9, 11, 0b00111111010),  // 56  0011111101s
    (false,  6,  2, false,  1,  5, 11, 0b01011111000),  // 57  0101111100s
    (false,  7,  2, false,  3,  3, 11, 0b01011111010),  // 58  0101111101s
    (false,  8,  2, false,  6,  2, 11, 0b01101111000),  // 59  0110111100s
    (false,  4,  3, false,  7,  2, 11, 0b01101111010),  // 60  0110111101s
    (false,  2,  4, false,  8,  2, 11, 0b01110111000),  // 61  0111011100s
    (false,  1,  7, false,  9,  2, 11, 0b01110111010),  // 62  0111011101s
    (false,  0, 12, false, 18,  1, 11, 0b01111011000),  // 63  0111101100s
    (false,  0, 13, false, 19,  1, 11, 0b01111011010),  // 64  0111101101s
    (false,  0, 14, false, 20,  1, 11, 0b01111101000),  // 65  0111110100s
    (true,  21,  1, true,  21,  1, 11, 0b01111101010),  // 66  0111110101s
    (true,  22,  1, true,  22,  1, 11, 0b01111110000),  // 67  0111111000s
    (true,  23,  1, true,  23,  1, 11, 0b01111110010),  // 68  0111111001s
    (true,  24,  1, true,  24,  1, 11, 0b10000000100),  // 69  1000000010s
    (true,  25,  1, true,  25,  1, 11, 0b10000000110),  // 70  1000000011s
    (false, 13,  1, false,  0, 10, 12, 0b001111111000), // 71  00111111100s
    (false,  9,  2, false,  0, 11, 12, 0b001111111010), // 72  00111111101s
    (false,  5,  3, false,  1,  6, 12, 0b010111111000), // 73  01011111100s
    (false,  6,  3, false,  2,  4, 12, 0b010111111010), // 74  01011111101s
    (false,  7,  3, false,  4,  3, 12, 0b011011111000), // 75  01101111100s
    (false,  3,  4, false,  5,  3, 12, 0b011011111010), // 76  01101111101s
    (false,  2,  5, false, 10,  2, 12, 0b011101111000), // 77  01110111100s
    (false,  2,  6, false, 21,  1, 12, 0b011101111010), // 78  01110111101s
    (false,  1,  8, false, 22,  1, 12, 0b011110111000), // 79  01111011100s
    (false,  1,  9, false, 23,  1, 12, 0b011110111010), // 80  01111011101s
    (false,  0, 15, false, 24,  1, 12, 0b011111011000), // 81  01111101100s
    (false,  0, 16, false, 25,  1, 12, 0b011111011010), // 82  01111101101s
    (false,  0, 17, false, 26,  1, 12, 0b011111101000), // 83  01111110100s
    (true,   0,  3, true,   0,  3, 12, 0b011111101010), // 84  01111110101s
    (true,   2,  2, true,   2,  2, 12, 0b011111110000), // 85  01111111000s
    (true,  26,  1, true,  26,  1, 12, 0b011111110010), // 86  01111111001s
    (true,  27,  1, true,  27,  1, 12, 0b100000000100), // 87  10000000010s
    (true,  28,  1, true,  28,  1, 12, 0b100000000110), // 88  10000000011s
    (false, 10,  2, false,  0, 12, 13, 0b0011111111000),// 89  001111111100s
    (false,  4,  4, false,  1,  7, 13, 0b0011111111010),// 90  001111111101s
    (false,  5,  4, false,  2,  5, 13, 0b0101111111000),// 91  010111111100s
    (false,  6,  4, false,  3,  4, 13, 0b0101111111010),// 92  010111111101s
    (false,  3,  5, false,  6,  3, 13, 0b0110111111000),// 93  011011111100s
    (false,  4,  5, false,  7,  3, 13, 0b0110111111010),// 94  011011111101s
    (false,  1, 10, false, 11,  2, 13, 0b0111011111000),// 95  011101111100s
    (false,  0, 18, false, 27,  1, 13, 0b0111011111010),// 96  011101111101s
    (false,  0, 19, false, 28,  1, 13, 0b0111101111000),// 97  011110111100s
    (false,  0, 22, false, 29,  1, 13, 0b0111101111010),// 98  011110111101s
    (true,   1,  3, true,   1,  3, 13, 0b0111110111000),// 99  011111011100s
    (true,   3,  2, true,   3,  2, 13, 0b0111110111010),// 100 011111011101s
    (true,   4,  2, true,   4,  2, 13, 0b0111111011000),// 101 011111101100s
    (true,  29,  1, true,  29,  1, 13, 0b0111111011010),// 102 011111101101s
    (true,  30,  1, true,  30,  1, 13, 0b0111111101000),// 103 011111110100s
    (true,  31,  1, true,  31,  1, 13, 0b0111111101010),// 104 011111110101s
    (true,  32,  1, true,  32,  1, 13, 0b0111111110000),// 105 011111111000s
    (true,  33,  1, true,  33,  1, 13, 0b0111111110010),// 106 011111111001s
    (true,  34,  1, true,  34,  1, 13, 0b1000000000100),// 107 100000000010s
    (true,  35,  1, true,  35,  1, 13, 0b1000000000110),// 108 100000000011s
    (false, 14,  1, false,  0, 13, 14, 0b00111111111000),  // 109 0011111111100s
    (false, 15,  1, false,  0, 14, 14, 0b00111111111010),  // 110 0011111111101s
    (false, 11,  2, false,  0, 15, 14, 0b01011111111000),  // 111 0101111111100s
    (false,  8,  3, false,  0, 16, 14, 0b01011111111010),  // 112 0101111111101s
    (false,  9,  3, false,  1,  8, 14, 0b01101111111000),  // 113 0110111111100s
    (false,  7,  4, false,  3,  5, 14, 0b01101111111010),  // 114 0110111111101s
    (false,  3,  6, false,  4,  4, 14, 0b01110111111000),  // 115 0111011111100s
    (false,  2,  7, false,  5,  4, 14, 0b01110111111010),  // 116 0111011111101s
    (false,  2,  8, false,  8,  3, 14, 0b01111011111000),  // 117 0111101111100s
    (false,  2,  9, false, 12,  2, 14, 0b01111011111010),  // 118 0111101111101s
    (false,  1, 11, false, 30,  1, 14, 0b01111101111000),  // 119 0111110111100s
    (false,  0, 20, false, 31,  1, 14, 0b01111101111010),  // 120 0111110111101s
    (false,  0, 21, false, 32,  1, 14, 0b01111110111000),  // 121 0111111011100s
    (false,  0, 23, false, 33,  1, 14, 0b01111110111010),  // 122 0111111011101s
    (true,   0,  4, true,   0,  4, 14, 0b01111111011000),  // 123 0111111101100s
    (true,   5,  2, true,   5,  2, 14, 0b01111111011010),  // 124 0111111101101s
    (true,   6,  2, true,   6,  2, 14, 0b01111111101000),  // 125 0111111110100s
    (true,   7,  2, true,   7,  2, 14, 0b01111111101010),  // 126 0111111110101s
    (true,   8,  2, true,   8,  2, 14, 0b01111111110000),  // 127 0111111111000s
    (true,   9,  2, true,   9,  2, 14, 0b01111111110010),  // 128 0111111111001s
    (true,  36,  1, true,  36,  1, 14, 0b10000000000100),  // 129 1000000000010s
    (true,  37,  1, true,  37,  1, 14, 0b10000000000110),  // 130 1000000000011s
    (false, 16,  1, false,  0, 17, 15, 0b001111111111000), // 131 00111111111100s
    (false, 17,  1, false,  0, 18, 15, 0b001111111111010), // 132 00111111111101s
    (false, 18,  1, false,  1,  9, 15, 0b010111111111000), // 133 01011111111100s
    (false,  8,  4, false,  1, 10, 15, 0b010111111111010), // 134 01011111111101s
    (false,  5,  5, false,  2,  6, 15, 0b011011111111000), // 135 01101111111100s
    (false,  4,  6, false,  2,  7, 15, 0b011011111111010), // 136 01101111111101s
    (false,  5,  6, false,  3,  6, 15, 0b011101111111000), // 137 01110111111100s
    (false,  3,  7, false,  6,  4, 15, 0b011101111111010), // 138 01110111111101s
    (false,  3,  8, false,  9,  3, 15, 0b011110111111000), // 139 01111011111100s
    (false,  2, 10, false, 13,  2, 15, 0b011110111111010), // 140 01111011111101s
    (false,  2, 11, false, 14,  2, 15, 0b011111011111000), // 141 01111101111100s
    (false,  1, 12, false, 15,  2, 15, 0b011111011111010), // 142 01111101111101s
    (false,  1, 13, false, 16,  2, 15, 0b011111101111000), // 143 01111110111100s
    (false,  0, 24, false, 34,  1, 15, 0b011111101111010), // 144 01111110111101s
    (false,  0, 25, false, 35,  1, 15, 0b011111110111000), // 145 01111111011100s
    (false,  0, 26, false, 36,  1, 15, 0b011111110111010), // 146 01111111011101s
    (true,   0,  5, true,   0,  5, 15, 0b011111111011000), // 147 01111111101100s
    (true,   1,  4, true,   1,  4, 15, 0b011111111011010), // 148 01111111101101s
    (true,  10,  2, true,  10,  2, 15, 0b011111111101000), // 149 01111111110100s
    (true,  11,  2, true,  11,  2, 15, 0b011111111101010), // 150 01111111110101s
    (true,  12,  2, true,  12,  2, 15, 0b011111111110000), // 151 01111111111000s
    (true,  38,  1, true,  38,  1, 15, 0b011111111110010), // 152 01111111111001s
    (true,  39,  1, true,  39,  1, 15, 0b100000000000100), // 153 10000000000010s
    (true,  40,  1, true,  40,  1, 15, 0b100000000000110), // 154 10000000000011s
    (false,  0, 27, false,  0, 19, 16, 0b0011111111111000),// 155 001111111111100s
    (false,  3,  9, false,  3,  7, 16, 0b0011111111111010),// 156 001111111111101s
    (false,  6,  5, false,  4,  5, 16, 0b0101111111111000),// 157 010111111111100s
    (false,  7,  5, false,  7,  4, 16, 0b0101111111111010),// 158 010111111111101s
    (false,  9,  4, false, 17,  2, 16, 0b0110111111111000),// 159 011011111111100s
    (false, 12,  2, false, 37,  1, 16, 0b0110111111111010),// 160 011011111111101s
    (false, 19,  1, false, 38,  1, 16, 0b0111011111111000),// 161 011101111111100s
    (true,   1,  5, true,   1,  5, 16, 0b0111011111111010),// 162 011101111111101s
    (true,   2,  3, true,   2,  3, 16, 0b0111101111111000),// 163 011110111111100s
    (true,  13,  2, true,  13,  2, 16, 0b0111101111111010),// 164 011110111111101s
    (true,  41,  1, true,  41,  1, 16, 0b0111110111111000),// 165 011111011111100s
    (true,  42,  1, true,  42,  1, 16, 0b0111110111111010),// 166 011111011111101s
    (true,  43,  1, true,  43,  1, 16, 0b0111111011111000),// 167 011111101111100s
    (true,  44,  1, true,  44,  1, 16, 0b0111111011111010),// 168 011111101111101s
];

/// Built-once view of [`RVLC_TABLE_RAW`] with the trailing sign bit
/// stripped from each codeword. Indexed 0..=168.
fn table() -> &'static [RvlcEntry] {
    static CELL: OnceLock<Vec<RvlcEntry>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut v = Vec::with_capacity(RVLC_TABLE_RAW.len());
        for &(il, ir, ilv, el, er, elv, bits, code_with_sign) in RVLC_TABLE_RAW {
            v.push(RvlcEntry {
                prefix_bits: bits - 1,
                prefix_code: code_with_sign >> 1,
                intra: (il, ir, ilv),
                inter: (el, er, elv),
            });
        }
        v
    })
    .as_slice()
}

// -------------------------------------------------------------------------
// Escape-sequence constants
// -------------------------------------------------------------------------

/// Opening escape — `00001`, MSB-first 5 bits.
const ESCAPE_OPEN: u32 = 0b00001;
const ESCAPE_OPEN_BITS: u32 = 5;

/// Closing escape prefix — `0000`, MSB-first 4 bits. Followed by the
/// LEVEL sign bit `s` (`0` = positive, `1` = negative).
const ESCAPE_CLOSE_PREFIX: u32 = 0b0000;
const ESCAPE_CLOSE_PREFIX_BITS: u32 = 4;

/// Total bits of an RVLC escape codeword (open + LAST + RUN + marker +
/// LEVEL + marker + close prefix + sign).
pub const ESCAPE_TOTAL_BITS: u32 =
    ESCAPE_OPEN_BITS + 1 + 6 + 1 + 11 + 1 + ESCAPE_CLOSE_PREFIX_BITS + 1;

// -------------------------------------------------------------------------
// Encode side
// -------------------------------------------------------------------------

/// Look up `(last, run, |level|)` in the intra column. Returns
/// `Some((prefix_bits, prefix_code))` when the triplet has a short
/// codeword, `None` when the encoder must fall back to an escape.
fn lookup_intra(last: bool, run: u8, level_abs: u16) -> Option<(u8, u32)> {
    for e in table() {
        if e.intra == (last, run, level_abs) {
            return Some((e.prefix_bits, e.prefix_code));
        }
    }
    None
}

/// Same as [`lookup_intra`] but indexes the inter column.
fn lookup_inter(last: bool, run: u8, level_abs: u16) -> Option<(u8, u32)> {
    for e in table() {
        if e.inter == (last, run, level_abs) {
            return Some((e.prefix_bits, e.prefix_code));
        }
    }
    None
}

/// Emit one RVLC escape sequence per Table B.23 / B.24 / B.25:
/// `00001 LAST(1) RUN(6) marker LEVEL(11) marker 0000 sign`.
///
/// `level` must be non-zero and within the 11-bit unsigned LEVEL range
/// `[1, 2047]`. `run` is the 6-bit unsigned RUN.
fn encode_escape(bw: &mut BitWriter, last: bool, run: u8, level: i32) {
    debug_assert!(level != 0, "RVLC escape: LEVEL must be non-zero");
    let abs = level.unsigned_abs();
    debug_assert!((1..=2047).contains(&abs), "RVLC escape: LEVEL out of range");
    bw.write_bits(ESCAPE_OPEN, ESCAPE_OPEN_BITS);
    bw.write_bits(if last { 1 } else { 0 }, 1);
    bw.write_bits(run as u32 & 0x3F, 6);
    bw.write_bits(1, 1); // marker
    bw.write_bits(abs, 11);
    bw.write_bits(1, 1); // marker
    bw.write_bits(ESCAPE_CLOSE_PREFIX, ESCAPE_CLOSE_PREFIX_BITS);
    bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
}

/// Emit one (last, run, level) symbol through the RVLC writer — short
/// codeword + sign when listed in Table B.23, otherwise an escape.
/// Selects the intra column (`is_intra = true`) or inter column
/// (`is_intra = false`).
fn emit_symbol(bw: &mut BitWriter, is_intra: bool, last: bool, run: u8, level: i32) {
    let abs = level.unsigned_abs() as u16;
    let lookup = if is_intra {
        lookup_intra(last, run, abs)
    } else {
        lookup_inter(last, run, abs)
    };
    if let Some((bits, code)) = lookup {
        bw.write_bits(code, bits as u32);
        bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
        return;
    }
    encode_escape(bw, last, run, level);
}

/// Walk an intra block (zigzag, scan starts at index 1 — DC is sent
/// separately) and emit one RVLC symbol per non-zero coefficient. Mirrors
/// `encoder::write_intra_ac` modulo the codeword table.
pub fn write_intra_ac(bw: &mut BitWriter, block: &[i32; 64]) -> Result<()> {
    let mut last_nz: Option<usize> = None;
    for i in 1..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        return Err(Error::other(
            "mpeg4 rvlc: AC walk requested but block is all zero",
        ));
    };
    let mut run = 0u8;
    let mut i = 1;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        emit_symbol(bw, true, last, run, lv);
        run = 0;
        i += 1;
    }
    Ok(())
}

/// Walk an inter block (zigzag, scan starts at index 0 — inter blocks
/// have no separately-coded DC) and emit one RVLC symbol per non-zero
/// coefficient. Mirrors `pvop::write_inter_ac` modulo the codeword
/// table.
pub fn write_inter_ac(bw: &mut BitWriter, block: &[i32; 64]) {
    let mut last_nz: Option<usize> = None;
    for i in 0..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        return;
    };
    let mut run = 0u8;
    let mut i = 0;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        emit_symbol(bw, false, last, run, lv);
        run = 0;
        i += 1;
    }
}

// -------------------------------------------------------------------------
// Decode side (forward only — error-recovery reverse decode is a future
// follow-up; for our purposes the decoder needs to consume what we emit).
// -------------------------------------------------------------------------

/// One decoded RVLC symbol = `(last, run, signed_level)`.
type DecodedSym = (bool, u8, i32);

/// Decode one RVLC symbol forward from `br`. Selects the intra or inter
/// column. Handles both short codewords and escapes.
fn decode_symbol(br: &mut BitReader<'_>, is_intra: bool) -> Result<DecodedSym> {
    // Peek up to 16 bits (the longest short codeword in B.23).
    let max_bits = 16u32;
    let remaining = br.bits_remaining() as u32;
    let peek_bits = max_bits.min(remaining);
    if peek_bits == 0 {
        return Err(Error::invalid("mpeg4 rvlc: no bits available"));
    }
    let peeked = br.peek_u32(peek_bits)?;
    // Left-justify into max_bits so we can extract a fixed-width prefix
    // without re-shifting per entry.
    let peeked_full = peeked << (max_bits - peek_bits);

    // Detect escape opening — `00001` (5 bits).
    if peek_bits >= ESCAPE_OPEN_BITS {
        let top5 = peeked_full >> (max_bits - ESCAPE_OPEN_BITS);
        if top5 == ESCAPE_OPEN {
            return decode_escape_body(br);
        }
    }

    // Linear scan of short codewords.
    for e in table() {
        let bits = e.prefix_bits as u32;
        if bits > peek_bits {
            continue;
        }
        let prefix = peeked_full >> (max_bits - bits);
        if prefix == e.prefix_code {
            br.consume(bits)?;
            let sign = br.read_u1()? as i32;
            let (last, run, abs) = if is_intra { e.intra } else { e.inter };
            let level = if sign == 1 { -(abs as i32) } else { abs as i32 };
            return Ok((last, run, level));
        }
    }
    Err(Error::invalid("mpeg4 rvlc: no matching codeword"))
}

/// Decode the body of an RVLC escape sequence — assumes the opening
/// `00001` has NOT been consumed. Reads the entire 30-bit escape and
/// returns the recovered triplet.
fn decode_escape_body(br: &mut BitReader<'_>) -> Result<DecodedSym> {
    // Consume the opening escape.
    br.consume(ESCAPE_OPEN_BITS)?;
    let last = br.read_u1()? == 1;
    let run = br.read_u32(6)? as u8;
    let m1 = br.read_u1()?;
    if m1 != 1 {
        return Err(Error::invalid("mpeg4 rvlc: missing marker after RUN"));
    }
    let abs = br.read_u32(11)? as i32;
    if abs == 0 {
        return Err(Error::invalid("mpeg4 rvlc: escape LEVEL=0 forbidden"));
    }
    let m2 = br.read_u1()?;
    if m2 != 1 {
        return Err(Error::invalid("mpeg4 rvlc: missing marker after LEVEL"));
    }
    let close = br.read_u32(ESCAPE_CLOSE_PREFIX_BITS)?;
    if close != ESCAPE_CLOSE_PREFIX {
        return Err(Error::invalid("mpeg4 rvlc: bad closing escape prefix"));
    }
    let sign = br.read_u1()? as i32;
    let level = if sign == 1 { -abs } else { abs };
    Ok((last, run, level))
}

/// Decode an intra block's AC coefficients into `block` (placed at
/// `block[scan[i]]` for scan index `i`). Returns the index of the last
/// non-zero scan position. Mirrors `block::decode_intra_ac`.
pub fn decode_intra_ac(
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    scan: &[usize; 64],
) -> Result<Option<usize>> {
    let mut i: usize = 1;
    loop {
        if i > 63 {
            return Err(Error::invalid("mpeg4 rvlc intra: AC overrun"));
        }
        let (last, run, level) = decode_symbol(br, true)?;
        i = i.saturating_add(run as usize);
        if i > 63 {
            return Err(Error::invalid("mpeg4 rvlc intra: AC run overflow"));
        }
        block[scan[i]] = level;
        if last {
            return Ok(Some(i));
        }
        i += 1;
        if i > 63 {
            return Ok(Some(i - 1));
        }
    }
}

/// Decode an inter block's AC coefficients. Mirrors
/// `block::decode_inter_ac` modulo the codeword table.
pub fn decode_inter_ac(
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    scan: &[usize; 64],
) -> Result<Option<usize>> {
    let mut i: usize = 0;
    loop {
        if i > 63 {
            return Err(Error::invalid("mpeg4 rvlc inter: AC overrun"));
        }
        let (last, run, level) = decode_symbol(br, false)?;
        i = i.saturating_add(run as usize);
        if i > 63 {
            return Err(Error::invalid("mpeg4 rvlc inter: AC run overflow"));
        }
        block[scan[i]] = level;
        if last {
            return Ok(Some(i));
        }
        i += 1;
        if i > 63 {
            return Ok(Some(i - 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the table has exactly 169 short entries.
    #[test]
    fn table_has_169_entries() {
        assert_eq!(table().len(), 169);
    }

    /// Sanity: every entry's prefix is `prefix_bits` wide (no high bits set).
    #[test]
    fn prefix_width_matches() {
        for (i, e) in table().iter().enumerate() {
            let max = if e.prefix_bits >= 32 {
                u32::MAX
            } else {
                (1u32 << e.prefix_bits) - 1
            };
            assert!(
                e.prefix_code <= max,
                "entry {i} prefix {:#x} doesn't fit in {} bits",
                e.prefix_code,
                e.prefix_bits
            );
        }
    }

    /// No codeword is a prefix of another — required for any VLC. The
    /// escape opening (`00001`) and closing (`0000`) share the leading
    /// `0000` pattern; we verify that no short codeword starts with
    /// `0000` so the forward parser can cleanly distinguish escape vs.
    /// short on that 4-bit window.
    #[test]
    fn no_short_codeword_starts_with_0000() {
        for (i, e) in table().iter().enumerate() {
            if e.prefix_bits >= 4 {
                let top4 = e.prefix_code >> (e.prefix_bits as u32 - 4);
                assert_ne!(
                    top4,
                    0,
                    "entry {i} prefix {:#0width$b} starts with 0000",
                    e.prefix_code,
                    width = e.prefix_bits as usize + 2
                );
            }
        }
    }

    /// Every codeword must be prefix-unique against every other codeword
    /// in its own column (intra-vs-intra and inter-vs-inter). The
    /// columns are decoded in different contexts — there's no
    /// cross-column constraint. We test by verifying that no two table
    /// rows have the same `(prefix_bits, prefix_code)` pair (the spec
    /// gives each row a unique prefix).
    #[test]
    fn prefix_codes_unique() {
        let t = table();
        for i in 0..t.len() {
            for j in (i + 1)..t.len() {
                assert_ne!(
                    (t[i].prefix_bits, t[i].prefix_code),
                    (t[j].prefix_bits, t[j].prefix_code),
                    "entries {i} and {j} share a prefix"
                );
            }
        }
    }

    /// Round-trip a single short intra symbol — the simplest one,
    /// `(LAST=0, RUN=0, LEVEL=+1)` (entry 0 in the table).
    #[test]
    fn intra_short_roundtrip_smoke() {
        let mut bw = BitWriter::new();
        emit_symbol(&mut bw, true, false, 0, 1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let (last, run, level) = decode_symbol(&mut br, true).expect("decode");
        assert_eq!((last, run, level), (false, 0, 1));
    }

    /// Round-trip a single short inter symbol — entry 0 again, but the
    /// inter column maps the same prefix to `(LAST=0, RUN=0, LEVEL=+1)`
    /// too (table B.23 happens to agree at index 0).
    #[test]
    fn inter_short_roundtrip_smoke() {
        let mut bw = BitWriter::new();
        emit_symbol(&mut bw, false, false, 0, 1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let (last, run, level) = decode_symbol(&mut br, false).expect("decode");
        assert_eq!((last, run, level), (false, 0, 1));
    }

    /// Round-trip a negative short symbol — the sign bit must round-trip.
    #[test]
    fn negative_sign_roundtrip() {
        let mut bw = BitWriter::new();
        emit_symbol(&mut bw, true, false, 0, -1);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let (last, run, level) = decode_symbol(&mut br, true).expect("decode");
        assert_eq!((last, run, level), (false, 0, -1));
    }

    /// Round-trip an escape — pick `(LAST=1, RUN=63, LEVEL=+2047)`,
    /// which is well outside the short table.
    #[test]
    fn escape_roundtrip() {
        let mut bw = BitWriter::new();
        emit_symbol(&mut bw, true, true, 63, 2047);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let (last, run, level) = decode_symbol(&mut br, true).expect("decode");
        assert_eq!((last, run, level), (true, 63, 2047));
    }

    /// Escape bit width matches the spec constant.
    #[test]
    fn escape_total_bits_is_30() {
        assert_eq!(ESCAPE_TOTAL_BITS, 30);
        let mut bw = BitWriter::new();
        encode_escape(&mut bw, false, 5, 100);
        // BitWriter counts bits — finish() pads to byte boundary, so
        // measure via the bit_position helper if available; otherwise
        // confirm via byte count:
        let bytes = bw.finish();
        // 30 bits → ceil(30/8) = 4 bytes (with padding).
        assert!(bytes.len() == 4 || bytes.len() == 5);
    }

    /// Round-trip a full intra AC walk through `write_intra_ac` /
    /// `decode_intra_ac` with a hand-crafted block.
    #[test]
    fn full_intra_ac_roundtrip() {
        // Place a few coefficients at scan positions 1, 3, 7 (last).
        let mut block = [0i32; 64];
        block[ZIGZAG[1]] = 5;
        block[ZIGZAG[3]] = -2;
        block[ZIGZAG[7]] = 1;
        let mut bw = BitWriter::new();
        write_intra_ac(&mut bw, &block).expect("encode");
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut decoded = [0i32; 64];
        let last = decode_intra_ac(&mut br, &mut decoded, &ZIGZAG).expect("decode");
        assert_eq!(last, Some(7));
        assert_eq!(decoded[ZIGZAG[1]], 5);
        assert_eq!(decoded[ZIGZAG[3]], -2);
        assert_eq!(decoded[ZIGZAG[7]], 1);
    }

    /// Round-trip a full inter AC walk including a coefficient at scan
    /// index 0 (which is illegal for intra but legal for inter).
    #[test]
    fn full_inter_ac_roundtrip() {
        let mut block = [0i32; 64];
        block[ZIGZAG[0]] = 3;
        block[ZIGZAG[2]] = -1;
        block[ZIGZAG[10]] = 1;
        let mut bw = BitWriter::new();
        write_inter_ac(&mut bw, &block);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut decoded = [0i32; 64];
        let last = decode_inter_ac(&mut br, &mut decoded, &ZIGZAG).expect("decode");
        assert_eq!(last, Some(10));
        assert_eq!(decoded[ZIGZAG[0]], 3);
        assert_eq!(decoded[ZIGZAG[2]], -1);
        assert_eq!(decoded[ZIGZAG[10]], 1);
    }

    /// Mixed short + escape walk — at least one coefficient triggers
    /// the escape path because its (last, run, level) isn't in B.23.
    #[test]
    fn mixed_short_and_escape_roundtrip() {
        let mut block = [0i32; 64];
        block[ZIGZAG[1]] = 1; // short
        block[ZIGZAG[63]] = 100; // escape (last=1, run=61, level=100)
        let mut bw = BitWriter::new();
        write_intra_ac(&mut bw, &block).expect("encode");
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let mut decoded = [0i32; 64];
        let last = decode_intra_ac(&mut br, &mut decoded, &ZIGZAG).expect("decode");
        assert_eq!(last, Some(63));
        assert_eq!(decoded[ZIGZAG[1]], 1);
        assert_eq!(decoded[ZIGZAG[63]], 100);
    }
}

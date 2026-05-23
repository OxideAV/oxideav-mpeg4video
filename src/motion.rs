//! §6.2.6.2 `motion_vector(mode)` bitstream decode plus the §7.6.3
//! general motion-vector decoding process (differential-MV
//! reconstruction + predictor add + modulo wrap).
//!
//! Round 6 stopped at the end of the B-VOP macroblock header prefix —
//! `dbquant` — leaving the bit reader positioned at the start of the
//! `motion_vector("…")` body. This module decodes that body for the
//! `"forward"` / `"backward"` / `"direct"` modes and reconstructs the
//! final motion-vector component from a caller-supplied predictor.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.6.2 `motion_vector(mode)` syntax — `horizontal_mv_data` /
//!   `horizontal_mv_residual` / `vertical_mv_data` /
//!   `vertical_mv_residual`, with the residuals gated on
//!   `(vop_fcode != 1) && (mv_data != 0)`.
//! * §6.3.6.2 — `horizontal_mv_data` / `vertical_mv_data` are VLC codes
//!   from Table B.12; `*_mv_residual` are `r_size`-bit unsigned
//!   integers with `r_size = vop_fcode - 1`.
//! * §7.6.3 General motion vector decoding process — the
//!   `r_size`/`f`/`high`/`low`/`range` recurrence, the
//!   `MVD = (Abs(mv_data) - 1) * f + residual + 1` reconstruction, the
//!   predictor add `MVx = Px + MVDx`, and the modulo wrap into
//!   `[low:high]`. Per the §7.6.3 note the `mv_data` value is **two
//!   times** the "vector differences" column of Table B.12.
//! * Table B.12 — VLC table for MVD (65 rows, "vector differences"
//!   −16 … +16 in half-pel steps; the on-wire `mv_data` is the doubled
//!   integer −32 … +32). The +32 (`16`) row "shall not be used when
//!   vop_fcode == 1".
//! * Table 7-9 — `[low:high]` range per `vop_fcode` (a cross-check on
//!   the `f`-derived bounds).
//!
//! ## Out of scope (this round)
//!
//! * The motion-vector **predictor** itself (median of three
//!   neighbouring block vectors, §7.6.5 / §7.6.6) — the predictor is a
//!   caller-supplied input here. This module reconstructs the
//!   component value *given* a predictor; computing the predictor from
//!   the neighbourhood is a later round.
//! * The "direct" mode's predictor scaling from the co-located P-VOP MV
//!   (§7.6.6) — `"direct"` here decodes only its `mv_data` pair (no
//!   residuals, exactly as the §6.2.6.2 syntax shows) and reconstructs
//!   the *delta* via §7.6.3 with `Px = Py = 0`; the scaled-predictor
//!   combination is a later round.
//! * The four-MV (`inter4v`) per-block decode loop — the caller invokes
//!   [`decode_motion_vector`] once per block.
//! * Interlaced field motion vectors (`field_prediction`) and the
//!   half-precision vertical adjustment of §7.6.3's closing paragraph.

use crate::bitreader::{BitReader, BitReaderError};

/// Prediction mode passed to [`decode_motion_vector`], matching the
/// `mode` argument of the §6.2.6.2 `motion_vector(mode)` syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvMode {
    /// `"direct"` — both components are a bare `mv_data` VLC with no
    /// residual (the §6.2.6.2 `if (mode == "direct")` branch).
    Direct,
    /// `"forward"` — each component is `mv_data` plus an optional
    /// `r_size`-bit residual, gated on `vop_fcode_forward`.
    Forward,
    /// `"backward"` — same shape as forward but gated on
    /// `vop_fcode_backward`.
    Backward,
}

/// Errors produced by the motion-vector decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionParseError {
    /// The supplied bit reader ran out mid-field.
    Truncated,
    /// The leading bits did not match any code in Table B.12. The
    /// next-13-bits window we tried to match is reported for
    /// diagnostics.
    InvalidMvData {
        /// The next-13-bits window (right-aligned) we tried to match.
        window: u16,
    },
    /// `vop_fcode` was outside the valid `1..=7` range (Table 7-9). A
    /// value of `0` is forbidden by §6.3.5; values above `7` exceed the
    /// table.
    InvalidFcode(u8),
}

impl core::fmt::Display for MotionParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MotionParseError::Truncated => write!(f, "motion vector field truncated"),
            MotionParseError::InvalidMvData { window } => {
                write!(f, "invalid mv_data prefix (next-13-bits = 0b{window:013b})")
            }
            MotionParseError::InvalidFcode(c) => {
                write!(f, "vop_fcode {c} outside valid range 1..=7")
            }
        }
    }
}

impl std::error::Error for MotionParseError {}

impl From<BitReaderError> for MotionParseError {
    fn from(_: BitReaderError) -> Self {
        MotionParseError::Truncated
    }
}

/// A reconstructed differential motion vector (the §7.6.3 `MVDx`,
/// `MVDy`), in half-sample units when `quarter_sample == 0` or
/// quarter-sample units when `quarter_sample == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVectorDelta {
    /// Reconstructed horizontal differential component (`MVDx`).
    pub dx: i32,
    /// Reconstructed vertical differential component (`MVDy`).
    pub dy: i32,
}

/// A reconstructed (final) motion vector — the §7.6.3 `MVx`, `MVy`
/// after the predictor add and the modulo wrap into `[low:high]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVector {
    /// Reconstructed horizontal component (`MVx`).
    pub x: i32,
    /// Reconstructed vertical component (`MVy`).
    pub y: i32,
}

// ---------------------------------------------------------------------------
// Table B.12 — VLC table for MVD (ISO/IEC 14496-2:2004 §B.1.3).
//
// Each entry is `(code_bits, code_len, mv_data)` where `mv_data` is the
// **doubled** integer the §7.6.3 note specifies: `mv_data = 2 * (vector
// differences column)`. The "vector differences" run from −16 to +16 in
// 0.5 steps, so the on-wire `mv_data` runs from −32 to +32.
//
// The table is symmetric in length but not bit-pattern, so it is
// transcribed in full. A 13-bit window is enough to disambiguate every
// row (the longest codes are 13 bits).
// ---------------------------------------------------------------------------

/// One Table B.12 row: `(code_bits, code_len, mv_data)`.
///
/// The binary literals are written with no digit grouping: the codes
/// are variable-length VLC prefixes (1..13 bits), so uniform 4-bit
/// nibble grouping would misrepresent the bit boundaries. The authentic
/// nibble layout from the spec table is reproduced in the trailing
/// `// "bits" → half-pel` comment on each row.
#[rustfmt::skip]
const MVD_TABLE: &[(u16, u8, i32)] = &[
    // Negative half — "vector differences" −16 … −0.5.
    (0b0000000000101, 13, -32), // 0000 0000 0010 1 → -16
    (0b0000000000111, 13, -31), // 0000 0000 0011 1 → -15.5
    (0b000000000101,  12, -30), // 0000 0000 0101   → -15
    (0b000000000111,  12, -29), // 0000 0000 0111   → -14.5
    (0b000000001001,  12, -28), // 0000 0000 1001   → -14
    (0b000000001011,  12, -27), // 0000 0000 1011   → -13.5
    (0b000000001101,  12, -26), // 0000 0000 1101   → -13
    (0b000000001111,  12, -25), // 0000 0000 1111   → -12.5
    (0b00000001001,   11, -24), // 0000 0001 001    → -12
    (0b00000001011,   11, -23), // 0000 0001 011    → -11.5
    (0b00000001101,   11, -22), // 0000 0001 101    → -11
    (0b00000001111,   11, -21), // 0000 0001 111    → -10.5
    (0b00000010001,   11, -20), // 0000 0010 001    → -10
    (0b00000010011,   11, -19), // 0000 0010 011    → -9.5
    (0b00000010101,   11, -18), // 0000 0010 101    → -9
    (0b00000010111,   11, -17), // 0000 0010 111    → -8.5
    (0b00000011001,   11, -16), // 0000 0011 001    → -8
    (0b00000011011,   11, -15), // 0000 0011 011    → -7.5
    (0b00000011101,   11, -14), // 0000 0011 101    → -7
    (0b00000011111,   11, -13), // 0000 0011 111    → -6.5
    (0b00000100001,   11, -12), // 0000 0100 001    → -6
    (0b00000100011,   11, -11), // 0000 0100 011    → -5.5
    (0b0000010011,    10, -10), // 0000 0100 11     → -5
    (0b0000010101,    10, -9),  // 0000 0101 01     → -4.5
    (0b0000010111,    10, -8),  // 0000 0101 11     → -4
    (0b00000111,      8,  -7),  // 0000 0111        → -3.5
    (0b00001001,      8,  -6),  // 0000 1001        → -3
    (0b00001011,      8,  -5),  // 0000 1011        → -2.5
    (0b0000111,       7,  -4),  // 0000 111         → -2
    (0b00011,         5,  -3),  // 0001 1           → -1.5
    (0b0011,          4,  -2),  // 0011             → -1
    (0b011,           3,  -1),  // 011              → -0.5
    // Zero.
    (0b1,             1,  0),   // 1                → 0
    // Positive half — "vector differences" +0.5 … +16.
    (0b010,           3,  1),   // 010              → 0.5
    (0b0010,          4,  2),   // 0010             → 1
    (0b00010,         5,  3),   // 0001 0           → 1.5
    (0b0000110,       7,  4),   // 0000 110         → 2
    (0b00001010,      8,  5),   // 0000 1010        → 2.5
    (0b00001000,      8,  6),   // 0000 1000        → 3
    (0b00000110,      8,  7),   // 0000 0110        → 3.5
    (0b0000010110,    10, 8),   // 0000 0101 10     → 4
    (0b0000010100,    10, 9),   // 0000 0101 00     → 4.5
    (0b0000010010,    10, 10),  // 0000 0100 10     → 5
    (0b00000100010,   11, 11),  // 0000 0100 010    → 5.5
    (0b00000100000,   11, 12),  // 0000 0100 000    → 6
    (0b00000011110,   11, 13),  // 0000 0011 110    → 6.5
    (0b00000011100,   11, 14),  // 0000 0011 100    → 7
    (0b00000011010,   11, 15),  // 0000 0011 010    → 7.5
    (0b00000011000,   11, 16),  // 0000 0011 000    → 8
    (0b00000010110,   11, 17),  // 0000 0010 110    → 8.5
    (0b00000010100,   11, 18),  // 0000 0010 100    → 9
    (0b00000010010,   11, 19),  // 0000 0010 010    → 9.5
    (0b00000010000,   11, 20),  // 0000 0010 000    → 10
    (0b00000001110,   11, 21),  // 0000 0001 110    → 10.5
    (0b00000001100,   11, 22),  // 0000 0001 100    → 11
    (0b00000001010,   11, 23),  // 0000 0001 010    → 11.5
    (0b00000001000,   11, 24),  // 0000 0001 000    → 12
    (0b000000001110,  12, 25),  // 0000 0000 1110   → 12.5
    (0b000000001100,  12, 26),  // 0000 0000 1100   → 13
    (0b000000001010,  12, 27),  // 0000 0000 1010   → 13.5
    (0b000000001000,  12, 28),  // 0000 0000 1000   → 14
    (0b000000000110,  12, 29),  // 0000 0000 0110   → 14.5
    (0b000000000100,  12, 30),  // 0000 0000 0100   → 15
    (0b0000000000110, 13, 31),  // 0000 0000 0011 0 → 15.5
    (0b0000000000100, 13, 32),  // 0000 0000 0010 0 → 16 (forbidden when vop_fcode==1)
];

/// The widest Table B.12 code is 13 bits.
const MVD_MAX_CODE_LEN: usize = 13;

/// Decode one `mv_data` VLC (Table B.12) from `br`. Returns the doubled
/// integer `mv_data` value (`= 2 * vector_differences`).
fn decode_mv_data(br: &mut BitReader<'_>) -> Result<i32, MotionParseError> {
    let remaining = br.remaining_bits().min(MVD_MAX_CODE_LEN);
    if remaining == 0 {
        return Err(MotionParseError::Truncated);
    }
    let window = br.next_bits(remaining)? as u16;
    let window_len = remaining as u8;
    for &(code, len, mv_data) in MVD_TABLE {
        if (len as usize) <= remaining {
            let shift = window_len - len;
            if (window >> shift) == code {
                br.skip_bits(len as usize)?;
                return Ok(mv_data);
            }
        }
    }
    Err(MotionParseError::InvalidMvData { window })
}

/// Reconstruct one differential component from a decoded `mv_data` and
/// its (optional) residual, per the §7.6.3 recurrence.
///
/// `f = 1 << (vop_fcode - 1)`. When `f == 1` or `mv_data == 0` the
/// component is `mv_data` itself; otherwise it is
/// `(Abs(mv_data) - 1) * f + residual + 1`, negated when `mv_data < 0`.
fn reconstruct_component(mv_data: i32, residual: i32, f: i32) -> i32 {
    if f == 1 || mv_data == 0 {
        mv_data
    } else {
        let magnitude = (mv_data.abs() - 1) * f + residual + 1;
        if mv_data < 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// `f`, `low`, `high`, `range` for a given `vop_fcode` (§7.6.3).
///
/// Returns `Err(InvalidFcode)` when `vop_fcode` is outside `1..=7`.
fn fcode_bounds(vop_fcode: u8) -> Result<(i32, i32, i32, i32), MotionParseError> {
    if !(1..=7).contains(&vop_fcode) {
        return Err(MotionParseError::InvalidFcode(vop_fcode));
    }
    let r_size = i32::from(vop_fcode) - 1;
    let f = 1i32 << r_size;
    let high = 32 * f - 1;
    let low = -32 * f;
    let range = 64 * f;
    Ok((f, low, high, range))
}

/// Decode one `motion_vector(mode)` body and reconstruct its
/// differential motion vector (`MVDx`, `MVDy`) per §6.2.6.2 + §7.6.3.
///
/// `vop_fcode` is `vop_fcode_forward` for [`MvMode::Forward`] /
/// [`MvMode::Direct`] and `vop_fcode_backward` for
/// [`MvMode::Backward`]; the caller picks the right one from the VOP
/// header. The residuals are read only when `vop_fcode != 1 && mv_data
/// != 0`, exactly as the §6.2.6.2 syntax gates them. The `"direct"`
/// branch never reads residuals.
///
/// On return the bit reader is positioned immediately after the
/// (last) component's residual / `mv_data`.
pub fn decode_motion_vector_delta(
    br: &mut BitReader<'_>,
    mode: MvMode,
    vop_fcode: u8,
) -> Result<MotionVectorDelta, MotionParseError> {
    let (f, _low, _high, _range) = fcode_bounds(vop_fcode)?;
    let reads_residual = mode != MvMode::Direct && f != 1;

    let h_data = decode_mv_data(br)?;
    let h_residual = if reads_residual && h_data != 0 {
        br.read_bits(r_size_bits(vop_fcode))? as i32
    } else {
        0
    };
    let v_data = decode_mv_data(br)?;
    let v_residual = if reads_residual && v_data != 0 {
        br.read_bits(r_size_bits(vop_fcode))? as i32
    } else {
        0
    };

    Ok(MotionVectorDelta {
        dx: reconstruct_component(h_data, h_residual, f),
        dy: reconstruct_component(v_data, v_residual, f),
    })
}

/// `r_size = vop_fcode - 1`, the residual width in bits (§6.3.6.2).
fn r_size_bits(vop_fcode: u8) -> usize {
    usize::from(vop_fcode.saturating_sub(1))
}

/// Combine a reconstructed differential MV with a predictor `(px, py)`
/// and apply the §7.6.3 modulo wrap into the `[low:high]` range
/// (Table 7-9), producing the final `(MVx, MVy)`.
///
/// `vop_fcode` selects the wrap range. The predictor itself is the
/// caller's responsibility (median of neighbouring block vectors,
/// §7.6.5) and is out of scope for this module.
pub fn reconstruct_motion_vector(
    delta: MotionVectorDelta,
    px: i32,
    py: i32,
    vop_fcode: u8,
) -> Result<MotionVector, MotionParseError> {
    let (_f, low, high, range) = fcode_bounds(vop_fcode)?;
    Ok(MotionVector {
        x: wrap_component(px + delta.dx, low, high, range),
        y: wrap_component(py + delta.dy, low, high, range),
    })
}

/// The §7.6.3 modulo wrap: add `range` once if below `low`, subtract
/// `range` once if above `high`.
fn wrap_component(mut value: i32, low: i32, high: i32, range: i32) -> i32 {
    if value < low {
        value += range;
    }
    if value > high {
        value -= range;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer matching the spec's bslbf / uimsbf
    /// convention. Mirrors the helpers in the other test modules.
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
    fn mvd_table_has_65_unique_rows() {
        assert_eq!(MVD_TABLE.len(), 65);
        // mv_data values cover the contiguous range -32..=32.
        let mut seen: Vec<i32> = MVD_TABLE.iter().map(|&(_, _, v)| v).collect();
        seen.sort_unstable();
        let expected: Vec<i32> = (-32..=32).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn mvd_table_is_prefix_free() {
        // No code may be a prefix of another, else the linear scan is
        // ambiguous. Compare every ordered pair.
        for (i, &(ci, li, _)) in MVD_TABLE.iter().enumerate() {
            for (j, &(cj, lj, _)) in MVD_TABLE.iter().enumerate() {
                if i == j {
                    continue;
                }
                if li <= lj {
                    let shift = lj - li;
                    assert_ne!(
                        cj >> shift,
                        ci,
                        "code {ci:b}/{li} is a prefix of {cj:b}/{lj}"
                    );
                }
            }
        }
    }

    #[test]
    fn decodes_zero_mv_data() {
        // Code `1` → mv_data 0.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        assert_eq!(decode_mv_data(&mut br).unwrap(), 0);
    }

    #[test]
    fn decodes_full_b12_table_round_trip() {
        for &(code, len, mv_data) in MVD_TABLE {
            let mut w = BitWriter::new();
            w.write_bits(code as u32, len as usize);
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let got = decode_mv_data(&mut br).unwrap();
            assert_eq!(got, mv_data, "code={code:b} len={len}");
            assert_eq!(br.bit_position(), len as usize, "consumed wrong bit count");
        }
    }

    #[test]
    fn fcode_bounds_match_table_7_9() {
        // Table 7-9: vop_fcode → [low:high].
        let expected = [
            (1u8, -32i32, 31i32),
            (2, -64, 63),
            (3, -128, 127),
            (4, -256, 255),
            (5, -512, 511),
            (6, -1024, 1023),
            (7, -2048, 2047),
        ];
        for (fc, low, high) in expected {
            let (_f, got_low, got_high, _range) = fcode_bounds(fc).unwrap();
            assert_eq!(got_low, low, "fcode={fc}");
            assert_eq!(got_high, high, "fcode={fc}");
        }
    }

    #[test]
    fn fcode_zero_and_eight_rejected() {
        assert_eq!(
            fcode_bounds(0).unwrap_err(),
            MotionParseError::InvalidFcode(0)
        );
        assert_eq!(
            fcode_bounds(8).unwrap_err(),
            MotionParseError::InvalidFcode(8)
        );
    }

    #[test]
    fn forward_fcode1_no_residual() {
        // vop_fcode == 1 → f == 1, so no residual bits. mv_data pair
        // (h=2 from code 0010, v=-2 from code 0011). MVD == mv_data.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4); // h_data = 2
        w.write_bits(0b0011, 4); // v_data = -2
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Forward, 1).unwrap();
        assert_eq!(mvd.dx, 2);
        assert_eq!(mvd.dy, -2);
    }

    #[test]
    fn forward_fcode2_reads_residual() {
        // vop_fcode == 2 → f == 2, r_size == 1 (one residual bit each).
        // h_data = 2 (code 0010), residual = 1 → MVDx = (|2|-1)*2 + 1 + 1
        //   = 2*1 + 2 = 4.
        // v_data = -2 (code 0011), residual = 0 → MVDy = -((|−2|-1)*2 + 0
        //   + 1) = -(2 + 1) = -3.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4); // h_data = 2
        w.write_bits(0b1, 1); // h_residual = 1
        w.write_bits(0b0011, 4); // v_data = -2
        w.write_bits(0b0, 1); // v_residual = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Forward, 2).unwrap();
        assert_eq!(mvd.dx, 4);
        assert_eq!(mvd.dy, -3);
    }

    #[test]
    fn forward_fcode2_zero_mv_data_skips_residual() {
        // h_data = 0 (code 1) → no residual even though f != 1.
        // v_data = 2 (code 0010), residual = 1 → MVDy = (2-1)*2 + 1 + 1
        //   = 4.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // h_data = 0 → no residual
        w.write_bits(0b0010, 4); // v_data = 2
        w.write_bits(0b1, 1); // v_residual = 1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Forward, 2).unwrap();
        assert_eq!(mvd.dx, 0);
        assert_eq!(mvd.dy, 4);
    }

    #[test]
    fn direct_mode_never_reads_residual() {
        // Even with vop_fcode != 1, the direct branch reads only the
        // two mv_data VLCs. h_data = 2 (0010), v_data = -2 (0011). With
        // no residual the §7.6.3 reconstruction still applies (f != 1),
        // giving MVDx = (2-1)*2 + 0 + 1 = 3, MVDy = -((2-1)*2 + 0 + 1)
        // = -3.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4); // h_data = 2
        w.write_bits(0b0011, 4); // v_data = -2 — no residuals follow
        w.write_bits(0b1010_1010, 8); // sentinel: next field
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Direct, 2).unwrap();
        assert_eq!(mvd.dx, 3);
        assert_eq!(mvd.dy, -3);
        // Confirm the reader stopped right after the two 4-bit VLCs.
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    #[test]
    fn backward_mode_uses_supplied_fcode() {
        // Backward mode behaves like forward but the caller passes
        // vop_fcode_backward. fcode == 1 here → no residual.
        let mut w = BitWriter::new();
        w.write_bits(0b011, 3); // h_data = -1
        w.write_bits(0b010, 3); // v_data = 1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Backward, 1).unwrap();
        assert_eq!(mvd.dx, -1);
        assert_eq!(mvd.dy, 1);
    }

    #[test]
    fn reconstruct_adds_predictor_no_wrap() {
        let delta = MotionVectorDelta { dx: 4, dy: -3 };
        let mv = reconstruct_motion_vector(delta, 2, 5, 2).unwrap();
        // fcode 2 → range [-64,63]; 2+4=6 and 5-3=2 stay in range.
        assert_eq!(mv.x, 6);
        assert_eq!(mv.y, 2);
    }

    #[test]
    fn reconstruct_wraps_below_low() {
        // fcode 1 → [-32,31], range 64. Px + MVDx = -30 + -10 = -40 < -32
        // → +64 → 24.
        let delta = MotionVectorDelta { dx: -10, dy: 0 };
        let mv = reconstruct_motion_vector(delta, -30, 0, 1).unwrap();
        assert_eq!(mv.x, 24);
        assert_eq!(mv.y, 0);
    }

    #[test]
    fn reconstruct_wraps_above_high() {
        // fcode 1 → [-32,31], range 64. Px + MVDx = 30 + 10 = 40 > 31
        // → -64 → -24.
        let delta = MotionVectorDelta { dx: 10, dy: 0 };
        let mv = reconstruct_motion_vector(delta, 30, 0, 1).unwrap();
        assert_eq!(mv.x, -24);
    }

    #[test]
    fn reconstruct_boundary_values_unchanged() {
        // Exactly low / high stay put.
        let (_f, low, high, _range) = fcode_bounds(1).unwrap();
        let mv =
            reconstruct_motion_vector(MotionVectorDelta { dx: 0, dy: 0 }, low, high, 1).unwrap();
        assert_eq!(mv.x, low);
        assert_eq!(mv.y, high);
    }

    #[test]
    fn full_pipeline_decode_then_reconstruct() {
        // Decode a forward MV (fcode 2), then add a predictor and wrap.
        // h_data = 2 (0010), residual 1 → MVDx = 4.
        // v_data = -2 (0011), residual 0 → MVDy = -3.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4);
        w.write_bits(0b1, 1);
        w.write_bits(0b0011, 4);
        w.write_bits(0b0, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Forward, 2).unwrap();
        let mv = reconstruct_motion_vector(mvd, 60, -62, 2).unwrap();
        // fcode 2 → [-64,63], range 128.
        // x: 60 + 4 = 64 > 63 → -128 → -64.
        // y: -62 + -3 = -65 < -64 → +128 → 63.
        assert_eq!(mv.x, -64);
        assert_eq!(mv.y, 63);
    }

    #[test]
    fn truncated_mv_data_errors() {
        let data: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_mv_data(&mut br).unwrap_err(),
            MotionParseError::Truncated
        );
    }

    #[test]
    fn invalid_mv_data_errors() {
        // 13 zero bits never match a Table B.12 code (every code has a
        // 1 bit somewhere in the first 13 positions).
        let mut w = BitWriter::new();
        w.write_bits(0, 13);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        assert!(matches!(
            decode_mv_data(&mut br).unwrap_err(),
            MotionParseError::InvalidMvData { .. }
        ));
    }

    #[test]
    fn truncated_mid_residual_errors() {
        // vop_fcode 7 → r_size 6. h_data = 2 (0010) then EOF before the
        // 6 residual bits.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4);
        let data = w.buf; // only 4 bits laid down, byte zero-padded
        let mut br = BitReader::new(&data);
        // After h_data the reader has 4 bits left in the byte (zeros),
        // which is fewer than the 6 residual bits → truncation.
        let err = decode_motion_vector_delta(&mut br, MvMode::Forward, 7).unwrap_err();
        assert_eq!(err, MotionParseError::Truncated);
    }

    #[test]
    fn invalid_fcode_propagates_from_delta_decode() {
        let data = vec![0xFF];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_motion_vector_delta(&mut br, MvMode::Forward, 0).unwrap_err(),
            MotionParseError::InvalidFcode(0)
        );
    }

    #[test]
    fn error_display_covers_all_variants() {
        let cases = [
            MotionParseError::Truncated,
            MotionParseError::InvalidMvData { window: 0 },
            MotionParseError::InvalidFcode(9),
        ];
        for e in cases {
            assert!(!format!("{e}").is_empty());
        }
    }

    #[test]
    fn reconstruct_component_helper_matches_spec_examples() {
        // f == 1 path: component is mv_data verbatim.
        assert_eq!(reconstruct_component(5, 0, 1), 5);
        assert_eq!(reconstruct_component(-5, 99, 1), -5);
        // mv_data == 0 path: component is 0 regardless of f.
        assert_eq!(reconstruct_component(0, 3, 4), 0);
        // general path, positive: (|3|-1)*4 + 2 + 1 = 8 + 3 = 11.
        assert_eq!(reconstruct_component(3, 2, 4), 11);
        // general path, negative: -((|−3|-1)*4 + 2 + 1) = -11.
        assert_eq!(reconstruct_component(-3, 2, 4), -11);
    }
}

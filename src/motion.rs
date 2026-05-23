//! Motion-vector bitstream decode — §6.2.6.2 `motion_vector(mode)` plus
//! the §7.6.3 differential-MV reconstruction, against Table B.12 from
//! ISO/IEC 14496-2:2004 §B.1.3.
//!
//! Round 6 left the bit reader positioned at the start of the
//! `motion_vector("…")` body for both P-VOP and B-VOP macroblocks. This
//! module consumes that body: for each requested component it decodes
//! the `*_mv_data` VLC (Table B.12), conditionally reads the
//! `*_mv_residual` field (when `vop_fcode != 1 && mv_data != 0`), and
//! reconstructs the differential motion vector component `MVD` exactly
//! per the equivalent process in §7.6.3.
//!
//! ## What lands this round
//!
//! * [`decode_mv_data`] — Table B.12 prefix VLC → the signed `mv_data`
//!   integer (the spec's "two times the value found in the vector
//!   differences column", i.e. in half-sample units, range
//!   `[-32, 32]`).
//! * [`decode_mv_component`] — one (`mv_data`, optional `mv_residual`)
//!   pair → reconstructed `MVD` component per §7.6.3.
//! * [`decode_motion_vector`] — the full `motion_vector(mode)` syntax
//!   for `forward` / `backward` / `direct`, returning a typed
//!   [`MotionVectorDelta`] of the two reconstructed `MVD` components.
//! * [`apply_predictor`] — the §7.6.3 `MV = P + MVD` step with the
//!   `[low:high]` modulo wrap (Table 7-9). The motion-vector *predictor*
//!   `(Px, Py)` itself depends on neighbouring-macroblock grid state,
//!   which is not yet modelled; the caller supplies the predictor and
//!   this helper applies the final wrap. The decode of the bitstream
//!   (everything above) is independent of the predictor.
//!
//! ## Out of scope (this round)
//!
//! * Predictor derivation `(Px, Py)` — the median-of-three / candidate
//!   selection from §7.6.2 needs the per-macroblock MV grid, which the
//!   crate does not yet maintain.
//! * Direct-mode MV scaling from the co-located P-VOP forward vector
//!   (§7.6.5) — `direct` mode here decodes only the bitstream `MVD`
//!   delta (the syntax reads exactly two `*_mv_data` VLCs with no
//!   residual); the temporal scaling is a later round.
//! * The interlaced field-prediction extra MV pair (`field_prediction`
//!   in §6.2.6.3) — gated out ahead of the call.
//! * 8×8 (Inter4V) per-block MV iteration — the caller decides how many
//!   times to invoke [`decode_motion_vector`]; this module decodes a
//!   single MV.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by the
//! agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.6.2 `motion_vector(mode)` syntax — the `direct` / `forward` /
//!   `backward` branches with `horizontal_mv_data` / `vertical_mv_data`
//!   (1-13 vlclbf) and `horizontal_mv_residual` / `vertical_mv_residual`
//!   (1-6 uimsbf).
//! * §6.3.6.2 semantics — `r_size = vop_fcode - 1`.
//! * §7.6.3 general motion-vector decoding process — the `f`,
//!   `r_size`, `high`, `low`, `range` reconstruction and the
//!   `mv_data == 2 * (vector difference)` note.
//! * Table B.12 — VLC for MVD.
//! * Table 7-9 — `[low:high]` range per `vop_fcode`.

use crate::bitreader::{BitReader, BitReaderError};

/// The `mode` argument of the §6.2.6.2 `motion_vector(mode)` syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvMode {
    /// `motion_vector("direct")` — two `*_mv_data` VLCs, no residual
    /// fields (the residual gate is omitted in the direct branch of the
    /// syntax). The reconstructed delta is the "direct-mode delta MV"
    /// the spec adds to the scaled co-located vector in §7.6.5.
    Direct,
    /// `motion_vector("forward")` — uses `vop_fcode_forward`. Residual
    /// fields present when `vop_fcode_forward != 1 && mv_data != 0`.
    Forward,
    /// `motion_vector("backward")` — uses `vop_fcode_backward`. Residual
    /// fields present when `vop_fcode_backward != 1 && mv_data != 0`.
    Backward,
}

/// A reconstructed differential motion vector — the `(MVDx, MVDy)` pair
/// from §7.6.3, in half-sample units when `quarter_sample == 0` or
/// quarter-sample units when `quarter_sample == 1`.
///
/// This is the bitstream-derived *delta*; the final motion vector is
/// `(Px + MVDx, Py + MVDy)` with the modulo wrap, computed by
/// [`apply_predictor`] once the predictor is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVectorDelta {
    /// Horizontal component `MVDx`.
    pub mvdx: i32,
    /// Vertical component `MVDy`.
    pub mvdy: i32,
}

/// Errors produced by the motion-vector decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionParseError {
    /// The supplied bit reader ran out mid-field.
    Truncated,
    /// The next bits did not match any Table B.12 codeword. The
    /// 13-bit window we tried to match (right-aligned) is reported.
    InvalidMvData {
        /// The next-up-to-13-bits window we tried to match.
        window: u16,
    },
    /// `vop_fcode` was outside the legal 1..=7 range (§6.3.5 forbids 0;
    /// Table 7-9 tops out at 7).
    InvalidFcode(u8),
}

impl core::fmt::Display for MotionParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MotionParseError::Truncated => write!(f, "motion vector body truncated"),
            MotionParseError::InvalidMvData { window } => {
                write!(
                    f,
                    "invalid mv_data VLC (next-bits window = 0b{window:013b})"
                )
            }
            MotionParseError::InvalidFcode(fc) => {
                write!(f, "vop_fcode out of range (got {fc}, valid 1..=7)")
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

// ---------------------------------------------------------------------------
// Table B.12 — VLC for MVD. Verbatim from ISO/IEC 14496-2:2004 §B.1.3.
//
// Each entry is (code_bits, code_len, mv_data). `mv_data` is the spec's
// "two times the value found in the vector differences column" (§7.6.3):
// the table's vector-difference values run -16..+16 in 0.5 steps, so the
// stored mv_data values run -32..+32 in unit steps. The codes are
// prefix-free, so a single pass that matches each row's top `len` bits
// against the right-aligned peek window identifies the codeword
// unambiguously — at most one row can match.
// ---------------------------------------------------------------------------

/// `(code, code_len_bits, mv_data)`. 65 rows, ordered as printed in the
/// spec (vector difference -16 .. +16).
///
/// The binary literals are grouped to mirror Table B.12's printed
/// codewords (a leading run of 4-bit groups followed by the remaining
/// 1-3 bits), so the irregular `unusual_byte_groupings` shape is
/// intentional — it keeps each Rust literal byte-for-byte alignable
/// against the spec page.
#[allow(clippy::unusual_byte_groupings)]
const MV_VLC_TABLE: &[(u16, u8, i16)] = &[
    (0b0000_0000_0010_1, 13, -32),
    (0b0000_0000_0011_1, 13, -31),
    (0b0000_0000_0101, 12, -30),
    (0b0000_0000_0111, 12, -29),
    (0b0000_0000_1001, 12, -28),
    (0b0000_0000_1011, 12, -27),
    (0b0000_0000_1101, 12, -26),
    (0b0000_0000_1111, 12, -25),
    (0b0000_0001_001, 11, -24),
    (0b0000_0001_011, 11, -23),
    (0b0000_0001_101, 11, -22),
    (0b0000_0001_111, 11, -21),
    (0b0000_0010_001, 11, -20),
    (0b0000_0010_011, 11, -19),
    (0b0000_0010_101, 11, -18),
    (0b0000_0010_111, 11, -17),
    (0b0000_0011_001, 11, -16),
    (0b0000_0011_011, 11, -15),
    (0b0000_0011_101, 11, -14),
    (0b0000_0011_111, 11, -13),
    (0b0000_0100_001, 11, -12),
    (0b0000_0100_011, 11, -11),
    (0b0000_0100_11, 10, -10),
    (0b0000_0101_01, 10, -9),
    (0b0000_0101_11, 10, -8),
    (0b0000_0111, 8, -7),
    (0b0000_1001, 8, -6),
    (0b0000_1011, 8, -5),
    (0b0000_111, 7, -4),
    (0b0001_1, 5, -3),
    (0b0011, 4, -2),
    (0b011, 3, -1),
    (0b1, 1, 0),
    (0b010, 3, 1),
    (0b0010, 4, 2),
    (0b0001_0, 5, 3),
    (0b0000_110, 7, 4),
    (0b0000_1010, 8, 5),
    (0b0000_1000, 8, 6),
    (0b0000_0110, 8, 7),
    (0b0000_0101_10, 10, 8),
    (0b0000_0101_00, 10, 9),
    (0b0000_0100_10, 10, 10),
    (0b0000_0100_010, 11, 11),
    (0b0000_0100_000, 11, 12),
    (0b0000_0011_110, 11, 13),
    (0b0000_0011_100, 11, 14),
    (0b0000_0011_010, 11, 15),
    (0b0000_0011_000, 11, 16),
    (0b0000_0010_110, 11, 17),
    (0b0000_0010_100, 11, 18),
    (0b0000_0010_010, 11, 19),
    (0b0000_0010_000, 11, 20),
    (0b0000_0001_110, 11, 21),
    (0b0000_0001_100, 11, 22),
    (0b0000_0001_010, 11, 23),
    (0b0000_0001_000, 11, 24),
    (0b0000_0000_1110, 12, 25),
    (0b0000_0000_1100, 12, 26),
    (0b0000_0000_1010, 12, 27),
    (0b0000_0000_1000, 12, 28),
    (0b0000_0000_0110, 12, 29),
    (0b0000_0000_0100, 12, 30),
    (0b0000_0000_0011_0, 13, 31),
    (0b0000_0000_0010_0, 13, 32),
];

/// Decode one `*_mv_data` VLC (Table B.12). Returns the signed `mv_data`
/// integer (two times the vector-difference column, range `[-32, 32]`).
///
/// The codes are prefix-free, so a single peek of up to 13 bits followed
/// by a longest-match scan unambiguously identifies the codeword. We
/// scan the table once, matching each row's `len` MSBs against the
/// right-aligned peek window; only a true codeword can match (the
/// prefix-free property guarantees at most one row matches).
pub fn decode_mv_data(br: &mut BitReader<'_>) -> Result<i16, MotionParseError> {
    let available = br.remaining_bits().min(13);
    if available == 0 {
        return Err(MotionParseError::Truncated);
    }
    let window = br.next_bits(available)? as u16;
    let window_len = available as u8;
    for &(code, len, mv_data) in MV_VLC_TABLE {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                br.skip_bits(len as usize)?;
                return Ok(mv_data);
            }
        }
    }
    // No row matched within the available bits. If we had fewer than 13
    // bits this is a truncation; otherwise the bitstream is malformed.
    if window_len < 13 {
        Err(MotionParseError::Truncated)
    } else {
        Err(MotionParseError::InvalidMvData { window })
    }
}

/// Reconstruct one differential-MV component `MVD` per §7.6.3, given the
/// already-decoded `mv_data` VLC value and the relevant `vop_fcode`.
///
/// Reads the `*_mv_residual` field from `br` only when
/// `vop_fcode != 1 && mv_data != 0`, exactly as the §6.2.6.2 syntax
/// gates it. `r_size = vop_fcode - 1` residual bits are consumed in that
/// case.
///
/// Returns the reconstructed `MVD` component.
pub fn decode_mv_component(
    br: &mut BitReader<'_>,
    mv_data: i16,
    vop_fcode: u8,
) -> Result<i32, MotionParseError> {
    if !(1..=7).contains(&vop_fcode) {
        return Err(MotionParseError::InvalidFcode(vop_fcode));
    }
    let r_size = vop_fcode - 1;
    let f: i32 = 1 << r_size;

    // §6.2.6.2: residual present iff (vop_fcode != 1) && (mv_data != 0).
    // §7.6.3: when f == 1 || mv_data == 0, MVD == mv_data with no
    // residual. The two conditions coincide (f == 1 ⇔ vop_fcode == 1).
    if f == 1 || mv_data == 0 {
        return Ok(mv_data as i32);
    }

    let residual = br.read_bits(r_size as usize)? as i32;
    let abs = (mv_data as i32).abs();
    let mut mvd = (abs - 1) * f + residual + 1;
    if mv_data < 0 {
        mvd = -mvd;
    }
    Ok(mvd)
}

/// Decode a full `motion_vector(mode)` body (§6.2.6.2) into a typed
/// [`MotionVectorDelta`].
///
/// * `Direct` — reads `horizontal_mv_data` then `vertical_mv_data`, with
///   **no** residual fields (the direct branch of the syntax omits the
///   residual). The reconstruction therefore returns the raw mv_data
///   values widened to `i32`.
/// * `Forward` — uses `fcode_forward`; residual gated as in §6.2.6.2.
/// * `Backward` — uses `fcode_backward`; residual gated as in §6.2.6.2.
///
/// The bit reader is left positioned immediately after the final field
/// of the body.
pub fn decode_motion_vector(
    br: &mut BitReader<'_>,
    mode: MvMode,
    fcode_forward: u8,
    fcode_backward: u8,
) -> Result<MotionVectorDelta, MotionParseError> {
    match mode {
        MvMode::Direct => {
            // The direct branch reads two raw mv_data VLCs and no
            // residual fields (§6.2.6.2). The delta is the table value
            // directly; the spec adds it to the scaled co-located MV in
            // §7.6.5 (out of scope this round).
            let hx = decode_mv_data(br)?;
            let vy = decode_mv_data(br)?;
            Ok(MotionVectorDelta {
                mvdx: hx as i32,
                mvdy: vy as i32,
            })
        }
        MvMode::Forward => {
            if !(1..=7).contains(&fcode_forward) {
                return Err(MotionParseError::InvalidFcode(fcode_forward));
            }
            let hx = decode_mv_data(br)?;
            let mvdx = decode_mv_component(br, hx, fcode_forward)?;
            let vy = decode_mv_data(br)?;
            let mvdy = decode_mv_component(br, vy, fcode_forward)?;
            Ok(MotionVectorDelta { mvdx, mvdy })
        }
        MvMode::Backward => {
            if !(1..=7).contains(&fcode_backward) {
                return Err(MotionParseError::InvalidFcode(fcode_backward));
            }
            let hx = decode_mv_data(br)?;
            let mvdx = decode_mv_component(br, hx, fcode_backward)?;
            let vy = decode_mv_data(br)?;
            let mvdy = decode_mv_component(br, vy, fcode_backward)?;
            Ok(MotionVectorDelta { mvdx, mvdy })
        }
    }
}

/// Apply the §7.6.3 `MV = P + MVD` step with the `[low:high]` modulo
/// wrap (Table 7-9) to a single component.
///
/// `predictor` is the caller-supplied component of `(Px, Py)`; deriving
/// the predictor from the neighbouring-macroblock MV grid is out of
/// scope this round (§7.6.2). `vop_fcode` selects the wrap range:
///
/// ```text
/// r_size = vop_fcode - 1
/// f      = 1 << r_size
/// high   = 32 * f - 1
/// low    = -32 * f
/// range  = 64 * f
/// MV = predictor + MVD
/// if MV < low  { MV += range }
/// if MV > high { MV -= range }
/// ```
///
/// Returns the wrapped final motion-vector component, guaranteed to lie
/// in `[low, high]` for any in-range predictor + delta.
pub fn apply_predictor(predictor: i32, mvd: i32, vop_fcode: u8) -> Result<i32, MotionParseError> {
    if !(1..=7).contains(&vop_fcode) {
        return Err(MotionParseError::InvalidFcode(vop_fcode));
    }
    let r_size = vop_fcode - 1;
    let f: i32 = 1 << r_size;
    let high = 32 * f - 1;
    let low = -32 * f;
    let range = 64 * f;
    let mut mv = predictor + mvd;
    if mv < low {
        mv += range;
    }
    if mv > high {
        mv -= range;
    }
    Ok(mv)
}

#[cfg(test)]
// Binary literals in the tests mirror the spec's Table B.12 codeword
// groupings, so the irregular byte-grouping shape is intentional.
#[allow(clippy::unusual_byte_groupings)]
mod tests {
    use super::*;

    /// MSB-first bit writer matching the spec's bslbf / uimsbf
    /// convention. Mirrors the helper in `bvop::tests` / `macroblock`.
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
    fn mv_vlc_table_is_complete_and_prefix_free() {
        // 65 rows, mv_data covers -32..=32 exactly once.
        assert_eq!(MV_VLC_TABLE.len(), 65);
        let mut seen = [false; 65];
        for &(_, _, mv) in MV_VLC_TABLE {
            assert!((-32..=32).contains(&mv), "mv_data {mv} out of range");
            seen[(mv + 32) as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "mv_data values not contiguous");

        // Prefix-free: no code is a prefix of another.
        for &(ca, la, _) in MV_VLC_TABLE {
            for &(cb, lb, _) in MV_VLC_TABLE {
                if la < lb {
                    // ca is candidate prefix of cb: compare top `la` bits.
                    let shift = lb - la;
                    if (cb >> shift) == ca {
                        // Only legitimate if they are literally the same
                        // row, which can't happen for la < lb.
                        panic!("code 0b{ca:b} (len {la}) is a prefix of 0b{cb:b} (len {lb})");
                    }
                }
            }
        }
    }

    #[test]
    fn decode_mv_data_round_trips_every_row() {
        for &(code, len, mv_data) in MV_VLC_TABLE {
            let mut w = BitWriter::new();
            w.write_bits(code as u32, len as usize);
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let got = decode_mv_data(&mut br).unwrap();
            assert_eq!(got, mv_data, "code=0b{code:b} len={len}");
            // The reader should have advanced exactly `len` bits.
            assert_eq!(br.bit_position(), len as usize, "code=0b{code:b}");
        }
    }

    #[test]
    fn decode_mv_data_zero_is_single_one_bit() {
        // mv_data == 0 is the code `1` (one bit).
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        assert_eq!(decode_mv_data(&mut br).unwrap(), 0);
        assert_eq!(br.bit_position(), 1);
    }

    #[test]
    fn decode_mv_data_truncated_empty() {
        let data: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_mv_data(&mut br).unwrap_err(),
            MotionParseError::Truncated
        );
    }

    #[test]
    fn decode_mv_data_invalid_full_window() {
        // 13 bits of all-zero is not a valid codeword (the longest
        // all-prefix-zero codes still end in a 1 within 13 bits).
        let data = [0x00, 0x00];
        let mut br = BitReader::new(&data);
        let err = decode_mv_data(&mut br).unwrap_err();
        assert!(
            matches!(err, MotionParseError::InvalidMvData { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn mv_component_fcode1_is_passthrough() {
        // f == 1 → MVD == mv_data, no residual read.
        for mv_data in [-32i16, -3, 0, 5, 32] {
            let data = [0xFFu8]; // residual bits, must NOT be consumed
            let mut br = BitReader::new(&data);
            let got = decode_mv_component(&mut br, mv_data, 1).unwrap();
            assert_eq!(got, mv_data as i32, "mv_data={mv_data}");
            assert_eq!(br.bit_position(), 0, "no residual should be read");
        }
    }

    #[test]
    fn mv_component_zero_mv_data_skips_residual() {
        // mv_data == 0 → MVD == 0 regardless of fcode, residual skipped.
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        let got = decode_mv_component(&mut br, 0, 4).unwrap();
        assert_eq!(got, 0);
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn mv_component_fcode2_positive() {
        // §7.6.3: fcode=2 → f=2, r_size=1. mv_data=4, residual=1 →
        // (4-1)*2 + 1 + 1 = 8.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // residual = 1 (r_size = 1 bit)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let got = decode_mv_component(&mut br, 4, 2).unwrap();
        assert_eq!(got, 8);
        assert_eq!(br.bit_position(), 1, "exactly r_size=1 bit consumed");
    }

    #[test]
    fn mv_component_fcode2_negative() {
        // mv_data=-4, residual=0 → -((4-1)*2 + 0 + 1) = -7.
        let mut w = BitWriter::new();
        w.write_bits(0b0, 1); // residual = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let got = decode_mv_component(&mut br, -4, 2).unwrap();
        assert_eq!(got, -7);
    }

    #[test]
    fn mv_component_fcode3_residual_width() {
        // fcode=3 → f=4, r_size=2. mv_data=2, residual=3 →
        // (2-1)*4 + 3 + 1 = 8.
        let mut w = BitWriter::new();
        w.write_bits(0b11, 2); // residual = 3 (2 bits)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let got = decode_mv_component(&mut br, 2, 3).unwrap();
        assert_eq!(got, 8);
        assert_eq!(br.bit_position(), 2, "exactly r_size=2 bits consumed");
    }

    #[test]
    fn mv_component_rejects_bad_fcode() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_mv_component(&mut br, 4, 0).unwrap_err(),
            MotionParseError::InvalidFcode(0)
        );
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_mv_component(&mut br, 4, 8).unwrap_err(),
            MotionParseError::InvalidFcode(8)
        );
    }

    #[test]
    fn mv_component_truncated_residual() {
        // fcode=4 → r_size=3 residual bits, but provide none after the
        // (already-decoded) mv_data.
        let data: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_mv_component(&mut br, 5, 4).unwrap_err(),
            MotionParseError::Truncated
        );
    }

    #[test]
    fn decode_motion_vector_forward_fcode1() {
        // fcode=1 → both components are raw mv_data, no residuals.
        // hx code for mv_data=-2 is 0b0011 (4 bits); vy for mv_data=3 is
        // 0b0001_0 (5 bits).
        let mut w = BitWriter::new();
        w.write_bits(0b0011, 4); // hx mv_data = -2
        w.write_bits(0b00010, 5); // vy mv_data = 3
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector(&mut br, MvMode::Forward, 1, 1).unwrap();
        assert_eq!(mvd, MotionVectorDelta { mvdx: -2, mvdy: 3 });
        assert_eq!(br.bit_position(), 9);
    }

    #[test]
    fn decode_motion_vector_forward_fcode2_with_residuals() {
        // fcode=2: hx mv_data=4 (code 0b0000_110 = 7 bits) + residual 1
        // (1 bit) → 8; vy mv_data=-4 (code 0b0011 = 4 bits) + residual 0
        // (1 bit) → -7.
        let mut w = BitWriter::new();
        w.write_bits(0b0000_110, 7); // hx mv_data = 4
        w.write_bits(0b1, 1); // hx residual = 1
        w.write_bits(0b0011, 4); // vy mv_data = -2 (code 0b0011)
        w.write_bits(0b0, 1); // vy residual = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector(&mut br, MvMode::Forward, 2, 1).unwrap();
        // hx: (4-1)*2 + 1 + 1 = 8. vy mv_data=-2: (2-1)*2 + 0 + 1 = 3,
        // negated → -3.
        assert_eq!(mvd, MotionVectorDelta { mvdx: 8, mvdy: -3 });
    }

    #[test]
    fn decode_motion_vector_backward_uses_backward_fcode() {
        // Backward mode uses fcode_backward (here 2); fcode_forward is
        // ignored. hx mv_data=0 (code 0b1) → no residual, MVD=0;
        // vy mv_data=2 (code 0b0010 = 4 bits) + residual=1 →
        // (2-1)*2 + 1 + 1 = 4.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // hx mv_data = 0
        w.write_bits(0b0010, 4); // vy mv_data = 2
        w.write_bits(0b1, 1); // vy residual = 1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector(&mut br, MvMode::Backward, 1, 2).unwrap();
        assert_eq!(mvd, MotionVectorDelta { mvdx: 0, mvdy: 4 });
    }

    #[test]
    fn decode_motion_vector_direct_reads_no_residual() {
        // Direct mode: two raw mv_data VLCs, no residual fields even
        // though we pass a non-1 fcode (it must be ignored).
        let mut w = BitWriter::new();
        w.write_bits(0b011, 3); // hx mv_data = -1
        w.write_bits(0b010, 3); // vy mv_data = 1
                                // Sentinel: if a residual were (wrongly) read, it would consume
                                // these bits.
        w.write_bits(0b1010_1010, 8);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mvd = decode_motion_vector(&mut br, MvMode::Direct, 4, 4).unwrap();
        assert_eq!(mvd, MotionVectorDelta { mvdx: -1, mvdy: 1 });
        // Exactly 6 bits consumed → sentinel intact.
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    #[test]
    fn decode_motion_vector_rejects_bad_forward_fcode() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_motion_vector(&mut br, MvMode::Forward, 0, 1).unwrap_err(),
            MotionParseError::InvalidFcode(0)
        );
    }

    #[test]
    fn decode_motion_vector_rejects_bad_backward_fcode() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_motion_vector(&mut br, MvMode::Backward, 1, 9).unwrap_err(),
            MotionParseError::InvalidFcode(9)
        );
    }

    #[test]
    fn apply_predictor_no_wrap() {
        // fcode=1 → [low,high] = [-32,31]. p=0, mvd=10 → 10.
        assert_eq!(apply_predictor(0, 10, 1).unwrap(), 10);
    }

    #[test]
    fn apply_predictor_wraps_high() {
        // fcode=1: p=30, mvd=5 → 35 > 31 → 35 - 64 = -29.
        assert_eq!(apply_predictor(30, 5, 1).unwrap(), -29);
    }

    #[test]
    fn apply_predictor_wraps_low() {
        // fcode=1: p=-30, mvd=-5 → -35 < -32 → -35 + 64 = 29.
        assert_eq!(apply_predictor(-30, -5, 1).unwrap(), 29);
    }

    #[test]
    fn apply_predictor_fcode2_range() {
        // fcode=2 → [low,high] = [-64,63], range=128. p=60, mvd=10 →
        // 70 > 63 → 70 - 128 = -58.
        assert_eq!(apply_predictor(60, 10, 2).unwrap(), -58);
    }

    #[test]
    fn apply_predictor_rejects_bad_fcode() {
        assert_eq!(
            apply_predictor(0, 0, 0).unwrap_err(),
            MotionParseError::InvalidFcode(0)
        );
    }

    #[test]
    fn error_display_covers_all_variants() {
        let cases = [
            MotionParseError::Truncated,
            MotionParseError::InvalidMvData { window: 0 },
            MotionParseError::InvalidFcode(0),
        ];
        for e in cases {
            assert!(!format!("{e}").is_empty());
        }
    }

    #[test]
    fn bit_reader_error_maps_to_truncated() {
        let e: MotionParseError = BitReaderError::EndOfStream.into();
        assert_eq!(e, MotionParseError::Truncated);
    }
}

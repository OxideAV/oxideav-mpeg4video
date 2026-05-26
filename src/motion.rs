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
//! * §7.6.5 Vector decoding processing in progressive P-/S(GMC)-VOP —
//!   the median-filter predictor. The three candidate predictors
//!   (`MV1`, `MV2`, `MV3`) are resolved by the four validity rules of
//!   §7.6.5 and combined component-wise with `Px = Median(MV1x, MV2x,
//!   MV3x)` / `Py = Median(MV1y, MV2y, MV3y)`. The worked example in
//!   the spec — `MV1=(-2,3)`, `MV2=(1,5)`, `MV3=(-1,7)` → `Px=-1`,
//!   `Py=5` — pins the `Median(a, b, c)` definition as the middle of
//!   three (the §4.1 operator clause does not list `Median`). The
//!   resolved `(Px, Py)` feeds straight into
//!   [`reconstruct_motion_vector`].
//!
//! ## Out of scope (this round)
//!
//! * **Gathering** the candidate predictors from the spatial
//!   neighbourhood (the `MV1`/`MV2`/`MV3` block positions of
//!   Figure 7-34, the four-MV vs single-MV cases, and the
//!   S(GMC)-VOP `mcsel == '1'` averaged-vector substitution of
//!   §7.8.7.3). Figure 7-34 is a diagram with no textual position list
//!   in the spec, so the spatial layout is a later round; this module
//!   resolves and medians candidates the caller has already gathered,
//!   marking transparent / out-of-VOP / out-of-packet neighbours as
//!   `None`.
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

// ---------------------------------------------------------------------------
// §7.6.5 — median-filter motion-vector predictor (progressive P-/S(GMC)-VOP).
// ---------------------------------------------------------------------------

/// `Median(a, b, c)` — the middle value of three integers (§7.6.5).
///
/// The §4.1 arithmetic-operator clause does not define `Median`; the
/// §7.6.5 worked example (`Median(-2, 1, -1) = -1`,
/// `Median(3, 5, 7) = 5`) fixes it as the order statistic that discards
/// the smallest and the largest of the three.
fn median3(a: i32, b: i32, c: i32) -> i32 {
    // max(min(a,b), min(max(a,b), c)) — the middle of three.
    a.max(b).min(a.min(b).max(c))
}

/// Resolve the three §7.6.5 candidate predictors into concrete vectors.
///
/// Each input slot is `Some(vector)` when the corresponding spatial
/// neighbour block vector is **valid** (the neighbour exists, is
/// non-transparent, and is inside the current VOP / video packet /
/// GOB), or `None` when it is **not valid** (transparent neighbour,
/// transparent block of the current MB, or a neighbour outside the
/// current VOP / video packet / GOB — all "treated as transparent" per
/// the §7.6.5 note).
///
/// The four §7.6.5 decision rules are then applied:
///
/// 1. A valid candidate keeps its block vector.
/// 2. If exactly **one** candidate is not valid, it is set to zero.
/// 3. If exactly **two** candidates are not valid, they are set to the
///    one remaining valid candidate.
/// 4. If **all three** candidates are not valid, they are set to zero.
///
/// Returns the resolved `[MV1, MV2, MV3]` triple.
fn resolve_candidates(candidates: [Option<MotionVector>; 3]) -> [MotionVector; 3] {
    let zero = MotionVector { x: 0, y: 0 };
    let valid_count = candidates.iter().filter(|c| c.is_some()).count();

    match valid_count {
        // Rule 4: all three invalid → all zero.
        0 => [zero, zero, zero],
        // Rule 3: exactly two invalid → the invalid pair takes the one
        // remaining valid candidate.
        1 => {
            let only = candidates
                .iter()
                .find_map(|c| *c)
                .expect("valid_count == 1 implies a Some");
            [only, only, only]
        }
        // Rule 2: exactly one invalid → that one becomes zero; the two
        // valid candidates keep their vectors. (Rule 1 for the valid
        // ones is the `unwrap_or(zero)` identity here.)
        // Rule 1 (all valid, valid_count == 3) also flows through this
        // arm: every slot is `Some`, so each keeps its block vector.
        _ => [
            candidates[0].unwrap_or(zero),
            candidates[1].unwrap_or(zero),
            candidates[2].unwrap_or(zero),
        ],
    }
}

/// Compute the §7.6.5 motion-vector predictor `(Px, Py)` from the three
/// spatial candidate predictors.
///
/// The candidates are first resolved by the four §7.6.5 validity rules
/// (see [`resolve_candidates`] — `None` marks an invalid / transparent
/// neighbour), then combined component-wise:
///
/// ```text
/// Px = Median(MV1x, MV2x, MV3x)
/// Py = Median(MV1y, MV2y, MV3y)
/// ```
///
/// The returned vector is the `(Px, Py)` predictor that
/// [`reconstruct_motion_vector`] adds to the decoded differential MV.
/// Gathering the candidates from the spatial neighbourhood
/// (Figure 7-34 positions) is the caller's responsibility and out of
/// scope for this module.
pub fn predict_motion_vector(candidates: [Option<MotionVector>; 3]) -> MotionVector {
    let [mv1, mv2, mv3] = resolve_candidates(candidates);
    MotionVector {
        x: median3(mv1.x, mv2.x, mv3.x),
        y: median3(mv1.y, mv2.y, mv3.y),
    }
}

// ---------------------------------------------------------------------------
// §7.8.7.3 — S(GMC)-VOP averaged-vector substitution.
//
// Macroblocks with `mcsel == 1` do not carry their own block motion vector;
// pel-wise motion vectors are produced by sprite warping (§7.8.5). When such
// a macroblock is referenced as a neighbour for the §7.6.5 median predictor
// (or as the co-located reference for B-VOP direct mode, §7.6.6), the spec
// substitutes the per-pixel-averaged motion vector
//
//     AMVx = (Σ MVx(x,y)) // Nb           AMVy = (Σ MVy(x,y)) // Nb
//
// with `Nb = 256` (all 16×16 luminance pixels). The averaged value is then
// quantised to the target sub-pel grid (half-pel when `quarter_sample == 0`,
// quarter-pel when `quarter_sample == 1`) and clipped to the Table 7-9
// `[low:high]` range for the supplied `vop_fcode`.
//
// The spec's `//` operator (§3.4) is integer division with rounding to the
// nearest integer; half-integer values are rounded **away from zero**. The
// pel-wise input MVs arrive in a caller-defined fixed-point unit; the
// `pel_denominator` parameter tells this function what fraction of a pel one
// integer step represents.
// ---------------------------------------------------------------------------

/// The §7.8.7.3 luminance-block pixel count (`Nb = 256`, all 16×16 pixels
/// of one macroblock).
pub const AMV_PIXEL_COUNT: usize = 256;

/// Integer division with rounding to the nearest integer, ties **away
/// from zero** (the spec's `//` operator, §3.4). `denom` must be positive.
fn rdiv_away(num: i64, denom: i64) -> i64 {
    debug_assert!(denom > 0);
    let half = denom / 2;
    if num >= 0 {
        (num + half) / denom
    } else {
        -((-num + half) / denom)
    }
}

/// Compute the §7.8.7.3 averaged motion vector for one S(GMC)-VOP
/// `mcsel == 1` macroblock and quantise it to the target sub-pel grid.
///
/// `pel_mvs_x` / `pel_mvs_y` are the 16×16 = 256 luminance pel-wise motion
/// vectors produced by sprite warping (§7.8.5), each measured in
/// `1 / pel_denominator` of a pel. `pel_denominator` is the caller's
/// fixed-point grid (e.g. `16` for sixteenth-pel sprite warping); it must
/// be a multiple of `2` when `quarter_sample == 0` and a multiple of `4`
/// when `quarter_sample == 1`, so the AMV expressed in the output unit is
/// representable by the spec's `//` rounding rule.
///
/// The returned [`MotionVector`] is the candidate predictor in
/// **half-pel units** when `quarter_sample == false` or **quarter-pel
/// units** when `quarter_sample == true` — the same unit the rest of
/// this module uses for `MotionVector` values, and the unit the
/// [`fcode_bounds`] `[low:high]` range applies to. Out-of-range values
/// are clipped per the §7.8.7.3 final sentence ("If the quantised AMV
/// is outside the motion vector range specified by f_code, it is
/// clipped in the range").
///
/// Returns [`MotionParseError::InvalidFcode`] when `vop_fcode` is
/// outside `1..=7` and [`MotionParseError::Truncated`] (re-used as a
/// generic "bad-input" sentinel) when `pel_denominator` is zero or not
/// a multiple of the required grid step.
pub fn averaged_motion_vector(
    pel_mvs_x: &[i64; AMV_PIXEL_COUNT],
    pel_mvs_y: &[i64; AMV_PIXEL_COUNT],
    pel_denominator: u32,
    quarter_sample: bool,
    vop_fcode: u8,
) -> Result<MotionVector, MotionParseError> {
    let (_f, low, high, _range) = fcode_bounds(vop_fcode)?;
    if pel_denominator == 0 {
        return Err(MotionParseError::Truncated);
    }
    // Output unit: half-pel (denom 2) or quarter-pel (denom 4). The caller
    // must supply a `pel_denominator` whose product with `Nb = 256` and
    // whose ratio with the output unit yield exact integer arithmetic; this
    // is the §7.8.7.3 quantisation precondition.
    let out_unit: u32 = if quarter_sample { 4 } else { 2 };
    if pel_denominator % out_unit != 0 {
        return Err(MotionParseError::Truncated);
    }
    // Real AMV = sum / (Nb * pel_denominator) pels.
    // Result in `1/out_unit`-pel units = (sum * out_unit) // (Nb * pel_denominator).
    let nb = AMV_PIXEL_COUNT as i64;
    let denom_out: i64 = nb * i64::from(pel_denominator / out_unit);
    let sum_x: i64 = pel_mvs_x.iter().copied().sum();
    let sum_y: i64 = pel_mvs_y.iter().copied().sum();
    let mut amv_x = rdiv_away(sum_x, denom_out);
    let mut amv_y = rdiv_away(sum_y, denom_out);
    let low = i64::from(low);
    let high = i64::from(high);
    amv_x = amv_x.clamp(low, high);
    amv_y = amv_y.clamp(low, high);
    Ok(MotionVector {
        x: amv_x as i32,
        y: amv_y as i32,
    })
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

    // ----- §7.6.5 median-filter predictor --------------------------------

    #[test]
    fn median3_picks_middle_value() {
        // The §7.6.5 worked example components.
        assert_eq!(median3(-2, 1, -1), -1);
        assert_eq!(median3(3, 5, 7), 5);
        // Order-independence: the median is invariant under permutation.
        assert_eq!(median3(7, 3, 5), 5);
        assert_eq!(median3(5, 7, 3), 5);
        assert_eq!(median3(1, -2, -1), -1);
        // Duplicates: the repeated value is the middle.
        assert_eq!(median3(4, 4, 9), 4);
        assert_eq!(median3(9, 4, 4), 4);
        assert_eq!(median3(2, 2, 2), 2);
        // Negative-spanning triple.
        assert_eq!(median3(-10, 0, 10), 0);
        assert_eq!(median3(-5, -5, -1), -5);
    }

    #[test]
    fn predict_matches_spec_worked_example() {
        // §7.6.5: MV1=(-2,3), MV2=(1,5), MV3=(-1,7) → Px=-1, Py=5.
        let candidates = [
            Some(MotionVector { x: -2, y: 3 }),
            Some(MotionVector { x: 1, y: 5 }),
            Some(MotionVector { x: -1, y: 7 }),
        ];
        let p = predict_motion_vector(candidates);
        assert_eq!(p.x, -1);
        assert_eq!(p.y, 5);
    }

    #[test]
    fn rule1_all_valid_medians_each_component() {
        // All three valid: each component is its own median.
        let resolved = resolve_candidates([
            Some(MotionVector { x: -2, y: 3 }),
            Some(MotionVector { x: 1, y: 5 }),
            Some(MotionVector { x: -1, y: 7 }),
        ]);
        assert_eq!(resolved[0], MotionVector { x: -2, y: 3 });
        assert_eq!(resolved[1], MotionVector { x: 1, y: 5 });
        assert_eq!(resolved[2], MotionVector { x: -1, y: 7 });
    }

    #[test]
    fn rule2_one_invalid_becomes_zero() {
        // Exactly one candidate invalid → it is set to zero; the two
        // valid candidates keep their vectors.
        let resolved = resolve_candidates([
            Some(MotionVector { x: 4, y: -4 }),
            None,
            Some(MotionVector { x: -6, y: 6 }),
        ]);
        assert_eq!(resolved[0], MotionVector { x: 4, y: -4 });
        assert_eq!(resolved[1], MotionVector { x: 0, y: 0 });
        assert_eq!(resolved[2], MotionVector { x: -6, y: 6 });
        // Predictor: Px = Median(4, 0, -6) = 0; Py = Median(-4, 0, 6) = 0.
        let p = predict_motion_vector([
            Some(MotionVector { x: 4, y: -4 }),
            None,
            Some(MotionVector { x: -6, y: 6 }),
        ]);
        assert_eq!(p.x, 0);
        assert_eq!(p.y, 0);
    }

    #[test]
    fn rule3_two_invalid_take_third() {
        // Exactly two invalid → both set to the one remaining valid
        // candidate. The predictor then equals that candidate (the
        // median of three identical values).
        for slot in 0..3 {
            let mut c = [None, None, None];
            let only = MotionVector { x: 7, y: -9 };
            c[slot] = Some(only);
            let resolved = resolve_candidates(c);
            assert_eq!(resolved, [only, only, only], "valid in slot {slot}");
            let p = predict_motion_vector(c);
            assert_eq!(p, only, "predictor equals the sole valid candidate");
        }
    }

    #[test]
    fn rule4_all_invalid_is_zero() {
        let resolved = resolve_candidates([None, None, None]);
        let zero = MotionVector { x: 0, y: 0 };
        assert_eq!(resolved, [zero, zero, zero]);
        let p = predict_motion_vector([None, None, None]);
        assert_eq!(p, zero);
    }

    #[test]
    fn predict_feeds_reconstruct_motion_vector() {
        // End-to-end: median predictor → §7.6.3 add + wrap. With
        // candidates (-2,3),(1,5),(-1,7) the predictor is (-1,5);
        // adding a delta of (4,-3) under fcode 2 gives (3,2), in range.
        let candidates = [
            Some(MotionVector { x: -2, y: 3 }),
            Some(MotionVector { x: 1, y: 5 }),
            Some(MotionVector { x: -1, y: 7 }),
        ];
        let p = predict_motion_vector(candidates);
        let delta = MotionVectorDelta { dx: 4, dy: -3 };
        let mv = reconstruct_motion_vector(delta, p.x, p.y, 2).unwrap();
        assert_eq!(mv.x, 3);
        assert_eq!(mv.y, 2);
    }

    // ----- §7.8.7.3 averaged-vector substitution -----------------------

    /// Build a constant pel-wise grid of `(vx, vy)` repeated 256 times.
    fn flat_pel_grid(vx: i64, vy: i64) -> ([i64; AMV_PIXEL_COUNT], [i64; AMV_PIXEL_COUNT]) {
        ([vx; AMV_PIXEL_COUNT], [vy; AMV_PIXEL_COUNT])
    }

    #[test]
    fn amv_rdiv_away_zero_and_positive_half() {
        // Spec §3.4: `//` rounds half away from zero. 3//2 = 2, -3//2 = -2.
        assert_eq!(rdiv_away(3, 2), 2);
        assert_eq!(rdiv_away(-3, 2), -2);
        assert_eq!(rdiv_away(0, 256), 0);
        assert_eq!(rdiv_away(128, 256), 1); // exact half rounds up
        assert_eq!(rdiv_away(-128, 256), -1); // exact half rounds down
        assert_eq!(rdiv_away(127, 256), 0); // just below half
    }

    #[test]
    fn amv_invalid_fcode_rejected() {
        let (x, y) = flat_pel_grid(0, 0);
        assert_eq!(
            averaged_motion_vector(&x, &y, 16, false, 0),
            Err(MotionParseError::InvalidFcode(0))
        );
        assert_eq!(
            averaged_motion_vector(&x, &y, 16, false, 8),
            Err(MotionParseError::InvalidFcode(8))
        );
    }

    #[test]
    fn amv_zero_pel_denominator_rejected() {
        let (x, y) = flat_pel_grid(0, 0);
        assert_eq!(
            averaged_motion_vector(&x, &y, 0, false, 1),
            Err(MotionParseError::Truncated)
        );
    }

    #[test]
    fn amv_pel_denominator_must_match_output_grid() {
        let (x, y) = flat_pel_grid(0, 0);
        // half-pel output needs `pel_denominator` divisible by 2; 3 fails.
        assert_eq!(
            averaged_motion_vector(&x, &y, 3, false, 1),
            Err(MotionParseError::Truncated)
        );
        // quarter-pel output needs `pel_denominator` divisible by 4; 2 fails.
        assert_eq!(
            averaged_motion_vector(&x, &y, 2, true, 1),
            Err(MotionParseError::Truncated)
        );
    }

    #[test]
    fn amv_flat_grid_round_trips_constant_vector() {
        // 256 copies of (x=32 sixteenths, y=-48 sixteenths) = (2.0, -3.0) pels.
        // Sum_x = 8192, Sum_y = -12288, denom_out (half-pel) = 256 * (16/2) = 2048.
        // AMV_half_x = 8192 // 2048 = 4 (= 2.0 pels), AMV_half_y = -12288 // 2048 = -6 (= -3.0 pels).
        let (x, y) = flat_pel_grid(32, -48);
        let amv = averaged_motion_vector(&x, &y, 16, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 4, y: -6 });
    }

    #[test]
    fn amv_flat_grid_quarter_sample_grid() {
        // Same input but quarter-sample output. denom_out = 256 * (16/4) = 1024.
        // AMV_q_x = 8192 // 1024 = 8 (= 2.0 pels), AMV_q_y = -12288 // 1024 = -12.
        let (x, y) = flat_pel_grid(32, -48);
        let amv = averaged_motion_vector(&x, &y, 16, true, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 8, y: -12 });
    }

    #[test]
    fn amv_quantises_quarter_pel_bin_low_edge_to_zero() {
        // Real value 0.24 pel → quarter-sample bin [0.125, 0.375) → 0.25.
        // Wait: 0.24 is in [0.125, 0.375) so it rounds to 0.25 = 1 quarter-pel.
        // Encode 0.24 pel as 256 copies of value 4 in 16ths (= 0.25 pel exactly is 4).
        // Use value 1 in 16ths (= 0.0625 pel), in [0, 0.125) → 0.
        // sum = 256, denom_out_q = 1024 → 256 // 1024 = 0.
        let (x, y) = flat_pel_grid(1, 0);
        let amv = averaged_motion_vector(&x, &y, 16, true, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn amv_quantises_quarter_pel_bin_inside_to_quarter() {
        // value 3 in 16ths = 0.1875 pel, in [0.125, 0.375) → 0.25 = 1 quarter-pel.
        // sum = 256 * 3 = 768. denom_out_q = 1024. 768 // 1024 = 1 (= 0.25 pel).
        let (x, y) = flat_pel_grid(3, 0);
        let amv = averaged_motion_vector(&x, &y, 16, true, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 1, y: 0 });
    }

    #[test]
    fn amv_quantises_half_pel_bin_low_edge() {
        // Real value 0.24 pel → half-sample bin [0, 0.25) → 0.
        // value 3 in 16ths = 0.1875 pel, in [0, 0.25) → 0 half-pels.
        // sum = 768. denom_out_h = 256 * 8 = 2048. 768 // 2048 = 0.
        let (x, y) = flat_pel_grid(3, 0);
        let amv = averaged_motion_vector(&x, &y, 16, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn amv_quantises_half_pel_bin_boundary_inclusive() {
        // Real value exactly 0.25 pel → half-sample bin [0.25, 0.75) → 0.5.
        // value 4 in 16ths = 0.25 pel. sum = 1024. 1024 // 2048 = 1 (half-pel = 0.5).
        let (x, y) = flat_pel_grid(4, 0);
        let amv = averaged_motion_vector(&x, &y, 16, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 1, y: 0 });
    }

    #[test]
    fn amv_quantises_half_pel_unit_inclusive_endpoint() {
        // Real value exactly 1.0 pel → half-sample bin [0.75, 1.0] → 1.0.
        // value 16 in 16ths. sum = 4096. 4096 // 2048 = 2 (half-pel = 1.0).
        let (x, y) = flat_pel_grid(16, 0);
        let amv = averaged_motion_vector(&x, &y, 16, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 2, y: 0 });
    }

    #[test]
    fn amv_negative_mirrors_positive_quantisation() {
        // Negative version of the 0.5-pel boundary: value -4 in 16ths.
        // sum = -1024 → -1024 // 2048 = -1 (since half rounds away from zero).
        let (x, y) = flat_pel_grid(-4, -16);
        let amv = averaged_motion_vector(&x, &y, 16, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: -1, y: -2 });
    }

    #[test]
    fn amv_clipped_to_fcode_range_high() {
        // fcode 1 → [-32, 31] half-pels. Pel grid 16 in 16ths = 1.0 pel
        // = 2 half-pels — well within range. Push beyond: 256 in 16ths
        // = 16.0 pels = 32 half-pels — exceeds high (31), clip to 31.
        let (x, y) = flat_pel_grid(256, 0);
        let amv = averaged_motion_vector(&x, &y, 16, false, 1).unwrap();
        assert_eq!(amv, MotionVector { x: 31, y: 0 });
    }

    #[test]
    fn amv_clipped_to_fcode_range_low() {
        // fcode 1 → [-32, 31]. Value -257 in 16ths = -16.0625 pel
        // = -32.125 half-pels → rounds to -32 (no clip yet, exactly at low).
        // Push further: -300 in 16ths → sum = -76800; /2048 = -37.5 → -38 → clip to -32.
        let (x, y) = flat_pel_grid(-300, 0);
        let amv = averaged_motion_vector(&x, &y, 16, false, 1).unwrap();
        assert_eq!(amv, MotionVector { x: -32, y: 0 });
    }

    #[test]
    fn amv_mixed_pixel_grid_sums_correctly() {
        // Half the macroblock at vx=8 (16ths), half at vx=-8.
        // sum_x = 128*8 + 128*(-8) = 0 → AMV_x = 0.
        let mut x = [0i64; AMV_PIXEL_COUNT];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = if i < 128 { 8 } else { -8 };
        }
        let y = [0i64; AMV_PIXEL_COUNT];
        let amv = averaged_motion_vector(&x, &y, 16, true, 5).unwrap();
        assert_eq!(amv, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn amv_eighth_pel_denominator_quarter_sample() {
        // pel_denominator = 8 (eighth-pel input) with quarter-sample output.
        // 8 % 4 == 0 → accepted. value 2 in eighths = 0.25 pel exactly.
        // sum = 512. denom_out_q = 256 * (8/4) = 512. AMV = 1 quarter-pel.
        let (x, y) = flat_pel_grid(2, 0);
        let amv = averaged_motion_vector(&x, &y, 8, true, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 1, y: 0 });
    }

    #[test]
    fn amv_pixel_count_constant_matches_spec() {
        // §7.8.7.3 note: "Nb is always 256 into the following expression".
        assert_eq!(AMV_PIXEL_COUNT, 256);
    }
}

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
//!   neighbourhood is owned by [`crate::mv_predictor_grid`] (the
//!   `MV1`/`MV2`/`MV3` block positions of Figure 7-34 and the four-MV
//!   vs single-MV cases); this module resolves and medians candidates
//!   the caller has already gathered, marking transparent / out-of-VOP
//!   / out-of-packet neighbours as `None`. The S(GMC)-VOP
//!   `mcsel == '1'` averaged-vector substitution of §7.8.7.3 is
//!   handled by [`averaged_motion_vector`].
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
    /// `field_prediction == 1` was supplied in a context the §6.2.6 /
    /// §6.2.6.3 syntax cannot produce — a direct-mode body (the
    /// §6.2.6.3 `mb_type != "1"` clause keeps the bit from ever being
    /// coded for a direct macroblock and the §6.2.6 direct line has no
    /// second invocation), a four-MV `inter4v` macroblock, or an intra
    /// macroblock (the §6.2.6.3 outer guard requires
    /// `derived_mb_type < 2`).
    InvalidFieldPredictionContext,
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
            MotionParseError::InvalidFieldPredictionContext => {
                write!(
                    f,
                    "field_prediction set in a context §6.2.6/§6.2.6.3 cannot produce"
                )
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
/// Gather the candidates from the spatial neighbourhood via
/// [`crate::mv_predictor_grid::MvGrid::predictor_candidates`].
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
/// fixed-point grid (`1` for integer pel, `2` for half-pel input,
/// `4` for quarter-pel input, `16` for the sixteenth-pel grid used by
/// the §7.8.5 sub-pel warping pipeline, etc.) and must be non-zero;
/// any positive value yields well-defined integer arithmetic via the
/// spec's `//` rounding.
///
/// The returned [`MotionVector`] is the candidate predictor in
/// **half-pel units** when `quarter_sample == false` or **quarter-pel
/// units** when `quarter_sample == true` — the same unit the rest of
/// this module uses for `MotionVector` values, and the unit the
/// Table 7-9 `[low:high]` range applies to. Out-of-range values are
/// clipped per the §7.8.7.3 final sentence ("If the quantised AMV is
/// outside the motion vector range specified by f_code, it is clipped
/// in the range").
///
/// Returns [`MotionParseError::InvalidFcode`] when `vop_fcode` is
/// outside `1..=7` and [`MotionParseError::Truncated`] (re-used as a
/// generic "bad-input" sentinel) when `pel_denominator` is zero.
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
    // Real AMV = sum / (Nb * pel_denominator) pels.
    // Output integer in `1 / out_unit`-pel units is
    //     (out_unit * sum) // (Nb * pel_denominator)
    // with `//` the §3.4 round-half-away-from-zero division.
    let nb = AMV_PIXEL_COUNT as i64;
    let out_unit: i64 = if quarter_sample { 4 } else { 2 };
    let denom_out: i64 = nb * i64::from(pel_denominator);
    let sum_x: i64 = pel_mvs_x.iter().copied().sum();
    let sum_y: i64 = pel_mvs_y.iter().copied().sum();
    let mut amv_x = rdiv_away(out_unit * sum_x, denom_out);
    let mut amv_y = rdiv_away(out_unit * sum_y, denom_out);
    let low = i64::from(low);
    let high = i64::from(high);
    amv_x = amv_x.clamp(low, high);
    amv_y = amv_y.clamp(low, high);
    Ok(MotionVector {
        x: amv_x as i32,
        y: amv_y as i32,
    })
}

// ---------------------------------------------------------------------------
// §7.6.9.5.2 — Direct-mode forward + backward motion-vector derivation
// (ISO/IEC 14496-2:2004 third edition).
//
// A direct-mode B-VOP macroblock does not carry its own forward / backward
// MV pair on the wire. Instead, the spec linearly scales the motion vector
// `MV` of the co-located macroblock in the temporally next anchor VOP
// (the most recent I-, P-, or S(GMC)-VOP) by the §7.6.7 temporal-reference
// difference `TRB / TRD`, then applies a single delta vector `MVD`
// (decoded via `decode_motion_vector_delta` with `MvMode::Direct`):
//
//     MVF = (TRB * MV) / TRD + MVD
//     MVB = (MVD == 0) ? ((TRB - TRD) * MV) / TRD
//                      : MVF - MV
//
// where the division `/` is the §3.4 truncation-toward-zero integer
// division (matching Rust's `i32::Div` semantics on signed operands).
//
// `MV` is the co-located block vector after §7.6.1.6 vector padding,
// applied independently to each of the four luminance blocks
// `i = 0,1,2,3` (Figure 6-8 sub-block order). When the co-located
// macroblock is part of an S(GMC)-VOP with `mcsel == 1` the spec
// substitutes the §7.8.7.3 averaged motion vector for `MV` (the caller
// already produces that via `averaged_motion_vector`). When the
// co-located macroblock is transparent or the slot is otherwise
// unavailable, the spec falls back to `MV = (0, 0)` and direct mode
// stays enabled (§7.6.9.5.1 final sentence).
//
// `TRB` is the temporal-reference difference between the current B-VOP
// and the previous (forward) anchor VOP; `TRD` is the temporal-reference
// difference between the temporally next (backward) anchor and the
// previous anchor — `TRD` is therefore the full inter-anchor span and
// is strictly positive whenever direct mode applies (the §7.6.7
// numbering convention places consecutive B-VOPs strictly between two
// anchors, so the next anchor's `modulo_time_base`-extended TR is
// strictly greater than the previous anchor's).
//
// Quarter-sample handling (§7.6.9.5.2 fourth paragraph): when `MV` is
// in quarter-pel units and `MVD` is in half-pel units, `MV` is first
// halved component-wise and rounded to the nearest half-pel position
// via Table 7-13 — exactly the same reduction
// `quarter_sample::reduce_qpel_to_half_pel_chroma` performs for
// chrominance MV reduction. The conversion happens **before** the
// `TRB * MV / TRD` multiplication so the entire formula runs in a
// single, consistent sub-pel grid.
//
// This module does not gather `MV` from the reference VOP's macroblock
// grid (that bookkeeping is the caller's responsibility, alongside the
// §7.6.1.6 vector padding); it expects a resolved `MV` and produces the
// `(MVF, MVB)` pair the §7.6.9.5.3 prediction-block generator consumes.
// ---------------------------------------------------------------------------

/// Whether the supplied co-located MV is already on the same sub-pel
/// grid as the delta MV, or needs the §7.6.9.5.2 quarter-to-half-pel
/// conversion described in the fourth paragraph of the subclause.
///
/// `Match` covers the two homogeneous cases (`MV` and `MVD` both in
/// half-pel units when `quarter_sample == 0`, or both in quarter-pel
/// units when `quarter_sample == 1`). `QpelMvToHalfPel` covers the
/// fourth-paragraph mismatch ("MV components of the co-located macroblock
/// are given in quarter sample units and the components MVDx and MVDy of
/// the delta vector are given in half sample units"), in which case
/// `MV` is divided by 2 and rounded via Table 7-13 before the linear
/// scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectMvUnits {
    /// The co-located MV and the delta MV share the same sub-pel grid;
    /// no pre-scaling conversion is needed.
    Match,
    /// The co-located MV is in quarter-pel units but the delta MV is
    /// in half-pel units; halve `MV` componentwise via Table 7-13
    /// before the `TRB * MV / TRD` linear scaling.
    QpelMvToHalfPel,
}

/// Co-located reference-MV state for the §7.6.9.5.2 direct-mode
/// derivation.
///
/// `Mv` carries the resolved block vector after the §7.6.1.6 vector
/// padding step (or the §7.8.7.3 averaged MV for the
/// co-located-`mcsel == 1` S(GMC)-VOP case — the caller decides which
/// substitution applies and passes the resolved vector in).
///
/// `TransparentOrAbsent` covers the §7.6.9.5.1 final-sentence fallback:
/// "If the co-located macroblock is transparent and thus the MVs are
/// not available, the direct mode is still enabled by setting MV
/// vectors to zero vectors." The caller uses this variant when the
/// reference grid slot is transparent / out-of-bounds / unavailable;
/// the derivation then runs with `MV = (0, 0)`, the delta MVD passing
/// through unchanged into both `MVF` and `MVB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectCoLocatedMv {
    /// A resolved (and §7.6.1.6-padded) reference-block motion vector.
    Mv(MotionVector),
    /// The co-located block is transparent or otherwise unavailable;
    /// substitute `MV = (0, 0)` per §7.6.9.5.1's final sentence.
    TransparentOrAbsent,
}

/// Errors specific to the §7.6.9.5.2 direct-mode MV derivation.
///
/// Generic motion-vector parse errors (out-of-range `vop_fcode`,
/// truncated bit stream) keep flowing through [`MotionParseError`];
/// `DirectMvError` extends the surface with the temporal-reference
/// preconditions §7.6.7 imposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectMvError {
    /// `trd` was zero (or non-positive). §7.6.7 places the next
    /// anchor strictly after the previous anchor whenever direct mode
    /// applies, so a `TRD` of zero indicates a stream-level error
    /// (two consecutive anchors with identical extended TR).
    InvalidTrd(i32),
    /// `trb` was outside `0..=trd`. The B-VOP lies temporally between
    /// the two anchors, so `0 <= trb <= trd` always holds — a
    /// violation indicates a malformed temporal-reference sequence.
    TrbOutOfRange {
        /// The supplied `TRB` value.
        trb: i32,
        /// The supplied `TRD` value (the upper bound for `TRB`).
        trd: i32,
    },
}

impl core::fmt::Display for DirectMvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DirectMvError::InvalidTrd(v) => write!(
                f,
                "direct-mode TRD must be positive (§7.6.7), got TRD = {v}"
            ),
            DirectMvError::TrbOutOfRange { trb, trd } => write!(
                f,
                "direct-mode TRB must satisfy 0 <= TRB <= TRD, got TRB = {trb}, TRD = {trd}"
            ),
        }
    }
}

impl std::error::Error for DirectMvError {}

/// Resolved §7.6.9.5.2 direct-mode (`MVF`, `MVB`) pair for one
/// luminance sub-block.
///
/// `forward` is the forward motion vector `MVF` (predicted from the
/// previous anchor); `backward` is `MVB` (predicted from the temporally
/// next anchor). Units match the input delta MV's units (half-pel when
/// `quarter_sample == 0`, quarter-pel when `quarter_sample == 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectModeMv {
    /// The forward motion vector `MVF` per §7.6.9.5.2.
    pub forward: MotionVector,
    /// The backward motion vector `MVB` per §7.6.9.5.2.
    pub backward: MotionVector,
}

/// §7.6.9.5.2 fourth-paragraph quarter→half-pel reduction.
///
/// Halves each component of `mv` (truncating toward `-∞` to match the
/// spec's `/ 2` on a quarter-pel grid) and applies Table 7-13's
/// fractional rounding via
/// [`crate::quarter_sample::reduce_qpel_to_half_pel_chroma`]. Public
/// so callers can pre-apply the reduction once per macroblock (e.g.
/// when the reference is four block-vectors and the delta `MVD` is
/// half-pel) and pass an already-converted `MV` directly into
/// [`direct_mode_motion_vector`] with [`DirectMvUnits::Match`].
pub fn direct_mode_reduce_qpel_to_half_pel(mv: MotionVector) -> MotionVector {
    use crate::quarter_sample::reduce_qpel_to_half_pel_chroma;
    MotionVector {
        x: reduce_qpel_to_half_pel_chroma(mv.x),
        y: reduce_qpel_to_half_pel_chroma(mv.y),
    }
}

/// Derive the §7.6.9.5.2 direct-mode forward + backward motion vectors
/// for one luminance sub-block.
///
/// `co_located` is the resolved reference-MV state (a §7.6.1.6-padded
/// block vector, the §7.8.7.3 averaged MV, or the transparent/absent
/// zero-MV fallback). `mvd` is the single shared delta motion vector
/// decoded via [`decode_motion_vector_delta`] with [`MvMode::Direct`].
/// `trb` and `trd` are the §7.6.7 temporal-reference distances
/// (`TRB`: current B-VOP to previous anchor; `TRD`: temporally next
/// anchor to previous anchor — see §7.6.9.5.2 paragraph 3).
///
/// `units` selects the §7.6.9.5.2 fourth-paragraph quarter→half-pel
/// reduction. Pass [`DirectMvUnits::Match`] when both `MV` and `MVD`
/// share a sub-pel grid; pass [`DirectMvUnits::QpelMvToHalfPel`] when
/// the bitstream is `quarter_sample == 1` but the delta MV is
/// half-pel.
///
/// The result is **not** clipped to the Table 7-9 `[low:high]` range —
/// the §7.6.9.5.2 formulas operate algebraically and rely on the
/// linear-scaling factor `TRB / TRD ∈ [0, 1]` to keep the magnitude
/// bounded relative to `MV`; the prediction-block-generator step that
/// follows (§7.6.9.5.3) consumes the algebraic value directly.
pub fn direct_mode_motion_vector(
    co_located: DirectCoLocatedMv,
    mvd: MotionVectorDelta,
    trb: i32,
    trd: i32,
    units: DirectMvUnits,
) -> Result<DirectModeMv, DirectMvError> {
    if trd <= 0 {
        return Err(DirectMvError::InvalidTrd(trd));
    }
    if trb < 0 || trb > trd {
        return Err(DirectMvError::TrbOutOfRange { trb, trd });
    }

    // §7.6.9.5.1 final sentence: transparent / absent co-located block
    // → MV = (0, 0). Direct mode stays enabled; delta passes through.
    let mv = match co_located {
        DirectCoLocatedMv::Mv(v) => v,
        DirectCoLocatedMv::TransparentOrAbsent => MotionVector { x: 0, y: 0 },
    };

    // §7.6.9.5.2 fourth-paragraph quarter→half-pel reduction: when the
    // co-located MV is in quarter-pel units and the delta is in
    // half-pel units, MV is divided by 2 and rounded via Table 7-13
    // before the linear scaling. The zero MV from the transparent-or-
    // absent fallback is invariant under the reduction.
    let mv = match units {
        DirectMvUnits::Match => mv,
        DirectMvUnits::QpelMvToHalfPel => direct_mode_reduce_qpel_to_half_pel(mv),
    };

    // The §7.6.9.5.2 linear-scaling formulas, in 64-bit signed
    // arithmetic to keep the intermediate `TRB * MV` product safe for
    // any combination of `vop_fcode <= 7` magnitudes and the largest
    // permissible TR difference. The §3.4 `/` operator is integer
    // division with truncation toward zero, matching Rust's `i32::Div`
    // (and therefore `i64::Div`) on signed operands.
    let trb_i = i64::from(trb);
    let trd_i = i64::from(trd);
    let mvx = i64::from(mv.x);
    let mvy = i64::from(mv.y);
    let mvdx = i64::from(mvd.dx);
    let mvdy = i64::from(mvd.dy);

    let mvfx = (trb_i * mvx) / trd_i + mvdx;
    let mvfy = (trb_i * mvy) / trd_i + mvdy;

    let mvbx = if mvdx == 0 {
        ((trb_i - trd_i) * mvx) / trd_i
    } else {
        mvfx - mvx
    };
    let mvby = if mvdy == 0 {
        ((trb_i - trd_i) * mvy) / trd_i
    } else {
        mvfy - mvy
    };

    // The four scaled components stay representable in i32: the
    // unscaled MV is already bounded by Table 7-9's [-2048, 2047]
    // half-pel range (worst case at `vop_fcode == 7`), `TRB <= TRD`
    // keeps the multiply-and-divide factor in `[0, 1]`, and the delta
    // MVD shares the same Table 7-9 range. Their algebraic sum sits
    // comfortably inside i32 even at the worst-case extremes.
    Ok(DirectModeMv {
        forward: MotionVector {
            x: mvfx as i32,
            y: mvfy as i32,
        },
        backward: MotionVector {
            x: mvbx as i32,
            y: mvby as i32,
        },
    })
}

// ---------------------------------------------------------------------------
// §6.2.5 `motion_coding(mode, type_of_mb)` / §6.2.6 P-VOP MB-level MV-body driver
// ---------------------------------------------------------------------------
//
// The §6.2.5 `motion_coding(mode, type_of_mb)` syntax wraps one or four
// invocations of the §6.2.6.2 `motion_vector(mode)` body. The §6.2.6 P-VOP
// macroblock-layer text inlines the same pattern:
//
//   if ((derived_mb_type == 0 || derived_mb_type == 1)
//       && (vop_coding_type == "P" || (vop_coding_type == "S" && !mcsel))) {
//         motion_vector("forward")
//         // [interlaced field_prediction repeat — later round]
//   }
//   if (derived_mb_type == 2) {
//         for (j = 0; j < 4; j++)
//             if (!transparent_block(j))
//                   motion_vector("forward")
//   }
//
// This module decodes that block list. For the rectangular shape the
// transparency mask is statically all-opaque (§6.2.6 + §6.1.3.4 NOTE 2 — every
// 8x8 sub-block is opaque), so the §6.2.6 "if (!transparent_block(j))" guard
// always evaluates true; binary-shape transparent-sub-block elision is later-
// round work.

/// One macroblock's worth of decoded `motion_vector(mode)` bodies, after
/// §7.6.3 differential-component reconstruction. The number of populated
/// slots is the §6.2.5 `type_of_mb`-1 cardinality: one for 1-MV macroblocks
/// (`derived_mb_type == 0 / 1`), four for the `inter4v` case
/// (`derived_mb_type == 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionCodingDeltas {
    /// `type_of_mb == 1` — the per-macroblock single `MotionVectorDelta`.
    OneMv(MotionVectorDelta),
    /// `type_of_mb == 2` — four 8x8 per-sub-block `MotionVectorDelta`s in
    /// Figure 6-8 raster order (`0 = TL`, `1 = TR`, `2 = BL`, `3 = BR`).
    FourMv([MotionVectorDelta; 4]),
}

impl MotionCodingDeltas {
    /// Return the deltas as a slice (1 or 4 entries) regardless of variant.
    pub fn as_slice(&self) -> &[MotionVectorDelta] {
        match self {
            MotionCodingDeltas::OneMv(d) => core::slice::from_ref(d),
            MotionCodingDeltas::FourMv(arr) => arr,
        }
    }
}

/// `type_of_mb` argument for [`motion_coding`] — the §6.2.5 cardinality.
///
/// The numeric value matches the spec's `type_of_mb` integer, where the
/// `type_of_mb == 2` branch fires four `motion_vector(mode)` invocations
/// per the §6.2.5 syntax `if (type_of_mb == 2) for (i = 0; i < 3; i++)
/// motion_vector(mode)` (the unconditional opening call counts as the
/// first, the loop adds three more for a total of four).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeOfMb {
    /// `type_of_mb == 1` — one `motion_vector(mode)` invocation.
    One,
    /// `type_of_mb == 2` — four `motion_vector(mode)` invocations.
    Four,
}

/// Decode the §6.2.5 `motion_coding(mode, type_of_mb)` body — one (or four)
/// `motion_vector(mode)` invocations, each producing a reconstructed
/// `MotionVectorDelta` per §6.2.6.2 + §7.6.3.
///
/// `vop_fcode` is `vop_fcode_forward` for [`MvMode::Forward`] /
/// [`MvMode::Direct`] and `vop_fcode_backward` for [`MvMode::Backward`];
/// the caller picks the right one from the VOP header. The deltas are not
/// added to a predictor here — the caller pairs each delta with the right
/// §7.6.5 predictor via [`reconstruct_motion_vector`] (the predictor is a
/// block-level concept and the per-block predictor depends on previously-
/// decoded MBs, which is out of scope for the syntax-walk this function
/// performs).
///
/// On return the bit reader is positioned immediately after the last
/// component's residual / `mv_data`.
pub fn motion_coding(
    br: &mut BitReader<'_>,
    mode: MvMode,
    type_of_mb: TypeOfMb,
    vop_fcode: u8,
) -> Result<MotionCodingDeltas, MotionParseError> {
    // The first invocation always fires (§6.2.5 unconditional opening call).
    let first = decode_motion_vector_delta(br, mode, vop_fcode)?;
    match type_of_mb {
        TypeOfMb::One => Ok(MotionCodingDeltas::OneMv(first)),
        TypeOfMb::Four => {
            let second = decode_motion_vector_delta(br, mode, vop_fcode)?;
            let third = decode_motion_vector_delta(br, mode, vop_fcode)?;
            let fourth = decode_motion_vector_delta(br, mode, vop_fcode)?;
            Ok(MotionCodingDeltas::FourMv([first, second, third, fourth]))
        }
    }
}

/// Decode the §6.2.6 P-VOP macroblock-level motion-vector body for a
/// non-interlaced rectangular-shape macroblock, dispatching on
/// `derived_mb_type` per the §6.2.6 syntax:
///
/// * `Inter` / `InterQ` (`derived_mb_type == 0 || 1`) → one
///   `motion_vector("forward")` body.
/// * `Inter4V` (`derived_mb_type == 2`) → four `motion_vector("forward")`
///   bodies (one per 8x8 luma sub-block).
/// * `Intra` / `IntraQ` → no MV body; returns `None`.
///
/// Intra macroblocks carry no motion vectors per §6.2.6 (the
/// `(derived_mb_type == 0 || derived_mb_type == 1)` and
/// `(derived_mb_type == 2)` gates exclude the intra branches), so the
/// function returns `Ok(None)` for them — the caller skips straight to the
/// `for (i = 0; i < block_count; i++) block(i)` loop.
///
/// `vop_fcode` is `vop_fcode_forward` from the VOP header (the §6.2.6
/// P-VOP MB-level MV body is always forward).
///
/// **Out of scope (this round):**
/// * Interlaced `field_prediction` — the §6.2.6 line
///   `if (interlaced && field_prediction) motion_vector("forward")` fires
///   a second invocation per field; this driver assumes
///   `interlaced == false` and returns the frame-mode result.
/// * S(GMC)-VOP `mcsel == 1` macroblocks — the §6.2.6 outer gate
///   `(vop_coding_type == "S" && !mcsel)` excludes them; the caller must
///   check `mcsel` and route the macroblock to the §7.8.5 sprite-warping
///   path instead of invoking this function.
/// * Binary-shape `transparent_block(j)` elision — rectangular shape
///   guarantees every sub-block is opaque (§6.1.3.4 NOTE 2 / §6.2.6
///   rectangular branch).
pub fn decode_p_macroblock_motion_vectors(
    br: &mut BitReader<'_>,
    derived_mb_type: crate::macroblock::DerivedMbType,
    vop_fcode_forward: u8,
) -> Result<Option<MotionCodingDeltas>, MotionParseError> {
    use crate::macroblock::DerivedMbType;
    let type_of_mb = match derived_mb_type {
        DerivedMbType::Inter | DerivedMbType::InterQ => TypeOfMb::One,
        DerivedMbType::Inter4V => TypeOfMb::Four,
        DerivedMbType::Intra | DerivedMbType::IntraQ => return Ok(None),
    };
    let deltas = motion_coding(br, MvMode::Forward, type_of_mb, vop_fcode_forward)?;
    Ok(Some(deltas))
}

// ---------------------------------------------------------------------------
// §6.2.6 binary-shape `transparent_block(j)` elision for the four-MV branch
// ---------------------------------------------------------------------------
//
// §6.2.6 P-VOP / S(GMC)-VOP macroblock-layer text spells the `inter4v`
// branch (`derived_mb_type == 2`) as:
//
//   if (derived_mb_type == 2) {
//         for (j = 0; j < 4; j++)
//             if (!transparent_block(j))
//                   motion_vector("forward")
//   }
//
// The §5.2.7 definition of `transparent_block(j)` returns 1 when the j-th
// 8x8 sub-block consists only of transparent pixels (the §6.1.3.4 binary-
// shape grid spells the per-block opacity from the decimated luma shape).
// When the shape is rectangular, every j is opaque and the loop fires four
// `motion_vector("forward")` invocations — exactly the existing
// [`decode_p_macroblock_motion_vectors`] / [`motion_coding`] behaviour for
// [`TypeOfMb::Four`].
//
// For binary-shape VOPs some sub-blocks can be transparent, and the
// bitstream omits the `motion_vector("forward")` body for those sub-blocks
// entirely. The §6.2.5 `motion_coding(mode, type_of_mb)` syntax is *not*
// what fires for the four-MV inter4v case under binary shape — §6.2.6
// inlines the `if (!transparent_block(j))` guard directly, so the four
// sub-block bodies are not bundled into a single `motion_coding` call.
//
// This module exposes the per-sub-block guard. The 8x8 block index `j`
// follows the §6.2.6 / Figure 6-8 raster order (`0 = TL`, `1 = TR`,
// `2 = BL`, `3 = BR`) — the same convention used by
// [`MotionCodingDeltas::FourMv`] and by [`crate::scan`] / Figure 6-8.

/// Per-sub-block opacity mask for the §6.2.6 four-MV inter4v branch.
///
/// Each entry corresponds to one 8x8 luma sub-block of the §6.2.6
/// macroblock in Figure 6-8 raster order (`0 = TL`, `1 = TR`, `2 = BL`,
/// `3 = BR`). The boolean encodes the §5.2.7 `transparent_block(j)`
/// **negation** — `opaque[j] == true` means "block j carries at least one
/// opaque pixel" (`!transparent_block(j)`).
///
/// Rectangular-shape VOPs always pass `[true; 4]` (§6.1.3.4 NOTE 2 —
/// every sub-block of a rectangular-shape macroblock is opaque); binary-
/// shape VOPs derive the four flags from the §6.1.3.4 decoded binary
/// shape (one entry per 8x8 luma quadrant — opaque if any of the 64
/// shape samples is opaque).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryShapeBlockOpacity {
    /// Figure 6-8 raster-order opacity flags (`!transparent_block(j)`).
    pub opaque: [bool; 4],
}

impl BinaryShapeBlockOpacity {
    /// All four sub-blocks opaque — the §6.1.3.4 NOTE 2 rectangular case.
    /// Equivalent to passing this mask through
    /// [`decode_p_macroblock_motion_vectors_with_shape`] and producing
    /// four `motion_vector("forward")` bodies, matching the existing
    /// [`decode_p_macroblock_motion_vectors`] behaviour for
    /// [`crate::macroblock::DerivedMbType::Inter4V`].
    pub const ALL_OPAQUE: Self = Self { opaque: [true; 4] };

    /// Build from a per-sub-block opacity array. `opaque[j] == true`
    /// means sub-block `j` (Figure 6-8 raster order) carries at least
    /// one opaque pixel.
    pub const fn new(opaque: [bool; 4]) -> Self {
        Self { opaque }
    }

    /// Number of `motion_vector("forward")` bodies the §6.2.6 inter4v
    /// branch will decode under this opacity mask. Equals
    /// `opaque.iter().filter(|o| **o).count()` — zero through four.
    pub const fn motion_vector_invocation_count(self) -> usize {
        let mut n = 0;
        let mut i = 0;
        while i < 4 {
            if self.opaque[i] {
                n += 1;
            }
            i += 1;
        }
        n
    }
}

/// Per-sub-block decoded motion-vector deltas for the §6.2.6 four-MV
/// inter4v branch under binary-shape `transparent_block(j)` elision.
///
/// Each slot corresponds to one 8x8 luma sub-block in Figure 6-8 raster
/// order (`0 = TL`, `1 = TR`, `2 = BL`, `3 = BR`). A `Some` entry holds
/// the §7.6.3 reconstructed [`MotionVectorDelta`]; a `None` entry means
/// the §6.2.6 `if (!transparent_block(j))` guard suppressed the
/// `motion_vector("forward")` body for that sub-block (no bits were read
/// from the bitstream for it).
///
/// For a fully-opaque mask (rectangular shape or
/// [`BinaryShapeBlockOpacity::ALL_OPAQUE`]) this is the same four
/// `MotionVectorDelta`s as
/// [`MotionCodingDeltas::FourMv`] — converted via
/// [`BinaryShapeFourMv::to_motion_coding_deltas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryShapeFourMv {
    /// Optional per-sub-block delta in Figure 6-8 raster order.
    pub deltas: [Option<MotionVectorDelta>; 4],
}

impl BinaryShapeFourMv {
    /// Iterate over the populated `(j, delta)` pairs (`j` in Figure 6-8
    /// raster order).
    pub fn iter_present(&self) -> impl Iterator<Item = (usize, MotionVectorDelta)> + '_ {
        self.deltas
            .iter()
            .enumerate()
            .filter_map(|(j, d)| d.map(|d| (j, d)))
    }

    /// Convert to a [`MotionCodingDeltas::FourMv`] view when **all four
    /// slots are populated**. Returns `None` when any sub-block was
    /// elided by `transparent_block(j) == 1`.
    ///
    /// Use this on the rectangular-shape / all-opaque path to share the
    /// downstream reconstruction logic with
    /// [`decode_p_macroblock_motion_vectors`] (§6.1.3.4 NOTE 2).
    pub fn to_motion_coding_deltas(&self) -> Option<MotionCodingDeltas> {
        match self.deltas {
            [Some(a), Some(b), Some(c), Some(d)] => Some(MotionCodingDeltas::FourMv([a, b, c, d])),
            _ => None,
        }
    }
}

/// Binary-shape-aware variant of [`decode_p_macroblock_motion_vectors`]
/// for the §6.2.6 four-MV inter4v branch.
///
/// This implements the §6.2.6 line `for (j = 0; j < 4; j++) if
/// (!transparent_block(j)) motion_vector("forward")` directly: the
/// caller supplies the four-block opacity mask derived from the
/// §6.1.3.4 decoded binary shape (or
/// [`BinaryShapeBlockOpacity::ALL_OPAQUE`] for rectangular VOPs), and
/// this routine reads one `motion_vector("forward")` body per opaque
/// sub-block, skipping transparent ones outright (no bits are consumed
/// for elided sub-blocks).
///
/// `derived_mb_type` is from the §6.2.6 / §7.4.4 macroblock layer.
///
/// * [`crate::macroblock::DerivedMbType::Inter4V`] — fires the
///   §6.2.6 `if (derived_mb_type == 2)` loop, returning
///   `Ok(Some(BinaryShapeFourMv))` with one delta per opaque
///   sub-block.
/// * [`crate::macroblock::DerivedMbType::Inter`] /
///   [`crate::macroblock::DerivedMbType::InterQ`] — the single-MV
///   branch; this variant is **not** the right routine (the
///   single-MV macroblock-level body has no per-sub-block
///   `transparent_block` gate per §6.2.6, just the macroblock-level
///   `transparent_mb()` gate handled by the macroblock-layer
///   walker). Callers must route these through
///   [`decode_p_macroblock_motion_vectors`] /
///   [`motion_coding`] with [`TypeOfMb::One`]; this function returns
///   `Ok(None)` so the caller can detect the misroute.
/// * [`crate::macroblock::DerivedMbType::Intra`] /
///   [`crate::macroblock::DerivedMbType::IntraQ`] — no MV body per
///   §6.2.6; returns `Ok(None)`.
///
/// **Out of scope (this round):**
/// * Interlaced `field_prediction` — the §6.2.6 inter4v branch is the
///   non-interlaced four-MV path; the `if (interlaced && field_prediction)`
///   line follows the §6.2.6 single-MV branch instead and is later-round
///   work.
/// * S(GMC)-VOP `mcsel == 1` macroblocks — gated out the same way as
///   in [`decode_p_macroblock_motion_vectors`]; the caller must check
///   `mcsel` first.
/// * Computing the opacity mask from the decoded binary shape — this
///   function takes the mask as input. The §6.1.3.4 binary-shape
///   decoder and the §6.2.6 mapping to per-sub-block opacity are
///   later-round work; [`crate::chroma_shape`] handles the related
///   luma-to-chroma shape decimation but not the luma-to-block
///   `transparent_block(j)` derivation.
pub fn decode_p_macroblock_motion_vectors_with_shape(
    br: &mut BitReader<'_>,
    derived_mb_type: crate::macroblock::DerivedMbType,
    vop_fcode_forward: u8,
    opacity: BinaryShapeBlockOpacity,
) -> Result<Option<BinaryShapeFourMv>, MotionParseError> {
    use crate::macroblock::DerivedMbType;
    match derived_mb_type {
        DerivedMbType::Inter4V => {
            let mut deltas: [Option<MotionVectorDelta>; 4] = [None; 4];
            for (j, slot) in deltas.iter_mut().enumerate() {
                if opacity.opaque[j] {
                    *slot = Some(decode_motion_vector_delta(
                        br,
                        MvMode::Forward,
                        vop_fcode_forward,
                    )?);
                }
            }
            Ok(Some(BinaryShapeFourMv { deltas }))
        }
        DerivedMbType::Inter
        | DerivedMbType::InterQ
        | DerivedMbType::Intra
        | DerivedMbType::IntraQ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// §6.2.6 interlaced field-prediction second `motion_vector(mode)` invocation
// ---------------------------------------------------------------------------
//
// The §6.2.6 macroblock syntax fires a second body right after the first
// whenever the macroblock is field predicted:
//
//   motion_vector("forward")
//   if (interlaced && field_prediction)
//         motion_vector("forward")
//
// (P-VOP / S(GMC)-VOP 1-MV branch; the B-VOP branch has the analogous
// forward and backward pairs). §7.7.2.1 fixes the pair's order: the
// decoder "shall first extract the differential motion vectors
// ((MVDx f1, MVDy f1) and (MVDx f2, MVDy f2) for top and bottom fields of
// a field predicted macroblock, respectively)" — the first body is the
// top-field differential, the second the bottom-field differential.
// Table 7-14 and the §7.7.2.2 field-mode pseudo code (`MVD[0]` updates
// `PMV[0]` = top field forward, `MVD[1]` updates `PMV[1]` = bottom field
// forward; backward uses `PMV[2]`/`PMV[3]` in the same top-then-bottom
// order, and the field-bidirectional loop walks `MVD[0..3]` as forward
// top, forward bottom, backward top, backward bottom) confirm the same
// ordering for B-VOPs.
//
// Both invocations of a pair share the direction's single fcode
// (`vop_fcode_forward` for the forward pair, `vop_fcode_backward` for the
// backward pair) — the §6.2.6.2 residual gating depends only on `mode`.
//
// Semantics deferred to a later round: the §7.7.2.1 reconstruction
// `MVx fi = MVDx fi + Px` / `MVy fi = 2 * (MVDy fi + (Py / 2))` (both
// fields share one predictor; the vertical differential is coded in
// field coordinates and the reconstructed vertical component is always
// an even frame-coordinate integer) and the §7.7.2 field motion
// compensation itself.

/// The two §6.2.6.2 bodies of one field-predicted prediction direction,
/// in bitstream order — §7.7.2.1: first body = top-field differential
/// (`MVD f1`), second body = bottom-field differential (`MVD f2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMvPair {
    /// Top-field differential (`MVDx f1`, `MVDy f1`) — the first body.
    pub top: MotionVectorDelta,
    /// Bottom-field differential (`MVDx f2`, `MVDy f2`) — the second
    /// body.
    pub bottom: MotionVectorDelta,
}

/// Decode the §6.2.6 field-predicted pair of `motion_vector(mode)`
/// bodies — the unconditional first invocation plus the `if (interlaced
/// && field_prediction)` second one — assigning top/bottom per §7.7.2.1.
///
/// `mode` must be [`MvMode::Forward`] or [`MvMode::Backward`]; the
/// §6.2.6 direct line has no second invocation and §6.2.6.3's
/// `mb_type != "1"` clause keeps `field_prediction` from ever being
/// coded for a direct macroblock, so [`MvMode::Direct`] yields
/// [`MotionParseError::InvalidFieldPredictionContext`] without
/// consuming bits.
pub fn decode_field_motion_vector_pair(
    br: &mut BitReader<'_>,
    mode: MvMode,
    vop_fcode: u8,
) -> Result<FieldMvPair, MotionParseError> {
    if mode == MvMode::Direct {
        return Err(MotionParseError::InvalidFieldPredictionContext);
    }
    let top = decode_motion_vector_delta(br, mode, vop_fcode)?;
    let bottom = decode_motion_vector_delta(br, mode, vop_fcode)?;
    Ok(FieldMvPair { top, bottom })
}

/// A P-VOP macroblock's decoded MV bodies, frame- or field-predicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PMbMotionVectors {
    /// Frame prediction (`field_prediction == 0`, or a progressive
    /// VOL) — the one-or-four delta view of [`MotionCodingDeltas`].
    Frame(MotionCodingDeltas),
    /// Field prediction (`field_prediction == 1`) — one forward
    /// differential per field, top then bottom (§7.7.2.1).
    Field(FieldMvPair),
}

/// Decode the §6.2.6 P-VOP macroblock-level motion-vector body for a
/// rectangular-shape macroblock in an interlaced VOL, honouring the
/// `if (interlaced && field_prediction) motion_vector("forward")`
/// second invocation.
///
/// `field_prediction` is the §6.2.6.3 bit the caller obtained from the
/// macroblock header's `interlaced_information()` body
/// (`InterlacedInformation::field_prediction.is_some()`); passing
/// `false` reproduces [`decode_p_macroblock_motion_vectors`] exactly
/// (the second invocation never fires). Since §6.2.6.3 only codes the
/// bit when `derived_mb_type < 2`, a `true` flag combined with an
/// `inter4v` or intra `derived_mb_type` is rejected with
/// [`MotionParseError::InvalidFieldPredictionContext`] before any bit
/// is consumed.
pub fn decode_p_macroblock_motion_vectors_interlaced(
    br: &mut BitReader<'_>,
    derived_mb_type: crate::macroblock::DerivedMbType,
    vop_fcode_forward: u8,
    field_prediction: bool,
) -> Result<Option<PMbMotionVectors>, MotionParseError> {
    use crate::macroblock::DerivedMbType;
    match derived_mb_type {
        DerivedMbType::Inter | DerivedMbType::InterQ => {
            if field_prediction {
                let pair = decode_field_motion_vector_pair(br, MvMode::Forward, vop_fcode_forward)?;
                Ok(Some(PMbMotionVectors::Field(pair)))
            } else {
                let delta = decode_motion_vector_delta(br, MvMode::Forward, vop_fcode_forward)?;
                Ok(Some(PMbMotionVectors::Frame(MotionCodingDeltas::OneMv(
                    delta,
                ))))
            }
        }
        DerivedMbType::Inter4V => {
            if field_prediction {
                return Err(MotionParseError::InvalidFieldPredictionContext);
            }
            let deltas = motion_coding(br, MvMode::Forward, TypeOfMb::Four, vop_fcode_forward)?;
            Ok(Some(PMbMotionVectors::Frame(deltas)))
        }
        DerivedMbType::Intra | DerivedMbType::IntraQ => {
            if field_prediction {
                return Err(MotionParseError::InvalidFieldPredictionContext);
            }
            Ok(None)
        }
    }
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
            MotionParseError::InvalidFieldPredictionContext,
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
    fn amv_integer_pel_input_accepted() {
        // pel_denominator == 1 (integer-pel input). Real AMV = sum / 256
        // pels. Output in half-pels = (2 * sum) // 256.
        // sum = 256 * 3 (flat MV of 3 pels) → AMV_half_x = 768 // 256 = 6.
        let (x, y) = flat_pel_grid(3, -2);
        let amv = averaged_motion_vector(&x, &y, 1, false, 7).unwrap();
        // 256*3=768; 2*768=1536; 1536/(256*1)=6.
        // 256*-2=-512; 2*-512=-1024; -1024/(256)= -4.
        assert_eq!(amv, MotionVector { x: 6, y: -4 });
    }

    #[test]
    fn amv_mismatched_pel_denominator_still_rounds_correctly() {
        // `pel_denominator == 3` is not a multiple of 2 — historically a
        // precondition error, but the spec's `//` operator is defined for
        // any positive denominator, so the function now accepts it and
        // rounds to the half-pel grid via the spec's away-from-zero rule.
        // Real value 1/3 pel (=0.333…) → half-sample bin [0.25, 0.75) → 0.5.
        // sum = 256 (flat MV of 1 in third-pels = 0.333…).
        // AMV_half = (2 * 256) // (256 * 3) = 512 // 768 → away-from-zero
        // round of 512/768 = 0.667 → 1 half-pel.
        let (x, y) = flat_pel_grid(1, 0);
        let amv = averaged_motion_vector(&x, &y, 3, false, 7).unwrap();
        assert_eq!(amv, MotionVector { x: 1, y: 0 });
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

    // ─────────────────── §7.6.9.5.2 direct-mode MV derivation ─────────

    #[test]
    fn direct_mv_trd_zero_rejected() {
        let err = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 4, y: -2 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            0,
            0,
            DirectMvUnits::Match,
        )
        .unwrap_err();
        assert_eq!(err, DirectMvError::InvalidTrd(0));
    }

    #[test]
    fn direct_mv_trd_negative_rejected() {
        let err = direct_mode_motion_vector(
            DirectCoLocatedMv::TransparentOrAbsent,
            MotionVectorDelta { dx: 0, dy: 0 },
            0,
            -1,
            DirectMvUnits::Match,
        )
        .unwrap_err();
        assert_eq!(err, DirectMvError::InvalidTrd(-1));
    }

    #[test]
    fn direct_mv_trb_out_of_range_rejected() {
        let err = direct_mode_motion_vector(
            DirectCoLocatedMv::TransparentOrAbsent,
            MotionVectorDelta { dx: 0, dy: 0 },
            5,
            3,
            DirectMvUnits::Match,
        )
        .unwrap_err();
        assert_eq!(err, DirectMvError::TrbOutOfRange { trb: 5, trd: 3 });

        let err = direct_mode_motion_vector(
            DirectCoLocatedMv::TransparentOrAbsent,
            MotionVectorDelta { dx: 0, dy: 0 },
            -1,
            3,
            DirectMvUnits::Match,
        )
        .unwrap_err();
        assert_eq!(err, DirectMvError::TrbOutOfRange { trb: -1, trd: 3 });
    }

    #[test]
    fn direct_mv_trb_equals_trd_makes_forward_full_mv() {
        // TRB == TRD: B-VOP coincides temporally with the next anchor.
        // MVF = MV + MVD; MVB (when MVD == 0) = 0 * MV / TRD = 0.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 6, y: -4 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            3,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 6, y: -4 });
        assert_eq!(got.backward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn direct_mv_trb_zero_makes_forward_only_mvd() {
        // TRB == 0: B-VOP coincides temporally with the previous anchor.
        // MVF = 0 + MVD = MVD; MVB (when MVD == 0) = (-TRD)*MV/TRD = -MV.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 6, y: -4 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            0,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 0, y: 0 });
        assert_eq!(got.backward, MotionVector { x: -6, y: 4 });
    }

    #[test]
    fn direct_mv_zero_delta_canonical_split() {
        // Worked example: MV = 9, TRD = 3, TRB = 1, MVD = 0.
        // MVF = (1*9)/3 = 3.
        // MVB = ((1-3)*9)/3 = -6 (MVD==0 path).
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 9, y: -9 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 3, y: -3 });
        assert_eq!(got.backward, MotionVector { x: -6, y: 6 });
        // Identity check: MVF + MVB = (TRB - TRD)*MV/TRD + TRB*MV/TRD
        //              = ((TRB - TRD + TRB) * MV) / TRD  (because
        // the divisions are both exact when TRD | MV).
        assert_eq!(got.forward.x + got.backward.x, -3);
    }

    #[test]
    fn direct_mv_nonzero_delta_uses_subtract_branch() {
        // MVD != 0 → MVB = MVF - MV (not the scaled formula).
        // MV = 9, TRD = 3, TRB = 2, MVDx = +1.
        // MVF = (2*9)/3 + 1 = 6 + 1 = 7.
        // MVB = MVF - MV = 7 - 9 = -2.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 9, y: 9 }),
            MotionVectorDelta { dx: 1, dy: 0 },
            2,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 7, y: 6 });
        // Note: dy == 0 still uses the scaled formula on the y axis.
        // MVBx = MVF - MV = -2; MVBy = ((2-3)*9)/3 = -3.
        assert_eq!(got.backward, MotionVector { x: -2, y: -3 });
    }

    #[test]
    fn direct_mv_per_component_branch_independence() {
        // Mixed deltas: dx != 0 takes the subtract branch, dy == 0
        // takes the scaled-formula branch — independently per axis.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 12, y: -6 }),
            MotionVectorDelta { dx: -3, dy: 0 },
            1,
            4,
            DirectMvUnits::Match,
        )
        .unwrap();
        // MVFx = (1*12)/4 + (-3) = 3 - 3 = 0; subtract branch → MVBx = 0 - 12 = -12.
        // MVFy = (1*-6)/4 + 0 = -1 (truncation toward zero: -6/4 = -1, not -2);
        // MVBy uses the scaled formula (dy == 0): ((1-4)*-6)/4 = 18/4 = 4.
        assert_eq!(got.forward, MotionVector { x: 0, y: -1 });
        assert_eq!(got.backward, MotionVector { x: -12, y: 4 });
    }

    #[test]
    fn direct_mv_truncation_toward_zero_per_3_4() {
        // §3.4 `/` is integer division with truncation toward zero
        // (not floor!). Verify both signs of the dividend.
        // MV = 7, TRD = 4, TRB = 1, MVD = 0.
        // MVF = (1*7)/4 = 1 (7/4 truncates to 1, not 2).
        // MV = -7 → MVF = (1*-7)/4 = -1 (-7/4 truncates to -1, not -2).
        let pos = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 7, y: 0 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            4,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(pos.forward.x, 1);

        let neg = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: -7, y: 0 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            4,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(neg.forward.x, -1);
    }

    #[test]
    fn direct_mv_transparent_co_located_uses_zero_mv() {
        // §7.6.9.5.1 final sentence: transparent / absent co-located
        // → substitute MV = (0, 0). MVF = MVD; MVB (MVD != 0) = MVD - 0 = MVD.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::TransparentOrAbsent,
            MotionVectorDelta { dx: 4, dy: -2 },
            2,
            5,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 4, y: -2 });
        assert_eq!(got.backward, MotionVector { x: 4, y: -2 });
    }

    #[test]
    fn direct_mv_transparent_with_zero_delta_yields_zero_pair() {
        // Transparent + MVD == 0 → both MVF and MVB are zero (the
        // §7.6.9.6 skipped-MB case the spec spells out explicitly).
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::TransparentOrAbsent,
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 0, y: 0 });
        assert_eq!(got.backward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn direct_mv_qpel_to_halfpel_halves_mv_via_table_7_13() {
        // QpelMvToHalfPel: MV is halved per Table 7-13 BEFORE scaling.
        // MV = (5, -5) quarter-pel → halved to (3, -3) half-pel
        //   (per Table 7-13: 5 = 4 + 1 → 2*2 + 1 = 3; -5 → -3).
        // Then with TRB = 1, TRD = 2, MVD = 0:
        //   MVF = (1 * 3) / 2 = 1 (truncating); MVBy similarly.
        //   MVBx = ((1 - 2) * 3) / 2 = -1.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 5, y: -5 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            2,
            DirectMvUnits::QpelMvToHalfPel,
        )
        .unwrap();
        // Sanity-check the conversion as exposed via the helper.
        let halved = direct_mode_reduce_qpel_to_half_pel(MotionVector { x: 5, y: -5 });
        assert_eq!(halved, MotionVector { x: 3, y: -3 });
        // The derivation should match the manual computation above.
        assert_eq!(got.forward, MotionVector { x: 1, y: -1 });
        assert_eq!(got.backward, MotionVector { x: -1, y: 1 });
    }

    #[test]
    fn direct_mv_qpel_match_path_does_not_reduce() {
        // Same input as the QpelMvToHalfPel case but with Match: no
        // reduction; the formula runs on the full quarter-pel MV.
        // MV = (5, -5), TRB = 1, TRD = 2, MVD = 0.
        // MVF = (1*5)/2 = 2 (truncates); MVB = ((1-2)*5)/2 = -2.
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 5, y: -5 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            2,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 2, y: -2 });
        assert_eq!(got.backward, MotionVector { x: -2, y: 2 });
    }

    #[test]
    fn direct_mv_reduce_qpel_invariant_on_zero() {
        // Sanity: reducing the zero MV stays zero (so transparent +
        // QpelMvToHalfPel is a no-op against transparent + Match).
        assert_eq!(
            direct_mode_reduce_qpel_to_half_pel(MotionVector { x: 0, y: 0 }),
            MotionVector { x: 0, y: 0 }
        );
    }

    #[test]
    fn direct_mv_reduce_qpel_matches_table_7_13_chroma_helper() {
        // The helper must produce the same numeric reduction as
        // `quarter_sample::reduce_qpel_to_half_pel_chroma`, component-wise.
        use crate::quarter_sample::reduce_qpel_to_half_pel_chroma;
        for &x in &[-9i32, -5, -4, -1, 0, 1, 4, 5, 9] {
            for &y in &[-9i32, -5, -4, -1, 0, 1, 4, 5, 9] {
                let got = direct_mode_reduce_qpel_to_half_pel(MotionVector { x, y });
                assert_eq!(got.x, reduce_qpel_to_half_pel_chroma(x), "x = {x}");
                assert_eq!(got.y, reduce_qpel_to_half_pel_chroma(y), "y = {y}");
            }
        }
    }

    #[test]
    fn direct_mv_skipped_p_vop_reconstruction() {
        // §7.6.9.6: "If the co-located macroblock in the most recently
        // decoded I- or P-VOP is skipped, the current B-macroblock is
        // treated as the forward mode with the zero motion vector
        // (MVFx, MVFy). If the modb equals to '1' the current B-
        // macroblock is reconstructed by using the direct mode with
        // zero delta vector." A skipped P-VOP MB implies MV = (0, 0)
        // and the direct-mode derivation with MVD = 0 must yield the
        // zero MVF / MVB pair (the zero-delta forward-mode equivalent).
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 0, y: 0 }),
            MotionVectorDelta { dx: 0, dy: 0 },
            1,
            3,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 0, y: 0 });
        assert_eq!(got.backward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn direct_mv_error_displays_format_human_readable() {
        let s = format!("{}", DirectMvError::InvalidTrd(0));
        assert!(s.contains("TRD"));
        assert!(s.contains("0"));
        let s = format!("{}", DirectMvError::TrbOutOfRange { trb: 5, trd: 3 });
        assert!(s.contains("TRB"));
        assert!(s.contains("TRD"));
    }

    #[test]
    fn direct_mv_decoded_delta_round_trip_via_mvmode_direct() {
        // End-to-end: read a direct-mode MVD off the wire. The spec
        // pins `f_code == 1` for direct mode (§7.6.3 closing paragraph,
        // "in the case of the direct mode the f_code is always one"),
        // so f = 1 and `reconstruct_component` returns `mv_data`
        // verbatim. h = +2 (code 0010), v = -2 (code 0011) → MVD = (2, -2).
        // Then derive direct-mode MVs with MV = (4, -4), TRB = 1, TRD = 4.
        // MVFx: dx != 0 path → (1*4)/4 + 2 = 1 + 2 = 3; MVBx = MVFx - MV = -1.
        // MVFy: dy != 0 path → (1*-4)/4 + -2 = -1 + -2 = -3; MVBy = MVFy - MV = 1.
        let mut w = BitWriter::new();
        w.write_bits(0b0010, 4); // h_data = 2
        w.write_bits(0b0011, 4); // v_data = -2
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let mvd = decode_motion_vector_delta(&mut br, MvMode::Direct, 1).unwrap();
        assert_eq!(mvd, MotionVectorDelta { dx: 2, dy: -2 });
        let got = direct_mode_motion_vector(
            DirectCoLocatedMv::Mv(MotionVector { x: 4, y: -4 }),
            mvd,
            1,
            4,
            DirectMvUnits::Match,
        )
        .unwrap();
        assert_eq!(got.forward, MotionVector { x: 3, y: -3 });
        assert_eq!(got.backward, MotionVector { x: -1, y: 1 });
    }

    // -----------------------------------------------------------------
    // §6.2.5 motion_coding / §6.2.6 P-VOP MV-body driver tests.
    // -----------------------------------------------------------------

    /// Write one Table B.12 row by `mv_data`. Panics if `mv_data` is
    /// outside `-32..=32` — the table only ranges over the on-wire
    /// doubled-integer domain.
    fn write_mv_data(w: &mut BitWriter, mv_data: i32) {
        let (code, len, _) = MVD_TABLE
            .iter()
            .copied()
            .find(|&(_, _, v)| v == mv_data)
            .unwrap_or_else(|| panic!("mv_data {mv_data} not in MVD_TABLE"));
        w.write_bits(u32::from(code), usize::from(len));
    }

    #[test]
    fn motion_coding_one_mv_forward_fcode_one() {
        // Single motion_vector("forward") body, vop_fcode == 1 → no
        // residuals. Encode (h=+2, v=-2) on the wire.
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2); // 0010 → +1 vector difference (mv_data 2)
        write_mv_data(&mut w, -2); // 0011 → -1 vector difference (mv_data -2)
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = motion_coding(&mut br, MvMode::Forward, TypeOfMb::One, 1).unwrap();
        assert_eq!(
            got,
            MotionCodingDeltas::OneMv(MotionVectorDelta { dx: 2, dy: -2 })
        );
        assert_eq!(got.as_slice().len(), 1);
    }

    #[test]
    fn motion_coding_four_mv_forward_fcode_one() {
        // Four motion_vector("forward") bodies, all (h=0, v=0). fcode == 1
        // means no residuals are read; the on-wire encoding is just 8
        // copies of the zero VLC (`1`).
        let mut w = BitWriter::new();
        for _ in 0..8 {
            write_mv_data(&mut w, 0);
        }
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = motion_coding(&mut br, MvMode::Forward, TypeOfMb::Four, 1).unwrap();
        let zeros = MotionVectorDelta { dx: 0, dy: 0 };
        assert_eq!(
            got,
            MotionCodingDeltas::FourMv([zeros, zeros, zeros, zeros])
        );
        assert_eq!(got.as_slice().len(), 4);
    }

    #[test]
    fn motion_coding_four_mv_distinct_deltas() {
        // Four distinct (dx, dy) deltas: (0,0) (1,-1) (-1,1) (0,1).
        // All at fcode == 1 to avoid the residual gate.
        let mut w = BitWriter::new();
        let deltas = [(0i32, 0i32), (2, -2), (-2, 2), (0, 2)];
        for &(h, v) in &deltas {
            write_mv_data(&mut w, h);
            write_mv_data(&mut w, v);
        }
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = motion_coding(&mut br, MvMode::Forward, TypeOfMb::Four, 1).unwrap();
        if let MotionCodingDeltas::FourMv(four) = got {
            assert_eq!(four[0], MotionVectorDelta { dx: 0, dy: 0 });
            assert_eq!(four[1], MotionVectorDelta { dx: 2, dy: -2 });
            assert_eq!(four[2], MotionVectorDelta { dx: -2, dy: 2 });
            assert_eq!(four[3], MotionVectorDelta { dx: 0, dy: 2 });
        } else {
            panic!("expected FourMv variant");
        }
    }

    #[test]
    fn motion_coding_with_fcode_two_reads_residuals() {
        // vop_fcode == 2 → r_size = 1 bit; residuals are read when
        // mv_data != 0. Encode one component as mv_data = 2 + residual 1,
        // the other as mv_data = -2 + residual 0.
        // Reconstruction (§7.6.3): |mv_data| = 2 → (2-1)*f + res + 1 = f + res + 1.
        // With f = 2: h = (2-1)*2 + 1 + 1 = 4; v = -((2-1)*2 + 0 + 1) = -3.
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2); // h_data
        w.write_bits(1, 1); // h_residual
        write_mv_data(&mut w, -2); // v_data
        w.write_bits(0, 1); // v_residual
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = motion_coding(&mut br, MvMode::Forward, TypeOfMb::One, 2).unwrap();
        assert_eq!(
            got,
            MotionCodingDeltas::OneMv(MotionVectorDelta { dx: 4, dy: -3 })
        );
    }

    #[test]
    fn motion_coding_truncated_mid_third_block() {
        // Encode three valid (h, v) pairs and an h_data for the fourth
        // block — the fourth block's v_data is missing, and the encoded
        // pattern leaves zero remaining bits in the reader so the next
        // VLC attempt reports Truncated. The (0,0) zero VLC at fcode==1
        // is a single `1` bit per component; 7 of them is 7 bits, but
        // we deliberately use a (h,v) pair whose v_data needs more
        // bits than the buffer has left after byte alignment.
        let mut w = BitWriter::new();
        // First three blocks: 6 zero VLCs (1 bit each) = 6 bits.
        for _ in 0..6 {
            write_mv_data(&mut w, 0);
        }
        // Fourth block's h_data: another zero (1 bit, total 7 bits).
        write_mv_data(&mut w, 0);
        // Now align (pads to 8 bits with `0`s — these padding bits do
        // NOT match any non-empty Table B.12 prefix on a fresh decode).
        w.align();
        // Truncate the buffer to exactly the bits we wrote (no spare
        // byte), so the fourth block's v_data has only one zero-padding
        // bit available; a single `0` is not a valid Table B.12 prefix
        // and a longer prefix triggers Truncated.
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let err = motion_coding(&mut br, MvMode::Forward, TypeOfMb::Four, 1).unwrap_err();
        // The fourth-block v_data is either InvalidMvData (when the
        // padding bit aligns to a non-VLC pattern) or Truncated (when
        // the decoder runs off the end of the buffer). Both are valid
        // error signatures for a malformed stream; we only need to
        // confirm the wrapper propagates *some* MotionParseError from
        // the fourth invocation rather than treating it as a successful
        // decode.
        assert!(
            matches!(
                err,
                MotionParseError::Truncated | MotionParseError::InvalidMvData { .. }
            ),
            "expected Truncated or InvalidMvData, got {err:?}"
        );
    }

    #[test]
    fn motion_coding_rejects_invalid_fcode_eagerly() {
        // fcode == 0 is rejected by `decode_motion_vector_delta` even
        // before the first VLC read; the wrapper must propagate it.
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let err = motion_coding(&mut br, MvMode::Forward, TypeOfMb::One, 0).unwrap_err();
        assert!(matches!(err, MotionParseError::InvalidFcode(0)));
        // The reader must not have advanced.
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn decode_p_mb_mvs_inter_one_mv() {
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        write_mv_data(&mut w, -2);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_p_macroblock_motion_vectors(&mut br, DerivedMbType::Inter, 1).unwrap();
        assert_eq!(
            got,
            Some(MotionCodingDeltas::OneMv(MotionVectorDelta {
                dx: 2,
                dy: -2
            }))
        );
    }

    #[test]
    fn decode_p_mb_mvs_interq_one_mv() {
        // InterQ (derived_mb_type == 1) also fires a single
        // motion_vector("forward").
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 0);
        write_mv_data(&mut w, 0);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_p_macroblock_motion_vectors(&mut br, DerivedMbType::InterQ, 1).unwrap();
        assert_eq!(
            got,
            Some(MotionCodingDeltas::OneMv(MotionVectorDelta {
                dx: 0,
                dy: 0
            }))
        );
    }

    #[test]
    fn decode_p_mb_mvs_inter4v_four_mvs() {
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        for _ in 0..8 {
            write_mv_data(&mut w, 0);
        }
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_p_macroblock_motion_vectors(&mut br, DerivedMbType::Inter4V, 1).unwrap();
        let zeros = MotionVectorDelta { dx: 0, dy: 0 };
        assert_eq!(
            got,
            Some(MotionCodingDeltas::FourMv([zeros, zeros, zeros, zeros]))
        );
    }

    #[test]
    fn decode_p_mb_mvs_intra_no_mv_no_bits_consumed() {
        // Intra and IntraQ both return Ok(None) without touching the
        // reader; the macroblock walker proceeds straight to block(i).
        use crate::macroblock::DerivedMbType;
        let bytes = [0xFFu8; 4];
        for ty in [DerivedMbType::Intra, DerivedMbType::IntraQ] {
            let mut br = BitReader::new(&bytes);
            let got = decode_p_macroblock_motion_vectors(&mut br, ty, 1).unwrap();
            assert_eq!(got, None);
            assert_eq!(br.bit_position(), 0, "must not consume bits for {:?}", ty);
        }
    }

    #[test]
    fn motion_coding_one_mv_round_trips_with_reconstruct_motion_vector() {
        // End-to-end: wire bytes → motion_coding → reconstruct_motion_vector
        // composes the §6.2.6.2 + §7.6.3 path with a caller predictor.
        // fcode=2, mv_data=+2 with residual 0 → (2-1)*2 + 0 + 1 = 3.
        // Predictor (Px, Py) = (5, -5) → final MV = (5+3, -5+3) = (8, -2)
        // (no Table 7-9 wrap at fcode=2 / range = 128).
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        w.write_bits(0, 1);
        write_mv_data(&mut w, 2);
        w.write_bits(0, 1);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let deltas = motion_coding(&mut br, MvMode::Forward, TypeOfMb::One, 2).unwrap();
        let MotionCodingDeltas::OneMv(d) = deltas else {
            panic!("expected OneMv");
        };
        assert_eq!(d, MotionVectorDelta { dx: 3, dy: 3 });
        let mv = reconstruct_motion_vector(d, 5, -5, 2).unwrap();
        assert_eq!(mv, MotionVector { x: 8, y: -2 });
    }

    #[test]
    fn type_of_mb_distinct_from_intra_returns_none() {
        // Sanity property: a derived_mb_type that has TypeOfMb::None
        // (intra) must yield Ok(None); a derived_mb_type that has
        // TypeOfMb::One or Four must yield Ok(Some) with the matching
        // cardinality. Exhaustive over the 5 variants.
        use crate::macroblock::DerivedMbType;
        let zero_inter_bytes = {
            // 8 zero VLC `1`s = 8 bits = 0xFF; enough to satisfy any
            // 1-MV or 4-MV decode at fcode == 1.
            vec![0xFFu8, 0xFFu8]
        };
        for (ty, expected_len) in [
            (DerivedMbType::Inter, Some(1)),
            (DerivedMbType::InterQ, Some(1)),
            (DerivedMbType::Inter4V, Some(4)),
            (DerivedMbType::Intra, None),
            (DerivedMbType::IntraQ, None),
        ] {
            let mut br = BitReader::new(&zero_inter_bytes);
            let got = decode_p_macroblock_motion_vectors(&mut br, ty, 1).unwrap();
            assert_eq!(got.map(|d| d.as_slice().len()), expected_len, "ty={:?}", ty);
        }
    }

    // ---- §6.2.6 binary-shape transparent_block(j) elision ----

    #[test]
    fn binary_shape_block_opacity_count_helper() {
        assert_eq!(
            BinaryShapeBlockOpacity::ALL_OPAQUE.motion_vector_invocation_count(),
            4
        );
        assert_eq!(
            BinaryShapeBlockOpacity::new([false; 4]).motion_vector_invocation_count(),
            0
        );
        assert_eq!(
            BinaryShapeBlockOpacity::new([true, false, true, false])
                .motion_vector_invocation_count(),
            2
        );
        assert_eq!(
            BinaryShapeBlockOpacity::new([true, true, true, false])
                .motion_vector_invocation_count(),
            3
        );
    }

    #[test]
    fn shape_aware_decode_matches_existing_path_on_all_opaque_mask() {
        // With ALL_OPAQUE, the shape-aware Inter4V routine must consume
        // the same bits and produce the same per-sub-block deltas as
        // decode_p_macroblock_motion_vectors (TypeOfMb::Four). Compare
        // bit position after decode + each slot.
        use crate::macroblock::DerivedMbType;
        // Four MV bodies at fcode == 1: each is two 4-bit codes (h=2 /
        // v=-2 from MVD_TABLE — code 0010 / 0011). Four pairs = 32 bits.
        let mut w = BitWriter::new();
        for _ in 0..4 {
            w.write_bits(0b0010, 4); // h = 2
            w.write_bits(0b0011, 4); // v = -2
        }
        w.align();
        let data = w.buf;

        let mut br_existing = BitReader::new(&data);
        let existing =
            decode_p_macroblock_motion_vectors(&mut br_existing, DerivedMbType::Inter4V, 1)
                .unwrap()
                .unwrap();

        let mut br_new = BitReader::new(&data);
        let new = decode_p_macroblock_motion_vectors_with_shape(
            &mut br_new,
            DerivedMbType::Inter4V,
            1,
            BinaryShapeBlockOpacity::ALL_OPAQUE,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            br_existing.bit_position(),
            br_new.bit_position(),
            "both paths must consume the same bits",
        );
        let lifted = new
            .to_motion_coding_deltas()
            .expect("all four slots populated under ALL_OPAQUE mask");
        assert_eq!(lifted, existing);
    }

    #[test]
    fn shape_aware_decode_elides_transparent_blocks() {
        // Mask: only j=0 (TL) and j=2 (BL) opaque — two MV bodies decoded,
        // j=1 / j=3 slots stay None. Bit consumption must equal exactly
        // two MV bodies' worth (no bits read for the elided slots).
        use crate::macroblock::DerivedMbType;

        let mut w = BitWriter::new();
        // j=0 (TL): h = 2, v = -2 (8 bits at fcode 1).
        w.write_bits(0b0010, 4);
        w.write_bits(0b0011, 4);
        // j=2 (BL): h = -2, v = 2 (codes 0011 / 0010).
        w.write_bits(0b0011, 4);
        w.write_bits(0b0010, 4);
        w.align();
        let data = w.buf;

        let mut br = BitReader::new(&data);
        let opacity = BinaryShapeBlockOpacity::new([true, false, true, false]);
        let got = decode_p_macroblock_motion_vectors_with_shape(
            &mut br,
            DerivedMbType::Inter4V,
            1,
            opacity,
        )
        .unwrap()
        .unwrap();

        // Two slots populated, two elided.
        assert_eq!(got.deltas[0], Some(MotionVectorDelta { dx: 2, dy: -2 }));
        assert_eq!(got.deltas[1], None);
        assert_eq!(got.deltas[2], Some(MotionVectorDelta { dx: -2, dy: 2 }));
        assert_eq!(got.deltas[3], None);

        // Exactly 16 bits consumed (two MV pairs × 8 bits each).
        assert_eq!(br.bit_position(), 16);

        // The lift-to-FourMv conversion must refuse when slots are elided.
        assert!(got.to_motion_coding_deltas().is_none());

        // iter_present yields exactly the two populated entries in order.
        let present: Vec<_> = got.iter_present().collect();
        assert_eq!(present.len(), 2);
        assert_eq!(present[0].0, 0);
        assert_eq!(present[1].0, 2);
    }

    #[test]
    fn shape_aware_decode_all_transparent_consumes_no_bits() {
        // §6.2.6 with every sub-block transparent: the inter4v loop fires
        // zero motion_vector("forward") bodies — no bits are consumed.
        use crate::macroblock::DerivedMbType;
        let data: Vec<u8> = vec![0xAA, 0xBB, 0xCC];
        let mut br = BitReader::new(&data);
        let got = decode_p_macroblock_motion_vectors_with_shape(
            &mut br,
            DerivedMbType::Inter4V,
            1,
            BinaryShapeBlockOpacity::new([false; 4]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(got.deltas, [None, None, None, None]);
        assert_eq!(br.bit_position(), 0);
        assert!(got.to_motion_coding_deltas().is_none());
        assert_eq!(got.iter_present().count(), 0);
    }

    #[test]
    fn shape_aware_decode_routes_non_inter4v_to_none() {
        // §6.2.6 only routes derived_mb_type == 2 through the
        // transparent_block(j) loop. Inter / InterQ / Intra / IntraQ
        // must surface Ok(None) so the caller routes to the single-MV
        // (decode_p_macroblock_motion_vectors) or no-MV path instead —
        // and no bits get consumed regardless of the supplied mask.
        use crate::macroblock::DerivedMbType;
        for ty in [
            DerivedMbType::Inter,
            DerivedMbType::InterQ,
            DerivedMbType::Intra,
            DerivedMbType::IntraQ,
        ] {
            let data: Vec<u8> = vec![0xFF, 0xFF];
            let mut br = BitReader::new(&data);
            let got = decode_p_macroblock_motion_vectors_with_shape(
                &mut br,
                ty,
                1,
                BinaryShapeBlockOpacity::ALL_OPAQUE,
            )
            .unwrap();
            assert!(got.is_none(), "ty={:?}", ty);
            assert_eq!(br.bit_position(), 0, "ty={:?} consumed bits", ty);
        }
    }

    // -----------------------------------------------------------------
    // §6.2.6 field-prediction second-invocation tests.
    // -----------------------------------------------------------------

    #[test]
    fn field_mv_pair_orders_top_then_bottom() {
        // §7.7.2.1: first body = top field, second = bottom field.
        // fcode == 1 → no residuals. Encode top (h=+2, v=-2), bottom
        // (h=0, v=+2): 4+4+1+4 = 13 bits.
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        write_mv_data(&mut w, -2);
        write_mv_data(&mut w, 0);
        write_mv_data(&mut w, 2);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_field_motion_vector_pair(&mut br, MvMode::Forward, 1).unwrap();
        assert_eq!(got.top, MotionVectorDelta { dx: 2, dy: -2 });
        assert_eq!(got.bottom, MotionVectorDelta { dx: 0, dy: 2 });
        assert_eq!(br.bit_position(), 13);
    }

    #[test]
    fn field_mv_pair_fcode_two_reads_residuals_in_both_bodies() {
        // vop_fcode == 2 → r_size = 1; residual read per non-zero
        // mv_data in BOTH bodies (the pair shares the direction's
        // fcode). Top: h_data=2 + res 1 → (2-1)*2+1+1 = +4; v_data=0
        // (no residual) → 0. Bottom: h_data=-2 + res 0 → -3;
        // v_data=2 + res 1 → +4.
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        w.write_bits(1, 1);
        write_mv_data(&mut w, 0);
        write_mv_data(&mut w, -2);
        w.write_bits(0, 1);
        write_mv_data(&mut w, 2);
        w.write_bits(1, 1);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_field_motion_vector_pair(&mut br, MvMode::Backward, 2).unwrap();
        assert_eq!(got.top, MotionVectorDelta { dx: 4, dy: 0 });
        assert_eq!(got.bottom, MotionVectorDelta { dx: -3, dy: 4 });
        // 4+1+1 + 4+1+4+1 = 16 bits.
        assert_eq!(br.bit_position(), 16);
    }

    #[test]
    fn field_mv_pair_rejects_direct_mode_without_consuming_bits() {
        // §6.2.6 has no second invocation on the direct line; §6.2.6.3
        // never codes field_prediction when mb_type == "1".
        let data: Vec<u8> = vec![0xFF, 0xFF];
        let mut br = BitReader::new(&data);
        let got = decode_field_motion_vector_pair(&mut br, MvMode::Direct, 1);
        assert_eq!(
            got.unwrap_err(),
            MotionParseError::InvalidFieldPredictionContext
        );
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn field_mv_pair_truncated_mid_second_body() {
        // Top body complete (h=+2, v=-2 → 8 bits), bottom h_data an
        // 8-bit code (mv_data -6) so the buffer ends exactly on a byte
        // boundary — the bottom v_data then has zero bits left →
        // Truncated.
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        write_mv_data(&mut w, -2);
        write_mv_data(&mut w, -6);
        w.align();
        let bytes = w.buf;
        assert_eq!(bytes.len(), 2);
        let mut br = BitReader::new(&bytes);
        let got = decode_field_motion_vector_pair(&mut br, MvMode::Forward, 1);
        assert_eq!(got.unwrap_err(), MotionParseError::Truncated);
    }

    #[test]
    fn p_mb_interlaced_inter_field_prediction_decodes_pair() {
        // P-VOP Inter MB, field_prediction == 1 → the §6.2.6 second
        // invocation fires: two forward bodies, top then bottom.
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        write_mv_data(&mut w, -2);
        write_mv_data(&mut w, 0);
        write_mv_data(&mut w, 2);
        write_mv_data(&mut w, 2);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got =
            decode_p_macroblock_motion_vectors_interlaced(&mut br, DerivedMbType::Inter, 1, true)
                .unwrap()
                .unwrap();
        assert_eq!(
            got,
            PMbMotionVectors::Field(FieldMvPair {
                top: MotionVectorDelta { dx: -2, dy: 0 },
                bottom: MotionVectorDelta { dx: 2, dy: 2 },
            })
        );
        assert_eq!(br.bit_position(), 13);
    }

    #[test]
    fn p_mb_interlaced_frame_matches_progressive_walker() {
        // field_prediction == 0 → identical decode to the round-37
        // progressive walker on the same bytes.
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        write_mv_data(&mut w, 2);
        write_mv_data(&mut w, -2);
        w.align();
        let bytes = w.buf;

        let mut br_a = BitReader::new(&bytes);
        let interlaced = decode_p_macroblock_motion_vectors_interlaced(
            &mut br_a,
            DerivedMbType::InterQ,
            1,
            false,
        )
        .unwrap()
        .unwrap();
        let mut br_b = BitReader::new(&bytes);
        let progressive = decode_p_macroblock_motion_vectors(&mut br_b, DerivedMbType::InterQ, 1)
            .unwrap()
            .unwrap();
        assert_eq!(interlaced, PMbMotionVectors::Frame(progressive));
        assert_eq!(br_a.bit_position(), br_b.bit_position());
    }

    #[test]
    fn p_mb_interlaced_inter4v_frame_decodes_four_bodies() {
        // inter4v never field-predicts; with field_prediction == 0 the
        // four-body §6.2.5 motion_coding path runs unchanged.
        use crate::macroblock::DerivedMbType;
        let mut w = BitWriter::new();
        for _ in 0..8 {
            write_mv_data(&mut w, 0);
        }
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_p_macroblock_motion_vectors_interlaced(
            &mut br,
            DerivedMbType::Inter4V,
            1,
            false,
        )
        .unwrap()
        .unwrap();
        let zeros = MotionVectorDelta { dx: 0, dy: 0 };
        assert_eq!(
            got,
            PMbMotionVectors::Frame(MotionCodingDeltas::FourMv([zeros; 4]))
        );
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn p_mb_interlaced_rejects_field_prediction_on_inter4v_and_intra() {
        // §6.2.6.3 only codes field_prediction when derived_mb_type < 2
        // — a true flag with inter4v / intra rows is a caller bug and
        // must not consume bits.
        use crate::macroblock::DerivedMbType;
        for ty in [
            DerivedMbType::Inter4V,
            DerivedMbType::Intra,
            DerivedMbType::IntraQ,
        ] {
            let data: Vec<u8> = vec![0xFF, 0xFF];
            let mut br = BitReader::new(&data);
            let got = decode_p_macroblock_motion_vectors_interlaced(&mut br, ty, 1, true);
            assert_eq!(
                got.unwrap_err(),
                MotionParseError::InvalidFieldPredictionContext,
                "ty={:?}",
                ty
            );
            assert_eq!(br.bit_position(), 0, "ty={:?} consumed bits", ty);
        }
    }

    #[test]
    fn p_mb_interlaced_intra_frame_yields_none() {
        // Intra rows carry no MV body regardless of interlacing.
        use crate::macroblock::DerivedMbType;
        let data: Vec<u8> = vec![0xFF, 0xFF];
        let mut br = BitReader::new(&data);
        let got =
            decode_p_macroblock_motion_vectors_interlaced(&mut br, DerivedMbType::Intra, 1, false)
                .unwrap();
        assert!(got.is_none());
        assert_eq!(br.bit_position(), 0);
    }
}

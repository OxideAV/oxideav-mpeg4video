//! §7.4.3 spatial DC / AC predictor for intra macroblocks.
//!
//! This module implements the four sub-sections of ISO/IEC 14496-2:2004
//! §7.4.3:
//!
//! * §7.4.3.1 — DC prediction direction. Three neighbours `A` (left),
//!   `B` (above-left), and `C` (above) supply their already-decoded
//!   inverse-quantised DC values `F[0][0]`. The predictor direction is
//!   `C` (above) when `|FA - FB| < |FB - FC|`, otherwise `A` (left).
//!   Neighbours that fall outside the VOP / video packet, or that
//!   belong to a non-intra macroblock, are treated as having
//!   `F[0][0] = 2^(bits_per_pixel + 2)`.
//! * §7.4.3.2 — Adaptive DC coefficient prediction. The chosen
//!   neighbour's `F[0][0]` is divided by `dc_scaler` (Table 7-1,
//!   piece-wise linear in `quantiser_scale`, with separate Type 1
//!   (luminance) and Type 2 (chrominance) formulas) and added to the
//!   stream-side `PQFX[0][0]` to yield `QFX[0][0]`.
//! * §7.4.3.3 — Adaptive AC coefficient prediction. When
//!   `ac_pred_flag == 1`, the first row (predictor C) or first column
//!   (predictor A) of the chosen neighbour, scaled by the ratio of the
//!   neighbour's `QpA`/`QpC` to the current block's `QpX`, is added to
//!   `PQFX`'s first row or column.
//! * §7.4.3.4 — Saturation of `QF[v][u]` to the range `[-2048, 2047]`.
//!
//! All numeric values are sourced from ISO/IEC 14496-2:2004 (3rd
//! edition):
//!
//! * §7.4.3 page 242 — the four sub-sections and Figure 7-5 (the three
//!   neighbours `A`, `B`, `C`).
//! * §7.4.3.1 page 242 — the default-neighbour rule
//!   `F[0][0] = 2^(bits_per_pixel + 2)`.
//! * Table 7-1 page 246 — the non-linear `dc_scaler` table.
//! * §7.4.3.4 page 243 — the saturation range `[-2048, 2047]`.

use crate::texture::DcComponent;

/// One of the three §7.4.3.1 neighbour positions used to derive the
/// DC and AC prediction. Positions follow Figure 7-5:
///
/// ```text
///        B   C
///        A   X
/// ```
///
/// where `X` is the block being decoded. `A` is the horizontally
/// adjacent (left) block, `C` is the vertically adjacent (above) block,
/// and `B` is the diagonal (above-left) block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighbourPosition {
    /// Block `A` — the horizontally adjacent (left) block.
    Left,
    /// Block `B` — the above-left (diagonal) block.
    AboveLeft,
    /// Block `C` — the vertically adjacent (above) block.
    Above,
}

/// The §7.4.3.1 DC-prediction direction chosen for one block.
///
/// Re-exported as [`crate::scan::DcPredictionDirection`] via the
/// existing scan-module enum; this module owns the §7.4.3.1
/// selection logic that produces the value.
pub use crate::scan::DcPredictionDirection;

/// The data carried per neighbouring block when running the §7.4.3
/// predictor. The two inverse-quantised first-row / first-column AC
/// coefficients are stored separately so the caller can supply just
/// the row for predictor `C` and just the column for predictor `A`
/// without paying for the unused side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourBlock {
    /// The inverse-quantised DC value `F[0][0]` of the neighbour.
    pub dc: i32,
    /// The neighbour's quantiser scale `Qp` used when its AC
    /// coefficients were quantised. Required by §7.4.3.3.
    pub qp: u32,
    /// The first row `QF[0][1..=7]` of the neighbour, used when this
    /// block is the predictor for the current block's `C`-direction
    /// AC prediction. `None` if the neighbour is outside the VOP /
    /// video packet — §7.4.3.3 says "all the prediction coefficients
    /// of that block are assumed to be zero" in that case.
    pub first_row: Option<[i32; 7]>,
    /// The first column `QF[1..=7][0]` of the neighbour, used when
    /// this block is the predictor for the current block's
    /// `A`-direction AC prediction. Same out-of-bounds rule as
    /// `first_row`.
    pub first_column: Option<[i32; 7]>,
}

/// Compute the §7.4.3.1 default DC value used when a neighbour is
/// outside the VOP / video packet, or when it belongs to a non-intra
/// macroblock.
///
/// `F[0][0]` is assumed to take the value `2^(bits_per_pixel + 2)`.
/// `bits_per_pixel` is the §6.3.3 VOL field with a default of `8`
/// (per the `not_8_bit == 0` path) and a §6.3.3 valid range of
/// `4..=12`, so the result fits comfortably in an `i32`.
///
/// # Panics
///
/// Panics if `bits_per_pixel` exceeds `29`, which would overflow the
/// shift; this is well outside the §6.3.3 valid range.
pub fn default_neighbour_dc(bits_per_pixel: u32) -> i32 {
    assert!(
        bits_per_pixel < 30,
        "bits_per_pixel = {bits_per_pixel} is out of range"
    );
    1i32 << (bits_per_pixel + 2)
}

/// Compute the Table 7-1 `dc_scaler` value for the given block
/// component and quantiser-scale.
///
/// The table is piece-wise linear in `quantiser_scale`:
///
/// | `quantiser_scale` | Luminance (Type 1) | Chrominance (Type 2) |
/// |-------------------|--------------------|----------------------|
/// | `1..=4`           | `8`                | `8`                  |
/// | `5..=8`           | `2 * qs`           | `(qs + 13) / 2`      |
/// | `9..=24`          | `qs + 8`           | `(qs + 13) / 2`      |
/// | `>= 25`           | `2 * qs - 16`      | `qs - 6`             |
///
/// The chrominance row in Table 7-1 has a single formula
/// `(qs + 13) / 2` spanning the `5..=24` range; the layout merges the
/// `5..=8` and `9..=24` columns into one cell. The text immediately
/// preceding the table confirms the per-component split into Type 1
/// (luminance) and Type 2 (chrominance) blocks; the `short_video_header
/// == 1` path is handled separately by the §7.4.1.1 caller (fixed
/// `dc_scaler = 8`).
///
/// # Panics
///
/// Panics on `quantiser_scale == 0`; §7.4.4.2 forbids a zero scale
/// (the valid range is `1..=2^quant_precision - 1`).
pub fn dc_scaler(component: DcComponent, quantiser_scale: u32) -> u32 {
    assert!(quantiser_scale > 0, "quantiser_scale must be > 0");
    match component {
        DcComponent::Luminance => match quantiser_scale {
            1..=4 => 8,
            5..=8 => 2 * quantiser_scale,
            9..=24 => quantiser_scale + 8,
            _ => 2 * quantiser_scale - 16,
        },
        DcComponent::Chrominance => match quantiser_scale {
            1..=4 => 8,
            5..=24 => (quantiser_scale + 13) / 2,
            _ => quantiser_scale - 6,
        },
    }
}

/// Apply the §7.4.3.1 direction-selection rule.
///
/// ```text
/// if (|FA[0][0] - FB[0][0]| < |FB[0][0] - FC[0][0]|)
///     predict from block C        // FromAbove
/// else
///     predict from block A        // FromLeft
/// ```
///
/// `fa`, `fb`, `fc` are the inverse-quantised DC values of blocks
/// `A` (left), `B` (above-left), `C` (above). Use
/// [`default_neighbour_dc`] to supply the value for any neighbour
/// outside the VOP / video packet or in a non-intra macroblock.
pub fn select_dc_direction(fa: i32, fb: i32, fc: i32) -> DcPredictionDirection {
    let d_ab = (fa - fb).unsigned_abs();
    let d_bc = (fb - fc).unsigned_abs();
    if d_ab < d_bc {
        DcPredictionDirection::FromAbove
    } else {
        DcPredictionDirection::FromLeft
    }
}

/// Reconstruct the §7.4.3.2 quantised DC value `QFX[0][0]` for one
/// block.
///
/// ```text
/// if (predict from block C)
///     QFX[0][0] = PQFX[0][0] + FC[0][0] // dc_scaler
/// else
///     QFX[0][0] = PQFX[0][0] + FA[0][0] // dc_scaler
/// ```
///
/// `pqfx_dc` is the bitstream-side decoded differential placed at scan
/// position 0 (from [`crate::texture::decode_intra_dc`]). `fa_dc` /
/// `fc_dc` are the inverse-quantised DC values of the two horizontal /
/// vertical neighbours (use [`default_neighbour_dc`] when a neighbour
/// is unavailable). `dc_scaler_x` is the result of [`dc_scaler`] for
/// the *current* block `X`.
///
/// The `//` operator in the spec (§4.1 arithmetic operators) denotes
/// integer division with rounding to the *nearest* integer,
/// half-integer values away from zero (`3//2 == 2`, `-3//2 == -2`) —
/// see [`div_round_half_away`].
///
/// # Panics
///
/// Panics on `dc_scaler_x == 0`; [`dc_scaler`] never returns zero.
pub fn predict_intra_dc(
    pqfx_dc: i32,
    direction: DcPredictionDirection,
    fa_dc: i32,
    fc_dc: i32,
    dc_scaler_x: u32,
) -> i32 {
    assert!(dc_scaler_x > 0, "dc_scaler_x must be > 0");
    let chosen = match direction {
        DcPredictionDirection::FromAbove => fc_dc,
        DcPredictionDirection::FromLeft => fa_dc,
    };
    pqfx_dc + div_round_half_away(chosen, dc_scaler_x as i32)
}

/// §4.1 `//` operator: integer division with rounding to the nearest
/// integer, half-integer values rounded away from zero (`3//2 == 2`,
/// `-3//2 == -2`). `d` must be positive.
#[inline]
fn div_round_half_away(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// Reconstruct the §7.4.3.3 first column of quantised AC coefficients
/// `QFX[1..=7][0]` when the §7.4.3.1 direction is `A` (left).
///
/// ```text
/// QFX[v][0] = PQFX[v][0] + (QFA[v][0] * QpA) // QpX     v = 1 to 7
/// ```
///
/// If `qfa_col` is `None` — block `A` is outside the VOP / video
/// packet — all prediction coefficients are taken as zero and the
/// function returns `pqfx_col` unchanged (§7.4.3.3).
///
/// # Panics
///
/// Panics on `qp_x == 0`.
pub fn predict_intra_ac_column(
    pqfx_col: [i32; 7],
    qfa_col: Option<[i32; 7]>,
    qp_a: u32,
    qp_x: u32,
) -> [i32; 7] {
    assert!(qp_x > 0, "qp_x must be > 0");
    let Some(qfa) = qfa_col else {
        return pqfx_col;
    };
    let qp_a = qp_a as i32;
    let qp_x = qp_x as i32;
    let mut out = [0i32; 7];
    for v in 0..7 {
        out[v] = pqfx_col[v] + div_round_half_away(qfa[v] * qp_a, qp_x);
    }
    out
}

/// Reconstruct the §7.4.3.3 first row of quantised AC coefficients
/// `QFX[0][1..=7]` when the §7.4.3.1 direction is `C` (above).
///
/// ```text
/// QFX[0][u] = PQFX[0][u] + (QFC[0][u] * QpC) // QpX     u = 1 to 7
/// ```
///
/// Same out-of-bounds rule as [`predict_intra_ac_column`]: if
/// `qfc_row` is `None`, returns `pqfx_row` unchanged.
///
/// # Panics
///
/// Panics on `qp_x == 0`.
pub fn predict_intra_ac_row(
    pqfx_row: [i32; 7],
    qfc_row: Option<[i32; 7]>,
    qp_c: u32,
    qp_x: u32,
) -> [i32; 7] {
    assert!(qp_x > 0, "qp_x must be > 0");
    let Some(qfc) = qfc_row else {
        return pqfx_row;
    };
    let qp_c = qp_c as i32;
    let qp_x = qp_x as i32;
    let mut out = [0i32; 7];
    for u in 0..7 {
        out[u] = pqfx_row[u] + div_round_half_away(qfc[u] * qp_c, qp_x);
    }
    out
}

/// Apply the §7.4.3.4 saturation: clamp `QF[v][u]` to `[-2048, 2047]`.
///
/// "The quantised coefficients resulting from the DC and AC Prediction
/// are saturated to lie in the range \[-2048, 2047\]."
pub fn saturate_qf(value: i32) -> i32 {
    value.clamp(-2048, 2047)
}

/// Apply [`saturate_qf`] to every cell of an 8×8 block in place.
pub fn saturate_block(block: &mut [[i32; 8]; 8]) {
    for row in block.iter_mut() {
        for cell in row.iter_mut() {
            *cell = saturate_qf(*cell);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- default_neighbour_dc -----

    #[test]
    fn default_dc_at_8_bpp() {
        // 2^(8+2) = 1024.
        assert_eq!(default_neighbour_dc(8), 1024);
    }

    #[test]
    fn default_dc_at_4_bpp_minimum() {
        // §6.3.3 valid range minimum: 2^(4+2) = 64.
        assert_eq!(default_neighbour_dc(4), 64);
    }

    #[test]
    fn default_dc_at_12_bpp_maximum() {
        // §6.3.3 valid range maximum: 2^(12+2) = 16384.
        assert_eq!(default_neighbour_dc(12), 16384);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn default_dc_overflows_for_excessive_bpp() {
        let _ = default_neighbour_dc(31);
    }

    // ----- dc_scaler — Table 7-1 -----

    #[test]
    fn dc_scaler_luminance_boundary_rows() {
        // Quantiser 1..=4 -> 8 flat.
        for qs in 1..=4 {
            assert_eq!(dc_scaler(DcComponent::Luminance, qs), 8, "qs = {qs}");
        }
        // Quantiser 5..=8 -> 2*qs: 10, 12, 14, 16.
        assert_eq!(dc_scaler(DcComponent::Luminance, 5), 10);
        assert_eq!(dc_scaler(DcComponent::Luminance, 6), 12);
        assert_eq!(dc_scaler(DcComponent::Luminance, 7), 14);
        assert_eq!(dc_scaler(DcComponent::Luminance, 8), 16);
        // Quantiser 9..=24 -> qs + 8: 17, ..., 32.
        assert_eq!(dc_scaler(DcComponent::Luminance, 9), 17);
        assert_eq!(dc_scaler(DcComponent::Luminance, 24), 32);
        // Quantiser >=25 -> 2*qs - 16: 34, ..., 46 at qs=31.
        assert_eq!(dc_scaler(DcComponent::Luminance, 25), 34);
        assert_eq!(dc_scaler(DcComponent::Luminance, 31), 46);
    }

    #[test]
    fn dc_scaler_luminance_is_monotone() {
        // Sanity-check: dc_scaler grows monotonically across the full
        // 5-bit quantiser range (1..=31, the largest range for
        // quant_precision == 5).
        let mut prev = 0u32;
        for qs in 1..=31 {
            let v = dc_scaler(DcComponent::Luminance, qs);
            assert!(v >= prev, "qs = {qs}: dc_scaler = {v} < prev = {prev}");
            prev = v;
        }
    }

    #[test]
    fn dc_scaler_chrominance_boundary_rows() {
        // Quantiser 1..=4 -> 8 flat.
        for qs in 1..=4 {
            assert_eq!(dc_scaler(DcComponent::Chrominance, qs), 8, "qs = {qs}");
        }
        // Quantiser 5..=24 -> (qs + 13) / 2.
        assert_eq!(dc_scaler(DcComponent::Chrominance, 5), 9); // (5+13)/2 = 9
        assert_eq!(dc_scaler(DcComponent::Chrominance, 6), 9); // (6+13)/2 = 9 (truncated)
        assert_eq!(dc_scaler(DcComponent::Chrominance, 7), 10); // (7+13)/2 = 10
        assert_eq!(dc_scaler(DcComponent::Chrominance, 8), 10); // (8+13)/2 = 10
        assert_eq!(dc_scaler(DcComponent::Chrominance, 24), 18); // (24+13)/2 = 18
                                                                 // Quantiser >=25 -> qs - 6.
        assert_eq!(dc_scaler(DcComponent::Chrominance, 25), 19);
        assert_eq!(dc_scaler(DcComponent::Chrominance, 31), 25);
    }

    #[test]
    #[should_panic(expected = "quantiser_scale must be > 0")]
    fn dc_scaler_rejects_zero_quantiser() {
        let _ = dc_scaler(DcComponent::Luminance, 0);
    }

    // ----- select_dc_direction -----

    #[test]
    fn dc_direction_horizontal_gradient_picks_above() {
        // |FA - FB| small (= 0), |FB - FC| large (= 100). The
        // horizontal differences are small, so the DC value is
        // changing along the C direction; predict from C.
        //
        // (Per §7.4.3.1: smaller |FA - FB| -> predict from C.)
        let dir = select_dc_direction(50, 50, 150);
        assert_eq!(dir, DcPredictionDirection::FromAbove);
    }

    #[test]
    fn dc_direction_vertical_gradient_picks_left() {
        // |FA - FB| large (= 100), |FB - FC| small (= 0). The vertical
        // differences are small; predict from A.
        let dir = select_dc_direction(150, 50, 50);
        assert_eq!(dir, DcPredictionDirection::FromLeft);
    }

    #[test]
    fn dc_direction_tie_picks_left() {
        // The rule uses strict `<`; equal differences -> "else" -> A.
        let dir = select_dc_direction(50, 50, 50);
        assert_eq!(dir, DcPredictionDirection::FromLeft);
    }

    #[test]
    fn dc_direction_uses_absolute_differences() {
        // |FA - FB| = 10, |FB - FC| = 30. Smaller horizontal diff -> C.
        let dir = select_dc_direction(60, 50, 80);
        assert_eq!(dir, DcPredictionDirection::FromAbove);
        // Mirror with negative values.
        let dir = select_dc_direction(-60, -50, -80);
        assert_eq!(dir, DcPredictionDirection::FromAbove);
    }

    #[test]
    fn dc_direction_default_neighbours_yield_left() {
        // When all neighbours fall back to default_neighbour_dc (so
        // FA == FB == FC), the tie rule selects A.
        let d = default_neighbour_dc(8);
        let dir = select_dc_direction(d, d, d);
        assert_eq!(dir, DcPredictionDirection::FromLeft);
    }

    // ----- predict_intra_dc -----

    #[test]
    fn predict_dc_from_above_divides_fc_by_dc_scaler() {
        // PQFX[0][0] = 5; FC = 64; dc_scaler = 8 -> 5 + 64/8 = 5 + 8 = 13.
        // fa = 999 is ignored on the FromAbove path.
        let qfx00 = predict_intra_dc(5, DcPredictionDirection::FromAbove, 999, 64, 8);
        assert_eq!(qfx00, 13);
    }

    #[test]
    fn predict_dc_from_left_divides_fa_by_dc_scaler() {
        // PQFX[0][0] = -2; FA = 80; dc_scaler = 10 -> -2 + 80/10 = -2 + 8 = 6.
        // fc = 999 is ignored on the FromLeft path.
        let qfx00 = predict_intra_dc(-2, DcPredictionDirection::FromLeft, 80, 999, 10);
        assert_eq!(qfx00, 6);
    }

    #[test]
    fn predict_dc_rounds_to_nearest_half_away_from_zero() {
        // §4.1 `//`: 5 // 8 rounds to 1 (0.625 → nearest is 1);
        // -5 // 8 rounds to -1. 3 // 8 rounds to 0 (0.375 → nearest 0).
        // The exact half 4 // 8 rounds away from zero to 1 (and -4 // 8
        // to -1).
        assert_eq!(
            predict_intra_dc(0, DcPredictionDirection::FromLeft, 5, 0, 8),
            1
        );
        assert_eq!(
            predict_intra_dc(0, DcPredictionDirection::FromLeft, -5, 0, 8),
            -1
        );
        assert_eq!(
            predict_intra_dc(0, DcPredictionDirection::FromLeft, 3, 0, 8),
            0
        );
        assert_eq!(
            predict_intra_dc(0, DcPredictionDirection::FromLeft, 4, 0, 8),
            1
        );
        assert_eq!(
            predict_intra_dc(0, DcPredictionDirection::FromLeft, -4, 0, 8),
            -1
        );
    }

    #[test]
    #[should_panic(expected = "dc_scaler_x must be > 0")]
    fn predict_dc_rejects_zero_scaler() {
        let _ = predict_intra_dc(0, DcPredictionDirection::FromAbove, 0, 0, 0);
    }

    // ----- predict_intra_ac_column -----

    #[test]
    fn predict_ac_column_scales_by_qp_ratio() {
        // QpA == QpX -> column passes through unchanged on top of pqfx_col.
        let pqfx_col = [1, 0, 0, 0, 0, 0, 0];
        let qfa_col = [4, 8, 12, 16, 20, 24, 28];
        let out = predict_intra_ac_column(pqfx_col, Some(qfa_col), 5, 5);
        // (q * 5) / 5 == q, so out[v] = pqfx_col[v] + qfa_col[v].
        assert_eq!(out, [5, 8, 12, 16, 20, 24, 28]);
    }

    #[test]
    fn predict_ac_column_scales_with_different_qp() {
        // QpA = 10, QpX = 5 -> doubled. PQFX zero -> out == 2*qfa.
        let pqfx_col = [0; 7];
        let qfa_col = [3, 0, 1, 7, -2, 0, 11];
        let out = predict_intra_ac_column(pqfx_col, Some(qfa_col), 10, 5);
        // (3 * 10) / 5 = 6, (1 * 10) / 5 = 2, (7 * 10) / 5 = 14, etc.
        assert_eq!(out, [6, 0, 2, 14, -4, 0, 22]);
    }

    #[test]
    fn predict_ac_column_rounds_to_nearest() {
        // §4.1 `//` with QpA = 3, QpX = 5:
        // (1*3)//5  = 3//5   → 1  (0.6 nearest 1);
        // (-1*3)//5 = -3//5  → -1;
        // (2*3)//5  = 6//5   → 1  (1.2 nearest 1);
        // (-2*3)//5 = -6//5  → -1;
        // (6*3)//5  = 18//5  → 4  (3.6 nearest 4);
        // (-6*3)//5 = -18//5 → -4.
        let pqfx_col = [0; 7];
        let qfa_col = [1, -1, 2, -2, 6, -6, 0];
        let out = predict_intra_ac_column(pqfx_col, Some(qfa_col), 3, 5);
        assert_eq!(out, [1, -1, 1, -1, 4, -4, 0]);
    }

    #[test]
    fn predict_ac_column_rounds_exact_half_away_from_zero() {
        // QpA = 1, QpX = 2: (1*1)//2 = 0.5 → 1; (-1*1)//2 = -0.5 → -1;
        // (3*1)//2 = 1.5 → 2; (-3*1)//2 = -1.5 → -2.
        let pqfx_col = [0; 7];
        let qfa_col = [1, -1, 3, -3, 0, 2, -2];
        let out = predict_intra_ac_column(pqfx_col, Some(qfa_col), 1, 2);
        assert_eq!(out, [1, -1, 2, -2, 0, 1, -1]);
    }

    #[test]
    fn predict_ac_column_missing_neighbour_returns_pqfx() {
        // §7.4.3.3: out-of-VOP neighbour -> prediction coefficients zero
        // -> output is pqfx_col unchanged.
        let pqfx_col = [7, -3, 2, 0, 1, -1, 9];
        let out = predict_intra_ac_column(pqfx_col, None, 99, 1);
        assert_eq!(out, pqfx_col);
    }

    #[test]
    #[should_panic(expected = "qp_x must be > 0")]
    fn predict_ac_column_rejects_zero_qp_x() {
        let _ = predict_intra_ac_column([0; 7], Some([0; 7]), 1, 0);
    }

    // ----- predict_intra_ac_row -----

    #[test]
    fn predict_ac_row_scales_by_qp_ratio() {
        let pqfx_row = [0, 1, 0, 0, 0, 0, 0];
        let qfc_row = [4, 8, 12, 16, 20, 24, 28];
        let out = predict_intra_ac_row(pqfx_row, Some(qfc_row), 7, 7);
        // (q * 7) / 7 == q, so out[u] = pqfx_row[u] + qfc_row[u].
        assert_eq!(out, [4, 9, 12, 16, 20, 24, 28]);
    }

    #[test]
    fn predict_ac_row_missing_neighbour_returns_pqfx() {
        let pqfx_row = [9, 8, 7, 6, 5, 4, 3];
        let out = predict_intra_ac_row(pqfx_row, None, 99, 1);
        assert_eq!(out, pqfx_row);
    }

    #[test]
    #[should_panic(expected = "qp_x must be > 0")]
    fn predict_ac_row_rejects_zero_qp_x() {
        let _ = predict_intra_ac_row([0; 7], Some([0; 7]), 1, 0);
    }

    // ----- saturate_qf / saturate_block -----

    #[test]
    fn saturate_clamps_to_spec_range() {
        assert_eq!(saturate_qf(0), 0);
        assert_eq!(saturate_qf(2047), 2047);
        assert_eq!(saturate_qf(-2048), -2048);
        assert_eq!(saturate_qf(2048), 2047);
        assert_eq!(saturate_qf(-2049), -2048);
        assert_eq!(saturate_qf(i32::MAX), 2047);
        assert_eq!(saturate_qf(i32::MIN), -2048);
    }

    #[test]
    fn saturate_block_applies_clamp_to_every_cell() {
        let mut block = [[0i32; 8]; 8];
        block[0][0] = 3000;
        block[7][7] = -3000;
        block[3][4] = 42;
        saturate_block(&mut block);
        assert_eq!(block[0][0], 2047);
        assert_eq!(block[7][7], -2048);
        assert_eq!(block[3][4], 42);
    }

    // ----- NeighbourBlock smoke test -----

    #[test]
    fn neighbour_block_carries_dc_qp_and_optional_sides() {
        let nb = NeighbourBlock {
            dc: 128,
            qp: 5,
            first_row: Some([1, 2, 3, 4, 5, 6, 7]),
            first_column: None,
        };
        assert_eq!(nb.dc, 128);
        assert_eq!(nb.qp, 5);
        assert_eq!(nb.first_row.unwrap()[0], 1);
        assert!(nb.first_column.is_none());
    }

    #[test]
    fn neighbour_position_variants_are_distinct() {
        let a = NeighbourPosition::Left;
        let b = NeighbourPosition::AboveLeft;
        let c = NeighbourPosition::Above;
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // ----- integration: a full DC predictor pass on one block -----

    #[test]
    fn full_dc_predictor_pass_matches_spec_example() {
        // Scenario: Luminance block X with quantiser_scale = 5 (so
        // dc_scaler(X) = 2*5 = 10). Neighbours:
        //   FA = 100, FB = 100, FC = 200.
        // Direction: |FA-FB| = 0 < |FB-FC| = 100 -> predict from C.
        // Bitstream-side differential PQFX[0][0] = 3.
        // QFX[0][0] = 3 + FC / dc_scaler = 3 + 200 / 10 = 3 + 20 = 23.
        let dc_x = dc_scaler(DcComponent::Luminance, 5);
        assert_eq!(dc_x, 10);
        let dir = select_dc_direction(100, 100, 200);
        assert_eq!(dir, DcPredictionDirection::FromAbove);
        let qfx00 = predict_intra_dc(3, dir, 100, 200, dc_x);
        assert_eq!(qfx00, 23);
    }

    #[test]
    fn full_dc_predictor_pass_from_left_with_chrominance() {
        // Scenario: Chrominance block, quantiser_scale = 7
        // -> dc_scaler = (7+13)/2 = 10. Neighbours FA=200, FB=100, FC=110.
        // |FA-FB| = 100, |FB-FC| = 10 -> "else" -> predict from A.
        // PQFX[0][0] = -1; QFX[0][0] = -1 + 200/10 = -1 + 20 = 19.
        let dc_x = dc_scaler(DcComponent::Chrominance, 7);
        assert_eq!(dc_x, 10);
        let dir = select_dc_direction(200, 100, 110);
        assert_eq!(dir, DcPredictionDirection::FromLeft);
        let qfx00 = predict_intra_dc(-1, dir, 200, 110, dc_x);
        assert_eq!(qfx00, 19);
    }
}

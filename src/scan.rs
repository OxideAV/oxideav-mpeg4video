//! §7.4.2 inverse scan — convert the one-dimensional decoded coefficient
//! stream `QFS[n]` into a two-dimensional `PQF[v][u]` 8×8 block.
//!
//! The §7.4.1.2 EVENT loop produces a sequence of `(LAST, RUN, LEVEL)`
//! triples. Expanding each triple into a position in the
//! zigzag-ordered 64-coefficient array `QFS[n]` is straightforward
//! (`run` zeros, then `level`, repeated until `LAST == 1`). §7.4.2
//! then maps `QFS[n]` into the two-dimensional `PQF[v][u]` using one of
//! three scan patterns shown in Figure 7-4.
//!
//! ## Scan-pattern selection (§7.4.2)
//!
//! * Non-intra blocks → always [`ScanType::Zigzag`].
//! * Intra blocks with `ac_pred_flag == 0` → [`ScanType::Zigzag`] for
//!   every block in the macroblock.
//! * Intra blocks with `ac_pred_flag == 1`: per-block. The §7.4.3.1
//!   DC-prediction direction picks the scan (perpendicular to the
//!   prediction edge):
//!   * Prediction from block A (horizontally adjacent, "left") →
//!     [`ScanType::AlternateVertical`] for the current block.
//!   * Prediction from block C (vertically adjacent, "above") →
//!     [`ScanType::AlternateHorizontal`] for the current block.
//!
//! See [`select_scan_type`] for the encoded rule. Predictor gathering
//! and the DC-direction decision themselves remain later-round work;
//! once those land, the caller threads their direction into
//! `select_scan_type` to pick the right table.
//!
//! ## The §7.4.2 modified inverse scan (`sadct_disable == 0`)
//!
//! For a non-rectangular VOP with `sadct_disable == 0`, an 8×8 block
//! whose number of opaque pels is below 64 is reconstructed with the
//! shape-adaptive DCT (SA-DCT). The number of SA-DCT coefficients in
//! such a block equals `opaque_pels`, and the coefficients are *not*
//! distributed over the whole 8×8 grid — each row `v` only holds
//! `coeff_width[v]` coefficients, packed against the left edge.
//! §7.4.2 therefore replaces the textbook scan with a `coeff_width`
//! aware variant that walks the chosen scan path, writes a decoded
//! coefficient only at positions `u < coeff_width[v]`, and forces
//! every other position to zero (NOTE 1 — a stray non-zero at an
//! SA-DCT-undefined position would confuse subsequent AC prediction).
//!
//! The auxiliary `coeff_width[v]` array and `opaque_pels` total are
//! derived from the decoded binary shape `f_shape[y][x]` per Annex A
//! §A.3.2 step I-S1 — see [`ShapeParams::from_shape`]. The modified
//! scan itself is [`modified_inverse_scan`]; [`events_to_pqf_sadct`]
//! is the one-call convenience.
//!
//! ## Out of scope (this round)
//!
//! * The §7.4.3 spatial DC/AC predictor that supplies the per-block
//!   prediction direction. `select_scan_type` is parameterised on a
//!   pre-resolved direction so it doesn't depend on the missing
//!   gathering.
//! * The inverse SA-DCT transform itself (§7.3.5 / Annex A §A.3.2
//!   steps I-S2..I-S4). This module supplies the `PQF[v][u]` layout
//!   and the `coeff_width[]` / `opaque_pels` shape parameters that
//!   that transform consumes; the transform body is later-round work.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §7.4.2 — the inverse-scan algorithm, the `inv_scan_u[scan_type]`
//!   / `inv_scan_v[scan_type]` arrays, the
//!   `PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] = QFS[n]`
//!   loop, the per-block scan-selection rule keyed on the
//!   `ac_pred_flag` and the §7.4.3.1 prediction direction, and the
//!   `sadct_disable == 0` modified-scan pseudocode plus its NOTE 1
//!   zero-fill constraint.
//! * Annex A §A.3.2 step I-S1 — the derivation of `coeff_width[v]`,
//!   `shift_shape[y][x]`, `pels_height[x]` and `opaque_pels` from the
//!   decoded shape `f_shape[y][x]`.
//! * Figure 7-4 — the three 8×8 scan tables: (a) Alternate-Horizontal,
//!   (b) Alternate-Vertical, (c) Zigzag.

use crate::sample_padding::SamplePresence;
use crate::texture::AcEvent;

/// One of the three §7.4.2 scan patterns of Figure 7-4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanType {
    /// Figure 7-4 (a) — Alternate-Horizontal scan. Used by an intra
    /// block whose §7.4.3.1 DC prediction is taken from block C
    /// (the vertically adjacent / above block) — §7.4.2: the scan runs
    /// perpendicular to the predicted first row.
    AlternateHorizontal,
    /// Figure 7-4 (b) — Alternate-Vertical scan. Used by an intra
    /// block whose §7.4.3.1 DC prediction is taken from block A
    /// (the horizontally adjacent / left block).
    AlternateVertical,
    /// Figure 7-4 (c) — Zigzag scan. Used for every non-intra block,
    /// and for every block in an intra macroblock with
    /// `ac_pred_flag == 0`.
    Zigzag,
}

/// The §7.4.3.1 DC-prediction direction surfaced by the predictor pass.
/// Captured here as an opaque enum so [`select_scan_type`] doesn't
/// require the predictor implementation (which lands later) to exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcPredictionDirection {
    /// "Predict from block A" — the horizontally adjacent (left) block.
    FromLeft,
    /// "Predict from block C" — the vertically adjacent (above) block.
    FromAbove,
}

/// Figure 7-4 (a) — Alternate-Horizontal scan table. The grid is laid
/// out as `ALT_HORIZONTAL[row][col] = scan_index`; the inverse-scan
/// loop reads `PQF[row][col] = QFS[scan_index]`.
///
/// Numbers transcribed verbatim from Figure 7-4 of ISO/IEC 14496-2:2004.
#[rustfmt::skip]
const ALT_HORIZONTAL: [[u8; 8]; 8] = [
    [ 0,  1,  2,  3, 10, 11, 12, 13],
    [ 4,  5,  8,  9, 17, 16, 15, 14],
    [ 6,  7, 19, 18, 26, 27, 28, 29],
    [20, 21, 24, 25, 30, 31, 32, 33],
    [22, 23, 34, 35, 42, 43, 44, 45],
    [36, 37, 40, 41, 46, 47, 48, 49],
    [38, 39, 50, 51, 56, 57, 58, 59],
    [52, 53, 54, 55, 60, 61, 62, 63],
];

/// Figure 7-4 (b) — Alternate-Vertical scan table.
#[rustfmt::skip]
const ALT_VERTICAL: [[u8; 8]; 8] = [
    [ 0,  4,  6, 20, 22, 36, 38, 52],
    [ 1,  5,  7, 21, 23, 37, 39, 53],
    [ 2,  8, 19, 24, 34, 40, 50, 54],
    [ 3,  9, 18, 25, 35, 41, 51, 55],
    [10, 17, 26, 30, 42, 46, 56, 60],
    [11, 16, 27, 31, 43, 47, 57, 61],
    [12, 15, 28, 32, 44, 48, 58, 62],
    [13, 14, 29, 33, 45, 49, 59, 63],
];

/// Figure 7-4 (c) — Zigzag scan table. This is the same zigzag order
/// used by JPEG / MPEG-1 / MPEG-2.
#[rustfmt::skip]
const ZIGZAG: [[u8; 8]; 8] = [
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63],
];

/// Returns the `[8][8]` scan-index grid for the requested scan pattern.
/// Cell `(row, col)` holds the scan index `n` such that the n-th value
/// of `QFS[]` lands at `PQF[row][col]`.
fn scan_grid(scan_type: ScanType) -> &'static [[u8; 8]; 8] {
    match scan_type {
        ScanType::AlternateHorizontal => &ALT_HORIZONTAL,
        ScanType::AlternateVertical => &ALT_VERTICAL,
        ScanType::Zigzag => &ZIGZAG,
    }
}

/// Pick the §7.4.2 scan type for one block given `ac_pred_flag` and the
/// §7.4.3.1 DC-prediction direction.
///
/// * Non-intra blocks → caller supplies any value; the function is
///   only consulted for intra blocks. (Pass `is_intra == false` to get
///   the non-intra default of [`ScanType::Zigzag`] for completeness.)
/// * Intra + `ac_pred_flag == false` → [`ScanType::Zigzag`].
/// * Intra + `ac_pred_flag == true` + DC predictor from left (A, the
///   horizontally adjacent block) → [`ScanType::AlternateVertical`].
/// * Intra + `ac_pred_flag == true` + DC predictor from above (C, the
///   vertically adjacent block) → [`ScanType::AlternateHorizontal`].
///
/// §7.4.2: "if the DC prediction refers to the horizontally adjacent
/// block, alternate-vertical scan is selected for the current block.
/// Otherwise (for DC prediction referring to vertically adjacent
/// block), alternate-horizontal scan is used" — the scan runs
/// *perpendicular* to the prediction edge, front-loading the
/// frequencies the §7.4.3.3 predictor row/column does not cover.
///
/// `dc_direction` is ignored when `ac_pred_flag == false` or when
/// `is_intra == false`. The caller resolves the direction per
/// §7.4.3.1 (compare `|FA - FB|` vs `|FB - FC|`) and threads it in
/// here.
pub fn select_scan_type(
    is_intra: bool,
    ac_pred_flag: bool,
    dc_direction: DcPredictionDirection,
) -> ScanType {
    if !is_intra || !ac_pred_flag {
        return ScanType::Zigzag;
    }
    match dc_direction {
        DcPredictionDirection::FromLeft => ScanType::AlternateVertical,
        DcPredictionDirection::FromAbove => ScanType::AlternateHorizontal,
    }
}

/// Expand a sequence of §7.4.1.2 `(LAST, RUN, LEVEL)` EVENTs into the
/// §7.4.1 one-dimensional `QFS[64]` coefficient array (zero-padded
/// past the final EVENT).
///
/// Each EVENT places `run` zeros at the current scan position, then
/// writes `level` at the next position. The optional intra-DC value
/// supplied via `intra_dc` (the result of §7.4.1.1) lands at `QFS[0]`
/// and the AC EVENT stream walks positions 1..=63; if `intra_dc` is
/// `None` the EVENT stream walks positions 0..=63 (the §7.4.1.2 path
/// taken when the intra-AC VLC is in use per §7.4.1.4, or for inter
/// blocks).
///
/// Returns the populated array. A defensive check on the running
/// position keeps a malformed EVENT stream from writing past
/// `QFS[63]`; the function returns [`InverseScanError::Overflow`] in
/// that case rather than panicking.
pub fn events_to_qfs(
    events: &[AcEvent],
    intra_dc: Option<i32>,
) -> Result<[i32; 64], InverseScanError> {
    let mut qfs = [0i32; 64];
    let mut pos = match intra_dc {
        Some(dc) => {
            qfs[0] = dc;
            1usize
        }
        None => 0,
    };
    for ev in events {
        let run = ev.run as usize;
        // Skip `run` zeros, then write LEVEL at `pos + run`.
        let target = pos
            .checked_add(run)
            .ok_or(InverseScanError::Overflow { position: pos })?;
        if target >= 64 {
            return Err(InverseScanError::Overflow { position: target });
        }
        qfs[target] = ev.level;
        pos = target + 1;
    }
    Ok(qfs)
}

/// Apply the §7.4.2 inverse scan: convert the one-dimensional `QFS[64]`
/// stream into the two-dimensional `PQF[v][u]` 8×8 block under the
/// supplied scan pattern.
///
/// The mapping is the textbook §7.4.2 loop:
///
/// ```text
/// for (n = 0; n < 64; n++)
///     PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] = QFS[n];
/// ```
pub fn inverse_scan(qfs: &[i32; 64], scan_type: ScanType) -> [[i32; 8]; 8] {
    let grid = scan_grid(scan_type);
    let mut pqf = [[0i32; 8]; 8];
    for v in 0..8 {
        for u in 0..8 {
            let n = grid[v][u] as usize;
            pqf[v][u] = qfs[n];
        }
    }
    pqf
}

/// The Annex A §A.3.2 step I-S1 shape parameters derived from a decoded
/// 8×8 binary shape block `f_shape[y][x]`, used to drive the §7.4.2
/// modified inverse scan (and, in a later round, the inverse SA-DCT
/// transform itself).
///
/// `coeff_width[v]` gives the number of SA-DCT coefficients available
/// in row `v` of `PQF[v][u]`; `opaque_pels` is the total number of
/// opaque samples in the block (which equals the number of SA-DCT
/// coefficients, per §7.4.2 NOTE 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeParams {
    /// `coeff_width[v]` — coefficients available in each of the eight
    /// rows of `PQF[v][u]`. Each element lies in `0..=8`.
    pub coeff_width: [u8; 8],
    /// `opaque_pels` — the total opaque-sample count of the block,
    /// `0..=64`. Equals `coeff_width.iter().sum()`.
    pub opaque_pels: u8,
}

impl ShapeParams {
    /// Derive [`ShapeParams`] from an 8×8 binary shape block
    /// `f_shape[y][x]` per Annex A §A.3.2 step I-S1.
    ///
    /// Mirroring the spec pseudocode: for each column `x`, opaque pels
    /// (`f_shape[y][x] == 255`) are vertically shifted to the top to
    /// form `shift_shape[y][x]`; `coeff_width[v]` then counts the
    /// opaque cells in row `v` of `shift_shape`. Because the per-column
    /// shift packs `pels_height[x]` opaque cells against the top,
    /// `coeff_width[v]` equals the number of columns whose
    /// `pels_height[x] > v` — i.e. the number of columns with strictly
    /// more than `v` opaque samples.
    pub fn from_shape(f_shape: &[[SamplePresence; 8]; 8]) -> Self {
        // I-S1: pels_height[x] = opaque count in column x; opaque_pels
        // = total. shift_shape[v][x] is opaque iff v < pels_height[x],
        // so coeff_width[v] = #{ x : pels_height[x] > v }.
        let mut pels_height = [0u8; 8];
        let mut opaque_pels = 0u16;
        for x in 0..8 {
            let mut count = 0u8;
            for row in f_shape.iter() {
                if row[x].is_opaque() {
                    count += 1;
                }
            }
            pels_height[x] = count;
            opaque_pels += count as u16;
        }
        let mut coeff_width = [0u8; 8];
        for (v, cw) in coeff_width.iter_mut().enumerate() {
            *cw = pels_height.iter().filter(|&&h| (h as usize) > v).count() as u8;
        }
        ShapeParams {
            coeff_width,
            opaque_pels: opaque_pels as u8,
        }
    }
}

/// Apply the §7.4.2 **modified** inverse scan used when
/// `sadct_disable == 0` and the block has fewer than 64 opaque pels.
///
/// The spec loop:
///
/// ```text
/// coeff_count = 0;
/// for (n = 0; n < 64; n++) {
///     PQF[inv_scan_v[st][n]][inv_scan_u[st][n]] = 0;
///     if (coeff_width[inv_scan_v[st][n]] > inv_scan_u[st][n]) {
///         PQF[inv_scan_v[st][n]][inv_scan_u[st][n]] = QFS[coeff_count];
///         coeff_count++;
///     }
/// }
/// ```
///
/// `QFS[0..opaque_pels]` carries the decoded SA-DCT coefficients in
/// scan order (the EVENT stream produced exactly `opaque_pels`
/// coefficients for a conformant block); every `PQF[v][u]` outside the
/// `u < coeff_width[v]` region is forced to zero so that subsequent AC
/// prediction is not confused (§7.4.2 NOTE 1).
pub fn modified_inverse_scan(
    qfs: &[i32; 64],
    scan_type: ScanType,
    coeff_width: &[u8; 8],
) -> [[i32; 8]; 8] {
    let grid = scan_grid(scan_type);
    // Invert the `[v][u] = n` grid into the scan-order `n -> (v, u)`
    // mapping the spec's `inv_scan_v[st][n]` / `inv_scan_u[st][n]`
    // arrays provide.
    let mut pos_of_n = [(0u8, 0u8); 64];
    for (v, row) in grid.iter().enumerate() {
        for (u, &cell) in row.iter().enumerate() {
            pos_of_n[cell as usize] = (v as u8, u as u8);
        }
    }
    let mut pqf = [[0i32; 8]; 8];
    let mut coeff_count = 0usize;
    for &(v, u) in pos_of_n.iter() {
        let (v, u) = (v as usize, u as usize);
        // PQF is already zero; only fill the in-shape positions.
        if (coeff_width[v] as usize) > u {
            pqf[v][u] = qfs[coeff_count];
            coeff_count += 1;
        }
    }
    pqf
}

/// One-shot SA-DCT path: expand AC EVENTs (+ optional intra-DC) into a
/// `QFS[]` stream, then apply the §7.4.2 modified inverse scan using
/// `coeff_width[]` derived from the block's binary shape.
///
/// The EVENTs of a conformant SA-DCT block carry exactly `opaque_pels`
/// coefficients packed at `QFS[0..opaque_pels]`; the modified scan
/// distributes them across the in-shape `PQF[v][u]` positions and
/// zero-fills the rest.
pub fn events_to_pqf_sadct(
    events: &[AcEvent],
    intra_dc: Option<i32>,
    scan_type: ScanType,
    coeff_width: &[u8; 8],
) -> Result<[[i32; 8]; 8], InverseScanError> {
    let qfs = events_to_qfs(events, intra_dc)?;
    Ok(modified_inverse_scan(&qfs, scan_type, coeff_width))
}

/// One-shot: expand AC EVENTs (+ optional intra-DC) into a
/// scanned 8×8 `PQF[v][u]` block in a single call.
pub fn events_to_pqf(
    events: &[AcEvent],
    intra_dc: Option<i32>,
    scan_type: ScanType,
) -> Result<[[i32; 8]; 8], InverseScanError> {
    let qfs = events_to_qfs(events, intra_dc)?;
    Ok(inverse_scan(&qfs, scan_type))
}

/// Errors produced when expanding EVENTs into `QFS[]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseScanError {
    /// The cumulative `RUN + 1` exceeded the 64-coefficient block size.
    /// The first out-of-range write position is reported for
    /// diagnostics; a conformant stream never produces this.
    Overflow {
        /// The position at which the malformed EVENT tried to write.
        position: usize,
    },
}

impl core::fmt::Display for InverseScanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InverseScanError::Overflow { position } => {
                write!(
                    f,
                    "AC EVENT stream walked past coefficient 63 (target = {position})"
                )
            }
        }
    }
}

impl std::error::Error for InverseScanError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every scan grid must contain each index 0..=63 exactly once.
    fn assert_grid_is_permutation(grid: &[[u8; 8]; 8], name: &str) {
        let mut seen = [false; 64];
        for row in grid.iter() {
            for &cell in row.iter() {
                assert!(cell < 64, "{name}: cell value {cell} out of range");
                assert!(!seen[cell as usize], "{name}: duplicate index {cell}");
                seen[cell as usize] = true;
            }
        }
        assert!(
            seen.iter().all(|s| *s),
            "{name}: missing one or more indices"
        );
    }

    #[test]
    fn alt_horizontal_grid_is_permutation() {
        assert_grid_is_permutation(&ALT_HORIZONTAL, "alt-horizontal");
    }

    #[test]
    fn alt_vertical_grid_is_permutation() {
        assert_grid_is_permutation(&ALT_VERTICAL, "alt-vertical");
    }

    #[test]
    fn zigzag_grid_is_permutation() {
        assert_grid_is_permutation(&ZIGZAG, "zigzag");
    }

    #[test]
    fn zigzag_starts_in_standard_order() {
        // The canonical JPEG zigzag begins
        //   (0,0) → (0,1) → (1,0) → (2,0) → (1,1) → (0,2) → (0,3) → (1,2) …
        // i.e. PQF[0][0]=0, PQF[0][1]=1, PQF[1][0]=2, PQF[2][0]=3,
        //      PQF[1][1]=4, PQF[0][2]=5.
        assert_eq!(ZIGZAG[0][0], 0);
        assert_eq!(ZIGZAG[0][1], 1);
        assert_eq!(ZIGZAG[1][0], 2);
        assert_eq!(ZIGZAG[2][0], 3);
        assert_eq!(ZIGZAG[1][1], 4);
        assert_eq!(ZIGZAG[0][2], 5);
        // And the DC always lands at scan index 0 in (0,0).
        assert_eq!(ZIGZAG[0][0], 0);
        // The last-scanned position is the bottom-right corner.
        assert_eq!(ZIGZAG[7][7], 63);
    }

    #[test]
    fn alt_vertical_first_column_walks_down() {
        // Figure 7-4 (b) first column reads 0,1,2,3,10,11,12,13 — the
        // alternate-vertical scan starts by walking down column 0.
        let col0: Vec<u8> = (0..8).map(|v| ALT_VERTICAL[v][0]).collect();
        assert_eq!(col0, vec![0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn alt_horizontal_first_row_walks_right() {
        // Figure 7-4 (a) first row reads 0,1,2,3,10,11,12,13 — the
        // alternate-horizontal scan starts by walking right along
        // row 0.
        assert_eq!(ALT_HORIZONTAL[0], [0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn alt_horizontal_and_alt_vertical_are_transposes() {
        // §7.4.2 design intent: alt-horizontal and alt-vertical are
        // each other's transpose (one walks rows-first, the other
        // columns-first).
        for v in 0..8 {
            for u in 0..8 {
                assert_eq!(
                    ALT_HORIZONTAL[v][u], ALT_VERTICAL[u][v],
                    "(v={v}, u={u}) alt-horizontal vs alt-vertical transpose mismatch"
                );
            }
        }
    }

    #[test]
    fn inverse_scan_zigzag_places_dc_at_origin() {
        let mut qfs = [0i32; 64];
        qfs[0] = 42;
        let pqf = inverse_scan(&qfs, ScanType::Zigzag);
        assert_eq!(pqf[0][0], 42);
        for (v, row) in pqf.iter().enumerate() {
            for (u, &cell) in row.iter().enumerate() {
                if (v, u) != (0, 0) {
                    assert_eq!(cell, 0);
                }
            }
        }
    }

    #[test]
    fn inverse_scan_zigzag_first_few_positions() {
        // Place 1,2,3,4,5,6 at QFS[0..6]; check they land at the
        // expected zigzag positions.
        let mut qfs = [0i32; 64];
        for (n, slot) in qfs.iter_mut().enumerate().take(6) {
            *slot = (n + 1) as i32;
        }
        let pqf = inverse_scan(&qfs, ScanType::Zigzag);
        assert_eq!(pqf[0][0], 1);
        assert_eq!(pqf[0][1], 2);
        assert_eq!(pqf[1][0], 3);
        assert_eq!(pqf[2][0], 4);
        assert_eq!(pqf[1][1], 5);
        assert_eq!(pqf[0][2], 6);
    }

    #[test]
    fn inverse_scan_alt_vertical_first_column() {
        // QFS[0..=3] placed at (0..=3, 0) (column 0, rows 0..3).
        let mut qfs = [0i32; 64];
        qfs[0] = 10;
        qfs[1] = 20;
        qfs[2] = 30;
        qfs[3] = 40;
        let pqf = inverse_scan(&qfs, ScanType::AlternateVertical);
        assert_eq!(pqf[0][0], 10);
        assert_eq!(pqf[1][0], 20);
        assert_eq!(pqf[2][0], 30);
        assert_eq!(pqf[3][0], 40);
    }

    #[test]
    fn inverse_scan_alt_horizontal_first_row() {
        let mut qfs = [0i32; 64];
        qfs[0] = 10;
        qfs[1] = 20;
        qfs[2] = 30;
        qfs[3] = 40;
        let pqf = inverse_scan(&qfs, ScanType::AlternateHorizontal);
        assert_eq!(pqf[0][0], 10);
        assert_eq!(pqf[0][1], 20);
        assert_eq!(pqf[0][2], 30);
        assert_eq!(pqf[0][3], 40);
    }

    #[test]
    fn inverse_scan_last_position_is_bottom_right_for_all_scans() {
        // For every scan pattern, QFS[63] lands at PQF[7][7] (the
        // last-scanned cell is always the bottom-right of the block).
        let mut qfs = [0i32; 64];
        qfs[63] = 999;
        for st in [
            ScanType::AlternateHorizontal,
            ScanType::AlternateVertical,
            ScanType::Zigzag,
        ] {
            let pqf = inverse_scan(&qfs, st);
            assert_eq!(pqf[7][7], 999, "scan {st:?}");
        }
    }

    #[test]
    fn events_to_qfs_with_intra_dc() {
        // intra_dc=5; one AC EVENT (run 2, level=-3, LAST). DC at
        // QFS[0]=5, two zeros at QFS[1..3], LEVEL at QFS[3]=-3, rest 0.
        let events = [AcEvent {
            last: true,
            run: 2,
            level: -3,
        }];
        let qfs = events_to_qfs(&events, Some(5)).unwrap();
        assert_eq!(qfs[0], 5);
        assert_eq!(qfs[1], 0);
        assert_eq!(qfs[2], 0);
        assert_eq!(qfs[3], -3);
        for &v in &qfs[4..] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn events_to_qfs_without_intra_dc_walks_from_zero() {
        // EVENT(run 0, level 7), EVENT(run 1, level -1, LAST):
        // QFS[0]=7, QFS[1]=0, QFS[2]=-1.
        let events = [
            AcEvent {
                last: false,
                run: 0,
                level: 7,
            },
            AcEvent {
                last: true,
                run: 1,
                level: -1,
            },
        ];
        let qfs = events_to_qfs(&events, None).unwrap();
        assert_eq!(qfs[0], 7);
        assert_eq!(qfs[1], 0);
        assert_eq!(qfs[2], -1);
        for &v in &qfs[3..] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn events_to_qfs_overflow_rejected() {
        // RUN 64 from position 0 → target 64, out of range.
        let events = [AcEvent {
            last: true,
            run: 64,
            level: 1,
        }];
        let err = events_to_qfs(&events, None).unwrap_err();
        assert!(matches!(err, InverseScanError::Overflow { position: 64 }));
    }

    #[test]
    fn events_to_qfs_overflow_with_intra_dc() {
        // intra_dc consumed pos=1; RUN 63 from there → target 64.
        let events = [AcEvent {
            last: true,
            run: 63,
            level: 1,
        }];
        let err = events_to_qfs(&events, Some(99)).unwrap_err();
        assert!(matches!(err, InverseScanError::Overflow { position: 64 }));
    }

    #[test]
    fn events_to_pqf_one_call_round_trip() {
        // intra_dc=8 (zigzag → PQF[0][0]=8); one EVENT placing level=2
        // at scan-position 1 (zigzag → PQF[0][1]=2).
        let events = [AcEvent {
            last: true,
            run: 0,
            level: 2,
        }];
        let pqf = events_to_pqf(&events, Some(8), ScanType::Zigzag).unwrap();
        assert_eq!(pqf[0][0], 8);
        assert_eq!(pqf[0][1], 2);
        // Everything else zero.
        for (v, row) in pqf.iter().enumerate() {
            for (u, &cell) in row.iter().enumerate() {
                if (v, u) != (0, 0) && (v, u) != (0, 1) {
                    assert_eq!(cell, 0);
                }
            }
        }
    }

    #[test]
    fn select_scan_intra_no_acpred_is_zigzag() {
        // Direction is ignored when ac_pred_flag == false.
        for dir in [
            DcPredictionDirection::FromAbove,
            DcPredictionDirection::FromLeft,
        ] {
            assert_eq!(select_scan_type(true, false, dir), ScanType::Zigzag);
        }
    }

    #[test]
    fn select_scan_non_intra_is_zigzag() {
        for ac in [true, false] {
            for dir in [
                DcPredictionDirection::FromAbove,
                DcPredictionDirection::FromLeft,
            ] {
                assert_eq!(select_scan_type(false, ac, dir), ScanType::Zigzag);
            }
        }
    }

    #[test]
    fn select_scan_intra_acpred_from_above_is_alt_horizontal() {
        // §7.4.2: DC prediction from the vertically adjacent block →
        // alternate-horizontal scan (perpendicular to the predicted
        // first row).
        assert_eq!(
            select_scan_type(true, true, DcPredictionDirection::FromAbove),
            ScanType::AlternateHorizontal
        );
    }

    #[test]
    fn select_scan_intra_acpred_from_left_is_alt_vertical() {
        // §7.4.2: DC prediction from the horizontally adjacent block →
        // alternate-vertical scan.
        assert_eq!(
            select_scan_type(true, true, DcPredictionDirection::FromLeft),
            ScanType::AlternateVertical
        );
    }

    /// Build an 8×8 `f_shape` from a row-major boolean grid
    /// (`true` == opaque) for the SA-DCT tests.
    fn shape(rows: [[bool; 8]; 8]) -> [[SamplePresence; 8]; 8] {
        let mut s = [[SamplePresence::Transparent; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                if rows[y][x] {
                    s[y][x] = SamplePresence::Opaque;
                }
            }
        }
        s
    }

    #[test]
    fn shape_params_full_opaque_is_8x8() {
        // A fully opaque block → every column has height 8, so every
        // row has 8 coefficients and opaque_pels == 64.
        let s = shape([[true; 8]; 8]);
        let p = ShapeParams::from_shape(&s);
        assert_eq!(p.coeff_width, [8; 8]);
        assert_eq!(p.opaque_pels, 64);
        // Invariant from §7.4.2 NOTE 1: opaque_pels == sum(coeff_width).
        let sum: u16 = p.coeff_width.iter().map(|&c| c as u16).sum();
        assert_eq!(p.opaque_pels as u16, sum);
    }

    #[test]
    fn shape_params_all_transparent_is_zero() {
        let s = shape([[false; 8]; 8]);
        let p = ShapeParams::from_shape(&s);
        assert_eq!(p.coeff_width, [0; 8]);
        assert_eq!(p.opaque_pels, 0);
    }

    #[test]
    fn shape_params_top_left_quadrant() {
        // Opaque 4×4 top-left quadrant. Each of columns 0..4 has 4
        // opaque pels (rows 0..4); columns 4..8 have 0. After the
        // vertical shift, shift_shape rows 0..4 each have 4 opaque
        // cells (one per opaque column), rows 4..8 have none.
        let mut rows = [[false; 8]; 8];
        for r in rows.iter_mut().take(4) {
            for c in r.iter_mut().take(4) {
                *c = true;
            }
        }
        let p = ShapeParams::from_shape(&shape(rows));
        assert_eq!(p.coeff_width, [4, 4, 4, 4, 0, 0, 0, 0]);
        assert_eq!(p.opaque_pels, 16);
    }

    #[test]
    fn shape_params_vertical_shift_packs_to_top() {
        // Opaque pels are NOT contiguous within a column: column 0 has
        // opaque at rows 1 and 5 (height 2), all others transparent.
        // The vertical shift packs them to rows 0,1 of shift_shape, so
        // coeff_width = [1, 1, 0, 0, 0, 0, 0, 0].
        let mut rows = [[false; 8]; 8];
        rows[1][0] = true;
        rows[5][0] = true;
        let p = ShapeParams::from_shape(&shape(rows));
        assert_eq!(p.coeff_width, [1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(p.opaque_pels, 2);
    }

    #[test]
    fn shape_params_staircase_columns() {
        // Column x has (x+1) opaque pels (rows 0..=x). pels_height =
        // [1,2,3,4,5,6,7,8]. coeff_width[v] = #cols with height > v:
        //   v=0 → 8, v=1 → 7, … v=7 → 1.
        let mut rows = [[false; 8]; 8];
        for (x, _) in (0..8).enumerate() {
            for r in rows.iter_mut().take(x + 1) {
                r[x] = true;
            }
        }
        let p = ShapeParams::from_shape(&shape(rows));
        assert_eq!(p.coeff_width, [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(p.opaque_pels, 36);
    }

    #[test]
    fn modified_scan_full_width_matches_plain_scan() {
        // coeff_width all 8 → no position is excluded, and the
        // coefficients are written in scan order, which is exactly the
        // standard inverse scan of QFS in scan order. Build QFS = the
        // scan-order ramp so plain inverse_scan(ramp) lays 1..=64 out
        // in scan order; the modified scan must produce the same grid.
        let mut qfs = [0i32; 64];
        for (n, slot) in qfs.iter_mut().enumerate() {
            *slot = (n + 1) as i32;
        }
        let plain = inverse_scan(&qfs, ScanType::Zigzag);
        let modified = modified_inverse_scan(&qfs, ScanType::Zigzag, &[8; 8]);
        assert_eq!(plain, modified);
    }

    #[test]
    fn modified_scan_zero_fills_out_of_shape_positions() {
        // coeff_width = [2,1,0,...]: row 0 keeps u in {0,1}, row 1
        // keeps u==0, every other position must be zero — even if a
        // hostile QFS would otherwise place values there. We feed a
        // QFS of all-ones; only 3 positions (2 + 1 coefficients) may
        // be non-zero.
        let qfs = [1i32; 64];
        let cw = [2u8, 1, 0, 0, 0, 0, 0, 0];
        let pqf = modified_inverse_scan(&qfs, ScanType::Zigzag, &cw);
        let mut nonzero = 0;
        for (v, row) in pqf.iter().enumerate() {
            for (u, &cell) in row.iter().enumerate() {
                let in_shape = (cw[v] as usize) > u;
                if in_shape {
                    assert_eq!(cell, 1, "in-shape ({v},{u}) should hold a coeff");
                    nonzero += 1;
                } else {
                    assert_eq!(cell, 0, "out-of-shape ({v},{u}) must be zero");
                }
            }
        }
        assert_eq!(nonzero, 3, "exactly opaque_pels coefficients survive");
    }

    #[test]
    fn modified_scan_packs_coeffs_in_scan_order() {
        // coeff_width = [3,0,...]: only PQF[0][0..3] are in-shape. The
        // zigzag scan visits (0,0)→(0,1)→(1,0)→(2,0)→(1,1)→(0,2)→…
        // Of those, the in-shape positions are (0,0), (0,1), (0,2) at
        // coeff_count 0, 1, 2 respectively (others skipped). So
        // QFS[0]→(0,0), QFS[1]→(0,1), QFS[2]→(0,2).
        let mut qfs = [0i32; 64];
        qfs[0] = 11;
        qfs[1] = 22;
        qfs[2] = 33;
        let cw = [3u8, 0, 0, 0, 0, 0, 0, 0];
        let pqf = modified_inverse_scan(&qfs, ScanType::Zigzag, &cw);
        assert_eq!(pqf[0][0], 11);
        assert_eq!(pqf[0][1], 22);
        assert_eq!(pqf[0][2], 33);
        // Nothing else.
        for (v, row) in pqf.iter().enumerate() {
            for (u, &cell) in row.iter().enumerate() {
                if !(v == 0 && u < 3) {
                    assert_eq!(cell, 0);
                }
            }
        }
    }

    #[test]
    fn modified_scan_coeff_count_equals_opaque_pels() {
        // The number of non-zero positions the modified scan can fill
        // equals sum(coeff_width) == opaque_pels. Feed all-ones QFS so
        // every fillable position becomes non-zero and count them.
        let rows = {
            let mut r = [[false; 8]; 8];
            // L-shape: column 0 full (8), row 0 full (8) → overlap at
            // (0,0). pels_height col0 = 8, cols1..8 each = 1.
            for r0 in r.iter_mut() {
                r0[0] = true;
            }
            for c in r[0].iter_mut() {
                *c = true;
            }
            r
        };
        let p = ShapeParams::from_shape(&shape(rows));
        let qfs = [1i32; 64];
        let pqf = modified_inverse_scan(&qfs, ScanType::Zigzag, &p.coeff_width);
        let nonzero: usize = pqf.iter().flatten().filter(|&&c| c != 0).count();
        assert_eq!(nonzero, p.opaque_pels as usize);
    }

    #[test]
    fn events_to_pqf_sadct_round_trip() {
        // coeff_width = [2,1,0,...] (opaque_pels = 3). EVENTs produce
        // QFS = [5, 0, -2] (DC 5; one EVENT run 1 level -2). The
        // modified scan lays them at the first three in-shape zigzag
        // positions: (0,0)=5, (0,1)=0, (1,0)=-2.
        let cw = [2u8, 1, 0, 0, 0, 0, 0, 0];
        let events = [AcEvent {
            last: true,
            run: 1,
            level: -2,
        }];
        let pqf = events_to_pqf_sadct(&events, Some(5), ScanType::Zigzag, &cw).unwrap();
        assert_eq!(pqf[0][0], 5);
        assert_eq!(pqf[0][1], 0);
        assert_eq!(pqf[1][0], -2);
        // All other cells zero.
        for (v, row) in pqf.iter().enumerate() {
            for (u, &cell) in row.iter().enumerate() {
                if !matches!((v, u), (0, 0) | (0, 1) | (1, 0)) {
                    assert_eq!(cell, 0);
                }
            }
        }
    }

    #[test]
    fn events_to_pqf_sadct_propagates_overflow() {
        let cw = [8u8; 8];
        let events = [AcEvent {
            last: true,
            run: 64,
            level: 1,
        }];
        let err = events_to_pqf_sadct(&events, None, ScanType::Zigzag, &cw).unwrap_err();
        assert!(matches!(err, InverseScanError::Overflow { position: 64 }));
    }

    #[test]
    fn inverse_scan_error_displays() {
        let e = InverseScanError::Overflow { position: 64 };
        let s = format!("{e}");
        assert!(s.contains("past coefficient 63"));
        assert!(s.contains("64"));
    }
}

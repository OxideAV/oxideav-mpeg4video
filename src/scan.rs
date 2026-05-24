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
//!   DC-prediction direction picks the scan:
//!   * Prediction from block C (vertically adjacent, "above") →
//!     [`ScanType::AlternateVertical`] for the current block.
//!   * Prediction from block A (horizontally adjacent, "left") →
//!     [`ScanType::AlternateHorizontal`] for the current block.
//!
//! See [`select_scan_type`] for the encoded rule. Predictor gathering
//! and the DC-direction decision themselves remain later-round work;
//! once those land, the caller threads their direction into
//! `select_scan_type` to pick the right table.
//!
//! ## Out of scope (this round)
//!
//! * The `sadct_disable == 0` modified inverse scan for
//!   non-rectangular VOPs (the `coeff_width[]` aware variant in the
//!   §7.4.2 pseudocode). The rectangular path lands here.
//! * The §7.4.3 spatial DC/AC predictor that supplies the per-block
//!   prediction direction. `select_scan_type` is parameterised on a
//!   pre-resolved direction so it doesn't depend on the missing
//!   gathering.
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
//!   `ac_pred_flag` and the §7.4.3.1 prediction direction.
//! * Figure 7-4 — the three 8×8 scan tables: (a) Alternate-Horizontal,
//!   (b) Alternate-Vertical, (c) Zigzag.

use crate::texture::AcEvent;

/// One of the three §7.4.2 scan patterns of Figure 7-4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanType {
    /// Figure 7-4 (a) — Alternate-Horizontal scan. Used by an intra
    /// block whose §7.4.3.1 DC prediction is taken from block A
    /// (the horizontally adjacent / left block).
    AlternateHorizontal,
    /// Figure 7-4 (b) — Alternate-Vertical scan. Used by an intra
    /// block whose §7.4.3.1 DC prediction is taken from block C
    /// (the vertically adjacent / above block).
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
/// * Intra + `ac_pred_flag == true` + DC predictor from above (C) →
///   [`ScanType::AlternateVertical`].
/// * Intra + `ac_pred_flag == true` + DC predictor from left (A) →
///   [`ScanType::AlternateHorizontal`].
///
/// `dc_direction` is ignored when `ac_pred_flag == false` or when
/// `is_intra == false`. When the predictor gathering pass lands, the
/// caller resolves the direction per §7.4.3.1 (compare
/// `|FA - FB|` vs `|FB - FC|`) and threads it in here.
pub fn select_scan_type(
    is_intra: bool,
    ac_pred_flag: bool,
    dc_direction: DcPredictionDirection,
) -> ScanType {
    if !is_intra || !ac_pred_flag {
        return ScanType::Zigzag;
    }
    match dc_direction {
        DcPredictionDirection::FromAbove => ScanType::AlternateVertical,
        DcPredictionDirection::FromLeft => ScanType::AlternateHorizontal,
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
    fn select_scan_intra_acpred_from_above_is_alt_vertical() {
        assert_eq!(
            select_scan_type(true, true, DcPredictionDirection::FromAbove),
            ScanType::AlternateVertical
        );
    }

    #[test]
    fn select_scan_intra_acpred_from_left_is_alt_horizontal() {
        assert_eq!(
            select_scan_type(true, true, DcPredictionDirection::FromLeft),
            ScanType::AlternateHorizontal
        );
    }

    #[test]
    fn inverse_scan_error_displays() {
        let e = InverseScanError::Overflow { position: 64 };
        let s = format!("{e}");
        assert!(s.contains("past coefficient 63"));
        assert!(s.contains("64"));
    }
}

//! The staged black-box input/output vectors of the integer
//! row–column inverse transform ("variant RC",
//! `docs/video/mpeg4-visual/integer-idct-ecosystem-variant.md` §9.1).
//!
//! `tests/fixtures/integer-idct-vectors/vectors.txt` is a byte-for-byte
//! copy of the staged `docs/video/mpeg4-visual/fixtures/
//! integer-idct-vectors/vectors.txt` (SHA-256
//! `b55e863cd3367a77496f95cc9aa57a171eaa89edd6bc6c4cf696dc2efc5da45e`):
//! 128 lines in input / expected-output pairs, 64 signed decimal
//! integers per line in row-major order, raw 16-bit transform results
//! with no output saturation. Pair `n` belongs to class `n mod 4`:
//! dense-small, sparse, DC-only, full-range (the last exercises the
//! 16-bit truncation of the row-pass result and the 32-bit wrap of the
//! column-pass products).

use oxideav_mpeg4video::variant_rc::{idct_8x8_variant_rc, variant_rc_raw};

const VECTORS: &str = include_str!("fixtures/integer-idct-vectors/vectors.txt");

type Block = [[i32; 8]; 8];

fn parse_block(line: &str, line_no: usize) -> Block {
    let values: Vec<i32> = line
        .split(' ')
        .map(|t| {
            t.parse()
                .unwrap_or_else(|e| panic!("line {line_no}: {t:?}: {e}"))
        })
        .collect();
    assert_eq!(values.len(), 64, "line {line_no}: 64 integers");
    let mut block = [[0i32; 8]; 8];
    for (i, v) in values.into_iter().enumerate() {
        block[i / 8][i % 8] = v;
    }
    block
}

fn pairs() -> Vec<(Block, Block)> {
    let lines: Vec<&str> = VECTORS.lines().collect();
    assert_eq!(lines.len(), 128, "128 lines");
    lines
        .chunks(2)
        .enumerate()
        .map(|(n, pair)| {
            (
                parse_block(pair[0], 2 * n + 1),
                parse_block(pair[1], 2 * n + 2),
            )
        })
        .collect()
}

#[test]
fn every_staged_vector_is_reproduced_exactly() {
    let pairs = pairs();
    assert_eq!(pairs.len(), 64);
    let mut matched = 0;
    for (n, (input, expected)) in pairs.iter().enumerate() {
        let got = variant_rc_raw(input).map(|row| row.map(i32::from));
        assert_eq!(&got, expected, "pair {n} (class {})", n % 4);
        matched += 1;
    }
    assert_eq!(matched, 64);
}

#[test]
fn vector_classes_are_the_documented_ones() {
    // The set's own description, so a truncated or reordered copy is
    // noticed: DC-only pairs have exactly one non-zero input, the
    // full-range pairs reach beyond the 8-bit output range, and no
    // pair has a row 0 of (−4, 0, …, 0) — the empty-row rule is pinned
    // by its own unit test instead.
    let mut largest_output = 0;
    for (n, (input, expected)) in pairs().iter().enumerate() {
        let non_zero = input.iter().flatten().filter(|&&c| c != 0).count();
        match n % 4 {
            0 => assert!((61..=64).contains(&non_zero), "pair {n}"),
            1 => assert!((8..=19).contains(&non_zero), "pair {n}"),
            2 => assert!(non_zero == 1 && input[0][0] != 0, "pair {n}"),
            _ => assert!(
                input.iter().flatten().all(|c| (-2048..=2048).contains(c)),
                "pair {n}"
            ),
        }
        assert_ne!(input[0], [-4, 0, 0, 0, 0, 0, 0, 0], "pair {n}");
        for &v in expected.iter().flatten() {
            largest_output = largest_output.max(v.abs());
        }
    }
    assert_eq!(largest_output, 2252);
}

#[test]
fn saturating_entry_point_clamps_the_same_results() {
    for (n, (input, expected)) in pairs().iter().enumerate() {
        // An input of 2048 lies outside the dequantiser's range but
        // inside the transform's 16-bit domain: no input saturation.
        let got = idct_8x8_variant_rc(input, 8);
        let want = expected.map(|row| row.map(|v| v.clamp(-256, 255)));
        assert_eq!(got, want, "pair {n}");
    }
}

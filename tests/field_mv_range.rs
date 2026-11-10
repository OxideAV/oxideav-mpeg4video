//! Erratum E4 (`docs/video/mpeg4-visual/mpeg4-visual-errata.md`): the
//! Table 7-9 `[low:high]` range bounds the **vertical component of a
//! field motion vector in field units** — the §7.6.3 reduction applies
//! to `Py / 2 + MVDy` before the §7.7.2.1 doubling.
//!
//! The fixtures are the staged
//! `docs/video/mpeg4-visual/fixtures/interlaced-field-mv-range/` set
//! (provenance + SHA-256 in `tests/fixtures/NOTES.md`): two interlaced
//! I-P-P streams (`vop_fcode_forward` 1 and 2) whose forced field
//! vectors sit on and beyond the frame-unit boundary, the decode the
//! rule yields (equal to the producing encoder's own reconstruction),
//! and a per-field-vector trace of predictor, differential and the
//! vector each candidate reading produces.

use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::field_motion::reconstruct_field_motion_vectors_wrapped;
use oxideav_mpeg4video::motion::{FieldMvPair, MotionVectorDelta};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

const W: usize = 352;
const H: usize = 288;

fn assert_stream_exact(name: &str, options: DecodeOptions) {
    let stream = fixture(&format!("field_mv_range_{name}.m4v"));
    let expected = fixture(&format!("field_mv_range_{name}.yuv"));
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    let mut frames = dec.decode(&stream).expect("stream decodes");
    frames.extend(dec.flush());
    let frame_len = W * H * 3 / 2;
    assert_eq!(frames.len(), 3, "{name}: I P P");
    assert_eq!(expected.len(), 3 * frame_len);
    for (i, frame) in frames.iter().enumerate() {
        let exp = &expected[i * frame_len..(i + 1) * frame_len];
        let (y, c) = exp.split_at(W * H);
        let (u, v) = c.split_at(W * H / 4);
        assert!(frame.luma_samples() == y, "{name}: VOP {i} luma differs");
        assert!(frame.cb_samples() == u, "{name}: VOP {i} Cb differs");
        assert!(frame.cr_samples() == v, "{name}: VOP {i} Cr differs");
    }
}

#[test]
fn fcode1_stream_decodes_exactly() {
    assert_stream_exact("fcode1", DecodeOptions::spec());
}

#[test]
fn fcode2_stream_decodes_exactly() {
    assert_stream_exact("fcode2", DecodeOptions::spec());
}

#[test]
fn ecosystem_compat_decodes_the_same_vectors() {
    // The field-unit range is not an ecosystem divergence: both modes
    // reproduce the fixture decode.
    assert_stream_exact("fcode1", DecodeOptions::ecosystem());
    assert_stream_exact("fcode2", DecodeOptions::ecosystem());
}

/// One row of the staged trace.
struct TraceRow {
    fcode: u8,
    py_frame: i32,
    mvdy_field: i32,
    rule_c_frame: i32,
    rule_a_frame: i32,
    rule_b_frame: i32,
    forced: Option<i32>,
}

fn trace_rows() -> Vec<TraceRow> {
    let text = String::from_utf8(fixture("field_mv_range_trace.csv")).expect("utf-8 trace");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("column {name}"))
    };
    let (c_fcode, c_py, c_mvd) = (col("vop_fcode"), col("Py_frame"), col("MVDy_field"));
    let (c_c, c_a, c_b) = (col("ruleC_frame"), col("ruleA_frame"), col("ruleB_frame"));
    let c_forced = col("encoder_forced_frame");
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let int = |i: usize| f[i].parse::<i32>().expect("integer field");
            TraceRow {
                fcode: int(c_fcode) as u8,
                py_frame: int(c_py),
                mvdy_field: int(c_mvd),
                rule_c_frame: int(c_c),
                rule_a_frame: int(c_a),
                rule_b_frame: int(c_b),
                forced: (!f[c_forced].is_empty()).then(|| int(c_forced)),
            }
        })
        .collect()
}

#[test]
fn trace_rows_reconstruct_through_the_field_unit_reduction() {
    let rows = trace_rows();
    assert_eq!(rows.len(), 134);
    let (mut beyond_frame_range, mut reduced_sum) = (0, 0);
    for (i, row) in rows.iter().enumerate() {
        let delta = MotionVectorDelta {
            dx: 0,
            dy: row.mvdy_field,
        };
        let pair = FieldMvPair {
            top: delta,
            bottom: delta,
        };
        let mvs = reconstruct_field_motion_vectors_wrapped(pair, 0, row.py_frame, row.fcode);
        assert_eq!(mvs.top.y, row.rule_c_frame, "row {i}");
        assert_eq!(mvs.bottom.y, row.rule_c_frame, "row {i}");
        if let Some(forced) = row.forced {
            // What the producing encoder aimed at is what we decode.
            assert_eq!(mvs.top.y, forced, "row {i}");
        }
        // The vector is a field-unit member of [low:high], doubled.
        let f = 1i32 << (row.fcode - 1);
        assert!((-32 * f..=32 * f - 1).contains(&(mvs.top.y / 2)), "row {i}");
        assert_eq!(mvs.top.y % 2, 0, "row {i}");
        beyond_frame_range += usize::from(row.rule_a_frame != row.rule_c_frame);
        reduced_sum += usize::from(row.rule_b_frame != row.rule_c_frame);
    }
    // The trace discriminates the three candidate readings: 24 rows
    // where a frame-unit reduction would differ, 4 where leaving the
    // sum unreduced would.
    assert_eq!(beyond_frame_range, 24);
    assert_eq!(reduced_sum, 4);
}

//! §7.7.2.1 quarter-sample field-interpolation probe pins.
//!
//! The interlaced+qpel conformance stream (`ilaced_qpel_ip_64x64`)
//! went bit-exact once two aspects of the quarter-sample field
//! motion-compensation geometry — which the printed spec text does
//! not pin (see `field_motion::field_motion_compensate_one_reference_qpel`
//! and `field_motion::field_chroma_dy_qpel`) — were arbitrated by
//! black-box pixel comparison over **constructed probe streams**:
//!
//! 1. each 16-wide luma field block interpolates as **two 8×8
//!    §7.6.2.2 blocks**, each with its own Figure 7-30 boundary
//!    mirroring (a single 16-wide interpolation mispredicts isolated
//!    samples in the columns whose FIR taps span the centre seam);
//! 2. the chroma field MV's vertical quarter → half halving **floors
//!    on the field grid** (`Div2Round(mv_y >> 2)`); the truncating
//!    alternative mispredicts every probe with a negative odd
//!    field-grid component.
//!
//! Each probe appends one hand-written P-VOP to the conformant
//! interlaced+qpel fixture's configuration headers + I-VOP: every
//! macroblock is skipped except one field-predicted macroblock with
//! chosen field reference selections and field MVDs and no residual,
//! so the P frame isolates exactly one field-MC geometry case. The
//! expected outputs (`tests/fixtures/fq_probe_*.yuv`) are the
//! black-box reference decodes of these streams (generation command +
//! SHA-256 in `tests/fixtures/NOTES.md`); the probe bitstreams are
//! rebuilt deterministically by this test.
//!
//! The probes must decode **bit-exactly** in both behaviour modes —
//! the arbitrated geometry is a default spec reading, not an
//! ecosystem-compat behaviour.

use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;

const MB_COLS: usize = 4;
const MB_ROWS: usize = 4;
/// Probe macroblock (col, row) — interior, so §7.6.4 edge clamping
/// never engages for the probed vectors.
const PROBE_MB: (usize, usize) = (1, 1);

/// MSB-first bit writer.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit: u8,
    acc: u8,
}

impl BitWriter {
    fn write_bits(&mut self, value: u32, n: usize) {
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) as u8;
            self.acc = (self.acc << 1) | b;
            self.bit += 1;
            if self.bit == 8 {
                self.bytes.push(self.acc);
                self.acc = 0;
                self.bit = 0;
            }
        }
    }
    /// Stuffing to the byte boundary ('0' then '1's).
    fn finish(mut self) -> Vec<u8> {
        if self.bit != 0 {
            self.write_bits(0, 1);
            while self.bit != 0 {
                self.write_bits(1, 1);
            }
        }
        self.bytes
    }
}

/// Table B.12 `(code, len, mv_data)` encode rows for the probe MVDs
/// (`vop_fcode == 1`, so `mv_data` is the differential itself). Only
/// the short codes the probes use are needed.
#[rustfmt::skip]
const MVD_CODES: &[(u16, u8, i32)] = &[
    (0b00000111, 8, -7), (0b00001001, 8, -6), (0b00001011, 8, -5),
    (0b0000111, 7, -4), (0b00011, 5, -3), (0b0011, 4, -2), (0b011, 3, -1),
    (0b1, 1, 0),
    (0b010, 3, 1), (0b0010, 4, 2), (0b00010, 5, 3), (0b0000110, 7, 4),
    (0b00001010, 8, 5), (0b00001000, 8, 6), (0b00000110, 8, 7),
];

fn write_mvd(w: &mut BitWriter, mv_data: i32) {
    let &(code, len, _) = MVD_CODES
        .iter()
        .find(|&&(_, _, v)| v == mv_data)
        .expect("probe mv_data in encode table");
    w.write_bits(u32::from(code), usize::from(len));
}

struct ProbeCfg {
    /// (forward_top_field_reference, forward_bottom_field_reference).
    refs: (u8, u8),
    /// Top-field MVD (x, y): quarter-pel horizontal / field-grid
    /// quarter-pel vertical differentials (predictor is zero — every
    /// other macroblock is skipped).
    top: (i32, i32),
    /// Bottom-field MVD (x, y).
    bottom: (i32, i32),
    /// Expected reference decode under `tests/fixtures/`.
    expected: &'static str,
}

/// The pinned probe set: seam coverage (half-pel and mixed quarter
/// fractions), cross-field reference selection, and negative odd
/// field-grid vertical components (the chroma floor-vs-truncate
/// discriminator).
const PROBES: &[ProbeCfg] = &[
    ProbeCfg {
        refs: (0, 0),
        top: (2, 0),
        bottom: (2, 0),
        expected: "fq_probe_half_seam.yuv",
    },
    ProbeCfg {
        refs: (0, 0),
        top: (1, 1),
        bottom: (3, 3),
        expected: "fq_probe_diag.yuv",
    },
    ProbeCfg {
        refs: (0, 1),
        top: (-3, 5),
        bottom: (6, -2),
        expected: "fq_probe_mixed.yuv",
    },
    ProbeCfg {
        refs: (0, 0),
        top: (-1, -1),
        bottom: (-5, -6),
        expected: "fq_probe_neg.yuv",
    },
    ProbeCfg {
        refs: (0, 0),
        top: (5, 3),
        bottom: (7, -3),
        expected: "fq_probe_odd_a.yuv",
    },
    ProbeCfg {
        refs: (0, 0),
        top: (-7, 7),
        bottom: (-4, -7),
        expected: "fq_probe_odd_c.yuv",
    },
    ProbeCfg {
        refs: (1, 0),
        top: (6, -1),
        bottom: (5, 1),
        expected: "fq_probe_odd_d.yuv",
    },
];

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Headers + GOV + I-VOP of the interlaced+qpel conformance fixture:
/// everything before the second VOP start code.
fn anchor_prefix() -> Vec<u8> {
    let stream = fixture("ilaced_qpel_ip_64x64.m4v");
    let mut vop_starts = stream
        .windows(4)
        .enumerate()
        .filter(|(_, w)| w == &[0, 0, 1, 0xB6])
        .map(|(i, _)| i);
    let _first = vop_starts.next().expect("I-VOP start code");
    let second = vop_starts.next().expect("second VOP start code");
    stream[..second].to_vec()
}

/// Append the probe P-VOP: header (P, tick 1, rounding 0, fcode 1),
/// then the §6.2.6 macroblock walk — all `not_coded` except the probe
/// macroblock (inter, cbp 0, `field_prediction == 1`, the two §6.2.6.2
/// field `motion_vector()` bodies, no residual).
fn build_probe(cfg: &ProbeCfg) -> Vec<u8> {
    let mut out = anchor_prefix();
    let mut w = BitWriter::default();
    w.write_bits(0x0000_01B6, 32);
    w.write_bits(0b01, 2); // vop_coding_type: P
    w.write_bits(0, 1); // modulo_time_base terminator
    w.write_bits(1, 1); // marker
    w.write_bits(1, 5); // vop_time_increment (resolution 25 → 5 bits)
    w.write_bits(1, 1); // marker
    w.write_bits(1, 1); // vop_coded
    w.write_bits(0, 1); // vop_rounding_type
    w.write_bits(0, 3); // intra_dc_vlc_thr
    w.write_bits(1, 1); // top_field_first (matches the anchor)
    w.write_bits(0, 1); // alternate_vertical_scan_flag
    w.write_bits(4, 5); // vop_quant
    w.write_bits(1, 3); // vop_fcode_forward
    for row in 0..MB_ROWS {
        for col in 0..MB_COLS {
            if (col, row) != PROBE_MB {
                w.write_bits(1, 1); // not_coded
                continue;
            }
            w.write_bits(0, 1); // not_coded = 0
            w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00 (Table B.7)
            w.write_bits(0b11, 2); // cbpy: inter pattern 0 (Table B.8)
                                   // interlaced_information(): cbp == 0 and derived_mb_type 0
                                   // → no dct_type bit (§6.2.6.3 first gate).
            w.write_bits(1, 1); // field_prediction = 1
            w.write_bits(u32::from(cfg.refs.0), 1); // forward_top_field_reference
            w.write_bits(u32::from(cfg.refs.1), 1); // forward_bottom_field_reference
            write_mvd(&mut w, cfg.top.0);
            write_mvd(&mut w, cfg.top.1);
            write_mvd(&mut w, cfg.bottom.0);
            write_mvd(&mut w, cfg.bottom.1);
        }
    }
    out.extend_from_slice(&w.finish());
    out
}

#[test]
fn field_qpel_probe_streams_are_bit_exact_in_both_modes() {
    for cfg in PROBES {
        let stream = build_probe(cfg);
        let expected = fixture(cfg.expected);
        for options in [DecodeOptions::spec(), DecodeOptions::ecosystem()] {
            let mut dec = Mpeg4VideoDecoder::with_options(options);
            let mut frames = dec
                .decode(&stream)
                .unwrap_or_else(|e| panic!("{}: {e}", cfg.expected));
            frames.extend(dec.flush());
            assert_eq!(frames.len(), 2, "{}: frame count", cfg.expected);
            let mut ours = Vec::with_capacity(expected.len());
            for frame in &frames {
                ours.extend_from_slice(frame.luma_samples());
                ours.extend_from_slice(frame.cb_samples());
                ours.extend_from_slice(frame.cr_samples());
            }
            assert_eq!(ours.len(), expected.len(), "{}: yuv length", cfg.expected);
            let differing = ours
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differing, 0,
                "{} ({options:?}): {differing} samples differ from the reference decode",
                cfg.expected
            );
        }
    }
}

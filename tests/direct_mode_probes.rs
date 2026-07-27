//! §7.7.2.2 interlaced-direct derivation probe pins.
//!
//! Constructed probe streams that pin the ecosystem-compat
//! interlaced-direct model (`crate::compat`, behaviour 1) against the
//! black-box reference decoder with **non-zero co-located field
//! motion vectors over textured anchors** — the configuration the
//! real conformance corpus could not isolate (its deviating direct
//! macroblocks sit over flat co-located regions, so the anchor MV
//! state was not pixel-determinable until now).
//!
//! Each probe is the interlaced conformance fixture's configuration
//! headers + I-VOP, then a hand-written P-VOP whose macroblock (1,1)
//! is field-predicted with chosen field reference selections and
//! field MVs (every other macroblock skipped), then a hand-written
//! B-VOP whose macroblock (1,1) is a direct macroblock — either the
//! zero-bit `modb == "1"` form or an explicit `modb == "01"` /
//! `mb_type == "1"` with a chosen `MVD[0]` (macroblocks whose
//! co-located future macroblock is skipped transmit zero bits per
//! §6.2.6). The expected outputs (`tests/fixtures/dm_probe_*.yuv`)
//! are the black-box reference decodes (provenance + SHA-256 in
//! `tests/fixtures/NOTES.md`).
//!
//! ## What the probes established (this round's arbitration)
//!
//! * **`MVD[0]` transmitted and non-zero, or absent (`modb "1"`)**:
//!   the reference decoder evaluates the erratum-corrected §7.7.2.2
//!   derivation with the co-located field MVs read as **zero** even
//!   when the bitstream-reconstructed co-located vectors are
//!   provably non-zero — the compat model is confirmed
//!   unconditionally for these forms (`compat` decode is bit-exact
//!   up to the interlaced-intra near-tie samples).
//! * **`MVD[0]` transmitted and exactly (0, 0)**: the reference
//!   decoder instead runs **progressive** direct mode over the
//!   co-located frame vector `Div2Round(MVf1 + MVf2)` (the §7.7.2.1
//!   CASE 2/3 field → frame conversion): forward
//!   `(TRB * MV) / TRD`, backward `((TRB − TRD) * MV) / TRD`, 16×16
//!   frame MC — verified sample-exact from the oracle's own anchor
//!   frames for three distinct co-located MV sets. Neither decode
//!   mode reproduces this today: the spec default keeps the printed
//!   field derivation (as it must), and the compat mode's
//!   zero-co-located field model predates this finding; whether the
//!   compat mode should adopt the transmitted-zero-`MVD[0]` branch
//!   is left to a project ruling (the corpus' conformance streams
//!   contain no such macroblock). The `dm_probe_mvd0_*` pins below
//!   record the measured envelopes of both modes so any future
//!   behaviour change is deliberate.

use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::vol::parse_video_object_layer;
use oxideav_mpeg4video::vop::{parse_video_object_plane_header, VopContext};

const MB_COLS: usize = 4;
const MB_ROWS: usize = 4;
const PROBE_MB: (usize, usize) = (1, 1);

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
/// (`vop_fcode == 1`).
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

struct DirectProbe {
    /// P-VOP probe MB (forward_top_field_reference, forward_bottom_field_reference).
    p_refs: (u8, u8),
    /// P-VOP field MVDs (predictor zero): half-pel horizontal,
    /// field-grid half-pel vertical.
    p_top: (i32, i32),
    p_bottom: (i32, i32),
    /// B-VOP direct macroblock form: `None` = zero-bit `modb "1"`;
    /// `Some(mvd)` = explicit `modb "01"` + `mb_type "1"` + MVD[0].
    mvd: Option<(i32, i32)>,
    /// Expected reference decode under `tests/fixtures/`.
    expected: &'static str,
    /// Spec-mode envelope `(max_tol, count_tol)`; `(1, 4)` means
    /// near-tie-only.
    spec_envelope: (u32, usize),
    /// Compat-mode envelope.
    compat_envelope: (u32, usize),
}

/// The pinned probe set. Spec mode legitimately deviates on every
/// non-zero-co-located probe (it keeps the printed `MV[i]` term);
/// compat matches the oracle up to the interlaced-intra near-tie
/// samples on the zero-co-located-model forms, and carries a
/// documented envelope on the transmitted-zero-`MVD[0]` probes (see
/// the module docs). Envelopes are measured values + headroom 0.
const PROBES: &[DirectProbe] = &[
    // MVD[0] non-zero → compat model confirmed.
    DirectProbe {
        p_refs: (0, 0),
        p_top: (2, 1),
        p_bottom: (-2, 2),
        mvd: Some((1, -1)),
        expected: "dm_probe_mvdnz.yuv",
        spec_envelope: (64, 161),
        compat_envelope: (1, 4),
    },
    DirectProbe {
        p_refs: (1, 0),
        p_top: (-3, -2),
        p_bottom: (4, 3),
        mvd: Some((0, 2)),
        expected: "dm_probe_xref.yuv",
        spec_envelope: (85, 159),
        compat_envelope: (1, 4),
    },
    DirectProbe {
        p_refs: (0, 1),
        p_top: (5, 0),
        p_bottom: (0, -4),
        mvd: Some((-2, 0)),
        expected: "dm_probe_mixed.yuv",
        spec_envelope: (63, 178),
        compat_envelope: (1, 4),
    },
    // MVD[0] absent (zero-bit modb "1") → compat model confirmed.
    DirectProbe {
        p_refs: (0, 0),
        p_top: (2, 1),
        p_bottom: (-2, 2),
        mvd: None,
        expected: "dm_probe_modb1.yuv",
        spec_envelope: (52, 118),
        compat_envelope: (1, 4),
    },
    // Zero co-located control: both modes agree with the oracle.
    DirectProbe {
        p_refs: (0, 0),
        p_top: (0, 0),
        p_bottom: (0, 0),
        mvd: Some((3, -2)),
        expected: "dm_probe_zero_ctl.yuv",
        spec_envelope: (1, 4),
        compat_envelope: (1, 4),
    },
    // MVD[0] transmitted as (0, 0) → the oracle runs progressive
    // direct over Div2Round(MVf1 + MVf2); neither mode reproduces it
    // (documented divergence, ruling pending — see module docs).
    DirectProbe {
        p_refs: (0, 0),
        p_top: (2, 1),
        p_bottom: (-2, 2),
        mvd: Some((0, 0)),
        expected: "dm_probe_mvd0_a.yuv",
        spec_envelope: (112, 378),
        compat_envelope: (112, 380),
    },
    DirectProbe {
        p_refs: (0, 1),
        p_top: (4, -2),
        p_bottom: (3, 1),
        mvd: Some((0, 0)),
        expected: "dm_probe_mvd0_b.yuv",
        spec_envelope: (43, 71),
        compat_envelope: (51, 159),
    },
    DirectProbe {
        p_refs: (1, 1),
        p_top: (-5, 3),
        p_bottom: (2, -3),
        mvd: Some((0, 0)),
        expected: "dm_probe_mvd0_c.yuv",
        spec_envelope: (120, 258),
        compat_envelope: (113, 195),
    },
];

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Build one probe stream over the `ilaced_ipb_64x64.m4v` anchor.
fn build_probe(cfg: &DirectProbe) -> Vec<u8> {
    let stream = fixture("ilaced_ipb_64x64.m4v");
    let mut vol_off = None;
    let mut vops = vec![];
    for i in 0..stream.len() - 3 {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            let b = stream[i + 3];
            if (0x20..=0x2F).contains(&b) && vol_off.is_none() {
                vol_off = Some(i);
            }
            if b == 0xB6 {
                vops.push(i);
            }
        }
    }
    let vol = parse_video_object_layer(&stream[vol_off.expect("vol start")..], 1).expect("vol");
    assert!(vol.interlaced && !vol.quarter_sample);
    let resolution = vol.time_increment_resolution;
    let mut tinc_bits = 1usize;
    while (1u32 << tinc_bits) < u32::from(resolution) {
        tinc_bits += 1;
    }
    let ctx = VopContext::from_vol(&vol);
    let ivop = parse_video_object_plane_header(&stream[vops[0]..], resolution, ctx).expect("i-vop");
    let i_tinc = u32::from(ivop.time_increment);
    let tff = u32::from(ivop.top_field_first);
    let mut out = stream[..vops[1]].to_vec();

    // ---- P-VOP at t = i_tinc + 2: all skipped except the probe MB
    // (field-predicted, cbp 0, chosen refs + field MVDs). ----
    let mut w = BitWriter::default();
    w.write_bits(0x0000_01B6, 32);
    w.write_bits(0b01, 2); // P
    w.write_bits(0, 1); // modulo_time_base terminator
    w.write_bits(1, 1); // marker
    w.write_bits(i_tinc + 2, tinc_bits);
    w.write_bits(1, 1); // marker
    w.write_bits(1, 1); // vop_coded
    w.write_bits(0, 1); // vop_rounding_type
    w.write_bits(0, 3); // intra_dc_vlc_thr
    w.write_bits(tff, 1); // top_field_first
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
            w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00
            w.write_bits(0b11, 2); // cbpy: inter pattern 0
            w.write_bits(1, 1); // field_prediction = 1
            w.write_bits(u32::from(cfg.p_refs.0), 1);
            w.write_bits(u32::from(cfg.p_refs.1), 1);
            write_mvd(&mut w, cfg.p_top.0);
            write_mvd(&mut w, cfg.p_top.1);
            write_mvd(&mut w, cfg.p_bottom.0);
            write_mvd(&mut w, cfg.p_bottom.1);
        }
    }
    out.extend_from_slice(&w.finish());

    // ---- B-VOP at t = i_tinc + 1: only the probe MB transmits bits
    // (its co-located future MB is coded; all others are skipped →
    // §6.2.6 co_located_not_coded zero-bit macroblocks). ----
    let mut w = BitWriter::default();
    w.write_bits(0x0000_01B6, 32);
    w.write_bits(0b10, 2); // B
    w.write_bits(0, 1); // modulo_time_base terminator
    w.write_bits(1, 1); // marker
    w.write_bits(i_tinc + 1, tinc_bits);
    w.write_bits(1, 1); // marker
    w.write_bits(1, 1); // vop_coded
    w.write_bits(0, 3); // intra_dc_vlc_thr
    w.write_bits(tff, 1); // top_field_first
    w.write_bits(0, 1); // alternate_vertical_scan_flag
    w.write_bits(4, 5); // vop_quant
    w.write_bits(1, 3); // vop_fcode_forward
    w.write_bits(1, 3); // vop_fcode_backward
    match cfg.mvd {
        None => w.write_bits(0b1, 1), // modb "1": zero-bit direct
        Some(mvd) => {
            w.write_bits(0b01, 2); // modb "01": mb_type present, no cbpb
            w.write_bits(0b1, 1); // mb_type "1": direct
            write_mvd(&mut w, mvd.0);
            write_mvd(&mut w, mvd.1);
        }
    }
    out.extend_from_slice(&w.finish());
    out
}

#[test]
fn interlaced_direct_probe_streams_match_their_pinned_envelopes() {
    for cfg in PROBES {
        let stream = build_probe(cfg);
        let expected = fixture(cfg.expected);
        for (options, (max_tol, cnt_tol)) in [
            (DecodeOptions::spec(), cfg.spec_envelope),
            (DecodeOptions::ecosystem(), cfg.compat_envelope),
        ] {
            let mut dec = Mpeg4VideoDecoder::with_options(options);
            let mut frames = dec
                .decode(&stream)
                .unwrap_or_else(|e| panic!("{}: {e}", cfg.expected));
            frames.extend(dec.flush());
            assert_eq!(frames.len(), 3, "{}: display frames", cfg.expected);
            let mut ours = Vec::with_capacity(expected.len());
            for frame in &frames {
                ours.extend_from_slice(frame.luma_samples());
                ours.extend_from_slice(frame.cb_samples());
                ours.extend_from_slice(frame.cr_samples());
            }
            assert_eq!(ours.len(), expected.len(), "{}: yuv length", cfg.expected);
            let mut max = 0u32;
            let mut differing = 0usize;
            for (&a, &b) in ours.iter().zip(expected.iter()) {
                let d = (i32::from(a) - i32::from(b)).unsigned_abs();
                if d > 0 {
                    differing += 1;
                }
                max = max.max(d);
            }
            assert!(
                max <= max_tol && differing <= cnt_tol,
                "{} ({options:?}): max {max} (tol {max_tol}), {differing} samples \
                 differ (tol {cnt_tol})",
                cfg.expected
            );
        }
    }
}

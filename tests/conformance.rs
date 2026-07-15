//! Real-stream conformance: decode reference-encoder-produced MPEG-4
//! Visual elementary streams and compare every sample against the
//! reference decode — **bit-exactly** wherever the oracle permits.
//!
//! The fixtures under `tests/fixtures/` were produced by a black-box
//! reference encoder together with the matching raw `yuv420p` reference
//! decode of each stream in display order (generation commands +
//! SHA-256 in `tests/fixtures/NOTES.md`). No external implementation
//! source was consulted — the binaries were used opaquely as
//! encode/decode oracles.
//!
//! The reference decodes were produced with the oracle's
//! floating-point IDCT, i.e. the mathematical Annex A.1 transform —
//! the same ideal transform this decoder evaluates in `f64` — so the
//! two decoders agree bit-for-bit on every stream, with two measured,
//! bounded exceptions:
//!
//! * **Near-tie IDCT samples** (`assert_stream_near_exact`): the
//!   oracle's transform runs in single precision; on isolated samples
//!   whose ideal spatial value sits within ~1e-5 of a rounding
//!   boundary (measured: 12.5000007, 238.4999993) its float error can
//!   cross the boundary while our `f64` evaluation rounds the ideal
//!   value correctly. Affected streams carry a per-stream budget of
//!   ±1-valued differing samples; everything else must match exactly.
//! * **§7.4.4.5 mismatch control on intra blocks** (`mq_ipb_64x64`):
//!   the oracle applies the method-1 mismatch toggle to non-intra
//!   blocks only (verified per-block-class — skipping the intra toggle
//!   collapses the stream's differences to a handful of near-ties).
//!   §7.4.4.5 contains no intra exemption, so this decoder keeps the
//!   spec behaviour **by default** and the assertion carries a ±1
//!   envelope; the opt-in ecosystem-compat mode
//!   (`DecodeOptions::ecosystem`, see `oxideav_mpeg4video::compat`)
//!   reproduces the oracle bit-for-bit up to the near-ties — see the
//!   `compat_*` tests at the end of this file.
//!
//! Two interlaced tool axes remain outside the exact envelope with
//! root-caused, documented deviations — see their tests below. The
//! §7.7.2.2 interlaced-direct deviation is likewise covered by the
//! compat mode (29 of the corpus' 30 interlaced-direct macroblocks go
//! bit-exact under it).

use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Split a raw `yuv420p` dump into per-frame (Y, U, V) plane slices.
fn yuv_frames(data: &[u8], w: usize, h: usize) -> Vec<(&[u8], &[u8], &[u8])> {
    let y_len = w * h;
    let c_len = (w / 2) * (h / 2);
    let frame_len = y_len + 2 * c_len;
    assert_eq!(data.len() % frame_len, 0, "yuv length must be whole frames");
    data.chunks_exact(frame_len)
        .map(|f| (&f[..y_len], &f[y_len..y_len + c_len], &f[y_len + c_len..]))
        .collect()
}

fn decode_stream(name: &str, options: DecodeOptions) -> Vec<DecodedFrame> {
    let stream = fixture(name);
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    let mut frames = dec
        .decode(&stream)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    frames.extend(dec.flush());
    frames
}

/// Compare one decoded frame to the reference planes; return
/// `(max_abs_diff, differing_sample_count, total_samples)`.
fn diff_stats(frame: &DecodedFrame, y: &[u8], u: &[u8], v: &[u8]) -> (u32, usize, usize) {
    let ours = [frame.luma_samples(), frame.cb_samples(), frame.cr_samples()];
    let theirs = [y, u, v];
    let mut max = 0u32;
    let mut differing = 0usize;
    let mut total = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        assert_eq!(a.len(), b.len(), "plane size mismatch");
        for (&x, &r) in a.iter().zip(b.iter()) {
            let d = (i32::from(x) - i32::from(r)).unsigned_abs();
            if d > 0 {
                differing += 1;
            }
            max = max.max(d);
            total += 1;
        }
    }
    (max, differing, total)
}

/// Whole-stream diff against the reference decode:
/// `(max_abs_diff, differing_samples, total_samples)` over every frame,
/// with frame count and plane sizes asserted exact.
fn stream_diff(m4v: &str, yuv: &str, w: usize, h: usize) -> (u32, usize, usize) {
    stream_diff_with(m4v, yuv, w, h, DecodeOptions::spec())
}

/// [`stream_diff`] under an explicit [`crate::compat`] behaviour
/// selection.
fn stream_diff_with(
    m4v: &str,
    yuv: &str,
    w: usize,
    h: usize,
    options: DecodeOptions,
) -> (u32, usize, usize) {
    let frames = decode_stream(m4v, options);
    let reference = fixture(yuv);
    let reference = yuv_frames(&reference, w, h);
    assert_eq!(
        frames.len(),
        reference.len(),
        "{m4v}: display-order frame count"
    );
    let mut max = 0u32;
    let mut differing = 0usize;
    let mut total = 0usize;
    for (frame, &(y, u, v)) in frames.iter().zip(reference.iter()) {
        let (m, d, t) = diff_stats(frame, y, u, v);
        max = max.max(m);
        differing += d;
        total += t;
    }
    (max, differing, total)
}

/// Every sample of every frame must equal the reference decode.
fn assert_stream_exact(m4v: &str, yuv: &str, w: usize, h: usize) {
    let (max, differing, total) = stream_diff(m4v, yuv, w, h);
    assert!(
        differing == 0,
        "{m4v}: expected bit-exact decode, got {differing}/{total} \
         differing samples (max abs diff {max})"
    );
}

/// Bit-exact except for a bounded budget of near-tie samples: at most
/// `tie_budget` samples may differ, each by exactly 1 (the oracle's
/// single-precision IDCT crossing a rounding boundary the ideal value
/// sits within ~1e-5 of — see the module docs).
fn assert_stream_near_exact(m4v: &str, yuv: &str, w: usize, h: usize, tie_budget: usize) {
    assert_stream_near_exact_with(m4v, yuv, w, h, tie_budget, DecodeOptions::spec());
}

/// [`assert_stream_near_exact`] under an explicit behaviour selection.
fn assert_stream_near_exact_with(
    m4v: &str,
    yuv: &str,
    w: usize,
    h: usize,
    tie_budget: usize,
    options: DecodeOptions,
) {
    let (max, differing, total) = stream_diff_with(m4v, yuv, w, h, options);
    assert!(
        max <= 1 && differing <= tie_budget,
        "{m4v} ({options:?}): expected <= {tie_budget} near-tie +/-1 samples, got \
         {differing}/{total} differing (max abs diff {max})"
    );
}

/// Bounded, root-caused deviation from the oracle (documented at the
/// call site): max per-sample difference and differing-sample fraction
/// must stay within the measured envelope.
fn assert_stream_bounded(m4v: &str, yuv: &str, w: usize, h: usize, max_tol: u32, frac_tol: f64) {
    assert_stream_bounded_with(m4v, yuv, w, h, max_tol, frac_tol, DecodeOptions::spec());
}

/// [`assert_stream_bounded`] under an explicit behaviour selection.
#[allow(clippy::too_many_arguments)]
fn assert_stream_bounded_with(
    m4v: &str,
    yuv: &str,
    w: usize,
    h: usize,
    max_tol: u32,
    frac_tol: f64,
    options: DecodeOptions,
) {
    let (max, differing, total) = stream_diff_with(m4v, yuv, w, h, options);
    let frac = differing as f64 / total as f64;
    assert!(
        max <= max_tol && frac <= frac_tol,
        "{m4v} ({options:?}): max abs diff {max} (tol {max_tol}), {differing}/{total} \
         samples differ ({frac:.5} > {frac_tol})"
    );
}

// ───────────────────────── bit-exact streams ─────────────────────────

#[test]
fn intra_only_stream_is_bit_exact() {
    // 3 intra frames with §6.2.5.2 video packets.
    assert_stream_exact("intra_64x64.m4v", "intra_64x64.yuv", 64, 64);
}

#[test]
fn ip_stream_is_bit_exact() {
    // I + 4 P frames: half-pel MC, §7.6.5 chroma derivation, and the
    // §7.4.3 DC/AC prediction with the §4.1 `//` rounding.
    assert_stream_exact("ip_64x64.m4v", "ip_64x64.yuv", 64, 64);
}

#[test]
fn ipb_stream_is_bit_exact() {
    // I/P/B with 2 consecutive B-VOPs: §6.1.3.8 reorder, §7.6.7
    // TRB/TRD, §6.2.6 co_located_not_coded zero-bit macroblocks, B-VOP
    // video packets, and the §7.6.9 modes as single 16×16 MC blocks.
    assert_stream_exact("ipb_64x64.m4v", "ipb_64x64.yuv", 64, 64);
}

#[test]
fn qpel_ip_stream_is_bit_exact() {
    // §7.6.2.2 quarter-sample mode: 8-tap FIR half-pel stage, bilinear
    // quarter positions, Figure 7-30 block-boundary mirroring, and the
    // §7.6.5 quarter-mode chroma reduction (K = 1).
    assert_stream_exact("qpel_ip_64x64.m4v", "qpel_ip_64x64.yuv", 64, 64);
}

#[test]
fn qpel_ipb_stream_is_bit_exact() {
    // Quarter-sample I/P/B: explicit B modes as 16×16 quarter-pel MC
    // blocks; §7.6.9.5 direct macroblocks derive on the quarter grid
    // (co-located MV and MVD[0] both quarter-pel) and compensate 8×8
    // blocks through §7.6.2.2 with its boundary mirroring.
    assert_stream_exact("qpel_ipb_64x64.m4v", "qpel_ipb_64x64.yuv", 64, 64);
}

#[test]
fn qpel_mv4_ipb_stream_is_bit_exact() {
    // Quarter-sample + four-MV anchors: the §7.6.5 quarter-mode K = 4
    // chroma reduction (§4.1 truncating per-vector halving onto the
    // sixteenth grid / Table 7-10) and §7.6.9.5.2 direct mode over
    // 4-MV co-located macroblocks with per-sub-block quarter-grid
    // (MVF[i], MVB[i]) pairs.
    assert_stream_exact("qpel_mv4_ipb_64x64.m4v", "qpel_mv4_ipb_64x64.yuv", 64, 64);
}

#[test]
fn aic_ipb_stream_is_bit_exact() {
    // §7.4.3.3 AC prediction: ac_pred_flag == 1 intra macroblocks with
    // the §7.4.2 direction-dependent scan (DC-from-left →
    // alternate-vertical, DC-from-above → alternate-horizontal) and the
    // §4.1 `//` quantiser rescale of the predicted row/column.
    assert_stream_exact("aic_ipb_64x64.m4v", "aic_ipb_64x64.yuv", 64, 64);
}

#[test]
fn altscan_ipb_stream_is_bit_exact() {
    // §6.3.5 alternate_vertical_scan_flag on a progressive VOL: every
    // block of every VOP inverse-scans with the Figure 7-4 (b) pattern.
    assert_stream_exact("altscan_ipb_64x64.m4v", "altscan_ipb_64x64.yuv", 64, 64);
}

#[test]
fn dp_ipb_stream_is_bit_exact() {
    // §6.2.5.3 data partitioning: per-packet dc_marker / motion_marker
    // partition walks, §E.1.2 prediction resets, header-partition intra
    // DC feeding the texture partition; B-VOPs on the combined syntax.
    assert_stream_exact("dp_ipb_64x64.m4v", "dp_ipb_64x64.yuv", 64, 64);
}

#[test]
fn ipb_176x144_stream_is_bit_exact() {
    // QCIF I/P/B with ~120-byte video packets: resync-marker handling
    // at a realistic frame size (99 macroblocks per VOP).
    assert_stream_exact("ipb_176x144.m4v", "ipb_176x144.yuv", 176, 144);
}

#[test]
fn interlaced_ip2_stream_is_bit_exact() {
    // Interlaced I + P with strong real motion (bottom-field-first):
    // §7.7.2.1 field MC, CASE 1/2/3 predictors, and the
    // `2*Div2Round(MVy/2)` field-chroma vertical derivation.
    assert_stream_exact("ilaced_ip2_64x64.m4v", "ilaced_ip2_64x64.yuv", 64, 64);
}

// ───────────── bit-exact up to near-tie IDCT samples ─────────────

#[test]
fn mv4_ipb_stream_is_bit_exact_up_to_near_ties() {
    // Four-MV anchors + B-VOPs (half-sample): per-sub-block §7.6.9.5.2
    // direct MVs, K = 4 chroma reduction. One near-tie sample (ideal
    // value 238.4999993) persists across the 7 frames.
    assert_stream_near_exact("mv4_ipb_64x64.m4v", "mv4_ipb_64x64.yuv", 64, 64, 7);
}

#[test]
fn interlaced_intra_stream_is_bit_exact_up_to_near_ties() {
    // Interlaced VOL, intra-only: §7.7.1 field DCT. One near-tie
    // sample (ideal value 12.5000007) per frame.
    assert_stream_near_exact(
        "ilaced_intra_64x64.m4v",
        "ilaced_intra_64x64.yuv",
        64,
        64,
        3,
    );
}

#[test]
fn interlaced_alternate_scan_intra_stream_is_bit_exact_up_to_near_ties() {
    // As above plus §6.3.5 alternate_vertical_scan_flag; same near-tie
    // sample.
    assert_stream_near_exact(
        "ilaced_altscan_intra_64x64.m4v",
        "ilaced_altscan_intra_64x64.yuv",
        64,
        64,
        3,
    );
}

#[test]
fn interlaced_ip_stream_is_bit_exact_up_to_near_ties() {
    // Interlaced I + 4 P with field motion estimation: §7.7.2.1
    // field-predicted macroblocks, §7.7.1 field DCT. The intra
    // near-tie sample propagates through the P chain (5 frames).
    assert_stream_near_exact("ilaced_ip_64x64.m4v", "ilaced_ip_64x64.yuv", 64, 64, 5);
}

// ─────────────── bounded, root-caused oracle deviations ───────────────

#[test]
fn mpeg_quant_ipb_stream_matches_within_intra_mismatch_envelope() {
    // quant_type == 1 (method-1 inverse quantisation, default matrices)
    // I/P/B. §7.4.4.5 requires the mismatch toggle on every method-1
    // block; the oracle applies it to non-intra blocks only (verified
    // per-block-class — with the intra toggle suppressed this stream
    // collapses to 4 near-tie samples). The F[7][7] LSB difference on
    // intra blocks ripples through the IDCT as scattered ±1 samples
    // (measured 3062/73728 ≈ 4.2%, max 1). This decoder follows the
    // printed clause by default; `DecodeOptions::ecosystem` opts into
    // the oracle's behaviour (see the `compat_*` tests below).
    assert_stream_bounded("mq_ipb_64x64.m4v", "mq_ipb_64x64.yuv", 64, 64, 1, 0.05);
}

#[test]
fn interlaced_ipb_stream_matches_within_interlaced_direct_envelope() {
    // Interlaced I/P/B: every macroblock class is bit-exact EXCEPT the
    // §7.7.2.2 interlaced-direct macroblocks (two isolated MBs per
    // B-VOP here) — see the 176×144 fixture below for the root cause —
    // plus the interlaced-intra near-tie sample. Measured: 650/43008
    // ≈ 1.5%, max 58.
    assert_stream_bounded(
        "ilaced_ipb_64x64.m4v",
        "ilaced_ipb_64x64.yuv",
        64,
        64,
        60,
        0.02,
    );
}

#[test]
fn interlaced_direct_bframes_stream_matches_within_interlaced_direct_envelope() {
    // 176×144, 25 VOPs (3 I / 6 P / 16 B), 30 interlaced-direct MBs
    // across 11 B-VOPs (fixture staged in
    // docs/video/mpeg4-visual/fixtures/interlaced-direct-bframes/,
    // refs #176). Classification matches the fixture's per-VOP
    // interlaced-direct distribution 30/30; every I-/P-VOP and every
    // non-direct interlaced-B macroblock class is bit-exact.
    //
    // Root cause of the remaining deviation, solved this round by
    // exhaustive per-macroblock-field search over the field-MC
    // candidate space (12 uniquely-determined macroblock-fields, all
    // consistent): the oracle evaluates the E1/E2-corrected §7.7.2.2
    // pseudo code with the co-located field motion vectors substituted
    // by ZERO — its derived vectors reduce to mvf[i] = mvb[i] = MVD[0]
    // on the field grid (e.g. MVD[0].y = -21 → exactly 21 frame lines
    // in both fields), while the forward reference fields still follow
    // the co-located macroblock's field selections and the backward
    // ones the spec's (0, 1) literals. §7.7.2.2 states the vectors are
    // "calculated from the forward field motion vectors of the
    // co-located future reference VOP", so this decoder keeps the
    // spec's MV[i] term by default (with the field-grid vertical
    // arithmetic the §7.7.2 evenness invariant requires); macroblocks
    // whose co-located field MVs are non-zero therefore legitimately
    // differ from this oracle, and `DecodeOptions::ecosystem` opts
    // into the zero-co-located derivation (see the `compat_*` tests
    // below). Measured: 6202/570240 ≈ 0.65%, max 114 (chroma).
    assert_stream_bounded(
        "ilaced_direct_176x144.m4v",
        "ilaced_direct_176x144.yuv",
        176,
        144,
        120,
        0.01,
    );
}

#[test]
fn interlaced_qpel_ip_stream_matches_within_field_qpel_envelope() {
    // Interlaced field motion estimation + quarter-sample (ildct +
    // ilme + qpel), I + P chain. Isolated field-predicted (and
    // neighbouring) macroblocks diverge from the oracle: an exhaustive
    // search over (top MV, bottom MV, reference fields, rounding) with
    // our §7.6.2.2-on-field-grid interpolation reproduces NO candidate
    // for the failing macroblocks, i.e. the oracle's quarter-sample
    // *field* interpolation cascade itself differs from our reading of
    // the §7.6.2.2 process applied per field — not the vectors. The
    // spec text does not pin the field-grid FIR/mirroring geometry
    // precisely; resolving it needs a dedicated arbitration pass (or a
    // clean-room trace of §7.7.2.1 quarter-sample field interpolation).
    // Measured: 1184/73728 ≈ 1.6%, max 111 (chroma).
    assert_stream_bounded(
        "ilaced_qpel_ip_64x64.m4v",
        "ilaced_qpel_ip_64x64.yuv",
        64,
        64,
        120,
        0.02,
    );
}

// ───────────── ecosystem-compat mode (`DecodeOptions::ecosystem`) ─────────────
//
// The opt-in compat mode reproduces the black-box-observed ecosystem
// behaviour on the two documented spec divergences (see `crate::compat`):
// the §7.4.4.5 method-1 mismatch toggle is skipped on intra blocks, and
// the §7.7.2.2 interlaced-direct derivation reads the co-located field
// MVs as zero. The tests below pin what each previously-deviating
// stream measures under compat, and that compat leaves the spec-exact
// streams untouched.

#[test]
fn compat_mpeg_quant_stream_collapses_to_near_ties() {
    // Spec mode carries a 3062-sample ±1 envelope on this stream (the
    // intra mismatch toggle, see the spec-mode test above). With the
    // compat intra exemption the whole difference collapses to the 4
    // remaining single-precision IDCT near-ties (measured 4/73728,
    // max 1).
    assert_stream_near_exact_with(
        "mq_ipb_64x64.m4v",
        "mq_ipb_64x64.yuv",
        64,
        64,
        4,
        DecodeOptions::ecosystem(),
    );
}

#[test]
fn compat_interlaced_ipb_stream_collapses_to_near_ties() {
    // Spec mode carries a 650-sample max-58 envelope (two isolated
    // §7.7.2.2 interlaced-direct macroblocks per B-VOP). With the
    // compat zero-co-located derivation every direct macroblock goes
    // bit-exact; what remains is the interlaced-intra near-tie sample
    // and its motion-compensated propagation (measured 7/43008,
    // max 1).
    assert_stream_near_exact_with(
        "ilaced_ipb_64x64.m4v",
        "ilaced_ipb_64x64.yuv",
        64,
        64,
        7,
        DecodeOptions::ecosystem(),
    );
}

#[test]
fn compat_interlaced_direct_bframes_stream_tightens_to_one_macroblock() {
    // Spec mode carries a 6202-sample max-114 envelope across the 30
    // interlaced-direct macroblocks. With the compat derivation 29 of
    // the 30 go bit-exact; measured 2777/950400 ≈ 0.29 % (max 64),
    // of which everything except ONE macroblock is ±1 near-tie
    // propagation through the last GOP's anchor chain.
    //
    // The residual macroblock (display frame 19, mb (6,8)) is
    // root-caused as far as black-box pixels allow: an exhaustive
    // per-field search over (MV, reference-field, mode, residual
    // placement) determines its oracle reconstruction uniquely as
    // bidirectional field MC with mvb[0] = (-3,36) against the top
    // backward field and mvb[1] = (-2,20) against the bottom — values
    // consistent with the *printed* §7.7.2.2 formulas (including the
    // literal `MVD[i]` gate reading an untransmitted MVD[1] as zero)
    // evaluated from a co-located field-MV set of ((4,-48), (3..4,-30))
    // — while the bitstream-reconstructed co-located MVs are
    // ((8,-48), (1,30)). The co-located anchor region is flat, so the
    // oracle's internal anchor MV state cannot be determined from
    // pixels; settling whether the ecosystem derivation is
    // conditional-zero or real-MV-with-different-anchor-state needs
    // the staged fixture's parser dump (docs ask filed).
    assert_stream_bounded_with(
        "ilaced_direct_176x144.m4v",
        "ilaced_direct_176x144.yuv",
        176,
        144,
        64,
        0.005,
        DecodeOptions::ecosystem(),
    );
}

#[test]
fn compat_is_a_no_op_for_the_interlaced_qpel_stream() {
    // Neither compat behaviour is exercised here (quant_type == 0, no
    // interlaced-direct macroblocks): the compat decode must be
    // sample-identical to the spec decode, and the §7.7.2.1
    // field-qpel envelope (docs ask #279) is unchanged.
    let spec = decode_stream("ilaced_qpel_ip_64x64.m4v", DecodeOptions::spec());
    let compat = decode_stream("ilaced_qpel_ip_64x64.m4v", DecodeOptions::ecosystem());
    assert_eq!(spec.len(), compat.len());
    for (i, (a, b)) in spec.iter().zip(compat.iter()).enumerate() {
        assert_eq!(a.luma_samples(), b.luma_samples(), "frame {i} luma");
        assert_eq!(a.cb_samples(), b.cb_samples(), "frame {i} cb");
        assert_eq!(a.cr_samples(), b.cr_samples(), "frame {i} cr");
    }
    assert_stream_bounded_with(
        "ilaced_qpel_ip_64x64.m4v",
        "ilaced_qpel_ip_64x64.yuv",
        64,
        64,
        120,
        0.02,
        DecodeOptions::ecosystem(),
    );
}

#[test]
fn compat_keeps_the_bit_exact_streams_bit_exact() {
    // Progressive streams without method-1 quantisation exercise
    // neither compat behaviour — the pinned bit-exact results must
    // hold identically under compat.
    for (m4v, yuv, w, h) in [
        ("ipb_64x64.m4v", "ipb_64x64.yuv", 64, 64),
        ("qpel_mv4_ipb_64x64.m4v", "qpel_mv4_ipb_64x64.yuv", 64, 64),
        ("dp_ipb_64x64.m4v", "dp_ipb_64x64.yuv", 64, 64),
        ("ilaced_ip2_64x64.m4v", "ilaced_ip2_64x64.yuv", 64, 64),
    ] {
        let (max, differing, total) = stream_diff_with(m4v, yuv, w, h, DecodeOptions::ecosystem());
        assert!(
            differing == 0,
            "{m4v} under compat: {differing}/{total} differ (max {max})"
        );
    }
}

#[test]
fn compat_keeps_the_near_exact_streams_within_their_budgets() {
    // The near-tie streams contain no method-1 blocks and no
    // interlaced-direct macroblocks: their spec-mode budgets hold
    // unchanged under compat.
    for (m4v, yuv, budget) in [
        ("mv4_ipb_64x64.m4v", "mv4_ipb_64x64.yuv", 7usize),
        ("ilaced_intra_64x64.m4v", "ilaced_intra_64x64.yuv", 3),
        (
            "ilaced_altscan_intra_64x64.m4v",
            "ilaced_altscan_intra_64x64.yuv",
            3,
        ),
        ("ilaced_ip_64x64.m4v", "ilaced_ip_64x64.yuv", 5),
    ] {
        assert_stream_near_exact_with(m4v, yuv, 64, 64, budget, DecodeOptions::ecosystem());
    }
}

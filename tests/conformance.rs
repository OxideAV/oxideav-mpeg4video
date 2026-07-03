//! Real-stream conformance: decode reference-encoder-produced MPEG-4
//! Visual elementary streams and compare every sample against the
//! reference decode.
//!
//! The fixtures under `tests/fixtures/` were produced by a black-box
//! reference encoder (64×64 `testsrc` pattern, Simple Profile settings)
//! together with the matching raw `yuv420p` reference decode of each
//! stream in display order. No external implementation source was
//! consulted — the binaries were used opaquely as encode/decode oracles.
//!
//! Tolerance: ISO/IEC 14496-2 Annex A.1 specifies the IDCT only up to
//! the IEEE 1180-1990 accuracy envelope, so two conforming decoders may
//! legitimately differ by ±1 per sample per intra pass, and prediction
//! drift can accumulate a little further across a P/B chain. The
//! assertions therefore allow a small per-sample tolerance and a tiny
//! fraction of differing samples, while requiring exact frame counts
//! and plane sizes.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;

const W: usize = 64;
const H: usize = 64;
const Y_LEN: usize = W * H;
const C_LEN: usize = (W / 2) * (H / 2);

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Split a raw `yuv420p` dump into per-frame (Y, U, V) plane slices.
fn yuv_frames(data: &[u8]) -> Vec<(&[u8], &[u8], &[u8])> {
    let frame_len = Y_LEN + 2 * C_LEN;
    assert_eq!(data.len() % frame_len, 0, "yuv length must be whole frames");
    data.chunks_exact(frame_len)
        .map(|f| (&f[..Y_LEN], &f[Y_LEN..Y_LEN + C_LEN], &f[Y_LEN + C_LEN..]))
        .collect()
}

fn decode_stream(name: &str) -> Vec<DecodedFrame> {
    let stream = fixture(name);
    let mut dec = Mpeg4VideoDecoder::new();
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

fn assert_stream_matches(m4v: &str, yuv: &str, max_tol: u32, frac_tol: f64) {
    let frames = decode_stream(m4v);
    let reference = fixture(yuv);
    let reference = yuv_frames(&reference);
    assert_eq!(
        frames.len(),
        reference.len(),
        "{m4v}: display-order frame count"
    );
    for (idx, (frame, &(y, u, v))) in frames.iter().zip(reference.iter()).enumerate() {
        let (max, differing, total) = diff_stats(frame, y, u, v);
        let frac = differing as f64 / total as f64;
        assert!(
            max <= max_tol && frac <= frac_tol,
            "{m4v} frame {idx}: max abs diff {max} (tol {max_tol}), \
             {differing}/{total} samples differ ({frac:.4} > {frac_tol})"
        );
    }
}

// Tolerance rationale: our IDCT is the double-precision orthonormal
// Annex A.1 transform (accurate to the ideal within rounding); the
// reference decoder uses an integer IDCT approximation that is itself
// allowed to deviate by ±1 per sample under IEEE 1180. Two conforming
// decoders can therefore legitimately differ by up to 2 per sample on
// an intra pass (opposite-direction rounding), with about a third of
// the busy-texture samples differing by 1. Prediction chains carry the
// per-pass difference forward, so a long P/B chain may show isolated
// samples one step further out. Frame counts, display order, and
// motion/header decode must be exact — any real decode error shows up
// as a large localized diff, far outside these envelopes.

#[test]
fn intra_only_stream_matches_reference_decode() {
    // 3 intra frames (with §6.2.5.2 video packets): the twin-IDCT
    // envelope only.
    assert_stream_matches("intra_64x64.m4v", "intra_64x64.yuv", 2, 0.40);
}

#[test]
fn ip_stream_matches_reference_decode() {
    // I + 4 P frames: half-pel MC and the §7.6.5 chroma-MV derivation
    // are exact integer math, so only IDCT rounding flows through the
    // prediction chain (no drift growth across the GOP).
    assert_stream_matches("ip_64x64.m4v", "ip_64x64.yuv", 2, 0.40);
}

#[test]
fn ipb_stream_matches_reference_decode() {
    // I/P/B with 2 consecutive B-VOPs: exercises the §6.1.3.8 reorder,
    // §7.6.7 TRB/TRD derivation, the §6.2.6 co_located_not_coded
    // zero-bit B macroblocks, B-VOP video packets (18-bit resync
    // markers + ahead-of-raster resume points), and the §7.6.9
    // prediction modes. A couple of samples reach 3 after two
    // prediction passes.
    assert_stream_matches("ipb_64x64.m4v", "ipb_64x64.yuv", 3, 0.40);
}

#[test]
fn qpel_ip_stream_matches_reference_decode() {
    // §7.6.2.2 quarter-sample mode (VOL `quarter_sample == 1`): I + 4 P
    // frames encoded with quarter-pel motion. Exercises the 8-tap FIR
    // half-pel stage, the bilinear quarter positions, the Figure 7-30
    // per-block boundary mirroring, and the §7.6.5 quarter→half chroma
    // MV reduction (K = 1 on the eighth-sample grid / Table 7-12). The
    // FIR is exact integer math, so only the twin-IDCT envelope flows
    // through the prediction chain — isolated samples reach 3 after
    // four prediction passes, as in the I/P/B stream.
    assert_stream_matches("qpel_ip_64x64.m4v", "qpel_ip_64x64.yuv", 3, 0.40);
}

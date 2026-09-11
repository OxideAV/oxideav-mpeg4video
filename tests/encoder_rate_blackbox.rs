//! Black-box pins of the budget-driven rate control: registry-built
//! streams whose per-macroblock `dquant` / `dbquant` steps are driven
//! by the GOP budget (one-pass, and two-pass through the direct-API
//! statistics round trip) — deterministic against the committed
//! fixtures, decoded bit-exactly by the crate's own decoder against
//! the opaque reference decode (`tests/fixtures/NOTES.md`), with the
//! achieved rate and PSNR reported.

use oxideav_core::Encoder as _;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

const W: usize = 96;
const H: usize = 64;
const FRAMES: usize = 24;
const BITRATE: u32 = 300_000;

/// A panning textured scene whose second half is busier (a second
/// texture layer switches on at frame 12), so the budget has a
/// complexity step to absorb.
fn picture(frame_index: usize) -> Planes {
    let (cw, ch) = (W / 2, H / 2);
    let scene = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5).rem_euclid(160)
            + (x * 3 - y * 11).rem_euclid(97)
            + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
        (30 + v.rem_euclid(200)) as u8
    };
    let n = frame_index as i64;
    let mut y = vec![0u8; W * H];
    for row in 0..H {
        for col in 0..W {
            let mut v = i32::from(scene(col as i64 + n * 2, row as i64 + n));
            if frame_index >= 12 {
                v += ((col * 5 + row * 3 + frame_index) % 23) as i32 - 11;
            }
            y[row * W + col] = v.clamp(16, 235) as u8;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = scene(col as i64 + n, row as i64) / 2 + 60;
            cr[row * cw + col] = 128 + ((col as i64 * 3 + n) % 9) as u8 * 4;
        }
    }
    (y, cb, cr)
}

fn params(extra: &[(&str, &str)]) -> oxideav_core::CodecParameters {
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(W as u32);
    params.height = Some(H as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    let mut opts = oxideav_core::CodecOptions::default()
        .set("bitrate", BITRATE.to_string())
        .set("gop-size", "12")
        .set("bf", "2")
        .set("mb-aq", "true")
        .set("four-mv", "true")
        .set("qp", "6");
    for (k, v) in extra {
        opts = opts.set(*k, v.to_string());
    }
    params.options = opts;
    params
}

fn run(
    mut enc: oxideav_mpeg4video::encoder::Mpeg4VideoEncoder,
) -> (Vec<u8>, oxideav_mpeg4video::encoder::Mpeg4VideoEncoder) {
    for k in 0..FRAMES {
        let (y, cb, cr) = picture(k);
        let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
            pts: None,
            planes: vec![
                oxideav_core::VideoPlane { stride: W, data: y },
                oxideav_core::VideoPlane {
                    stride: W / 2,
                    data: cb,
                },
                oxideav_core::VideoPlane {
                    stride: W / 2,
                    data: cr,
                },
            ],
        });
        enc.send_frame(&frame).unwrap();
    }
    enc.flush().unwrap();
    let mut stream = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => stream.extend_from_slice(&p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    (stream, enc)
}

fn decode_all(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn pin_fixture(name: &str, bytes: &[u8]) {
    let path = fixture_path(name);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
        return;
    }
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        committed == bytes,
        "{name}: encoder output drifted from the committed fixture \
         ({} vs {} bytes) — regenerate + re-run the black-box decode",
        bytes.len(),
        committed.len()
    );
}

fn diff_against_yuv(frames: &[DecodedFrame], yuv: &[u8]) -> (usize, u32, usize) {
    let frame_len = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(yuv.len(), frames.len() * frame_len, "reference frame count");
    let mut differing = 0usize;
    let mut max = 0u32;
    let mut total = 0usize;
    for (k, f) in frames.iter().enumerate() {
        let r = &yuv[k * frame_len..(k + 1) * frame_len];
        let c = (W / 2) * (H / 2);
        for (ours, theirs) in [
            (f.luma_samples(), &r[..W * H]),
            (f.cb_samples(), &r[W * H..W * H + c]),
            (f.cr_samples(), &r[W * H + c..]),
        ] {
            for (&a, &b) in ours.iter().zip(theirs.iter()) {
                total += 1;
                let d = (i32::from(a) - i32::from(b)).unsigned_abs();
                if d != 0 {
                    differing += 1;
                    max = max.max(d);
                }
            }
        }
    }
    (differing, max, total)
}

/// Luma PSNR of the decoded frames against the source pictures.
fn luma_psnr(frames: &[DecodedFrame]) -> f64 {
    let mut sse = 0f64;
    for (k, f) in frames.iter().enumerate() {
        let (y, _, _) = picture(k);
        for (&a, &b) in f.luma_samples().iter().zip(y.iter()) {
            let d = f64::from(a) - f64::from(b);
            sse += d * d;
        }
    }
    let mse = sse / (frames.len() * W * H) as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn check(name: &str, stream: &[u8]) {
    pin_fixture(&format!("{name}.m4v"), stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let frames = decode_all(stream);
    assert_eq!(frames.len(), FRAMES);
    let rate = stream.len() as f64 * 8.0 / (FRAMES as f64 / 25.0);
    let ratio = rate / f64::from(BITRATE);
    let psnr = luma_psnr(&frames);
    println!("{name}: {rate:.0} b/s (×{ratio:.3} of target), luma PSNR {psnr:.2} dB");
    assert!(
        (0.92..=1.08).contains(&ratio),
        "{name}: rate ratio {ratio:.3}"
    );
    let yuv = std::fs::read(fixture_path(&format!("{name}.yuv"))).unwrap();
    let (differing, max, total) = diff_against_yuv(&frames, &yuv);
    assert_eq!(
        (differing, max),
        (0, 0),
        "{name}: {differing}/{total} samples differ from the reference decode (max {max})"
    );
}

/// One-pass budget mode: the GOP planner + per-macroblock regulator
/// drive `dquant` on I/P and `dbquant` on B macroblocks; the reference
/// decode is bit-exact.
#[test]
fn blackbox_budget_one_pass_ipb_is_bit_exact() {
    let enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params(&[])).unwrap();
    let (stream, _) = run(enc);
    check("enc_ipb_rcbudget_96x64", &stream);
}

/// Two-pass through the direct API: the first run's statistics feed a
/// second encoder (`with_first_pass_stats`), whose stream is pinned and
/// bit-exact in the reference decode.
#[test]
fn blackbox_budget_two_pass_ipb_is_bit_exact() {
    let first = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params(&[])).unwrap();
    let (_, first) = run(first);
    let stats = first.first_pass_stats().clone();
    assert_eq!(stats.frames.len(), FRAMES);
    let second = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params(&[]))
        .unwrap()
        .with_first_pass_stats(stats)
        .unwrap();
    let (stream, _) = run(second);
    check("enc_ipb_rc2pass_96x64", &stream);
}

/// Budget mode on an interlaced VOL (ecosystem-compat B syntax so the
/// reference reads the stream as we do): budget-driven `dquant` on
/// field-DCT / field-predicted I/P macroblocks and `dbquant` on
/// interlaced B macroblocks; the reference decode is bit-exact.
#[test]
fn blackbox_budget_interlaced_compat_ipb_is_bit_exact() {
    let enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params(&[
        ("interlaced", "true"),
        ("ecosystem-compat", "true"),
        ("qpel", "true"),
    ]))
    .unwrap();
    let (stream, _) = run(enc);
    check("enc_ilaced_rcbudget_compat_96x64", &stream);
}

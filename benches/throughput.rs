//! Decode / encode throughput over caller-supplied streams.
//!
//! No stream is bundled: the bench reads its inputs from the
//! environment and prints nothing but a skip notice when they are
//! absent.
//!
//! ```text
//! MPEG4VIDEO_BENCH_DECODE=a.m4v:b.m4v cargo bench --bench throughput
//! MPEG4VIDEO_BENCH_ENCODE=1280x720:src.yuv[:key=value,...] cargo bench --bench throughput
//! ```
//!
//! `MPEG4VIDEO_BENCH_DECODE` is a `:`-separated list of elementary
//! streams; each is decoded `MPEG4VIDEO_BENCH_REPEAT` times (default 3)
//! and the best wall time is reported together with an FNV-1a hash of
//! every decoded sample (the bit-identity witness across changes).
//! `MPEG4VIDEO_BENCH_COMPAT=1` decodes in the ecosystem-compat mode and
//! `MPEG4VIDEO_BENCH_INVERSE_TRANSFORM=exact|variant-rc|from-stream`
//! forces the inverse-transform selection (default: the mode's own).
//! `MPEG4VIDEO_BENCH_ENCODE` names a raw `yuv420p` file with its
//! dimensions and optional encoder options; the hash covers the
//! emitted bytes.

use std::time::Instant;

use oxideav_core::{CodecId, CodecOptions, CodecParameters, Error, Frame, VideoFrame, VideoPlane};
use oxideav_mpeg4video::compat::{DecodeOptions, InverseTransformSelection};
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn repeats() -> usize {
    std::env::var("MPEG4VIDEO_BENCH_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// The decode options selected by the environment (see the file
/// header).
fn decode_options() -> DecodeOptions {
    let compat = std::env::var("MPEG4VIDEO_BENCH_COMPAT").is_ok_and(|v| v == "1");
    let options = DecodeOptions::spec().with_ecosystem_compat(compat);
    match std::env::var("MPEG4VIDEO_BENCH_INVERSE_TRANSFORM").as_deref() {
        Ok("exact") => options.with_inverse_transform(InverseTransformSelection::Exact),
        Ok("variant-rc") => options.with_inverse_transform(InverseTransformSelection::VariantRc),
        Ok("from-stream") => options.with_inverse_transform(InverseTransformSelection::FromStream),
        _ => options,
    }
}

fn bench_decode(path: &str) {
    let Ok(stream) = std::fs::read(path) else {
        println!("decode {path}: unreadable, skipped");
        return;
    };
    let mut best = f64::INFINITY;
    let mut frames_out = 0usize;
    let mut hash = 0u64;
    for _ in 0..repeats() {
        let start = Instant::now();
        let mut dec = Mpeg4VideoDecoder::with_options(decode_options());
        let mut frames = match dec.decode(&stream) {
            Ok(f) => f,
            Err(e) => {
                println!("decode {path}: {e}");
                return;
            }
        };
        frames.extend(dec.flush());
        best = best.min(start.elapsed().as_secs_f64());
        frames_out = frames.len();
        if hash == 0 {
            hash = 0xCBF2_9CE4_8422_2325;
            for f in &frames {
                fnv1a(&mut hash, f.luma_samples());
                fnv1a(&mut hash, f.cb_samples());
                fnv1a(&mut hash, f.cr_samples());
            }
        }
    }
    println!(
        "decode {path}: {frames_out} frames, {:.1} fps ({:.2} ms/frame), hash {hash:016x}",
        frames_out as f64 / best,
        1e3 * best / frames_out.max(1) as f64
    );
}

fn bench_encode(spec: &str) {
    let mut parts = spec.splitn(3, ':');
    let (Some(dims), Some(path)) = (parts.next(), parts.next()) else {
        println!("encode {spec}: expected WxH:path[:options]");
        return;
    };
    let options = parts.next().unwrap_or("");
    let Some((w, h)) = dims
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<usize>().ok()?, h.parse::<usize>().ok()?)))
    else {
        println!("encode {spec}: bad dimensions");
        return;
    };
    let Ok(yuv) = std::fs::read(path) else {
        println!("encode {path}: unreadable, skipped");
        return;
    };
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let frame_len = w * h + 2 * cw * ch;
    let count = yuv.len() / frame_len;
    let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    let mut bag = CodecOptions::default();
    for kv in options.split(',').filter(|s| !s.is_empty()) {
        if let Some((k, v)) = kv.split_once('=') {
            bag = bag.set(k, v);
        }
    }
    params.options = bag;
    let mut best = f64::INFINITY;
    let mut hash = 0u64;
    let mut bytes = 0usize;
    for _ in 0..repeats() {
        let mut enc = match oxideav_mpeg4video::encoder::make_encoder(&params) {
            Ok(e) => e,
            Err(e) => {
                println!("encode {spec}: {e}");
                return;
            }
        };
        hash = 0xCBF2_9CE4_8422_2325;
        bytes = 0;
        let start = Instant::now();
        let mut drain = |enc: &mut Box<dyn oxideav_core::Encoder>| loop {
            match enc.receive_packet() {
                Ok(p) => {
                    bytes += p.data.len();
                    fnv1a(&mut hash, &p.data);
                }
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        };
        for k in 0..count {
            let f = &yuv[k * frame_len..(k + 1) * frame_len];
            let frame = Frame::Video(VideoFrame {
                pts: Some(k as i64),
                planes: vec![
                    VideoPlane {
                        stride: w,
                        data: f[..w * h].to_vec(),
                    },
                    VideoPlane {
                        stride: cw,
                        data: f[w * h..w * h + cw * ch].to_vec(),
                    },
                    VideoPlane {
                        stride: cw,
                        data: f[w * h + cw * ch..].to_vec(),
                    },
                ],
            });
            enc.send_frame(&frame).expect("send_frame");
            drain(&mut enc);
        }
        enc.flush().expect("flush");
        drain(&mut enc);
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!(
        "encode {dims} {path} [{options}]: {count} frames, {:.2} fps ({:.1} ms/frame), {bytes} bytes, hash {hash:016x}",
        count as f64 / best,
        1e3 * best / count.max(1) as f64
    );
}

fn main() {
    let decode = std::env::var("MPEG4VIDEO_BENCH_DECODE").unwrap_or_default();
    let encode = std::env::var("MPEG4VIDEO_BENCH_ENCODE").unwrap_or_default();
    if decode.is_empty() && encode.is_empty() {
        println!(
            "throughput: set MPEG4VIDEO_BENCH_DECODE / MPEG4VIDEO_BENCH_ENCODE (see the file header); skipped"
        );
        return;
    }
    for path in decode.split(':').filter(|s| !s.is_empty()) {
        bench_decode(path);
    }
    for spec in encode.split(';').filter(|s| !s.is_empty()) {
        bench_encode(spec);
    }
}

//! End-to-end benchmark: encode one MPEG-4 Part 2 I-VOP + decode back.
//!
//! Exercises the full pipeline — VOS/VO/VOL/VOP header parsing, MB loop,
//! VLC walks, block dequant + IDCT, and picture plane writes. A 256×256
//! frame is ≈256 macroblocks, representative of low-res streaming content
//! while still running in well under a second per iteration.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg4video::{decoder::make_decoder, encoder::make_encoder, CODEC_ID_STR};

const W: u32 = 256;
const H: u32 = 256;

fn synth_frame(phase: u32) -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let v = ((col as u32 * 3) + (row as u32 * 2) + phase) as i32
                + ((col as i32 ^ row as i32) & 31);
            y[row * w + col] = (v.clamp(0, 255)) as u8;
        }
    }
    let mut cb = vec![128u8; cw * ch];
    let mut cr = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = 128u8.wrapping_add((col as u8) & 31);
            cr[row * cw + col] = 128u8.wrapping_sub((row as u8) & 31);
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
        pts: Some(phase as i64),
        time_base: TimeBase::new(1, 25),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn encode_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(25, 1));
    p
}

fn encode_gop(frames: &[VideoFrame]) -> Vec<u8> {
    let params = encode_params();
    let mut enc = make_encoder(&params).expect("encoder");
    let mut data = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        loop {
            match enc.receive_packet() {
                Ok(p) => data.extend_from_slice(&p.data),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("enc: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("enc: {e}"),
        }
    }
    data
}

fn bench_encode_i(c: &mut Criterion) {
    let frame = synth_frame(0);
    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("mpeg4_i_256x256", |b| {
        b.iter(|| encode_gop(std::slice::from_ref(&frame)));
    });
    group.finish();
}

fn bench_decode_i(c: &mut Criterion) {
    let frame = synth_frame(0);
    let bytes = encode_gop(std::slice::from_ref(&frame));
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("mpeg4_i_256x256", |b| {
        b.iter(|| {
            let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            let mut dec = make_decoder(&params).expect("decoder");
            let pkt = Packet::new(0, TimeBase::new(1, 25), bytes.clone());
            dec.send_packet(&pkt).expect("send");
            dec.flush().expect("flush");
            let mut frames = 0;
            loop {
                match dec.receive_frame() {
                    Ok(_) => frames += 1,
                    Err(Error::NeedMore) | Err(Error::Eof) => break,
                    Err(e) => panic!("dec: {e}"),
                }
            }
            frames
        });
    });
    group.finish();
}

fn bench_decode_ippp(c: &mut Criterion) {
    // 3-frame IPP GOP to exercise the P-VOP decode path (motion
    // compensation + inter texture).
    let frames: Vec<VideoFrame> = (0..3).map(synth_frame).collect();
    let bytes = encode_gop(&frames);
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(3));
    group.bench_function("mpeg4_ippp_256x256", |b| {
        b.iter(|| {
            let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            let mut dec = make_decoder(&params).expect("decoder");
            let pkt = Packet::new(0, TimeBase::new(1, 25), bytes.clone());
            dec.send_packet(&pkt).expect("send");
            dec.flush().expect("flush");
            let mut frames = 0;
            loop {
                match dec.receive_frame() {
                    Ok(_) => frames += 1,
                    Err(Error::NeedMore) | Err(Error::Eof) => break,
                    Err(e) => panic!("dec: {e}"),
                }
            }
            frames
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode_i, bench_decode_i, bench_decode_ippp);
criterion_main!(benches);

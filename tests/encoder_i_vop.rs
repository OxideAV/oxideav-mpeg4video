//! I-VOP encoder validation: self-encode → decode through the crate's
//! own end-to-end decoder → sample-exact agreement with the encoder's
//! closed-loop reconstruction, plus rate/distortion sanity.
//!
//! The synthetic sources are fully deterministic (gradients + an LCG
//! texture field) so every assertion is reproducible byte-for-byte.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// Deterministic LCG (numerical recipes constants).
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// A deterministic 4:2:0 test picture: smooth gradients with a block
/// of LCG texture and a frame-indexed moving bright square (so
/// successive frames differ).
struct TestPicture {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: usize,
    height: usize,
}

impl TestPicture {
    fn synthesise(width: usize, height: usize, frame_index: usize) -> Self {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let mut y = vec![0u8; width * height];
        let mut state = 0x1234_5678u32 ^ (frame_index as u32).wrapping_mul(0x9E37_79B9);
        for row in 0..height {
            for col in 0..width {
                let grad = (row * 3 + col * 2) as i32 % 200 + 20;
                let noise = (lcg(&mut state) >> 28) as i32; // 0..=15
                let mut v = grad + noise;
                // Moving bright square.
                let bx = (frame_index * 7) % width.max(1);
                if col >= bx && col < (bx + 12).min(width) && row >= 8 && row < 20.min(height) {
                    v = 235;
                }
                y[row * width + col] = v.clamp(16, 235) as u8;
            }
        }
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for row in 0..ch {
            for col in 0..cw {
                cb[row * cw + col] = (96 + ((row + col + frame_index) % 64)) as u8;
                cr[row * cw + col] = (160u32.wrapping_sub((row * 2 + col) as u32 % 48)) as u8;
            }
        }
        Self {
            y,
            cb,
            cr,
            width,
            height,
        }
    }

    fn view(&self) -> FrameView<'_> {
        FrameView {
            y: &self.y,
            cb: &self.cb,
            cr: &self.cr,
            width: self.width,
            height: self.height,
        }
    }
}

/// PSNR of the visible area of `recon` against the source plane set.
fn psnr(pic: &TestPicture, recon: &DecodedFrame) -> f64 {
    let rw = recon.width();
    let rcw = rw / 2;
    let mut se = 0f64;
    let mut n = 0usize;
    for row in 0..pic.height {
        for col in 0..pic.width {
            let d = f64::from(pic.y[row * pic.width + col])
                - f64::from(recon.luma_samples()[row * rw + col]);
            se += d * d;
            n += 1;
        }
    }
    let (cw, ch) = (pic.width.div_ceil(2), pic.height.div_ceil(2));
    for (src, rec) in [(&pic.cb, recon.cb_samples()), (&pic.cr, recon.cr_samples())] {
        for row in 0..ch {
            for col in 0..cw {
                let d = f64::from(src[row * cw + col]) - f64::from(rec[row * rcw + col]);
                se += d * d;
                n += 1;
            }
        }
    }
    let mse = se / n as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Encode `frames` I-VOPs and return (stream bytes, per-VOP encoder
/// reconstructions).
fn encode_stream(
    cfg: &EncoderConfig,
    qp: u32,
    frames: usize,
) -> (Vec<u8>, Vec<DecodedFrame>, Vec<TestPicture>) {
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let mut stream = headers;
    let mut recons = Vec::new();
    let mut pics = Vec::new();
    let res = cfg.time_increment_resolution;
    for k in 0..frames {
        let pic = TestPicture::synthesise(usize::from(cfg.width), usize::from(cfg.height), k);
        let ticks = k as u32; // one tick per frame
        let modulo = ticks / u32::from(res);
        let prev_modulo = if k == 0 {
            0
        } else {
            (ticks - 1) / u32::from(res)
        };
        let (unit, recon) = encode_i_vop(
            &vol,
            cfg,
            &pic.view(),
            modulo - prev_modulo,
            (ticks % u32::from(res)) as u16,
            qp,
        );
        stream.extend_from_slice(&unit);
        recons.push(recon);
        pics.push(pic);
    }
    (stream, recons, pics)
}

fn assert_frames_match(decoded: &[DecodedFrame], recons: &[DecodedFrame]) {
    assert_eq!(decoded.len(), recons.len(), "frame count");
    for (k, (d, r)) in decoded.iter().zip(recons.iter()).enumerate() {
        assert_eq!(d.width(), r.width());
        assert_eq!(d.height(), r.height());
        assert_eq!(d.luma_samples(), r.luma_samples(), "frame {k} luma");
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

fn decode_full(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

#[test]
fn intra_stream_self_decodes_sample_exact_method2() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        ..EncoderConfig::default()
    };
    let (stream, recons, pics) = encode_stream(&cfg, 4, 3);
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);
    for (pic, recon) in pics.iter().zip(recons.iter()) {
        let p = psnr(pic, recon);
        assert!(p > 34.0, "qp4 method-2 PSNR {p:.2} dB below floor");
    }
}

#[test]
fn intra_stream_self_decodes_sample_exact_method1() {
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        quant_type: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, pics) = encode_stream(&cfg, 5, 2);
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);
    for (pic, recon) in pics.iter().zip(recons.iter()) {
        let p = psnr(pic, recon);
        assert!(p > 32.0, "qp5 method-1 PSNR {p:.2} dB below floor");
    }
}

#[test]
fn partial_edge_macroblocks_encode_and_decode() {
    // 40×24 → a 3×2 macroblock grid with 8-sample partial edges.
    let cfg = EncoderConfig {
        width: 40,
        height: 24,
        ..EncoderConfig::default()
    };
    let (stream, recons, pics) = encode_stream(&cfg, 6, 2);
    let decoded = decode_full(&stream);
    assert_eq!(decoded[0].width(), 48, "decoded frame covers the MB grid");
    assert_eq!(decoded[0].height(), 32);
    assert_frames_match(&decoded, &recons);
    let p = psnr(&pics[0], &recons[0]);
    assert!(p > 30.0, "qp6 PSNR {p:.2} dB below floor");
}

#[test]
fn encoding_is_deterministic() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        ..EncoderConfig::default()
    };
    let (a, _, _) = encode_stream(&cfg, 4, 2);
    let (b, _, _) = encode_stream(&cfg, 4, 2);
    assert_eq!(a, b, "same input must produce identical bytes");
}

#[test]
fn ac_prediction_never_costs_bits() {
    let base = EncoderConfig {
        width: 64,
        height: 64,
        ac_prediction: false,
        ..EncoderConfig::default()
    };
    let with_pred = EncoderConfig {
        ac_prediction: true,
        ..base
    };
    let (off, _, _) = encode_stream(&base, 4, 2);
    let (on, recons, _) = encode_stream(&with_pred, 4, 2);
    assert!(
        on.len() <= off.len(),
        "cost-decided ac_pred grew the stream: {} > {}",
        on.len(),
        off.len()
    );
    // And the ac_pred stream still decodes sample-exact.
    let decoded = decode_full(&on);
    assert_frames_match(&decoded, &recons);
}

#[test]
fn quantiser_sweep_rate_and_distortion_are_monotone_ish() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        ..EncoderConfig::default()
    };
    let mut sizes = Vec::new();
    let mut psnrs = Vec::new();
    for qp in [2u32, 8, 31] {
        let (stream, recons, pics) = encode_stream(&cfg, qp, 1);
        let decoded = decode_full(&stream);
        assert_frames_match(&decoded, &recons);
        sizes.push(stream.len());
        psnrs.push(psnr(&pics[0], &recons[0]));
    }
    assert!(
        sizes[0] > sizes[1] && sizes[1] > sizes[2],
        "rate falls with qp: {sizes:?}"
    );
    assert!(
        psnrs[0] > psnrs[1] && psnrs[1] > psnrs[2],
        "PSNR falls with qp: {psnrs:?}"
    );
}

//! §7.8 basic static sprites, both directions: the encoder emits a
//! static-sprite VOL (sprite I-VOP + trajectory-only S-VOPs), the
//! stream decoder stores the sprite and reconstructs every S-VOP by
//! the §7.8.4–§7.8.6 warp — sample-exact against the encoder's closed
//! loop for every warp class (stationary, translation, affine family,
//! perspective), with the trajectory fit recovering known motion.

use oxideav_mpeg4video::decoder::{Mpeg4VideoDecoder, StreamDecodeError};
use oxideav_mpeg4video::framestore::DecodedFrame;
use oxideav_mpeg4video::ivop_encode::{EncoderConfig, FrameView, StaticSpriteVol};
use oxideav_mpeg4video::sprite_encode::{StaticSpriteEncoder, write_static_sprite_vop};
use oxideav_mpeg4video::vol::SpriteWarpingAccuracy;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A smooth, aperiodic panorama (integer triangle waves).
fn panorama(w: usize, h: usize) -> Planes {
    let tri = |v: i64, period: i64| -> i64 { (v.rem_euclid(period) - period / 2).abs() };
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let (x, r) = (col as i64, row as i64);
            // The ramp keeps the triangle-wave lattice from repeating.
            let v = 10
                + tri(3 * x + 2 * r, 127)
                + tri(5 * r - 2 * x, 89)
                + tri(x + 3 * r, 53)
                + x / 4
                + r / 3;
            y[row * w + col] = v.clamp(0, 255) as u8;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = (96 + tri(col as i64 + row as i64, 41)) as u8;
            cr[row * cw + col] = (150 - tri(2 * col as i64 - row as i64, 37)) as u8;
        }
    }
    (y, cb, cr)
}

/// The `w × h` window of `pano` (width `pw`) whose top-left luminance
/// sample is `(x, y)` (even).
fn window(pano: &Planes, pw: usize, x: usize, y: usize, w: usize, h: usize) -> Planes {
    let cpw = pw.div_ceil(2);
    let mut luma = Vec::with_capacity(w * h);
    for row in 0..h {
        luma.extend_from_slice(&pano.0[(y + row) * pw + x..(y + row) * pw + x + w]);
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut cb = Vec::with_capacity(cw * ch);
    let mut cr = Vec::with_capacity(cw * ch);
    for row in 0..ch {
        let at = (y / 2 + row) * cpw + x / 2;
        cb.extend_from_slice(&pano.1[at..at + cw]);
        cr.extend_from_slice(&pano.2[at..at + cw]);
    }
    (luma, cb, cr)
}

fn view(p: &Planes, w: usize, h: usize) -> FrameView<'_> {
    FrameView {
        y: &p.0,
        cb: &p.1,
        cr: &p.2,
        width: w,
        height: h,
    }
}

fn decode_all(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

fn assert_exact(frames: &[DecodedFrame], recons: &[DecodedFrame]) {
    assert_eq!(frames.len(), recons.len(), "frame count");
    for (k, (d, r)) in frames.iter().zip(recons.iter()).enumerate() {
        assert!(d.luma_samples() == r.luma_samples(), "frame {k} luma");
        assert!(d.cb_samples() == r.cb_samples(), "frame {k} cb");
        assert!(d.cr_samples() == r.cr_samples(), "frame {k} cr");
    }
}

fn mean_abs_luma(frame: &DecodedFrame, source: &Planes, w: usize, h: usize) -> f64 {
    let stride = frame.luma_samples().len() / (h.div_ceil(16) * 16);
    let mut sum = 0u64;
    for row in 0..h {
        for col in 0..w {
            sum += u64::from(
                frame.luma_samples()[row * stride + col].abs_diff(source.0[row * w + col]),
            );
        }
    }
    sum as f64 / (w * h) as f64
}

/// A camera pan over a larger sprite at a non-zero sprite origin: the
/// fit recovers the pan exactly, the decode equals the closed loop and
/// the displayed pictures are the panorama windows up to the sprite's
/// quantisation error.
#[test]
fn pan_over_a_panorama_round_trips_and_fits() {
    let (sw, sh, w, h) = (176usize, 112usize, 64usize, 48usize);
    let (left, top) = (-24i16, 10i16);
    let pano = panorama(sw, sh);
    let cfg = EncoderConfig::default()
        .with_width(w as u16)
        .with_height(h as u16)
        .with_static_sprite(Some(
            StaticSpriteVol::new(sw as u16, sh as u16).with_origin(left, top),
        ));
    let (mut enc, mut stream) = StaticSpriteEncoder::new(&cfg, &view(&pano, sw, sh), 2);
    let mut recons = Vec::new();
    let path = [
        (40usize, 20usize),
        (46, 22),
        (52, 26),
        (60, 30),
        (72, 36),
        (90, 44),
    ];
    for (k, &(x, y)) in path.iter().enumerate() {
        let src = window(&pano, sw, x, y, w, h);
        let (unit, recon, trajectory) = enc.encode_vop(&view(&src, w, h), 0, k as u16);
        // §7.8.4: du[0] / dv[0] are the picture origin's position in
        // the sprite addressing space, in half samples.
        assert_eq!(
            trajectory[0],
            [
                2 * (i32::from(left) + x as i32),
                2 * (i32::from(top) + y as i32)
            ],
            "frame {k}"
        );
        assert!(mean_abs_luma(&recon, &src, w, h) < 2.0, "frame {k}");
        stream.extend_from_slice(&unit);
        recons.push(recon);
    }
    let frames = decode_all(&stream);
    // The sprite I-VOP is stored, never displayed.
    assert_exact(&frames, &recons);
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.pts_ticks(), Some(k as i64));
    }
}

/// Every warp class through caller-chosen trajectories, at two warp
/// accuracies and with a sprite that is not a whole number of
/// macroblocks: the decode equals the encoder's closed loop.
#[test]
fn every_warp_class_decodes_sample_exact() {
    let (sw, sh, w, h) = (150usize, 100usize, 48usize, 32usize);
    let pano = panorama(sw, sh);
    let trajectories: [[[i32; 2]; 4]; 3] = [
        [[60, 40], [0, 0], [0, 0], [0, 0]],
        [[61, 37], [5, -3], [-4, 6], [3, 2]],
        [[90, 20], [-9, 4], [7, 11], [-6, -5]],
    ];
    for accuracy in [
        SpriteWarpingAccuracy::HalfPel,
        SpriteWarpingAccuracy::SixteenthPel,
    ] {
        for points in 0..=4u8 {
            let cfg = EncoderConfig::default()
                .with_width(w as u16)
                .with_height(h as u16)
                .with_static_sprite(Some(
                    StaticSpriteVol::new(sw as u16, sh as u16)
                        .with_warping_points(points)
                        .with_accuracy(accuracy),
                ));
            let (mut enc, mut stream) = StaticSpriteEncoder::new(&cfg, &view(&pano, sw, sh), 4);
            let mut recons = Vec::new();
            for (k, trajectory) in trajectories.iter().enumerate() {
                let (unit, recon) = enc.encode_vop_with_trajectory(trajectory, 0, k as u16);
                stream.extend_from_slice(&unit);
                recons.push(recon);
            }
            assert_exact(&decode_all(&stream), &recons);
            if points >= 2 {
                // The model matters: a deformation shows.
                assert!(recons[1].luma_samples() != recons[0].luma_samples());
            }
        }
    }
}

/// The fit recovers a known three-point (affine) warp: the picture is
/// the decoder's own warp of the sprite under a chosen trajectory, and
/// the fitted stream reproduces it.
#[test]
fn affine_fit_recovers_a_known_warp() {
    let (sw, sh, w, h) = (160usize, 112usize, 64usize, 48usize);
    let pano = panorama(sw, sh);
    let cfg = EncoderConfig::default()
        .with_width(w as u16)
        .with_height(h as u16)
        .with_static_sprite(Some(
            StaticSpriteVol::new(sw as u16, sh as u16).with_warping_points(3),
        ));
    let (mut reference, _) = StaticSpriteEncoder::new(&cfg, &view(&pano, sw, sh), 2);
    let (mut enc, mut stream) = StaticSpriteEncoder::new(&cfg, &view(&pano, sw, sh), 2);
    let truth = [[72, 44], [6, -2], [3, 5], [0, 0]];
    let (_, target) = reference.encode_vop_with_trajectory(&truth, 0, 0);
    let src: Planes = (
        target.luma_samples().to_vec(),
        target.cb_samples().to_vec(),
        target.cr_samples().to_vec(),
    );
    let (unit, recon, fitted) = enc.encode_vop(&view(&src, w, h), 0, 0);
    stream.extend_from_slice(&unit);
    let err = mean_abs_luma(&recon, &src, w, h);
    // The best pure translation of the same picture, for scale.
    let translation_cfg = cfg.with_static_sprite(Some(
        StaticSpriteVol::new(sw as u16, sh as u16).with_warping_points(1),
    ));
    let (mut translation, _) = StaticSpriteEncoder::new(&translation_cfg, &view(&pano, sw, sh), 2);
    let (_, translated, _) = translation.encode_vop(&view(&src, w, h), 0, 0);
    let translation_err = mean_abs_luma(&translated, &src, w, h);
    assert!(
        err < 1.5 && 3.0 * err < translation_err,
        "fitted {fitted:?} vs {truth:?}: mean |diff| {err} (translation only {translation_err})"
    );
    assert_exact(&decode_all(&stream), &[recon]);
}

/// The registry path: `static-sprite` emits the sprite with the first
/// packet and one S-VOP per frame; the registry decoder returns one
/// picture per input frame.
#[test]
fn registry_static_sprite_stream_round_trips() {
    use oxideav_core::{
        CodecId, CodecOptions, CodecParameters, Error, Frame, VideoFrame, VideoPlane,
    };
    let (w, h) = (64usize, 48usize);
    let pano = panorama(w + 16, h + 16);
    let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.options = CodecOptions::default()
        .set("static-sprite", "true")
        .set("qp", "2");
    let mut enc = oxideav_mpeg4video::encoder::make_encoder(&params).unwrap();
    let mut stream = Vec::new();
    let mut sources = Vec::new();
    for k in 0..5usize {
        // The scene drifts by (2, 1) samples per frame; frame 0 is the
        // sprite.
        let src = window(&pano, w + 16, 2 * k, 2 * (k / 2), w, h);
        let frame = Frame::Video(VideoFrame {
            pts: Some(k as i64),
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: src.0.clone(),
                },
                VideoPlane {
                    stride: w / 2,
                    data: src.1.clone(),
                },
                VideoPlane {
                    stride: w / 2,
                    data: src.2.clone(),
                },
            ],
        });
        enc.send_frame(&frame).unwrap();
        sources.push(src);
    }
    enc.flush().unwrap();
    let mut packets = 0;
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                assert_eq!(p.flags.keyframe, packets == 0);
                stream.extend_from_slice(&p.data);
                packets += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    assert_eq!(packets, 5);
    let frames = decode_all(&stream);
    assert_eq!(frames.len(), 5);
    // Frame 0 is the sprite itself under the identity warp; the later
    // frames show the sprite displaced (edge samples replicate, so
    // only the first is held to the tight bound).
    assert!(mean_abs_luma(&frames[0], &sources[0], w, h) < 2.0);
    for (f, s) in frames.iter().zip(&sources).skip(1) {
        assert!(mean_abs_luma(f, s, w, h) < 8.0);
    }
    // Incompatible tools are refused with the clause.
    params.options = CodecOptions::default()
        .set("static-sprite", "true")
        .set("bf", "2");
    let err = match oxideav_mpeg4video::encoder::make_encoder(&params) {
        Err(e) => e,
        Ok(_) => panic!("static-sprite + bf must be rejected"),
    };
    assert!(err.to_string().contains("7.8.1"), "{err}");
}

/// Hostile / out-of-contract streams fail with typed errors.
#[test]
fn malformed_static_sprite_streams_are_rejected() {
    let (sw, sh, w, h) = (64usize, 48usize, 32usize, 32usize);
    let pano = panorama(sw, sh);
    let cfg = EncoderConfig::default()
        .with_width(w as u16)
        .with_height(h as u16)
        .with_static_sprite(Some(StaticSpriteVol::new(sw as u16, sh as u16)));
    let (_, head) = StaticSpriteEncoder::new(&cfg, &view(&pano, sw, sh), 4);
    let headers = oxideav_mpeg4video::ivop_encode::write_configuration_headers(&cfg);
    let s_vop = write_static_sprite_vop(25, 0, 1, 0, 1, &[[4, 4], [0, 0], [0, 0], [0, 0]]);

    // An S-VOP before the sprite I-VOP has nothing to warp.
    let mut stream = headers.clone();
    stream.extend_from_slice(&s_vop);
    let mut dec = Mpeg4VideoDecoder::new();
    assert!(matches!(
        dec.decode(&stream),
        Err(StreamDecodeError::MissingAnchor)
    ));

    // A truncated trajectory is a parse error, not a panic.
    for cut in 5..s_vop.len() {
        let mut stream = head.clone();
        stream.extend_from_slice(&s_vop[..cut]);
        let mut dec = Mpeg4VideoDecoder::new();
        let _ = dec.decode(&stream);
    }

    // Every single-bit corruption of the sprite head and the S-VOP
    // decodes or errors — never panics.
    let mut whole = head.clone();
    whole.extend_from_slice(&s_vop);
    for byte in headers.len().saturating_sub(12)..whole.len().min(headers.len() + 40) {
        for bit in 0..8 {
            let mut bad = whole.clone();
            bad[byte] ^= 1 << bit;
            let mut dec = Mpeg4VideoDecoder::new();
            let _ = dec.decode(&bad);
        }
    }
}

/// Regression (found by the `encode_roundtrip` fuzz target): on
/// unstructured content a four-point fit could settle on a trajectory
/// whose perspective denominator vanishes at a sample the fit's
/// sub-grid does not visit. A fitted trajectory now has to define the
/// warp over the whole coded picture.
#[test]
fn perspective_fit_on_unstructured_content_stays_defined() {
    use oxideav_core::{
        CodecId, CodecOptions, CodecParameters, Error, Frame, VideoFrame, VideoPlane,
    };
    let (w, h) = (64usize, 32usize);
    let payload = [0x7fu8, 0x06, 0x65, 0xff, 0xff, 0xfe];
    let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.options = CodecOptions::default()
        .set("qp", "21")
        .set("static-sprite", "true")
        .set("sprite-points", "4");
    let mut enc = oxideav_mpeg4video::encoder::make_encoder(&params).unwrap();
    let (cw, ch) = (w / 2, h / 2);
    for k in 0..3usize {
        let sample = |i: usize| -> u8 {
            payload[(i * 7 + k * 13) % payload.len()]
                .wrapping_add(((i / w) as u8).wrapping_mul(k as u8))
        };
        let frame = Frame::Video(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: (0..w * h).map(sample).collect(),
                },
                VideoPlane {
                    stride: cw,
                    data: (0..cw * ch).map(|i| sample(i + 1)).collect(),
                },
                VideoPlane {
                    stride: cw,
                    data: (0..cw * ch).map(|i| sample(i + 2)).collect(),
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
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    assert_eq!(decode_all(&stream).len(), 3);
}

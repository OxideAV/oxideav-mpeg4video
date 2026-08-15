//! B-VOP encoder validation: I/P/B self-encode → decode through the
//! crate's own end-to-end decoder → sample-exact agreement with the
//! encoder's closed-loop reconstructions, plus §7.6.9 mode behaviour,
//! reorder timing, and determinism.

use oxideav_core::Encoder as _;
use oxideav_mpeg4video::bvop_encode::{encode_b_vop, BVopEncodeStats};
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// The translating scene of the P-VOP tests: a fixed textured
/// background sampled at a (2, 1)-pel per-frame offset + a static
/// corner block.
fn picture(w: usize, h: usize, frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w / 2, h / 2);
    let bg = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5) % 160 + ((x / 9 + y / 7) % 13) * 6;
        (40 + v.rem_euclid(170)) as u8
    };
    let (ox, oy) = ((frame_index * 2) as i64, frame_index as i64);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    for row in 0..8.min(h) {
        for col in 0..8.min(w) {
            y[row * w + col] = 200;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + 64;
            cr[row * cw + col] = 128;
        }
    }
    (y, cb, cr)
}

fn decode_full(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

fn assert_frames_match(decoded: &[DecodedFrame], recons: &[DecodedFrame]) {
    assert_eq!(decoded.len(), recons.len(), "frame count");
    for (k, (d, r)) in decoded.iter().zip(recons.iter()).enumerate() {
        assert_eq!(d.luma_samples(), r.luma_samples(), "frame {k} luma");
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

/// Build an I B P coded stream (display I, B, P) with the low-level
/// entry points; return (stream, display-order recons, B stats).
fn build_ibp(cfg: &EncoderConfig, qp: u32) -> (Vec<u8>, Vec<DecodedFrame>, BVopEncodeStats) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let pics: Vec<_> = (0..3).map(|k| picture(w, h, k)).collect();
    let view = |k: usize| FrameView {
        y: &pics[k].0,
        cb: &pics[k].1,
        cr: &pics[k].2,
        width: w,
        height: h,
    };

    let mut store = FrameStore::new();
    let mut stream = headers;

    // I at ticks 0.
    let (i_unit, i_recon) = encode_i_vop(&vol, cfg, &view(0), 0, 0, qp);
    store.push_anchor(i_recon.clone());
    stream.extend_from_slice(&i_unit);

    // P (display index 2) at ticks 2 — coded before the B.
    let reference = store.backward().unwrap().clone();
    let (p_unit, _stats) = encode_p_vop(&vol, cfg, &view(2), &reference, 0, 2, qp);
    let (p_recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);

    // B (display index 1) at ticks 1: TRB = 1, TRD = 2.
    let (b_unit, b_recon, b_stats) =
        encode_b_vop(&vol, cfg, &view(1), &store, Some(&motion), 1, 2, 0, 1, qp);
    stream.extend_from_slice(&b_unit);

    (stream, vec![i_recon, b_recon, p_recon], b_stats)
}

#[test]
fn ibp_stream_self_decodes_sample_exact() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        b_vops: true,
        ..EncoderConfig::default()
    };
    assert_eq!(cfg.profile_and_level(), 0xF3, "B-VOPs select ASP");
    let (stream, recons, stats) = build_ibp(&cfg, 4);
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);
    // The B-VOP must actually exercise B machinery: some macroblocks
    // coded (any mode), and the static corner + translating background
    // make cheap direct/forward modes attractive.
    let coded = stats.modb_one + stats.direct + stats.forward + stats.backward + stats.interpolated;
    assert!(coded > 0, "B-VOP coded no macroblocks: {stats:?}");
}

#[test]
fn ibp_stream_qpel_and_method1_self_decode() {
    for (quant_type, quarter_sample) in [(true, false), (false, true), (true, true)] {
        let cfg = EncoderConfig {
            width: 64,
            height: 48,
            quant_type,
            quarter_sample,
            b_vops: true,
            ..EncoderConfig::default()
        };
        let (stream, recons, _stats) = build_ibp(&cfg, 5);
        let decoded = decode_full(&stream);
        assert_frames_match(&decoded, &recons);
    }
}

#[test]
fn ibp_byte_deterministic() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        b_vops: true,
        ..EncoderConfig::default()
    };
    let (a, _, _) = build_ibp(&cfg, 4);
    let (b, _, _) = build_ibp(&cfg, 4);
    assert_eq!(a, b);
}

/// A B-VOP over a fully-static scene against a fully-skipped P anchor
/// must transmit zero bits for every macroblock (§6.2.6
/// co_located_not_coded) — the smallest legal B-VOP.
#[test]
fn static_scene_b_vop_is_all_zero_bit() {
    let cfg = EncoderConfig {
        width: 48,
        height: 48,
        b_vops: true,
        ..EncoderConfig::default()
    };
    let (w, h) = (48usize, 48usize);
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let (y, cb, cr) = picture(w, h, 0);
    let view = FrameView {
        y: &y,
        cb: &cb,
        cr: &cr,
        width: w,
        height: h,
    };
    let mut store = FrameStore::new();
    let mut stream = headers;
    let (i_unit, i_recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
    store.push_anchor(i_recon.clone());
    stream.extend_from_slice(&i_unit);
    let reference = store.backward().unwrap().clone();
    let (p_unit, p_stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, 2, 4);
    assert_eq!(p_stats.skipped, 9, "static P must be all-skip");
    let (p_recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);
    let (b_unit, b_recon, b_stats) =
        encode_b_vop(&vol, &cfg, &view, &store, Some(&motion), 1, 2, 0, 1, 4);
    assert_eq!(b_stats.zero_bit, 9, "every co-located MB is skipped");
    // Header + stuffing only.
    assert!(
        b_unit.len() <= 8,
        "all-zero-bit B unit is {} B",
        b_unit.len()
    );
    stream.extend_from_slice(&b_unit);
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &[i_recon, b_recon, p_recon]);
}

/// Registry-level bf > 0: reordered packets (anchor first, then its
/// B run), display-order decode, Annex D-style dts, keyframe cadence.
#[test]
fn registry_encoder_bf2_round_trips_in_display_order() {
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default().set("bf", "2");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();

    let n = 8usize;
    let sources: Vec<_> = (0..n).map(|k| picture(64, 64, k)).collect();
    for (y, cb, cr) in &sources {
        let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
            pts: None,
            planes: vec![
                oxideav_core::VideoPlane {
                    stride: 64,
                    data: y.clone(),
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: cb.clone(),
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: cr.clone(),
                },
            ],
        });
        enc.send_frame(&frame).unwrap();
    }
    enc.flush().unwrap();

    let mut stream = Vec::new();
    let mut pts_list = Vec::new();
    let mut dts_list = Vec::new();
    let mut keyframes = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                pts_list.push(p.pts.unwrap());
                dts_list.push(p.dts.unwrap());
                keyframes.push(p.flags.keyframe);
                stream.extend_from_slice(&p.data);
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    // 8 frames → 8 packets; coding order I0 P3 B1 B2 P6 B4 B5 (flush) P7.
    assert_eq!(pts_list.len(), n);
    assert_eq!(pts_list, vec![0, 3, 1, 2, 6, 4, 5, 7]);
    assert_eq!(
        keyframes,
        vec![true, false, false, false, false, false, false, false]
    );
    // dts must be monotonically increasing and never exceed pts.
    for k in 1..dts_list.len() {
        assert!(
            dts_list[k] > dts_list[k - 1],
            "dts not monotonic at {k}: {dts_list:?}"
        );
    }
    for (p, d) in pts_list.iter().zip(&dts_list) {
        assert!(d <= p, "dts {d} exceeds pts {p}");
    }

    // The stream decodes to n frames in display order, matching the
    // sources closely (closed-loop exactness is asserted at the unit
    // level; here we pin display order + fidelity).
    let decoded = decode_full(&stream);
    assert_eq!(decoded.len(), n);
    for (k, frame) in decoded.iter().enumerate() {
        assert_eq!(
            frame.pts_ticks(),
            Some(k as i64),
            "display order broken at {k}"
        );
        let (y, _, _) = &sources[k];
        let rw = frame.width();
        let mut se = 0f64;
        for row in 0..64 {
            for col in 0..64 {
                let d =
                    f64::from(y[row * 64 + col]) - f64::from(frame.luma_samples()[row * rw + col]);
                se += d * d;
            }
        }
        let psnr = 10.0 * (255.0f64 * 255.0 / (se / (64.0 * 64.0))).log10();
        assert!(psnr > 30.0, "frame {k} luma PSNR {psnr:.2} dB below floor");
    }
}

//! Per-macroblock quantiser modulation: activity-classed `dquant`
//! (I-/P-VOPs, `intra+q` / `inter+q` types) and `dbquant` (B-VOPs)
//! emission, the §6.3.7 running quantiser mirrored on both sides,
//! sample-exact through the crate's own decoder.

use oxideav_mpeg4video::bvop_encode::encode_b_vop;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// A picture with every activity class: a flat left band, a smooth
/// gradient, a textured band and a full-range noise band, all
/// translating by (2, 1) pels per frame so P-/B-VOPs have motion.
fn mixed_picture(w: usize, h: usize, frame_index: usize) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let band = w / 4;
    let mut y = vec![0u8; w * h];
    let (ox, oy) = ((frame_index * 2) as i64, frame_index as i64);
    for row in 0..h {
        for col in 0..w {
            let (x, yy) = (col as i64 + ox, row as i64 + oy);
            let bx = (x.rem_euclid(w as i64) as usize) / band.max(1);
            let v: i64 = match bx {
                0 => 120,
                1 => 60 + (x + yy).rem_euclid(64) * 2,
                2 => 40 + ((x * 7 + yy * 5).rem_euclid(160)),
                _ => {
                    let mut s = (x as u32).wrapping_mul(0x9E37_79B9)
                        ^ (yy as u32).wrapping_mul(0x85EB_CA6B);
                    16 + i64::from(lcg(&mut s) >> 24) * 219 / 255
                }
            };
            y[row * w + col] = v.clamp(16, 235) as u8;
        }
    }
    let cb = vec![110u8; cw * ch];
    let cr = vec![140u8; cw * ch];
    (y, cb, cr)
}

fn vol_of(cfg: &EncoderConfig) -> (Vec<u8>, oxideav_mpeg4video::vol::VolHeader) {
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    (headers, vol)
}

fn assert_self_decodes(stream: &[u8], recons: &[DecodedFrame]) {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), recons.len(), "frame count");
    for (k, (d, r)) in frames.iter().zip(recons.iter()).enumerate() {
        assert_eq!(d.luma_samples(), r.luma_samples(), "frame {k} luma");
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

/// I + P + P with adaptive quantisation: `inter+q` / `intra+q` are
/// emitted, the stream differs from the constant-quantiser one, and
/// the decode is sample-exact.
#[test]
fn ip_dquant_round_trips_and_is_emitted() {
    let (w, h) = (96usize, 48usize);
    for quant_type in [false, true] {
        let cfg = EncoderConfig {
            width: w as u16,
            height: h as u16,
            quant_type,
            adaptive_quant: true,
            ..EncoderConfig::default()
        };
        let plain = EncoderConfig {
            adaptive_quant: false,
            ..cfg
        };
        let (headers, vol) = vol_of(&cfg);
        let mut stream = headers.clone();
        let mut plain_stream = headers;
        let mut recons = Vec::new();
        let mut store = FrameStore::new();
        let mut plain_store = FrameStore::new();
        let mut dquants = 0;
        for k in 0..3usize {
            let (y, cb, cr) = mixed_picture(w, h, k);
            let view = FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: w,
                height: h,
            };
            if k == 0 {
                let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 8);
                let (plain_unit, plain_recon) = encode_i_vop(&vol, &plain, &view, 0, 0, 8);
                assert_ne!(unit, plain_unit, "I-VOP dquant must change the stream");
                store.push_anchor(recon.clone());
                plain_store.push_anchor(plain_recon);
                stream.extend_from_slice(&unit);
                plain_stream.extend_from_slice(&plain_unit);
                recons.push(recon);
            } else {
                let reference = store.backward().unwrap().clone();
                let (unit, stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, 8);
                dquants += stats.dquant;
                let (recon, _) = reconstruct_own_p_vop_with_motion(&vol, &unit, &mut store);
                stream.extend_from_slice(&unit);
                recons.push(recon);
            }
        }
        assert!(
            dquants > 0,
            "P-VOPs must carry dquant (m{})",
            2 - u8::from(quant_type)
        );
        assert_self_decodes(&stream, &recons);
    }
}

/// With `four_mv` on, inter4v macroblocks (no `inter4v+q` type in
/// Table B.7) keep the running quantiser and the walk stays exact.
#[test]
fn dquant_with_inter4v_and_qpel_round_trips() {
    let (w, h) = (80usize, 48usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        four_mv: true,
        quarter_sample: true,
        adaptive_quant: true,
        ..EncoderConfig::default()
    };
    let (headers, vol) = vol_of(&cfg);
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..4usize {
        let (y, cb, cr) = mixed_picture(w, h, k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 6);
            store.push_anchor(recon.clone());
            stream.extend_from_slice(&unit);
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, _stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, 6);
            let (recon, _) = reconstruct_own_p_vop_with_motion(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }
    assert_self_decodes(&stream, &recons);
}

/// B-VOP `dbquant`: non-zero Table 6-33 steps are emitted on
/// non-direct coded macroblocks and the decode is sample-exact.
#[test]
fn b_vop_dbquant_round_trips_and_is_emitted() {
    let (w, h) = (96usize, 48usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        b_vops: true,
        adaptive_quant: true,
        ..EncoderConfig::default()
    };
    let (headers, vol) = vol_of(&cfg);
    let mut stream = headers;
    let mut store = FrameStore::new();
    let pics: Vec<Planes> = (0..3).map(|k| mixed_picture(w, h, k)).collect();
    let view = |k: usize| FrameView {
        y: &pics[k].0,
        cb: &pics[k].1,
        cr: &pics[k].2,
        width: w,
        height: h,
    };
    let (i_unit, i_recon) = encode_i_vop(&vol, &cfg, &view(0), 0, 0, 8);
    store.push_anchor(i_recon.clone());
    stream.extend_from_slice(&i_unit);
    let reference = store.backward().unwrap().clone();
    let (p_unit, _) = encode_p_vop(&vol, &cfg, &view(2), &reference, 0, 2, 8);
    let (p_recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);
    let (b_unit, b_recon, stats) =
        encode_b_vop(&vol, &cfg, &view(1), &store, Some(&motion), 1, 2, 0, 1, 8);
    stream.extend_from_slice(&b_unit);
    assert!(
        stats.dbquant > 0,
        "B-VOP must carry dbquant steps ({stats:?})"
    );
    // Display order I0 B1 P2.
    assert_self_decodes(&stream, &[i_recon, b_recon, p_recon]);
}

/// The registry option drives every VOP type; the stream stays
/// byte-deterministic and decodes.
#[test]
fn registry_mb_aq_option() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 48usize);
    let build = |aq: bool| -> Vec<u8> {
        let mut params =
            oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
        params.width = Some(w as u32);
        params.height = Some(h as u32);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        params.options = oxideav_core::CodecOptions::default()
            .set("mb-aq", if aq { "true" } else { "false" })
            .set("bf", "2")
            .set("qp", "10");
        let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
        for k in 0..6usize {
            let (y, cb, cr) = mixed_picture(w, h, k);
            let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
                pts: None,
                planes: vec![
                    oxideav_core::VideoPlane { stride: w, data: y },
                    oxideav_core::VideoPlane {
                        stride: w / 2,
                        data: cb,
                    },
                    oxideav_core::VideoPlane {
                        stride: w / 2,
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
        stream
    };
    let aq = build(true);
    assert_eq!(aq, build(true), "byte-deterministic");
    assert_ne!(aq, build(false));
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&aq).expect("own stream must decode");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), 6);
}

//! GMC (sprite trajectory) emission: S(GMC)-VOP anchors with one
//! §7.8.4 warping point — the global translation lands in the
//! `sprite_trajectory()`, `mcsel` splits GMC from local macroblocks,
//! and everything decodes sample-exact through the crate's own
//! decoder walk.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::svop_encode::{encode_s_vop, reconstruct_own_s_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A textured background translating globally by `shift` pels per
/// frame, plus a small stationary logo block (local motion differs
/// from global there).
fn picture(w: usize, h: usize, frame_index: usize, shift: (i64, i64)) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let bg = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5).rem_euclid(160) + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
        (40 + v.rem_euclid(170)) as u8
    };
    let (ox, oy) = (frame_index as i64 * shift.0, frame_index as i64 * shift.1);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    // Stationary bright logo in the top-left 16×16.
    for row in 0..16.min(h) {
        for col in 0..16.min(w) {
            y[row * w + col] = 210;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let cr = vec![130u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + 64;
        }
    }
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

fn decode_all(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

fn assert_exact(frames: &[DecodedFrame], recons: &[DecodedFrame]) {
    assert_eq!(frames.len(), recons.len(), "frame count");
    for (k, (d, r)) in frames.iter().zip(recons.iter()).enumerate() {
        assert_eq!(d.luma_samples(), r.luma_samples(), "frame {k} luma");
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

/// I + 3 S(GMC): the trajectory carries the global translation in
/// half-sample units, GMC macroblocks dominate, the stationary logo
/// goes local or intra, and the decode is sample-exact.
#[test]
fn global_translation_lands_in_the_trajectory() {
    let (w, h) = (96usize, 64usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        gmc: true,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (headers, vol) = vol_of(&cfg);
    assert!(matches!(
        vol.sprite_enable,
        oxideav_mpeg4video::vol::SpriteEnable::Gmc
    ));
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..4usize {
        let (y, cb, cr) = picture(w, h, k, (7, 3));
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 5);
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, stats) = encode_s_vop(&vol, &cfg, &view, &reference, 0, k as u16, 5);
            // (7, 3) pels per frame = (14, 6) half-sample units.
            assert_eq!(stats.trajectory, (14, 6), "S-VOP {k}: {stats:?}");
            assert!(
                stats.gmc + stats.gmc_skipped > stats.local + stats.intra,
                "S-VOP {k}: GMC must dominate ({stats:?})"
            );
            let (recon, _) = reconstruct_own_s_vop_with_motion(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }
    assert_exact(&decode_all(&stream), &recons);
}

/// A stationary scene: the trajectory is (0, 0) and every macroblock
/// collapses to a `not_coded` GMC copy — the S-VOP is tiny.
#[test]
fn stationary_scene_collapses_to_gmc_copies() {
    let (w, h) = (48usize, 48usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        gmc: true,
        ..EncoderConfig::default()
    };
    let (headers, vol) = vol_of(&cfg);
    let (y, cb, cr) = picture(w, h, 0, (0, 0));
    let view = FrameView {
        y: &y,
        cb: &cb,
        cr: &cr,
        width: w,
        height: h,
    };
    let mut store = FrameStore::new();
    let (i_unit, i_recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
    store.push_anchor(i_recon.clone());
    let reference = store.backward().unwrap().clone();
    let (s_unit, stats) = encode_s_vop(&vol, &cfg, &view, &reference, 0, 1, 4);
    assert_eq!(stats.trajectory, (0, 0));
    assert_eq!(stats.gmc_skipped, (w / 16) * (h / 16), "{stats:?}");
    assert!(
        s_unit.len() < 24,
        "all-copy S unit is {} bytes",
        s_unit.len()
    );
    let mut stream = headers;
    stream.extend_from_slice(&i_unit);
    stream.extend_from_slice(&s_unit);
    let (s_recon, motion) = reconstruct_own_s_vop_with_motion(&vol, &s_unit, &mut store);
    // Skipped GMC macroblocks contribute their averaged MV (here zero).
    assert!(motion.iter().all(
        |m| matches!(m, oxideav_mpeg4video::pvop_mv::PvopMbMotion::OneMv(v) if v.x == 0 && v.y == 0)
    ));
    assert_exact(&decode_all(&stream), &[i_recon, s_recon]);
}

/// GMC with quarter-sample, adaptive quant and video packets, plus
/// B-VOPs between S anchors, through the registry encoder.
#[test]
fn registry_gmc_full_toolset_round_trips() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let build = || -> Vec<u8> {
        let mut params =
            oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
        params.width = Some(w as u32);
        params.height = Some(h as u32);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        params.options = oxideav_core::CodecOptions::default()
            .set("gmc", "true")
            .set("qpel", "true")
            .set("mb-aq", "true")
            .set("bf", "2")
            .set("fcode", "2")
            .set("packet-bits", "600");
        let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
        for k in 0..6usize {
            let (y, cb, cr) = picture(w, h, k, (5, 2));
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
    let stream = build();
    assert_eq!(stream, build(), "byte-deterministic");
    // The stream contains S-VOPs (coding type 0b11 in the two bits
    // after each VOP start code, except the first I).
    let s_vops = stream
        .windows(5)
        .filter(|win| win[..4] == [0, 0, 1, 0xB6] && win[4] >> 6 == 0b11)
        .count();
    assert!(s_vops >= 1, "expected S-VOPs in the stream");
    let frames = decode_all(&stream);
    assert_eq!(frames.len(), 6);

    // gmc + data-partitioned is rejected.
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(32);
    params.height = Some(32);
    params.options = oxideav_core::CodecOptions::default()
        .set("gmc", "true")
        .set("data-partitioned", "true");
    assert!(oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).is_err());
}

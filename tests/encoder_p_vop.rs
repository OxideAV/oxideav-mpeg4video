//! P-VOP encoder validation: I+P self-encode → decode through the
//! crate's own end-to-end decoder → sample-exact agreement with the
//! encoder's closed-loop reconstructions, plus motion/skip behaviour
//! and rate sanity.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop, PVopEncodeStats};
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// A translating scene: a fixed textured background sampled at a
/// per-frame offset (global motion of (2, 1) pels per frame), so P
/// frames have real motion to chase.
struct Scene {
    width: usize,
    height: usize,
}

impl Scene {
    fn background(&self, x: i64, y: i64) -> u8 {
        // Smooth deterministic texture, independent of frame index.
        let v = (x * 7 + y * 5) % 160 + ((x / 9 + y / 7) % 13) * 6;
        (40 + v.rem_euclid(170)) as u8
    }

    fn picture(&self, frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (self.width, self.height);
        let (cw, ch) = (w / 2, h / 2);
        let (ox, oy) = ((frame_index * 2) as i64, frame_index as i64);
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = self.background(col as i64 + ox, row as i64 + oy);
            }
        }
        // Mild static noise on the first frame only would break motion;
        // keep planes purely translated + a static logo block so some
        // area is unchanged (skips).
        for row in 0..8.min(h) {
            for col in 0..8.min(w) {
                y[row * w + col] = 200;
            }
        }
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for row in 0..ch {
            for col in 0..cw {
                cb[row * cw + col] =
                    self.background(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + 64;
                cr[row * cw + col] = 128;
            }
        }
        (y, cb, cr)
    }
}

struct Encoded {
    stream: Vec<u8>,
    recons: Vec<DecodedFrame>,
    stats: Vec<PVopEncodeStats>,
    unit_sizes: Vec<usize>,
}

fn encode_ip_stream(cfg: &EncoderConfig, qp: u32, frames: usize, scene: &Scene) -> Encoded {
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let mut stream = headers;
    let mut recons = Vec::new();
    let mut stats = Vec::new();
    let mut unit_sizes = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..frames {
        let (y, cb, cr) = scene.picture(k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: scene.width,
            height: scene.height,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, cfg, &view, 0, 0, qp);
            unit_sizes.push(unit.len());
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().expect("anchor present").clone();
            let (unit, st) = encode_p_vop(&vol, cfg, &view, &reference, 0, k as u16, qp);
            unit_sizes.push(unit.len());
            let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
            stats.push(st);
        }
    }
    Encoded {
        stream,
        recons,
        stats,
        unit_sizes,
    }
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

fn psnr_luma(src: &[u8], recon: &DecodedFrame, w: usize, h: usize) -> f64 {
    let rw = recon.width();
    let mut se = 0f64;
    for row in 0..h {
        for col in 0..w {
            let d = f64::from(src[row * w + col]) - f64::from(recon.luma_samples()[row * rw + col]);
            se += d * d;
        }
    }
    let mse = se / (w * h) as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn ip_stream_self_decodes_sample_exact_with_motion() {
    let scene = Scene {
        width: 64,
        height: 64,
    };
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        ..EncoderConfig::default()
    };
    let enc = encode_ip_stream(&cfg, 4, 6, &scene);
    let decoded = decode_full(&enc.stream);
    assert_frames_match(&decoded, &enc.recons);

    // The P frames must actually use inter coding (global translation
    // of (2, 1) pels per frame is inside the search range).
    let inter_total: usize = enc.stats.iter().map(|s| s.inter + s.skipped).sum();
    let intra_total: usize = enc.stats.iter().map(|s| s.intra).sum();
    assert!(
        inter_total > intra_total * 3,
        "motion estimation barely used: inter+skip {inter_total} vs intra {intra_total}"
    );

    // P units must be much cheaper than the I unit.
    let i_size = enc.unit_sizes[0];
    for (k, &p_size) in enc.unit_sizes[1..].iter().enumerate() {
        assert!(
            p_size < i_size,
            "P unit {k} ({p_size} B) not smaller than the I unit ({i_size} B)"
        );
    }

    // Fidelity floor on each reconstructed frame.
    for (k, recon) in enc.recons.iter().enumerate() {
        let (y, _, _) = scene.picture(k);
        let p = psnr_luma(&y, recon, 64, 64);
        assert!(p > 32.0, "frame {k} luma PSNR {p:.2} dB below floor");
    }
}

#[test]
fn static_scene_collapses_to_skips() {
    let scene = Scene {
        width: 48,
        height: 48,
    };
    let cfg = EncoderConfig {
        width: 48,
        height: 48,
        ..EncoderConfig::default()
    };
    // Static: same picture every frame (frame_index fixed at 0).
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let (y, cb, cr) = scene.picture(0);
    let view = FrameView {
        y: &y,
        cb: &cb,
        cr: &cr,
        width: 48,
        height: 48,
    };
    let mut store = FrameStore::new();
    let mut stream = headers;
    let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
    stream.extend_from_slice(&unit);
    store.push_anchor(recon.clone());
    let reference = store.backward().unwrap().clone();
    let (p_unit, stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, 1, 4);
    // Every macroblock of an unchanged frame must be skipped: 3×3 MBs.
    assert_eq!(
        stats,
        PVopEncodeStats {
            skipped: 9,
            inter: 0,
            inter4v: 0,
            intra: 0,
            dquant: 0,
            packets: 0,
        }
    );
    // A fully-skipped P-VOP is tiny (header + 9 bits + stuffing).
    assert!(
        p_unit.len() < 16,
        "all-skip P unit is {} bytes",
        p_unit.len()
    );
    let p_recon = reconstruct_own_p_vop(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);
    // Skipped MBs copy the reference exactly.
    assert_eq!(p_recon.luma_samples(), recon.luma_samples());
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &[recon, p_recon]);
}

/// A scene whose 8×8 blocks alternate between two motion fields (a
/// checkerboard of (+2, +1)- and (-2, 0)-per-frame translations), so a
/// 16×16 macroblock straddles divergent motion and §6.3.7 inter4v wins.
fn divergent_picture(w: usize, h: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let scene = Scene {
        width: w,
        height: h,
    };
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let parity = (row / 8 + col / 8) % 2;
            let (ox, oy) = if parity == 0 {
                ((k * 2) as i64, k as i64)
            } else {
                (-((k * 2) as i64), 0)
            };
            y[row * w + col] = scene.background(col as i64 + ox, row as i64 + oy);
        }
    }
    let cb = vec![100u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    (y, cb, cr)
}

#[test]
fn divergent_block_motion_selects_inter4v_and_self_decodes() {
    let (w, h) = (64usize, 64usize);
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        four_mv: true,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let mut store = FrameStore::new();
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut four_mv_total = 0usize;
    for k in 0..4 {
        let (y, cb, cr) = divergent_picture(w, h, k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, 4);
            four_mv_total += stats.inter4v;
            let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }
    assert!(
        four_mv_total > 0,
        "divergent per-block motion never selected inter4v"
    );
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);

    // With four-mv disabled the same content still self-decodes (and
    // emits no inter4v macroblocks).
    let cfg_off = EncoderConfig {
        four_mv: false,
        ..cfg
    };
    let mut store = FrameStore::new();
    let mut stream = write_configuration_headers(&cfg_off);
    let mut recons = Vec::new();
    for k in 0..3 {
        let (y, cb, cr) = divergent_picture(w, h, k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg_off, &view, 0, 0, 4);
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, stats) = encode_p_vop(&vol, &cfg_off, &view, &reference, 0, k as u16, 4);
            assert_eq!(stats.inter4v, 0, "four_mv off must never emit inter4v");
            let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);
}

#[test]
fn four_mv_stream_is_byte_deterministic() {
    let scene = Scene {
        width: 64,
        height: 48,
    };
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        four_mv: true,
        ..EncoderConfig::default()
    };
    let a = encode_ip_stream(&cfg, 4, 4, &scene);
    let b = encode_ip_stream(&cfg, 4, 4, &scene);
    assert_eq!(a.stream, b.stream);
    let decoded = decode_full(&a.stream);
    assert_frames_match(&decoded, &a.recons);
}

#[test]
fn qpel_ip_stream_self_decodes_sample_exact() {
    let scene = Scene {
        width: 64,
        height: 64,
    };
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    assert_eq!(cfg.profile_and_level(), 0xF3, "qpel selects ASP");
    let enc = encode_ip_stream(&cfg, 4, 5, &scene);
    let decoded = decode_full(&enc.stream);
    assert_frames_match(&decoded, &enc.recons);
    let inter_total: usize = enc.stats.iter().map(|s| s.inter + s.skipped).sum();
    assert!(inter_total > 0, "qpel P-VOPs never used inter coding");
    for (k, recon) in enc.recons.iter().enumerate() {
        let (y, _, _) = scene.picture(k);
        let p = psnr_luma(&y, recon, 64, 64);
        assert!(p > 32.0, "qpel frame {k} luma PSNR {p:.2} dB below floor");
    }
    // Determinism.
    let again = encode_ip_stream(&cfg, 4, 5, &scene);
    assert_eq!(enc.stream, again.stream);
}

#[test]
fn qpel_four_mv_divergent_motion_self_decodes() {
    let (w, h) = (64usize, 64usize);
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        four_mv: true,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    assert!(vol.quarter_sample, "emitted VOL must carry quarter_sample");
    assert_eq!(vol.video_object_layer_verid, 2);

    let mut store = FrameStore::new();
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut four_mv_total = 0usize;
    for k in 0..4 {
        let (y, cb, cr) = divergent_picture(w, h, k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, 4);
            four_mv_total += stats.inter4v;
            let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }
    assert!(four_mv_total > 0, "qpel+4MV never selected inter4v");
    let decoded = decode_full(&stream);
    assert_frames_match(&decoded, &recons);
}

#[test]
fn ip_determinism_and_method1() {
    let scene = Scene {
        width: 64,
        height: 48,
    };
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        quant_type: true,
        ..EncoderConfig::default()
    };
    let a = encode_ip_stream(&cfg, 5, 4, &scene);
    let b = encode_ip_stream(&cfg, 5, 4, &scene);
    assert_eq!(
        a.stream, b.stream,
        "same input must produce identical bytes"
    );
    let decoded = decode_full(&a.stream);
    assert_frames_match(&decoded, &a.recons);
}

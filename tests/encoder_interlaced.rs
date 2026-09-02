//! Interlaced encoder tools: `interlaced` VOL, per-macroblock field
//! DCT (`dct_type`), §7.7.2.1 field-predicted P macroblocks, the
//! §7.7.2.2 field / interlaced-direct B modes, `top_field_first` and
//! `alternate_vertical_scan_flag` — every emitted stream decodes
//! sample-exact through the crate's own decoder against the encoder's
//! closed-loop reconstructions, and the committed fixtures pin the
//! byte-exact streams alongside their black-box reference decodes
//! (`tests/fixtures/NOTES.md`).

use oxideav_mpeg4video::bvop_interlaced_encode::{
    encode_b_vop_interlaced, BVopInterlacedEncodeStats,
};
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{
    encode_p_vop, reconstruct_own_p_vop_with_anchor_motion, PVopEncodeStats,
};
use oxideav_mpeg4video::svop_encode::{
    encode_s_vop, reconstruct_own_s_vop_with_anchor_motion, SVopEncodeStats,
};
use oxideav_mpeg4video::vol::parse_video_object_layer;
use oxideav_mpeg4video::vop_decode::AnchorMbMotion;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A textured scene whose two fields move independently: the top
/// field (even lines) translates by `top` pels per frame, the bottom
/// field (odd lines) by `bottom` — the interlaced signature no single
/// frame vector fits, so field DCT and field motion prediction win
/// wherever the two velocities differ. A stationary logo block in the
/// top-left corner is frame-coherent (both fields identical), where
/// frame modes stay competitive.
fn picture(w: usize, h: usize, frame_index: usize, top: (i64, i64), bottom: (i64, i64)) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let scene = |x: i64, y: i64| -> u8 {
        let v = (x * 5 + y * 3).rem_euclid(120) + ((x.div_euclid(7) + y.div_euclid(5)) % 11) * 9;
        (30 + v.rem_euclid(190)) as u8
    };
    let n = frame_index as i64;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        let shift = if row % 2 == 0 { top } else { bottom };
        let (ox, oy) = (n * shift.0, n * shift.1);
        for col in 0..w {
            y[row * w + col] = scene(col as i64 + ox, (row as i64) + oy);
        }
    }
    for row in 0..16.min(h) {
        for col in 0..16.min(w) {
            y[row * w + col] = 200 + ((row / 4 + col / 4) % 2) as u8 * 20;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        let shift = if row % 2 == 0 { top } else { bottom };
        for col in 0..cw {
            cb[row * cw + col] = scene(col as i64 + n * shift.0 / 2, row as i64) / 2 + 60;
            cr[row * cw + col] = 128 + ((col as i64 + n) % 7) as u8 * 3;
        }
    }
    (y, cb, cr)
}

/// [`picture`] with the bottom-field velocity applied only to the
/// right half of the picture: the left half is frame-coherent (a
/// global pan the GMC trajectory captures), the right half has the
/// interlaced signature (field-predicted local macroblocks win).
fn picture_split(
    w: usize,
    h: usize,
    frame_index: usize,
    top: (i64, i64),
    bottom: (i64, i64),
) -> Planes {
    let (mut y, cb, cr) = picture(w, h, frame_index, top, top);
    let (y_right, _, _) = picture(w, h, frame_index, top, bottom);
    for row in 0..h {
        for col in w / 2..w {
            y[row * w + col] = y_right[row * w + col];
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

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// With `OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1` the stream is written to
/// `tests/fixtures/` (the black-box reference decode must then be
/// regenerated per `tests/fixtures/NOTES.md`); otherwise it must match
/// the committed bytes.
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

/// Compare decoded frames against a raw yuv420p reference decode.
/// Returns `(differing samples, max abs diff, total)`.
fn diff_against_yuv(
    frames: &[DecodedFrame],
    yuv: &[u8],
    w: usize,
    h: usize,
) -> (usize, u32, usize) {
    let frame_len = w * h + 2 * (w / 2) * (h / 2);
    assert_eq!(yuv.len(), frames.len() * frame_len, "reference frame count");
    let mut differing = 0usize;
    let mut max = 0u32;
    let mut total = 0usize;
    for (k, f) in frames.iter().enumerate() {
        let r = &yuv[k * frame_len..(k + 1) * frame_len];
        let (ry, ru, rv) = (
            &r[..w * h],
            &r[w * h..w * h + (w / 2) * (h / 2)],
            &r[w * h + (w / 2) * (h / 2)..],
        );
        for (ours, theirs) in [
            (f.luma_samples(), ry),
            (f.cb_samples(), ru),
            (f.cr_samples(), rv),
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

/// Encode I + P… with the direct API; returns the stream, the
/// closed-loop recons and the P stats.
fn encode_ip(
    cfg: &EncoderConfig,
    frames: usize,
    motion: ((i64, i64), (i64, i64)),
    qp: u32,
) -> (Vec<u8>, Vec<DecodedFrame>, Vec<PVopEncodeStats>) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let (headers, vol) = vol_of(cfg);
    assert!(vol.interlaced);
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut stats = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..frames {
        let (y, cb, cr) = picture(w, h, k, motion.0, motion.1);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, cfg, &view, 0, 0, qp);
            stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            recons.push(recon);
        } else {
            let reference = store.backward().unwrap().clone();
            let (unit, st) = encode_p_vop(&vol, cfg, &view, &reference, 0, k as u16, qp);
            let (recon, _) = reconstruct_own_p_vop_with_anchor_motion(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
            stats.push(st);
        }
    }
    (stream, recons, stats)
}

/// I + 3 P over field-rate motion: field prediction and field DCT
/// both fire, the VOP headers carry the interlace flags, and the
/// decode is sample-exact.
#[test]
fn interlaced_ip_field_tools_fire_and_round_trip() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        interlaced: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_ip(&cfg, 4, ((2, 0), (0, 0)), 4);
    let field: usize = stats.iter().map(|s| s.field).sum();
    let field_dct: usize = stats.iter().map(|s| s.field_dct).sum();
    assert!(field > 0, "field prediction never chosen: {stats:?}");
    assert!(field_dct > 0, "field DCT never chosen: {stats:?}");
    let frames = decode_all(&stream);
    assert_exact(&frames, &recons);
    assert_eq!(cfg.profile_and_level(), 0xF3, "interlaced is an ASP tool");
}

/// The same content with quarter-sample motion: field-qpel through the
/// §7.6.2.2 per-8×8 field cascade, still sample-exact.
#[test]
fn interlaced_qpel_ip_round_trips() {
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        interlaced: true,
        quarter_sample: true,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_ip(&cfg, 4, ((1, 0), (2, 2)), 5);
    assert!(
        stats.iter().map(|s| s.field).sum::<usize>() > 0,
        "{stats:?}"
    );
    assert_exact(&decode_all(&stream), &recons);
}

/// `alternate_vertical_scan_flag` + `top_field_first == 0`: every
/// block takes the alternate-vertical scan on both the intra and the
/// inter paths; the flags survive the VOP-header round trip.
#[test]
fn interlaced_alternate_scan_bottom_field_first_round_trips() {
    let cfg = EncoderConfig {
        width: 48,
        height: 48,
        interlaced: true,
        alternate_scan: true,
        top_field_first: false,
        four_mv: true,
        adaptive_quant: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, _stats) = encode_ip(&cfg, 3, ((0, 2), (1, 0)), 6);
    let frames = decode_all(&stream);
    assert_exact(&frames, &recons);
    // Parse the first VOP header back: interlace flags as configured.
    let (_, vol) = vol_of(&cfg);
    let pos = stream
        .windows(4)
        .position(|w| w == [0, 0, 1, 0xB6])
        .unwrap();
    let vop = oxideav_mpeg4video::vop::parse_video_object_plane_header(
        &stream[pos..],
        vol.time_increment_resolution,
        oxideav_mpeg4video::vop::VopContext::from_vol(&vol),
    )
    .unwrap();
    assert!(!vop.top_field_first);
    assert!(vop.alternate_vertical_scan);
}

/// Encode I P B B P B B with the interlaced B encoder; returns the
/// stream, the closed-loop recons in display order and the B stats.
fn encode_ipbb(
    cfg: &EncoderConfig,
    motion: ((i64, i64), (i64, i64)),
    qp: u32,
    ecosystem_compat: bool,
) -> (Vec<u8>, Vec<DecodedFrame>, Vec<BVopInterlacedEncodeStats>) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let (headers, vol) = vol_of(cfg);
    let mut stream = headers;
    let mut recons: Vec<Option<DecodedFrame>> = vec![None; 7];
    let mut b_stats = Vec::new();
    let mut store = FrameStore::new();
    let view_of = |k: usize| picture(w, h, k, motion.0, motion.1);
    // Display order 0..7, coding order: I0 P3 B1 B2 P6 B4 B5.
    let (y, cb, cr) = view_of(0);
    let (unit, recon) = encode_i_vop(
        &vol,
        cfg,
        &FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        },
        0,
        0,
        qp,
    );
    stream.extend_from_slice(&unit);
    store.push_anchor(recon.clone());
    recons[0] = Some(recon);
    for (anchor, bs) in [(3usize, [1usize, 2usize]), (6, [4, 5])] {
        let (y, cb, cr) = view_of(anchor);
        let reference = store.backward().unwrap().clone();
        let (unit, _) = encode_p_vop(
            &vol,
            cfg,
            &FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: w,
                height: h,
            },
            &reference,
            0,
            anchor as u16,
            qp,
        );
        let (recon, motion): (DecodedFrame, Vec<AnchorMbMotion>) =
            reconstruct_own_p_vop_with_anchor_motion(&vol, &unit, &mut store);
        stream.extend_from_slice(&unit);
        recons[anchor] = Some(recon);
        for b in bs {
            let (y, cb, cr) = view_of(b);
            let (unit, recon, st) = encode_b_vop_interlaced(
                &vol,
                cfg,
                &FrameView {
                    y: &y,
                    cb: &cb,
                    cr: &cr,
                    width: w,
                    height: h,
                },
                &store,
                Some(&motion),
                (b - (anchor - 3)) as i32,
                3,
                0,
                b as u16,
                qp,
                ecosystem_compat,
            );
            stream.extend_from_slice(&unit);
            recons[b] = Some(recon);
            b_stats.push(st);
        }
    }
    let recons: Vec<DecodedFrame> = recons.into_iter().map(|r| r.unwrap()).collect();
    (stream, recons, b_stats)
}

/// Interlaced B-VOPs (spec-literal tool set): the field modes and the
/// §7.7.2.2 interlaced-direct mode both fire, and the stream decodes
/// sample-exact through the interlaced B walk.
#[test]
fn interlaced_ipbb_spec_modes_fire_and_round_trip() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        interlaced: true,
        b_vops: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_ipbb(&cfg, ((2, 0), (0, 0)), 4, false);
    let field: usize = stats
        .iter()
        .map(|s| s.field_forward + s.field_backward + s.field_bidirectional)
        .sum();
    let idirect: usize = stats.iter().map(|s| s.interlaced_direct).sum();
    assert!(field > 0, "no field B mode chosen: {stats:?}");
    assert!(idirect > 0, "no interlaced-direct macroblock: {stats:?}");
    assert!(stats.iter().all(|s| s.compat_direct_suppressed == 0));
    assert_exact(&decode_all(&stream), &recons);
}

/// The ecosystem-compat emission never codes direct mode over a
/// field-predicted co-located macroblock; the stream is still
/// sample-exact through our (spec) decoder — and, being free of the
/// §7.7.2.2 divergence, identical under the ecosystem-compat decode.
#[test]
fn interlaced_ipbb_compat_emission_avoids_interlaced_direct() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        interlaced: true,
        b_vops: true,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_ipbb(&cfg, ((1, 2), (2, 0)), 5, true);
    assert!(stats.iter().all(|s| s.interlaced_direct == 0), "{stats:?}");
    assert!(
        stats
            .iter()
            .map(|s| s.compat_direct_suppressed)
            .sum::<usize>()
            > 0,
        "{stats:?}"
    );
    assert_exact(&decode_all(&stream), &recons);
    let mut dec =
        Mpeg4VideoDecoder::with_options(oxideav_mpeg4video::compat::DecodeOptions::ecosystem());
    let mut compat = dec.decode(&stream).unwrap();
    compat.extend(dec.flush());
    assert_exact(&compat, &recons);
}

/// Registry path: `interlaced` (+ `alt-scan`, `bf`, `mb-aq`,
/// `packet-bits`) produces a decodable byte-deterministic stream with
/// an interlaced VOL; the incompatible combinations are rejected.
#[test]
fn registry_interlaced_options() {
    use oxideav_core::Encoder as _;
    let (w, h) = (64usize, 48usize);
    let build = || -> Vec<u8> {
        let mut params =
            oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
        params.width = Some(w as u32);
        params.height = Some(h as u32);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        params.options = oxideav_core::CodecOptions::default()
            .set("interlaced", "true")
            .set("alt-scan", "true")
            .set("bf", "2")
            .set("mb-aq", "true")
            .set("packet-bits", "700");
        let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
        for k in 0..6usize {
            let (y, cb, cr) = picture(w, h, k, (2, 0), (0, 0));
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
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).unwrap();
    frames.extend(dec.flush());
    assert_eq!(frames.len(), 6);
    assert!(dec.vol().unwrap().interlaced);

    // interlaced + data-partitioned is rejected (the decoder has no
    // interlaced data-partitioned walk).
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(32);
    params.height = Some(32);
    params.options = oxideav_core::CodecOptions::default()
        .set("interlaced", "true")
        .set("data-partitioned", "true");
    assert!(
        oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).is_err(),
        "interlaced + data-partitioned must be rejected"
    );
}

/// Black-box pin: the interlaced I+P stream is byte-deterministic and
/// the reference decoder reproduces our closed-loop reconstruction
/// bit-exactly (commands + SHA-256 in `tests/fixtures/NOTES.md`).
#[test]
fn blackbox_interlaced_ip_stream_is_bit_exact() {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        interlaced: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, _) = encode_ip(&cfg, 4, ((2, 0), (0, 0)), 4);
    pin_fixture("enc_ilaced_ip_64x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_ilaced_ip_64x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&recons, &yuv, 64, 64);
    assert_eq!(
        (differing, max),
        (0, 0),
        "interlaced I+P: {differing}/{total} samples differ (max {max})"
    );
}

/// Black-box pin: the ecosystem-compat interlaced I/P/B stream (field
/// B modes, no interlaced-direct) — the reference decode matches our
/// closed loop bit-exactly.
#[test]
fn blackbox_interlaced_ipbb_compat_stream_is_bit_exact() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        interlaced: true,
        b_vops: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, _) = encode_ipbb(&cfg, ((2, 0), (0, 0)), 4, true);
    pin_fixture("enc_ilaced_ipbb_compat_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_ilaced_ipbb_compat_96x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&recons, &yuv, 96, 64);
    assert_eq!(
        (differing, max),
        (0, 0),
        "interlaced compat I/P/B: {differing}/{total} samples differ (max {max})"
    );
}

/// Black-box pin of the spec-literal interlaced I/P/B stream (with
/// §7.7.2.2 interlaced-direct macroblocks, quarter-sample). The
/// reference decoder reads that one derivation differently
/// (`crate::compat` divergence 1: co-located field MVs taken as
/// zero), so its decode differs from our spec closed loop **only**
/// inside interlaced-direct macroblocks, and the crate's own
/// ecosystem-compat decode of the same stream reproduces the reference
/// bit-exactly — every other tool in the stream (field DCT, field
/// prediction in P and B, frame modes, progressive direct) is
/// bit-exact under both readings.
#[test]
fn blackbox_interlaced_ipbb_spec_stream_diverges_only_in_interlaced_direct() {
    use oxideav_mpeg4video::compat::DecodeOptions;
    let (w, h) = (96usize, 64usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        interlaced: true,
        b_vops: true,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_ipbb(&cfg, ((1, 2), (2, 0)), 5, false);
    assert!(
        stats.iter().map(|s| s.interlaced_direct).sum::<usize>() > 0,
        "{stats:?}"
    );
    pin_fixture("enc_ilaced_ipbb_spec_qpel_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_ilaced_ipbb_spec_qpel_96x64.yuv")).unwrap();
    // Spec decode == closed loop (already asserted elsewhere); the
    // ecosystem-compat decode is the reference decoder's reading.
    let mut dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
    let mut eco = dec.decode(&stream).unwrap();
    eco.extend(dec.flush());
    let (differing, max, total) = diff_against_yuv(&eco, &yuv, w, h);
    assert_eq!(
        (differing, max),
        (0, 0),
        "ecosystem-compat decode vs reference: {differing}/{total} differ (max {max})"
    );
    // Every macroblock where the spec closed loop departs from the
    // reference is one where the two readings differ (an
    // interlaced-direct macroblock); anchors are bit-exact.
    let frame_len = w * h * 3 / 2;
    let (mbw, mbh) = (w / 16, h / 16);
    let mut divergent_mbs = 0usize;
    for (k, spec) in recons.iter().enumerate() {
        let r = &yuv[k * frame_len..(k + 1) * frame_len];
        for mb in 0..mbw * mbh {
            let (mx, my) = (mb % mbw, mb / mbw);
            let mut spec_vs_ref = false;
            let mut spec_vs_eco = false;
            for y in 0..16 {
                for x in 0..16 {
                    let i = (my * 16 + y) * w + mx * 16 + x;
                    spec_vs_ref |= spec.luma_samples()[i] != r[i];
                    spec_vs_eco |= spec.luma_samples()[i] != eco[k].luma_samples()[i];
                }
            }
            for y in 0..8 {
                for x in 0..8 {
                    let i = (my * 8 + y) * (w / 2) + mx * 8 + x;
                    spec_vs_ref |= spec.cb_samples()[i] != r[w * h + i]
                        || spec.cr_samples()[i] != r[w * h + (w / 2) * (h / 2) + i];
                    spec_vs_eco |= spec.cb_samples()[i] != eco[k].cb_samples()[i]
                        || spec.cr_samples()[i] != eco[k].cr_samples()[i];
                }
            }
            assert!(
                !spec_vs_ref || spec_vs_eco,
                "frame {k} MB {mb}: reference differs from the spec decode outside the \
                 interlaced-direct divergence"
            );
            if k % 3 == 0 {
                assert!(!spec_vs_ref, "anchor frame {k} MB {mb} must be bit-exact");
            }
            divergent_mbs += usize::from(spec_vs_ref);
        }
    }
    assert!(
        divergent_mbs > 0,
        "the stream carries divergent direct macroblocks"
    );
}

/// Encode I S B B S with an interlaced GMC VOL: S(GMC) anchors whose
/// local macroblocks may be §7.7.2.1 field-predicted, GMC macroblocks
/// frame-predicted per §7.8.7.2 (field DCT on any residual), and
/// interlaced B-VOPs over the S anchors' field motion.
fn encode_isbb(
    cfg: &EncoderConfig,
    motion: ((i64, i64), (i64, i64)),
    qp: u32,
    ecosystem_compat: bool,
) -> (
    Vec<u8>,
    Vec<DecodedFrame>,
    Vec<SVopEncodeStats>,
    Vec<BVopInterlacedEncodeStats>,
) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let (headers, vol) = vol_of(cfg);
    assert!(
        vol.interlaced
            && matches!(
                vol.sprite_enable,
                oxideav_mpeg4video::vol::SpriteEnable::Gmc
            )
    );
    let mut stream = headers;
    let mut recons: Vec<Option<DecodedFrame>> = vec![None; 7];
    let mut s_stats = Vec::new();
    let mut b_stats = Vec::new();
    let mut store = FrameStore::new();
    let view_of = |k: usize| picture_split(w, h, k, motion.0, motion.1);
    let (y, cb, cr) = view_of(0);
    let (unit, recon) = encode_i_vop(
        &vol,
        cfg,
        &FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        },
        0,
        0,
        qp,
    );
    stream.extend_from_slice(&unit);
    store.push_anchor(recon.clone());
    recons[0] = Some(recon);
    for (anchor, bs) in [(3usize, [1usize, 2usize]), (6, [4, 5])] {
        let (y, cb, cr) = view_of(anchor);
        let reference = store.backward().unwrap().clone();
        let (unit, st) = encode_s_vop(
            &vol,
            cfg,
            &FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: w,
                height: h,
            },
            &reference,
            0,
            anchor as u16,
            qp,
        );
        let (recon, anchor_motion): (DecodedFrame, Vec<AnchorMbMotion>) =
            reconstruct_own_s_vop_with_anchor_motion(&vol, &unit, &mut store);
        stream.extend_from_slice(&unit);
        recons[anchor] = Some(recon);
        s_stats.push(st);
        for b in bs {
            let (y, cb, cr) = view_of(b);
            let (unit, recon, st) = encode_b_vop_interlaced(
                &vol,
                cfg,
                &FrameView {
                    y: &y,
                    cb: &cb,
                    cr: &cr,
                    width: w,
                    height: h,
                },
                &store,
                Some(&anchor_motion),
                (b - (anchor - 3)) as i32,
                3,
                0,
                b as u16,
                qp,
                ecosystem_compat,
            );
            stream.extend_from_slice(&unit);
            recons[b] = Some(recon);
            b_stats.push(st);
        }
    }
    let recons: Vec<DecodedFrame> = recons.into_iter().map(|r| r.unwrap()).collect();
    (stream, recons, s_stats, b_stats)
}

/// Interlaced S(GMC)-VOPs: the pan the trajectory carries predicts the
/// field that moves with it (GMC macroblocks), the other field's
/// macroblocks go field-predicted local (§7.7.2.1 over the GMC
/// neighbours' averaged-MV candidates), field DCT fires, and the
/// interlaced B-VOPs over the S anchors decode sample-exact.
#[test]
fn interlaced_gmc_isbb_round_trips() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        interlaced: true,
        gmc: true,
        b_vops: true,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, s_stats, b_stats) = encode_isbb(&cfg, ((2, 0), (2, 2)), 5, false);
    let gmc: usize = s_stats.iter().map(|s| s.gmc + s.gmc_skipped).sum();
    let field: usize = s_stats.iter().map(|s| s.field).sum();
    let field_dct: usize = s_stats.iter().map(|s| s.field_dct).sum();
    assert!(gmc > 0, "no GMC macroblock: {s_stats:?}");
    assert!(
        field > 0,
        "no field-predicted local macroblock: {s_stats:?}"
    );
    assert!(field_dct > 0, "no field DCT: {s_stats:?}");
    assert!(
        b_stats
            .iter()
            .map(|s| s.field_forward + s.field_backward + s.field_bidirectional)
            .sum::<usize>()
            > 0,
        "{b_stats:?}"
    );
    assert_exact(&decode_all(&stream), &recons);
}

/// Black-box pin: interlaced GMC I/S/B/B/S/B/B through the registry
/// (`interlaced` + `gmc` + `bf 2` + `ecosystem-compat`), reference
/// decode bit-exact.
#[test]
fn blackbox_interlaced_gmc_isbb_compat_stream_is_bit_exact() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default()
        .set("interlaced", "true")
        .set("gmc", "true")
        .set("fcode", "2")
        .set("bf", "2")
        .set("ecosystem-compat", "true")
        .set("qp", "5");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..7usize {
        let (y, cb, cr) = picture_split(w, h, k, (2, 0), (2, 2));
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
    // The stream carries S-VOPs (coding type 0b11 after each VOP start
    // code).
    let s_vops = stream
        .windows(5)
        .filter(|win| win[..4] == [0, 0, 1, 0xB6] && win[4] >> 6 == 0b11)
        .count();
    assert!(s_vops >= 2, "expected S-VOPs in the stream");
    pin_fixture("enc_isb_ilaced_gmc_compat_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let frames = decode_all(&stream);
    let yuv = std::fs::read(fixture_path("enc_isb_ilaced_gmc_compat_96x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&frames, &yuv, w, h);
    assert_eq!(
        (differing, max),
        (0, 0),
        "interlaced GMC I/S/B: {differing}/{total} samples differ (max {max})"
    );
}

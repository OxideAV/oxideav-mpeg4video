//! GMC with two and three warping points: the encoder fits the §7.8.5
//! similarity / affine model to its motion field, emits the
//! multi-point `sprite_trajectory()`, and the S(GMC)-VOPs decode
//! sample-exact through the crate's own warp (the decoder as oracle),
//! plus the black-box pins of `tests/fixtures/NOTES.md`.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::svop_encode::{
    encode_s_vop, reconstruct_own_s_vop_with_motion, SVopEncodeStats,
};
use oxideav_mpeg4video::vol::parse_video_object_layer;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// The global motion of a synthetic scene: integer fixed-point so the
/// pictures are byte-deterministic on every platform.
#[derive(Clone, Copy)]
enum Motion {
    /// Zoom about the origin by `(64 + n·zoom) / 64` plus a pan.
    Zoom { zoom: i64, pan: (i64, i64) },
    /// Rotation by `n / 32` radians (small-angle, fixed-point) plus a pan.
    Rotate { pan: (i64, i64) },
}

/// Two incommensurate diagonal ramps plus a coarse tile pattern: no
/// small shift maps the texture onto itself, so block matching finds
/// the true displacement rather than a period-aligned alias.
fn scene(x: i64, y: i64) -> u8 {
    let v = (x * 7 + y * 5).rem_euclid(160)
        + (x * 3 - y * 11).rem_euclid(97)
        + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
    (40 + v.rem_euclid(170)) as u8
}

/// Sample the scene at a fractional source position given in 1/64
/// pel (bilinear, integer arithmetic) — so a fractional global
/// displacement produces the blended samples a warp predicts rather
/// than a nearest-neighbour staircase.
fn sample64(sx: i64, sy: i64) -> u8 {
    let (ix, iy) = (sx.div_euclid(64), sy.div_euclid(64));
    let (fx, fy) = (sx.rem_euclid(64), sy.rem_euclid(64));
    let s = |dx: i64, dy: i64| -> i64 { i64::from(scene(ix + dx, iy + dy)) };
    let top = s(0, 0) * (64 - fx) + s(1, 0) * fx;
    let bot = s(0, 1) * (64 - fx) + s(1, 1) * fx;
    ((top * (64 - fy) + bot * fy + 2048) / 4096) as u8
}

fn picture(w: usize, h: usize, frame_index: usize, motion: Motion) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let n = frame_index as i64;
    // Source position of picture point (x, y) in 1/64 pel.
    let map = |x: i64, y: i64| -> (i64, i64) {
        match motion {
            Motion::Zoom { zoom, pan } => (
                x * (64 + n * zoom) + n * pan.0 * 64,
                y * (64 + n * zoom) + n * pan.1 * 64,
            ),
            Motion::Rotate { pan } => (
                x * 64 - n * y * 2 + n * pan.0 * 64,
                y * 64 + n * x * 2 + n * pan.1 * 64,
            ),
        }
    };
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let (sx, sy) = map(col as i64, row as i64);
            y[row * w + col] = sample64(sx, sy);
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            let (sx, sy) = map(col as i64 * 2, row as i64 * 2);
            cb[row * cw + col] = sample64(sx / 2, sy / 2) / 2 + 64;
            cr[row * cw + col] = 128 + (sample64(sy / 2, sx / 2) % 40);
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
        let c = (w / 2) * (h / 2);
        for (ours, theirs) in [
            (f.luma_samples(), &r[..w * h]),
            (f.cb_samples(), &r[w * h..w * h + c]),
            (f.cr_samples(), &r[w * h + c..]),
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

/// I + `frames - 1` S(GMC) pictures; returns the stream, recons and
/// the S stats.
fn encode_is(
    cfg: &EncoderConfig,
    frames: usize,
    motion: Motion,
    qp: u32,
) -> (Vec<u8>, Vec<DecodedFrame>, Vec<SVopEncodeStats>) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let (headers, vol) = vol_of(cfg);
    assert_eq!(vol.no_of_sprite_warping_points, Some(cfg.gmc_points));
    let mut stream = headers;
    let mut recons = Vec::new();
    let mut stats = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..frames {
        let (y, cb, cr) = picture(w, h, k, motion);
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
            let (unit, st) = encode_s_vop(&vol, cfg, &view, &reference, 0, k as u16, qp);
            let (recon, _) = reconstruct_own_s_vop_with_motion(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
            stats.push(st);
        }
    }
    (stream, recons, stats)
}

/// Zoom + pan through a three-point (affine) trajectory: the fitted
/// `du[1]` / `dv[2]` carry the scale, GMC dominates the mode decision,
/// and the decode is sample-exact.
#[test]
fn affine_zoom_lands_in_a_three_point_trajectory() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        gmc: true,
        gmc_points: 3,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_is(
        &cfg,
        4,
        Motion::Zoom {
            zoom: 1,
            pan: (2, 1),
        },
        5,
    );
    for (k, st) in stats.iter().enumerate() {
        assert_eq!(st.points.count, 3);
        // Zoom by 1/64 per frame over W = 96 pels ≈ 1.5 pels = 3
        // half-samples of extra displacement at (W, 0); over H = 64
        // ≈ 1 pel = 2 half-samples at (0, H).
        assert!((2..=4).contains(&st.points.points[1][0]), "{st:?}");
        assert!((1..=3).contains(&st.points.points[2][1]), "{st:?}");
        // On the pristine reference the warp beats every local vector;
        // on the reconstructed references the local search can exploit
        // quantisation noise, but GMC stays a substantial share.
        let gmc = st.gmc + st.gmc_skipped;
        if k == 0 {
            assert!(
                gmc > st.local + st.intra,
                "S-VOP {k}: GMC must dominate: {st:?}"
            );
        } else {
            assert!(gmc * 4 >= (st.local + st.intra + gmc), "S-VOP {k}: {st:?}");
        }
    }
    assert_exact(&decode_all(&stream), &recons);
}

/// Rotation + pan through a two-point (similarity) trajectory.
#[test]
fn similarity_rotation_lands_in_a_two_point_trajectory() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        gmc: true,
        gmc_points: 2,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_is(&cfg, 4, Motion::Rotate { pan: (1, 0) }, 4);
    for (k, st) in stats.iter().enumerate() {
        assert_eq!(st.points.count, 2);
        // The rotation by 1/32 rad moves (W, 0) by ≈ W/32 = 3 pels
        // downwards: dv[1] ≈ 6 half-samples.
        assert!((4..=8).contains(&st.points.points[1][1]), "{st:?}");
        let gmc = st.gmc + st.gmc_skipped;
        if k == 0 {
            assert!(
                gmc > st.local + st.intra,
                "S-VOP {k}: GMC must dominate: {st:?}"
            );
        } else {
            assert!(gmc * 4 >= (st.local + st.intra + gmc), "S-VOP {k}: {st:?}");
        }
    }
    assert_exact(&decode_all(&stream), &recons);
}

/// A pure pan still fits the multi-point models (zero warp terms).
#[test]
fn pure_pan_fits_the_affine_model_as_translation() {
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        gmc: true,
        gmc_points: 3,
        ..EncoderConfig::default()
    };
    let (stream, recons, stats) = encode_is(
        &cfg,
        3,
        Motion::Zoom {
            zoom: 0,
            pan: (3, 2),
        },
        4,
    );
    for st in &stats {
        assert_eq!(st.points.points[0], [6, 4], "{st:?}");
        assert_eq!(st.points.points[1], [0, 0], "{st:?}");
        assert_eq!(st.points.points[2], [0, 0], "{st:?}");
    }
    assert_exact(&decode_all(&stream), &recons);
}

/// Registry: `gmc-points` is validated and threads through to the VOL.
#[test]
fn registry_gmc_points_option() {
    let mut p = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    p.width = Some(64);
    p.height = Some(48);
    p.options = oxideav_core::CodecOptions::default()
        .set("gmc", "true")
        .set("gmc-points", "3");
    let enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&p).unwrap();
    use oxideav_core::Encoder as _;
    let extradata = &enc.output_params().extradata;
    let pos = extradata
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&extradata[pos..], 0xF3).unwrap();
    assert_eq!(vol.no_of_sprite_warping_points, Some(3));
    p.options = oxideav_core::CodecOptions::default()
        .set("gmc", "true")
        .set("gmc-points", "4");
    assert!(oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&p).is_err());
}

/// Black-box pin: the three-point zoom stream. Every trajectory
/// component is positive and the warp keeps every §7.8.7.3 averaged
/// MV positive too, so the reference decode is expected bit-exact
/// against the spec closed loop.
#[test]
fn blackbox_three_point_gmc_stream() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        gmc: true,
        gmc_points: 3,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, _) = encode_is(
        &cfg,
        4,
        Motion::Zoom {
            zoom: 1,
            pan: (2, 1),
        },
        5,
    );
    pin_fixture("enc_is_gmc3_zoom_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_is_gmc3_zoom_96x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&recons, &yuv, 96, 64);
    assert_eq!(
        (differing, max),
        (0, 0),
        "three-point GMC: {differing}/{total} samples differ (max {max})"
    );
}

/// Black-box pin: the two-point rotation stream — exact up to one
/// near-tie sample of the intra picture.
#[test]
fn blackbox_two_point_gmc_stream() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        gmc: true,
        gmc_points: 2,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let (stream, recons, _) = encode_is(&cfg, 4, Motion::Rotate { pan: (1, 0) }, 4);
    pin_fixture("enc_is_gmc2_rot_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_is_gmc2_rot_96x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&recons, &yuv, 96, 64);
    // One near-tie IDCT sample of the I picture (the reference
    // decoder's single-precision transform crossing a rounding
    // boundary, `tests/fixtures/NOTES.md`) and its GMC-propagated
    // copy in the first S picture; the two-point warp itself is exact
    // (the later S pictures are bit-exact).
    assert!(
        differing <= 2 && max <= 1,
        "two-point GMC: {differing}/{total} samples differ (max {max})"
    );
}

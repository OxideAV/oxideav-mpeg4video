//! Black-box cross-check of the I-VOP encoder: the committed streams
//! under `tests/fixtures/` were produced by THIS encoder from the
//! deterministic synthetic source below, then decoded by the opaque
//! reference binary with the floating-point IDCT selected (commands +
//! SHA-256 in `tests/fixtures/NOTES.md`). The tests assert
//!
//! 1. **byte determinism** — re-encoding the same synthetic input
//!    reproduces the committed stream byte-for-byte (any encoder
//!    change that alters the bitstream must regenerate the fixture
//!    and re-run the black-box decode), and
//! 2. **decode agreement** — the crate's own decoder reproduces the
//!    reference decode of our stream **bit-exactly**, i.e. an
//!    independent decoder accepts and identically interprets what we
//!    emit.
//!
//! No external implementation source was consulted — the reference
//! binary was invoked as an opaque validator only.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::FrameStore;
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop};
use oxideav_mpeg4video::vol::parse_video_object_layer;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Fixture (re)generation hook: with `OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1`
/// the reproduction tests write the freshly built stream to
/// `tests/fixtures/` instead of comparing (the black-box reference
/// decode must then be regenerated per `tests/fixtures/NOTES.md`).
/// Returns `true` when the fixture was (re)written.
fn maybe_write_fixture(name: &str, bytes: &[u8]) -> bool {
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_none() {
        return false;
    }
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
    true
}

/// Deterministic LCG (numerical recipes constants).
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// The deterministic 64×64 4:2:0 source shared with the fixture
/// generation run — gradients, an LCG texture field, and a
/// frame-indexed bright square.
fn synthesise(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (width, height) = (64usize, 64usize);
    let (cw, ch) = (width / 2, height / 2);
    let mut y = vec![0u8; width * height];
    let mut state = 0x1234_5678u32 ^ (frame_index as u32).wrapping_mul(0x9E37_79B9);
    for row in 0..height {
        for col in 0..width {
            let grad = (row * 3 + col * 2) as i32 % 200 + 20;
            let noise = (lcg(&mut state) >> 28) as i32;
            let mut v = grad + noise;
            let bx = (frame_index * 7) % width;
            if col >= bx && col < (bx + 12).min(width) && (8..20).contains(&row) {
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
    (y, cb, cr)
}

/// Build the fixture stream: 3 I-VOPs, 64×64, the given quantisation
/// method, qp 4, resolution 25, cost-decided AC prediction.
fn build_stream(quant_type: bool) -> Vec<u8> {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        quant_type,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let mut stream = headers;
    for k in 0..3usize {
        let (y, cb, cr) = synthesise(k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 64,
            height: 64,
        };
        let (unit, _recon) = encode_i_vop(&vol, &cfg, &view, 0, k as u16, 4);
        stream.extend_from_slice(&unit);
    }
    stream
}

/// Translating-scene picture for the I+P fixture: a fixed background
/// sampled at a per-frame (2, 1)-pel offset plus a static corner
/// block, 64×64.
fn ip_picture(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
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
    for row in 0..8 {
        for col in 0..8 {
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

/// Build the I+P fixture stream: 1 I-VOP + 5 P-VOPs, 64×64, method-2
/// quantisation, qp 4.
fn build_ip_stream() -> Vec<u8> {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let mut stream = headers;
    let mut store = FrameStore::new();
    for k in 0..6usize {
        let (y, cb, cr) = ip_picture(k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 64,
            height: 64,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 4);
            store.push_anchor(recon);
            stream.extend_from_slice(&unit);
        } else {
            let reference = store.backward().expect("anchor present").clone();
            let (unit, _stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, 4);
            let _ = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
        }
    }
    stream
}

/// A checkerboard of divergent 8×8-block motion fields (the 4MV
/// exercise scene): even-parity blocks translate by (+2, +1) pels per
/// frame, odd-parity blocks by (-2, 0).
fn divergent_picture(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);
    let bg = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5) % 160 + ((x / 9 + y / 7) % 13) * 6;
        (40 + v.rem_euclid(170)) as u8
    };
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let parity = (row / 8 + col / 8) % 2;
            let (ox, oy) = if parity == 0 {
                ((frame_index * 2) as i64, frame_index as i64)
            } else {
                (-((frame_index * 2) as i64), 0)
            };
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    let cb = vec![100u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    (y, cb, cr)
}

/// Sub-pel translating scene for the quarter-sample fixture: a smooth
/// integer-arithmetic texture defined on the quarter-pel grid, sampled
/// at a per-frame offset of (3, 1) **quarter** pels — real fractional
/// motion for the §7.6.2.2 path to chase.
fn qpel_picture(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);
    // Quarter-grid texture: triangle waves in two directions, smooth
    // at the quarter step (pure integer arithmetic — deterministic).
    let bg_q = |xq: i64, yq: i64| -> u8 {
        let t1 = (xq * 3 + yq * 2).rem_euclid(512);
        let t2 = (xq - yq * 4).rem_euclid(768);
        let tri1 = (t1 - 256).abs(); // 0..=256
        let tri2 = (t2 - 384).abs() / 2; // 0..=192
        (32 + (tri1 * 2 + tri2) / 5) as u8
    };
    let (oxq, oyq) = ((frame_index * 3) as i64, frame_index as i64);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg_q(col as i64 * 4 + oxq, row as i64 * 4 + oyq);
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg_q(col as i64 * 8 + oxq, row as i64 * 8 + oyq) / 2 + 64;
        }
    }
    (y, cb, cr)
}

/// One planar 4:2:0 synthetic picture (`(y, cb, cr)` planes).
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A 96×48 picture spanning every `crate::mb_quant` activity class —
/// a flat band, a smooth gradient, a textured band and a full-range
/// noise band — translating by (2, 1) pels per frame.
fn mixed_activity_picture(frame_index: usize) -> Planes {
    let (w, h) = (96usize, 48usize);
    let (cw, ch) = (w / 2, h / 2);
    let band = w / 4;
    let mut y = vec![0u8; w * h];
    let (ox, oy) = ((frame_index * 2) as i64, frame_index as i64);
    for row in 0..h {
        for col in 0..w {
            let (x, yy) = (col as i64 + ox, row as i64 + oy);
            let bx = (x.rem_euclid(w as i64) as usize) / band;
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
    (y, vec![110u8; cw * ch], vec![140u8; cw * ch])
}

/// A 96×64 textured background translating by (20, 5) pels per frame
/// — a displacement outside the `fcode == 1` Table 7-9 range, so the
/// `fcode > 1` `r_size`-bit residual form of `motion_vector()` is
/// exercised on every P-VOP.
fn long_motion_picture(frame_index: usize) -> Planes {
    let (w, h) = (96usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);
    let bg = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5).rem_euclid(160) + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
        (40 + v.rem_euclid(170)) as u8
    };
    let (ox, oy) = (frame_index as i64 * 20, frame_index as i64 * 5);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + 64;
        }
    }
    (y, cb, cr)
}

/// Build an I+4P stream over `picture` with the given tool set.
fn build_tooled_ip_stream(cfg: &EncoderConfig, picture: fn(usize) -> Planes) -> Vec<u8> {
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let mut stream = headers;
    let mut store = FrameStore::new();
    for k in 0..5usize {
        let (y, cb, cr) = picture(k);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 64,
            height: 64,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, cfg, &view, 0, 0, 4);
            store.push_anchor(recon);
            stream.extend_from_slice(&unit);
        } else {
            let reference = store.backward().expect("anchor present").clone();
            let (unit, _stats) = encode_p_vop(&vol, cfg, &view, &reference, 0, k as u16, 4);
            let _ = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
        }
    }
    stream
}

fn build_4mv_stream() -> Vec<u8> {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        four_mv: true,
        ..EncoderConfig::default()
    };
    build_tooled_ip_stream(&cfg, divergent_picture)
}

fn build_qpel_stream() -> Vec<u8> {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    build_tooled_ip_stream(&cfg, qpel_picture)
}

fn build_qpel_4mv_stream() -> Vec<u8> {
    let cfg = EncoderConfig {
        width: 64,
        height: 64,
        four_mv: true,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    build_tooled_ip_stream(&cfg, divergent_picture)
}

/// Build the I/P/B fixture stream through the registry encoder
/// (`bf == 2`, so the coded order is I0 P3 B1 B2 P5 B4 with a flush
/// tail): 6 frames of the translating scene, method 2, qp 4.
fn build_ipb_stream() -> Vec<u8> {
    build_registry_stream(
        oxideav_core::CodecOptions::default().set("bf", "2"),
        ip_picture,
    )
}

/// The combined-tools sibling of [`build_ipb_stream`]: quarter-sample
/// + inter4v + B-VOPs in one stream over the divergent-motion scene.
fn build_ipb_qpel4mv_stream() -> Vec<u8> {
    build_registry_stream(
        oxideav_core::CodecOptions::default()
            .set("bf", "2")
            .set("qpel", "true")
            .set("four-mv", "true"),
        divergent_picture,
    )
}

fn build_registry_stream(
    options: oxideav_core::CodecOptions,
    picture: fn(usize) -> Planes,
) -> Vec<u8> {
    build_registry_stream_dims(options, picture, 64, 64)
}

/// Per-macroblock adaptive quantisation (`dquant` on I/P, `dbquant`
/// on B) + inter4v + `bf` 2 over the mixed-activity scene.
fn build_aq_ipb_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default()
            .set("mb-aq", "true")
            .set("four-mv", "true")
            .set("bf", "2")
            .set("qp", "10"),
        mixed_activity_picture,
        96,
        48,
    )
}

/// Video packets (~500-bit target, HEC alternating) in a combined-
/// syntax I/P/B stream with `fcode` 2.
fn build_vp_ipb_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default()
            .set("packet-bits", "500")
            .set("fcode", "2")
            .set("bf", "2"),
        long_motion_picture,
        96,
        64,
    )
}

/// Data partitioning (dc_marker / motion_marker) + video packets +
/// per-macroblock dquant, I/P only.
fn build_dp_ip_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default()
            .set("packet-bits", "400")
            .set("data-partitioned", "true")
            .set("mb-aq", "true"),
        mixed_activity_picture,
        96,
        48,
    )
}

/// Data partitioning + reversible VLCs + video packets + dquant /
/// dbquant + inter4v + `fcode` 2 + B-VOPs (combined syntax inside the
/// partitioned VOL).
fn build_dp_rvlc_ipb_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default()
            .set("packet-bits", "400")
            .set("data-partitioned", "true")
            .set("rvlc", "true")
            .set("mb-aq", "true")
            .set("four-mv", "true")
            .set("fcode", "2")
            .set("bf", "2"),
        mixed_activity_picture,
        96,
        48,
    )
}

/// `fcode` 2 half-sample I/P over the 20-pel-per-frame scene.
fn build_fcode2_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default().set("fcode", "2"),
        long_motion_picture,
        96,
        64,
    )
}

/// `fcode` 3 + quarter-sample + inter4v + B-VOPs over the same scene
/// (forward/backward B vectors under the wide range too).
fn build_fcode3_qpel_ipb_stream() -> Vec<u8> {
    build_registry_stream_dims(
        oxideav_core::CodecOptions::default()
            .set("fcode", "3")
            .set("qpel", "true")
            .set("four-mv", "true")
            .set("bf", "2"),
        long_motion_picture,
        96,
        64,
    )
}

fn build_registry_stream_dims(
    options: oxideav_core::CodecOptions,
    picture: fn(usize) -> Planes,
    w: usize,
    h: usize,
) -> Vec<u8> {
    use oxideav_core::Encoder as _;
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = options;
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..6usize {
        let (y, cb, cr) = picture(k);
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
            Err(e) => panic!("unexpected {e}"),
        }
    }
    stream
}

fn assert_own_decode_matches_reference(m4v: &str, yuv: &str) {
    assert_own_decode_matches_reference_dims(m4v, yuv, 64, 64);
}

fn assert_own_decode_matches_reference_dims(m4v: &str, yuv: &str, w: usize, h: usize) {
    let stream = fixture(m4v);
    let reference = fixture(yuv);
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("committed stream must decode");
    frames.extend(dec.flush());
    let (cw, ch) = (w / 2, h / 2);
    let frame_len = w * h + 2 * cw * ch;
    assert_eq!(
        reference.len(),
        frames.len() * frame_len,
        "{yuv}: frame count"
    );
    for (k, frame) in frames.iter().enumerate() {
        let base = k * frame_len;
        assert_eq!(
            frame.luma_samples(),
            &reference[base..base + w * h],
            "{m4v}: frame {k} luma vs reference decode"
        );
        assert_eq!(
            frame.cb_samples(),
            &reference[base + w * h..base + w * h + cw * ch],
            "{m4v}: frame {k} cb"
        );
        assert_eq!(
            frame.cr_samples(),
            &reference[base + w * h + cw * ch..base + frame_len],
            "{m4v}: frame {k} cr"
        );
    }
}

#[test]
fn method2_stream_reproduces_committed_fixture() {
    assert_eq!(
        build_stream(false),
        fixture("enc_intra_m2_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn method1_stream_reproduces_committed_fixture() {
    assert_eq!(
        build_stream(true),
        fixture("enc_intra_m1_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn method2_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_intra_m2_64x64.m4v", "enc_intra_m2_64x64.yuv");
}

/// The method-1 stream hits the documented §7.4.4.5 spec-vs-ecosystem
/// divergence (see `crate::compat`): the reference decoder applies
/// the mismatch toggle to non-intra blocks only, so the **ecosystem**
/// mode must be bit-exact while the literal-spec decode carries a
/// bounded ±1 envelope on the toggled-`F[7][7]` blocks.
#[test]
fn method1_stream_decodes_against_reference_decoder_per_compat_contract() {
    use oxideav_mpeg4video::compat::DecodeOptions;
    let stream = fixture("enc_intra_m1_64x64.m4v");
    let reference = fixture("enc_intra_m1_64x64.yuv");
    let frame_len = 64 * 64 + 2 * 32 * 32;
    let mut stats = Vec::new();
    for opts in [DecodeOptions::ecosystem(), DecodeOptions::spec()] {
        let mut dec = Mpeg4VideoDecoder::with_options(opts);
        let mut frames = dec.decode(&stream).expect("committed stream must decode");
        frames.extend(dec.flush());
        assert_eq!(reference.len(), frames.len() * frame_len);
        let (mut differing, mut max) = (0usize, 0i32);
        for (k, frame) in frames.iter().enumerate() {
            let ours: Vec<u8> = frame
                .luma_samples()
                .iter()
                .chain(frame.cb_samples())
                .chain(frame.cr_samples())
                .copied()
                .collect();
            for (a, b) in ours
                .iter()
                .zip(&reference[k * frame_len..(k + 1) * frame_len])
            {
                let d = (i32::from(*a) - i32::from(*b)).abs();
                if d > 0 {
                    differing += 1;
                }
                max = max.max(d);
            }
        }
        stats.push((differing, max));
    }
    assert_eq!(
        stats[0],
        (0, 0),
        "ecosystem-compat decode must reproduce the reference bit-exactly"
    );
    // Literal-spec decode: the intra mismatch toggle flips isolated
    // samples by at most ±1. The count is a property of the pinned
    // fixture — regenerate both fixture files if the encoder changes.
    assert_eq!(stats[1].1, 1, "spec-mode envelope must stay ±1");
    assert_eq!(
        stats[1].0, 834,
        "spec-mode differing-sample count drifted; re-measure after regenerating fixtures"
    );
}

#[test]
fn ip_stream_reproduces_committed_fixture() {
    assert_eq!(
        build_ip_stream(),
        fixture("enc_ip_m2_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ip_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ip_m2_64x64.m4v", "enc_ip_m2_64x64.yuv");
}

#[test]
fn four_mv_stream_reproduces_committed_fixture() {
    let built = build_4mv_stream();
    if maybe_write_fixture("enc_ip_4mv_64x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ip_4mv_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn four_mv_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ip_4mv_64x64.m4v", "enc_ip_4mv_64x64.yuv");
}

#[test]
fn qpel_stream_reproduces_committed_fixture() {
    let built = build_qpel_stream();
    if maybe_write_fixture("enc_ip_qpel_64x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ip_qpel_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn qpel_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ip_qpel_64x64.m4v", "enc_ip_qpel_64x64.yuv");
}

#[test]
fn qpel_4mv_stream_reproduces_committed_fixture() {
    let built = build_qpel_4mv_stream();
    if maybe_write_fixture("enc_ip_qpel4mv_64x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ip_qpel4mv_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn qpel_4mv_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ip_qpel4mv_64x64.m4v", "enc_ip_qpel4mv_64x64.yuv");
}

#[test]
fn ipb_stream_reproduces_committed_fixture() {
    let built = build_ipb_stream();
    if maybe_write_fixture("enc_ipb_64x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ipb_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ipb_64x64.m4v", "enc_ipb_64x64.yuv");
}

#[test]
fn ipb_qpel_4mv_stream_reproduces_committed_fixture() {
    let built = build_ipb_qpel4mv_stream();
    if maybe_write_fixture("enc_ipb_qpel4mv_64x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_qpel4mv_64x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ipb_qpel_4mv_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference("enc_ipb_qpel4mv_64x64.m4v", "enc_ipb_qpel4mv_64x64.yuv");
}

#[test]
fn fcode2_stream_reproduces_committed_fixture() {
    let built = build_fcode2_stream();
    if maybe_write_fixture("enc_ip_fcode2_96x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ip_fcode2_96x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn fcode2_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ip_fcode2_96x64.m4v",
        "enc_ip_fcode2_96x64.yuv",
        96,
        64,
    );
}

#[test]
fn fcode3_qpel_ipb_stream_reproduces_committed_fixture() {
    let built = build_fcode3_qpel_ipb_stream();
    if maybe_write_fixture("enc_ipb_fcode3_qpel4mv_96x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_fcode3_qpel4mv_96x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn fcode3_qpel_ipb_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ipb_fcode3_qpel4mv_96x64.m4v",
        "enc_ipb_fcode3_qpel4mv_96x64.yuv",
        96,
        64,
    );
}

#[test]
fn aq_ipb_stream_reproduces_committed_fixture() {
    let built = build_aq_ipb_stream();
    if maybe_write_fixture("enc_ipb_aq4mv_96x48.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_aq4mv_96x48.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn aq_ipb_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ipb_aq4mv_96x48.m4v",
        "enc_ipb_aq4mv_96x48.yuv",
        96,
        48,
    );
}

#[test]
fn ipb_vp_fcode2_96x64_stream_reproduces_committed_fixture() {
    let built = build_vp_ipb_stream();
    if maybe_write_fixture("enc_ipb_vp_fcode2_96x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_vp_fcode2_96x64.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ipb_vp_fcode2_96x64_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ipb_vp_fcode2_96x64.m4v",
        "enc_ipb_vp_fcode2_96x64.yuv",
        96,
        64,
    );
}

#[test]
fn ip_dp_aq_96x48_stream_reproduces_committed_fixture() {
    let built = build_dp_ip_stream();
    if maybe_write_fixture("enc_ip_dp_aq_96x48.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ip_dp_aq_96x48.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ip_dp_aq_96x48_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ip_dp_aq_96x48.m4v",
        "enc_ip_dp_aq_96x48.yuv",
        96,
        48,
    );
}

#[test]
fn ipb_dprvlc_aq4mv_96x48_stream_reproduces_committed_fixture() {
    let built = build_dp_rvlc_ipb_stream();
    if maybe_write_fixture("enc_ipb_dprvlc_aq4mv_96x48.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("enc_ipb_dprvlc_aq4mv_96x48.m4v"),
        "encoder output drifted from the black-box-validated fixture; \
         regenerate the fixture AND its reference decode"
    );
}

#[test]
fn ipb_dprvlc_aq4mv_96x48_stream_decodes_bit_exact_against_reference_decoder() {
    assert_own_decode_matches_reference_dims(
        "enc_ipb_dprvlc_aq4mv_96x48.m4v",
        "enc_ipb_dprvlc_aq4mv_96x48.yuv",
        96,
        48,
    );
}

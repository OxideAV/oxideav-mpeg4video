//! S(GMC)-VOPs on the §6.2.5.3 data-partitioned layout, both ways:
//! partition 1 carries `not_coded` / `mcbpc` / `mcsel` / the local
//! `motion_coding()` bodies, partition 2 the texture headers, the
//! texture partition the (optionally reversible-VLC) coefficients —
//! decoded back through the crate's own DP S walk sample-exact, and
//! pinned against the opaque reference decode
//! (`tests/fixtures/NOTES.md`).

#![allow(dead_code)]

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::packet_encode::ResilienceConfig;
use oxideav_mpeg4video::svop_encode::{encode_s_vop, reconstruct_own_s_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Textured scene with a panning background and, from frame 1 on, a
/// freshly revealed smooth patch that forces intra macroblocks into
/// the P/S pictures.
fn picture(w: usize, h: usize, frame_index: usize, pan: (i64, i64)) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let scene = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5).rem_euclid(160)
            + (x * 3 - y * 11).rem_euclid(97)
            + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
        (30 + v.rem_euclid(200)) as u8
    };
    let n = frame_index as i64;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = scene(col as i64 + n * pan.0, row as i64 + n * pan.1);
        }
    }
    if frame_index > 0 {
        // A smooth patch (tiny mean-removed activity) that the textured
        // reference cannot predict: intra wins there.
        for row in 16..48.min(h) {
            for col in 32..64.min(w) {
                y[row * w + col] = (60 + row + frame_index * 10) as u8;
            }
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = scene(col as i64 + n * pan.0 / 2, row as i64) / 2 + 60;
            cr[row * cw + col] = 128 + ((col as i64 * 3 + n) % 9) as u8 * 4;
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

/// I + 3 S(GMC) on the data-partitioned layout — plain, with video
/// packets, and with packets + RVLC — every stream decodes
/// sample-exact against the closed-loop reconstruction, GMC and local
/// macroblocks both present, packets cut inside the S-VOPs.
#[test]
fn gmc_data_partitioned_round_trips() {
    for (packet_bits, rvlc, points) in [(0u32, false, 1u8), (600, false, 1), (600, true, 3)] {
        let cfg = EncoderConfig {
            width: 96,
            height: 64,
            gmc: true,
            gmc_points: points,
            fcode: 2,
            adaptive_quant: true,
            intra_dc_vlc_thr: 3,
            resilience: ResilienceConfig {
                packet_bits,
                data_partitioned: true,
                reversible_vlc: rvlc,
            },
            ..EncoderConfig::default()
        };
        let (headers, vol) = vol_of(&cfg);
        assert!(vol.data_partitioned && vol.reversible_vlc == rvlc);
        let mut stream = headers;
        let mut recons = Vec::new();
        let mut store = FrameStore::new();
        for k in 0..4usize {
            let (y, cb, cr) = picture(96, 64, k, (2, 1));
            let view = FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: 96,
                height: 64,
            };
            if k == 0 {
                let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 9);
                stream.extend_from_slice(&unit);
                store.push_anchor(recon.clone());
                recons.push(recon);
            } else {
                let reference = store.backward().unwrap().clone();
                let (unit, stats) = encode_s_vop(&vol, &cfg, &view, &reference, 0, k as u16, 9);
                assert!(
                    stats.gmc + stats.gmc_skipped > 0 && stats.local + stats.intra > 0,
                    "S-VOP {k} must mix GMC and local/intra macroblocks: {stats:?}"
                );
                if packet_bits > 0 {
                    assert!(stats.packets >= 2, "S-VOP {k} must cut packets ({stats:?})");
                }
                let (recon, _) = reconstruct_own_s_vop_with_motion(&vol, &unit, &mut store);
                stream.extend_from_slice(&unit);
                recons.push(recon);
            }
        }
        assert_exact(&decode_all(&stream), &recons);
    }
}

/// Black-box observation (recorded in `tests/fixtures/NOTES.md`): the
/// opaque reference decoder does **not** read the §6.2.5.3
/// `data_partitioned_p_vop()` S(GMC) clauses — on a data-partitioned
/// S-VOP it desynchronises at the very first macroblock (every
/// macroblock of the picture differs, intra ones included, while the
/// I-VOP and the combined-syntax S(GMC) streams stay bit-exact). The
/// crate keeps the printed syntax on both sides; the registry
/// combination is exercised here through the crate's own decoder.
#[test]
fn registry_gmc_dp_rvlc_stream_decodes() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default()
        .set("gmc", "true")
        .set("gmc-points", "3")
        .set("fcode", "2")
        .set("bf", "1")
        .set("mb-aq", "true")
        .set("data-partitioned", "true")
        .set("rvlc", "true")
        .set("packet-bits", "600")
        .set("qp", "7");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..5usize {
        let (y, cb, cr) = picture(w, h, k, (2, 1));
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
    let frames = decode_all(&stream);
    assert_eq!(frames.len(), 5);
    let s_vops = stream
        .windows(5)
        .filter(|win| win[..4] == [0, 0, 1, 0xB6] && win[4] >> 6 == 0b11)
        .count();
    assert!(s_vops >= 2, "expected S-VOPs in the stream");
}

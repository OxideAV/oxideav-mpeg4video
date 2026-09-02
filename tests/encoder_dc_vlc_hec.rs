//! `intra_dc_vlc_thr` (Table 6-25) on the encoder — explicit thresholds
//! (DC differentials riding the AC VLC above the running quantiser
//! threshold, in combined and data-partitioned layouts), the measured
//! election on I-VOPs — and the S(GMC)-VOP video-packet HEC body with
//! its `sprite_trajectory()` restatement, both decoded back through
//! the crate's own decoder and pinned against the black-box reference
//! decode (`tests/fixtures/NOTES.md`).

use oxideav_mpeg4video::bitreader::BitReader;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, encode_i_vop_elect_thr, encode_i_vop_with_thr, write_configuration_headers,
    EncoderConfig, FrameView,
};
use oxideav_mpeg4video::packet_encode::ResilienceConfig;
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop};
use oxideav_mpeg4video::svop_encode::{encode_s_vop, reconstruct_own_s_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;
use oxideav_mpeg4video::vop::{parse_vop_header_body, VopContext};

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

/// The `intra_dc_vlc_thr` of every VOP header in `stream`.
fn vop_thresholds(stream: &[u8], vol: &oxideav_mpeg4video::vol::VolHeader) -> Vec<u8> {
    let mut starts: Vec<usize> = stream
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == [0, 0, 1, 0xB6])
        .map(|(i, _)| i)
        .collect();
    starts.push(stream.len());
    starts
        .windows(2)
        .map(|w| {
            let mut br = BitReader::new(&stream[w[0]..w[1]]);
            br.read_bits(32).unwrap();
            parse_vop_header_body(
                &mut br,
                vol.time_increment_resolution,
                VopContext::from_vol(vol),
            )
            .unwrap()
            .intra_dc_vlc_thr
        })
        .collect()
}

/// I + 2 P at an explicit threshold: the AC-VLC DC path (7), a
/// mid-table threshold straddled by the adaptive quantiser (3 → 17),
/// combined and data-partitioned, all sample-exact.
#[test]
fn explicit_thresholds_round_trip_in_both_layouts() {
    for (thr, qp, dp) in [
        (7u8, 6u32, false),
        (3, 17, false),
        (7, 9, true),
        (3, 17, true),
    ] {
        let cfg = EncoderConfig {
            width: 96,
            height: 64,
            adaptive_quant: true,
            intra_dc_vlc_thr: thr,
            resilience: ResilienceConfig {
                packet_bits: if dp { 700 } else { 0 },
                data_partitioned: dp,
                reversible_vlc: dp,
            },
            ..EncoderConfig::default()
        };
        let (headers, vol) = vol_of(&cfg);
        let mut stream = headers;
        let mut recons = Vec::new();
        let mut store = FrameStore::new();
        for k in 0..3usize {
            let (y, cb, cr) = picture(96, 64, k, (2, 1));
            let view = FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: 96,
                height: 64,
            };
            if k == 0 {
                let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, qp);
                stream.extend_from_slice(&unit);
                store.push_anchor(recon.clone());
                recons.push(recon);
            } else {
                let reference = store.backward().unwrap().clone();
                let (unit, stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, qp);
                assert!(
                    stats.intra > 0,
                    "thr {thr}: P-VOP {k} needs intra MBs ({stats:?})"
                );
                let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
                stream.extend_from_slice(&unit);
                recons.push(recon);
            }
        }
        assert_eq!(vop_thresholds(&stream, &vol), vec![thr; 3]);
        assert_exact(&decode_all(&stream), &recons);
    }
}

/// The election codes the I-VOP both ways and keeps the smaller unit;
/// the explicit variants bracket it.
#[test]
fn election_keeps_the_cheaper_variant() {
    let cfg = EncoderConfig {
        width: 64,
        height: 48,
        ..EncoderConfig::default()
    };
    let (_, vol) = vol_of(&cfg);
    for qp in [2u32, 8, 20, 31] {
        let (y, cb, cr) = picture(64, 48, 0, (0, 0));
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 64,
            height: 48,
        };
        let (dc, _) = encode_i_vop_with_thr(&vol, &cfg, &view, 0, 0, qp, 0);
        let (ac, _) = encode_i_vop_with_thr(&vol, &cfg, &view, 0, 0, qp, 7);
        let (elected, recon, thr) = encode_i_vop_elect_thr(&vol, &cfg, &view, 0, 0, qp);
        assert!(thr == 0 || thr == 7);
        assert_eq!(elected.len(), dc.len().min(ac.len()), "qp {qp}");
        assert_eq!(elected, if thr == 7 { ac } else { dc });
        let mut s = write_configuration_headers(&cfg);
        s.extend_from_slice(&elected);
        assert_exact(&decode_all(&s), &[recon]);
    }
}

/// Registry: `dc-vlc-thr` is validated, `auto-dc-vlc` carries the
/// elected threshold into the following P-VOP headers.
#[test]
fn registry_dc_vlc_options() {
    use oxideav_core::Encoder as _;
    let (w, h) = (64usize, 48usize);
    let encode =
        |opts: oxideav_core::CodecOptions| -> (Vec<u8>, oxideav_mpeg4video::vol::VolHeader) {
            let mut params =
                oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
            params.width = Some(w as u32);
            params.height = Some(h as u32);
            params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
            params.options = opts;
            let mut enc =
                oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
            let extradata = enc.output_params().extradata.clone();
            let pos = extradata
                .windows(4)
                .position(|x| x == [0, 0, 1, 0x20])
                .unwrap();
            let vol = parse_video_object_layer(&extradata[pos..], 0x03)
                .unwrap_or_else(|_| parse_video_object_layer(&extradata[pos..], 0xF3).unwrap());
            for k in 0..4usize {
                let (y, cb, cr) = picture(w, h, k, (1, 0));
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
            (stream, vol)
        };
    let (stream, vol) = encode(oxideav_core::CodecOptions::default().set("dc-vlc-thr", "7"));
    assert_eq!(vop_thresholds(&stream, &vol), vec![7; 4]);
    assert_eq!(decode_all(&stream).len(), 4);
    let (stream, vol) = encode(
        oxideav_core::CodecOptions::default()
            .set("auto-dc-vlc", "true")
            .set("qp", "24"),
    );
    let thrs = vop_thresholds(&stream, &vol);
    assert!(
        thrs.iter().all(|&t| t == thrs[0]),
        "carried to every VOP: {thrs:?}"
    );
    assert_eq!(decode_all(&stream).len(), 4);
    let mut p = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    p.width = Some(32);
    p.height = Some(32);
    p.options = oxideav_core::CodecOptions::default().set("dc-vlc-thr", "8");
    assert!(oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&p).is_err());
}

/// S(GMC)-VOP video packets now carry the HEC body with its
/// `sprite_trajectory()` restatement (one and three warping points)
/// — the decoder's packet-header parser consumes it and the stream
/// decodes sample-exact.
#[test]
fn gmc_packets_carry_the_hec_trajectory() {
    for points in [1u8, 3] {
        let cfg = EncoderConfig {
            width: 96,
            height: 64,
            gmc: true,
            gmc_points: points,
            fcode: 2,
            intra_dc_vlc_thr: 7,
            resilience: ResilienceConfig {
                packet_bits: 500,
                ..Default::default()
            },
            ..EncoderConfig::default()
        };
        let (headers, vol) = vol_of(&cfg);
        let mut stream = headers;
        let mut recons = Vec::new();
        let mut store = FrameStore::new();
        for k in 0..3usize {
            let (y, cb, cr) = picture(96, 64, k, (2, 1));
            let view = FrameView {
                y: &y,
                cb: &cb,
                cr: &cr,
                width: 96,
                height: 64,
            };
            if k == 0 {
                let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 6);
                stream.extend_from_slice(&unit);
                store.push_anchor(recon.clone());
                recons.push(recon);
            } else {
                let reference = store.backward().unwrap().clone();
                let (unit, stats) = encode_s_vop(&vol, &cfg, &view, &reference, 0, k as u16, 6);
                assert!(
                    stats.packets >= 2,
                    "{points} points: S-VOP {k} must cut packets ({stats:?})"
                );
                let (recon, _) = reconstruct_own_s_vop_with_motion(&vol, &unit, &mut store);
                stream.extend_from_slice(&unit);
                recons.push(recon);
            }
        }
        assert_exact(&decode_all(&stream), &recons);
    }
}

/// Black-box pin: AC-VLC intra DC (`intra_dc_vlc_thr == 7`) + S(GMC)
/// video packets with HEC bodies (trajectory restated) + B-VOPs —
/// the reference decode is bit-exact against our closed loop.
#[test]
fn blackbox_dcvlc7_gmc_hec_stream_is_bit_exact() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default()
        .set("gmc", "true")
        .set("fcode", "2")
        .set("bf", "1")
        .set("dc-vlc-thr", "7")
        .set("mb-aq", "true")
        .set("packet-bits", "500")
        .set("qp", "6");
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
    pin_fixture("enc_isb_dcvlc7_hec_96x64.m4v", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let frames = decode_all(&stream);
    let yuv = std::fs::read(fixture_path("enc_isb_dcvlc7_hec_96x64.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&frames, &yuv, w, h);
    assert_eq!(
        (differing, max),
        (0, 0),
        "dcvlc7 + GMC HEC + B: {differing}/{total} samples differ (max {max})"
    );
}

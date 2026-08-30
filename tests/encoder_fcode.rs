//! `fcode > 1` motion ranges: the encoder's §7.6 search window and
//! Table 7-9 range follow `vop_fcode_forward` / `vop_fcode_backward`,
//! long displacements are actually found and emitted through the
//! `r_size`-bit residual form of §6.2.6.2 `motion_vector()`, and the
//! result decodes sample-exact through the crate's own decoder.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{
    encode_p_vop, reconstruct_own_p_vop_with_motion, PVopEncodeStats,
};
use oxideav_mpeg4video::pvop_mv::PvopMbMotion;
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// A textured background translating by `(shift_x, shift_y)` pels
/// per frame — far beyond the ±8-pel `fcode == 1` window.
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
    let mut cb = vec![0u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + 64;
        }
    }
    (y, cb, cr)
}

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

struct Encoded {
    stream: Vec<u8>,
    recons: Vec<DecodedFrame>,
    stats: Vec<PVopEncodeStats>,
    motion: Vec<Vec<PvopMbMotion>>,
}

fn encode_ip(cfg: &EncoderConfig, frames: usize, shift: (i64, i64), qp: u32) -> Encoded {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let mut out = Encoded {
        stream: headers,
        recons: Vec::new(),
        stats: Vec::new(),
        motion: Vec::new(),
    };
    let mut store = FrameStore::new();
    for k in 0..frames {
        let (y, cb, cr) = picture(w, h, k, shift);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        if k == 0 {
            let (unit, recon) = encode_i_vop(&vol, cfg, &view, 0, 0, qp);
            out.stream.extend_from_slice(&unit);
            store.push_anchor(recon.clone());
            out.recons.push(recon);
        } else {
            let reference = store.backward().expect("anchor present").clone();
            let (unit, st) = encode_p_vop(&vol, cfg, &view, &reference, 0, k as u16, qp);
            let (recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &unit, &mut store);
            out.stream.extend_from_slice(&unit);
            out.recons.push(recon);
            out.stats.push(st);
            out.motion.push(motion);
        }
    }
    out
}

fn assert_self_decodes(enc: &Encoded) {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&enc.stream).expect("own stream must decode");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), enc.recons.len(), "frame count");
    for (k, (d, r)) in frames.iter().zip(enc.recons.iter()).enumerate() {
        assert_eq!(d.luma_samples(), r.luma_samples(), "frame {k} luma");
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

/// The largest |component| (in MV units) over every decoded inter
/// vector of a P-VOP.
fn max_component(motion: &[PvopMbMotion]) -> i32 {
    motion
        .iter()
        .flat_map(|m| match m {
            PvopMbMotion::OneMv(v) => vec![*v],
            PvopMbMotion::FourMv(vs) => vs.to_vec(),
            _ => Vec::new(),
        })
        .map(|v| v.x.abs().max(v.y.abs()))
        .max()
        .unwrap_or(0)
}

/// The P-VOP header's `vop_fcode_forward` of every P unit in `stream`.
fn header_fcodes(stream: &[u8], vol_fcode_expected: u8) -> usize {
    use oxideav_mpeg4video::vop::{parse_vop_header_body, VopCodingType, VopContext};
    let pos = stream
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&stream[pos..], 0x03).unwrap();
    let mut count = 0;
    let mut at = 0;
    while let Some(off) = stream[at..].windows(4).position(|w| w == [0, 0, 1, 0xB6]) {
        let start = at + off;
        let mut br = oxideav_mpeg4video::bitreader::BitReader::new(&stream[start + 4..]);
        let vop = parse_vop_header_body(
            &mut br,
            vol.time_increment_resolution,
            VopContext::from_vol(&vol),
        )
        .unwrap();
        if matches!(vop.coding_type, VopCodingType::P) {
            assert_eq!(vop.fcode_fwd, vol_fcode_expected, "P-VOP fcode");
            count += 1;
        }
        at = start + 4;
    }
    count
}

#[test]
fn fcode2_half_sample_finds_20_pel_translation() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let enc = encode_ip(&cfg, 3, (20, 5), 4);
    assert_self_decodes(&enc);
    assert_eq!(header_fcodes(&enc.stream, 2), 2);
    for (k, m) in enc.motion.iter().enumerate() {
        // 20 pels = 40 half-sample units — unrepresentable under
        // fcode 1 (|MV| <= 31).
        assert!(
            max_component(m) >= 40,
            "P-VOP {k}: expected a 20-pel vector, max component {}",
            max_component(m)
        );
        assert!(
            enc.stats[k].inter + enc.stats[k].inter4v > enc.stats[k].intra,
            "P-VOP {k}: motion should dominate intra ({:?})",
            enc.stats[k]
        );
    }
}

#[test]
fn fcode3_quarter_sample_finds_24_pel_translation() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        fcode: 3,
        quarter_sample: true,
        ..EncoderConfig::default()
    };
    let enc = encode_ip(&cfg, 3, (24, -3), 4);
    assert_self_decodes(&enc);
    for (k, m) in enc.motion.iter().enumerate() {
        // 24 pels = 96 quarter-sample units — beyond the fcode-2
        // quarter range (|MV| <= 63) and the fcode-1 one.
        assert!(
            max_component(m) >= 96,
            "P-VOP {k}: expected a 24-pel vector, max component {}",
            max_component(m)
        );
    }
}

#[test]
fn fcode2_with_inter4v_stays_in_range_and_exact() {
    let cfg = EncoderConfig {
        width: 80,
        height: 48,
        fcode: 2,
        four_mv: true,
        ..EncoderConfig::default()
    };
    let enc = encode_ip(&cfg, 4, (13, 9), 6);
    assert_self_decodes(&enc);
    for m in &enc.motion {
        // Table 7-9 fcode 2: every component in [-64, 63].
        assert!(max_component(m) <= 64);
    }
}

#[test]
fn every_fcode_round_trips_a_small_grid() {
    for fcode in 1..=7u8 {
        for quarter_sample in [false, true] {
            let cfg = EncoderConfig {
                width: 48,
                height: 32,
                fcode,
                quarter_sample,
                ..EncoderConfig::default()
            };
            let enc = encode_ip(&cfg, 3, (5, 2), 8);
            assert_self_decodes(&enc);
            let (low, high) = {
                let f = 1i32 << (fcode - 1);
                (-32 * f, 32 * f - 1)
            };
            for m in &enc.motion {
                for v in m.iter().flat_map(|m| match m {
                    PvopMbMotion::OneMv(v) => vec![*v],
                    PvopMbMotion::FourMv(vs) => vs.to_vec(),
                    _ => Vec::new(),
                }) {
                    assert!(
                        (low..=high).contains(&v.x) && (low..=high).contains(&v.y),
                        "fcode {fcode}: {v:?} outside Table 7-9 [{low}, {high}]"
                    );
                }
            }
        }
    }
}

#[test]
fn registry_fcode_option_drives_p_and_b_vops() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default()
        .set("fcode", "3")
        .set("bf", "1");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..5usize {
        let (y, cb, cr) = picture(w, h, k, (18, 4));
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
    // Coded order I0 P2 B1 P4 B3: both P headers carry fcode 3, and
    // the B headers carry it on both directions (the decoder parses
    // both — any mismatch would break the decode below).
    assert_eq!(header_fcodes(&stream, 3), 2);
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("own stream must decode");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), 5);

    // Rejects out-of-range fcode.
    params.options = oxideav_core::CodecOptions::default().set("fcode", "8");
    assert!(oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).is_err());
}

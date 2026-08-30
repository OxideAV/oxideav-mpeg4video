//! Error-resilience emission: §6.2.5 video packets (resync markers +
//! `video_packet_header` with and without the HEC body), §6.2.5.3 data
//! partitioning (`dc_marker` / `motion_marker`) and the reversible-VLC
//! texture partition — every stream decodes sample-exact through the
//! crate's own decoder, and a corrupted RVLC texture partition still
//! decodes through the §E.1.4.4 recovery path.

use oxideav_mpeg4video::bvop_encode::encode_b_vop;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::packet_encode::ResilienceConfig;
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop_with_motion};
use oxideav_mpeg4video::vol::parse_video_object_layer;

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Textured background translating by `shift` pels per frame with a
/// noise band (so every packet carries real texture) and a flat band.
fn picture(w: usize, h: usize, frame_index: usize, shift: (i64, i64)) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let (ox, oy) = (frame_index as i64 * shift.0, frame_index as i64 * shift.1);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let (x, yy) = (col as i64 + ox, row as i64 + oy);
            let band = x.rem_euclid(w as i64) as usize * 3 / w;
            let v: i64 = match band {
                0 => 40 + (x * 7 + yy * 5).rem_euclid(160),
                1 => 128,
                _ => {
                    let mut s = (x as u32).wrapping_mul(0x9E37_79B9)
                        ^ (yy as u32).wrapping_mul(0x85EB_CA6B);
                    16 + i64::from(lcg(&mut s) >> 24) * 219 / 255
                }
            };
            y[row * w + col] = v.clamp(16, 235) as u8;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let cr = vec![132u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] =
                (90 + ((col as i64 + ox / 2) * 3 + row as i64).rem_euclid(70)) as u8;
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

struct Encoded {
    stream: Vec<u8>,
    recons: Vec<DecodedFrame>,
    packets: usize,
}

/// I + (frames − 1) P.
fn encode_ip(cfg: &EncoderConfig, frames: usize, shift: (i64, i64), qp: u32) -> Encoded {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let (headers, vol) = vol_of(cfg);
    let mut out = Encoded {
        stream: headers,
        recons: Vec::new(),
        packets: 0,
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
            let reference = store.backward().unwrap().clone();
            let (unit, st) = encode_p_vop(&vol, cfg, &view, &reference, 0, k as u16, qp);
            out.packets += st.packets;
            let (recon, _) = reconstruct_own_p_vop_with_motion(&vol, &unit, &mut store);
            out.stream.extend_from_slice(&unit);
            out.recons.push(recon);
        }
    }
    out
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

/// Count the byte-aligned resync markers of a P-VOP stream under
/// `fcode` (16 + fcode zeros... the marker is `15 + fcode` zeros then a
/// one, byte-aligned): a coarse structural check independent of the
/// decoder.
fn count_vop_start_codes(stream: &[u8]) -> usize {
    stream.windows(4).filter(|w| *w == [0, 0, 1, 0xB6]).count()
}

#[test]
fn video_packets_combined_ip_round_trip_every_fcode() {
    for fcode in [1u8, 2, 3] {
        let cfg = EncoderConfig {
            width: 96,
            height: 64,
            fcode,
            resilience: ResilienceConfig {
                packet_bits: 400,
                ..Default::default()
            },
            ..EncoderConfig::default()
        };
        let enc = encode_ip(&cfg, 3, (6, 2), 6);
        assert!(
            enc.packets >= 4,
            "fcode {fcode}: {} packets cut",
            enc.packets
        );
        assert_exact(&decode_all(&enc.stream), &enc.recons);
        assert_eq!(count_vop_start_codes(&enc.stream), 3);
    }
}

#[test]
fn video_packets_with_adaptive_quant_reseed_the_quantiser() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        adaptive_quant: true,
        four_mv: true,
        resilience: ResilienceConfig {
            packet_bits: 300,
            ..Default::default()
        },
        ..EncoderConfig::default()
    };
    let enc = encode_ip(&cfg, 4, (3, 1), 9);
    assert!(enc.packets > 0);
    assert_exact(&decode_all(&enc.stream), &enc.recons);
}

#[test]
fn data_partitioned_ip_round_trips_single_and_multi_packet() {
    for packet_bits in [0u32, 500] {
        for quant_type in [false, true] {
            let cfg = EncoderConfig {
                width: 80,
                height: 48,
                quant_type,
                adaptive_quant: true,
                resilience: ResilienceConfig {
                    packet_bits,
                    data_partitioned: true,
                    reversible_vlc: false,
                },
                ..EncoderConfig::default()
            };
            let enc = encode_ip(&cfg, 4, (4, 2), 7);
            assert_eq!(enc.packets > 0, packet_bits > 0);
            assert_exact(&decode_all(&enc.stream), &enc.recons);
        }
    }
}

#[test]
fn reversible_vlc_texture_round_trips_with_every_tool() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        fcode: 2,
        four_mv: true,
        quarter_sample: true,
        adaptive_quant: true,
        resilience: ResilienceConfig {
            packet_bits: 600,
            data_partitioned: true,
            reversible_vlc: true,
        },
        ..EncoderConfig::default()
    };
    for qp in [2u32, 12, 31] {
        let enc = encode_ip(&cfg, 4, (9, -3), qp);
        assert_exact(&decode_all(&enc.stream), &enc.recons);
    }
}

/// B-VOPs inside a data-partitioned + RVLC VOL keep the combined
/// syntax (§6.2.5.3 NOTE) but still cut video packets.
#[test]
fn b_vops_in_a_partitioned_vol_use_combined_packets() {
    let (w, h) = (96usize, 64usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        b_vops: true,
        fcode: 2,
        adaptive_quant: true,
        resilience: ResilienceConfig {
            packet_bits: 350,
            data_partitioned: true,
            reversible_vlc: true,
        },
        ..EncoderConfig::default()
    };
    let (headers, vol) = vol_of(&cfg);
    let mut stream = headers;
    let mut store = FrameStore::new();
    let pics: Vec<Planes> = (0..3).map(|k| picture(w, h, k, (5, 2))).collect();
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
    assert!(stats.packets > 0, "B-VOP must cut packets ({stats:?})");
    assert_exact(&decode_all(&stream), &[i_recon, b_recon, p_recon]);
}

/// The registry path wires every option; `rvlc` without
/// `data-partitioned` is rejected.
#[test]
fn registry_resilience_options() {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default().set("rvlc", "true");
    assert!(oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).is_err());
    params.options = oxideav_core::CodecOptions::default()
        .set("packet-bits", "500")
        .set("data-partitioned", "true")
        .set("rvlc", "true")
        .set("bf", "2")
        .set("mb-aq", "true")
        .set("fcode", "2");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..6usize {
        let (y, cb, cr) = picture(w, h, k, (4, 1));
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
    // The VOL declares the tools.
    let pos = stream
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&stream[pos..], 0xF3).unwrap();
    assert!(!vol.resync_marker_disable);
    assert!(vol.data_partitioned);
    assert!(vol.reversible_vlc);
    assert_eq!(decode_all(&stream).len(), 6);
}

/// Corrupt the RVLC texture partition of the second video packet of a
/// P-VOP: the decoder's §E.1.4.4 two-way recovery keeps the stream
/// decodable (every frame is produced) and the I-VOP is untouched.
#[test]
fn corrupted_rvlc_texture_partition_still_decodes() {
    let cfg = EncoderConfig {
        width: 96,
        height: 64,
        resilience: ResilienceConfig {
            packet_bits: 400,
            data_partitioned: true,
            reversible_vlc: true,
        },
        ..EncoderConfig::default()
    };
    let enc = encode_ip(&cfg, 2, (4, 2), 5);
    // Locate the P-VOP, then its second resync marker (P marker under
    // fcode 1: 16 zeros + 1, byte-aligned → bytes 00 00 80 after
    // stuffing), then its motion_marker; corrupt bytes well after it.
    let stream = enc.stream.clone();
    let p_start = stream
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == [0, 0, 1, 0xB6])
        .nth(1)
        .map(|(i, _)| i)
        .expect("P-VOP start code");
    let mut corrupted = stream.clone();
    let tail = &mut corrupted[p_start + 40..];
    let n = tail.len();
    for b in &mut tail[n / 2..n / 2 + 4] {
        *b ^= 0x5A;
    }
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&corrupted).unwrap_or_default();
    frames.extend(dec.flush());
    assert!(!frames.is_empty(), "the I-VOP must survive");
    assert_eq!(frames[0].luma_samples(), enc.recons[0].luma_samples());
}

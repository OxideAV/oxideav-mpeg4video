//! Rate-control validation: the registry encoder under a bitrate
//! target must (a) signal `vbv_parameters` in the VOL, (b) land the
//! measured rate near the target, (c) satisfy the Annex D item-9 VBV
//! constraints under an independent re-simulation of the model from
//! the emitted packets, and (d) stay deterministic and decodable.

use oxideav_core::Encoder as _;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// A moving textured scene busy enough that the quantiser has real
/// work: translating background + a bouncing bright square + noise
/// bands.
fn picture(w: usize, h: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w / 2, h / 2);
    let bg = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5) % 160 + ((x / 5 + y / 3) % 17) * 5;
        (30 + v.rem_euclid(190)) as u8
    };
    let (ox, oy) = ((k * 3) as i64, (k * 2) as i64);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    let bx = (k * 11) % (w - 12);
    let by = (k * 5) % (h - 12);
    for row in by..by + 12 {
        for col in bx..bx + 12 {
            y[row * w + col] = 240;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = bg(col as i64 * 2 + ox, row as i64) / 2 + 60;
            cr[row * cw + col] = 128 + ((col + k) % 16) as u8;
        }
    }
    (y, cb, cr)
}

fn encode(bitrate: u32, frames: usize, bf: u32) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    let mut opts = oxideav_core::CodecOptions::default().set("bitrate", bitrate.to_string());
    if bf > 0 {
        opts = opts.set("bf", bf.to_string());
    }
    params.options = opts;
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    let extradata = enc.output_params().extradata.clone();
    for k in 0..frames {
        let (y, cb, cr) = picture(64, 64, k);
        let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
            pts: None,
            planes: vec![
                oxideav_core::VideoPlane {
                    stride: 64,
                    data: y,
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: cb,
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: cr,
                },
            ],
        });
        enc.send_frame(&frame).unwrap();
    }
    enc.flush().unwrap();
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    (packets, extradata)
}

#[test]
fn vol_signals_vbv_parameters() {
    let (_packets, extradata) = encode(200_000, 1, 0);
    let pos = extradata
        .windows(4)
        .position(|w| w == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&extradata[pos..], 0x03).unwrap();
    let ctrl = vol.vol_control.expect("vol_control_parameters present");
    assert!(ctrl.low_delay, "no B-VOPs → low_delay = 1");
    let vbv = ctrl.vbv.expect("vbv_parameters present");
    assert_eq!(vbv.bit_rate, 200_000u32.div_ceil(400));
    assert_eq!(vbv.vbv_buffer_size, (200_000u32 * 2).div_ceil(16384));
    assert_eq!(vbv.vbv_occupancy, 170 * vbv.vbv_buffer_size);
}

#[test]
fn measured_rate_lands_near_target_and_vbv_never_underflows() {
    let frames = 50usize;
    for bitrate in [150_000u32, 400_000] {
        let (packets, _extradata) = encode(bitrate, frames, 0);
        assert_eq!(packets.len(), frames);

        // Measured rate over the whole run (25 fps → 2 s of video).
        let total_bits: u64 = packets.iter().map(|p| p.len() as u64 * 8).sum();
        let duration = frames as f64 / 25.0;
        let measured = total_bits as f64 / duration;
        let ratio = measured / f64::from(bitrate);
        assert!(
            (0.6..=1.1).contains(&ratio),
            "measured {measured:.0} b/s vs target {bitrate} (ratio {ratio:.2})"
        );

        // Independent Annex D re-simulation from the emitted packet
        // sizes: item 8 recurrence, item 9 constraints. The first
        // packet carries the configuration run, matching d_0's
        // definition (item 5); the initial occupancy is the signalled
        // default 170 × vbv_buffer_size.
        let buffer_units = (u64::from(bitrate) * 2).div_ceil(16384);
        let b_cap = 16384.0 * buffer_units as f64;
        let occupancy0 = 64.0 * (170.0 * buffer_units as f64);
        let refill = f64::from(bitrate) / 25.0;
        // The config bits are inside packet 0, and item 8's b_0 seed
        // adds them to the occupancy before removing d_0 — so seeding
        // with `occupancy0 + config_bits` and removing whole packets
        // reproduces the model; the config bits are part of packet 0
        // here, hence occupancy0 must be measured against d_0 which
        // includes them. Extract the config length from the packet
        // stream: it precedes the first VOP start code prefix 0x1B6.
        let first = &packets[0];
        let config_len = first
            .windows(4)
            .position(|w| w == [0, 0, 1, 0xB6])
            .expect("first packet contains a VOP");
        let mut buf = occupancy0 + config_len as f64 * 8.0;
        for (i, p) in packets.iter().enumerate() {
            let d = p.len() as f64 * 8.0;
            assert!(d < b_cap, "packet {i}: d_i must stay below B");
            assert!(
                d <= buf + 1e-9,
                "packet {i}: VBV underflow (d = {d}, occupancy = {buf:.1})"
            );
            buf = (buf - d + refill).min(b_cap);
            assert!(buf >= 0.0);
        }
    }
}

#[test]
fn higher_bitrate_spends_more_bits() {
    let frames = 30usize;
    let sizes: Vec<u64> = [120_000u32, 300_000, 700_000]
        .iter()
        .map(|&r| {
            let (packets, _) = encode(r, frames, 0);
            packets.iter().map(|p| p.len() as u64).sum()
        })
        .collect();
    assert!(
        sizes[0] < sizes[1] && sizes[1] < sizes[2],
        "stream sizes must grow with the bitrate target: {sizes:?}"
    );
}

#[test]
fn rate_controlled_stream_decodes_and_is_deterministic() {
    let (packets_a, _) = encode(250_000, 20, 2);
    let (packets_b, _) = encode(250_000, 20, 2);
    assert_eq!(packets_a, packets_b, "rate control must be deterministic");
    let stream: Vec<u8> = packets_a.into_iter().flatten().collect();
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("rate-controlled stream decodes");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), 20);
    // Display order intact under bf 2 + rate control.
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.pts_ticks(), Some(k as i64));
    }
}

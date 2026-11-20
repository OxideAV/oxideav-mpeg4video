//! Encoder round trip under fuzzer-chosen tool sets: the bytes pick the
//! picture size, the rate-control / resilience / motion / interlace
//! options and the content of two to four tiny frames; the registry
//! encoder must produce a stream the crate's own decoder reads back
//! into exactly as many frames (every writer path — budget-driven
//! dquant / dbquant, video packets, data partitioning incl. S(GMC),
//! RVLC, interlaced tools, short header, static-sprite VOLs with
//! fitted trajectories — is exercised).
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::Encoder as _;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 || data.len() > 4096 {
        return;
    }
    let b = data;
    let short_header = b[0] & 0x80 != 0;
    let (w, h) = if short_header {
        (128usize, 96usize)
    } else {
        (
            16 * (1 + usize::from(b[0] & 3)),
            16 * (1 + usize::from((b[0] >> 2) & 1)),
        )
    };
    let mut opts = oxideav_core::CodecOptions::default()
        .set("qp", (1 + u32::from(b[1] & 31)).to_string())
        .set("gop-size", (1 + u32::from(b[2] & 7)).to_string());
    let bitrate = match b[3] & 3 {
        0 => 0u32,
        1 => 30_000,
        2 => 200_000,
        _ => 1_500_000,
    };
    if bitrate > 0 {
        opts = opts
            .set("bitrate", bitrate.to_string())
            .set("rc-mode", if b[3] & 4 != 0 { "vop" } else { "budget" })
            .set("rc-band", (u32::from(b[3] >> 3) & 15).to_string());
    }
    if b[4] & 1 != 0 {
        opts = opts.set("mb-aq", "true");
    }
    // A basic static-sprite VOL (§7.8) takes none of the motion /
    // rate / resilience tools below.
    let static_sprite = !short_header && b[0] & 0x40 != 0;
    // N-bit (§6.3.3 not_8_bit): its object type has neither sprites
    // nor the short header.
    let n_bit = !short_header && !static_sprite && b[3] & 0x80 != 0;
    let depth = [4u32, 5, 6, 7, 9, 10, 11, 12][usize::from(b[1] >> 5)];
    let precision = 3 + (u32::from(b[2] >> 3) % 7);
    let (format, format_bits) = match (n_bit, depth) {
        (false, _) => (oxideav_core::PixelFormat::Yuv420P, 8),
        (true, 0..=8) => (oxideav_core::PixelFormat::Yuv420P, 8),
        (true, 9 | 10) => (oxideav_core::PixelFormat::Yuv420P10Le, 10),
        (true, _) => (oxideav_core::PixelFormat::Yuv420P12Le, 12),
    };
    if static_sprite {
        opts = oxideav_core::CodecOptions::default()
            .set("qp", (1 + u32::from(b[1] & 31)).to_string())
            .set("static-sprite", "true")
            .set("sprite-points", (u32::from(b[2]) % 5).to_string());
        if b[7] & 4 != 0 {
            opts = opts.set("mpeg-quant", "true");
        }
        opts = opts.set("dc-vlc-thr", (u32::from(b[7] >> 4) & 7).to_string());
    } else if short_header {
        opts = opts.set("short-header", "true");
        if b[4] & 2 != 0 {
            opts = opts.set("gob-headers", "false");
        }
    } else {
        if n_bit {
            // An N-bit VOL: the depth, the quantiser precision and a
            // quantiser inside the range that precision can signal.
            let max_qp = (1u32 << precision) - 1;
            opts = opts
                .set("bits-per-pixel", depth.to_string())
                .set("quant-precision", precision.to_string())
                .set("qp", (1 + u32::from(b[1] & 31) % max_qp).to_string());
        }
        let dp = b[4] & 2 != 0;
        let interlaced = b[4] & 4 != 0 && !dp && !n_bit;
        if dp {
            opts = opts.set("data-partitioned", "true");
            if b[4] & 8 != 0 {
                opts = opts.set("rvlc", "true");
            }
        }
        if b[4] & 16 != 0 {
            opts = opts.set("packet-bits", (100 + 50 * u32::from(b[5] & 15)).to_string());
        }
        if b[4] & 32 != 0 && !n_bit {
            opts = opts
                .set("gmc", "true")
                .set("gmc-points", (1 + u32::from(b[5] >> 4) % 3).to_string());
        }
        if b[4] & 64 != 0 {
            opts = opts.set("bf", (1 + u32::from(b[6] & 1)).to_string());
        }
        if b[4] & 128 != 0 && !n_bit {
            opts = opts.set("qpel", "true");
        }
        if b[6] & 2 != 0 {
            opts = opts.set("four-mv", "true");
        }
        if b[6] & 4 != 0 {
            opts = opts.set("fcode", (1 + u32::from(b[6] >> 3) % 3).to_string());
        }
        if interlaced {
            opts = opts.set("interlaced", "true");
            if b[7] & 1 != 0 {
                opts = opts.set("alt-scan", "true");
            }
            if b[7] & 2 != 0 {
                opts = opts.set("ecosystem-compat", "true");
            }
        }
        if b[7] & 4 != 0 {
            opts = opts.set("mpeg-quant", "true");
        }
        if b[7] & 8 != 0 {
            opts = opts.set("auto-dc-vlc", "true");
        } else {
            opts = opts.set("dc-vlc-thr", (u32::from(b[7] >> 4) & 7).to_string());
        }
    }
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(format);
    params.options = opts;
    let Ok(mut enc) = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params) else {
        return;
    };
    let frames = 2 + usize::from(b[7] >> 6) % 3;
    let payload = &b[8..];
    let (cw, ch) = (w / 2, h / 2);
    for k in 0..frames {
        let sample = |i: usize| -> u8 {
            if payload.is_empty() {
                128
            } else {
                payload[(i * 7 + k * 13) % payload.len()]
                    .wrapping_add(((i / w) as u8).wrapping_mul(k as u8))
            }
        };
        // One plane in the input format: bytes, or 16-bit
        // little-endian words spanning the format's whole range.
        let plane = |len: usize, offset: usize, width: usize| -> oxideav_core::VideoPlane {
            if format_bits > 8 {
                let data = (0..len)
                    .flat_map(|i| {
                        let hi = u16::from(sample(i + offset));
                        let lo = u16::from(sample(i + offset + 3));
                        (((hi << 8) | lo) >> (16 - format_bits)).to_le_bytes()
                    })
                    .collect();
                oxideav_core::VideoPlane {
                    stride: 2 * width,
                    data,
                }
            } else {
                oxideav_core::VideoPlane {
                    stride: width,
                    data: (0..len).map(|i| sample(i + offset)).collect(),
                }
            }
        };
        let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
            pts: None,
            planes: vec![plane(w * h, 0, w), plane(cw * ch, 1, cw), plane(cw * ch, 2, cw)],
        });
        enc.send_frame(&frame).expect("frame accepted");
    }
    enc.flush().expect("flush");
    let mut stream = Vec::new();
    let mut packets = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                stream.extend_from_slice(&p.data);
                packets += 1;
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert_eq!(packets, frames);
    let mut dec = Mpeg4VideoDecoder::new();
    let mut out = dec.decode(&stream).expect("own stream decodes");
    out.extend(dec.flush());
    assert_eq!(out.len(), frames, "decoded frame count");
});

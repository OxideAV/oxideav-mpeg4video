//! N-bit video (§6.3.3 `not_8_bit`, `bits_per_pixel` 4..=12,
//! `quant_precision` 3..=9): I/P/B self-encode at every sample depth →
//! decode through the crate's own stream decoder → sample-exact
//! agreement with the encoder's closed-loop reconstructions, the
//! fidelity the depth allows, and the 16-bit output planes.

use oxideav_mpeg4video::bvop_encode::encode_b_vop_any;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    EncoderConfig, FrameView, encode_i_vop_any, write_configuration_headers,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop_any, reconstruct_own_p_vop_with_motion};
use oxideav_mpeg4video::sample::Sample;
use oxideav_mpeg4video::vol::parse_video_object_layer;

/// A translating textured scene at `bpp` bits per sample (the full
/// code range is used), as 16-bit working planes.
fn picture(w: usize, h: usize, frame_index: usize, bpp: u32) -> [Vec<u16>; 3] {
    let (cw, ch) = (w / 2, h / 2);
    let max = (1i64 << bpp) - 1;
    let bg = |x: i64, y: i64| -> u16 {
        // A smooth ramp plus texture, spanning the whole range.
        let ramp = (x * 3 + y * 2).rem_euclid(97) * max / 96;
        let tex = ((x * 7 + y * 5) % 23) * max / 64;
        ((ramp * 3 + tex) / 4).clamp(0, max) as u16
    };
    let (ox, oy) = ((frame_index * 2) as i64, frame_index as i64);
    let mut y = vec![0u16; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = bg(col as i64 + ox, row as i64 + oy);
        }
    }
    for row in 0..8.min(h) {
        for col in 0..8.min(w) {
            y[row * w + col] = max as u16;
        }
    }
    let mut cb = vec![0u16; cw * ch];
    let mut cr = vec![0u16; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] =
                bg(col as i64 + ox / 2, row as i64 + oy / 2) / 2 + (max / 4) as u16;
            cr[row * cw + col] = (max as u16).div_ceil(2);
        }
    }
    [y, cb, cr]
}

fn store_planes<S: Sample>(planes: &[Vec<u16>; 3]) -> [Vec<S>; 3] {
    planes
        .clone()
        .map(|p| p.iter().map(|&v| S::narrow(i32::from(v))).collect())
}

fn decode_full(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("own stream must decode");
    frames.extend(dec.flush());
    frames
}

/// The three planes of a decoded frame as 16-bit values, whatever the
/// store.
fn planes_of(frame: &DecodedFrame) -> [Vec<u16>; 3] {
    match (
        frame.luma_samples16(),
        frame.cb_samples16(),
        frame.cr_samples16(),
    ) {
        (Some(y), Some(cb), Some(cr)) => [y.to_vec(), cb.to_vec(), cr.to_vec()],
        _ => [
            frame.luma_samples().iter().map(|&v| u16::from(v)).collect(),
            frame.cb_samples().iter().map(|&v| u16::from(v)).collect(),
            frame.cr_samples().iter().map(|&v| u16::from(v)).collect(),
        ],
    }
}

fn psnr(a: &[u16], b: &[u16], bpp: u32) -> f64 {
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let peak = f64::from((1u32 << bpp) - 1);
    10.0 * (peak * peak * a.len() as f64 / sse).log10()
}

/// Build an I B P coded stream (display I, B, P); returns the stream
/// and the display-order closed-loop reconstructions.
fn build_ibp<S: Sample>(cfg: &EncoderConfig, qp: u32) -> (Vec<u8>, Vec<DecodedFrame>) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let bpp = u32::from(cfg.bits_per_pixel);
    let headers = write_configuration_headers(cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    assert_eq!(vol.bits_per_pixel, cfg.bits_per_pixel);
    assert_eq!(vol.quant_precision, cfg.quant_precision);

    let pics: Vec<[Vec<S>; 3]> = (0..3)
        .map(|k| store_planes::<S>(&picture(w, h, k, bpp)))
        .collect();
    let view = |k: usize| FrameView {
        y: &pics[k][0],
        cb: &pics[k][1],
        cr: &pics[k][2],
        width: w,
        height: h,
    };

    let mut store = FrameStore::new();
    let mut stream = headers;

    let (i_unit, i_recon) = encode_i_vop_any(&vol, cfg, &view(0), 0, 0, qp);
    store.push_anchor(i_recon.clone());
    stream.extend_from_slice(&i_unit);

    let reference = store.backward().unwrap().clone();
    let (p_unit, _stats) = encode_p_vop_any(&vol, cfg, &view(2), &reference, 0, 2, qp);
    let (p_recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);

    let (b_unit, b_recon, _b_stats) =
        encode_b_vop_any(&vol, cfg, &view(1), &store, Some(&motion), 1, 2, 0, 1, qp);
    stream.extend_from_slice(&b_unit);

    (stream, vec![i_recon, b_recon, p_recon])
}

/// Encode + decode one configuration; checks closed-loop agreement
/// and returns the worst luma PSNR against the source.
fn round_trip(cfg: &EncoderConfig, qp: u32) -> f64 {
    let bpp = u32::from(cfg.bits_per_pixel);
    let (stream, recons) = if bpp > 8 {
        build_ibp::<u16>(cfg, qp)
    } else {
        build_ibp::<u8>(cfg, qp)
    };
    let decoded = decode_full(&stream);
    assert_eq!(decoded.len(), recons.len(), "frame count at {bpp} bits");
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let mut worst = f64::INFINITY;
    for (k, (d, r)) in decoded.iter().zip(recons.iter()).enumerate() {
        assert_eq!(d.bits_per_pixel(), bpp, "frame {k} depth");
        assert_eq!(d.is_wide(), bpp > 8, "frame {k} store");
        assert_eq!(planes_of(d), planes_of(r), "frame {k} at {bpp} bits");
        let max = (1u16 << bpp) - 1;
        let got = planes_of(d);
        assert!(
            got.iter().flatten().all(|&v| v <= max),
            "frame {k}: a sample exceeds the {bpp}-bit range"
        );
        let src = picture(w, h, k, bpp);
        // Compare the visible area (the stores are macroblock-padded).
        let stride = d.width();
        let mut vis = Vec::with_capacity(w * h);
        for row in 0..h {
            vis.extend_from_slice(&got[0][row * stride..row * stride + w]);
        }
        worst = worst.min(psnr(&vis, &src[0], bpp));
    }
    worst
}

fn n_bit_cfg(bpp: u8, precision: u8) -> EncoderConfig {
    EncoderConfig::default()
        .with_width(64)
        .with_height(48)
        .with_b_vops(true)
        .with_bit_depth(bpp, precision)
}

#[test]
fn every_depth_round_trips_sample_exact() {
    for bpp in 4..=12u8 {
        if bpp == 8 {
            continue;
        }
        let cfg = n_bit_cfg(bpp, 5);
        assert_eq!(cfg.profile_and_level(), 0x42, "N-bit profile");
        assert_eq!(cfg.video_object_type(), 0x05, "N-bit object type");
        // A quantiser of about 1/64 of the range step keeps the
        // comparison meaningful at every depth.
        let qp = (1u32 << bpp.saturating_sub(6)).clamp(1, 31);
        let worst = round_trip(&cfg, qp);
        // The same relative quantiser lands near 36.8 dB at every
        // depth it can be expressed at; the shallow depths bottom out
        // on their own step (about 6 dB per bit).
        let floor = (6.0 * f64::from(bpp)).min(35.0);
        assert!(
            worst > floor,
            "{bpp} bits at qp {qp}: luma PSNR {worst:.2} dB"
        );
    }
}

#[test]
fn fine_quantiser_is_near_lossless_at_ten_bits() {
    let worst = round_trip(&n_bit_cfg(10, 5), 1);
    assert!(worst > 55.0, "10 bits at qp 1: luma PSNR {worst:.2} dB");
}

#[test]
fn every_quant_precision_carries_its_quantiser_range() {
    for precision in 3..=9u8 {
        let cfg = n_bit_cfg(12, precision);
        let max_qp = cfg.max_quantiser();
        assert_eq!(max_qp, (1 << precision) - 1);
        // The largest quantiser the precision can signal, and one in
        // the middle of its range.
        for qp in [max_qp, max_qp.div_ceil(2)] {
            let worst = round_trip(&cfg, qp);
            assert!(worst.is_finite() && worst > 12.0, "qp {qp}: {worst:.2} dB");
        }
    }
}

#[test]
fn tool_combinations_round_trip_at_ten_and_twelve_bits() {
    for bpp in [10u8, 12] {
        for (quant_type, four_mv, ac_pred) in [
            (true, false, true),
            (false, true, false),
            (true, true, true),
        ] {
            let cfg = n_bit_cfg(bpp, 7)
                .with_quant_type(quant_type)
                .with_four_mv(four_mv)
                .with_ac_prediction(ac_pred)
                .with_fcode(2);
            let worst = round_trip(&cfg, 24);
            assert!(
                worst > 30.0,
                "{bpp} bits {quant_type}/{four_mv}/{ac_pred}: {worst:.2}"
            );
        }
    }
}

#[test]
fn adaptive_quantisation_stays_inside_a_short_quantiser_range() {
    // quant_precision 3: the running quantiser saturates at 7, and
    // the dquant / dbquant plan has to follow the decoder's clip.
    for precision in [3u8, 4, 9] {
        let cfg = n_bit_cfg(10, precision).with_adaptive_quant(true);
        let qp = cfg.max_quantiser().min(40);
        let worst = round_trip(&cfg, qp);
        assert!(worst.is_finite(), "precision {precision}");
    }
}

// ---------------------------------------------------------------
// Registry path: framework frames in, packets out, frames back.
// ---------------------------------------------------------------

/// The picture of [`picture`] as a framework frame in `format`
/// (samples scaled to the format's own depth).
fn framework_frame(
    w: usize,
    h: usize,
    k: usize,
    bpp: u32,
    format_bits: u32,
) -> oxideav_core::Frame {
    let planes = picture(w, h, k, bpp);
    let shift = format_bits - bpp;
    let pack = |p: &Vec<u16>, pw: usize| -> oxideav_core::VideoPlane {
        if format_bits > 8 {
            oxideav_core::VideoPlane {
                stride: 2 * pw,
                data: p.iter().flat_map(|&v| (v << shift).to_le_bytes()).collect(),
            }
        } else {
            oxideav_core::VideoPlane {
                stride: pw,
                data: p.iter().map(|&v| (v << shift) as u8).collect(),
            }
        }
    };
    oxideav_core::Frame::Video(oxideav_core::VideoFrame {
        pts: None,
        planes: vec![
            pack(&planes[0], w),
            pack(&planes[1], w / 2),
            pack(&planes[2], w / 2),
        ],
    })
}

/// Encode `frames` pictures through the registry encoder and decode
/// the packets through the registry decoder; returns the decoded
/// frames' luma planes brought back to `bpp` bits, and the decoder's
/// stated pixel format.
fn registry_round_trip(
    format: oxideav_core::PixelFormat,
    format_bits: u32,
    bpp: u32,
    frames: usize,
    extra: &[(&str, &str)],
) -> (Vec<Vec<u16>>, oxideav_core::PixelFormat, usize) {
    let (w, h) = (64usize, 48usize);
    let id = oxideav_core::CodecId::new("mpeg4video");
    let mut params = oxideav_core::CodecParameters::video(id.clone());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(format);
    params.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    let mut opts = oxideav_core::CodecOptions::default();
    for (k, v) in extra {
        opts = opts.set(*k, v.to_string());
    }
    params.options = opts;
    let mut enc = oxideav_mpeg4video::encoder::make_encoder(&params).unwrap();
    assert_eq!(enc.output_params().pixel_format, Some(format));
    for k in 0..frames {
        enc.send_frame(&framework_frame(w, h, k, bpp, format_bits))
            .unwrap();
    }
    enc.flush().unwrap();

    let mut dparams = oxideav_core::CodecParameters::video(id);
    dparams.extradata = enc.output_params().extradata.clone();
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dparams).unwrap();
    let stated = dec.output_pixel_format().unwrap();
    let mut out = Vec::new();
    let mut bytes = 0usize;
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<Vec<u16>>| loop {
        match dec.receive_frame() {
            Ok(oxideav_core::Frame::Video(v)) => {
                let luma = &v.planes[0];
                let shift = format_bits - bpp;
                let mut plane = Vec::with_capacity(w * h);
                for row in 0..h {
                    for col in 0..w {
                        let s = if format_bits > 8 {
                            let o = row * luma.stride + 2 * col;
                            u16::from_le_bytes([luma.data[o], luma.data[o + 1]])
                        } else {
                            u16::from(luma.data[row * luma.stride + col])
                        };
                        // The scaling is an exact shift both ways.
                        assert_eq!(s & ((1 << shift) - 1), 0, "low bits of a scaled sample");
                        plane.push(s >> shift);
                    }
                }
                out.push(plane);
            }
            Ok(_) => panic!("video decoder produced a non-video frame"),
            Err(oxideav_core::Error::NeedMore | oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    };
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                bytes += p.data.len();
                dec.send_packet(&p).unwrap();
                drain(&mut dec, &mut out);
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    dec.flush().unwrap();
    drain(&mut dec, &mut out);
    (out, stated, bytes)
}

#[test]
fn registry_codes_ten_and_twelve_bit_formats() {
    for (format, format_bits) in [
        (oxideav_core::PixelFormat::Yuv420P10Le, 10u32),
        (oxideav_core::PixelFormat::Yuv420P12Le, 12),
    ] {
        let qp = (1u32 << (format_bits - 6)).to_string();
        let (frames, stated, _) = registry_round_trip(
            format,
            format_bits,
            format_bits,
            7,
            &[("bf", "2"), ("quant-precision", "7"), ("qp", &qp)],
        );
        assert_eq!(stated, format, "decoder labels the frames from the VOL");
        assert_eq!(frames.len(), 7);
        for (k, got) in frames.iter().enumerate() {
            let src = picture(64, 48, k, format_bits);
            let p = psnr(got, &src[0], format_bits);
            assert!(p > 34.0, "{format_bits}-bit frame {k}: {p:.2} dB");
        }
    }
}

#[test]
fn registry_codes_odd_depths_inside_the_containing_format() {
    for (format, format_bits, bpp) in [
        (oxideav_core::PixelFormat::Yuv420P, 8u32, 6u32),
        (oxideav_core::PixelFormat::Yuv420P10Le, 10, 9),
        (oxideav_core::PixelFormat::Yuv420P12Le, 12, 11),
    ] {
        let depth = bpp.to_string();
        let (frames, stated, _) = registry_round_trip(
            format,
            format_bits,
            bpp,
            4,
            &[("bits-per-pixel", &depth), ("qp", "2")],
        );
        assert_eq!(stated, format);
        assert_eq!(frames.len(), 4);
        for (k, got) in frames.iter().enumerate() {
            let src = picture(64, 48, k, bpp);
            let p = psnr(got, &src[0], bpp);
            assert!(p > 30.0, "{bpp}-bit frame {k}: {p:.2} dB");
        }
    }
}

#[test]
fn registry_rate_control_reaches_coarse_quantisers_at_twelve_bits() {
    // 12-bit content needs quantisers far above 31 to reach a low
    // rate; quant-precision 9 opens the range to 511.
    let run = |bitrate: &str| {
        registry_round_trip(
            oxideav_core::PixelFormat::Yuv420P12Le,
            12,
            12,
            25,
            &[
                ("bitrate", bitrate),
                ("quant-precision", "9"),
                ("qp", "64"),
                ("gop-size", "12"),
            ],
        )
    };
    let (frames_lo, _, bytes_lo) = run("60000");
    let (frames_hi, _, bytes_hi) = run("240000");
    assert_eq!(frames_lo.len(), 25);
    assert_eq!(frames_hi.len(), 25);
    for (bytes, target) in [(bytes_lo, 60_000.0), (bytes_hi, 240_000.0)] {
        let rate = bytes as f64 * 8.0; // 25 frames at 25 fps = one second
        let ratio = rate / target;
        assert!(
            (0.7..1.3).contains(&ratio),
            "measured {rate:.0} b/s against {target:.0}"
        );
    }
    assert!(bytes_hi > 2 * bytes_lo);
}

#[test]
fn registry_refuses_tools_outside_the_n_bit_object_type() {
    let id = oxideav_core::CodecId::new("mpeg4video");
    for tool in ["qpel", "gmc", "interlaced", "static-sprite"] {
        let mut params = oxideav_core::CodecParameters::video(id.clone());
        params.width = Some(64);
        params.height = Some(48);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P10Le);
        params.options = oxideav_core::CodecOptions::default().set(tool, "true");
        assert!(
            oxideav_mpeg4video::encoder::make_encoder(&params).is_err(),
            "{tool} at 10 bits"
        );
    }
    // quant-precision belongs to the N-bit VOL; qp is bounded by it.
    let mut params = oxideav_core::CodecParameters::video(id.clone());
    params.width = Some(64);
    params.height = Some(48);
    params.options = oxideav_core::CodecOptions::default().set("quant-precision", "7");
    assert!(oxideav_mpeg4video::encoder::make_encoder(&params).is_err());
    params.options = oxideav_core::CodecOptions::default().set("qp", "40");
    assert!(oxideav_mpeg4video::encoder::make_encoder(&params).is_err());
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P10Le);
    params.options = oxideav_core::CodecOptions::default()
        .set("qp", "40")
        .set("quant-precision", "6");
    assert!(oxideav_mpeg4video::encoder::make_encoder(&params).is_ok());
}

// ---------------------------------------------------------------
// The motion-compensation primitives over the two sample stores.
// ---------------------------------------------------------------

use oxideav_mpeg4video::half_sample::{ReferenceVop, interpolate_block};
use oxideav_mpeg4video::quarter_sample::{
    interpolate_block_qpel_any, interpolate_block_qpel_field_any,
};

fn noise_plane(w: usize, h: usize, max: u32, seed: u32) -> Vec<u16> {
    let mut state = seed;
    (0..w * h)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) % (max + 1)) as u16
        })
        .collect()
}

#[test]
fn both_stores_interpolate_identically_on_eight_bit_content() {
    let (w, h) = (40usize, 36usize);
    let wide = noise_plane(w, h, 255, 7);
    let narrow: Vec<u8> = wide.iter().map(|&v| v as u8).collect();
    let rn = ReferenceVop::new(&narrow[..], w, h).unwrap();
    let rw = ReferenceVop::new(&wide[..], w, h).unwrap();
    let widen = |v: Vec<u8>| -> Vec<u16> { v.into_iter().map(u16::from).collect() };
    for rc in 0..2u8 {
        for (mvx, mvy) in [
            (0, 0),
            (1, 0),
            (0, 1),
            (3, 5),
            (-7, 2),
            (-90, -90),
            (90, 90),
        ] {
            for (ox, oy) in [(8, 8), (0, 0), (24, 20)] {
                assert_eq!(
                    widen(interpolate_block(&rn, mvx, mvy, ox, oy, 16, 16, rc)),
                    interpolate_block(&rw, mvx, mvy, ox, oy, 16, 16, rc),
                    "half-sample mv ({mvx}, {mvy}) at ({ox}, {oy}) rc {rc}"
                );
                for (bw, bh) in [(16usize, 16usize), (8, 8)] {
                    assert_eq!(
                        widen(interpolate_block_qpel_any(
                            &rn, mvx, mvy, ox, oy, bw, bh, rc, 8
                        )),
                        interpolate_block_qpel_any(&rw, mvx, mvy, ox, oy, bw, bh, rc, 8),
                        "quarter-sample mv ({mvx}, {mvy}) at ({ox}, {oy}) rc {rc}"
                    );
                }
                for field in 0..2 {
                    assert_eq!(
                        widen(interpolate_block_qpel_field_any(
                            &rn,
                            mvx,
                            2 * mvy,
                            ox,
                            oy,
                            16,
                            8,
                            field,
                            rc,
                            8
                        )),
                        interpolate_block_qpel_field_any(
                            &rw,
                            mvx,
                            2 * mvy,
                            ox,
                            oy,
                            16,
                            8,
                            field,
                            rc,
                            8
                        ),
                        "field quarter-sample mv ({mvx}, {mvy}) field {field}"
                    );
                }
            }
        }
    }
}

#[test]
fn quarter_sample_filter_clips_to_the_sample_depth() {
    // §7.6.2.2: each filter stage is clipped to [0, 2^bpp - 1]. A
    // checkerboard of extremes overshoots both ways.
    for bpp in [9u32, 10, 12] {
        let max = (1u16 << bpp) - 1;
        let (w, h) = (32usize, 32usize);
        let plane: Vec<u16> = (0..w * h)
            .map(|i| {
                if (i % w / 2 + i / w / 2) % 2 == 0 {
                    max
                } else {
                    0
                }
            })
            .collect();
        let r = ReferenceVop::new(&plane[..], w, h).unwrap();
        let mut seen_max = false;
        let mut seen_zero = false;
        for (mvx, mvy) in [(2, 0), (0, 2), (2, 2), (1, 3), (3, 1)] {
            let out = interpolate_block_qpel_any(&r, mvx, mvy, 8, 8, 16, 16, 0, bpp);
            assert!(
                out.iter().all(|&v| v <= max),
                "{bpp} bits mv ({mvx}, {mvy})"
            );
            seen_max |= out.contains(&max);
            seen_zero |= out.contains(&0);
        }
        assert!(
            seen_max && seen_zero,
            "{bpp} bits: both clip bounds are reached"
        );
    }
}

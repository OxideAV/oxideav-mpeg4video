//! Encoder robustness sweep: adversarial content (full-range noise,
//! saturation extremes, checkerboards that maximise high-frequency
//! coefficients) across sizes, quantisers and both quantisation
//! methods, in I+P GOPs — every stream must decode through the
//! crate's own decoder **sample-exactly** against the encoder's
//! closed-loop reconstructions.

use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::{DecodedFrame, FrameStore};
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::pvop_encode::{encode_p_vop, reconstruct_own_p_vop};
use oxideav_mpeg4video::vol::parse_video_object_layer;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Adversarial picture kinds.
enum Kind {
    /// Full-range LCG noise — dense AC spectra, exercises escapes.
    Noise,
    /// 1×1 checkerboard of 0/255 — maximal highest-frequency energy.
    Checkerboard,
    /// Flat extremes — saturated DC, zero AC.
    Flat(u8),
    /// Vertical hard edges every 4 samples.
    Bars,
}

fn synthesise(kind: &Kind, w: usize, h: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut state = seed;
    let mut plane = |pw: usize, ph: usize, chroma: bool| -> Vec<u8> {
        let mut out = vec![0u8; pw * ph];
        for row in 0..ph {
            for col in 0..pw {
                out[row * pw + col] = match kind {
                    Kind::Noise => (lcg(&mut state) >> 24) as u8,
                    Kind::Checkerboard => {
                        if (row + col) % 2 == 0 {
                            255
                        } else {
                            0
                        }
                    }
                    Kind::Flat(v) => {
                        if chroma {
                            128
                        } else {
                            *v
                        }
                    }
                    Kind::Bars => {
                        if (col / 4) % 2 == 0 {
                            255
                        } else {
                            0
                        }
                    }
                };
            }
        }
        out
    };
    (plane(w, h, false), plane(cw, ch, true), plane(cw, ch, true))
}

/// Encode a 4-frame I P I P stream and assert exact self-decode.
fn roundtrip(kind: Kind, w: u16, h: u16, qp: u32, quant_type: bool) {
    let cfg = EncoderConfig {
        width: w,
        height: h,
        quant_type,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let mut stream = headers;
    let mut recons: Vec<DecodedFrame> = Vec::new();
    let mut store = FrameStore::new();
    for k in 0..4usize {
        let (y, cb, cr) = synthesise(
            &kind,
            usize::from(w),
            usize::from(h),
            0xC0FF_EE00 ^ k as u32,
        );
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: usize::from(w),
            height: usize::from(h),
        };
        if k % 2 == 0 {
            let (unit, recon) = encode_i_vop(&vol, &cfg, &view, 0, k as u16, qp);
            store.push_anchor(recon.clone());
            stream.extend_from_slice(&unit);
            recons.push(recon);
        } else {
            let reference = store.backward().expect("anchor present").clone();
            let (unit, _stats) = encode_p_vop(&vol, &cfg, &view, &reference, 0, k as u16, qp);
            let recon = reconstruct_own_p_vop(&vol, &unit, &mut store);
            stream.extend_from_slice(&unit);
            recons.push(recon);
        }
    }

    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("own stream must decode");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), recons.len(), "{w}x{h} qp{qp}: frame count");
    for (k, (d, r)) in frames.iter().zip(recons.iter()).enumerate() {
        assert_eq!(
            d.luma_samples(),
            r.luma_samples(),
            "{w}x{h} qp{qp} m{}: frame {k} luma",
            if quant_type { 1 } else { 2 }
        );
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

#[test]
fn noise_survives_extreme_quantisers_method2() {
    for qp in [1u32, 13, 31] {
        roundtrip(Kind::Noise, 32, 32, qp, false);
    }
}

#[test]
fn noise_survives_extreme_quantisers_method1() {
    for qp in [1u32, 13, 31] {
        roundtrip(Kind::Noise, 32, 32, qp, true);
    }
}

#[test]
fn checkerboard_maximal_high_frequency() {
    roundtrip(Kind::Checkerboard, 48, 32, 1, false);
    roundtrip(Kind::Checkerboard, 48, 32, 1, true);
    roundtrip(Kind::Checkerboard, 48, 32, 31, false);
}

#[test]
fn flat_extremes_saturate_dc() {
    roundtrip(Kind::Flat(0), 32, 32, 1, false);
    roundtrip(Kind::Flat(255), 32, 32, 1, false);
    roundtrip(Kind::Flat(255), 32, 32, 31, true);
}

#[test]
fn hard_edges_and_odd_grid_sizes() {
    // Partial edge macroblocks in both dimensions.
    roundtrip(Kind::Bars, 40, 24, 2, false);
    roundtrip(Kind::Bars, 24, 40, 2, true);
    roundtrip(Kind::Noise, 17, 17, 7, false);
}

#[test]
fn minimum_size_single_macroblock() {
    roundtrip(Kind::Noise, 16, 16, 4, false);
    roundtrip(Kind::Checkerboard, 16, 16, 31, true);
}

/// I/B/P round-trip with the full round-443 tool set (inter4v +
/// quarter-sample + B-VOPs) on adversarial content — every stream
/// self-decodes sample-exact against the closed-loop reconstructions.
fn roundtrip_ibp(kind: Kind, w: u16, h: u16, qp: u32, quant_type: bool) {
    use oxideav_mpeg4video::bvop_encode::encode_b_vop;
    use oxideav_mpeg4video::pvop_encode::reconstruct_own_p_vop_with_motion;
    let cfg = EncoderConfig {
        width: w,
        height: h,
        quant_type,
        four_mv: true,
        quarter_sample: true,
        b_vops: true,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .expect("VOL start code");
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();

    let pics: Vec<_> = (0..3)
        .map(|k| {
            synthesise(
                &kind,
                usize::from(w),
                usize::from(h),
                0xBEEF_0000 ^ k as u32,
            )
        })
        .collect();
    let view = |k: usize| FrameView {
        y: &pics[k].0,
        cb: &pics[k].1,
        cr: &pics[k].2,
        width: usize::from(w),
        height: usize::from(h),
    };

    let mut store = FrameStore::new();
    let mut stream = headers;
    let (i_unit, i_recon) = encode_i_vop(&vol, &cfg, &view(0), 0, 0, qp);
    store.push_anchor(i_recon.clone());
    stream.extend_from_slice(&i_unit);
    let reference = store.backward().unwrap().clone();
    let (p_unit, _stats) = encode_p_vop(&vol, &cfg, &view(2), &reference, 0, 2, qp);
    let (p_recon, motion) = reconstruct_own_p_vop_with_motion(&vol, &p_unit, &mut store);
    stream.extend_from_slice(&p_unit);
    let (b_unit, b_recon, _bstats) =
        encode_b_vop(&vol, &cfg, &view(1), &store, Some(&motion), 1, 2, 0, 1, qp);
    stream.extend_from_slice(&b_unit);

    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("own stream must decode");
    frames.extend(dec.flush());
    let recons = [i_recon, b_recon, p_recon];
    assert_eq!(frames.len(), recons.len(), "{w}x{h} qp{qp}: frame count");
    for (k, (d, r)) in frames.iter().zip(recons.iter()).enumerate() {
        assert_eq!(
            d.luma_samples(),
            r.luma_samples(),
            "{w}x{h} qp{qp} m{}: frame {k} luma",
            if quant_type { 1 } else { 2 }
        );
        assert_eq!(d.cb_samples(), r.cb_samples(), "frame {k} cb");
        assert_eq!(d.cr_samples(), r.cr_samples(), "frame {k} cr");
    }
}

#[test]
fn full_toolset_ibp_survives_adversarial_content() {
    roundtrip_ibp(Kind::Noise, 32, 32, 1, false);
    roundtrip_ibp(Kind::Noise, 32, 32, 31, true);
    roundtrip_ibp(Kind::Checkerboard, 48, 32, 1, false);
    roundtrip_ibp(Kind::Flat(255), 32, 32, 31, false);
    // Partial-edge and single-macroblock grids.
    roundtrip_ibp(Kind::Bars, 40, 24, 2, true);
    roundtrip_ibp(Kind::Noise, 17, 17, 7, false);
    roundtrip_ibp(Kind::Noise, 16, 16, 13, true);
}

/// Registry-level full tool set (four-mv + qpel + bf + rate control)
/// on noise: decodes to the right frame count in display order and is
/// byte-deterministic.
#[test]
fn registry_full_toolset_with_rate_control() {
    use oxideav_core::Encoder as _;
    let run = || -> Vec<Vec<u8>> {
        let mut params =
            oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
        params.width = Some(48);
        params.height = Some(48);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        params.options = oxideav_core::CodecOptions::default()
            .set("four-mv", "true")
            .set("qpel", "true")
            .set("bf", "2")
            .set("bitrate", "400000");
        let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
        for k in 0..8usize {
            let (y, cb, cr) = synthesise(&Kind::Noise, 48, 48, 0xAB00 ^ k as u32);
            let frame = oxideav_core::Frame::Video(oxideav_core::VideoFrame {
                pts: None,
                planes: vec![
                    oxideav_core::VideoPlane {
                        stride: 48,
                        data: y,
                    },
                    oxideav_core::VideoPlane {
                        stride: 24,
                        data: cb,
                    },
                    oxideav_core::VideoPlane {
                        stride: 24,
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
        packets
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "full tool set must stay deterministic");
    let stream: Vec<u8> = a.into_iter().flatten().collect();
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(&stream).expect("stream decodes");
    frames.extend(dec.flush());
    assert_eq!(frames.len(), 8);
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.pts_ticks(), Some(k as i64), "display order at {k}");
    }
}

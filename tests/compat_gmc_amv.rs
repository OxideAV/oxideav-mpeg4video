//! Compat divergence 3 (§7.8.7.3 GMC averaged MV): the deployed
//! reference decoder derives each non-positive averaged-MV component
//! one MV-grid unit lower than the spec quantisation (zero included:
//! 0 → −1; strictly positive components are exact). The
//! committed `dec_sgmc_*` fixture pairs pin the divergence and prove
//! the opt-in ecosystem-compat mode reproduces the reference decode
//! **bit-exactly** while the default stays spec-literal.
//!
//! Fixture provenance (commands + SHA-256 in `tests/fixtures/NOTES.md`):
//! the streams are built deterministically below; the `.yuv` sides are
//! black-box reference decodes with the floating-point IDCT.

use oxideav_mpeg4video::bitwriter::BitWriter;
use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;
use oxideav_mpeg4video::ivop_encode::{
    encode_i_vop, write_configuration_headers, EncoderConfig, FrameView,
};
use oxideav_mpeg4video::svop_encode::write_s_vop_header;
use oxideav_mpeg4video::vlc_encode::{put_cbpy, put_mcbpc_p, put_motion_vector};
use oxideav_mpeg4video::vol::parse_video_object_layer;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn maybe_write_fixture(name: &str, bytes: &[u8]) -> bool {
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_none() {
        return false;
    }
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
    true
}

/// I-VOP of a diagonal ramp, then a crafted S(GMC)-VOP with a negative
/// trajectory: macroblock 0 is GMC-coded (`mcsel == 1`, empty cbp),
/// macroblock 1 is local with a zero MVD — its reconstructed motion
/// vector *is* macroblock 0's averaged MV (the only valid §7.6.5
/// candidate in row 0), which is where the divergence bites — and the
/// rest are `not_coded` GMC copies.
fn craft_probe_stream(quarter_sample: bool, du: i32, dv: i32) -> Vec<u8> {
    let (w, h) = (64usize, 64usize);
    let cfg = EncoderConfig {
        width: w as u16,
        height: h as u16,
        gmc: true,
        quarter_sample,
        fcode: 2,
        ..EncoderConfig::default()
    };
    let headers = write_configuration_headers(&cfg);
    let pos = headers
        .windows(4)
        .position(|win| win == [0, 0, 1, 0x20])
        .unwrap();
    let vol = parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = ((c * 3 + r * 7) % 251) as u8;
        }
    }
    let cb = vec![128u8; w * h / 4];
    let cr = vec![128u8; w * h / 4];
    let view = FrameView {
        y: &y,
        cb: &cb,
        cr: &cr,
        width: w,
        height: h,
    };
    let (i_unit, _recon) = encode_i_vop(&vol, &cfg, &view, 0, 0, 2);
    let mut stream = headers;
    stream.extend_from_slice(&i_unit);
    let mut bw = BitWriter::new();
    write_s_vop_header(&mut bw, 25, 0, 1, 4, 2, du, dv);
    for i in 0..(w / 16) * (h / 16) {
        if i == 0 {
            bw.write_bit(false); // not_coded = 0
            put_mcbpc_p(&mut bw, 0, 0); // inter, cbpc 0
            bw.write_bit(true); // mcsel = 1
            put_cbpy(&mut bw, 0, false);
        } else if i == 1 {
            bw.write_bit(false);
            put_mcbpc_p(&mut bw, 0, 0);
            bw.write_bit(false); // mcsel = 0
            put_cbpy(&mut bw, 0, false);
            put_motion_vector(&mut bw, 0, 0, 2); // zero MVD
        } else {
            bw.write_bit(true); // not_coded (GMC copy)
        }
    }
    bw.next_start_code();
    stream.extend_from_slice(bw.as_bytes());
    stream
}

/// The 20-pel-per-frame scene the encoder's ±8-pel `fcode == 1` search
/// cannot track — its dominant-motion trajectory goes negative, so
/// GMC averaged MVs feed divergent §7.6.5 medians throughout.
fn negtraj_picture(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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

fn build_negtraj_stream() -> Vec<u8> {
    use oxideav_core::Encoder as _;
    let (w, h) = (96usize, 64usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.options = oxideav_core::CodecOptions::default().set("gmc", "true");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    for k in 0..4usize {
        let (y, cb, cr) = negtraj_picture(k);
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
    stream
}

fn decode(stream: &[u8], opts: DecodeOptions) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::with_options(opts);
    let mut frames = dec.decode(stream).expect("stream must decode");
    frames.extend(dec.flush());
    frames
}

fn assert_matches_reference(stream: &[u8], yuv: &str, w: usize, h: usize) -> usize {
    let reference = fixture(yuv);
    let (cw, ch) = (w / 2, h / 2);
    let frame_len = w * h + 2 * cw * ch;
    let eco = decode(stream, DecodeOptions::ecosystem());
    assert_eq!(reference.len(), eco.len() * frame_len, "{yuv}: frame count");
    for (k, f) in eco.iter().enumerate() {
        let base = k * frame_len;
        assert_eq!(
            f.luma_samples(),
            &reference[base..base + w * h],
            "{yuv}: ecosystem frame {k} luma"
        );
        assert_eq!(
            f.cb_samples(),
            &reference[base + w * h..base + w * h + cw * ch],
            "{yuv}: ecosystem frame {k} cb"
        );
        assert_eq!(
            f.cr_samples(),
            &reference[base + w * h + cw * ch..base + frame_len],
            "{yuv}: ecosystem frame {k} cr"
        );
    }
    // The spec-literal decode must differ (the divergence is real) —
    // return how many samples of the last frame differ.
    let spec = decode(stream, DecodeOptions::spec());
    let last = spec.len() - 1;
    let base = last * frame_len;
    spec[last]
        .luma_samples()
        .iter()
        .enumerate()
        .filter(|&(i, &s)| s != reference[base + i])
        .count()
}

#[test]
fn half_pel_probe_pins_the_negative_amv_rule() {
    let built = craft_probe_stream(false, -3, -7);
    if maybe_write_fixture("dec_sgmc_negamv_hp_64x64.m4v", &built) {
        return;
    }
    assert_eq!(built, fixture("dec_sgmc_negamv_hp_64x64.m4v"));
    let spec_diffs = assert_matches_reference(&built, "dec_sgmc_negamv_hp_64x64.yuv", 64, 64);
    assert!(
        spec_diffs > 0,
        "spec mode must differ on the diverging local macroblock"
    );
}

#[test]
fn quarter_pel_probe_pins_the_negative_amv_rule() {
    let built = craft_probe_stream(true, -4, -10);
    if maybe_write_fixture("dec_sgmc_negamv_qp_64x64.m4v", &built) {
        return;
    }
    assert_eq!(built, fixture("dec_sgmc_negamv_qp_64x64.m4v"));
    let spec_diffs = assert_matches_reference(&built, "dec_sgmc_negamv_qp_64x64.yuv", 64, 64);
    assert!(spec_diffs > 0);
}

/// A full encoder-produced S(GMC) stream whose trajectories go
/// negative: under ecosystem-compat every frame reproduces the
/// reference decode bit-exactly — evidence the negative-AMV rule is
/// the *only* GMC divergence in play.
#[test]
fn negative_trajectory_stream_reproduces_committed_fixture() {
    let built = build_negtraj_stream();
    if maybe_write_fixture("dec_sgmc_negtraj_96x64.m4v", &built) {
        return;
    }
    assert_eq!(
        built,
        fixture("dec_sgmc_negtraj_96x64.m4v"),
        "encoder output drifted; regenerate the fixture AND its reference decode"
    );
}

#[test]
fn negative_trajectory_stream_decodes_bit_exact_under_ecosystem_mode() {
    let stream = fixture("dec_sgmc_negtraj_96x64.m4v");
    let spec_diffs = assert_matches_reference(&stream, "dec_sgmc_negtraj_96x64.yuv", 96, 64);
    assert!(
        spec_diffs > 100,
        "the spec-literal decode should diverge broadly here ({spec_diffs})"
    );
}

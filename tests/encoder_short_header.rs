//! §6.2.5.2 short header (H.263-compatible) — encoder and decoder:
//! self-encoded I/P pictures decode sample-exact against the closed
//! loop through the stream decoder's short-header path (with and
//! without GOB headers), the registry `short-header` option, and the
//! black-box pins (`tests/fixtures/NOTES.md`): our stream decoded by
//! the reference decoder, and a reference-encoder-produced stream
//! decoded by ours.

use oxideav_mpeg4video::bitwriter::BitWriter;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::framestore::DecodedFrame;
use oxideav_mpeg4video::ivop_encode::{EncoderConfig, FrameView};
use oxideav_mpeg4video::short_header_encode::{
    encode_short_header_picture, write_short_video_end_marker,
};

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A textured scene panning by `shift` pels per frame with a
/// stationary bright block; sharp detail so intra blocks exercise the
/// Type-4 escape and long EVENT runs.
fn picture(w: usize, h: usize, frame_index: usize, shift: (i64, i64)) -> Planes {
    let (cw, ch) = (w / 2, h / 2);
    let scene = |x: i64, y: i64| -> u8 {
        let v = (x * 7 + y * 5).rem_euclid(160) + ((x.div_euclid(9) + y.div_euclid(7)) % 13) * 6;
        (30 + v.rem_euclid(200)) as u8
    };
    let n = frame_index as i64;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = scene(col as i64 + n * shift.0, row as i64 + n * shift.1);
        }
    }
    for row in 20..36.min(h) {
        for col in 40..56.min(w) {
            y[row * w + col] = if (row + col) % 2 == 0 { 250 } else { 5 };
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = scene(col as i64 + n * shift.0 / 2, row as i64) / 2 + 60;
            cr[row * cw + col] = 128 + ((col as i64 * 3 + n) % 9) as u8 * 4;
        }
    }
    (y, cb, cr)
}

fn decode_all(stream: &[u8]) -> Vec<DecodedFrame> {
    let mut dec = Mpeg4VideoDecoder::new();
    let mut frames = dec.decode(stream).expect("stream must decode");
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

/// Encode `frames` short-header pictures (I then P…, keyframe every
/// `gop`), returning the stream (end marker appended) and the recons.
fn encode_stream(
    cfg: &EncoderConfig,
    frames: usize,
    gop: usize,
    shift: (i64, i64),
    qp: u32,
) -> (Vec<u8>, Vec<DecodedFrame>) {
    let (w, h) = (usize::from(cfg.width), usize::from(cfg.height));
    let mut stream = Vec::new();
    let mut recons: Vec<DecodedFrame> = Vec::new();
    for k in 0..frames {
        let (y, cb, cr) = picture(w, h, k, shift);
        let view = FrameView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: w,
            height: h,
        };
        let reference = if k % gop == 0 { None } else { recons.last() };
        let (unit, recon, stats) =
            encode_short_header_picture(cfg, &view, reference, (k * 2) as u8, qp);
        if reference.is_some() {
            assert!(
                stats.inter > 0,
                "picture {k}: no inter macroblock ({stats:?})"
            );
        } else {
            assert_eq!(stats.intra, (w / 16) * (h / 16));
        }
        stream.extend_from_slice(&unit);
        recons.push(recon);
    }
    let mut tail = BitWriter::new();
    write_short_video_end_marker(&mut tail);
    stream.extend_from_slice(tail.as_bytes());
    (stream, recons)
}

#[test]
fn qcif_ip_with_gob_headers_round_trips() {
    let cfg = EncoderConfig {
        width: 176,
        height: 144,
        short_header: true,
        ..EncoderConfig::default()
    };
    let (stream, recons) = encode_stream(&cfg, 5, 3, (2, 1), 5);
    // No MPEG-4 start code anywhere; the picture starts are
    // byte-aligned short_video_start_markers.
    assert!(!stream.windows(3).any(|w| w == [0, 0, 1]));
    assert_eq!(
        oxideav_mpeg4video::short_header::scan_short_header_pictures(&stream).len(),
        5
    );
    let frames = decode_all(&stream);
    assert_exact(&frames, &recons);
    // temporal_reference ticks (30000/1001 Hz) follow the 2-per-frame
    // cadence the test wrote.
    let ticks: Vec<i64> = frames.iter().map(|f| f.pts_ticks().unwrap()).collect();
    assert_eq!(ticks, vec![0, 2, 4, 6, 8]);
}

#[test]
fn sub_qcif_without_gob_headers_round_trips_with_adaptive_quant() {
    let cfg = EncoderConfig {
        width: 128,
        height: 96,
        short_header: true,
        gob_headers: false,
        adaptive_quant: true,
        ..EncoderConfig::default()
    };
    let (stream, recons) = encode_stream(&cfg, 4, 4, (-3, 2), 8);
    assert_exact(&decode_all(&stream), &recons);
}

#[test]
fn cif_intra_only_round_trips_at_quantiser_extremes() {
    let cfg = EncoderConfig {
        width: 352,
        height: 288,
        short_header: true,
        ..EncoderConfig::default()
    };
    for qp in [1u32, 31] {
        let (stream, recons) = encode_stream(&cfg, 1, 1, (0, 0), qp);
        assert_exact(&decode_all(&stream), &recons);
    }
}

/// Registry path: `short-header` streams have no extradata, an
/// `H263` tag, decode through the same registry decoder, and reject
/// non-Table-6-29 sizes and non-fixed tools.
#[test]
fn registry_short_header_option() {
    use oxideav_core::Encoder as _;
    let (w, h) = (176usize, 144usize);
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.frame_rate = Some(oxideav_core::Rational::new(15, 1));
    params.options = oxideav_core::CodecOptions::default()
        .set("short-header", "true")
        .set("gop-size", "4")
        .set("mb-aq", "true");
    let mut enc = oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&params).unwrap();
    assert!(enc.output_params().extradata.is_empty());
    assert_eq!(
        enc.output_params().tag,
        Some(oxideav_core::CodecTag::fourcc(b"H263"))
    );
    let mut stream = Vec::new();
    for k in 0..6usize {
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
    let mut keyframes = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                keyframes.push(p.flags.keyframe);
                stream.extend_from_slice(&p.data);
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert_eq!(keyframes, vec![true, false, false, false, true, false]);
    let frames = decode_all(&stream);
    assert_eq!(frames.len(), 6);
    // 15 fps → temporal_reference advances by 2 ticks per frame.
    assert_eq!(frames[1].pts_ticks(), Some(2));

    // Registry decoder without extradata takes the short-header path.
    let mut dparams =
        oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    dparams.width = Some(w as u32);
    dparams.height = Some(h as u32);
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dparams).unwrap();
    let pkt = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), stream.clone());
    dec.send_packet(&pkt).unwrap();
    dec.flush().unwrap();
    let mut n = 0;
    loop {
        match dec.receive_frame() {
            Ok(_) => n += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert_eq!(n, 6);

    for (bad_w, bad_h, opts) in [
        (64u32, 64u32, vec![]),
        (176, 144, vec![("bf", "1")]),
        (176, 144, vec![("qpel", "true")]),
        (176, 144, vec![("interlaced", "true")]),
        (176, 144, vec![("fcode", "2")]),
    ] {
        let mut p = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
        p.width = Some(bad_w);
        p.height = Some(bad_h);
        let mut o = oxideav_core::CodecOptions::default().set("short-header", "true");
        for (k, v) in opts {
            o = o.set(k, v);
        }
        p.options = o;
        assert!(
            oxideav_mpeg4video::encoder::Mpeg4VideoEncoder::from_params(&p).is_err(),
            "{bad_w}x{bad_h} must be rejected"
        );
    }
}

/// Black-box pin: our QCIF I/P/P/I/P short-header stream decodes
/// bit-exactly through the reference decoder.
#[test]
fn blackbox_our_short_header_stream_is_bit_exact() {
    let cfg = EncoderConfig {
        width: 176,
        height: 144,
        short_header: true,
        adaptive_quant: true,
        ..EncoderConfig::default()
    };
    let (stream, recons) = encode_stream(&cfg, 5, 3, (2, 1), 5);
    pin_fixture("enc_sh_ippip_176x144.h263", &stream);
    if std::env::var_os("OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES").is_some() {
        return;
    }
    let yuv = std::fs::read(fixture_path("enc_sh_ippip_176x144.yuv")).unwrap();
    let (differing, max, total) = diff_against_yuv(&recons, &yuv, 176, 144);
    assert_eq!(
        (differing, max),
        (0, 0),
        "short header: {differing}/{total} samples differ (max {max})"
    );
}

/// Black-box pin: a reference-encoder-produced short-header stream
/// (I P P I P, QCIF) decodes bit-exactly through ours.
#[test]
fn blackbox_reference_short_header_stream_decodes_bit_exact() {
    let stream = std::fs::read(fixture_path("sh_ipp_176x144.h263")).unwrap();
    let yuv = std::fs::read(fixture_path("sh_ipp_176x144.yuv")).unwrap();
    let frames = decode_all(&stream);
    let (differing, max, total) = diff_against_yuv(&frames, &yuv, 176, 144);
    assert_eq!(
        (differing, max),
        (0, 0),
        "reference short header: {differing}/{total} samples differ (max {max})"
    );
}

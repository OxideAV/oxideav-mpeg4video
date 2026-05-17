//! Encoder-side `video_packet_header()` (§6.3.5.2) integration tests
//! (round 73).
//!
//! Exercises the `resync_marker_period` codec option:
//! * Self-roundtrip: encode → decode → PSNR + frame count, end-to-end
//!   through the marker-emitting + marker-consuming paths.
//! * VOL bit verification: `resync_marker_disable = 0` when the
//!   period is non-zero.
//! * Marker-byte search: the well-known 17-zeros-and-a-one pattern
//!   (for a P-VOP with `f_code = 1`) appears in the I-VOP and P-VOP
//!   bodies at least once.
//! * Per-encoder option-validation gates: `resync_marker_period>0 +
//!   dp=1` and `... + sprite_static=1` are rejected at factory time.
//! * ffmpeg cross-decode (skipped if ffmpeg is missing).

use std::process::Command;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

/// Synthesise a deterministic 96×96 frame stream (36 MBs per VOP — gives
/// us room for several resync packets per VOP when the period is small).
fn make_frame(idx: u32, width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; cw * ch];
    let mut cr = vec![128u8; cw * ch];
    let shift = idx as i32;
    let shift_v = (idx / 2) as i32;
    for row in 0..h {
        for col in 0..w {
            let x = col as i32 - shift;
            let yy = row as i32 - shift_v;
            let base = ((x.rem_euclid(64) * 3) + (yy.rem_euclid(48) * 2)) as u8;
            let bump = ((x.rem_euclid(16) as u8).wrapping_mul(2))
                .wrapping_add((yy.rem_euclid(16) as u8).wrapping_mul(3));
            y[row * w + col] = base.wrapping_add(bump);
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            let x = col as i32 - shift / 2;
            let yy = row as i32 - shift_v / 2;
            cb[row * cw + col] =
                (128i32 + (x.rem_euclid(16)) - (yy.rem_euclid(16))).clamp(0, 255) as u8;
            cr[row * cw + col] =
                (128i32 + (yy.rem_euclid(16)) - (x.rem_euclid(16))).clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        pts: Some(idx as i64),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn flatten_frame(v: &VideoFrame) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &v.planes {
        out.extend_from_slice(&p.data);
    }
    out
}

fn build_encoder_with_period(width: u32, height: u32, period: u32) -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params
        .options
        .insert("resync_marker_period", period.to_string());
    oxideav_mpeg4video::encoder::make_encoder(&params).expect("build resync encoder")
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq: u64 = 0;
    for i in 0..n {
        let d = a[i] as i32 - b[i] as i32;
        sum_sq += (d * d) as u64;
    }
    let mse = sum_sq as f64 / n as f64;
    if mse == 0.0 {
        return 100.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn command_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Locate the first VOL start code (0x000001 + 0x20..=0x2F) and parse
/// the VOL payload through the public parser.
fn parse_vol_from_packet(bytes: &[u8]) -> oxideav_mpeg4video::headers::vol::VideoObjectLayer {
    let mut vol_start = None;
    for i in 0..bytes.len().saturating_sub(4) {
        if bytes[i] == 0
            && bytes[i + 1] == 0
            && bytes[i + 2] == 1
            && (0x20..=0x2F).contains(&bytes[i + 3])
        {
            vol_start = Some(i + 4);
            break;
        }
    }
    let vol_payload_start = vol_start.expect("VOL start code not found");
    let mut br = oxideav_core::bits::BitReader::new(&bytes[vol_payload_start..]);
    oxideav_mpeg4video::headers::vol::parse_vol(&mut br).expect("parse VOL")
}

/// Encode 8 frames (1 I + 7 P) with `resync_marker_period = 6`, decode
/// them back through our own decoder, and check per-frame PSNR > 30 dB.
/// 96×96 = 6×6 = 36 MBs per VOP → 5 mid-VOP splits per frame.
#[test]
fn resync_self_roundtrip_psnr_passes() {
    let (width, height) = (96u32, 96u32);
    let num_frames = 8u32;

    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder_with_period(width, height, 6);
    let mut packets: Vec<Packet> = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }

    let mut es = Vec::new();
    for pkt in &packets {
        es.extend_from_slice(&pkt.data);
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("make_decoder");
    let in_pkt = Packet::new(0, TimeBase::new(1, 24), es.clone());
    dec.send_packet(&in_pkt).expect("send_packet");
    dec.flush().expect("flush decoder");

    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => decoded.push(v),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e:?}"),
        }
    }

    assert_eq!(
        decoded.len(),
        num_frames as usize,
        "decoded frame count mismatch"
    );
    for (i, (src, dec)) in src_frames.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(&flatten_frame(src), &flatten_frame(dec));
        assert!(p > 30.0, "frame {i}: PSNR {p:.2} dB below 30 dB threshold",);
    }
}

/// The VOL header must advertise `resync_marker_disable = 0` when the
/// encoder is configured with a non-zero period.
#[test]
fn resync_vol_advertises_resync_enabled() {
    let (width, height) = (96u32, 96u32);
    let mut enc = build_encoder_with_period(width, height, 6);
    enc.send_frame(&Frame::Video(make_frame(0, width, height)))
        .expect("send 0");
    let pkt = enc.receive_packet().expect("first packet");
    // Parse the bitstream as a real decoder would and verify the VOL
    // bit. We use a low-level header parser to read just the VOL.
    let vol = parse_vol_from_packet(&pkt.data);
    assert!(
        !vol.resync_marker_disable,
        "VOL must set resync_marker_disable = 0 when resync_marker_period > 0",
    );
}

/// The 17-zero resync-marker prefix (P-VOP, f_code = 1) must appear at
/// least once in the encoded P-VOP body when the period is set so that
/// at least one mid-VOP split occurs.
#[test]
fn resync_marker_pattern_present_in_p_vop_body() {
    let (width, height) = (96u32, 96u32);
    // 36 MBs / period 6 = 5 internal markers per P-VOP.
    let mut enc = build_encoder_with_period(width, height, 6);
    enc.send_frame(&Frame::Video(make_frame(0, width, height)))
        .expect("send 0");
    enc.send_frame(&Frame::Video(make_frame(1, width, height)))
        .expect("send 1");
    enc.flush().expect("flush");
    let _ = enc.receive_packet().expect("I-VOP packet"); // I-VOP packet (includes VOL)
    let p_pkt = enc.receive_packet().expect("P-VOP packet");
    // P-VOP with f_code = 1 → 16 zeros + 1 trailing 1 (per
    // video_packet_prefix_length). The marker is byte-aligned after the
    // stuffing, so it appears as two zero bytes (16 bits) followed by a
    // byte whose top bit is 1.
    let data = &p_pkt.data;
    let mut markers_found = 0;
    // Skip the VOP-header prefix (just past the 4-byte VOP start code).
    let start = data
        .windows(4)
        .position(|w| w == [0x00, 0x00, 0x01, 0xB6])
        .map(|p| p + 4)
        .unwrap_or(0);
    let mut i = start;
    while i + 3 <= data.len() {
        // Two zero bytes followed by a byte with the top bit set is
        // exactly the prefix produced by an `align_with_one_zero_then_ones`
        // padded boundary, then 16 zero bits, then the marker `1`.
        if data[i] == 0x00 && data[i + 1] == 0x00 && (data[i + 2] & 0x80) != 0 {
            // Filter out the VOP / Object start code (0x00 0x00 0x01).
            if data[i + 2] != 0x01 {
                markers_found += 1;
            }
        }
        i += 1;
    }
    assert!(
        markers_found >= 4,
        "expected several resync markers in the P-VOP body (got {markers_found})",
    );
}

/// `resync_marker_period > 0` is mutually exclusive with `dp = 1` and
/// `sprite_static = 1`. The factory must reject both combinations with
/// a clear error.
#[test]
fn resync_with_incompatible_options_rejected() {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.options.insert("resync_marker_period", "8");
    params.options.insert("dp", "1");
    let err = oxideav_mpeg4video::encoder::make_encoder(&params).err();
    assert!(
        matches!(err, Some(oxideav_core::Error::Unsupported(_))),
        "resync + dp must be rejected, got {err:?}",
    );

    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.options.insert("resync_marker_period", "8");
    params.options.insert("sprite_static", "1");
    let err = oxideav_mpeg4video::encoder::make_encoder(&params).err();
    assert!(
        matches!(err, Some(oxideav_core::Error::Unsupported(_))),
        "resync + sprite_static must be rejected, got {err:?}",
    );
}

/// `resync_marker_period = 0` (default) produces a bitstream that is
/// byte-identical to what round-72 emitted: VOL has
/// `resync_marker_disable = 1`, no mid-VOP markers, no extra bytes.
/// We assert the headline byte: `resync_marker_disable = 1`.
#[test]
fn resync_default_zero_keeps_legacy_behaviour() {
    let (width, height) = (96u32, 96u32);
    let mut enc = build_encoder_with_period(width, height, 0);
    enc.send_frame(&Frame::Video(make_frame(0, width, height)))
        .expect("send 0");
    let pkt = enc.receive_packet().expect("first packet");
    let vol = parse_vol_from_packet(&pkt.data);
    assert!(
        vol.resync_marker_disable,
        "default behaviour: resync_marker_disable must remain 1",
    );
}

/// ffmpeg cross-decode: hand our resync-mode ES to ffmpeg and assert
/// (a) ffmpeg accepts the stream and (b) the decoded I-VOP and P-VOP
/// match our source within 30 dB / 25 dB PSNR.
#[test]
fn resync_ffmpeg_decode() {
    if !command_exists("ffmpeg") {
        eprintln!("ffmpeg missing — skipping resync ffmpeg interop test");
        return;
    }
    let (width, height) = (96u32, 96u32);
    let num_frames = 4u32;
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    // 36 MBs / period 9 = 3 internal markers per VOP — exercises the
    // marker emit path without exploding bitstream size.
    let mut enc = build_encoder_with_period(width, height, 9);
    let mut es = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => es.extend_from_slice(&pkt.data),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }

    let tmp = std::env::temp_dir();
    let es_path = tmp.join("oxideav_resync_ours.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_resync_ffmpeg.yuv");
    let _ = std::fs::remove_file(&yuv_out);
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "m4v",
            "-i",
            es_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            yuv_out.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode our resync-mode stream",
    );
    let ffmpeg_decoded = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame_bytes = (width as usize * height as usize * 3) / 2;
    assert!(
        ffmpeg_decoded.len() >= per_frame_bytes,
        "ffmpeg resync-mode output too small: {} bytes",
        ffmpeg_decoded.len(),
    );
    let src0 = flatten_frame(&src_frames[0]);
    let ff0 = &ffmpeg_decoded[0..per_frame_bytes];
    let p0 = psnr(&src0, ff0);
    eprintln!("ffmpeg resync decode frame 0 (I-VOP): PSNR = {p0:.2} dB");
    assert!(
        p0 > 30.0,
        "ffmpeg resync I-VOP PSNR {p0:.2} dB below 30 dB — bitstream is malformed",
    );
    let src1 = flatten_frame(&src_frames[1]);
    let ff1 = &ffmpeg_decoded[per_frame_bytes..2 * per_frame_bytes];
    let p1 = psnr(&src1, ff1);
    eprintln!("ffmpeg resync decode frame 1 (first P-VOP): PSNR = {p1:.2} dB");
    assert!(
        p1 > 25.0,
        "ffmpeg resync first P-VOP PSNR {p1:.2} dB below 25 dB",
    );
    let _ = &es_path;
}

/// Smaller resync period (every 4 MBs on a 96×96 frame = 9 internal
/// splits per VOP) still round-trips through our decoder. This pins
/// the marker emission at high frequency and exercises the
/// slice_first_mb threading on every other MB row.
#[test]
fn resync_high_frequency_period_still_roundtrips() {
    let (width, height) = (96u32, 96u32);
    let num_frames = 4u32;
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }
    let mut enc = build_encoder_with_period(width, height, 4);
    let mut es = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => es.extend_from_slice(&p.data),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("make_decoder");
    let in_pkt = Packet::new(0, TimeBase::new(1, 24), es);
    dec.send_packet(&in_pkt).expect("send_packet");
    dec.flush().expect("flush dec");
    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => decoded.push(v),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e:?}"),
        }
    }
    assert_eq!(decoded.len(), num_frames as usize);
    for (i, (src, dec)) in src_frames.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(&flatten_frame(src), &flatten_frame(dec));
        assert!(p > 30.0, "frame {i}: PSNR {p:.2} dB below 30 dB");
    }
}

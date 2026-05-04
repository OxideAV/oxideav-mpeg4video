//! Data-partitioning encoder integration tests (round 21).
//!
//! Exercises the `dp` codec option:
//! * Self-roundtrip: encode → decode → PSNR + frame count.
//! * VOL bit verification: `data_partitioned = 1` + `resync_marker_disable = 0`.
//! * DC and motion marker bytes appear in the emitted bitstream.
//! * ffmpeg cross-decode (skipped if ffmpeg is missing).

use std::process::Command;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

/// Synthesise a deterministic 64×64 frame stream — same recipe as
/// `tests/p_vop.rs::make_frame` so the DP and combined-mode tests
/// exercise comparable content.
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
            let base = ((x.rem_euclid(64) * 4) + (yy.rem_euclid(48) * 2)) as u8;
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

fn build_encoder_dp(width: u32, height: u32) -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.options.insert("dp", "1");
    oxideav_mpeg4video::encoder::make_encoder(&params).expect("build dp encoder")
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

/// Encode 8 frames (1 I + 7 P) under data partitioning, decode them
/// back through our own decoder, and check per-frame PSNR > 30 dB.
#[test]
fn dp_self_roundtrip_psnr_passes() {
    let (width, height) = (64u32, 64u32);
    let num_frames = 8u32;

    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder_dp(width, height);
    let mut packets: Vec<Packet> = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    assert!(
        packets.len() >= num_frames as usize,
        "DP encoder produced {} packets, expected ≥{}",
        packets.len(),
        num_frames
    );
    assert!(
        packets[0].flags.keyframe,
        "first DP packet should be keyframe"
    );

    // Concatenate ES + decode through our decoder.
    let mut es = Vec::new();
    for pkt in &packets {
        es.extend_from_slice(&pkt.data);
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("build decoder");
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
        "DP decoder returned {} frames, expected {}",
        decoded.len(),
        num_frames
    );

    let mut min_psnr = f64::INFINITY;
    for (i, (src, dec)) in src_frames.iter().zip(decoded.iter()).enumerate() {
        let s = flatten_frame(src);
        let d = flatten_frame(dec);
        let p = psnr(&s, &d);
        eprintln!("DP roundtrip frame {i}: PSNR = {p:.2} dB");
        min_psnr = min_psnr.min(p);
    }
    assert!(
        min_psnr > 30.0,
        "DP roundtrip min-PSNR {min_psnr:.2} dB below 30 dB"
    );
}

/// Verify the VOL bytes carry `data_partitioned = 1` and
/// `resync_marker_disable = 0`. Catches any regression in the
/// VOL writer's bit ordering for DP-on streams.
#[test]
fn dp_vol_advertises_partitioning() {
    use oxideav_core::bits::BitReader;
    let mut enc = build_encoder_dp(64, 64);
    let f = make_frame(0, 64, 64);
    enc.send_frame(&Frame::Video(f)).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("first packet");
    let bytes = pkt.data.clone();
    // Find the VOL start code (0x000001 + 0x20..0x2F) — the encoder uses 0x20.
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
    // Re-parse the VOL payload through the public parser — that gives us
    // the high-level fields back, which is what we want to verify.
    let vol = oxideav_mpeg4video::headers::vol::parse_vol(&mut BitReader::new(
        &bytes[vol_payload_start..],
    ))
    .expect("VOL parse");
    assert!(
        vol.data_partitioned,
        "VOL did not advertise data_partitioned=1"
    );
    assert!(
        !vol.resync_marker_disable,
        "VOL must clear resync_marker_disable when data_partitioned=1"
    );
    assert!(
        !vol.reversible_vlc,
        "DP-only encoder must keep RVLC=0; opt in via `rvlc=1` (round-22)"
    );
}

/// Sanity-check that the DC marker (19 bits) appears in the I-VOP body
/// after the VOP header. Catches catastrophic emit-order regressions
/// before they show up as decoder mismatches.
#[test]
fn dp_i_vop_emits_dc_marker() {
    let mut enc = build_encoder_dp(48, 48);
    let f = make_frame(0, 48, 48);
    enc.send_frame(&Frame::Video(f)).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("first packet");
    // Brute-force scan: at every bit alignment, peek the next 19 bits
    // and compare them to `DC_MARKER`. This mirrors what a spec-strict
    // DP decoder does to find the marker (`next_bits()` in §6.2.5.3).
    let dc = oxideav_mpeg4video::dp::DC_MARKER;
    let total_bits = pkt.data.len() * 8;
    let mut found = false;
    for bit_off in 0..total_bits.saturating_sub(19) {
        let mut acc: u32 = 0;
        for i in 0..19 {
            let b = bit_off + i;
            let bit = (pkt.data[b / 8] >> (7 - (b % 8))) & 1;
            acc = (acc << 1) | (bit as u32);
        }
        if acc == dc {
            found = true;
            break;
        }
    }
    assert!(found, "DC marker not found in DP I-VOP byte stream");
}

/// ffmpeg cross-decode: hand our DP-mode ES to ffmpeg and assert
/// (a) ffmpeg accepts the stream and (b) the decoded I-VOP matches
/// our source within 30 dB PSNR.
#[test]
fn dp_ffmpeg_decode() {
    if !command_exists("ffmpeg") {
        eprintln!("ffmpeg missing — skipping DP ffmpeg interop test");
        return;
    }
    let (width, height) = (64u32, 64u32);
    let num_frames = 8u32;
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder_dp(width, height);
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
    let es_path = tmp.join("oxideav_dp_ours.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_dp_ffmpeg.yuv");
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
        "ffmpeg failed to decode our DP-mode stream"
    );
    let ffmpeg_decoded = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame_bytes = (width as usize * height as usize * 3) / 2;
    assert!(
        ffmpeg_decoded.len() >= per_frame_bytes,
        "ffmpeg DP-mode output too small: {} bytes",
        ffmpeg_decoded.len()
    );
    // I-VOP must be near-bit-exact (no MV / motion drift).
    let src0 = flatten_frame(&src_frames[0]);
    let ff0 = &ffmpeg_decoded[0..per_frame_bytes];
    let p0 = psnr(&src0, ff0);
    eprintln!("ffmpeg DP decode frame 0 (I-VOP): PSNR = {p0:.2} dB");
    assert!(
        p0 > 30.0,
        "ffmpeg DP I-VOP PSNR {p0:.2} dB below 30 dB — bitstream is malformed"
    );
    // First P-VOP — small drift expected from ffmpeg's integer IDCT vs
    // our float IDCT, but a clean DP layout should still pass 25 dB.
    let src1 = flatten_frame(&src_frames[1]);
    let ff1 = &ffmpeg_decoded[per_frame_bytes..2 * per_frame_bytes];
    let p1 = psnr(&src1, ff1);
    eprintln!("ffmpeg DP decode frame 1 (first P-VOP): PSNR = {p1:.2} dB");
    assert!(
        p1 > 25.0,
        "ffmpeg DP first P-VOP PSNR {p1:.2} dB below 25 dB"
    );
    let _ = &es_path;
}

/// Sanity-check that the motion marker (17 bits) appears in the body
/// of the FIRST P-VOP after the I-VOP.
#[test]
fn dp_p_vop_emits_motion_marker() {
    let (w, h) = (48u32, 48u32);
    let mut enc = build_encoder_dp(w, h);
    // Send 2 frames: 1 I-VOP + 1 P-VOP.
    enc.send_frame(&Frame::Video(make_frame(0, w, h)))
        .expect("send 0");
    enc.send_frame(&Frame::Video(make_frame(1, w, h)))
        .expect("send 1");
    enc.flush().expect("flush");
    let _i_pkt = enc.receive_packet().expect("I-VOP packet");
    let p_pkt = enc.receive_packet().expect("P-VOP packet");
    let mm = oxideav_mpeg4video::dp::MOTION_MARKER;
    let total_bits = p_pkt.data.len() * 8;
    let mut found = false;
    for bit_off in 0..total_bits.saturating_sub(17) {
        let mut acc: u32 = 0;
        for i in 0..17 {
            let b = bit_off + i;
            let bit = (p_pkt.data[b / 8] >> (7 - (b % 8))) & 1;
            acc = (acc << 1) | (bit as u32);
        }
        if acc == mm {
            found = true;
            break;
        }
    }
    assert!(found, "motion marker not found in DP P-VOP byte stream");
}

/// Reject DP combined with QPel / GMC / B-frames — these aren't yet
/// plumbed into the DP body emitter (see `dp.rs` follow-up notes).
#[test]
fn dp_rejects_unsupported_combos() {
    for opt in [("qpel", "1"), ("gmc", "1"), ("bf", "2")] {
        let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
        params.media_type = MediaType::Video;
        params.width = Some(64);
        params.height = Some(64);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(24, 1));
        params.options.insert("dp", "1");
        params.options.insert(opt.0, opt.1);
        let r = oxideav_mpeg4video::encoder::make_encoder(&params);
        assert!(
            r.is_err(),
            "DP + {}={} should be rejected by the encoder factory",
            opt.0,
            opt.1
        );
    }
}

/// Build a high-contrast checker pattern with a "phase" parameter — two
/// visually unrelated frames so a hard switch between them simulates a
/// scene change. Mirrors `make_scene` in `tests/p_vop.rs` so this test
/// exercises the same intra-in-P decision path under DP layout.
fn make_scene(phase: u32, width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let cb = vec![128u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    for row in 0..h {
        for col in 0..w {
            let v = match phase {
                0 => ((col * 5) ^ (row * 11)) as u8,
                _ => 255u8.wrapping_sub(((col * 13) ^ (row * 7)) as u8),
            };
            y[row * w + col] = v;
        }
    }
    VideoFrame {
        pts: Some(0),
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

/// Encode an 8-frame scene-change clip under DP, verify the encoder
/// picked at least one Intra-in-P MB (Table B-13 rows 4..=7) inside the
/// post-cut P-VOP, then roundtrip through (a) our decoder and (b)
/// ffmpeg. The intra MB syntax inside `data_partitioned_p_vop()`
/// (§6.2.5.3) puts intra DC values into part 2 (after motion_marker)
/// and intra AC walks into part 3 — this test covers the spec-mandated
/// routing for `derived_mb_type >= 3` MBs in a DP P-VOP body.
#[test]
fn dp_p_vop_intra_in_p_scene_change_roundtrip() {
    let (width, height) = (32u32, 32u32);
    let num_frames = 8u32;
    let mut src: Vec<VideoFrame> = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        let phase = if i < 4 { 0 } else { 1 };
        let mut f = make_scene(phase, width, height);
        f.pts = Some(i as i64);
        src.push(f);
    }

    let mut enc = build_encoder_dp(width, height);
    for f in &src {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    let mut packets: Vec<Packet> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    assert!(packets.len() >= num_frames as usize);

    // Find the post-cut P-VOP (frame 4 packet) and verify at least one
    // Intra-in-P MCBPC codeword (Table B-13 rows 4..=7) appears in its
    // body. Codewords:
    //   row 4 (Intra, cbpc=0): 5 bits = 0b00011
    //   row 5 (Intra, cbpc=1): 8 bits = 0b00000100
    //   row 6 (Intra, cbpc=2): 8 bits = 0b00000011
    //   row 7 (Intra, cbpc=3): 7 bits = 0b0000011
    // We do a coarse "8 leading zero bits in a row" scan as a proxy for
    // any of the longer Intra rows (5 / 6 / 7 all start with 5+ zeros
    // and don't appear in any Inter row 0..=3 / 16..=19). The
    // self-roundtrip + ffmpeg check below validates the syntax exactly.
    let cut_pkt = &packets[4];
    let mut intra_seen = false;
    for byte_idx in 0..cut_pkt.data.len().saturating_sub(2) {
        // Look at any 16-bit window for `00000011` or `00000100` (the
        // Intra row 4..=7 prefixes) at any bit alignment.
        let w16 = ((cut_pkt.data[byte_idx] as u16) << 8) | cut_pkt.data[byte_idx + 1] as u16;
        for bit_off in 0..9u32 {
            let win8 = ((w16 >> (8 - bit_off)) & 0xFF) as u8;
            if win8 == 0b00000011 || win8 == 0b00000100 {
                intra_seen = true;
                break;
            }
        }
        if intra_seen {
            break;
        }
    }
    assert!(
        intra_seen,
        "post-cut DP P-VOP packet contains no Intra-in-P MCBPC prefix"
    );

    // Self-roundtrip through our decoder.
    let mut es = Vec::new();
    for pkt in &packets {
        es.extend_from_slice(&pkt.data);
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("build decoder");
    let in_pkt = Packet::new(0, TimeBase::new(1, 24), es.clone());
    dec.send_packet(&in_pkt).expect("send_packet");
    dec.flush().expect("flush decoder");
    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(f)) => decoded.push(f),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e:?}"),
        }
    }
    assert_eq!(
        decoded.len(),
        num_frames as usize,
        "DP intra-in-P self-roundtrip dropped frames"
    );
    // PSNR per frame — the cut frame stays high because intra-in-P
    // recovers the post-cut content exactly (no MC drift).
    for (i, (s, d)) in src.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(&flatten_frame(s), &flatten_frame(d));
        eprintln!("DP intra-in-P frame {i}: PSNR = {p:.2} dB");
        assert!(
            p > 25.0,
            "DP intra-in-P frame {i} PSNR {p:.2} dB below 25 dB"
        );
    }

    // ffmpeg cross-decode — the spec-conformant DP P-VOP body with
    // mixed intra+inter MBs must roundtrip cleanly.
    if !command_exists("ffmpeg") {
        eprintln!("ffmpeg missing — skipping DP intra-in-P ffmpeg interop test");
        return;
    }
    let tmp = std::env::temp_dir();
    let es_path = tmp.join("oxideav_dp_intra_p_scene.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_dp_intra_p_scene_ffmpeg.yuv");
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
        "ffmpeg failed to decode our DP intra-in-P scene-change stream"
    );
    let ff = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame = (width as usize * height as usize * 3) / 2;
    assert!(
        ff.len() >= per_frame * num_frames as usize,
        "ffmpeg DP intra-in-P output too small: {} bytes for {} frames",
        ff.len(),
        num_frames
    );
    let src4 = flatten_frame(&src[4]);
    let ff4 = &ff[4 * per_frame..5 * per_frame];
    let p4 = psnr(&src4, ff4);
    eprintln!("ffmpeg DP intra-in-P cut-frame PSNR = {p4:.2} dB");
    assert!(
        p4 > 25.0,
        "ffmpeg DP intra-in-P cut-frame PSNR {p4:.2} below 25 dB — intra-in-P syntax mismatch"
    );
}

/// Build a frame whose 8×8 luma blocks each translate by a *different*
/// per-block offset frame-to-frame. The 1MV mode can only pick a single
/// MV per 16×16 MB so it cannot match all four sub-blocks; the encoder's
/// 4MV mode-decision (§7.5.7 / §7.6.7) wins and emits Inter4MV MCBPC
/// codewords (Table B-13 rows 16..=19) with four MVDs per MB. Mirrors
/// `make_subblock_motion_frame` in `tests/p_vop.rs`.
fn make_subblock_motion_frame(
    idx: u32,
    width: u32,
    height: u32,
    base_pattern: &[u8],
) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let cb = vec![128u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    for by in 0..(h / 8) {
        for bx in 0..(w / 8) {
            let dx = ((bx + idx as usize * 2) % 4) as i32 - 2;
            let dy = ((by + idx as usize * 3) % 4) as i32 - 2;
            for j in 0..8usize {
                for i in 0..8usize {
                    let sx = (bx * 8 + i) as i32 + dx;
                    let sy = (by * 8 + j) as i32 + dy;
                    let sx = sx.rem_euclid(w as i32) as usize;
                    let sy = sy.rem_euclid(h as i32) as usize;
                    y[(by * 8 + j) * w + bx * 8 + i] = base_pattern[sy * w + sx];
                }
            }
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

/// Encode a 4-frame sub-MB-motion clip under DP, verify the encoder
/// picked at least one Inter4MV MCBPC codeword (Table B-13 rows
/// 16..=19) inside a P-VOP body, then roundtrip through (a) our DP
/// decoder and (b) ffmpeg. The Inter4MV MB syntax inside
/// `data_partitioned_p_vop()` (§6.3.7) puts FOUR MVDs per MB into part 1
/// (instead of one) — the motion partition's marker still appears
/// downstream because the per-MB MCBPC + 4 MV components are
/// uniqueness-disjoint from the 17-bit `motion_marker` pattern. This
/// test exercises the spec-mandated routing for Inter4MV MBs in a DP
/// P-VOP body.
#[test]
fn dp_p_vop_inter4mv_roundtrip() {
    let (width, height) = (32u32, 32u32);
    let num_frames = 4u32;
    // Build a static base pattern (deterministic checker-style noise) so
    // the per-block "translation" actually has visible content to track.
    let w = width as usize;
    let h = height as usize;
    let mut base = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            base[y * w + x] = ((x * 7) ^ (y * 13) ^ (x * y / 4)) as u8;
        }
    }
    let src: Vec<VideoFrame> = (0..num_frames)
        .map(|i| make_subblock_motion_frame(i, width, height, &base))
        .collect();

    let mut enc = build_encoder_dp(width, height);
    for f in &src {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    let mut packets: Vec<Packet> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    assert!(packets.len() >= num_frames as usize);

    // Inter4MV MCBPC codewords (Table B-13 rows 16..=19):
    //   row 16 (Inter4MV, cbpc=0): 3 bits = 0b010
    //   row 17 (Inter4MV, cbpc=1): 7 bits = 0b0000101
    //   row 18 (Inter4MV, cbpc=2): 7 bits = 0b0000100
    //   row 19 (Inter4MV, cbpc=3): 8 bits = 0b00000101
    // The cbpc=0 prefix `010` is also a sub-pattern of plenty of other
    // codewords, so we look for the longer row-17/18/19 prefixes which
    // share `0000010` as their leading 7 bits — distinct from any Inter
    // (rows 0..=3) or Intra (rows 4..=7) prefix in B-13.
    let mut inter4mv_seen = false;
    for pkt in &packets[1..] {
        // Look for `0000010` at any bit alignment in the P-VOP packets.
        for byte_idx in 0..pkt.data.len().saturating_sub(2) {
            let w16 = ((pkt.data[byte_idx] as u16) << 8) | pkt.data[byte_idx + 1] as u16;
            for bit_off in 0..9u32 {
                let win7 = ((w16 >> (9 - bit_off)) & 0x7F) as u8;
                if win7 == 0b0000010 {
                    inter4mv_seen = true;
                    break;
                }
            }
            if inter4mv_seen {
                break;
            }
        }
        if inter4mv_seen {
            break;
        }
    }
    assert!(
        inter4mv_seen,
        "no Inter4MV MCBPC prefix found in DP P-VOP packets — \
         the encoder did not exercise Inter4MV"
    );

    // Self-roundtrip through our decoder.
    let mut es = Vec::new();
    for pkt in &packets {
        es.extend_from_slice(&pkt.data);
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("build decoder");
    let in_pkt = Packet::new(0, TimeBase::new(1, 24), es.clone());
    dec.send_packet(&in_pkt).expect("send_packet");
    dec.flush().expect("flush decoder");
    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(f)) => decoded.push(f),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e:?}"),
        }
    }
    assert_eq!(
        decoded.len(),
        num_frames as usize,
        "DP Inter4MV self-roundtrip dropped frames"
    );
    let mut min_psnr = f64::INFINITY;
    for (i, (s, d)) in src.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(&flatten_frame(s), &flatten_frame(d));
        eprintln!("DP Inter4MV frame {i}: PSNR = {p:.2} dB");
        min_psnr = min_psnr.min(p);
    }
    assert!(
        min_psnr > 25.0,
        "DP Inter4MV self-roundtrip min-PSNR {min_psnr:.2} dB below 25 dB"
    );

    // Hard-asserted ffmpeg cross-decode.
    if !command_exists("ffmpeg") {
        panic!("ffmpeg required for DP Inter4MV cross-decode test");
    }
    let tmp = std::env::temp_dir();
    let es_path = tmp.join("oxideav_dp_inter4mv.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_dp_inter4mv_ffmpeg.yuv");
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
        "ffmpeg failed to decode our DP Inter4MV stream"
    );
    let ff = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame = (width as usize * height as usize * 3) / 2;
    assert!(
        ff.len() >= per_frame * num_frames as usize,
        "ffmpeg DP Inter4MV output too small: {} bytes for {} frames",
        ff.len(),
        num_frames
    );
    // The I-VOP is bit-near-exact; the post-I P-VOPs carry the Inter4MV
    // motion. Validate every frame so a regression in any of part-1
    // (4 MVs) / part-2 (cbpy) / part-3 (AC walks) shows up.
    let mut ff_min = f64::INFINITY;
    for (i, src_frame) in src.iter().enumerate() {
        let s = flatten_frame(src_frame);
        let f = &ff[i * per_frame..(i + 1) * per_frame];
        let p = psnr(&s, f);
        eprintln!("ffmpeg DP Inter4MV frame {i}: PSNR = {p:.2} dB");
        ff_min = ff_min.min(p);
    }
    assert!(
        ff_min > 25.0,
        "ffmpeg DP Inter4MV min-PSNR {ff_min:.2} dB below 25 dB — \
         Inter4MV under DP decoded incorrectly by ffmpeg"
    );
}

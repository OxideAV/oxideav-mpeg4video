//! Reversible VLC (round 22) encoder integration tests.
//!
//! Exercises the `rvlc` codec option in combination with `dp`:
//! * Self-roundtrip: encode → decode → PSNR + frame count.
//! * VOL bit verification: `reversible_vlc = 1` only when both `dp=1`
//!   and `rvlc=1` are set.
//! * The encoder factory rejects `rvlc=1 dp=0` (per ISO/IEC 14496-2
//!   §6.2.5: RVLC requires DP).
//! * ffmpeg cross-decode: hand our DP+RVLC ES to ffmpeg and assert it
//!   accepts it.
//! * Bit-overhead measurement: emit the same content with and without
//!   RVLC and confirm the overhead is small (< ~50% of bytes).

use std::process::Command;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

/// Same content recipe as `tests/dp.rs::make_frame` so the RVLC and
/// non-RVLC tests exercise comparable bitstreams.
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

fn build_encoder(width: u32, height: u32, rvlc: bool) -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.options.insert("dp", "1");
    if rvlc {
        params.options.insert("rvlc", "1");
    }
    oxideav_mpeg4video::encoder::make_encoder(&params).expect("build dp+rvlc encoder")
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

/// Encode a stream with DP+RVLC, decode through our own decoder, and
/// confirm per-frame PSNR remains above 30 dB. Mirrors
/// `dp::dp_self_roundtrip_psnr_passes` for the RVLC writer/reader path.
#[test]
fn rvlc_self_roundtrip_psnr_passes() {
    let (width, height) = (64u32, 64u32);
    let num_frames = 8u32;

    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder(width, height, true);
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
        "RVLC encoder produced {} packets, expected ≥{}",
        packets.len(),
        num_frames
    );
    assert!(
        packets[0].flags.keyframe,
        "first DP+RVLC packet should be keyframe"
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
        "RVLC decoder returned {} frames, expected {}",
        decoded.len(),
        num_frames
    );

    let mut min_psnr = f64::INFINITY;
    for (i, (src, dec)) in src_frames.iter().zip(decoded.iter()).enumerate() {
        let s = flatten_frame(src);
        let d = flatten_frame(dec);
        let p = psnr(&s, &d);
        eprintln!("RVLC roundtrip frame {i}: PSNR = {p:.2} dB");
        min_psnr = min_psnr.min(p);
    }
    assert!(
        min_psnr > 30.0,
        "RVLC roundtrip min-PSNR {min_psnr:.2} dB below 30 dB"
    );
}

/// Verify the VOL bytes carry both `data_partitioned = 1` and
/// `reversible_vlc = 1` when `rvlc=1` is on.
#[test]
fn rvlc_vol_advertises_reversible_vlc() {
    use oxideav_core::bits::BitReader;
    let mut enc = build_encoder(64, 64, true);
    let f = make_frame(0, 64, 64);
    enc.send_frame(&Frame::Video(f)).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("first packet");
    let bytes = pkt.data.clone();
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
    let vol = oxideav_mpeg4video::headers::vol::parse_vol(&mut BitReader::new(
        &bytes[vol_payload_start..],
    ))
    .expect("VOL parse");
    assert!(
        vol.data_partitioned,
        "VOL did not advertise data_partitioned=1"
    );
    assert!(
        vol.reversible_vlc,
        "VOL did not advertise reversible_vlc=1 with rvlc=1"
    );
    assert!(
        !vol.resync_marker_disable,
        "VOL must clear resync_marker_disable when DP is on"
    );
}

/// `rvlc=1` without `dp=1` must be rejected by the encoder factory
/// per the spec rule that RVLC requires DP (§6.2.5).
#[test]
fn rvlc_without_dp_is_rejected() {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.options.insert("rvlc", "1");
    let r = oxideav_mpeg4video::encoder::make_encoder(&params);
    assert!(r.is_err(), "rvlc=1 without dp=1 must be rejected");
}

/// Bit-overhead measurement: encode the same content with `dp=1
/// rvlc=0` and `dp=1 rvlc=1`, then assert the RVLC stream isn't
/// dramatically larger than the non-RVLC stream. Empirically the cost
/// is in the single-digit-percent range on natural content; we leave
/// generous headroom (≤ 60% overhead) so the test isn't fragile to
/// table tuning.
#[test]
fn rvlc_bit_overhead_is_modest() {
    let (width, height) = (64u32, 64u32);
    let num_frames = 6u32;

    let measure = |rvlc: bool| -> usize {
        let mut enc = build_encoder(width, height, rvlc);
        let mut total = 0usize;
        for i in 0..num_frames {
            let f = make_frame(i, width, height);
            enc.send_frame(&Frame::Video(f)).expect("send_frame");
        }
        enc.flush().expect("flush");
        loop {
            match enc.receive_packet() {
                Ok(pkt) => total += pkt.data.len(),
                Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("receive_packet: {e:?}"),
            }
        }
        total
    };

    let baseline = measure(false);
    let with_rvlc = measure(true);
    let overhead = if baseline == 0 {
        0.0
    } else {
        (with_rvlc as f64 - baseline as f64) * 100.0 / baseline as f64
    };
    eprintln!(
        "DP non-RVLC: {} bytes, DP+RVLC: {} bytes, overhead: {:+.2}%",
        baseline, with_rvlc, overhead
    );
    assert!(
        overhead < 60.0,
        "RVLC overhead {overhead:.2}% exceeds the 60% sanity cap"
    );
}

/// Error-recovery acceptance test (round 24, task #175).
///
/// Demonstrates the headline RVLC benefit: when the AC-coefficient
/// region of the bitstream takes deliberate bit-level damage, the RVLC
/// path recovers MORE pixel area (= more blocks whose coefficients
/// match the original exactly) than the standard Tcoef path does. The
/// reason: with RVLC we can walk backward from the END of the damaged
/// region by parsing the bit-reversed buffer (Annex E.1.4.4 strategy
/// 1/3). Standard Tcoef has no reverse-decode property — it can only
/// walk forward, and a parser that walks past damaged bytes typically
/// emits garbage coefficients without raising any error, since the
/// table is dense and most random bit patterns parse as valid VLCs.
///
/// The test builds a synthetic K-block AC partition twice (RVLC +
/// standard Tcoef) over the SAME coefficient stream, corrupts a byte
/// in the middle, then measures how many blocks each path can
/// reproduce BIT-EXACTLY (= blocks whose decoded coefficient array
/// equals the original). This is the most defensible "pixel area
/// recovered" metric because any deviation in coefficients leads to
/// wrong pels.
///
/// Pass condition: RVLC recovers strictly more bit-exact blocks than
/// standard Tcoef.
#[test]
fn rvlc_corruption_recovery_beats_baseline() {
    use oxideav_core::bits::{BitReader, BitWriter};
    use oxideav_mpeg4video::block;
    use oxideav_mpeg4video::headers::vol::ZIGZAG;
    use oxideav_mpeg4video::rvlc;

    // Build K test blocks with non-trivial coefficient patterns. A mix
    // of low- and high-frequency taps spreads codeword lengths across
    // the table.
    let k_blocks = 16usize;
    let make_block = |seed: usize| -> [i32; 64] {
        let mut b = [0i32; 64];
        b[ZIGZAG[1]] = ((seed % 5) as i32) + 1;
        b[ZIGZAG[3]] = -(((seed * 7) % 4) as i32) - 1;
        if seed % 3 == 0 {
            b[ZIGZAG[8]] = 2;
        }
        if seed % 4 == 0 {
            b[ZIGZAG[16]] = -1;
        }
        b[ZIGZAG[20 + (seed % 10)]] = 1; // last-non-zero
        b
    };
    let originals: Vec<[i32; 64]> = (0..k_blocks).map(make_block).collect();

    // ------------------------------------------------------------------
    // Encode K blocks under each codec table; capture per-block bit lengths.
    // ------------------------------------------------------------------
    let mut bw_rvlc = BitWriter::new();
    let mut block_lens_rvlc: Vec<u64> = Vec::new();
    let mut prev_pos = 0u64;
    for orig in originals.iter().take(k_blocks) {
        rvlc::write_intra_ac(&mut bw_rvlc, orig).expect("rvlc encode");
        let pos = bw_rvlc.bit_position();
        block_lens_rvlc.push(pos - prev_pos);
        prev_pos = pos;
    }
    let total_bits_rvlc = bw_rvlc.bit_position();
    let bytes_rvlc = bw_rvlc.finish();

    let mut bw_std = BitWriter::new();
    let mut prev_pos = 0u64;
    let mut block_lens_std: Vec<u64> = Vec::new();
    for orig in originals.iter().take(k_blocks) {
        oxideav_mpeg4video::encoder::write_intra_ac(&mut bw_std, orig).expect("std encode");
        let pos = bw_std.bit_position();
        block_lens_std.push(pos - prev_pos);
        prev_pos = pos;
    }
    let total_bits_std = bw_std.bit_position();
    let bytes_std = bw_std.finish();

    eprintln!(
        "synthetic AC partition: RVLC = {} bits ({} bytes), std = {} bits ({} bytes)",
        total_bits_rvlc,
        bytes_rvlc.len(),
        total_bits_std,
        bytes_std.len()
    );

    // ------------------------------------------------------------------
    // Corrupt a window of bytes covering the byte that contains the
    // middle of the partition. Both streams take damage in the same
    // RELATIVE region (around byte index `len/2`). A multi-byte
    // window guarantees the damage spans at least one codeword
    // boundary so the parser cannot just step over it.
    // ------------------------------------------------------------------
    fn corrupt(bytes: &[u8]) -> Vec<u8> {
        let mut out = bytes.to_vec();
        let mid = out.len() / 2;
        for off in 0..3.min(out.len() - mid) {
            out[mid + off] ^= 0xA5;
        }
        out
    }
    let corrupt_rvlc = corrupt(&bytes_rvlc);
    let corrupt_std = corrupt(&bytes_std);

    // ------------------------------------------------------------------
    // Forward decode + bit-exact comparison against the original blocks.
    // We count a block as "recovered" only when the decoded coefficient
    // array matches the original exactly. A best-effort walker that
    // emits garbage coefficients past the damage point gets credited
    // with zero past the damage.
    // ------------------------------------------------------------------
    fn forward_recover_rvlc(bytes: &[u8], total_bits: u64, originals: &[[i32; 64]]) -> Vec<bool> {
        let mut br = BitReader::new(bytes);
        let mut ok = vec![false; originals.len()];
        for (k, orig) in originals.iter().enumerate() {
            if br.bit_position() >= total_bits {
                break;
            }
            let mut decoded = [0i32; 64];
            match rvlc::try_decode_intra_ac(&mut br, &mut decoded, &ZIGZAG) {
                Ok(_) => ok[k] = decoded == *orig,
                Err(_) => break,
            }
        }
        ok
    }
    fn forward_recover_std(bytes: &[u8], total_bits: u64, originals: &[[i32; 64]]) -> Vec<bool> {
        let mut br = BitReader::new(bytes);
        let mut ok = vec![false; originals.len()];
        for (k, orig) in originals.iter().enumerate() {
            if br.bit_position() >= total_bits {
                break;
            }
            let mut decoded = [0i32; 64];
            match block::decode_intra_ac(&mut br, &mut decoded, &ZIGZAG) {
                Ok(_) => ok[k] = decoded == *orig,
                Err(_) => break,
            }
        }
        ok
    }

    // ------------------------------------------------------------------
    // Reverse decode for RVLC: bit-reverse the corrupted bytes, walk
    // forward through the reversed buffer block-by-block from the END
    // (= forward through the reversed buffer, but indexed from the
    // END of `originals`). Bit-exact comparison again.
    // ------------------------------------------------------------------
    fn reverse_recover_rvlc(
        bytes: &[u8],
        total_bits: u64,
        block_lens: &[u64],
        originals: &[[i32; 64]],
    ) -> Vec<bool> {
        let rev = rvlc::bit_reverse_buffer(bytes, total_bits as usize);
        let mut br = BitReader::new(&rev);
        let mut ok = vec![false; originals.len()];
        // Iterate from the LAST original block backward — block_lens
        // is in forward emission order so iterate in reverse.
        for (idx_from_end, &nb) in block_lens.iter().rev().enumerate() {
            let k = originals.len() - 1 - idx_from_end;
            let mut decoded = [0i32; 64];
            match rvlc::decode_intra_ac_reverse(&mut br, nb, &mut decoded, &ZIGZAG) {
                Ok(_) => ok[k] = decoded == originals[k],
                Err(_) => break,
            }
        }
        ok
    }

    let fwd_rvlc = forward_recover_rvlc(&corrupt_rvlc, total_bits_rvlc, &originals);
    let fwd_std = forward_recover_std(&corrupt_std, total_bits_std, &originals);
    let rev_rvlc =
        reverse_recover_rvlc(&corrupt_rvlc, total_bits_rvlc, &block_lens_rvlc, &originals);

    // Combined per-block recovery: RVLC = forward OR reverse; std = forward only.
    let combined_rvlc: Vec<bool> = fwd_rvlc
        .iter()
        .zip(rev_rvlc.iter())
        .map(|(f, r)| *f || *r)
        .collect();
    let combined_std = fwd_std.clone();

    let n_rvlc = combined_rvlc.iter().filter(|x| **x).count();
    let n_std = combined_std.iter().filter(|x| **x).count();

    eprintln!(
        "recovery: RVLC fwd {} + rev {} → combined {}/{} bit-exact; \
         std fwd {}/{} bit-exact",
        fwd_rvlc.iter().filter(|x| **x).count(),
        rev_rvlc.iter().filter(|x| **x).count(),
        n_rvlc,
        k_blocks,
        n_std,
        k_blocks
    );
    eprintln!("  RVLC fwd mask: {:?}", fwd_rvlc);
    eprintln!("  RVLC rev mask: {:?}", rev_rvlc);
    eprintln!("  std  fwd mask: {:?}", fwd_std);

    assert!(
        n_rvlc > n_std,
        "RVLC recovered {} bit-exact blocks vs std {} (k = {}); \
         expected RVLC > std",
        n_rvlc,
        n_std,
        k_blocks
    );
}

/// ffmpeg cross-decode: hand our DP+RVLC ES to ffmpeg and assert
/// (a) ffmpeg accepts the stream and (b) the decoded I-VOP matches our
/// source within 30 dB PSNR (same bar as the DP-only test).
#[test]
fn rvlc_ffmpeg_decode() {
    if !command_exists("ffmpeg") {
        eprintln!("ffmpeg missing — skipping RVLC ffmpeg interop test");
        return;
    }
    let (width, height) = (64u32, 64u32);
    let num_frames = 8u32;
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder(width, height, true);
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
    let es_path = tmp.join("oxideav_rvlc_ours.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_rvlc_ffmpeg.yuv");
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
        "ffmpeg failed to decode our DP+RVLC stream"
    );
    let ffmpeg_decoded = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame_bytes = (width as usize * height as usize * 3) / 2;
    assert!(
        ffmpeg_decoded.len() >= per_frame_bytes,
        "ffmpeg DP+RVLC output too small: {} bytes",
        ffmpeg_decoded.len()
    );
    let src0 = flatten_frame(&src_frames[0]);
    let ff0 = &ffmpeg_decoded[0..per_frame_bytes];
    let p0 = psnr(&src0, ff0);
    eprintln!("ffmpeg DP+RVLC decode frame 0 (I-VOP): PSNR = {p0:.2} dB");
    assert!(
        p0 > 30.0,
        "ffmpeg DP+RVLC I-VOP PSNR {p0:.2} dB below 30 dB"
    );
}

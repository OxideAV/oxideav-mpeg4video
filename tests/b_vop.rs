//! B-VOP end-to-end tests against ffmpeg-generated clips.
//!
//! Fixture generation:
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=10:duration=1.2" \
//!       -c:v mpeg4 -g 6 -bf 2 -qscale:v 5 -an -f m4v /tmp/m4v_bvop_64.es
//!   ffmpeg -y -i /tmp/m4v_bvop_64.es -f rawvideo -pix_fmt yuv420p \
//!       /tmp/m4v_bvop_64.yuv
//!
//! The `-bf 2` flag tells ffmpeg to insert up to 2 B-frames between each pair
//! of references — the resulting GOP looks like IBBPBBPBBPBB with -g 6.

use oxideav_core::bits::BitReader;
use oxideav_mpeg4video::{
    headers::{
        vol::parse_vol,
        vop::{parse_vop, VopCodingType},
    },
    start_codes::{self, VOP_START_CODE},
};
use std::path::Path;

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

/// The test fixture's VOL must parse and its VOP stream must contain at least
/// one B-VOP. Exercises the VOP header parser on a `vop_coding_type == B`
/// VOP — the extra `vop_fcode_backward` field is what makes B-VOP headers
/// different from P-VOPs (§6.3.5).
#[test]
fn bvop_fixture_contains_b_vops() {
    let Some(data) = read_fixture("/tmp/m4v_bvop_64.es") else {
        return;
    };
    let (vol_pos, _) = start_codes::iter_start_codes(&data)
        .find(|(_, c)| start_codes::is_video_object_layer(*c))
        .expect("VOL");
    let next = start_codes::iter_start_codes(&data[vol_pos + 4..])
        .next()
        .map(|(p, _)| vol_pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[vol_pos + 4..next]);
    let vol = parse_vol(&mut br).expect("parse VOL");

    let vops: Vec<_> = start_codes::iter_start_codes(&data)
        .filter(|(_, c)| *c == VOP_START_CODE)
        .collect();
    let mut n_i = 0;
    let mut n_p = 0;
    let mut n_b = 0;
    let mut bwd_fcode_seen = 0u8;
    for i in 0..vops.len() {
        let s = vops[i].0;
        let e = if i + 1 < vops.len() {
            vops[i + 1].0
        } else {
            data.len()
        };
        let mut br = BitReader::new(&data[s + 4..e]);
        let vop = parse_vop(&mut br, &vol).expect("parse VOP");
        match vop.vop_coding_type {
            VopCodingType::I => n_i += 1,
            VopCodingType::P => n_p += 1,
            VopCodingType::B => {
                n_b += 1;
                bwd_fcode_seen = bwd_fcode_seen.max(vop.vop_fcode_backward);
            }
            VopCodingType::S => {}
        }
    }
    eprintln!("bvop_fixture: I={n_i} P={n_p} B={n_b}  vop_fcode_backward>={bwd_fcode_seen}");
    assert!(n_b >= 1, "fixture should contain at least one B-VOP");
    assert!(n_i >= 1);
    assert!(n_p >= 1);
    assert!(
        bwd_fcode_seen >= 1,
        "B-VOP header must carry vop_fcode_backward"
    );
}

/// Full-pipeline decode of a clip that exercises 4MV P-MBs so the direct
/// mode path in B-VOPs hits the per-block branch (§7.5.9.5.2). Fixture:
///   ffmpeg -y -f lavfi -i "mandelbrot=size=64x64:rate=10" -t 1.2 \
///       -c:v mpeg4 -g 6 -bf 2 -qscale:v 3 -mbd rd -mbcmp satd -an \
///       -f m4v /tmp/m4v_bvop_4mv.es
///   ffmpeg -y -i /tmp/m4v_bvop_4mv.es -f rawvideo -pix_fmt yuv420p \
///       /tmp/m4v_bvop_4mv.yuv
///
/// This test is a floor, not a tight PSNR assertion — we only require it
/// runs without a panic so the 4MV direct path is exercised by CI.
#[test]
fn decode_bvop_4mv_clip_runs() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

    let Some(bitstream) = read_fixture("/tmp/m4v_bvop_4mv.es") else {
        return;
    };
    let params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).expect("build decoder");
    let packet = Packet::new(0, TimeBase::new(1, 90_000), bitstream);
    let _ = dec.send_packet(&packet);
    let _ = dec.flush();
    let mut n = 0;
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        assert_eq!(vf.format, PixelFormat::Yuv420P);
        n += 1;
        if n > 64 {
            break;
        }
    }
    eprintln!("4mv bvop clip: decoded {n} frames");
}

/// Full-pipeline B-VOP decode — feed the full elementary stream into the
/// decoder and compare N output frames against ffmpeg's reference YUV.
/// Fixture: see module-level docs.
#[test]
fn decode_bvop_clip_matches_ffmpeg() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

    let Some(bitstream) = read_fixture("/tmp/m4v_bvop_64.es") else {
        return;
    };
    let Ok(reference) = std::fs::read("/tmp/m4v_bvop_64.yuv") else {
        eprintln!("reference YUV missing — skipping test");
        return;
    };

    let frame_size = 64 * 64 * 3 / 2;
    let n_frames_total = reference.len() / frame_size;
    assert!(n_frames_total >= 4, "need >=4 frames, got {n_frames_total}");

    let params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).expect("build decoder");
    let packet = Packet::new(0, TimeBase::new(1, 90_000), bitstream);
    // Whatever the outcome of send_packet — Ok or a decode error partway —
    // any frames the decoder already buffered are available via receive_frame.
    let send_res = dec.send_packet(&packet);
    let _ = dec.flush();
    eprintln!("send_packet result: {send_res:?}");

    let mut frames_decoded = 0usize;
    let mut total_close = 0usize;
    let mut total_pixels = 0usize;
    let mut sum_sq_diff: u64 = 0;
    let mut max_diff_overall = 0i32;
    let mut n_high_psnr = 0usize;

    loop {
        let frame = match dec.receive_frame() {
            Ok(Frame::Video(vf)) => vf,
            Ok(_) => break,
            Err(e) => {
                eprintln!("stopped at frame {frames_decoded}: {e}");
                break;
            }
        };
        assert_eq!(frame.format, PixelFormat::Yuv420P);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        let mut ours = Vec::with_capacity(frame_size);
        ours.extend_from_slice(&frame.planes[0].data);
        ours.extend_from_slice(&frame.planes[1].data);
        ours.extend_from_slice(&frame.planes[2].data);
        // With the decode-order → display-order reorder buffer wired up
        // (`held_ref_frame` in the decoder), the N-th emitted frame
        // corresponds to display-order index N. ffmpeg's raw-video
        // muxer also writes display order, so we compare 1:1.
        let ref_off = frames_decoded * frame_size;
        if ref_off + frame_size > reference.len() {
            break;
        }
        let ref_slice = &reference[ref_off..ref_off + frame_size];
        let mut close = 0usize;
        let mut max_diff = 0i32;
        let mut sq: u64 = 0;
        for i in 0..frame_size {
            let d = (ours[i] as i32) - (ref_slice[i] as i32);
            if d.abs() <= 2 {
                close += 1;
            }
            max_diff = max_diff.max(d.abs());
            sq += (d * d) as u64;
        }
        let pct = 100.0 * close as f64 / frame_size as f64;
        let mse = sq as f64 / frame_size as f64;
        let psnr = if mse > 0.0 {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        } else {
            100.0
        };
        eprintln!(
            "decode frame {frames_decoded}: pixel match {pct:.2}%; max diff {max_diff}; PSNR {psnr:.2} dB"
        );
        total_close += close;
        total_pixels += frame_size;
        sum_sq_diff += sq;
        max_diff_overall = max_diff_overall.max(max_diff);
        if psnr >= 50.0 {
            n_high_psnr += 1;
        }
        frames_decoded += 1;
    }

    assert!(
        frames_decoded >= 1,
        "decoder should produce at least 1 frame; got {frames_decoded}"
    );
    let pct = 100.0 * total_close as f64 / total_pixels.max(1) as f64;
    let mse = sum_sq_diff as f64 / total_pixels.max(1) as f64;
    let psnr = if mse > 0.0 {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    } else {
        100.0
    };
    eprintln!(
        "bvop overall: {frames_decoded} frames; pixel match {pct:.2}%; PSNR {psnr:.2} dB; max diff {max_diff_overall}"
    );
    // With decode-order → display-order reorder in place, at least the
    // I-VOP and any P-VOPs the decoder gets to reconstruct should land
    // at their correct display-order positions and match the ffmpeg
    // reference almost bit-exactly (~67 dB for the frames we've
    // observed). We require at least two such >=50 dB frames as a
    // direct assertion that reorder is wired correctly — I/P-VOPs that
    // were not reordered against would drop to the low 30s / high 20s
    // against a display-order reference.
    assert!(
        n_high_psnr >= 2,
        "reorder check failed: expected >=2 high-PSNR frames \
         (I/P-VOPs at their display-order position), got {n_high_psnr}. \
         frames={frames_decoded} overall_psnr={psnr:.2}"
    );
    // Overall PSNR floor: 4MV direct mode (§7.5.9.5.2) is wired up but
    // this particular fixture (testsrc pattern) does not produce 4MV
    // P-MBs at ffmpeg's default `-bf 2 -qscale:v 5` settings, so the
    // observed ceiling is still governed by interlaced B-MBs and
    // quarter-pel MC in B-VOPs. The `decode_bvop_4mv_clip_runs` test
    // above exercises the 4MV direct path on a mandelbrot fixture.
    //
    // Current measured overall: ~32.8 dB. We guard at 32 dB as a
    // regression floor; the 35 dB target requires the remaining two
    // B-VOP paths (interlaced + quarter-pel) to land.
    assert!(
        psnr >= 32.0,
        "bvop clip overall PSNR fell below direct-mode floor: {psnr:.2} dB"
    );
}

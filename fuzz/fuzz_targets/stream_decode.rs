//! Whole elementary streams through `Mpeg4VideoDecoder`: configuration
//! headers, VOP headers, the I/P/B/S(GMC) walks (progressive and
//! interlaced), video-packet headers with HEC bodies, data partitioning
//! and RVLC recovery. A fixed VOS/VOL prefix is prepended in half of
//! the runs so the macroblock layers get exercised on small pictures.
//! A third run places the bytes behind a complete basic static-sprite
//! head (VOL + sprite I-VOP), so they become trajectory-only S-VOPs
//! driving the §7.8.4–§7.8.6 warps (affine family and perspective)
//! with arbitrary reference points.
#![no_main]

use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::ivop_encode::{EncoderConfig, FrameView, StaticSpriteVol};
use oxideav_mpeg4video::sprite_encode::StaticSpriteEncoder;
use oxideav_mpeg4video::vol::SpriteWarpingAccuracy;

/// Configuration headers + sprite I-VOP of a 32×16 static-sprite VOL
/// over a 40×24 sprite, per warping-point count (0..=4) and two warp
/// accuracies.
fn sprite_heads() -> &'static Vec<Vec<u8>> {
    static HEADS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    HEADS.get_or_init(|| {
        let (sw, sh) = (40usize, 24usize);
        let y: Vec<u8> = (0..sw * sh).map(|i| (i * 7 + (i / sw) * 31) as u8).collect();
        let c: Vec<u8> = (0..(sw / 2) * (sh / 2)).map(|i| (i * 5) as u8).collect();
        let view = FrameView {
            y: &y,
            cb: &c,
            cr: &c,
            width: sw,
            height: sh,
        };
        let mut heads = Vec::new();
        for accuracy in [SpriteWarpingAccuracy::HalfPel, SpriteWarpingAccuracy::SixteenthPel] {
            for points in 0..=4u8 {
                let cfg = EncoderConfig::default()
                    .with_width(32)
                    .with_height(16)
                    .with_static_sprite(Some(
                        StaticSpriteVol::new(sw as u16, sh as u16)
                            .with_origin(-4, 2)
                            .with_warping_points(points)
                            .with_accuracy(accuracy),
                    ));
                heads.push(StaticSpriteEncoder::new(&cfg, &view, 6).1);
            }
        }
        heads
    })
}

/// A 32×32 verid-2 VOL (quarter_sample off, resync markers enabled,
/// GMC 3 points) followed by an I-VOP start code: the fuzzer's bytes
/// become the VOP header + macroblock layer and any following units.
const PREFIX: &[u8] = &[
    0x00, 0x00, 0x01, 0xB0, 0xF3, // VOS, ASP@L3
    0x00, 0x00, 0x01, 0xB5, 0x09, // VO: video ID
    0x00, 0x00, 0x01, 0x00, // video_object 0
    0x00, 0x00, 0x01, 0x20, // VOL 0
];

fuzz_target!(|data: &[u8]| {
    if data.len() > 8192 {
        return;
    }
    let mut dec = Mpeg4VideoDecoder::new();
    // Raw bytes: exercises the scanners and every header parser.
    let _ = dec.decode(data);
    let _ = dec.flush();
    // Prefixed: the fuzzer shapes the VOL body and the VOPs.
    let mut prefixed = PREFIX.to_vec();
    prefixed.extend_from_slice(data);
    let mut dec = Mpeg4VideoDecoder::new();
    let _ = dec.decode(&prefixed);
    let _ = dec.flush();
    // Static sprite: the bytes are the units after the sprite I-VOP,
    // opening as an S-VOP (`vop_coding_type == 11`, time fields from
    // the first byte's low bits).
    if let Some((&select, body)) = data.split_first() {
        let heads = sprite_heads();
        let mut stream = heads[usize::from(select) % heads.len()].clone();
        stream.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6]);
        // `11` S, modulo_time_base `0`, marker, 5-bit increment,
        // marker, vop_coded 1, intra_dc_vlc_thr 000, then the
        // fuzzer's trajectory bits.
        stream.push(0b1101_0000 | ((select >> 4) & 0x0F));
        stream.push(0b0110_0000 | (body.first().copied().unwrap_or(0) & 0x0F));
        stream.extend_from_slice(body);
        let mut dec = Mpeg4VideoDecoder::new();
        let _ = dec.decode(&stream);
        let _ = dec.flush();
    }
});

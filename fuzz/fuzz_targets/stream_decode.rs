//! Whole elementary streams through `Mpeg4VideoDecoder`: configuration
//! headers, VOP headers, the I/P/B/S(GMC) walks (progressive and
//! interlaced), video-packet headers with HEC bodies, data partitioning
//! and RVLC recovery. A fixed VOS/VOL prefix is prepended in half of
//! the runs so the macroblock layers get exercised on small pictures.
//! A third run places the bytes behind a complete basic static-sprite
//! head (VOL + sprite I-VOP), so they become trajectory-only S-VOPs
//! driving the §7.8.4–§7.8.6 warps (affine family and perspective)
//! with arbitrary reference points.
//!
//! The decode options rotate with the input's first byte over the
//! literal-spec mode, the ecosystem-compat mode and both forced
//! inverse-transform selections, so every macroblock walk also runs
//! through the integer row–column transform. Two further runs aim at
//! the stream-evidence rule: the bytes as the payload of a user-data
//! unit (hostile signatures and digit runs), and the prefixed stream
//! behind a signed unit so a from-stream decoder switches transforms
//! before the fuzzer's VOPs.
#![no_main]

use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use oxideav_mpeg4video::compat::{DecodeOptions, InverseTransformSelection};
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
        let y: Vec<u8> = (0..sw * sh)
            .map(|i| (i * 7 + (i / sw) * 31) as u8)
            .collect();
        let c: Vec<u8> = (0..(sw / 2) * (sh / 2)).map(|i| (i * 5) as u8).collect();
        let view = FrameView {
            y: &y,
            cb: &c,
            cr: &c,
            width: sw,
            height: sh,
        };
        let mut heads = Vec::new();
        for accuracy in [
            SpriteWarpingAccuracy::HalfPel,
            SpriteWarpingAccuracy::SixteenthPel,
        ] {
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

/// The decode options one input runs with, from its first byte.
fn options_of(select: u8) -> DecodeOptions {
    match select % 6 {
        0 => DecodeOptions::spec(),
        1 => DecodeOptions::ecosystem(),
        2 => DecodeOptions::spec().with_inverse_transform(InverseTransformSelection::VariantRc),
        3 => {
            DecodeOptions::ecosystem().with_inverse_transform(InverseTransformSelection::VariantRc)
        }
        4 => DecodeOptions::spec().with_inverse_transform(InverseTransformSelection::FromStream),
        _ => DecodeOptions::ecosystem().with_inverse_transform(InverseTransformSelection::Exact),
    }
}

/// What the stream-evidence rule should make of a user-data payload,
/// written independently of the decoder: the signature at offset 0,
/// then a decimal digit run that fits 32 bits.
fn expected_build(payload: &[u8]) -> Option<u32> {
    // A user-data unit ends at the next start-code prefix.
    let end = payload
        .windows(3)
        .position(|w| w == [0, 0, 1])
        .unwrap_or(payload.len());
    let payload = &payload[..end];
    if payload.len() < 5 || &payload[..4] != b"XviD" {
        return None;
    }
    let mut value: u64 = 0;
    let mut digits = 0;
    for &b in &payload[4..] {
        if !b.is_ascii_digit() {
            break;
        }
        digits += 1;
        value = (value * 10 + u64::from(b - b'0')).min(1 << 40);
    }
    (digits > 0 && value <= u64::from(u32::MAX)).then_some(value as u32)
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 8192 {
        return;
    }
    let options = options_of(data.first().copied().unwrap_or(0));
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    // Raw bytes: exercises the scanners and every header parser.
    let _ = dec.decode(data);
    let _ = dec.flush();
    // Prefixed: the fuzzer shapes the VOL body and the VOPs.
    let mut prefixed = PREFIX.to_vec();
    prefixed.extend_from_slice(data);
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    let _ = dec.decode(&prefixed);
    let _ = dec.flush();
    // Signed: the same stream behind a matching user-data unit, under
    // the from-stream selection of the compat mode — the walks run on
    // the integer transform from the first VOP.
    let mut signed = b"\x00\x00\x01\xB2XviD0050".to_vec();
    signed.extend_from_slice(&prefixed);
    let mut dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
    let _ = dec.decode(&signed);
    let _ = dec.flush();
    // A later matching unit in the fuzzer's bytes may change the
    // reported build; nothing can withdraw it.
    assert!(dec.detected_encoder_build().is_some());
    assert_eq!(
        dec.active_inverse_transform(),
        InverseTransformSelection::VariantRc
    );
    // Hostile user data: the bytes as one unit's payload. The rule's
    // verdict matches the independent reading, the literal-spec
    // decoder never leaves the exact transform, and the compat decoder
    // switches exactly when a build was found.
    let mut unit = vec![0x00, 0x00, 0x01, 0xB2];
    unit.extend_from_slice(data);
    let no_later_unit = !data.windows(3).any(|w| w == [0, 0, 1]);
    for options in [DecodeOptions::spec(), DecodeOptions::ecosystem()] {
        let mut dec = Mpeg4VideoDecoder::with_options(options);
        let _ = dec.decode(&unit);
        if no_later_unit {
            let want = expected_build(data);
            assert_eq!(dec.detected_encoder_build(), want);
            let variant = options.ecosystem_compat && want.is_some();
            assert_eq!(
                dec.active_inverse_transform() == InverseTransformSelection::VariantRc,
                variant
            );
        }
        if !options.ecosystem_compat {
            assert_eq!(
                dec.active_inverse_transform(),
                InverseTransformSelection::Exact
            );
        }
    }
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
        let mut dec = Mpeg4VideoDecoder::with_options(options);
        let _ = dec.decode(&stream);
        let _ = dec.flush();
    }
});

//! Pure-Rust MPEG-4 Part 2 video (ISO/IEC 14496-2) decoder.
//!
//! Scope:
//! * VOS / Visual Object / Video Object Layer / Video Object Plane header
//!   parsing for Advanced Simple Profile (ASP) levels 1-5.
//! * `CodecParameters` population from the VOL.
//! * **I-VOP** decode — AC/DC prediction + H.263 / MPEG-4 dequantisation
//!   + IDCT.
//! * **P-VOP** decode — half-pel and quarter-pel motion compensation,
//!   single-MV mode (4MV path is implemented but rarely triggered by
//!   typical encoders), inter texture reconstruction, MV-median prediction
//!   with first-slice-line special cases, and skipped-MB pass-through.
//! * Video-packet resync markers (§6.3.5.2) — detect-and-consume with
//!   forward-MB-num validation to avoid false positives.
//! * One reference frame held in the decoder; refreshed by each
//!   I-VOP/P-VOP.
//!
//! * **B-VOP** decode — bidirectional prediction with Direct / Forward /
//!   Backward / Interpolated modes and co-located MV scaling (§7.6.5).
//!
//! * **Decode-order → display-order reorder** — one-slot reorder buffer
//!   inside the decoder holds each I/P reference until the next I/P
//!   arrives, so B-VOPs decoded between two references are released
//!   into the ready queue AHEAD of the newer reference. `flush()`
//!   drains the held reference at end-of-stream. Per-VOP display PTS is
//!   synthesised from the bitstream's `modulo_time_base` /
//!   `vop_time_increment` + VOL `vop_time_increment_resolution` and
//!   rescaled into the packet time base.
//!
//! * **GMC (Global Motion Compensation)** — VOL `sprite_enable == 2`,
//!   P-VOP-level `sprite_trajectory()` with 1..=4 warp points, per-MB
//!   `mcsel` flag selecting between local (translational) and global
//!   (warp) motion. Bilinear sampling of the warped reference per
//!   §7.7.5 / §7.7.6. `sprite_brightness_change == 1` not yet applied
//!   during reconstruction but the VLC (Table 11-33) is decoded.
//!
//! * **Interlaced field coding (§6.2.7.3, §7.6)** — VOL + VOP-level
//!   interlaced flags are parsed, MB-layer `interlaced_information()`
//!   consumes `dct_type`, `field_prediction`, and the four field-
//!   reference bits. Field-DCT intra/inter MBs run through an 8-line
//!   field-row reorder (`interlaced::field_dct_reorder_mb`), field-
//!   predicted P-VOP MBs sample the reference through
//!   `field_predict_luma_block`. B-VOP field-predicted MBs decode two
//!   MVs per direction (top + bottom field, §7.6.2.2) — the MVs are
//!   averaged for the frame-MV predictor chain and stored as
//!   `field_mvs_fwd`/`field_mvs_bwd` for downstream field sampling.
//!   B-VOP `dct_type == 1` residuals run through
//!   `field_dct_reorder_mb` before being added to the MC output.
//!   `alternate_vertical_scan_flag == 1` routes all AC decode (intra
//!   and inter) through the alt-vertical scan.
//!
//! Header-level parse coverage (no reconstruction yet):
//! * **Static-sprite VOL** — `sprite_enable == 1` parses the full
//!   Table 6-7 payload (rectangle + warping + brightness +
//!   low_latency) and exposes `SpriteRect`. S-VOP decode still returns
//!   `Unsupported` at the VOP layer.
//!
//! Out of scope (returns `Unsupported`):
//! * S-VOPs / static-sprite reconstruction (`sprite_enable == 1`) —
//!   the I-VOP canvas + S-VOP warp onto canvas is not yet decoded.
//! * Scalability, data partitioning, reversible VLCs.
//! * MPEG-4 Studio / AVC Simple profiles.
//! * Encoder: I-VOPs only — P / B / S VOPs are out of scope (§6.2.5).
//!
//! The crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(feature = "nightly", feature(portable_simd))]

pub mod bits_ext;
pub mod block;
pub mod bvop;
pub mod decoder;
pub mod encoder;
pub mod gmc;
pub mod headers;
pub mod inter;
pub mod interlaced;
pub mod iq;
pub mod mb;
pub mod mc;
pub mod pvop;
pub mod resync;
pub mod simd;
pub mod start_codes;
pub mod tables;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag, ProbeContext};

/// The canonical oxideav codec id for MPEG-4 Part 2 video.
///
/// Note: this matches the ISO standard name. Container-level FourCCs like
/// `XVID`, `DIVX`, `DX50`, `MP4V`, `FMP4` are all this codec.
pub const CODEC_ID_STR: &str = "mpeg4video";

/// Probe used by the codec registry: returns a confidence in
/// `0.0..=1.0` that the context describes an MPEG-4 Part 2 stream.
/// Returns `1.0` when `0x000001` + a VOS / VOL / VOP / GOV / user-data
/// marker byte is present in the inspected header or packet bytes,
/// `0.0` when bytes were supplied and no start code was found, and
/// `0.5` when no bytes are available (weak evidence — the tag itself
/// already suggests this codec).
pub fn probe_is_mpeg4_part2(ctx: &ProbeContext) -> f32 {
    match ctx.packet.or(ctx.header) {
        Some(d) if has_start_code(d) => 1.0,
        Some(_) => 0.0,
        None => 0.5,
    }
}

/// Exposed for the decoder's first-packet sniff — inspects raw packet
/// bytes directly rather than going through a [`ProbeContext`].
pub(crate) fn has_packet_start_code(data: &[u8]) -> bool {
    has_start_code(data)
}

fn has_start_code(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let scan = &data[..data.len().min(2048)];
    for w in scan.windows(4) {
        if w[0] == 0x00 && w[1] == 0x00 && w[2] == 0x01 {
            let marker = w[3];
            // Visual-Object-Sequence / VO / VOL / VOP / GOV / user-data.
            if matches!(marker, 0xB0..=0xBF) || (0x20..=0x2F).contains(&marker) {
                return true;
            }
        }
    }
    false
}

/// Register this codec's decoder + I/P-VOP encoder with a registry,
/// including the container tags it claims.
///
/// The codec is registered twice:
///   * First registration — decoder + encoder + the unambiguous ISO
///     FourCCs (no probe, confidence defaults to 1.0).
///   * Second registration — tag-only, covers the mislabelling-prone
///     DIV3-family FourCCs with `probe_is_mpeg4_part2` so DIV3 files
///     that actually hold ISO Part 2 bytes resolve here, while files
///     that hold MS-MPEG4 bytes lose to `oxideav-msmpeg4`'s probe.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mpeg4video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(4096, 4096);
    let id = CodecId::new(CODEC_ID_STR);

    // Unambiguous MPEG-4 Part 2 FourCCs — probe omitted (confidence 1.0).
    reg.register(
        CodecInfo::new(id.clone())
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tags([
                CodecTag::fourcc(b"XVID"),
                CodecTag::fourcc(b"DIVX"),
                CodecTag::fourcc(b"DX50"),
                CodecTag::fourcc(b"MP4V"),
                CodecTag::fourcc(b"FMP4"),
                CodecTag::fourcc(b"3IV2"),
                CodecTag::fourcc(b"M4S2"),
                CodecTag::fourcc(b"MP4S"),
                CodecTag::fourcc(b"DIVF"),
                CodecTag::fourcc(b"BLZ0"),
                CodecTag::fourcc(b"DX40"),
                CodecTag::fourcc(b"RMP4"),
                CodecTag::fourcc(b"SMP4"),
                CodecTag::fourcc(b"UMP4"),
                CodecTag::fourcc(b"WV1F"),
                CodecTag::fourcc(b"XVIX"),
                CodecTag::fourcc(b"DXGM"),
            ]),
    );

    // Mislabelling-fallback claims: the MS-MPEG4 FourCC pool. Tag-only
    // (no factories — those are already registered above). The probe
    // returns 1.0 only when ISO start codes are present, letting
    // msmpeg4's mirror probe win on genuine MS-MPEG4 bytes.
    reg.register(CodecInfo::new(id).probe(probe_is_mpeg4_part2).tags([
        CodecTag::fourcc(b"DIV3"),
        CodecTag::fourcc(b"DIV4"),
        CodecTag::fourcc(b"DIV5"),
        CodecTag::fourcc(b"DIV6"),
        CodecTag::fourcc(b"MP43"),
        CodecTag::fourcc(b"MPG3"),
        CodecTag::fourcc(b"AP41"),
    ]));
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    fn ctx<'a>(tag: &'a CodecTag, data: &'a [u8]) -> ProbeContext<'a> {
        ProbeContext::new(tag).packet(data)
    }

    #[test]
    fn probe_accepts_vos_start_code() {
        let data = [0x00, 0x00, 0x01, 0xB0, 0x01, 0x00];
        let t = CodecTag::fourcc(b"DIV3");
        assert!(probe_is_mpeg4_part2(&ctx(&t, &data)) >= 0.99);
    }

    #[test]
    fn probe_accepts_vol_start_code() {
        let data = [0x00, 0x00, 0x01, 0x20, 0x08, 0x08, 0x40];
        let t = CodecTag::fourcc(b"DIV3");
        assert!(probe_is_mpeg4_part2(&ctx(&t, &data)) >= 0.99);
    }

    #[test]
    fn probe_accepts_vop_start_code() {
        let data = [0x00, 0x00, 0x01, 0xB6, 0x40, 0x00];
        let t = CodecTag::fourcc(b"DIV3");
        assert!(probe_is_mpeg4_part2(&ctx(&t, &data)) >= 0.99);
    }

    #[test]
    fn probe_rejects_msmpeg4_picture_header() {
        let data = [0x85u8, 0x3F, 0xD4, 0x80, 0x00, 0xA2, 0x10, 0xFF];
        let t = CodecTag::fourcc(b"DIV3");
        assert!(probe_is_mpeg4_part2(&ctx(&t, &data)) < 0.01);
    }

    #[test]
    fn probe_rejects_too_short_bytes() {
        let t = CodecTag::fourcc(b"DIV3");
        assert!(probe_is_mpeg4_part2(&ctx(&t, &[0x00, 0x00, 0x01])) < 0.01);
        assert!(probe_is_mpeg4_part2(&ctx(&t, &[])) < 0.01);
    }

    #[test]
    fn probe_without_data_is_middling() {
        let t = CodecTag::fourcc(b"DIV3");
        let c = ProbeContext::new(&t);
        let conf = probe_is_mpeg4_part2(&c);
        assert!((0.0..=1.0).contains(&conf));
        assert!(conf > 0.0 && conf < 1.0);
    }

    #[test]
    fn registered_tag_claims_route_correctly() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        // XVID: unambiguous ISO FourCC, resolves to mpeg4video.
        let xvid = CodecTag::fourcc(b"XVID");
        assert_eq!(
            reg.resolve_tag_ref(&ProbeContext::new(&xvid))
                .map(|c| c.as_str()),
            Some(CODEC_ID_STR),
        );
        // DIV3 with ISO bytes: probe returns 1.0.
        let iso = [0x00u8, 0x00, 0x01, 0xB0, 0x01, 0x00];
        let div3 = CodecTag::fourcc(b"DIV3");
        let c = ProbeContext::new(&div3).packet(&iso);
        assert_eq!(
            reg.resolve_tag_ref(&c).map(|c| c.as_str()),
            Some(CODEC_ID_STR),
        );
        // DIV3 with MS-MPEG4 bytes: probe returns 0.0, no other
        // claim in this crate, resolve returns None. (The msmpeg4
        // crate's mirror probe would win when registered alongside.)
        let ms = [0x85u8, 0x3F, 0xD4, 0x80, 0x00, 0xA2];
        let c = ProbeContext::new(&div3).packet(&ms);
        assert!(reg.resolve_tag_ref(&c).is_none());
    }
}


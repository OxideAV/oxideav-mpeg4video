//! Pure-Rust MPEG-4 Part 2 video (ISO/IEC 14496-2) decoder.
//!
//! Scope:
//! * VOS / Visual Object / Video Object Layer / Video Object Plane header
//!   parsing for Advanced Simple Profile (ASP) levels 1-5.
//! * `CodecParameters` population from the VOL.
//! * **I-VOP** decode — AC/DC prediction + H.263 / MPEG-4 dequantisation
//!   + IDCT.
//! * **P-VOP** decode — half-pel motion compensation, single-MV mode (4MV
//!   path is implemented but rarely triggered by typical encoders), inter
//!   texture reconstruction, MV-median prediction with first-slice-line
//!   special cases, and skipped-MB pass-through.
//! * Video-packet resync markers (§6.3.5.2) — detect-and-consume with
//!   forward-MB-num validation to avoid false positives.
//! * One reference frame held in the decoder; refreshed by each
//!   I-VOP/P-VOP.
//!
//! Out of scope (returns `Unsupported`):
//! * B-VOPs (bidirectional prediction).
//! * S-VOPs (sprites), GMC.
//! * Quarter-pel motion (`quarter_sample` rejected at VOL parse time).
//! * Interlaced field coding, scalability, data partitioning, reversible
//!   VLCs.
//! * MPEG-4 Studio / AVC Simple profiles.
//! * Encoder: I-VOPs only — P / B / S VOPs are out of scope (§6.2.5).
//!
//! The crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bits_ext;
pub mod block;
pub mod decoder;
pub mod encoder;
pub mod headers;
pub mod inter;
pub mod iq;
pub mod mb;
pub mod mc;
pub mod pvop;
pub mod resync;
pub mod start_codes;
pub mod tables;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// The canonical oxideav codec id for MPEG-4 Part 2 video.
///
/// Note: this matches the ISO standard name. Container-level FourCCs like
/// `XVID`, `DIVX`, `DX50`, `MP4V`, `FMP4` are all this codec.
pub const CODEC_ID_STR: &str = "mpeg4video";

/// Probe used by the codec registry: accepts any bitstream that
/// carries an MPEG-4 Part 2 start-code prefix (`0x000001` followed
/// by a Visual-Object-Sequence / VOL / VOP / GOV / user-data marker
/// byte) in its first 2 KB. Used to catch the "DIV3 FourCC on an
/// actually-ISO stream" mislabel case — the registry's priority walk
/// tries msmpeg4v3's claim first (higher priority), and when its
/// probe rejects the ISO bytes we fall through to this one.
pub fn probe_is_mpeg4_part2(data: &[u8]) -> bool {
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
/// **Tag ownership:**
///
/// * Priority 10 (unambiguous ISO FourCCs, no probe needed): `XVID`,
///   `DIVX`, `DX50`, `MP4V`, `FMP4`, `3IV2`, `M4S2`, `MP4S`, `DIVF`,
///   `BLZ0`, `DX40`, `RMP4`, `SMP4`, `UMP4`, `WV1F`, `XVIX`, `DXGM`.
/// * Priority 5 (mislabelling fallback with probe): the MS-MPEG4 FourCC
///   pool `DIV3`, `DIV4`, `DIV5`, `DIV6`, `MP43`, `MPG3`, `AP41`. The
///   `oxideav-msmpeg4` crate claims the same FourCCs at priority 10
///   with a "bitstream has no ISO start code" probe, so the registry
///   resolves DIV3 files with ACTUAL MS-MPEG4 data to msmpeg4v3 and
///   DIV3 files mislabelled over real MPEG-4 Part 2 to this crate.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mpeg4video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(4096, 4096);
    let id = CodecId::new(CODEC_ID_STR);
    reg.register_decoder_impl(id.clone(), caps.clone(), decoder::make_decoder);
    // Encoder produces I-VOPs + P-VOPs; advertise non-intra-only.
    reg.register_encoder_impl(id.clone(), caps, encoder::make_encoder);

    // Unambiguous MPEG-4 Part 2 FourCCs — priority 10, no probe needed.
    for fcc in &[
        b"XVID", b"DIVX", b"DX50", b"MP4V", b"FMP4", b"3IV2", b"M4S2", b"MP4S", b"DIVF", b"BLZ0",
        b"DX40", b"RMP4", b"SMP4", b"UMP4", b"WV1F", b"XVIX", b"DXGM",
    ] {
        reg.claim_tag(id.clone(), CodecTag::fourcc(fcc), 10, None);
    }

    // Mislabelling-fallback claims: the MS-MPEG4 FourCC pool. Lower
    // priority than oxideav-msmpeg4's claims, with a probe that
    // accepts only genuine ISO bitstreams.
    for fcc in &[
        b"DIV3", b"DIV4", b"DIV5", b"DIV6", b"MP43", b"MPG3", b"AP41",
    ] {
        reg.claim_tag(
            id.clone(),
            CodecTag::fourcc(fcc),
            5,
            Some(probe_is_mpeg4_part2),
        );
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn probe_accepts_vos_start_code() {
        let data = [0x00, 0x00, 0x01, 0xB0, 0x01, 0x00];
        assert!(probe_is_mpeg4_part2(&data));
    }

    #[test]
    fn probe_accepts_vol_start_code() {
        let data = [0x00, 0x00, 0x01, 0x20, 0x08, 0x08, 0x40];
        assert!(probe_is_mpeg4_part2(&data));
    }

    #[test]
    fn probe_accepts_vop_start_code() {
        let data = [0x00, 0x00, 0x01, 0xB6, 0x40, 0x00];
        assert!(probe_is_mpeg4_part2(&data));
    }

    #[test]
    fn probe_rejects_msmpeg4_picture_header() {
        let data = [0x85u8, 0x3F, 0xD4, 0x80, 0x00, 0xA2, 0x10, 0xFF];
        assert!(!probe_is_mpeg4_part2(&data));
    }

    #[test]
    fn probe_rejects_too_short() {
        assert!(!probe_is_mpeg4_part2(&[0x00, 0x00, 0x01]));
        assert!(!probe_is_mpeg4_part2(&[]));
    }

    #[test]
    fn registered_tag_claims_route_correctly() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        // XVID: unambiguous ISO FourCC, resolves to mpeg4video without
        // needing probe data.
        assert_eq!(
            reg.resolve_tag(&CodecTag::fourcc(b"XVID"), None)
                .map(|c| c.as_str()),
            Some(CODEC_ID_STR),
        );
        // DIV3 with ISO bytes: falls into the mislabel-fallback claim.
        let iso = [0x00u8, 0x00, 0x01, 0xB0, 0x01, 0x00];
        assert_eq!(
            reg.resolve_tag(&CodecTag::fourcc(b"DIV3"), Some(&iso))
                .map(|c| c.as_str()),
            Some(CODEC_ID_STR),
        );
        // DIV3 with MS-MPEG4 bytes: probe rejects, no other claim in
        // this crate, resolve returns None. (In practice the registry
        // would also hold msmpeg4v3's claim at higher priority, which
        // would accept the bytes.)
        let ms = [0x85u8, 0x3F, 0xD4, 0x80, 0x00, 0xA2];
        assert!(reg
            .resolve_tag(&CodecTag::fourcc(b"DIV3"), Some(&ms))
            .is_none());
    }
}

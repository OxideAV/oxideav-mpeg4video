//! Whole elementary streams through `Mpeg4VideoDecoder`: configuration
//! headers, VOP headers, the I/P/B/S(GMC) walks (progressive and
//! interlaced), video-packet headers with HEC bodies, data partitioning
//! and RVLC recovery. A fixed VOS/VOL prefix is prepended in half of
//! the runs so the macroblock layers get exercised on small pictures.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;

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
});

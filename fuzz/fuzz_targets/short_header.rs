//! §6.2.5.2 short-header parse + decode on arbitrary bytes: the
//! picture-header parser, the GOB / macroblock walk (every source
//! format, GOB headers present or absent, `pei`/`psupp` runs, Type-4
//! escapes, intra DC FLC edge values) and the stream decoder's
//! VOL-less auto-detection must never panic or loop.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_mpeg4video::bitreader::BitReader;
use oxideav_mpeg4video::decoder::Mpeg4VideoDecoder;
use oxideav_mpeg4video::short_header::{
    decode_short_header_macroblocks, parse_short_header_picture, SHORT_VIDEO_START_MARKER,
};

fuzz_target!(|data: &[u8]| {
    // Direct parser + macroblock walk with the marker prepended so the
    // fuzzer spends its budget past the 22-bit gate.
    let mut unit = vec![0u8, 0u8, 0x80u8];
    unit.extend_from_slice(data);
    if let Some(first) = unit.get_mut(2) {
        *first |= ((SHORT_VIDEO_START_MARKER & 0x3F) << 2) as u8 & 0x80;
    }
    let mut br = BitReader::new(&unit);
    if let Ok(pic) = parse_short_header_picture(&mut br) {
        // Bound the walk to the small formats: 4CIF / 16CIF pictures
        // are too large to reconstruct in a fuzz iteration.
        let (w, h) = pic.source_format.dimensions();
        if u32::from(w) * u32::from(h) <= 352 * 288 {
            let _ = decode_short_header_macroblocks(&mut br, &pic);
        }
    }
    // Whole-stream path: auto-detection + reference chain.
    if data.len() <= 4096 {
        let mut dec = Mpeg4VideoDecoder::new();
        let _ = dec.decode(&unit);
        let _ = dec.flush();
    }
});

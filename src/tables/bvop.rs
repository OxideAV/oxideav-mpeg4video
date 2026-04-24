//! B-VOP VLC tables (ISO/IEC 14496-2:2004 §7.6.5 / Table B.3 / Table B.4).
//!
//! * **MODB** (Table B.3, final 2004 edition) — 1- or 2-bit prefix per B-MB.
//!   - `1`    → no mb_type, no cbpb — macroblock takes the default
//!     (direct-mode forward+backward prediction with zero delta,
//!     no residual). Per §7.6.9.6: "If the modb equals to '1' the
//!     current B-macroblock is reconstructed by using the direct mode
//!     with zero delta vector."
//!   - `01`   → mb_type present, cbpb absent (all-zero cbpb).
//!   - `00`   → mb_type present AND cbpb present.
//!
//!   The syntax at §6.2.7 states:
//!     - if (modb != '1')  mb_type     (i.e. mb_type read iff bits start with 0)
//!     - if (modb == '00') cbpb        (i.e. cbpb read iff TWO leading 0 bits)
//!
//!   NOTE: The 2nd-edition committee draft Table 11-3 used the inverted
//!   codes (`0` / `10` / `11`). The 2004 final standard flipped the
//!   prefixes; real ffmpeg-encoded streams conform to the 2004 codes, so
//!   decoding with the draft codes loses sync on any non-direct B-MB.
//!
//!   Our decoded MODB value:
//!     * 0 → skipped / default (no mbtype, no cbpb) — bit "1"
//!     * 1 → mbtype only                            — bits "01"
//!     * 2 → mbtype + cbpb                          — bits "00"
//!
//! * **MBTYPE** (Table B.4, non-scalable B-VOPs) — 1..=4 bit prefix per
//!   non-skipped B-MB. Unchanged between draft and 2004 spec.
//!   - `1`     → direct (scaled co-located MV plus optional delta)
//!   - `01`    → interpolate MC+Q (mvd_forward AND mvd_backward)
//!   - `001`   → backward MC+Q  (mvd_backward only)
//!   - `0001`  → forward MC+Q   (mvd_forward only)
//!
//!   Our decoded MBTYPE value:
//!     * 0 → Direct
//!     * 1 → Interpolated
//!     * 2 → Backward
//!     * 3 → Forward

use std::sync::OnceLock;

use crate::tables::vlc::VlcEntry;

/// MODB value codes.
pub const MODB_SKIPPED: u8 = 0;
pub const MODB_MBTYPE_ONLY: u8 = 1;
pub const MODB_MBTYPE_CBPB: u8 = 2;

/// MBTYPE value codes.
pub const MBTYPE_DIRECT: u8 = 0;
pub const MBTYPE_INTERPOLATED: u8 = 1;
pub const MBTYPE_BACKWARD: u8 = 2;
pub const MBTYPE_FORWARD: u8 = 3;

const MODB_ROWS: [(u8, u32, u8); 3] = [
    (1, 0b1, MODB_SKIPPED),
    (2, 0b01, MODB_MBTYPE_ONLY),
    (2, 0b00, MODB_MBTYPE_CBPB),
];

const MBTYPE_ROWS: [(u8, u32, u8); 4] = [
    (1, 0b1, MBTYPE_DIRECT),
    (2, 0b01, MBTYPE_INTERPOLATED),
    (3, 0b001, MBTYPE_BACKWARD),
    (4, 0b0001, MBTYPE_FORWARD),
];

pub fn modb_table() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        MODB_ROWS
            .iter()
            .map(|&(b, c, v)| VlcEntry::new(b, c, v))
            .collect()
    })
    .as_slice()
}

pub fn mbtype_table() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        MBTYPE_ROWS
            .iter()
            .map(|&(b, c, v)| VlcEntry::new(b, c, v))
            .collect()
    })
    .as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::vlc;
    use oxideav_core::bits::BitReader;

    #[test]
    fn modb_decodes_all_variants() {
        // 2004 Table B.3 codes:
        // "1"  → skipped / default (direct, zero delta, no residual)
        let mut br = BitReader::new(&[0x80]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_SKIPPED);
        // "01" → mbtype only
        let mut br = BitReader::new(&[0x40]);
        assert_eq!(
            vlc::decode(&mut br, modb_table()).unwrap(),
            MODB_MBTYPE_ONLY
        );
        // "00" → mbtype + cbpb
        let mut br = BitReader::new(&[0x00]);
        assert_eq!(
            vlc::decode(&mut br, modb_table()).unwrap(),
            MODB_MBTYPE_CBPB
        );
    }

    #[test]
    fn mbtype_decodes_all_variants() {
        // "1" → direct
        let mut br = BitReader::new(&[0x80]);
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_DIRECT);
        // "01" → interpolated
        let mut br = BitReader::new(&[0x40]);
        assert_eq!(
            vlc::decode(&mut br, mbtype_table()).unwrap(),
            MBTYPE_INTERPOLATED
        );
        // "001" → backward
        let mut br = BitReader::new(&[0x20]);
        assert_eq!(
            vlc::decode(&mut br, mbtype_table()).unwrap(),
            MBTYPE_BACKWARD
        );
        // "0001" → forward
        let mut br = BitReader::new(&[0x10]);
        assert_eq!(
            vlc::decode(&mut br, mbtype_table()).unwrap(),
            MBTYPE_FORWARD
        );
    }
}

//! B-VOP VLC tables (ISO/IEC 14496-2 §7.6.5 / Table 11-3 / Table 11-4).
//!
//! * **MODB** (Table 11-3) — 1- or 2-bit prefix per B-MB.
//!   - `0`    → no mb_type, no cbpb — macroblock takes the default
//!     (direct-mode forward+backward prediction with zero delta,
//!     no residual).
//!   - `10`   → mb_type present, cbpb absent (all-zero cbpb).
//!   - `11`   → mb_type present AND cbpb present.
//!
//!   Our decoded MODB value:
//!     * 0 → skipped / default (no mbtype, no cbpb)
//!     * 1 → mbtype only
//!     * 2 → mbtype + cbpb
//!
//! * **MBTYPE** (Table 11-4, non-scalable B-VOPs) — 1..=4 bit prefix per
//!   non-skipped B-MB.
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
    (1, 0b0, MODB_SKIPPED),
    (2, 0b10, MODB_MBTYPE_ONLY),
    (2, 0b11, MODB_MBTYPE_CBPB),
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
        // "0" → skipped / default
        let mut br = BitReader::new(&[0x00]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_SKIPPED);
        // "10" → mbtype only
        let mut br = BitReader::new(&[0x80]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_MBTYPE_ONLY);
        // "11" → mbtype + cbpb
        let mut br = BitReader::new(&[0xC0]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_MBTYPE_CBPB);
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
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_FORWARD);
    }
}

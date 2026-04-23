//! B-VOP VLC tables (ISO/IEC 14496-2 §7.6.5 / Annex B).
//!
//! * **MODB** (Table B-16) — 1- or 2-bit prefix per B-MB.
//!   - `1`    → skipped (no MBTYPE, no CBPB, MV inherited via direct mode at 0).
//!   - `00`   → MBTYPE follows AND CBPB follows.
//!   - `01`   → MBTYPE follows, CBPB absent (all-zero cbpb).
//!
//!   Our decoded MODB value:
//!     * 0 → skipped
//!     * 1 → "01" case (mbtype only)
//!     * 2 → "00" case (mbtype + cbpb)
//!
//! * **MBTYPE** (Table B-18) — 1..=4 bit prefix per non-skipped B-MB.
//!   - `1`     → Interpolated (mvd_forward AND mvd_backward)
//!   - `01`    → Backward only (mvd_backward)
//!   - `001`   → Forward only (mvd_forward)
//!   - `0001`  → Direct (scaled co-located MV + optional mvd_forward)
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
    (1, 0b1, MBTYPE_INTERPOLATED),
    (2, 0b01, MBTYPE_BACKWARD),
    (3, 0b001, MBTYPE_FORWARD),
    (4, 0b0001, MBTYPE_DIRECT),
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
        // "1" → skipped
        let mut br = BitReader::new(&[0x80]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_SKIPPED);
        // "01" → mbtype only
        let mut br = BitReader::new(&[0x40]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_MBTYPE_ONLY);
        // "00" → mbtype + cbpb
        let mut br = BitReader::new(&[0x00]);
        assert_eq!(vlc::decode(&mut br, modb_table()).unwrap(), MODB_MBTYPE_CBPB);
    }

    #[test]
    fn mbtype_decodes_all_variants() {
        let mut br = BitReader::new(&[0x80]);
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_INTERPOLATED);
        let mut br = BitReader::new(&[0x40]);
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_BACKWARD);
        let mut br = BitReader::new(&[0x20]);
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_FORWARD);
        let mut br = BitReader::new(&[0x10]);
        assert_eq!(vlc::decode(&mut br, mbtype_table()).unwrap(), MBTYPE_DIRECT);
    }
}

//! Minimal MSB-first bit writer used by the encoder's §6.2 syntax
//! emitters.
//!
//! Mirror of [`crate::bitreader::BitReader`]: ISO/IEC 14496-2 writes
//! integers most-significant-bit-first inside each byte (`bslbf` /
//! `uimsbf`, §6.1.1), and this writer produces exactly that layout.
//! Besides raw `write_bits`, it implements the §5.2.4
//! `next_start_code()` stuffing discipline — one `0` bit followed by
//! `1` bits up to the next byte boundary — which every syntactic unit
//! emits before the following start code (and which
//! [`crate::bitreader::BitReader`]-side consumers skip transparently).
//!
//! Provenance: behaviour is derived from the bit-stream conventions in
//! §5.2.3/§5.2.4 and §6.1.1 of ISO/IEC 14496-2:2004 (3rd edition),
//! read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.

/// MSB-first bit writer.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Number of valid bits already written. `bit_len / 8` full bytes
    /// live in `bytes`; when `bit_len % 8 != 0` the trailing partial
    /// byte is the last element of `bytes` with its unused low bits
    /// zero.
    bit_len: usize,
}

impl BitWriter {
    /// Fresh, empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of bits written so far.
    pub fn bit_position(&self) -> usize {
        self.bit_len
    }

    /// Whether the current position lies on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_len % 8 == 0
    }

    /// Append the low `n` bits of `value` (1..=32), most significant
    /// first — the §6.1.1 `uimsbf` layout. `n == 0` writes nothing.
    ///
    /// # Panics
    ///
    /// Panics if `n > 32` or if `value` has bits set above bit `n - 1`
    /// (an encoder-side value that does not fit its declared field
    /// width is always a caller bug worth failing loudly on).
    pub fn write_bits(&mut self, value: u32, n: usize) {
        assert!(n <= 32, "write_bits: n = {n} exceeds 32");
        if n < 32 {
            assert!(
                value < (1u32 << n),
                "write_bits: value {value:#x} does not fit in {n} bits"
            );
        }
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            if self.bit_len % 8 == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.last_mut().expect("push above guarantees a byte");
            *last |= bit << (7 - (self.bit_len % 8));
            self.bit_len += 1;
        }
    }

    /// Append a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(u32::from(bit), 1);
    }

    /// Append every bit written so far into `other` (bit-exact, no
    /// alignment) — how the partition writers of a video packet are
    /// spliced into the unit.
    pub fn append(&mut self, other: &BitWriter) {
        let full = other.bit_len / 8;
        for &byte in &other.bytes[..full] {
            self.write_bits(u32::from(byte), 8);
        }
        let rem = other.bit_len % 8;
        if rem != 0 {
            let tail = other.bytes[full] >> (8 - rem);
            self.write_bits(u32::from(tail), rem);
        }
    }

    /// Append a `marker_bit` (always `1`, §6.2.x).
    pub fn write_marker(&mut self) {
        self.write_bits(1, 1);
    }

    /// §5.2.4 `next_start_code()` stuffing: one `0` bit, then `1` bits
    /// until byte-aligned. From an aligned position this emits the full
    /// `01111111` stuffing byte (`0x7F`), exactly as the pseudo-code's
    /// unconditional `zero_bit` implies; from `k` bits into a byte it
    /// emits `8 - k` stuffing bits.
    pub fn next_start_code(&mut self) {
        self.write_bits(0, 1);
        while !self.is_byte_aligned() {
            self.write_bits(1, 1);
        }
    }

    /// Pad with **zero** bits to the next byte boundary (a no-op when
    /// already aligned) — the §6.2.5.2 short-header alignment rule
    /// ("zero to seven zero-valued bits") before
    /// `short_video_start_marker`, `gob_resync_marker` and
    /// `short_video_end_marker`.
    pub fn align_zero(&mut self) {
        while !self.is_byte_aligned() {
            self.write_bits(0, 1);
        }
    }

    /// Append a 32-bit start code (`0x000001xx`) — caller must be
    /// byte-aligned (emit [`Self::next_start_code`] first).
    ///
    /// # Panics
    ///
    /// Panics when unaligned: a misaligned start code is never valid.
    pub fn write_start_code(&mut self, code: u32) {
        assert!(
            self.is_byte_aligned(),
            "start code must be byte-aligned (call next_start_code first)"
        );
        self.write_bits(code, 32);
    }

    /// Consume the writer and return the bytes. The caller is
    /// responsible for having stuffed to a byte boundary
    /// ([`Self::next_start_code`]); any trailing partial byte is
    /// zero-padded (which a conformant emitter never leaves behind).
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Borrow the bytes written so far (trailing partial byte included,
    /// zero-padded).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn writes_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011, 4);
        w.write_bits(0b0000_0101, 8);
        assert_eq!(w.as_bytes(), &[0b1011_0000, 0b0101_0000]);
        assert_eq!(w.bit_position(), 12);
    }

    #[test]
    fn round_trips_through_bitreader() {
        let mut w = BitWriter::new();
        w.write_bits(0x12_34_56, 24);
        w.write_bits(5, 3);
        w.write_bits(0, 1);
        w.write_bits(0xFFF, 12);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(24).unwrap(), 0x12_34_56);
        assert_eq!(r.read_bits(3).unwrap(), 5);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.read_bits(12).unwrap(), 0xFFF);
    }

    #[test]
    fn next_start_code_from_aligned_emits_stuffing_byte() {
        let mut w = BitWriter::new();
        w.write_bits(0xAB, 8);
        w.next_start_code();
        assert_eq!(w.as_bytes(), &[0xAB, 0b0111_1111]);
    }

    #[test]
    fn next_start_code_mid_byte_pads_to_boundary() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.next_start_code();
        // 101 + 0 + 1111 = 1010_1111.
        assert_eq!(w.as_bytes(), &[0b1010_1111]);
        assert!(w.is_byte_aligned());
    }

    #[test]
    fn stuffing_is_transparent_to_reader_alignment_skip() {
        // A reader that aligns past the stuffing lands on the next
        // start code.
        let mut w = BitWriter::new();
        w.write_bits(0b11, 2); // some payload tail
        w.next_start_code();
        w.write_start_code(0x0000_01B6);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        r.read_bits(2).unwrap();
        r.align_to_byte();
        assert_eq!(r.read_bits(32).unwrap(), 0x0000_01B6);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn oversized_value_panics() {
        let mut w = BitWriter::new();
        w.write_bits(0b100, 2);
    }
}

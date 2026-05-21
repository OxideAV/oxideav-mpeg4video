//! Minimal MSB-first bit reader used by the §6.2 header parsers.
//!
//! The MPEG-4 Visual bitstream specification (ISO/IEC 14496-2) writes
//! integers most-significant-bit-first inside each byte. This reader
//! consumes a byte slice in that order and exposes the small set of
//! primitives the §6.2 parser needs (`read_bits`, `next_bits`, byte
//! alignment, position queries). It is intentionally tiny; the goal
//! is to keep the parser readable, not to win micro-benchmarks.
//!
//! The reader is fallible: any attempt to read past the end of the
//! supplied slice returns [`BitReaderError::EndOfStream`]. Callers
//! map this into the crate's [`crate::Error::Truncated`] variant.
//!
//! Provenance: behaviour is derived from the bit-stream conventions
//! described in §6.1.1 and the syntax tables in §6.2 of ISO/IEC
//! 14496-2 (3rd edition, 2004), which the agent read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.

/// Errors produced by [`BitReader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitReaderError {
    /// The reader was asked for more bits than the underlying slice
    /// can provide.
    EndOfStream,
    /// A read of more than 32 bits was requested — the reader stores
    /// its accumulator in a `u64` window large enough for any §6.2
    /// header field (the widest is 30-bit `bit_rate`, transmitted as
    /// two 15-bit halves).
    TooManyBits,
}

/// MSB-first bit reader.
#[derive(Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit offset from the start of `data`. `bit_pos / 8` is the
    /// current byte index; `bit_pos % 8` is the count of bits already
    /// consumed from the high end of that byte.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice as an MSB-first bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Total number of bits available in the wrapped slice.
    pub fn len_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Number of bits still unread.
    pub fn remaining_bits(&self) -> usize {
        self.len_bits().saturating_sub(self.bit_pos)
    }

    /// Current absolute bit position (0-indexed from the start of the
    /// slice). Useful for diagnostics.
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Whether `bit_pos` currently lies on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos % 8 == 0
    }

    /// Skip `n` bits without returning their value.
    pub fn skip_bits(&mut self, n: usize) -> Result<(), BitReaderError> {
        if self.remaining_bits() < n {
            return Err(BitReaderError::EndOfStream);
        }
        self.bit_pos += n;
        Ok(())
    }

    /// Advance `bit_pos` to the next byte boundary, doing nothing if
    /// it is already aligned.
    pub fn align_to_byte(&mut self) {
        let rem = self.bit_pos % 8;
        if rem != 0 {
            self.bit_pos += 8 - rem;
        }
    }

    /// Read `n` bits (1..=32) and return them as a `u32`. The first
    /// bit consumed becomes the most significant bit of the result.
    pub fn read_bits(&mut self, n: usize) -> Result<u32, BitReaderError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(BitReaderError::TooManyBits);
        }
        if self.remaining_bits() < n {
            return Err(BitReaderError::EndOfStream);
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            let byte = self.data[self.bit_pos / 8];
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            value = (value << 1) | u32::from(bit);
            self.bit_pos += 1;
        }
        Ok(value)
    }

    /// Read a single bit as a `bool`.
    pub fn read_bool(&mut self) -> Result<bool, BitReaderError> {
        Ok(self.read_bits(1)? == 1)
    }

    /// Peek `n` bits without advancing the position.
    pub fn next_bits(&self, n: usize) -> Result<u32, BitReaderError> {
        let mut clone = BitReader {
            data: self.data,
            bit_pos: self.bit_pos,
        };
        clone.read_bits(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first_across_byte_boundary() {
        // 0xB0 = 1011_0000. Reading 4 bits should yield 0b1011 = 11.
        let mut br = BitReader::new(&[0xB0, 0x55]);
        assert_eq!(br.read_bits(4).unwrap(), 0b1011);
        // Next 8 bits span the rest of byte 0 (4 bits) and high half
        // of byte 1 (4 bits): 0000_0101 = 5.
        assert_eq!(br.read_bits(8).unwrap(), 0b0000_0101);
    }

    #[test]
    fn read_bits_24_assembles_three_bytes() {
        let mut br = BitReader::new(&[0x12, 0x34, 0x56]);
        assert_eq!(br.read_bits(24).unwrap(), 0x12_34_56);
    }

    #[test]
    fn skip_and_align() {
        let mut br = BitReader::new(&[0xFF, 0xFF, 0xFF]);
        br.skip_bits(3).unwrap();
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn next_bits_does_not_advance() {
        let br = BitReader::new(&[0xAA]);
        assert_eq!(br.next_bits(4).unwrap(), 0xA);
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn read_past_end_errors() {
        let mut br = BitReader::new(&[0x01]);
        assert!(br.read_bits(9).is_err());
    }

    #[test]
    fn zero_bit_read_returns_zero() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_bits(0).unwrap(), 0);
        assert_eq!(br.bit_position(), 0);
    }
}

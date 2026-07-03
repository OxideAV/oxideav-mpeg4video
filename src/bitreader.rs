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
#[derive(Debug, Clone)]
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

/// MSB-first bit reader that consumes a slice from a fixed end position
/// toward the start — the reverse-direction reader the Annex E.1.4.4
/// two-way RVLC decode runs after a forward error is detected.
///
/// The reverse decode of a Reversible VLC (§E.1.3) reads the last bit of
/// the DCT-coefficient region first, then the next-to-last, and so on.
/// The bits *emitted* by [`read_bit_reverse`] are therefore in
/// stream order but visited tail-first: the first call returns the bit
/// at `end_bit - 1`, the second the bit at `end_bit - 2`, etc. A caller
/// reconstructing a codeword shifts each emitted bit into the LSB of an
/// accumulator, which yields the *bit-reversed* forward codeword — the
/// quantity Table B.23 is matched against in the backward direction.
///
/// Provenance: the reverse-decode discipline is described in §E.1.3 /
/// §E.1.4.4 of ISO/IEC 14496-2:2004 (3rd edition), read from
/// `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
#[derive(Debug, Clone)]
pub struct BackwardBitReader<'a> {
    data: &'a [u8],
    /// Absolute bit index, from the start of `data`, of the *most
    /// recently consumed* boundary. The next bit to be read is at
    /// `cur - 1`; reading stops once `cur` reaches `start`.
    cur: usize,
    /// Lower bound (inclusive) — the first bit of the region. Reading
    /// past it returns [`BitReaderError::EndOfStream`].
    start: usize,
}

impl<'a> BackwardBitReader<'a> {
    /// Wrap `data`, reading the half-open absolute bit range
    /// `[start_bit, end_bit)` from `end_bit` backward toward `start_bit`.
    ///
    /// `start_bit <= end_bit <= data.len() * 8` must hold; the caller is
    /// responsible for these bounds (they come from the video-packet
    /// Tcoef-region delimiters).
    pub fn new(data: &'a [u8], start_bit: usize, end_bit: usize) -> Self {
        debug_assert!(start_bit <= end_bit);
        debug_assert!(end_bit <= data.len() * 8);
        Self {
            data,
            cur: end_bit,
            start: start_bit,
        }
    }

    /// Number of bits still unread (between `start` and the current
    /// position).
    pub fn remaining_bits(&self) -> usize {
        self.cur.saturating_sub(self.start)
    }

    /// Absolute bit position of the next bit to be read minus one — i.e.
    /// the boundary just below the last-consumed bit. Useful for
    /// diagnostics and for computing how many bits the backward decode
    /// consumed (`end_bit - bit_position()`).
    pub fn bit_position(&self) -> usize {
        self.cur
    }

    /// Read the single bit immediately preceding the current position and
    /// advance toward the start. The returned bit is the actual stream
    /// bit (not inverted); the caller decides how to assemble it.
    pub fn read_bit_reverse(&mut self) -> Result<u8, BitReaderError> {
        if self.cur <= self.start {
            return Err(BitReaderError::EndOfStream);
        }
        self.cur -= 1;
        let byte = self.data[self.cur / 8];
        Ok((byte >> (7 - (self.cur % 8))) & 1)
    }

    /// Read `n` bits (1..=32) in the backward direction and assemble them
    /// so the result equals the value those `n` stream bits would have if
    /// read **forward** (most-significant bit first) starting at the
    /// position `n` bits before the current one.
    ///
    /// In other words this returns the natural forward value of the
    /// `n`-bit field that ends at the current position — used for the
    /// fixed-length sub-fields of the Type-5 escape, whose bit order is
    /// unchanged by direction.
    pub fn read_bits_value(&mut self, n: usize) -> Result<u32, BitReaderError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(BitReaderError::TooManyBits);
        }
        if self.remaining_bits() < n {
            return Err(BitReaderError::EndOfStream);
        }
        // Read tail-first into the LSB end, then the assembled value has
        // its bits reversed relative to forward order; reversing the low
        // `n` bits restores forward order.
        let mut rev: u32 = 0;
        for _ in 0..n {
            let bit = self.read_bit_reverse()?;
            rev = (rev << 1) | u32::from(bit);
        }
        // `rev` holds bits in tail-first order; reverse the low n bits.
        let mut value: u32 = 0;
        for i in 0..n {
            value |= ((rev >> i) & 1) << (n - 1 - i);
        }
        Ok(value)
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

    #[test]
    fn backward_reader_emits_bits_tail_first() {
        // 0b1011_0010. Reading backward from bit 8 should emit the bits
        // 0,1,0,0,1,1,0,1 (the byte read right-to-left).
        let mut br = BackwardBitReader::new(&[0b1011_0010], 0, 8);
        let mut got = Vec::new();
        for _ in 0..8 {
            got.push(br.read_bit_reverse().unwrap());
        }
        assert_eq!(got, vec![0, 1, 0, 0, 1, 1, 0, 1]);
        assert!(br.read_bit_reverse().is_err());
    }

    #[test]
    fn backward_read_bits_value_recovers_forward_field() {
        // 0xAB = 1010_1011. The low 4 bits (1011 = 0xB) end at bit 8;
        // reading 4 bits backward in forward order yields 0xB. The high
        // 4 bits (1010 = 0xA) then yield 0xA.
        let mut br = BackwardBitReader::new(&[0xAB], 0, 8);
        assert_eq!(br.read_bits_value(4).unwrap(), 0xB);
        assert_eq!(br.read_bits_value(4).unwrap(), 0xA);
    }

    #[test]
    fn backward_reader_respects_start_bound() {
        let mut br = BackwardBitReader::new(&[0xFF, 0xFF], 4, 12);
        assert_eq!(br.remaining_bits(), 8);
        for _ in 0..8 {
            br.read_bit_reverse().unwrap();
        }
        assert_eq!(br.remaining_bits(), 0);
        assert!(br.read_bit_reverse().is_err());
    }
}

//! §6.2.5.4 / §7.8.3 low-latency static-sprite piece transmission.
//!
//! When `sprite_enable == "static"` and `low_latency_sprite_enable == 1`
//! the sprite object is *not* decoded as a single I-VOP at VOL
//! initialisation. Instead the bitstream interleaves the visible S-VOP
//! payloads with a stream of *sprite pieces*: rectangular regions of the
//! sprite-object buffer (object-pieces) and texture refinements
//! (update-pieces), each one carried by `decode_sprite_piece()`
//! (§6.2.5.4). The per-S-VOP §6.2.5 syntax runs the loop
//!
//! ```text
//! if (sprite_transmit_mode != "stop" && low_latency_sprite_enable) {
//!     do {
//!         sprite_transmit_mode            // 2 bits, Table 6-26
//!         if (sprite_transmit_mode == "piece" || == "update")
//!             decode_sprite_piece()
//!     } while (sprite_transmit_mode != "stop" && != "pause")
//! }
//! ```
//!
//! (spec lines 4335..=4356). This module parses the *structural* shell of
//! that loop: the [`SpriteTransmitMode`] code (§6.2.5 / Table 6-26), the
//! [`decode_sprite_piece`] header (`piece_quant` / `piece_width` /
//! `piece_height` / marker / `piece_xoffset` / `piece_yoffset`), and the
//! `brightness_change_factor()` codeword (§6.2.5.4, Table B.35). The
//! per-macroblock `sprite_shape_texture()` body — object-pieces use a
//! subset of the I-VOP macroblock syntax, update-pieces a subset of the
//! P-VOP inter syntax (§7.8.3) — is dispatched through the existing
//! macroblock-layer entry points by the frame walker; here we expose the
//! piece geometry it needs (`piece_width × piece_height` macroblocks at
//! `(piece_xoffset, piece_yoffset)`).
//!
//! All field widths are from the §6.2.5.4 `decode_sprite_piece()` table:
//! `piece_quant` 5 bits, `piece_width` / `piece_height` /
//! `piece_xoffset` / `piece_yoffset` 9 bits each, with one `marker_bit`
//! between `piece_height` and `piece_xoffset`.

use crate::bitreader::BitReader;

/// §6.2.5 / Table 6-26 sprite transmission mode (`sprite_transmit_mode`,
/// a 2-bit code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteTransmitMode {
    /// `00` — all object and quality-update pieces for the entire VOL
    /// have been sent. Terminates the per-S-VOP piece loop.
    Stop,
    /// `01` — an object-piece follows (`decode_sprite_piece()` using a
    /// subset of the I-VOP macroblock syntax, §7.8.3).
    Piece,
    /// `10` — an update-piece follows (`decode_sprite_piece()` using a
    /// subset of the P-VOP inter macroblock syntax, §7.8.3.2).
    Update,
    /// `11` — all sprite-object and quality-update pieces for the current
    /// VOP have been sent. Suspends the loop until the next S-VOP.
    Pause,
}

impl SpriteTransmitMode {
    /// Decode the 2-bit `sprite_transmit_mode` code per Table 6-26.
    #[inline]
    pub fn from_code(code: u32) -> Self {
        match code & 0b11 {
            0b00 => SpriteTransmitMode::Stop,
            0b01 => SpriteTransmitMode::Piece,
            0b10 => SpriteTransmitMode::Update,
            _ => SpriteTransmitMode::Pause,
        }
    }

    /// Whether reaching this mode terminates the per-S-VOP `do {…} while`
    /// loop (§6.2.5): both `stop` and `pause` are loop exits.
    #[inline]
    pub fn terminates_loop(self) -> bool {
        matches!(self, SpriteTransmitMode::Stop | SpriteTransmitMode::Pause)
    }

    /// Whether this mode is followed by a `decode_sprite_piece()` body
    /// (`piece` and `update`).
    #[inline]
    pub fn has_piece(self) -> bool {
        matches!(self, SpriteTransmitMode::Piece | SpriteTransmitMode::Update)
    }
}

/// Errors raised while parsing the low-latency sprite-piece syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpritePieceError {
    /// The bitstream ran out mid-field.
    Truncated,
    /// The `marker_bit` between `piece_height` and `piece_xoffset` was
    /// `0` (§6.2 marker convention requires `1`).
    MarkerBitMissing,
    /// `piece_quant` was `0`; the §11475 note constrains it to the
    /// `1..=31` quantiser-scale range.
    InvalidPieceQuant,
    /// A `piece_width` / `piece_height` of `0` was decoded; a piece
    /// covers at least one macroblock.
    EmptyPiece,
    /// `brightness_change_factor_size` ran past the 4-bit Table B.35
    /// maximum without a terminating `0`.
    BrightnessSizeOverflow,
}

impl core::fmt::Display for SpritePieceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpritePieceError::Truncated => write!(f, "sprite piece truncated"),
            SpritePieceError::MarkerBitMissing => {
                write!(f, "decode_sprite_piece marker_bit was 0 (expected 1)")
            }
            SpritePieceError::InvalidPieceQuant => {
                write!(f, "piece_quant was 0 (valid range 1..=31)")
            }
            SpritePieceError::EmptyPiece => {
                write!(f, "piece_width/piece_height was 0")
            }
            SpritePieceError::BrightnessSizeOverflow => {
                write!(
                    f,
                    "brightness_change_factor_size exceeded the 4-bit Table B.35 maximum"
                )
            }
        }
    }
}

impl std::error::Error for SpritePieceError {}

/// The §6.2.5.4 `decode_sprite_piece()` header (everything before
/// `sprite_shape_texture()`): the geometry of one sprite piece in
/// macroblock units plus its quantiser scale.
///
/// `piece_xoffset` / `piece_yoffset` place the `piece_width ×
/// piece_height` macroblock region into the sprite-object buffer at the
/// decoder (§11487/§11489). The texture body that follows is dispatched
/// per [`SpriteTransmitMode`]: object-pieces (`piece`) skip already-sent
/// macroblocks via `send_mb()`, update-pieces (`update`) use the P-VOP
/// `not_coded` bit (§7.8.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpritePieceHeader {
    /// `piece_quant` (§11477): quantiser scale `1..=31`.
    pub piece_quant: u8,
    /// `piece_width` in macroblock units.
    pub piece_width: u16,
    /// `piece_height` in macroblock units.
    pub piece_height: u16,
    /// `piece_xoffset` in macroblock units from the left edge of the
    /// sprite object.
    pub piece_xoffset: u16,
    /// `piece_yoffset` in macroblock units from the top edge.
    pub piece_yoffset: u16,
}

impl SpritePieceHeader {
    /// Number of macroblocks the `sprite_shape_texture()` body iterates
    /// over (`piece_width * piece_height`).
    #[inline]
    pub fn macroblock_count(&self) -> usize {
        self.piece_width as usize * self.piece_height as usize
    }
}

/// Parse the §6.2.5.4 `decode_sprite_piece()` header.
///
/// Field order (spec lines 4947..=4954): `piece_quant` (5), `piece_width`
/// (9), `piece_height` (9), `marker_bit` (1), `piece_xoffset` (9),
/// `piece_yoffset` (9). Leaves `br` positioned at the start of the
/// `sprite_shape_texture()` body.
pub fn decode_sprite_piece(br: &mut BitReader<'_>) -> Result<SpritePieceHeader, SpritePieceError> {
    let piece_quant = br.read_bits(5).map_err(|_| SpritePieceError::Truncated)? as u8;
    if piece_quant == 0 {
        return Err(SpritePieceError::InvalidPieceQuant);
    }
    let piece_width = br.read_bits(9).map_err(|_| SpritePieceError::Truncated)? as u16;
    let piece_height = br.read_bits(9).map_err(|_| SpritePieceError::Truncated)? as u16;
    if piece_width == 0 || piece_height == 0 {
        return Err(SpritePieceError::EmptyPiece);
    }
    let marker = br.read_bits(1).map_err(|_| SpritePieceError::Truncated)?;
    if marker != 1 {
        return Err(SpritePieceError::MarkerBitMissing);
    }
    let piece_xoffset = br.read_bits(9).map_err(|_| SpritePieceError::Truncated)? as u16;
    let piece_yoffset = br.read_bits(9).map_err(|_| SpritePieceError::Truncated)? as u16;
    Ok(SpritePieceHeader {
        piece_quant,
        piece_width,
        piece_height,
        piece_xoffset,
        piece_yoffset,
    })
}

/// Decode a §6.2.5.4 `brightness_change_factor()` codeword (Table B.35).
///
/// The codeword is a VLC `brightness_change_factor_size` (`0`, `10`,
/// `110`, `1110`, `1111`) selecting a value-magnitude band, followed by a
/// fixed-length `brightness_change_factor_code` of
/// `brightness_change_factor_size` bits (with `1110`/`1111` both being
/// "size 4" but selecting two disjoint 9-/10-bit bands). The decoded
/// signed factor feeds the §7.8.6 `Y = (Y · (factor + 100)) // 100`
/// brightness post-adjustment.
///
/// Per Table B.35 the bands are:
/// * size 1 (`0`): `-16..=-1, 1..=16` — 5-bit code (sign in the band).
/// * size 2 (`10`): `-48..=-17, 17..=48` — 6-bit code.
/// * size 3 (`110`): `-112..=-49, 49..=112` — 7-bit code.
/// * size 4-low (`1110`): `113..=624` — 9-bit code (positive only).
/// * size 4-high (`1111`): `625..=1648` — 10-bit code (positive only).
///
/// For the symmetric bands (sizes 1..=3) the `n`-bit code's top bit
/// selects the sign: a set top bit is the positive half
/// (`base + (code - half)`), a clear top bit is the negative half
/// (`-(base + (half-1 - code))`), where `base` is the band's smallest
/// magnitude and `half == 2^(n-1)`.
pub fn decode_brightness_change_factor(br: &mut BitReader<'_>) -> Result<i32, SpritePieceError> {
    // Read the size VLC: count leading 1s (max 4), stop at the 0 — except
    // the all-four-1s case `1111` which has no terminating 0.
    let mut ones = 0u32;
    loop {
        if ones == 4 {
            // `1111` — size-4-high band, no terminating 0.
            break;
        }
        let bit = br.read_bits(1).map_err(|_| SpritePieceError::Truncated)?;
        if bit == 0 {
            break;
        }
        ones += 1;
    }

    // Map the unary-ish prefix to (code_bits, value_base, signed_band).
    match ones {
        0 => decode_symmetric_band(br, 5, 1),
        1 => decode_symmetric_band(br, 6, 17),
        2 => decode_symmetric_band(br, 7, 49),
        3 => {
            // `1110` — size-4-low band: 9-bit code → 113..=624.
            let code = br.read_bits(9).map_err(|_| SpritePieceError::Truncated)? as i32;
            Ok(113 + code)
        }
        4 => {
            // `1111` — size-4-high band: 10-bit code → 625..=1648.
            let code = br.read_bits(10).map_err(|_| SpritePieceError::Truncated)? as i32;
            Ok(625 + code)
        }
        _ => Err(SpritePieceError::BrightnessSizeOverflow),
    }
}

/// Decode an `n`-bit code from a symmetric Table B.35 band whose smallest
/// magnitude is `base`. The band covers `[-(base + half - 1) .. -base]`
/// and `[base .. base + half - 1]` where `half == 2^(n-1)`.
#[inline]
fn decode_symmetric_band(
    br: &mut BitReader<'_>,
    n: usize,
    base: i32,
) -> Result<i32, SpritePieceError> {
    let code = br.read_bits(n).map_err(|_| SpritePieceError::Truncated)? as i32;
    let half = 1i32 << (n - 1);
    if code >= half {
        // Top bit set → positive half: smallest positive at code==half.
        Ok(base + (code - half))
    } else {
        // Top bit clear → negative half: most negative at code==0.
        Ok(-(base + (half - 1 - code)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer mirroring `bslbf` / `uimsbf`.
    struct BitWriter {
        buf: Vec<u8>,
        bit: u8,
        cur: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                bit: 0,
                cur: 0,
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for k in (0..n).rev() {
                let b = ((value >> k) & 1) as u8;
                self.cur |= b << (7 - self.bit);
                self.bit += 1;
                if self.bit == 8 {
                    self.buf.push(self.cur);
                    self.cur = 0;
                    self.bit = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bit != 0 {
                self.buf.push(self.cur);
            }
            self.buf
        }
    }

    #[test]
    fn transmit_mode_codes() {
        assert_eq!(
            SpriteTransmitMode::from_code(0b00),
            SpriteTransmitMode::Stop
        );
        assert_eq!(
            SpriteTransmitMode::from_code(0b01),
            SpriteTransmitMode::Piece
        );
        assert_eq!(
            SpriteTransmitMode::from_code(0b10),
            SpriteTransmitMode::Update
        );
        assert_eq!(
            SpriteTransmitMode::from_code(0b11),
            SpriteTransmitMode::Pause
        );
    }

    #[test]
    fn transmit_mode_loop_predicates() {
        assert!(SpriteTransmitMode::Stop.terminates_loop());
        assert!(SpriteTransmitMode::Pause.terminates_loop());
        assert!(!SpriteTransmitMode::Piece.terminates_loop());
        assert!(!SpriteTransmitMode::Update.terminates_loop());

        assert!(SpriteTransmitMode::Piece.has_piece());
        assert!(SpriteTransmitMode::Update.has_piece());
        assert!(!SpriteTransmitMode::Stop.has_piece());
        assert!(!SpriteTransmitMode::Pause.has_piece());
    }

    #[test]
    fn sprite_piece_header_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(7, 5); // piece_quant = 7
        w.write_bits(11, 9); // piece_width = 11
        w.write_bits(9, 9); // piece_height = 9
        w.write_bits(1, 1); // marker
        w.write_bits(2, 9); // piece_xoffset = 2
        w.write_bits(3, 9); // piece_yoffset = 3
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let hdr = decode_sprite_piece(&mut br).unwrap();
        assert_eq!(hdr.piece_quant, 7);
        assert_eq!(hdr.piece_width, 11);
        assert_eq!(hdr.piece_height, 9);
        assert_eq!(hdr.piece_xoffset, 2);
        assert_eq!(hdr.piece_yoffset, 3);
        assert_eq!(hdr.macroblock_count(), 99);
    }

    #[test]
    fn piece_quant_zero_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(0, 5); // piece_quant = 0
        w.write_bits(1, 9);
        w.write_bits(1, 9);
        w.write_bits(1, 1);
        w.write_bits(0, 9);
        w.write_bits(0, 9);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_sprite_piece(&mut br).unwrap_err(),
            SpritePieceError::InvalidPieceQuant
        );
    }

    #[test]
    fn empty_piece_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(5, 5);
        w.write_bits(0, 9); // piece_width = 0
        w.write_bits(4, 9);
        w.write_bits(1, 1);
        w.write_bits(0, 9);
        w.write_bits(0, 9);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_sprite_piece(&mut br).unwrap_err(),
            SpritePieceError::EmptyPiece
        );
    }

    #[test]
    fn missing_marker_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(5, 5);
        w.write_bits(4, 9);
        w.write_bits(4, 9);
        w.write_bits(0, 1); // bad marker
        w.write_bits(0, 9);
        w.write_bits(0, 9);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(
            decode_sprite_piece(&mut br).unwrap_err(),
            SpritePieceError::MarkerBitMissing
        );
    }

    #[test]
    fn brightness_size1_band_endpoints() {
        // size 1 (`0` prefix), 5-bit code. Band: -16..=-1, 1..=16.
        // base=1, half=16. code 10000 (16) → +1; code 11111 (31) → +16.
        // code 01111 (15) → -1; code 00000 (0) → -16.
        for (code, expected) in [(0b10000, 1), (0b11111, 16), (0b01111, -1), (0b00000, -16)] {
            let mut w = BitWriter::new();
            w.write_bits(0, 1); // size prefix `0`
            w.write_bits(code, 5);
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(
                decode_brightness_change_factor(&mut br).unwrap(),
                expected,
                "size1 code={code:05b}"
            );
        }
    }

    #[test]
    fn brightness_size2_band_endpoints() {
        // size 2 (`10` prefix), 6-bit code. Band: -48..=-17, 17..=48.
        // base=17, half=32. code 100000 (32) → +17; code 111111 (63) → +48.
        // code 011111 (31) → -17; code 000000 (0) → -48.
        for (code, expected) in [
            (0b100000, 17),
            (0b111111, 48),
            (0b011111, -17),
            (0b000000, -48),
        ] {
            let mut w = BitWriter::new();
            w.write_bits(0b10, 2);
            w.write_bits(code, 6);
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(
                decode_brightness_change_factor(&mut br).unwrap(),
                expected,
                "size2 code={code:06b}"
            );
        }
    }

    #[test]
    fn brightness_size3_band_endpoints() {
        // size 3 (`110` prefix), 7-bit code. Band: -112..=-49, 49..=112.
        // base=49, half=64. code 1000000 (64) → +49; code 1111111 → +112.
        for (code, expected) in [(0b1000000, 49), (0b1111111, 112), (0b0000000, -112)] {
            let mut w = BitWriter::new();
            w.write_bits(0b110, 3);
            w.write_bits(code, 7);
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(
                decode_brightness_change_factor(&mut br).unwrap(),
                expected,
                "size3 code={code:07b}"
            );
        }
    }

    #[test]
    fn brightness_size4_low_band() {
        // size 4-low (`1110` prefix), 9-bit code → 113 + code, 113..=624.
        let mut w = BitWriter::new();
        w.write_bits(0b1110, 4);
        w.write_bits(0, 9);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_brightness_change_factor(&mut br).unwrap(), 113);

        let mut w = BitWriter::new();
        w.write_bits(0b1110, 4);
        w.write_bits(511, 9); // 113 + 511 = 624
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_brightness_change_factor(&mut br).unwrap(), 624);
    }

    #[test]
    fn brightness_size4_high_band() {
        // size 4-high (`1111` prefix), 10-bit code → 625 + code, 625..=1648.
        let mut w = BitWriter::new();
        w.write_bits(0b1111, 4);
        w.write_bits(0, 10);
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_brightness_change_factor(&mut br).unwrap(), 625);

        let mut w = BitWriter::new();
        w.write_bits(0b1111, 4);
        w.write_bits(1023, 10); // 625 + 1023 = 1648
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        assert_eq!(decode_brightness_change_factor(&mut br).unwrap(), 1648);
    }

    #[test]
    fn truncated_piece_is_error() {
        let buf = [0xFFu8; 1];
        let mut br = BitReader::new(&buf);
        // 5 bits piece_quant fits, then width needs 9 — only 3 left.
        assert_eq!(
            decode_sprite_piece(&mut br).unwrap_err(),
            SpritePieceError::Truncated
        );
    }
}

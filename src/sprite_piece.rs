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
    /// §7.8.3.2: an update-piece refines a macroblock whose corresponding
    /// object macroblock has not yet been transmitted (an update MB cannot
    /// precede its object MB; the carried `(x, y)` is the offending grid
    /// position).
    UpdateBeforeObject(usize, usize),
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
            SpritePieceError::UpdateBeforeObject(x, y) => {
                write!(
                    f,
                    "update-piece MB ({x}, {y}) precedes its object MB (§7.8.3.2)"
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

/// §7.8.3.1 sprite-object-buffer macroblock occupancy tracker.
///
/// The static-sprite object buffer is a `sprite_width_in_mb ×
/// sprite_height_in_mb` macroblock grid assembled progressively from
/// object-pieces. For an object-piece, `sprite_shape_texture()` iterates
/// the piece's `piece_width × piece_height` macroblocks in raster order
/// and consults `send_mb()` per macroblock: a macroblock that was already
/// transmitted by an earlier piece (a *hole* in this piece's bitstream)
/// returns `1` and carries **no** `macroblock()` body — the decoder
/// retrieves it from the earlier piece (§7.8.3.1). A fresh macroblock
/// returns `0` and is decoded from this piece's bitstream.
///
/// This tracker maintains the per-macroblock "already-sent" bitmap so a
/// caller walking a piece can answer `send_mb()` and mark macroblocks as
/// they are transmitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpriteObjectBuffer {
    width_mb: usize,
    height_mb: usize,
    /// `sent[y * width_mb + x] == true` once MB `(x, y)` of the sprite
    /// object grid has been transmitted by some piece.
    sent: Vec<bool>,
}

impl SpriteObjectBuffer {
    /// Create an empty object buffer of `width_mb × height_mb`
    /// macroblocks (all macroblocks un-sent). Returns `None` for a
    /// zero-area grid.
    pub fn new(width_mb: usize, height_mb: usize) -> Option<Self> {
        if width_mb == 0 || height_mb == 0 {
            return None;
        }
        Some(SpriteObjectBuffer {
            width_mb,
            height_mb,
            sent: vec![false; width_mb * height_mb],
        })
    }

    /// `send_mb()` (§11475): `true` if the sprite-object macroblock at
    /// grid position `(x, y)` was already transmitted by an earlier
    /// piece. Out-of-grid coordinates are treated as already-sent (they
    /// are never decoded from a piece).
    #[inline]
    pub fn send_mb(&self, x: usize, y: usize) -> bool {
        if x >= self.width_mb || y >= self.height_mb {
            return true;
        }
        self.sent[y * self.width_mb + x]
    }

    /// Mark the sprite-object macroblock at grid `(x, y)` as transmitted.
    /// Out-of-grid coordinates are ignored.
    #[inline]
    pub fn mark_sent(&mut self, x: usize, y: usize) {
        if x < self.width_mb && y < self.height_mb {
            self.sent[y * self.width_mb + x] = true;
        }
    }

    /// Walk one object-piece's `piece_width × piece_height` macroblock
    /// region (top-left at `(piece_xoffset, piece_yoffset)` in
    /// sprite-object grid coordinates) in raster order, returning the
    /// grid positions of the macroblocks that are *new* (`send_mb()` is
    /// `0`) — the ones that carry a `macroblock()` body in this piece's
    /// bitstream — and marking them as sent.
    ///
    /// Macroblocks that are holes (`send_mb()` is `1`, already sent or
    /// out of grid) are skipped, mirroring the §7.8.3.1
    /// `if (!send_mb()) macroblock()` guard.
    pub fn object_piece_new_macroblocks(
        &mut self,
        header: &SpritePieceHeader,
    ) -> Vec<(usize, usize)> {
        let mut new_mbs = Vec::new();
        let x0 = header.piece_xoffset as usize;
        let y0 = header.piece_yoffset as usize;
        for dy in 0..header.piece_height as usize {
            for dx in 0..header.piece_width as usize {
                let x = x0 + dx;
                let y = y0 + dy;
                if !self.send_mb(x, y) {
                    self.mark_sent(x, y);
                    new_mbs.push((x, y));
                }
            }
        }
        new_mbs
    }

    /// Whether every macroblock of the sprite-object grid has been
    /// transmitted (the object is fully assembled).
    pub fn is_complete(&self) -> bool {
        self.sent.iter().all(|&s| s)
    }

    /// Validate an **update**-piece against this object buffer (§7.8.3.2):
    /// every macroblock the update-piece touches must already exist in the
    /// object buffer (an update MB cannot precede its object MB). The
    /// caller supplies the per-macroblock `not_coded` flags of the update
    /// piece in raster order (length `piece_width * piece_height`):
    /// `not_coded == false` means the macroblock is *refined*.
    ///
    /// Returns the grid positions of the refined macroblocks
    /// (`not_coded == false`). Object MBs are **not** marked sent — an
    /// update refines existing texture, it does not establish a new MB.
    /// Errors with [`SpritePieceError::UpdateBeforeObject`] for the first
    /// touched MB whose object MB is missing, or
    /// [`SpritePieceError::Truncated`] if the `not_coded` slice length does
    /// not match the piece geometry.
    pub fn update_piece_refined_macroblocks(
        &self,
        header: &SpritePieceHeader,
        not_coded: &[bool],
    ) -> Result<Vec<(usize, usize)>, SpritePieceError> {
        if not_coded.len() != header.macroblock_count() {
            return Err(SpritePieceError::Truncated);
        }
        let x0 = header.piece_xoffset as usize;
        let y0 = header.piece_yoffset as usize;
        let mut refined = Vec::new();
        for dy in 0..header.piece_height as usize {
            for dx in 0..header.piece_width as usize {
                let x = x0 + dx;
                let y = y0 + dy;
                // §7.8.3.2: the update MB's object MB must already exist.
                if !self.send_mb(x, y) {
                    return Err(SpritePieceError::UpdateBeforeObject(x, y));
                }
                let idx = dy * header.piece_width as usize + dx;
                if !not_coded[idx] {
                    refined.push((x, y));
                }
            }
        }
        Ok(refined)
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

/// One iteration of the §6.2.5 low-latency piece loop: the
/// `sprite_transmit_mode` that was read and — when that mode carried a
/// `decode_sprite_piece()` body — the decoded header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpritePieceLoopEntry {
    /// The `sprite_transmit_mode` read at the top of this iteration.
    pub mode: SpriteTransmitMode,
    /// The `decode_sprite_piece()` header, present iff `mode.has_piece()`.
    pub header: Option<SpritePieceHeader>,
}

/// Maximum piece-loop iterations before the driver bails with
/// [`SpritePieceError::Truncated`]. A well-formed stream terminates at a
/// `stop` / `pause` mode; the cap guards against a malformed loop that
/// never emits one.
const MAX_PIECE_ITERATIONS: usize = 4096;

/// Drive the per-S-VOP §6.2.5 low-latency piece loop (spec lines
/// 4335..=4356).
///
/// The §6.2.5 syntax only enters the loop when the *prior*
/// `sprite_transmit_mode` was not `stop` (the VOL initialises it to
/// `piece`, §10903) and `low_latency_sprite_enable == 1`. The caller is
/// responsible for that outer gate; this driver runs the inner
/// `do {…} while` once entered:
///
/// ```text
/// do {
///     sprite_transmit_mode           // 2 bits
///     if (mode == piece || mode == update)
///         decode_sprite_piece()      // header + sprite_shape_texture()
/// } while (mode != stop && mode != pause)
/// ```
///
/// Because the `sprite_shape_texture()` body is a full macroblock-layer
/// walk (object-pieces use the I-VOP subset, update-pieces the P-VOP
/// inter subset, §7.8.3), the driver delegates advancing past each piece
/// body to `skip_body`: it is called with the decoded header after the
/// header is read and must leave `br` positioned at the next
/// `sprite_transmit_mode` (or `next_start_code()` boundary). For a
/// header-only structural walk (no texture present, e.g. unit tests or a
/// caller that records geometry without decoding pixels) pass a no-op.
///
/// Returns one [`SpritePieceLoopEntry`] per iteration, in order. The
/// final entry's `mode` is always a loop terminator (`stop` or `pause`)
/// with no header.
pub fn drive_sprite_piece_loop<F>(
    br: &mut BitReader<'_>,
    mut skip_body: F,
) -> Result<Vec<SpritePieceLoopEntry>, SpritePieceError>
where
    F: FnMut(&mut BitReader<'_>, &SpritePieceHeader) -> Result<(), SpritePieceError>,
{
    let mut entries = Vec::new();
    for _ in 0..MAX_PIECE_ITERATIONS {
        let code = br.read_bits(2).map_err(|_| SpritePieceError::Truncated)?;
        let mode = SpriteTransmitMode::from_code(code);
        let header = if mode.has_piece() {
            let hdr = decode_sprite_piece(br)?;
            skip_body(br, &hdr)?;
            Some(hdr)
        } else {
            None
        };
        let terminates = mode.terminates_loop();
        entries.push(SpritePieceLoopEntry { mode, header });
        if terminates {
            return Ok(entries);
        }
    }
    Err(SpritePieceError::Truncated)
}

/// The decoded §6.2.5 static-sprite S-VOP sprite block: the
/// `sprite_trajectory()` (`du[i]`/`dv[i]` pairs, `None` when
/// `no_of_sprite_warping_points == 0`), the optional
/// `brightness_change_factor()`, and the low-latency piece loop entries
/// (empty when `low_latency_sprite_enable == 0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticSpriteVopBlock {
    /// `no_of_sprite_warping_points` active points.
    pub warping_points: u8,
    /// `[du[i], dv[i]]` warping-vector pairs (only the first
    /// `warping_points` entries are valid).
    pub trajectory: [[i32; 2]; 4],
    /// `brightness_change_factor()` value (§7.8.6), `None` when
    /// `sprite_brightness_change == 0`.
    pub brightness_change: Option<i32>,
    /// The §6.2.5 low-latency piece-loop entries, in order. Empty when
    /// `low_latency_sprite_enable == 0` (no loop is present).
    pub piece_loop: Vec<SpritePieceLoopEntry>,
}

/// Parse the §6.2.5 static-sprite S-VOP sprite block (spec lines
/// 4328..=4357): `sprite_trajectory()`, optional
/// `brightness_change_factor()`, and — when `low_latency_sprite_enable`
/// — the `do { sprite_transmit_mode; … } while` piece loop.
///
/// The §6.2.5 `vop()` syntax reaches this block after `intra_dc_vlc_thr`
/// (and the interlaced fields) for `sprite_enable == "static" &&
/// vop_coding_type == "S"`, and follows it with `next_start_code();
/// return()` — i.e. a static S-VOP carries *no* macroblock-layer texture
/// of its own; all sprite samples arrive through the piece stream. This
/// parser stops at the end of the piece loop, leaving `br` at the
/// `next_start_code()` boundary the caller aligns.
///
/// `no_of_sprite_warping_points` / `sprite_brightness_change` /
/// `low_latency_sprite_enable` are VOL-header fields the caller supplies.
/// `skip_body` advances past each piece's `sprite_shape_texture()` body
/// (see [`drive_sprite_piece_loop`]).
pub fn parse_static_sprite_vop_block<F>(
    br: &mut BitReader<'_>,
    no_of_sprite_warping_points: u8,
    sprite_brightness_change: bool,
    low_latency_sprite_enable: bool,
    skip_body: F,
) -> Result<StaticSpriteVopBlock, SpritePieceError>
where
    F: FnMut(&mut BitReader<'_>, &SpritePieceHeader) -> Result<(), SpritePieceError>,
{
    // sprite_trajectory() — present iff no_of_sprite_warping_points > 0.
    let (warping_points, trajectory) = if no_of_sprite_warping_points > 0 {
        crate::sprite::decode_sprite_trajectory_static(br, no_of_sprite_warping_points)
            .map_err(|_| SpritePieceError::Truncated)?
    } else {
        (0, [[0i32; 2]; 4])
    };

    // brightness_change_factor() — present iff sprite_brightness_change.
    let brightness_change = if sprite_brightness_change {
        Some(decode_brightness_change_factor(br)?)
    } else {
        None
    };

    // Low-latency piece loop — present iff low_latency_sprite_enable. The
    // §10903 note initialises sprite_transmit_mode to "piece" at VOL
    // start, so the `!= "stop"` guard is satisfied on first entry.
    let piece_loop = if low_latency_sprite_enable {
        drive_sprite_piece_loop(br, skip_body)?
    } else {
        Vec::new()
    };

    Ok(StaticSpriteVopBlock {
        warping_points,
        trajectory,
        brightness_change,
        piece_loop,
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

    /// Emit a `decode_sprite_piece()` header into a writer (no body).
    fn write_piece_header(w: &mut BitWriter, q: u32, pw: u32, ph: u32, xo: u32, yo: u32) {
        w.write_bits(q, 5);
        w.write_bits(pw, 9);
        w.write_bits(ph, 9);
        w.write_bits(1, 1); // marker
        w.write_bits(xo, 9);
        w.write_bits(yo, 9);
    }

    #[test]
    fn loop_single_piece_then_stop() {
        // piece(01) + header, then stop(00). No texture body (skip_body
        // is a no-op).
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // mode = piece
        write_piece_header(&mut w, 3, 4, 4, 0, 0);
        w.write_bits(0b00, 2); // mode = stop
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let entries = drive_sprite_piece_loop(&mut br, |_, _| Ok(())).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].mode, SpriteTransmitMode::Piece);
        assert_eq!(entries[0].header.unwrap().piece_quant, 3);
        assert_eq!(entries[0].header.unwrap().macroblock_count(), 16);
        assert_eq!(entries[1].mode, SpriteTransmitMode::Stop);
        assert!(entries[1].header.is_none());
    }

    #[test]
    fn loop_piece_then_update_then_pause() {
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // piece
        write_piece_header(&mut w, 5, 2, 3, 1, 1);
        w.write_bits(0b10, 2); // update
        write_piece_header(&mut w, 7, 2, 3, 1, 1);
        w.write_bits(0b11, 2); // pause
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let entries = drive_sprite_piece_loop(&mut br, |_, _| Ok(())).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].mode, SpriteTransmitMode::Piece);
        assert_eq!(entries[1].mode, SpriteTransmitMode::Update);
        assert_eq!(entries[1].header.unwrap().piece_quant, 7);
        assert_eq!(entries[2].mode, SpriteTransmitMode::Pause);
    }

    #[test]
    fn loop_immediate_stop_no_pieces() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // stop right away
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let entries = drive_sprite_piece_loop(&mut br, |_, _| Ok(())).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].mode, SpriteTransmitMode::Stop);
    }

    #[test]
    fn loop_skip_body_advances_reader() {
        // Piece carries a 13-bit "texture" filler; skip_body must consume
        // it so the next read lands on the stop code.
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2);
        write_piece_header(&mut w, 1, 1, 1, 0, 0);
        w.write_bits(0b1010101010101, 13); // mock body
        w.write_bits(0b00, 2); // stop
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let entries = drive_sprite_piece_loop(&mut br, |br, _hdr| {
            br.skip_bits(13).map_err(|_| SpritePieceError::Truncated)
        })
        .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].mode, SpriteTransmitMode::Stop);
    }

    /// Emit a `warping_mv_code(dmv)`: unary SSS, FLC, marker.
    fn write_warping(w: &mut BitWriter, sss: u32, code: u32) {
        for _ in 0..sss {
            w.write_bits(1, 1);
        }
        w.write_bits(0, 1);
        if sss != 0 {
            w.write_bits(code, sss as usize);
        }
        w.write_bits(1, 1); // marker
    }

    #[test]
    fn static_block_trajectory_brightness_and_loop() {
        // 2 warping points + brightness + a single piece then stop.
        let mut w = BitWriter::new();
        write_warping(&mut w, 1, 1); // du0 = +1
        write_warping(&mut w, 1, 0); // dv0 = -1
        write_warping(&mut w, 2, 0b10); // du1 = +2
        write_warping(&mut w, 2, 0b01); // dv1 = -2
        w.write_bits(0, 1); // brightness size 1
        w.write_bits(0b10000, 5); // brightness +1
        w.write_bits(0b01, 2); // transmit_mode = piece
        write_piece_header(&mut w, 4, 2, 2, 0, 0);
        w.write_bits(0b00, 2); // stop
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let block = parse_static_sprite_vop_block(&mut br, 2, true, true, |_, _| Ok(())).unwrap();
        assert_eq!(block.warping_points, 2);
        assert_eq!(block.trajectory[0], [1, -1]);
        assert_eq!(block.trajectory[1], [2, -2]);
        assert_eq!(block.brightness_change, Some(1));
        assert_eq!(block.piece_loop.len(), 2);
        assert_eq!(block.piece_loop[0].mode, SpriteTransmitMode::Piece);
        assert_eq!(block.piece_loop[1].mode, SpriteTransmitMode::Stop);
    }

    #[test]
    fn static_block_no_lowlatency_skips_loop() {
        // 0 points, no brightness, no low-latency ⇒ empty block, no bits
        // consumed beyond the (absent) trajectory.
        let buf = [0xFFu8; 2];
        let mut br = BitReader::new(&buf);
        let block = parse_static_sprite_vop_block(&mut br, 0, false, false, |_, _| Ok(())).unwrap();
        assert_eq!(block.warping_points, 0);
        assert!(block.brightness_change.is_none());
        assert!(block.piece_loop.is_empty());
        assert_eq!(br.bit_position(), 0, "no bits consumed");
    }

    #[test]
    fn object_buffer_first_piece_all_new() {
        // A fresh 4×4 object buffer: a 2×2 piece at (0,0) is all-new.
        let mut buf = SpriteObjectBuffer::new(4, 4).unwrap();
        let hdr = SpritePieceHeader {
            piece_quant: 5,
            piece_width: 2,
            piece_height: 2,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        let new = buf.object_piece_new_macroblocks(&hdr);
        assert_eq!(new, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert!(buf.send_mb(0, 0));
        assert!(!buf.send_mb(2, 0));
        assert!(!buf.is_complete());
    }

    #[test]
    fn object_buffer_overlapping_piece_has_holes() {
        // First piece fills (0,0)..(1,1). A second 3×3 piece at (0,0)
        // overlaps it: the overlapping MBs are holes (already sent), only
        // the new ones carry a body.
        let mut buf = SpriteObjectBuffer::new(4, 4).unwrap();
        let p1 = SpritePieceHeader {
            piece_quant: 5,
            piece_width: 2,
            piece_height: 2,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        buf.object_piece_new_macroblocks(&p1);
        let p2 = SpritePieceHeader {
            piece_quant: 5,
            piece_width: 3,
            piece_height: 3,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        let new = buf.object_piece_new_macroblocks(&p2);
        // (0,0),(1,0),(0,1),(1,1) are holes; the L-shaped frontier is new.
        assert_eq!(new, vec![(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)]);
    }

    #[test]
    fn object_buffer_completes_when_grid_filled() {
        let mut buf = SpriteObjectBuffer::new(2, 2).unwrap();
        let hdr = SpritePieceHeader {
            piece_quant: 1,
            piece_width: 2,
            piece_height: 2,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        buf.object_piece_new_macroblocks(&hdr);
        assert!(buf.is_complete());
    }

    #[test]
    fn object_buffer_out_of_grid_is_sent() {
        let buf = SpriteObjectBuffer::new(2, 2).unwrap();
        // A coordinate beyond the grid is always "sent" (never decoded).
        assert!(buf.send_mb(5, 0));
        assert!(buf.send_mb(0, 9));
    }

    #[test]
    fn object_buffer_rejects_zero_area() {
        assert!(SpriteObjectBuffer::new(0, 4).is_none());
        assert!(SpriteObjectBuffer::new(4, 0).is_none());
    }

    #[test]
    fn update_piece_refines_existing_macroblocks() {
        // Object piece fills a 2×2 region; an update piece over the same
        // region refines the MBs whose not_coded == false.
        let mut buf = SpriteObjectBuffer::new(4, 4).unwrap();
        let obj = SpritePieceHeader {
            piece_quant: 5,
            piece_width: 2,
            piece_height: 2,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        buf.object_piece_new_macroblocks(&obj);
        // Update over the same region: refine (0,0) and (1,1), skip the
        // other two (not_coded = true).
        let not_coded = [false, true, true, false];
        let refined = buf
            .update_piece_refined_macroblocks(&obj, &not_coded)
            .unwrap();
        assert_eq!(refined, vec![(0, 0), (1, 1)]);
        // Update does not establish new MBs — completeness unchanged.
        assert!(!buf.is_complete());
    }

    #[test]
    fn update_before_object_is_rejected() {
        // An update piece touching a MB the object buffer has not yet seen
        // must error (§7.8.3.2: the first piece must be an object-piece).
        let buf = SpriteObjectBuffer::new(4, 4).unwrap();
        let hdr = SpritePieceHeader {
            piece_quant: 5,
            piece_width: 1,
            piece_height: 1,
            piece_xoffset: 2,
            piece_yoffset: 1,
        };
        assert_eq!(
            buf.update_piece_refined_macroblocks(&hdr, &[false])
                .unwrap_err(),
            SpritePieceError::UpdateBeforeObject(2, 1)
        );
    }

    #[test]
    fn update_piece_length_mismatch_is_truncated() {
        let mut buf = SpriteObjectBuffer::new(2, 2).unwrap();
        let hdr = SpritePieceHeader {
            piece_quant: 1,
            piece_width: 2,
            piece_height: 2,
            piece_xoffset: 0,
            piece_yoffset: 0,
        };
        buf.object_piece_new_macroblocks(&hdr);
        // Wrong not_coded length (3, expected 4).
        assert_eq!(
            buf.update_piece_refined_macroblocks(&hdr, &[false, false, false])
                .unwrap_err(),
            SpritePieceError::Truncated
        );
    }

    #[test]
    fn static_block_brightness_only_when_flagged() {
        // 1 warping point, no brightness flag ⇒ brightness absent, the
        // bits after the trajectory belong to the loop.
        let mut w = BitWriter::new();
        write_warping(&mut w, 1, 1); // du0 = +1
        write_warping(&mut w, 1, 1); // dv0 = +1
        w.write_bits(0b00, 2); // transmit_mode = stop (low-latency on)
        let buf = w.finish();
        let mut br = BitReader::new(&buf);
        let block = parse_static_sprite_vop_block(&mut br, 1, false, true, |_, _| Ok(())).unwrap();
        assert_eq!(block.warping_points, 1);
        assert_eq!(block.trajectory[0], [1, 1]);
        assert!(block.brightness_change.is_none());
        assert_eq!(block.piece_loop.len(), 1);
        assert_eq!(block.piece_loop[0].mode, SpriteTransmitMode::Stop);
    }
}

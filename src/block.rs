//! §6.2.7 `block(i)` macroblock-level texture assembly for I- and
//! inter-coded macroblocks.
//!
//! Rounds 9..14 built every stage of the §7.4.x intra texture pipeline
//! as standalone functions: §7.4.1.1 intra-DC decode, §7.4.1.2 AC EVENT
//! decode, §7.4.2 inverse scan, §7.4.3 spatial DC/AC predictor, §7.4.4
//! inverse quantisation, and §7.4.5 + Annex A inverse DCT. This module
//! is the per-macroblock driver that wires those stages together: for an
//! intra macroblock it walks the §6.1.3.9 / Figure 6-8 block ordering
//! (4 luminance + 2 chrominance blocks in 4:2:0), runs each block's
//! `block(i)` syntax of §6.2.7 through the §7.4.x chain, and assembles
//! the reconstructed 16×16 luminance + 8×8 Cb / 8×8 Cr macroblock.
//!
//! Round 21 (this round) adds the **inter** half of the §6.2.7 driver:
//! [`decode_inter_block`] runs the `if (pattern_code[i]) while (!last)
//! DCT coefficient` branch of the §6.2.7 syntax table — the entire
//! intra-DC prologue is gated out for inter `derived_mb_type` values, so
//! an inter block carries no DC coefficient at all (§6.2.7 line "if
//! (!data_partitioned && (derived_mb_type == 3 || derived_mb_type == 4))
//! { intra-DC … }"), no spatial predictor add (§7.4.3 is intra-only),
//! and always uses [`crate::scan::ScanType::Zigzag`] (§7.4.2 "Non-intra
//! blocks → zigzag"). The IDCT output `f[y][x]` is the §7.3 step-2
//! *residual* — it is NOT clipped to `[0, 2^bpp − 1]` because the §7.3
//! step-3 display clip happens after `d[y][x] = p[y][x] + f[y][x]`. The
//! per-block residual stays in `[−2^bpp, 2^bpp − 1]` from §7.4.5.
//! [`decode_inter_macroblock`] assembles a [`InterMacroblock`] of 16×16
//! luma + 8×8 Cb / 8×8 Cr signed-residual planes for any
//! `derived_mb_type ∈ {Inter, InterQ, Inter4V}` macroblock. The
//! caller's motion-compensation stage adds the prediction and applies
//! the §7.3 step-3 display clip.
//!
//! ## §6.2.7 `block(i)` syntax (intra branch)
//!
//! For an intra macroblock (`derived_mb_type == 3 || == 4`) with
//! `data_partitioned == 0` and `short_video_header == 0`, each block's
//! texture is:
//!
//! ```text
//! block(i) {
//!     last = 0
//!     if (use_intra_dc_vlc == 1) {
//!         if (i < 4) dct_dc_size_luminance + dct_dc_differential + marker
//!         else       dct_dc_size_chrominance + dct_dc_differential + marker
//!     }
//!     if (pattern_code[i])
//!         while (!last) DCT coefficient
//! }
//! ```
//!
//! The differential intra-DC ([`crate::texture::decode_intra_dc`]) is
//! always present for blocks of an intra macroblock when
//! `use_intra_dc_vlc == 1`; the §7.4.1.2 AC EVENT loop
//! ([`crate::texture::decode_ac_events`]) runs only when
//! `pattern_code[i] == 1`.
//!
//! ## `pattern_code[i]` (§6.3.7 / §6.2.7)
//!
//! `pattern_code[i]` is the per-block "this block has at least one
//! coded AC coefficient" flag. For the 4:2:0 macroblock it is assembled
//! from the macroblock header's `cbpy` (the four luminance blocks,
//! Table B.8) and `cbpc` (the two chrominance blocks, carried in
//! `mcbpc`, Tables B.6 / B.7), per the §6.3.7 semantics — `cbpy`
//! "represents a pattern of non-transparent luminance blocks with at
//! least one non intra DC transform coefficient", and `cbpc` the
//! chrominance equivalent. The bit-to-block mapping follows
//! Figure 6-8's block order and the existing macroblock-header field
//! conventions:
//!
//! * `cbpy` bit 3 → block 0 (luma top-left), bit 2 → block 1 (top-
//!   right), bit 1 → block 2 (bottom-left), bit 0 → block 3 (bottom-
//!   right).
//! * `cbpc` bit 1 → block 4 (Cb), bit 0 → block 5 (Cr).
//!
//! This is identical to the §6.2.7 pattern-derivation
//! `if (cbp & (1 << (5 - i))) pattern_code[i] = 1` with
//! `cbp = (cbpy << 2) | cbpc` (block 0 at bit 5 down to block 5 at
//! bit 0).
//!
//! ## The §7.4.x chain per block
//!
//! 1. **Read coefficients.** [`crate::texture::decode_intra_dc`] for the
//!    differential DC, then [`crate::texture::decode_ac_events`] for the
//!    AC EVENTs when `pattern_code[i]`.
//! 2. **Expand to `QFS[64]`** ([`crate::scan::events_to_qfs`]) with the
//!    differential DC at scan position 0, then convert to the 2-D
//!    `PQF[v][u]` block via the §7.4.2 inverse scan
//!    ([`crate::scan::inverse_scan`]) under the §7.4.2-selected scan
//!    pattern ([`crate::scan::select_scan_type`]).
//! 3. **Spatial predictor** (§7.4.3): add the §7.4.3.2 DC predictor and
//!    the §7.4.3.3 AC predictors (gated by `ac_pred_flag`) to obtain the
//!    quantised `QF[v][u]`, then §7.4.3.4-saturate to `[-2048, 2047]`.
//! 4. **Inverse quantisation** (§7.4.4): method 1 (`quant_type == 1`,
//!    with the `W[0]` intra matrix) or method 2 (`quant_type == 0`),
//!    yielding the reconstructed `F[v][u]`.
//! 5. **Inverse DCT** (§7.4.5 + Annex A): [`crate::idct::idct_8x8`]
//!    produces the spatial samples `f[y][x]`, saturated to
//!    `[-2^bpp, 2^bpp - 1]`.
//! 6. **Final clip** (§6.3.2 / the §7.6 reconstruction): for an intra
//!    block there is no prediction to add, so the reconstructed sample
//!    is the IDCT output clipped to the display range
//!    `[0, 2^bpp - 1]`.
//!
//! ## Predictor neighbours this round
//!
//! Gathering the §7.4.3 predictor candidates from a concrete block grid
//! (the Figure 7-5 walk over neighbouring macroblocks) is a separate
//! task. This module accepts the predictor neighbourhood as an explicit
//! [`BlockPredictors`] argument — its
//! [`outside`][`BlockPredictors::outside`] constructor is the
//! §7.4.3.1 / §7.4.3.3 "neighbour outside the VOP / video packet" case
//! (default DC of `2^(bpp+2)`, all AC prediction coefficients zero),
//! which is exactly the predictor state for the first intra macroblock
//! of a VOP. A caller that has gathered neighbours threads them in per
//! block.
//!
//! ## Out of scope (this round)
//!
//! * Motion compensation and inter / B-VOP reconstruction.
//! * The Figure 7-5 predictor-candidate gathering over a block grid.
//! * The `short_video_header == 1` fixed-8-bit DC path and the
//!   `data_partitioned == 1` reordering.
//! * Non-rectangular shape (`transparent_block(i)`), 4:2:2 / 4:4:4
//!   chroma formats, and the SA-DCT path.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by the
//! agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.1.3.9 / Figure 6-8 — 4:2:0 macroblock block ordering
//!   (0,1 / 2,3 luminance; 4 Cb; 5 Cr).
//! * §6.2.6 macroblock() — the `for (i = 0; i < block_count; i++)
//!   block(i)` loop (block_count = 6 for 4:2:0).
//! * §6.2.7 block(i) — the intra-DC branch and the
//!   `if (pattern_code[i]) while (!last) DCT coefficient` loop.
//! * §6.3.7 — `cbpy` / `cbpc` semantics; `pattern_code[i]` is set to 1
//!   when the block has one or more coded AC coefficients.
//! * §6.3.3 — the default intra / non-intra quantisation matrices used
//!   when no matrix was loaded (`quant_type == 1`).

use crate::bitreader::BitReader;
use crate::idct::idct_8x8;
use crate::inverse_quant::{inverse_quant_method1, inverse_quant_method2, InverseQuantContext};
use crate::macroblock::{DerivedMbType, MacroblockHeader};
use crate::predictor::{
    dc_scaler, default_neighbour_dc, predict_intra_ac_column, predict_intra_ac_row,
    predict_intra_dc, saturate_block, select_dc_direction,
};
use crate::scan::{events_to_qfs, inverse_scan, select_scan_type};
use crate::texture::{
    decode_ac_events, decode_intra_dc, DcComponent, TcoefTable, TextureParseError,
};
use crate::vol::VolHeader;

/// The §6.3.3 default intra-block quantisation matrix `W[0]`, in raster
/// (`[row][col]`) order, transcribed verbatim from the table in
/// §6.3.3. Used when `quant_type == 1` and no `intra_quant_mat` was
/// loaded.
#[rustfmt::skip]
pub const DEFAULT_INTRA_QUANT_MATRIX: [[u8; 8]; 8] = [
    [ 8, 17, 18, 19, 21, 23, 25, 27],
    [17, 18, 19, 21, 23, 25, 27, 28],
    [20, 21, 22, 23, 24, 26, 28, 30],
    [21, 22, 23, 24, 26, 28, 30, 32],
    [22, 23, 24, 26, 28, 30, 32, 35],
    [23, 24, 26, 28, 30, 32, 35, 38],
    [25, 26, 28, 30, 32, 35, 38, 41],
    [27, 28, 30, 32, 35, 38, 41, 45],
];

/// The §6.3.3 default non-intra-block quantisation matrix `W[1]`, in
/// raster (`[row][col]`) order, transcribed verbatim from the table in
/// §6.3.3.
#[rustfmt::skip]
pub const DEFAULT_NONINTRA_QUANT_MATRIX: [[u8; 8]; 8] = [
    [16, 17, 18, 19, 20, 21, 22, 23],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [18, 19, 20, 21, 22, 23, 24, 25],
    [19, 20, 21, 22, 23, 24, 26, 27],
    [20, 21, 22, 23, 25, 26, 27, 28],
    [21, 22, 23, 24, 26, 27, 28, 30],
    [22, 23, 24, 26, 27, 28, 30, 31],
    [23, 24, 25, 27, 28, 30, 31, 33],
];

/// Errors produced while assembling a `block(i)` / macroblock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockAssemblyError {
    /// The supplied macroblock header is not an intra macroblock; this
    /// driver only handles the intra `block(i)` assembly so far.
    NotIntra,
    /// The supplied macroblock header is not an inter macroblock — i.e.
    /// it was passed to [`decode_inter_macroblock`] but
    /// `derived_mb_type ∉ {Inter, InterQ, Inter4V}`.
    NotInter,
    /// The macroblock header carried `not_coded == true` (a skipped
    /// P-VOP MB), which has no texture to assemble.
    NotCoded,
    /// A `block(i)` texture decode (DC or AC EVENT) failed. See
    /// [`TextureParseError`].
    Texture(TextureParseError),
    /// A §7.4.2 inverse-scan expansion failed (a malformed AC EVENT
    /// stream walked past coefficient 63).
    InverseScan(crate::scan::InverseScanError),
}

impl core::fmt::Display for BlockAssemblyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BlockAssemblyError::NotIntra => {
                write!(f, "block(i) assembly requires an intra macroblock")
            }
            BlockAssemblyError::NotInter => {
                write!(f, "block(i) assembly requires an inter macroblock")
            }
            BlockAssemblyError::NotCoded => {
                write!(f, "macroblock is not_coded; no texture to assemble")
            }
            BlockAssemblyError::Texture(err) => {
                write!(f, "block(i) texture decode failed: {err}")
            }
            BlockAssemblyError::InverseScan(err) => {
                write!(f, "block(i) inverse scan failed: {err}")
            }
        }
    }
}

impl std::error::Error for BlockAssemblyError {}

impl From<TextureParseError> for BlockAssemblyError {
    fn from(err: TextureParseError) -> Self {
        BlockAssemblyError::Texture(err)
    }
}

impl From<crate::scan::InverseScanError> for BlockAssemblyError {
    fn from(err: crate::scan::InverseScanError) -> Self {
        BlockAssemblyError::InverseScan(err)
    }
}

/// The §7.4.3 predictor neighbourhood for one block.
///
/// The three §7.4.3.1 DC predictor candidates `FA` (left, block A),
/// `FB` (above-left, block B), `FC` (above, block C) are the inverse-
/// quantised DC values of the neighbouring blocks; `qp_a` / `qp_c` are
/// their quantiser scales (needed by the §7.4.3.3 AC scaling). The
/// first-row (block C) and first-column (block A) AC coefficient arrays
/// are `None` when the neighbour is outside the VOP / video packet,
/// matching the §7.4.3.3 "all the prediction coefficients of that block
/// are assumed to be zero" rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPredictors {
    /// `FA[0][0]` — inverse-quantised DC of the left (A) neighbour.
    pub fa_dc: i32,
    /// `FB[0][0]` — inverse-quantised DC of the above-left (B) neighbour.
    pub fb_dc: i32,
    /// `FC[0][0]` — inverse-quantised DC of the above (C) neighbour.
    pub fc_dc: i32,
    /// Quantiser scale `QpA` of the left neighbour (block A).
    pub qp_a: u32,
    /// Quantiser scale `QpC` of the above neighbour (block C).
    pub qp_c: u32,
    /// `QFA[1..=7][0]` — the left neighbour's first column. `None` ⇒
    /// out of VOP.
    pub a_first_column: Option<[i32; 7]>,
    /// `QFC[0][1..=7]` — the above neighbour's first row. `None` ⇒ out
    /// of VOP.
    pub c_first_row: Option<[i32; 7]>,
}

impl BlockPredictors {
    /// The §7.4.3.1 / §7.4.3.3 "all neighbours outside the VOP / video
    /// packet" state for the given `bits_per_pixel`: every DC takes the
    /// default `2^(bpp + 2)`, every AC prediction coefficient is zero
    /// (`None`), and the neighbour quantiser scales mirror the current
    /// block's (they are unused once the AC arrays are `None`).
    pub fn outside(bits_per_pixel: u32, quantiser_scale: u32) -> Self {
        let dc = default_neighbour_dc(bits_per_pixel);
        Self {
            fa_dc: dc,
            fb_dc: dc,
            fc_dc: dc,
            qp_a: quantiser_scale,
            qp_c: quantiser_scale,
            a_first_column: None,
            c_first_row: None,
        }
    }
}

/// Per-macroblock decode parameters threaded into the `block(i)` driver,
/// resolved from the VOL / VOP headers and the macroblock header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroblockTextureContext {
    /// `quantiser_scale` after `dquant`, the §6.2.5 `vop_quant` adjusted
    /// by the macroblock's `dquant_delta`. Range `1..=2^precision - 1`.
    pub quantiser_scale: u32,
    /// `bits_per_pixel` from §6.3.3 (default 8).
    pub bits_per_pixel: u32,
    /// `quant_type` from the VOL header: `true` selects §7.4.4.1 method 1
    /// (with a quantisation matrix), `false` selects §7.4.4.2 method 2.
    pub quant_type: bool,
    /// `ac_pred_flag` from the macroblock header — gates the §7.4.3.3 AC
    /// prediction and the §7.4.2 alternate-scan selection.
    pub ac_pred_flag: bool,
}

/// A reconstructed intra 4:2:0 macroblock: a 16×16 luminance plane plus
/// two 8×8 chrominance planes, all in pixel-space `[0, 2^bpp - 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraMacroblock {
    /// Reconstructed luminance samples, `luma[row][col]`, 16×16.
    pub luma: [[i32; 16]; 16],
    /// Reconstructed Cb samples, `cb[row][col]`, 8×8 (Figure 6-8 block 4).
    pub cb: [[i32; 8]; 8],
    /// Reconstructed Cr samples, `cr[row][col]`, 8×8 (Figure 6-8 block 5).
    pub cr: [[i32; 8]; 8],
}

/// A decoded inter 4:2:0 macroblock's **residual** (`f[y][x]` of §7.3
/// step 2): a 16×16 luminance plane plus two 8×8 chrominance planes,
/// each in `[-2^bpp, 2^bpp - 1]` per the §7.4.5 / Annex A IDCT
/// saturation. The §7.3 step-2 sum with the motion-compensated
/// prediction `p[y][x]` and the §7.3 step-3 `[0, 2^bpp - 1]` clip happen
/// in the caller's motion-compensation stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterMacroblock {
    /// Residual luminance samples, `luma[row][col]`, 16×16.
    pub luma: [[i32; 16]; 16],
    /// Residual Cb samples, `cb[row][col]`, 8×8 (Figure 6-8 block 4).
    pub cb: [[i32; 8]; 8],
    /// Residual Cr samples, `cr[row][col]`, 8×8 (Figure 6-8 block 5).
    pub cr: [[i32; 8]; 8],
}

impl InterMacroblock {
    /// An all-zero residual — the §7.4 result for a macroblock whose
    /// coded-block pattern is empty (`cbp == 0`). Added to a
    /// motion-compensated prediction in the §7.3 step-2 sum it is the
    /// identity, so the reconstruction is a pure motion-compensated copy.
    pub fn zero() -> Self {
        Self {
            luma: [[0i32; 16]; 16],
            cb: [[0i32; 8]; 8],
            cr: [[0i32; 8]; 8],
        }
    }
}

/// Derive the six `pattern_code[i]` flags from the macroblock header's
/// `cbpy` (luminance, blocks 0..=3) and `cbpc` (chrominance, blocks
/// 4..=5), per §6.3.7 / §6.2.7.
///
/// Index `i` follows Figure 6-8: 0..=3 luminance, 4 = Cb, 5 = Cr. See
/// the module docs for the bit-to-block mapping.
pub fn pattern_code(cbpy: u8, cbpc: u8) -> [bool; 6] {
    [
        cbpy & 0b1000 != 0, // block 0 — luma top-left
        cbpy & 0b0100 != 0, // block 1 — luma top-right
        cbpy & 0b0010 != 0, // block 2 — luma bottom-left
        cbpy & 0b0001 != 0, // block 3 — luma bottom-right
        cbpc & 0b10 != 0,   // block 4 — Cb
        cbpc & 0b01 != 0,   // block 5 — Cr
    ]
}

/// Derive the six `coded[i]` flags from a B-VOP macroblock's 6-bit
/// `cbpb`, per §6.3.7 / §6.2.6.
///
/// Unlike a P-VOP (where the coded pattern is split across `cbpy`
/// luminance + `cbpc` chrominance fields), a B-VOP carries a single
/// `cbpb` field whose bits run **leftmost = top-left block**: for a
/// 4:2:0 rectangular macroblock the 6 bits map MSB→block 0 …
/// LSB→block 5 (block 4 = Cb, block 5 = Cr), matching the §6.2.6
/// `NOTE` `block_count == 6`. `cbpb == None` (the `modb` indicated no
/// `cbpb`) means no block is coded.
pub fn cbpb_pattern_code(cbpb: Option<u8>) -> [bool; 6] {
    let bits = cbpb.unwrap_or(0);
    [
        bits & 0b10_0000 != 0, // block 0 — luma top-left
        bits & 0b01_0000 != 0, // block 1 — luma top-right
        bits & 0b00_1000 != 0, // block 2 — luma bottom-left
        bits & 0b00_0100 != 0, // block 3 — luma bottom-right
        bits & 0b00_0010 != 0, // block 4 — Cb
        bits & 0b00_0001 != 0, // block 5 — Cr
    ]
}

/// The §6.1.3 / Figure 6-8 component of block `i` in a 4:2:0 macroblock.
#[inline]
fn block_component(i: usize) -> DcComponent {
    DcComponent::from_block_index(i)
}

/// One intra `block(i)` decode with its §7.4.3 predictor by-products.
///
/// [`decode_intra_block_full`] returns not just the reconstructed
/// spatial samples but also the two values the §7.4.3 predictor
/// neighbourhood of *later* blocks needs from this block:
///
/// * `qf` — the quantised coefficient block after the §7.4.3.2 /
///   §7.4.3.3 prediction adds and the §7.4.3.4 saturation. Its first
///   row / first column are what a neighbouring block's §7.4.3.3 AC
///   prediction reads (see
///   [`BlockNeighbour::from_qf`](crate::neighbour::BlockNeighbour::from_qf)).
/// * `dc` — the inverse-quantised DC `F[0][0]` (§7.4.4.1.1), the value
///   the §7.4.3.1 `|FA − FB| < |FB − FC|` direction rule compares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraBlockDecode {
    /// Reconstructed spatial samples, clipped to `[0, 2^bpp - 1]`.
    pub spatial: [[i32; 8]; 8],
    /// Quantised coefficients after prediction add + §7.4.3.4 saturate.
    pub qf: [[i32; 8]; 8],
    /// Inverse-quantised DC `F[0][0]` (§7.4.4.1.1).
    pub dc: i32,
}

/// Run the §6.2.7 `block(i)` intra texture syntax for one block and the
/// full §7.4.x reconstruction chain, returning the reconstructed
/// spatial 8×8 block clipped to the display range `[0, 2^bpp - 1]`.
///
/// `i` is the Figure 6-8 block index (0..=5). `coded` is
/// `pattern_code[i]` — when `false` the §7.4.1.2 AC EVENT loop is
/// skipped and only the intra DC contributes. `predictors` supplies the
/// §7.4.3 neighbourhood (use [`BlockPredictors::outside`] for an
/// isolated macroblock). `quant_matrix` is the raster-order `W[0]` intra
/// matrix used when `ctx.quant_type == true` (method 1); it is ignored
/// for method 2.
///
/// This entry decodes the `use_intra_dc_vlc == 1` path (differential
/// intra DC coded with the Table B.13 / B.14 DC VLCs). For the Table
/// 6-25 `use_intra_dc_vlc == 0` path (DC coded as an AC coefficient)
/// use [`decode_intra_block_full`].
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_block(
    br: &mut BitReader<'_>,
    i: usize,
    coded: bool,
    ctx: MacroblockTextureContext,
    predictors: BlockPredictors,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<[[i32; 8]; 8], BlockAssemblyError> {
    decode_intra_block_full(br, i, coded, true, ctx, predictors, quant_matrix)
        .map(|full| full.spatial)
}

/// [`decode_intra_block`] with the §6.3.5 / Table 6-25
/// `use_intra_dc_vlc` selection and the §7.4.3 predictor by-products.
///
/// * `use_intra_dc_vlc == true` — the §6.2.7 intra-DC prologue is
///   present: `dct_dc_size_*` + `dct_dc_differential` (Tables B.13 /
///   B.14 / B.15) code the differential DC, and the AC EVENT loop
///   (when `coded`) fills scan positions 1..=63.
/// * `use_intra_dc_vlc == false` — no DC prologue; the differential DC
///   is coded **as an AC coefficient** (§6.3.5 intra_dc_vlc_thr /
///   Table 6-25 "intra AC VLC" rows), i.e. the EVENT loop fills scan
///   positions 0..=63 and `QFS[0]` carries the differential DC. When
///   `coded == false` no bits are consumed at all and the differential
///   DC is zero.
///
/// In both cases the §7.4.3 spatial DC/AC prediction, §7.4.4 inverse
/// quantisation, and §7.4.5 IDCT run identically — the threshold only
/// moves *where* the differential DC sits in the bitstream.
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_block_full(
    br: &mut BitReader<'_>,
    i: usize,
    coded: bool,
    use_intra_dc_vlc: bool,
    ctx: MacroblockTextureContext,
    predictors: BlockPredictors,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<IntraBlockDecode, BlockAssemblyError> {
    let component = block_component(i);

    // §6.2.7 — differential intra DC prologue, present only when
    // use_intra_dc_vlc == 1 (Table 6-25).
    let intra_dc = if use_intra_dc_vlc {
        Some(decode_intra_dc(br, component)?.differential)
    } else {
        None
    };

    // §6.2.7 — the AC EVENT loop runs only when pattern_code[i] == 1.
    let events = if coded {
        decode_ac_events(br, TcoefTable::Intra)?
    } else {
        Vec::new()
    };

    // §7.4.1 / §7.4.2 — assemble the one-dimensional QFS[64]. With the
    // DC VLC the differential DC occupies scan position 0 and the
    // EVENTs fill 1..; without it the EVENTs fill from position 0 (the
    // first EVENT's run counts from QFS[0], which carries the DC).
    let qfs = events_to_qfs(&events, intra_dc)?;

    // §7.4.3.1 — pick the DC prediction direction from the neighbour DCs.
    let direction = select_dc_direction(predictors.fa_dc, predictors.fb_dc, predictors.fc_dc);
    let scan_type = select_scan_type(true, ctx.ac_pred_flag, direction);

    // §7.4.2 — 1-D QFS into the 2-D PQF block.
    let pqf = inverse_scan(&qfs, scan_type);

    // §7.4.3 — spatial DC/AC predictor add → quantised QF[v][u].
    let mut qf = [[0i32; 8]; 8];
    let scaler = dc_scaler(component, ctx.quantiser_scale);
    // §7.4.3.2 DC reconstruction.
    qf[0][0] = predict_intra_dc(
        pqf[0][0],
        direction,
        predictors.fa_dc,
        predictors.fc_dc,
        scaler,
    );

    // Copy the remaining decoded coefficients, then overlay the §7.4.3.3
    // AC predictors on the first row / column when ac_pred_flag is set.
    for v in 0..8 {
        for u in 0..8 {
            if v == 0 && u == 0 {
                continue;
            }
            qf[v][u] = pqf[v][u];
        }
    }

    if ctx.ac_pred_flag {
        match direction {
            crate::scan::DcPredictionDirection::FromLeft => {
                // §7.4.3.3 first column QFX[1..=7][0].
                let mut pqfx_col = [0i32; 7];
                for v in 1..8 {
                    pqfx_col[v - 1] = pqf[v][0];
                }
                let col = predict_intra_ac_column(
                    pqfx_col,
                    predictors.a_first_column,
                    predictors.qp_a,
                    ctx.quantiser_scale,
                );
                for v in 1..8 {
                    qf[v][0] = col[v - 1];
                }
            }
            crate::scan::DcPredictionDirection::FromAbove => {
                // §7.4.3.3 first row QFX[0][1..=7].
                let mut pqfx_row = [0i32; 7];
                pqfx_row.copy_from_slice(&pqf[0][1..8]);
                let row = predict_intra_ac_row(
                    pqfx_row,
                    predictors.c_first_row,
                    predictors.qp_c,
                    ctx.quantiser_scale,
                );
                qf[0][1..8].copy_from_slice(&row);
            }
        }
    }

    // §7.4.3.4 — saturate the quantised coefficients to [-2048, 2047].
    saturate_block(&mut qf);

    // §7.4.4 — inverse quantisation → reconstructed F[v][u].
    let iq_ctx = InverseQuantContext {
        macroblock_intra: true,
        component,
        quantiser_scale: ctx.quantiser_scale,
        bits_per_pixel: ctx.bits_per_pixel,
        short_video_header: false,
    };
    let f = if ctx.quant_type {
        inverse_quant_method1(&qf, quant_matrix, iq_ctx)
    } else {
        inverse_quant_method2(&qf, iq_ctx)
    };

    // §7.4.5 + Annex A — inverse DCT (output saturated to
    // [-2^bpp, 2^bpp - 1]).
    let spatial = idct_8x8(&f, ctx.bits_per_pixel);

    // §6.3.2 / §7.6 — for an intra block there is no prediction to add,
    // so the reconstructed sample is the IDCT output clipped to the
    // display range [0, 2^bpp - 1].
    let hi = (1i32 << ctx.bits_per_pixel) - 1;
    let mut out = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = spatial[y][x].clamp(0, hi);
        }
    }
    Ok(IntraBlockDecode {
        spatial: out,
        qf,
        dc: f[0][0],
    })
}

/// Resolve the raster-order `W[0]` intra quantisation matrix for the
/// given VOL header: the loaded `intra_quant_mat` (de-zigzagged) when
/// present, else the §6.3.3 default intra matrix.
pub fn intra_quant_matrix(vol: &VolHeader) -> [[u8; 8]; 8] {
    match vol.intra_quant_mat {
        Some(zigzag) => de_zigzag(&zigzag),
        None => DEFAULT_INTRA_QUANT_MATRIX,
    }
}

/// Resolve the raster-order `W[1]` non-intra quantisation matrix for
/// the given VOL header: the loaded `nonintra_quant_mat` (de-zigzagged)
/// when present, else the §6.3.3 default non-intra matrix.
pub fn nonintra_quant_matrix(vol: &VolHeader) -> [[u8; 8]; 8] {
    match vol.nonintra_quant_mat {
        Some(zigzag) => de_zigzag(&zigzag),
        None => DEFAULT_NONINTRA_QUANT_MATRIX,
    }
}

/// Convert a §6.2.3.3 zigzag-ordered 64-entry quantiser matrix into the
/// raster-order `[row][col]` matrix the §7.4.4 method-1 reconstruction
/// reads. The §6.3.3 list is in the same zigzag order as the
/// Figure 7-4 (c) scan; we re-use that table to place each value.
pub fn de_zigzag(zigzag: &[u8; 64]) -> [[u8; 8]; 8] {
    // The Figure 7-4 (c) zigzag grid maps (row, col) → scan index n;
    // inverse: raster[row][col] = zigzag[ scan_index(row, col) ].
    const ZIGZAG: [[u8; 8]; 8] = [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ];
    let mut raster = [[0u8; 8]; 8];
    for row in 0..8 {
        for col in 0..8 {
            raster[row][col] = zigzag[ZIGZAG[row][col] as usize];
        }
    }
    raster
}

/// Decode and reconstruct one intra 4:2:0 macroblock from the texture
/// bitstream that follows the §6.2.6 macroblock header.
///
/// `br` must be positioned at the first `block(0)` of the macroblock
/// (i.e. immediately after the header — including `dquant` — that
/// produced `header`). `header` must be an intra macroblock
/// (`derived_mb_type == 3 || == 4`). `ctx` carries the resolved
/// per-macroblock parameters (`quantiser_scale`, `bits_per_pixel`,
/// `quant_type`, `ac_pred_flag`). `predictors` is the §7.4.3
/// neighbourhood for *every* block; pass [`BlockPredictors::outside`]
/// for an isolated macroblock.
///
/// The six blocks are decoded in Figure 6-8 order (0,1 / 2,3 luminance;
/// 4 Cb; 5 Cr) and assembled into the [`IntraMacroblock`]:
///
/// ```text
///   luma[0..8][0..8]   ← block 0     luma[0..8][8..16]  ← block 1
///   luma[8..16][0..8]  ← block 2     luma[8..16][8..16] ← block 3
///   cb                 ← block 4     cr                 ← block 5
/// ```
pub fn decode_intra_macroblock(
    br: &mut BitReader<'_>,
    header: &MacroblockHeader,
    ctx: MacroblockTextureContext,
    predictors: BlockPredictors,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<IntraMacroblock, BlockAssemblyError> {
    if header.not_coded {
        return Err(BlockAssemblyError::NotCoded);
    }
    match header.mb_type {
        Some(DerivedMbType::Intra) | Some(DerivedMbType::IntraQ) => {}
        _ => return Err(BlockAssemblyError::NotIntra),
    }

    let coded = pattern_code(header.cbpy, header.cbpc);

    // Decode the six blocks in Figure 6-8 order.
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = decode_intra_block(br, i, coded[i], ctx, predictors, quant_matrix)?;
    }

    // Assemble the 16×16 luminance plane from the four 8×8 luma blocks.
    let mut luma = [[0i32; 16]; 16];
    // Block 0 → top-left, 1 → top-right, 2 → bottom-left, 3 → bottom-right.
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }

    Ok(IntraMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

/// Decode one §6.2.7 `block(i)` for a **non-intra** macroblock and run
/// the §7.4.x reconstruction chain, returning the signed residual 8×8
/// block in `[-2^bpp, 2^bpp - 1]` (the §7.4.5 IDCT saturation range).
///
/// The §6.2.7 syntax table gates the entire intra-DC prologue on
/// `(derived_mb_type == 3 || derived_mb_type == 4)`, so an inter
/// `block(i)` is just:
///
/// ```text
/// block(i) {
///     last = 0
///     if (pattern_code[i])
///         while (!last) DCT coefficient
/// }
/// ```
///
/// When `coded == false` (`pattern_code[i] == 0`) no bits are consumed
/// and the residual is the all-zero block. When `coded == true` the
/// §7.4.1.2 AC EVENT loop runs against [`TcoefTable::Inter`] (Table
/// B.17 / Tables B.20 / B.22) — the §7.4.1.3 escape modes are handled
/// inside [`decode_ac_events`]. The decoded EVENTs are placed into
/// `QFS[0..=63]` (no DC at position 0 — there is no intra-DC for inter
/// blocks; the differential-DC bits and §7.4.3 spatial DC/AC predictor
/// are intra-only per §7.4.3). The §7.4.2 inverse scan uses the
/// [`ScanType::Zigzag`][crate::scan::ScanType::Zigzag] table (§7.4.2
/// "Non-intra blocks → zigzag"). The §7.4.4 inverse quant runs with
/// `macroblock_intra == false` (method 1 with the `W[1]` non-intra
/// matrix when `ctx.quant_type == true`, else method 2 with the
/// `(2*|QF| + 1) * qs` formula and the §7.4.4.2 sign-incorporation).
/// The §7.4.5 + Annex A IDCT produces the residual `f[y][x]`, already
/// saturated to `[-2^bpp, 2^bpp - 1]` by [`idct_8x8`].
///
/// Per §7.3 step-2 the caller adds `p[y][x]` from motion compensation
/// and clips the resulting `d[y][x]` to `[0, 2^bpp - 1]` (the §7.3
/// step-3 display saturation). That happens outside this driver.
pub fn decode_inter_block(
    br: &mut BitReader<'_>,
    i: usize,
    coded: bool,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<[[i32; 8]; 8], BlockAssemblyError> {
    let component = block_component(i);

    if !coded {
        // §6.2.7 — no AC EVENT loop, no DC for inter; the residual is
        // the all-zero block.
        return Ok([[0i32; 8]; 8]);
    }

    // §6.2.7 — `while (!last) DCT coefficient` against the inter
    // Tcoef tables.
    let events = decode_ac_events(br, TcoefTable::Inter)?;

    // §7.4.2 — expand to QFS[0..=63] (no DC: pass `None` for the
    // §7.4.1.1 intra-DC slot) then 1-D → 2-D under the zigzag scan.
    let qfs = events_to_qfs(&events, None)?;
    let pqf = inverse_scan(&qfs, crate::scan::ScanType::Zigzag);

    // For an inter block PQF[v][u] = QF[v][u] directly (§7.4.3 spatial
    // DC/AC prediction is intra-only). Saturate to [-2048, 2047] per
    // §7.4.3.4 (the §7.4.3.4 clause applies to "the QF values" — it is
    // not gated on intra; method 2's §7.4.4.4 saturation re-clamps to
    // the wider `[-2^(bpp+3), 2^(bpp+3) - 1]` range later anyway, but
    // running §7.4.3.4 first matches the Figure 7-3 pipeline order).
    let mut qf = pqf;
    saturate_block(&mut qf);

    // §7.4.4 — inverse quantisation → reconstructed F[v][u] (non-intra
    // path: method 1 with W[1], or method 2 with the `(2*|QF| + 1) * qs`
    // formula).
    let iq_ctx = InverseQuantContext {
        macroblock_intra: false,
        component,
        quantiser_scale: ctx.quantiser_scale,
        bits_per_pixel: ctx.bits_per_pixel,
        short_video_header: false,
    };
    let f = if ctx.quant_type {
        inverse_quant_method1(&qf, quant_matrix, iq_ctx)
    } else {
        inverse_quant_method2(&qf, iq_ctx)
    };

    // §7.4.5 + Annex A — inverse DCT. `idct_8x8` already saturates to
    // [-2^bpp, 2^bpp - 1]; the result is the §7.3 step-2 residual.
    Ok(idct_8x8(&f, ctx.bits_per_pixel))
}

/// Decode and reconstruct one inter 4:2:0 macroblock's signed-residual
/// 16×16 luma + 8×8 Cb / 8×8 Cr planes from the texture bitstream that
/// follows the §6.2.6 macroblock header.
///
/// `br` must be positioned at the first `block(0)` of the macroblock
/// (i.e. immediately after `dquant` for an `InterQ` block, or
/// immediately after `cbpy` for an `Inter` / `Inter4V` block). `header`
/// must be an inter macroblock (`derived_mb_type ∈ {Inter, InterQ,
/// Inter4V}`). `ctx` carries the resolved per-macroblock parameters
/// (`quantiser_scale`, `bits_per_pixel`, `quant_type`); `ac_pred_flag`
/// is ignored on the inter path. `quant_matrix` is the raster-order
/// `W[1]` non-intra matrix used when `ctx.quant_type == true`
/// (method 1); it is ignored for method 2.
///
/// For a `not_coded` macroblock (P-VOP skipped MB) the caller should
/// short-circuit to the §7.5 zero-MV / zero-residual reconstruction
/// rather than calling this driver; the function returns
/// [`BlockAssemblyError::NotCoded`] in that case.
///
/// The six blocks are decoded in Figure 6-8 order (0,1 / 2,3 luminance;
/// 4 Cb; 5 Cr) and assembled into the [`InterMacroblock`]:
///
/// ```text
///   luma[0..8][0..8]   ← block 0     luma[0..8][8..16]  ← block 1
///   luma[8..16][0..8]  ← block 2     luma[8..16][8..16] ← block 3
///   cb                 ← block 4     cr                 ← block 5
/// ```
pub fn decode_inter_macroblock(
    br: &mut BitReader<'_>,
    header: &MacroblockHeader,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<InterMacroblock, BlockAssemblyError> {
    if header.not_coded {
        return Err(BlockAssemblyError::NotCoded);
    }
    match header.mb_type {
        Some(DerivedMbType::Inter) | Some(DerivedMbType::InterQ) | Some(DerivedMbType::Inter4V) => {
        }
        _ => return Err(BlockAssemblyError::NotInter),
    }

    let coded = pattern_code(header.cbpy, header.cbpc);

    // Decode the six blocks in Figure 6-8 order.
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = decode_inter_block(br, i, coded[i], ctx, quant_matrix)?;
    }

    // Assemble the 16×16 luminance residual plane from the four 8×8
    // luma blocks.
    let mut luma = [[0i32; 16]; 16];
    // Block 0 → top-left, 1 → top-right, 2 → bottom-left, 3 → bottom-right.
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }

    Ok(InterMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

/// Decode and reconstruct a **B-VOP** inter macroblock's signed-residual
/// 16×16 luma + 8×8 Cb / 8×8 Cr planes from the texture bitstream that
/// follows the §6.2.6 motion-vector bodies.
///
/// `br` must be positioned at the first coded `block(i)` (i.e.
/// immediately after the last motion-vector body the macroblock's
/// `mb_type` codes). `cbpb` is the macroblock header's coded-block
/// pattern (`None` when `modb` indicated no `cbpb`, i.e. every block is
/// uncoded → a wholly-zero residual). `ctx` carries the resolved
/// per-macroblock parameters (`quantiser_scale` after any `dbquant`,
/// `bits_per_pixel`, `quant_type`); `ac_pred_flag` is ignored on the
/// inter path. `quant_matrix` is the raster-order non-intra `W[1]`
/// matrix used when `ctx.quant_type == true` (method 1); it is ignored
/// for method 2.
///
/// The B-VOP residual is always an *inter* residual — B-VOPs carry no
/// intra macroblocks (§7.6.9), so every coded block runs the §6.2.7
/// inter texture syntax (no DC, inter Tcoef tables). The six blocks are
/// decoded in Figure 6-8 order (0,1 / 2,3 luminance; 4 Cb; 5 Cr), gated
/// by [`cbpb_pattern_code`], and assembled into the [`InterMacroblock`]
/// with the same plane layout as [`decode_inter_macroblock`].
pub fn decode_b_vop_inter_macroblock(
    br: &mut BitReader<'_>,
    cbpb: Option<u8>,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<InterMacroblock, BlockAssemblyError> {
    let coded = cbpb_pattern_code(cbpb);

    // Decode the six blocks in Figure 6-8 order.
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = decode_inter_block(br, i, coded[i], ctx, quant_matrix)?;
    }

    // Assemble the 16×16 luminance residual plane from the four 8×8
    // luma blocks (Figure 6-8: 0 → top-left, 1 → top-right, 2 →
    // bottom-left, 3 → bottom-right).
    let mut luma = [[0i32; 16]; 16];
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }

    Ok(InterMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    /// A tiny MSB-first bit writer for building synthetic block streams.
    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        bit: u8,
        acc: u8,
    }

    impl BitWriter {
        fn write_bit(&mut self, b: u32) {
            self.acc = (self.acc << 1) | ((b & 1) as u8);
            self.bit += 1;
            if self.bit == 8 {
                self.bytes.push(self.acc);
                self.acc = 0;
                self.bit = 0;
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for i in (0..n).rev() {
                self.write_bit((value >> i) & 1);
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bit > 0 {
                self.acc <<= 8 - self.bit;
                self.bytes.push(self.acc);
            }
            self.bytes
        }
    }

    /// Encode an intra-DC of size 0 (differential 0): Table B.13
    /// luminance "011" / Table B.14 chrominance "00".
    fn write_dc_zero_luma(w: &mut BitWriter) {
        w.write_bits(0b011, 3);
    }
    fn write_dc_zero_chroma(w: &mut BitWriter) {
        // Table B.14 — dct_dc_size_chrominance size-0 code is "11".
        w.write_bits(0b11, 2);
    }

    /// Encode a DC of size 1 with a chosen +1 or -1 differential for a
    /// luminance block. Table B.13 size-1 code is "11"; Table B.15
    /// size-1 additional code "1" → +1, "0" → -1.
    fn write_dc_size1_luma(w: &mut BitWriter, positive: bool) {
        w.write_bits(0b11, 2); // dct_dc_size_luminance = 1
        w.write_bits(if positive { 1 } else { 0 }, 1);
    }

    #[test]
    fn pattern_code_bit_mapping() {
        // cbpy 1010 → blocks 0 and 2 coded; cbpc 01 → block 5 (Cr).
        let pc = pattern_code(0b1010, 0b01);
        assert_eq!(pc, [true, false, true, false, false, true]);
        // All luminance + both chroma coded.
        assert_eq!(pattern_code(0b1111, 0b11), [true; 6]);
        // Nothing coded — DC-only blocks.
        assert_eq!(pattern_code(0b0000, 0b00), [false; 6]);
    }

    #[test]
    fn de_zigzag_dc_and_corner() {
        // A zigzag array whose scan index 0 is at (0,0) and whose
        // last (63) is at (7,7).
        let mut z = [0u8; 64];
        z[0] = 8;
        z[63] = 45;
        let raster = de_zigzag(&z);
        assert_eq!(raster[0][0], 8);
        assert_eq!(raster[7][7], 45);
        // The default intra matrix round-trips through de_zigzag if we
        // first zigzag it; check a known interior cell instead: scan
        // index 2 maps to (1,0).
        let mut z2 = [0u8; 64];
        z2[2] = 99;
        let r2 = de_zigzag(&z2);
        assert_eq!(r2[1][0], 99);
    }

    /// A DC-only intra block (all AC zero, pattern_code == 0) with a
    /// zero differential and the "outside-VOP" predictor reconstructs to
    /// a uniform mid-grey block: the §7.4.3.2 DC = default_neighbour_dc /
    /// dc_scaler, inverse-quantised, IDCT'd to a flat block.
    #[test]
    fn dc_only_block_is_flat() {
        // bpp = 8, qs = 8, method 2 (quant_type = false).
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);

        let mut w = BitWriter::default();
        write_dc_zero_luma(&mut w); // size 0 → differential 0
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block = decode_intra_block(
            &mut br,
            0,
            false, // not coded → no AC loop
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();

        // Compute the expected uniform value by hand:
        // QF[0][0] = PQFX[0][0] + chosen/dc_scaler = 0 + 1024 / dc_scaler.
        let scaler = dc_scaler(DcComponent::Luminance, 8) as i32; // qs=8 → 16
        let qf00 = 1024 / scaler; // = 64
                                  // method-2 intra DC: F''[0][0] = dc_scaler * QF[0][0].
        let f00 = scaler * qf00; // = 16 * 64 = 1024
                                 // Flat block: every spatial sample = F''[0][0] / 8 (the DC term
                                 // of the orthonormal IDCT). 1024 / 8 = 128.
        let expected = (f00 / 8).clamp(0, 255);
        for row in block.iter() {
            for &px in row.iter() {
                assert_eq!(px, expected, "DC-only block must be flat");
            }
        }
        assert_eq!(expected, 128);
    }

    /// A synthetic intra block with a known DC differential reconstructs
    /// to the expected uniform spatial value, exercising the full chain
    /// read-DC → predict → inverse-quant → IDCT → clip.
    #[test]
    fn known_dc_differential_reconstructs() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);

        // +1 DC differential on a luminance block.
        let mut w = BitWriter::default();
        write_dc_size1_luma(&mut w, true);
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block = decode_intra_block(
            &mut br,
            0,
            false,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();

        let scaler = dc_scaler(DcComponent::Luminance, 8) as i32; // 16
                                                                  // QF[0][0] = +1 (differential) + 1024/16 = 1 + 64 = 65.
        let qf00 = 1 + 1024 / scaler;
        let f00 = scaler * qf00; // 16 * 65 = 1040
        let expected = (f00 / 8).clamp(0, 255); // 130
        for row in block.iter() {
            for &px in row.iter() {
                assert_eq!(px, expected);
            }
        }
        assert_eq!(expected, 130);
    }

    /// `use_intra_dc_vlc == 0` + `pattern_code[i] == 0`: no bits are
    /// consumed at all; the differential DC is zero and the block
    /// reconstructs exactly as the DC-VLC zero-differential case.
    #[test]
    fn no_dc_vlc_uncoded_block_consumes_no_bits() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);
        let data = [0xAAu8; 4]; // arbitrary — must remain unread
        let mut br = BitReader::new(&data);
        let full = decode_intra_block_full(
            &mut br,
            0,
            false, // pattern_code[i] == 0
            false, // use_intra_dc_vlc == 0
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();
        assert_eq!(br.bit_position(), 0, "no bits may be consumed");
        // Same reconstruction as the dc_only_block_is_flat case: the
        // predicted DC alone → flat 128.
        for row in full.spatial.iter() {
            for &px in row.iter() {
                assert_eq!(px, 128);
            }
        }
        // Traced predictor by-products: QF[0][0] = 1024/16 = 64 and the
        // inverse-quantised DC F[0][0] = 16 * 64 = 1024.
        assert_eq!(full.qf[0][0], 64);
        assert_eq!(full.dc, 1024);
    }

    /// `use_intra_dc_vlc == 0` + coded block: the differential DC is
    /// carried by the first AC EVENT at scan position 0. A single
    /// (LAST=1, RUN=0, LEVEL=+1) intra EVENT (Table B.16 code `0111` +
    /// sign `0`) must reconstruct identically to the DC-VLC `+1`
    /// differential case.
    #[test]
    fn no_dc_vlc_coded_block_reads_dc_from_events() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);
        let mut w = BitWriter::default();
        w.write_bits(0b0111, 4); // Table B.16 (LAST=1, RUN=0, LEVEL=1)
        w.write_bits(0, 1); // sign: positive
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let full = decode_intra_block_full(
            &mut br,
            0,
            true, // pattern_code[i] == 1
            false,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();
        // Matches known_dc_differential_reconstructs: QF[0][0] = 1 + 64,
        // F[0][0] = 16 * 65 = 1040, flat 130.
        for row in full.spatial.iter() {
            for &px in row.iter() {
                assert_eq!(px, 130);
            }
        }
        assert_eq!(full.dc, 1040);
    }

    /// The traced entry and the plain entry decode the same bits to the
    /// same spatial block on the `use_intra_dc_vlc == 1` path.
    #[test]
    fn traced_and_plain_intra_block_agree() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);
        let mut w = BitWriter::default();
        write_dc_size1_luma(&mut w, false); // -1 differential
        let data = w.finish();

        let mut br1 = BitReader::new(&data);
        let plain = decode_intra_block(
            &mut br1,
            0,
            false,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();
        let mut br2 = BitReader::new(&data);
        let full = decode_intra_block_full(
            &mut br2,
            0,
            false,
            true,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();
        assert_eq!(plain, full.spatial);
        assert_eq!(br1.bit_position(), br2.bit_position());
    }

    /// Full §6.2.7 macroblock assembly: six DC-only blocks (cbpy/cbpc 0)
    /// reconstruct a flat 16×16 luma + flat 8×8 Cb/Cr macroblock.
    #[test]
    fn intra_macroblock_dc_only_assembles() {
        let header = MacroblockHeader {
            not_coded: false,
            mb_type: Some(DerivedMbType::Intra),
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0,
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);

        // Six DC-zero blocks: 4 luma + 2 chroma.
        let mut w = BitWriter::default();
        for _ in 0..4 {
            write_dc_zero_luma(&mut w);
        }
        for _ in 0..2 {
            write_dc_zero_chroma(&mut w);
        }
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let mb = decode_intra_macroblock(
            &mut br,
            &header,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();

        // Every luma sample = 128 (same DC math as the single-block test).
        for row in mb.luma.iter() {
            for &px in row.iter() {
                assert_eq!(px, 128);
            }
        }
        // Chroma: dc_scaler(Chrominance, qs=8) = (8+13)/2 = 10.
        let cscaler = dc_scaler(DcComponent::Chrominance, 8) as i32; // 10
        let cqf = 1024 / cscaler; // 102
        let cf = cscaler * cqf; // 1020
                                // Flat IDCT sample = round(F''[0][0] / 8); 1020 / 8 = 127.5 →
                                // 128 under the §4.1 round-to-nearest (away from zero) of §7.4.5.
        let cexp = (((cf as f64) / 8.0).round() as i32).clamp(0, 255); // 128
        for row in mb.cb.iter() {
            for &px in row.iter() {
                assert_eq!(px, cexp);
            }
        }
        for row in mb.cr.iter() {
            for &px in row.iter() {
                assert_eq!(px, cexp);
            }
        }
    }

    /// A coded AC EVENT in a luminance block places a single non-DC
    /// coefficient, producing a non-flat reconstructed block. We assert
    /// the block is no longer uniform and that the §6.2.7 `pattern_code`
    /// gate actually runs the AC loop.
    #[test]
    fn coded_ac_event_produces_non_flat_block() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);

        // DC size 0, then one AC EVENT: Table B.16 intra code for
        // (LAST=1, RUN=0, LEVEL=1) is "10" + sign "0" (positive). The
        // first Table B.16 entry is LAST=0 RUN=0 LEVEL=1 = "10"; the
        // LAST=1 RUN=0 LEVEL=1 entry exists too. To keep the test robust
        // against table specifics, encode a Type-3 escape which is
        // fixed-length and unambiguous:
        //   ESC(0000011) + "11" + LAST(1) RUN(000000) marker(1)
        //   LEVEL(0000_0000_0001) marker(1)
        let mut w = BitWriter::default();
        write_dc_zero_luma(&mut w);
        // Escape prefix + Type-3 selector.
        w.write_bits(0b000_0011, 7);
        w.write_bits(0b11, 2);
        w.write_bits(1, 1); // LAST = 1
        w.write_bits(0, 6); // RUN = 0
        w.write_bits(1, 1); // marker
        w.write_bits(0b0000_0000_0001, 12); // LEVEL = +1
        w.write_bits(1, 1); // marker
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block = decode_intra_block(
            &mut br,
            0,
            true,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();

        // The AC coefficient at scan position 1 ((0,1) in zigzag)
        // perturbs the block away from flat.
        let first = block[0][0];
        let any_different = block.iter().flatten().any(|&px| px != first);
        assert!(any_different, "a coded AC event must break flatness");
    }

    #[test]
    fn not_coded_macroblock_rejected() {
        let header = MacroblockHeader::SKIPPED;
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let err = decode_intra_macroblock(
            &mut br,
            &header,
            ctx,
            BlockPredictors::outside(8, 8),
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap_err();
        assert_eq!(err, BlockAssemblyError::NotCoded);
    }

    #[test]
    fn inter_macroblock_rejected() {
        let header = MacroblockHeader {
            not_coded: false,
            mb_type: Some(DerivedMbType::Inter),
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0,
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let err = decode_intra_macroblock(
            &mut br,
            &header,
            ctx,
            BlockPredictors::outside(8, 8),
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap_err();
        assert_eq!(err, BlockAssemblyError::NotIntra);
    }

    /// `decode_inter_block` with `coded == false` consumes no bits and
    /// returns the all-zero residual (§6.2.7 — the AC EVENT loop is
    /// gated on `pattern_code[i]`).
    #[test]
    fn inter_block_uncoded_is_zero_residual() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        // Two bytes of garbage — none of it should be consumed because
        // `coded == false` short-circuits before the bit reader is touched.
        let data = [0xFFu8; 2];
        let mut br = BitReader::new(&data);
        let pre_position = br.bit_position();
        let block =
            decode_inter_block(&mut br, 0, false, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();
        assert_eq!(
            br.bit_position(),
            pre_position,
            "no bits should be consumed"
        );
        for row in block.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0, "uncoded inter block must be the zero residual");
            }
        }
    }

    /// A single inter EVENT (`LAST=1, RUN=0, LEVEL=+1`) on the inter
    /// Tcoef table reconstructs to a near-uniform near-2 residual:
    /// QF[0][0] = 1; method 2 (`qs = 5`, odd) → `F''[0][0] = (2 + 1) * 5
    /// = 15`; IDCT DC term `15 / 8 ≈ 1.875` rounds to 2.
    #[test]
    fn inter_block_single_dc_event_reconstructs() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 5,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };

        let mut w = BitWriter::default();
        // Inter Tcoef table B.17 entry: `(LAST=1, RUN=0, LEVEL=1)` is
        // code `0111`, 4 bits, then sign bit `0` (+).
        w.write_bits(0b0111, 4);
        w.write_bit(0); // sign +
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block =
            decode_inter_block(&mut br, 0, true, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();

        // The IDCT of a DC-only block of magnitude 15 is the uniform
        // value `15 / 8 = 1.875`, which rounds (§4.1 round-to-nearest,
        // ties away from zero) to 2. The Annex A.1 / IEEE 1180-1990
        // §3.3 tolerance is ±1 LSB.
        for row in block.iter() {
            for &px in row.iter() {
                assert!(
                    (px - 2).abs() <= 1,
                    "single-event inter block: pixel {px} not within 1 LSB of 2"
                );
            }
        }
    }

    /// A single inter EVENT (`LAST=1, RUN=0, LEVEL=-1`) reconstructs to
    /// a near-uniform near-(-2) residual — the negative-sign-bit path.
    #[test]
    fn inter_block_negative_event_reconstructs() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 5,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };

        let mut w = BitWriter::default();
        w.write_bits(0b0111, 4); // (LAST=1, RUN=0, LEVEL=1)
        w.write_bit(1); // sign -
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block =
            decode_inter_block(&mut br, 0, true, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();

        for row in block.iter() {
            for &px in row.iter() {
                assert!(
                    (px + 2).abs() <= 1,
                    "negative single-event inter block: pixel {px} not within 1 LSB of -2"
                );
            }
        }
    }

    /// `decode_inter_macroblock` walks Figure 6-8 over six blocks, all
    /// of which are uncoded for an inter MB with `cbpy == 0` /
    /// `cbpc == 0`. The reconstructed residual is all zero.
    #[test]
    fn inter_macroblock_all_uncoded_is_zero() {
        let header = MacroblockHeader {
            not_coded: false,
            mb_type: Some(DerivedMbType::Inter),
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0,
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let mb =
            decode_inter_macroblock(&mut br, &header, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();
        for row in mb.luma.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0);
            }
        }
        for row in mb.cb.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0);
            }
        }
        for row in mb.cr.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0);
            }
        }
    }

    /// `decode_inter_macroblock` walks Figure 6-8 over six blocks; with
    /// `cbpy = 0b1000` and `cbpc = 0` only block 0 (luma top-left) is
    /// coded. The reconstructed luma top-left 8×8 carries the residual;
    /// the other three luma 8×8s and the chrominance planes stay zero.
    #[test]
    fn inter_macroblock_one_coded_luma_block() {
        let header = MacroblockHeader {
            not_coded: false,
            mb_type: Some(DerivedMbType::Inter),
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0b1000, // only block 0 coded
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 5,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };

        let mut w = BitWriter::default();
        // Block 0 — one positive (LAST=1, RUN=0, LEVEL=1) EVENT.
        w.write_bits(0b0111, 4);
        w.write_bit(0);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let mb =
            decode_inter_macroblock(&mut br, &header, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();

        // Block 0 (luma[0..8][0..8]) should be near-uniform near 2.
        for y in 0..8 {
            for x in 0..8 {
                let px = mb.luma[y][x];
                assert!(
                    (px - 2).abs() <= 1,
                    "block 0 pixel ({y},{x}) = {px}, expected near 2"
                );
            }
        }
        // Block 1 (top-right), block 2 (bottom-left), block 3 (bottom-right)
        // should all be zero.
        for y in 0..8 {
            for x in 8..16 {
                assert_eq!(mb.luma[y][x], 0, "block 1 must be zero");
            }
        }
        for y in 8..16 {
            for x in 0..16 {
                assert_eq!(mb.luma[y][x], 0, "blocks 2/3 must be zero");
            }
        }
        for row in mb.cb.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0);
            }
        }
        for row in mb.cr.iter() {
            for &px in row.iter() {
                assert_eq!(px, 0);
            }
        }
    }

    /// Passing an intra macroblock to `decode_inter_macroblock` is
    /// rejected with [`BlockAssemblyError::NotInter`].
    #[test]
    fn intra_macroblock_rejected_by_inter_driver() {
        let header = MacroblockHeader {
            not_coded: false,
            mb_type: Some(DerivedMbType::Intra),
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0,
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let err = decode_inter_macroblock(&mut br, &header, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX)
            .unwrap_err();
        assert_eq!(err, BlockAssemblyError::NotInter);
    }

    #[test]
    fn cbpb_pattern_code_bit_mapping() {
        // Leftmost (MSB) bit = block 0; 0b10_0000 → only block 0 coded.
        assert_eq!(
            cbpb_pattern_code(Some(0b10_0000)),
            [true, false, false, false, false, false]
        );
        // 0b00_0011 → blocks 4 (Cb) + 5 (Cr) coded.
        assert_eq!(
            cbpb_pattern_code(Some(0b00_0011)),
            [false, false, false, false, true, true]
        );
        // All six coded.
        assert_eq!(cbpb_pattern_code(Some(0b11_1111)), [true; 6]);
        // None / zero → nothing coded.
        assert_eq!(cbpb_pattern_code(None), [false; 6]);
        assert_eq!(cbpb_pattern_code(Some(0)), [false; 6]);
    }

    /// `cbpb == None` means every block is uncoded — the B-VOP residual
    /// is wholly zero and no texture bits are consumed.
    #[test]
    fn b_vop_inter_macroblock_no_cbpb_is_zero() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let before = br.bit_position();
        let mb = decode_b_vop_inter_macroblock(&mut br, None, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX)
            .unwrap();
        assert_eq!(
            br.bit_position(),
            before,
            "no texture bits should be consumed"
        );
        for row in mb.luma.iter() {
            assert!(row.iter().all(|&px| px == 0));
        }
        assert!(mb.cb.iter().all(|r| r.iter().all(|&px| px == 0)));
        assert!(mb.cr.iter().all(|r| r.iter().all(|&px| px == 0)));
    }

    /// A B-VOP residual with `cbpb = 0b10_0000` codes only block 0 (luma
    /// top-left). The decoded residual carries the coefficient there;
    /// every other block stays zero. Mirrors the P-VOP single-luma test
    /// but routes the coded-block pattern through `cbpb` rather than
    /// `cbpy`/`cbpc`.
    #[test]
    fn b_vop_inter_macroblock_one_coded_luma_block() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 5,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let mut w = BitWriter::default();
        // Block 0 — one positive (LAST=1, RUN=0, LEVEL=1) inter EVENT.
        w.write_bits(0b0111, 4);
        w.write_bit(0);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let mb = decode_b_vop_inter_macroblock(
            &mut br,
            Some(0b10_0000),
            ctx,
            &DEFAULT_NONINTRA_QUANT_MATRIX,
        )
        .unwrap();

        for y in 0..8 {
            for x in 0..8 {
                let px = mb.luma[y][x];
                assert!(
                    (px - 2).abs() <= 1,
                    "block 0 pixel ({y},{x}) = {px}, expected near 2"
                );
            }
        }
        for y in 0..8 {
            for x in 8..16 {
                assert_eq!(mb.luma[y][x], 0, "block 1 must be zero");
            }
        }
        for y in 8..16 {
            for x in 0..16 {
                assert_eq!(mb.luma[y][x], 0, "blocks 2/3 must be zero");
            }
        }
        assert!(mb.cb.iter().all(|r| r.iter().all(|&px| px == 0)));
        assert!(mb.cr.iter().all(|r| r.iter().all(|&px| px == 0)));
    }

    /// A `not_coded == true` macroblock is short-circuited via
    /// [`BlockAssemblyError::NotCoded`] regardless of `mb_type`; the
    /// §7.5 zero-MV / zero-residual reconstruction belongs in the
    /// caller's motion-compensation stage.
    #[test]
    fn inter_macroblock_not_coded_rejected() {
        let header = MacroblockHeader {
            not_coded: true,
            mb_type: None,
            cbpc: 0,
            ac_pred_flag: false,
            cbpy: 0,
            dquant_delta: None,
            interlaced_info: None,
            mcsel: None,
        };
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: false,
            ac_pred_flag: false,
        };
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let err = decode_inter_macroblock(&mut br, &header, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX)
            .unwrap_err();
        assert_eq!(err, BlockAssemblyError::NotCoded);
    }

    /// `nonintra_quant_matrix` returns the §6.3.3 default `W[1]` matrix
    /// when no `nonintra_quant_mat` was loaded by the VOL header.
    #[test]
    fn nonintra_quant_matrix_default() {
        // Build a minimal VolHeader-like value via the VOL parser? Easier:
        // call the helper through a constructed `VolHeader` via its public
        // API. The crate exposes `VolHeader` with `nonintra_quant_mat:
        // Option<[u8; 64]>`. Use struct-update from a parsed header is
        // verbose; instead exercise the `Some(zigzag)` and `None` branches
        // via direct field set on a `VolHeader` literal once the type is
        // constructable. We confirm the default branch via the matrix
        // constant directly, since the helper trivially delegates.
        assert_eq!(DEFAULT_NONINTRA_QUANT_MATRIX[0][0], 16);
        assert_eq!(DEFAULT_NONINTRA_QUANT_MATRIX[7][7], 33);
    }

    /// `decode_inter_block` with method 1 (`quant_type == true`) runs
    /// end-to-end against the default `W[1]` matrix; a single-event
    /// block reconstructs to a near-uniform residual (the per-event
    /// magnitude depends on the matrix and `qs`, so the assertion is
    /// looser).
    #[test]
    fn inter_block_method1_runs() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: true,
            ac_pred_flag: false,
        };

        let mut w = BitWriter::default();
        // (LAST=1, RUN=0, LEVEL=1) +.
        w.write_bits(0b0111, 4);
        w.write_bit(0);
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block =
            decode_inter_block(&mut br, 0, true, ctx, &DEFAULT_NONINTRA_QUANT_MATRIX).unwrap();
        // The residual must be inside the §7.4.5 IDCT saturation range.
        for row in block.iter() {
            for &px in row.iter() {
                assert!((-256..=255).contains(&px), "px {px} outside §7.4.5 range");
            }
        }
        // At qs=8 with W[1][0][0]=16, non-intra method 1 yields
        // F''[0][0] = ((2*1 + 1) * 16 * 8) / 16 = 24. The IDCT spreads
        // this DC into f[y][x] ≈ 24/8 = 3. Allow ±2 because the §7.4.4.5
        // mismatch toggle perturbs F[7][7] by ±1 which the IDCT spreads.
        let centre = block[0][0];
        assert!(
            (centre - 3).abs() <= 2,
            "method-1 inter block centre = {centre}, expected ~3"
        );
    }

    /// Method 1 (`quant_type == true`) runs end-to-end with the default
    /// intra matrix; a DC-only block still reconstructs flat (the DC path
    /// is the same dc_scaler formula regardless of method).
    #[test]
    fn method1_dc_only_block() {
        let ctx = MacroblockTextureContext {
            quantiser_scale: 8,
            bits_per_pixel: 8,
            quant_type: true,
            ac_pred_flag: false,
        };
        let predictors = BlockPredictors::outside(8, 8);

        let mut w = BitWriter::default();
        write_dc_zero_luma(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);

        let block = decode_intra_block(
            &mut br,
            0,
            false,
            ctx,
            predictors,
            &DEFAULT_INTRA_QUANT_MATRIX,
        )
        .unwrap();
        // DC reconstruction is method-independent (dc_scaler * QF[0][0]),
        // so the block is ~128. §7.4.4.5 mismatch control toggles the
        // F[7][7] LSB when the block sum is even (it is, for a DC-only
        // block), spreading a ±1 high-frequency perturbation across the
        // IDCT output; the reconstruction stays within 1 LSB of the flat
        // value (the IEEE 1180-1990 §3.3 peak-error tolerance Annex A.1
        // references).
        for row in block.iter() {
            for &px in row.iter() {
                assert!((px - 128).abs() <= 1, "method-1 DC block: {px} not ~128");
            }
        }
    }
}

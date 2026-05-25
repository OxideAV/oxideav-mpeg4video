//! §6.2.7 `block(i)` macroblock-level texture assembly for intra I-VOP
//! macroblocks.
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

/// The §6.1.3 / Figure 6-8 component of block `i` in a 4:2:0 macroblock.
#[inline]
fn block_component(i: usize) -> DcComponent {
    DcComponent::from_block_index(i)
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
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_block(
    br: &mut BitReader<'_>,
    i: usize,
    coded: bool,
    ctx: MacroblockTextureContext,
    predictors: BlockPredictors,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<[[i32; 8]; 8], BlockAssemblyError> {
    let component = block_component(i);

    // §6.2.7 — differential intra DC (always present for an intra block
    // when use_intra_dc_vlc == 1, the path we decode).
    let dc = decode_intra_dc(br, component)?;

    // §6.2.7 — the AC EVENT loop runs only when pattern_code[i] == 1.
    let events = if coded {
        decode_ac_events(br, TcoefTable::Intra)?
    } else {
        Vec::new()
    };

    // §7.4.1 / §7.4.2 — assemble the one-dimensional QFS[64] (differential
    // DC at scan position 0) and choose the §7.4.2 scan pattern.
    let qfs = events_to_qfs(&events, Some(dc.differential))?;

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
    Ok(out)
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

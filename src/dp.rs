//! Data partitioning emit + decode for MPEG-4 Part 2 (ISO/IEC 14496-2
//! §6.2.6 / §6.3.7 `data_partitioned_motion_shape_texture()`).
//!
//! Data partitioning rearranges the per-MB syntax inside a video packet
//! so that motion-/header-style fields (which dominate decode integrity)
//! sit in part 1, and DCT-coefficient fields (which only affect the
//! reconstructed pels) sit in part 2. A spec-defined "marker" sequence
//! separates the two halves so a decoder that loses synchronisation in
//! one half can still recover the other:
//!
//! * **DC marker** — 19 bits `110 1011 0000 0000 0001` (= `0x6B001`),
//!   used in I-VOPs.
//! * **Motion marker** — 17 bits `1 1111 0000 0000 0001` (= `0x1F001`),
//!   used in P-VOPs and S(GMC)-VOPs.
//!
//! Both markers are constructed so that they cannot appear in any
//! valid concatenation of part-1 codewords (DC VLC + sign + marker for
//! I-VOPs; not_coded + mcbpc + mcsel + MV for P-VOPs); the spec refers
//! to this as the marker "uniqueness" property.
//!
//! Per-VOP layout (§6.2.5.3 + §6.2.6):
//!
//! ```text
//! data_partitioned_i_vop():
//!   for each MB:
//!     mcbpc                       (Table B-10)
//!     [dquant if Intra+Q]
//!     for each block (Y0..Y3,Cb,Cr):
//!       dct_dc_size_lumi/chrom    (Table B-12 / B-13)
//!       dc_differential
//!       marker_bit if size > 8
//!   dc_marker (19 bits)
//!   for each MB:
//!     ac_pred_flag (1)
//!     cbpy         (Table B-9)
//!   for each MB:
//!     for each coded block:  AC walk (Table B-16)
//! ```
//!
//! ```text
//! data_partitioned_p_vop():
//!   for each MB:
//!     not_coded
//!     if !not_coded:
//!       mcbpc                     (Table B-13)
//!       [mcsel if S(GMC) + Inter/InterQ]
//!       motion_coding(forward)    (MV — same Table B-12 magnitude
//!                                  + sign + residual layout used by the
//!                                  combined-mode encoder)
//!   motion_marker (17 bits)
//!   for each MB:
//!     if !not_coded:
//!       [ac_pred_flag if Intra]
//!       cbpy                      (Table B-9)
//!       [dquant if InterQ/IntraQ]
//!       [intra DC VLCs if Intra && use_intra_dc_vlc]
//!   for each MB:
//!     for each coded block:  AC walk (Table B-16 intra / B-17 inter)
//! ```
//!
//! Encoder scope here: a whole picture is emitted as ONE video packet
//! using the DP layout. No mid-picture `video_packet_header()` splits —
//! the spec allows this (a "video packet" can contain every MB) and it
//! keeps the encoder + tests focused on the DP layout itself rather
//! than on multi-packet resync. The VOL emits
//! `resync_marker_disable = 0` so a future encoder pass can introduce
//! mid-picture splits without re-flipping the VOL bit.
//!
//! Decoder scope: matching reverse — parse part 1 collecting per-MB
//! state, expect the marker, parse part 2's `(ac_pred_flag, cbpy)` per
//! MB, then parse the per-MB AC walks. Reconstructs pictures bit-exact
//! to the combined-mode decoder when fed our own DP-encoded bitstreams.

use oxideav_core::{Error, Result, VideoFrame};

use oxideav_core::bits::{BitReader, BitWriter};

use crate::block::{choose_dc_predictor, decode_intra_ac, decode_intra_dc_diff};
use crate::encoder::{
    fdct8x8, load_block_samples, lookup_neighbour_dcs, quantise_ac_intra_h263, round_div,
    update_neighbour, write_cbpy, write_intra_ac, write_intra_dc_diff, write_mcbpc_intra,
    write_recon_to_picture,
};
use crate::headers::vol::{VideoObjectLayer, ZIGZAG};
use crate::headers::vop::VideoObjectPlane;
use crate::iq::dc_scaler;
use crate::mb::{IVopPicture, PredGrid};
use crate::tables::{cbpy as cbpy_tab, mcbpc, vlc};

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Drain `n_bits` from `br` into a fresh byte-aligned buffer (MSB-first
/// bit layout, trailing bits zero-padded). Used by the RVLC strategy
/// picker to anchor a bit-misaligned AC partition into a buffer that
/// `bit_reverse_buffer` can consume. After this call the BitReader is
/// positioned `n_bits` further into its underlying stream.
fn drain_to_aligned_buffer(br: &mut BitReader<'_>, n_bits: u64) -> Vec<u8> {
    let n_out_bytes = (n_bits as usize).div_ceil(8);
    let mut out = vec![0u8; n_out_bytes];
    let mut remaining = n_bits;
    let mut byte_idx = 0usize;
    while remaining >= 8 {
        // Best-effort — if the reader runs short the result stays zero.
        let b = br.read_u32(8).unwrap_or(0) as u8;
        out[byte_idx] = b;
        byte_idx += 1;
        remaining -= 8;
    }
    if remaining > 0 {
        let tail = br.read_u32(remaining as u32).unwrap_or(0);
        // Tail is right-aligned in the low `remaining` bits; shift it
        // up so the partition's last bits sit at the high end of the
        // last output byte.
        let shift = 8 - remaining as u32;
        out[byte_idx] = ((tail << shift) & 0xFF) as u8;
    }
    out
}

// -------------------------------------------------------------------------
// Marker bit patterns (§6.3.7)
// -------------------------------------------------------------------------

/// **DC marker** — 19-bit tag `110 1011 0000 0000 0001` separating the
/// header/DC partition from the AC partition in `data_partitioned_i_vop()`.
pub const DC_MARKER: u32 = 0b110_1011_0000_0000_0001;
/// Bit width of [`DC_MARKER`] when emitted MSB-first.
pub const DC_MARKER_BITS: u32 = 19;

/// **Motion marker** — 17-bit tag `1 1111 0000 0000 0001` separating
/// the motion partition from the texture partition in
/// `data_partitioned_p_vop()`.
pub const MOTION_MARKER: u32 = 0b1_1111_0000_0000_0001;
/// Bit width of [`MOTION_MARKER`] when emitted MSB-first.
pub const MOTION_MARKER_BITS: u32 = 17;

// -------------------------------------------------------------------------
// I-VOP DP encode
// -------------------------------------------------------------------------

/// Per-MB intermediate state captured during the encode pre-pass for
/// data_partitioned_i_vop(). Carries everything needed to (a) emit Part
/// 1's DC bits, (b) emit Part 2's `(ac_pred_flag, cbpy)`, (c) emit Part
/// 2's AC walks, and (d) reconstruct samples into the picture.
struct IMbDp {
    cbpc: u8,
    cbpy: u8,
    /// `(dc_diff, recon_dc_pel)` per block. `recon_dc_pel` is the post-
    /// prediction reconstructed DC in pel domain — used when stamping
    /// the IDCT output into the picture.
    dc: [(i32, i32); 6],
    luma_coded: [bool; 4],
    chroma_coded: [bool; 2],
    /// AC levels per block in natural order (`[ZIGZAG[i]]` is scan i).
    ac_levels: [[i32; 64]; 6],
}

/// Encode one I-VOP body using `data_partitioned_i_vop()` layout
/// (§6.2.5.3) and return the reconstructed picture so it can serve as
/// the next P-VOP reference. Mirrors `encoder::encode_i_vop_body_and_reconstruct`
/// modulo bit ordering.
///
/// `reversible_vlc` selects the AC-walk writer: when `true`, the per-MB
/// AC partition is emitted via [`crate::rvlc::write_intra_ac`] (Table
/// B.23) instead of [`encoder::write_intra_ac`] (Table B.16). The VOL
/// header must already advertise `reversible_vlc = 1`.
pub fn encode_i_vop_body_dp_and_reconstruct(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: u32,
    height: u32,
    vop_quant: u32,
    reversible_vlc: bool,
) -> Result<IVopPicture> {
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let mb_total = mb_w * mb_h;

    let mut grid = PredGrid::new(mb_w, mb_h);
    let mut pic = IVopPicture::new(width, height);
    let mut mbs: Vec<IMbDp> = Vec::with_capacity(mb_total);

    // ---- Pre-pass: DCT + quant + DC predict + reconstruct, per MB.
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb = compute_intra_mb_dp(v, width, height, mb_x, mb_y, vop_quant, &mut grid);
            // Stamp reconstruction into picture.
            for blk in 0..6 {
                stamp_intra_block_recon(&mb, blk, mb_x, mb_y, vop_quant, &mut pic);
            }
            mbs.push(mb);
        }
    }

    // ---- Part 1 emit: per-MB { mcbpc, DC VLCs }.
    for mb in &mbs {
        // No dquant emit (encoder uses constant quant — mb_type stays = 3,
        // the "Intra" MCBPC group, not Intra+Q).
        write_mcbpc_intra(bw, mb.cbpc);
        for (blk, (diff, _recon)) in mb.dc.iter().enumerate() {
            write_intra_dc_diff(bw, blk, *diff);
        }
    }

    // ---- Marker.
    bw.write_bits(DC_MARKER, DC_MARKER_BITS);

    // ---- Part 2 emit: per-MB { ac_pred_flag, cbpy }.
    //
    // CBPY here uses the **raw** Table B-9 VLC value as the coded mask
    // (matches our combined-mode I-VOP path in `encoder::write_cbpy`).
    // This stays bit-exact through our own decoder (`dp::decode_ivop_dp`).
    // ffmpeg-interop note: ffmpeg appears to invert cbpy in DP I-VOP
    // mode (XOR with 0xF) — our self-roundtrip uses the spec-literal
    // raw mask, which is what `data_partitioned_i_vop()` Table 6-21
    // describes. The interop gap is tracked as a round-22 follow-up.
    for mb in &mbs {
        bw.write_bits(0, 1); // ac_pred_flag = 0 (encoder never predicts ACs)
        write_cbpy(bw, mb.cbpy);
    }

    // ---- Part 3 emit: per-MB { AC walks for each coded block }.
    for mb in &mbs {
        for blk in 0..6 {
            let coded = if blk < 4 {
                mb.luma_coded[blk]
            } else {
                mb.chroma_coded[blk - 4]
            };
            if !coded {
                continue;
            }
            if reversible_vlc {
                crate::rvlc::write_intra_ac(bw, &mb.ac_levels[blk])?;
            } else {
                write_intra_ac(bw, &mb.ac_levels[blk])?;
            }
        }
    }

    Ok(pic)
}

/// Pre-compute one intra MB worth of DCT/quant/DC-prediction state for
/// later DP emission. Mirrors the pre-emission half of
/// `encoder::encode_intra_mb_inner` (without writing any bits).
fn compute_intra_mb_dp(
    v: &VideoFrame,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
) -> IMbDp {
    // Read source samples for each of the six 8×8 blocks.
    let mut blocks = [[0i32; 64]; 6];
    for blk in 0..6 {
        load_block_samples(v, width, height, mb_x, mb_y, blk, &mut blocks[blk]);
    }
    // Forward DCT.
    let mut dct = [[0i32; 64]; 6];
    for blk in 0..6 {
        let mut f = [0.0f32; 64];
        for i in 0..64 {
            f[i] = blocks[blk][i] as f32;
        }
        fdct8x8(&mut f);
        for i in 0..64 {
            dct[blk][i] = f[i].round() as i32;
        }
    }
    // Quantise DC (round-to-nearest) + ACs (H.263 intra).
    let mut dc_units = [0i32; 6];
    let mut ac_levels = [[0i32; 64]; 6];
    for blk in 0..6 {
        let scale = dc_scaler(blk, quant) as i32;
        let dc_q = round_div(dct[blk][0], scale).clamp(-2048, 2047);
        dc_units[blk] = dc_q;
        for i in 1..64 {
            ac_levels[blk][i] =
                quantise_ac_intra_h263(dct[blk][i], quant as i32).clamp(-2047, 2047);
        }
    }
    // Compute coded flags.
    let mut luma_coded = [false; 4];
    let mut chroma_coded = [false; 2];
    for blk in 0..4 {
        luma_coded[blk] = ac_levels[blk][1..64].iter().any(|&v| v != 0);
    }
    chroma_coded[0] = ac_levels[4][1..64].iter().any(|&v| v != 0);
    chroma_coded[1] = ac_levels[5][1..64].iter().any(|&v| v != 0);
    let cbpc = ((chroma_coded[0] as u8) << 1) | (chroma_coded[1] as u8);
    let mut cbpy: u8 = 0;
    for (i, &c) in luma_coded.iter().enumerate() {
        if c {
            cbpy |= 1 << (3 - i);
        }
    }
    // DC differential + reconstruction per block, updating the predictor
    // grid as we go (so the next MB's DC predictor sees the same
    // neighbour DCs the decoder will).
    let mut dc = [(0i32, 0i32); 6];
    for blk in 0..6 {
        let (left, top_left, top) = lookup_neighbour_dcs(blk, mb_x, mb_y, grid);
        let (predicted_dc_pel, _dir) = choose_dc_predictor(left, top_left, top);
        let scale = dc_scaler(blk, quant) as i32;
        let pred_units = (predicted_dc_pel + scale / 2) / scale;
        let dc_diff = dc_units[blk] - pred_units;
        let recon_dc = (dc_units[blk] * scale).clamp(0, 2047);
        dc[blk] = (dc_diff, recon_dc);
        update_neighbour(grid, blk, mb_x, mb_y, recon_dc, quant as u8);
    }
    IMbDp {
        cbpc,
        cbpy,
        dc,
        luma_coded,
        chroma_coded,
        ac_levels,
    }
}

/// Reconstruct one block of an `IMbDp` into the picture (same recipe
/// the decoder will run from our bitstream). `quant` is the constant
/// quantiser the picture was emitted with (no dquant in DP encoder).
fn stamp_intra_block_recon(
    mb: &IMbDp,
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    pic: &mut IVopPicture,
) {
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let mut coeffs = mb.ac_levels[blk];
    for i in 1..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        let abs = l.abs();
        let mut val = 2 * q * abs + q_plus;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
    }
    coeffs[0] = mb.dc[blk].1.clamp(-2048, 2047);
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = coeffs[i] as f32;
    }
    crate::block::idct8x8(&mut f);
    write_recon_to_picture(pic, blk, mb_x, mb_y, &f);
}

// -------------------------------------------------------------------------
// I-VOP DP decode
// -------------------------------------------------------------------------

/// Decode one I-VOP body emitted under `data_partitioned_i_vop()`
/// (§6.2.5.3). Mirrors `decoder::decode_ivop_pic` modulo bit ordering.
/// Treats the entire picture as one video packet (matches the
/// encoder's emit policy); mid-picture `video_packet_header()` resync
/// is not yet supported in DP mode.
pub fn decode_ivop_dp(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
) -> Result<IVopPicture> {
    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mb_total = mb_w * mb_h;

    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut grid = PredGrid::new(mb_w, mb_h);
    let quant = vop.vop_quant; // constant — no dquant in our DP encoder
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };

    // Per-MB state carried between part 1 and part 2.
    struct IMbState {
        cbpc: u8,
        dc_diff: [i32; 6],
        recon_dc_units: [i32; 6],
        // populated in part 2:
        cbpy: u8,
        ac_pred: bool,
        // populated in part 3:
        ac_levels: [[i32; 64]; 6],
    }
    let mut mbs: Vec<IMbState> = Vec::with_capacity(mb_total);

    // ---- Part 1: per-MB { mcbpc, DC VLCs }.
    for mb_idx in 0..mb_total {
        let mb_x = mb_idx % mb_w;
        let mb_y = mb_idx / mb_w;
        let mcbpc_v = loop {
            let v = vlc::decode(br, mcbpc::i_table())?;
            if v != mcbpc::STUFFING {
                break v;
            }
        };
        let cbpc = if mcbpc_v < 4 {
            mcbpc_v as u8
        } else if mcbpc_v < 8 {
            // Intra+Q with dquant — read+ignore to keep parsing alive.
            let _d = br.read_u32(2)?;
            (mcbpc_v - 4) as u8
        } else {
            return Err(Error::invalid("mpeg4 DP I: invalid mcbpc"));
        };
        let mut dc_diff = [0i32; 6];
        let mut recon_dc_units = [0i32; 6];
        for blk in 0..6 {
            let (left, top_left, top) = lookup_neighbour_dcs(blk, mb_x, mb_y, &grid);
            let (predicted_dc_pel, _dir) = choose_dc_predictor(left, top_left, top);
            let scale = dc_scaler(blk, quant) as i32;
            let pred_units = (predicted_dc_pel + scale / 2) / scale;
            let diff = decode_intra_dc_diff(br, blk)?;
            let units = pred_units + diff;
            let recon_pel = (units * scale).clamp(0, 2047);
            update_neighbour(&mut grid, blk, mb_x, mb_y, recon_pel, quant as u8);
            dc_diff[blk] = diff;
            recon_dc_units[blk] = units;
        }
        mbs.push(IMbState {
            cbpc,
            dc_diff,
            recon_dc_units,
            cbpy: 0,
            ac_pred: false,
            ac_levels: [[0i32; 64]; 6],
        });
    }

    // ---- DC marker.
    let m = br.read_u32(DC_MARKER_BITS)?;
    if m != DC_MARKER {
        return Err(Error::invalid("mpeg4 DP I: dc_marker mismatch"));
    }

    // ---- Part 2: per-MB { ac_pred_flag, cbpy }.
    for mb in mbs.iter_mut() {
        mb.ac_pred = br.read_u1()? == 1;
        mb.cbpy = vlc::decode(br, cbpy_tab::table())? as u8;
    }

    // ---- Part 3: per-MB { AC walks for each coded block }.
    //
    // When `reversible_vlc = 1` we route the partition through the
    // §E.1.4.4.2.1 strategy 1-4 picker: forward + reverse walkers cover
    // both halves of the partition and the picker selects between
    // overlap (Strategy 2/4) and gap-conceal (Strategy 1/3). When
    // `reversible_vlc = 0` we keep the simple per-block forward walker.
    if vol.reversible_vlc {
        // Build a flat block descriptor list mirroring the per-MB coded
        // mask. Each entry is `(mb_idx, blk_idx)` so we can scatter the
        // picker's output back into the per-MB state vector.
        let mut descs: Vec<crate::rvlc::RvlcBlockDesc> = Vec::new();
        let mut slots: Vec<(usize, usize)> = Vec::new();
        for (mb_idx, mb) in mbs.iter().enumerate() {
            let cbpc = mb.cbpc;
            let luma_coded = [
                (mb.cbpy >> 3) & 1 != 0,
                (mb.cbpy >> 2) & 1 != 0,
                (mb.cbpy >> 1) & 1 != 0,
                mb.cbpy & 1 != 0,
            ];
            let chroma_coded = [(cbpc >> 1) & 1 != 0, cbpc & 1 != 0];
            for blk in 0..6 {
                let coded = if blk < 4 {
                    luma_coded[blk]
                } else {
                    chroma_coded[blk - 4]
                };
                if !coded {
                    continue;
                }
                descs.push(crate::rvlc::RvlcBlockDesc { is_intra: true });
                slots.push((mb_idx, blk));
            }
        }
        let mut out_blocks: Vec<[i32; 64]> = vec![[0i32; 64]; descs.len()];
        // Drain the AC partition into a fresh byte-aligned buffer by
        // reading the remaining bits out of the BitReader. The DP I-VOP
        // encoder emits the entire picture as one video packet, so
        // everything past Part 2 belongs to the AC partition.
        let total_bits = br.bits_remaining();
        let aligned = drain_to_aligned_buffer(br, total_bits);
        let (_stats, _outcomes) = crate::rvlc::decode_rvlc_ac_partition(
            &aligned,
            0,
            total_bits,
            &descs,
            &ZIGZAG,
            &mut out_blocks,
        )?;
        // Scatter results back into per-MB AC slots.
        for ((mb_idx, blk), coeffs) in slots.into_iter().zip(out_blocks) {
            mbs[mb_idx].ac_levels[blk] = coeffs;
        }
    } else {
        for mb in mbs.iter_mut() {
            let cbpc = mb.cbpc;
            let luma_coded = [
                (mb.cbpy >> 3) & 1 != 0,
                (mb.cbpy >> 2) & 1 != 0,
                (mb.cbpy >> 1) & 1 != 0,
                mb.cbpy & 1 != 0,
            ];
            let chroma_coded = [(cbpc >> 1) & 1 != 0, cbpc & 1 != 0];
            for blk in 0..6 {
                let coded = if blk < 4 {
                    luma_coded[blk]
                } else {
                    chroma_coded[blk - 4]
                };
                if !coded {
                    continue;
                }
                // We disabled AC prediction at encode time — use the
                // default zigzag scan.
                decode_intra_ac(br, &mut mb.ac_levels[blk], &ZIGZAG)?;
            }
        }
    }

    // ---- Reconstruct each MB.
    for (mb_idx, mb) in mbs.iter().enumerate() {
        let mb_x = mb_idx % mb_w;
        let mb_y = mb_idx / mb_w;
        for blk in 0..6 {
            let mut coeffs = mb.ac_levels[blk];
            for i in 1..64 {
                let l = coeffs[i];
                if l == 0 {
                    continue;
                }
                let abs = l.abs();
                let mut val = 2 * q * abs + q_plus;
                if l < 0 {
                    val = -val;
                }
                coeffs[i] = val.clamp(-2048, 2047);
            }
            let scale = dc_scaler(blk, quant) as i32;
            let recon_dc = (mb.recon_dc_units[blk] * scale).clamp(0, 2047);
            coeffs[0] = recon_dc;
            let mut f = [0.0f32; 64];
            for i in 0..64 {
                f[i] = coeffs[i] as f32;
            }
            crate::block::idct8x8(&mut f);
            write_recon_to_picture(&mut pic, blk, mb_x, mb_y, &f);
        }
    }

    // Suppress unused warnings (dc_diff and ac_pred carry information that
    // a future AC-prediction-aware decoder will consume).
    for m in &mbs {
        let _ = (&m.dc_diff, m.ac_pred);
    }

    Ok(pic)
}

// -------------------------------------------------------------------------
// P-VOP DP encode
// -------------------------------------------------------------------------

use crate::encoder::write_mcbpc_p_intra;
use crate::inter::{MbMotion, MvGrid};
use crate::pvop::{
    estimate_and_encode_mb, intra_cost_proxy, reset_pred_grid_mb, wrap_mvd, write_inter_ac,
    write_mcbpc_inter, write_mcbpc_inter4mv, write_mv_component, write_recon_to_pic, PMbEncoding,
    INTRA_IN_P_BIAS, INTRA_MARGIN,
};

/// Per-MB DP P-VOP encoding state — either an inter MB carrying motion
/// information + inter residual, or an intra-in-P MB carrying the same
/// state an I-VOP intra MB would (DC differentials, AC levels). The
/// distinction drives part-1 emission (intra MBs skip the MV) and
/// part-2 emission (intra MBs emit `ac_pred_flag` + 6 intra DC VLCs
/// after their cbpy, per `data_partitioned_p_vop()` §6.2.5.3 lines
/// `derived_mb_type >= 3` branches).
enum PMbDp {
    /// Inter MB (Inter / Inter+Q / Inter4MV in mb_type taxonomy). Emits
    /// either MCBPC group 0..=3 (Inter, one MV) or group 16..=19
    /// (Inter4MV, four MVs), then inter cbpy + inter AC walks. The DP
    /// encoder picks 1MV vs 4MV per the same SAD+lambda criterion the
    /// combined-mode P-VOP encoder uses. GMC is rejected at the factory.
    /// Boxed to keep the enum's stack footprint small (PMbEncoding
    /// embeds the 16×16 luma + 2×8×8 chroma reconstruction buffers +
    /// 6×64 AC arrays, ~2 KiB).
    Inter(Box<PMbEncoding>),
    /// Intra-in-P MB (Table B-13 mb_type=3, MCBPC rows 4..=7). Emits
    /// MCBPC group 4..=7, NO MV, then ac_pred_flag + raw cbpy + 6 intra
    /// DC differentials in part 2, intra AC walks in part 3. Boxed for
    /// the same reason as `Inter` (6×64 AC arrays).
    Intra(Box<IMbDp>),
}

/// Encode one P-VOP body using `data_partitioned_p_vop()` layout
/// (§6.2.5.3 / §6.3.7) and return the reconstructed picture + MV grid.
///
/// Scope: 1MV-Inter + Inter4MV + Intra-in-P. GMC is rejected at the
/// encoder factory and doesn't reach here. Inter4MV is picked
/// automatically when the chosen-mode SAD wins by more than the
/// `FOURMV_LAMBDA` bit-cost penalty (§7.5.7), exactly like the
/// combined-mode encoder; the per-MB MCBPC switches to Table B-13
/// rows 16..=19 (`Inter4MV`) and four MVDs are emitted in part 1
/// instead of one. Skipped MBs (`not_coded = 1`) are emitted exactly
/// like the combined-mode encoder. The picture is treated as one
/// video packet — no mid-VOP `video_packet_header()` splits.
///
/// Intra-in-P decision matches the combined-mode P-VOP encoder
/// (`pvop::encode_p_vop_body_with_grid`): the chosen-mode inter SAD
/// is compared against an intra MAD proxy + `INTRA_IN_P_BIAS +
/// INTRA_MARGIN`, and the MB is re-encoded as intra when the proxy
/// shows a clear scene-change / occlusion. Intra MBs route their DC
/// differentials into the part-2 partition (after `motion_marker`)
/// and their AC coefficients into the part-3 partition, since intra
/// MBs carry no motion data; this is the spec-mandated routing for
/// `derived_mb_type >= 3` in `data_partitioned_p_vop()`.
///
/// `reversible_vlc` selects the per-MB AC writer: when `true`, AC
/// walks emit through [`crate::rvlc::write_intra_ac`] /
/// [`crate::rvlc::write_inter_ac`] (Table B.23) instead of the plain
/// Tables B.16 / B.17. The VOL header must already advertise
/// `reversible_vlc = 1`.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_vop_body_dp_with_grid(
    bw: &mut BitWriter,
    v: &VideoFrame,
    width: u32,
    height: u32,
    reference: &IVopPicture,
    vop_quant: u32,
    f_code_fwd: u8,
    rounding_type: bool,
    reversible_vlc: bool,
) -> Result<(IVopPicture, MvGrid)> {
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut pic = IVopPicture::new(width, height);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);
    // PredGrid tracks intra DC predictor state (§7.4.3.1). Intra-in-P
    // MBs update it; inter / skipped MBs reset their slot to defaults
    // so a later intra MB reads `dc=1024, is_intra=false` from inter
    // neighbours (matches `inter::reset_pred_grid_mb` in the decoder).
    let mut pred_grid = PredGrid::new(mb_w, mb_h);

    // Per-MB encoding state (inter or intra). We materialise the full
    // decision in this pre-pass so part-1 / part-2 / part-3 emit can
    // walk the MBs in scan order three times without re-running ME.
    let mut mbs: Vec<PMbDp> = Vec::with_capacity(mb_w * mb_h);

    // ---- Pre-pass: estimate, residual-encode, reconstruct each MB.
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // 1MV / 4MV mode-decision path. `estimate_and_encode_mb`
            // runs the same SAD+lambda heuristic the combined-mode
            // P-VOP encoder uses; the chosen mode lands in `four_mv`
            // and the per-block MVs in `mv4_half`. GMC is rejected
            // upstream (warp = None), QPel is rejected upstream
            // (quarter_sample = false), so the predictor is purely
            // half-pel translational across both 1MV and 4MV.
            let inter_mb = estimate_and_encode_mb(
                v,
                width,
                height,
                reference,
                mb_x,
                mb_y,
                vop_quant,
                rounding_type,
                &mv_grid,
                false,
                None,
            )?;

            // Intra-in-P decision (§6.3.7): mirrors
            // `pvop::encode_p_vop_body_with_grid`. Skip MBs that the
            // inter decision marked as `not_coded` — they stay inter
            // (skipped MBs carry no residual to compare).
            let intra_cost = intra_cost_proxy(v, width, height, mb_x, mb_y);
            let inter_cost = inter_mb.inter_luma_sad;
            let prefer_intra = !inter_mb.skipped
                && inter_cost
                    > intra_cost
                        .saturating_add(INTRA_IN_P_BIAS)
                        .saturating_add(INTRA_MARGIN);

            if prefer_intra {
                // Re-encode as intra. `compute_intra_mb_dp` runs the
                // same DCT + quant + DC-prediction recipe the I-VOP
                // DP path uses, and updates the predictor grid as the
                // decoder will see it. Reconstruction goes into `pic`
                // (so future MBs / VOPs can reference it).
                let intra_mb =
                    compute_intra_mb_dp(v, width, height, mb_x, mb_y, vop_quant, &mut pred_grid);
                for blk in 0..6 {
                    stamp_intra_block_recon(&intra_mb, blk, mb_x, mb_y, vop_quant, &mut pic);
                }
                // Intra MBs contribute (0,0) to the MV grid for the
                // next MB's median predictor (§7.6.7 step 3) and are
                // NOT considered `not_coded`. Mirrors the combined-
                // mode encoder's intra-in-P MV-grid update.
                mv_grid.set(
                    mb_x,
                    mb_y,
                    MbMotion {
                        mv: [(0, 0); 4],
                        four_mv: false,
                        not_coded: false,
                    },
                );
                mbs.push(PMbDp::Intra(Box::new(intra_mb)));
            } else {
                // Stash recon → pic, update predictor grids, record MV.
                // For 4MV MBs, all four per-block MVs go into the grid so
                // the next MB's median predictor (§7.5.7) sees them.
                write_recon_to_pic(&mut pic, &inter_mb, mb_x, mb_y);
                reset_pred_grid_mb(&mut pred_grid, mb_x, mb_y);
                let motion = if inter_mb.four_mv {
                    MbMotion {
                        mv: inter_mb.mv4_half,
                        four_mv: true,
                        not_coded: false,
                    }
                } else {
                    MbMotion {
                        mv: [inter_mb.mv_half; 4],
                        four_mv: false,
                        not_coded: inter_mb.skipped,
                    }
                };
                mv_grid.set(mb_x, mb_y, motion);
                mbs.push(PMbDp::Inter(Box::new(inter_mb)));
            }
        }
    }

    // ---- Part 1 emit: per-MB { not_coded, [mcbpc, MV if inter+!not_coded] }.
    // We rebuild a fresh predictor grid for the median MV predictor —
    // emission walks the MBs in scan order and feeds the just-emitted
    // MV into the grid for the next MB's prediction (mirrors the
    // combined-mode emitter in `pvop::emit_p_mb`).
    let mut emit_grid = MvGrid::new(mb_w, mb_h);
    for (mb_idx, mb) in mbs.iter().enumerate() {
        let mb_x = mb_idx % mb_w;
        let mb_y = mb_idx / mb_w;
        match mb {
            PMbDp::Inter(inter) => {
                if inter.skipped {
                    bw.write_bits(1, 1); // not_coded = 1
                                         // Skipped MBs contribute (0,0) to the grid for future
                                         // predictors.
                    emit_grid.set(mb_x, mb_y, MbMotion::default());
                    continue;
                }
                bw.write_bits(0, 1); // not_coded = 0
                let cbpc = ((inter.chroma_coded[0] as u8) << 1) | (inter.chroma_coded[1] as u8);
                if inter.four_mv {
                    // Inter4MV MCBPC (Table B-13 rows 16..=19). The
                    // decoder's `decompose_inter` reads group 4 as
                    // `PMbType::Inter4MV` and switches to the 4MV
                    // motion-decode branch.
                    write_mcbpc_inter4mv(bw, cbpc);
                } else {
                    write_mcbpc_inter(bw, cbpc);
                }
                // (mcsel is a no-op when GMC isn't enabled, and DP rejects GMC.)
                // 1MV: one MVD predicted by the median over (left, top,
                // top-right) on the in-progress emit grid.
                // 4MV: four MVDs, one per 8×8 block. Per §7.6.2 fig 7-6,
                // block k's median predictor may reference blocks 0..k-1
                // of THIS MB, so we commit each freshly-emitted MV to the
                // emit grid before predicting the next block (mirrors the
                // combined-mode `emit_p_mb` 4MV path).
                let range = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
                if inter.four_mv {
                    let mut committed = MbMotion {
                        mv: [(0, 0); 4],
                        four_mv: true,
                        not_coded: false,
                    };
                    for blk in 0..4 {
                        emit_grid.set(mb_x, mb_y, committed);
                        let (px, py) =
                            crate::inter::predict_mv_full(&emit_grid, mb_x, mb_y, blk, true, 0, 0);
                        let (mvx, mvy) = inter.mv4_half[blk];
                        let dx = wrap_mvd(mvx - px, range);
                        let dy = wrap_mvd(mvy - py, range);
                        write_mv_component(bw, dx, f_code_fwd);
                        write_mv_component(bw, dy, f_code_fwd);
                        committed.mv[blk] = (mvx, mvy);
                    }
                    emit_grid.set(mb_x, mb_y, committed);
                } else {
                    let (px, py) =
                        crate::inter::predict_mv_full(&emit_grid, mb_x, mb_y, 0, false, 0, 0);
                    let (mvx, mvy) = inter.mv_half;
                    let dx = wrap_mvd(mvx - px, range);
                    let dy = wrap_mvd(mvy - py, range);
                    write_mv_component(bw, dx, f_code_fwd);
                    write_mv_component(bw, dy, f_code_fwd);
                    emit_grid.set(
                        mb_x,
                        mb_y,
                        MbMotion {
                            mv: [(mvx, mvy); 4],
                            four_mv: false,
                            not_coded: false,
                        },
                    );
                }
            }
            PMbDp::Intra(intra) => {
                // Intra-in-P (Table B-13 rows 4..=7). Emit not_coded=0
                // + Intra MCBPC. NO motion vector — the spec gates the
                // motion_coding() call on `derived_mb_type < 2 ||
                // derived_mb_type == 2` (i.e. Inter / Inter+Q /
                // Inter4MV); Intra MBs (mb_type 3, IntraQ 4) skip it.
                bw.write_bits(0, 1);
                write_mcbpc_p_intra(bw, intra.cbpc);
                // Intra MBs contribute (0,0) to future MV predictors.
                emit_grid.set(
                    mb_x,
                    mb_y,
                    MbMotion {
                        mv: [(0, 0); 4],
                        four_mv: false,
                        not_coded: false,
                    },
                );
            }
        }
    }

    // ---- Motion marker.
    bw.write_bits(MOTION_MARKER, MOTION_MARKER_BITS);

    // ---- Part 2 emit: per-MB { ac_pred_flag if intra, cbpy,
    //                            6 intra DC VLCs if intra }.
    for mb in &mbs {
        match mb {
            PMbDp::Inter(inter) => {
                if inter.skipped {
                    continue;
                }
                // CBPY for inter MBs is bit-inverted (decoder XORs with 0xF).
                let mut mask: u8 = 0;
                for (i, &c) in inter.luma_coded.iter().enumerate() {
                    if c {
                        mask |= 1 << (3 - i);
                    }
                }
                write_cbpy(bw, mask ^ 0xF);
            }
            PMbDp::Intra(intra) => {
                // ac_pred_flag = 0 (encoder never predicts ACs — same
                // policy as the I-VOP DP path).
                bw.write_bits(0, 1);
                // Intra cbpy uses the RAW Table B-9 value as the coded
                // mask (matches our combined-mode I-VOP path in
                // `encoder::write_cbpy`). The DP I-VOP path also emits
                // the raw value; we stay consistent here.
                write_cbpy(bw, intra.cbpy);
                // No dquant — the DP encoder uses constant quant
                // (mb_type stays = 3, the "Intra" group, never Intra+Q).
                // Six intra DC differentials (Y0..Y3, Cb, Cr) — same
                // VLC layout the I-VOP DP path uses, since
                // `intra_dc_vlc_thr = 0` ⇒ `use_intra_dc_vlc = true`
                // for every quant we emit (threshold 99 vs quant ≤ 31).
                for (blk, (diff, _recon)) in intra.dc.iter().enumerate() {
                    write_intra_dc_diff(bw, blk, *diff);
                }
            }
        }
    }

    // ---- Part 3 emit: per-MB { AC walks for each coded block }.
    //
    // Inter MBs: `pvop::write_inter_ac` (Table B-17) or RVLC variant.
    // Intra MBs: `encoder::write_intra_ac` (Table B-16) or RVLC variant.
    for mb in &mbs {
        match mb {
            PMbDp::Inter(inter) => {
                if inter.skipped {
                    continue;
                }
                for blk in 0..6 {
                    let coded = if blk < 4 {
                        inter.luma_coded[blk]
                    } else {
                        inter.chroma_coded[blk - 4]
                    };
                    if !coded {
                        continue;
                    }
                    if reversible_vlc {
                        crate::rvlc::write_inter_ac(bw, &inter.ac_levels[blk]);
                    } else {
                        write_inter_ac(bw, &inter.ac_levels[blk]);
                    }
                }
            }
            PMbDp::Intra(intra) => {
                for blk in 0..6 {
                    let coded = if blk < 4 {
                        intra.luma_coded[blk]
                    } else {
                        intra.chroma_coded[blk - 4]
                    };
                    if !coded {
                        continue;
                    }
                    if reversible_vlc {
                        crate::rvlc::write_intra_ac(bw, &intra.ac_levels[blk])?;
                    } else {
                        write_intra_ac(bw, &intra.ac_levels[blk])?;
                    }
                }
            }
        }
    }

    Ok((pic, mv_grid))
}

// -------------------------------------------------------------------------
// P-VOP DP decode
// -------------------------------------------------------------------------

/// Decode one P-VOP body emitted under `data_partitioned_p_vop()`.
/// Round-22 scope mirrors the encoder: 1MV-Inter + Intra-in-P, no GMC,
/// no dquant, no interlace. Treats the entire picture as one video
/// packet.
pub fn decode_pvop_dp_with_grid(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
    reference: &IVopPicture,
) -> Result<(IVopPicture, MvGrid)> {
    use crate::block::decode_inter_ac;
    use crate::inter::decode_mv_component;
    use crate::iq::INTRA_DC_VLC_THR_TABLE;
    use crate::mc::luma_mv_to_chroma;
    use crate::pvop::{predict_chroma_block, predict_luma_mb};

    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mb_total = mb_w * mb_h;

    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);
    // Intra DC predictor grid — mirrors the combined-mode decoder
    // (`inter::reset_pred_grid_mb`). Inter / skipped MBs reset their
    // slot so a later intra MB reads `dc=1024, is_intra=false` from
    // inter neighbours.
    let mut pred_grid = PredGrid::new(mb_w, mb_h);

    let quant = vop.vop_quant;
    let f_code = vop.vop_fcode_forward.max(1);
    let rounding = vop.rounding_type;
    let thr = INTRA_DC_VLC_THR_TABLE[vop.intra_dc_vlc_thr as usize] as u32;
    let use_intra_dc_vlc = quant < thr.max(1);

    /// Per-MB carrier flowing through parts 1 → 2 → 3 → reconstruct.
    /// `Skipped` is `not_coded = 1` (no residual, copy reference at
    /// MV(0,0)). `Inter` carries the inter motion + residual state
    /// (1MV when `four_mv = false`, four per-block MVs when `true`);
    /// `Intra` carries the intra DC differentials + residual state.
    enum PMbState {
        Skipped,
        Inter {
            cbpc: u8,
            /// Per-block luma MVs. For 1MV MBs all four entries hold the
            /// single MV; for Inter4MV MBs they hold the four per-block
            /// MVs in block order 0=(0,0) 1=(8,0) 2=(0,8) 3=(8,8).
            mv4: [(i32, i32); 4],
            four_mv: bool,
            cbpy_inv: u8, // raw VLC value (inter MBs invert with ^0xF)
            ac_levels: [[i32; 64]; 6],
        },
        Intra {
            cbpc: u8,
            ac_pred: bool,
            cbpy: u8, // raw VLC value (intra MBs use it directly)
            dc_diff: [i32; 6],
            ac_levels: [[i32; 64]; 6],
        },
    }

    let mut mbs: Vec<PMbState> = Vec::with_capacity(mb_total);

    // ---- Part 1: per-MB { not_coded, [mcbpc + (MV if inter)] }.
    for mb_idx in 0..mb_total {
        let mb_x = mb_idx % mb_w;
        let mb_y = mb_idx / mb_w;
        let not_coded = br.read_u1()? == 1;
        if not_coded {
            mv_grid.set(
                mb_x,
                mb_y,
                MbMotion {
                    not_coded: true,
                    ..MbMotion::default()
                },
            );
            mbs.push(PMbState::Skipped);
            continue;
        }
        let mcbpc_v = loop {
            let v = vlc::decode(br, mcbpc::p_table())?;
            if v != mcbpc::INTER_STUFFING {
                break v;
            }
        };
        let (mb_type, cbpc) = mcbpc::decompose_inter(mcbpc_v);
        match mb_type {
            mcbpc::PMbType::Inter => {
                // (No mcsel — GMC is rejected at encoder factory time.)
                // Motion vector with median predictor (slice_first_mb =
                // (0,0) since this packet starts at MB 0).
                let (px, py) = crate::inter::predict_mv_full(&mv_grid, mb_x, mb_y, 0, false, 0, 0);
                let mvx = decode_mv_component(br, f_code, px)?;
                let mvy = decode_mv_component(br, f_code, py)?;
                mv_grid.set(
                    mb_x,
                    mb_y,
                    MbMotion {
                        mv: [(mvx, mvy); 4],
                        four_mv: false,
                        not_coded: false,
                    },
                );
                mbs.push(PMbState::Inter {
                    cbpc,
                    mv4: [(mvx, mvy); 4],
                    four_mv: false,
                    cbpy_inv: 0,
                    ac_levels: [[0i32; 64]; 6],
                });
            }
            mcbpc::PMbType::Inter4MV => {
                // Inter4MV (Table B-13 rows 16..=19, group `value >> 2 == 4`).
                // Decode four MVDs in block order. Per §7.6.2 fig 7-6
                // each block's predictor may reference the prior blocks
                // of THIS MB, so commit each freshly-decoded MV into
                // the grid before predicting the next block. Mirrors
                // `inter::decode_p_mb`'s 4MV path.
                let mut motion = MbMotion {
                    mv: [(0, 0); 4],
                    four_mv: true,
                    not_coded: false,
                };
                for blk in 0..4 {
                    mv_grid.set(mb_x, mb_y, motion);
                    let (px, py) =
                        crate::inter::predict_mv_full(&mv_grid, mb_x, mb_y, blk, true, 0, 0);
                    let mvx = decode_mv_component(br, f_code, px)?;
                    let mvy = decode_mv_component(br, f_code, py)?;
                    motion.mv[blk] = (mvx, mvy);
                }
                mv_grid.set(mb_x, mb_y, motion);
                mbs.push(PMbState::Inter {
                    cbpc,
                    mv4: motion.mv,
                    four_mv: true,
                    cbpy_inv: 0,
                    ac_levels: [[0i32; 64]; 6],
                });
            }
            mcbpc::PMbType::Intra => {
                // Intra-in-P (Table B-13 mb_type = 3). No motion vector
                // in part 1 — the spec's `data_partitioned_p_vop()`
                // gates `motion_coding()` on `derived_mb_type < 2 ||
                // == 2`. cbpy + ac_pred_flag + DC VLCs land in part 2.
                // Intra MBs contribute (0,0) to the median predictor
                // for future inter MBs (§7.6.7 step 3).
                mv_grid.set(
                    mb_x,
                    mb_y,
                    MbMotion {
                        mv: [(0, 0); 4],
                        four_mv: false,
                        not_coded: false,
                    },
                );
                mbs.push(PMbState::Intra {
                    cbpc,
                    ac_pred: false,
                    cbpy: 0,
                    dc_diff: [0; 6],
                    ac_levels: [[0i32; 64]; 6],
                });
            }
            other => {
                // Round-22 DP decoder rejects InterQ / IntraQ /
                // Inter4MV / Inter4MV+Q — none are emitted by our
                // encoder (DP holds vop_quant constant and rejects
                // 4MV / GMC at the factory).
                return Err(Error::unsupported(format!(
                    "mpeg4 DP P: unsupported mb_type {:?}",
                    other
                )));
            }
        }
    }

    // ---- Motion marker.
    let m = br.read_u32(MOTION_MARKER_BITS)?;
    if m != MOTION_MARKER {
        return Err(Error::invalid("mpeg4 DP P: motion_marker mismatch"));
    }

    // ---- Part 2: per-MB { [ac_pred_flag if intra], cbpy,
    //                       [6 intra DC VLCs if intra && use_intra_dc_vlc] }.
    for mb in mbs.iter_mut() {
        match mb {
            PMbState::Skipped => continue,
            PMbState::Inter { cbpy_inv, .. } => {
                *cbpy_inv = vlc::decode(br, cbpy_tab::table())? as u8;
            }
            PMbState::Intra {
                ac_pred,
                cbpy,
                dc_diff,
                ..
            } => {
                *ac_pred = br.read_u1()? == 1;
                *cbpy = vlc::decode(br, cbpy_tab::table())? as u8;
                // No dquant (mb_type stays = 3 in our encoder).
                if use_intra_dc_vlc {
                    for blk in 0..6 {
                        dc_diff[blk] = decode_intra_dc_diff(br, blk)?;
                    }
                }
            }
        }
    }

    // ---- Part 3: per-MB { AC walks for each coded block }.
    //
    // RVLC path: route the AC partition through the §E.1.4.4.2.1
    // strategy 1-4 picker (forward + reverse walkers, then merge by
    // strategy). Non-RVLC path: simple per-block forward walker.
    if vol.reversible_vlc {
        // Build a flat block descriptor list across all coded blocks
        // (intra + inter) in emission order. Each `(mb_idx, blk_idx)`
        // entry tells us where to scatter the picker's output.
        let mut descs: Vec<crate::rvlc::RvlcBlockDesc> = Vec::new();
        let mut slots: Vec<(usize, usize)> = Vec::new();
        for (mb_idx, mb) in mbs.iter().enumerate() {
            match mb {
                PMbState::Skipped => continue,
                PMbState::Inter { cbpc, cbpy_inv, .. } => {
                    let cbpy_mask = *cbpy_inv ^ 0xF;
                    let luma_coded = [
                        (cbpy_mask >> 3) & 1 != 0,
                        (cbpy_mask >> 2) & 1 != 0,
                        (cbpy_mask >> 1) & 1 != 0,
                        cbpy_mask & 1 != 0,
                    ];
                    let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                    for blk in 0..6 {
                        let coded = if blk < 4 {
                            luma_coded[blk]
                        } else {
                            chroma_coded[blk - 4]
                        };
                        if !coded {
                            continue;
                        }
                        descs.push(crate::rvlc::RvlcBlockDesc { is_intra: false });
                        slots.push((mb_idx, blk));
                    }
                }
                PMbState::Intra { cbpc, cbpy, .. } => {
                    let luma_coded = [
                        (*cbpy >> 3) & 1 != 0,
                        (*cbpy >> 2) & 1 != 0,
                        (*cbpy >> 1) & 1 != 0,
                        *cbpy & 1 != 0,
                    ];
                    let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                    for blk in 0..6 {
                        let coded = if blk < 4 {
                            luma_coded[blk]
                        } else {
                            chroma_coded[blk - 4]
                        };
                        if !coded {
                            continue;
                        }
                        descs.push(crate::rvlc::RvlcBlockDesc { is_intra: true });
                        slots.push((mb_idx, blk));
                    }
                }
            }
        }
        let mut out_blocks: Vec<[i32; 64]> = vec![[0i32; 64]; descs.len()];
        let total_bits = br.bits_remaining();
        let aligned = drain_to_aligned_buffer(br, total_bits);
        let (_stats, _outcomes) = crate::rvlc::decode_rvlc_ac_partition(
            &aligned,
            0,
            total_bits,
            &descs,
            &ZIGZAG,
            &mut out_blocks,
        )?;
        for ((mb_idx, blk), coeffs) in slots.into_iter().zip(out_blocks) {
            match &mut mbs[mb_idx] {
                PMbState::Skipped => unreachable!(),
                PMbState::Inter { ac_levels, .. } => ac_levels[blk] = coeffs,
                PMbState::Intra { ac_levels, .. } => ac_levels[blk] = coeffs,
            }
        }
    } else {
        for mb in mbs.iter_mut() {
            match mb {
                PMbState::Skipped => continue,
                PMbState::Inter {
                    cbpc,
                    cbpy_inv,
                    ac_levels,
                    ..
                } => {
                    let cbpy_mask = *cbpy_inv ^ 0xF;
                    let luma_coded = [
                        (cbpy_mask >> 3) & 1 != 0,
                        (cbpy_mask >> 2) & 1 != 0,
                        (cbpy_mask >> 1) & 1 != 0,
                        cbpy_mask & 1 != 0,
                    ];
                    let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                    for blk in 0..6 {
                        let coded = if blk < 4 {
                            luma_coded[blk]
                        } else {
                            chroma_coded[blk - 4]
                        };
                        if !coded {
                            continue;
                        }
                        decode_inter_ac(br, &mut ac_levels[blk], &ZIGZAG)?;
                    }
                }
                PMbState::Intra {
                    cbpc,
                    cbpy,
                    ac_levels,
                    ..
                } => {
                    let luma_coded = [
                        (*cbpy >> 3) & 1 != 0,
                        (*cbpy >> 2) & 1 != 0,
                        (*cbpy >> 1) & 1 != 0,
                        *cbpy & 1 != 0,
                    ];
                    let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                    for blk in 0..6 {
                        let coded = if blk < 4 {
                            luma_coded[blk]
                        } else {
                            chroma_coded[blk - 4]
                        };
                        if !coded {
                            continue;
                        }
                        decode_intra_ac(br, &mut ac_levels[blk], &ZIGZAG)?;
                    }
                }
            }
        }
    }

    // ---- Reconstruct each MB.
    for (mb_idx, mb) in mbs.iter().enumerate() {
        let mb_x = mb_idx % mb_w;
        let mb_y = mb_idx / mb_w;
        match mb {
            PMbState::Skipped => {
                // Copy 16×16 luma + 8×8 chroma at MV(0,0). Intra
                // predictor neighbour for this MB stays at the
                // default `is_intra=false, dc=1024` (no work needed:
                // PredGrid is already initialised that way and we
                // never write to a skipped MB's slot).
                copy_skipped_mb(&mut pic, reference, mb_x, mb_y);
            }
            PMbState::Inter {
                cbpc,
                mv4,
                four_mv,
                cbpy_inv,
                ac_levels,
            } => {
                let cbpy_mask = *cbpy_inv ^ 0xF;
                let luma_coded = [
                    (cbpy_mask >> 3) & 1 != 0,
                    (cbpy_mask >> 2) & 1 != 0,
                    (cbpy_mask >> 1) & 1 != 0,
                    cbpy_mask & 1 != 0,
                ];
                let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                // Build luma MB predictor. 1MV: one MV applied to all
                // four luma blocks via `predict_luma_mb`. 4MV: each 8×8
                // block uses its own MV per `predict_luma_mb_4mv`.
                let mut pred_y = [0u8; 256];
                if *four_mv {
                    crate::pvop::predict_luma_mb_4mv(
                        reference,
                        mb_x,
                        mb_y,
                        *mv4,
                        rounding,
                        &mut pred_y,
                    );
                } else {
                    let (mvx, mvy) = mv4[0];
                    predict_luma_mb(reference, mb_x, mb_y, mvx, mvy, rounding, &mut pred_y);
                }
                let q = quant as i32;
                for blk in 0..4 {
                    let (sub_x, sub_y) = match blk {
                        0 => (0usize, 0usize),
                        1 => (8, 0),
                        2 => (0, 8),
                        3 => (8, 8),
                        _ => unreachable!(),
                    };
                    let mut block_pred = [0u8; 64];
                    for j in 0..8 {
                        for i in 0..8 {
                            block_pred[j * 8 + i] = pred_y[(sub_y + j) * 16 + (sub_x + i)];
                        }
                    }
                    let recon =
                        add_inter_residual_block(&block_pred, &ac_levels[blk], q, luma_coded[blk]);
                    let px = mb_x * 16 + sub_x;
                    let py = mb_y * 16 + sub_y;
                    for j in 0..8 {
                        for i in 0..8 {
                            pic.y[(py + j) * pic.y_stride + (px + i)] = recon[j * 8 + i];
                        }
                    }
                }
                // Chroma — 1MV uses the single luma MV scaled per §7.5.4;
                // 4MV uses the average of the four luma MVs (§7.5.9.5).
                let (cmx, cmy) = if *four_mv {
                    let sx: i32 = mv4.iter().map(|(x, _)| *x).sum();
                    let sy: i32 = mv4.iter().map(|(_, y)| *y).sum();
                    (luma_mv_to_chroma(sx / 4), luma_mv_to_chroma(sy / 4))
                } else {
                    let (mvx, mvy) = mv4[0];
                    (luma_mv_to_chroma(mvx), luma_mv_to_chroma(mvy))
                };
                let mut pred_cb = [0u8; 64];
                let mut pred_cr = [0u8; 64];
                predict_chroma_block(
                    &reference.cb,
                    reference.c_stride,
                    mb_x,
                    mb_y,
                    cmx,
                    cmy,
                    rounding,
                    &mut pred_cb,
                );
                predict_chroma_block(
                    &reference.cr,
                    reference.c_stride,
                    mb_x,
                    mb_y,
                    cmx,
                    cmy,
                    rounding,
                    &mut pred_cr,
                );
                let recon_cb =
                    add_inter_residual_block(&pred_cb, &ac_levels[4], q, chroma_coded[0]);
                let recon_cr =
                    add_inter_residual_block(&pred_cr, &ac_levels[5], q, chroma_coded[1]);
                let cx = mb_x * 8;
                let cy = mb_y * 8;
                for j in 0..8 {
                    for i in 0..8 {
                        pic.cb[(cy + j) * pic.c_stride + (cx + i)] = recon_cb[j * 8 + i];
                        pic.cr[(cy + j) * pic.c_stride + (cx + i)] = recon_cr[j * 8 + i];
                    }
                }
                // Inter MBs reset the intra DC predictor slot (so a
                // later intra MB sees `is_intra=false`).
                crate::pvop::reset_pred_grid_mb(&mut pred_grid, mb_x, mb_y);
            }
            PMbState::Intra {
                cbpc,
                cbpy,
                dc_diff,
                ac_levels,
                ac_pred: _,
            } => {
                // Intra-in-P reconstruction. Mirror the I-VOP DP
                // decoder's per-block recipe: predict DC from the
                // PredGrid, apply the differential, dequantise ACs,
                // IDCT, write to picture.
                let q = quant as i32;
                let q_plus = if q & 1 == 1 { q } else { q - 1 };
                let luma_coded = [
                    (*cbpy >> 3) & 1 != 0,
                    (*cbpy >> 2) & 1 != 0,
                    (*cbpy >> 1) & 1 != 0,
                    *cbpy & 1 != 0,
                ];
                let chroma_coded = [(*cbpc >> 1) & 1 != 0, *cbpc & 1 != 0];
                for blk in 0..6 {
                    let (left, top_left, top) = lookup_neighbour_dcs(blk, mb_x, mb_y, &pred_grid);
                    let (predicted_dc_pel, _dir) = choose_dc_predictor(left, top_left, top);
                    let scale = dc_scaler(blk, quant) as i32;
                    let pred_units = (predicted_dc_pel + scale / 2) / scale;
                    let units = pred_units + dc_diff[blk];
                    let recon_dc = (units * scale).clamp(0, 2047);
                    update_neighbour(&mut pred_grid, blk, mb_x, mb_y, recon_dc, quant as u8);
                    let coded = if blk < 4 {
                        luma_coded[blk]
                    } else {
                        chroma_coded[blk - 4]
                    };
                    let mut coeffs = if coded { ac_levels[blk] } else { [0i32; 64] };
                    for i in 1..64 {
                        let l = coeffs[i];
                        if l == 0 {
                            continue;
                        }
                        let abs = l.abs();
                        let mut val = 2 * q * abs + q_plus;
                        if l < 0 {
                            val = -val;
                        }
                        coeffs[i] = val.clamp(-2048, 2047);
                    }
                    coeffs[0] = recon_dc;
                    let mut f = [0.0f32; 64];
                    for i in 0..64 {
                        f[i] = coeffs[i] as f32;
                    }
                    crate::block::idct8x8(&mut f);
                    write_recon_to_picture(&mut pic, blk, mb_x, mb_y, &f);
                }
            }
        }
    }

    Ok((pic, mv_grid))
}

/// Mirror of `inter::copy_skipped_mb` for the DP P-VOP decoder. Copies
/// the 16×16 luma + 8×8 chroma blocks at MV(0,0) from `reference` into
/// `pic` at `(mb_x, mb_y)`.
fn copy_skipped_mb(pic: &mut IVopPicture, reference: &IVopPicture, mb_x: usize, mb_y: usize) {
    let px = mb_x * 16;
    let py = mb_y * 16;
    for j in 0..16 {
        for i in 0..16 {
            pic.y[(py + j) * pic.y_stride + (px + i)] =
                reference.y[(py + j) * reference.y_stride + (px + i)];
        }
    }
    let cx = mb_x * 8;
    let cy = mb_y * 8;
    for j in 0..8 {
        for i in 0..8 {
            pic.cb[(cy + j) * pic.c_stride + (cx + i)] =
                reference.cb[(cy + j) * reference.c_stride + (cx + i)];
            pic.cr[(cy + j) * pic.c_stride + (cx + i)] =
                reference.cr[(cy + j) * reference.c_stride + (cx + i)];
        }
    }
}

/// Inter residual reconstruction: H.263 inter dequant + IDCT, then add
/// to predictor + clip to u8.
fn add_inter_residual_block(pred: &[u8; 64], levels: &[i32; 64], q: i32, coded: bool) -> [u8; 64] {
    if !coded {
        return *pred;
    }
    // H.263 inter dequant: recon = sign(l) * (2*Q*|l| + Q_plus) for l != 0,
    // 0 otherwise. Q_plus = Q for odd Q, Q-1 for even.
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let mut coeffs = *levels;
    for i in 0..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        let abs = l.abs();
        let mut val = 2 * q * abs + q_plus;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
    }
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = coeffs[i] as f32;
    }
    crate::block::idct8x8(&mut f);
    let mut out = [0u8; 64];
    for i in 0..64 {
        let r = f[i].round() as i32 + pred[i] as i32;
        out[i] = r.clamp(0, 255) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_bit_patterns() {
        // Verify spec-given bit patterns.
        // DC marker = 110 1011 0000 0000 0001 (binary, 19 bits).
        assert_eq!(DC_MARKER, 0b110_1011_0000_0000_0001);
        assert_eq!(DC_MARKER_BITS, 19);
        // Motion marker = 1 1111 0000 0000 0001 (binary, 17 bits).
        assert_eq!(MOTION_MARKER, 0b1_1111_0000_0000_0001);
        assert_eq!(MOTION_MARKER_BITS, 17);
    }
}

//! Interlaced field coding support (ISO/IEC 14496-2 §6.2.7.3, §7.6).
//!
//! When the VOL advertises `interlaced == 1` AND the VOP advertises
//! `interlaced == 1`, each MB carries an extra `interlaced_information()`
//! payload after the `dquant` field. The flags it contains are:
//!
//! * `dct_type` — 1-bit, present when the MB has coded blocks OR is intra.
//!   * 0 = frame DCT coding (rows are interleaved top/bottom fields as
//!     usual),
//!   * 1 = field DCT coding (the top four rows of each 8×8 luma block come
//!     from one field, the bottom four rows from the other).
//! * `field_prediction` — 1-bit, only for non-intra single-MV (and
//!   non-direct B-VOP) MBs. Enables field-based motion compensation: the
//!   MB uses TWO MVs (one per field) instead of one.
//! * `forward_{top,bottom}_field_reference` — 1-bit each; pick which field
//!   of the forward reference each half of the MB pulls from.
//! * `backward_{top,bottom}_field_reference` — same, for the backward
//!   reference (B-VOP only).
//!
//! Field-DCT reconstruction (§7.6.1): after IDCT the 8×8 block is treated
//! as the top or bottom field of the MB; rows 0..=3 map to even frame rows
//! and rows 4..=7 map to odd frame rows (i.e. the MB's row order is
//! [0,2,4,6,8,10,12,14] and [1,3,5,7,9,11,13,15]). Chrominance is
//! unaffected by `dct_type`.
//!
//! Alternate-vertical scan (§7.4.3.3 — Table 7-3c): when
//! `vop.alternate_vertical_scan == true`, all intra AND inter blocks in
//! the VOP use `ALTERNATE_VERTICAL_SCAN` irrespective of AC prediction
//! direction. The helper `choose_scan_interlaced()` in this module
//! formalises the selection.

use oxideav_core::Result;

use crate::headers::vol::{ALTERNATE_HORIZONTAL_SCAN, ALTERNATE_VERTICAL_SCAN, ZIGZAG};
use crate::headers::vop::{VideoObjectPlane, VopCodingType};
use oxideav_core::bits::BitReader;

/// Per-MB interlaced information decoded from §6.2.7.3.
#[derive(Clone, Copy, Debug, Default)]
pub struct InterlacedInfo {
    /// `dct_type` — true = field DCT coding, false = frame DCT coding.
    pub dct_type: bool,
    /// `field_prediction` — true = two-MV (per-field) MC.
    pub field_prediction: bool,
    /// Forward top field ref. (0 = top field, 1 = bottom field).
    pub forward_top_field_ref: bool,
    /// Forward bottom field ref. (0 = top field, 1 = bottom field).
    pub forward_bottom_field_ref: bool,
    /// Backward top field ref. (B-VOP only).
    pub backward_top_field_ref: bool,
    /// Backward bottom field ref. (B-VOP only).
    pub backward_bottom_field_ref: bool,
}

/// Role of this MB w.r.t. reading interlaced_information(). Mirrors the
/// "derived_mbtype" switch in §6.2.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbClass {
    /// I-VOP MB or P-VOP intra MB (derived_mbtype == 3 or 4). `dct_type`
    /// is always present; `field_prediction` is never.
    Intra,
    /// P-VOP inter (single-MV) MB (derived_mbtype == 0 or 1). `dct_type`
    /// present iff `cbp != 0`; `field_prediction` present if interlaced.
    InterSingleMv,
    /// P-VOP 4MV inter MB (derived_mbtype == 2). `dct_type` present iff
    /// `cbp != 0`; `field_prediction` never.
    Inter4Mv,
    /// B-VOP non-direct MB. `dct_type` present iff `cbp != 0`;
    /// `field_prediction` present.
    BVopNonDirect,
    /// B-VOP direct MB. `dct_type` present iff `cbp != 0`;
    /// `field_prediction` is never.
    BVopDirect,
}

/// Parse `interlaced_information()` per §6.2.7.3. `cbp_nonzero` is true when
/// either CBPY or CBPC has any coded block. `vop_type` provides the B/P/I
/// context for field_prediction gating.
pub fn parse_interlaced_information(
    br: &mut BitReader<'_>,
    class: MbClass,
    vop_type: VopCodingType,
    cbp_nonzero: bool,
) -> Result<InterlacedInfo> {
    let mut info = InterlacedInfo::default();

    // dct_type present when: MB is intra (class == Intra) OR cbp != 0.
    if matches!(class, MbClass::Intra) || cbp_nonzero {
        info.dct_type = br.read_u1()? == 1;
    }

    // field_prediction: only for non-intra single-MV P-VOP MBs and
    // non-direct B-VOP MBs.
    let field_pred_allowed = match class {
        MbClass::InterSingleMv => vop_type == VopCodingType::P,
        MbClass::BVopNonDirect => vop_type == VopCodingType::B,
        _ => false,
    };
    if field_pred_allowed {
        info.field_prediction = br.read_u1()? == 1;
        if info.field_prediction {
            // Forward top/bottom field reference — present if the MB is
            // not backward-only predicted.
            if vop_type == VopCodingType::P || class == MbClass::BVopNonDirect {
                // In a P-VOP all field-predicted MBs use forward references.
                // In a B-VOP `backward-only` mode is `mb_type == "001"` but
                // at this layer we don't distinguish between the B-VOP
                // sub-modes; callers that need backward-only suppression
                // of the forward refs must override the bits they don't
                // actually consume. We read them per the committee-draft
                // grammar rather than gate further.
                info.forward_top_field_ref = br.read_u1()? == 1;
                info.forward_bottom_field_ref = br.read_u1()? == 1;
            }
            if vop_type == VopCodingType::B && class == MbClass::BVopNonDirect {
                info.backward_top_field_ref = br.read_u1()? == 1;
                info.backward_bottom_field_ref = br.read_u1()? == 1;
            }
        }
    }

    Ok(info)
}

/// Select the coefficient scan order for an interlaced VOP.
///
/// * When `vop.alternate_vertical_scan == true`: always use
///   `ALTERNATE_VERTICAL_SCAN` regardless of AC prediction direction.
/// * Otherwise fall back to the progressive choice (zigzag / alt-horizontal
///   / alt-vertical per AC prediction direction).
pub fn choose_scan_interlaced(
    vop: &VideoObjectPlane,
    ac_pred: bool,
    dir: Option<crate::block::PredDir>,
) -> &'static [usize; 64] {
    if vop.alternate_vertical_scan {
        return &ALTERNATE_VERTICAL_SCAN;
    }
    if !ac_pred {
        return &ZIGZAG;
    }
    match dir {
        Some(crate::block::PredDir::Left) => &ALTERNATE_VERTICAL_SCAN,
        Some(crate::block::PredDir::Top) => &ALTERNATE_HORIZONTAL_SCAN,
        None => &ZIGZAG,
    }
}

/// Reorder an 8x8 block's rows after IDCT to undo field-DCT mapping.
///
/// When an MB is field-DCT coded, the four luminance blocks are actually
/// [Y_top_left, Y_top_right, Y_bottom_left, Y_bottom_right] where the
/// top two come from the top field (even frame rows 0,2,4,6,8,10,12,14)
/// and the bottom two come from the bottom field (odd frame rows
/// 1,3,5,7,9,11,13,15). The IDCT output is a field sample — rows 0..=7
/// within the 8x8 are FIELD rows. When writing to the MB's frame buffer
/// we must interleave top-field and bottom-field lines.
///
/// This helper operates at the **MB level**, not the per-block level:
/// given four 8x8 luma blocks laid out as field-DCT (blk0=top-left-top-
/// field, blk1=top-right-top-field, blk2=bot-left-bot-field,
/// blk3=bot-right-bot-field), assemble the 16×16 frame-organised output
/// into `out_mb`.
pub fn field_dct_reorder_mb(
    top_left: &[i32; 64],
    top_right: &[i32; 64],
    bot_left: &[i32; 64],
    bot_right: &[i32; 64],
    out_mb: &mut [i32; 256],
) {
    // Top field (blk0 = left, blk1 = right) contributes to frame rows
    // 0, 2, 4, 6, 8, 10, 12, 14.
    // Bottom field (blk2 = left, blk3 = right) contributes to frame rows
    // 1, 3, 5, 7, 9, 11, 13, 15.
    for field_row in 0..8 {
        let top_frame_row = field_row * 2;
        let bot_frame_row = field_row * 2 + 1;
        // Top field left half.
        for col in 0..8 {
            out_mb[top_frame_row * 16 + col] = top_left[field_row * 8 + col];
        }
        // Top field right half.
        for col in 0..8 {
            out_mb[top_frame_row * 16 + 8 + col] = top_right[field_row * 8 + col];
        }
        // Bottom field left half.
        for col in 0..8 {
            out_mb[bot_frame_row * 16 + col] = bot_left[field_row * 8 + col];
        }
        // Bottom field right half.
        for col in 0..8 {
            out_mb[bot_frame_row * 16 + 8 + col] = bot_right[field_row * 8 + col];
        }
    }
}

/// Sample a full 16x16 macroblock prediction from a reference using
/// two field motion vectors (top + bottom), per §7.6.2.2's
/// `field_motion_compensate_one_reference()` pseudocode.
///
/// The luma MB is assembled row-interleaved — even rows (0, 2, 4, …, 14)
/// come from the top-field MC, odd rows (1, 3, …, 15) from the
/// bottom-field MC. Each field-MC call samples an 8x16 strip from the
/// reference's top or bottom field, selected by `top_field_ref` /
/// `bot_field_ref`.
///
/// `(mb_px, mb_py)` are the top-left frame coordinates of the current
/// MB. `mv_top = (x, y_frame_half)` and `mv_bot = (x, y_frame_half)`
/// are the field MVs expressed in the frame-coordinate half-pel units
/// carried on the MV grid (vertical components already scaled by 2).
///
/// Writes into `out_mb` at row stride 16.
pub fn field_predict_luma_mb(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_h: usize,
    mb_px: i32,
    mb_py: i32,
    mv_top: (i32, i32),
    mv_bot: (i32, i32),
    top_field_ref: bool,
    bot_field_ref: bool,
    rounding: bool,
    out_mb: &mut [u8; 256],
) {
    // Each field-MC call samples an 8-row field strip into a 16-col-wide
    // buffer, then we interleave the two strips into a 16x16 frame MB.
    //
    // `field_predict_luma_block` takes 8x8 blocks — we call it four times,
    // two per field (left + right halves of the MB).
    //
    // Vertical MV in field coordinates is mv.y / 2 (frame half-pel divided
    // by 2, since one row in the field plane = two rows in the frame
    // plane). The frame-coordinate base y translates to field-plane y by
    // dividing by 2.
    let field_py = mb_py / 2;

    // Top-field 8x16 strip — two 8x8 blocks side by side.
    for (x_off, out_col) in [(0i32, 0usize), (8i32, 8usize)] {
        let mut blk = [0u8; 64];
        field_predict_luma_block(
            ref_plane,
            ref_stride,
            ref_h,
            mb_px + x_off,
            field_py,
            mv_top.0,
            mv_top.1 / 2,
            rounding,
            top_field_ref,
            &mut blk,
        );
        for r in 0..8 {
            let frame_row = r * 2; // top-field: even rows 0,2,4,...,14
            for c in 0..8 {
                out_mb[frame_row * 16 + out_col + c] = blk[r * 8 + c];
            }
        }
    }

    // Bottom-field 8x16 strip — same layout, but writes odd frame rows.
    for (x_off, out_col) in [(0i32, 0usize), (8i32, 8usize)] {
        let mut blk = [0u8; 64];
        field_predict_luma_block(
            ref_plane,
            ref_stride,
            ref_h,
            mb_px + x_off,
            field_py,
            mv_bot.0,
            mv_bot.1 / 2,
            rounding,
            bot_field_ref,
            &mut blk,
        );
        for r in 0..8 {
            let frame_row = r * 2 + 1; // bottom-field: odd rows 1,3,...,15
            for c in 0..8 {
                out_mb[frame_row * 16 + out_col + c] = blk[r * 8 + c];
            }
        }
    }
}

/// Chroma counterpart to `field_predict_luma_mb` — samples an 8x8 chroma
/// MB from a reference using two field MVs. The chroma MVs are
/// `Div2Round`-reduced from the luma MVs by the caller.
pub fn field_predict_chroma_mb(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_h: usize,
    c_px: i32,
    c_py: i32,
    mv_top_c: (i32, i32),
    mv_bot_c: (i32, i32),
    top_field_ref: bool,
    bot_field_ref: bool,
    rounding: bool,
    out_mb: &mut [u8; 64],
) {
    // Chroma is 8x8 per MB (4:2:0). Split into two 8x4 field strips:
    // rows 0,2,4,6 from top field, rows 1,3,5,7 from bottom field.
    //
    // We sample each field as a single 8x4 strip via `field_predict_luma_block`
    // (which is structurally field-based even for luma-sized blocks), writing
    // only the 4 rows relevant to the field.
    let field_stride = ref_stride * 2;
    let field_h = ref_h / 2;
    // Top-field 8x4 strip.
    let field_base_top = if top_field_ref { ref_stride } else { 0 };
    let mut tmp_top = [0u8; 64];
    crate::mc::predict_block(
        &ref_plane[field_base_top.min(ref_plane.len())..],
        field_stride,
        ref_stride as i32,
        field_h as i32,
        c_px,
        c_py / 2,
        mv_top_c.0,
        mv_top_c.1 / 2,
        8,
        rounding,
        &mut tmp_top,
        8,
    );
    // Bottom-field 8x4 strip.
    let field_base_bot = if bot_field_ref { ref_stride } else { 0 };
    let mut tmp_bot = [0u8; 64];
    crate::mc::predict_block(
        &ref_plane[field_base_bot.min(ref_plane.len())..],
        field_stride,
        ref_stride as i32,
        field_h as i32,
        c_px,
        c_py / 2,
        mv_bot_c.0,
        mv_bot_c.1 / 2,
        8,
        rounding,
        &mut tmp_bot,
        8,
    );
    for r in 0..4 {
        for c in 0..8 {
            out_mb[(r * 2) * 8 + c] = tmp_top[r * 8 + c];
            out_mb[(r * 2 + 1) * 8 + c] = tmp_bot[r * 8 + c];
        }
    }
}

/// Field-based luma MC sample: predict an 8x8 field block from the
/// reference, scaling the vertical MV by 2 so it addresses a field row
/// (rather than a frame row). The caller positions `block_py_field` such
/// that it's the field-row index within the reference's field plane
/// (0..=frame_h/2 - 1). The reference pool is treated as two separate
/// fields.
///
/// Returns the predicted 8x8 field block in `out` at row stride 8.
pub fn field_predict_luma_block(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_h: usize,
    blk_px: i32,
    field_py: i32,
    mv_x_half: i32,
    mv_y_field_half: i32,
    rounding: bool,
    which_ref_field: bool,
    out: &mut [u8; 64],
) {
    // The reference field is the odd or even lines of the reference frame,
    // depending on `which_ref_field`. We construct a logical "field plane"
    // by sampling every other line starting at 0 (top field) or 1 (bottom).
    // This is equivalent to calling `predict_block` on that field plane
    // with stride=2*ref_stride and half-pel MV.
    let field_stride = ref_stride * 2;
    let field_base_offset = if which_ref_field { ref_stride } else { 0 };
    // Field plane height in rows.
    let field_h = ref_h / 2;

    // The mv_y_field_half is already in half-pel units within the FIELD
    // (so integer pel step is 2 — odd values are half-pel).
    crate::mc::predict_block(
        &ref_plane[field_base_offset.min(ref_plane.len())..],
        field_stride,
        ref_stride as i32,
        field_h as i32,
        blk_px,
        field_py,
        mv_x_half,
        mv_y_field_half,
        8,
        rounding,
        out,
        8,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::vop::{VideoObjectPlane, VopCodingType};
    use oxideav_core::bits::{BitReader, BitWriter};

    fn make_vop(av: bool) -> VideoObjectPlane {
        VideoObjectPlane {
            vop_coding_type: VopCodingType::I,
            modulo_time_base: 0,
            vop_time_increment: 0,
            vop_coded: true,
            rounding_type: false,
            intra_dc_vlc_thr: 0,
            vop_quant: 1,
            vop_fcode_forward: 1,
            vop_fcode_backward: 1,
            width: 16,
            height: 16,
            sprite_trajectory: None,
            interlaced: true,
            top_field_first: true,
            alternate_vertical_scan: av,
            brightness_change_factor: 0,
        }
    }

    #[test]
    fn intra_mb_reads_dct_type_only() {
        // dct_type = 1, then whatever.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1);
        bw.write_bits(0, 7); // filler
        let data = bw.finish();
        let mut br = BitReader::new(&data);
        let info = parse_interlaced_information(&mut br, MbClass::Intra, VopCodingType::I, false)
            .expect("parse");
        assert!(info.dct_type);
        assert!(!info.field_prediction);
    }

    #[test]
    fn inter_single_mv_no_cbp_no_dct_type() {
        // dct_type is skipped when cbp_nonzero=false and class is inter
        // single-MV. Only field_prediction is read.
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1); // field_prediction=0
        bw.write_bits(0, 7);
        let data = bw.finish();
        let mut br = BitReader::new(&data);
        let info =
            parse_interlaced_information(&mut br, MbClass::InterSingleMv, VopCodingType::P, false)
                .expect("parse");
        assert!(!info.dct_type);
        assert!(!info.field_prediction);
    }

    #[test]
    fn inter_single_mv_with_cbp_and_field_pred() {
        // dct_type=1, field_prediction=1, forward_top=0, forward_bot=1.
        let mut bw = BitWriter::new();
        bw.write_bits(1, 1); // dct_type=1
        bw.write_bits(1, 1); // field_prediction=1
        bw.write_bits(0, 1); // forward_top=0
        bw.write_bits(1, 1); // forward_bot=1
        let data = bw.finish();
        let mut br = BitReader::new(&data);
        let info =
            parse_interlaced_information(&mut br, MbClass::InterSingleMv, VopCodingType::P, true)
                .expect("parse");
        assert!(info.dct_type);
        assert!(info.field_prediction);
        assert!(!info.forward_top_field_ref);
        assert!(info.forward_bottom_field_ref);
    }

    #[test]
    fn alt_vertical_scan_overrides_direction() {
        let vop = make_vop(true);
        // With alt-vertical scan on, we always return ALTERNATE_VERTICAL_SCAN.
        let scan = choose_scan_interlaced(&vop, true, Some(crate::block::PredDir::Top));
        assert_eq!(scan[0], ALTERNATE_VERTICAL_SCAN[0]);
    }

    #[test]
    fn no_alt_vertical_scan_falls_back_to_zigzag() {
        let vop = make_vop(false);
        let scan = choose_scan_interlaced(&vop, false, None);
        assert_eq!(scan[1], ZIGZAG[1]);
    }

    #[test]
    fn field_predict_luma_mb_interleaves_fields() {
        // Build a 32x32 reference where even rows are all 50 and odd rows
        // are all 200 — i.e. "top field" = 50-valued plane, "bottom field"
        // = 200-valued plane.
        let mut reference = vec![0u8; 32 * 32];
        for r in 0..32 {
            let v = if r & 1 == 0 { 50 } else { 200 };
            for c in 0..32 {
                reference[r * 32 + c] = v;
            }
        }
        let mut out = [0u8; 256];
        // Zero MVs; top-field-ref=0 (top field of ref = even rows = 50),
        // bot-field-ref=1 (bottom field of ref = odd rows = 200).
        field_predict_luma_mb(
            &reference,
            32,
            32,
            0,
            0,
            (0, 0),
            (0, 0),
            false,
            true,
            false,
            &mut out,
        );
        // Expected MB: even rows filled with top-field sample (50),
        // odd rows filled with bottom-field sample (200).
        for r in 0..16 {
            let expect = if r & 1 == 0 { 50 } else { 200 };
            for c in 0..16 {
                assert_eq!(
                    out[r * 16 + c],
                    expect,
                    "row {r} col {c}: got {} expected {expect}",
                    out[r * 16 + c]
                );
            }
        }
    }

    #[test]
    fn field_predict_chroma_mb_interleaves_fields() {
        // 16x16 reference chroma plane, even rows = 100, odd rows = 220.
        let mut reference = vec![0u8; 16 * 16];
        for r in 0..16 {
            let v = if r & 1 == 0 { 100 } else { 220 };
            for c in 0..16 {
                reference[r * 16 + c] = v;
            }
        }
        let mut out = [0u8; 64];
        field_predict_chroma_mb(
            &reference,
            16,
            16,
            0,
            0,
            (0, 0),
            (0, 0),
            false,
            true,
            false,
            &mut out,
        );
        for r in 0..8 {
            let expect = if r & 1 == 0 { 100 } else { 220 };
            for c in 0..8 {
                assert_eq!(out[r * 8 + c], expect);
            }
        }
    }

    #[test]
    fn field_dct_reorder_mb_layout() {
        // Fill blocks with recognisable values so we can verify where they
        // land in the output.
        let mut tl = [0i32; 64];
        let mut tr = [0i32; 64];
        let mut bl = [0i32; 64];
        let mut br_ = [0i32; 64];
        for i in 0..64 {
            tl[i] = 100 + i as i32;
            tr[i] = 200 + i as i32;
            bl[i] = 300 + i as i32;
            br_[i] = 400 + i as i32;
        }
        let mut out = [0i32; 256];
        field_dct_reorder_mb(&tl, &tr, &bl, &br_, &mut out);
        // Frame row 0 should be tl row 0 + tr row 0.
        assert_eq!(out[0], tl[0]);
        assert_eq!(out[8], tr[0]);
        // Frame row 1 should be bl row 0 + br_ row 0.
        assert_eq!(out[16], bl[0]);
        assert_eq!(out[24], br_[0]);
        // Frame row 14 should be tl row 7 (top field index 7).
        assert_eq!(out[14 * 16], tl[7 * 8]);
        // Frame row 15 should be bl row 7.
        assert_eq!(out[15 * 16], bl[7 * 8]);
    }
}

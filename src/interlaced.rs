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

use crate::headers::vol::{
    ALTERNATE_HORIZONTAL_SCAN, ALTERNATE_VERTICAL_SCAN, ZIGZAG,
};
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
        let info = parse_interlaced_information(
            &mut br,
            MbClass::InterSingleMv,
            VopCodingType::P,
            false,
        )
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
        let info = parse_interlaced_information(
            &mut br,
            MbClass::InterSingleMv,
            VopCodingType::P,
            true,
        )
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

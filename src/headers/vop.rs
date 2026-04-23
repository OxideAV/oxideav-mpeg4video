//! Video Object Plane header (ISO/IEC 14496-2 §6.2.5).

use oxideav_core::{Error, Result};

use crate::bits_ext::BitReaderExt;
use crate::gmc::{decode_sprite_trajectory, SpriteTrajectory};
use crate::headers::vol::{ShapeType, VideoObjectLayer};
use oxideav_core::bits::BitReader;

/// `vop_coding_type` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VopCodingType {
    /// I-VOP — intra-only.
    I,
    /// P-VOP — predicted from the previous reference.
    P,
    /// B-VOP — bidirectional.
    B,
    /// S-VOP — sprite VOP.
    S,
}

/// Parsed VOP header fields relevant to this decoder.
#[derive(Clone, Debug)]
pub struct VideoObjectPlane {
    pub vop_coding_type: VopCodingType,
    pub modulo_time_base: u32,
    pub vop_time_increment: u32,
    pub vop_coded: bool,
    pub rounding_type: bool,
    pub intra_dc_vlc_thr: u8,
    pub vop_quant: u32,
    pub vop_fcode_forward: u8,
    pub vop_fcode_backward: u8,
    /// Rectangle of the VOP — currently we only decode full frames. If the
    /// bitstream uses non-zero horizontal/vertical spatial reference the
    /// caller still receives the values here.
    pub width: u32,
    pub height: u32,
    /// GMC sprite trajectory — present only when the VOL advertises
    /// `sprite_enable == 2` (GMC) and this VOP is a P-VOP with
    /// `no_of_sprite_warping_points > 0`. `None` otherwise.
    pub sprite_trajectory: Option<SpriteTrajectory>,
    /// VOP-level interlaced flag (§6.2.5). Only present when the VOL
    /// advertises `interlaced == 1`. When set, the VOP may contain
    /// interlaced (field-coded) macroblocks alongside progressive ones.
    pub interlaced: bool,
    /// `top_field_first` — only meaningful when `interlaced == true`.
    /// 1 = the top field is displayed first, 0 = bottom field first.
    pub top_field_first: bool,
    /// `alternate_vertical_scan_flag` — only meaningful when
    /// `interlaced == true`. When set, interlaced intra blocks use the
    /// alternate-vertical scan (Table 7-3c) instead of zigzag.
    pub alternate_vertical_scan: bool,
    /// Brightness-change factor decoded when the VOL advertises
    /// `sprite_brightness_change == 1` AND this VOP is an S-VOP. The
    /// value is in 1/100ths; sample reconstruction multiplies by
    /// `(factor * 0.01 + 1)`. 0 when absent.
    pub brightness_change_factor: i32,
}

/// Parse the VOP header that follows a 0x000001B6 start code.
///
/// `vol` provides the per-sequence context needed to decode variable-length
/// fields (`vop_time_increment_bits`, shape type, etc.).
pub fn parse_vop(br: &mut BitReader<'_>, vol: &VideoObjectLayer) -> Result<VideoObjectPlane> {
    let vop_coding_type = match br.read_u32(2)? {
        0 => VopCodingType::I,
        1 => VopCodingType::P,
        2 => VopCodingType::B,
        3 => VopCodingType::S,
        _ => unreachable!(),
    };

    // modulo_time_base — 1-bits terminated by a 0-bit.
    let mut modulo_time_base = 0u32;
    loop {
        let b = br.read_u1()?;
        if b == 0 {
            break;
        }
        modulo_time_base += 1;
        if modulo_time_base > 60 {
            return Err(Error::invalid("mpeg4 VOP: modulo_time_base runaway"));
        }
    }

    br.read_marker()?;
    let vop_time_increment = br.read_u32(vol.vop_time_increment_bits)?;
    br.read_marker()?;

    let vop_coded = br.read_u1()? == 1;
    if !vop_coded {
        return Ok(VideoObjectPlane {
            vop_coding_type,
            modulo_time_base,
            vop_time_increment,
            vop_coded,
            rounding_type: false,
            intra_dc_vlc_thr: 0,
            vop_quant: 0,
            vop_fcode_forward: 0,
            vop_fcode_backward: 0,
            width: vol.width,
            height: vol.height,
            sprite_trajectory: None,
            interlaced: false,
            top_field_first: false,
            alternate_vertical_scan: false,
            brightness_change_factor: 0,
        });
    }

    // Only rectangular shape reaches this point (checked in VOL parser).
    debug_assert_eq!(vol.shape, ShapeType::Rectangular);

    // vop_rounding_type — present only for P-VOPs (§6.2.5).
    let rounding_type = if vop_coding_type == VopCodingType::P {
        br.read_u1()? == 1
    } else {
        false
    };

    // reduced_resolution_vop flag (rarely used; skip if enabled).
    if vol.reduced_resolution_vop_enable
        && matches!(vop_coding_type, VopCodingType::I | VopCodingType::P)
    {
        let _reduced = br.read_u1()?;
    }

    // intra_dc_vlc_thr — 3 bits.
    let intra_dc_vlc_thr = br.read_u32(3)? as u8;

    // VOP-level interlaced fields (§6.2.5). Present only when the VOL
    // advertises `interlaced == 1`.
    let (interlaced, top_field_first, alternate_vertical_scan) = if vol.interlaced {
        let il = br.read_u1()? == 1;
        if il {
            let tff = br.read_u1()? == 1;
            let avs = br.read_u1()? == 1;
            (il, tff, avs)
        } else {
            (false, false, false)
        }
    } else {
        (false, false, false)
    };

    // vop_quant — quant_precision bits (default 5).
    let vop_quant = br.read_u32(vol.quant_precision as u32)?;
    if vop_quant == 0 {
        return Err(Error::invalid("mpeg4 VOP: vop_quant = 0"));
    }

    let mut fcode_fwd = 0u8;
    let mut fcode_bwd = 0u8;
    if matches!(vop_coding_type, VopCodingType::P | VopCodingType::S) {
        fcode_fwd = br.read_u32(3)? as u8;
    }
    if vop_coding_type == VopCodingType::B {
        fcode_fwd = br.read_u32(3)? as u8;
        fcode_bwd = br.read_u32(3)? as u8;
    }

    // GMC trajectory (§7.7.4 amendment). When the VOL uses
    // `sprite_enable == 2` and the current VOP is a P-VOP, the next
    // `no_of_sprite_warping_points` trajectory pairs follow the fcode
    // field and precede the MB-layer. S-VOPs carry the same trajectory
    // but we don't decode S-VOPs yet.
    let sprite_trajectory =
        if vol.sprite_enable == 2 && vop_coding_type == VopCodingType::P
            && vol.no_of_sprite_warping_points > 0
        {
            Some(decode_sprite_trajectory(br, vol)?)
        } else {
            None
        };

    Ok(VideoObjectPlane {
        vop_coding_type,
        modulo_time_base,
        vop_time_increment,
        vop_coded,
        rounding_type,
        intra_dc_vlc_thr,
        vop_quant,
        vop_fcode_forward: fcode_fwd,
        vop_fcode_backward: fcode_bwd,
        width: vol.width,
        height: vol.height,
        sprite_trajectory,
        interlaced,
        top_field_first,
        alternate_vertical_scan,
        brightness_change_factor: 0,
    })
}

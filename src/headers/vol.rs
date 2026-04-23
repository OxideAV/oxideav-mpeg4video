//! Video Object Layer header (ISO/IEC 14496-2 §6.2.3).
//!
//! The VOL carries the per-sequence picture geometry, frame rate, quantisation
//! type selector and shape information. Populating `CodecParameters` requires a
//! successfully parsed VOL.

use oxideav_core::{Error, Result};

use crate::bits_ext::BitReaderExt;
use oxideav_core::bits::BitReader;

/// `aspect_ratio_info` — Table 6-12.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AspectRatioInfo {
    Square,
    Par12_11,
    Par10_11,
    Par16_11,
    Par40_33,
    Extended { par_width: u8, par_height: u8 },
    Reserved(u8),
}

/// `video_object_layer_shape` — Table 6-14.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType {
    Rectangular = 0,
    Binary = 1,
    BinaryOnly = 2,
    GrayScale = 3,
}

/// `chroma_format` — Table 6-15.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    Yuv420,
    Reserved(u8),
}

/// Static-sprite canvas rectangle (Table 6-7). Present in the VOL only
/// when `sprite_enable == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpriteRect {
    /// Sprite canvas width in luma pels.
    pub width: u32,
    /// Sprite canvas height in luma pels.
    pub height: u32,
    /// Sprite canvas left coordinate (signed 13-bit).
    pub left: i32,
    /// Sprite canvas top coordinate (signed 13-bit).
    pub top: i32,
}

/// Parsed Video Object Layer header. Only the subset of fields this decoder
/// consumes or exposes is stored — unparsed optional sub-blocks (scalability,
/// sprite VOPs, complexity-estimation) are read-then-discarded.
#[derive(Clone, Debug)]
pub struct VideoObjectLayer {
    pub random_accessible_vol: bool,
    pub video_object_type_indication: u8,
    pub is_object_layer_identifier: bool,
    pub verid: u8,
    pub priority: u8,
    pub aspect_ratio_info: AspectRatioInfo,
    /// VOL control parameters — bitrate / VBV etc. Not exposed.
    pub vol_control_parameters: bool,
    pub chroma_format: ChromaFormat,
    pub low_delay: bool,
    pub vbv_parameters_present: bool,
    pub shape: ShapeType,
    /// `vop_time_increment_resolution` — ticks per second, used to reconstruct
    /// VOP timestamps. Also serves as the rational numerator for frame_rate.
    pub vop_time_increment_resolution: u32,
    /// Number of bits used by `vop_time_increment` in each VOP header — the
    /// smallest integer that can hold (resolution - 1).
    pub vop_time_increment_bits: u32,
    /// If true the VOL has a fixed inter-VOP time increment (used as the
    /// denominator for frame_rate).
    pub fixed_vop_rate: bool,
    pub fixed_vop_time_increment: u32,
    /// Coded picture width (in luma pels).
    pub width: u32,
    /// Coded picture height (in luma pels).
    pub height: u32,
    pub interlaced: bool,
    pub obmc_disable: bool,
    /// `sprite_enable` — 1 bit when `verid == 1`, 2 bits otherwise.
    /// Canonical values (ISO 14496-2 + amendments):
    ///   0 = no sprite (default).
    ///   1 = static sprite (requires S-VOPs — not yet decoded).
    ///   2 = GMC (global motion compensation for P-VOPs).
    ///   3 = reserved.
    pub sprite_enable: u8,
    /// GMC / sprite trajectory: number of warp points (0..=4).
    /// 0 = identity, 1 = translation, 2/3 = affine, 4 = perspective.
    /// Present only when `sprite_enable` is 1 or 2.
    pub no_of_sprite_warping_points: u8,
    /// `sprite_warping_accuracy` (Table 6-8): 0 = 1/2-pel, 1 = 1/4-pel,
    /// 2 = 1/8-pel, 3 = 1/16-pel. Present only when `sprite_enable` is
    /// 1 or 2.
    pub sprite_warping_accuracy: u8,
    /// `sprite_brightness_change`. Rejected by the decoder when true —
    /// we parse but do not apply the brightness delta (trivially rare
    /// in practice; every GMC bitstream we've seen keeps it cleared).
    pub sprite_brightness_change: bool,
    /// `low_latency_sprite_enable`. Only meaningful for static-sprite
    /// mode (`sprite_enable == 1`); GMC uses regular P-VOPs.
    pub low_latency_sprite_enable: bool,
    /// Static-sprite rectangle — `(width, height, left, top)` of the
    /// sprite canvas as advertised by the VOL when `sprite_enable == 1`.
    /// All four values are 13-bit unsigned / signed integers per Table
    /// 6-7. `None` for `sprite_enable != 1`.
    pub sprite_rect: Option<SpriteRect>,
    pub not_8_bit: bool,
    pub quant_precision: u8,
    pub bits_per_pixel: u8,
    /// Quantisation type: false = H.263 quant, true = MPEG-4 (matrix) quant.
    pub mpeg_quant: bool,
    pub intra_quant_matrix: Option<[u8; 64]>,
    pub non_intra_quant_matrix: Option<[u8; 64]>,
    pub quarter_sample: bool,
    pub complexity_estimation_disable: bool,
    pub resync_marker_disable: bool,
    pub data_partitioned: bool,
    pub reversible_vlc: bool,
    pub newpred_enable: bool,
    pub reduced_resolution_vop_enable: bool,
    pub scalability: bool,
}

/// Default MPEG-4 intra quant matrix — Table 7-1.
pub const DEFAULT_INTRA_QUANT_MATRIX: [u8; 64] = [
    8, 17, 18, 19, 21, 23, 25, 27, 17, 18, 19, 21, 23, 25, 27, 28, 20, 21, 22, 23, 24, 26, 28, 30,
    21, 22, 23, 24, 26, 28, 30, 32, 22, 23, 24, 26, 28, 30, 32, 35, 23, 24, 26, 28, 30, 32, 35, 38,
    25, 26, 28, 30, 32, 35, 38, 41, 27, 28, 30, 32, 35, 38, 41, 45,
];

/// Default MPEG-4 non-intra quant matrix — Table 7-2.
pub const DEFAULT_NON_INTRA_QUANT_MATRIX: [u8; 64] = [
    16, 17, 18, 19, 20, 21, 22, 23, 17, 18, 19, 20, 21, 22, 23, 24, 18, 19, 20, 21, 22, 23, 24, 25,
    19, 20, 21, 22, 23, 24, 26, 27, 20, 21, 22, 23, 25, 26, 27, 28, 21, 22, 23, 24, 26, 27, 28, 30,
    22, 23, 24, 26, 27, 28, 30, 31, 23, 24, 25, 27, 28, 30, 31, 33,
];

/// Default zig-zag scan order (§7.4.3.3 Table 7-3a).
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Alternate-horizontal scan (§7.4.3.3 Table 7-3b) — used when AC prediction
/// direction is "top" (a horizontal DC gradient — the block is best described
/// row-first).
pub const ALTERNATE_HORIZONTAL_SCAN: [usize; 64] = [
    0, 1, 2, 3, 8, 9, 16, 17, 10, 11, 4, 5, 6, 7, 15, 14, 13, 12, 19, 18, 24, 25, 32, 33, 26, 27,
    20, 21, 22, 23, 28, 29, 30, 31, 34, 35, 40, 41, 48, 49, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
    50, 51, 56, 57, 58, 59, 52, 53, 54, 55, 60, 61, 62, 63,
];

/// Alternate-vertical scan (§7.4.3.3 Table 7-3c) — used when AC prediction
/// direction is "left" (a vertical DC gradient).
pub const ALTERNATE_VERTICAL_SCAN: [usize; 64] = [
    0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3, 11, 4, 12, 19,
    27, 34, 42, 50, 58, 35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29, 36, 44, 52, 60, 37, 45, 53,
    61, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63,
];

fn bits_needed(max_value: u32) -> u32 {
    // Number of bits required to represent `max_value`. Minimum 1 bit per spec.
    if max_value == 0 {
        return 1;
    }
    32 - max_value.leading_zeros()
}

/// Read a signed 13-bit two's-complement integer from the bitstream.
/// Used for `sprite_left` / `sprite_top` (Table 6-7).
fn read_signed_13(br: &mut BitReader<'_>) -> Result<i32> {
    let raw = br.read_u32(13)? as i32;
    // Sign-extend: bit 12 is the sign bit.
    Ok(if raw & (1 << 12) != 0 {
        raw - (1 << 13)
    } else {
        raw
    })
}

fn parse_aspect_ratio(br: &mut BitReader<'_>) -> Result<AspectRatioInfo> {
    let code = br.read_u32(4)? as u8;
    Ok(match code {
        1 => AspectRatioInfo::Square,
        2 => AspectRatioInfo::Par12_11,
        3 => AspectRatioInfo::Par10_11,
        4 => AspectRatioInfo::Par16_11,
        5 => AspectRatioInfo::Par40_33,
        0xF => {
            let par_width = br.read_u32(8)? as u8;
            let par_height = br.read_u32(8)? as u8;
            AspectRatioInfo::Extended {
                par_width,
                par_height,
            }
        }
        other => AspectRatioInfo::Reserved(other),
    })
}

/// Parse a VOL header payload. `br` must be positioned just past the
/// `0x000001[20..2F]` start code.
///
/// This implementation covers rectangular-shape ASP bitstreams — the dominant
/// case for XVID/DivX/FMP4. Scalable / binary-shape / sprite-only paths return
/// `Error::Unsupported`.
pub fn parse_vol(br: &mut BitReader<'_>) -> Result<VideoObjectLayer> {
    let random_accessible_vol = br.read_u1()? == 1;
    let video_object_type_indication = br.read_u32(8)? as u8;

    let is_object_layer_identifier = br.read_u1()? == 1;
    let mut verid = 1u8;
    let mut priority = 0u8;
    if is_object_layer_identifier {
        verid = br.read_u32(4)? as u8;
        priority = br.read_u32(3)? as u8;
    }

    let aspect_ratio_info = parse_aspect_ratio(br)?;

    let vol_control_parameters = br.read_u1()? == 1;
    let mut chroma_format = ChromaFormat::Yuv420;
    let mut low_delay = false;
    let mut vbv_parameters_present = false;
    if vol_control_parameters {
        let cf = br.read_u32(2)? as u8;
        chroma_format = match cf {
            1 => ChromaFormat::Yuv420,
            other => ChromaFormat::Reserved(other),
        };
        low_delay = br.read_u1()? == 1;
        let vbv_parameters = br.read_u1()? == 1;
        if vbv_parameters {
            vbv_parameters_present = true;
            // first_half_bit_rate (15) + marker + latter_half_bit_rate (15) + marker
            br.skip(15)?;
            br.read_marker()?;
            br.skip(15)?;
            br.read_marker()?;
            // first_half_vbv_buffer_size (15) + marker + latter (3)
            br.skip(15)?;
            br.read_marker()?;
            br.skip(3)?;
            // first_half_vbv_occupancy (11) + marker + latter (15) + marker
            br.skip(11)?;
            br.read_marker()?;
            br.skip(15)?;
            br.read_marker()?;
        }
    }

    let shape_code = br.read_u32(2)? as u8;
    let shape = match shape_code {
        0 => ShapeType::Rectangular,
        1 => ShapeType::Binary,
        2 => ShapeType::BinaryOnly,
        3 => ShapeType::GrayScale,
        _ => unreachable!(),
    };
    if shape != ShapeType::Rectangular {
        return Err(Error::unsupported(
            "mpeg4 non-rectangular shape VOL: follow-up",
        ));
    }

    br.read_marker()?; // marker
    let vop_time_increment_resolution = br.read_u32(16)?;
    br.read_marker()?; // marker
    if vop_time_increment_resolution == 0 {
        return Err(Error::invalid(
            "mpeg4 VOL: vop_time_increment_resolution = 0",
        ));
    }
    let vop_time_increment_bits = bits_needed(vop_time_increment_resolution - 1).max(1);

    let fixed_vop_rate = br.read_u1()? == 1;
    let mut fixed_vop_time_increment = 1u32;
    if fixed_vop_rate {
        fixed_vop_time_increment = br.read_u32(vop_time_increment_bits)?;
        if fixed_vop_time_increment == 0 {
            fixed_vop_time_increment = 1;
        }
    }

    // Only for Rectangular shape: width/height.
    br.read_marker()?;
    let width = br.read_u32(13)?;
    br.read_marker()?;
    let height = br.read_u32(13)?;
    br.read_marker()?;

    let interlaced = br.read_u1()? == 1;
    let obmc_disable = br.read_u1()? == 1;

    // sprite_enable — 1 bit in verid<=1, 2 bits in verid>=2 per spec corrigendum.
    let sprite_enable = if verid == 1 {
        br.read_u32(1)? as u8
    } else {
        br.read_u32(2)? as u8
    };
    // Sprite-specific VOL fields. Bitstream layout per ISO/IEC 14496-2
    // §6.2.3 + amendment 1 (GMC):
    //   if (sprite_enable == 1 || sprite_enable == 2) {
    //       if (sprite_enable == 1) {  /* static sprite rectangle */
    //           sprite_width(13), marker, sprite_height(13), marker,
    //           sprite_left(13), marker, sprite_top(13), marker;
    //       }
    //       no_of_sprite_warping_points(6);
    //       sprite_warping_accuracy(2);
    //       sprite_brightness_change(1);
    //       if (sprite_enable == 1) low_latency_sprite_enable(1);
    //   }
    let mut no_of_sprite_warping_points = 0u8;
    let mut sprite_warping_accuracy = 0u8;
    let mut sprite_brightness_change = false;
    let mut low_latency_sprite_enable = false;
    let mut sprite_rect = None;
    match sprite_enable {
        0 => { /* no sprite */ }
        1 => {
            // Static sprite (§6.2.3 + Table 6-7). Parse the canvas
            // rectangle + trajectory/accuracy/brightness fields so the
            // VOL structure is fully exposed to consumers. The S-VOP
            // decode path itself is rejected downstream.
            let sprite_width = br.read_u32(13)?;
            br.read_marker()?;
            let sprite_height = br.read_u32(13)?;
            br.read_marker()?;
            let sprite_left = read_signed_13(br)?;
            br.read_marker()?;
            let sprite_top = read_signed_13(br)?;
            br.read_marker()?;
            sprite_rect = Some(SpriteRect {
                width: sprite_width,
                height: sprite_height,
                left: sprite_left,
                top: sprite_top,
            });
            no_of_sprite_warping_points = br.read_u32(6)? as u8;
            if no_of_sprite_warping_points > 4 {
                return Err(Error::invalid(format!(
                    "mpeg4 static-sprite VOL: no_of_sprite_warping_points={no_of_sprite_warping_points} > 4"
                )));
            }
            sprite_warping_accuracy = br.read_u32(2)? as u8;
            sprite_brightness_change = br.read_u1()? == 1;
            low_latency_sprite_enable = br.read_u1()? == 1;
        }
        2 => {
            // GMC-enabled — VOL carries only the trajectory-related
            // fields; no static-sprite rectangle. Validated: values
            // > 4 for the warp count are invalid per Table 6-7.
            no_of_sprite_warping_points = br.read_u32(6)? as u8;
            if no_of_sprite_warping_points > 4 {
                return Err(Error::invalid(format!(
                    "mpeg4 GMC VOL: no_of_sprite_warping_points={no_of_sprite_warping_points} > 4"
                )));
            }
            sprite_warping_accuracy = br.read_u32(2)? as u8;
            sprite_brightness_change = br.read_u1()? == 1;
            // `sprite_brightness_change == 1` is now permitted. When a
            // GMC P-VOP carries this flag the VOP header includes a
            // `brightness_change_factor()` VLC (§6.2.5.6 / Table 11-33);
            // the factor is applied during warp-sample reconstruction.
        }
        _ => {
            return Err(Error::invalid(format!(
                "mpeg4 VOL: reserved sprite_enable={sprite_enable}"
            )));
        }
    }

    let not_8_bit = br.read_u1()? == 1;
    let (quant_precision, bits_per_pixel) = if not_8_bit {
        (br.read_u32(4)? as u8, br.read_u32(4)? as u8)
    } else {
        (5u8, 8u8)
    };

    let mpeg_quant = br.read_u1()? == 1;
    let mut intra_q: Option<[u8; 64]> = None;
    let mut non_intra_q: Option<[u8; 64]> = None;
    if mpeg_quant {
        if br.read_u1()? == 1 {
            intra_q = Some(read_quant_matrix(br)?);
        }
        if br.read_u1()? == 1 {
            non_intra_q = Some(read_quant_matrix(br)?);
        }
    }

    // quarter_sample was added in verid>=2 (MPEG-4 Part 2 / 2000 corrigendum).
    // verid==1 streams never carry this bit.
    let quarter_sample = if verid != 1 {
        br.read_u1()? == 1
    } else {
        false
    };

    let complexity_estimation_disable = br.read_u1()? == 1;
    if !complexity_estimation_disable {
        return Err(Error::unsupported(
            "mpeg4 complexity_estimation_header: follow-up",
        ));
    }

    let resync_marker_disable = br.read_u1()? == 1;
    let data_partitioned = br.read_u1()? == 1;
    let mut reversible_vlc = false;
    if data_partitioned {
        reversible_vlc = br.read_u1()? == 1;
    }

    // newpred_enable and reduced_resolution_vop_enable were added in verid>=2.
    let (newpred_enable, reduced_resolution_vop_enable) = if verid != 1 {
        let np = br.read_u1()? == 1;
        if np {
            return Err(Error::unsupported("mpeg4 newpred_enable: follow-up"));
        }
        let rr = br.read_u1()? == 1;
        (np, rr)
    } else {
        (false, false)
    };

    let scalability = br.read_u1()? == 1;
    if scalability {
        return Err(Error::unsupported("mpeg4 scalable VOL: follow-up"));
    }

    Ok(VideoObjectLayer {
        random_accessible_vol,
        video_object_type_indication,
        is_object_layer_identifier,
        verid,
        priority,
        aspect_ratio_info,
        vol_control_parameters,
        chroma_format,
        low_delay,
        vbv_parameters_present,
        shape,
        vop_time_increment_resolution,
        vop_time_increment_bits,
        fixed_vop_rate,
        fixed_vop_time_increment,
        width,
        height,
        interlaced,
        obmc_disable,
        sprite_enable,
        no_of_sprite_warping_points,
        sprite_warping_accuracy,
        sprite_brightness_change,
        low_latency_sprite_enable,
        sprite_rect,
        not_8_bit,
        quant_precision,
        bits_per_pixel,
        mpeg_quant,
        intra_quant_matrix: intra_q,
        non_intra_quant_matrix: non_intra_q,
        quarter_sample,
        complexity_estimation_disable,
        resync_marker_disable,
        data_partitioned,
        reversible_vlc,
        newpred_enable,
        reduced_resolution_vop_enable,
        scalability,
    })
}

/// Parse a custom quant matrix (§6.2.3). 1..=64 signed-8-bit values,
/// terminated by a zero value; remaining entries default.
fn read_quant_matrix(br: &mut BitReader<'_>) -> Result<[u8; 64]> {
    let mut m = [0u8; 64];
    let mut last_nonzero = 8u8;
    for i in 0..64 {
        let v = br.read_u32(8)? as u8;
        if v == 0 {
            // Fill remainder with the last non-zero value per §6.2.3.
            for j in i..64 {
                m[ZIGZAG[j]] = last_nonzero;
            }
            return Ok(m);
        }
        m[ZIGZAG[i]] = v;
        last_nonzero = v;
    }
    Ok(m)
}

impl VideoObjectLayer {
    /// Number of macroblocks horizontally (16 pels each), rounded up.
    pub fn mb_width(&self) -> u32 {
        self.width.div_ceil(16)
    }
    /// Number of macroblocks vertically (16 pels each), rounded up.
    pub fn mb_height(&self) -> u32 {
        self.height.div_ceil(16)
    }

    /// Rational frame rate as (num, den) in seconds. Returns (resolution, 1)
    /// if no fixed rate is advertised — this is the best the VOL header gives
    /// us in that case.
    pub fn frame_rate(&self) -> (i64, i64) {
        if self.fixed_vop_rate && self.fixed_vop_time_increment != 0 {
            (
                self.vop_time_increment_resolution as i64,
                self.fixed_vop_time_increment as i64,
            )
        } else {
            (self.vop_time_increment_resolution as i64, 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    #[test]
    fn bits_needed_basic() {
        assert_eq!(bits_needed(0), 1);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(9), 4);
        assert_eq!(bits_needed(15), 4);
        assert_eq!(bits_needed(16), 5);
    }

    /// Helper: build a minimal VOL payload with the given sprite_enable
    /// and verid so we can exercise the GMC-path branches.
    fn build_vol_bytes(sprite_enable: u8, verid: u8) -> Vec<u8> {
        build_vol_bytes_ex(sprite_enable, verid, false, false)
    }

    /// Extended helper: additionally toggles `sprite_brightness_change`
    /// and VOL-level `interlaced` so we can exercise the new parse paths.
    fn build_vol_bytes_ex(
        sprite_enable: u8,
        verid: u8,
        brightness: bool,
        interlaced: bool,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1); // random_accessible_vol
        bw.write_bits(1, 8); // video_object_type_indication = 1
        bw.write_bits(1, 1); // is_object_layer_identifier
        bw.write_bits(verid as u32, 4); // verid
        bw.write_bits(0, 3); // priority
        bw.write_bits(1, 4); // aspect_ratio_info = square
        bw.write_bits(0, 1); // vol_control_parameters
        bw.write_bits(0, 2); // shape = rectangular
        bw.write_bits(1, 1); // marker
        bw.write_bits(30, 16); // vop_time_increment_resolution
        bw.write_bits(1, 1); // marker
        bw.write_bits(0, 1); // fixed_vop_rate
        bw.write_bits(1, 1); // marker
        bw.write_bits(32, 13); // width
        bw.write_bits(1, 1); // marker
        bw.write_bits(32, 13); // height
        bw.write_bits(1, 1); // marker
        bw.write_bits(if interlaced { 1 } else { 0 }, 1); // interlaced
        bw.write_bits(1, 1); // obmc_disable
        // sprite_enable — 1 bit in verid==1, 2 bits otherwise.
        if verid == 1 {
            bw.write_bits(sprite_enable as u32, 1);
        } else {
            bw.write_bits(sprite_enable as u32, 2);
        }
        if sprite_enable == 2 {
            bw.write_bits(2, 6); // no_of_sprite_warping_points = 2
            bw.write_bits(1, 2); // sprite_warping_accuracy = 1/4-pel
            bw.write_bits(if brightness { 1 } else { 0 }, 1); // sprite_brightness_change
        }
        bw.write_bits(0, 1); // not_8_bit
        bw.write_bits(0, 1); // mpeg_quant
        if verid != 1 {
            bw.write_bits(0, 1); // quarter_sample
        }
        bw.write_bits(1, 1); // complexity_estimation_disable
        bw.write_bits(1, 1); // resync_marker_disable
        bw.write_bits(0, 1); // data_partitioned
        if verid != 1 {
            bw.write_bits(0, 1); // newpred_enable
            bw.write_bits(0, 1); // reduced_resolution_vop_enable
        }
        bw.write_bits(0, 1); // scalability
        bw.finish()
    }

    #[test]
    fn vol_parse_accepts_sprite_enable_1() {
        // sprite_enable = 1 = static sprite. The VOL parser now
        // consumes the full sprite-rectangle + warping fields so
        // downstream consumers see the whole VOL structure; actual
        // S-VOP decode is still rejected at the decoder layer.
        let bytes = build_vol_bytes_static_sprite(1, 2);
        let mut br = BitReader::new(&bytes);
        let vol = parse_vol(&mut br).expect("static-sprite VOL must parse");
        assert_eq!(vol.sprite_enable, 1);
        let rect = vol.sprite_rect.expect("sprite_rect populated");
        // Test helper writes width=256, height=128, left=0, top=0.
        assert_eq!(rect.width, 256);
        assert_eq!(rect.height, 128);
        assert_eq!(rect.left, 0);
        assert_eq!(rect.top, 0);
        assert_eq!(vol.no_of_sprite_warping_points, 1);
        assert!(vol.low_latency_sprite_enable);
    }

    /// Build a VOL with `sprite_enable = 1` and a static-sprite
    /// rectangle. Separate from `build_vol_bytes_ex` since the sprite
    /// payload differs between sprite_enable=1 (rectangle + warping +
    /// brightness + low_latency) and sprite_enable=2 (warping +
    /// brightness only).
    fn build_vol_bytes_static_sprite(warping_points: u8, verid: u8) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1); // random_accessible_vol
        bw.write_bits(1, 8); // video_object_type_indication
        bw.write_bits(1, 1); // is_object_layer_identifier
        bw.write_bits(verid as u32, 4);
        bw.write_bits(0, 3); // priority
        bw.write_bits(1, 4); // aspect_ratio_info = square
        bw.write_bits(0, 1); // vol_control_parameters
        bw.write_bits(0, 2); // shape = rectangular
        bw.write_bits(1, 1); // marker
        bw.write_bits(30, 16); // vop_time_increment_resolution
        bw.write_bits(1, 1); // marker
        bw.write_bits(0, 1); // fixed_vop_rate
        bw.write_bits(1, 1); // marker
        bw.write_bits(32, 13); // width
        bw.write_bits(1, 1); // marker
        bw.write_bits(32, 13); // height
        bw.write_bits(1, 1); // marker
        bw.write_bits(0, 1); // interlaced
        bw.write_bits(1, 1); // obmc_disable
        // sprite_enable — 2 bits (verid != 1). Value 1 = static sprite.
        bw.write_bits(1, 2);
        // Static-sprite rectangle: width, height, left (signed), top (signed).
        bw.write_bits(256, 13);
        bw.write_bits(1, 1);
        bw.write_bits(128, 13);
        bw.write_bits(1, 1);
        bw.write_bits(0, 13);
        bw.write_bits(1, 1);
        bw.write_bits(0, 13);
        bw.write_bits(1, 1);
        bw.write_bits(warping_points as u32, 6);
        bw.write_bits(1, 2); // sprite_warping_accuracy = 1/4-pel
        bw.write_bits(0, 1); // sprite_brightness_change
        bw.write_bits(1, 1); // low_latency_sprite_enable
        bw.write_bits(0, 1); // not_8_bit
        bw.write_bits(0, 1); // mpeg_quant
        if verid != 1 {
            bw.write_bits(0, 1); // quarter_sample
        }
        bw.write_bits(1, 1); // complexity_estimation_disable
        bw.write_bits(1, 1); // resync_marker_disable
        bw.write_bits(0, 1); // data_partitioned
        if verid != 1 {
            bw.write_bits(0, 1); // newpred_enable
            bw.write_bits(0, 1); // reduced_resolution_vop_enable
        }
        bw.write_bits(0, 1); // scalability
        bw.finish()
    }

    #[test]
    fn vol_parse_accepts_sprite_enable_2_gmc() {
        // sprite_enable = 2 = GMC. Parser must succeed and expose
        // the warping-point count + accuracy.
        let bytes = build_vol_bytes(2, 2);
        let mut br = BitReader::new(&bytes);
        let vol = parse_vol(&mut br).expect("GMC VOL must parse");
        assert_eq!(vol.sprite_enable, 2);
        assert_eq!(vol.no_of_sprite_warping_points, 2);
        assert_eq!(vol.sprite_warping_accuracy, 1);
        assert!(!vol.sprite_brightness_change);
    }

    #[test]
    fn vol_parse_accepts_sprite_brightness_change() {
        // Previously any sprite_enable=2 stream with sprite_brightness_change=1
        // was rejected as unsupported. After the brightness_change_factor
        // VLC landed, the VOL parser simply records the flag and leaves
        // application to the VOP-level / warp-reconstruction code.
        let bytes = build_vol_bytes_ex(2, 2, true, false);
        let mut br = BitReader::new(&bytes);
        let vol = parse_vol(&mut br).expect("sprite_brightness_change=1 must parse");
        assert!(vol.sprite_brightness_change);
        assert_eq!(vol.sprite_enable, 2);
    }

    #[test]
    fn vol_parse_accepts_interlaced_flag() {
        // VOL-level interlaced=1 must parse cleanly (decoder-level rejection
        // of field-coded MBs happens at the VOP layer, not the VOL).
        let bytes = build_vol_bytes_ex(0, 2, false, true);
        let mut br = BitReader::new(&bytes);
        let vol = parse_vol(&mut br).expect("interlaced VOL must parse");
        assert!(vol.interlaced);
    }

    #[test]
    fn vol_parse_rejects_reserved_sprite_enable_3() {
        // sprite_enable = 3 is reserved — parser must flag invalid.
        let bytes = build_vol_bytes(3, 2);
        let mut br = BitReader::new(&bytes);
        let err = parse_vol(&mut br).unwrap_err();
        assert!(err.to_string().contains("reserved sprite_enable=3"));
    }
}

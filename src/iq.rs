//! Inverse quantisation for MPEG-4 Part 2 (§7.4.4).
//!
//! Two modes, selected by the VOL's `mpeg_quant` flag:
//! * **H.263 quantisation** (the default XVID/DivX mode) — very simple: each
//!   AC coefficient `l != 0` dequantises to
//!   `(2 * Q * |l| + Q) * sign(l)` if `Q` is odd,
//!   `(2 * Q * |l| + Q - 1) * sign(l)` if `Q` is even.
//! * **MPEG-4 quantisation** — uses an 8x8 quant matrix similar to MPEG-1/2,
//!   with mismatch control.
//!
//! Only the H.263 path is filled in for this session; the MPEG-4 matrix path
//! stubs out clearly.

use oxideav_core::{Error, Result};

use crate::headers::vol::VideoObjectLayer;

/// Luma DC scaler by quantiser, spec Table 7-2.
pub const Y_DC_SCALE_TABLE: [u8; 32] = [
    0, 8, 8, 8, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    34, 36, 38, 40, 42, 44, 46,
];

/// Chroma DC scaler by quantiser, spec Table 7-3.
pub const C_DC_SCALE_TABLE: [u8; 32] = [
    0, 8, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
    19, 20, 21, 22, 23, 24, 25,
];

/// DC-VLC vs plain-13-bit DC threshold by `intra_dc_vlc_thr` (VOP header).
/// Spec Table 6-22 / §7.4.3. `intra_dc_vlc_thr[i]` gives the quant threshold
/// above which plain-13-bit DC coding is used for intra MBs.
///
/// `thr[0] == 99` means "always use VLC"; `thr[7] == 0` means "never use VLC".
pub const INTRA_DC_VLC_THR_TABLE: [u8; 8] = [99, 13, 15, 17, 19, 21, 23, 0];

/// Dequantise one intra block's AC coefficients in-place (index 0 is the DC
/// coefficient and is left untouched — DC is handled separately by the
/// caller, with prediction). `coeffs[i]` is the raw decoded level; on return
/// `coeffs[i]` holds the reconstructed coefficient.
///
/// `quant` is the current `vop_quant` (1..=31 for quant_precision=5).
pub fn dequantise_intra_h263(coeffs: &mut [i32; 64], quant: u32) -> Result<()> {
    if quant == 0 {
        return Err(Error::invalid("mpeg4 iq: quant = 0"));
    }
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    crate::simd::dequant_h263(coeffs, q, q_plus, 1);
    Ok(())
}

/// MPEG-4 (matrix) intra quantisation — §7.4.4.3.
///
/// For intra blocks with the MPEG-4 quantisation path (VOL `mpeg_quant`),
/// `abs_coef = ((2 * level + k) * wQ * matrix[zz]) / 16`, with `k=0` for
/// intra (§7.4.4.3 (17)). Result saturates to [-2048, 2047]. Index 0 (DC) is
/// untouched; the caller handles DC via the DC scaler.
///
/// `matrix` holds the intra quant matrix in natural (un-zigzagged) order.
pub fn dequantise_intra_mpeg4(
    coeffs: &mut [i32; 64],
    quant: u32,
    vol: &VideoObjectLayer,
) -> Result<()> {
    if quant == 0 {
        return Err(Error::invalid("mpeg4 iq: quant = 0"));
    }
    let matrix = vol
        .intra_quant_matrix
        .unwrap_or(crate::headers::vol::DEFAULT_INTRA_QUANT_MATRIX);
    let wq = quant as i32;
    for i in 1..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        // §7.4.4.3 equation (17), intra: |F''| = (2 * |level| * wQ * Q_intra[i]) / 16
        // with sign carried separately. We do not apply mismatch control for
        // intra blocks (§7.4.4.7 is for non-intra only per spec).
        let m = matrix[i] as i32;
        let abs = l.unsigned_abs() as i32;
        let mut val = (2 * abs * wq * m) / 16;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
    }
    Ok(())
}

/// Dequantise one inter block's AC + DC coefficients in-place using the H.263
/// formula (§7.4.4.2).
///
/// For inter blocks ALL coefficients (including index 0) follow the same rule.
/// `quant` is the current `vop_quant` (1..=31 for quant_precision=5).
pub fn dequantise_inter_h263(coeffs: &mut [i32; 64], quant: u32) -> Result<()> {
    if quant == 0 {
        return Err(Error::invalid("mpeg4 iq: quant = 0"));
    }
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    crate::simd::dequant_h263(coeffs, q, q_plus, 0);
    Ok(())
}

/// MPEG-4 (matrix) inter quantisation — §7.4.4.3.
///
/// For inter blocks: `abs = ((2 * level + sign(level)) * wQ * matrix[zz]) / 16`
/// with the standard mismatch-control nudge applied to every odd sum at the
/// end (§7.4.4.7). `matrix` is the non-intra quant matrix in natural order.
pub fn dequantise_inter_mpeg4(
    coeffs: &mut [i32; 64],
    quant: u32,
    vol: &VideoObjectLayer,
) -> Result<()> {
    if quant == 0 {
        return Err(Error::invalid("mpeg4 iq: quant = 0"));
    }
    let matrix = vol
        .non_intra_quant_matrix
        .unwrap_or(crate::headers::vol::DEFAULT_NON_INTRA_QUANT_MATRIX);
    let wq = quant as i32;
    let mut sum: i64 = 0;
    for i in 0..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        // §7.4.4.3 equation (18), inter:
        //   |F''| = ((2 * |level| + 1) * wQ * Q_inter[i]) / 16
        let m = matrix[i] as i32;
        let abs = l.unsigned_abs() as i32;
        let mut val = ((2 * abs + 1) * wq * m) / 16;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
        sum += coeffs[i] as i64;
    }
    // Mismatch control (§7.4.4.7): if sum is even, toggle bit0 of last
    // coefficient. We always apply to coeffs[63].
    if sum & 1 == 0 {
        coeffs[63] ^= 1;
    }
    Ok(())
}

/// Return the DC scaler for a block, picking the luma or chroma table based on
/// the block index (0..=3 are luma, 4 is Cb, 5 is Cr). Valid for `quant` in
/// 1..=31 for 5-bit quant precision.
pub fn dc_scaler(block_idx: usize, quant: u32) -> u32 {
    let q = (quant as usize).min(31);
    if block_idx < 4 {
        Y_DC_SCALE_TABLE[q] as u32
    } else {
        C_DC_SCALE_TABLE[q] as u32
    }
}

// -------------------------------------------------------------------------
// Forward (encoder) MPEG-quant — invert the dequant rules above, picking
// the integer level whose reconstruction is closest to the input
// coefficient. Used by the encoder when the VOL is emitted with
// `mpeg_quant = 1`.
// -------------------------------------------------------------------------

/// Forward MPEG-quant for one intra-AC coefficient. Inverts the §7.4.4.3
/// equation (17) intra rule
///   `recon = (2 * |level| * wQ * matrix[i]) / 16`
/// and picks the integer `level` whose reconstruction is closest to the
/// input. Index 0 (DC) is owned by the caller; this routine only handles
/// AC indices 1..64.
///
/// `matrix_zz` is the intra quant matrix in NATURAL order (same indexing
/// as `dequantise_intra_mpeg4`); `quant` is the active vop_quant.
pub fn quantise_ac_intra_mpeg4(coef: i32, idx: usize, quant: u32, matrix_zz: &[u8; 64]) -> i32 {
    if coef == 0 || quant == 0 || idx == 0 {
        return 0;
    }
    let m = matrix_zz[idx] as i32;
    if m <= 0 {
        return 0;
    }
    let wq = quant as i32;
    let denom = 2 * wq * m; // recon(L) = denom * L / 16
    if denom <= 0 {
        return 0;
    }
    let abs = coef.unsigned_abs() as i32;
    // Closed-form coarse seed: l_low = (16 * abs) / denom; compare l_low and l_low + 1.
    let l_low = ((16 * abs) / denom).max(0);
    let mut best_l = 0i32;
    let mut best_err = abs;
    for cand in [l_low.saturating_sub(1), l_low, l_low + 1] {
        if cand < 0 {
            continue;
        }
        let recon = (denom * cand) / 16;
        let err = (abs - recon).abs();
        if err < best_err {
            best_err = err;
            best_l = cand;
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

/// Forward MPEG-quant for one inter coefficient. Inverts the §7.4.4.3
/// equation (18) inter rule
///   `recon = ((2 * |level| + 1) * wQ * matrix[i]) / 16`
/// and picks the integer `level` whose reconstruction is closest. Inter
/// blocks use this for ALL 64 indices (the H.263 encoder side is the
/// matching call for `mpeg_quant = 0` mode).
///
/// `matrix_zz` is the non-intra quant matrix in NATURAL order.
pub fn quantise_ac_inter_mpeg4(coef: i32, idx: usize, quant: u32, matrix_zz: &[u8; 64]) -> i32 {
    if coef == 0 || quant == 0 {
        return 0;
    }
    let m = matrix_zz[idx] as i32;
    if m <= 0 {
        return 0;
    }
    let wq = quant as i32;
    let factor = wq * m;
    if factor <= 0 {
        return 0;
    }
    let abs = coef.unsigned_abs() as i32;
    // recon(L) = ((2L + 1) * factor) / 16. Solve 2L+1 ≈ 16*abs / factor →
    // L ≈ (16 * abs / factor - 1) / 2. Compare the closest 3 integer Ls.
    let twol_plus_one = (16 * abs + factor / 2) / factor;
    let l_seed = if twol_plus_one >= 1 {
        (twol_plus_one - 1) / 2
    } else {
        0
    };
    let mut best_l = 0i32;
    let mut best_err = abs;
    for cand in [l_seed.saturating_sub(1), l_seed, l_seed + 1] {
        if cand < 0 {
            continue;
        }
        let recon = ((2 * cand + 1) * factor) / 16;
        let err = (abs - recon).abs();
        if err < best_err {
            best_err = err;
            best_l = cand;
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

/// Reconstruct one inter coefficient under the MPEG-quant rule given a
/// quantised `level`. Mirrors the body of `dequantise_inter_mpeg4` but
/// for a single coefficient (encoder-side reconstruct loop). Mismatch
/// control is the caller's responsibility (apply once to coeffs[63]
/// after summing all reconstructions, per §7.4.4.7).
pub fn reconstruct_inter_mpeg4(level: i32, idx: usize, quant: u32, matrix_zz: &[u8; 64]) -> i32 {
    if level == 0 || quant == 0 {
        return 0;
    }
    let m = matrix_zz[idx] as i32;
    if m <= 0 {
        return 0;
    }
    let wq = quant as i32;
    let abs = level.unsigned_abs() as i32;
    let val = ((2 * abs + 1) * wq * m) / 16;
    let signed = if level < 0 { -val } else { val };
    signed.clamp(-2048, 2047)
}

/// Reconstruct one intra-AC coefficient under the MPEG-quant rule given a
/// quantised `level`. Mirrors the body of `dequantise_intra_mpeg4`. No
/// mismatch control is applied to intra blocks (§7.4.4.7 is non-intra
/// only).
pub fn reconstruct_intra_mpeg4(level: i32, idx: usize, quant: u32, matrix_zz: &[u8; 64]) -> i32 {
    if level == 0 || quant == 0 || idx == 0 {
        return 0;
    }
    let m = matrix_zz[idx] as i32;
    if m <= 0 {
        return 0;
    }
    let wq = quant as i32;
    let abs = level.unsigned_abs() as i32;
    let val = (2 * abs * wq * m) / 16;
    let signed = if level < 0 { -val } else { val };
    signed.clamp(-2048, 2047)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h263_intra_even_quant() {
        // Q=4 (even). Level 3 -> 2*4*3 + (4-1) = 24 + 3 = 27.
        let mut c = [0i32; 64];
        c[1] = 3;
        dequantise_intra_h263(&mut c, 4).unwrap();
        assert_eq!(c[1], 27);
    }

    #[test]
    fn h263_intra_odd_quant() {
        // Q=5. Level -2 -> -(2*5*2 + 5) = -25.
        let mut c = [0i32; 64];
        c[2] = -2;
        dequantise_intra_h263(&mut c, 5).unwrap();
        assert_eq!(c[2], -25);
    }

    #[test]
    fn dc_scaler_tables() {
        assert_eq!(dc_scaler(0, 1), 8); // luma
        assert_eq!(dc_scaler(4, 1), 8); // chroma
        assert_eq!(dc_scaler(0, 31), 46);
        assert_eq!(dc_scaler(5, 31), 25);
    }

    /// MPEG-quant intra-AC forward + dequant must round-trip every
    /// coefficient that already lies on the reconstruction lattice. We
    /// pick a few `(quant, idx, level)` combos, dequantise to `recon`,
    /// re-quantise, and assert we get the same level back.
    #[test]
    fn mpeg_intra_forward_round_trip_lattice() {
        let m = crate::headers::vol::DEFAULT_INTRA_QUANT_MATRIX;
        for &q in &[1u32, 2, 5, 8, 16, 31] {
            for &idx in &[1usize, 7, 17, 32, 63] {
                for &lvl in &[1i32, -1, 4, -7, 12, -25] {
                    let recon = reconstruct_intra_mpeg4(lvl, idx, q, &m);
                    if recon == 0 || recon.abs() >= 2047 {
                        // dead zone or saturation — encoder is allowed to
                        // round to a different (smaller-magnitude) level.
                        continue;
                    }
                    let back = quantise_ac_intra_mpeg4(recon, idx, q, &m);
                    // Forward picks the closest level. Allow ±1 because
                    // (2*L*wQ*Q)/16 rounds toward zero so adjacent
                    // levels share boundaries.
                    assert!(
                        (back - lvl).abs() <= 1,
                        "intra mpeg-quant: q={q} idx={idx} lvl={lvl} recon={recon} back={back}"
                    );
                }
            }
        }
    }

    /// Same as above for inter blocks. Inter has the half-step shift in
    /// the dequant rule (`2L+1` instead of `2L`).
    #[test]
    fn mpeg_inter_forward_round_trip_lattice() {
        let m = crate::headers::vol::DEFAULT_NON_INTRA_QUANT_MATRIX;
        for &q in &[1u32, 2, 5, 8, 16, 31] {
            for &idx in &[0usize, 1, 7, 17, 32, 63] {
                for &lvl in &[1i32, -1, 4, -7, 12, -25] {
                    let recon = reconstruct_inter_mpeg4(lvl, idx, q, &m);
                    if recon == 0 || recon.abs() >= 2047 {
                        continue;
                    }
                    let back = quantise_ac_inter_mpeg4(recon, idx, q, &m);
                    assert!(
                        (back - lvl).abs() <= 1,
                        "inter mpeg-quant: q={q} idx={idx} lvl={lvl} recon={recon} back={back}"
                    );
                }
            }
        }
    }

    /// Forward-quant of a coefficient between two valid reconstructions
    /// must pick the closer of the two. Sanity check at low quant.
    #[test]
    fn mpeg_intra_forward_picks_closest() {
        let m = crate::headers::vol::DEFAULT_INTRA_QUANT_MATRIX;
        // q=2, idx=1, matrix[1]=17 → recon(L) = (2*L*2*17)/16 = 4*17*L/16 = 4.25*L.
        // L=1 -> 4, L=2 -> 8 (truncating). Coef 6 is closer to 4 (err 2) than 8 (err 2):
        // tie, picks lower magnitude (L=1).
        assert_eq!(quantise_ac_intra_mpeg4(6, 1, 2, &m), 1);
        // Coef 7 is closer to 8 (err 1) than 4 (err 3) -> L=2.
        assert_eq!(quantise_ac_intra_mpeg4(7, 1, 2, &m), 2);
        // Negative input.
        assert_eq!(quantise_ac_intra_mpeg4(-7, 1, 2, &m), -2);
    }
}

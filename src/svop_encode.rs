//! Rectangular progressive **S(GMC)-VOP encoder** — global motion
//! compensation with one §7.8.4 warping point at half-pel accuracy.
//!
//! An S(GMC)-VOP is the P-VOP macroblock layer plus the §6.3.6 `mcsel`
//! flag: each inter macroblock predicts either from the §7.8.7.1
//! GMC-warped reference (`mcsel == 1`, no `motion_vector()` bodies) or
//! through the plain §7.6 local-MC path (`mcsel == 0`; inter4v never
//! codes `mcsel`). A `not_coded` S macroblock is a GMC copy (implied
//! `mcsel == 1`, zero residual) — *not* the P-VOP zero-MV copy.
//!
//! The encoder:
//!
//! * estimates the **global translation** as the most common per-MB
//!   §7.6 motion vector (the mode of the estimate field), emitted as
//!   the §6.2.5 `sprite_trajectory()` `(du[0], dv[0])` pair in
//!   half-sample units (`i0' = (s/2)·du[0]` — one warping point is a
//!   pure translation of `du/2` pels whatever `s` is);
//! * scores, per macroblock, the GMC prediction (built by the
//!   decoder's own [`gmc_prediction_macroblock`]) against the local
//!   §7.6 candidates and the intra activity, with a small bias toward
//!   GMC (an `mcsel == 1` macroblock spends no motion bits);
//! * mirrors the decoder's predictor bookkeeping: a GMC macroblock
//!   records the §7.8.7.3 **averaged motion vector** into the shared
//!   [`MvGrid`] exactly as `MvDriver::record_gmc_macroblock` does.
//!
//! The emitted unit is decoded back through
//! [`crate::vop_decode::decode_s_gmc_vop_macroblocks`] +
//! [`assemble_s_gmc_vop_frame`] — the same closed loop as the I/P/B
//! encoders.
//!
//! Provenance: §6.2.5 (`sprite_trajectory()`, `warping_mv_code()`),
//! §6.2.6/§6.3.6 (`mcsel`, S(GMC) `not_coded`), §7.8.4/§7.8.7 (warp
//! geometry, GMC prediction, averaged MV) of ISO/IEC 14496-2:2004
//! (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! via the crate's decoder transcriptions.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::nonintra_quant_matrix;
use crate::data_partition::use_intra_dc_vlc;
use crate::framestore::{DecodedFrame, FrameStore};
use crate::ivop_encode::{EncoderConfig, FrameView};
use crate::motion::{predict_motion_vector, MotionVector};
use crate::mv_predictor_grid::MvGrid;
use crate::neighbour::IntraBlockGrid;
use crate::packet_encode::{Layout, MbFields, PacketVopInfo, PacketWriter};
use crate::pvop_encode::{
    estimate_motion, intra_activity, intra_mb_in_p_fields, sample_mode_of, source_luma_mb,
    PVopEncodeStats,
};
use crate::pvop_mv::PvopMbMotion;
use crate::s_gmc_recon::{gmc_prediction_macroblock, GmcReferencePlanes};
use crate::sprite::SpriteTrajectory;
use crate::texture::AcEvent;
use crate::vol::SpriteWarpingAccuracy;
use crate::vop::{parse_vop_header_body, vop_time_increment_bits, VopCodingType, VopContext};
use crate::vop_decode::{gmc_averaged_mv, AnchorMbMotion};
use crate::warp::WarpGeometry;

const VOP_START_CODE: u32 = 0x0000_01B6;

/// Emit one §6.2.5 `warping_mv_code()`: the Table B.34 `dmv_length`
/// VLC (`00` → 0, `010`..`110` → 1..=5, then `SSS − 3` one-bits and a
/// `0` for 6..=14), the `SSS`-bit `dmv_code` when `SSS != 0`, and the
/// trailing `marker_bit`. Exact inverse of
/// [`crate::sprite::decode_warping_mv_code`].
///
/// # Panics
///
/// Panics when `|dmv|` exceeds the Table B.34 range (`2^14 - 1`).
pub fn put_warping_mv_code(bw: &mut BitWriter, dmv: i32) {
    let magnitude = dmv.unsigned_abs();
    assert!(magnitude < (1 << 14), "warping dmv {dmv} out of range");
    if dmv == 0 {
        bw.write_bits(0b00, 2); // dmv_length = 0
        bw.write_marker();
        return;
    }
    let sss = 32 - magnitude.leading_zeros(); // |dmv| in [2^(SSS-1), 2^SSS - 1]
    match sss {
        1..=5 => bw.write_bits(sss + 1, 3), // 010, 011, 100, 101, 110
        _ => {
            for _ in 0..(sss - 3) {
                bw.write_bit(true);
            }
            bw.write_bit(false);
        }
    }
    let span = (1i64 << sss) - 1;
    let code = if dmv > 0 {
        i64::from(dmv)
    } else {
        i64::from(dmv) + span
    };
    bw.write_bits(code as u32, sss as usize);
    bw.write_marker();
}

/// Emit a §6.2.5 S(GMC)-VOP header through `vop_fcode_forward`,
/// including the `sprite_trajectory()` (`trajectory.count` warping
/// points, each `du[i]` then `dv[i]` as `warping_mv_code()`). The
/// writer is left mid-unit — the macroblock walk follows.
#[allow(clippy::too_many_arguments)]
pub fn write_s_vop_header(
    bw: &mut BitWriter,
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    quant: u32,
    fcode: u8,
    trajectory: &SpriteTrajectory,
    intra_dc_vlc_thr: u8,
    interlace: Option<crate::ivop_encode::VopInterlaceFlags>,
) {
    bw.write_start_code(VOP_START_CODE);
    bw.write_bits(0b11, 2); // vop_coding_type = S
    for _ in 0..modulo_time_base {
        bw.write_bit(true);
    }
    bw.write_bit(false);
    bw.write_marker();
    bw.write_bits(
        u32::from(time_increment),
        usize::from(vop_time_increment_bits(resolution)),
    );
    bw.write_marker();
    bw.write_bit(true); // vop_coded = 1
    bw.write_bit(false); // vop_rounding_type = 0 (S(GMC) carries it like P)
    assert!(intra_dc_vlc_thr <= 7, "intra_dc_vlc_thr is a 3-bit field");
    bw.write_bits(u32::from(intra_dc_vlc_thr), 3); // intra_dc_vlc_thr (Table 6-25)
    if let Some(flags) = interlace {
        flags.write(bw); // top_field_first + alternate_vertical_scan_flag
    }
    assert!(
        (1..=3).contains(&trajectory.count),
        "GMC trajectories carry 1..=3 points"
    );
    // sprite_trajectory(): no_of_sprite_warping_points (du, dv) pairs.
    for point in &trajectory.points[..usize::from(trajectory.count)] {
        put_warping_mv_code(bw, point[0]);
        put_warping_mv_code(bw, point[1]);
    }
    assert!((1..=31).contains(&quant), "vop_quant {quant} out of range");
    bw.write_bits(quant, 5);
    assert!((1..=7).contains(&fcode), "vop_fcode_forward out of range");
    bw.write_bits(u32::from(fcode), 3);
}

/// Per-VOP S(GMC) encode statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SVopEncodeStats {
    /// `not_coded` GMC copies (implied `mcsel == 1`, zero residual).
    pub gmc_skipped: usize,
    /// Coded `mcsel == 1` macroblocks.
    pub gmc: usize,
    /// `mcsel == 0` local-MC macroblocks (1-MV).
    pub local: usize,
    /// inter4v macroblocks (no `mcsel` in the syntax).
    pub inter4v: usize,
    /// Intra macroblocks.
    pub intra: usize,
    /// §7.7.2.1 field-predicted local macroblocks (interlaced VOLs).
    pub field: usize,
    /// Macroblocks coded with the §7.7.1 field DCT.
    pub field_dct: usize,
    /// Macroblocks that carried a `dquant`.
    pub dquant: usize,
    /// Video packets cut inside the VOP.
    pub packets: usize,
    /// The emitted trajectory `(du[0], dv[0])` in half-sample units.
    pub trajectory: (i32, i32),
    /// The full emitted `sprite_trajectory()` (`count` points).
    pub points: SpriteTrajectory,
}

/// The most common estimated motion vector across the macroblock grid
/// — the global translation candidate, in the VOL's MV units.
fn dominant_motion(mvs: &[(MotionVector, u32)]) -> MotionVector {
    let mut best = MotionVector { x: 0, y: 0 };
    let mut best_count = 0usize;
    for (i, &(mv, _)) in mvs.iter().enumerate() {
        let count = mvs.iter().filter(|&&(m, _)| m == mv).count();
        if count > best_count || (count == best_count && i == 0) {
            best_count = count;
            best = mv;
        }
    }
    best
}

/// Solve the `n × n` normal equations `a · x = b` in place by Gaussian
/// elimination with partial pivoting; `None` when singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (cell, &p) in a[row][col..].iter_mut().zip(pivot_row[col..].iter()) {
                *cell -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut acc = b[col];
        for k in col + 1..n {
            acc -= a[col][k] * x[k];
        }
        x[col] = acc / a[col][col];
    }
    Some(x)
}

/// A fitted global-motion model: the displacement (in pels) of the
/// picture point `(x, y)` is `(a·x + b·y + tx, c·x + d·y + ty)`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GlobalMotion {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

impl GlobalMotion {
    fn displacement(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }
}

/// Least-squares fit of the §7.8.5 model with `points` warping points
/// to the per-macroblock motion field (`samples`: macroblock-centre
/// coordinates and their displacement, in pels). Two points fit the
/// similarity `(a·x − b·y + tx, b·x + a·y + ty)`, three the full
/// affine; one point is the plain mean (the caller uses the mode
/// instead). `None` when the system is singular.
fn fit_global_motion(samples: &[(f64, f64, f64, f64)], points: u8) -> Option<GlobalMotion> {
    if samples.is_empty() {
        return None;
    }
    match points {
        2 => {
            // Unknowns (a, b, tx, ty); rows: dx = a x − b y + tx,
            // dy = b x + a y + ty.
            let mut ata = vec![vec![0.0; 4]; 4];
            let mut atb = vec![0.0; 4];
            for &(x, y, dx, dy) in samples {
                for (row, rhs) in [([x, -y, 1.0, 0.0], dx), ([y, x, 0.0, 1.0], dy)] {
                    for i in 0..4 {
                        for j in 0..4 {
                            ata[i][j] += row[i] * row[j];
                        }
                        atb[i] += row[i] * rhs;
                    }
                }
            }
            let v = solve(ata, atb)?;
            Some(GlobalMotion {
                a: v[0],
                b: -v[1],
                c: v[1],
                d: v[0],
                tx: v[2],
                ty: v[3],
            })
        }
        3 => {
            // Two independent 3-unknown systems over (x, y, 1).
            let mut ata = vec![vec![0.0; 3]; 3];
            let mut atx = vec![0.0; 3];
            let mut aty = vec![0.0; 3];
            for &(x, y, dx, dy) in samples {
                let row = [x, y, 1.0];
                for i in 0..3 {
                    for j in 0..3 {
                        ata[i][j] += row[i] * row[j];
                    }
                    atx[i] += row[i] * dx;
                    aty[i] += row[i] * dy;
                }
            }
            let vx = solve(ata.clone(), atx)?;
            let vy = solve(ata, aty)?;
            Some(GlobalMotion {
                a: vx[0],
                b: vx[1],
                tx: vx[2],
                c: vy[0],
                d: vy[1],
                ty: vy[2],
            })
        }
        _ => {
            let n = samples.len() as f64;
            let (sx, sy) = samples
                .iter()
                .fold((0.0, 0.0), |acc, &(_, _, dx, dy)| (acc.0 + dx, acc.1 + dy));
            Some(GlobalMotion {
                a: 0.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                tx: sx / n,
                ty: sy / n,
            })
        }
    }
}

/// Fit the global motion (one outlier-rejecting refit) and express it
/// as the §7.8.4 `sprite_trajectory()`: `du[0]/dv[0]` = twice the
/// displacement of `(0, 0)`, `du[1]/dv[1]` that of `(W, 0)` minus
/// `du[0]`, `du[2]/dv[2]` that of `(0, H)` minus `du[0]` — all in
/// half-sample integers (the §7.8.4 reference-point formulas with
/// `i0 = j0 = 0`). Falls back to the plain translation when the fit is
/// degenerate.
fn fit_trajectory(
    local_mvs: &[(MotionVector, u32)],
    mb_width: usize,
    mb_height: usize,
    units_per_pel: i32,
    width: u32,
    height: u32,
    points: u8,
) -> SpriteTrajectory {
    let samples: Vec<(f64, f64, f64, f64)> = local_mvs
        .iter()
        .enumerate()
        .map(|(idx, &(mv, _))| {
            let (col, row) = (idx % mb_width, idx / mb_width);
            (
                (col * 16 + 8) as f64,
                (row * 16 + 8) as f64,
                f64::from(mv.x) / f64::from(units_per_pel),
                f64::from(mv.y) / f64::from(units_per_pel),
            )
        })
        .collect();
    let _ = mb_height;
    // Robust fit: the motion field of a block matcher carries alias
    // vectors (self-similar texture, coarse wide-window lattice hits),
    // so start from the *mode* of the field — the dominant translation
    // — keep the macroblocks within a generous band of it, fit the
    // model there, then tighten to the macroblocks the model explains
    // within a pel and refit.
    let mode_mv = dominant_motion(local_mvs);
    let (mx, my) = (
        f64::from(mode_mv.x) / f64::from(units_per_pel),
        f64::from(mode_mv.y) / f64::from(units_per_pel),
    );
    let translation = GlobalMotion {
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        tx: mx,
        ty: my,
    };
    let band: Vec<_> = samples
        .iter()
        .copied()
        .filter(|&(_, _, dx, dy)| (dx - mx).abs() <= 2.5 && (dy - my).abs() <= 2.5)
        .collect();
    let min_samples = usize::from(points) + 1;
    let mut model = translation;
    if band.len() >= min_samples {
        if let Some(first) = fit_global_motion(&band, points) {
            model = first;
            let inliers: Vec<_> = samples
                .iter()
                .copied()
                .filter(|&(x, y, dx, dy)| {
                    let (px, py) = first.displacement(x, y);
                    (px - dx).abs() <= 1.0 && (py - dy).abs() <= 1.0
                })
                .collect();
            if inliers.len() >= min_samples {
                if let Some(second) = fit_global_motion(&inliers, points) {
                    model = second;
                }
            }
        }
    }
    let half = |v: f64| -> i32 { (2.0 * v).round().clamp(-16000.0, 16000.0) as i32 };
    let (d0x, d0y) = model.displacement(0.0, 0.0);
    let du0 = half(d0x);
    let dv0 = half(d0y);
    let mut traj = SpriteTrajectory {
        count: points,
        points: [[du0, dv0], [0, 0], [0, 0]],
    };
    if points >= 2 {
        let (d1x, d1y) = model.displacement(f64::from(width), 0.0);
        traj.points[1] = [half(d1x) - du0, half(d1y) - dv0];
    }
    if points >= 3 {
        let (d2x, d2y) = model.displacement(0.0, f64::from(height));
        traj.points[2] = [half(d2x) - du0, half(d2y) - dv0];
    }
    // A warp whose corner displacements differ from the translation by
    // at most one half-sample is within the motion field's own
    // resolution — the translation explains it equally; keep the pure
    // translation (the point count is fixed by the VOL).
    if traj.points[1..]
        .iter()
        .all(|p| p[0].abs() <= 1 && p[1].abs() <= 1)
    {
        traj.points[1] = [0, 0];
        traj.points[2] = [0, 0];
    }
    traj
}

/// Total luminance SAD of the §7.8.7.1 GMC prediction of every
/// macroblock under `trajectory` — the cost the trajectory refinement
/// minimises (the decoder's own warp, so what is measured is exactly
/// the prediction a decoder forms).
fn total_gmc_sad(
    trajectory: &SpriteTrajectory,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    planes: &GmcReferencePlanes<'_>,
) -> u64 {
    let (mb_width, mb_height) = cfg.mb_dimensions();
    let geometry = WarpGeometry::decode(
        trajectory,
        u32::from(cfg.width),
        u32::from(cfg.height),
        SpriteWarpingAccuracy::HalfPel,
    );
    let mut total = 0u64;
    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let src = source_luma_mb(frame, mb_row, mb_col);
            let pred = gmc_prediction_macroblock(
                &geometry,
                planes,
                (mb_col * 16) as i64,
                (mb_row * 16) as i64,
                0,
                8,
            );
            total += u64::from(sad_against(&src, &pred));
        }
    }
    total
}

/// Refine a fitted multi-point trajectory by coordinate descent on the
/// decoder's own warp: every active `du[i]` / `dv[i]` is nudged by
/// ±1 and ±2 half-samples while the total GMC SAD keeps dropping. The
/// least-squares fit works on a motion field quantised to the MV grid
/// and can land half a sample off; this closes that gap on the actual
/// prediction error.
fn refine_trajectory(
    mut trajectory: SpriteTrajectory,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    planes: &GmcReferencePlanes<'_>,
) -> SpriteTrajectory {
    let count = usize::from(trajectory.count);
    let mut best = total_gmc_sad(&trajectory, cfg, frame, planes);
    let bound = (1i32 << 14) - 1;
    for _pass in 0..8 {
        let mut improved = false;
        for point in 0..count {
            for axis in 0..2 {
                for step in [-1i32, 1, -2, 2] {
                    let mut cand = trajectory;
                    cand.points[point][axis] =
                        (cand.points[point][axis] + step).clamp(-bound, bound);
                    let cost = total_gmc_sad(&cand, cfg, frame, planes);
                    if cost < best {
                        best = cost;
                        trajectory = cand;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    trajectory
}

/// SAD of the source macroblock against an arbitrary prediction.
fn sad_against(src: &[[i32; 16]; 16], pred: &crate::reconstruct::InterPredictionMacroblock) -> u32 {
    let mut sad = 0u32;
    for (j, row) in src.iter().enumerate() {
        for (i, &s) in row.iter().enumerate() {
            sad += (s - pred.luma[j][i]).unsigned_abs();
        }
    }
    sad
}

/// Encode one rectangular progressive S(GMC)-VOP against `reference`
/// (the closed-loop reconstruction of the previous anchor). Returns
/// the emitted unit and the mode statistics.
pub fn encode_s_vop(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    reference: &DecodedFrame,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
) -> (Vec<u8>, SVopEncodeStats) {
    assert!((1..=31).contains(&qp), "vop_quant {qp} out of range");
    assert!(cfg.gmc, "encode_s_vop needs a GMC VOL");
    let (mb_width, mb_height) = cfg.mb_dimensions();
    let w_intra = crate::block::intra_quant_matrix(vol);
    let w_inter = nonintra_quant_matrix(vol);
    let mode = sample_mode_of(vol);
    let fcode = cfg.fcode;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();
    let gmc_planes = GmcReferencePlanes {
        luma: reference.luma_reference(),
        cb: reference.cb_reference(),
        cr: reference.cr_reference(),
    };

    // ---- Pass 1: per-MB local motion estimates + the global mode ----
    let mut local_mvs: Vec<(MotionVector, u32)> = Vec::with_capacity(mb_width * mb_height);
    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);
            local_mvs.push(estimate_motion(&src, &luma_ref, mb_x, mb_y, mode, fcode));
        }
    }
    let units_per_pel = if vol.quarter_sample { 4 } else { 2 };
    let points = cfg.gmc_points.clamp(1, 3);
    assert_eq!(
        vol.no_of_sprite_warping_points,
        Some(points),
        "VOL and config disagree on no_of_sprite_warping_points"
    );
    let trajectory = if points == 1 {
        let global = dominant_motion(&local_mvs);
        // §7.8.4: du[0]/dv[0] are half-sample units for any s (i0' =
        // (s/2)·du). Quarter-sample local MVs quantise to the half grid.
        // The trajectory is kept inside the Table 7-9 range for the VOP's
        // fcode so the §7.8.7.3 averaged-MV clip can never fire: for a
        // one-point warp the AMV *is* the translation, and a clipped AMV
        // would make every neighbour's §7.6.5 median depend on the exact
        // clip behaviour — the encoder simply never emits that corner.
        let (low, high) = crate::pvop_encode::mv_range(fcode);
        let (amv_low, amv_high) = if vol.quarter_sample {
            // AMV is derived in quarter-sample units (2·du).
            (low / 2, high / 2)
        } else {
            (low, high)
        };
        let (du, dv) = if vol.quarter_sample {
            (global.x / 2, global.y / 2)
        } else {
            (global.x, global.y)
        };
        let (du, dv) = (du.clamp(amv_low, amv_high), dv.clamp(amv_low, amv_high));
        SpriteTrajectory {
            count: 1,
            points: [[du, dv], [0, 0], [0, 0]],
        }
    } else {
        // §7.8.5 similarity (2 points) / affine (3 points): least-squares
        // fit of the per-macroblock motion field. The per-macroblock
        // §7.8.7.3 averaged MV now varies across the picture; the
        // decoder's own derivation (clip included) is mirrored below.
        let fitted = fit_trajectory(
            &local_mvs,
            mb_width,
            mb_height,
            units_per_pel,
            u32::from(cfg.width),
            u32::from(cfg.height),
            points,
        );
        refine_trajectory(fitted, cfg, frame, &gmc_planes)
    };
    let (du, dv) = (trajectory.points[0][0], trajectory.points[0][1]);
    let geometry = WarpGeometry::decode(
        &trajectory,
        u32::from(cfg.width),
        u32::from(cfg.height),
        SpriteWarpingAccuracy::HalfPel,
    );

    let mut bw = BitWriter::new();
    write_s_vop_header(
        &mut bw,
        cfg.time_increment_resolution,
        modulo_time_base,
        time_increment,
        qp,
        fcode,
        &trajectory,
        cfg.intra_dc_vlc_thr,
        cfg.vop_interlace(),
    );
    let mut pw = PacketWriter::new(
        bw,
        cfg.resilience,
        PacketVopInfo {
            coding_type: VopCodingType::S,
            fcode_fwd: fcode,
            fcode_bwd: 0,
            modulo_time_base,
            time_increment,
            time_increment_bits: vop_time_increment_bits(cfg.time_increment_resolution),
            intra_dc_vlc_thr: cfg.intra_dc_vlc_thr,
            total_macroblocks: (mb_width * mb_height) as u32,
            interlaced: cfg.interlaced,
            sprite_trajectory: Some(trajectory),
        },
        Layout::Combined,
    );

    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut mv_grid = MvGrid::new(mb_height, mb_width);
    let mut stats = SVopEncodeStats {
        trajectory: (du, dv),
        points: trajectory,
        ..Default::default()
    };
    let vop_qp = qp;
    let mut running_qp = vop_qp;

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let idx = mb_row * mb_width + mb_col;
            if pw.maybe_cut(idx, running_qp) {
                intra_grid = IntraBlockGrid::new(mb_height, mb_width);
                mv_grid = MvGrid::new(mb_height, mb_width);
            }
            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);
            let activity = intra_activity(&src);
            let plan_quant = |running: u32| -> (u32, Option<i8>) {
                if cfg.adaptive_quant {
                    let class = crate::mb_quant::activity_class(activity);
                    crate::mb_quant::plan_dquant(running, crate::mb_quant::target_qp(vop_qp, class))
                } else {
                    (running, None)
                }
            };

            // GMC candidate: the decoder's own §7.8.7.1 prediction.
            let gmc_pred = gmc_prediction_macroblock(
                &geometry,
                &gmc_planes,
                i64::from(mb_x),
                i64::from(mb_y),
                0,
                8,
            );
            let gmc_sad = sad_against(&src, &gmc_pred);
            let (local_mv, frame_local_sad) = local_mvs[idx];
            // §7.7.2.1 field-predicted local candidate (interlaced VOL):
            // the GMC neighbours' averaged MVs count as frame candidates
            // of the shared predictor.
            let field_local = if cfg.interlaced {
                let candidates = mv_grid
                    .field_predictor_candidates(mb_row, mb_col)
                    .expect("grid coordinates in range");
                let predictor = crate::motion::predict_field_motion_vector(candidates);
                let top = crate::field_encode::estimate_field_motion(
                    &src, &luma_ref, mb_x, mb_y, false, predictor, mode, fcode,
                );
                let bottom = crate::field_encode::estimate_field_motion(
                    &src, &luma_ref, mb_x, mb_y, true, predictor, mode, fcode,
                );
                (top.sad + bottom.sad + crate::pvop_encode::FIELD_MODE_BIAS < frame_local_sad)
                    .then_some((predictor, top, bottom))
            } else {
                None
            };
            let local_sad = field_local.map_or(frame_local_sad, |(_, t, b)| t.sad + b.sad);

            // Mode decision: GMC saves the motion_vector() body, so it
            // gets a small preference; intra wins on flat-vs-motion
            // activity exactly as in the P walk.
            // The preference grows with the quantiser: the saved
            // motion_vector() body is worth more distortion at a
            // coarser step (a plain SAD-per-bit proxy, 8 SAD per
            // quantiser step on top of the flat 64).
            let choose_gmc = gmc_sad <= local_sad.saturating_add(64 + 8 * vop_qp);
            let inter_sad = gmc_sad.min(local_sad);
            let choose_intra = activity + 512 < inter_sad;

            if choose_intra {
                stats.intra += 1;
                let (qp, dquant) = plan_quant(running_qp);
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                let fields = intra_mb_in_p_fields(
                    &pw,
                    frame,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    qp,
                    cfg,
                    &w_intra,
                    use_intra_dc_vlc(cfg.intra_dc_vlc_thr, qp),
                    dquant,
                );
                pw.push(&fields);
                continue;
            }

            let (qp, dquant) = plan_quant(running_qp);

            // Build the chosen prediction + quantise the residual.
            let prediction = if choose_gmc {
                gmc_pred
            } else if let Some((_, top, bottom)) = field_local {
                let mvs = crate::field_motion::FieldMotionVectors {
                    top: top.mv,
                    bottom: bottom.mv,
                };
                match mode {
                    crate::bvop_prediction::BVopSampleMode::HalfPel => {
                        crate::field_motion::field_motion_compensate_one_reference(
                            &luma_ref,
                            &cb_ref,
                            &cr_ref,
                            mvs,
                            top.ref_field,
                            bottom.ref_field,
                            mb_x,
                            mb_y,
                            0,
                        )
                    }
                    crate::bvop_prediction::BVopSampleMode::QuarterPel { bits_per_pixel } => {
                        crate::field_motion::field_motion_compensate_one_reference_qpel(
                            &luma_ref,
                            &cb_ref,
                            &cr_ref,
                            mvs,
                            top.ref_field,
                            bottom.ref_field,
                            mb_x,
                            mb_y,
                            0,
                            bits_per_pixel,
                        )
                    }
                }
            } else {
                crate::pvop_mv::predict_inter_macroblock(
                    PvopMbMotion::OneMv(local_mv),
                    &luma_ref,
                    &cb_ref,
                    &cr_ref,
                    mb_x,
                    mb_y,
                    0,
                    mode,
                )
                .expect("inter motion always yields a prediction")
            };
            let (res_luma, res_cb, res_cr) =
                crate::pvop_encode::macroblock_residual(frame, mb_row, mb_col, &prediction);
            // §7.7.1 dct_type election (interlaced VOL; GMC macroblocks
            // included — §7.8.7.2 fixes only their *prediction* to the
            // frame warp).
            let field_dct = cfg.interlaced && crate::ivop_encode::elect_field_dct(&res_luma);
            let res_luma = if field_dct {
                crate::ivop_encode::field_dct_luma(&res_luma)
            } else {
                res_luma
            };
            let events = crate::pvop_encode::quantise_inter_residual(
                &res_luma,
                &res_cb,
                &res_cr,
                qp,
                cfg.quant_type,
                &w_inter,
                crate::pvop_encode::inter_scan(cfg),
            );
            let all_zero = events.iter().all(|e| e.is_empty());
            let field_dct = field_dct && !all_zero;

            if choose_gmc {
                // The decoder records the §7.8.7.3 averaged MV for the
                // neighbours' §7.6.5 medians.
                let amv = gmc_averaged_mv(
                    &geometry,
                    i64::from(mb_x),
                    i64::from(mb_y),
                    vol.quarter_sample,
                    fcode,
                    false, // the encoder mirrors the spec-literal decode
                )
                .expect("AMV derivation cannot fail on a valid fcode");
                mv_grid
                    .record_one_mv(mb_row, mb_col, amv)
                    .expect("grid coordinates in range");
                if all_zero {
                    // §6.3.6 S(GMC) not_coded: a GMC copy.
                    stats.gmc_skipped += 1;
                    pw.push(&MbFields {
                        not_coded: true,
                        mb_type: 0,
                        cbpc: 0,
                        cbpy: 0,
                        ac_pred_flag: false,
                        dquant: None,
                        mcsel: None,
                        mvds: Vec::new(),
                        fcode,
                        intra_dc: None,
                        blocks: Default::default(),
                        interlaced: None,
                    });
                    continue;
                }
                stats.gmc += 1;
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
            } else {
                stats.local += 1;
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
            }

            let cbpy = (u8::from(!events[0].is_empty()) << 3)
                | (u8::from(!events[1].is_empty()) << 2)
                | (u8::from(!events[2].is_empty()) << 1)
                | u8::from(!events[3].is_empty());
            let cbpc = (u8::from(!events[4].is_empty()) << 1) | u8::from(!events[5].is_empty());
            if field_dct {
                stats.field_dct += 1;
            }
            let mut field_refs = None;
            let mvds = if choose_gmc {
                Vec::new()
            } else if let Some((predictor, top, bottom)) = field_local {
                stats.field += 1;
                mv_grid
                    .record_field(mb_row, mb_col, top.mv, bottom.mv)
                    .expect("grid coordinates in range");
                field_refs = Some((top.ref_field, bottom.ref_field));
                vec![
                    crate::field_encode::field_mv_differential(top.mv, predictor),
                    crate::field_encode::field_mv_differential(bottom.mv, predictor),
                ]
            } else {
                let candidates = mv_grid
                    .predictor_candidates(mb_row, mb_col, 0)
                    .expect("grid coordinates in range");
                let predictor = predict_motion_vector(candidates);
                mv_grid
                    .record_one_mv(mb_row, mb_col, local_mv)
                    .expect("grid coordinates in range");
                vec![(local_mv.x - predictor.x, local_mv.y - predictor.y)]
            };
            let blocks: [Vec<AcEvent>; 6] = events
                .try_into()
                .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
            pw.push(&MbFields {
                not_coded: false,
                mb_type: if dquant.is_some() { 1 } else { 0 },
                cbpc,
                cbpy,
                ac_pred_flag: false,
                dquant,
                mcsel: Some(choose_gmc),
                mvds,
                fcode,
                intra_dc: None,
                blocks,
                interlaced: cfg
                    .interlaced
                    .then_some(crate::packet_encode::InterlacedMbInfo {
                        field_dct,
                        field_refs,
                    }),
            });
        }
    }
    stats.packets = pw.packets_cut();
    (pw.finish(), stats)
}

/// Decode an emitted S(GMC)-VOP unit through the crate's decoder walk,
/// advance `store`'s anchor chain, and return the reconstruction plus
/// the per-macroblock motion the following B-VOPs' §7.6.9.5.1 /
/// §7.6.9.6 co-located substitution consumes (the same mapping the
/// stream decoder applies: a skipped GMC macroblock contributes its
/// averaged MV; a coded GMC or intra macroblock the zero-vector
/// fallback).
pub fn reconstruct_own_s_vop_with_motion(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &mut FrameStore,
) -> (DecodedFrame, Vec<PvopMbMotion>) {
    let (frame, motion) = reconstruct_own_s_vop_with_anchor_motion(vol, unit, store);
    (frame, motion.iter().map(|m| m.progressive()).collect())
}

/// [`reconstruct_own_s_vop_with_motion`] keeping the interlaced shape
/// of the decoded motion: a §7.7.2.1 field-predicted local macroblock
/// surfaces as [`AnchorMbMotion::Field`] (the §7.7.2.2 interlaced-direct
/// source of a following interlaced B-VOP).
pub fn reconstruct_own_s_vop_with_anchor_motion(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &mut FrameStore,
) -> (DecodedFrame, Vec<AnchorMbMotion>) {
    let (mb_width, mb_height) = (
        usize::from(vol.width).div_ceil(16),
        usize::from(vol.height).div_ceil(16),
    );
    let mut br = BitReader::new(unit);
    let sc = br.read_bits(32).expect("unit starts with a start code");
    assert_eq!(sc, VOP_START_CODE, "encoder emitted a malformed unit");
    let vop = parse_vop_header_body(
        &mut br,
        vol.time_increment_resolution,
        VopContext::from_vol(vol),
    )
    .expect("own S-VOP header must parse");
    assert!(matches!(vop.coding_type, VopCodingType::S));
    let (entries, geometry) = crate::vop_decode::decode_s_gmc_vop_macroblocks(
        &mut br,
        vol,
        &vop,
        crate::compat::DecodeOptions::spec(),
    )
    .expect("own S-VOP payload must decode");
    let motion = entries
        .iter()
        .map(|e| match e {
            crate::frame_decode::SGmcMbContent::Local { motion, .. } => {
                AnchorMbMotion::Frame(*motion)
            }
            crate::frame_decode::SGmcMbContent::FieldLocal {
                mvs,
                top_field_ref,
                bottom_field_ref,
                ..
            } => AnchorMbMotion::Field {
                mvs: *mvs,
                top_ref: *top_field_ref,
                bottom_ref: *bottom_field_ref,
            },
            crate::frame_decode::SGmcMbContent::Gmc {
                amv,
                not_coded: true,
                ..
            } => AnchorMbMotion::Frame(PvopMbMotion::OneMv(*amv)),
            crate::frame_decode::SGmcMbContent::Gmc {
                not_coded: false, ..
            }
            | crate::frame_decode::SGmcMbContent::Intra(_) => {
                AnchorMbMotion::Frame(PvopMbMotion::Intra)
            }
        })
        .collect();
    let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
        store,
        mb_width,
        mb_height,
        &entries,
        &geometry,
        vop.rounding_type,
        sample_mode_of(vol),
        8,
    )
    .expect("own S-VOP must assemble")
    .clone();
    store.push_anchor(frame.clone());
    (frame, motion)
}

/// Statistics bridge so callers tracking [`PVopEncodeStats`]-shaped
/// numbers can fold an S-VOP in (skips = GMC copies, inter = GMC +
/// local).
pub fn as_p_stats(stats: &SVopEncodeStats) -> PVopEncodeStats {
    PVopEncodeStats {
        skipped: stats.gmc_skipped,
        inter: stats.gmc + stats.local,
        inter4v: stats.inter4v,
        field: stats.field,
        field_dct: stats.field_dct,
        intra: stats.intra,
        dquant: stats.dquant,
        packets: stats.packets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::decode_warping_mv_code;

    #[test]
    fn affine_fit_recovers_exact_models() {
        // d(x, y) = (0.02 x + 0.01 y + 2, -0.01 x + 0.03 y + 1) sampled
        // on a 6×4 macroblock grid.
        let mut samples = Vec::new();
        for row in 0..4 {
            for col in 0..6 {
                let (x, y) = ((col * 16 + 8) as f64, (row * 16 + 8) as f64);
                samples.push((x, y, 0.02 * x + 0.01 * y + 2.0, -0.01 * x + 0.03 * y + 1.0));
            }
        }
        let m = fit_global_motion(&samples, 3).unwrap();
        assert!(
            (m.a - 0.02).abs() < 1e-9 && (m.b - 0.01).abs() < 1e-9,
            "{m:?}"
        );
        assert!(
            (m.c + 0.01).abs() < 1e-9 && (m.d - 0.03).abs() < 1e-9,
            "{m:?}"
        );
        assert!(
            (m.tx - 2.0).abs() < 1e-9 && (m.ty - 1.0).abs() < 1e-9,
            "{m:?}"
        );
        // Similarity: d = (a x − b y + tx, b x + a y + ty).
        let sim: Vec<_> = samples
            .iter()
            .map(|&(x, y, _, _)| (x, y, 0.02 * x - 0.03 * y + 2.0, 0.03 * x + 0.02 * y + 1.0))
            .collect();
        let m = fit_global_motion(&sim, 2).unwrap();
        assert!(
            (m.a - 0.02).abs() < 1e-9 && (m.b + 0.03).abs() < 1e-9,
            "{m:?}"
        );
        assert!(
            (m.c - 0.03).abs() < 1e-9 && (m.d - 0.02).abs() < 1e-9,
            "{m:?}"
        );
        // Trajectory of the affine model over a 96×64 picture, in
        // half-samples: (0,0) → (4, 2); (96,0) → (2·(1.92+2), 2·(−0.96+1)).
        let mvs: Vec<(MotionVector, u32)> = samples
            .iter()
            .map(|&(_, _, dx, dy)| {
                (
                    MotionVector {
                        x: (dx * 2.0).round() as i32,
                        y: (dy * 2.0).round() as i32,
                    },
                    0,
                )
            })
            .collect();
        let t = fit_trajectory(&mvs, 6, 4, 2, 96, 64, 3);
        assert_eq!(t.points[0], [4, 2], "{t:?}");
        assert!(
            (t.points[1][0] - 4).abs() <= 1 && (t.points[1][1] + 2).abs() <= 1,
            "{t:?}"
        );
        assert!(
            (t.points[2][0] - 1).abs() <= 1 && (t.points[2][1] - 4).abs() <= 1,
            "{t:?}"
        );
    }

    #[test]
    fn warping_mv_codes_round_trip() {
        for v in (-16383..=16383).step_by(7).chain([-16383, -1, 0, 1, 16383]) {
            let mut bw = BitWriter::new();
            put_warping_mv_code(&mut bw, v);
            bw.next_start_code();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert_eq!(decode_warping_mv_code(&mut br).unwrap(), v, "dmv {v}");
        }
    }

    #[test]
    fn s_vop_header_round_trips_with_trajectory() {
        let cfg = EncoderConfig {
            width: 64,
            height: 48,
            gmc: true,
            ..EncoderConfig::default()
        };
        let headers = crate::ivop_encode::write_configuration_headers(&cfg);
        let pos = headers
            .windows(4)
            .position(|w| w == [0, 0, 1, 0x20])
            .unwrap();
        let vol =
            crate::vol::parse_video_object_layer(&headers[pos..], cfg.profile_and_level()).unwrap();
        assert!(matches!(vol.sprite_enable, crate::vol::SpriteEnable::Gmc));
        assert_eq!(vol.no_of_sprite_warping_points, Some(1));

        let mut bw = BitWriter::new();
        write_s_vop_header(
            &mut bw,
            25,
            1,
            7,
            9,
            2,
            &SpriteTrajectory {
                count: 1,
                points: [[-6, 3], [0, 0], [0, 0]],
            },
            0,
            None,
        );
        bw.next_start_code();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(32).unwrap(), VOP_START_CODE);
        let vop = parse_vop_header_body(&mut br, 25, VopContext::from_vol(&vol)).unwrap();
        assert!(matches!(vop.coding_type, VopCodingType::S));
        assert_eq!(vop.quant, 9);
        assert_eq!(vop.fcode_fwd, 2);
        let traj = vop.sprite_trajectory.expect("trajectory present");
        assert_eq!(traj.count, 1);
        assert_eq!(traj.points[0], [-6, 3]);
    }
}

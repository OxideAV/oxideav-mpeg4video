//! The XviD integer inverse DCT (Walken's design: a row pass with per-row cosine tables and
//! rounding terms, then a tangent and √2 butterfly column pass in 16-bit arithmetic), as FFmpeg
//! runs it for every stream whose user data names XviD. Reimplemented here from the published
//! algorithm and its fixed-point constants, not copied, so that XviD streams decode bit for bit
//! as FFmpeg decodes them: the exact `f64` transform in [`crate::idct`] is conformant but drifts
//! up to two levels per long GOP against the transform the encoder used. The shortcuts the
//! reference takes are kept because they change results: a DC-only row that scales to zero is
//! left untouched and counted empty, and rows 3 to 7 the row pass found empty are not read by the
//! column pass at all.

const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 6;
/// Per-row rounding terms (1 << (COL_SHIFT + ROW_SHIFT - 1) for row 0, then the fixed-point
/// values 1.75683487303, 1.10355339059, 0.587788325588, 0, 0.058658283817, 0.25, 0.25).
const ROUND: [i32; 8] = [65536, 3597, 2260, 1203, 0, 120, 512, 512];
/// Cosine tables, one per row (rows 0 and 4, 1 and 7, 2 and 6, 3 and 5 share them).
const TAB04: [i32; 7] = [22725, 21407, 19266, 16384, 12873, 8867, 4520];
const TAB17: [i32; 7] = [31521, 29692, 26722, 22725, 17855, 12299, 6270];
const TAB26: [i32; 7] = [29692, 27969, 25172, 21407, 16819, 11585, 5906];
const TAB35: [i32; 7] = [26722, 25172, 22654, 19266, 15137, 10426, 5315];
const ROW_TABLES: [&[i32; 7]; 8] = [
    &TAB04, &TAB17, &TAB26, &TAB35, &TAB04, &TAB35, &TAB26, &TAB17,
];
/// tan(π/16), tan(2π/16), tan(3π/16) and √2 in 16-bit fixed point.
const TAN1: i32 = 0x32EC;
const TAN2: i32 = 0x6A0A;
const TAN3: i32 = 0xAB0E;
const SQRT2: i32 = 0x5A82;

/// 16-bit fixed-point multiply with the reference's 32-bit wraparound.
#[inline]
fn mult(c: i32, x: i32) -> i32 {
    c.wrapping_mul(x) >> 16
}

/// One row pass in place. Returns whether the row holds anything after scaling, as the
/// reference reports it; a DC-only row whose scaled value is zero is left as it was.
fn idct_row(row: &mut [i16; 8], tab: &[i32; 7], rnd: i32) -> bool {
    let [c1, c2, c3, c4, c5, c6, c7] = *tab;
    let r = |k: usize| i32::from(row[k]);
    let (in0, in1, in2, in3, in4, in5, in6, in7) = (r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(7));
    let right = in5 | in6 | in7;
    let left = in1 | in2 | in3;
    let shift = |v: i32| (v >> ROW_SHIFT) as i16;
    if right | in4 == 0 {
        let k = c4.wrapping_mul(in0).wrapping_add(rnd);
        if left != 0 {
            let a0 = k.wrapping_add(c2.wrapping_mul(in2));
            let a1 = k.wrapping_add(c6.wrapping_mul(in2));
            let a2 = k.wrapping_sub(c6.wrapping_mul(in2));
            let a3 = k.wrapping_sub(c2.wrapping_mul(in2));
            let b0 = c1.wrapping_mul(in1).wrapping_add(c3.wrapping_mul(in3));
            let b1 = c3.wrapping_mul(in1).wrapping_sub(c7.wrapping_mul(in3));
            let b2 = c5.wrapping_mul(in1).wrapping_sub(c1.wrapping_mul(in3));
            let b3 = c7.wrapping_mul(in1).wrapping_sub(c5.wrapping_mul(in3));
            *row = [
                shift(a0.wrapping_add(b0)),
                shift(a1.wrapping_add(b1)),
                shift(a2.wrapping_add(b2)),
                shift(a3.wrapping_add(b3)),
                shift(a3.wrapping_sub(b3)),
                shift(a2.wrapping_sub(b2)),
                shift(a1.wrapping_sub(b1)),
                shift(a0.wrapping_sub(b0)),
            ];
        } else {
            let a0 = k >> ROW_SHIFT;
            if a0 == 0 {
                return false;
            }
            *row = [a0 as i16; 8];
        }
    } else if left | right == 0 {
        let a0 = shift(rnd.wrapping_add(c4.wrapping_mul(in0.wrapping_add(in4))));
        let a1 = shift(rnd.wrapping_add(c4.wrapping_mul(in0.wrapping_sub(in4))));
        *row = [a0, a1, a1, a0, a0, a1, a1, a0];
    } else {
        let k = c4.wrapping_mul(in0).wrapping_add(rnd);
        let a0 = k
            .wrapping_add(c2.wrapping_mul(in2))
            .wrapping_add(c4.wrapping_mul(in4))
            .wrapping_add(c6.wrapping_mul(in6));
        let a1 = k
            .wrapping_add(c6.wrapping_mul(in2))
            .wrapping_sub(c4.wrapping_mul(in4))
            .wrapping_sub(c2.wrapping_mul(in6));
        let a2 = k
            .wrapping_sub(c6.wrapping_mul(in2))
            .wrapping_sub(c4.wrapping_mul(in4))
            .wrapping_add(c2.wrapping_mul(in6));
        let a3 = k
            .wrapping_sub(c2.wrapping_mul(in2))
            .wrapping_add(c4.wrapping_mul(in4))
            .wrapping_sub(c6.wrapping_mul(in6));
        let b0 = c1
            .wrapping_mul(in1)
            .wrapping_add(c3.wrapping_mul(in3))
            .wrapping_add(c5.wrapping_mul(in5))
            .wrapping_add(c7.wrapping_mul(in7));
        let b1 = c3
            .wrapping_mul(in1)
            .wrapping_sub(c7.wrapping_mul(in3))
            .wrapping_sub(c1.wrapping_mul(in5))
            .wrapping_sub(c5.wrapping_mul(in7));
        let b2 = c5
            .wrapping_mul(in1)
            .wrapping_sub(c1.wrapping_mul(in3))
            .wrapping_add(c7.wrapping_mul(in5))
            .wrapping_add(c3.wrapping_mul(in7));
        let b3 = c7
            .wrapping_mul(in1)
            .wrapping_sub(c5.wrapping_mul(in3))
            .wrapping_add(c3.wrapping_mul(in5))
            .wrapping_sub(c1.wrapping_mul(in7));
        *row = [
            shift(a0.wrapping_add(b0)),
            shift(a1.wrapping_add(b1)),
            shift(a2.wrapping_add(b2)),
            shift(a3.wrapping_add(b3)),
            shift(a3.wrapping_sub(b3)),
            shift(a2.wrapping_sub(b2)),
            shift(a1.wrapping_sub(b1)),
            shift(a0.wrapping_sub(b0)),
        ];
    }
    true
}

/// One column pass over `col` (the eight row-pass results of one column), reading only the
/// first `live_rows` rows, as the reference's three column variants do for 8, 4 or 3 live rows.
fn idct_col(col: &mut [i16; 8], live_rows: usize) {
    let c = |k: usize| if k < live_rows { i32::from(col[k]) } else { 0 };
    // Odd part.
    let (in1, in3, in5, in7) = (c(1), c(3), c(5), c(7));
    let mut mm0 = mult(TAN1, in7).wrapping_add(in1);
    let mut mm1 = mult(TAN1, in1).wrapping_sub(in7);
    let mut mm2 = mult(TAN3, in5).wrapping_add(in3);
    let mut mm3 = mult(TAN3, in3).wrapping_sub(in5);
    let mut mm7 = mm0.wrapping_add(mm2);
    let mut mm4 = mm1.wrapping_sub(mm3);
    mm0 = mm0.wrapping_sub(mm2);
    mm1 = mm1.wrapping_add(mm3);
    let mut mm6 = mm0.wrapping_add(mm1);
    let mut mm5 = mm0.wrapping_sub(mm1);
    mm5 = mult(SQRT2, mm5).wrapping_mul(2);
    mm6 = mult(SQRT2, mm6).wrapping_mul(2);
    // Even part.
    let (in0, in2, in4, in6) = (c(0), c(2), c(4), c(6));
    mm3 = mult(TAN2, in6).wrapping_add(in2);
    mm2 = mult(TAN2, in2).wrapping_sub(in6);
    mm0 = in0.wrapping_add(in4);
    mm1 = in0.wrapping_sub(in4);
    let butterfly = |a: &mut i32, b: &mut i32| {
        let sum = a.wrapping_add(*b);
        *b = a.wrapping_sub(*b);
        *a = sum;
    };
    butterfly(&mut mm0, &mut mm3);
    butterfly(&mut mm0, &mut mm7);
    let out = |v: i32| (v >> COL_SHIFT) as i16;
    col[0] = out(mm0);
    col[7] = out(mm7);
    butterfly(&mut mm3, &mut mm4);
    col[3] = out(mm3);
    col[4] = out(mm4);
    butterfly(&mut mm1, &mut mm2);
    butterfly(&mut mm1, &mut mm6);
    col[1] = out(mm1);
    col[6] = out(mm6);
    butterfly(&mut mm2, &mut mm5);
    col[2] = out(mm2);
    col[5] = out(mm5);
}

/// The transform in place on a row-major 8×8 block of 16-bit coefficients, exactly as the
/// reference computes it.
pub fn xvid_idct_in_place(block: &mut [i16; 64]) {
    let mut rows = 0x07u32;
    for (v, (tab, rnd)) in ROW_TABLES.iter().zip(ROUND).enumerate() {
        let row: &mut [i16; 8] = (&mut block[v * 8..v * 8 + 8]).try_into().expect("eight");
        if idct_row(row, tab, rnd) {
            rows |= 1 << v;
        }
    }
    let live_rows = if rows & 0xF0 != 0 {
        8
    } else if rows & 0x08 != 0 {
        4
    } else {
        3
    };
    for x in 0..8 {
        let mut col = [0i16; 8];
        for y in 0..8 {
            col[y] = block[y * 8 + x];
        }
        idct_col(&mut col, live_rows);
        for y in 0..8 {
            block[y * 8 + x] = col[y];
        }
    }
}

/// [`crate::idct::idct_8x8`]'s shape over the XviD transform: `coefficients[v][u]` in, `f[y][x]`
/// out, saturated to `[-2^bpp, 2^bpp - 1]` like the exact transform (the reference clips only
/// after adding the prediction; with predictions inside the sample range the two clips agree).
pub fn xvid_idct_8x8(coefficients: &[[i32; 8]; 8], bits_per_pixel: u32) -> [[i32; 8]; 8] {
    let mut block = [0i16; 64];
    for (v, row) in coefficients.iter().enumerate() {
        for (u, &f) in row.iter().enumerate() {
            block[v * 8 + u] = f.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
    xvid_idct_in_place(&mut block);
    let (lo, hi) = crate::idct::idct_saturation_bounds(bits_per_pixel);
    let mut out = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = i32::from(block[y * 8 + x]).clamp(lo, hi);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks and their transforms produced by running FFmpeg's `ff_xvid_idct` (libavcodec/xvididct.c)
    /// on seeded pseudo-random coefficients: dense small values, sparse values, DC-only blocks and
    /// full-range values in turn. Input line, then output line, per block.
    const VECTORS: &str = include_str!("../tests/fixtures/xvid_idct_vectors.txt");

    fn blocks(text: &str) -> Vec<([i16; 64], [i16; 64])> {
        let lines: Vec<[i16; 64]> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: Vec<i16> = l.split_whitespace().map(|n| n.parse().unwrap()).collect();
                v.try_into().expect("64 values")
            })
            .collect();
        lines.chunks_exact(2).map(|p| (p[0], p[1])).collect()
    }

    #[test]
    fn matches_the_reference_on_every_committed_vector() {
        let pairs = blocks(VECTORS);
        assert_eq!(pairs.len(), 64);
        for (n, (input, want)) in pairs.iter().enumerate() {
            let mut got = *input;
            xvid_idct_in_place(&mut got);
            assert_eq!(&got[..], &want[..], "block {n}: {input:?}");
        }
    }

    /// A larger run of the same harness, when its file is given (development check).
    #[test]
    fn matches_the_reference_on_a_local_vector_file_when_present() {
        let Ok(path) = std::env::var("XVID_IDCT_VECTORS") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let pairs = blocks(&text);
        assert!(pairs.len() > 64);
        let bad = pairs
            .iter()
            .enumerate()
            .filter(|(_, (input, want))| {
                let mut got = *input;
                xvid_idct_in_place(&mut got);
                got[..] != want[..]
            })
            .map(|(n, _)| n)
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "{} of {} blocks differ, first {:?}",
            bad.len(),
            pairs.len(),
            &bad[..bad.len().min(5)]
        );
    }

    #[test]
    fn a_zero_block_stays_zero_and_the_shaped_entry_saturates() {
        let mut zero = [0i16; 64];
        xvid_idct_in_place(&mut zero);
        assert_eq!(zero, [0i16; 64]);
        let mut huge = [[0i32; 8]; 8];
        huge[0][0] = 2047;
        let out = xvid_idct_8x8(&huge, 8);
        assert!(out.iter().flatten().all(|&v| v == 255), "{out:?}");
    }
}

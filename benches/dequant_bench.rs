//! Micro-benchmarks for the H.263 dequantiser.
//!
//! The dequantiser runs once per 8×8 block (6× per macroblock) and is the
//! simplest SIMD-friendly loop in the codec: it's dominated by a constant
//! linear rescale `sign(l) * (2*Q*|l| + q_plus)` with a skip-on-zero gate
//! and a saturating clamp.
//!
//! Three profiles:
//! * `sparse`  — few non-zero coefficients (typical high-quant P block).
//! * `medium`  — ~30% non-zero (typical I block after dequant).
//! * `dense`   — all 64 non-zero (worst case: low-quant content).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_mpeg4video::simd::{self, chunked, scalar};

fn mk_block(density: &str, seed: u32) -> [i32; 64] {
    let mut b = [0i32; 64];
    match density {
        "sparse" => {
            // DC + 3 AC coefs.
            b[0] = 15;
            b[1] = -3;
            b[8] = 2;
            b[16] = 4;
        }
        "medium" => {
            for (i, slot) in b.iter_mut().enumerate() {
                if (i * seed as usize + 13) % 3 == 0 {
                    let v = (((seed.wrapping_mul(i as u32 + 1)) % 41) as i32) - 20;
                    *slot = v;
                }
            }
        }
        "dense" => {
            for (i, slot) in b.iter_mut().enumerate() {
                let v = (((seed.wrapping_mul(i as u32 + 1)) % 41) as i32) - 20;
                *slot = v.max(1); // force non-zero
            }
        }
        _ => unreachable!(),
    }
    b
}

fn bench_dequant(c: &mut Criterion) {
    let n = 1000usize;
    let q = 5i32; // typical
    let q_plus = q; // odd

    for density in ["sparse", "medium", "dense"] {
        let blocks: Vec<[i32; 64]> = (0..n)
            .map(|i| mk_block(density, i as u32 * 2654435761))
            .collect();
        let mut group = c.benchmark_group(format!("dequant_h263/{density}"));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("scalar", n), |b| {
            b.iter(|| {
                let mut local = blocks.clone();
                for blk in local.iter_mut() {
                    scalar::dequant_h263(blk, q, q_plus, 0);
                }
                local
            });
        });
        group.bench_function(BenchmarkId::new("chunked", n), |b| {
            b.iter(|| {
                let mut local = blocks.clone();
                for blk in local.iter_mut() {
                    chunked::dequant_h263(blk, q, q_plus, 0);
                }
                local
            });
        });
        group.bench_function(BenchmarkId::new("default", n), |b| {
            b.iter(|| {
                let mut local = blocks.clone();
                for blk in local.iter_mut() {
                    simd::dequant_h263(blk, q, q_plus, 0);
                }
                local
            });
        });
        group.finish();
    }
}

fn bench_add_residual(c: &mut Criterion) {
    let n = 1000usize;
    let preds: Vec<[u8; 64]> =
        (0..n).map(|i| std::array::from_fn(|k| ((i + k) as u8).wrapping_mul(3))).collect();
    let residuals: Vec<[i32; 64]> = (0..n)
        .map(|i| std::array::from_fn(|k| ((i as i32) * 5 + (k as i32) * 3) - 150))
        .collect();

    let mut group = c.benchmark_group("add_residual_clip_block");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar", n), |b| {
        b.iter(|| {
            let mut dst = vec![0u8; 16 * n];
            for i in 0..n {
                scalar::add_residual_clip_block(&preds[i], &residuals[i], &mut dst, i * 8, 16);
            }
            dst
        });
    });
    group.bench_function(BenchmarkId::new("chunked", n), |b| {
        b.iter(|| {
            let mut dst = vec![0u8; 16 * n];
            for i in 0..n {
                chunked::add_residual_clip_block(&preds[i], &residuals[i], &mut dst, i * 8, 16);
            }
            dst
        });
    });
    group.bench_function(BenchmarkId::new("default", n), |b| {
        b.iter(|| {
            let mut dst = vec![0u8; 16 * n];
            for i in 0..n {
                simd::add_residual_clip_block(&preds[i], &residuals[i], &mut dst, i * 8, 16);
            }
            dst
        });
    });
    group.finish();
}

criterion_group!(benches, bench_dequant, bench_add_residual);
criterion_main!(benches);

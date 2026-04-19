//! Micro-benchmarks for the 8×8 forward and inverse DCT kernels.
//!
//! We time batches of 1000 blocks per iteration so the per-iteration cost
//! matches the scale of a real frame (≈1500 blocks for CIF, ≈6000 for SD)
//! and is dominated by the kernel rather than loop overhead.
//!
//! Three variants per transform:
//! * `scalar`   — the straightforward 3-deep loop reference.
//! * `chunked`  — fixed-size `[f32; 8]` arrays + inner unrolled loops.
//!   LLVM reliably lowers this to AVX2 / NEON on release builds.
//! * `default`  — whatever the crate dispatches (chunked on stable,
//!   std::simd on nightly with `--features nightly`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_mpeg4video::simd::{self, chunked, scalar};

fn mk_dct_block(seed: u32) -> [f32; 64] {
    // DC + sparse low-frequency AC, shape mirrors a dequantised real block.
    let mut b = [0.0f32; 64];
    b[0] = 800.0 + (seed as f32 * 0.37) % 200.0;
    for i in 1..10 {
        let v = (((seed.wrapping_mul(i as u32 + 1)) % 201) as i32 - 100) as f32;
        b[(i * 3) % 64] = v;
    }
    b
}

fn mk_sample_block(seed: u32) -> [f32; 64] {
    let mut b = [0.0f32; 64];
    for j in 0..8 {
        for i in 0..8 {
            let noise =
                ((seed.wrapping_mul(17 + i as u32).wrapping_add(j as u32 * 13)) % 16) as f32;
            b[j * 8 + i] = 64.0 + (i + j) as f32 * 10.0 + noise;
        }
    }
    b
}

fn bench_idct(c: &mut Criterion) {
    let n = 1000usize;
    let blocks: Vec<[f32; 64]> = (0..n)
        .map(|i| mk_dct_block(i as u32 * 2654435761))
        .collect();
    let mut group = c.benchmark_group("idct8x8");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                scalar::idct8x8(blk);
            }
            local
        });
    });
    group.bench_function(BenchmarkId::new("chunked", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                chunked::idct8x8(blk);
            }
            local
        });
    });
    group.bench_function(BenchmarkId::new("default", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                simd::idct8x8(blk);
            }
            local
        });
    });
    group.finish();
}

fn bench_fdct(c: &mut Criterion) {
    let n = 1000usize;
    let blocks: Vec<[f32; 64]> = (0..n)
        .map(|i| mk_sample_block(i as u32 * 2654435761))
        .collect();
    let mut group = c.benchmark_group("fdct8x8");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                scalar::fdct8x8(blk);
            }
            local
        });
    });
    group.bench_function(BenchmarkId::new("chunked", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                chunked::fdct8x8(blk);
            }
            local
        });
    });
    group.bench_function(BenchmarkId::new("default", n), |b| {
        b.iter(|| {
            let mut local = blocks.clone();
            for blk in local.iter_mut() {
                simd::fdct8x8(blk);
            }
            local
        });
    });
    group.finish();
}

criterion_group!(benches, bench_idct, bench_fdct);
criterion_main!(benches);

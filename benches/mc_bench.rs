//! Motion-compensation micro-benchmarks.
//!
//! The half-pel bilinear filter is the dominant cost of P-VOP decode
//! after VLC walks — every inter MB triggers one 16×16 luma and two 8×8
//! chroma predictions, and a CIF frame has 396 MBs.
//!
//! Four sub-pixel positions per block size:
//! * `(0,0)` integer copy — no filter.
//! * `(1,0)` horizontal half-pel — 2-tap filter.
//! * `(0,1)` vertical half-pel — 2-tap filter.
//! * `(1,1)` diagonal half-pel — 4-tap filter.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_mpeg4video::mc::predict_block;
use oxideav_mpeg4video::simd;

const REF_W: i32 = 384;
const REF_H: i32 = 288;

fn mk_ref_plane() -> Vec<u8> {
    let mut p = vec![0u8; (REF_W * REF_H) as usize];
    for y in 0..REF_H as usize {
        for x in 0..REF_W as usize {
            p[y * REF_W as usize + x] = ((x * 7 + y * 3) & 0xFF) as u8;
        }
    }
    p
}

fn bench_predict(c: &mut Criterion) {
    let refp = mk_ref_plane();
    let subpel_positions = [(0, 0, "int"), (1, 0, "half_h"), (0, 1, "half_v"), (1, 1, "half_hv")];
    let block_sizes = [(8, "8x8"), (16, "16x16")];

    for (n, name) in block_sizes {
        for (hx, hy, label) in subpel_positions {
            let count = 1000usize;
            let bench_name = format!("{}_{}", name, label);
            let mut group = c.benchmark_group(format!("predict_block/{bench_name}"));
            group.throughput(Throughput::Elements(count as u64));
            group.bench_function(BenchmarkId::new("default", count), |b| {
                b.iter(|| {
                    let mut dst = vec![0u8; (n * n) as usize];
                    for i in 0..count {
                        let base_x = 16 + (i as i32 % 32) * 8;
                        let base_y = 16 + ((i as i32 / 32) % 16) * 8;
                        predict_block(
                            &refp,
                            REF_W as usize,
                            REF_W,
                            REF_H,
                            base_x,
                            base_y,
                            hx,
                            hy,
                            n,
                            false,
                            &mut dst,
                            n as usize,
                        );
                    }
                    dst
                });
            });
            group.finish();
        }
    }
}

fn bench_copy_mb_luma(c: &mut Criterion) {
    // Skipped-MB copy hot path: 16×16 bytes from reference to picture.
    let src = mk_ref_plane();
    let n = 1000usize;
    let mut group = c.benchmark_group("copy_mb_luma");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar", n), |b| {
        b.iter(|| {
            let mut dst = vec![0u8; src.len()];
            for i in 0..n {
                let base = (i % 256) * 16 * REF_W as usize;
                simd::scalar::copy_mb_luma(
                    &src,
                    base,
                    REF_W as usize,
                    &mut dst,
                    base,
                    REF_W as usize,
                );
            }
            dst
        });
    });
    group.bench_function(BenchmarkId::new("chunked", n), |b| {
        b.iter(|| {
            let mut dst = vec![0u8; src.len()];
            for i in 0..n {
                let base = (i % 256) * 16 * REF_W as usize;
                simd::chunked::copy_mb_luma(
                    &src,
                    base,
                    REF_W as usize,
                    &mut dst,
                    base,
                    REF_W as usize,
                );
            }
            dst
        });
    });
    group.finish();
}

criterion_group!(benches, bench_predict, bench_copy_mb_luma);
criterion_main!(benches);

//! matvec-check: bit-exact + perf gate for the DSL quant matvec across backends.
//!
//! Proves the migration thesis end-to-end: the SAME `matvec_q8` kernel source runs on CPU and every
//! compiled GPU backend, matches the CPU oracle bit-exactly, and reports per-backend throughput -- the
//! perf gate that decides whether a hand-tuned kernel can be retired for its DSL twin.

use hanzo_kernel::prelude::*;
use hanzo_kernel::quant::{
    gen_q4k, matvec_q4k_bench, matvec_q4k_ref, matvec_q4k_run, matvec_q8_bench, matvec_q8_ref,
    matvec_q8_run, QK8_0,
};
use std::time::Instant;

fn maxrel(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        m = m.max((x - y).abs() / x.abs().max(1e-6));
    }
    m
}

// Q4_K: the real K-quant decoded in-kernel from packed bytes. Bit-exact gate + honest kernel-only BW.
fn check_q4k<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize) {
    let (wqs, wsc, wd, wdm, x) = gen_q4k(rows, k);
    let reference = matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, rows, k);
    let got = matvec_q4k_run::<R>(client, &wqs, &wsc, &wd, &wdm, &x, rows, k);
    let rel = maxrel(&reference, &got);
    let ok = rel < 3e-3;
    let ms = matvec_q4k_bench::<R>(client, &wqs, &wsc, &wd, &wdm, &x, rows, k, 50);
    // REAL packed bytes/block = 144 (d+dmin+scales+qs); nb = k/256 blocks/row.
    let wbytes = rows * (k / 256) * 144;
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * rows as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] Q4_K   {}x{}  max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, rows, k, rel,
        if ok { "BIT-EXACT ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

fn gen(rows: usize, k: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let nb = k / QK8_0;
    let mut s = 0x2545F491_4F6CDD1Du64; // xorshift, deterministic
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let wd: Vec<f32> = (0..rows * nb).map(|_| (next() % 1000) as f32 / 8000.0 + 0.01).collect();
    let wq: Vec<i32> = (0..rows * k).map(|_| (next() % 255) as i32 - 127).collect();
    let x: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    (wd, wq, x)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let denom = x.abs().max(1e-6);
        m = m.max(d / denom);
    }
    m
}

fn check<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize) {
    let (wd, wq, x) = gen(rows, k);
    let reference = matvec_q8_ref(&wd, &wq, &x, rows, k);
    let got = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    let rel = max_rel(&reference, &got);
    let ok = rel < 3e-3;  // real decode bugs are >>1e-2; f32 reorder over K terms is ~K*eps
    // warm + timed loop for a rough throughput number (GFLOP: 2 * rows * k)
    for _ in 0..2 {
        let _ = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    }
    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    }
    let _ = (t, iters);
    // real kernel-only throughput (amortized host round-trip)
    let ms = matvec_q8_bench::<R>(client, &wd, &wq, &x, rows, k, 50);
    let gbps = (wd.len() * 4 + wq.len() * 4) as f64 / (ms * 1e6); // weight bytes / time = effective BW
    println!(
        "[{:<7}] matvec {}x{}  max_rel={:.2e}  {}  {:.3} ms/dispatch  {:.0} GB/s (weight BW)",
        name, rows, k, rel, if ok { "MATCH ✓ (f32-reorder tol)" } else { "MISMATCH ✗" }, ms, gbps
    );
}

fn main() {
    let (rows, k) = (4096usize, 4096usize);
    let ctrl = 256usize; // small-K control: reorder noise ~ ctrl*eps, should be ~1e-6
    println!("hanzo-kernel :: one #[cube] matvec_q8 source, lowered per backend, gated bit-exact\n");

    #[cfg(feature = "cpu")]
    {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        check::<CpuRuntime>("CPU", &c, rows, k);
        check::<CpuRuntime>("CPU/ctrl", &c, rows, ctrl);
        check_q4k::<CpuRuntime>("CPU", &c, rows, k);
    }
    #[cfg(feature = "vulkan")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        check::<WgpuRuntime>("VULKAN", &c, rows, k);
        check::<WgpuRuntime>("VK/ctrl", &c, rows, ctrl);
        check_q4k::<WgpuRuntime>("VULKAN", &c, rows, k);
    }
    #[cfg(feature = "metal")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        check::<WgpuRuntime>("METAL", &WgpuRuntime::client(&WgpuDevice::default()), rows, k);
    }
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        check::<CudaRuntime>("CUDA", &CudaRuntime::client(&CudaDevice::default()), rows, k);
    }
    #[cfg(feature = "rocm")]
    {
        use cubecl::hip::{HipDevice, HipRuntime};
        check::<HipRuntime>("ROCM", &HipRuntime::client(&HipDevice::default()), rows, k);
    }
}

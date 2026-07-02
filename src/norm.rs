//! Norm ops in the DSL: RMSNorm and LayerNorm, one source -> every backend.
//!
//! One thread per row: each invocation reduces its own row (mean / mean-square) then normalizes it.
//! Correctness-first; the block-per-row shared-mem reduction (see `quant::matvec_q8_dp4a_blk`) is the
//! perf follow-up with the identical shape. `n` (the normalized dimension) is comptime, so the bounded
//! loops lower cleanly and no runtime `.len()` metadata buffer is needed for it.

use crate::prelude::*;

/// RMSNorm over the last dim: `out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`, per row of `n`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rms_norm<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let mut ss = F::new(0.0);
        for i in 0..n {
            let v = x[base + i];
            ss += v * v;
        }
        let denom = (ss / F::cast_from(n as u32) + eps[0]).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
}

/// LayerNorm over the last dim: `out[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i]`, per row of `n`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn layer_norm<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let ninv = F::new(1.0) / F::cast_from(n as u32);
        let mut sum = F::new(0.0);
        for i in 0..n {
            sum += x[base + i];
        }
        let mean = sum * ninv;
        let mut var = F::new(0.0);
        for i in 0..n {
            let d = x[base + i] - mean;
            var += d * d;
        }
        let denom = (var * ninv + eps[0]).sqrt();
        for i in 0..n {
            out[base + i] = (x[base + i] - mean) / denom * w[i] + b[i];
        }
    }
}

/// Host launch for RMSNorm, generic over the runtime (CPU / Vulkan / Metal / CUDA / ROCm).
pub fn rms_norm_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        rms_norm::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            n,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Host launch for LayerNorm.
pub fn layer_norm_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        layer_norm::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(bh.clone(), b.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            n,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle for RMSNorm -- the trusted reference the DSL kernel is gated against.
pub fn rms_norm_ref(x: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for row in 0..rows {
        let base = row * n;
        let ss: f32 = (0..n).map(|i| x[base + i] * x[base + i]).sum();
        let denom = (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
    out
}

/// CPU oracle for LayerNorm.
pub fn layer_norm_ref(x: &[f32], w: &[f32], b: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for row in 0..rows {
        let base = row * n;
        let mean: f32 = (0..n).map(|i| x[base + i]).sum::<f32>() / n as f32;
        let var: f32 = (0..n).map(|i| (x[base + i] - mean).powi(2)).sum::<f32>() / n as f32;
        let denom = (var + eps).sqrt();
        for i in 0..n {
            out[base + i] = (x[base + i] - mean) / denom * w[i] + b[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
            .fold(0.0, f32::max)
    }

    fn data(rows: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut s = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        let x: Vec<f32> = (0..rows * n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next() * 0.5 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|_| next() * 0.1).collect();
        (x, w, b)
    }

    #[test]
    fn rms_norm_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, _) = data(rows, n);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = rms_norm_run::<CpuRuntime>(&c, &x, &w, rows, n, EPS);
        let want = rms_norm_ref(&x, &w, rows, n, EPS);
        let rel = max_rel(&want, &got);
        eprintln!("[rms_norm  CPU] {rows}x{n} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "rms_norm max_rel {rel}");
    }

    #[test]
    fn layer_norm_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, b) = data(rows, n);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = layer_norm_run::<CpuRuntime>(&c, &x, &w, &b, rows, n, EPS);
        let want = layer_norm_ref(&x, &w, &b, rows, n, EPS);
        let rel = max_rel(&want, &got);
        eprintln!("[layer_norm CPU] {rows}x{n} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "layer_norm max_rel {rel}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn norm_metal_bit_exact() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, b) = data(rows, n);
        let c = WgpuRuntime::client(&WgpuDevice::default());
        let r = rms_norm_run::<WgpuRuntime>(&c, &x, &w, rows, n, EPS);
        let l = layer_norm_run::<WgpuRuntime>(&c, &x, &w, &b, rows, n, EPS);
        let rr = max_rel(&rms_norm_ref(&x, &w, rows, n, EPS), &r);
        let lr = max_rel(&layer_norm_ref(&x, &w, &b, rows, n, EPS), &l);
        eprintln!("[rms_norm  METAL] max_rel={rr:.2e}  [layer_norm METAL] max_rel={lr:.2e}");
        assert!(rr < 2e-3 && lr < 2e-3);
    }
}

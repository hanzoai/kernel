//! Quantized matvec kernels in the DSL -- the bulk of the ~52k hand-written lines is this family.
//!
//! Q8_0 first (the simplest K-independent block): 32 int8 quants + one f16 scale per block. This
//! proves the pattern -- quant decode + contraction, one source, bit-exact, every backend. Q4_K / Q6_K
//! and the int8-dp4a fast path (`Line<i8>.dot`) follow the identical shape with more bit-twiddling.

use crate::prelude::*;

/// Q8_0 block size (weights per scale).
pub const QK8_0: usize = 32;

/// Q8_0 matvec: `out[row] = sum_k (wd[block] * wq[k]) * x[k]`, one invocation per output row.
/// `wd`: scales, one per 32-weight block, `[rows * k/32]`. `wq`: int8 quants widened to i32, `[rows * k]`.
/// The decode (`wd * wq`) and the contraction are the exact math each backend hand-writes today.
#[cube(launch_unchecked)]
pub fn matvec_q8<F: Float>(
    wd: &Array<F>,
    wq: &Array<i32>,
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let nb = k / 32;
        let wbase = row * k;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for i in 0..k {
            let d = wd[dbase + i / 32];
            let q = F::cast_from(wq[wbase + i]);
            acc += d * q * x[i];
        }
        out[row] = acc;
    }
}

/// Host launch, generic over the runtime (CPU / Vulkan / Metal / CUDA / ROCm -- all from this one fn).
pub fn matvec_q8_run<R: Runtime>(
    client: &ComputeClient<R>,
    wd: &[f32],
    wq: &[i32],
    x: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    // One thread per output row; round the grid up to cover all rows.
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        matvec_q8::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(grid, 1, 1),
            CubeDim::new_1d(block),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    }
    let bytes = client.read_one_unchecked(oh);
    f32::from_bytes(&bytes).to_vec()
}

/// CPU oracle: the trusted reference the DSL kernel is gated against (bit-exact within f32 reorder).
pub fn matvec_q8_ref(wd: &[f32], wq: &[i32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    (0..rows)
        .map(|row| {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += wd[row * nb + i / 32] * wq[row * k + i] as f32 * x[i];
            }
            acc
        })
        .collect()
}

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
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
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
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
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

/// Kernel-only throughput: buffers created ONCE, `iters` dispatches enqueued back-to-back, one final
/// sync (the read). This amortizes the host round-trip so the number reflects the KERNEL, not the API
/// -- the real gate for "can the DSL kernel retire the hand-tuned one". Returns mean ms/dispatch.
pub fn matvec_q8_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wd: &[f32],
    wq: &[i32],
    x: &[f32],
    rows: usize,
    k: usize,
    iters: usize,
) -> f64 {
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8::launch_unchecked::<f32, R>(
            c,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone()); // sync warmup
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh); // force completion of all enqueued dispatches
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
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

// ============================================================================================
// Q4_K -- the real K-quant, decoded IN-KERNEL from packed bytes (the bulk of the hand-written lines).
// Block = 144 bytes / 256 weights: d(f16) dmin(f16) scales[12] qs[128]. Host extracts the two f16
// super-block scalars (per-block metadata, like the hand-tuned kernels); the KERNEL does the real work
// -- 6-bit scale/min unpack (get_scale_min_k4) + 4-bit quant extract via bit ops. Layout passed as u32:
//   wqs: qs, 32 u32/block (128 bytes)    wsc: scales, 3 u32/block (12 bytes)
//   wd:  d,  1 f32/block                  wdm: dmin, 1 f32/block
// ============================================================================================
pub const QK_K: usize = 256;

#[device]
fn byte_at(a: &Array<u32>, base: usize, i: usize) -> u32 {
    (a[base + i / 4] >> ((8 * (i % 4)) as u32)) & 255
}

// get_scale_min_k4 scale component (llama.cpp), from the 12 packed scale bytes at scbase.
#[device]
fn q4k_sc(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = byte_at(wsc, scbase, j) & 63;
    if j >= 4 {
        r = (byte_at(wsc, scbase, j + 4) & 15) | ((byte_at(wsc, scbase, j - 4) >> 6) << 4);
    }
    r
}
// get_scale_min_k4 min component.
#[device]
fn q4k_m(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = byte_at(wsc, scbase, j + 4) & 63;
    if j >= 4 {
        r = (byte_at(wsc, scbase, j + 4) >> 4) | ((byte_at(wsc, scbase, j) >> 6) << 4);
    }
    r
}

/// Q4_K matvec, one invocation per output row. Bit-identical decode order to `BlockQ4K::to_float`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q4k<F: Float>(
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    wdm: &Array<F>,
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let nb = k / 256;
        let mut acc = F::new(0.0);
        for b in 0..nb {
            let blk = row * nb + b;
            let qbase = blk * 32;
            let scbase = blk * 3;
            let d = wd[blk];
            let dmin = wdm[blk];
            let xbase = b * 256;
            for g in 0..4 {
                let is = g * 2;
                let d1 = d * F::cast_from(q4k_sc(wsc, scbase, is));
                let mm1 = dmin * F::cast_from(q4k_m(wsc, scbase, is));
                let d2 = d * F::cast_from(q4k_sc(wsc, scbase, is + 1));
                let mm2 = dmin * F::cast_from(q4k_m(wsc, scbase, is + 1));
                for qi in 0..32 {
                    let qb = byte_at(wqs, qbase, g * 32 + qi);
                    let wlo = d1 * F::cast_from(qb & 15) - mm1;
                    acc += wlo * x[xbase + g * 64 + qi];
                    let whi = d2 * F::cast_from(qb >> 4) - mm2;
                    acc += whi * x[xbase + g * 64 + 32 + qi];
                }
            }
        }
        out[row] = acc;
    }
}

/// Host launch for the Q4_K matvec (all runtimes).
pub fn matvec_q4k_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize,
) -> Vec<f32> {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        matvec_q4k::launch_unchecked::<f32, R>(
            client, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

pub fn matvec_q4k_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize, iters: usize,
) -> f64 {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q4k::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[inline]
fn cpu_byte(a: &[u32], base: usize, i: usize) -> u32 { (a[base + i / 4] >> (8 * (i % 4))) & 255 }
#[inline]
fn cpu_sc(wsc: &[u32], scbase: usize, j: usize) -> u32 {
    if j < 4 { cpu_byte(wsc, scbase, j) & 63 }
    else { (cpu_byte(wsc, scbase, j + 4) & 15) | ((cpu_byte(wsc, scbase, j - 4) >> 6) << 4) }
}
#[inline]
fn cpu_m(wsc: &[u32], scbase: usize, j: usize) -> u32 {
    if j < 4 { cpu_byte(wsc, scbase, j + 4) & 63 }
    else { (cpu_byte(wsc, scbase, j + 4) >> 4) | ((cpu_byte(wsc, scbase, j) >> 6) << 4) }
}

/// CPU oracle for Q4_K, same packed inputs as the kernel (bit-for-bit BlockQ4K::to_float order).
pub fn matvec_q4k_ref(wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 256;
    (0..rows).map(|row| {
        let mut acc = 0.0f32;
        for b in 0..nb {
            let blk = row * nb + b;
            let (qbase, scbase) = (blk * 32, blk * 3);
            let (d, dmin) = (wd[blk], wdm[blk]);
            let xbase = b * 256;
            for g in 0..4 {
                let is = g * 2;
                let d1 = d * cpu_sc(wsc, scbase, is) as f32;
                let mm1 = dmin * cpu_m(wsc, scbase, is) as f32;
                let d2 = d * cpu_sc(wsc, scbase, is + 1) as f32;
                let mm2 = dmin * cpu_m(wsc, scbase, is + 1) as f32;
                for qi in 0..32 {
                    let qb = cpu_byte(wqs, qbase, g * 32 + qi);
                    acc += (d1 * (qb & 15) as f32 - mm1) * x[xbase + g * 64 + qi];
                    acc += (d2 * (qb >> 4) as f32 - mm2) * x[xbase + g * 64 + 32 + qi];
                }
            }
        }
        acc
    }).collect()
}

/// Deterministic valid Q4_K test data (packed u32 layout + f16-rounded d/dmin + activation).
pub fn gen_q4k(rows: usize, k: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let nb = k / 256;
    let nblk = rows * nb;
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let wqs: Vec<u32> = (0..nblk * 32).map(|_| next() as u32).collect(); // 128 qs bytes/block
    let wsc: Vec<u32> = (0..nblk * 3).map(|_| next() as u32).collect();  // 12 scale bytes/block
    // f16-round d/dmin so the CPU ref (which the kernel matches) uses the true stored precision
    let wd: Vec<f32> = (0..nblk).map(|_| half::f16::from_f32((next() % 1000) as f32 / 20000.0 + 0.002).to_f32()).collect();
    let wdm: Vec<f32> = (0..nblk).map(|_| half::f16::from_f32((next() % 1000) as f32 / 40000.0).to_f32()).collect();
    let x: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    (wqs, wsc, wd, wdm, x)
}

// ============================================================================================
// dp4a matvec: the inner product uses Vector<i32,4>.dot -> SPIR-V OpSDot (hardware integer dot,
// SPV_KHR_integer_dot_product), verified emitted. Integer activations (xq) so the int dot is exact;
// out[row] = sum_block wd[block] * dot(qw_group, xq_group). Bit-exact vs matvec_q8_dp4a_ref.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a<F: Float>(
    wq: &Array<Vector<i32, Const<4>>>, // int8 weights, grouped x4  [rows * k/4]
    xq: &Array<Vector<i32, Const<4>>>, // int activation, grouped x4 [k/4]
    wd: &Array<F>,                     // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let ng = k / 4;
        let nb = k / 32;
        let wbase = row * ng;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for g in 0..ng {
            let dp = wq[wbase + g].dot(xq[g]); // OpSDot: 4 int8*int products -> i32
            acc += wd[dbase + g / 8] * F::cast_from(dp);
        }
        out[row] = acc;
    }
}

pub fn matvec_q8_dp4a_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i32], xq: &[i32], wd: &[f32], rows: usize, k: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xqh = client.create_from_slice(i32::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32; let grid = (rows as u32).div_ceil(block);
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

pub fn matvec_q8_dp4a_ref(wq: &[i32], xq: &[i32], wd: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    (0..rows).map(|row| {
        let mut acc = 0.0f32;
        for i in 0..k { acc += wd[row * nb + i / 32] * (wq[row * k + i] * xq[i]) as f32; }
        acc
    }).collect()
}

// ============================================================================================
// PACKED dp4a matvec: weights stored as Vector<i8,4> (4 BYTES/group -- the real int8 footprint,
// vs the i32x4 kernel's 16 bytes), cast to Vector<i32,4> in-register (lane-wise sign-extend), then
// .dot() -> OpSDot. Same hardware dp4a, 4x less weight memory traffic. Matvec is weight-bandwidth-
// bound, so this is THE lever toward hand-tuned parity. Bit-exact vs matvec_q8_dp4a_ref.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a_i8<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>, // int8 weights, packed x4  [rows * k/4], 4 bytes/group
    xq: &Array<Vector<i8, Const<4>>>, // int8 activation, packed x4 [k/4]
    wd: &Array<F>,                    // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let ng = k / 4;
        let nb = k / 32;
        let wbase = row * ng;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for g in 0..ng {
            let wi = Vector::<i32, Const<4>>::cast_from(wq[wbase + g]);
            let xi = Vector::<i32, Const<4>>::cast_from(xq[g]);
            let dp = wi.dot(xi); // OpSDot on the widened i32x4 (loaded from 4-byte packed int8)
            acc += wd[dbase + g / 8] * F::cast_from(dp);
        }
        out[row] = acc;
    }
}

/// Host for the packed-int8 dp4a matvec. Weights + activation are real int8 (`&[i8]`), 4 bytes/group.
pub fn matvec_q8_dp4a_i8_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i8], xq: &[i8], wd: &[f32], rows: usize, k: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32; let grid = (rows as u32).div_ceil(block);
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a_i8::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

// ============================================================================================
// BLOCK-per-row dp4a matvec: the bandwidth-bound pattern. One thread block owns one output row;
// its `nt` threads stride over the k/4 groups so ADJACENT threads read ADJACENT Vector<i8,4>
// (consecutive 4-byte loads -> coalesced 256B transactions), vs the one-thread-per-row kernel whose
// adjacent threads are a whole row (k/4 groups) apart -> uncoalesced. Partials are tree-reduced in
// shared memory. This is what takes the DSL dp4a from latency-bound to weight-bandwidth-bound.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a_blk<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>, // int8 weights, packed x4  [rows * k/4]
    xq: &Array<Vector<i8, Const<4>>>, // int8 activation, packed x4 [k/4]
    wd: &Array<F>,                    // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] nt: usize, // threads per block (one block per row)
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let ng = k / 4;
    let wbase = row * ng;
    let dbase = row * (k / 32);
    let mut partial = F::new(0.0);
    let per = ng / nt; // groups per thread (ng is a multiple of nt); bounded for-loop lowers cleanly
    for j in 0..per {
        let g = j * nt + t;
        let wi = Vector::<i32, Const<4>>::cast_from(wq[wbase + g]);
        let xi = Vector::<i32, Const<4>>::cast_from(xq[g]);
        partial += wd[dbase + g / 8] * F::cast_from(wi.dot(xi)); // OpSDot per group
    }
    let mut smem = SharedMemory::<F>::new(nt);
    smem[t] = partial;
    sync_cube();
    let mut stride = CUBE_DIM / 2; // runtime (== nt); a comptime `nt/2` can't be mutated (RuntimeCell)
    while stride > 0 {
        if UNIT_POS < stride {
            let v = smem[(UNIT_POS + stride) as usize];
            smem[t] += v;
        }
        sync_cube();
        stride /= 2;
    }
    if t == 0 { out[row] = smem[0]; }
}

/// Host for the block-per-row dp4a matvec. `nt` threads cooperate per row (coalesced + reduced).
pub fn matvec_q8_dp4a_blk_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i8], xq: &[i8], wd: &[f32], rows: usize, k: usize, nt: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a_blk::launch_unchecked::<f32, R>(
            c, Grid::Static(rows as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k, nt,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

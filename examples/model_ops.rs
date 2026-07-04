//! Using the built-in op library from a downstream crate — the common case when you fork a model
//! (e.g. a Zen architecture) and want the transformer ops without writing kernels yourself.
//!
//!   cargo run --example model_ops
//!
//! Every op is one call, runs on the CPU reference here, and lowers to CUDA / Metal / Vulkan / ROCm
//! unchanged. `use hanzo_kernel::prelude::*;` brings the `Runtime` trait (and everything else) into
//! scope — you always want it.

use hanzo_kernel::cubecl::cpu::{CpuDevice, CpuRuntime};
use hanzo_kernel::prelude::*;

fn main() {
    let client = CpuRuntime::client(&CpuDevice::default());

    // RMSNorm — every transformer layer.
    let normed = hanzo_kernel::norm::rms_norm_run::<CpuRuntime>(
        &client,
        &[1.0, 2.0, 3.0, 4.0], // x  (rows=1, n=4)
        &[1.0; 4],             // weight
        1,                     // rows
        4,                     // n (hidden size)
        1e-6,                  // eps
    );
    println!("rms_norm      = {normed:?}");

    // LayerNorm — the same, with a bias.
    let ln = hanzo_kernel::norm::layer_norm_run::<CpuRuntime>(
        &client, &[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4], 1, 4, 1e-6,
    );
    println!("layer_norm    = {ln:?}");

    // RoPE — both conventions. `interleaved=false` is GPT-NeoX, `true` is GPT-J.
    let rope = hanzo_kernel::rope::rope_run::<CpuRuntime>(
        &client, &[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0], &[0.0, 0.0], 1, 4, false,
    );
    println!("rope (neox)   = {rope:?}");

    // Quantized matvec (Q8) — the decode-hot-path contraction.
    let mv = hanzo_kernel::quant::matvec_q8_run::<CpuRuntime>(
        &client, &[0.1; 4], &[1; 16], &[1.0; 4], 4, 4,
    );
    println!("matvec_q8     = {mv:?}");

    assert!(normed.iter().chain(&ln).chain(&rope).chain(&mv).all(|v| v.is_finite()));
    println!("\nall ops ran on the CPU runtime — the same source lowers to every GPU backend.");
}

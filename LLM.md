# LLM.md — hanzoai/kernel
Guidance for AI agents working in this repo.

## What this is
`hanzo-kernel` — cross-device GPU runtime primitives in **Rust**. Write a kernel
once (`#[kernel(targets(...))]`) and lower it to CUDA, ROCm, Vulkan, Metal,
WebGPU, and CPU from one source, at or above hand-tuned speed. Ships a bit-exact
op library: `quant`, `norm`, `rope`, `attn`, `gdn`, `fuse`.

## Canonical role
A **canonical implementation repo** — the real code and authoritative docs live
here. A low-level Rust primitive in the Hanzo inference stack (consumed by
`hanzoai/ml` and `hanzoai/engine`); part of the Rust ecosystem
(engine · ml · node · net · router · **kernel** · pqc · evm · mcp · cli). Not an
SDK line — discovery/wrapper repos link here, never copy the impl. DRY: one
impl, one place.

## Install / run
```toml
hanzo-kernel = "0.2"   # feature: cpu (default) | vulkan | metal | cuda | rocm
```
- `cargo run --example hello_kernel` — author a `#[kernel]`.
- `cargo run --example model_ops` — run the built-in op library.
- `cargo run --release --bin matvec-check --no-default-features --features "cpu,vulkan"` — correctness + throughput gate.

## Entry points
`src/lib.rs` → `prelude`; op modules `norm`/`rope`/`quant`/`attn`/`gdn`/`fuse`;
`examples/`. CubeCL is the lowering engine, named in exactly one `Cargo.toml`.

## Brand rules (hard)
- Hanzo is the **Open AI Cloud** — a full AI SDK / cloud, **never** an "LLM
  gateway" and never positioned vs LiteLLM. `/v1/` paths, never `/api/`.
- **Zen** models are our own family — never name upstream models.

Canonical SDK model: `~/work/hanzo/SDK-ARCHITECTURE.md`.

## License

Relicensed from BSD-3-Clause to the dual `MIT OR Apache-2.0` grant under
HIP-0137 ("One License", `hanzoai/hips`). `LICENSE` states the dual grant;
`LICENSE-MIT` and `LICENSE-APACHE` carry the full texts. The original BSD
copyright line — `Hanzo AI, Inc.` — carries over verbatim into
`LICENSE-MIT`: the relicense changes the grant, not the copyright record.

//! hanzo-kernel: Hanzo's first-party GPU kernel DSL.
//!
//! Write a kernel ONCE in Rust; it lowers to CUDA / ROCm / Vulkan / Metal. This crate is the
//! first-party facade over the CubeCL lowering engine: kernel source names only `hanzo_kernel::*`,
//! never `cubecl`. "Values, not places" -- CubeCL is the implementation value; `hanzo_kernel` is the
//! stable namespace we build against, so the engine can be upgraded (or forked) without touching a
//! single kernel.
//!
//! The perf primitives our hand-tuned kernels rely on are all here and lower to the native
//! instruction on each backend:
//!   - `dot` on a vectorized `Line<i8>` -> dp4a / OpSDotAccSat (int8 4-way dot)
//!   - `cmma` -> tensor cores (WMMA / cooperative-matrix / simdgroup)
//!   - `SharedMemory`, `plane_*` (subgroup reduce/shuffle), `Atomic`, barriers
//!
//! MIGRATION POLICY (perf-gated, never a downgrade): the DSL provides COVERAGE -- every quant type on
//! every backend, from one source, killing the "same op, N impls, N numeric behaviors" bug class. A
//! hand-tuned kernel is replaced by its DSL twin ONLY when the DSL version is bit-exact AND within perf
//! noise (bench-gated). Where a hand-tuned kernel still wins, it stays as the specialized peak path and
//! the DSL is the portable fallback for the backends that lack a tuned version.

/// The kernel-authoring surface. Kernels write `use hanzo_kernel::prelude::*;` and `#[cube]`.
/// (`cube` is re-exported here so the `cubecl` name never appears in kernel source.)
pub mod prelude {
    pub use cubecl::prelude::*;
    pub use cubecl::{cube, CubeCount, CubeDim};
}

pub use cubecl;

pub mod quant;

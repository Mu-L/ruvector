//! Kernel operations for quantized inference.
//!
//! This module provides the core mathematical operations:
//! - Quantized GEMM (int8 matrix multiplication)
//! - INT4 quantization (2× memory reduction)
//! - Layer normalization
//! - Activation functions
//! - Benchmark utilities

pub mod qgemm;
pub mod norm;
pub mod bench_utils;
pub mod quant4;

pub use qgemm::{qgemm_i8, qgemm_i8_simd};
pub use norm::{layer_norm, layer_norm_inplace, rms_norm};
pub use bench_utils::{Timer, BenchStats, BenchConfig, run_benchmark, compute_gflops, compute_bandwidth_gbps};
pub use quant4::{
    pack_int4, unpack_int4, quantize_f32_to_int4, dequantize_int4_to_f32,
    Int4Weights, BlockInt4Weights, int4_gemv, int4_gemm,
};

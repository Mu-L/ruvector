//! Model Architectures for RuvLLM
//!
//! This module contains model architecture implementations optimized for
//! various hardware targets including Apple Neural Engine (ANE), Metal GPU,
//! and CPU.
//!
//! ## Available Models
//!
//! | Model | Architecture | Params | ANE Optimized | Use Case |
//! |-------|--------------|--------|---------------|----------|
//! | RuvLTRA | Qwen 0.5B | 500M | Yes | Edge inference, mobile |
//!
//! ## Model Selection Guide
//!
//! ```text
//! Model Size vs Performance:
//!
//!   RuvLTRA (0.5B)  ████████░░  Good quality, fast inference
//!                              ANE: 38 TOPS, ~200 tok/s
//!
//!   Phi-3 (3B)      ██████████  High quality, moderate speed
//!                              GPU: Metal, ~50 tok/s
//!
//!   Qwen 1.8B       █████████░  Balanced quality/speed
//!                              GPU: Metal, ~80 tok/s
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::models::ruvltra::{RuvLtraConfig, RuvLtraModel};
//!
//! // Create model with default Qwen 0.5B config
//! let config = RuvLtraConfig::default();
//! let model = RuvLtraModel::new(&config)?;
//!
//! // Run inference
//! let logits = model.forward(&input_ids, &positions, None)?;
//! ```

pub mod ruvltra;

// Re-export main types
pub use ruvltra::{
    // Configuration
    RuvLtraConfig,
    AneOptimization,
    QuantizationType,
    MemoryLayout,
    // Model components
    RuvLtraModel,
    RuvLtraAttention,
    RuvLtraMLP,
    RuvLtraDecoderLayer,
    // Utilities
    RuvLtraModelInfo,
    AneDispatcher,
};

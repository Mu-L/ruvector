//! # RuvLLM WASM - Browser-Compatible LLM Inference Runtime
//!
//! This crate provides WebAssembly bindings for the RuvLLM inference runtime,
//! enabling LLM inference directly in web browsers.
//!
//! ## Features
//!
//! - **KV Cache Management**: Two-tier KV cache with FP16 tail and quantized store
//! - **Memory Pooling**: Efficient buffer reuse for minimal allocation overhead
//! - **Chat Templates**: Support for Llama3, Mistral, Qwen, Phi, Gemma formats
//! - **TypeScript-Friendly**: All types have getter/setter methods for easy JS interop
//!
//! ## Quick Start (JavaScript)
//!
//! ```javascript
//! import init, { RuvLLMWasm, GenerateConfig, ChatMessageWasm, ChatTemplateWasm } from 'ruvllm-wasm';
//!
//! async function main() {
//!     // Initialize WASM module
//!     await init();
//!
//!     // Create inference engine
//!     const llm = new RuvLLMWasm();
//!     llm.initialize();
//!
//!     // Format a chat conversation
//!     const template = ChatTemplateWasm.llama3();
//!     const messages = [
//!         ChatMessageWasm.system("You are a helpful assistant."),
//!         ChatMessageWasm.user("What is WebAssembly?"),
//!     ];
//!     const prompt = template.format(messages);
//!
//!     console.log("Formatted prompt:", prompt);
//!
//!     // KV Cache management
//!     const config = new KvCacheConfigWasm();
//!     config.tailLength = 256;
//!     const kvCache = new KvCacheWasm(config);
//!
//!     const stats = kvCache.stats();
//!     console.log("Cache stats:", stats.toJson());
//! }
//!
//! main();
//! ```
//!
//! ## Building
//!
//! ```bash
//! # Build for browser (bundler target)
//! wasm-pack build --target bundler
//!
//! # Build for Node.js
//! wasm-pack build --target nodejs
//!
//! # Build for web (no bundler)
//! wasm-pack build --target web
//! ```
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+
//! | JavaScript/TS     |---->| wasm-bindgen      |
//! | Application       |     | Bindings          |
//! +-------------------+     +-------------------+
//!                                   |
//!                                   v
//!                           +-------------------+
//!                           | RuvLLM Core       |
//!                           | (Rust WASM)       |
//!                           +-------------------+
//!                                   |
//!                                   v
//!                           +-------------------+
//!                           | Memory Pool       |
//!                           | KV Cache          |
//!                           | Chat Templates    |
//!                           +-------------------+
//! ```
//!
//! ## Memory Management
//!
//! The WASM module uses efficient memory management strategies:
//!
//! - **Arena Allocator**: O(1) bump allocation for inference temporaries
//! - **Buffer Pool**: Pre-allocated buffers in size classes (1KB-256KB)
//! - **Two-Tier KV Cache**: FP32 tail + u8 quantized store
//!
//! ## Browser Compatibility
//!
//! Requires browsers with WebAssembly support:
//! - Chrome 57+
//! - Firefox 52+
//! - Safari 11+
//! - Edge 16+

#![warn(missing_docs)]
#![warn(clippy::all)]

use wasm_bindgen::prelude::*;

pub mod bindings;
pub mod utils;

// Re-export all bindings
pub use bindings::*;
pub use utils::{error, log, now_ms, set_panic_hook, warn, Timer};

/// Initialize the WASM module.
///
/// This should be called once at application startup to set up
/// panic hooks and any other initialization.
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
}

/// Perform a simple health check.
///
/// Returns true if the WASM module is functioning correctly.
#[wasm_bindgen(js_name = healthCheck)]
pub fn health_check() -> bool {
    // Verify we can create basic structures
    let arena = bindings::InferenceArenaWasm::new(1024);
    arena.capacity() >= 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_config_defaults() {
        let config = bindings::GenerateConfig::new();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_chat_message() {
        let msg = bindings::ChatMessageWasm::user("Hello");
        assert_eq!(msg.role(), "user");
        assert_eq!(msg.content(), "Hello");
    }

    #[test]
    fn test_chat_template_detection() {
        let template = bindings::ChatTemplateWasm::detect_from_model_id("meta-llama/Llama-3-8B");
        assert_eq!(template.name(), "llama3");
    }

    #[test]
    fn test_kv_cache_config() {
        let mut config = bindings::KvCacheConfigWasm::new();
        config.set_tail_length(512);
        assert_eq!(config.tail_length(), 512);
    }

    #[test]
    fn test_arena_creation() {
        let arena = bindings::InferenceArenaWasm::new(4096);
        assert!(arena.capacity() >= 4096);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_buffer_pool() {
        let pool = bindings::BufferPoolWasm::new();
        pool.prewarm_all(2);
        assert!(pool.hit_rate() >= 0.0);
    }

    // RuvLLMWasm::new() calls set_panic_hook which uses wasm-bindgen,
    // so skip this test on non-wasm32 targets
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_ruvllm_wasm() {
        let mut llm = bindings::RuvLLMWasm::new();
        assert!(!llm.is_initialized());
        llm.initialize().unwrap();
        assert!(llm.is_initialized());
    }
}

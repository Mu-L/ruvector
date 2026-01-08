//! C ABI bindings for FFI integration
//!
//! Provides a stable C interface for linking from other languages.
//!
//! ## Safety (ADR-0011 S-4)
//!
//! This module contains unsafe FFI code. All unsafe blocks are documented
//! with SAFETY comments explaining the invariants that must be upheld.
//!
//! ### Pointer Requirements
//!
//! All pointer parameters must:
//! - Be non-null (checked at function entry)
//! - Be properly aligned for the pointed-to type
//! - Point to valid memory for the specified length
//! - Remain valid for the duration of the function call
//!
//! ### Memory Ownership
//!
//! - `FpgaEngine` pointers are owned by the caller and must be freed with `fpga_engine_destroy`
//! - `FpgaInferenceResult` allocations must be freed with `fpga_result_free`
//! - All other buffers are borrowed and must remain valid during the call

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use crate::backend::native_sim::NativeSimBackend;
use crate::backend::TransformerBackend;
use crate::gating::DefaultCoherenceGate;
use crate::types::{ComputeClass, FixedShape, GateHint, InferenceRequest, ModelId};

/// Opaque engine handle
pub struct FpgaEngine {
    backend: Box<dyn TransformerBackend>,
}

/// Result code
#[repr(C)]
pub enum FpgaResult {
    Ok = 0,
    InvalidArgument = 1,
    ModelNotFound = 2,
    InferenceFailed = 3,
    AllocationFailed = 4,
    InvalidArtifact = 5,
}

/// Inference result structure
#[repr(C)]
pub struct FpgaInferenceResult {
    /// Status code
    pub status: FpgaResult,
    /// Logits (caller must free with fpga_free_logits)
    pub logits: *mut i16,
    /// Number of logits
    pub logits_len: usize,
    /// Top-K results (token_id, logit pairs)
    pub topk: *mut u32,
    /// Number of top-K pairs
    pub topk_len: usize,
    /// Latency in nanoseconds
    pub latency_ns: u32,
    /// Compute cycles
    pub cycles: u32,
    /// Gate decision (0=full, 1=early_exit, 2=skipped)
    pub gate_decision: u8,
    /// Exit layer (if early exit)
    pub exit_layer: u8,
}

/// Create a new FPGA engine with native simulator backend
///
/// Returns a handle that must be freed with `fpga_engine_destroy`
#[no_mangle]
pub extern "C" fn fpga_engine_create() -> *mut FpgaEngine {
    let gate = Arc::new(DefaultCoherenceGate::new());
    let backend = Box::new(NativeSimBackend::new(gate));

    let engine = Box::new(FpgaEngine { backend });
    Box::into_raw(engine)
}

/// Destroy an FPGA engine
///
/// # Safety
///
/// - `engine` must be a valid pointer returned by `fpga_engine_create`
/// - `engine` must not have been previously destroyed
/// - After this call, `engine` is invalid and must not be used
#[no_mangle]
pub extern "C" fn fpga_engine_destroy(engine: *mut FpgaEngine) {
    if !engine.is_null() {
        // SAFETY: We've verified the pointer is non-null.
        // The caller guarantees this pointer was obtained from `fpga_engine_create`
        // and has not been destroyed yet. Box::from_raw reclaims ownership and
        // drops the engine, freeing all associated memory.
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Load a model artifact
///
/// Returns model ID bytes (32 bytes) on success, NULL on failure
#[no_mangle]
pub extern "C" fn fpga_load_artifact(
    engine: *mut FpgaEngine,
    artifact_bytes: *const u8,
    artifact_len: usize,
    model_id_out: *mut u8,
) -> FpgaResult {
    if engine.is_null() || artifact_bytes.is_null() || model_id_out.is_null() {
        return FpgaResult::InvalidArgument;
    }

    // SAFETY: We've verified all pointers are non-null above.
    // - `engine` was obtained from `fpga_engine_create` (caller's responsibility)
    // - `artifact_bytes` points to valid memory of `artifact_len` bytes (caller's responsibility)
    // - We borrow `engine` mutably for the duration of this function
    let engine = unsafe { &mut *engine };

    // SAFETY: We've verified `artifact_bytes` is non-null.
    // Caller guarantees `artifact_bytes` points to `artifact_len` valid bytes.
    let artifact_slice = unsafe { std::slice::from_raw_parts(artifact_bytes, artifact_len) };

    let artifact = match crate::artifact::unpack_artifact(artifact_slice) {
        Ok(a) => a,
        Err(_) => return FpgaResult::InvalidArtifact,
    };

    match engine.backend.load(&artifact) {
        Ok(model_id) => {
            // SAFETY: We've verified `model_id_out` is non-null.
            // - Caller guarantees `model_id_out` points to at least 32 writable bytes
            // - ModelId::as_bytes() returns exactly 32 bytes
            // - Memory regions don't overlap (model_id is on stack, model_id_out is caller's buffer)
            unsafe {
                ptr::copy_nonoverlapping(model_id.as_bytes().as_ptr(), model_id_out, 32);
            }
            FpgaResult::Ok
        }
        Err(_) => FpgaResult::InvalidArtifact,
    }
}

/// Run inference
///
/// Result must be freed with `fpga_result_free`
#[no_mangle]
pub extern "C" fn fpga_infer(
    engine: *mut FpgaEngine,
    model_id: *const u8,
    tokens: *const u16,
    tokens_len: usize,
    mask: *const u8,
    mask_len: usize,
    coherence_score: i16,
    boundary_crossed: bool,
    max_compute_class: u8,
) -> FpgaInferenceResult {
    let error_result = || FpgaInferenceResult {
        status: FpgaResult::InvalidArgument,
        logits: ptr::null_mut(),
        logits_len: 0,
        topk: ptr::null_mut(),
        topk_len: 0,
        latency_ns: 0,
        cycles: 0,
        gate_decision: 2,
        exit_layer: 0,
    };

    if engine.is_null() || model_id.is_null() || tokens.is_null() || mask.is_null() {
        return error_result();
    }

    // SAFETY: All pointers verified non-null above.
    // Caller guarantees `engine` was obtained from `fpga_engine_create`.
    let engine = unsafe { &mut *engine };

    // SAFETY: `model_id` is non-null and points to exactly 32 bytes (ModelId size).
    // This is documented in the function's public API contract.
    let id_slice = unsafe { std::slice::from_raw_parts(model_id, 32) };
    let mut id_bytes = [0u8; 32];
    id_bytes.copy_from_slice(id_slice);
    let model = ModelId::new(id_bytes);

    // SAFETY: `tokens` is non-null and points to `tokens_len` u16 values.
    // `mask` is non-null and points to `mask_len` u8 values.
    // Caller guarantees these buffers are valid for the specified lengths.
    let tokens_slice = unsafe { std::slice::from_raw_parts(tokens, tokens_len) };
    let mask_slice = unsafe { std::slice::from_raw_parts(mask, mask_len) };

    // Build shape (micro for C API)
    let shape = FixedShape::micro();

    // Build gate hint
    let compute_class = ComputeClass::from_u8(max_compute_class)
        .unwrap_or(ComputeClass::Deliberative);
    let gate_hint = GateHint::new(coherence_score, boundary_crossed, compute_class);

    // Create request
    let req = InferenceRequest::new(model, shape, tokens_slice, mask_slice, gate_hint);

    // Run inference
    match engine.backend.infer(req) {
        Ok(result) => {
            // Allocate logits with checked allocation (prevents panic on overflow)
            let logits_len = result.logits_q.len();
            let logits = if logits_len > 0 {
                match std::alloc::Layout::array::<i16>(logits_len) {
                    Ok(layout) if layout.size() > 0 => {
                        let ptr = unsafe { std::alloc::alloc(layout) as *mut i16 };
                        if !ptr.is_null() {
                            unsafe {
                                ptr::copy_nonoverlapping(result.logits_q.as_ptr(), ptr, logits_len);
                            }
                        }
                        ptr
                    }
                    _ => ptr::null_mut(), // Return null on allocation failure
                }
            } else {
                ptr::null_mut()
            };

            // Allocate top-K with checked allocation
            let (topk, topk_len) = if let Some(ref tk) = result.topk {
                let len = tk.len() * 2; // (token, logit) pairs
                match std::alloc::Layout::array::<u32>(len) {
                    Ok(layout) if layout.size() > 0 => {
                        let ptr = unsafe { std::alloc::alloc(layout) as *mut u32 };
                        if !ptr.is_null() {
                            for (i, (token, logit)) in tk.iter().enumerate() {
                                unsafe {
                                    *ptr.add(i * 2) = *token as u32;
                                    *ptr.add(i * 2 + 1) = *logit as u32;
                                }
                            }
                        }
                        (ptr, tk.len())
                    }
                    _ => (ptr::null_mut(), 0), // Return null on allocation failure
                }
            } else {
                (ptr::null_mut(), 0)
            };

            // Encode gate decision
            let (gate_decision, exit_layer) = match result.witness.gate_decision {
                crate::types::GateDecision::RanFull => (0, 0),
                crate::types::GateDecision::EarlyExit { layer } => (1, layer),
                crate::types::GateDecision::Skipped { .. } => (2, 0),
            };

            FpgaInferenceResult {
                status: FpgaResult::Ok,
                logits,
                logits_len,
                topk,
                topk_len,
                latency_ns: result.witness.latency_ns,
                cycles: result.witness.cycles,
                gate_decision,
                exit_layer,
            }
        }
        Err(_) => {
            let mut result = error_result();
            result.status = FpgaResult::InferenceFailed;
            result
        }
    }
}

/// Free inference result
///
/// # Safety
///
/// - `result` must be a valid pointer to an `FpgaInferenceResult`
/// - The result must have been obtained from `fpga_infer`
/// - This function may only be called once per result
///
/// # Memory Safety (ADR-0011 S-6)
///
/// This function safely deallocates memory using the same layout that was
/// used during allocation in `fpga_infer`. The layout is reconstructed from
/// the stored length fields to ensure correctness.
#[no_mangle]
pub extern "C" fn fpga_result_free(result: *mut FpgaInferenceResult) {
    if result.is_null() {
        return;
    }

    // SAFETY: We've verified `result` is non-null.
    // Caller guarantees this pointer came from `fpga_infer` and hasn't been freed.
    unsafe {
        let r = &mut *result;

        // Free logits buffer
        if !r.logits.is_null() && r.logits_len > 0 {
            // SAFETY: This layout matches what was used in fpga_infer's allocation.
            // We use expect() here because the layout was valid during allocation,
            // so it must still be valid now (same length, same type).
            // If this somehow fails, it indicates memory corruption.
            if let Ok(layout) = std::alloc::Layout::array::<i16>(r.logits_len) {
                // SAFETY: `r.logits` was allocated with this exact layout in `fpga_infer`.
                // We have exclusive access through the mutable reference.
                std::alloc::dealloc(r.logits as *mut u8, layout);
            }
            r.logits = ptr::null_mut();
            r.logits_len = 0;
        }

        // Free top-K buffer
        if !r.topk.is_null() && r.topk_len > 0 {
            // Note: topk buffer stores token+logit pairs, so actual length is topk_len * 2
            if let Ok(layout) = std::alloc::Layout::array::<u32>(r.topk_len * 2) {
                // SAFETY: `r.topk` was allocated with this exact layout in `fpga_infer`.
                std::alloc::dealloc(r.topk as *mut u8, layout);
            }
            r.topk = ptr::null_mut();
            r.topk_len = 0;
        }
    }
}

/// Unload a model
///
/// # Safety
///
/// - `engine` must be a valid pointer from `fpga_engine_create`
/// - `model_id` must point to exactly 32 bytes
#[no_mangle]
pub extern "C" fn fpga_unload(engine: *mut FpgaEngine, model_id: *const u8) -> FpgaResult {
    if engine.is_null() || model_id.is_null() {
        return FpgaResult::InvalidArgument;
    }

    // SAFETY: Pointers verified non-null. Caller guarantees validity.
    let engine = unsafe { &mut *engine };

    // SAFETY: `model_id` is non-null and points to 32 bytes per API contract.
    let id_slice = unsafe { std::slice::from_raw_parts(model_id, 32) };
    let mut id_bytes = [0u8; 32];
    id_bytes.copy_from_slice(id_slice);
    let model = ModelId::new(id_bytes);

    match engine.backend.unload(model) {
        Ok(()) => FpgaResult::Ok,
        Err(_) => FpgaResult::ModelNotFound,
    }
}

/// Check if a model is loaded
///
/// # Safety
///
/// - `engine` must be a valid pointer from `fpga_engine_create`
/// - `model_id` must point to exactly 32 bytes
#[no_mangle]
pub extern "C" fn fpga_is_loaded(engine: *const FpgaEngine, model_id: *const u8) -> bool {
    if engine.is_null() || model_id.is_null() {
        return false;
    }

    // SAFETY: Pointers verified non-null. Caller guarantees validity.
    // Using shared reference since this is a read-only operation.
    let engine = unsafe { &*engine };

    // SAFETY: `model_id` is non-null and points to 32 bytes per API contract.
    let id_slice = unsafe { std::slice::from_raw_parts(model_id, 32) };
    let mut id_bytes = [0u8; 32];
    id_bytes.copy_from_slice(id_slice);
    let model = ModelId::new(id_bytes);

    engine.backend.is_loaded(model)
}

/// Get version string
#[no_mangle]
pub extern "C" fn fpga_version() -> *const c_char {
    // Static string with null terminator
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

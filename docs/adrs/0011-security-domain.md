# ADR-0011: Security Domain Fixes

**Status:** Implemented
**Date:** 2026-01-08
**Priority:** P1 - Critical
**Parent ADR:** [ADR-0010](./0010fixes.md)
**GitHub Issue:** [#108](https://github.com/ruvnet/ruvector/issues/108)

---

## Context

The comprehensive system review (ADR-0010) identified several security vulnerabilities that require immediate attention. RuVector's current security posture scores **82/100 (B+)**, with critical gaps in authentication, input validation, and network security.

### Current Security Metrics

| Category | Finding | Risk Level |
|----------|---------|------------|
| MCP Authentication | None implemented | CRITICAL |
| Path Validation | No sanitization | CRITICAL |
| FFI Safety | Missing pointer checks | MODERATE |
| CORS Policy | Permissive | MODERATE |
| Rate Limiting | Not implemented | MODERATE |
| Memory Safety | Deallocation risks | LOW |

---

## Decision

Implement a comprehensive security hardening strategy organized into the following aggregates following Domain-Driven Design principles.

---

## Aggregate 1: Authentication & Authorization

### S-1: MCP Endpoint Authentication

**Decision:** Implement token-based authentication with optional mTLS support.

**Current State:**
```rust
// ruvector-mcp/src/transport.rs
// No authentication layer - all endpoints publicly accessible
pub async fn start_server(config: &ServerConfig) -> Result<()> {
    let app = Router::new()
        .route("/", post(handle_request))
        .layer(CorsLayer::permissive());  // No auth
    // ...
}
```

**Target State:**
```rust
// Middleware-based authentication
pub struct AuthMiddleware {
    token_validator: Arc<dyn TokenValidator>,
}

impl AuthMiddleware {
    pub fn bearer_token() -> Self {
        Self { token_validator: Arc::new(BearerTokenValidator::new()) }
    }

    pub fn mtls(ca_cert: &Path) -> Self {
        Self { token_validator: Arc::new(MtlsValidator::new(ca_cert)) }
    }
}

// Configuration
#[derive(Deserialize)]
pub struct AuthConfig {
    pub mode: AuthMode,  // "none", "bearer", "mtls"
    pub token_file: Option<PathBuf>,
    pub ca_cert: Option<PathBuf>,
}
```

**Acceptance Criteria:**
- [ ] Create `AuthMiddleware` trait and implementations
- [ ] Add `BearerTokenValidator` with HMAC-SHA256 signature verification
- [ ] Add `MtlsValidator` for mutual TLS authentication
- [ ] Configuration via `settings.toml` and environment variables
- [ ] Add authentication bypass for localhost in development mode
- [ ] Document authentication setup in README
- [ ] Add integration tests for auth flows

**Files to Modify:**
- `ruvector-mcp/src/transport.rs`
- `ruvector-mcp/src/middleware/auth.rs` (new)
- `ruvector-mcp/src/config.rs`

---

### S-2: CORS Restriction

**Decision:** Replace permissive CORS with configurable whitelist.

**Current State:**
```rust
.layer(CorsLayer::permissive())  // Allows all origins
```

**Target State:**
```rust
fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    match config.mode {
        CorsMode::Restrictive => {
            CorsLayer::new()
                .allow_origin(config.allowed_origins.iter()
                    .map(|o| o.parse::<HeaderValue>().unwrap())
                    .collect::<Vec<_>>())
                .allow_methods([Method::POST, Method::OPTIONS])
                .allow_headers([CONTENT_TYPE, AUTHORIZATION])
                .max_age(Duration::from_secs(3600))
        }
        CorsMode::Development => CorsLayer::permissive(),
    }
}

#[derive(Deserialize)]
pub struct CorsConfig {
    pub mode: CorsMode,  // "restrictive" or "development"
    pub allowed_origins: Vec<String>,
}
```

**Acceptance Criteria:**
- [ ] Default to restrictive CORS in production
- [ ] Configuration via `settings.toml`
- [ ] Support wildcard subdomains (e.g., `*.example.com`)
- [ ] Add unit tests for CORS configuration

---

## Aggregate 2: Input Validation

### S-3: Path Traversal Prevention

**Decision:** Implement canonical path resolution with directory whitelist.

**Current State:**
```rust
// ruvector-mcp/src/handlers.rs
// Line 464: No validation
std::fs::copy(&params.db_path, &params.backup_path)?;

// Line ~200: Direct path assignment
db_options.storage_path = params.path.clone();
```

**Target State:**
```rust
use std::path::{Path, PathBuf};

/// Validates that a path is within the allowed directories
pub struct PathValidator {
    allowed_dirs: Vec<PathBuf>,
}

impl PathValidator {
    pub fn new(allowed_dirs: Vec<PathBuf>) -> Self {
        Self { allowed_dirs }
    }

    pub fn validate(&self, path: &Path) -> Result<PathBuf, SecurityError> {
        // Resolve to canonical path
        let canonical = path.canonicalize()
            .map_err(|_| SecurityError::InvalidPath(path.to_path_buf()))?;

        // Check path doesn't contain traversal
        if path.to_string_lossy().contains("..") {
            return Err(SecurityError::PathTraversal(path.to_path_buf()));
        }

        // Verify within allowed directories
        let allowed = self.allowed_dirs.iter()
            .any(|dir| canonical.starts_with(dir));

        if !allowed {
            return Err(SecurityError::PathOutsideAllowed {
                path: canonical,
                allowed: self.allowed_dirs.clone(),
            });
        }

        Ok(canonical)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Invalid path: {0}")]
    InvalidPath(PathBuf),

    #[error("Path traversal attempt detected: {0}")]
    PathTraversal(PathBuf),

    #[error("Path {path} is outside allowed directories: {allowed:?}")]
    PathOutsideAllowed { path: PathBuf, allowed: Vec<PathBuf> },
}
```

**Acceptance Criteria:**
- [ ] Create `PathValidator` utility
- [ ] Integrate with all file-handling MCP handlers
- [ ] Configure allowed directories via settings
- [ ] Add comprehensive unit tests including:
  - `../` traversal attempts
  - Symlink resolution
  - Absolute paths outside allowed dirs
  - Unicode/encoded path manipulation
- [ ] Return clear error messages without exposing system paths

**Files to Modify:**
- `ruvector-mcp/src/security/path.rs` (new)
- `ruvector-mcp/src/handlers.rs`
- `ruvector-mcp/src/config.rs`

---

### S-4: FFI Pointer Validation

**Decision:** Add comprehensive pointer validation and SAFETY documentation.

**Current State:**
```rust
// ruvector-fpga-transformer/src/ffi/c_abi.rs:89-90
// No validation that pointer is valid or properly aligned
unsafe {
    let data = std::slice::from_raw_parts(ptr, len);  // Unsafe!
}
```

**Target State:**
```rust
/// Validates a raw pointer before use
///
/// # Safety
/// This function checks basic pointer validity but cannot guarantee
/// the pointer points to valid memory. Caller must ensure the pointer
/// was obtained from a valid allocation.
#[inline]
pub fn validate_ptr<T>(ptr: *const T, len: usize) -> Result<(), FfiError> {
    // Null check
    if ptr.is_null() {
        return Err(FfiError::NullPointer);
    }

    // Alignment check
    if (ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(FfiError::MisalignedPointer {
            ptr: ptr as usize,
            required_alignment: std::mem::align_of::<T>(),
        });
    }

    // Size overflow check
    let byte_len = len.checked_mul(std::mem::size_of::<T>())
        .ok_or(FfiError::SizeOverflow)?;

    // Reasonable size bounds (configurable)
    if byte_len > MAX_FFI_BUFFER_SIZE {
        return Err(FfiError::BufferTooLarge(byte_len));
    }

    Ok(())
}

// Usage with SAFETY comment
/// # Safety
/// - `ptr` must be valid for reads of `len * size_of::<f32>()` bytes
/// - `ptr` must be properly aligned for f32
/// - The memory must not be mutated during the lifetime of the returned slice
pub unsafe fn create_slice(ptr: *const f32, len: usize) -> Result<&[f32], FfiError> {
    validate_ptr(ptr, len)?;
    // SAFETY: We've validated the pointer is non-null, aligned, and within size bounds.
    // The caller guarantees the memory is valid and won't be mutated.
    Ok(std::slice::from_raw_parts(ptr, len))
}
```

**Acceptance Criteria:**
- [ ] Create `validate_ptr<T>()` utility function
- [ ] Add SAFETY comments to all 25+ unsafe blocks
- [ ] Add debug assertions for pointer validity
- [ ] Document invariants that callers must uphold
- [ ] Add unit tests for edge cases (null, misaligned, overflow)

**Files to Modify:**
- `ruvector-fpga-transformer/src/ffi/c_abi.rs`
- `ruvector-fpga-transformer/src/ffi/mod.rs`

---

## Aggregate 3: Network Security

### S-5: Rate Limiting

**Decision:** Implement token bucket rate limiting for MCP endpoints.

**Target State:**
```rust
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;

pub struct RateLimitConfig {
    /// Requests per second for read operations
    pub read_rps: NonZeroU32,
    /// Requests per second for write operations
    pub write_rps: NonZeroU32,
    /// Requests per second for file operations
    pub file_rps: NonZeroU32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            read_rps: NonZeroU32::new(1000).unwrap(),
            write_rps: NonZeroU32::new(100).unwrap(),
            file_rps: NonZeroU32::new(10).unwrap(),
        }
    }
}

pub struct RateLimitMiddleware {
    read_limiter: RateLimiter<...>,
    write_limiter: RateLimiter<...>,
    file_limiter: RateLimiter<...>,
}
```

**Acceptance Criteria:**
- [ ] Integrate `governor` crate for rate limiting
- [ ] Categorize endpoints by operation type
- [ ] Return `429 Too Many Requests` with `Retry-After` header
- [ ] Per-IP rate limiting (optional)
- [ ] Configuration via settings

---

## Aggregate 4: Memory Safety

### S-6: Deallocation Safety

**Decision:** Store allocation metadata to prevent mismatched deallocation.

**Current State:**
```rust
// ruvector-fpga-transformer/src/ffi/c_abi.rs:244-254
unsafe {
    std::alloc::dealloc(r.logits as *mut u8, layout);
}
// Assumes layout matches allocation - no verification
```

**Target State:**
```rust
/// Tracked allocation that stores its layout for safe deallocation
pub struct TrackedAllocation<T> {
    ptr: *mut T,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T> TrackedAllocation<T> {
    pub fn new(count: usize) -> Result<Self, AllocError> {
        let layout = Layout::array::<T>(count)?;
        // SAFETY: Layout is valid and non-zero
        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(AllocError::OutOfMemory);
        }
        Ok(Self { ptr, layout, _marker: PhantomData })
    }

    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
}

impl<T> Drop for TrackedAllocation<T> {
    fn drop(&mut self) {
        // SAFETY: Layout matches what was used in allocation
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout); }
    }
}
```

**Acceptance Criteria:**
- [ ] Create `TrackedAllocation<T>` wrapper
- [ ] Migrate FFI allocations to use tracked wrapper
- [ ] Add Drop implementation for automatic cleanup
- [ ] Consider using `Box<[T]>` or `Vec<T>` where possible

---

## Implementation Status

All security fixes have been implemented in the `ruvector-security` crate.

### Completed Items

| Issue | Status | Implementation |
|-------|--------|----------------|
| S-1: MCP Authentication | ✅ Done | `ruvector-security/src/auth.rs` - `AuthMiddleware`, `BearerTokenValidator` |
| S-2: CORS Restriction | ✅ Done | `ruvector-security/src/cors.rs` - `CorsConfig`, `build_cors_layer()` |
| S-3: Path Traversal Prevention | ✅ Done | `ruvector-security/src/path.rs` - `PathValidator` |
| S-4: FFI Pointer Validation | ✅ Done | `ruvector-security/src/ffi.rs` - `validate_ptr()`, SAFETY comments in `c_abi.rs` |
| S-5: Rate Limiting | ✅ Done | `ruvector-security/src/rate_limit.rs` - `RateLimiter` |
| S-6: Deallocation Safety | ✅ Done | `ruvector-security/src/ffi.rs` - `TrackedAllocation<T>` |

### Integration Points

- `ruvector-server/src/lib.rs` - Uses `build_cors_layer()` for configurable CORS
- `ruvector-cli/src/mcp/transport.rs` - Uses `CorsConfig` for SSE transport
- `ruvector-fpga-transformer/src/ffi/c_abi.rs` - All unsafe blocks documented with SAFETY comments

### Test Coverage

26 unit tests covering:
- Bearer token validation with constant-time comparison
- CORS origin validation (restrictive, development, wildcard)
- Path traversal detection (`..\`, null bytes, symlinks)
- FFI pointer validation (null, alignment, overflow)
- Rate limiting (token bucket, per-IP, disabled)
- TrackedAllocation lifecycle

## Original Implementation Plan

### Phase 1: Critical (Week 1)
1. **S-1: MCP Authentication** - Token-based auth middleware
2. **S-3: Path Traversal Prevention** - PathValidator implementation

### Phase 2: High Priority (Week 2)
3. **S-2: CORS Restriction** - Configurable whitelist
4. **S-4: FFI Pointer Validation** - SAFETY comments and checks

### Phase 3: Medium Priority (Week 3)
5. **S-5: Rate Limiting** - Token bucket implementation
6. **S-6: Deallocation Safety** - TrackedAllocation wrapper

---

## Consequences

### Positive
- Security score improves from 82/100 to target 95/100
- Protection against common attack vectors (OWASP Top 10)
- Audit-ready codebase with documented safety invariants
- Production-ready network security posture

### Negative
- Slight performance overhead from validation checks
- Breaking change for existing MCP clients (auth required)
- Increased code complexity in FFI layer

### Mitigation
- Validation checks are O(1) with minimal impact
- Provide migration guide for auth changes
- Clear documentation for FFI safety contracts

---

## References

- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [Rust Unsafe Guidelines](https://rust-lang.github.io/unsafe-code-guidelines/)
- [Tower Middleware](https://docs.rs/tower/latest/tower/)
- [Governor Rate Limiting](https://docs.rs/governor/latest/governor/)

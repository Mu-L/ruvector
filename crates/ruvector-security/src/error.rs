//! Security error types

use std::path::PathBuf;

/// Result type for security operations
pub type SecurityResult<T> = std::result::Result<T, SecurityError>;

/// Security-related errors
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    /// Invalid path provided
    #[error("Invalid path: {0}")]
    InvalidPath(PathBuf),

    /// Path traversal attempt detected
    #[error("Path traversal attempt detected: {0}")]
    PathTraversal(PathBuf),

    /// Path is outside allowed directories
    #[error("Path {path} is outside allowed directories")]
    PathOutsideAllowed {
        path: PathBuf,
        allowed: Vec<PathBuf>,
    },

    /// Path contains invalid characters
    #[error("Path contains invalid characters: {0}")]
    InvalidPathCharacters(PathBuf),

    /// Symlink detected when not allowed
    #[error("Symlink detected: {0}")]
    SymlinkDetected(PathBuf),

    /// Authentication required but not provided
    #[error("Authentication required")]
    AuthenticationRequired,

    /// Invalid authentication token
    #[error("Invalid authentication token")]
    InvalidToken,

    /// Token has expired
    #[error("Token has expired")]
    TokenExpired,

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after_secs} seconds")]
    RateLimitExceeded { retry_after_secs: u64 },

    /// FFI null pointer
    #[error("Null pointer passed to FFI function")]
    NullPointer,

    /// FFI misaligned pointer
    #[error("Misaligned pointer: address {ptr:#x} requires {required_alignment}-byte alignment")]
    MisalignedPointer { ptr: usize, required_alignment: usize },

    /// FFI size overflow
    #[error("Size overflow in FFI operation")]
    SizeOverflow,

    /// FFI buffer too large
    #[error("Buffer too large: {0} bytes exceeds maximum")]
    BufferTooLarge(usize),

    /// Allocation failure
    #[error("Memory allocation failed")]
    AllocationFailed,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

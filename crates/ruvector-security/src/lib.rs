//! RuVector Security Utilities
//!
//! This crate provides security primitives for the RuVector vector database:
//!
//! - **Path Validation**: Prevents path traversal attacks (S-3)
//! - **Authentication**: Token-based authentication middleware (S-1)
//! - **CORS Configuration**: Configurable CORS policies (S-2)
//! - **Rate Limiting**: Token bucket rate limiter (S-5)
//! - **FFI Safety**: Safe pointer handling utilities (S-4, S-6)
//!
//! # Example
//!
//! ```rust,no_run
//! use ruvector_security::{PathValidator, SecurityConfig};
//!
//! let validator = PathValidator::new(vec!["/data".into()]);
//! assert!(validator.validate("/data/vectors.db").is_ok());
//! assert!(validator.validate("/etc/passwd").is_err());
//! ```

pub mod auth;
pub mod cors;
pub mod error;
pub mod ffi;
pub mod path;
pub mod rate_limit;

pub use auth::{AuthConfig, AuthMiddleware, AuthMode, BearerTokenValidator, TokenValidator};
pub use cors::{CorsConfig, CorsMode};
pub use error::{SecurityError, SecurityResult};
pub use ffi::{validate_ptr, FfiError, TrackedAllocation};
pub use path::PathValidator;
pub use rate_limit::{RateLimitConfig, RateLimiter};

/// Security configuration combining all security settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub auth: AuthConfig,
    /// CORS configuration
    pub cors: CorsConfig,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Allowed paths for file operations
    pub allowed_paths: Vec<std::path::PathBuf>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth: AuthConfig::default(),
            cors: CorsConfig::default(),
            rate_limit: RateLimitConfig::default(),
            allowed_paths: vec![std::path::PathBuf::from(".")],
        }
    }
}

impl SecurityConfig {
    /// Create a new security configuration for development
    pub fn development() -> Self {
        Self {
            auth: AuthConfig {
                mode: AuthMode::None,
                ..Default::default()
            },
            cors: CorsConfig {
                mode: CorsMode::Development,
                ..Default::default()
            },
            rate_limit: RateLimitConfig::default(),
            allowed_paths: vec![std::path::PathBuf::from(".")],
        }
    }

    /// Create a new security configuration for production
    pub fn production(token: String, allowed_origins: Vec<String>) -> Self {
        Self {
            auth: AuthConfig {
                mode: AuthMode::Bearer,
                token: Some(token),
                ..Default::default()
            },
            cors: CorsConfig {
                mode: CorsMode::Restrictive,
                allowed_origins,
                ..Default::default()
            },
            rate_limit: RateLimitConfig::default(),
            allowed_paths: vec![std::path::PathBuf::from("./data")],
        }
    }

    /// Create a path validator from this config
    pub fn path_validator(&self) -> PathValidator {
        PathValidator::new(self.allowed_paths.clone())
    }
}

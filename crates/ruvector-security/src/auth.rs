//! Authentication middleware and token validation
//!
//! Provides bearer token authentication for MCP endpoints.

use crate::error::{SecurityError, SecurityResult};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Authentication mode
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AuthMode {
    /// No authentication (development only)
    #[default]
    None,
    /// Bearer token authentication
    Bearer,
    /// Mutual TLS (not yet implemented)
    Mtls,
}

/// Authentication configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuthConfig {
    /// Authentication mode
    pub mode: AuthMode,
    /// Bearer token (for Bearer mode)
    #[serde(skip_serializing)]
    pub token: Option<String>,
    /// Secret key for HMAC validation
    #[serde(skip_serializing)]
    pub secret_key: Option<String>,
    /// Token expiry in seconds (0 = no expiry)
    pub token_expiry_secs: u64,
    /// Allow localhost without auth (development)
    pub allow_localhost: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            mode: AuthMode::None,
            token: None,
            secret_key: None,
            token_expiry_secs: 0,
            allow_localhost: true,
        }
    }
}

/// Token validator trait
pub trait TokenValidator: Send + Sync {
    /// Validate a token and return Ok if valid
    fn validate(&self, token: &str) -> SecurityResult<()>;
}

/// Bearer token validator using constant-time comparison
#[derive(Clone)]
pub struct BearerTokenValidator {
    /// Expected token hash
    token_hash: Vec<u8>,
    /// HMAC key for hashing
    hmac_key: Vec<u8>,
}

impl BearerTokenValidator {
    /// Create a new bearer token validator
    pub fn new(expected_token: &str) -> Self {
        // Generate a random key for HMAC
        let mut hmac_key = vec![0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut hmac_key);

        // Hash the expected token
        let mut mac = HmacSha256::new_from_slice(&hmac_key).expect("HMAC can take key of any size");
        mac.update(expected_token.as_bytes());
        let token_hash = mac.finalize().into_bytes().to_vec();

        Self {
            token_hash,
            hmac_key,
        }
    }

    /// Create from environment variable
    pub fn from_env(env_var: &str) -> Option<Self> {
        std::env::var(env_var).ok().map(|token| Self::new(&token))
    }
}

impl TokenValidator for BearerTokenValidator {
    fn validate(&self, token: &str) -> SecurityResult<()> {
        // Hash the provided token
        let mut mac =
            HmacSha256::new_from_slice(&self.hmac_key).expect("HMAC can take key of any size");
        mac.update(token.as_bytes());
        let token_hash = mac.finalize().into_bytes();

        // Constant-time comparison to prevent timing attacks
        if token_hash.ct_eq(&self.token_hash).into() {
            Ok(())
        } else {
            Err(SecurityError::InvalidToken)
        }
    }
}

/// Authentication middleware
#[derive(Clone)]
pub struct AuthMiddleware {
    validator: Option<Arc<dyn TokenValidator>>,
    config: AuthConfig,
}

impl AuthMiddleware {
    /// Create new auth middleware with configuration
    pub fn new(config: AuthConfig) -> Self {
        let validator: Option<Arc<dyn TokenValidator>> = match &config.mode {
            AuthMode::None => None,
            AuthMode::Bearer => config
                .token
                .as_ref()
                .map(|t| Arc::new(BearerTokenValidator::new(t)) as Arc<dyn TokenValidator>),
            AuthMode::Mtls => {
                tracing::warn!("mTLS authentication not yet implemented, falling back to None");
                None
            }
        };

        Self { validator, config }
    }

    /// Create middleware that requires no authentication (development)
    pub fn none() -> Self {
        Self::new(AuthConfig::default())
    }

    /// Create middleware with bearer token
    pub fn bearer(token: &str) -> Self {
        Self::new(AuthConfig {
            mode: AuthMode::Bearer,
            token: Some(token.to_string()),
            ..Default::default()
        })
    }

    /// Create from environment variable
    pub fn from_env(env_var: &str) -> Self {
        match std::env::var(env_var) {
            Ok(token) if !token.is_empty() => Self::bearer(&token),
            _ => Self::none(),
        }
    }

    /// Validate a request's authorization header
    pub fn validate_header(&self, auth_header: Option<&str>) -> SecurityResult<()> {
        // If no authentication required, allow all
        if self.config.mode == AuthMode::None {
            return Ok(());
        }

        // Get validator or fail
        let validator = self
            .validator
            .as_ref()
            .ok_or(SecurityError::AuthenticationRequired)?;

        // Parse authorization header
        let header = auth_header.ok_or(SecurityError::AuthenticationRequired)?;

        // Extract bearer token
        let token = header
            .strip_prefix("Bearer ")
            .or_else(|| header.strip_prefix("bearer "))
            .ok_or(SecurityError::InvalidToken)?;

        validator.validate(token)
    }

    /// Check if a remote address should bypass authentication
    pub fn is_localhost_allowed(&self, remote_addr: &str) -> bool {
        if !self.config.allow_localhost {
            return false;
        }

        remote_addr.starts_with("127.0.0.1")
            || remote_addr.starts_with("::1")
            || remote_addr.starts_with("localhost")
    }

    /// Get the authentication mode
    pub fn mode(&self) -> &AuthMode {
        &self.config.mode
    }
}

impl Default for AuthMiddleware {
    fn default() -> Self {
        Self::none()
    }
}

/// Generate a secure random token
pub fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
    BASE64.encode(bytes)
}

/// Generate a token with a specific prefix for identification
pub fn generate_prefixed_token(prefix: &str) -> String {
    format!("{}_{}", prefix, generate_token())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bearer_token_validation() {
        let token = "my_secret_token";
        let validator = BearerTokenValidator::new(token);

        assert!(validator.validate(token).is_ok());
        assert!(validator.validate("wrong_token").is_err());
    }

    #[test]
    fn test_auth_middleware_bearer() {
        let token = "test_token_12345";
        let middleware = AuthMiddleware::bearer(token);

        // Valid token
        assert!(middleware
            .validate_header(Some(&format!("Bearer {}", token)))
            .is_ok());

        // Invalid token
        assert!(middleware
            .validate_header(Some("Bearer wrong_token"))
            .is_err());

        // Missing header
        assert!(middleware.validate_header(None).is_err());
    }

    #[test]
    fn test_auth_middleware_none() {
        let middleware = AuthMiddleware::none();

        // All requests should pass
        assert!(middleware.validate_header(None).is_ok());
        assert!(middleware.validate_header(Some("anything")).is_ok());
    }

    #[test]
    fn test_generate_token() {
        let token1 = generate_token();
        let token2 = generate_token();

        // Tokens should be different
        assert_ne!(token1, token2);

        // Tokens should be reasonable length
        assert!(token1.len() >= 40);
    }

    #[test]
    fn test_localhost_bypass() {
        let config = AuthConfig {
            mode: AuthMode::Bearer,
            token: Some("secret".to_string()),
            allow_localhost: true,
            ..Default::default()
        };
        let middleware = AuthMiddleware::new(config);

        assert!(middleware.is_localhost_allowed("127.0.0.1:8080"));
        assert!(middleware.is_localhost_allowed("::1:8080"));
        assert!(!middleware.is_localhost_allowed("192.168.1.1:8080"));
    }
}

//! CORS (Cross-Origin Resource Sharing) configuration
//!
//! Provides configurable CORS policies for production and development.

use std::time::Duration;

/// CORS mode
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CorsMode {
    /// Restrictive CORS (production default)
    #[default]
    Restrictive,
    /// Permissive CORS (development only)
    Development,
    /// Custom CORS configuration
    Custom,
}

/// CORS configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorsConfig {
    /// CORS mode
    pub mode: CorsMode,
    /// Allowed origins (for Restrictive/Custom mode)
    pub allowed_origins: Vec<String>,
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Exposed headers
    pub exposed_headers: Vec<String>,
    /// Allow credentials
    pub allow_credentials: bool,
    /// Max age in seconds
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            mode: CorsMode::Restrictive,
            allowed_origins: vec![],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            exposed_headers: vec!["X-Request-ID".to_string(), "X-RateLimit-Remaining".to_string()],
            allow_credentials: false,
            max_age_secs: 3600,
        }
    }
}

impl CorsConfig {
    /// Create a development CORS configuration (permissive)
    pub fn development() -> Self {
        Self {
            mode: CorsMode::Development,
            ..Default::default()
        }
    }

    /// Create a production CORS configuration
    pub fn production(allowed_origins: Vec<String>) -> Self {
        Self {
            mode: CorsMode::Restrictive,
            allowed_origins,
            allow_credentials: true,
            ..Default::default()
        }
    }

    /// Add an allowed origin
    pub fn allow_origin(mut self, origin: impl Into<String>) -> Self {
        self.allowed_origins.push(origin.into());
        self
    }

    /// Set allowed methods
    pub fn allow_methods(mut self, methods: Vec<String>) -> Self {
        self.allowed_methods = methods;
        self
    }

    /// Get max age as Duration
    pub fn max_age(&self) -> Duration {
        Duration::from_secs(self.max_age_secs)
    }

    /// Check if an origin is allowed
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        match self.mode {
            CorsMode::Development => true,
            CorsMode::Restrictive | CorsMode::Custom => {
                self.allowed_origins.iter().any(|allowed| {
                    if allowed == "*" {
                        return true;
                    }
                    // Support wildcard subdomains like *.example.com
                    if let Some(suffix) = allowed.strip_prefix("*.") {
                        return origin.ends_with(suffix)
                            || origin == format!("https://{}", suffix)
                            || origin == format!("http://{}", suffix);
                    }
                    origin == allowed
                })
            }
        }
    }
}

/// Build a tower-http CORS layer from configuration
#[cfg(feature = "middleware")]
pub fn build_cors_layer(
    config: &CorsConfig,
) -> tower_http::cors::CorsLayer {
    use http::header::{HeaderName, HeaderValue};
    use http::Method;
    use tower_http::cors::CorsLayer;

    match config.mode {
        CorsMode::Development => CorsLayer::permissive(),
        CorsMode::Restrictive | CorsMode::Custom => {
            let mut layer = CorsLayer::new();

            // Set allowed origins
            if config.allowed_origins.is_empty() {
                // No origins = block all cross-origin requests
                layer = layer.allow_origin(tower_http::cors::AllowOrigin::list(std::iter::empty::<HeaderValue>()));
            } else if config.allowed_origins.iter().any(|o| o == "*") {
                layer = layer.allow_origin(tower_http::cors::Any);
            } else {
                let origins: Vec<HeaderValue> = config
                    .allowed_origins
                    .iter()
                    .filter_map(|o| o.parse().ok())
                    .collect();
                layer = layer.allow_origin(origins);
            }

            // Set allowed methods
            let methods: Vec<Method> = config
                .allowed_methods
                .iter()
                .filter_map(|m| m.parse().ok())
                .collect();
            layer = layer.allow_methods(methods);

            // Set allowed headers
            let headers: Vec<HeaderName> = config
                .allowed_headers
                .iter()
                .filter_map(|h| h.parse().ok())
                .collect();
            layer = layer.allow_headers(headers);

            // Set exposed headers
            let exposed: Vec<HeaderName> = config
                .exposed_headers
                .iter()
                .filter_map(|h| h.parse().ok())
                .collect();
            layer = layer.expose_headers(exposed);

            // Set credentials
            if config.allow_credentials {
                layer = layer.allow_credentials(true);
            }

            // Set max age
            layer = layer.max_age(config.max_age());

            layer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_development_allows_all() {
        let config = CorsConfig::development();
        assert!(config.is_origin_allowed("https://example.com"));
        assert!(config.is_origin_allowed("http://localhost:3000"));
    }

    #[test]
    fn test_restrictive_checks_origins() {
        let config = CorsConfig::production(vec!["https://app.example.com".to_string()]);

        assert!(config.is_origin_allowed("https://app.example.com"));
        assert!(!config.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_wildcard_subdomain() {
        let config = CorsConfig::production(vec!["*.example.com".to_string()]);

        assert!(config.is_origin_allowed("https://app.example.com"));
        assert!(config.is_origin_allowed("https://api.example.com"));
        assert!(!config.is_origin_allowed("https://example.org"));
    }
}

//! ruvector-server: REST API server for rUvector vector database
//!
//! This crate provides a REST API server built on axum for interacting with rUvector.
//!
//! ## Security Features (ADR-0011)
//!
//! - Configurable CORS policies (restrictive by default in production)
//! - Bearer token authentication support
//! - Rate limiting for API endpoints
//! - Path validation for file operations

pub mod error;
pub mod routes;
pub mod state;

use axum::{routing::get, Router};
use ruvector_security::{cors::build_cors_layer, AuthConfig, AuthMiddleware, CorsConfig, CorsMode, RateLimitConfig, RateLimiter};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::{compression::CompressionLayer, trace::TraceLayer};

pub use error::{Error, Result};
pub use state::AppState;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Enable CORS
    pub enable_cors: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// CORS configuration
    #[serde(default)]
    pub cors: CorsConfig,
    /// Authentication configuration
    #[serde(default)]
    pub auth: AuthConfig,
    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 6333,
            enable_cors: true,
            enable_compression: true,
            cors: CorsConfig::default(),
            auth: AuthConfig::default(),
            rate_limit: RateLimitConfig::default(),
        }
    }
}

impl Config {
    /// Create development configuration (permissive CORS, no auth)
    pub fn development() -> Self {
        Self {
            cors: CorsConfig {
                mode: CorsMode::Development,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create production configuration with security enabled
    pub fn production(allowed_origins: Vec<String>, auth_token: Option<String>) -> Self {
        Self {
            cors: CorsConfig {
                mode: CorsMode::Restrictive,
                allowed_origins,
                allow_credentials: true,
                ..Default::default()
            },
            auth: AuthConfig {
                mode: if auth_token.is_some() {
                    ruvector_security::AuthMode::Bearer
                } else {
                    ruvector_security::AuthMode::None
                },
                token: auth_token,
                allow_localhost: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Main server structure
pub struct RuvectorServer {
    config: Config,
    state: AppState,
}

impl RuvectorServer {
    /// Create a new server instance with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            state: AppState::new(),
        }
    }

    /// Create a new server instance with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            state: AppState::new(),
        }
    }

    /// Build the router with all routes
    fn build_router(&self) -> Router {
        let mut router = Router::new()
            .route("/health", get(routes::health::health_check))
            .route("/ready", get(routes::health::readiness))
            .nest("/collections", routes::collections::routes())
            .merge(routes::points::routes())
            .with_state(self.state.clone());

        // Add middleware layers
        router = router.layer(TraceLayer::new_for_http());

        if self.config.enable_compression {
            router = router.layer(CompressionLayer::new());
        }

        // Apply security CORS layer (S-2: Configurable CORS)
        if self.config.enable_cors {
            let cors = build_cors_layer(&self.config.cors);
            router = router.layer(cors);
        }

        router
    }

    /// Start the server
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to bind or start
    pub async fn start(self) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {}", e)))?;

        let router = self.build_router();

        tracing::info!("Starting ruvector-server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Server(format!("Failed to bind to {}: {}", addr, e)))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| Error::Server(format!("Server error: {}", e)))?;

        Ok(())
    }
}

impl Default for RuvectorServer {
    fn default() -> Self {
        Self::new()
    }
}

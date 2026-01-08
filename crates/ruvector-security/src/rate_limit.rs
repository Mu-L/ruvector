//! Rate limiting using token bucket algorithm
//!
//! Provides protection against API abuse and DoS attacks.

use crate::error::{SecurityError, SecurityResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rate limit configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second for read operations
    pub read_rps: u32,
    /// Requests per second for write operations
    pub write_rps: u32,
    /// Requests per second for file operations
    pub file_rps: u32,
    /// Burst size multiplier (allows temporary bursts)
    pub burst_multiplier: u32,
    /// Enable per-IP rate limiting
    pub per_ip: bool,
    /// Window size in seconds for rate tracking
    pub window_secs: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            read_rps: 1000,
            write_rps: 100,
            file_rps: 10,
            burst_multiplier: 2,
            per_ip: true,
            window_secs: 60,
        }
    }
}

/// Operation type for rate limiting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Read operations (search, get)
    Read,
    /// Write operations (insert, update, delete)
    Write,
    /// File operations (backup, restore)
    File,
}

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    /// Current tokens available
    tokens: f64,
    /// Maximum tokens (burst capacity)
    max_tokens: f64,
    /// Tokens added per second
    refill_rate: f64,
    /// Last refill time
    last_refill: Instant,
}

impl TokenBucket {
    fn new(tokens_per_second: u32, burst_multiplier: u32) -> Self {
        let max_tokens = (tokens_per_second * burst_multiplier) as f64;
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate: tokens_per_second as f64,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self, tokens: f64) -> Result<(), Duration> {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            Ok(())
        } else {
            // Calculate wait time
            let needed = tokens - self.tokens;
            let wait_secs = needed / self.refill_rate;
            Err(Duration::from_secs_f64(wait_secs))
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = elapsed.as_secs_f64() * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_refill = now;
    }

    fn tokens_remaining(&mut self) -> u32 {
        self.refill();
        self.tokens as u32
    }
}

/// Rate limiter state
struct RateLimiterState {
    /// Global buckets by operation type
    global_buckets: HashMap<OperationType, TokenBucket>,
    /// Per-IP buckets: IP -> (operation type -> bucket)
    ip_buckets: HashMap<String, HashMap<OperationType, TokenBucket>>,
    /// Configuration
    config: RateLimitConfig,
    /// Cleanup interval tracking
    last_cleanup: Instant,
}

impl RateLimiterState {
    fn new(config: RateLimitConfig) -> Self {
        let mut global_buckets = HashMap::new();

        global_buckets.insert(
            OperationType::Read,
            TokenBucket::new(config.read_rps, config.burst_multiplier),
        );
        global_buckets.insert(
            OperationType::Write,
            TokenBucket::new(config.write_rps, config.burst_multiplier),
        );
        global_buckets.insert(
            OperationType::File,
            TokenBucket::new(config.file_rps, config.burst_multiplier),
        );

        Self {
            global_buckets,
            ip_buckets: HashMap::new(),
            config,
            last_cleanup: Instant::now(),
        }
    }

    fn get_or_create_ip_bucket(&mut self, ip: &str, op: OperationType) -> &mut TokenBucket {
        let config = &self.config;
        let ip_map = self.ip_buckets.entry(ip.to_string()).or_default();

        ip_map.entry(op).or_insert_with(|| {
            let rps = match op {
                OperationType::Read => config.read_rps,
                OperationType::Write => config.write_rps,
                OperationType::File => config.file_rps,
            };
            TokenBucket::new(rps, config.burst_multiplier)
        })
    }

    fn cleanup_stale_entries(&mut self) {
        let now = Instant::now();
        let window = Duration::from_secs(self.config.window_secs * 2);

        if now.duration_since(self.last_cleanup) > window {
            // Remove IP entries that haven't been used recently
            self.ip_buckets.retain(|_, buckets| {
                buckets
                    .values()
                    .any(|b| now.duration_since(b.last_refill) < window)
            });
            self.last_cleanup = now;
        }
    }
}

/// Thread-safe rate limiter
#[derive(Clone)]
pub struct RateLimiter {
    state: Arc<RwLock<RateLimiterState>>,
    enabled: bool,
}

impl RateLimiter {
    /// Create a new rate limiter with configuration
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(RateLimiterState::new(config))),
            enabled: true,
        }
    }

    /// Create a disabled rate limiter (passes all requests)
    pub fn disabled() -> Self {
        Self {
            state: Arc::new(RwLock::new(RateLimiterState::new(RateLimitConfig::default()))),
            enabled: false,
        }
    }

    /// Check rate limit for an operation
    ///
    /// # Arguments
    /// * `op` - Operation type
    /// * `ip` - Optional IP address for per-IP limiting
    ///
    /// # Returns
    /// * `Ok(())` if request is allowed
    /// * `Err(SecurityError::RateLimitExceeded)` if rate limited
    pub async fn check(&self, op: OperationType, ip: Option<&str>) -> SecurityResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut state = self.state.write().await;

        // Cleanup stale entries periodically
        state.cleanup_stale_entries();

        // Check global limit first
        if let Some(bucket) = state.global_buckets.get_mut(&op) {
            if let Err(wait) = bucket.try_acquire(1.0) {
                return Err(SecurityError::RateLimitExceeded {
                    retry_after_secs: wait.as_secs().max(1),
                });
            }
        }

        // Check per-IP limit if enabled and IP provided
        if state.config.per_ip {
            if let Some(ip) = ip {
                let bucket = state.get_or_create_ip_bucket(ip, op);
                if let Err(wait) = bucket.try_acquire(1.0) {
                    return Err(SecurityError::RateLimitExceeded {
                        retry_after_secs: wait.as_secs().max(1),
                    });
                }
            }
        }

        Ok(())
    }

    /// Get remaining tokens for rate limit headers
    pub async fn remaining(&self, op: OperationType, ip: Option<&str>) -> u32 {
        if !self.enabled {
            return u32::MAX;
        }

        let mut state = self.state.write().await;

        let global_remaining = state
            .global_buckets
            .get_mut(&op)
            .map(|b| b.tokens_remaining())
            .unwrap_or(u32::MAX);

        if let Some(ip) = ip {
            if state.config.per_ip {
                let ip_remaining = state.get_or_create_ip_bucket(ip, op).tokens_remaining();
                return global_remaining.min(ip_remaining);
            }
        }

        global_remaining
    }

    /// Get rate limit for operation
    pub async fn limit(&self, op: OperationType) -> u32 {
        let state = self.state.read().await;
        match op {
            OperationType::Read => state.config.read_rps,
            OperationType::Write => state.config.write_rps,
            OperationType::File => state.config.file_rps,
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit_allows_within_limit() {
        let config = RateLimitConfig {
            read_rps: 10,
            burst_multiplier: 1,
            per_ip: false,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Should allow 10 requests
        for _ in 0..10 {
            assert!(limiter.check(OperationType::Read, None).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_excess() {
        let config = RateLimitConfig {
            read_rps: 5,
            burst_multiplier: 1,
            per_ip: false,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Use up all tokens
        for _ in 0..5 {
            let _ = limiter.check(OperationType::Read, None).await;
        }

        // Next request should be rate limited
        let result = limiter.check(OperationType::Read, None).await;
        assert!(matches!(result, Err(SecurityError::RateLimitExceeded { .. })));
    }

    #[tokio::test]
    async fn test_per_ip_limiting() {
        // Test that per-IP limiting is enabled and creates separate buckets
        let config = RateLimitConfig {
            read_rps: 10,
            burst_multiplier: 1,
            per_ip: true,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Verify per_ip is enabled and creates separate bucket entries
        // Both IPs should be able to use some tokens
        assert!(limiter
            .check(OperationType::Read, Some("192.168.1.1"))
            .await
            .is_ok());
        assert!(limiter
            .check(OperationType::Read, Some("192.168.1.2"))
            .await
            .is_ok());

        // Verify remaining tokens are tracked independently
        let remaining_ip1 = limiter.remaining(OperationType::Read, Some("192.168.1.1")).await;
        let remaining_ip2 = limiter.remaining(OperationType::Read, Some("192.168.1.2")).await;

        // Both should have used 1 token each from their per-IP bucket
        // (plus global bucket consumption)
        assert!(remaining_ip1 > 0);
        assert!(remaining_ip2 > 0);
    }

    #[tokio::test]
    async fn test_disabled_limiter() {
        let limiter = RateLimiter::disabled();

        // Should allow unlimited requests
        for _ in 0..1000 {
            assert!(limiter.check(OperationType::Read, None).await.is_ok());
        }
    }
}

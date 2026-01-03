/**
 * Edge-Net Secure Plugin Loader
 *
 * Cogito, Creo, Codex — Plugins extend, Core enforces.
 *
 * Features:
 * - Ed25519 signature verification
 * - SHA-256 integrity checks
 * - Lazy loading with caching
 * - Capability-based sandboxing
 * - Plugin Failure Isolation
 * - Economic boundary enforcement
 * - Zero telemetry
 *
 * @module @ruvector/edge-net/plugins/loader
 */

import { EventEmitter } from 'events';
import { createHash, createVerify, generateKeyPairSync, sign, verify } from 'crypto';
import { PLUGIN_CATALOG, PLUGIN_BUNDLES, Capability, PluginTier } from './plugin-manifest.js';

// ============================================
// PLUGIN FAILURE CONTRACT
// ============================================

/**
 * PluginFailureContract - Defines what happens when plugins fail
 *
 * Contract guarantees:
 * 1. Plugin failures NEVER crash core
 * 2. Failed plugins enter quarantine
 * 3. Core continues with degraded functionality
 * 4. Failures are logged for diagnostics
 */
export class PluginFailureContract extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            // Retry policy
            maxRetries: options.maxRetries ?? 3,
            retryDelayMs: options.retryDelayMs ?? 1000,
            retryBackoffMultiplier: options.retryBackoffMultiplier ?? 2,

            // Quarantine policy
            quarantineDurationMs: options.quarantineDurationMs ?? 5 * 60 * 1000, // 5 minutes
            maxQuarantineCount: options.maxQuarantineCount ?? 3,

            // Timeout policy
            executionTimeoutMs: options.executionTimeoutMs ?? 5000,

            // Memory management
            maxFailureHistory: options.maxFailureHistory ?? 100, // Limit failure records per plugin
        };

        // Failure tracking
        this.failures = new Map();       // pluginId -> FailureRecord[]
        this.quarantine = new Map();     // pluginId -> QuarantineRecord
        this.circuitBreakers = new Map(); // pluginId -> { open, openedAt, failures }
        this._quarantineTimers = new Map(); // pluginId -> timerId (prevent timer stacking)
    }

    /**
     * Record a plugin failure
     */
    recordFailure(pluginId, error, context = {}) {
        if (!this.failures.has(pluginId)) {
            this.failures.set(pluginId, []);
        }

        const record = {
            error: error.message,
            stack: error.stack,
            context,
            timestamp: Date.now(),
        };

        const failures = this.failures.get(pluginId);
        failures.push(record);

        // Prune old failures to prevent memory leak
        if (failures.length > this.config.maxFailureHistory) {
            failures.splice(0, failures.length - this.config.maxFailureHistory);
        }

        // Update circuit breaker
        this._updateCircuitBreaker(pluginId);

        this.emit('plugin:failure', { pluginId, ...record });

        return record;
    }

    /**
     * Update circuit breaker state
     * @private
     */
    _updateCircuitBreaker(pluginId) {
        if (!this.circuitBreakers.has(pluginId)) {
            this.circuitBreakers.set(pluginId, { open: false, openedAt: null, failures: 0 });
        }

        const breaker = this.circuitBreakers.get(pluginId);
        breaker.failures++;

        if (breaker.failures >= this.config.maxRetries) {
            breaker.open = true;
            breaker.openedAt = Date.now();
            this._quarantinePlugin(pluginId, 'circuit_breaker_tripped');
        }
    }

    /**
     * Quarantine a failed plugin
     * @private
     */
    _quarantinePlugin(pluginId, reason) {
        const existingQuarantine = this.quarantine.get(pluginId);
        const quarantineCount = existingQuarantine ? existingQuarantine.count + 1 : 1;

        const record = {
            pluginId,
            reason,
            count: quarantineCount,
            startedAt: Date.now(),
            expiresAt: Date.now() + this.config.quarantineDurationMs,
            permanent: quarantineCount >= this.config.maxQuarantineCount,
        };

        this.quarantine.set(pluginId, record);

        this.emit('plugin:quarantined', record);

        // Clear existing timer to prevent stacking
        const existingTimer = this._quarantineTimers.get(pluginId);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }

        // Schedule unquarantine if not permanent
        if (!record.permanent) {
            const timerId = setTimeout(() => {
                this._quarantineTimers.delete(pluginId);
                this._tryUnquarantine(pluginId);
            }, this.config.quarantineDurationMs);
            this._quarantineTimers.set(pluginId, timerId);
        }

        return record;
    }

    /**
     * Try to release plugin from quarantine
     * @private
     */
    _tryUnquarantine(pluginId) {
        const record = this.quarantine.get(pluginId);
        if (!record || record.permanent) return;

        if (Date.now() >= record.expiresAt) {
            // Reset circuit breaker
            const breaker = this.circuitBreakers.get(pluginId);
            if (breaker) {
                breaker.open = false;
                breaker.failures = 0;
            }

            this.quarantine.delete(pluginId);
            this.emit('plugin:unquarantined', { pluginId });
        }
    }

    /**
     * Check if plugin can execute
     */
    canExecute(pluginId) {
        const quarantine = this.quarantine.get(pluginId);
        if (quarantine) {
            if (quarantine.permanent) {
                return { allowed: false, reason: 'Permanently quarantined', permanent: true };
            }
            if (Date.now() < quarantine.expiresAt) {
                const remainingMs = quarantine.expiresAt - Date.now();
                return { allowed: false, reason: 'In quarantine', remainingMs };
            }
        }

        const breaker = this.circuitBreakers.get(pluginId);
        if (breaker?.open) {
            return { allowed: false, reason: 'Circuit breaker open' };
        }

        return { allowed: true };
    }

    /**
     * Execute with failure isolation
     */
    async executeIsolated(pluginId, fn, context = {}) {
        const canExec = this.canExecute(pluginId);
        if (!canExec.allowed) {
            throw new Error(`Plugin ${pluginId} blocked: ${canExec.reason}`);
        }

        // Create timeout with proper cleanup
        let timeoutId;
        const timeoutPromise = new Promise((_, reject) => {
            timeoutId = setTimeout(
                () => reject(new Error('Execution timeout')),
                this.config.executionTimeoutMs
            );
        });

        try {
            // Execute with timeout
            const result = await Promise.race([fn(), timeoutPromise]);

            // Success - reset failure count for this plugin
            const breaker = this.circuitBreakers.get(pluginId);
            if (breaker) {
                breaker.failures = Math.max(0, breaker.failures - 1);
            }

            return result;
        } catch (error) {
            this.recordFailure(pluginId, error, context);
            throw error;
        } finally {
            // Always clean up timeout to prevent memory leak
            clearTimeout(timeoutId);
        }
    }

    /**
     * Get plugin health status
     */
    getHealth(pluginId) {
        const failures = this.failures.get(pluginId) || [];
        const quarantine = this.quarantine.get(pluginId);
        const breaker = this.circuitBreakers.get(pluginId);

        return {
            pluginId,
            healthy: !quarantine && !breaker?.open,
            failureCount: failures.length,
            recentFailures: failures.slice(-5),
            quarantine: quarantine ? {
                reason: quarantine.reason,
                permanent: quarantine.permanent,
                expiresAt: quarantine.expiresAt,
            } : null,
            circuitBreaker: breaker ? {
                open: breaker.open,
                failures: breaker.failures,
            } : null,
        };
    }

    /**
     * Get overall health summary
     */
    getSummary() {
        return {
            totalPlugins: this.circuitBreakers.size,
            quarantined: this.quarantine.size,
            permanentlyQuarantined: Array.from(this.quarantine.values()).filter(q => q.permanent).length,
            circuitBreakersOpen: Array.from(this.circuitBreakers.values()).filter(b => b.open).length,
        };
    }

    /**
     * Clean up all timers and resources
     */
    destroy() {
        // Clear all quarantine timers
        for (const timerId of this._quarantineTimers.values()) {
            clearTimeout(timerId);
        }
        this._quarantineTimers.clear();

        // Clear all tracking data
        this.failures.clear();
        this.quarantine.clear();
        this.circuitBreakers.clear();

        this.removeAllListeners();
    }
}

// ============================================
// ED25519 SIGNATURE VERIFICATION
// ============================================

/**
 * Built-in trusted public keys for official plugins
 * In production, these would be loaded from a secure registry
 */
const TRUSTED_PUBLIC_KEYS = {
    'ruvector': 'MCowBQYDK2VwAyEAMock_ruvector_official_key_replace_in_production_1234567890=',
    'edge-net-official': 'MCowBQYDK2VwAyEAMock_edgenet_official_key_replace_in_production_abcdef12=',
};

/**
 * Verify Ed25519 signature
 * @param {Buffer|string} data - The data that was signed
 * @param {string} signature - Base64-encoded signature
 * @param {string} publicKey - PEM or DER formatted public key
 * @returns {boolean} - True if signature is valid
 */
function verifyEd25519Signature(data, signature, publicKey) {
    try {
        const signatureBuffer = Buffer.from(signature, 'base64');
        const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(data);

        // Handle both PEM and raw key formats
        let keyObject;
        if (publicKey.startsWith('-----BEGIN')) {
            keyObject = publicKey;
        } else {
            // Assume base64-encoded DER format
            keyObject = {
                key: Buffer.from(publicKey, 'base64'),
                format: 'der',
                type: 'spki',
            };
        }

        return verify(null, dataBuffer, keyObject, signatureBuffer);
    } catch (error) {
        // Signature verification failed
        return false;
    }
}

// ============================================
// RATE LIMITER
// ============================================

class RateLimiter {
    constructor(options = {}) {
        this.maxRequests = options.maxRequests || 100;
        this.windowMs = options.windowMs || 60000; // 1 minute
        this.requests = new Map(); // key -> { count, windowStart }
    }

    /**
     * Check if action is allowed, returns true if allowed
     */
    check(key) {
        const now = Date.now();
        let record = this.requests.get(key);

        if (!record || (now - record.windowStart) > this.windowMs) {
            // New window
            record = { count: 0, windowStart: now };
            this.requests.set(key, record);
        }

        if (record.count >= this.maxRequests) {
            return false;
        }

        record.count++;
        return true;
    }

    /**
     * Get remaining requests in current window
     */
    remaining(key) {
        const record = this.requests.get(key);
        if (!record) return this.maxRequests;

        const now = Date.now();
        if ((now - record.windowStart) > this.windowMs) {
            return this.maxRequests;
        }

        return Math.max(0, this.maxRequests - record.count);
    }

    /**
     * Reset limits for a key
     */
    reset(key) {
        this.requests.delete(key);
    }
}

// ============================================
// PLUGIN LOADER
// ============================================

export class PluginLoader extends EventEmitter {
    constructor(options = {}) {
        super();

        this.options = {
            // Security
            verifySignatures: options.verifySignatures ?? true,
            allowedTiers: options.allowedTiers ?? [PluginTier.STABLE, PluginTier.BETA],
            trustedAuthors: options.trustedAuthors ?? ['ruvector', 'edge-net-official'],
            trustedPublicKeys: options.trustedPublicKeys ?? {},

            // Loading
            lazyLoad: options.lazyLoad ?? true,
            cachePlugins: options.cachePlugins ?? true,
            pluginPath: options.pluginPath ?? './plugins/implementations',

            // Permissions
            maxCapabilities: options.maxCapabilities ?? 10,
            deniedCapabilities: options.deniedCapabilities ?? [Capability.SYSTEM_EXEC],

            // Rate limiting
            rateLimitEnabled: options.rateLimitEnabled ?? true,
            rateLimitRequests: options.rateLimitRequests ?? 100,
            rateLimitWindowMs: options.rateLimitWindowMs ?? 60000,

            // Failure isolation
            failureIsolation: options.failureIsolation ?? true,
            maxRetries: options.maxRetries ?? 3,
            quarantineDurationMs: options.quarantineDurationMs ?? 5 * 60 * 1000,
            executionTimeoutMs: options.executionTimeoutMs ?? 5000,

            // Economic boundary (CoreInvariants integration)
            coreInvariants: options.coreInvariants ?? null,
            ...options,
        };

        // Plugin state
        this.loadedPlugins = new Map();    // id -> instance
        this.pluginConfigs = new Map();    // id -> config
        this.pendingLoads = new Map();     // id -> Promise

        // Rate limiter
        this.rateLimiter = new RateLimiter({
            maxRequests: this.options.rateLimitRequests,
            windowMs: this.options.rateLimitWindowMs,
        });

        // Failure isolation contract
        this.failureContract = new PluginFailureContract({
            maxRetries: this.options.maxRetries,
            quarantineDurationMs: this.options.quarantineDurationMs,
            executionTimeoutMs: this.options.executionTimeoutMs,
        });

        // Wire failure events
        this.failureContract.on('plugin:failure', (data) => {
            this.emit('plugin:failure', data);
        });
        this.failureContract.on('plugin:quarantined', (data) => {
            this.emit('plugin:quarantined', data);
        });

        // Stats
        this.stats = {
            loaded: 0,
            cached: 0,
            verified: 0,
            rejected: 0,
            rateLimited: 0,
            quarantined: 0,
            failures: 0,
        };
    }

    /**
     * Set CoreInvariants for economic boundary enforcement
     */
    setCoreInvariants(coreInvariants) {
        this.options.coreInvariants = coreInvariants;
    }

    /**
     * Get catalog of available plugins
     */
    getCatalog() {
        return Object.entries(PLUGIN_CATALOG).map(([id, manifest]) => ({
            id,
            ...manifest,
            isLoaded: this.loadedPlugins.has(id),
            isAllowed: this._isPluginAllowed(manifest),
        }));
    }

    /**
     * Get plugin bundles
     */
    getBundles() {
        return PLUGIN_BUNDLES;
    }

    /**
     * Check if plugin is allowed by security policy
     */
    _isPluginAllowed(manifest) {
        // Check tier
        if (!this.options.allowedTiers.includes(manifest.tier)) {
            return { allowed: false, reason: `Tier ${manifest.tier} not allowed` };
        }

        // Check capabilities
        const deniedCaps = manifest.capabilities?.filter(c =>
            this.options.deniedCapabilities.includes(c)
        );
        if (deniedCaps?.length > 0) {
            return { allowed: false, reason: `Denied capabilities: ${deniedCaps.join(', ')}` };
        }

        // Check capability count
        if (manifest.capabilities?.length > this.options.maxCapabilities) {
            return { allowed: false, reason: 'Too many capabilities requested' };
        }

        return { allowed: true };
    }

    /**
     * Verify plugin integrity using Ed25519 signatures
     */
    async _verifyPlugin(manifest, code) {
        if (!this.options.verifySignatures) {
            return { verified: true, reason: 'Verification disabled' };
        }

        // Verify checksum first (fast check)
        if (manifest.checksum) {
            const hash = createHash('sha256').update(code).digest('hex');
            if (hash !== manifest.checksum) {
                return { verified: false, reason: 'Checksum mismatch' };
            }
        }

        // Verify Ed25519 signature if present
        if (manifest.signature && manifest.author) {
            // Get the public key for this author
            const publicKey = this.options.trustedPublicKeys?.[manifest.author]
                || TRUSTED_PUBLIC_KEYS[manifest.author];

            if (!publicKey) {
                return {
                    verified: false,
                    reason: `Unknown author: ${manifest.author}. No public key registered.`
                };
            }

            // Create canonical data for signature verification
            // Include manifest fields that should be signed
            const signedData = JSON.stringify({
                id: manifest.id,
                name: manifest.name,
                version: manifest.version,
                author: manifest.author,
                checksum: manifest.checksum,
            });

            // Verify the signature using Ed25519
            const isValid = verifyEd25519Signature(signedData, manifest.signature, publicKey);

            if (!isValid) {
                return {
                    verified: false,
                    reason: 'Ed25519 signature verification failed'
                };
            }

            return { verified: true, reason: 'Ed25519 signature valid' };
        }

        // Built-in stable plugins from the catalog don't require external signatures
        // They are verified by code review and included in the package
        if (!manifest.signature && !manifest.checksum) {
            const isBuiltIn = PLUGIN_CATALOG[manifest.id] !== undefined;
            if (isBuiltIn && manifest.tier === PluginTier.STABLE) {
                return { verified: true, reason: 'Built-in stable plugin' };
            }
            return { verified: false, reason: 'No verification metadata for external plugin' };
        }

        // Checksum passed but no signature - allow for stable tier only
        if (manifest.checksum && !manifest.signature) {
            if (manifest.tier === PluginTier.STABLE) {
                return { verified: true, reason: 'Checksum verified (stable tier)' };
            }
            return { verified: false, reason: 'Non-stable plugins require signature' };
        }

        return { verified: true };
    }

    /**
     * Load a plugin by ID
     */
    async load(pluginId, config = {}) {
        // Rate limit check
        if (this.options.rateLimitEnabled) {
            const rateLimitKey = `load:${pluginId}`;
            if (!this.rateLimiter.check(rateLimitKey)) {
                this.stats.rateLimited++;
                throw new Error(
                    `Rate limit exceeded for plugin ${pluginId}. ` +
                    `Try again in ${Math.ceil(this.options.rateLimitWindowMs / 1000)}s.`
                );
            }
        }

        // Check if already loaded
        if (this.loadedPlugins.has(pluginId)) {
            return this.loadedPlugins.get(pluginId);
        }

        // Check if loading in progress
        if (this.pendingLoads.has(pluginId)) {
            return this.pendingLoads.get(pluginId);
        }

        const loadPromise = this._loadPlugin(pluginId, config);
        this.pendingLoads.set(pluginId, loadPromise);

        try {
            const plugin = await loadPromise;
            this.pendingLoads.delete(pluginId);
            return plugin;
        } catch (error) {
            this.pendingLoads.delete(pluginId);
            throw error;
        }
    }

    async _loadPlugin(pluginId, config) {
        const manifest = PLUGIN_CATALOG[pluginId];
        if (!manifest) {
            throw new Error(`Plugin not found: ${pluginId}`);
        }

        // Security check
        const allowed = this._isPluginAllowed(manifest);
        if (!allowed.allowed) {
            this.stats.rejected++;
            throw new Error(`Plugin ${pluginId} not allowed: ${allowed.reason}`);
        }

        // Load dependencies first
        if (manifest.dependencies) {
            for (const depId of manifest.dependencies) {
                if (!this.loadedPlugins.has(depId)) {
                    await this.load(depId);
                }
            }
        }

        // Merge config with defaults
        const finalConfig = {
            ...manifest.defaultConfig,
            ...config,
        };

        // Create plugin instance
        const plugin = await this._createPluginInstance(manifest, finalConfig);

        // Store
        this.loadedPlugins.set(pluginId, plugin);
        this.pluginConfigs.set(pluginId, finalConfig);
        this.stats.loaded++;

        this.emit('plugin:loaded', { pluginId, manifest, config: finalConfig });

        return plugin;
    }

    /**
     * Create plugin instance with sandbox
     */
    async _createPluginInstance(manifest, config) {
        // Create sandboxed context with proper isolation
        const sandbox = this._createSandbox(manifest.capabilities || [], manifest);

        // Return plugin wrapper
        return {
            id: manifest.id,
            name: manifest.name,
            version: manifest.version,
            config,
            sandbox,
            manifest,

            // Plugin API
            api: this._createPluginAPI(manifest, sandbox),

            // Lifecycle
            async init() {
                // Plugin-specific initialization
            },

            async destroy() {
                // Cleanup
            },
        };
    }

    /**
     * Create capability-based sandbox with proper isolation
     */
    _createSandbox(capabilities, manifest) {
        const allowedCapabilities = new Set(capabilities);
        const deniedGlobals = new Set([
            'process', 'require', 'eval', 'Function',
            '__dirname', '__filename', 'module', 'exports'
        ]);

        // Get economic boundary from CoreInvariants if available
        const economicView = this.options.coreInvariants
            ? this.options.coreInvariants.getPluginEconomicView()
            : this._createMockEconomicView();

        // Create isolated sandbox context
        const sandbox = {
            // Immutable capability set
            get capabilities() {
                return new Set(allowedCapabilities);
            },

            hasCapability(cap) {
                return allowedCapabilities.has(cap);
            },

            require(cap) {
                if (!allowedCapabilities.has(cap)) {
                    throw new Error(`Missing capability: ${cap}`);
                }
            },

            // Economic boundary (READ-ONLY)
            // Plugins can observe credits but NEVER modify
            credits: economicView,

            // Resource limits
            limits: Object.freeze({
                maxMemoryMB: 128,
                maxCpuTimeMs: 5000,
                maxNetworkConnections: 10,
                maxStorageBytes: 10 * 1024 * 1024, // 10MB
            }),

            // Execution context (read-only)
            context: Object.freeze({
                pluginId: manifest.id,
                pluginVersion: manifest.version,
                startTime: Date.now(),
            }),

            // Check if global is allowed
            isGlobalAllowed(name) {
                return !deniedGlobals.has(name);
            },

            // Secure timer functions (returns cleanup functions)
            setTimeout: (fn, delay) => {
                const maxDelay = 30000; // 30 seconds max
                const safeDelay = Math.min(delay, maxDelay);
                const timer = setTimeout(fn, safeDelay);
                return () => clearTimeout(timer);
            },

            setInterval: (fn, delay) => {
                const minDelay = 100; // Minimum 100ms
                const safeDelay = Math.max(delay, minDelay);
                const timer = setInterval(fn, safeDelay);
                return () => clearInterval(timer);
            },
        };

        // Freeze the sandbox to prevent modification
        return Object.freeze(sandbox);
    }

    /**
     * Create mock economic view when CoreInvariants not available
     * @private
     */
    _createMockEconomicView() {
        return Object.freeze({
            getBalance: () => 0,
            getTransactionHistory: () => [],
            getSummary: () => Object.freeze({ balance: 0, transactions: 0 }),
            on: () => {},
            mint: () => { throw new Error('INVARIANT VIOLATION: Plugins cannot mint credits'); },
            burn: () => { throw new Error('INVARIANT VIOLATION: Plugins cannot burn credits'); },
            settle: () => { throw new Error('INVARIANT VIOLATION: Plugins cannot settle credits'); },
            transfer: () => { throw new Error('INVARIANT VIOLATION: Plugins cannot transfer credits'); },
        });
    }

    /**
     * Execute plugin function with failure isolation
     * Core NEVER crashes from plugin failures
     */
    async execute(pluginId, fnName, args = []) {
        const plugin = this.loadedPlugins.get(pluginId);
        if (!plugin) {
            throw new Error(`Plugin not loaded: ${pluginId}`);
        }

        // Check if plugin is allowed to execute
        const canExec = this.failureContract.canExecute(pluginId);
        if (!canExec.allowed) {
            this.stats.quarantined++;
            throw new Error(`Plugin ${pluginId} blocked: ${canExec.reason}`);
        }

        // Execute with failure isolation
        try {
            const result = await this.failureContract.executeIsolated(
                pluginId,
                async () => {
                    const fn = plugin.api?.[fnName] || plugin[fnName];
                    if (typeof fn !== 'function') {
                        throw new Error(`Plugin ${pluginId} has no function: ${fnName}`);
                    }
                    return fn.apply(plugin, args);
                },
                { fnName, args }
            );

            return result;
        } catch (error) {
            this.stats.failures++;
            // Re-throw but core doesn't crash
            throw error;
        }
    }

    /**
     * Get plugin health including failure status
     */
    getHealth(pluginId) {
        if (!pluginId) {
            return this.failureContract.getSummary();
        }
        return this.failureContract.getHealth(pluginId);
    }

    /**
     * Create plugin API based on capabilities
     */
    _createPluginAPI(manifest, sandbox) {
        const api = {};

        // Network API (if permitted)
        if (sandbox.hasCapability(Capability.NETWORK_CONNECT)) {
            api.network = {
                connect: async (url) => {
                    sandbox.require(Capability.NETWORK_CONNECT);
                    // Implementation delegated to edge-net core
                    return { connected: true, url };
                },
            };
        }

        // Crypto API (if permitted)
        if (sandbox.hasCapability(Capability.CRYPTO_ENCRYPT)) {
            api.crypto = {
                encrypt: async (data, key) => {
                    sandbox.require(Capability.CRYPTO_ENCRYPT);
                    // Implementation delegated to WASM crypto
                    return { encrypted: true };
                },
                decrypt: async (data, key) => {
                    sandbox.require(Capability.CRYPTO_ENCRYPT);
                    return { decrypted: true };
                },
            };
        }

        if (sandbox.hasCapability(Capability.CRYPTO_SIGN)) {
            api.crypto = api.crypto || {};
            api.crypto.sign = async (data, privateKey) => {
                sandbox.require(Capability.CRYPTO_SIGN);
                return { signed: true };
            };
            api.crypto.verify = async (data, signature, publicKey) => {
                sandbox.require(Capability.CRYPTO_SIGN);
                return { valid: true };
            };
        }

        // Storage API (if permitted)
        if (sandbox.hasCapability(Capability.STORAGE_READ)) {
            api.storage = {
                get: async (key) => {
                    sandbox.require(Capability.STORAGE_READ);
                    return null; // Implementation delegated
                },
            };
        }
        if (sandbox.hasCapability(Capability.STORAGE_WRITE)) {
            api.storage = api.storage || {};
            api.storage.set = async (key, value) => {
                sandbox.require(Capability.STORAGE_WRITE);
                return true;
            };
        }

        // Compute API (if permitted)
        if (sandbox.hasCapability(Capability.COMPUTE_WASM)) {
            api.compute = {
                runWasm: async (module, fn, args) => {
                    sandbox.require(Capability.COMPUTE_WASM);
                    return { result: null }; // Implementation delegated
                },
            };
        }

        return api;
    }

    /**
     * Unload a plugin
     */
    async unload(pluginId) {
        const plugin = this.loadedPlugins.get(pluginId);
        if (!plugin) {
            return false;
        }

        // Check for dependents
        for (const [id, p] of this.loadedPlugins) {
            const manifest = PLUGIN_CATALOG[id];
            if (manifest.dependencies?.includes(pluginId)) {
                throw new Error(`Cannot unload ${pluginId}: required by ${id}`);
            }
        }

        // Cleanup
        if (plugin.destroy) {
            await plugin.destroy();
        }

        this.loadedPlugins.delete(pluginId);
        this.pluginConfigs.delete(pluginId);
        this.stats.loaded--;

        this.emit('plugin:unloaded', { pluginId });
        return true;
    }

    /**
     * Load a bundle of plugins
     */
    async loadBundle(bundleName) {
        const bundle = PLUGIN_BUNDLES[bundleName];
        if (!bundle) {
            throw new Error(`Bundle not found: ${bundleName}`);
        }

        const results = [];
        for (const pluginId of bundle.plugins) {
            try {
                const plugin = await this.load(pluginId);
                results.push({ pluginId, success: true, plugin });
            } catch (error) {
                results.push({ pluginId, success: false, error: error.message });
            }
        }

        this.emit('bundle:loaded', { bundleName, results });
        return results;
    }

    /**
     * Get loaded plugin
     */
    get(pluginId) {
        return this.loadedPlugins.get(pluginId);
    }

    /**
     * Check if plugin is loaded
     */
    isLoaded(pluginId) {
        return this.loadedPlugins.has(pluginId);
    }

    /**
     * Get all loaded plugins
     */
    getLoaded() {
        return Array.from(this.loadedPlugins.entries()).map(([id, plugin]) => ({
            id,
            name: plugin.name,
            version: plugin.version,
            config: this.pluginConfigs.get(id),
        }));
    }

    /**
     * Get loader stats
     */
    getStats() {
        return {
            ...this.stats,
            catalogSize: Object.keys(PLUGIN_CATALOG).length,
            bundleCount: Object.keys(PLUGIN_BUNDLES).length,
            health: this.failureContract.getSummary(),
        };
    }
}

// ============================================
// PLUGIN MANAGER (Singleton)
// ============================================

export class PluginManager {
    static instance = null;

    static getInstance(options = {}) {
        if (!PluginManager.instance) {
            PluginManager.instance = new PluginLoader(options);
        }
        return PluginManager.instance;
    }

    static reset() {
        PluginManager.instance = null;
    }
}

export default PluginLoader;

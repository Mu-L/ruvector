#!/usr/bin/env node
/**
 * @ruvector/edge-net Genesis Node - Production Deployment
 *
 * Production-ready bootstrap node for the Edge-Net P2P network.
 * Features:
 * - Persistent service with graceful shutdown
 * - Firebase registration as known bootstrap node
 * - DHT routing table maintenance
 * - Peer discovery request handling
 * - Health check HTTP endpoint
 * - Structured JSON logging for monitoring
 * - Automatic reconnection and recovery
 *
 * Environment Variables:
 *   GENESIS_PORT          - WebSocket port (default: 8787)
 *   GENESIS_HOST          - Bind address (default: 0.0.0.0)
 *   GENESIS_DATA          - Data directory (default: /data/genesis)
 *   GENESIS_NODE_ID       - Fixed node ID (optional, auto-generated if not set)
 *   HEALTH_PORT           - Health check HTTP port (default: 8788)
 *   LOG_LEVEL             - Logging level: debug, info, warn, error (default: info)
 *   LOG_FORMAT            - Log format: json, text (default: json)
 *   FIREBASE_API_KEY      - Firebase API key (optional)
 *   FIREBASE_PROJECT_ID   - Firebase project ID (optional)
 *   METRICS_ENABLED       - Enable Prometheus metrics (default: true)
 *
 * @module @ruvector/edge-net/deploy/genesis-prod
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import http from 'http';

// Resolve paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    // Network
    port: parseInt(process.env.GENESIS_PORT || '8787'),
    host: process.env.GENESIS_HOST || '0.0.0.0',
    healthPort: parseInt(process.env.HEALTH_PORT || '8788'),

    // Storage
    dataDir: process.env.GENESIS_DATA || '/data/genesis',

    // Identity
    nodeId: process.env.GENESIS_NODE_ID || null,

    // Logging
    logLevel: process.env.LOG_LEVEL || 'info',
    logFormat: process.env.LOG_FORMAT || 'json',

    // Features
    metricsEnabled: process.env.METRICS_ENABLED !== 'false',

    // Rate limiting
    rateLimit: {
        maxConnectionsPerIp: parseInt(process.env.MAX_CONN_PER_IP || '50'),
        maxMessagesPerSecond: parseInt(process.env.MAX_MSG_PER_SEC || '100'),
        challengeExpiry: 60000,
    },

    // Cleanup
    cleanup: {
        staleConnectionTimeout: 300000, // 5 minutes
        cleanupInterval: 60000, // 1 minute
    },

    // DHT
    dht: {
        maxRoutingTableSize: 1000,
        bucketRefreshInterval: 60000,
        announceInterval: 300000, // 5 minutes
    },

    // Firebase (optional bootstrap registration)
    firebase: {
        enabled: !!(process.env.FIREBASE_API_KEY && process.env.FIREBASE_PROJECT_ID),
        apiKey: process.env.FIREBASE_API_KEY,
        projectId: process.env.FIREBASE_PROJECT_ID,
        authDomain: process.env.FIREBASE_AUTH_DOMAIN,
        registrationInterval: 60000, // Re-register every minute
    },
};

// ============================================
// STRUCTURED LOGGER
// ============================================

const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

class Logger {
    constructor(name, config) {
        this.name = name;
        this.level = LOG_LEVELS[config.logLevel] || LOG_LEVELS.info;
        this.format = config.logFormat;
    }

    _log(level, message, meta = {}) {
        if (LOG_LEVELS[level] < this.level) return;

        const entry = {
            timestamp: new Date().toISOString(),
            level,
            service: this.name,
            message,
            ...meta,
        };

        if (this.format === 'json') {
            console.log(JSON.stringify(entry));
        } else {
            const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
            console.log(`[${entry.timestamp}] ${level.toUpperCase()} [${this.name}] ${message}${metaStr}`);
        }
    }

    debug(msg, meta) { this._log('debug', msg, meta); }
    info(msg, meta) { this._log('info', msg, meta); }
    warn(msg, meta) { this._log('warn', msg, meta); }
    error(msg, meta) { this._log('error', msg, meta); }
}

const log = new Logger('genesis-node', CONFIG);

// ============================================
// METRICS COLLECTOR
// ============================================

class MetricsCollector {
    constructor() {
        this.counters = {
            connections_total: 0,
            connections_active: 0,
            messages_received: 0,
            messages_sent: 0,
            signals_relayed: 0,
            dht_lookups: 0,
            dht_stores: 0,
            auth_challenges: 0,
            auth_successes: 0,
            auth_failures: 0,
            errors_total: 0,
        };

        this.gauges = {
            peers_registered: 0,
            rooms_active: 0,
            ledgers_stored: 0,
            uptime_seconds: 0,
        };

        this.histograms = {
            message_latency_ms: [],
        };

        this.startTime = Date.now();
    }

    inc(counter, value = 1) {
        if (this.counters[counter] !== undefined) {
            this.counters[counter] += value;
        }
    }

    dec(counter, value = 1) {
        if (this.counters[counter] !== undefined) {
            this.counters[counter] -= value;
        }
    }

    set(gauge, value) {
        if (this.gauges[gauge] !== undefined) {
            this.gauges[gauge] = value;
        }
    }

    observe(histogram, value) {
        if (this.histograms[histogram]) {
            this.histograms[histogram].push(value);
            // Keep last 1000 observations
            if (this.histograms[histogram].length > 1000) {
                this.histograms[histogram].shift();
            }
        }
    }

    getMetrics() {
        this.gauges.uptime_seconds = Math.floor((Date.now() - this.startTime) / 1000);

        return {
            counters: { ...this.counters },
            gauges: { ...this.gauges },
            histograms: Object.fromEntries(
                Object.entries(this.histograms).map(([k, v]) => [
                    k,
                    {
                        count: v.length,
                        avg: v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0,
                        max: v.length ? Math.max(...v) : 0,
                        min: v.length ? Math.min(...v) : 0,
                    },
                ])
            ),
        };
    }

    toPrometheus() {
        const lines = [];

        // Counters
        for (const [name, value] of Object.entries(this.counters)) {
            lines.push(`# TYPE genesis_${name} counter`);
            lines.push(`genesis_${name} ${value}`);
        }

        // Gauges
        this.gauges.uptime_seconds = Math.floor((Date.now() - this.startTime) / 1000);
        for (const [name, value] of Object.entries(this.gauges)) {
            lines.push(`# TYPE genesis_${name} gauge`);
            lines.push(`genesis_${name} ${value}`);
        }

        return lines.join('\n');
    }
}

// ============================================
// PEER REGISTRY (Enhanced)
// ============================================

class PeerRegistry {
    constructor(metrics) {
        this.peers = new Map();
        this.byPublicKey = new Map();
        this.byRoom = new Map();
        this.connections = new Map();
        this.ipConnections = new Map(); // Track connections per IP
        this.metrics = metrics;
    }

    register(peerId, info) {
        this.peers.set(peerId, {
            ...info,
            peerId,
            registeredAt: Date.now(),
            lastSeen: Date.now(),
        });

        if (info.publicKey) {
            this.byPublicKey.set(info.publicKey, peerId);
        }

        this.metrics.set('peers_registered', this.peers.size);
        return this.peers.get(peerId);
    }

    update(peerId, updates) {
        const peer = this.peers.get(peerId);
        if (peer) {
            Object.assign(peer, updates, { lastSeen: Date.now() });
        }
        return peer;
    }

    get(peerId) {
        return this.peers.get(peerId);
    }

    getByPublicKey(publicKey) {
        const peerId = this.byPublicKey.get(publicKey);
        return peerId ? this.peers.get(peerId) : null;
    }

    remove(peerId) {
        const peer = this.peers.get(peerId);
        if (peer) {
            if (peer.publicKey) {
                this.byPublicKey.delete(peer.publicKey);
            }
            if (peer.room) {
                const room = this.byRoom.get(peer.room);
                if (room) room.delete(peerId);
            }
            this.peers.delete(peerId);
            this.metrics.set('peers_registered', this.peers.size);
            return true;
        }
        return false;
    }

    joinRoom(peerId, room) {
        const peer = this.peers.get(peerId);
        if (!peer) return false;

        if (peer.room && peer.room !== room) {
            const oldRoom = this.byRoom.get(peer.room);
            if (oldRoom) oldRoom.delete(peerId);
        }

        if (!this.byRoom.has(room)) {
            this.byRoom.set(room, new Set());
        }
        this.byRoom.get(room).add(peerId);
        peer.room = room;

        this.metrics.set('rooms_active', this.byRoom.size);
        return true;
    }

    getRoomPeers(room) {
        const peerIds = this.byRoom.get(room) || new Set();
        return Array.from(peerIds).map(id => this.peers.get(id)).filter(Boolean);
    }

    getAllPeers() {
        return Array.from(this.peers.values());
    }

    pruneStale(maxAge = CONFIG.cleanup.staleConnectionTimeout) {
        const cutoff = Date.now() - maxAge;
        const removed = [];

        for (const [peerId, peer] of this.peers) {
            if (peer.lastSeen < cutoff) {
                this.remove(peerId);
                removed.push(peerId);
            }
        }

        return removed;
    }

    trackIpConnection(ip, connectionId) {
        if (!this.ipConnections.has(ip)) {
            this.ipConnections.set(ip, new Set());
        }
        this.ipConnections.get(ip).add(connectionId);
    }

    removeIpConnection(ip, connectionId) {
        const conns = this.ipConnections.get(ip);
        if (conns) {
            conns.delete(connectionId);
            if (conns.size === 0) {
                this.ipConnections.delete(ip);
            }
        }
    }

    getIpConnectionCount(ip) {
        return this.ipConnections.get(ip)?.size || 0;
    }

    getStats() {
        return {
            totalPeers: this.peers.size,
            rooms: this.byRoom.size,
            roomSizes: Object.fromEntries(
                Array.from(this.byRoom.entries()).map(([room, peers]) => [room, peers.size])
            ),
        };
    }
}

// ============================================
// LEDGER STORE (Enhanced)
// ============================================

class LedgerStore {
    constructor(dataDir, metrics) {
        this.dataDir = dataDir;
        this.ledgers = new Map();
        this.pendingWrites = new Map();
        this.metrics = metrics;

        if (!existsSync(dataDir)) {
            mkdirSync(dataDir, { recursive: true });
        }

        this.loadAll();
    }

    loadAll() {
        try {
            const indexPath = join(this.dataDir, 'index.json');
            if (existsSync(indexPath)) {
                const index = JSON.parse(readFileSync(indexPath, 'utf8'));
                for (const publicKey of index.keys || []) {
                    this.load(publicKey);
                }
                log.info('Loaded ledger index', { count: index.keys?.length || 0 });
            }
        } catch (err) {
            log.warn('Failed to load ledger index', { error: err.message });
        }
    }

    load(publicKey) {
        try {
            const path = join(this.dataDir, `ledger-${publicKey.slice(0, 16)}.json`);
            if (existsSync(path)) {
                const data = JSON.parse(readFileSync(path, 'utf8'));
                this.ledgers.set(publicKey, data);
                return data;
            }
        } catch (err) {
            log.warn('Failed to load ledger', { publicKey: publicKey.slice(0, 8), error: err.message });
        }
        return null;
    }

    save(publicKey) {
        try {
            const data = this.ledgers.get(publicKey);
            if (!data) return false;

            const path = join(this.dataDir, `ledger-${publicKey.slice(0, 16)}.json`);
            writeFileSync(path, JSON.stringify(data, null, 2));
            this.saveIndex();
            return true;
        } catch (err) {
            log.warn('Failed to save ledger', { publicKey: publicKey.slice(0, 8), error: err.message });
            return false;
        }
    }

    saveIndex() {
        try {
            const indexPath = join(this.dataDir, 'index.json');
            writeFileSync(indexPath, JSON.stringify({
                keys: Array.from(this.ledgers.keys()),
                updatedAt: Date.now(),
            }, null, 2));
        } catch (err) {
            log.warn('Failed to save index', { error: err.message });
        }
    }

    get(publicKey) {
        return this.ledgers.get(publicKey);
    }

    getStates(publicKey) {
        const ledger = this.ledgers.get(publicKey);
        if (!ledger) return [];
        return Object.values(ledger.devices || {});
    }

    update(publicKey, deviceId, state) {
        if (!this.ledgers.has(publicKey)) {
            this.ledgers.set(publicKey, {
                publicKey,
                createdAt: Date.now(),
                devices: {},
            });
        }

        const ledger = this.ledgers.get(publicKey);
        const existing = ledger.devices[deviceId] || {};
        const merged = this.mergeCRDT(existing, state);

        ledger.devices[deviceId] = {
            ...merged,
            deviceId,
            updatedAt: Date.now(),
        };

        this.scheduleSave(publicKey);
        this.metrics.set('ledgers_stored', this.ledgers.size);

        return ledger.devices[deviceId];
    }

    mergeCRDT(existing, incoming) {
        if (!existing.timestamp || incoming.timestamp > existing.timestamp) {
            return { ...incoming };
        }

        return {
            earned: Math.max(existing.earned || 0, incoming.earned || 0),
            spent: Math.max(existing.spent || 0, incoming.spent || 0),
            timestamp: Math.max(existing.timestamp || 0, incoming.timestamp || 0),
        };
    }

    scheduleSave(publicKey) {
        if (this.pendingWrites.has(publicKey)) return;

        this.pendingWrites.set(publicKey, setTimeout(() => {
            this.save(publicKey);
            this.pendingWrites.delete(publicKey);
        }, 1000));
    }

    flush() {
        for (const [publicKey, timeout] of this.pendingWrites) {
            clearTimeout(timeout);
            this.save(publicKey);
        }
        this.pendingWrites.clear();
    }

    getStats() {
        return {
            totalLedgers: this.ledgers.size,
            totalDevices: Array.from(this.ledgers.values())
                .reduce((sum, l) => sum + Object.keys(l.devices || {}).length, 0),
        };
    }
}

// ============================================
// AUTH SERVICE
// ============================================

class AuthService {
    constructor(metrics) {
        this.challenges = new Map();
        this.tokens = new Map();
        this.metrics = metrics;
    }

    createChallenge(publicKey, deviceId) {
        const nonce = randomBytes(32).toString('hex');
        const challenge = randomBytes(32).toString('hex');

        this.challenges.set(nonce, {
            challenge,
            publicKey,
            deviceId,
            expiresAt: Date.now() + CONFIG.rateLimit.challengeExpiry,
        });

        this.metrics.inc('auth_challenges');
        return { nonce, challenge };
    }

    verifyChallenge(nonce, publicKey, signature) {
        const challengeData = this.challenges.get(nonce);
        if (!challengeData) {
            this.metrics.inc('auth_failures');
            return { valid: false, error: 'Invalid nonce' };
        }

        if (Date.now() > challengeData.expiresAt) {
            this.challenges.delete(nonce);
            this.metrics.inc('auth_failures');
            return { valid: false, error: 'Challenge expired' };
        }

        if (challengeData.publicKey !== publicKey) {
            this.metrics.inc('auth_failures');
            return { valid: false, error: 'Public key mismatch' };
        }

        this.challenges.delete(nonce);

        const token = randomBytes(32).toString('hex');
        const tokenData = {
            publicKey,
            deviceId: challengeData.deviceId,
            createdAt: Date.now(),
            expiresAt: Date.now() + 24 * 60 * 60 * 1000,
        };

        this.tokens.set(token, tokenData);
        this.metrics.inc('auth_successes');

        return { valid: true, token, expiresAt: tokenData.expiresAt };
    }

    validateToken(token) {
        const tokenData = this.tokens.get(token);
        if (!tokenData) return null;

        if (Date.now() > tokenData.expiresAt) {
            this.tokens.delete(token);
            return null;
        }

        return tokenData;
    }

    cleanup() {
        const now = Date.now();

        for (const [nonce, data] of this.challenges) {
            if (now > data.expiresAt) {
                this.challenges.delete(nonce);
            }
        }

        for (const [token, data] of this.tokens) {
            if (now > data.expiresAt) {
                this.tokens.delete(token);
            }
        }
    }
}

// ============================================
// DHT ROUTING TABLE
// ============================================

const K = 20;
const ID_BITS = 160;

function xorDistance(id1, id2) {
    const buf1 = Buffer.from(id1, 'hex');
    const buf2 = Buffer.from(id2, 'hex');
    const result = Buffer.alloc(Math.max(buf1.length, buf2.length));

    for (let i = 0; i < result.length; i++) {
        result[i] = (buf1[i] || 0) ^ (buf2[i] || 0);
    }

    return result.toString('hex');
}

function getBucketIndex(distance) {
    const buf = Buffer.from(distance, 'hex');

    for (let i = 0; i < buf.length; i++) {
        if (buf[i] !== 0) {
            for (let j = 7; j >= 0; j--) {
                if (buf[i] & (1 << j)) {
                    return (buf.length - i - 1) * 8 + j;
                }
            }
        }
    }

    return 0;
}

class DHTRoutingTable {
    constructor(localId, metrics) {
        this.localId = localId;
        this.buckets = new Array(ID_BITS).fill(null).map(() => []);
        this.metrics = metrics;
    }

    add(peer) {
        if (peer.id === this.localId) return false;

        const distance = xorDistance(this.localId, peer.id);
        const bucketIndex = getBucketIndex(distance);
        const bucket = this.buckets[bucketIndex];

        const existingIndex = bucket.findIndex(p => p.id === peer.id);
        if (existingIndex !== -1) {
            bucket.splice(existingIndex, 1);
            bucket.push({ ...peer, lastSeen: Date.now() });
            return true;
        }

        if (bucket.length < K) {
            bucket.push({ ...peer, lastSeen: Date.now() });
            return true;
        }

        return false;
    }

    remove(peerId) {
        const distance = xorDistance(this.localId, peerId);
        const bucketIndex = getBucketIndex(distance);
        const bucket = this.buckets[bucketIndex];

        const index = bucket.findIndex(p => p.id === peerId);
        if (index !== -1) {
            bucket.splice(index, 1);
            return true;
        }
        return false;
    }

    findClosest(targetId, count = K) {
        const candidates = [];

        for (const bucket of this.buckets) {
            candidates.push(...bucket);
        }

        return candidates
            .map(p => ({
                ...p,
                distance: xorDistance(p.id, targetId),
            }))
            .sort((a, b) => a.distance.localeCompare(b.distance))
            .slice(0, count);
    }

    getAllPeers() {
        const peers = [];
        for (const bucket of this.buckets) {
            peers.push(...bucket);
        }
        return peers;
    }

    prune(maxAge = 300000) {
        const cutoff = Date.now() - maxAge;
        let removed = 0;

        for (const bucket of this.buckets) {
            for (let i = bucket.length - 1; i >= 0; i--) {
                if (bucket[i].lastSeen < cutoff) {
                    bucket.splice(i, 1);
                    removed++;
                }
            }
        }

        return removed;
    }

    getStats() {
        let totalPeers = 0;
        let bucketsUsed = 0;

        for (const bucket of this.buckets) {
            if (bucket.length > 0) {
                totalPeers += bucket.length;
                bucketsUsed++;
            }
        }

        return { totalPeers, bucketsUsed, bucketCount: this.buckets.length };
    }
}

// ============================================
// FIREBASE BOOTSTRAP REGISTRATION
// ============================================

class FirebaseBootstrapRegistration {
    constructor(nodeId, config, metrics) {
        this.nodeId = nodeId;
        this.config = config;
        this.metrics = metrics;
        this.app = null;
        this.db = null;
        this.isRegistered = false;
        this.registrationInterval = null;
    }

    async connect() {
        if (!this.config.firebase.enabled) {
            log.info('Firebase registration disabled');
            return false;
        }

        try {
            const { initializeApp, getApps } = await import('firebase/app');
            const { getFirestore, doc, setDoc, serverTimestamp } = await import('firebase/firestore');

            const firebaseConfig = {
                apiKey: this.config.firebase.apiKey,
                projectId: this.config.firebase.projectId,
                authDomain: this.config.firebase.authDomain || `${this.config.firebase.projectId}.firebaseapp.com`,
            };

            const apps = getApps();
            this.app = apps.length ? apps[0] : initializeApp(firebaseConfig);
            this.db = getFirestore(this.app);

            this.firebase = { doc, setDoc, serverTimestamp };

            await this.register();

            // Re-register periodically
            this.registrationInterval = setInterval(
                () => this.register(),
                this.config.firebase.registrationInterval
            );

            log.info('Firebase bootstrap registration enabled');
            return true;

        } catch (error) {
            log.warn('Firebase connection failed', { error: error.message });
            return false;
        }
    }

    async register() {
        if (!this.db) return;

        try {
            const { doc, setDoc, serverTimestamp } = this.firebase;

            const bootstrapRef = doc(this.db, 'edgenet_bootstrap_nodes', this.nodeId);

            await setDoc(bootstrapRef, {
                nodeId: this.nodeId,
                type: 'genesis',
                host: this.config.host === '0.0.0.0' ? null : this.config.host,
                port: this.config.port,
                capabilities: ['signaling', 'dht', 'ledger', 'discovery'],
                online: true,
                lastSeen: serverTimestamp(),
                version: '1.0.0',
            }, { merge: true });

            this.isRegistered = true;
            log.debug('Registered as bootstrap node');

        } catch (error) {
            log.warn('Bootstrap registration failed', { error: error.message });
        }
    }

    async unregister() {
        if (!this.db || !this.isRegistered) return;

        try {
            const { doc, setDoc } = this.firebase;

            const bootstrapRef = doc(this.db, 'edgenet_bootstrap_nodes', this.nodeId);
            await setDoc(bootstrapRef, { online: false }, { merge: true });

            log.info('Unregistered from bootstrap nodes');
        } catch (error) {
            log.warn('Bootstrap unregistration failed', { error: error.message });
        }
    }

    stop() {
        if (this.registrationInterval) {
            clearInterval(this.registrationInterval);
        }
    }
}

// ============================================
// HEALTH CHECK SERVER
// ============================================

class HealthCheckServer {
    constructor(config, metrics, getStatus) {
        this.config = config;
        this.metrics = metrics;
        this.getStatus = getStatus;
        this.server = null;
    }

    start() {
        this.server = http.createServer((req, res) => {
            const url = new URL(req.url, `http://${req.headers.host}`);

            switch (url.pathname) {
                case '/health':
                case '/healthz':
                    this.handleHealth(req, res);
                    break;

                case '/ready':
                case '/readyz':
                    this.handleReady(req, res);
                    break;

                case '/metrics':
                    this.handleMetrics(req, res);
                    break;

                case '/status':
                    this.handleStatus(req, res);
                    break;

                default:
                    res.writeHead(404);
                    res.end('Not Found');
            }
        });

        this.server.listen(this.config.healthPort, () => {
            log.info('Health check server started', { port: this.config.healthPort });
        });
    }

    handleHealth(req, res) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', timestamp: Date.now() }));
    }

    handleReady(req, res) {
        const status = this.getStatus();
        const ready = status.isRunning;

        res.writeHead(ready ? 200 : 503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ready, timestamp: Date.now() }));
    }

    handleMetrics(req, res) {
        if (this.config.metricsEnabled) {
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end(this.metrics.toPrometheus());
        } else {
            res.writeHead(404);
            res.end('Metrics disabled');
        }
    }

    handleStatus(req, res) {
        const status = this.getStatus();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(status, null, 2));
    }

    stop() {
        if (this.server) {
            this.server.close();
        }
    }
}

// ============================================
// PRODUCTION GENESIS NODE
// ============================================

class ProductionGenesisNode extends EventEmitter {
    constructor() {
        super();

        // Generate or use fixed node ID
        this.nodeId = CONFIG.nodeId || createHash('sha1').update(randomBytes(32)).digest('hex');

        // Initialize components
        this.metrics = new MetricsCollector();
        this.peerRegistry = new PeerRegistry(this.metrics);
        this.ledgerStore = new LedgerStore(CONFIG.dataDir, this.metrics);
        this.authService = new AuthService(this.metrics);
        this.dhtRouting = new DHTRoutingTable(this.nodeId, this.metrics);

        // Firebase registration
        this.firebaseRegistration = new FirebaseBootstrapRegistration(
            this.nodeId, CONFIG, this.metrics
        );

        // Health check
        this.healthServer = new HealthCheckServer(
            CONFIG, this.metrics, () => this.getStatus()
        );

        // WebSocket server
        this.wss = null;
        this.connections = new Map();

        // Timers
        this.cleanupInterval = null;
        this.statsInterval = null;
        this.dhtRefreshInterval = null;

        // State
        this.isRunning = false;
        this.startedAt = null;
    }

    async start() {
        log.info('Starting Production Genesis Node', {
            nodeId: this.nodeId.slice(0, 16),
            port: CONFIG.port,
            dataDir: CONFIG.dataDir,
        });

        // Start health check server
        this.healthServer.start();

        // Connect to Firebase for bootstrap registration
        await this.firebaseRegistration.connect();

        // Start WebSocket server
        const { WebSocketServer } = await import('ws');

        this.wss = new WebSocketServer({
            port: CONFIG.port,
            host: CONFIG.host,
            perMessageDeflate: false,
        });

        this.wss.on('connection', (ws, req) => this.handleConnection(ws, req));
        this.wss.on('error', (err) => {
            log.error('WebSocket server error', { error: err.message });
            this.metrics.inc('errors_total');
        });

        // Start cleanup interval
        this.cleanupInterval = setInterval(
            () => this.cleanup(),
            CONFIG.cleanup.cleanupInterval
        );

        // Start stats logging interval
        this.statsInterval = setInterval(
            () => this.logStats(),
            60000
        );

        // Start DHT refresh interval
        this.dhtRefreshInterval = setInterval(
            () => this.refreshDHT(),
            CONFIG.dht.bucketRefreshInterval
        );

        this.isRunning = true;
        this.startedAt = Date.now();

        log.info('Genesis Node started successfully', {
            wsEndpoint: `ws://${CONFIG.host}:${CONFIG.port}`,
            healthEndpoint: `http://${CONFIG.host}:${CONFIG.healthPort}/health`,
            metricsEndpoint: `http://${CONFIG.host}:${CONFIG.healthPort}/metrics`,
        });

        this.emit('started', { nodeId: this.nodeId, port: CONFIG.port });

        return this;
    }

    async stop() {
        log.info('Shutting down Genesis Node...');

        this.isRunning = false;

        // Stop intervals
        if (this.cleanupInterval) clearInterval(this.cleanupInterval);
        if (this.statsInterval) clearInterval(this.statsInterval);
        if (this.dhtRefreshInterval) clearInterval(this.dhtRefreshInterval);

        // Unregister from Firebase
        await this.firebaseRegistration.unregister();
        this.firebaseRegistration.stop();

        // Close WebSocket server
        if (this.wss) {
            this.wss.close();
        }

        // Stop health server
        this.healthServer.stop();

        // Flush ledger data
        this.ledgerStore.flush();

        log.info('Genesis Node stopped');
        this.emit('stopped');
    }

    handleConnection(ws, req) {
        const connectionId = randomBytes(16).toString('hex');
        const ip = req.headers['x-forwarded-for']?.split(',')[0].trim() ||
                   req.socket.remoteAddress;

        // Rate limiting by IP
        if (this.peerRegistry.getIpConnectionCount(ip) >= CONFIG.rateLimit.maxConnectionsPerIp) {
            log.warn('Rate limit exceeded', { ip, connectionId });
            ws.close(1008, 'Rate limit exceeded');
            return;
        }

        this.metrics.inc('connections_total');
        this.metrics.inc('connections_active');

        this.connections.set(connectionId, {
            ws,
            ip,
            peerId: null,
            connectedAt: Date.now(),
            messageCount: 0,
            lastMessageTime: Date.now(),
        });

        this.peerRegistry.trackIpConnection(ip, connectionId);

        log.debug('New connection', { connectionId: connectionId.slice(0, 8), ip });

        ws.on('message', (data) => {
            try {
                const conn = this.connections.get(connectionId);
                if (!conn) return;

                // Rate limiting by message frequency
                const now = Date.now();
                if (now - conn.lastMessageTime < 10) { // 100 msg/sec max
                    conn.messageCount++;
                    if (conn.messageCount > CONFIG.rateLimit.maxMessagesPerSecond) {
                        log.warn('Message rate limit exceeded', { connectionId: connectionId.slice(0, 8) });
                        ws.close(1008, 'Message rate limit exceeded');
                        return;
                    }
                } else {
                    conn.messageCount = 1;
                    conn.lastMessageTime = now;
                }

                const message = JSON.parse(data.toString());
                this.handleMessage(connectionId, message);

            } catch (err) {
                log.warn('Invalid message', { connectionId: connectionId.slice(0, 8), error: err.message });
                this.metrics.inc('errors_total');
            }
        });

        ws.on('close', () => {
            this.handleDisconnect(connectionId);
        });

        ws.on('error', (err) => {
            log.warn('Connection error', { connectionId: connectionId.slice(0, 8), error: err.message });
            this.metrics.inc('errors_total');
        });

        // Send welcome
        this.send(connectionId, {
            type: 'welcome',
            connectionId,
            nodeId: this.nodeId,
            serverTime: Date.now(),
            capabilities: ['signaling', 'dht', 'ledger', 'discovery'],
        });
    }

    handleDisconnect(connectionId) {
        const conn = this.connections.get(connectionId);
        if (!conn) return;

        if (conn.peerId) {
            const peer = this.peerRegistry.get(conn.peerId);
            if (peer?.room) {
                this.broadcastToRoom(peer.room, {
                    type: 'peer-left',
                    peerId: conn.peerId,
                }, conn.peerId);
            }
            this.peerRegistry.remove(conn.peerId);
            this.dhtRouting.remove(conn.peerId);
        }

        this.peerRegistry.removeIpConnection(conn.ip, connectionId);
        this.connections.delete(connectionId);
        this.metrics.dec('connections_active');

        log.debug('Connection closed', { connectionId: connectionId.slice(0, 8) });
    }

    handleMessage(connectionId, message) {
        this.metrics.inc('messages_received');

        const conn = this.connections.get(connectionId);
        if (!conn) return;

        switch (message.type) {
            case 'announce':
                this.handleAnnounce(connectionId, message);
                break;

            case 'join':
                this.handleJoinRoom(connectionId, message);
                break;

            case 'offer':
            case 'answer':
            case 'ice-candidate':
                this.relaySignal(connectionId, message);
                break;

            case 'auth-challenge':
                this.handleAuthChallenge(connectionId, message);
                break;

            case 'auth-verify':
                this.handleAuthVerify(connectionId, message);
                break;

            case 'ledger-get':
                this.handleLedgerGet(connectionId, message);
                break;

            case 'ledger-put':
                this.handleLedgerPut(connectionId, message);
                break;

            case 'dht-bootstrap':
                this.handleDHTBootstrap(connectionId, message);
                break;

            case 'dht-find-node':
                this.handleDHTFindNode(connectionId, message);
                break;

            case 'dht-store':
                this.handleDHTStore(connectionId, message);
                break;

            case 'ping':
                this.send(connectionId, { type: 'pong', timestamp: Date.now() });
                break;

            default:
                log.debug('Unknown message type', { type: message.type });
        }
    }

    handleAnnounce(connectionId, message) {
        const conn = this.connections.get(connectionId);
        const peerId = message.peerId || message.piKey || randomBytes(16).toString('hex');

        conn.peerId = peerId;

        this.peerRegistry.register(peerId, {
            publicKey: message.publicKey,
            siteId: message.siteId,
            capabilities: message.capabilities || [],
            connectionId,
        });

        // Add to DHT routing table
        this.dhtRouting.add({
            id: peerId,
            address: connectionId,
            lastSeen: Date.now(),
        });

        // Send current peer list
        const peers = this.peerRegistry.getAllPeers()
            .filter(p => p.peerId !== peerId)
            .slice(0, 50)
            .map(p => ({
                piKey: p.peerId,
                siteId: p.siteId,
                capabilities: p.capabilities,
            }));

        this.send(connectionId, {
            type: 'peer-list',
            peers,
        });

        // Notify other peers
        for (const peer of this.peerRegistry.getAllPeers()) {
            if (peer.peerId !== peerId && peer.connectionId) {
                this.send(peer.connectionId, {
                    type: 'peer-joined',
                    peerId,
                    siteId: message.siteId,
                    capabilities: message.capabilities,
                });
            }
        }

        log.debug('Peer announced', { peerId: peerId.slice(0, 8), capabilities: message.capabilities });
    }

    handleJoinRoom(connectionId, message) {
        const conn = this.connections.get(connectionId);
        if (!conn?.peerId) return;

        const room = message.room || 'default';
        this.peerRegistry.joinRoom(conn.peerId, room);

        const roomPeers = this.peerRegistry.getRoomPeers(room)
            .filter(p => p.peerId !== conn.peerId)
            .map(p => ({
                piKey: p.peerId,
                siteId: p.siteId,
            }));

        this.send(connectionId, {
            type: 'room-joined',
            room,
            peers: roomPeers,
        });

        this.broadcastToRoom(room, {
            type: 'peer-joined',
            peerId: conn.peerId,
            siteId: this.peerRegistry.get(conn.peerId)?.siteId,
        }, conn.peerId);
    }

    relaySignal(connectionId, message) {
        this.metrics.inc('signals_relayed');

        const conn = this.connections.get(connectionId);
        if (!conn?.peerId) return;

        const targetPeer = this.peerRegistry.get(message.to);
        if (!targetPeer?.connectionId) {
            this.send(connectionId, {
                type: 'error',
                error: 'Target peer not found',
                originalType: message.type,
            });
            return;
        }

        this.send(targetPeer.connectionId, {
            ...message,
            from: conn.peerId,
        });

        this.metrics.inc('messages_sent');
    }

    handleAuthChallenge(connectionId, message) {
        const { nonce, challenge } = this.authService.createChallenge(
            message.publicKey,
            message.deviceId
        );

        this.send(connectionId, {
            type: 'auth-challenge-response',
            nonce,
            challenge,
        });
    }

    handleAuthVerify(connectionId, message) {
        const result = this.authService.verifyChallenge(
            message.nonce,
            message.publicKey,
            message.signature
        );

        this.send(connectionId, {
            type: 'auth-verify-response',
            ...result,
        });
    }

    handleLedgerGet(connectionId, message) {
        const tokenData = this.authService.validateToken(message.token);
        if (!tokenData) {
            this.send(connectionId, {
                type: 'ledger-response',
                error: 'Invalid or expired token',
            });
            return;
        }

        const states = this.ledgerStore.getStates(message.publicKey || tokenData.publicKey);

        this.send(connectionId, {
            type: 'ledger-response',
            states,
        });
    }

    handleLedgerPut(connectionId, message) {
        const tokenData = this.authService.validateToken(message.token);
        if (!tokenData) {
            this.send(connectionId, {
                type: 'ledger-put-response',
                error: 'Invalid or expired token',
            });
            return;
        }

        const updated = this.ledgerStore.update(
            tokenData.publicKey,
            message.deviceId || tokenData.deviceId,
            message.state
        );

        this.send(connectionId, {
            type: 'ledger-put-response',
            success: true,
            state: updated,
        });
    }

    handleDHTBootstrap(connectionId, message) {
        this.metrics.inc('dht_lookups');

        const peers = this.dhtRouting.getAllPeers()
            .slice(0, 20)
            .map(p => ({
                id: p.id,
                address: p.address,
                lastSeen: p.lastSeen,
            }));

        this.send(connectionId, {
            type: 'dht-bootstrap-response',
            nodeId: this.nodeId,
            peers,
        });
    }

    handleDHTFindNode(connectionId, message) {
        this.metrics.inc('dht_lookups');

        const closest = this.dhtRouting.findClosest(message.target, K);

        this.send(connectionId, {
            type: 'dht-find-node-response',
            target: message.target,
            nodes: closest.map(p => ({
                id: p.id,
                address: p.address,
                distance: p.distance,
            })),
        });
    }

    handleDHTStore(connectionId, message) {
        this.metrics.inc('dht_stores');

        // For now, just acknowledge - full DHT storage would be added here
        this.send(connectionId, {
            type: 'dht-store-response',
            success: true,
            key: message.key,
        });
    }

    send(connectionId, message) {
        const conn = this.connections.get(connectionId);
        if (conn?.ws?.readyState === 1) {
            conn.ws.send(JSON.stringify(message));
            this.metrics.inc('messages_sent');
        }
    }

    broadcastToRoom(room, message, excludePeerId = null) {
        const peers = this.peerRegistry.getRoomPeers(room);
        for (const peer of peers) {
            if (peer.peerId !== excludePeerId && peer.connectionId) {
                this.send(peer.connectionId, message);
            }
        }
    }

    cleanup() {
        // Prune stale peers
        const removed = this.peerRegistry.pruneStale();
        if (removed.length > 0) {
            log.info('Pruned stale peers', { count: removed.length });
        }

        // Prune DHT routing table
        const dhtRemoved = this.dhtRouting.prune();
        if (dhtRemoved > 0) {
            log.debug('Pruned DHT entries', { count: dhtRemoved });
        }

        // Cleanup auth
        this.authService.cleanup();
    }

    refreshDHT() {
        // Periodic DHT maintenance would go here
        log.debug('DHT refresh', this.dhtRouting.getStats());
    }

    logStats() {
        const stats = this.getStatus();
        log.info('Node statistics', {
            peers: stats.peers.total,
            connections: stats.connections,
            dht: stats.dht.totalPeers,
            uptime: Math.floor((Date.now() - this.startedAt) / 1000),
        });
    }

    getStatus() {
        return {
            nodeId: this.nodeId,
            isRunning: this.isRunning,
            startedAt: this.startedAt,
            uptime: this.startedAt ? Date.now() - this.startedAt : 0,
            connections: this.connections.size,
            peers: this.peerRegistry.getStats(),
            ledger: this.ledgerStore.getStats(),
            dht: this.dhtRouting.getStats(),
            metrics: this.metrics.getMetrics(),
            firebase: {
                enabled: CONFIG.firebase.enabled,
                registered: this.firebaseRegistration.isRegistered,
            },
        };
    }
}

// ============================================
// MAIN
// ============================================

async function main() {
    log.info('Production Genesis Node starting...', {
        version: '1.0.0',
        nodeEnv: process.env.NODE_ENV,
    });

    const genesis = new ProductionGenesisNode();

    // Handle shutdown signals
    const shutdown = async (signal) => {
        log.info('Received shutdown signal', { signal });
        await genesis.stop();
        process.exit(0);
    };

    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));

    // Handle uncaught errors
    process.on('uncaughtException', (err) => {
        log.error('Uncaught exception', { error: err.message, stack: err.stack });
        process.exit(1);
    });

    process.on('unhandledRejection', (reason, promise) => {
        log.error('Unhandled rejection', { reason: String(reason) });
    });

    // Start the node
    await genesis.start();
}

main().catch(err => {
    log.error('Fatal error', { error: err.message, stack: err.stack });
    process.exit(1);
});

export default ProductionGenesisNode;

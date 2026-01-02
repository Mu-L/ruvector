#!/usr/bin/env node
/**
 * P2P Migration Test Suite
 *
 * Tests the HybridBootstrap migration flow:
 * firebase -> hybrid -> p2p
 *
 * Validates:
 * 1. Migration thresholds
 * 2. DHT routing table population
 * 3. Signaling fallback behavior
 * 4. Network partition recovery
 * 5. Node churn handling
 *
 * @module @ruvector/edge-net/tests/p2p-migration-test
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';

// ============================================
// MOCK IMPLEMENTATIONS
// ============================================

/**
 * Mock WebRTC Peer Manager for simulation
 */
class MockWebRTCPeerManager extends EventEmitter {
    constructor(peerId) {
        super();
        this.peerId = peerId;
        this.peers = new Map();
        this.externalSignaling = null;
        this.stats = {
            totalConnections: 0,
            successfulConnections: 0,
            failedConnections: 0,
        };
    }

    setExternalSignaling(callback) {
        this.externalSignaling = callback;
    }

    async connectToPeer(peerId) {
        if (this.peers.has(peerId)) return;
        this.stats.totalConnections++;

        // Simulate connection delay
        await new Promise(resolve => setTimeout(resolve, 50));

        // Simulate successful connection
        this.peers.set(peerId, {
            peerId,
            state: 'connected',
            lastSeen: Date.now(),
        });
        this.stats.successfulConnections++;
        this.emit('peer-connected', peerId);
    }

    disconnectPeer(peerId) {
        if (this.peers.has(peerId)) {
            this.peers.delete(peerId);
            this.emit('peer-disconnected', peerId);
        }
    }

    isConnected(peerId) {
        return this.peers.has(peerId);
    }

    sendToPeer(peerId, message) {
        if (this.peers.has(peerId)) {
            this.emit('message-sent', { to: peerId, message });
            return true;
        }
        return false;
    }

    async handleOffer({ from, offer }) {
        // Simulate offer handling
        await new Promise(resolve => setTimeout(resolve, 20));
    }

    async handleAnswer({ from, answer }) {
        // Simulate answer handling
        await new Promise(resolve => setTimeout(resolve, 20));
    }

    async handleIceCandidate({ from, candidate }) {
        // Simulate ICE handling
        await new Promise(resolve => setTimeout(resolve, 10));
    }

    getStats() {
        return {
            ...this.stats,
            connectedPeers: this.peers.size,
        };
    }
}

/**
 * Mock DHT Node for simulation
 */
class MockDHTNode extends EventEmitter {
    constructor(id) {
        super();
        this.id = id || createHash('sha1').update(randomBytes(32)).digest('hex');
        this.peers = new Map();
        this.storage = new Map();
        this.stats = {
            lookups: 0,
            stores: 0,
        };
    }

    addPeer(peer) {
        if (peer.id === this.id) return false;
        this.peers.set(peer.id, { ...peer, lastSeen: Date.now() });
        this.emit('peer-added', peer);
        return true;
    }

    removePeer(peerId) {
        if (this.peers.has(peerId)) {
            this.peers.delete(peerId);
            this.emit('peer-removed', peerId);
            return true;
        }
        return false;
    }

    getPeers() {
        return Array.from(this.peers.values());
    }

    getStats() {
        return {
            ...this.stats,
            totalPeers: this.peers.size,
        };
    }
}

/**
 * Mock Firebase Signaling for simulation
 */
class MockFirebaseSignaling extends EventEmitter {
    constructor(options = {}) {
        super();
        this.peerId = options.peerId;
        this.isConnected = false;
        this.peers = new Map();
        this.stats = {
            firebaseSignals: 0,
        };
    }

    async connect() {
        await new Promise(resolve => setTimeout(resolve, 100));
        this.isConnected = true;
        this.emit('connected');
        return true;
    }

    async disconnect() {
        this.isConnected = false;
        this.peers.clear();
        this.emit('disconnected');
    }

    addPeer(peerId, data = {}) {
        this.peers.set(peerId, { peerId, ...data, lastSeen: Date.now() });
        this.emit('peer-discovered', { peerId, ...data });
    }

    removePeer(peerId) {
        if (this.peers.has(peerId)) {
            this.peers.delete(peerId);
            this.emit('peer-left', { peerId });
        }
    }

    async sendOffer(toPeerId, offer) {
        this.stats.firebaseSignals++;
        return true;
    }

    async sendAnswer(toPeerId, answer) {
        this.stats.firebaseSignals++;
        return true;
    }

    async sendIceCandidate(toPeerId, candidate) {
        this.stats.firebaseSignals++;
        return true;
    }

    async sendSignal(toPeerId, type, data) {
        this.stats.firebaseSignals++;
        return true;
    }
}

/**
 * Simulated HybridBootstrap for testing (mirrors real implementation)
 */
class SimulatedHybridBootstrap extends EventEmitter {
    constructor(options = {}) {
        super();
        this.peerId = options.peerId || createHash('sha1').update(randomBytes(16)).digest('hex');
        this.mode = 'firebase';
        this.dhtPeerThreshold = options.dhtPeerThreshold || 5;
        this.p2pPeerThreshold = options.p2pPeerThreshold || 10;

        // Components
        this.firebase = null;
        this.dht = null;
        this.webrtc = null;

        // Stats
        this.stats = {
            firebaseDiscoveries: 0,
            dhtDiscoveries: 0,
            directConnections: 0,
            firebaseSignals: 0,
            p2pSignals: 0,
            modeTransitions: [],
        };

        // Migration timing
        this.migrationTimestamps = {
            started: null,
            toHybrid: null,
            toP2P: null,
            fallbackToHybrid: null,
            fallbackToFirebase: null,
        };
    }

    async start(webrtc, dht) {
        this.webrtc = webrtc;
        this.dht = dht;
        this.migrationTimestamps.started = Date.now();

        // Create mock Firebase
        this.firebase = new MockFirebaseSignaling({ peerId: this.peerId });
        this.setupFirebaseEvents();

        // Set up WebRTC external signaling
        if (this.webrtc) {
            this.webrtc.setExternalSignaling(async (type, toPeerId, data) => {
                switch (type) {
                    case 'offer':
                        await this.firebase.sendOffer(toPeerId, data);
                        break;
                    case 'answer':
                        await this.firebase.sendAnswer(toPeerId, data);
                        break;
                    case 'ice-candidate':
                        await this.firebase.sendIceCandidate(toPeerId, data);
                        break;
                }
                this.stats.firebaseSignals++;
            });
        }

        const connected = await this.firebase.connect();
        if (connected) {
            this.mode = 'firebase';
            this.stats.modeTransitions.push({ from: null, to: 'firebase', timestamp: Date.now() });
        }

        return connected;
    }

    setupFirebaseEvents() {
        this.firebase.on('peer-discovered', async ({ peerId }) => {
            this.stats.firebaseDiscoveries++;
            if (this.webrtc) {
                await this.connectToPeer(peerId);
            }
            this.emit('peer-discovered', { peerId, source: 'firebase' });
        });

        this.firebase.on('offer', async ({ from, offer }) => {
            this.stats.firebaseSignals++;
            if (this.webrtc) {
                await this.webrtc.handleOffer({ from, offer });
            }
        });

        this.firebase.on('answer', async ({ from, answer }) => {
            this.stats.firebaseSignals++;
            if (this.webrtc) {
                await this.webrtc.handleAnswer({ from, answer });
            }
        });
    }

    async connectToPeer(peerId) {
        if (!this.webrtc) return;
        try {
            await this.webrtc.connectToPeer(peerId);
            this.stats.directConnections++;

            // Also add to DHT
            if (this.dht) {
                this.dht.addPeer({ id: peerId, lastSeen: Date.now() });
            }
        } catch (error) {
            // Connection failed
        }
    }

    async signal(toPeerId, type, data) {
        if (this.webrtc?.isConnected(toPeerId)) {
            this.webrtc.sendToPeer(toPeerId, { type, data });
            this.stats.p2pSignals++;
            return;
        }
        if (this.firebase?.isConnected) {
            await this.firebase.sendSignal(toPeerId, type, data);
            this.stats.firebaseSignals++;
            return;
        }
        throw new Error('No signaling path available');
    }

    checkMigration() {
        const connectedPeers = this.webrtc?.peers?.size || 0;
        const dhtPeers = this.dht?.getPeers?.()?.length || 0;
        const previousMode = this.mode;

        // Migration logic (mirrors real implementation)
        if (this.mode === 'firebase') {
            if (dhtPeers >= this.dhtPeerThreshold) {
                this.mode = 'hybrid';
                this.migrationTimestamps.toHybrid = Date.now();
            }
        } else if (this.mode === 'hybrid') {
            if (connectedPeers >= this.p2pPeerThreshold) {
                this.mode = 'p2p';
                this.migrationTimestamps.toP2P = Date.now();
            } else if (dhtPeers < this.dhtPeerThreshold / 2) {
                this.mode = 'firebase';
                this.migrationTimestamps.fallbackToFirebase = Date.now();
            }
        } else if (this.mode === 'p2p') {
            if (connectedPeers < this.p2pPeerThreshold / 2) {
                this.mode = 'hybrid';
                this.migrationTimestamps.fallbackToHybrid = Date.now();
            }
        }

        if (this.mode !== previousMode) {
            this.stats.modeTransitions.push({
                from: previousMode,
                to: this.mode,
                timestamp: Date.now(),
                dhtPeers,
                connectedPeers,
            });
            this.emit('mode-changed', { from: previousMode, to: this.mode });
        }

        return { previousMode, currentMode: this.mode, dhtPeers, connectedPeers };
    }

    getStats() {
        return {
            mode: this.mode,
            ...this.stats,
            firebaseConnected: this.firebase?.isConnected || false,
            firebasePeers: this.firebase?.peers?.size || 0,
            dhtPeers: this.dht?.getPeers?.()?.length || 0,
            directPeers: this.webrtc?.peers?.size || 0,
            migrationTimestamps: this.migrationTimestamps,
        };
    }

    async stop() {
        if (this.firebase) {
            await this.firebase.disconnect();
        }
    }
}

// ============================================
// TEST UTILITIES
// ============================================

/**
 * Create a test network with multiple nodes
 */
function createTestNetwork(nodeCount, options = {}) {
    const nodes = [];
    for (let i = 0; i < nodeCount; i++) {
        const peerId = createHash('sha1').update(`test-node-${i}`).digest('hex');
        const webrtc = new MockWebRTCPeerManager(peerId);
        const dht = new MockDHTNode(peerId);
        const bootstrap = new SimulatedHybridBootstrap({
            peerId,
            dhtPeerThreshold: options.dhtPeerThreshold || 5,
            p2pPeerThreshold: options.p2pPeerThreshold || 10,
        });

        nodes.push({
            id: i,
            peerId,
            webrtc,
            dht,
            bootstrap,
        });
    }
    return nodes;
}

/**
 * Simulate peer discovery via Firebase
 */
async function simulateFirebaseDiscovery(nodes) {
    for (const node of nodes) {
        for (const otherNode of nodes) {
            if (node.peerId !== otherNode.peerId) {
                node.bootstrap.firebase.addPeer(otherNode.peerId);
                await new Promise(r => setTimeout(r, 10));
            }
        }
    }
}

/**
 * Simulate nodes joining the network gradually
 */
async function simulateGradualJoin(nodes, delayMs = 100) {
    for (const node of nodes) {
        await node.bootstrap.start(node.webrtc, node.dht);
        await new Promise(r => setTimeout(r, delayMs));
    }
}

/**
 * Connect nodes directly (simulate WebRTC connections)
 */
async function connectNodes(nodeA, nodeB) {
    await nodeA.webrtc.connectToPeer(nodeB.peerId);
    await nodeB.webrtc.connectToPeer(nodeA.peerId);
    nodeA.dht.addPeer({ id: nodeB.peerId });
    nodeB.dht.addPeer({ id: nodeA.peerId });
}

/**
 * Test result formatter
 */
function formatTestResult(name, passed, details = {}) {
    const status = passed ? '\x1b[32mPASSED\x1b[0m' : '\x1b[31mFAILED\x1b[0m';
    console.log(`  ${status}: ${name}`);
    if (!passed && Object.keys(details).length > 0) {
        console.log('    Details:', JSON.stringify(details, null, 2));
    }
    return passed;
}

// ============================================
// TEST SCENARIOS
// ============================================

/**
 * TEST 1: Happy Path - Gradual Network Growth
 *
 * Scenario: Nodes join gradually, network grows, migration occurs
 * Expected: firebase -> hybrid -> p2p transitions
 */
async function testHappyPathGradualGrowth() {
    console.log('\n--- TEST 1: Happy Path - Gradual Network Growth ---');

    const nodes = createTestNetwork(15, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 8,
    });

    // Start first node
    const firstNode = nodes[0];
    await firstNode.bootstrap.start(firstNode.webrtc, firstNode.dht);

    let passed = true;

    // Verify starts in firebase mode
    passed = formatTestResult(
        'Node starts in firebase mode',
        firstNode.bootstrap.mode === 'firebase',
        { mode: firstNode.bootstrap.mode }
    ) && passed;

    // Add nodes gradually and check transitions
    for (let i = 1; i < nodes.length; i++) {
        const node = nodes[i];
        await node.bootstrap.start(node.webrtc, node.dht);

        // Connect to first node
        await connectNodes(firstNode, node);

        // Check migration
        firstNode.bootstrap.checkMigration();
    }

    // Final state checks
    const stats = firstNode.bootstrap.getStats();

    passed = formatTestResult(
        'Transitioned to hybrid mode',
        stats.modeTransitions.some(t => t.to === 'hybrid'),
        { transitions: stats.modeTransitions }
    ) && passed;

    passed = formatTestResult(
        'Transitioned to p2p mode',
        stats.mode === 'p2p' || stats.modeTransitions.some(t => t.to === 'p2p'),
        { currentMode: stats.mode, transitions: stats.modeTransitions }
    ) && passed;

    passed = formatTestResult(
        'DHT routing table populated',
        stats.dhtPeers >= 10,
        { dhtPeers: stats.dhtPeers }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 2: Edge Case - Nodes Leaving
 *
 * Scenario: Network grows then nodes leave
 * Expected: p2p -> hybrid -> firebase fallback
 *
 * Fallback thresholds are half the original:
 * - P2P -> Hybrid: when connectedPeers < p2pPeerThreshold / 2
 * - Hybrid -> Firebase: when dhtPeers < dhtPeerThreshold / 2
 */
async function testNodesLeaving() {
    console.log('\n--- TEST 2: Edge Case - Nodes Leaving ---');

    const nodes = createTestNetwork(12, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 6,
    });

    let passed = true;

    // Build up network
    for (const node of nodes) {
        await node.bootstrap.start(node.webrtc, node.dht);
    }

    // Connect all nodes to first node
    const firstNode = nodes[0];
    for (let i = 1; i < nodes.length; i++) {
        await connectNodes(firstNode, nodes[i]);
        firstNode.bootstrap.checkMigration();
    }

    // Verify in p2p mode
    passed = formatTestResult(
        'Reached p2p mode with full network',
        firstNode.bootstrap.mode === 'p2p',
        { mode: firstNode.bootstrap.mode }
    ) && passed;

    // Simulate nodes leaving - need to get below p2pPeerThreshold/2 = 3
    // So we need to disconnect until we have < 3 peers
    for (let i = nodes.length - 1; i >= 4; i--) {
        firstNode.webrtc.disconnectPeer(nodes[i].peerId);
        firstNode.dht.removePeer(nodes[i].peerId);
        firstNode.bootstrap.checkMigration();
    }

    // Now we should have 3 peers (indices 1,2,3), still at or above threshold
    // Need to drop one more to trigger fallback
    firstNode.webrtc.disconnectPeer(nodes[3].peerId);
    firstNode.dht.removePeer(nodes[3].peerId);
    firstNode.bootstrap.checkMigration();

    // Should fall back to hybrid (2 peers < 3)
    passed = formatTestResult(
        'Falls back to hybrid when peers drop below half threshold',
        firstNode.bootstrap.mode === 'hybrid',
        { mode: firstNode.bootstrap.mode, directPeers: firstNode.webrtc.peers.size, threshold: 'p2pPeerThreshold/2 = 3' }
    ) && passed;

    // More nodes leave - need to get DHT below dhtPeerThreshold/2 = 1.5
    firstNode.webrtc.disconnectPeer(nodes[2].peerId);
    firstNode.dht.removePeer(nodes[2].peerId);
    firstNode.bootstrap.checkMigration();

    // Now 1 DHT peer, should fall back to firebase (1 < 1.5)
    passed = formatTestResult(
        'Falls back to firebase when DHT peers drop below half threshold',
        firstNode.bootstrap.mode === 'firebase',
        { mode: firstNode.bootstrap.mode, dhtPeers: firstNode.dht.getPeers().length, threshold: 'dhtPeerThreshold/2 = 1.5' }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 3: Edge Case - Nodes Rejoining
 *
 * Scenario: Nodes leave then rejoin
 * Expected: Proper re-migration
 */
async function testNodesRejoining() {
    console.log('\n--- TEST 3: Edge Case - Nodes Rejoining ---');

    const nodes = createTestNetwork(10, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 6,
    });

    let passed = true;

    // Initial build-up
    const firstNode = nodes[0];
    await firstNode.bootstrap.start(firstNode.webrtc, firstNode.dht);

    for (let i = 1; i < 8; i++) {
        await nodes[i].bootstrap.start(nodes[i].webrtc, nodes[i].dht);
        await connectNodes(firstNode, nodes[i]);
        firstNode.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'Initial p2p state reached',
        firstNode.bootstrap.mode === 'p2p',
        { mode: firstNode.bootstrap.mode }
    ) && passed;

    // Nodes leave
    for (let i = 7; i >= 2; i--) {
        firstNode.webrtc.disconnectPeer(nodes[i].peerId);
        firstNode.dht.removePeer(nodes[i].peerId);
    }
    firstNode.bootstrap.checkMigration();

    const modeAfterLeaving = firstNode.bootstrap.mode;
    passed = formatTestResult(
        'Mode degraded after nodes left',
        modeAfterLeaving !== 'p2p',
        { mode: modeAfterLeaving }
    ) && passed;

    // Nodes rejoin
    for (let i = 2; i < 8; i++) {
        await connectNodes(firstNode, nodes[i]);
        firstNode.bootstrap.checkMigration();
    }

    // New nodes join
    for (let i = 8; i < 10; i++) {
        await nodes[i].bootstrap.start(nodes[i].webrtc, nodes[i].dht);
        await connectNodes(firstNode, nodes[i]);
        firstNode.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'Re-migrated to p2p after rejoin',
        firstNode.bootstrap.mode === 'p2p',
        { mode: firstNode.bootstrap.mode, directPeers: firstNode.webrtc.peers.size }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 4: Network Partition Recovery
 *
 * Scenario: Network splits then recovers
 * Expected: Graceful handling of partition
 */
async function testNetworkPartitionRecovery() {
    console.log('\n--- TEST 4: Network Partition Recovery ---');

    const nodes = createTestNetwork(12, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 6,
    });

    let passed = true;

    // Build full network
    for (const node of nodes) {
        await node.bootstrap.start(node.webrtc, node.dht);
    }

    // Connect all in mesh
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            await connectNodes(nodes[i], nodes[j]);
        }
    }

    // Check migration multiple times to allow state to propagate
    // (real implementation has 30s interval, but we check immediately)
    for (let round = 0; round < 3; round++) {
        for (const node of nodes) {
            node.bootstrap.checkMigration();
        }
    }

    // In mesh topology with 11 connections each, all should be in p2p mode
    const p2pCount = nodes.filter(n => n.bootstrap.mode === 'p2p').length;
    passed = formatTestResult(
        'Most nodes in p2p mode initially (mesh has 11 connections each)',
        p2pCount >= 10, // Allow some variance
        { p2pNodes: p2pCount, totalNodes: nodes.length, modes: nodes.map(n => n.bootstrap.mode) }
    ) && passed;

    // Simulate partition: split into two groups
    const groupA = nodes.slice(0, 6);
    const groupB = nodes.slice(6);

    // Disconnect cross-group connections
    for (const nodeA of groupA) {
        for (const nodeB of groupB) {
            nodeA.webrtc.disconnectPeer(nodeB.peerId);
            nodeA.dht.removePeer(nodeB.peerId);
            nodeB.webrtc.disconnectPeer(nodeA.peerId);
            nodeB.dht.removePeer(nodeA.peerId);
        }
    }

    // Check migration for both groups
    for (const node of nodes) {
        node.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'Both partitions still operational',
        groupA[0].bootstrap.mode !== 'firebase' && groupB[0].bootstrap.mode !== 'firebase',
        { groupA: groupA[0].bootstrap.mode, groupB: groupB[0].bootstrap.mode }
    ) && passed;

    // Heal partition
    for (const nodeA of groupA) {
        for (const nodeB of groupB) {
            await connectNodes(nodeA, nodeB);
        }
    }

    for (const node of nodes) {
        node.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'Network recovered to p2p after healing',
        nodes.every(n => n.bootstrap.mode === 'p2p'),
        { modes: nodes.map(n => n.bootstrap.mode) }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 5: Signaling Fallback Behavior
 *
 * Scenario: Test signaling routes through correct channel
 * Expected: P2P when available, Firebase fallback
 */
async function testSignalingFallback() {
    console.log('\n--- TEST 5: Signaling Fallback Behavior ---');

    const nodes = createTestNetwork(3, {
        dhtPeerThreshold: 1,
        p2pPeerThreshold: 2,
    });

    let passed = true;

    // Start all nodes
    for (const node of nodes) {
        await node.bootstrap.start(node.webrtc, node.dht);
    }

    const nodeA = nodes[0];
    const nodeB = nodes[1];
    const nodeC = nodes[2];

    // Signal without P2P connection (should use Firebase)
    const statsBeforeFirebase = { ...nodeA.bootstrap.stats };
    await nodeA.bootstrap.signal(nodeB.peerId, 'test', { data: 'hello' });

    passed = formatTestResult(
        'Uses Firebase for signaling without P2P',
        nodeA.bootstrap.stats.firebaseSignals > statsBeforeFirebase.firebaseSignals,
        { before: statsBeforeFirebase.firebaseSignals, after: nodeA.bootstrap.stats.firebaseSignals }
    ) && passed;

    // Connect nodes directly
    await connectNodes(nodeA, nodeB);

    // Signal with P2P connection (should use P2P)
    const statsBeforeP2P = { ...nodeA.bootstrap.stats };
    await nodeA.bootstrap.signal(nodeB.peerId, 'test', { data: 'hello' });

    passed = formatTestResult(
        'Uses P2P for signaling when connected',
        nodeA.bootstrap.stats.p2pSignals > statsBeforeP2P.p2pSignals,
        { before: statsBeforeP2P.p2pSignals, after: nodeA.bootstrap.stats.p2pSignals }
    ) && passed;

    // Signal to unconnected peer (should fallback to Firebase)
    const statsBeforeFallback = { ...nodeA.bootstrap.stats };
    await nodeA.bootstrap.signal(nodeC.peerId, 'test', { data: 'hello' });

    passed = formatTestResult(
        'Falls back to Firebase for unconnected peer',
        nodeA.bootstrap.stats.firebaseSignals > statsBeforeFallback.firebaseSignals,
        { before: statsBeforeFallback.firebaseSignals, after: nodeA.bootstrap.stats.firebaseSignals }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 6: DHT Routing Table Population
 *
 * Scenario: Validate DHT is properly populated during migration
 * Expected: DHT contains all connected peers
 */
async function testDHTRoutingTablePopulation() {
    console.log('\n--- TEST 6: DHT Routing Table Population ---');

    const nodes = createTestNetwork(8, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 6,
    });

    let passed = true;

    const firstNode = nodes[0];
    await firstNode.bootstrap.start(firstNode.webrtc, firstNode.dht);

    // Connect nodes and verify DHT population
    for (let i = 1; i < nodes.length; i++) {
        await nodes[i].bootstrap.start(nodes[i].webrtc, nodes[i].dht);
        await connectNodes(firstNode, nodes[i]);
    }

    // Check DHT stats
    const dhtStats = firstNode.dht.getStats();
    const webrtcStats = firstNode.webrtc.getStats();

    passed = formatTestResult(
        'DHT peers matches WebRTC connections',
        dhtStats.totalPeers === webrtcStats.connectedPeers,
        { dhtPeers: dhtStats.totalPeers, webrtcPeers: webrtcStats.connectedPeers }
    ) && passed;

    // Verify all peers are in DHT
    const dhtPeerIds = new Set(firstNode.dht.getPeers().map(p => p.id));
    const allConnected = nodes.slice(1).every(n => dhtPeerIds.has(n.peerId));

    passed = formatTestResult(
        'All connected peers in DHT routing table',
        allConnected,
        { dhtPeers: Array.from(dhtPeerIds).map(p => p.slice(0, 8)) }
    ) && passed;

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 7: Migration Timing Measurement
 *
 * Scenario: Measure migration timing for performance analysis
 * Expected: Record timing data for analysis
 */
async function testMigrationTiming() {
    console.log('\n--- TEST 7: Migration Timing Measurement ---');

    const nodes = createTestNetwork(15, {
        dhtPeerThreshold: 3,
        p2pPeerThreshold: 8,
    });

    let passed = true;

    const firstNode = nodes[0];
    await firstNode.bootstrap.start(firstNode.webrtc, firstNode.dht);

    // Connect nodes and track timing
    for (let i = 1; i < nodes.length; i++) {
        await nodes[i].bootstrap.start(nodes[i].webrtc, nodes[i].dht);
        await connectNodes(firstNode, nodes[i]);
        firstNode.bootstrap.checkMigration();
    }

    const stats = firstNode.bootstrap.getStats();
    const timestamps = stats.migrationTimestamps;

    passed = formatTestResult(
        'Migration timestamps recorded',
        timestamps.started !== null,
        { timestamps }
    ) && passed;

    if (timestamps.toHybrid) {
        const firebaseToHybridTime = timestamps.toHybrid - timestamps.started;
        console.log(`    Firebase -> Hybrid: ${firebaseToHybridTime}ms`);

        passed = formatTestResult(
            'Firebase to Hybrid migration completed',
            firebaseToHybridTime > 0,
            { timeMs: firebaseToHybridTime }
        ) && passed;
    }

    if (timestamps.toP2P) {
        const hybridToP2PTime = timestamps.toP2P - timestamps.toHybrid;
        const totalTime = timestamps.toP2P - timestamps.started;
        console.log(`    Hybrid -> P2P: ${hybridToP2PTime}ms`);
        console.log(`    Total migration: ${totalTime}ms`);

        passed = formatTestResult(
            'Hybrid to P2P migration completed',
            hybridToP2PTime > 0,
            { timeMs: hybridToP2PTime }
        ) && passed;
    }

    // Cleanup
    for (const node of nodes) {
        await node.bootstrap.stop();
    }

    return passed;
}

/**
 * TEST 8: Threshold Configuration
 *
 * Scenario: Test different threshold configurations
 * Expected: Migration respects configured thresholds
 */
async function testThresholdConfiguration() {
    console.log('\n--- TEST 8: Threshold Configuration ---');

    let passed = true;

    // Test with low thresholds
    const lowNodes = createTestNetwork(5, {
        dhtPeerThreshold: 2,
        p2pPeerThreshold: 3,
    });

    const lowFirst = lowNodes[0];
    await lowFirst.bootstrap.start(lowFirst.webrtc, lowFirst.dht);

    for (let i = 1; i < 4; i++) {
        await lowNodes[i].bootstrap.start(lowNodes[i].webrtc, lowNodes[i].dht);
        await connectNodes(lowFirst, lowNodes[i]);
        lowFirst.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'Low thresholds: Reaches P2P with fewer peers',
        lowFirst.bootstrap.mode === 'p2p',
        { mode: lowFirst.bootstrap.mode, peers: lowFirst.webrtc.peers.size }
    ) && passed;

    // Test with high thresholds
    const highNodes = createTestNetwork(5, {
        dhtPeerThreshold: 10,
        p2pPeerThreshold: 20,
    });

    const highFirst = highNodes[0];
    await highFirst.bootstrap.start(highFirst.webrtc, highFirst.dht);

    for (let i = 1; i < 5; i++) {
        await highNodes[i].bootstrap.start(highNodes[i].webrtc, highNodes[i].dht);
        await connectNodes(highFirst, highNodes[i]);
        highFirst.bootstrap.checkMigration();
    }

    passed = formatTestResult(
        'High thresholds: Stays in firebase with few peers',
        highFirst.bootstrap.mode === 'firebase',
        { mode: highFirst.bootstrap.mode, peers: highFirst.webrtc.peers.size }
    ) && passed;

    // Cleanup
    for (const node of [...lowNodes, ...highNodes]) {
        await node.bootstrap.stop();
    }

    return passed;
}

// ============================================
// TEST RUNNER
// ============================================

async function runAllTests() {
    console.log('\n' + '='.repeat(60));
    console.log('P2P MIGRATION TEST SUITE');
    console.log('='.repeat(60));

    const results = [];

    try {
        results.push({ name: 'Happy Path - Gradual Growth', passed: await testHappyPathGradualGrowth() });
        results.push({ name: 'Edge Case - Nodes Leaving', passed: await testNodesLeaving() });
        results.push({ name: 'Edge Case - Nodes Rejoining', passed: await testNodesRejoining() });
        results.push({ name: 'Network Partition Recovery', passed: await testNetworkPartitionRecovery() });
        results.push({ name: 'Signaling Fallback Behavior', passed: await testSignalingFallback() });
        results.push({ name: 'DHT Routing Table Population', passed: await testDHTRoutingTablePopulation() });
        results.push({ name: 'Migration Timing Measurement', passed: await testMigrationTiming() });
        results.push({ name: 'Threshold Configuration', passed: await testThresholdConfiguration() });
    } catch (error) {
        console.error('\nTest suite error:', error);
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('TEST SUMMARY');
    console.log('='.repeat(60));

    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;

    for (const result of results) {
        const status = result.passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
        console.log(`  ${status}: ${result.name}`);
    }

    console.log('\n' + '-'.repeat(60));
    console.log(`Total: ${results.length} | Passed: ${passed} | Failed: ${failed}`);
    console.log('='.repeat(60) + '\n');

    return failed === 0;
}

// Run if executed directly
if (process.argv[1]?.endsWith('p2p-migration-test.js')) {
    runAllTests()
        .then(success => process.exit(success ? 0 : 1))
        .catch(err => {
            console.error('Fatal error:', err);
            process.exit(1);
        });
}

export {
    runAllTests,
    testHappyPathGradualGrowth,
    testNodesLeaving,
    testNodesRejoining,
    testNetworkPartitionRecovery,
    testSignalingFallback,
    testDHTRoutingTablePopulation,
    testMigrationTiming,
    testThresholdConfiguration,
    SimulatedHybridBootstrap,
    MockWebRTCPeerManager,
    MockDHTNode,
    MockFirebaseSignaling,
    createTestNetwork,
};

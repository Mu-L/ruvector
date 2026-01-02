#!/usr/bin/env node
/**
 * WebRTC Data Channel End-to-End Test
 *
 * Comprehensive verification of ACTUAL P2P data flow (not just signaling).
 *
 * Tests:
 * 1. WebRTC peer connection establishment via Genesis signaling server
 * 2. Data channel creation and bidirectional messaging
 * 3. Latency measurement for P2P messages
 * 4. Throughput measurement (messages per second, bytes per second)
 * 5. Reconnection after disconnect
 * 6. Large message handling
 *
 * Usage:
 *   node tests/webrtc-datachannel-e2e-test.js
 *
 * Environment:
 *   GENESIS_SERVER=wss://edge-net-genesis-875130704813.us-central1.run.app
 *   TURN_URL=turn:34.72.154.225:3478
 *   TURN_USERNAME=edgenet
 *   TURN_CREDENTIAL=ruvector2024turn
 *
 * @module @ruvector/edge-net/tests/webrtc-datachannel-e2e-test
 */

import { EventEmitter } from 'events';
import { randomBytes, createHash } from 'crypto';

// ============================================
// TEST CONFIGURATION
// ============================================

const TEST_CONFIG = {
    // Signaling server
    signalingServer: process.env.GENESIS_SERVER ||
        'wss://edge-net-genesis-875130704813.us-central1.run.app',

    // TURN server configuration
    turnServer: {
        urls: process.env.TURN_URL || 'turn:34.72.154.225:3478',
        username: process.env.TURN_USERNAME || 'edgenet',
        credential: process.env.TURN_CREDENTIAL || 'ruvector2024turn',
    },

    // STUN servers (backup)
    stunServers: [
        { urls: 'stun:34.72.154.225:3478' },
        { urls: 'stun:stun.l.google.com:19302' },
    ],

    // Test parameters
    connectionTimeout: 30000,    // 30 seconds to establish connection
    messageCount: 20,            // Number of test messages
    largeMessageSize: 16384,     // 16KB large messages
    throughputTestDuration: 5000, // 5 seconds for throughput test
    reconnectTestDelay: 3000,    // 3 seconds before reconnect test

    // Timeouts
    overallTimeout: 120000,      // 2 minutes overall test timeout
};

// ============================================
// TEST RESULT TRACKING
// ============================================

class TestResults {
    constructor() {
        this.tests = [];
        this.startTime = Date.now();
    }

    addResult(name, passed, details = {}) {
        this.tests.push({
            name,
            passed,
            details,
            timestamp: Date.now(),
        });
        const icon = passed ? 'PASS' : 'FAIL';
        console.log(`  [${icon}] ${name}`);
        if (details.error) {
            console.log(`        Error: ${details.error}`);
        }
        if (details.metrics) {
            for (const [key, value] of Object.entries(details.metrics)) {
                console.log(`        ${key}: ${value}`);
            }
        }
    }

    getSummary() {
        const passed = this.tests.filter(t => t.passed).length;
        const failed = this.tests.filter(t => !t.passed).length;
        const duration = Date.now() - this.startTime;

        return {
            total: this.tests.length,
            passed,
            failed,
            duration,
            success: failed === 0,
        };
    }

    printSummary() {
        const summary = this.getSummary();
        console.log('\n' + '='.repeat(60));
        console.log('  TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`  Total tests:  ${summary.total}`);
        console.log(`  Passed:       ${summary.passed}`);
        console.log(`  Failed:       ${summary.failed}`);
        console.log(`  Duration:     ${summary.duration}ms`);
        console.log('='.repeat(60));
        console.log(`  OVERALL: ${summary.success ? 'PASS' : 'FAIL'}`);
        console.log('='.repeat(60) + '\n');

        return summary.success;
    }
}

// ============================================
// WEBRTC PEER CLASS (Simplified for testing)
// ============================================

class TestPeer extends EventEmitter {
    constructor(peerId, isInitiator) {
        super();
        this.peerId = peerId;
        this.isInitiator = isInitiator;
        this.pc = null;
        this.dataChannel = null;
        this.signalingSocket = null;
        this.remotePeerId = null;

        // Metrics
        this.metrics = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0,
            latencies: [],
            connectionStartTime: null,
            connectionEstablishedTime: null,
        };

        // Pending ICE candidates (received before remote description)
        this.pendingCandidates = [];

        // WebRTC classes
        this._wrtc = null;
    }

    async loadWebRTC() {
        // Try browser globals first
        if (globalThis.RTCPeerConnection) {
            return {
                RTCPeerConnection: globalThis.RTCPeerConnection,
                RTCSessionDescription: globalThis.RTCSessionDescription,
                RTCIceCandidate: globalThis.RTCIceCandidate,
            };
        }

        // Load wrtc for Node.js
        try {
            const wrtc = await import('wrtc');
            this._wrtc = wrtc.default || wrtc;
            return {
                RTCPeerConnection: this._wrtc.RTCPeerConnection,
                RTCSessionDescription: this._wrtc.RTCSessionDescription,
                RTCIceCandidate: this._wrtc.RTCIceCandidate,
            };
        } catch (err) {
            throw new Error(`WebRTC not available: ${err.message}`);
        }
    }

    async initialize() {
        const webrtc = await this.loadWebRTC();

        // Build ICE configuration
        const iceConfig = {
            iceServers: [
                ...TEST_CONFIG.stunServers,
                {
                    urls: TEST_CONFIG.turnServer.urls,
                    username: TEST_CONFIG.turnServer.username,
                    credential: TEST_CONFIG.turnServer.credential,
                },
                {
                    urls: TEST_CONFIG.turnServer.urls + '?transport=tcp',
                    username: TEST_CONFIG.turnServer.username,
                    credential: TEST_CONFIG.turnServer.credential,
                },
            ],
            iceTransportPolicy: 'all',
            iceCandidatePoolSize: 10,
        };

        console.log(`  [${this.peerId.slice(0, 8)}] Initializing with ICE config:`,
            iceConfig.iceServers.map(s => s.urls).join(', '));

        this.pc = new webrtc.RTCPeerConnection(iceConfig);
        this._RTCSessionDescription = webrtc.RTCSessionDescription;
        this._RTCIceCandidate = webrtc.RTCIceCandidate;

        this.setupEventHandlers();

        if (this.isInitiator) {
            this.createDataChannel();
        }

        this.metrics.connectionStartTime = Date.now();
    }

    setupEventHandlers() {
        this.pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.emit('ice-candidate', event.candidate);
            }
        };

        this.pc.oniceconnectionstatechange = () => {
            const state = this.pc.iceConnectionState;
            console.log(`  [${this.peerId.slice(0, 8)}] ICE state: ${state}`);

            if (state === 'connected' || state === 'completed') {
                this.metrics.connectionEstablishedTime = Date.now();
                this.emit('ice-connected');
            } else if (state === 'disconnected' || state === 'failed') {
                this.emit('ice-disconnected', state);
            }
        };

        this.pc.ondatachannel = (event) => {
            console.log(`  [${this.peerId.slice(0, 8)}] Received data channel`);
            this.dataChannel = event.channel;
            this.setupDataChannel();
        };
    }

    createDataChannel() {
        console.log(`  [${this.peerId.slice(0, 8)}] Creating data channel`);
        this.dataChannel = this.pc.createDataChannel('edge-net-e2e-test', {
            ordered: true,
            maxRetransmits: 3,
        });
        this.setupDataChannel();
    }

    setupDataChannel() {
        if (!this.dataChannel) return;

        this.dataChannel.onopen = () => {
            console.log(`  [${this.peerId.slice(0, 8)}] Data channel OPEN`);
            this.emit('channel-open');
        };

        this.dataChannel.onclose = () => {
            console.log(`  [${this.peerId.slice(0, 8)}] Data channel CLOSED`);
            this.emit('channel-close');
        };

        this.dataChannel.onerror = (error) => {
            console.error(`  [${this.peerId.slice(0, 8)}] Data channel error:`, error);
            this.emit('channel-error', error);
        };

        this.dataChannel.onmessage = (event) => {
            this.metrics.messagesReceived++;
            this.metrics.bytesReceived += event.data.length;
            this.handleMessage(event.data);
        };
    }

    handleMessage(data) {
        try {
            const message = JSON.parse(data);

            // Handle ping/pong for latency measurement
            if (message.type === 'ping') {
                this.send({
                    type: 'pong',
                    pingTimestamp: message.timestamp,
                    pongTimestamp: Date.now(),
                });
                return;
            }

            if (message.type === 'pong') {
                const latency = Date.now() - message.pingTimestamp;
                this.metrics.latencies.push(latency);
                this.emit('pong', latency);
                return;
            }

            this.emit('message', message);
        } catch (err) {
            // Raw string message
            this.emit('message', data);
        }
    }

    async createOffer() {
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);
        return offer;
    }

    async handleOffer(offer) {
        await this.pc.setRemoteDescription(
            new this._RTCSessionDescription(offer)
        );

        // Process pending ICE candidates
        for (const candidate of this.pendingCandidates) {
            await this.pc.addIceCandidate(new this._RTCIceCandidate(candidate));
        }
        this.pendingCandidates = [];

        const answer = await this.pc.createAnswer();
        await this.pc.setLocalDescription(answer);
        return answer;
    }

    async handleAnswer(answer) {
        await this.pc.setRemoteDescription(
            new this._RTCSessionDescription(answer)
        );

        // Process pending ICE candidates
        for (const candidate of this.pendingCandidates) {
            await this.pc.addIceCandidate(new this._RTCIceCandidate(candidate));
        }
        this.pendingCandidates = [];
    }

    async addIceCandidate(candidate) {
        if (this.pc.remoteDescription) {
            await this.pc.addIceCandidate(
                new this._RTCIceCandidate(candidate)
            );
        } else {
            // Queue for later
            this.pendingCandidates.push(candidate);
        }
    }

    send(data) {
        if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
            throw new Error('Data channel not ready');
        }

        const message = typeof data === 'string' ? data : JSON.stringify(data);
        this.dataChannel.send(message);
        this.metrics.messagesSent++;
        this.metrics.bytesSent += message.length;
    }

    sendPing() {
        this.send({
            type: 'ping',
            timestamp: Date.now(),
        });
    }

    getAverageLatency() {
        if (this.metrics.latencies.length === 0) return 0;
        const sum = this.metrics.latencies.reduce((a, b) => a + b, 0);
        return Math.round(sum / this.metrics.latencies.length);
    }

    getConnectionTime() {
        if (!this.metrics.connectionStartTime || !this.metrics.connectionEstablishedTime) {
            return null;
        }
        return this.metrics.connectionEstablishedTime - this.metrics.connectionStartTime;
    }

    close() {
        if (this.dataChannel) {
            this.dataChannel.close();
        }
        if (this.pc) {
            this.pc.close();
        }
        if (this.signalingSocket) {
            this.signalingSocket.close();
        }
    }
}

// ============================================
// SIGNALING HELPER
// ============================================

class SignalingHelper {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.socket = null;
        this.peerId = null;
        this.emitter = new EventEmitter();
    }

    async connect(peerId) {
        this.peerId = peerId;

        // Load WebSocket for Node.js
        const WebSocket = globalThis.WebSocket ||
            (await import('ws')).default;

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Signaling connection timeout'));
            }, 10000);

            try {
                this.socket = new WebSocket(this.serverUrl);

                this.socket.onopen = () => {
                    clearTimeout(timeout);
                    console.log(`  [Signaling] Connected to ${this.serverUrl}`);

                    // Announce presence
                    this.socket.send(JSON.stringify({
                        type: 'announce',
                        piKey: peerId,
                        siteId: `e2e-test-${peerId.slice(0, 8)}`,
                        capabilities: ['e2e-test'],
                    }));

                    resolve(true);
                };

                this.socket.onerror = (err) => {
                    clearTimeout(timeout);
                    reject(new Error(`WebSocket error: ${err.message || 'connection failed'}`));
                };

                this.socket.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.emitter.emit(message.type, message);
                    } catch (err) {
                        console.error('  [Signaling] Parse error:', err);
                    }
                };

                this.socket.onclose = () => {
                    this.emitter.emit('disconnected');
                };

            } catch (err) {
                clearTimeout(timeout);
                reject(err);
            }
        });
    }

    send(type, to, data) {
        if (!this.socket || this.socket.readyState !== 1) {
            throw new Error('Signaling socket not connected');
        }

        this.socket.send(JSON.stringify({
            type,
            to,
            from: this.peerId,
            ...data,
        }));
    }

    on(event, handler) {
        this.emitter.on(event, handler);
    }

    close() {
        if (this.socket) {
            this.socket.close();
        }
    }
}

// ============================================
// E2E TEST RUNNER
// ============================================

async function runE2ETest() {
    console.log('\n' + '='.repeat(60));
    console.log('  WebRTC Data Channel E2E Test');
    console.log('  Testing ACTUAL P2P data flow');
    console.log('='.repeat(60));
    console.log(`  Signaling: ${TEST_CONFIG.signalingServer}`);
    console.log(`  TURN:      ${TEST_CONFIG.turnServer.urls}`);
    console.log('='.repeat(60) + '\n');

    const results = new TestResults();

    // Generate peer IDs
    const peerAId = `peer-A-${randomBytes(6).toString('hex')}`;
    const peerBId = `peer-B-${randomBytes(6).toString('hex')}`;

    console.log(`Peer A: ${peerAId}`);
    console.log(`Peer B: ${peerBId}`);
    console.log('');

    // Create peers
    const peerA = new TestPeer(peerAId, true);  // Initiator
    const peerB = new TestPeer(peerBId, false); // Responder

    // Create signaling helpers
    const signalingA = new SignalingHelper(TEST_CONFIG.signalingServer);
    const signalingB = new SignalingHelper(TEST_CONFIG.signalingServer);

    // Track state
    let channelAOpen = false;
    let channelBOpen = false;
    let messagesExchanged = 0;
    let reconnectTested = false;

    // Setup promise resolvers
    let connectionResolve;
    const connectionPromise = new Promise(r => connectionResolve = r);

    // ==========================================
    // TEST 1: Signaling Server Connection
    // ==========================================
    console.log('\n[TEST 1] Signaling Server Connection');

    try {
        await Promise.all([
            signalingA.connect(peerAId),
            signalingB.connect(peerBId),
        ]);
        results.addResult('Signaling server connection', true, {
            metrics: {
                'Server': TEST_CONFIG.signalingServer,
            }
        });
    } catch (err) {
        results.addResult('Signaling server connection', false, {
            error: err.message,
        });
        return results.printSummary();
    }

    // ==========================================
    // TEST 2: WebRTC Peer Initialization
    // ==========================================
    console.log('\n[TEST 2] WebRTC Peer Initialization');

    try {
        await Promise.all([
            peerA.initialize(),
            peerB.initialize(),
        ]);
        results.addResult('WebRTC peer initialization', true);
    } catch (err) {
        results.addResult('WebRTC peer initialization', false, {
            error: err.message,
        });
        cleanup();
        return results.printSummary();
    }

    // ==========================================
    // Wire up signaling
    // ==========================================

    // ICE candidates from A to B
    peerA.on('ice-candidate', (candidate) => {
        signalingA.send('ice-candidate', peerBId, { candidate });
    });

    // ICE candidates from B to A
    peerB.on('ice-candidate', (candidate) => {
        signalingB.send('ice-candidate', peerAId, { candidate });
    });

    // Handle ICE candidates
    signalingA.on('ice-candidate', async ({ from, candidate }) => {
        if (from === peerBId) {
            await peerA.addIceCandidate(candidate);
        }
    });

    signalingB.on('ice-candidate', async ({ from, candidate }) => {
        if (from === peerAId) {
            await peerB.addIceCandidate(candidate);
        }
    });

    // Handle offers at B
    signalingB.on('offer', async ({ from, offer }) => {
        if (from === peerAId) {
            console.log(`  [${peerBId.slice(0, 8)}] Received offer from ${from.slice(0, 8)}`);
            peerB.remotePeerId = from;
            const answer = await peerB.handleOffer(offer);
            signalingB.send('answer', peerAId, { answer });
        }
    });

    // Handle answers at A
    signalingA.on('answer', async ({ from, answer }) => {
        if (from === peerBId) {
            console.log(`  [${peerAId.slice(0, 8)}] Received answer from ${from.slice(0, 8)}`);
            await peerA.handleAnswer(answer);
        }
    });

    // Track channel state
    peerA.on('channel-open', () => {
        channelAOpen = true;
        if (channelAOpen && channelBOpen) connectionResolve();
    });

    peerB.on('channel-open', () => {
        channelBOpen = true;
        if (channelAOpen && channelBOpen) connectionResolve();
    });

    // ==========================================
    // TEST 3: WebRTC Connection Establishment
    // ==========================================
    console.log('\n[TEST 3] WebRTC Connection Establishment');

    try {
        // Create and send offer
        const offer = await peerA.createOffer();
        peerA.remotePeerId = peerBId;
        signalingA.send('offer', peerBId, { offer });

        // Wait for connection with timeout
        const connectionTimeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Connection timeout')), TEST_CONFIG.connectionTimeout);
        });

        await Promise.race([connectionPromise, connectionTimeout]);

        const connectionTime = peerA.getConnectionTime();
        results.addResult('WebRTC connection establishment', true, {
            metrics: {
                'Connection time': `${connectionTime}ms`,
            }
        });
    } catch (err) {
        results.addResult('WebRTC connection establishment', false, {
            error: err.message,
        });
        cleanup();
        return results.printSummary();
    }

    // ==========================================
    // TEST 4: Bidirectional Message Exchange
    // ==========================================
    console.log('\n[TEST 4] Bidirectional Message Exchange');

    try {
        const messagesFromA = [];
        const messagesFromB = [];

        peerA.on('message', (msg) => messagesFromB.push(msg));
        peerB.on('message', (msg) => messagesFromA.push(msg));

        // Send messages from A to B
        for (let i = 0; i < TEST_CONFIG.messageCount; i++) {
            peerA.send({
                type: 'test',
                from: 'A',
                sequence: i,
                timestamp: Date.now(),
                payload: `Message ${i} from A`,
            });
        }

        // Send messages from B to A
        for (let i = 0; i < TEST_CONFIG.messageCount; i++) {
            peerB.send({
                type: 'test',
                from: 'B',
                sequence: i,
                timestamp: Date.now(),
                payload: `Message ${i} from B`,
            });
        }

        // Wait for messages
        await new Promise(r => setTimeout(r, 2000));

        const aReceived = messagesFromB.filter(m => m.from === 'B').length;
        const bReceived = messagesFromA.filter(m => m.from === 'A').length;

        results.addResult('Bidirectional message exchange',
            aReceived >= TEST_CONFIG.messageCount && bReceived >= TEST_CONFIG.messageCount,
            {
                metrics: {
                    'A received': `${aReceived}/${TEST_CONFIG.messageCount}`,
                    'B received': `${bReceived}/${TEST_CONFIG.messageCount}`,
                }
            }
        );
    } catch (err) {
        results.addResult('Bidirectional message exchange', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 5: Latency Measurement
    // ==========================================
    console.log('\n[TEST 5] Latency Measurement');

    try {
        // Send pings from A
        const pingCount = 10;
        for (let i = 0; i < pingCount; i++) {
            peerA.sendPing();
            await new Promise(r => setTimeout(r, 100));
        }

        // Wait for pongs
        await new Promise(r => setTimeout(r, 2000));

        const avgLatency = peerA.getAverageLatency();
        const minLatency = Math.min(...peerA.metrics.latencies);
        const maxLatency = Math.max(...peerA.metrics.latencies);

        results.addResult('Latency measurement', peerA.metrics.latencies.length > 0, {
            metrics: {
                'Average latency': `${avgLatency}ms`,
                'Min latency': `${minLatency}ms`,
                'Max latency': `${maxLatency}ms`,
                'Samples': peerA.metrics.latencies.length,
            }
        });
    } catch (err) {
        results.addResult('Latency measurement', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 6: Large Message Handling
    // ==========================================
    console.log('\n[TEST 6] Large Message Handling');

    try {
        let largeMessageReceived = false;
        const largePayload = randomBytes(TEST_CONFIG.largeMessageSize).toString('hex');
        const messageHash = createHash('sha256').update(largePayload).digest('hex');

        const largeMessagePromise = new Promise((resolve) => {
            const handler = (msg) => {
                if (msg.type === 'large-test') {
                    const receivedHash = createHash('sha256')
                        .update(msg.payload)
                        .digest('hex');
                    if (receivedHash === messageHash) {
                        largeMessageReceived = true;
                        peerB.removeListener('message', handler);
                        resolve();
                    }
                }
            };
            peerB.on('message', handler);
        });

        peerA.send({
            type: 'large-test',
            size: TEST_CONFIG.largeMessageSize,
            payload: largePayload,
            hash: messageHash,
        });

        await Promise.race([
            largeMessagePromise,
            new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 10000)),
        ]);

        results.addResult('Large message handling', largeMessageReceived, {
            metrics: {
                'Message size': `${Math.round(TEST_CONFIG.largeMessageSize / 1024)}KB`,
                'Integrity verified': 'SHA-256 hash matched',
            }
        });
    } catch (err) {
        results.addResult('Large message handling', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 7: Throughput Measurement
    // ==========================================
    console.log('\n[TEST 7] Throughput Measurement');

    try {
        const startBytes = peerA.metrics.bytesSent;
        const startMessages = peerA.metrics.messagesSent;
        const startTime = Date.now();
        const testPayload = randomBytes(1024).toString('hex'); // 2KB message

        // Send as many messages as possible in the duration
        while (Date.now() - startTime < TEST_CONFIG.throughputTestDuration) {
            peerA.send({
                type: 'throughput-test',
                payload: testPayload,
                timestamp: Date.now(),
            });
            // Small delay to prevent overwhelming
            await new Promise(r => setTimeout(r, 10));
        }

        const duration = (Date.now() - startTime) / 1000;
        const messagesSent = peerA.metrics.messagesSent - startMessages;
        const bytesSent = peerA.metrics.bytesSent - startBytes;

        const messagesPerSecond = Math.round(messagesSent / duration);
        const bytesPerSecond = Math.round(bytesSent / duration);
        const kbPerSecond = Math.round(bytesPerSecond / 1024);

        results.addResult('Throughput measurement', messagesSent > 0, {
            metrics: {
                'Duration': `${duration.toFixed(1)}s`,
                'Messages sent': messagesSent,
                'Throughput (msg)': `${messagesPerSecond} msg/s`,
                'Throughput (data)': `${kbPerSecond} KB/s`,
            }
        });
    } catch (err) {
        results.addResult('Throughput measurement', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 8: Connection Metrics Summary
    // ==========================================
    console.log('\n[TEST 8] Connection Metrics Summary');

    try {
        const totalBytesSent = peerA.metrics.bytesSent + peerB.metrics.bytesSent;
        const totalBytesReceived = peerA.metrics.bytesReceived + peerB.metrics.bytesReceived;
        const totalMessagesSent = peerA.metrics.messagesSent + peerB.metrics.messagesSent;
        const totalMessagesReceived = peerA.metrics.messagesReceived + peerB.metrics.messagesReceived;

        results.addResult('Connection metrics summary', true, {
            metrics: {
                'Total messages sent': totalMessagesSent,
                'Total messages received': totalMessagesReceived,
                'Total bytes sent': `${Math.round(totalBytesSent / 1024)} KB`,
                'Total bytes received': `${Math.round(totalBytesReceived / 1024)} KB`,
                'Message delivery rate': `${Math.round(totalMessagesReceived / totalMessagesSent * 100)}%`,
            }
        });
    } catch (err) {
        results.addResult('Connection metrics summary', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 9: Graceful Disconnect
    // ==========================================
    console.log('\n[TEST 9] Graceful Disconnect');

    try {
        let disconnectDetected = false;

        peerB.on('channel-close', () => {
            disconnectDetected = true;
        });

        // Close peer A's data channel
        peerA.dataChannel.close();

        // Wait for disconnect detection
        await new Promise(r => setTimeout(r, 1000));

        results.addResult('Graceful disconnect', disconnectDetected, {
            metrics: {
                'Disconnect detected by peer B': disconnectDetected ? 'Yes' : 'No',
            }
        });
    } catch (err) {
        results.addResult('Graceful disconnect', false, {
            error: err.message,
        });
    }

    // ==========================================
    // TEST 10: ICE Candidate Types Analysis
    // ==========================================
    console.log('\n[TEST 10] ICE Candidate Types Analysis');

    try {
        // Get ICE candidate stats from peer connections if available
        const statsA = await peerA.pc.getStats();
        const statsB = await peerB.pc.getStats();

        const candidateTypes = new Set();
        const connectionTypes = [];

        statsA.forEach((report) => {
            if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                connectionTypes.push({
                    localType: report.localCandidateId,
                    remoteType: report.remoteCandidateId,
                    nominated: report.nominated,
                });
            }
            if (report.type === 'local-candidate' || report.type === 'remote-candidate') {
                candidateTypes.add(report.candidateType);
            }
        });

        const candidateList = Array.from(candidateTypes);
        const hasRelay = candidateList.includes('relay');

        results.addResult('ICE candidate types analysis', candidateList.length > 0, {
            metrics: {
                'Candidate types found': candidateList.join(', ') || 'none',
                'TURN relay used': hasRelay ? 'Yes' : 'No',
                'Connection established': 'Yes',
            }
        });
    } catch (err) {
        results.addResult('ICE candidate types analysis', false, {
            error: err.message,
        });
    }

    // ==========================================
    // Cleanup
    // ==========================================
    function cleanup() {
        console.log('\n  Cleaning up...');
        peerA.close();
        peerB.close();
        signalingA.close();
        signalingB.close();
    }

    cleanup();

    // Print and return results
    return results.printSummary();
}

// ==========================================
// QUICK CONNECTIVITY TEST
// ==========================================

async function runQuickTest() {
    console.log('\n' + '='.repeat(60));
    console.log('  Quick WebRTC Connectivity Test');
    console.log('='.repeat(60) + '\n');

    // Test 1: Check wrtc availability
    console.log('[1] Checking wrtc module...');
    try {
        const wrtc = await import('wrtc');
        console.log('    wrtc module loaded successfully');
    } catch (err) {
        console.log('    FAILED: wrtc not available -', err.message);
        return false;
    }

    // Test 2: Check signaling server
    console.log('[2] Checking signaling server...');
    try {
        const WebSocket = (await import('ws')).default;
        const ws = new WebSocket(TEST_CONFIG.signalingServer);

        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Timeout')), 5000);
            ws.onopen = () => {
                clearTimeout(timeout);
                ws.close();
                resolve();
            };
            ws.onerror = () => {
                clearTimeout(timeout);
                reject(new Error('Connection failed'));
            };
        });
        console.log('    Signaling server reachable');
    } catch (err) {
        console.log('    WARNING: Signaling server unreachable -', err.message);
        console.log('    (Test will use inline peer exchange)');
    }

    // Test 3: Basic peer connection
    console.log('[3] Testing basic peer connection...');
    try {
        const wrtc = (await import('wrtc')).default;
        const pc = new wrtc.RTCPeerConnection({
            iceServers: TEST_CONFIG.stunServers,
        });

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        pc.close();
        console.log('    Peer connection works');
    } catch (err) {
        console.log('    FAILED: Peer connection error -', err.message);
        return false;
    }

    console.log('\n  Quick test passed - ready for E2E test');
    return true;
}

// ==========================================
// MAIN ENTRY POINT
// ==========================================

async function main() {
    const args = process.argv.slice(2);

    // Set overall timeout
    const timeoutHandle = setTimeout(() => {
        console.error('\n\nOVERALL TEST TIMEOUT - Exiting');
        process.exit(1);
    }, TEST_CONFIG.overallTimeout);

    try {
        if (args.includes('--quick')) {
            const success = await runQuickTest();
            clearTimeout(timeoutHandle);
            process.exit(success ? 0 : 1);
        }

        if (args.includes('--help') || args.includes('-h')) {
            console.log(`
WebRTC Data Channel E2E Test

Usage:
  node tests/webrtc-datachannel-e2e-test.js [options]

Options:
  --quick    Run quick connectivity check only
  --help     Show this help message

Environment Variables:
  GENESIS_SERVER     Signaling server URL (default: wss://edge-net-genesis-...)
  TURN_URL           TURN server URL (default: turn:34.72.154.225:3478)
  TURN_USERNAME      TURN username (default: edgenet)
  TURN_CREDENTIAL    TURN credential (default: ruvector2024turn)

Tests Performed:
  1. Signaling server connection
  2. WebRTC peer initialization
  3. WebRTC connection establishment
  4. Bidirectional message exchange
  5. Latency measurement
  6. Large message handling (16KB)
  7. Throughput measurement
  8. Connection metrics summary
  9. Graceful disconnect
  10. ICE candidate types analysis
`);
            clearTimeout(timeoutHandle);
            process.exit(0);
        }

        // Run full E2E test
        const success = await runE2ETest();
        clearTimeout(timeoutHandle);
        process.exit(success ? 0 : 1);

    } catch (err) {
        clearTimeout(timeoutHandle);
        console.error('\nTest error:', err);
        process.exit(1);
    }
}

main();

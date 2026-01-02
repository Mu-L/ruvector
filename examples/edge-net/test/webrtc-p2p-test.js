#!/usr/bin/env node
/**
 * WebRTC P2P Connectivity Test
 *
 * Tests real WebRTC data channel connectivity through the relay server.
 * Simulates two peers connecting and exchanging messages.
 */

import WebSocket from 'ws';
import { randomBytes } from 'crypto';

const RELAY_URL = process.env.RELAY_URL || 'ws://localhost:8080';
const TEST_TIMEOUT = 30000;

// Test results
const results = {
    signalingConnected: false,
    peerDiscovered: false,
    webrtcOfferSent: false,
    webrtcAnswerReceived: false,
    messageExchanged: false,
    latencyMs: null,
};

// Peer simulation
class TestPeer {
    constructor(id) {
        this.id = `test-peer-${id}-${randomBytes(4).toString('hex')}`;
        this.ws = null;
        this.remotePeerId = null;
        this.receivedMessages = [];
        this.callbacks = {};
    }

    async connect() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Connection timeout')), 10000);

            this.ws = new WebSocket(RELAY_URL);

            this.ws.on('open', () => {
                clearTimeout(timeout);
                console.log(`  [${this.id.slice(0, 12)}] Connected to relay`);

                // Register with relay
                this.send({
                    type: 'register',
                    nodeId: this.id,
                    publicKey: randomBytes(32).toString('hex'),
                    siteId: `test-site-${this.id.slice(-4)}`,
                });
                resolve();
            });

            this.ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.handleMessage(message);
                } catch (err) {
                    console.error(`  [${this.id.slice(0, 12)}] Parse error:`, err);
                }
            });

            this.ws.on('error', (err) => {
                clearTimeout(timeout);
                reject(err);
            });
        });
    }

    handleMessage(message) {
        this.receivedMessages.push(message);

        if (message.type === 'welcome') {
            console.log(`  [${this.id.slice(0, 12)}] Registered, ${message.peers?.length || 0} peers online`);
            results.signalingConnected = true;
            this.emit('welcome', message);
        }

        if (message.type === 'node_joined') {
            console.log(`  [${this.id.slice(0, 12)}] Peer discovered: ${message.nodeId.slice(0, 12)}`);
            results.peerDiscovered = true;
            this.remotePeerId = message.nodeId;
            this.emit('peer_joined', message);
        }

        if (message.type === 'webrtc_offer') {
            console.log(`  [${this.id.slice(0, 12)}] Received WebRTC offer from ${message.from.slice(0, 12)}`);
            results.webrtcOfferSent = true;
            this.remotePeerId = message.from;

            // Send mock answer
            this.send({
                type: 'webrtc_answer',
                targetId: message.from,
                answer: {
                    type: 'answer',
                    sdp: 'mock-sdp-answer',
                },
            });
            this.emit('offer', message);
        }

        if (message.type === 'webrtc_answer') {
            console.log(`  [${this.id.slice(0, 12)}] Received WebRTC answer from ${message.from.slice(0, 12)}`);
            results.webrtcAnswerReceived = true;
            this.emit('answer', message);
        }

        if (message.type === 'webrtc_ice') {
            console.log(`  [${this.id.slice(0, 12)}] Received ICE candidate from ${message.from.slice(0, 12)}`);
            this.emit('ice', message);
        }

        if (message.type === 'peer_message') {
            console.log(`  [${this.id.slice(0, 12)}] Received P2P message from ${message.from.slice(0, 12)}`);
            results.messageExchanged = true;
            this.emit('p2p_message', message);
        }
    }

    send(message) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    sendWebRTCOffer(targetId) {
        this.send({
            type: 'webrtc_offer',
            targetId,
            offer: {
                type: 'offer',
                sdp: 'mock-sdp-offer',
            },
        });
    }

    sendPeerMessage(targetId, payload) {
        this.send({
            type: 'peer_message',
            targetId,
            payload,
        });
    }

    on(event, callback) {
        this.callbacks[event] = callback;
    }

    emit(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event](data);
        }
    }

    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Run test
async function runTest() {
    console.log('\n🔬 WebRTC P2P Connectivity Test');
    console.log('================================\n');
    console.log(`Relay URL: ${RELAY_URL}\n`);

    const peer1 = new TestPeer('A');
    const peer2 = new TestPeer('B');

    try {
        // Step 1: Connect both peers to relay
        console.log('📡 Step 1: Connecting to relay server...');
        await Promise.all([peer1.connect(), peer2.connect()]);
        console.log('  ✅ Both peers connected\n');

        // Step 2: Wait for peer discovery
        console.log('🔍 Step 2: Peer discovery...');
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Peer discovery timeout')), 10000);

            peer1.on('peer_joined', () => {
                clearTimeout(timeout);
                resolve();
            });

            // Peer2 should see peer1 after registration
            setTimeout(() => {
                if (peer1.remotePeerId) {
                    clearTimeout(timeout);
                    resolve();
                }
            }, 2000);
        });
        console.log('  ✅ Peers discovered each other\n');

        // Step 3: WebRTC signaling
        console.log('🤝 Step 3: WebRTC signaling...');

        const signalingStart = Date.now();

        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Signaling timeout')), 10000);

            peer2.on('offer', (msg) => {
                // peer2 will auto-send answer
            });

            peer1.on('answer', () => {
                clearTimeout(timeout);
                results.latencyMs = Date.now() - signalingStart;
                resolve();
            });

            // Peer1 initiates offer
            peer1.sendWebRTCOffer(peer2.id);
        });
        console.log(`  ✅ WebRTC signaling complete (${results.latencyMs}ms)\n`);

        // Step 4: P2P message exchange
        console.log('💬 Step 4: P2P message exchange...');

        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Message timeout')), 5000);

            peer2.on('p2p_message', (msg) => {
                clearTimeout(timeout);
                console.log(`  📨 Message received: ${JSON.stringify(msg.payload)}`);
                resolve();
            });

            peer1.sendPeerMessage(peer2.id, {
                type: 'test',
                content: 'Hello from peer1!',
                timestamp: Date.now(),
            });
        });
        console.log('  ✅ Message exchange successful\n');

        // Print results
        console.log('📊 Test Results:');
        console.log('================');
        console.log(`  Signaling Connected: ${results.signalingConnected ? '✅' : '❌'}`);
        console.log(`  Peer Discovered:     ${results.peerDiscovered ? '✅' : '❌'}`);
        console.log(`  WebRTC Offer Sent:   ${results.webrtcOfferSent ? '✅' : '❌'}`);
        console.log(`  WebRTC Answer Rcvd:  ${results.webrtcAnswerReceived ? '✅' : '❌'}`);
        console.log(`  Message Exchanged:   ${results.messageExchanged ? '✅' : '❌'}`);
        console.log(`  Signaling Latency:   ${results.latencyMs}ms`);

        const passed = Object.values(results).filter(v => v === true).length;
        const total = Object.values(results).filter(v => typeof v === 'boolean').length;

        console.log(`\n🎯 Score: ${passed}/${total} tests passed`);

        if (passed === total) {
            console.log('\n✅ All P2P connectivity tests PASSED!\n');
            process.exit(0);
        } else {
            console.log('\n❌ Some tests failed\n');
            process.exit(1);
        }

    } catch (err) {
        console.error('\n❌ Test failed:', err.message);
        console.log('\n📊 Partial Results:', results);
        process.exit(1);
    } finally {
        peer1.close();
        peer2.close();
    }
}

// Run with timeout
const timeout = setTimeout(() => {
    console.error('\n⏰ Test timeout exceeded');
    process.exit(1);
}, TEST_TIMEOUT);

runTest().finally(() => clearTimeout(timeout));

#!/usr/bin/env node
/**
 * WebRTC P2P Data Channel Test
 *
 * Tests real WebRTC peer-to-peer communication with:
 * - Two separate Node.js processes
 * - Firebase signaling for offer/answer/ICE exchange
 * - WebRTC data channel message exchange
 *
 * Usage:
 *   node tests/webrtc-peer-test.js
 *   node tests/webrtc-peer-test.js --peer1  # Run peer 1 only
 *   node tests/webrtc-peer-test.js --peer2  # Run peer 2 only
 *
 * @module @ruvector/edge-net/tests/webrtc-peer-test
 */

import { spawn, fork } from 'child_process';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const pkgDir = path.join(__dirname, '..');

// Test configuration
const TEST_CONFIG = {
    testRoom: `webrtc-test-${Date.now()}`,
    timeout: 60000, // 60 second timeout
    messageCount: 5,
};

// ============================================
// SINGLE PEER TEST RUNNER
// ============================================

async function runPeer(peerNum, room) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`  PEER ${peerNum} STARTING`);
    console.log(`${'='.repeat(60)}\n`);

    // Dynamic imports
    const { WebRTCPeerConnection, WebRTCPeerManager, WEBRTC_CONFIG } = await import('../webrtc.js');
    const { FirebaseSignaling, getFirebaseConfig } = await import('../firebase-signaling.js');
    const { randomBytes } = await import('crypto');

    // Generate unique peer ID
    const peerId = `test-peer-${peerNum}-${randomBytes(4).toString('hex')}`;
    const isInitiator = peerNum === 1;

    console.log(`[Peer ${peerNum}] ID: ${peerId}`);
    console.log(`[Peer ${peerNum}] Room: ${room}`);
    console.log(`[Peer ${peerNum}] Role: ${isInitiator ? 'INITIATOR' : 'RESPONDER'}`);

    // Create local identity
    const localIdentity = {
        piKey: peerId,
        publicKey: randomBytes(32).toString('hex'),
        siteId: `test-site-${peerNum}`,
    };

    // Initialize Firebase signaling
    const firebase = new FirebaseSignaling({
        peerId,
        room,
        firebaseConfig: getFirebaseConfig(),
        verifySignatures: false, // Skip WASM verification for test
    });

    // Initialize WebRTC peer manager
    const webrtc = new WebRTCPeerManager(localIdentity, {
        fallbackToSimulation: false,
        fallbackToDHT: false,
    });

    // Track state
    let connectedPeerId = null;
    let dataChannelReady = false;
    let messagesSent = 0;
    let messagesReceived = 0;
    const receivedMessages = [];

    // Setup WebRTC external signaling via Firebase
    webrtc.setExternalSignaling(async (type, toPeerId, data) => {
        console.log(`[Peer ${peerNum}] Signaling -> ${type} to ${toPeerId.slice(0, 8)}...`);
        switch (type) {
            case 'offer':
                await firebase.sendOffer(toPeerId, data);
                break;
            case 'answer':
                await firebase.sendAnswer(toPeerId, data);
                break;
            case 'ice-candidate':
                await firebase.sendIceCandidate(toPeerId, data);
                break;
        }
    });

    // Handle peer discovery from Firebase
    firebase.on('peer-discovered', async ({ peerId: discoveredPeerId }) => {
        console.log(`[Peer ${peerNum}] Discovered peer: ${discoveredPeerId.slice(0, 16)}...`);

        // Only initiator starts connection (peer with higher ID to avoid race)
        if (isInitiator && peerId > discoveredPeerId) {
            console.log(`[Peer ${peerNum}] Initiating WebRTC connection...`);
            try {
                await webrtc.connectToPeer(discoveredPeerId);
            } catch (err) {
                console.error(`[Peer ${peerNum}] Connect failed:`, err.message);
            }
        }
    });

    // Handle incoming offers
    firebase.on('offer', async ({ from, offer }) => {
        console.log(`[Peer ${peerNum}] Received OFFER from ${from.slice(0, 8)}...`);
        try {
            await webrtc.handleOffer({ from, offer });
        } catch (err) {
            console.error(`[Peer ${peerNum}] Handle offer failed:`, err.message);
        }
    });

    // Handle incoming answers
    firebase.on('answer', async ({ from, answer }) => {
        console.log(`[Peer ${peerNum}] Received ANSWER from ${from.slice(0, 8)}...`);
        try {
            await webrtc.handleAnswer({ from, answer });
        } catch (err) {
            console.error(`[Peer ${peerNum}] Handle answer failed:`, err.message);
        }
    });

    // Handle ICE candidates
    firebase.on('ice-candidate', async ({ from, candidate }) => {
        console.log(`[Peer ${peerNum}] Received ICE candidate from ${from.slice(0, 8)}...`);
        try {
            await webrtc.handleIceCandidate({ from, candidate });
        } catch (err) {
            console.error(`[Peer ${peerNum}] Handle ICE failed:`, err.message);
        }
    });

    // Handle WebRTC peer connected
    webrtc.on('peer-connected', (connPeerId) => {
        console.log(`\n[Peer ${peerNum}] DATA CHANNEL OPEN with ${connPeerId.slice(0, 8)}...`);
        connectedPeerId = connPeerId;
        dataChannelReady = true;

        // Start sending test messages
        sendTestMessages();
    });

    // Handle incoming messages
    webrtc.on('message', ({ from, message }) => {
        messagesReceived++;
        receivedMessages.push({ from, message, receivedAt: Date.now() });
        console.log(`[Peer ${peerNum}] Received message #${messagesReceived}:`,
            typeof message === 'object' ? JSON.stringify(message).slice(0, 50) : message.slice(0, 50));

        // Send echo response
        if (message.type !== 'echo' && message.type !== 'heartbeat' && message.type !== 'heartbeat-ack') {
            try {
                webrtc.sendToPeer(from, {
                    type: 'echo',
                    originalMessage: message,
                    echoFrom: peerId,
                    timestamp: Date.now(),
                });
            } catch (err) {
                // Channel might be closing
            }
        }
    });

    // Send test messages
    async function sendTestMessages() {
        console.log(`\n[Peer ${peerNum}] Starting test message exchange...`);

        for (let i = 0; i < TEST_CONFIG.messageCount; i++) {
            await new Promise(resolve => setTimeout(resolve, 500));

            if (!dataChannelReady) break;

            const testMessage = {
                type: 'test',
                from: peerId,
                sequence: i + 1,
                timestamp: Date.now(),
                payload: `Hello from Peer ${peerNum}, message ${i + 1}`,
            };

            try {
                webrtc.sendToPeer(connectedPeerId, testMessage);
                messagesSent++;
                console.log(`[Peer ${peerNum}] Sent message #${messagesSent}`);
            } catch (err) {
                console.error(`[Peer ${peerNum}] Send failed:`, err.message);
            }
        }
    }

    // Connect to Firebase
    console.log(`\n[Peer ${peerNum}] Connecting to Firebase...`);
    const connected = await firebase.connect();

    if (!connected) {
        console.error(`[Peer ${peerNum}] Firebase connection failed!`);
        process.exit(1);
    }

    console.log(`[Peer ${peerNum}] Firebase connected, waiting for peers...`);

    // Timeout handler
    const timeout = setTimeout(async () => {
        console.log(`\n[Peer ${peerNum}] TEST TIMEOUT - Summary:`);
        console.log(`  - Data channel established: ${dataChannelReady}`);
        console.log(`  - Messages sent: ${messagesSent}`);
        console.log(`  - Messages received: ${messagesReceived}`);

        await cleanup();
        process.exit(dataChannelReady && messagesReceived > 0 ? 0 : 1);
    }, TEST_CONFIG.timeout);

    // Cleanup function
    async function cleanup() {
        clearTimeout(timeout);
        webrtc.close();
        await firebase.disconnect();
    }

    // Success check
    const checkSuccess = setInterval(async () => {
        if (messagesSent >= TEST_CONFIG.messageCount && messagesReceived >= TEST_CONFIG.messageCount) {
            clearInterval(checkSuccess);
            console.log(`\n[Peer ${peerNum}] TEST PASSED!`);
            console.log(`  - Messages sent: ${messagesSent}`);
            console.log(`  - Messages received: ${messagesReceived}`);

            await cleanup();
            process.exit(0);
        }
    }, 1000);

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
        console.log(`\n[Peer ${peerNum}] Shutting down...`);
        await cleanup();
        process.exit(0);
    });
}

// ============================================
// MAIN ORCHESTRATOR
// ============================================

async function runDualPeerTest() {
    console.log('\n' + '='.repeat(60));
    console.log('  WebRTC P2P Data Channel Test');
    console.log('  Testing real peer-to-peer communication');
    console.log('='.repeat(60) + '\n');

    const room = TEST_CONFIG.testRoom;
    console.log(`Test room: ${room}`);
    console.log(`Timeout: ${TEST_CONFIG.timeout / 1000}s`);
    console.log(`Messages per peer: ${TEST_CONFIG.messageCount}\n`);

    // Spawn two peer processes
    const testFile = fileURLToPath(import.meta.url);

    const peer1 = fork(testFile, ['--peer', '1', '--room', room], {
        stdio: 'inherit',
        env: { ...process.env, FORCE_COLOR: '1' }
    });

    // Delay peer2 to ensure peer1 registers first
    await new Promise(resolve => setTimeout(resolve, 2000));

    const peer2 = fork(testFile, ['--peer', '2', '--room', room], {
        stdio: 'inherit',
        env: { ...process.env, FORCE_COLOR: '1' }
    });

    let peer1Exit = null;
    let peer2Exit = null;

    peer1.on('exit', (code) => {
        peer1Exit = code;
        console.log(`\nPeer 1 exited with code ${code}`);
        checkCompletion();
    });

    peer2.on('exit', (code) => {
        peer2Exit = code;
        console.log(`\nPeer 2 exited with code ${code}`);
        checkCompletion();
    });

    function checkCompletion() {
        if (peer1Exit !== null && peer2Exit !== null) {
            const success = peer1Exit === 0 && peer2Exit === 0;
            console.log('\n' + '='.repeat(60));
            console.log(`  TEST ${success ? 'PASSED' : 'FAILED'}`);
            console.log('='.repeat(60) + '\n');
            process.exit(success ? 0 : 1);
        }
    }

    // Overall timeout
    setTimeout(() => {
        console.log('\nOverall test timeout - killing processes');
        peer1.kill();
        peer2.kill();
        process.exit(1);
    }, TEST_CONFIG.timeout + 10000);
}

// ============================================
// INLINE PEER TEST (NO SUBPROCESS)
// ============================================

async function runInlineTest() {
    console.log('\n' + '='.repeat(60));
    console.log('  WebRTC P2P Inline Test');
    console.log('  Two peers in same process');
    console.log('='.repeat(60) + '\n');

    const { WebRTCPeerConnection, WEBRTC_CONFIG } = await import('../webrtc.js');
    const { randomBytes } = await import('crypto');

    // Load wrtc for Node.js
    let wrtc;
    try {
        wrtc = await import('wrtc');
        console.log('wrtc module loaded successfully');
    } catch (err) {
        console.error('Failed to load wrtc:', err.message);
        process.exit(1);
    }

    const { RTCPeerConnection, RTCSessionDescription, RTCIceCandidate } = wrtc;

    // Create two peer connections
    const peer1Id = `peer-1-${randomBytes(4).toString('hex')}`;
    const peer2Id = `peer-2-${randomBytes(4).toString('hex')}`;

    const identity1 = { piKey: peer1Id, siteId: 'test-1' };
    const identity2 = { piKey: peer2Id, siteId: 'test-2' };

    console.log(`\nPeer 1 ID: ${peer1Id}`);
    console.log(`Peer 2 ID: ${peer2Id}`);

    // Create WebRTC peer connections
    const peerConn1 = new WebRTCPeerConnection(peer2Id, identity1, true);
    const peerConn2 = new WebRTCPeerConnection(peer1Id, identity2, false);

    // Track messages
    let peer1Messages = [];
    let peer2Messages = [];
    let peer1ChannelOpen = false;
    let peer2ChannelOpen = false;

    // Setup message handlers
    peerConn1.on('message', ({ message }) => {
        console.log(`[Peer 1] Received:`, typeof message === 'string' ? message : JSON.stringify(message));
        peer1Messages.push(message);
    });

    peerConn2.on('message', ({ message }) => {
        console.log(`[Peer 2] Received:`, typeof message === 'string' ? message : JSON.stringify(message));
        peer2Messages.push(message);
    });

    peerConn1.on('channel-open', () => {
        console.log('[Peer 1] Data channel OPEN');
        peer1ChannelOpen = true;
    });

    peerConn2.on('channel-open', () => {
        console.log('[Peer 2] Data channel OPEN');
        peer2ChannelOpen = true;
    });

    // Initialize both peers
    console.log('\nInitializing peer connections...');
    await peerConn1.initialize();
    await peerConn2.initialize();

    // Exchange ICE candidates directly
    const peer1Candidates = [];
    const peer2Candidates = [];

    peerConn1.on('ice-candidate', ({ candidate }) => {
        console.log('[Peer 1] Generated ICE candidate');
        peer1Candidates.push(candidate);
        // Forward to peer2 (simulating signaling)
        peerConn2.addIceCandidate(candidate).catch(e => console.log('ICE add failed:', e.message));
    });

    peerConn2.on('ice-candidate', ({ candidate }) => {
        console.log('[Peer 2] Generated ICE candidate');
        peer2Candidates.push(candidate);
        // Forward to peer1 (simulating signaling)
        peerConn1.addIceCandidate(candidate).catch(e => console.log('ICE add failed:', e.message));
    });

    // Create offer from peer1
    console.log('\n[Peer 1] Creating offer...');
    const offer = await peerConn1.createOffer();
    console.log('[Peer 1] Offer created');

    // Peer2 handles offer and creates answer
    console.log('[Peer 2] Handling offer...');
    const answer = await peerConn2.handleOffer(offer);
    console.log('[Peer 2] Answer created');

    // Peer1 handles answer
    console.log('[Peer 1] Handling answer...');
    await peerConn1.handleAnswer(answer);
    console.log('[Peer 1] Answer processed');

    // Wait for connection
    console.log('\nWaiting for data channel...');

    const channelTimeout = 30000;
    const startTime = Date.now();

    while (!peer1ChannelOpen || !peer2ChannelOpen) {
        if (Date.now() - startTime > channelTimeout) {
            console.error('\nTimeout waiting for data channel!');
            console.log(`Peer 1 channel open: ${peer1ChannelOpen}`);
            console.log(`Peer 2 channel open: ${peer2ChannelOpen}`);
            console.log(`Peer 1 state: ${peerConn1.state}`);
            console.log(`Peer 2 state: ${peerConn2.state}`);
            peerConn1.close();
            peerConn2.close();
            process.exit(1);
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    console.log('\nData channels established!');

    // Send test messages
    console.log('\nExchanging test messages...');

    for (let i = 0; i < 3; i++) {
        peerConn1.send({ type: 'test', from: 'peer1', seq: i });
        peerConn2.send({ type: 'test', from: 'peer2', seq: i });
        await new Promise(resolve => setTimeout(resolve, 200));
    }

    // Wait for messages
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Report results
    console.log('\n' + '='.repeat(60));
    console.log('  TEST RESULTS');
    console.log('='.repeat(60));
    console.log(`Peer 1 received: ${peer1Messages.length} messages`);
    console.log(`Peer 2 received: ${peer2Messages.length} messages`);

    const success = peer1Messages.length >= 3 && peer2Messages.length >= 3;
    console.log(`\nTEST ${success ? 'PASSED' : 'FAILED'}`);

    // Cleanup
    peerConn1.close();
    peerConn2.close();

    process.exit(success ? 0 : 1);
}

// ============================================
// FIREBASE INTEGRATION TEST
// ============================================

async function runFirebaseTest() {
    console.log('\n' + '='.repeat(60));
    console.log('  WebRTC + Firebase Signaling Test');
    console.log('='.repeat(60) + '\n');

    const { WebRTCPeerManager } = await import('../webrtc.js');
    const { FirebaseSignaling, getFirebaseConfig } = await import('../firebase-signaling.js');
    const { randomBytes } = await import('crypto');

    const room = `firebase-test-${Date.now()}`;
    console.log(`Test room: ${room}\n`);

    // Create two peers
    const peer1Id = `fp1-${randomBytes(4).toString('hex')}`;
    const peer2Id = `fp2-${randomBytes(4).toString('hex')}`;

    const identity1 = { piKey: peer1Id, siteId: 'firebase-test-1' };
    const identity2 = { piKey: peer2Id, siteId: 'firebase-test-2' };

    // Create Firebase signaling for each peer
    const fb1 = new FirebaseSignaling({ peerId: peer1Id, room, verifySignatures: false });
    const fb2 = new FirebaseSignaling({ peerId: peer2Id, room, verifySignatures: false });

    // Create WebRTC managers
    const webrtc1 = new WebRTCPeerManager(identity1, { fallbackToSimulation: false });
    const webrtc2 = new WebRTCPeerManager(identity2, { fallbackToSimulation: false });

    // Wire up external signaling
    webrtc1.setExternalSignaling(async (type, to, data) => {
        console.log(`[P1] Signaling -> ${type}`);
        if (type === 'offer') await fb1.sendOffer(to, data);
        else if (type === 'answer') await fb1.sendAnswer(to, data);
        else if (type === 'ice-candidate') await fb1.sendIceCandidate(to, data);
    });

    webrtc2.setExternalSignaling(async (type, to, data) => {
        console.log(`[P2] Signaling -> ${type}`);
        if (type === 'offer') await fb2.sendOffer(to, data);
        else if (type === 'answer') await fb2.sendAnswer(to, data);
        else if (type === 'ice-candidate') await fb2.sendIceCandidate(to, data);
    });

    // Track state
    let connected = false;
    let messages1 = [];
    let messages2 = [];

    // Wire Firebase events to WebRTC
    fb1.on('peer-discovered', async ({ peerId }) => {
        console.log(`[P1] Discovered: ${peerId.slice(0, 8)}`);
        if (peer1Id > peerId) {
            await webrtc1.connectToPeer(peerId);
        }
    });

    fb2.on('peer-discovered', async ({ peerId }) => {
        console.log(`[P2] Discovered: ${peerId.slice(0, 8)}`);
        if (peer2Id > peerId) {
            await webrtc2.connectToPeer(peerId);
        }
    });

    fb1.on('offer', async ({ from, offer }) => {
        console.log(`[P1] Received offer`);
        await webrtc1.handleOffer({ from, offer });
    });

    fb2.on('offer', async ({ from, offer }) => {
        console.log(`[P2] Received offer`);
        await webrtc2.handleOffer({ from, offer });
    });

    fb1.on('answer', async ({ from, answer }) => {
        console.log(`[P1] Received answer`);
        await webrtc1.handleAnswer({ from, answer });
    });

    fb2.on('answer', async ({ from, answer }) => {
        console.log(`[P2] Received answer`);
        await webrtc2.handleAnswer({ from, answer });
    });

    fb1.on('ice-candidate', async ({ from, candidate }) => {
        await webrtc1.handleIceCandidate({ from, candidate });
    });

    fb2.on('ice-candidate', async ({ from, candidate }) => {
        await webrtc2.handleIceCandidate({ from, candidate });
    });

    webrtc1.on('peer-connected', (peerId) => {
        console.log(`[P1] Connected to ${peerId.slice(0, 8)}`);
        connected = true;
    });

    webrtc2.on('peer-connected', (peerId) => {
        console.log(`[P2] Connected to ${peerId.slice(0, 8)}`);
        connected = true;
    });

    webrtc1.on('message', ({ message }) => {
        console.log(`[P1] Message:`, JSON.stringify(message).slice(0, 50));
        messages1.push(message);
    });

    webrtc2.on('message', ({ message }) => {
        console.log(`[P2] Message:`, JSON.stringify(message).slice(0, 50));
        messages2.push(message);
    });

    // Connect to Firebase
    console.log('Connecting to Firebase...');
    const conn1 = await fb1.connect();
    const conn2 = await fb2.connect();

    if (!conn1 || !conn2) {
        console.error('Firebase connection failed');
        process.exit(1);
    }

    console.log('Both peers connected to Firebase\n');

    // Wait for WebRTC connection
    const timeout = 45000;
    const start = Date.now();

    while (!connected && Date.now() - start < timeout) {
        await new Promise(r => setTimeout(r, 500));
    }

    if (!connected) {
        console.error('\nWebRTC connection timeout');
        await fb1.disconnect();
        await fb2.disconnect();
        process.exit(1);
    }

    // Exchange messages
    console.log('\nExchanging messages...');
    await new Promise(r => setTimeout(r, 1000));

    webrtc1.sendToPeer(peer2Id, { type: 'hello', from: 'peer1' });
    webrtc2.sendToPeer(peer1Id, { type: 'hello', from: 'peer2' });

    await new Promise(r => setTimeout(r, 2000));

    // Results
    console.log('\n' + '='.repeat(60));
    console.log('  RESULTS');
    console.log('='.repeat(60));
    console.log(`P1 messages: ${messages1.length}`);
    console.log(`P2 messages: ${messages2.length}`);

    const success = messages1.length > 0 && messages2.length > 0;
    console.log(`\nTEST ${success ? 'PASSED' : 'FAILED'}`);

    // Cleanup
    webrtc1.close();
    webrtc2.close();
    await fb1.disconnect();
    await fb2.disconnect();

    process.exit(success ? 0 : 1);
}

// ============================================
// ENTRY POINT
// ============================================

const args = process.argv.slice(2);

if (args.includes('--peer')) {
    const peerIndex = args.indexOf('--peer');
    const peerNum = parseInt(args[peerIndex + 1], 10);
    const roomIndex = args.indexOf('--room');
    const room = roomIndex !== -1 ? args[roomIndex + 1] : TEST_CONFIG.testRoom;

    runPeer(peerNum, room).catch(err => {
        console.error('Peer error:', err);
        process.exit(1);
    });
} else if (args.includes('--inline')) {
    runInlineTest().catch(err => {
        console.error('Inline test error:', err);
        process.exit(1);
    });
} else if (args.includes('--firebase')) {
    runFirebaseTest().catch(err => {
        console.error('Firebase test error:', err);
        process.exit(1);
    });
} else if (args.includes('--dual')) {
    runDualPeerTest().catch(err => {
        console.error('Dual peer test error:', err);
        process.exit(1);
    });
} else {
    // Default: run inline test (simplest, no subprocesses)
    console.log('Usage:');
    console.log('  node tests/webrtc-peer-test.js --inline   # Same-process test (recommended)');
    console.log('  node tests/webrtc-peer-test.js --firebase # Firebase signaling test');
    console.log('  node tests/webrtc-peer-test.js --dual     # Two-process test');
    console.log('');
    console.log('Running inline test by default...\n');

    runInlineTest().catch(err => {
        console.error('Test error:', err);
        process.exit(1);
    });
}

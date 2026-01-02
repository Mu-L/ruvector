#!/usr/bin/env node
/**
 * Edge-Net WebRTC P2P Implementation
 *
 * Real peer-to-peer communication using WebRTC data channels.
 * Replaces simulated P2P with actual network connectivity.
 *
 * Features:
 * - WebRTC data channels for P2P messaging
 * - ICE candidate handling with STUN/TURN
 * - WebSocket signaling with fallback
 * - Connection quality monitoring
 * - Automatic reconnection
 * - QDAG synchronization over data channels
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';

// WebRTC Configuration
export const WEBRTC_CONFIG = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun.cloudflare.com:3478' },
        { urls: 'stun:stun.services.mozilla.com:3478' },
    ],
    // Signaling server endpoints
    signalingServers: [
        'wss://edge-net-signal.ruvector.dev',
        'wss://signal.edge-net.io',
    ],
    // Fallback to local simulation if no signaling available
    fallbackToSimulation: true,
    // Connection timeouts
    connectionTimeout: 30000,
    reconnectDelay: 5000,
    maxReconnectAttempts: 5,
    // Data channel options
    dataChannelOptions: {
        ordered: true,
        maxRetransmits: 3,
    },
    // Heartbeat for connection health
    heartbeatInterval: 5000,
    heartbeatTimeout: 15000,
};

/**
 * WebRTC Peer Connection Manager
 *
 * Manages individual peer connections with ICE handling,
 * data channels, and connection lifecycle.
 */
export class WebRTCPeerConnection extends EventEmitter {
    constructor(peerId, localIdentity, isInitiator = false) {
        super();
        this.peerId = peerId;
        this.localIdentity = localIdentity;
        this.isInitiator = isInitiator;
        this.pc = null;
        this.dataChannel = null;
        this.state = 'new';
        this.iceCandidates = [];
        this.pendingCandidates = [];
        this.lastHeartbeat = Date.now();
        this.reconnectAttempts = 0;
        this.metrics = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesTransferred: 0,
            latency: [],
            connectionTime: null,
        };
    }

    /**
     * Initialize the RTCPeerConnection
     */
    async initialize() {
        // Use wrtc for Node.js or native WebRTC in browser
        const RTCPeerConnection = globalThis.RTCPeerConnection ||
            (await this.loadNodeWebRTC());

        if (!RTCPeerConnection) {
            throw new Error('WebRTC not available');
        }

        this.pc = new RTCPeerConnection({
            iceServers: WEBRTC_CONFIG.iceServers,
        });

        this.setupEventHandlers();

        if (this.isInitiator) {
            await this.createDataChannel();
        }

        return this;
    }

    /**
     * Load wrtc for Node.js environment
     */
    async loadNodeWebRTC() {
        try {
            const wrtc = await import('wrtc');
            return wrtc.RTCPeerConnection;
        } catch (err) {
            // wrtc not available, will use simulation
            console.warn('WebRTC not available in Node.js, using simulation');
            return null;
        }
    }

    /**
     * Setup RTCPeerConnection event handlers
     */
    setupEventHandlers() {
        // ICE candidate events
        this.pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.iceCandidates.push(event.candidate);
                this.emit('ice-candidate', {
                    peerId: this.peerId,
                    candidate: event.candidate,
                });
            }
        };

        this.pc.onicegatheringstatechange = () => {
            this.emit('ice-gathering-state', this.pc.iceGatheringState);
        };

        this.pc.oniceconnectionstatechange = () => {
            const state = this.pc.iceConnectionState;
            this.state = state;
            this.emit('connection-state', state);

            if (state === 'connected') {
                this.metrics.connectionTime = Date.now();
                this.startHeartbeat();
            } else if (state === 'disconnected' || state === 'failed') {
                this.handleDisconnection();
            }
        };

        // Data channel events (for non-initiator)
        this.pc.ondatachannel = (event) => {
            this.dataChannel = event.channel;
            this.setupDataChannel();
        };
    }

    /**
     * Create data channel (initiator only)
     */
    async createDataChannel() {
        this.dataChannel = this.pc.createDataChannel(
            'edge-net',
            WEBRTC_CONFIG.dataChannelOptions
        );
        this.setupDataChannel();
    }

    /**
     * Setup data channel event handlers
     */
    setupDataChannel() {
        if (!this.dataChannel) return;

        this.dataChannel.onopen = () => {
            this.emit('channel-open', this.peerId);
            console.log(`  📡 Data channel open with ${this.peerId.slice(0, 8)}...`);
        };

        this.dataChannel.onclose = () => {
            this.emit('channel-close', this.peerId);
        };

        this.dataChannel.onerror = (error) => {
            this.emit('channel-error', { peerId: this.peerId, error });
        };

        this.dataChannel.onmessage = (event) => {
            this.metrics.messagesReceived++;
            this.metrics.bytesTransferred += event.data.length;
            this.handleMessage(event.data);
        };
    }

    /**
     * Create and return an offer
     */
    async createOffer() {
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);
        return offer;
    }

    /**
     * Handle incoming offer and create answer
     */
    async handleOffer(offer) {
        await this.pc.setRemoteDescription(new RTCSessionDescription(offer));

        // Process any pending ICE candidates
        for (const candidate of this.pendingCandidates) {
            await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
        }
        this.pendingCandidates = [];

        const answer = await this.pc.createAnswer();
        await this.pc.setLocalDescription(answer);
        return answer;
    }

    /**
     * Handle incoming answer
     */
    async handleAnswer(answer) {
        await this.pc.setRemoteDescription(new RTCSessionDescription(answer));

        // Process any pending ICE candidates
        for (const candidate of this.pendingCandidates) {
            await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
        }
        this.pendingCandidates = [];
    }

    /**
     * Add ICE candidate
     */
    async addIceCandidate(candidate) {
        if (this.pc.remoteDescription) {
            await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
        } else {
            // Queue for later
            this.pendingCandidates.push(candidate);
        }
    }

    /**
     * Send message over data channel
     */
    send(data) {
        if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
            throw new Error('Data channel not ready');
        }

        const message = typeof data === 'string' ? data : JSON.stringify(data);
        this.dataChannel.send(message);
        this.metrics.messagesSent++;
        this.metrics.bytesTransferred += message.length;
    }

    /**
     * Handle incoming message
     */
    handleMessage(data) {
        try {
            const message = JSON.parse(data);

            // Handle heartbeat
            if (message.type === 'heartbeat') {
                this.lastHeartbeat = Date.now();
                this.send({ type: 'heartbeat-ack', timestamp: message.timestamp });
                return;
            }

            if (message.type === 'heartbeat-ack') {
                const latency = Date.now() - message.timestamp;
                this.metrics.latency.push(latency);
                if (this.metrics.latency.length > 100) {
                    this.metrics.latency.shift();
                }
                return;
            }

            this.emit('message', { peerId: this.peerId, message });
        } catch (err) {
            // Raw string message
            this.emit('message', { peerId: this.peerId, message: data });
        }
    }

    /**
     * Start heartbeat monitoring
     */
    startHeartbeat() {
        this.heartbeatTimer = setInterval(() => {
            if (this.dataChannel?.readyState === 'open') {
                this.send({ type: 'heartbeat', timestamp: Date.now() });
            }

            // Check for timeout
            if (Date.now() - this.lastHeartbeat > WEBRTC_CONFIG.heartbeatTimeout) {
                this.handleDisconnection();
            }
        }, WEBRTC_CONFIG.heartbeatInterval);
    }

    /**
     * Handle disconnection with reconnection logic
     */
    handleDisconnection() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }

        if (this.reconnectAttempts < WEBRTC_CONFIG.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.emit('reconnecting', {
                peerId: this.peerId,
                attempt: this.reconnectAttempts,
            });

            setTimeout(() => {
                this.emit('reconnect', this.peerId);
            }, WEBRTC_CONFIG.reconnectDelay * this.reconnectAttempts);
        } else {
            this.emit('disconnected', this.peerId);
        }
    }

    /**
     * Get connection metrics
     */
    getMetrics() {
        const avgLatency = this.metrics.latency.length > 0
            ? this.metrics.latency.reduce((a, b) => a + b, 0) / this.metrics.latency.length
            : 0;

        return {
            ...this.metrics,
            averageLatency: avgLatency,
            state: this.state,
            dataChannelState: this.dataChannel?.readyState || 'closed',
        };
    }

    /**
     * Close the connection
     */
    close() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }
        if (this.dataChannel) {
            this.dataChannel.close();
        }
        if (this.pc) {
            this.pc.close();
        }
        this.state = 'closed';
    }
}

/**
 * WebRTC Peer Manager
 *
 * Manages multiple peer connections, signaling, and network topology.
 */
export class WebRTCPeerManager extends EventEmitter {
    constructor(localIdentity, options = {}) {
        super();
        this.localIdentity = localIdentity;
        this.options = { ...WEBRTC_CONFIG, ...options };
        this.peers = new Map();
        this.signalingSocket = null;
        this.isConnected = false;
        this.mode = 'initializing'; // 'webrtc', 'simulation', 'hybrid'
        this.stats = {
            totalConnections: 0,
            successfulConnections: 0,
            failedConnections: 0,
            messagesRouted: 0,
        };
    }

    /**
     * Initialize the peer manager and connect to signaling
     */
    async initialize() {
        console.log('\n🌐 Initializing WebRTC P2P Network...');

        // Try to connect to signaling server
        const signalingConnected = await this.connectToSignaling();

        if (signalingConnected) {
            this.mode = 'webrtc';
            console.log('  ✅ WebRTC mode active - real P2P enabled');
        } else if (this.options.fallbackToSimulation) {
            this.mode = 'simulation';
            console.log('  ⚠️  Simulation mode - signaling unavailable');
        } else {
            throw new Error('Could not connect to signaling server');
        }

        // Announce our presence
        await this.announce();

        return this;
    }

    /**
     * Connect to WebSocket signaling server
     */
    async connectToSignaling() {
        // Check if WebSocket is available
        const WebSocket = globalThis.WebSocket ||
            (await this.loadNodeWebSocket());

        if (!WebSocket) {
            console.log('  ⚠️  WebSocket not available');
            return false;
        }

        for (const serverUrl of this.options.signalingServers) {
            try {
                const connected = await this.trySignalingServer(WebSocket, serverUrl);
                if (connected) return true;
            } catch (err) {
                console.log(`  ⚠️  Signaling server ${serverUrl} unavailable`);
            }
        }

        return false;
    }

    /**
     * Load ws for Node.js environment
     */
    async loadNodeWebSocket() {
        try {
            const ws = await import('ws');
            return ws.default || ws.WebSocket;
        } catch (err) {
            return null;
        }
    }

    /**
     * Try connecting to a specific signaling server
     */
    async trySignalingServer(WebSocket, serverUrl) {
        return new Promise((resolve) => {
            const timeout = setTimeout(() => {
                resolve(false);
            }, 5000);

            try {
                this.signalingSocket = new WebSocket(serverUrl);

                this.signalingSocket.onopen = () => {
                    clearTimeout(timeout);
                    console.log(`  📡 Connected to signaling: ${serverUrl}`);
                    this.setupSignalingHandlers();
                    this.isConnected = true;
                    resolve(true);
                };

                this.signalingSocket.onerror = () => {
                    clearTimeout(timeout);
                    resolve(false);
                };
            } catch (err) {
                clearTimeout(timeout);
                resolve(false);
            }
        });
    }

    /**
     * Setup signaling socket event handlers
     */
    setupSignalingHandlers() {
        this.signalingSocket.onmessage = async (event) => {
            try {
                const message = JSON.parse(event.data);
                await this.handleSignalingMessage(message);
            } catch (err) {
                console.error('Signaling message error:', err);
            }
        };

        this.signalingSocket.onclose = () => {
            this.isConnected = false;
            this.emit('signaling-disconnected');

            // Attempt reconnection
            setTimeout(() => this.connectToSignaling(), 5000);
        };
    }

    /**
     * Handle incoming signaling messages
     */
    async handleSignalingMessage(message) {
        switch (message.type) {
            case 'peer-joined':
                await this.handlePeerJoined(message);
                break;

            case 'offer':
                await this.handleOffer(message);
                break;

            case 'answer':
                await this.handleAnswer(message);
                break;

            case 'ice-candidate':
                await this.handleIceCandidate(message);
                break;

            case 'peer-list':
                await this.handlePeerList(message.peers);
                break;

            case 'peer-left':
                this.handlePeerLeft(message.peerId);
                break;
        }
    }

    /**
     * Announce presence to signaling server
     */
    async announce() {
        if (this.mode === 'simulation') {
            // Simulate some peers
            this.simulatePeers();
            return;
        }

        if (this.signalingSocket?.readyState === 1) {
            this.signalingSocket.send(JSON.stringify({
                type: 'announce',
                piKey: this.localIdentity.piKey,
                publicKey: this.localIdentity.publicKey,
                siteId: this.localIdentity.siteId,
                capabilities: ['compute', 'storage', 'verify'],
            }));
        }
    }

    /**
     * Simulate peers for offline/testing mode
     */
    simulatePeers() {
        const simulatedPeers = [
            { piKey: 'sim-peer-1-' + randomBytes(8).toString('hex'), siteId: 'sim-node-1' },
            { piKey: 'sim-peer-2-' + randomBytes(8).toString('hex'), siteId: 'sim-node-2' },
            { piKey: 'sim-peer-3-' + randomBytes(8).toString('hex'), siteId: 'sim-node-3' },
        ];

        for (const peer of simulatedPeers) {
            this.peers.set(peer.piKey, {
                piKey: peer.piKey,
                siteId: peer.siteId,
                state: 'simulated',
                lastSeen: Date.now(),
            });
        }

        console.log(`  📡 Simulated ${simulatedPeers.length} peers`);
        this.emit('peers-updated', this.getPeerList());
    }

    /**
     * Handle new peer joining
     */
    async handlePeerJoined(message) {
        const { peerId, publicKey, siteId } = message;

        // Don't connect to ourselves
        if (peerId === this.localIdentity.piKey) return;

        console.log(`  🔗 New peer: ${siteId} (${peerId.slice(0, 8)}...)`);

        // Initiate connection if we have higher ID (simple tiebreaker)
        if (this.localIdentity.piKey > peerId) {
            await this.connectToPeer(peerId);
        }

        this.emit('peer-joined', { peerId, siteId });
    }

    /**
     * Initiate connection to a peer
     */
    async connectToPeer(peerId) {
        if (this.peers.has(peerId)) return;

        this.stats.totalConnections++;

        try {
            const peerConnection = new WebRTCPeerConnection(
                peerId,
                this.localIdentity,
                true // initiator
            );

            await peerConnection.initialize();
            this.setupPeerHandlers(peerConnection);

            const offer = await peerConnection.createOffer();

            // Send offer via signaling
            this.signalingSocket.send(JSON.stringify({
                type: 'offer',
                to: peerId,
                from: this.localIdentity.piKey,
                offer,
            }));

            this.peers.set(peerId, peerConnection);
            this.emit('peers-updated', this.getPeerList());

        } catch (err) {
            this.stats.failedConnections++;
            console.error(`Failed to connect to ${peerId}:`, err.message);
        }
    }

    /**
     * Handle incoming offer
     */
    async handleOffer(message) {
        const { from, offer } = message;

        if (this.peers.has(from)) return;

        this.stats.totalConnections++;

        try {
            const peerConnection = new WebRTCPeerConnection(
                from,
                this.localIdentity,
                false // not initiator
            );

            await peerConnection.initialize();
            this.setupPeerHandlers(peerConnection);

            const answer = await peerConnection.handleOffer(offer);

            // Send answer via signaling
            this.signalingSocket.send(JSON.stringify({
                type: 'answer',
                to: from,
                from: this.localIdentity.piKey,
                answer,
            }));

            this.peers.set(from, peerConnection);
            this.emit('peers-updated', this.getPeerList());

        } catch (err) {
            this.stats.failedConnections++;
            console.error(`Failed to handle offer from ${from}:`, err.message);
        }
    }

    /**
     * Handle incoming answer
     */
    async handleAnswer(message) {
        const { from, answer } = message;
        const peerConnection = this.peers.get(from);

        if (peerConnection) {
            await peerConnection.handleAnswer(answer);
            this.stats.successfulConnections++;
        }
    }

    /**
     * Handle ICE candidate
     */
    async handleIceCandidate(message) {
        const { from, candidate } = message;
        const peerConnection = this.peers.get(from);

        if (peerConnection) {
            await peerConnection.addIceCandidate(candidate);
        }
    }

    /**
     * Handle peer list from server
     */
    async handlePeerList(peers) {
        for (const peer of peers) {
            if (peer.piKey !== this.localIdentity.piKey && !this.peers.has(peer.piKey)) {
                await this.connectToPeer(peer.piKey);
            }
        }
    }

    /**
     * Handle peer leaving
     */
    handlePeerLeft(peerId) {
        const peer = this.peers.get(peerId);
        if (peer) {
            if (peer.close) peer.close();
            this.peers.delete(peerId);
            this.emit('peer-left', peerId);
            this.emit('peers-updated', this.getPeerList());
        }
    }

    /**
     * Setup event handlers for a peer connection
     */
    setupPeerHandlers(peerConnection) {
        peerConnection.on('ice-candidate', ({ candidate }) => {
            if (this.signalingSocket?.readyState === 1) {
                this.signalingSocket.send(JSON.stringify({
                    type: 'ice-candidate',
                    to: peerConnection.peerId,
                    from: this.localIdentity.piKey,
                    candidate,
                }));
            }
        });

        peerConnection.on('channel-open', () => {
            this.stats.successfulConnections++;
            this.emit('peer-connected', peerConnection.peerId);
        });

        peerConnection.on('message', ({ message }) => {
            this.stats.messagesRouted++;
            this.emit('message', {
                from: peerConnection.peerId,
                message,
            });
        });

        peerConnection.on('disconnected', () => {
            this.peers.delete(peerConnection.peerId);
            this.emit('peer-disconnected', peerConnection.peerId);
            this.emit('peers-updated', this.getPeerList());
        });

        peerConnection.on('reconnect', async (peerId) => {
            this.peers.delete(peerId);
            await this.connectToPeer(peerId);
        });
    }

    /**
     * Send message to a specific peer
     */
    sendToPeer(peerId, message) {
        const peer = this.peers.get(peerId);
        if (peer && peer.send) {
            peer.send(message);
            return true;
        }
        return false;
    }

    /**
     * Broadcast message to all peers
     */
    broadcast(message) {
        let sent = 0;
        for (const [peerId, peer] of this.peers) {
            try {
                if (peer.send) {
                    peer.send(message);
                    sent++;
                }
            } catch (err) {
                // Peer not ready
            }
        }
        return sent;
    }

    /**
     * Get list of connected peers
     */
    getPeerList() {
        const peers = [];
        for (const [peerId, peer] of this.peers) {
            peers.push({
                peerId,
                state: peer.state || 'simulated',
                siteId: peer.siteId,
                lastSeen: peer.lastSeen || Date.now(),
                metrics: peer.getMetrics ? peer.getMetrics() : null,
            });
        }
        return peers;
    }

    /**
     * Get connection statistics
     */
    getStats() {
        return {
            ...this.stats,
            mode: this.mode,
            connectedPeers: this.peers.size,
            signalingConnected: this.isConnected,
        };
    }

    /**
     * Close all connections
     */
    close() {
        for (const [, peer] of this.peers) {
            if (peer.close) peer.close();
        }
        this.peers.clear();

        if (this.signalingSocket) {
            this.signalingSocket.close();
        }
    }
}

/**
 * QDAG Synchronizer
 *
 * Synchronizes QDAG contributions over WebRTC data channels.
 */
export class QDAGSynchronizer extends EventEmitter {
    constructor(peerManager, qdag) {
        super();
        this.peerManager = peerManager;
        this.qdag = qdag;
        this.syncState = new Map(); // Track sync state per peer
        this.pendingSync = new Set();
    }

    /**
     * Initialize synchronization
     */
    initialize() {
        // Listen for new peer connections
        this.peerManager.on('peer-connected', (peerId) => {
            this.requestSync(peerId);
        });

        // Listen for sync messages
        this.peerManager.on('message', ({ from, message }) => {
            this.handleSyncMessage(from, message);
        });

        // Periodic sync
        setInterval(() => this.syncWithPeers(), 10000);
    }

    /**
     * Request QDAG sync from a peer
     */
    requestSync(peerId) {
        const lastSync = this.syncState.get(peerId) || 0;

        this.peerManager.sendToPeer(peerId, {
            type: 'qdag_sync_request',
            since: lastSync,
            myTip: this.qdag?.getLatestHash() || null,
        });

        this.pendingSync.add(peerId);
    }

    /**
     * Handle incoming sync messages
     */
    handleSyncMessage(from, message) {
        if (message.type === 'qdag_sync_request') {
            this.handleSyncRequest(from, message);
        } else if (message.type === 'qdag_sync_response') {
            this.handleSyncResponse(from, message);
        } else if (message.type === 'qdag_contribution') {
            this.handleNewContribution(from, message);
        }
    }

    /**
     * Handle sync request from peer
     */
    handleSyncRequest(from, message) {
        const contributions = this.qdag?.getContributionsSince(message.since) || [];

        this.peerManager.sendToPeer(from, {
            type: 'qdag_sync_response',
            contributions,
            tip: this.qdag?.getLatestHash() || null,
        });
    }

    /**
     * Handle sync response from peer
     */
    handleSyncResponse(from, message) {
        this.pendingSync.delete(from);
        this.syncState.set(from, Date.now());

        if (message.contributions && message.contributions.length > 0) {
            let added = 0;
            for (const contrib of message.contributions) {
                if (this.qdag?.addContribution(contrib)) {
                    added++;
                }
            }

            if (added > 0) {
                this.emit('synced', { from, added });
            }
        }
    }

    /**
     * Handle new contribution broadcast
     */
    handleNewContribution(from, message) {
        if (this.qdag?.addContribution(message.contribution)) {
            this.emit('contribution-received', {
                from,
                contribution: message.contribution,
            });
        }
    }

    /**
     * Broadcast a new contribution to all peers
     */
    broadcastContribution(contribution) {
        this.peerManager.broadcast({
            type: 'qdag_contribution',
            contribution,
        });
    }

    /**
     * Sync with all connected peers
     */
    syncWithPeers() {
        const peers = this.peerManager.getPeerList();
        for (const peer of peers) {
            if (!this.pendingSync.has(peer.peerId)) {
                this.requestSync(peer.peerId);
            }
        }
    }
}

// Export default configuration for testing
export default {
    WebRTCPeerConnection,
    WebRTCPeerManager,
    QDAGSynchronizer,
    WEBRTC_CONFIG,
};

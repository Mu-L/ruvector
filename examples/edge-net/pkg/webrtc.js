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
 * - Configurable TURN/STUN via environment variables
 * - ICE connection diagnostics
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';

/**
 * Environment variable configuration for ICE servers
 *
 * Environment Variables:
 * - EDGE_NET_STUN_SERVERS: Comma-separated STUN server URLs
 * - EDGE_NET_TURN_URL: TURN server URL
 * - EDGE_NET_TURN_USERNAME: TURN username
 * - EDGE_NET_TURN_CREDENTIAL: TURN password/credential
 * - EDGE_NET_TURN_URL_TCP: TURN server URL for TCP transport
 * - EDGE_NET_ICE_TRANSPORT_POLICY: 'all' or 'relay' (force TURN)
 * - EDGE_NET_SIGNALING_SERVERS: Comma-separated signaling server URLs
 */

// Get environment safely (works in Node.js, browser with bundler support)
const getEnv = (key, defaultValue = '') => {
    if (typeof process !== 'undefined' && process.env) {
        return process.env[key] || defaultValue;
    }
    if (typeof globalThis !== 'undefined' && globalThis.__EDGE_NET_ENV__) {
        return globalThis.__EDGE_NET_ENV__[key] || defaultValue;
    }
    return defaultValue;
};

/**
 * Parse STUN servers from environment or use defaults
 */
function getStunServers() {
    const envStun = getEnv('EDGE_NET_STUN_SERVERS');
    if (envStun) {
        return envStun.split(',').map(url => ({ urls: url.trim() }));
    }
    // Default STUN servers (free, reliable)
    return [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun.cloudflare.com:3478' },
        { urls: 'stun:stun.services.mozilla.com:3478' },
        { urls: 'stun:stun.stunprotocol.org:3478' },
    ];
}

/**
 * Parse TURN servers from environment or use defaults
 */
function getTurnServers() {
    const turnUrl = getEnv('EDGE_NET_TURN_URL');
    const turnUsername = getEnv('EDGE_NET_TURN_USERNAME');
    const turnCredential = getEnv('EDGE_NET_TURN_CREDENTIAL');
    const turnUrlTcp = getEnv('EDGE_NET_TURN_URL_TCP');

    const servers = [];

    // Custom TURN from environment
    if (turnUrl && turnUsername && turnCredential) {
        servers.push({
            urls: turnUrl,
            username: turnUsername,
            credential: turnCredential,
        });

        // Add TCP transport variant if specified
        if (turnUrlTcp) {
            servers.push({
                urls: turnUrlTcp,
                username: turnUsername,
                credential: turnCredential,
            });
        }
    }

    // Add default free TURN servers if no custom ones
    if (servers.length === 0) {
        servers.push(
            // Metered.ca free tier (limited bandwidth, good for testing)
            {
                urls: 'turn:openrelay.metered.ca:80',
                username: 'openrelayproject',
                credential: 'openrelayproject',
            },
            {
                urls: 'turn:openrelay.metered.ca:443',
                username: 'openrelayproject',
                credential: 'openrelayproject',
            },
            {
                urls: 'turn:openrelay.metered.ca:443?transport=tcp',
                username: 'openrelayproject',
                credential: 'openrelayproject',
            },
            // Alternative relay
            {
                urls: 'turn:relay.metered.ca:80',
                username: 'e8a437a4c4d4e5f6a7b8c9d0',
                credential: 'freePublicTurnServer',
            }
        );
    }

    return servers;
}

/**
 * Get signaling servers from environment or use defaults
 */
function getSignalingServers() {
    const envSignaling = getEnv('EDGE_NET_SIGNALING_SERVERS');
    if (envSignaling) {
        return envSignaling.split(',').map(url => url.trim());
    }
    return [
        'ws://localhost:8787',                    // Local signaling server first
        'ws://127.0.0.1:8787',                    // Local alternative
        'wss://edge-net-signal.ruvector.dev',    // Production (when deployed)
    ];
}

/**
 * Build complete ICE configuration
 */
function buildIceConfig() {
    const iceTransportPolicy = getEnv('EDGE_NET_ICE_TRANSPORT_POLICY', 'all');

    return {
        iceServers: [...getStunServers(), ...getTurnServers()],
        iceTransportPolicy, // 'all' = STUN+TURN, 'relay' = TURN only
        iceCandidatePoolSize: 10, // Pre-gather candidates for faster connection
        bundlePolicy: 'max-bundle',
        rtcpMuxPolicy: 'require',
    };
}

// WebRTC Configuration
export const WEBRTC_CONFIG = {
    // Dynamic ICE configuration
    ...buildIceConfig(),
    // Signaling server endpoints (priority order)
    signalingServers: getSignalingServers(),
    // Fallback to local DHT if no signaling available
    fallbackToSimulation: false,
    fallbackToDHT: true,
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
    // DHT configuration
    dht: {
        bootstrapNodes: [],
        kBucketSize: 20,
        alpha: 3, // Parallel lookups
        refreshInterval: 60000,
    },
};

/**
 * ICE Server Providers Configuration
 *
 * Popular TURN/STUN providers with their configuration patterns.
 * Use these as reference for configuring your production environment.
 */
export const ICE_PROVIDERS = {
    // Self-hosted coturn (recommended for production)
    coturn: {
        name: 'Self-hosted coturn',
        config: (host, username, credential) => ([
            { urls: `stun:${host}:3478` },
            { urls: `turn:${host}:3478`, username, credential },
            { urls: `turn:${host}:3478?transport=tcp`, username, credential },
            { urls: `turns:${host}:5349`, username, credential }, // TLS
        ]),
        docs: 'https://github.com/coturn/coturn',
    },

    // Metered.ca (free tier available)
    metered: {
        name: 'Metered TURN',
        config: (apiKey) => ([
            { urls: 'stun:stun.relay.metered.ca:80' },
            {
                urls: 'turn:a.relay.metered.ca:80',
                username: apiKey,
                credential: apiKey,
            },
            {
                urls: 'turn:a.relay.metered.ca:443',
                username: apiKey,
                credential: apiKey,
            },
            {
                urls: 'turn:a.relay.metered.ca:443?transport=tcp',
                username: apiKey,
                credential: apiKey,
            },
        ]),
        pricing: 'Free: 500MB/month, Paid: from $0.40/GB',
        docs: 'https://www.metered.ca/stun-turn',
    },

    // Twilio (pay-as-you-go)
    twilio: {
        name: 'Twilio Network Traversal',
        // Note: Twilio requires fetching ephemeral credentials via API
        config: (username, credential) => ([
            { urls: 'stun:global.stun.twilio.com:3478' },
            {
                urls: 'turn:global.turn.twilio.com:3478?transport=udp',
                username,
                credential,
            },
            {
                urls: 'turn:global.turn.twilio.com:3478?transport=tcp',
                username,
                credential,
            },
            {
                urls: 'turn:global.turn.twilio.com:443?transport=tcp',
                username,
                credential,
            },
        ]),
        pricing: '$0.40/GB',
        docs: 'https://www.twilio.com/docs/stun-turn',
    },

    // Xirsys (global coverage)
    xirsys: {
        name: 'Xirsys',
        config: (username, credential, domain) => ([
            { urls: `stun:${domain}:3478` },
            { urls: `turn:${domain}:80?transport=udp`, username, credential },
            { urls: `turn:${domain}:3478?transport=udp`, username, credential },
            { urls: `turn:${domain}:80?transport=tcp`, username, credential },
            { urls: `turn:${domain}:3478?transport=tcp`, username, credential },
        ]),
        pricing: 'Free: 500MB/month, Paid: from $24.99/month',
        docs: 'https://xirsys.com/',
    },
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
        // WebRTC classes (set during initialize for Node.js compatibility)
        this._RTCSessionDescription = null;
        this._RTCIceCandidate = null;
    }

    /**
     * Initialize the RTCPeerConnection
     */
    async initialize() {
        // Use wrtc for Node.js or native WebRTC in browser
        const webrtcClasses = await this.loadWebRTCClasses();

        if (!webrtcClasses) {
            throw new Error('WebRTC not available');
        }

        const { RTCPeerConnection, RTCSessionDescription, RTCIceCandidate } = webrtcClasses;

        // Store classes for later use in handleOffer/handleAnswer/addIceCandidate
        this._RTCSessionDescription = RTCSessionDescription;
        this._RTCIceCandidate = RTCIceCandidate;

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
     * Load WebRTC classes (from browser globals or wrtc package)
     */
    async loadWebRTCClasses() {
        // Check for browser globals first
        if (globalThis.RTCPeerConnection) {
            return {
                RTCPeerConnection: globalThis.RTCPeerConnection,
                RTCSessionDescription: globalThis.RTCSessionDescription,
                RTCIceCandidate: globalThis.RTCIceCandidate,
            };
        }

        // Try to load wrtc for Node.js
        try {
            const wrtcModule = await import('wrtc');
            // wrtc exports everything under default in ESM
            const wrtc = wrtcModule.default || wrtcModule;
            return {
                RTCPeerConnection: wrtc.RTCPeerConnection,
                RTCSessionDescription: wrtc.RTCSessionDescription,
                RTCIceCandidate: wrtc.RTCIceCandidate,
            };
        } catch (err) {
            // wrtc not available
            console.warn('WebRTC not available in Node.js:', err.message);
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
        // Create RTCSessionDescription using stored class reference
        const RTCSessionDescription = this._RTCSessionDescription || globalThis.RTCSessionDescription;
        const RTCIceCandidate = this._RTCIceCandidate || globalThis.RTCIceCandidate;

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
        // Create RTCSessionDescription using stored class reference
        const RTCSessionDescription = this._RTCSessionDescription || globalThis.RTCSessionDescription;
        const RTCIceCandidate = this._RTCIceCandidate || globalThis.RTCIceCandidate;

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
        const RTCIceCandidate = this._RTCIceCandidate || globalThis.RTCIceCandidate;

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
        // External signaling callback (e.g., Firebase)
        this.externalSignaling = null;
    }

    /**
     * Set external signaling callback for Firebase or other signaling
     */
    setExternalSignaling(callback) {
        this.externalSignaling = callback;
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

            // Send offer via available signaling method
            const signalData = {
                type: 'offer',
                to: peerId,
                from: this.localIdentity.piKey,
                offer,
            };

            if (this.signalingSocket?.readyState === 1) {
                // WebSocket signaling
                this.signalingSocket.send(JSON.stringify(signalData));
            } else if (this.externalSignaling) {
                // External signaling (Firebase, etc.)
                await this.externalSignaling('offer', peerId, offer);
            } else {
                throw new Error('No signaling method available');
            }

            this.peers.set(peerId, peerConnection);
            this.emit('peers-updated', this.getPeerList());

        } catch (err) {
            this.stats.failedConnections++;
            console.error(`Failed to connect to ${peerId}:`, err.message);
            throw err;
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

            // Send answer via available signaling method
            if (this.signalingSocket?.readyState === 1) {
                this.signalingSocket.send(JSON.stringify({
                    type: 'answer',
                    to: from,
                    from: this.localIdentity.piKey,
                    answer,
                }));
            } else if (this.externalSignaling) {
                await this.externalSignaling('answer', from, answer);
            }

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
        peerConnection.on('ice-candidate', async ({ candidate }) => {
            // Forward ICE candidate via available signaling method
            if (this.signalingSocket?.readyState === 1) {
                this.signalingSocket.send(JSON.stringify({
                    type: 'ice-candidate',
                    to: peerConnection.peerId,
                    from: this.localIdentity.piKey,
                    candidate,
                }));
            } else if (this.externalSignaling) {
                // Use external signaling (Firebase, etc.)
                try {
                    await this.externalSignaling('ice-candidate', peerConnection.peerId, candidate);
                } catch (err) {
                    console.warn('Failed to send ICE candidate via external signaling:', err.message);
                }
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

/**
 * ICE Connection Diagnostics
 *
 * Provides comprehensive diagnostics for ICE connectivity issues,
 * helping identify NAT types, test STUN/TURN servers, and diagnose
 * connection failures.
 */
export class ICEDiagnostics extends EventEmitter {
    constructor(options = {}) {
        super();
        this.options = {
            timeout: options.timeout || 10000,
            iceServers: options.iceServers || WEBRTC_CONFIG.iceServers,
            verbose: options.verbose || false,
        };
        this.results = {
            startTime: null,
            endTime: null,
            natType: 'unknown',
            stunResults: [],
            turnResults: [],
            candidateTypes: [],
            errors: [],
            recommendations: [],
        };
    }

    /**
     * Run full ICE diagnostics suite
     */
    async runDiagnostics() {
        this.results.startTime = Date.now();
        this.log('Starting ICE diagnostics...');

        try {
            // Test STUN servers
            await this.testStunServers();

            // Test TURN servers
            await this.testTurnServers();

            // Determine NAT type
            this.determineNatType();

            // Generate recommendations
            this.generateRecommendations();

        } catch (err) {
            this.results.errors.push({
                phase: 'diagnostics',
                error: err.message,
            });
        }

        this.results.endTime = Date.now();
        this.results.duration = this.results.endTime - this.results.startTime;

        return this.results;
    }

    /**
     * Test STUN servers for reachability
     */
    async testStunServers() {
        const stunServers = this.options.iceServers.filter(
            server => server.urls && server.urls.startsWith('stun:')
        );

        this.log(`Testing ${stunServers.length} STUN servers...`);

        for (const server of stunServers) {
            const result = await this.testIceServer(server, 'stun');
            this.results.stunResults.push(result);
            this.emit('stun-result', result);
        }
    }

    /**
     * Test TURN servers for reachability and authentication
     */
    async testTurnServers() {
        const turnServers = this.options.iceServers.filter(
            server => server.urls && (
                server.urls.startsWith('turn:') ||
                server.urls.startsWith('turns:')
            )
        );

        this.log(`Testing ${turnServers.length} TURN servers...`);

        for (const server of turnServers) {
            const result = await this.testIceServer(server, 'turn');
            this.results.turnResults.push(result);
            this.emit('turn-result', result);
        }
    }

    /**
     * Test a single ICE server
     */
    async testIceServer(server, type) {
        const startTime = Date.now();
        const result = {
            url: server.urls,
            type,
            success: false,
            latency: null,
            candidateType: null,
            error: null,
        };

        try {
            // Get RTCPeerConnection
            const RTCPeerConnection = globalThis.RTCPeerConnection ||
                (await this.loadNodeWebRTC());

            if (!RTCPeerConnection) {
                result.error = 'WebRTC not available';
                return result;
            }

            const pc = new RTCPeerConnection({
                iceServers: [server],
                iceTransportPolicy: type === 'turn' ? 'relay' : 'all',
            });

            const candidatePromise = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('ICE gathering timeout'));
                }, this.options.timeout);

                pc.onicecandidate = (event) => {
                    if (event.candidate) {
                        clearTimeout(timeout);
                        resolve(event.candidate);
                    }
                };

                pc.onicegatheringstatechange = () => {
                    if (pc.iceGatheringState === 'complete') {
                        clearTimeout(timeout);
                        reject(new Error('No candidates gathered'));
                    }
                };
            });

            // Create data channel to trigger ICE gathering
            pc.createDataChannel('test');

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait for candidate
            const candidate = await candidatePromise;

            result.success = true;
            result.latency = Date.now() - startTime;
            result.candidateType = candidate.type;
            result.candidate = {
                type: candidate.type,
                protocol: candidate.protocol,
                address: candidate.address,
                port: candidate.port,
            };

            // Track candidate types
            if (!this.results.candidateTypes.includes(candidate.type)) {
                this.results.candidateTypes.push(candidate.type);
            }

            pc.close();

        } catch (err) {
            result.error = err.message;
            result.latency = Date.now() - startTime;
        }

        this.log(`  ${type.toUpperCase()} ${server.urls}: ${result.success ? 'OK' : 'FAILED'} (${result.latency}ms)`);
        return result;
    }

    /**
     * Load wrtc for Node.js environment
     */
    async loadNodeWebRTC() {
        try {
            const wrtc = await import('wrtc');
            return wrtc.RTCPeerConnection;
        } catch (err) {
            return null;
        }
    }

    /**
     * Determine NAT type based on gathered candidates
     */
    determineNatType() {
        const hasHost = this.results.candidateTypes.includes('host');
        const hasSrflx = this.results.candidateTypes.includes('srflx');
        const hasRelay = this.results.candidateTypes.includes('relay');

        if (hasHost && hasSrflx) {
            // Can get server reflexive = not symmetric NAT
            const stunSuccess = this.results.stunResults.filter(r => r.success).length;
            if (stunSuccess >= 2) {
                this.results.natType = 'full-cone';
            } else {
                this.results.natType = 'restricted-cone';
            }
        } else if (hasHost && !hasSrflx && hasRelay) {
            // Can only get relay = symmetric NAT
            this.results.natType = 'symmetric';
        } else if (hasHost && !hasSrflx && !hasRelay) {
            // No external connectivity
            this.results.natType = 'blocked';
        } else if (!hasHost) {
            // No host candidates = unusual configuration
            this.results.natType = 'unknown';
        }

        this.log(`NAT Type detected: ${this.results.natType}`);
    }

    /**
     * Generate recommendations based on diagnostics
     */
    generateRecommendations() {
        const recs = [];

        // Check STUN connectivity
        const stunSuccess = this.results.stunResults.filter(r => r.success).length;
        const stunTotal = this.results.stunResults.length;

        if (stunSuccess === 0) {
            recs.push({
                severity: 'error',
                message: 'No STUN servers reachable',
                action: 'Check firewall rules for UDP port 3478',
            });
        } else if (stunSuccess < stunTotal / 2) {
            recs.push({
                severity: 'warning',
                message: `Only ${stunSuccess}/${stunTotal} STUN servers reachable`,
                action: 'Some STUN servers may be blocked or unreliable',
            });
        }

        // Check TURN connectivity
        const turnSuccess = this.results.turnResults.filter(r => r.success).length;
        const turnTotal = this.results.turnResults.length;

        if (this.results.natType === 'symmetric' && turnSuccess === 0) {
            recs.push({
                severity: 'error',
                message: 'Symmetric NAT detected but no TURN servers available',
                action: 'Configure a TURN server for reliable connectivity',
            });
        } else if (turnSuccess === 0 && turnTotal > 0) {
            recs.push({
                severity: 'warning',
                message: 'No TURN servers reachable',
                action: 'Check TURN credentials and firewall rules',
            });
        }

        // NAT-specific recommendations
        if (this.results.natType === 'symmetric') {
            recs.push({
                severity: 'info',
                message: 'Symmetric NAT detected',
                action: 'TURN server is required for reliable P2P connections',
            });
        }

        if (this.results.natType === 'blocked') {
            recs.push({
                severity: 'error',
                message: 'Network appears to block WebRTC',
                action: 'Configure TURN over TCP/443 or use a WebSocket fallback',
            });
        }

        // Performance recommendations
        const avgLatency = this.calculateAverageLatency();
        if (avgLatency > 200) {
            recs.push({
                severity: 'warning',
                message: `High ICE latency (${avgLatency}ms average)`,
                action: 'Consider using geographically closer STUN/TURN servers',
            });
        }

        this.results.recommendations = recs;
    }

    /**
     * Calculate average latency across all tests
     */
    calculateAverageLatency() {
        const allResults = [
            ...this.results.stunResults,
            ...this.results.turnResults,
        ].filter(r => r.success && r.latency);

        if (allResults.length === 0) return 0;

        const total = allResults.reduce((sum, r) => sum + r.latency, 0);
        return Math.round(total / allResults.length);
    }

    /**
     * Get a summary report
     */
    getSummary() {
        const stunOk = this.results.stunResults.filter(r => r.success).length;
        const turnOk = this.results.turnResults.filter(r => r.success).length;

        return {
            status: this.getOverallStatus(),
            natType: this.results.natType,
            stunServers: `${stunOk}/${this.results.stunResults.length} reachable`,
            turnServers: `${turnOk}/${this.results.turnResults.length} reachable`,
            avgLatency: `${this.calculateAverageLatency()}ms`,
            candidateTypes: this.results.candidateTypes,
            recommendations: this.results.recommendations.length,
            duration: `${this.results.duration}ms`,
        };
    }

    /**
     * Determine overall connectivity status
     */
    getOverallStatus() {
        const stunOk = this.results.stunResults.some(r => r.success);
        const turnOk = this.results.turnResults.some(r => r.success);
        const hasErrors = this.results.recommendations.some(r => r.severity === 'error');

        if (hasErrors) return 'error';
        if (stunOk && turnOk) return 'good';
        if (stunOk || turnOk) return 'degraded';
        return 'failed';
    }

    /**
     * Format results as readable string
     */
    formatReport() {
        const summary = this.getSummary();
        const lines = [
            '================================================================',
            '            ICE CONNECTION DIAGNOSTICS REPORT                   ',
            '================================================================',
            `  Status:         ${summary.status.toUpperCase()}`,
            `  NAT Type:       ${summary.natType}`,
            `  STUN Servers:   ${summary.stunServers}`,
            `  TURN Servers:   ${summary.turnServers}`,
            `  Avg Latency:    ${summary.avgLatency}`,
            `  Candidates:     ${summary.candidateTypes.join(', ') || 'none'}`,
            '----------------------------------------------------------------',
        ];

        if (this.results.recommendations.length > 0) {
            lines.push('  RECOMMENDATIONS:');
            for (const rec of this.results.recommendations) {
                const icon = rec.severity === 'error' ? '[!]' :
                            rec.severity === 'warning' ? '[W]' : '[i]';
                lines.push(`  ${icon} ${rec.message}`);
                lines.push(`      -> ${rec.action}`);
            }
        }

        lines.push('================================================================');

        return lines.join('\n');
    }

    log(message) {
        if (this.options.verbose) {
            console.log(`[ICE] ${message}`);
        }
        this.emit('log', message);
    }
}

/**
 * Quick ICE connectivity test
 *
 * Usage:
 *   import { testIceConnectivity } from './webrtc.js';
 *   const result = await testIceConnectivity();
 *   console.log(result.formatReport());
 */
export async function testIceConnectivity(options = {}) {
    const diagnostics = new ICEDiagnostics({
        verbose: true,
        ...options,
    });
    await diagnostics.runDiagnostics();
    return diagnostics;
}

/**
 * Create custom ICE configuration for specific provider
 *
 * Usage:
 *   import { createIceConfig, ICE_PROVIDERS } from './webrtc.js';
 *   const config = createIceConfig('coturn', 'turn.example.com', 'user', 'pass');
 */
export function createIceConfig(provider, ...args) {
    if (!ICE_PROVIDERS[provider]) {
        throw new Error(`Unknown ICE provider: ${provider}. Available: ${Object.keys(ICE_PROVIDERS).join(', ')}`);
    }
    return {
        iceServers: ICE_PROVIDERS[provider].config(...args),
    };
}

// Export default configuration for testing
export default {
    WebRTCPeerConnection,
    WebRTCPeerManager,
    QDAGSynchronizer,
    ICEDiagnostics,
    WEBRTC_CONFIG,
    ICE_PROVIDERS,
    testIceConnectivity,
    createIceConfig,
};

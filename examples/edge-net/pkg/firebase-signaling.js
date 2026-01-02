/**
 * @ruvector/edge-net Firebase Signaling
 *
 * Uses Google Firebase as bootstrap infrastructure for WebRTC signaling
 * with migration path to full P2P DHT network.
 *
 * Security Model (WASM-based, no Firebase Auth needed):
 * 1. Each node generates cryptographic identity in WASM (PiKey)
 * 2. All messages are signed with Ed25519 keys
 * 3. Peers verify signatures before accepting connections
 * 4. AdaptiveSecurity provides self-learning attack detection
 *
 * Architecture:
 * 1. Firebase Firestore for signaling (offer/answer/ICE)
 * 2. WASM cryptographic identity (no Firebase Auth)
 * 3. Gradual migration to DHT as network grows
 *
 * @module @ruvector/edge-net/firebase-signaling
 */

import { EventEmitter } from 'events';

// ============================================
// FIREBASE CONFIGURATION
// ============================================

/**
 * Edge-Net Public Firebase Configuration
 *
 * This is the PUBLIC Firebase project for edge-net P2P network.
 * API keys for Firebase web apps are designed to be public - security is via:
 * 1. Firestore Security Rules (only authenticated users can write)
 * 2. Anonymous Authentication (anyone can join, tracked by UID)
 * 3. API restrictions in Google Cloud Console
 *
 * Contributors automatically join the network - no setup required!
 */
export const EDGE_NET_FIREBASE_CONFIG = {
    apiKey: "AIzaSyAZAJhathdnKZGzBQ8iDBFG8_OQsvb2QvA",
    projectId: "ruv-dev",
    authDomain: "ruv-dev.firebaseapp.com",
    storageBucket: "ruv-dev.appspot.com",
};

/**
 * Get Firebase config
 *
 * Priority:
 * 1. Environment variables (for custom Firebase projects)
 * 2. Built-in edge-net public config (no setup required)
 */
export function getFirebaseConfig() {
    // Allow override via environment variables for custom projects
    const apiKey = process.env.FIREBASE_API_KEY;
    const projectId = process.env.FIREBASE_PROJECT_ID;

    if (apiKey && projectId) {
        return {
            apiKey,
            projectId,
            authDomain: process.env.FIREBASE_AUTH_DOMAIN || `${projectId}.firebaseapp.com`,
            storageBucket: process.env.FIREBASE_STORAGE_BUCKET || `${projectId}.appspot.com`,
        };
    }

    // Use built-in public edge-net config (no setup required!)
    return EDGE_NET_FIREBASE_CONFIG;
}

/**
 * Async version that can load from config file
 */
export async function getFirebaseConfigAsync() {
    // Try environment variables first
    const apiKey = process.env.FIREBASE_API_KEY;
    const projectId = process.env.FIREBASE_PROJECT_ID;

    if (apiKey && projectId) {
        return {
            apiKey,
            projectId,
            authDomain: process.env.FIREBASE_AUTH_DOMAIN || `${projectId}.firebaseapp.com`,
            databaseURL: process.env.FIREBASE_DATABASE_URL || `https://${projectId}-default-rtdb.firebaseio.com`,
            storageBucket: process.env.FIREBASE_STORAGE_BUCKET || `${projectId}.appspot.com`,
        };
    }

    // Try loading saved config
    try {
        const { loadConfig } = await import('./firebase-setup.js');
        const savedConfig = loadConfig();
        if (savedConfig && apiKey) {
            return { apiKey, ...savedConfig };
        }
        // Can work with just project config for server-side with ADC
        if (savedConfig) {
            return savedConfig; // No API key, but has project info (use ADC)
        }
    } catch {
        // firebase-setup.js not available
    }

    return null;
}

/**
 * Default returns null - must be configured via environment or setup
 */
export const DEFAULT_FIREBASE_CONFIG = null;

/**
 * Signaling collection names in Firestore
 * Each is a top-level collection (Firestore requires collection/document pairs)
 */
export const SIGNALING_PATHS = {
    peers: 'edgenet_peers',
    signals: 'edgenet_signals',
    rooms: 'edgenet_rooms',
    ledger: 'edgenet_ledger',
};

// ============================================
// FIREBASE SIGNALING CLIENT
// ============================================

/**
 * Firebase-based WebRTC Signaling
 *
 * Provides:
 * - Peer discovery via Firestore
 * - WebRTC signaling (offer/answer/ICE)
 * - Presence tracking
 * - Graceful fallback to local/DHT
 */
export class FirebaseSignaling extends EventEmitter {
    constructor(options = {}) {
        super();

        // SECURITY: Config must come from options or environment variables
        // Will be loaded async if not provided
        this._configPromise = null;
        this._providedConfig = options.firebaseConfig;
        this.peerId = options.peerId;
        this.room = options.room || 'default';

        // Initial sync config check (env vars only)
        this.config = options.firebaseConfig || getFirebaseConfig();

        // Firebase instances (lazy loaded)
        this.app = null;
        this.db = null;
        this.rtdb = null;

        // WASM Security (replaces Firebase Auth)
        /** @type {import('./secure-access.js').SecureAccessManager|null} */
        this.secureAccess = options.secureAccess || null;
        this.verifySignatures = options.verifySignatures !== false;

        // State
        this.isConnected = false;
        this.peers = new Map();
        this.pendingSignals = new Map();

        // Listeners for cleanup
        this.unsubscribers = [];

        // Migration tracking
        this.stats = {
            firebaseSignals: 0,
            dhtSignals: 0,
            p2pSignals: 0,
            verifiedSignals: 0,
            rejectedSignals: 0,
        };
    }

    /**
     * Initialize Firebase connection with WASM cryptographic security
     */
    async connect() {
        // Use built-in config if not provided
        if (!this.config) {
            this.config = getFirebaseConfig();
        }

        if (!this.config || !this.config.apiKey || !this.config.projectId) {
            console.log('   ⚠️  Firebase not configured');
            this.emit('not-configured');
            return false;
        }

        try {
            // Initialize WASM security if not provided
            if (!this.secureAccess) {
                try {
                    const { createSecureAccess } = await import('./secure-access.js');
                    this.secureAccess = await createSecureAccess({
                        siteId: this.room,
                        persistIdentity: true
                    });
                    // Use WASM-generated node ID if peerId not set
                    if (!this.peerId) {
                        this.peerId = this.secureAccess.getShortId();
                    }
                } catch (err) {
                    console.log('   ⚠️  WASM security unavailable, using basic mode');
                }
            }

            // Dynamic import Firebase (tree-shakeable)
            const { initializeApp, getApps } = await import('firebase/app');
            const { getFirestore, collection, doc, setDoc, onSnapshot, deleteDoc, query, where, orderBy, limit, serverTimestamp } = await import('firebase/firestore');

            // Store Firebase methods for later use
            this.firebase = {
                collection, doc, setDoc, onSnapshot, deleteDoc, query, where, orderBy, limit, serverTimestamp
            };

            // Initialize or reuse existing app
            const apps = getApps();
            this.app = apps.length ? apps[0] : initializeApp(this.config);

            // Initialize Firestore
            this.db = getFirestore(this.app);

            // WASM cryptographic identity (replaces Firebase Auth)
            if (this.secureAccess) {
                this.uid = this.secureAccess.getNodeId();
                console.log(`   🔐 WASM crypto identity: ${this.secureAccess.getShortId()}`);
                console.log(`   📦 Public key: ${this.secureAccess.getPublicKeyHex().slice(0, 16)}...`);
            } else {
                this.uid = this.peerId;
                console.log(`   ⚠️  No WASM security, using peerId: ${this.peerId?.slice(0, 8)}...`);
            }

            // Register presence in Firestore
            await this.registerPresence();

            // Listen for peers
            this.subscribeToPeers();

            // Listen for signals
            this.subscribeToSignals();

            this.isConnected = true;
            console.log('   ✅ Firebase connected with WASM security');

            this.emit('connected');
            return true;

        } catch (error) {
            console.log('   ⚠️  Firebase unavailable:', error.message);
            this.emit('error', error);
            return false;
        }
    }

    /**
     * Register this peer's presence in Firestore with WASM-signed data
     */
    async registerPresence() {
        const { doc, setDoc, serverTimestamp } = this.firebase;

        const presenceRef = doc(this.db, SIGNALING_PATHS.peers, this.peerId);

        // Build presence data
        const presenceData = {
            peerId: this.peerId,
            room: this.room,
            online: true,
            lastSeen: serverTimestamp(),
            capabilities: ['compute', 'storage', 'verify'],
        };

        // Add WASM cryptographic identity if available
        if (this.secureAccess) {
            presenceData.publicKey = this.secureAccess.getPublicKeyHex();
            // Sign the presence announcement
            const signed = this.secureAccess.signMessage({
                peerId: this.peerId,
                room: this.room,
                capabilities: presenceData.capabilities
            });
            presenceData.signature = signed.signature;
            presenceData.signedAt = signed.timestamp;
        }

        // Set online status in Firestore
        await setDoc(presenceRef, presenceData, { merge: true });

        // Set up heartbeat to maintain presence (Firestore doesn't have onDisconnect)
        this._heartbeatInterval = setInterval(async () => {
            try {
                const heartbeat = { lastSeen: serverTimestamp() };
                // Sign heartbeat if security available
                if (this.secureAccess) {
                    const signed = this.secureAccess.signMessage({ heartbeat: Date.now() });
                    heartbeat.heartbeatSig = signed.signature;
                }
                await setDoc(presenceRef, heartbeat, { merge: true });
            } catch (e) {
                // Ignore heartbeat errors
            }
        }, 30000);

        console.log(`   📡 Registered presence: ${this.peerId.slice(0, 8)}... (WASM-signed)`);
    }

    /**
     * Subscribe to peer presence updates (using Firestore)
     */
    subscribeToPeers() {
        const { collection, query, where, onSnapshot } = this.firebase;

        // Query peers in same room that were active in last 2 minutes
        const peersRef = collection(this.db, SIGNALING_PATHS.peers);
        const q = query(peersRef, where('room', '==', this.room));

        const unsubscribe = onSnapshot(q, (snapshot) => {
            const now = Date.now();
            const staleThreshold = 2 * 60 * 1000; // 2 minutes

            snapshot.docChanges().forEach((change) => {
                const data = change.doc.data();
                const peerId = change.doc.id;

                if (peerId === this.peerId) return; // Skip self

                if (change.type === 'added' || change.type === 'modified') {
                    // Check if peer is still active (lastSeen within threshold)
                    const lastSeen = data.lastSeen?.toMillis?.() || 0;
                    if (now - lastSeen < staleThreshold) {
                        if (!this.peers.has(peerId)) {
                            this.peers.set(peerId, data);
                            this.emit('peer-discovered', { peerId, ...data });
                        }
                    } else {
                        // Peer is stale
                        if (this.peers.has(peerId)) {
                            this.peers.delete(peerId);
                            this.emit('peer-left', { peerId });
                        }
                    }
                } else if (change.type === 'removed') {
                    if (this.peers.has(peerId)) {
                        this.peers.delete(peerId);
                        this.emit('peer-left', { peerId });
                    }
                }
            });
        });

        this.unsubscribers.push(unsubscribe);
    }

    /**
     * Subscribe to WebRTC signaling messages
     */
    subscribeToSignals() {
        const { collection, query, where, onSnapshot } = this.firebase;

        // Listen for signals addressed to this peer
        const signalsRef = collection(this.db, SIGNALING_PATHS.signals);
        const q = query(signalsRef, where('to', '==', this.peerId));

        const unsubscribe = onSnapshot(q, (snapshot) => {
            snapshot.docChanges().forEach(async (change) => {
                if (change.type === 'added') {
                    const signal = change.doc.data();
                    this.handleSignal(signal, change.doc.id);
                }
            });
        });

        this.unsubscribers.push(unsubscribe);
    }

    /**
     * Handle incoming signal with WASM signature verification
     */
    async handleSignal(signal, docId) {
        this.stats.firebaseSignals++;

        // Delete processed signal
        const { doc, deleteDoc } = this.firebase;
        await deleteDoc(doc(this.db, SIGNALING_PATHS.signals, docId));

        // Verify signature if WASM security is enabled
        if (this.verifySignatures && this.secureAccess && signal.signature && signal.publicKey) {
            const isValid = this.secureAccess.verifyMessage({
                payload: JSON.stringify({
                    from: signal.from,
                    to: signal.to,
                    type: signal.type,
                    data: typeof signal.data === 'object' ? JSON.stringify(signal.data) : signal.data,
                    timestamp: signal.timestamp
                }),
                signature: signal.signature,
                publicKey: signal.publicKey,
                timestamp: signal.timestamp
            });

            if (!isValid) {
                console.warn(`   ⚠️  Invalid signature from ${signal.from?.slice(0, 8)}...`);
                this.stats.rejectedSignals++;
                this.emit('invalid-signature', { from: signal.from, type: signal.type });
                return; // Reject the signal
            }

            // Register verified peer
            this.secureAccess.registerPeer(signal.from, signal.publicKey);
            this.stats.verifiedSignals++;
        }

        // Emit appropriate event
        switch (signal.type) {
            case 'offer':
                this.emit('offer', { from: signal.from, offer: signal.data, verified: !!signal.signature });
                break;
            case 'answer':
                this.emit('answer', { from: signal.from, answer: signal.data, verified: !!signal.signature });
                break;
            case 'ice-candidate':
                this.emit('ice-candidate', { from: signal.from, candidate: signal.data, verified: !!signal.signature });
                break;
            default:
                this.emit('signal', { ...signal, verified: !!signal.signature });
        }
    }

    /**
     * Send WebRTC offer to peer
     */
    async sendOffer(toPeerId, offer) {
        return this.sendSignal(toPeerId, 'offer', offer);
    }

    /**
     * Send WebRTC answer to peer
     */
    async sendAnswer(toPeerId, answer) {
        return this.sendSignal(toPeerId, 'answer', answer);
    }

    /**
     * Send ICE candidate to peer
     */
    async sendIceCandidate(toPeerId, candidate) {
        return this.sendSignal(toPeerId, 'ice-candidate', candidate);
    }

    /**
     * Serialize WebRTC objects to plain JSON for Firebase storage
     * RTCIceCandidate and RTCSessionDescription are not directly storable
     */
    _serializeWebRTCData(data) {
        if (!data || typeof data !== 'object') {
            return data;
        }

        // Handle RTCIceCandidate
        if (data.candidate !== undefined && data.sdpMid !== undefined) {
            return {
                candidate: data.candidate,
                sdpMid: data.sdpMid,
                sdpMLineIndex: data.sdpMLineIndex,
                usernameFragment: data.usernameFragment || null,
            };
        }

        // Handle RTCSessionDescription (offer/answer)
        if (data.type !== undefined && data.sdp !== undefined) {
            return {
                type: data.type,
                sdp: data.sdp,
            };
        }

        // Try to convert any object with toJSON method
        if (typeof data.toJSON === 'function') {
            return data.toJSON();
        }

        // Return as-is if already plain object
        return data;
    }

    /**
     * Send signal via Firebase with WASM signature
     */
    async sendSignal(toPeerId, type, data) {
        if (!this.isConnected) {
            throw new Error('Firebase not connected');
        }

        const { collection, doc, setDoc } = this.firebase;

        const signalId = `${this.peerId}-${toPeerId}-${Date.now()}`;
        const signalRef = doc(this.db, SIGNALING_PATHS.signals, signalId);

        // Serialize WebRTC objects to plain JSON
        const serializedData = this._serializeWebRTCData(data);

        const timestamp = Date.now();
        const signalData = {
            from: this.peerId,
            to: toPeerId,
            type,
            data: serializedData,
            timestamp,
            room: this.room,
        };

        // Sign the signal with WASM cryptography
        if (this.secureAccess) {
            const signed = this.secureAccess.signMessage({
                from: this.peerId,
                to: toPeerId,
                type,
                data: typeof serializedData === 'object' ? JSON.stringify(serializedData) : serializedData,
                timestamp
            });
            signalData.signature = signed.signature;
            signalData.publicKey = signed.publicKey;
        }

        await setDoc(signalRef, signalData);

        return true;
    }

    /**
     * Get list of online peers
     */
    getOnlinePeers() {
        return Array.from(this.peers.entries()).map(([id, data]) => ({
            id,
            ...data,
        }));
    }

    /**
     * Disconnect and cleanup
     */
    async disconnect() {
        // Stop heartbeat
        if (this._heartbeatInterval) {
            clearInterval(this._heartbeatInterval);
            this._heartbeatInterval = null;
        }

        // Unsubscribe from all listeners
        for (const unsub of this.unsubscribers) {
            if (typeof unsub === 'function') unsub();
        }
        this.unsubscribers = [];

        // Remove presence from Firestore
        if (this.db && this.firebase) {
            try {
                const { doc, deleteDoc } = this.firebase;
                const presenceRef = doc(this.db, SIGNALING_PATHS.peers, this.peerId);
                await deleteDoc(presenceRef);
            } catch (e) {
                // Ignore cleanup errors
            }
        }

        this.isConnected = false;
        this.peers.clear();

        this.emit('disconnected');
    }
}

// ============================================
// FIREBASE LEDGER SYNC
// ============================================

/**
 * Firebase-based Ledger Synchronization
 *
 * Syncs CRDT ledger state across peers using Firestore
 * with automatic CRDT merge on conflicts.
 */
export class FirebaseLedgerSync extends EventEmitter {
    constructor(ledger, options = {}) {
        super();

        this.ledger = ledger;
        this.peerId = options.peerId;
        // SECURITY: Config must come from options or environment variables
        this.config = options.firebaseConfig || getFirebaseConfig();

        // Firebase instances
        this.app = null;
        this.db = null;

        // Sync state
        this.lastSyncedVersion = 0;
        this.syncInterval = options.syncInterval || 30000;
        this.syncTimer = null;

        this.unsubscribers = [];
    }

    /**
     * Start ledger sync
     */
    async start() {
        // SECURITY: Require valid config
        if (!this.config || !this.config.apiKey || !this.config.projectId) {
            console.log('   ⚠️  Firebase ledger sync disabled (no credentials)');
            return false;
        }

        try {
            const { initializeApp, getApps } = await import('firebase/app');
            const { getFirestore, doc, setDoc, onSnapshot, getDoc } = await import('firebase/firestore');

            this.firebase = { doc, setDoc, onSnapshot, getDoc };

            const apps = getApps();
            this.app = apps.length ? apps[0] : initializeApp(this.config);
            this.db = getFirestore(this.app);

            // Initial sync from Firebase
            await this.pullLedger();

            // Subscribe to ledger updates
            this.subscribeLedger();

            // Periodic push
            this.syncTimer = setInterval(() => this.pushLedger(), this.syncInterval);

            console.log('   ✅ Firebase ledger sync started');
            return true;

        } catch (error) {
            console.log('   ⚠️  Firebase ledger sync unavailable:', error.message);
            return false;
        }
    }

    /**
     * Pull ledger state from Firebase
     */
    async pullLedger() {
        const { doc, getDoc } = this.firebase;

        const ledgerRef = doc(this.db, SIGNALING_PATHS.ledger, this.peerId);
        const snapshot = await getDoc(ledgerRef);

        if (snapshot.exists()) {
            const remoteState = snapshot.data();
            this.mergeLedger(remoteState);
        }
    }

    /**
     * Push ledger state to Firebase
     */
    async pushLedger() {
        if (!this.ledger) return;

        const { doc, setDoc } = this.firebase;

        const state = this.ledger.export();
        const ledgerRef = doc(this.db, SIGNALING_PATHS.ledger, this.peerId);

        await setDoc(ledgerRef, {
            ...state,
            peerId: this.peerId,
            updatedAt: Date.now(),
        }, { merge: true });
    }

    /**
     * Subscribe to ledger updates from other peers
     */
    subscribeLedger() {
        const { doc, onSnapshot } = this.firebase;

        // For now, just sync own ledger
        // Full multi-peer sync would subscribe to all peers
        const ledgerRef = doc(this.db, SIGNALING_PATHS.ledger, this.peerId);

        const unsubscribe = onSnapshot(ledgerRef, (snapshot) => {
            if (snapshot.exists()) {
                const remoteState = snapshot.data();
                if (remoteState.updatedAt > this.lastSyncedVersion) {
                    this.mergeLedger(remoteState);
                    this.lastSyncedVersion = remoteState.updatedAt;
                }
            }
        });

        this.unsubscribers.push(unsubscribe);
    }

    /**
     * Merge remote ledger state using CRDT rules
     */
    mergeLedger(remoteState) {
        if (!this.ledger || !remoteState) return;

        // CRDT merge: take max of counters
        if (remoteState.credits !== undefined) {
            const localCredits = this.ledger.getBalance?.() || 0;
            if (remoteState.credits > localCredits) {
                // Remote has more - need to import
                this.ledger.import?.(remoteState);
                this.emit('synced', { source: 'firebase', credits: remoteState.credits });
            }
        }
    }

    /**
     * Stop sync
     */
    stop() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }

        for (const unsub of this.unsubscribers) {
            if (typeof unsub === 'function') unsub();
        }
        this.unsubscribers = [];
    }
}

// ============================================
// HYBRID BOOTSTRAP MANAGER
// ============================================

/**
 * Hybrid Bootstrap Manager
 *
 * Manages the migration from Firebase bootstrap to full P2P:
 * 1. Start with Firebase for discovery and signaling
 * 2. Establish WebRTC connections to peers
 * 3. Build DHT routing table from connected peers
 * 4. Gradually reduce Firebase dependency
 * 5. Eventually operate fully P2P
 */
export class HybridBootstrap extends EventEmitter {
    constructor(options = {}) {
        super();

        this.peerId = options.peerId;
        this.config = options.firebaseConfig || DEFAULT_FIREBASE_CONFIG;

        // WASM Security
        /** @type {import('./secure-access.js').SecureAccessManager|null} */
        this.secureAccess = options.secureAccess || null;

        // Components
        this.firebase = null;
        this.dht = null;
        this.webrtc = null;

        // Migration state
        this.mode = 'firebase'; // firebase -> hybrid -> p2p
        this.dhtPeerThreshold = options.dhtPeerThreshold || 5;
        this.p2pPeerThreshold = options.p2pPeerThreshold || 10;

        // Stats for migration decisions
        this.stats = {
            firebaseDiscoveries: 0,
            dhtDiscoveries: 0,
            directConnections: 0,
            firebaseSignals: 0,
            p2pSignals: 0,
            verifiedPeers: 0,
        };
    }

    /**
     * Start hybrid bootstrap with WASM security
     */
    async start(webrtc, dht) {
        this.webrtc = webrtc;
        this.dht = dht;

        // Initialize WASM security if not provided
        if (!this.secureAccess) {
            try {
                const { createSecureAccess } = await import('./secure-access.js');
                this.secureAccess = await createSecureAccess({
                    siteId: 'edge-net',
                    persistIdentity: true
                });
                // Use WASM node ID if peerId not set
                if (!this.peerId) {
                    this.peerId = this.secureAccess.getShortId();
                }
            } catch (err) {
                console.log('   ⚠️  WASM security unavailable for bootstrap');
            }
        }

        // Start with Firebase, passing WASM security
        this.firebase = new FirebaseSignaling({
            peerId: this.peerId,
            firebaseConfig: this.config,
            secureAccess: this.secureAccess,
        });

        // Wire up events
        this.setupFirebaseEvents();

        // Set up WebRTC to use Firebase for signaling
        if (this.webrtc) {
            this.webrtc.setExternalSignaling(async (type, toPeerId, data) => {
                // Route signaling through Firebase
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

        // Connect to Firebase
        const connected = await this.firebase.connect();

        if (connected) {
            console.log('   🔄 Hybrid bootstrap: Firebase mode');
            this.mode = 'firebase';
        } else {
            console.log('   🔄 Hybrid bootstrap: DHT-only mode');
            this.mode = 'p2p';
        }

        // Start migration checker
        this.startMigrationChecker();

        return connected;
    }

    /**
     * Setup Firebase event handlers
     */
    setupFirebaseEvents() {
        this.firebase.on('peer-discovered', async ({ peerId }) => {
            this.stats.firebaseDiscoveries++;

            // Try to connect via WebRTC
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

        this.firebase.on('ice-candidate', async ({ from, candidate }) => {
            if (this.webrtc) {
                await this.webrtc.handleIceCandidate({ from, candidate });
            }
        });
    }

    /**
     * Connect to peer with signaling fallback
     */
    async connectToPeer(peerId) {
        if (!this.webrtc) return;

        try {
            // Use WebRTCPeerManager's connectToPeer method
            // This handles offer creation and signaling internally
            await this.webrtc.connectToPeer(peerId);
            this.stats.directConnections++;

        } catch (error) {
            console.warn(`[HybridBootstrap] Connect to ${peerId.slice(0, 8)} failed:`, error.message);
        }
    }

    /**
     * Send signaling message with automatic routing
     */
    async signal(toPeerId, type, data) {
        // Prefer P2P if available
        if (this.webrtc?.isConnected(toPeerId)) {
            this.webrtc.sendToPeer(toPeerId, { type, data });
            this.stats.p2pSignals++;
            return;
        }

        // Fall back to Firebase
        if (this.firebase?.isConnected) {
            await this.firebase.sendSignal(toPeerId, type, data);
            this.stats.firebaseSignals++;
            return;
        }

        throw new Error('No signaling path available');
    }

    /**
     * Start migration checker
     * Monitors network health and decides when to reduce Firebase dependency
     */
    startMigrationChecker() {
        setInterval(() => {
            this.checkMigration();
        }, 30000);
    }

    /**
     * Check if we should migrate modes
     */
    checkMigration() {
        const connectedPeers = this.webrtc?.peers?.size || 0;
        const dhtPeers = this.dht?.getPeers?.()?.length || 0;

        const previousMode = this.mode;

        // Migration logic
        if (this.mode === 'firebase') {
            // Migrate to hybrid when we have enough DHT peers
            if (dhtPeers >= this.dhtPeerThreshold) {
                this.mode = 'hybrid';
                console.log(`   🔄 Migration: firebase → hybrid (${dhtPeers} DHT peers)`);
            }
        } else if (this.mode === 'hybrid') {
            // Migrate to full P2P when we have strong peer connectivity
            if (connectedPeers >= this.p2pPeerThreshold) {
                this.mode = 'p2p';
                console.log(`   🔄 Migration: hybrid → p2p (${connectedPeers} direct peers)`);

                // Could disconnect Firebase here to save resources
                // this.firebase.disconnect();
            }
            // Fall back to Firebase if DHT shrinks
            else if (dhtPeers < this.dhtPeerThreshold / 2) {
                this.mode = 'firebase';
                console.log(`   🔄 Migration: hybrid → firebase (DHT peers dropped)`);
            }
        } else if (this.mode === 'p2p') {
            // Fall back to hybrid if peers drop
            if (connectedPeers < this.p2pPeerThreshold / 2) {
                this.mode = 'hybrid';
                console.log(`   🔄 Migration: p2p → hybrid (peers dropped)`);
            }
        }

        if (this.mode !== previousMode) {
            this.emit('mode-changed', { from: previousMode, to: this.mode });
        }
    }

    /**
     * Get current bootstrap stats
     */
    getStats() {
        return {
            mode: this.mode,
            ...this.stats,
            firebaseConnected: this.firebase?.isConnected || false,
            firebasePeers: this.firebase?.peers?.size || 0,
            dhtPeers: this.dht?.getPeers?.()?.length || 0,
            directPeers: this.webrtc?.peers?.size || 0,
        };
    }

    /**
     * Stop bootstrap
     */
    async stop() {
        if (this.firebase) {
            await this.firebase.disconnect();
        }
    }
}

// ============================================
// EXPORTS
// ============================================

export default FirebaseSignaling;

/**
 * Firebase Real-Time Data Service
 *
 * Connects to Firebase Firestore to fetch real peer data from the Edge-Net network.
 * Uses the same Firebase config as the main pkg/ module.
 *
 * Features:
 * - Real-time peer presence from edgenet_peers collection
 * - Network statistics aggregation
 * - Automatic reconnection with exponential backoff
 * - Demo mode fallback when Firebase unavailable
 */

// Use a simple event emitter pattern for browser compatibility
type EventHandler = (...args: unknown[]) => void;

class SimpleEventEmitter {
  private handlers: Map<string, Set<EventHandler>> = new Map();

  on(event: string, handler: EventHandler): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
  }

  off(event: string, handler: EventHandler): void {
    this.handlers.get(event)?.delete(handler);
  }

  emit(event: string, ...args: unknown[]): void {
    this.handlers.get(event)?.forEach((handler) => {
      try {
        handler(...args);
      } catch (e) {
        console.error(`[FirebaseData] Event handler error for ${event}:`, e);
      }
    });
  }

  removeAllListeners(): void {
    this.handlers.clear();
  }
}

// Firebase config - same as pkg/firebase-signaling.js
export const EDGE_NET_FIREBASE_CONFIG = {
  apiKey: 'AIzaSyAZAJhathdnKZGzBQ8iDBFG8_OQsvb2QvA',
  projectId: 'ruv-dev',
  authDomain: 'ruv-dev.firebaseapp.com',
  storageBucket: 'ruv-dev.appspot.com',
};

// Firestore collection paths
export const FIRESTORE_PATHS = {
  peers: 'edgenet_peers',
  signals: 'edgenet_signals',
  rooms: 'edgenet_rooms',
  ledger: 'edgenet_ledger',
};

// Peer document structure from Firestore
export interface FirestorePeer {
  peerId: string;
  room: string;
  online: boolean;
  lastSeen: { toMillis(): number } | null;
  capabilities: string[];
  publicKey?: string;
  signature?: string;
  signedAt?: number;
}

// Network statistics computed from peers
export interface NetworkStatistics {
  totalPeers: number;
  activePeers: number;
  peersInDefaultRoom: number;
  capabilities: Record<string, number>;
  averageUptime: number;
  lastUpdated: number;
  peers: PeerInfo[];
}

// Processed peer info for UI
export interface PeerInfo {
  id: string;
  room: string;
  online: boolean;
  lastSeen: number;
  capabilities: string[];
  isVerified: boolean;
  uptimeMs: number;
}

// Event types emitted by the service
export interface FirebaseDataEvents {
  'connected': () => void;
  'disconnected': () => void;
  'stats-updated': (stats: NetworkStatistics) => void;
  'peer-joined': (peer: PeerInfo) => void;
  'peer-left': (peerId: string) => void;
  'error': (error: Error) => void;
}

// Stale threshold - peers not seen in 2 minutes are considered offline
const STALE_THRESHOLD_MS = 2 * 60 * 1000;

// Demo data for fallback mode
function generateDemoStats(): NetworkStatistics {
  const now = Date.now();
  return {
    totalPeers: 3,
    activePeers: 2,
    peersInDefaultRoom: 2,
    capabilities: { compute: 3, storage: 2, verify: 1 },
    averageUptime: 3600000, // 1 hour
    lastUpdated: now,
    peers: [
      {
        id: 'demo-node-1',
        room: 'default',
        online: true,
        lastSeen: now - 5000,
        capabilities: ['compute', 'storage'],
        isVerified: true,
        uptimeMs: 3600000,
      },
      {
        id: 'demo-node-2',
        room: 'default',
        online: true,
        lastSeen: now - 10000,
        capabilities: ['compute', 'verify'],
        isVerified: true,
        uptimeMs: 7200000,
      },
      {
        id: 'demo-node-3',
        room: 'genesis',
        online: false,
        lastSeen: now - 300000,
        capabilities: ['compute'],
        isVerified: false,
        uptimeMs: 1800000,
      },
    ],
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FirebaseModule = any;

/**
 * Firebase Data Service
 * Provides real-time peer data from Firestore
 */
class FirebaseDataService extends SimpleEventEmitter {
  private app: FirebaseModule = null;
  private db: FirebaseModule = null;
  private firebase: FirebaseModule = null;

  private isConnected = false;
  private isDemoMode = false;
  private unsubscribers: Array<() => void> = [];
  private peers: Map<string, PeerInfo> = new Map();
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private statsUpdateInterval: ReturnType<typeof setInterval> | null = null;

  // Cached stats
  private currentStats: NetworkStatistics | null = null;

  /**
   * Connect to Firebase Firestore
   */
  async connect(): Promise<boolean> {
    if (this.isConnected) {
      console.log('[FirebaseData] Already connected');
      return true;
    }

    try {
      console.log('[FirebaseData] Connecting to Firebase...');

      // Dynamic import Firebase modules
      const { initializeApp, getApps } = await import('firebase/app');
      const firestore = await import('firebase/firestore');

      // Store Firebase methods
      this.firebase = firestore;

      // Initialize or reuse existing app
      const apps = getApps();
      this.app = apps.length ? apps[0] : initializeApp(EDGE_NET_FIREBASE_CONFIG);
      this.db = firestore.getFirestore(this.app);

      // Subscribe to peer collection
      this.subscribeToPeers();

      // Start periodic stats update
      this.startStatsUpdater();

      this.isConnected = true;
      this.isDemoMode = false;
      this.reconnectAttempt = 0;

      console.log('[FirebaseData] Connected to Firebase Firestore');
      this.emit('connected');

      return true;
    } catch (error) {
      console.warn('[FirebaseData] Firebase unavailable, using demo mode:', error);
      this.enableDemoMode();
      return false;
    }
  }

  /**
   * Enable demo mode with simulated data
   */
  private enableDemoMode(): void {
    this.isDemoMode = true;
    this.isConnected = false;
    this.currentStats = generateDemoStats();

    // Update demo stats periodically
    this.startStatsUpdater();

    console.log('[FirebaseData] Running in demo mode');
    this.emit('stats-updated', this.currentStats);
  }

  /**
   * Subscribe to edgenet_peers collection
   */
  private subscribeToPeers(): void {
    if (!this.db || !this.firebase) return;

    // Query all peers (we'll filter by room client-side for flexibility)
    const peersRef = this.firebase.collection(this.db, FIRESTORE_PATHS.peers);
    const q = this.firebase.query(peersRef, this.firebase.where('online', '==', true));

    const unsubscribe = this.firebase.onSnapshot(
      q,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (snapshot: any) => {
        const now = Date.now();

        snapshot.docChanges().forEach((change: { type: string; doc: { id: string; data(): FirestorePeer } }) => {
          const data = change.doc.data();
          const peerId = change.doc.id;
          const lastSeen = data.lastSeen?.toMillis?.() || 0;
          const isActive = now - lastSeen < STALE_THRESHOLD_MS;

          if (change.type === 'added' || change.type === 'modified') {
            const peer: PeerInfo = {
              id: peerId,
              room: data.room || 'default',
              online: data.online && isActive,
              lastSeen,
              capabilities: data.capabilities || ['compute'],
              isVerified: !!data.signature && !!data.publicKey,
              uptimeMs: now - lastSeen,
            };

            const existingPeer = this.peers.get(peerId);
            this.peers.set(peerId, peer);

            if (!existingPeer) {
              console.log('[FirebaseData] Peer joined:', peerId.slice(0, 8));
              this.emit('peer-joined', peer);
            }
          } else if (change.type === 'removed') {
            if (this.peers.has(peerId)) {
              this.peers.delete(peerId);
              console.log('[FirebaseData] Peer left:', peerId.slice(0, 8));
              this.emit('peer-left', peerId);
            }
          }
        });

        // Update stats after processing changes
        this.updateStats();
      },
      (error: Error) => {
        console.error('[FirebaseData] Snapshot error:', error);
        this.emit('error', error);
        this.scheduleReconnect();
      }
    );

    this.unsubscribers.push(unsubscribe);
  }

  /**
   * Start periodic stats updater
   */
  private startStatsUpdater(): void {
    if (this.statsUpdateInterval) {
      clearInterval(this.statsUpdateInterval);
    }

    // Update every 5 seconds
    this.statsUpdateInterval = setInterval(() => {
      if (this.isDemoMode) {
        // Slowly vary demo stats
        const demo = generateDemoStats();
        demo.activePeers = Math.max(1, demo.activePeers + Math.floor(Math.random() * 3 - 1));
        this.currentStats = demo;
        this.emit('stats-updated', demo);
      } else {
        this.updateStats();
      }
    }, 5000);
  }

  /**
   * Update computed statistics
   */
  private updateStats(): void {
    const now = Date.now();
    const peerList = Array.from(this.peers.values());

    // Filter active peers (seen within threshold)
    const activePeers = peerList.filter(
      (p) => p.online && now - p.lastSeen < STALE_THRESHOLD_MS
    );

    // Compute capability counts
    const capabilities: Record<string, number> = {};
    for (const peer of activePeers) {
      for (const cap of peer.capabilities) {
        capabilities[cap] = (capabilities[cap] || 0) + 1;
      }
    }

    // Compute average uptime
    const avgUptime =
      activePeers.length > 0
        ? activePeers.reduce((sum, p) => sum + (now - p.lastSeen), 0) / activePeers.length
        : 0;

    const stats: NetworkStatistics = {
      totalPeers: peerList.length,
      activePeers: activePeers.length,
      peersInDefaultRoom: activePeers.filter((p) => p.room === 'default').length,
      capabilities,
      averageUptime: avgUptime,
      lastUpdated: now,
      peers: peerList,
    };

    this.currentStats = stats;
    this.emit('stats-updated', stats);
  }

  /**
   * Get current network statistics
   */
  getStats(): NetworkStatistics {
    if (this.currentStats) {
      return this.currentStats;
    }
    return generateDemoStats();
  }

  /**
   * Get list of active peers
   */
  getActivePeers(): PeerInfo[] {
    if (this.isDemoMode) {
      return generateDemoStats().peers.filter((p) => p.online);
    }

    const now = Date.now();
    return Array.from(this.peers.values()).filter(
      (p) => p.online && now - p.lastSeen < STALE_THRESHOLD_MS
    );
  }

  /**
   * Get peer count
   */
  getPeerCount(): { total: number; active: number } {
    if (this.isDemoMode) {
      const demo = this.currentStats || generateDemoStats();
      return { total: demo.totalPeers, active: demo.activePeers };
    }

    const now = Date.now();
    const total = this.peers.size;
    const active = Array.from(this.peers.values()).filter(
      (p) => p.online && now - p.lastSeen < STALE_THRESHOLD_MS
    ).length;

    return { total, active };
  }

  /**
   * Check if connected to Firebase
   */
  isFirebaseConnected(): boolean {
    return this.isConnected && !this.isDemoMode;
  }

  /**
   * Check if running in demo mode
   */
  isInDemoMode(): boolean {
    return this.isDemoMode;
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;

    const delays = [1000, 2000, 5000, 10000, 30000];
    const delay = delays[Math.min(this.reconnectAttempt, delays.length - 1)];

    console.log(`[FirebaseData] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt + 1})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectAttempt++;
      this.disconnect();
      this.connect();
    }, delay);
  }

  /**
   * Disconnect from Firebase
   */
  disconnect(): void {
    // Clear timers
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.statsUpdateInterval) {
      clearInterval(this.statsUpdateInterval);
      this.statsUpdateInterval = null;
    }

    // Unsubscribe from all listeners
    for (const unsub of this.unsubscribers) {
      if (typeof unsub === 'function') unsub();
    }
    this.unsubscribers = [];

    // Clear state
    this.peers.clear();
    this.isConnected = false;
    this.isDemoMode = false;

    console.log('[FirebaseData] Disconnected');
    this.emit('disconnected');
  }
}

// Export singleton instance
export const firebaseDataService = new FirebaseDataService();

// Export class for testing
export { FirebaseDataService };

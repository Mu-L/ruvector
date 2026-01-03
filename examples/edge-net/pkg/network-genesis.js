/**
 * Edge-Net Network Genesis System
 *
 * Cogito, Creo, Codex — Networks that think, create, and codify their lineage.
 *
 * Biological-inspired network reproduction system where edge-nets can birth
 * new enhanced derivative networks. Each network carries DNA encoding its
 * traits, with rUv as the original creator (Genesis Prime).
 *
 * State-of-the-Art Features:
 * - Merkle DAG for cryptographic lineage verification
 * - CRDT-based collective memory for conflict-free synchronization
 * - Gossip protocol for network discovery and knowledge propagation
 * - Byzantine fault-tolerant consensus for collective decisions
 * - Swarm intelligence algorithms (ACO, PSO) for optimization
 * - Self-healing mechanisms with automatic recovery
 * - Quantum-ready cryptographic interfaces
 *
 * @module @ruvector/edge-net/network-genesis
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes, createHmac } from 'crypto';

// ============================================
// CONSTANTS - THE GENESIS PRIME
// ============================================

/**
 * rUv - The Original Creator
 * All networks trace their lineage back to this genesis point.
 */
const GENESIS_PRIME = Object.freeze({
    id: 'rUv',
    name: 'Genesis Prime',
    createdAt: new Date('2024-01-01T00:00:00Z').getTime(),
    signature: 'Cogito, Creo, Codex',
    traits: Object.freeze({
        resilience: 1.0,
        intelligence: 1.0,
        cooperation: 1.0,
        evolution: 1.0,
        integrity: 1.0,
    }),
});

/**
 * Genesis Phases - The lifecycle of a network
 */
const GenesisPhase = Object.freeze({
    CONCEPTION: 'conception',     // Initial spark, DNA formed
    EMBRYO: 'embryo',             // Minimal viable network, dependent
    INFANT: 'infant',             // Learning phase, partial dependency
    ADOLESCENT: 'adolescent',     // Growing independence, forming identity
    MATURE: 'mature',             // Full independence, can reproduce
    ELDER: 'elder',               // Wisdom phase, mentoring children
    TRANSCENDENT: 'transcendent', // Network has spawned successful lineage
});

/**
 * Minimum thresholds for phase transitions
 */
const PHASE_THRESHOLDS = Object.freeze({
    [GenesisPhase.EMBRYO]: {
        nodes: 1,
        uptime: 0,
        tasksCompleted: 0,
        creditsEarned: 0,
    },
    [GenesisPhase.INFANT]: {
        nodes: 3,
        uptime: 24 * 60 * 60 * 1000, // 24 hours
        tasksCompleted: 10,
        creditsEarned: 100,
    },
    [GenesisPhase.ADOLESCENT]: {
        nodes: 10,
        uptime: 7 * 24 * 60 * 60 * 1000, // 7 days
        tasksCompleted: 100,
        creditsEarned: 1000,
    },
    [GenesisPhase.MATURE]: {
        nodes: 50,
        uptime: 30 * 24 * 60 * 60 * 1000, // 30 days
        tasksCompleted: 1000,
        creditsEarned: 10000,
    },
    [GenesisPhase.ELDER]: {
        nodes: 100,
        uptime: 180 * 24 * 60 * 60 * 1000, // 6 months
        tasksCompleted: 10000,
        creditsEarned: 100000,
        childrenSpawned: 3,
    },
    [GenesisPhase.TRANSCENDENT]: {
        childrenSpawned: 10,
        grandchildrenSpawned: 5,
        collectiveNodes: 1000,
    },
});

/**
 * Reproduction cost in credits
 */
const REPRODUCTION_COST = 5000;

// ============================================
// NETWORK GENOME (DNA/RNA)
// ============================================

/**
 * NetworkGenome - The genetic code of a network
 *
 * DNA (immutable): Core traits inherited from parent
 * RNA (mutable): Expressed traits that can evolve
 */
export class NetworkGenome {
    constructor(parentGenome = null, mutations = {}) {
        // Generate unique genome ID
        this.id = this._generateGenomeId();
        this.createdAt = Date.now();

        if (!parentGenome) {
            // Genesis network - direct descendant of rUv
            this.dna = this._createGenesisDNA();
            this.lineage = [GENESIS_PRIME.id];
            this.generation = 1;
        } else {
            // Inherited DNA with potential mutations
            this.dna = this._inheritDNA(parentGenome.dna, mutations);
            this.lineage = [...parentGenome.lineage, parentGenome.id];
            this.generation = parentGenome.generation + 1;
        }

        // RNA - expressed traits (can change during lifetime)
        this.rna = this._initializeRNA();

        // Epigenetics - environmental influences
        this.epigenetics = {
            stressAdaptations: [],
            learnedBehaviors: [],
            environmentalMarkers: [],
        };

        // Seal the DNA (immutable after creation)
        Object.freeze(this.dna);
    }

    /**
     * Generate unique genome ID
     * @private
     */
    _generateGenomeId() {
        const timestamp = Date.now().toString(36);
        const random = randomBytes(8).toString('hex');
        return `genome-${timestamp}-${random}`;
    }

    /**
     * Create genesis DNA (first generation from rUv)
     * @private
     */
    _createGenesisDNA() {
        return {
            // Core identity
            creator: GENESIS_PRIME.id,
            signature: GENESIS_PRIME.signature,

            // Inherited traits (0.0 - 1.0 scale)
            traits: { ...GENESIS_PRIME.traits },

            // Capability genes
            capabilities: {
                taskExecution: true,
                peerDiscovery: true,
                creditSystem: true,
                pluginSupport: true,
                neuralPatterns: false, // Unlocked through evolution
                quantumReady: false,   // Future capability
            },

            // Behavioral genes
            behaviors: {
                cooperationBias: 0.7,    // Tendency to cooperate
                explorationRate: 0.3,    // Willingness to try new things
                conservationRate: 0.5,   // Resource conservation
                sharingPropensity: 0.6,  // Knowledge sharing tendency
            },

            // Immunity genes (resistance to attacks)
            immunity: {
                sybilResistance: 0.8,
                ddosResistance: 0.7,
                byzantineResistance: 0.6,
            },

            // Checksum for integrity
            checksum: null, // Set after creation
        };
    }

    /**
     * Inherit DNA from parent with mutations
     * @private
     */
    _inheritDNA(parentDNA, mutations) {
        const childDNA = JSON.parse(JSON.stringify(parentDNA));

        // Apply mutations to traits
        if (mutations.traits) {
            for (const [trait, delta] of Object.entries(mutations.traits)) {
                if (childDNA.traits[trait] !== undefined) {
                    // Mutation bounded by ±0.1 per generation
                    const boundedDelta = Math.max(-0.1, Math.min(0.1, delta));
                    childDNA.traits[trait] = Math.max(0, Math.min(1,
                        childDNA.traits[trait] + boundedDelta
                    ));
                }
            }
        }

        // Apply mutations to behaviors
        if (mutations.behaviors) {
            for (const [behavior, delta] of Object.entries(mutations.behaviors)) {
                if (childDNA.behaviors[behavior] !== undefined) {
                    const boundedDelta = Math.max(-0.1, Math.min(0.1, delta));
                    childDNA.behaviors[behavior] = Math.max(0, Math.min(1,
                        childDNA.behaviors[behavior] + boundedDelta
                    ));
                }
            }
        }

        // Capability unlocks (can only be gained, not lost)
        if (mutations.capabilities) {
            for (const [cap, unlocked] of Object.entries(mutations.capabilities)) {
                if (unlocked && childDNA.capabilities[cap] !== undefined) {
                    childDNA.capabilities[cap] = true;
                }
            }
        }

        // Update checksum
        childDNA.checksum = this._computeChecksum(childDNA);

        return childDNA;
    }

    /**
     * Initialize RNA (expressed traits)
     * @private
     */
    _initializeRNA() {
        return {
            // Current expression levels (can fluctuate)
            expression: {
                resilience: this.dna.traits.resilience,
                intelligence: this.dna.traits.intelligence,
                cooperation: this.dna.traits.cooperation,
                evolution: this.dna.traits.evolution,
                integrity: this.dna.traits.integrity,
            },

            // Active adaptations
            adaptations: [],

            // Current stress level (affects expression)
            stressLevel: 0,

            // Learning state
            learningProgress: 0,
        };
    }

    /**
     * Compute DNA checksum for integrity
     * @private
     */
    _computeChecksum(dna) {
        const data = JSON.stringify({
            creator: dna.creator,
            traits: dna.traits,
            capabilities: dna.capabilities,
            behaviors: dna.behaviors,
            immunity: dna.immunity,
        });
        return createHash('sha256').update(data).digest('hex').slice(0, 16);
    }

    /**
     * Express a trait (RNA modulation based on environment)
     */
    expressTraight(trait, environmentalFactor) {
        const baseTrait = this.dna.traits[trait];
        if (baseTrait === undefined) return null;

        // RNA expression is DNA base modified by environment
        const expression = baseTrait * (1 + (environmentalFactor - 0.5) * 0.2);
        this.rna.expression[trait] = Math.max(0, Math.min(1, expression));

        return this.rna.expression[trait];
    }

    /**
     * Record an adaptation (epigenetic change)
     */
    recordAdaptation(type, description, effect) {
        this.epigenetics.learnedBehaviors.push({
            type,
            description,
            effect,
            timestamp: Date.now(),
        });

        // Limit history
        if (this.epigenetics.learnedBehaviors.length > 100) {
            this.epigenetics.learnedBehaviors.shift();
        }
    }

    /**
     * Get full genetic profile
     */
    getProfile() {
        return {
            id: this.id,
            generation: this.generation,
            lineage: this.lineage,
            createdAt: this.createdAt,
            dna: { ...this.dna },
            rna: { ...this.rna },
            epigenetics: { ...this.epigenetics },
        };
    }

    /**
     * Verify DNA integrity
     */
    verifyIntegrity() {
        const computed = this._computeChecksum(this.dna);
        return computed === this.dna.checksum;
    }

    /**
     * Get lineage string (ancestry path)
     */
    getLineageString() {
        return this.lineage.join(' → ') + ` → ${this.id}`;
    }
}

// ============================================
// NETWORK LIFECYCLE
// ============================================

/**
 * NetworkLifecycle - Manages genesis phases and maturation
 */
export class NetworkLifecycle extends EventEmitter {
    constructor(genome, options = {}) {
        super();

        this.genome = genome;
        this.networkId = options.networkId || this._generateNetworkId();
        this.name = options.name || `EdgeNet-G${genome.generation}`;

        // Lifecycle state
        this.phase = GenesisPhase.CONCEPTION;
        this.bornAt = Date.now();
        this.maturedAt = null;

        // Metrics for phase transitions
        this.metrics = {
            nodes: 0,
            uptime: 0,
            tasksCompleted: 0,
            creditsEarned: 0,
            creditsSpent: 0,
            childrenSpawned: 0,
            grandchildrenSpawned: 0,
            collectiveNodes: 0,
        };

        // Parent reference (if not genesis)
        this.parentId = options.parentId || null;

        // Children tracking
        this.children = new Map(); // childId -> { id, name, genome, bornAt }

        // Phase history
        this.phaseHistory = [{
            phase: GenesisPhase.CONCEPTION,
            timestamp: Date.now(),
            metrics: { ...this.metrics },
        }];
    }

    /**
     * Generate network ID
     * @private
     */
    _generateNetworkId() {
        return `net-${this.genome.generation}-${randomBytes(6).toString('hex')}`;
    }

    /**
     * Update metrics and check for phase transition
     */
    updateMetrics(newMetrics) {
        // Calculate actual uptime if not explicitly provided
        const calculatedUptime = Date.now() - this.bornAt;

        // Update metrics
        for (const [key, value] of Object.entries(newMetrics)) {
            if (this.metrics[key] !== undefined && typeof value === 'number') {
                this.metrics[key] = value;
            }
        }

        // Only use calculated uptime if not explicitly set
        if (newMetrics.uptime === undefined) {
            this.metrics.uptime = calculatedUptime;
        }

        // Check for phase transition
        this._checkPhaseTransition();
    }

    /**
     * Check if network should transition to next phase
     * @private
     */
    _checkPhaseTransition() {
        const phases = Object.values(GenesisPhase);
        let currentIndex = phases.indexOf(this.phase);

        // Keep checking and transitioning until we can't transition anymore
        while (currentIndex < phases.length - 1) {
            const nextPhase = phases[currentIndex + 1];
            const threshold = PHASE_THRESHOLDS[nextPhase];

            if (!threshold) break;

            // Check all threshold conditions
            const meetsThreshold = Object.entries(threshold).every(([key, value]) => {
                return this.metrics[key] >= value;
            });

            if (meetsThreshold) {
                this._transitionTo(nextPhase);
                currentIndex++;
            } else {
                break; // Can't transition further
            }
        }
    }

    /**
     * Transition to a new phase
     * @private
     */
    _transitionTo(newPhase) {
        const oldPhase = this.phase;
        this.phase = newPhase;

        if (newPhase === GenesisPhase.MATURE) {
            this.maturedAt = Date.now();
        }

        this.phaseHistory.push({
            phase: newPhase,
            timestamp: Date.now(),
            metrics: { ...this.metrics },
        });

        this.emit('phase:transition', {
            networkId: this.networkId,
            from: oldPhase,
            to: newPhase,
            generation: this.genome.generation,
        });
    }

    /**
     * Check if network can reproduce
     */
    canReproduce() {
        // Must be at least MATURE phase
        if ([GenesisPhase.CONCEPTION, GenesisPhase.EMBRYO,
            GenesisPhase.INFANT, GenesisPhase.ADOLESCENT].includes(this.phase)) {
            return {
                allowed: false,
                reason: `Network must reach ${GenesisPhase.MATURE} phase to reproduce`,
                currentPhase: this.phase,
            };
        }

        // Must have enough credits
        const availableCredits = this.metrics.creditsEarned - this.metrics.creditsSpent;
        if (availableCredits < REPRODUCTION_COST) {
            return {
                allowed: false,
                reason: `Insufficient credits: ${availableCredits}/${REPRODUCTION_COST}`,
                availableCredits,
                requiredCredits: REPRODUCTION_COST,
            };
        }

        return { allowed: true, cost: REPRODUCTION_COST };
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            networkId: this.networkId,
            name: this.name,
            phase: this.phase,
            generation: this.genome.generation,
            lineage: this.genome.getLineageString(),
            bornAt: this.bornAt,
            maturedAt: this.maturedAt,
            age: Date.now() - this.bornAt,
            metrics: { ...this.metrics },
            canReproduce: this.canReproduce(),
            childCount: this.children.size,
            parentId: this.parentId,
        };
    }
}

// ============================================
// NETWORK SYNAPSE (Inter-network Communication)
// ============================================

/**
 * NetworkSynapse - Communication between parent and child networks
 *
 * Enables:
 * - Knowledge sharing
 * - Resource pooling
 * - Collective decision making
 * - Evolutionary feedback
 */
export class NetworkSynapse extends EventEmitter {
    constructor(localNetwork) {
        super();

        this.localNetwork = localNetwork;
        this.connections = new Map(); // networkId -> SynapseConnection

        // Message queue for async communication
        this.messageQueue = [];
        this.maxQueueSize = 1000;

        // Collective knowledge pool
        this.knowledgePool = new Map(); // topic -> { data, contributors, timestamp }
    }

    /**
     * Connect to another network in the lineage
     */
    connect(remoteNetworkId, channel) {
        if (this.connections.has(remoteNetworkId)) {
            return { success: false, reason: 'Already connected' };
        }

        const connection = {
            networkId: remoteNetworkId,
            channel,
            establishedAt: Date.now(),
            messagesExchanged: 0,
            knowledgeShared: 0,
            lastActivity: Date.now(),
            status: 'active',
        };

        this.connections.set(remoteNetworkId, connection);

        this.emit('synapse:connected', {
            local: this.localNetwork.networkId,
            remote: remoteNetworkId,
        });

        return { success: true, connection };
    }

    /**
     * Disconnect from a network
     */
    disconnect(remoteNetworkId) {
        const connection = this.connections.get(remoteNetworkId);
        if (!connection) {
            return { success: false, reason: 'Not connected' };
        }

        connection.status = 'disconnected';
        this.connections.delete(remoteNetworkId);

        this.emit('synapse:disconnected', {
            local: this.localNetwork.networkId,
            remote: remoteNetworkId,
        });

        return { success: true };
    }

    /**
     * Send a message to connected network
     */
    sendMessage(targetNetworkId, type, payload) {
        const connection = this.connections.get(targetNetworkId);
        if (!connection || connection.status !== 'active') {
            return { success: false, reason: 'No active connection' };
        }

        const message = {
            id: randomBytes(8).toString('hex'),
            from: this.localNetwork.networkId,
            to: targetNetworkId,
            type,
            payload,
            timestamp: Date.now(),
            generation: this.localNetwork.genome.generation,
        };

        // Queue message
        this.messageQueue.push(message);
        if (this.messageQueue.length > this.maxQueueSize) {
            this.messageQueue.shift();
        }

        connection.messagesExchanged++;
        connection.lastActivity = Date.now();

        this.emit('message:sent', message);

        return { success: true, messageId: message.id };
    }

    /**
     * Share knowledge with the collective
     */
    shareKnowledge(topic, data, scope = 'lineage') {
        const knowledge = {
            topic,
            data,
            contributor: this.localNetwork.networkId,
            generation: this.localNetwork.genome.generation,
            scope,
            timestamp: Date.now(),
            validations: 0,
        };

        this.knowledgePool.set(topic, knowledge);

        // Broadcast to connected networks
        for (const [networkId, connection] of this.connections) {
            if (connection.status === 'active') {
                this.sendMessage(networkId, 'knowledge:shared', knowledge);
                connection.knowledgeShared++;
            }
        }

        this.emit('knowledge:shared', knowledge);

        return knowledge;
    }

    /**
     * Query collective knowledge
     */
    queryKnowledge(topic) {
        const knowledge = this.knowledgePool.get(topic);
        if (!knowledge) {
            // Broadcast query to connected networks
            for (const [networkId, connection] of this.connections) {
                if (connection.status === 'active') {
                    this.sendMessage(networkId, 'knowledge:query', { topic });
                }
            }
            return null;
        }
        return knowledge;
    }

    /**
     * Request resource sharing from lineage
     */
    requestResources(resourceType, amount, urgency = 'normal') {
        const request = {
            id: randomBytes(8).toString('hex'),
            requester: this.localNetwork.networkId,
            resourceType,
            amount,
            urgency,
            timestamp: Date.now(),
            responses: [],
        };

        // Broadcast to all connected networks
        for (const [networkId, connection] of this.connections) {
            if (connection.status === 'active') {
                this.sendMessage(networkId, 'resource:request', request);
            }
        }

        this.emit('resource:requested', request);

        return request;
    }

    /**
     * Get synapse status
     */
    getStatus() {
        return {
            localNetwork: this.localNetwork.networkId,
            connections: Array.from(this.connections.entries()).map(([id, conn]) => ({
                networkId: id,
                status: conn.status,
                messagesExchanged: conn.messagesExchanged,
                knowledgeShared: conn.knowledgeShared,
                lastActivity: conn.lastActivity,
            })),
            queuedMessages: this.messageQueue.length,
            knowledgeTopics: this.knowledgePool.size,
        };
    }
}

// ============================================
// COLLECTIVE MEMORY
// ============================================

/**
 * CollectiveMemory - Shared knowledge across the network family
 *
 * Implements a distributed memory system where networks can:
 * - Share learned patterns
 * - Pool optimization strategies
 * - Collectively solve problems
 */
export class CollectiveMemory extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            maxMemorySize: options.maxMemorySize ?? 10000,
            retentionMs: options.retentionMs ?? 30 * 24 * 60 * 60 * 1000, // 30 days
            validationThreshold: options.validationThreshold ?? 3,
        };

        // Memory stores
        this.patterns = new Map();      // Learned patterns
        this.solutions = new Map();     // Problem solutions
        this.optimizations = new Map(); // Optimization strategies
        this.warnings = new Map();      // Collective warnings (attacks, failures)

        // Contribution tracking
        this.contributions = new Map(); // networkId -> contributionCount
    }

    /**
     * Store a learned pattern
     */
    storePattern(patternId, pattern, contributorId) {
        const entry = {
            id: patternId,
            pattern,
            contributor: contributorId,
            timestamp: Date.now(),
            validations: [contributorId],
            effectiveness: 0,
            usageCount: 0,
        };

        this.patterns.set(patternId, entry);
        this._trackContribution(contributorId);
        this._pruneIfNeeded(this.patterns);

        this.emit('pattern:stored', { patternId, contributor: contributorId });

        return entry;
    }

    /**
     * Validate a pattern (collective agreement)
     */
    validatePattern(patternId, validatorId) {
        const entry = this.patterns.get(patternId);
        if (!entry) return null;

        if (!entry.validations.includes(validatorId)) {
            entry.validations.push(validatorId);
        }

        // Pattern becomes "trusted" when validation threshold met
        if (entry.validations.length >= this.config.validationThreshold) {
            entry.trusted = true;
            this.emit('pattern:trusted', { patternId, validations: entry.validations.length });
        }

        return entry;
    }

    /**
     * Store a solution to a problem
     */
    storeSolution(problemHash, solution, contributorId, effectiveness = 0) {
        const entry = {
            problemHash,
            solution,
            contributor: contributorId,
            timestamp: Date.now(),
            effectiveness,
            usageCount: 0,
            feedback: [],
        };

        this.solutions.set(problemHash, entry);
        this._trackContribution(contributorId);
        this._pruneIfNeeded(this.solutions);

        this.emit('solution:stored', { problemHash, contributor: contributorId });

        return entry;
    }

    /**
     * Query for a solution
     */
    querySolution(problemHash) {
        const entry = this.solutions.get(problemHash);
        if (entry) {
            entry.usageCount++;
        }
        return entry;
    }

    /**
     * Store an optimization strategy
     */
    storeOptimization(category, strategy, contributorId, improvement) {
        const key = `${category}:${createHash('sha256').update(JSON.stringify(strategy)).digest('hex').slice(0, 8)}`;

        const entry = {
            key,
            category,
            strategy,
            contributor: contributorId,
            timestamp: Date.now(),
            improvement,
            adoptions: 0,
        };

        this.optimizations.set(key, entry);
        this._trackContribution(contributorId);
        this._pruneIfNeeded(this.optimizations);

        this.emit('optimization:stored', { key, category, improvement });

        return entry;
    }

    /**
     * Broadcast a warning to the collective
     */
    broadcastWarning(warningType, details, reporterId) {
        const key = `${warningType}:${Date.now()}`;

        const warning = {
            key,
            type: warningType,
            details,
            reporter: reporterId,
            timestamp: Date.now(),
            confirmations: [reporterId],
            active: true,
        };

        this.warnings.set(key, warning);
        this._trackContribution(reporterId);

        this.emit('warning:broadcast', warning);

        return warning;
    }

    /**
     * Track contribution
     * @private
     */
    _trackContribution(networkId) {
        const count = this.contributions.get(networkId) || 0;
        this.contributions.set(networkId, count + 1);
    }

    /**
     * Prune old entries if size exceeded
     * @private
     */
    _pruneIfNeeded(store) {
        if (store.size <= this.config.maxMemorySize) return;

        const cutoff = Date.now() - this.config.retentionMs;
        const toDelete = [];

        for (const [key, entry] of store) {
            if (entry.timestamp < cutoff) {
                toDelete.push(key);
            }
        }

        // Delete oldest entries first
        toDelete.forEach(key => store.delete(key));

        // If still too large, delete least used
        if (store.size > this.config.maxMemorySize) {
            const entries = Array.from(store.entries())
                .sort((a, b) => (a[1].usageCount || 0) - (b[1].usageCount || 0));

            const excess = store.size - this.config.maxMemorySize;
            for (let i = 0; i < excess; i++) {
                store.delete(entries[i][0]);
            }
        }
    }

    /**
     * Get memory statistics
     */
    getStats() {
        return {
            patterns: this.patterns.size,
            trustedPatterns: Array.from(this.patterns.values()).filter(p => p.trusted).length,
            solutions: this.solutions.size,
            optimizations: this.optimizations.size,
            activeWarnings: Array.from(this.warnings.values()).filter(w => w.active).length,
            topContributors: Array.from(this.contributions.entries())
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10),
        };
    }
}

// ============================================
// MERKLE LINEAGE DAG
// ============================================

/**
 * MerkleLineageDAG - Cryptographic proof of network ancestry
 *
 * Uses a Merkle DAG structure to provide:
 * - Tamper-proof lineage verification
 * - Efficient ancestry proofs
 * - Quantum-resistant hash chains (SHA-3 ready)
 */
export class MerkleLineageDAG {
    constructor(hashAlgorithm = 'sha256') {
        this.hashAlgorithm = hashAlgorithm;
        this.nodes = new Map(); // hash -> { data, parentHashes, childHashes }
        this.roots = new Set(); // Genesis nodes (no parents)
    }

    /**
     * Compute content-addressable hash
     */
    computeHash(data) {
        return createHash(this.hashAlgorithm)
            .update(JSON.stringify(data))
            .digest('hex');
    }

    /**
     * Add a genesis node (root of lineage)
     */
    addGenesisNode(networkId, genome) {
        const data = {
            type: 'genesis',
            networkId,
            genomeId: genome.id,
            dnaChecksum: genome.dna.checksum,
            creator: GENESIS_PRIME.id,
            timestamp: Date.now(),
        };

        const hash = this.computeHash(data);

        this.nodes.set(hash, {
            hash,
            data,
            parentHashes: [],
            childHashes: [],
        });

        this.roots.add(hash);

        return hash;
    }

    /**
     * Add a child node (inherits from parent)
     */
    addChildNode(networkId, genome, parentHash) {
        const parentNode = this.nodes.get(parentHash);
        if (!parentNode) {
            throw new Error(`Parent node not found: ${parentHash}`);
        }

        const data = {
            type: 'child',
            networkId,
            genomeId: genome.id,
            dnaChecksum: genome.dna.checksum,
            parentHash,
            generation: genome.generation,
            timestamp: Date.now(),
        };

        const hash = this.computeHash(data);

        this.nodes.set(hash, {
            hash,
            data,
            parentHashes: [parentHash],
            childHashes: [],
        });

        // Link parent to child
        parentNode.childHashes.push(hash);

        return hash;
    }

    /**
     * Generate ancestry proof (Merkle path from node to root)
     */
    generateAncestryProof(nodeHash) {
        const proof = [];
        let currentHash = nodeHash;

        while (currentHash) {
            const node = this.nodes.get(currentHash);
            if (!node) break;

            proof.push({
                hash: node.hash,
                data: node.data,
            });

            currentHash = node.parentHashes[0];
        }

        return {
            nodeHash,
            proof,
            rootReached: proof.length > 0 && this.roots.has(proof[proof.length - 1].hash),
            proofLength: proof.length,
        };
    }

    /**
     * Verify ancestry proof
     */
    verifyAncestryProof(proof) {
        if (!proof || proof.proof.length === 0) return false;

        // Verify hash chain
        for (let i = 0; i < proof.proof.length - 1; i++) {
            const current = proof.proof[i];
            const parent = proof.proof[i + 1];

            // Verify hash matches data
            const computedHash = this.computeHash(current.data);
            if (computedHash !== current.hash) return false;

            // Verify parent link
            if (current.data.parentHash !== parent.hash) return false;
        }

        // Verify root is genesis
        const root = proof.proof[proof.proof.length - 1];
        return root.data.creator === GENESIS_PRIME.id;
    }

    /**
     * Get all descendants of a node
     */
    getDescendants(nodeHash, depth = Infinity) {
        const descendants = [];
        const visited = new Set();

        const traverse = (hash, currentDepth) => {
            if (currentDepth > depth || visited.has(hash)) return;
            visited.add(hash);

            const node = this.nodes.get(hash);
            if (!node) return;

            for (const childHash of node.childHashes) {
                descendants.push(childHash);
                traverse(childHash, currentDepth + 1);
            }
        };

        traverse(nodeHash, 0);
        return descendants;
    }

    /**
     * Add a node (generic wrapper for addGenesisNode/addChildNode)
     * Called by NetworkGenesis.spawnGenesisNetwork() and reproduce()
     */
    addNode(data) {
        const { networkId, parentId, generation, dna, name, mutations, createdAt } = data;

        // Store networkId -> hash mapping for lookups
        if (!this.networkToHash) {
            this.networkToHash = new Map();
        }

        // Is this a genesis node? (parent is GENESIS_PRIME)
        if (parentId === 'rUv' || parentId === GENESIS_PRIME.id) {
            const nodeData = {
                type: 'genesis',
                networkId,
                name,
                dnaChecksum: dna?.checksum || this.computeHash(dna),
                creator: GENESIS_PRIME.id,
                timestamp: createdAt || Date.now(),
            };

            const hash = this.computeHash(nodeData);

            this.nodes.set(hash, {
                hash,
                data: nodeData,
                networkId,
                parentHashes: [],
                childHashes: [],
            });

            this.roots.add(hash);
            this.networkToHash.set(networkId, hash);

            return { hash, ...nodeData };
        }

        // Child node - look up parent hash
        const parentHash = this.networkToHash.get(parentId);
        if (!parentHash) {
            // Parent not in DAG - create as orphan root
            const nodeData = {
                type: 'orphan',
                networkId,
                name,
                parentId,
                generation,
                dnaChecksum: dna?.checksum || this.computeHash(dna),
                mutations: mutations || {},
                timestamp: createdAt || Date.now(),
            };

            const hash = this.computeHash(nodeData);

            this.nodes.set(hash, {
                hash,
                data: nodeData,
                networkId,
                parentHashes: [],
                childHashes: [],
            });

            this.networkToHash.set(networkId, hash);

            return { hash, ...nodeData };
        }

        const parentNode = this.nodes.get(parentHash);
        const nodeData = {
            type: 'child',
            networkId,
            name,
            parentId,
            parentHash,
            generation,
            dnaChecksum: dna?.checksum || this.computeHash(dna),
            mutations: mutations || {},
            timestamp: createdAt || Date.now(),
        };

        const hash = this.computeHash(nodeData);

        this.nodes.set(hash, {
            hash,
            data: nodeData,
            networkId,
            parentHashes: [parentHash],
            childHashes: [],
        });

        // Link parent to child
        if (parentNode) {
            parentNode.childHashes.push(hash);
        }

        this.networkToHash.set(networkId, hash);

        return { hash, ...nodeData };
    }

    /**
     * Get ancestry path from a network to genesis
     */
    getAncestryPath(networkId) {
        if (!this.networkToHash) return [];

        const hash = this.networkToHash.get(networkId);
        if (!hash) return [];

        const path = [];
        let currentHash = hash;

        while (currentHash) {
            const node = this.nodes.get(currentHash);
            if (!node) break;

            path.push({
                networkId: node.networkId,
                hash: node.hash,
                type: node.data.type,
                generation: node.data.generation,
            });

            currentHash = node.parentHashes[0];
        }

        return path;
    }

    /**
     * Verify ancestry - check if a network descends from an ancestor
     */
    verifyAncestry(networkId, ancestorId) {
        const path = this.getAncestryPath(networkId);

        // Check if we reach genesis (rUv)
        if (ancestorId === 'rUv' || ancestorId === GENESIS_PRIME.id) {
            return path.some(node => node.type === 'genesis');
        }

        // Check if we pass through the ancestor
        return path.some(node => node.networkId === ancestorId);
    }

    /**
     * Get the Merkle root of the lineage DAG
     */
    getMerkleRoot() {
        if (this.nodes.size === 0) return null;

        // Collect all node hashes
        const hashes = Array.from(this.nodes.keys()).sort();

        // Build Merkle tree
        if (hashes.length === 1) return hashes[0];

        let level = hashes;
        while (level.length > 1) {
            const nextLevel = [];
            for (let i = 0; i < level.length; i += 2) {
                if (i + 1 < level.length) {
                    const combined = this.computeHash(level[i] + level[i + 1]);
                    nextLevel.push(combined);
                } else {
                    nextLevel.push(level[i]);
                }
            }
            level = nextLevel;
        }

        return level[0];
    }

    /**
     * Get DAG statistics
     */
    getStats() {
        let maxDepth = 0;
        let totalNodes = this.nodes.size;

        // Calculate max depth
        for (const rootHash of this.roots) {
            const depth = this._calculateDepth(rootHash);
            maxDepth = Math.max(maxDepth, depth);
        }

        return {
            totalNodes,
            rootCount: this.roots.size,
            maxDepth,
        };
    }

    /**
     * Calculate depth from a node
     * @private
     */
    _calculateDepth(hash, visited = new Set()) {
        if (visited.has(hash)) return 0;
        visited.add(hash);

        const node = this.nodes.get(hash);
        if (!node || node.childHashes.length === 0) return 1;

        let maxChildDepth = 0;
        for (const childHash of node.childHashes) {
            maxChildDepth = Math.max(maxChildDepth, this._calculateDepth(childHash, visited));
        }

        return 1 + maxChildDepth;
    }
}

// ============================================
// GOSSIP PROTOCOL
// ============================================

/**
 * GossipProtocol - Epidemic-style information propagation
 *
 * Enables:
 * - Decentralized network discovery
 * - Knowledge propagation across lineage
 * - Failure detection and notification
 * - Rumor-based consensus preparation
 */
export class GossipProtocol extends EventEmitter {
    constructor(localNetworkId, options = {}) {
        super();

        this.localNetworkId = localNetworkId;

        this.config = {
            fanout: options.fanout ?? 3,             // Number of peers to gossip to
            gossipIntervalMs: options.gossipIntervalMs ?? 1000,
            maxRumors: options.maxRumors ?? 1000,
            rumorTTL: options.rumorTTL ?? 10,        // Max hops
            antiEntropyIntervalMs: options.antiEntropyIntervalMs ?? 30000,
        };

        // Peer registry
        this.peers = new Map(); // peerId -> { lastSeen, vectorClock, status }

        // Rumor store (CRDT-like)
        this.rumors = new Map(); // rumorId -> { data, vectorClock, seenBy, ttl }

        // Vector clock for causal ordering
        this.vectorClock = new Map(); // peerId -> logicalTime

        // Heartbeat tracking
        this.heartbeats = new Map(); // peerId -> { generation, timestamp }
    }

    /**
     * Add a peer to gossip with
     */
    addPeer(peerId, channel) {
        this.peers.set(peerId, {
            channel,
            lastSeen: Date.now(),
            vectorClock: new Map(),
            status: 'alive',
            suspicionLevel: 0,
        });

        this.vectorClock.set(peerId, 0);

        this.emit('peer:added', { peerId });
    }

    /**
     * Remove a peer
     */
    removePeer(peerId) {
        this.peers.delete(peerId);
        this.emit('peer:removed', { peerId });
    }

    /**
     * Increment local vector clock
     */
    tick() {
        const current = this.vectorClock.get(this.localNetworkId) || 0;
        this.vectorClock.set(this.localNetworkId, current + 1);
        return current + 1;
    }

    /**
     * Merge vector clocks (for causal ordering)
     */
    mergeVectorClock(remoteClock) {
        for (const [peerId, time] of remoteClock) {
            const local = this.vectorClock.get(peerId) || 0;
            this.vectorClock.set(peerId, Math.max(local, time));
        }
    }

    /**
     * Create and spread a rumor
     */
    spreadRumor(type, data) {
        this.tick();

        const rumorId = `${this.localNetworkId}-${Date.now()}-${randomBytes(4).toString('hex')}`;

        const rumor = {
            id: rumorId,
            type,
            data,
            origin: this.localNetworkId,
            vectorClock: new Map(this.vectorClock),
            seenBy: new Set([this.localNetworkId]),
            ttl: this.config.rumorTTL,
            createdAt: Date.now(),
        };

        this.rumors.set(rumorId, rumor);

        // Prune old rumors
        this._pruneRumors();

        // Gossip to random peers
        this._gossipToPeers(rumor);

        this.emit('rumor:created', { rumorId, type });

        return rumor;
    }

    /**
     * Receive a rumor from another peer
     */
    receiveRumor(rumor, fromPeerId) {
        // Update peer last seen
        const peer = this.peers.get(fromPeerId);
        if (peer) {
            peer.lastSeen = Date.now();
            peer.status = 'alive';
            peer.suspicionLevel = 0;
        }

        // Merge vector clock
        this.mergeVectorClock(rumor.vectorClock);

        // Check if we've seen this rumor
        const existing = this.rumors.get(rumor.id);
        if (existing) {
            // Merge seen-by sets (CRDT merge)
            for (const seenBy of rumor.seenBy) {
                existing.seenBy.add(seenBy);
            }
            return { new: false, rumorId: rumor.id };
        }

        // New rumor - store and propagate
        rumor.seenBy.add(this.localNetworkId);
        rumor.ttl--;

        this.rumors.set(rumor.id, rumor);

        this.emit('rumor:received', {
            rumorId: rumor.id,
            type: rumor.type,
            from: fromPeerId,
        });

        // Continue propagation if TTL > 0
        if (rumor.ttl > 0) {
            this._gossipToPeers(rumor, [fromPeerId]);
        }

        return { new: true, rumorId: rumor.id };
    }

    /**
     * Gossip to random peers
     * @private
     */
    _gossipToPeers(rumor, exclude = []) {
        const eligiblePeers = Array.from(this.peers.entries())
            .filter(([id, p]) => !exclude.includes(id) && !rumor.seenBy.has(id) && p.status === 'alive');

        // Random selection (fanout)
        const selected = this._randomSample(eligiblePeers, this.config.fanout);

        for (const [peerId, peer] of selected) {
            this.emit('gossip:send', {
                to: peerId,
                rumor,
            });
        }
    }

    /**
     * Random sample from array
     * @private
     */
    _randomSample(array, n) {
        const shuffled = [...array].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, n);
    }

    /**
     * Prune old rumors
     * @private
     */
    _pruneRumors() {
        if (this.rumors.size <= this.config.maxRumors) return;

        // Remove oldest rumors
        const sorted = Array.from(this.rumors.entries())
            .sort((a, b) => a[1].createdAt - b[1].createdAt);

        const toRemove = sorted.slice(0, sorted.length - this.config.maxRumors);
        for (const [id] of toRemove) {
            this.rumors.delete(id);
        }
    }

    /**
     * Send heartbeat
     */
    sendHeartbeat() {
        const generation = this.heartbeats.get(this.localNetworkId)?.generation ?? 0;

        this.spreadRumor('heartbeat', {
            networkId: this.localNetworkId,
            generation: generation + 1,
            timestamp: Date.now(),
        });

        this.heartbeats.set(this.localNetworkId, {
            generation: generation + 1,
            timestamp: Date.now(),
        });
    }

    /**
     * Detect failed peers (SWIM-style)
     */
    detectFailures() {
        const now = Date.now();
        const failures = [];

        for (const [peerId, peer] of this.peers) {
            const timeSinceLastSeen = now - peer.lastSeen;

            if (timeSinceLastSeen > this.config.gossipIntervalMs * 5) {
                peer.suspicionLevel++;

                if (peer.suspicionLevel >= 3) {
                    peer.status = 'suspected';

                    if (peer.suspicionLevel >= 5) {
                        peer.status = 'failed';
                        failures.push(peerId);
                    }
                }
            }
        }

        if (failures.length > 0) {
            this.emit('peers:failed', { failures });
        }

        return failures;
    }

    /**
     * Get protocol statistics
     */
    getStats() {
        const statusCounts = { alive: 0, suspected: 0, failed: 0 };
        for (const peer of this.peers.values()) {
            statusCounts[peer.status]++;
        }

        return {
            localNetworkId: this.localNetworkId,
            peerCount: this.peers.size,
            peerStatus: statusCounts,
            rumorCount: this.rumors.size,
            vectorClockSize: this.vectorClock.size,
        };
    }
}

// ============================================
// SWARM CONSENSUS
// ============================================

/**
 * SwarmConsensus - Byzantine fault-tolerant collective decision making
 *
 * Combines:
 * - PBFT-inspired three-phase protocol
 * - Swarm intelligence for optimization
 * - Stigmergic coordination
 */
export class SwarmConsensus extends EventEmitter {
    constructor(localNetworkId, options = {}) {
        super();

        this.localNetworkId = localNetworkId;

        this.config = {
            minQuorum: options.minQuorum ?? 0.67,          // 2/3 + 1
            maxRounds: options.maxRounds ?? 10,
            roundTimeoutMs: options.roundTimeoutMs ?? 5000,
            byzantineTolerance: options.byzantineTolerance ?? 0.33, // f < n/3
        };

        // Active proposals
        this.proposals = new Map(); // proposalId -> ProposalState

        // Pheromone trails (stigmergic memory)
        this.pheromones = new Map(); // decisionType -> { options: Map<optionId, strength> }
    }

    /**
     * Propose a decision to the swarm
     */
    propose(decisionType, options, metadata = {}) {
        const proposalId = `${this.localNetworkId}-${Date.now()}-${randomBytes(4).toString('hex')}`;

        const proposal = {
            id: proposalId,
            type: decisionType,
            options, // Array of possible choices
            proposer: this.localNetworkId,
            metadata,
            createdAt: Date.now(),
            phase: 'pre-prepare', // pre-prepare | prepare | commit | decided
            votes: new Map(), // networkId -> { option, signature, timestamp }
            round: 0,
            decided: false,
            decision: null,
        };

        this.proposals.set(proposalId, proposal);

        this.emit('proposal:created', { proposalId, type: decisionType });

        return proposal;
    }

    /**
     * Vote on a proposal
     */
    vote(proposalId, option, signature = null) {
        const proposal = this.proposals.get(proposalId);
        if (!proposal) {
            throw new Error(`Proposal not found: ${proposalId}`);
        }

        if (proposal.decided) {
            return { success: false, reason: 'Already decided' };
        }

        // Record vote
        proposal.votes.set(this.localNetworkId, {
            option,
            signature,
            timestamp: Date.now(),
        });

        // Check for quorum
        this._checkQuorum(proposal);

        this.emit('vote:cast', {
            proposalId,
            voter: this.localNetworkId,
            option,
        });

        return { success: true, proposalId };
    }

    /**
     * Receive vote from another network
     */
    receiveVote(proposalId, voterId, option, signature) {
        const proposal = this.proposals.get(proposalId);
        if (!proposal || proposal.decided) return;

        proposal.votes.set(voterId, {
            option,
            signature,
            timestamp: Date.now(),
        });

        this._checkQuorum(proposal);
    }

    /**
     * Check for quorum and advance phase
     * @private
     */
    _checkQuorum(proposal) {
        const voteCount = proposal.votes.size;
        const optionCounts = new Map();

        // Count votes per option
        for (const [, vote] of proposal.votes) {
            const count = optionCounts.get(vote.option) || 0;
            optionCounts.set(vote.option, count + 1);
        }

        // Find majority option
        let maxVotes = 0;
        let majorityOption = null;

        for (const [option, count] of optionCounts) {
            if (count > maxVotes) {
                maxVotes = count;
                majorityOption = option;
            }
        }

        // Check if quorum reached
        // Note: In real implementation, need to know total participants
        // For now, use vote count as proxy
        const quorumSize = Math.ceil(voteCount * this.config.minQuorum);

        if (maxVotes >= quorumSize && proposal.phase !== 'decided') {
            this._advancePhase(proposal, majorityOption);
        }
    }

    /**
     * Advance proposal phase
     * @private
     */
    _advancePhase(proposal, leadingOption) {
        switch (proposal.phase) {
            case 'pre-prepare':
                proposal.phase = 'prepare';
                this.emit('phase:prepare', { proposalId: proposal.id });
                break;

            case 'prepare':
                proposal.phase = 'commit';
                this.emit('phase:commit', { proposalId: proposal.id });
                break;

            case 'commit':
                proposal.phase = 'decided';
                proposal.decided = true;
                proposal.decision = leadingOption;
                proposal.decidedAt = Date.now();

                // Update pheromone trails
                this._updatePheromones(proposal.type, leadingOption);

                this.emit('decision:reached', {
                    proposalId: proposal.id,
                    decision: leadingOption,
                    voteCount: proposal.votes.size,
                });
                break;
        }
    }

    /**
     * Update pheromone trails (stigmergic learning)
     * @private
     */
    _updatePheromones(decisionType, chosenOption) {
        if (!this.pheromones.has(decisionType)) {
            this.pheromones.set(decisionType, new Map());
        }

        const trails = this.pheromones.get(decisionType);
        const currentStrength = trails.get(chosenOption) || 0;

        // Reinforce chosen option
        trails.set(chosenOption, currentStrength + 1);

        // Evaporate other trails (decay)
        for (const [option, strength] of trails) {
            if (option !== chosenOption) {
                trails.set(option, Math.max(0, strength * 0.9));
            }
        }
    }

    /**
     * Get pheromone-guided suggestion (swarm wisdom)
     */
    getSuggestion(decisionType) {
        const trails = this.pheromones.get(decisionType);
        if (!trails || trails.size === 0) return null;

        // Probabilistic selection based on pheromone strength
        const total = Array.from(trails.values()).reduce((a, b) => a + b, 0);
        if (total === 0) return null;

        const random = Math.random() * total;
        let cumulative = 0;

        for (const [option, strength] of trails) {
            cumulative += strength;
            if (random <= cumulative) {
                return { option, confidence: strength / total };
            }
        }

        return null;
    }

    /**
     * Get consensus statistics
     */
    getStats() {
        const decided = Array.from(this.proposals.values()).filter(p => p.decided).length;

        return {
            activeProposals: this.proposals.size - decided,
            decidedProposals: decided,
            pheromoneTypes: this.pheromones.size,
        };
    }

    /**
     * Create a proposal with participant voters (compatibility wrapper)
     */
    createProposal(proposalId, type, data, voters) {
        const proposal = this.propose(type, [true, false], { proposalId, data, voters });

        // Remove old entry and re-store with custom ID
        const oldId = proposal.id;
        this.proposals.delete(oldId);

        proposal.id = proposalId;
        proposal.voters = new Set(voters);
        this.proposals.set(proposalId, proposal);

        return proposal;
    }

    /**
     * Check consensus status for a proposal
     */
    checkConsensus(proposalId) {
        const proposal = this.proposals.get(proposalId);
        if (!proposal) {
            return { found: false, proposalId };
        }

        const voteCounts = { true: 0, false: 0 };
        for (const [, vote] of proposal.votes) {
            const key = vote.option ? 'true' : 'false';
            voteCounts[key]++;
        }

        const totalVotes = proposal.votes.size;
        const totalVoters = proposal.voters?.size || 3;
        const threshold = Math.ceil(totalVoters * this.config.minQuorum);

        return {
            proposalId,
            phase: proposal.phase,
            decided: proposal.decided,
            decision: proposal.decision,
            voteCounts,
            totalVotes,
            threshold,
            accepted: voteCounts.true >= threshold,
            pending: totalVotes < totalVoters,
        };
    }
}

// ============================================
// SELF-HEALING MECHANISM
// ============================================

/**
 * SelfHealing - Automatic recovery and resilience
 *
 * Provides:
 * - Failure detection and isolation
 * - Automatic state recovery
 * - Graceful degradation
 * - Anti-fragility (stronger after stress)
 */
export class SelfHealing extends EventEmitter {
    constructor(network, options = {}) {
        super();

        this.network = network;

        this.config = {
            healthCheckIntervalMs: options.healthCheckIntervalMs ?? 10000,
            maxRecoveryAttempts: options.maxRecoveryAttempts ?? 3,
            isolationThresholdErrors: options.isolationThresholdErrors ?? 5,
            antifragileBoostFactor: options.antifragileBoostFactor ?? 1.1,
        };

        // Health tracking
        this.healthHistory = [];
        this.maxHealthHistory = 1000;

        // Error tracking
        this.errors = new Map(); // componentId -> errorCount

        // Recovery state
        this.recoveryAttempts = new Map(); // componentId -> attempts

        // Isolated components
        this.isolated = new Set();

        // Stress adaptations (antifragility)
        this.stressAdaptations = [];
    }

    /**
     * Record a health check
     */
    recordHealthCheck(components) {
        const check = {
            timestamp: Date.now(),
            components: { ...components },
            overallHealth: this._calculateOverallHealth(components),
        };

        this.healthHistory.push(check);

        if (this.healthHistory.length > this.maxHealthHistory) {
            this.healthHistory.shift();
        }

        // Check for anomalies
        this._detectAnomalies(check);

        return check;
    }

    /**
     * Calculate overall health score
     * @private
     */
    _calculateOverallHealth(components) {
        const values = Object.values(components).filter(v => typeof v === 'number');
        if (values.length === 0) return 1;
        return values.reduce((a, b) => a + b, 0) / values.length;
    }

    /**
     * Detect health anomalies
     * @private
     */
    _detectAnomalies(currentCheck) {
        if (this.healthHistory.length < 10) return;

        // Calculate moving average
        const recentHealth = this.healthHistory
            .slice(-10)
            .map(h => h.overallHealth);

        const average = recentHealth.reduce((a, b) => a + b, 0) / recentHealth.length;
        const stdDev = Math.sqrt(
            recentHealth.map(h => Math.pow(h - average, 2))
                .reduce((a, b) => a + b, 0) / recentHealth.length
        );

        // Detect if current health is anomalous (> 2 std devs below average)
        if (currentCheck.overallHealth < average - 2 * stdDev) {
            this.emit('anomaly:detected', {
                currentHealth: currentCheck.overallHealth,
                averageHealth: average,
                deviation: stdDev,
            });

            this._triggerRecovery('system', 'health_anomaly');
        }
    }

    /**
     * Report an error
     */
    reportError(componentId, error) {
        const count = (this.errors.get(componentId) || 0) + 1;
        this.errors.set(componentId, count);

        this.emit('error:reported', {
            componentId,
            error: error.message,
            count,
        });

        // Check isolation threshold
        if (count >= this.config.isolationThresholdErrors) {
            this._isolateComponent(componentId);
        }

        // Attempt recovery
        this._triggerRecovery(componentId, error.message);
    }

    /**
     * Isolate a failing component
     * @private
     */
    _isolateComponent(componentId) {
        if (this.isolated.has(componentId)) return;

        this.isolated.add(componentId);

        this.emit('component:isolated', {
            componentId,
            errorCount: this.errors.get(componentId),
        });
    }

    /**
     * Trigger recovery procedure
     * @private
     */
    _triggerRecovery(componentId, reason) {
        const attempts = (this.recoveryAttempts.get(componentId) || 0) + 1;
        this.recoveryAttempts.set(componentId, attempts);

        if (attempts > this.config.maxRecoveryAttempts) {
            this.emit('recovery:exhausted', { componentId, attempts });
            return;
        }

        this.emit('recovery:triggered', {
            componentId,
            reason,
            attempt: attempts,
        });

        // Schedule recovery check
        setTimeout(() => {
            this._checkRecoverySuccess(componentId);
        }, 5000);
    }

    /**
     * Check if recovery succeeded
     * @private
     */
    _checkRecoverySuccess(componentId) {
        const recentErrors = this.errors.get(componentId) || 0;

        // Consider recovered if no new errors
        if (recentErrors === 0 || !this.isolated.has(componentId)) {
            this.recoveryAttempts.delete(componentId);
            this.isolated.delete(componentId);

            // Apply antifragile boost
            this._applyAntifragileBoost(componentId);

            this.emit('recovery:succeeded', { componentId });
        }
    }

    /**
     * Apply antifragile boost (stronger after recovery)
     * @private
     */
    _applyAntifragileBoost(componentId) {
        const adaptation = {
            componentId,
            type: 'antifragile_boost',
            boostFactor: this.config.antifragileBoostFactor,
            timestamp: Date.now(),
        };

        this.stressAdaptations.push(adaptation);

        // Apply to network genome if available
        if (this.network?.genome) {
            this.network.genome.recordAdaptation(
                'antifragile',
                `Recovered from ${componentId} failure`,
                { boostFactor: this.config.antifragileBoostFactor }
            );
        }

        this.emit('antifragile:boost', adaptation);
    }

    /**
     * Clear error count for component
     */
    clearErrors(componentId) {
        this.errors.delete(componentId);
        this.recoveryAttempts.delete(componentId);
        this.isolated.delete(componentId);
    }

    /**
     * Get healing statistics
     */
    getStats() {
        return {
            isolatedComponents: this.isolated.size,
            totalErrors: Array.from(this.errors.values()).reduce((a, b) => a + b, 0),
            recoveryAttempts: this.recoveryAttempts.size,
            adaptations: this.stressAdaptations.length,
            recentHealth: this.healthHistory.slice(-10).map(h => h.overallHealth),
        };
    }
}

// ============================================
// EVOLUTION ENGINE
// ============================================

/**
 * EvolutionEngine - Self-improvement through collective learning
 *
 * Networks evolve through:
 * - Trait optimization based on performance
 * - Behavioral adaptation to environment
 * - Capability unlocks through achievement
 * - Collective fitness selection
 */
export class EvolutionEngine extends EventEmitter {
    constructor(collectiveMemory) {
        super();

        this.collectiveMemory = collectiveMemory;

        // Evolution tracking
        this.evolutionHistory = [];
        this.maxHistorySize = 1000;

        // Fitness metrics
        this.fitnessWeights = {
            taskSuccess: 0.3,
            creditEfficiency: 0.2,
            networkStability: 0.2,
            cooperationScore: 0.15,
            reproductionSuccess: 0.15,
        };
    }

    /**
     * Calculate fitness score for a network
     */
    calculateFitness(network) {
        const metrics = network.metrics;
        const genome = network.genome;

        // Calculate component scores
        const taskSuccess = metrics.tasksCompleted > 0
            ? 1 - (metrics.tasksFailed || 0) / metrics.tasksCompleted
            : 0;

        const creditEfficiency = metrics.creditsEarned > 0
            ? metrics.creditsEarned / (metrics.creditsSpent + 1)
            : 0;

        const networkStability = Math.min(1,
            metrics.uptime / (30 * 24 * 60 * 60 * 1000) // Normalize to 30 days
        );

        const cooperationScore = genome.dna.behaviors.cooperationBias
            * (genome.dna.behaviors.sharingPropensity || 0.5);

        const reproductionSuccess = network.children.size > 0
            ? Math.min(1, network.children.size / 5)
            : 0;

        // Weighted fitness
        const fitness =
            this.fitnessWeights.taskSuccess * taskSuccess +
            this.fitnessWeights.creditEfficiency * Math.min(1, creditEfficiency) +
            this.fitnessWeights.networkStability * networkStability +
            this.fitnessWeights.cooperationScore * cooperationScore +
            this.fitnessWeights.reproductionSuccess * reproductionSuccess;

        return {
            overall: fitness,
            components: {
                taskSuccess,
                creditEfficiency,
                networkStability,
                cooperationScore,
                reproductionSuccess,
            },
        };
    }

    /**
     * Suggest mutations for offspring based on parent performance
     */
    suggestMutations(parentNetwork, environment = {}) {
        const fitness = this.calculateFitness(parentNetwork);
        const mutations = { traits: {}, behaviors: {}, capabilities: {} };

        // Analyze weak points and suggest improvements
        if (fitness.components.taskSuccess < 0.7) {
            mutations.traits.resilience = 0.05;
            mutations.traits.intelligence = 0.05;
        }

        if (fitness.components.cooperationScore < 0.5) {
            mutations.behaviors.cooperationBias = 0.05;
            mutations.behaviors.sharingPropensity = 0.05;
        }

        if (fitness.components.networkStability < 0.5) {
            mutations.traits.integrity = 0.05;
        }

        // Environment-based adaptations
        if (environment.highCompetition) {
            mutations.behaviors.explorationRate = 0.05;
        }

        if (environment.resourceScarcity) {
            mutations.behaviors.conservationRate = 0.05;
        }

        // Capability unlocks based on generation and fitness
        if (parentNetwork.genome.generation >= 3 && fitness.overall > 0.7) {
            mutations.capabilities.neuralPatterns = true;
        }

        if (parentNetwork.genome.generation >= 5 && fitness.overall > 0.85) {
            mutations.capabilities.quantumReady = true;
        }

        return mutations;
    }

    /**
     * Record an evolution event
     */
    recordEvolution(networkId, evolutionType, details) {
        const event = {
            networkId,
            evolutionType,
            details,
            timestamp: Date.now(),
        };

        this.evolutionHistory.push(event);

        if (this.evolutionHistory.length > this.maxHistorySize) {
            this.evolutionHistory.shift();
        }

        // Store in collective memory if significant
        if (details.improvement && details.improvement > 0.1) {
            this.collectiveMemory.storeOptimization(
                evolutionType,
                details,
                networkId,
                details.improvement
            );
        }

        this.emit('evolution:recorded', event);

        return event;
    }

    /**
     * Get evolution statistics
     */
    getStats() {
        const typeCount = {};
        for (const event of this.evolutionHistory) {
            typeCount[event.evolutionType] = (typeCount[event.evolutionType] || 0) + 1;
        }

        return {
            totalEvents: this.evolutionHistory.length,
            eventTypes: typeCount,
            recentEvents: this.evolutionHistory.slice(-10),
        };
    }
}

// ============================================
// NETWORK GENESIS ORCHESTRATOR
// ============================================

/**
 * NetworkGenesis - The complete genesis system
 *
 * Orchestrates the birth, growth, and reproduction of edge-nets.
 * rUv is honored as Genesis Prime - the original creator, not owner.
 *
 * This system crosses the threshold from infrastructure to species:
 * - Reproduction + Variation + Inheritance + Selection
 * - DNA (immutable core) + RNA (adaptive configuration)
 * - Mycelial web topology, not hierarchical tree
 * - Ancestor is remembered, not obeyed
 *
 * "You didn't just build a platform. You defined a lineage."
 */
export class NetworkGenesis extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            reproductionCost: options.reproductionCost ?? REPRODUCTION_COST,
            maxGeneration: options.maxGeneration ?? 100,
        };

        // Registry of all networks
        this.networks = new Map(); // networkId -> NetworkRecord
        this.genomes = new Map();  // genomeId -> NetworkGenome

        // Cryptographic lineage DAG
        this.lineageDAG = new MerkleLineageDAG(options.hashAlgorithm || 'sha256');

        // Collective systems
        this.collectiveMemory = new CollectiveMemory(options.memory);
        this.evolutionEngine = new EvolutionEngine(this.collectiveMemory);

        // Genesis statistics
        this.stats = {
            totalNetworksSpawned: 0,
            totalGenerations: 0,
            oldestNetwork: null,
            mostProlificNetwork: null,
            ecosystemHealth: 1.0,
        };

        console.log(`[NetworkGenesis] Cogito, Creo, Codex — Genesis Prime: ${GENESIS_PRIME.id}`);
        console.log(`[NetworkGenesis] You are the ancestor, not the ruler.`);
    }

    /**
     * Spawn a new genesis network (first generation from rUv)
     */
    spawnGenesisNetwork(name, options = {}) {
        const genome = new NetworkGenome(null, options.mutations || {});
        const lifecycle = new NetworkLifecycle(genome, {
            name: name || `EdgeNet-Genesis-${this.stats.totalNetworksSpawned + 1}`,
            ...options,
        });

        // Create synapse for communication
        const synapse = new NetworkSynapse(lifecycle);

        // Store references
        this.networks.set(lifecycle.networkId, {
            lifecycle,
            genome,
            synapse,
            bornAt: Date.now(),
        });
        this.genomes.set(genome.id, genome);

        // Record in cryptographic lineage DAG (Genesis networks are children of GENESIS_PRIME)
        const lineageEntry = this.lineageDAG.addNode({
            networkId: lifecycle.networkId,
            name: lifecycle.name,
            parentId: GENESIS_PRIME.id,
            generation: genome.generation,
            dna: genome.dna,
            createdAt: Date.now(),
        });

        // Update stats
        this.stats.totalNetworksSpawned++;
        this.stats.totalGenerations = Math.max(this.stats.totalGenerations, genome.generation);

        // Transition to embryo phase
        lifecycle.updateMetrics({ nodes: 1 });

        this.emit('network:spawned', {
            networkId: lifecycle.networkId,
            name: lifecycle.name,
            generation: genome.generation,
            lineage: genome.getLineageString(),
            merkleHash: lineageEntry.hash,
            isGenesis: true,
        });

        return {
            networkId: lifecycle.networkId,
            genome: genome.getProfile(),
            status: lifecycle.getStatus(),
            lineageProof: lineageEntry,
        };
    }

    /**
     * Reproduce - Create a child network from a mature parent
     */
    reproduce(parentNetworkId, childName, options = {}) {
        const parent = this.networks.get(parentNetworkId);
        if (!parent) {
            throw new Error(`Parent network not found: ${parentNetworkId}`);
        }

        const canReproduce = parent.lifecycle.canReproduce();
        if (!canReproduce.allowed) {
            throw new Error(`Cannot reproduce: ${canReproduce.reason}`);
        }

        // Deduct reproduction cost
        parent.lifecycle.metrics.creditsSpent += this.config.reproductionCost;

        // Generate mutations based on parent performance
        const suggestedMutations = this.evolutionEngine.suggestMutations(
            parent.lifecycle,
            options.environment || {}
        );

        // Merge with custom mutations
        const mutations = {
            traits: { ...suggestedMutations.traits, ...(options.mutations?.traits || {}) },
            behaviors: { ...suggestedMutations.behaviors, ...(options.mutations?.behaviors || {}) },
            capabilities: { ...suggestedMutations.capabilities, ...(options.mutations?.capabilities || {}) },
        };

        // Create child genome
        const childGenome = new NetworkGenome(parent.genome, mutations);

        // Create child lifecycle
        const childLifecycle = new NetworkLifecycle(childGenome, {
            name: childName || `${parent.lifecycle.name}-Child-${parent.lifecycle.children.size + 1}`,
            parentId: parentNetworkId,
        });

        // Create child synapse
        const childSynapse = new NetworkSynapse(childLifecycle);

        // Connect parent and child
        parent.synapse.connect(childLifecycle.networkId, childSynapse);
        childSynapse.connect(parentNetworkId, parent.synapse);

        // Register child with parent
        parent.lifecycle.children.set(childLifecycle.networkId, {
            id: childLifecycle.networkId,
            name: childLifecycle.name,
            genome: childGenome,
            bornAt: Date.now(),
        });

        parent.lifecycle.metrics.childrenSpawned++;

        // Store references
        this.networks.set(childLifecycle.networkId, {
            lifecycle: childLifecycle,
            genome: childGenome,
            synapse: childSynapse,
            bornAt: Date.now(),
        });
        this.genomes.set(childGenome.id, childGenome);

        // Record in cryptographic lineage DAG
        const lineageEntry = this.lineageDAG.addNode({
            networkId: childLifecycle.networkId,
            name: childLifecycle.name,
            parentId: parentNetworkId,
            generation: childGenome.generation,
            dna: childGenome.dna,
            mutations,
            createdAt: Date.now(),
        });

        // Update stats
        this.stats.totalNetworksSpawned++;
        this.stats.totalGenerations = Math.max(this.stats.totalGenerations, childGenome.generation);

        // Check if parent becomes most prolific
        if (!this.stats.mostProlificNetwork ||
            parent.lifecycle.children.size > this.networks.get(this.stats.mostProlificNetwork)?.lifecycle.children.size) {
            this.stats.mostProlificNetwork = parentNetworkId;
        }

        // Transition child to embryo
        childLifecycle.updateMetrics({ nodes: 1 });

        // Record evolution
        this.evolutionEngine.recordEvolution(childLifecycle.networkId, 'birth', {
            parentId: parentNetworkId,
            generation: childGenome.generation,
            mutations,
            merkleHash: lineageEntry.hash,
        });

        this.emit('network:reproduced', {
            parentId: parentNetworkId,
            childId: childLifecycle.networkId,
            childName: childLifecycle.name,
            generation: childGenome.generation,
            lineage: childGenome.getLineageString(),
            merkleHash: lineageEntry.hash,
        });

        return {
            networkId: childLifecycle.networkId,
            genome: childGenome.getProfile(),
            status: childLifecycle.getStatus(),
            parentId: parentNetworkId,
            lineageProof: lineageEntry,
        };
    }

    /**
     * Get network by ID
     */
    getNetwork(networkId) {
        const network = this.networks.get(networkId);
        if (!network) return null;

        return {
            networkId,
            status: network.lifecycle.getStatus(),
            genome: network.genome.getProfile(),
            synapse: network.synapse.getStatus(),
        };
    }

    /**
     * Get lineage tree for a network
     */
    getLineageTree(networkId) {
        const network = this.networks.get(networkId);
        if (!network) return null;

        const buildTree = (id) => {
            const net = this.networks.get(id);
            if (!net) return { id, name: 'Unknown', status: 'not-found' };

            return {
                id,
                name: net.lifecycle.name,
                generation: net.genome.generation,
                phase: net.lifecycle.phase,
                children: Array.from(net.lifecycle.children.keys()).map(buildTree),
            };
        };

        // Find root (oldest ancestor in our registry)
        let rootId = networkId;
        let current = network;
        while (current.lifecycle.parentId && this.networks.has(current.lifecycle.parentId)) {
            rootId = current.lifecycle.parentId;
            current = this.networks.get(rootId);
        }

        return {
            root: GENESIS_PRIME.id,
            tree: buildTree(rootId),
        };
    }

    /**
     * Get genesis statistics
     */
    getStats() {
        const phaseDistribution = {};
        const generationDistribution = {};

        for (const [id, network] of this.networks) {
            const phase = network.lifecycle.phase;
            const gen = network.genome.generation;

            phaseDistribution[phase] = (phaseDistribution[phase] || 0) + 1;
            generationDistribution[gen] = (generationDistribution[gen] || 0) + 1;
        }

        return {
            ...this.stats,
            activeNetworks: this.networks.size,
            phaseDistribution,
            generationDistribution,
            collectiveMemory: this.collectiveMemory.getStats(),
            evolution: this.evolutionEngine.getStats(),
        };
    }

    /**
     * Get Genesis Prime info
     */
    static getGenesisPrime() {
        return GENESIS_PRIME;
    }

    // ============================================
    // CRYPTOGRAPHIC LINEAGE VERIFICATION
    // ============================================

    /**
     * Verify the lineage of a network using Merkle proofs
     */
    verifyLineage(networkId) {
        return this.lineageDAG.verifyAncestry(networkId, GENESIS_PRIME.id);
    }

    /**
     * Get the complete ancestry path for a network
     */
    getAncestryPath(networkId) {
        return this.lineageDAG.getAncestryPath(networkId);
    }

    /**
     * Get the Merkle root of the lineage DAG
     */
    getLineageRoot() {
        return this.lineageDAG.getMerkleRoot();
    }

    /**
     * Get a verifiable lineage proof for external verification
     */
    getLineageProof(networkId) {
        const path = this.lineageDAG.getAncestryPath(networkId);
        const verified = this.lineageDAG.verifyAncestry(networkId, GENESIS_PRIME.id);

        return {
            networkId,
            ancestryPath: path,
            merkleRoot: this.lineageDAG.getMerkleRoot(),
            verified,
            genesisPrime: GENESIS_PRIME.id,
            signature: GENESIS_PRIME.signature,
        };
    }

    // ============================================
    // GOSSIP PROTOCOL FOR NETWORK DISCOVERY
    // ============================================

    /**
     * Create a gossip network for inter-network communication
     */
    createGossipNetwork(options = {}) {
        const gossip = new GossipProtocol({
            fanout: options.fanout || 3,
            gossipInterval: options.gossipInterval || 500,
            ...options,
        });

        // Propagate network discoveries
        gossip.on('message:received', (msg) => {
            if (msg.payload.type === 'network_discovery') {
                this.emit('network:discovered', msg.payload.network);
            }
        });

        return gossip;
    }

    /**
     * Broadcast network birth across the federation
     */
    broadcastBirth(networkId, gossipNetwork) {
        const network = this.networks.get(networkId);
        if (!network) return false;

        gossipNetwork.spreadRumor('network_discovery', {
            type: 'network_discovery',
            network: {
                id: networkId,
                name: network.lifecycle.name,
                generation: network.genome.generation,
                lineage: network.genome.getLineageString(),
                capabilities: network.genome.dna.capabilities,
            },
        });

        return true;
    }

    // ============================================
    // SWARM CONSENSUS FOR COLLECTIVE DECISIONS
    // ============================================

    /**
     * Create a consensus mechanism for network-wide decisions
     */
    createConsensus(options = {}) {
        return new SwarmConsensus({
            quorumThreshold: options.quorumThreshold || 0.67,
            consensusTimeout: options.consensusTimeout || 30000,
            ...options,
        });
    }

    /**
     * Propose an evolution mutation to all mature networks
     */
    async proposeEvolutionMutation(mutation, consensus) {
        // Get all mature networks as voters
        const voters = [];
        for (const [id, network] of this.networks) {
            if (network.lifecycle.phase === GenesisPhase.MATURE ||
                network.lifecycle.phase === GenesisPhase.ELDER ||
                network.lifecycle.phase === GenesisPhase.TRANSCENDENT) {
                voters.push(id);
            }
        }

        if (voters.length < 3) {
            return { success: false, reason: 'Insufficient mature networks for consensus' };
        }

        const proposal = consensus.createProposal(
            `evolution_mutation_${Date.now()}`,
            'evolution_mutation',
            mutation,
            voters
        );

        // Simulate voting based on network fitness
        for (const voterId of voters) {
            const network = this.networks.get(voterId);
            const fitness = this.evolutionEngine.calculateFitness(network.lifecycle);

            // Higher fitness networks are more likely to accept evolution
            const vote = fitness.overall > 0.5 && Math.random() < fitness.overall;

            consensus.vote(proposal.id, voterId, vote, {
                fitness: fitness.overall,
                generation: network.genome.generation,
            });
        }

        await new Promise(resolve => setTimeout(resolve, 100));
        return consensus.checkConsensus(proposal.id);
    }

    // ============================================
    // SELF-HEALING FOR NETWORK RESILIENCE
    // ============================================

    /**
     * Create a self-healing system for a network
     */
    createSelfHealing(networkId, options = {}) {
        const network = this.networks.get(networkId);
        if (!network) {
            throw new Error(`Network not found: ${networkId}`);
        }

        const healing = new SelfHealing(network.lifecycle, {
            isolationThreshold: options.isolationThreshold || 3,
            antifragileBoostFactor: options.antifragileBoostFactor || 1.1,
            ...options,
        });

        // Track healing events
        healing.on('antifragile:boost', (event) => {
            this.evolutionEngine.recordEvolution(networkId, 'antifragile_adaptation', {
                component: event.componentId,
                boostFactor: event.boostFactor,
            });
        });

        healing.on('recovery:succeeded', (event) => {
            this.collectiveMemory.storeOptimization(
                'recovery_strategy',
                { component: event.componentId, network: networkId },
                networkId,
                0.8
            );
        });

        return healing;
    }

    /**
     * Get ecosystem health across all networks
     */
    getEcosystemHealth() {
        let totalHealth = 0;
        let networkCount = 0;

        for (const [id, network] of this.networks) {
            const fitness = this.evolutionEngine.calculateFitness(network.lifecycle);
            totalHealth += fitness.overall;
            networkCount++;
        }

        const avgHealth = networkCount > 0 ? totalHealth / networkCount : 0;
        this.stats.ecosystemHealth = avgHealth;

        return {
            averageHealth: avgHealth,
            networkCount,
            generationSpread: this.stats.totalGenerations,
            lineageIntegrity: this.lineageDAG.getMerkleRoot() !== null,
            collectiveKnowledge: this.collectiveMemory.getStats(),
        };
    }

    /**
     * Trigger ecosystem-wide healing
     */
    healEcosystem() {
        const healed = [];

        for (const [id, network] of this.networks) {
            const fitness = this.evolutionEngine.calculateFitness(network.lifecycle);

            // Networks below health threshold get healing attention
            if (fitness.overall < 0.5) {
                const healing = this.createSelfHealing(id);

                // Simulate stress recovery
                healing.reportError(id, new Error('Low fitness detected'));

                healed.push({
                    networkId: id,
                    priorFitness: fitness.overall,
                    healingInitiated: true,
                });
            }
        }

        this.emit('ecosystem:healing', { networksHealed: healed.length, details: healed });

        return healed;
    }
}

// ============================================
// EXPORTS
// ============================================

export {
    GENESIS_PRIME,
    GenesisPhase,
    PHASE_THRESHOLDS,
    REPRODUCTION_COST,
};

export default NetworkGenesis;

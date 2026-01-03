/**
 * Edge-Net Core Invariants
 *
 * Cogito, Creo, Codex — The system thinks collectively, creates through
 * contribution, and codifies trust in cryptographic proof.
 *
 * These invariants are NOT configurable. They define what Edge-net IS.
 *
 * @module @ruvector/edge-net/core-invariants
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';

// ============================================
// INVARIANT 1: ECONOMIC SETTLEMENT ISOLATION
// ============================================

/**
 * EconomicBoundary - Enforces that plugins can NEVER execute settlement
 *
 * Core operations (immutable):
 * - rUv minting
 * - rUv burning
 * - Credit settlement
 * - Balance enforcement
 * - Slashing execution
 *
 * Plugin operations (read-only):
 * - Pricing suggestions
 * - Reputation scoring
 * - Economic modeling
 * - Auction mechanisms
 */
export class EconomicBoundary {
    constructor(creditSystem) {
        this.creditSystem = creditSystem;
        this.settlementLock = false;
        this.coreOperationCounts = {
            mint: 0,
            burn: 0,
            settle: 0,
            slash: 0,
        };

        // Seal the boundary - plugins get a read-only proxy
        this._sealed = false;
    }

    /**
     * Get a read-only view for plugins
     * Plugins can observe but NEVER modify
     */
    getPluginView() {
        return Object.freeze({
            // Read-only accessors
            getBalance: (nodeId) => this.creditSystem.getBalance(nodeId),
            getTransactionHistory: (nodeId, limit) =>
                this.creditSystem.getTransactionHistory(nodeId, limit),
            getSummary: () => {
                const summary = this.creditSystem.getSummary();
                // Remove sensitive internal state
                delete summary.recentTransactions;
                return Object.freeze(summary);
            },

            // Event subscription (read-only)
            on: (event, handler) => {
                // Only allow observation events
                const allowedEvents = [
                    'credits-earned',
                    'credits-spent',
                    'insufficient-funds',
                ];
                if (allowedEvents.includes(event)) {
                    this.creditSystem.on(event, (data) => {
                        // Clone data to prevent mutation
                        handler(JSON.parse(JSON.stringify(data)));
                    });
                }
            },

            // Explicit denial methods (throw if called)
            mint: () => {
                throw new Error('INVARIANT VIOLATION: Plugins cannot mint credits');
            },
            burn: () => {
                throw new Error('INVARIANT VIOLATION: Plugins cannot burn credits');
            },
            settle: () => {
                throw new Error('INVARIANT VIOLATION: Plugins cannot settle credits');
            },
            transfer: () => {
                throw new Error('INVARIANT VIOLATION: Plugins cannot transfer credits');
            },
        });
    }

    /**
     * Core-only: Mint credits (bootstrap, rewards)
     * @private - Only callable from core system
     */
    _coreMint(nodeId, amount, reason, proofOfWork) {
        if (!proofOfWork) {
            throw new Error('INVARIANT: Minting requires proof of work');
        }

        this.coreOperationCounts.mint++;
        return this.creditSystem.ledger.credit(amount, JSON.stringify({
            type: 'core_mint',
            reason,
            proofHash: createHash('sha256').update(JSON.stringify(proofOfWork)).digest('hex'),
            timestamp: Date.now(),
        }));
    }

    /**
     * Core-only: Burn credits (slashing, expiry)
     * @private - Only callable from core system
     */
    _coreBurn(nodeId, amount, reason, evidence) {
        this.coreOperationCounts.burn++;
        return this.creditSystem.ledger.debit(amount, JSON.stringify({
            type: 'core_burn',
            reason,
            evidenceHash: evidence ? createHash('sha256').update(JSON.stringify(evidence)).digest('hex') : null,
            timestamp: Date.now(),
        }));
    }

    /**
     * Core-only: Execute slashing
     * @private - Only callable from core system
     */
    _coreSlash(nodeId, amount, violation, evidence) {
        this.coreOperationCounts.slash++;

        // Record slashing event
        const slashRecord = {
            type: 'slash',
            nodeId,
            amount,
            violation,
            evidenceHash: createHash('sha256').update(JSON.stringify(evidence)).digest('hex'),
            timestamp: Date.now(),
        };

        // Execute burn
        this._coreBurn(nodeId, amount, `Slashed: ${violation}`, evidence);

        return slashRecord;
    }
}

// ============================================
// INVARIANT 2: IDENTITY ANTI-SYBIL MEASURES
// ============================================

/**
 * IdentityFriction - Prevents Sybil attacks on the plugin marketplace
 *
 * Mechanisms:
 * - Delayed activation (24h window)
 * - Reputation warm-up (0.1 → 1.0 over 100 tasks)
 * - Stake requirement for priority
 * - Witness diversity requirement
 */
export class IdentityFriction extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            // Delayed activation
            activationDelayMs: options.activationDelayMs ?? 24 * 60 * 60 * 1000, // 24 hours

            // Reputation warm-up
            initialReputation: options.initialReputation ?? 0.1,
            maxReputation: options.maxReputation ?? 1.0,
            warmupTasks: options.warmupTasks ?? 100,

            // Stake requirements
            stakeForPriority: options.stakeForPriority ?? 10, // rUv
            stakeSlashPercent: options.stakeSlashPercent ?? 0.5, // 50%

            // Witness diversity
            minWitnessDiversity: options.minWitnessDiversity ?? 3,
            witnessDiversityWindowMs: options.witnessDiversityWindowMs ?? 7 * 24 * 60 * 60 * 1000, // 7 days
        };

        // Identity registry
        this.identities = new Map(); // nodeId -> IdentityRecord
        this._activationTimers = new Map(); // nodeId -> timerId (for cleanup)
    }

    /**
     * Register a new identity
     */
    registerIdentity(nodeId, publicKey) {
        if (this.identities.has(nodeId)) {
            throw new Error('Identity already registered');
        }

        const record = {
            nodeId,
            publicKeyHash: createHash('sha256').update(publicKey).digest('hex'),
            createdAt: Date.now(),
            activatedAt: null, // Set after delay
            reputation: this.config.initialReputation,
            tasksCompleted: 0,
            stake: 0,
            witnesses: [], // { nodeId, createdAt, attestedAt }
            status: 'pending', // pending | active | suspended | slashed
        };

        this.identities.set(nodeId, record);

        // Schedule activation with tracked timer
        const timerId = setTimeout(() => {
            this._activationTimers.delete(nodeId);
            this._activateIdentity(nodeId);
        }, this.config.activationDelayMs);
        this._activationTimers.set(nodeId, timerId);

        this.emit('identity:registered', { nodeId, activatesAt: Date.now() + this.config.activationDelayMs });

        return record;
    }

    /**
     * Activate identity after delay
     * @private
     */
    _activateIdentity(nodeId) {
        const record = this.identities.get(nodeId);
        if (!record) return;

        if (record.status === 'pending') {
            record.activatedAt = Date.now();
            record.status = 'active';
            this.emit('identity:activated', { nodeId });
        }
    }

    /**
     * Check if identity can execute tasks
     */
    canExecuteTasks(nodeId) {
        const record = this.identities.get(nodeId);
        if (!record) return { allowed: false, reason: 'Unknown identity' };

        if (record.status === 'pending') {
            const remainingMs = (record.createdAt + this.config.activationDelayMs) - Date.now();
            return {
                allowed: false,
                reason: 'Pending activation',
                remainingMs: Math.max(0, remainingMs),
            };
        }

        if (record.status === 'suspended' || record.status === 'slashed') {
            return { allowed: false, reason: `Identity ${record.status}` };
        }

        return { allowed: true, reputation: record.reputation };
    }

    /**
     * Record task completion and update reputation
     */
    recordTaskCompletion(nodeId, taskId, success, witnesses = []) {
        const record = this.identities.get(nodeId);
        if (!record) return null;

        if (success) {
            record.tasksCompleted++;

            // Warm-up reputation curve
            const progress = Math.min(record.tasksCompleted / this.config.warmupTasks, 1);
            const reputationRange = this.config.maxReputation - this.config.initialReputation;
            record.reputation = this.config.initialReputation + (reputationRange * progress);

            // Record witnesses
            for (const witness of witnesses) {
                const witnessRecord = this.identities.get(witness.nodeId);
                if (witnessRecord && this._isWitnessDiverse(record, witnessRecord)) {
                    record.witnesses.push({
                        nodeId: witness.nodeId,
                        createdAt: witnessRecord.createdAt,
                        attestedAt: Date.now(),
                    });
                }
            }
        }

        this.emit('task:recorded', {
            nodeId,
            taskId,
            success,
            reputation: record.reputation,
            tasksCompleted: record.tasksCompleted,
        });

        return record;
    }

    /**
     * Check witness diversity (different creation times)
     * @private
     */
    _isWitnessDiverse(identity, witness) {
        // Witnesses must be created at different times (not Sybil cluster)
        const timeDiff = Math.abs(identity.createdAt - witness.createdAt);
        return timeDiff > this.config.witnessDiversityWindowMs;
    }

    /**
     * Stake for priority access
     */
    stake(nodeId, amount) {
        const record = this.identities.get(nodeId);
        if (!record) throw new Error('Unknown identity');

        record.stake += amount;
        this.emit('identity:staked', { nodeId, amount, totalStake: record.stake });

        return record;
    }

    /**
     * Slash stake for violations
     */
    slashStake(nodeId, violation, evidence) {
        const record = this.identities.get(nodeId);
        if (!record) throw new Error('Unknown identity');

        const slashAmount = Math.floor(record.stake * this.config.stakeSlashPercent);
        record.stake -= slashAmount;

        // Reduce reputation
        record.reputation = Math.max(this.config.initialReputation, record.reputation * 0.5);

        if (record.stake <= 0 && record.reputation <= this.config.initialReputation) {
            record.status = 'slashed';
        }

        this.emit('identity:slashed', {
            nodeId,
            violation,
            slashAmount,
            remainingStake: record.stake,
            reputation: record.reputation,
            status: record.status,
        });

        return { slashAmount, record };
    }

    /**
     * Get identity status
     */
    getIdentity(nodeId) {
        const record = this.identities.get(nodeId);
        if (!record) return null;

        return {
            nodeId: record.nodeId,
            status: record.status,
            reputation: record.reputation,
            tasksCompleted: record.tasksCompleted,
            stake: record.stake,
            witnessCount: record.witnesses.length,
            createdAt: record.createdAt,
            activatedAt: record.activatedAt,
        };
    }

    /**
     * Check if identity has priority (staked)
     */
    hasPriority(nodeId) {
        const record = this.identities.get(nodeId);
        if (!record) return false;
        return record.stake >= this.config.stakeForPriority;
    }

    /**
     * Clean up all timers and resources
     */
    destroy() {
        // Clear all activation timers
        for (const timerId of this._activationTimers.values()) {
            clearTimeout(timerId);
        }
        this._activationTimers.clear();

        // Clear identity data
        this.identities.clear();

        this.removeAllListeners();
    }
}

// ============================================
// INVARIANT 3: VERIFIABLE WORK
// ============================================

/**
 * WorkVerifier - Ensures "Verifiable work or no reward"
 *
 * All credit issuance requires cryptographic proof of work completion.
 * Unverifiable claims are rejected, not trusted.
 */
export class WorkVerifier extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            // Redundancy for verification
            redundancyPercent: options.redundancyPercent ?? 0.05, // 5% of tasks
            redundancyThreshold: options.redundancyThreshold ?? 2, // 2 matching results

            // Challenger system
            challengeWindowMs: options.challengeWindowMs ?? 60 * 60 * 1000, // 1 hour
            challengeStake: options.challengeStake ?? 5, // rUv to challenge
            challengeReward: options.challengeReward ?? 10, // rUv if challenge succeeds

            // Result hashing
            hashAlgorithm: 'sha256',
        };

        // Pending verifications
        this.pendingWork = new Map(); // taskId -> WorkRecord
        this.challenges = new Map(); // taskId -> Challenge[]
    }

    /**
     * Submit work for verification
     */
    submitWork(taskId, nodeId, result, executionProof) {
        const resultHash = this._hashResult(result);
        const proofHash = createHash(this.config.hashAlgorithm)
            .update(JSON.stringify(executionProof))
            .digest('hex');

        const workRecord = {
            taskId,
            nodeId,
            resultHash,
            proofHash,
            submittedAt: Date.now(),
            challengeDeadline: Date.now() + this.config.challengeWindowMs,
            status: 'pending', // pending | verified | challenged | rejected
            redundantResults: [],
        };

        this.pendingWork.set(taskId, workRecord);

        // Check if this should be redundantly executed
        if (Math.random() < this.config.redundancyPercent) {
            workRecord.requiresRedundancy = true;
        }

        this.emit('work:submitted', { taskId, nodeId, resultHash });

        return workRecord;
    }

    /**
     * Submit redundant execution result
     */
    submitRedundantResult(taskId, nodeId, result) {
        const workRecord = this.pendingWork.get(taskId);
        if (!workRecord) throw new Error('Unknown task');

        const resultHash = this._hashResult(result);

        workRecord.redundantResults.push({
            nodeId,
            resultHash,
            submittedAt: Date.now(),
        });

        // Check for consensus
        const matchingResults = workRecord.redundantResults.filter(
            r => r.resultHash === workRecord.resultHash
        );

        if (matchingResults.length >= this.config.redundancyThreshold) {
            workRecord.status = 'verified';
            this.emit('work:verified', { taskId, method: 'redundancy' });
        }

        return workRecord;
    }

    /**
     * Challenge a work result
     */
    challenge(taskId, challengerNodeId, stake, evidence) {
        const workRecord = this.pendingWork.get(taskId);
        if (!workRecord) throw new Error('Unknown task');

        if (Date.now() > workRecord.challengeDeadline) {
            throw new Error('Challenge window closed');
        }

        if (stake < this.config.challengeStake) {
            throw new Error(`Insufficient stake: ${stake} < ${this.config.challengeStake}`);
        }

        const challenge = {
            challengerId: challengerNodeId,
            stake,
            evidenceHash: createHash(this.config.hashAlgorithm)
                .update(JSON.stringify(evidence))
                .digest('hex'),
            submittedAt: Date.now(),
            status: 'pending', // pending | accepted | rejected
        };

        if (!this.challenges.has(taskId)) {
            this.challenges.set(taskId, []);
        }
        this.challenges.get(taskId).push(challenge);

        workRecord.status = 'challenged';

        this.emit('work:challenged', { taskId, challengerId: challengerNodeId });

        return challenge;
    }

    /**
     * Resolve a challenge (requires arbitration)
     */
    resolveChallenge(taskId, challengeIndex, isValid, arbitrationProof) {
        const workRecord = this.pendingWork.get(taskId);
        const challenges = this.challenges.get(taskId);

        if (!workRecord || !challenges || !challenges[challengeIndex]) {
            throw new Error('Challenge not found');
        }

        const challenge = challenges[challengeIndex];

        if (isValid) {
            // Challenge succeeded - work was invalid
            challenge.status = 'accepted';
            workRecord.status = 'rejected';

            this.emit('challenge:accepted', {
                taskId,
                challengerId: challenge.challengerId,
                reward: this.config.challengeReward,
            });

            return {
                challengerReward: this.config.challengeReward + challenge.stake,
                workerSlash: true,
            };
        } else {
            // Challenge failed - work was valid
            challenge.status = 'rejected';
            workRecord.status = 'verified';

            this.emit('challenge:rejected', {
                taskId,
                challengerId: challenge.challengerId,
                stakeLost: challenge.stake,
            });

            return {
                challengerLoss: challenge.stake,
                workerReward: challenge.stake * 0.5, // Half of stake goes to worker
            };
        }
    }

    /**
     * Finalize work after challenge window
     */
    finalizeWork(taskId) {
        const workRecord = this.pendingWork.get(taskId);
        if (!workRecord) throw new Error('Unknown task');

        if (Date.now() < workRecord.challengeDeadline) {
            throw new Error('Challenge window still open');
        }

        if (workRecord.status === 'pending') {
            // No challenges and redundancy passed (if required)
            if (workRecord.requiresRedundancy) {
                const matchingResults = workRecord.redundantResults.filter(
                    r => r.resultHash === workRecord.resultHash
                );
                if (matchingResults.length < this.config.redundancyThreshold) {
                    workRecord.status = 'rejected';
                    this.emit('work:rejected', { taskId, reason: 'Insufficient redundancy' });
                    return workRecord;
                }
            }

            workRecord.status = 'verified';
            this.emit('work:finalized', { taskId });
        }

        return workRecord;
    }

    /**
     * Check if work is verified
     */
    isVerified(taskId) {
        const workRecord = this.pendingWork.get(taskId);
        return workRecord?.status === 'verified';
    }

    /**
     * Hash result deterministically
     * @private
     */
    _hashResult(result) {
        // Canonical JSON serialization with deep key sorting
        const sortKeys = (obj) => {
            if (obj === null || typeof obj !== 'object') return obj;
            if (Array.isArray(obj)) return obj.map(sortKeys);
            return Object.keys(obj).sort().reduce((acc, key) => {
                acc[key] = sortKeys(obj[key]);
                return acc;
            }, {});
        };
        const canonical = JSON.stringify(sortKeys(result));
        return createHash(this.config.hashAlgorithm).update(canonical).digest('hex');
    }

    /**
     * Clean up old work records to prevent memory leak
     */
    pruneOldRecords(maxAgeMs = 24 * 60 * 60 * 1000) {
        const cutoff = Date.now() - maxAgeMs;
        let pruned = 0;

        for (const [taskId, record] of this.pendingWork) {
            if (record.submittedAt < cutoff && record.status !== 'pending') {
                this.pendingWork.delete(taskId);
                this.challenges.delete(taskId);
                pruned++;
            }
        }

        return pruned;
    }
}

// ============================================
// INVARIANT 4: DEGRADATION OVER HALT
// ============================================

/**
 * DegradationController - Ensures system never halts
 *
 * The system degrades gracefully under load, attack, or partial failure.
 * Consistency is sacrificed before availability.
 */
export class DegradationController extends EventEmitter {
    constructor(options = {}) {
        super();

        this.config = {
            // Load thresholds
            warningLoadPercent: options.warningLoadPercent ?? 70,
            criticalLoadPercent: options.criticalLoadPercent ?? 90,
            maxLoadPercent: options.maxLoadPercent ?? 100,

            // Degradation levels
            levels: ['normal', 'elevated', 'degraded', 'emergency'],

            // Auto-recovery
            recoveryCheckIntervalMs: options.recoveryCheckIntervalMs ?? 30000,

            // Memory management
            maxHistorySize: options.maxHistorySize ?? 1000,
        };

        this.currentLevel = 'normal';
        // Protected metrics keys - prevents prototype pollution
        this._allowedMetricKeys = new Set(['cpuLoad', 'memoryUsage', 'pendingTasks', 'errorRate']);
        this.metrics = {
            cpuLoad: 0,
            memoryUsage: 0,
            pendingTasks: 0,
            errorRate: 0,
            lastUpdated: Date.now(),
        };

        this.degradationHistory = [];
    }

    /**
     * Update system metrics
     * Protected against prototype pollution - only allowed keys are updated
     */
    updateMetrics(metrics) {
        // Only copy allowed metric keys to prevent prototype pollution
        for (const key of this._allowedMetricKeys) {
            if (Object.hasOwn(metrics, key) && typeof metrics[key] === 'number') {
                this.metrics[key] = metrics[key];
            }
        }
        this.metrics.lastUpdated = Date.now();

        this._evaluateDegradation();
    }

    /**
     * Evaluate and apply degradation level
     * @private
     */
    _evaluateDegradation() {
        const load = this._calculateOverallLoad();
        const previousLevel = this.currentLevel;

        if (load >= this.config.maxLoadPercent) {
            this.currentLevel = 'emergency';
        } else if (load >= this.config.criticalLoadPercent) {
            this.currentLevel = 'degraded';
        } else if (load >= this.config.warningLoadPercent) {
            this.currentLevel = 'elevated';
        } else {
            this.currentLevel = 'normal';
        }

        if (this.currentLevel !== previousLevel) {
            this.degradationHistory.push({
                from: previousLevel,
                to: this.currentLevel,
                load,
                timestamp: Date.now(),
            });

            // Prune history to prevent memory leak
            if (this.degradationHistory.length > this.config.maxHistorySize) {
                this.degradationHistory.splice(0, this.degradationHistory.length - this.config.maxHistorySize);
            }

            this.emit('level:changed', {
                from: previousLevel,
                to: this.currentLevel,
                load,
            });
        }
    }

    /**
     * Calculate overall load
     * @private
     */
    _calculateOverallLoad() {
        return Math.max(
            this.metrics.cpuLoad || 0,
            this.metrics.memoryUsage || 0,
            (this.metrics.errorRate || 0) * 100
        );
    }

    /**
     * Get current degradation policy
     */
    getPolicy() {
        const policies = {
            normal: {
                acceptNewTasks: true,
                pluginsEnabled: true,
                redundancyEnabled: true,
                maxConcurrency: Infinity,
                taskTimeout: 30000,
            },
            elevated: {
                acceptNewTasks: true,
                pluginsEnabled: true,
                redundancyEnabled: false, // Disable redundancy to reduce load
                maxConcurrency: 100,
                taskTimeout: 20000,
            },
            degraded: {
                acceptNewTasks: true,
                pluginsEnabled: false, // Disable non-core plugins
                redundancyEnabled: false,
                maxConcurrency: 50,
                taskTimeout: 10000,
            },
            emergency: {
                acceptNewTasks: false, // Shed load
                pluginsEnabled: false,
                redundancyEnabled: false,
                maxConcurrency: 10,
                taskTimeout: 5000,
            },
        };

        return {
            level: this.currentLevel,
            ...policies[this.currentLevel],
        };
    }

    /**
     * Check if action is allowed under current policy
     */
    isAllowed(action) {
        const policy = this.getPolicy();

        switch (action) {
            case 'accept_task':
                return policy.acceptNewTasks;
            case 'load_plugin':
                return policy.pluginsEnabled;
            case 'redundant_execution':
                return policy.redundancyEnabled;
            default:
                return true;
        }
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            level: this.currentLevel,
            metrics: { ...this.metrics },
            policy: this.getPolicy(),
            history: this.degradationHistory.slice(-10),
        };
    }
}

// ============================================
// CORE SYSTEM ORCHESTRATOR
// ============================================

/**
 * CoreInvariants - Orchestrates all invariant enforcement
 *
 * Cogito, Creo, Codex
 */
export class CoreInvariants extends EventEmitter {
    constructor(creditSystem, options = {}) {
        super();

        // Initialize all invariant enforcers
        this.economicBoundary = new EconomicBoundary(creditSystem);
        this.identityFriction = new IdentityFriction(options.identity);
        this.workVerifier = new WorkVerifier(options.verification);
        this.degradationController = new DegradationController(options.degradation);

        // Cross-wire events
        this._wireEvents();

        console.log('[CoreInvariants] Cogito, Creo, Codex — Invariants initialized');
    }

    /**
     * Wire cross-component events
     * @private
     */
    _wireEvents() {
        // Slash identity when work is rejected
        this.workVerifier.on('work:rejected', ({ taskId }) => {
            const work = this.workVerifier.pendingWork.get(taskId);
            if (work) {
                this.identityFriction.slashStake(
                    work.nodeId,
                    'work_rejected',
                    { taskId }
                );
            }
        });

        // Emit unified events
        this.identityFriction.on('identity:slashed', (data) => {
            this.emit('invariant:slashing', data);
        });

        this.degradationController.on('level:changed', (data) => {
            this.emit('invariant:degradation', data);
        });
    }

    /**
     * Get plugin-safe view of economic system
     */
    getPluginEconomicView() {
        return this.economicBoundary.getPluginView();
    }

    /**
     * Register new identity with friction
     */
    registerIdentity(nodeId, publicKey) {
        return this.identityFriction.registerIdentity(nodeId, publicKey);
    }

    /**
     * Submit and verify work
     */
    submitWork(taskId, nodeId, result, proof) {
        // Check identity can execute
        const canExecute = this.identityFriction.canExecuteTasks(nodeId);
        if (!canExecute.allowed) {
            throw new Error(`Identity cannot execute: ${canExecute.reason}`);
        }

        // Check degradation allows
        if (!this.degradationController.isAllowed('accept_task')) {
            throw new Error('System in emergency mode, shedding load');
        }

        return this.workVerifier.submitWork(taskId, nodeId, result, proof);
    }

    /**
     * Update system load metrics
     */
    updateMetrics(metrics) {
        this.degradationController.updateMetrics(metrics);
    }

    /**
     * Get comprehensive status
     */
    getStatus() {
        return {
            degradation: this.degradationController.getStatus(),
            pendingVerifications: this.workVerifier.pendingWork.size,
            activeChallenges: this.workVerifier.challenges.size,
            identityCount: this.identityFriction.identities.size,
        };
    }
}

export default CoreInvariants;

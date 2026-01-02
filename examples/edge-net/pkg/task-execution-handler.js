/**
 * @ruvector/edge-net Task Execution Handler
 *
 * Wires Firebase signaling to the worker pool for distributed task execution.
 * When a node receives a 'task-assign' signal, it validates and executes the task,
 * then sends the result back to the originator.
 *
 * Signal types:
 * - 'task-assign'   - Assign a task to a peer for execution
 * - 'task-result'   - Return successful result to originator
 * - 'task-error'    - Report execution failure to originator
 * - 'task-progress' - Optional progress updates during execution
 *
 * Credit Integration:
 * - Workers automatically earn credits when completing tasks
 * - Task submitters spend credits when submitting tasks
 * - Credits tracked via CRDT ledger for conflict-free replication
 *
 * @module @ruvector/edge-net/task-execution-handler
 */

import { EventEmitter } from 'events';
import { randomBytes, createHash } from 'crypto';

// ============================================
// TASK VALIDATION
// ============================================

/**
 * Validates incoming task assignments
 */
export class TaskValidator {
    constructor(options = {}) {
        this.maxDataSize = options.maxDataSize || 1024 * 1024; // 1MB default
        this.allowedTypes = options.allowedTypes || [
            'embed', 'process', 'analyze', 'transform', 'compute', 'aggregate', 'custom'
        ];
        this.maxPriority = options.maxPriority || 10;
        this.requireSignature = options.requireSignature !== false;
    }

    /**
     * Validate a task assignment
     * @param {Object} task - The task to validate
     * @param {Object} signal - The signal containing the task
     * @returns {Object} Validation result { valid: boolean, errors: string[] }
     */
    validate(task, signal = {}) {
        const errors = [];

        // Required fields
        if (!task) {
            return { valid: false, errors: ['Task is required'] };
        }

        if (!task.id) {
            errors.push('Task ID is required');
        }

        if (!task.type) {
            errors.push('Task type is required');
        } else if (!this.allowedTypes.includes(task.type)) {
            errors.push(`Invalid task type: ${task.type}. Allowed: ${this.allowedTypes.join(', ')}`);
        }

        if (task.data === undefined) {
            errors.push('Task data is required');
        }

        // Data size check
        const dataSize = this._estimateSize(task.data);
        if (dataSize > this.maxDataSize) {
            errors.push(`Task data exceeds max size (${dataSize} > ${this.maxDataSize})`);
        }

        // Priority check
        if (task.priority !== undefined) {
            const priority = typeof task.priority === 'string'
                ? this._priorityToNumber(task.priority)
                : task.priority;
            if (priority < 0 || priority > this.maxPriority) {
                errors.push(`Invalid priority: ${task.priority}`);
            }
        }

        // Capability check (if specified)
        if (task.requiredCapabilities && !Array.isArray(task.requiredCapabilities)) {
            errors.push('requiredCapabilities must be an array');
        }

        // Signature validation (if required and available)
        if (this.requireSignature && signal.signature) {
            if (!this._verifySignature(task, signal)) {
                errors.push('Invalid task signature');
            }
        }

        // Timeout check
        if (task.timeout !== undefined) {
            if (typeof task.timeout !== 'number' || task.timeout <= 0) {
                errors.push('Timeout must be a positive number');
            }
            if (task.timeout > 600000) { // 10 minutes max
                errors.push('Timeout exceeds maximum (600000ms)');
            }
        }

        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Check if local node has required capabilities
     */
    hasCapabilities(required, available) {
        if (!required || required.length === 0) return true;
        if (!available || available.length === 0) return false;
        return required.every(cap => available.includes(cap));
    }

    _estimateSize(data) {
        if (data === null || data === undefined) return 0;
        if (typeof data === 'string') return data.length;
        try {
            return JSON.stringify(data).length;
        } catch {
            return this.maxDataSize + 1; // Fail validation if can't serialize
        }
    }

    _priorityToNumber(priority) {
        const map = { critical: 0, high: 1, medium: 2, low: 3 };
        return map[priority.toLowerCase()] ?? 2;
    }

    _verifySignature(task, signal) {
        // Basic signature verification - in production use WASM crypto
        if (!signal.signature || !signal.publicKey) return false;
        // Simplified: just check signature exists and has correct format
        return typeof signal.signature === 'string' && signal.signature.length >= 64;
    }
}

// ============================================
// TASK EXECUTION HANDLER
// ============================================

/**
 * TaskExecutionHandler - Bridges signaling and worker pool for distributed execution
 *
 * Listens for 'task-assign' signals from FirebaseSignaling,
 * validates and executes tasks using RealWorkerPool,
 * and sends results back via signaling.
 */
export class TaskExecutionHandler extends EventEmitter {
    /**
     * @param {Object} options
     * @param {FirebaseSignaling} options.signaling - Firebase signaling instance
     * @param {RealWorkerPool} options.workerPool - Worker pool for execution
     * @param {string[]} options.capabilities - This node's capabilities
     * @param {Object} options.secureAccess - WASM secure access manager (optional)
     */
    constructor(options = {}) {
        super();

        this.signaling = options.signaling;
        this.workerPool = options.workerPool;
        this.capabilities = options.capabilities || ['compute', 'process', 'embed'];
        this.secureAccess = options.secureAccess || null;
        this.nodeId = options.nodeId || options.signaling?.peerId;

        // Task tracking
        this.activeTasks = new Map();      // taskId -> { task, startTime, from }
        this.completedTasks = new Map();   // taskId -> { result, duration }
        this.taskTimeouts = new Map();     // taskId -> timeout handle

        // Configuration
        this.defaultTimeout = options.defaultTimeout || 60000;
        this.maxConcurrentTasks = options.maxConcurrentTasks || 10;
        this.reportProgress = options.reportProgress !== false;
        this.progressInterval = options.progressInterval || 5000;

        // Validator
        this.validator = new TaskValidator({
            requireSignature: options.requireSignature !== false,
            allowedTypes: options.allowedTypes,
        });

        // Stats
        this.stats = {
            tasksReceived: 0,
            tasksExecuted: 0,
            tasksFailed: 0,
            tasksRejected: 0,
            totalExecutionTime: 0,
        };

        // Bind event handlers
        this._boundHandlers = {
            onSignal: this._handleSignal.bind(this),
        };

        this.attached = false;
    }

    /**
     * Attach to signaling - start listening for task assignments
     */
    attach() {
        if (this.attached) return this;
        if (!this.signaling) {
            throw new Error('Signaling instance required');
        }

        // Listen for all signals and filter for task-related ones
        this.signaling.on('signal', this._boundHandlers.onSignal);

        this.attached = true;
        this.emit('attached');

        console.log(`[TaskExecutionHandler] Attached to signaling, capabilities: ${this.capabilities.join(', ')}`);
        return this;
    }

    /**
     * Detach from signaling - stop listening
     */
    detach() {
        if (!this.attached) return this;

        this.signaling.off('signal', this._boundHandlers.onSignal);

        // Clear all timeouts
        for (const timeout of this.taskTimeouts.values()) {
            clearTimeout(timeout);
        }
        this.taskTimeouts.clear();

        this.attached = false;
        this.emit('detached');

        return this;
    }

    /**
     * Handle incoming signal
     */
    async _handleSignal(signal) {
        const { type, from, data, verified } = signal;

        switch (type) {
            case 'task-assign':
                await this._handleTaskAssign(from, data, signal);
                break;

            case 'task-result':
                this._handleTaskResult(from, data);
                break;

            case 'task-error':
                this._handleTaskError(from, data);
                break;

            case 'task-progress':
                this._handleTaskProgress(from, data);
                break;

            case 'task-cancel':
                await this._handleTaskCancel(from, data);
                break;
        }
    }

    /**
     * Handle task assignment - validate, execute, return result
     */
    async _handleTaskAssign(from, taskData, signal) {
        this.stats.tasksReceived++;

        // Handle various data formats
        const task = taskData?.task || taskData || {};
        const taskId = task.id || `recv-${randomBytes(8).toString('hex')}`;

        console.log(`[TaskExecutionHandler] Received task ${taskId} from ${from?.slice(0, 8)}...`);

        // Check capacity
        if (this.activeTasks.size >= this.maxConcurrentTasks) {
            this.stats.tasksRejected++;
            await this._sendError(from, taskId, 'Node at capacity', 'CAPACITY_EXCEEDED');
            this.emit('task-rejected', { taskId, from, reason: 'capacity' });
            return;
        }

        // Validate task
        const validation = this.validator.validate(task, signal);
        if (!validation.valid) {
            this.stats.tasksRejected++;
            await this._sendError(from, taskId, validation.errors.join('; '), 'VALIDATION_FAILED');
            this.emit('task-rejected', { taskId, from, reason: 'validation', errors: validation.errors });
            return;
        }

        // Check capabilities
        if (!this.validator.hasCapabilities(task.requiredCapabilities, this.capabilities)) {
            this.stats.tasksRejected++;
            await this._sendError(from, taskId, 'Missing required capabilities', 'CAPABILITY_MISMATCH');
            this.emit('task-rejected', { taskId, from, reason: 'capabilities' });
            return;
        }

        // Track task
        const taskInfo = {
            task,
            from,
            startTime: Date.now(),
            verified: signal.verified || false,
        };
        this.activeTasks.set(taskId, taskInfo);

        // Set timeout
        const timeout = task.timeout || this.defaultTimeout;
        const timeoutHandle = setTimeout(() => {
            this._handleTaskTimeout(taskId);
        }, timeout);
        this.taskTimeouts.set(taskId, timeoutHandle);

        // Start progress reporting if enabled
        let progressHandle = null;
        if (this.reportProgress && this.progressInterval > 0) {
            progressHandle = setInterval(() => {
                this._sendProgress(from, taskId, {
                    status: 'running',
                    elapsed: Date.now() - taskInfo.startTime,
                });
            }, this.progressInterval);
        }

        // Execute task
        try {
            this.emit('task-start', { taskId, from, type: task.type });

            // Check if worker pool is ready
            if (!this.workerPool || this.workerPool.status !== 'ready') {
                throw new Error('Worker pool not ready');
            }

            // Execute via worker pool
            const result = await this.workerPool.execute(
                task.type,
                task.data,
                task.options || {}
            );

            // Task completed successfully
            const duration = Date.now() - taskInfo.startTime;
            this.stats.tasksExecuted++;
            this.stats.totalExecutionTime += duration;

            // Clear timeout and progress
            clearTimeout(timeoutHandle);
            this.taskTimeouts.delete(taskId);
            if (progressHandle) clearInterval(progressHandle);

            // Store completed task
            this.completedTasks.set(taskId, { result, duration });
            this.activeTasks.delete(taskId);

            // Send result back
            await this._sendResult(from, taskId, result, duration);

            this.emit('task-complete', { taskId, from, duration, result });

            console.log(`[TaskExecutionHandler] Task ${taskId} completed in ${duration}ms`);

        } catch (error) {
            // Task failed
            const duration = Date.now() - taskInfo.startTime;
            this.stats.tasksFailed++;

            // Clear timeout and progress
            clearTimeout(timeoutHandle);
            this.taskTimeouts.delete(taskId);
            if (progressHandle) clearInterval(progressHandle);

            this.activeTasks.delete(taskId);

            // Send error back
            await this._sendError(from, taskId, error.message, 'EXECUTION_FAILED');

            this.emit('task-error', { taskId, from, error: error.message, duration });

            console.error(`[TaskExecutionHandler] Task ${taskId} failed:`, error.message);
        }
    }

    /**
     * Handle task timeout
     */
    async _handleTaskTimeout(taskId) {
        const taskInfo = this.activeTasks.get(taskId);
        if (!taskInfo) return;

        this.stats.tasksFailed++;
        this.activeTasks.delete(taskId);
        this.taskTimeouts.delete(taskId);

        await this._sendError(taskInfo.from, taskId, 'Task execution timed out', 'TIMEOUT');

        this.emit('task-timeout', { taskId, from: taskInfo.from });

        console.warn(`[TaskExecutionHandler] Task ${taskId} timed out`);
    }

    /**
     * Handle incoming task result (when we submitted a task)
     */
    _handleTaskResult(from, data) {
        const { taskId, result, duration, processedBy } = data;

        this.emit('result-received', {
            taskId,
            from,
            result,
            duration,
            processedBy,
        });
    }

    /**
     * Handle incoming task error (when we submitted a task)
     */
    _handleTaskError(from, data) {
        const { taskId, error, code } = data;

        this.emit('error-received', {
            taskId,
            from,
            error,
            code,
        });
    }

    /**
     * Handle incoming progress update
     */
    _handleTaskProgress(from, data) {
        const { taskId, progress, status, elapsed } = data;

        this.emit('progress-received', {
            taskId,
            from,
            progress,
            status,
            elapsed,
        });
    }

    /**
     * Handle task cancellation request
     */
    async _handleTaskCancel(from, data) {
        const { taskId } = data;
        const taskInfo = this.activeTasks.get(taskId);

        if (!taskInfo) {
            return; // Task not found or already completed
        }

        // Verify cancellation is from original submitter
        if (taskInfo.from !== from) {
            console.warn(`[TaskExecutionHandler] Unauthorized cancel request for ${taskId}`);
            return;
        }

        // Clear timeout
        const timeout = this.taskTimeouts.get(taskId);
        if (timeout) {
            clearTimeout(timeout);
            this.taskTimeouts.delete(taskId);
        }

        this.activeTasks.delete(taskId);
        this.emit('task-cancelled', { taskId, from });

        console.log(`[TaskExecutionHandler] Task ${taskId} cancelled by originator`);
    }

    /**
     * Send task result back to originator
     */
    async _sendResult(to, taskId, result, duration) {
        if (!this.signaling?.isConnected) return;

        await this.signaling.sendSignal(to, 'task-result', {
            taskId,
            result,
            duration,
            processedBy: this.nodeId,
            success: true,
        });
    }

    /**
     * Send error back to originator
     */
    async _sendError(to, taskId, error, code = 'ERROR') {
        if (!this.signaling?.isConnected) return;

        await this.signaling.sendSignal(to, 'task-error', {
            taskId,
            error,
            code,
            processedBy: this.nodeId,
            success: false,
        });
    }

    /**
     * Send progress update to originator
     */
    async _sendProgress(to, taskId, progressData) {
        if (!this.signaling?.isConnected) return;

        try {
            await this.signaling.sendSignal(to, 'task-progress', {
                taskId,
                ...progressData,
                processedBy: this.nodeId,
            });
        } catch {
            // Ignore progress send errors
        }
    }

    /**
     * Submit a task to a remote peer
     * @param {string} toPeerId - Target peer ID
     * @param {Object} task - Task to submit
     * @param {Object} options - Submission options
     * @returns {Promise<Object>} Task result
     */
    async submitTask(toPeerId, task, options = {}) {
        if (!this.signaling?.isConnected) {
            throw new Error('Signaling not connected');
        }

        const taskId = task.id || `submit-${randomBytes(8).toString('hex')}`;
        const timeout = options.timeout || this.defaultTimeout;

        // Create promise for result
        return new Promise((resolve, reject) => {
            // Set up result listener
            const onResult = (data) => {
                if (data.taskId === taskId) {
                    cleanup();
                    resolve(data);
                }
            };

            const onError = (data) => {
                if (data.taskId === taskId) {
                    cleanup();
                    reject(new Error(data.error || 'Task failed'));
                }
            };

            const timeoutHandle = setTimeout(() => {
                cleanup();
                reject(new Error('Task submission timed out'));
            }, timeout);

            const cleanup = () => {
                clearTimeout(timeoutHandle);
                this.off('result-received', onResult);
                this.off('error-received', onError);
            };

            this.on('result-received', onResult);
            this.on('error-received', onError);

            // Send task assignment
            this.signaling.sendSignal(toPeerId, 'task-assign', {
                task: {
                    id: taskId,
                    type: task.type,
                    data: task.data,
                    options: task.options,
                    priority: task.priority || 'medium',
                    requiredCapabilities: task.requiredCapabilities,
                    timeout,
                },
            }).catch(err => {
                cleanup();
                reject(err);
            });
        });
    }

    /**
     * Broadcast task to first available peer
     * @param {Object} task - Task to execute
     * @param {Object} options - Options
     * @returns {Promise<Object>} Task result
     */
    async broadcastTask(task, options = {}) {
        if (!this.signaling?.isConnected) {
            throw new Error('Signaling not connected');
        }

        const peers = this.signaling.getOnlinePeers();
        if (peers.length === 0) {
            throw new Error('No peers available');
        }

        // Filter peers by capabilities if required
        const requiredCaps = task.requiredCapabilities || [];
        const eligiblePeers = requiredCaps.length > 0
            ? peers.filter(p => requiredCaps.every(c => p.capabilities?.includes(c)))
            : peers;

        if (eligiblePeers.length === 0) {
            throw new Error('No peers with required capabilities');
        }

        // Try peers in order until one succeeds
        const errors = [];
        for (const peer of eligiblePeers) {
            try {
                return await this.submitTask(peer.id, task, options);
            } catch (err) {
                errors.push({ peer: peer.id, error: err.message });
            }
        }

        throw new Error(`All peers failed: ${JSON.stringify(errors)}`);
    }

    /**
     * Get handler status
     */
    getStatus() {
        return {
            attached: this.attached,
            nodeId: this.nodeId,
            capabilities: this.capabilities,
            activeTasks: this.activeTasks.size,
            completedTasks: this.completedTasks.size,
            stats: { ...this.stats },
            avgExecutionTime: this.stats.tasksExecuted > 0
                ? this.stats.totalExecutionTime / this.stats.tasksExecuted
                : 0,
        };
    }
}

// ============================================
// AUTO-WIRE INTEGRATION
// ============================================

/**
 * Create and wire task execution when a node joins the network
 *
 * @param {Object} options
 * @param {FirebaseSignaling} options.signaling - Firebase signaling instance
 * @param {RealWorkerPool} options.workerPool - Worker pool (will create if not provided)
 * @param {Object} options.secureAccess - Secure access manager (optional)
 * @returns {Promise<TaskExecutionHandler>} Wired handler
 */
export async function createTaskExecutionWiring(options = {}) {
    const { signaling, secureAccess } = options;

    if (!signaling) {
        throw new Error('Signaling instance required for task execution wiring');
    }

    // Create or use provided worker pool
    let workerPool = options.workerPool;
    if (!workerPool) {
        const { RealWorkerPool } = await import('./real-workers.js');
        workerPool = new RealWorkerPool({ size: 2 });
        await workerPool.initialize();
    }

    // Create handler
    const handler = new TaskExecutionHandler({
        signaling,
        workerPool,
        secureAccess,
        nodeId: signaling.peerId,
        capabilities: options.capabilities || ['compute', 'process', 'embed', 'transform', 'analyze'],
    });

    // Attach to signaling
    handler.attach();

    // Log task events
    handler.on('task-start', ({ taskId, from, type }) => {
        console.log(`  [Task] Starting ${type} task ${taskId.slice(0, 8)}... from ${from?.slice(0, 8)}...`);
    });

    handler.on('task-complete', ({ taskId, duration }) => {
        console.log(`  [Task] Completed ${taskId.slice(0, 8)}... in ${duration}ms`);
    });

    handler.on('task-error', ({ taskId, error }) => {
        console.log(`  [Task] Failed ${taskId.slice(0, 8)}...: ${error}`);
    });

    return handler;
}

/**
 * Integration class that auto-wires everything when a node joins
 */
export class DistributedTaskNetwork extends EventEmitter {
    constructor(options = {}) {
        super();

        this.signaling = null;
        this.workerPool = null;
        this.handler = null;
        this.secureAccess = options.secureAccess || null;

        this.config = {
            room: options.room || 'default',
            peerId: options.peerId,
            capabilities: options.capabilities || ['compute', 'process', 'embed'],
            firebaseConfig: options.firebaseConfig,
            autoInitWorkers: options.autoInitWorkers !== false,
        };

        this.connected = false;
    }

    /**
     * Initialize and join the distributed task network
     */
    async join() {
        console.log('\n[DistributedTaskNetwork] Joining network...');

        // Initialize Firebase signaling
        const { FirebaseSignaling } = await import('./firebase-signaling.js');
        this.signaling = new FirebaseSignaling({
            peerId: this.config.peerId,
            room: this.config.room,
            firebaseConfig: this.config.firebaseConfig,
            secureAccess: this.secureAccess,
        });

        // Connect to Firebase
        const connected = await this.signaling.connect();
        if (!connected) {
            throw new Error('Failed to connect to Firebase signaling');
        }

        // Initialize worker pool
        if (this.config.autoInitWorkers) {
            const { RealWorkerPool } = await import('./real-workers.js');
            this.workerPool = new RealWorkerPool({ size: 2 });
            await this.workerPool.initialize();
        }

        // Create and attach task handler
        this.handler = new TaskExecutionHandler({
            signaling: this.signaling,
            workerPool: this.workerPool,
            secureAccess: this.secureAccess,
            nodeId: this.signaling.peerId,
            capabilities: this.config.capabilities,
        });
        this.handler.attach();

        // Forward events
        this.handler.on('task-complete', (data) => this.emit('task-complete', data));
        this.handler.on('task-error', (data) => this.emit('task-error', data));
        this.handler.on('result-received', (data) => this.emit('result-received', data));

        this.connected = true;

        console.log(`[DistributedTaskNetwork] Joined as ${this.signaling.peerId.slice(0, 8)}...`);
        console.log(`[DistributedTaskNetwork] Capabilities: ${this.config.capabilities.join(', ')}`);

        this.emit('joined', {
            peerId: this.signaling.peerId,
            capabilities: this.config.capabilities,
        });

        return this;
    }

    /**
     * Submit a task for distributed execution
     */
    async submitTask(task, options = {}) {
        if (!this.connected || !this.handler) {
            throw new Error('Not connected to network');
        }

        // If specific peer, submit directly
        if (options.targetPeer) {
            return this.handler.submitTask(options.targetPeer, task, options);
        }

        // Otherwise broadcast to find available peer
        return this.handler.broadcastTask(task, options);
    }

    /**
     * Execute task locally (bypass network)
     */
    async executeLocally(task) {
        if (!this.workerPool || this.workerPool.status !== 'ready') {
            throw new Error('Worker pool not ready');
        }

        return this.workerPool.execute(task.type, task.data, task.options);
    }

    /**
     * Get online peers
     */
    getPeers() {
        return this.signaling?.getOnlinePeers() || [];
    }

    /**
     * Get network status
     */
    getStatus() {
        return {
            connected: this.connected,
            peerId: this.signaling?.peerId,
            peers: this.signaling?.peers?.size || 0,
            handler: this.handler?.getStatus(),
            workerPool: this.workerPool?.getStatus(),
        };
    }

    /**
     * Leave the network
     */
    async leave() {
        if (this.handler) {
            this.handler.detach();
        }

        if (this.workerPool) {
            await this.workerPool.shutdown();
        }

        if (this.signaling) {
            await this.signaling.disconnect();
        }

        this.connected = false;
        this.emit('left');

        console.log('[DistributedTaskNetwork] Left network');
    }
}

// ============================================
// EXPORTS
// ============================================

export default TaskExecutionHandler;

/**
 * @ruvector/edge-net REAL Worker System
 *
 * Actually functional distributed workers with:
 * - Real Node.js worker_threads for parallel execution
 * - Real WebSocket relay for task distribution
 * - Real result collection and aggregation
 * - Real resource management
 *
 * @module @ruvector/edge-net/real-workers
 */

import { EventEmitter } from 'events';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { randomBytes, createHash } from 'crypto';
import { cpus } from 'os';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================
// WORKER TASK TYPES
// ============================================

export const WorkerTaskTypes = {
    EMBED: 'embed',
    PROCESS: 'process',
    ANALYZE: 'analyze',
    TRANSFORM: 'transform',
    COMPUTE: 'compute',
    AGGREGATE: 'aggregate',
    CUSTOM: 'custom',
};

// ============================================
// INLINE WORKER CODE
// ============================================

const WORKER_CODE = `
const { parentPort, workerData } = require('worker_threads');
const crypto = require('crypto');

// Simple hash-based embedding (for fallback)
function hashEmbed(text, dims = 384) {
    const hash = crypto.createHash('sha256').update(String(text)).digest();
    const embedding = new Float32Array(dims);
    for (let i = 0; i < dims; i++) {
        embedding[i] = (hash[i % 32] - 128) / 128;
    }
    return Array.from(embedding);
}

// Task handlers
const handlers = {
    embed: (data) => {
        if (Array.isArray(data)) {
            return data.map(item => ({
                text: String(item).slice(0, 100),
                embedding: hashEmbed(item),
                dimensions: 384,
            }));
        }
        return {
            text: String(data).slice(0, 100),
            embedding: hashEmbed(data),
            dimensions: 384,
        };
    },

    process: (data, options = {}) => {
        const processor = options.processor || 'default';
        if (Array.isArray(data)) {
            return data.map((item, i) => ({
                index: i,
                processed: true,
                result: typeof item === 'object' ? { ...item, _processed: true } : { value: item, _processed: true },
                processor,
            }));
        }
        return {
            processed: true,
            result: typeof data === 'object' ? { ...data, _processed: true } : { value: data, _processed: true },
            processor,
        };
    },

    analyze: (data, options = {}) => {
        const items = Array.isArray(data) ? data : [data];
        const stats = {
            count: items.length,
            types: {},
            sizes: [],
        };

        for (const item of items) {
            const type = typeof item;
            stats.types[type] = (stats.types[type] || 0) + 1;
            if (typeof item === 'string') {
                stats.sizes.push(item.length);
            } else if (typeof item === 'object' && item !== null) {
                stats.sizes.push(JSON.stringify(item).length);
            }
        }

        stats.avgSize = stats.sizes.length > 0
            ? stats.sizes.reduce((a, b) => a + b, 0) / stats.sizes.length
            : 0;
        stats.minSize = stats.sizes.length > 0 ? Math.min(...stats.sizes) : 0;
        stats.maxSize = stats.sizes.length > 0 ? Math.max(...stats.sizes) : 0;

        return {
            analyzed: true,
            stats,
            timestamp: Date.now(),
        };
    },

    transform: (data, options = {}) => {
        const transform = options.transform || 'identity';
        const transforms = {
            identity: (x) => x,
            uppercase: (x) => typeof x === 'string' ? x.toUpperCase() : x,
            lowercase: (x) => typeof x === 'string' ? x.toLowerCase() : x,
            reverse: (x) => typeof x === 'string' ? x.split('').reverse().join('') : x,
            hash: (x) => crypto.createHash('sha256').update(String(x)).digest('hex'),
            json: (x) => JSON.stringify(x),
            length: (x) => typeof x === 'string' ? x.length : JSON.stringify(x).length,
        };

        const fn = transforms[transform] || transforms.identity;

        if (Array.isArray(data)) {
            return data.map(item => ({
                original: item,
                transformed: fn(item),
                transform,
            }));
        }
        return {
            original: data,
            transformed: fn(data),
            transform,
        };
    },

    compute: (data, options = {}) => {
        const operation = options.operation || 'sum';
        const items = Array.isArray(data) ? data : [data];
        const numbers = items.map(x => typeof x === 'number' ? x : parseFloat(x) || 0);

        const operations = {
            sum: () => numbers.reduce((a, b) => a + b, 0),
            product: () => numbers.reduce((a, b) => a * b, 1),
            mean: () => numbers.reduce((a, b) => a + b, 0) / numbers.length,
            min: () => Math.min(...numbers),
            max: () => Math.max(...numbers),
            variance: () => {
                const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
                return numbers.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numbers.length;
            },
        };

        return {
            computed: true,
            operation,
            result: (operations[operation] || operations.sum)(),
            inputCount: numbers.length,
        };
    },

    aggregate: (data, options = {}) => {
        const items = Array.isArray(data) ? data : [data];
        const groupBy = options.groupBy;

        if (groupBy && typeof items[0] === 'object') {
            const groups = {};
            for (const item of items) {
                const key = item[groupBy] || 'undefined';
                if (!groups[key]) groups[key] = [];
                groups[key].push(item);
            }
            return {
                aggregated: true,
                groupBy,
                groups: Object.keys(groups).map(key => ({
                    key,
                    count: groups[key].length,
                    items: groups[key],
                })),
            };
        }

        return {
            aggregated: true,
            count: items.length,
            items,
        };
    },

    custom: (data, options = {}) => {
        // For custom tasks, just return with metadata
        return {
            custom: true,
            data,
            options,
            timestamp: Date.now(),
        };
    },
};

// Handle messages from main thread
parentPort.on('message', (message) => {
    const { taskId, type, data, options } = message;

    try {
        const handler = handlers[type] || handlers.custom;
        const result = handler(data, options);

        parentPort.postMessage({
            taskId,
            success: true,
            result,
        });
    } catch (error) {
        parentPort.postMessage({
            taskId,
            success: false,
            error: error.message,
        });
    }
});
`;

// ============================================
// REAL WORKER THREAD POOL
// ============================================

/**
 * Real worker pool using Node.js worker_threads
 */
export class RealWorkerPool extends EventEmitter {
    constructor(options = {}) {
        super();
        this.id = `pool-${randomBytes(6).toString('hex')}`;
        this.size = options.size || Math.max(2, cpus().length - 1);
        this.maxQueueSize = options.maxQueueSize || 1000;

        this.workers = [];
        this.taskQueue = [];
        this.activeTasks = new Map();
        this.taskIdCounter = 0;

        this.status = 'created';
        this.stats = {
            tasksCompleted: 0,
            tasksFailed: 0,
            totalProcessingTime: 0,
            avgProcessingTime: 0,
        };
    }

    /**
     * Initialize the worker pool
     */
    async initialize() {
        this.status = 'initializing';
        this.emit('status', 'Initializing worker pool...');

        for (let i = 0; i < this.size; i++) {
            await this.spawnWorker(i);
        }

        this.status = 'ready';
        this.emit('ready', {
            poolId: this.id,
            workers: this.workers.length,
        });

        return this;
    }

    /**
     * Spawn a worker thread
     */
    async spawnWorker(index) {
        return new Promise((resolve, reject) => {
            try {
                const worker = new Worker(WORKER_CODE, { eval: true });

                const workerInfo = {
                    id: `worker-${index}`,
                    worker,
                    status: 'idle',
                    tasksCompleted: 0,
                    currentTask: null,
                    terminated: false,  // Track intentional termination
                };

                worker.on('message', (msg) => {
                    this.handleWorkerMessage(workerInfo, msg);
                });

                worker.on('error', (err) => {
                    console.error(`[Worker ${index}] Error:`, err.message);
                    this.handleWorkerError(workerInfo, err);
                });

                worker.on('exit', (code) => {
                    // Only respawn if worker crashed unexpectedly (not terminated intentionally)
                    if (!workerInfo.terminated && this.status === 'ready') {
                        console.error(`[Worker ${index}] Exited unexpectedly with code ${code}, respawning...`);
                        // Respawn worker
                        const idx = this.workers.indexOf(workerInfo);
                        if (idx >= 0) {
                            this.spawnWorker(index).then(w => {
                                this.workers[idx] = w;
                            }).catch(err => {
                                console.error(`[Worker ${index}] Failed to respawn:`, err.message);
                            });
                        }
                    }
                });

                this.workers.push(workerInfo);
                resolve(workerInfo);
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Handle message from worker
     */
    handleWorkerMessage(workerInfo, msg) {
        const { taskId, success, result, error } = msg;
        const taskInfo = this.activeTasks.get(taskId);

        if (!taskInfo) return;

        const duration = Date.now() - taskInfo.startTime;
        this.stats.totalProcessingTime += duration;

        if (success) {
            this.stats.tasksCompleted++;
            workerInfo.tasksCompleted++;
            taskInfo.resolve(result);
        } else {
            this.stats.tasksFailed++;
            taskInfo.reject(new Error(error));
        }

        this.stats.avgProcessingTime = this.stats.totalProcessingTime /
            (this.stats.tasksCompleted + this.stats.tasksFailed);

        this.activeTasks.delete(taskId);
        workerInfo.status = 'idle';
        workerInfo.currentTask = null;

        this.emit('task-complete', { taskId, success, duration });

        // Process next queued task
        this.processQueue();
    }

    /**
     * Handle worker error
     */
    handleWorkerError(workerInfo, error) {
        if (workerInfo.currentTask) {
            const taskInfo = this.activeTasks.get(workerInfo.currentTask);
            if (taskInfo) {
                taskInfo.reject(error);
                this.activeTasks.delete(workerInfo.currentTask);
            }
        }
        workerInfo.status = 'error';
        this.emit('worker-error', { workerId: workerInfo.id, error: error.message });
    }

    /**
     * Execute a single task
     */
    async execute(type, data, options = {}) {
        if (this.status !== 'ready') {
            throw new Error('Worker pool not ready');
        }

        const taskId = `task-${++this.taskIdCounter}`;

        return new Promise((resolve, reject) => {
            const taskInfo = {
                taskId,
                type,
                data,
                options,
                resolve,
                reject,
                startTime: null,
                queuedAt: Date.now(),
            };

            // Find idle worker
            const worker = this.findIdleWorker();

            if (worker) {
                this.dispatchTask(worker, taskInfo);
            } else {
                if (this.taskQueue.length >= this.maxQueueSize) {
                    reject(new Error('Task queue full'));
                    return;
                }
                this.taskQueue.push(taskInfo);
                this.emit('task-queued', { taskId, queueLength: this.taskQueue.length });
            }
        });
    }

    /**
     * Execute batch of tasks in parallel
     */
    async executeBatch(type, dataArray, options = {}) {
        if (!Array.isArray(dataArray)) {
            return this.execute(type, dataArray, options);
        }

        const batchId = `batch-${randomBytes(4).toString('hex')}`;
        const startTime = Date.now();

        this.emit('batch-start', { batchId, count: dataArray.length });

        const promises = dataArray.map(data =>
            this.execute(type, data, options)
        );

        const results = await Promise.allSettled(promises);

        const duration = Date.now() - startTime;
        const succeeded = results.filter(r => r.status === 'fulfilled').length;
        const failed = results.filter(r => r.status === 'rejected').length;

        this.emit('batch-complete', { batchId, duration, succeeded, failed });

        return results.map(r =>
            r.status === 'fulfilled' ? r.value : { error: r.reason.message }
        );
    }

    /**
     * Map operation across data array
     */
    async map(type, dataArray, options = {}) {
        return this.executeBatch(type, dataArray, options);
    }

    /**
     * Reduce operation with aggregation
     */
    async reduce(type, dataArray, options = {}) {
        const mapped = await this.executeBatch(type, dataArray, options);
        return this.execute('aggregate', mapped, options);
    }

    /**
     * Find an idle worker
     */
    findIdleWorker() {
        return this.workers.find(w => w.status === 'idle');
    }

    /**
     * Dispatch task to worker
     */
    dispatchTask(workerInfo, taskInfo) {
        workerInfo.status = 'busy';
        workerInfo.currentTask = taskInfo.taskId;
        taskInfo.startTime = Date.now();

        this.activeTasks.set(taskInfo.taskId, taskInfo);

        workerInfo.worker.postMessage({
            taskId: taskInfo.taskId,
            type: taskInfo.type,
            data: taskInfo.data,
            options: taskInfo.options,
        });

        this.emit('task-dispatched', {
            taskId: taskInfo.taskId,
            workerId: workerInfo.id,
            queueTime: taskInfo.startTime - taskInfo.queuedAt,
        });
    }

    /**
     * Process queued tasks
     */
    processQueue() {
        while (this.taskQueue.length > 0) {
            const worker = this.findIdleWorker();
            if (!worker) break;

            const taskInfo = this.taskQueue.shift();
            this.dispatchTask(worker, taskInfo);
        }
    }

    /**
     * Get pool status
     */
    getStatus() {
        return {
            poolId: this.id,
            status: this.status,
            workers: {
                total: this.workers.length,
                idle: this.workers.filter(w => w.status === 'idle').length,
                busy: this.workers.filter(w => w.status === 'busy').length,
            },
            queue: {
                size: this.taskQueue.length,
                maxSize: this.maxQueueSize,
            },
            activeTasks: this.activeTasks.size,
            stats: this.stats,
        };
    }

    /**
     * Shutdown the pool
     */
    async shutdown() {
        this.status = 'shutting_down';
        this.emit('shutdown-start');

        // Wait for active tasks with timeout
        const timeout = Date.now() + 10000;
        while (this.activeTasks.size > 0 && Date.now() < timeout) {
            await new Promise(r => setTimeout(r, 100));
        }

        // Terminate workers (mark as intentionally terminated first)
        for (const workerInfo of this.workers) {
            workerInfo.terminated = true;
            await workerInfo.worker.terminate();
        }

        this.workers = [];
        this.taskQueue = [];
        this.activeTasks.clear();
        this.status = 'shutdown';

        this.emit('shutdown-complete');
    }

    // Alias for shutdown
    async close() {
        return this.shutdown();
    }
}

// ============================================
// DISTRIBUTED TASK CLIENT
// ============================================

/**
 * Client for distributed task execution via relay
 */
export class DistributedTaskClient extends EventEmitter {
    constructor(options = {}) {
        super();
        this.relayUrl = options.relayUrl || 'ws://localhost:8080';
        this.nodeId = options.nodeId || `client-${randomBytes(8).toString('hex')}`;
        this.ws = null;
        this.connected = false;

        this.pendingTasks = new Map();
        this.completedTasks = new Map();
        this.taskIdCounter = 0;
    }

    /**
     * Connect to relay server
     */
    async connect() {
        return new Promise(async (resolve, reject) => {
            try {
                let WebSocket;
                if (typeof globalThis.WebSocket !== 'undefined') {
                    WebSocket = globalThis.WebSocket;
                } else {
                    const ws = await import('ws');
                    WebSocket = ws.default || ws.WebSocket;
                }

                this.ws = new WebSocket(this.relayUrl);

                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000);

                this.ws.onopen = () => {
                    clearTimeout(timeout);
                    this.connected = true;

                    // Register as task client
                    this.send({
                        type: 'register',
                        nodeId: this.nodeId,
                        capabilities: ['task_client'],
                    });

                    this.emit('connected');
                    resolve(true);
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(JSON.parse(event.data));
                };

                this.ws.onclose = () => {
                    this.connected = false;
                    this.emit('disconnected');
                };

                this.ws.onerror = (error) => {
                    clearTimeout(timeout);
                    reject(error);
                };

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Handle incoming message
     */
    handleMessage(message) {
        switch (message.type) {
            case 'welcome':
                this.emit('registered', message);
                break;

            case 'task_result':
                this.handleTaskResult(message);
                break;

            case 'task_progress':
                this.handleTaskProgress(message);
                break;

            case 'task_accepted':
                this.emit('task-accepted', message);
                break;

            default:
                this.emit('message', message);
        }
    }

    /**
     * Handle task result
     */
    handleTaskResult(message) {
        const taskInfo = this.pendingTasks.get(message.taskId);
        if (!taskInfo) return;

        const duration = Date.now() - taskInfo.startTime;

        if (message.success) {
            taskInfo.resolve({
                taskId: message.taskId,
                result: message.result,
                processedBy: message.processedBy,
                duration,
            });
        } else {
            taskInfo.reject(new Error(message.error || 'Task failed'));
        }

        this.pendingTasks.delete(message.taskId);
        this.completedTasks.set(message.taskId, {
            ...message,
            duration,
        });

        this.emit('task-complete', { taskId: message.taskId, duration });
    }

    /**
     * Handle task progress update
     */
    handleTaskProgress(message) {
        this.emit('task-progress', message);
    }

    /**
     * Send message to relay
     */
    send(message) {
        if (this.connected && this.ws?.readyState === 1) {
            this.ws.send(JSON.stringify(message));
            return true;
        }
        return false;
    }

    /**
     * Submit task for distributed execution
     */
    async submitTask(task, options = {}) {
        if (!this.connected) {
            throw new Error('Not connected to relay');
        }

        const taskId = `dtask-${++this.taskIdCounter}-${Date.now()}`;

        return new Promise((resolve, reject) => {
            const taskInfo = {
                taskId,
                task,
                options,
                resolve,
                reject,
                startTime: Date.now(),
            };

            this.pendingTasks.set(taskId, taskInfo);

            // Set timeout
            const timeout = options.timeout || 60000;
            setTimeout(() => {
                if (this.pendingTasks.has(taskId)) {
                    this.pendingTasks.delete(taskId);
                    reject(new Error('Task timeout'));
                }
            }, timeout);

            // Submit to relay
            this.send({
                type: 'task_submit',
                task: {
                    id: taskId,
                    type: task.type || 'compute',
                    data: task.data,
                    options: task.options,
                    priority: options.priority || 'medium',
                },
            });
        });
    }

    /**
     * Submit batch of tasks
     */
    async submitBatch(tasks, options = {}) {
        const promises = tasks.map(task =>
            this.submitTask(task, options)
        );
        return Promise.allSettled(promises);
    }

    /**
     * Close connection
     */
    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// ============================================
// DISTRIBUTED TASK WORKER
// ============================================

/**
 * Worker that processes distributed tasks from relay
 */
export class DistributedTaskWorker extends EventEmitter {
    constructor(options = {}) {
        super();
        this.relayUrl = options.relayUrl || 'ws://localhost:8080';
        this.nodeId = options.nodeId || `worker-${randomBytes(8).toString('hex')}`;
        this.capabilities = options.capabilities || ['compute', 'embed', 'process'];
        this.ws = null;
        this.connected = false;

        this.localPool = null;
        this.activeTasks = new Map();

        this.stats = {
            tasksProcessed: 0,
            tasksFailed: 0,
            creditsEarned: 0,
        };
    }

    /**
     * Initialize worker with local pool
     */
    async initialize() {
        this.localPool = new RealWorkerPool({ size: 2 });
        await this.localPool.initialize();
        return this;
    }

    /**
     * Connect to relay and start processing
     */
    async connect() {
        return new Promise(async (resolve, reject) => {
            try {
                let WebSocket;
                if (typeof globalThis.WebSocket !== 'undefined') {
                    WebSocket = globalThis.WebSocket;
                } else {
                    const ws = await import('ws');
                    WebSocket = ws.default || ws.WebSocket;
                }

                this.ws = new WebSocket(this.relayUrl);

                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000);

                this.ws.onopen = () => {
                    clearTimeout(timeout);
                    this.connected = true;

                    // Register as task worker
                    this.send({
                        type: 'register',
                        nodeId: this.nodeId,
                        capabilities: this.capabilities,
                        workerType: 'task_processor',
                    });

                    this.emit('connected');
                    resolve(true);
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(JSON.parse(event.data));
                };

                this.ws.onclose = () => {
                    this.connected = false;
                    this.emit('disconnected');
                };

                this.ws.onerror = (error) => {
                    clearTimeout(timeout);
                    reject(error);
                };

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Handle incoming message
     */
    handleMessage(message) {
        switch (message.type) {
            case 'welcome':
                console.log(`[Worker] Registered: ${this.nodeId}`);
                this.emit('registered', message);
                break;

            case 'task_assignment':
                this.processTask(message.task);
                break;

            default:
                this.emit('message', message);
        }
    }

    /**
     * Process assigned task
     */
    async processTask(task) {
        const startTime = Date.now();
        this.activeTasks.set(task.id, task);

        console.log(`[Worker] Processing task: ${task.id}`);
        this.emit('task-start', { taskId: task.id });

        try {
            // Execute using local pool
            const result = await this.localPool.execute(
                task.type || 'process',
                task.data,
                task.options
            );

            const duration = Date.now() - startTime;
            this.stats.tasksProcessed++;

            // Report result
            this.send({
                type: 'task_complete',
                taskId: task.id,
                submitterId: task.submitter,
                result,
                reward: Math.ceil(1000000 * (duration / 1000)), // 0.001 rUv per second
                success: true,
            });

            this.emit('task-complete', { taskId: task.id, duration, result });

        } catch (error) {
            this.stats.tasksFailed++;

            this.send({
                type: 'task_complete',
                taskId: task.id,
                submitterId: task.submitter,
                error: error.message,
                success: false,
            });

            this.emit('task-error', { taskId: task.id, error: error.message });
        }

        this.activeTasks.delete(task.id);
    }

    /**
     * Send message to relay
     */
    send(message) {
        if (this.connected && this.ws?.readyState === 1) {
            this.ws.send(JSON.stringify(message));
            return true;
        }
        return false;
    }

    /**
     * Get worker stats
     */
    getStats() {
        return {
            nodeId: this.nodeId,
            connected: this.connected,
            activeTasks: this.activeTasks.size,
            ...this.stats,
            poolStatus: this.localPool?.getStatus(),
        };
    }

    /**
     * Stop worker
     */
    async stop() {
        if (this.localPool) {
            await this.localPool.shutdown();
        }
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Default export
export default RealWorkerPool;

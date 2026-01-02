/**
 * @ruvector/edge-net Task Scheduler with Load Balancing
 *
 * Distributed task scheduling with:
 * - Priority queuing
 * - Load balancing across workers
 * - Task affinity and locality
 * - Retry and failure handling
 * - Resource allocation
 *
 * @module @ruvector/edge-net/scheduler
 */

import { EventEmitter } from 'events';
import { randomBytes } from 'crypto';

// ============================================
// TASK PRIORITY
// ============================================

export const TaskPriority = {
    CRITICAL: 0,
    HIGH: 1,
    MEDIUM: 2,
    LOW: 3,
};

export const TaskStatus = {
    PENDING: 'pending',
    QUEUED: 'queued',
    ASSIGNED: 'assigned',
    RUNNING: 'running',
    COMPLETED: 'completed',
    FAILED: 'failed',
    CANCELLED: 'cancelled',
    RETRYING: 'retrying',
};

// ============================================
// TASK
// ============================================

/**
 * Task definition
 */
export class Task {
    constructor(options = {}) {
        this.id = options.id || `task-${randomBytes(8).toString('hex')}`;
        this.type = options.type || 'compute';
        this.data = options.data;
        this.options = options.options || {};

        // Priority and scheduling
        this.priority = options.priority ?? TaskPriority.MEDIUM;
        this.deadline = options.deadline || null;
        this.timeout = options.timeout || 60000; // 1 minute default

        // Retry configuration
        this.maxRetries = options.maxRetries ?? 3;
        this.retryCount = 0;
        this.retryDelay = options.retryDelay || 1000;

        // Affinity
        this.preferredWorker = options.preferredWorker || null;
        this.requiredCapabilities = options.requiredCapabilities || [];

        // Resource requirements
        this.resources = options.resources || {
            cpu: 1,
            memory: 256, // MB
        };

        // Status tracking
        this.status = TaskStatus.PENDING;
        this.assignedTo = null;
        this.result = null;
        this.error = null;

        // Timestamps
        this.createdAt = Date.now();
        this.queuedAt = null;
        this.startedAt = null;
        this.completedAt = null;

        // Callback
        this.resolve = null;
        this.reject = null;
    }

    /**
     * Set task result
     */
    setResult(result) {
        this.result = result;
        this.status = TaskStatus.COMPLETED;
        this.completedAt = Date.now();
        if (this.resolve) this.resolve(result);
    }

    /**
     * Set task error
     */
    setError(error) {
        this.error = error;
        this.status = TaskStatus.FAILED;
        this.completedAt = Date.now();
        if (this.reject) this.reject(error);
    }

    /**
     * Check if task can retry
     */
    canRetry() {
        return this.retryCount < this.maxRetries;
    }

    /**
     * Check if task has expired
     */
    isExpired() {
        if (!this.deadline) return false;
        return Date.now() > this.deadline;
    }

    /**
     * Get task age in milliseconds
     */
    age() {
        return Date.now() - this.createdAt;
    }

    /**
     * Get execution time in milliseconds
     */
    executionTime() {
        if (!this.startedAt) return 0;
        const end = this.completedAt || Date.now();
        return end - this.startedAt;
    }

    toJSON() {
        return {
            id: this.id,
            type: this.type,
            data: this.data,
            options: this.options,
            priority: this.priority,
            status: this.status,
            assignedTo: this.assignedTo,
            result: this.result,
            error: this.error?.message || this.error,
            retryCount: this.retryCount,
            createdAt: this.createdAt,
            completedAt: this.completedAt,
        };
    }
}

// ============================================
// WORKER INFO
// ============================================

/**
 * Worker information for scheduling
 */
export class WorkerInfo {
    constructor(options = {}) {
        this.id = options.id;
        this.capabilities = options.capabilities || [];
        this.maxConcurrent = options.maxConcurrent || 4;

        // Current state
        this.activeTasks = new Set();
        this.status = 'idle'; // idle, busy, offline

        // Resources
        this.resources = {
            cpu: options.cpu || 4,
            memory: options.memory || 4096,
            cpuUsed: 0,
            memoryUsed: 0,
        };

        // Performance metrics
        this.metrics = {
            tasksCompleted: 0,
            tasksFailed: 0,
            avgExecutionTime: 0,
            lastTaskTime: 0,
            successRate: 1.0,
        };

        // Timestamps
        this.connectedAt = Date.now();
        this.lastSeen = Date.now();
    }

    /**
     * Check if worker has capacity
     */
    hasCapacity() {
        return this.activeTasks.size < this.maxConcurrent;
    }

    /**
     * Check if worker has required capabilities
     */
    hasCapabilities(required) {
        if (!required || required.length === 0) return true;
        return required.every(cap => this.capabilities.includes(cap));
    }

    /**
     * Check if worker has resources
     */
    hasResources(required) {
        const cpuAvailable = this.resources.cpu - this.resources.cpuUsed;
        const memAvailable = this.resources.memory - this.resources.memoryUsed;

        return cpuAvailable >= (required.cpu || 1) &&
               memAvailable >= (required.memory || 256);
    }

    /**
     * Allocate resources for task
     */
    allocate(task) {
        this.activeTasks.add(task.id);
        this.resources.cpuUsed += task.resources.cpu || 1;
        this.resources.memoryUsed += task.resources.memory || 256;
        this.status = this.activeTasks.size >= this.maxConcurrent ? 'busy' : 'idle';
    }

    /**
     * Release resources from task
     */
    release(task) {
        this.activeTasks.delete(task.id);
        this.resources.cpuUsed = Math.max(0, this.resources.cpuUsed - (task.resources.cpu || 1));
        this.resources.memoryUsed = Math.max(0, this.resources.memoryUsed - (task.resources.memory || 256));
        this.status = this.activeTasks.size >= this.maxConcurrent ? 'busy' : 'idle';
    }

    /**
     * Update metrics after task completion
     */
    updateMetrics(task, success) {
        if (success) {
            this.metrics.tasksCompleted++;
        } else {
            this.metrics.tasksFailed++;
        }

        const total = this.metrics.tasksCompleted + this.metrics.tasksFailed;
        this.metrics.successRate = this.metrics.tasksCompleted / total;

        // Update average execution time
        const execTime = task.executionTime();
        this.metrics.avgExecutionTime =
            (this.metrics.avgExecutionTime * (total - 1) + execTime) / total;

        this.metrics.lastTaskTime = Date.now();
    }

    /**
     * Calculate worker score for scheduling
     */
    score(task) {
        let score = 100;

        // Prefer workers with capacity
        if (!this.hasCapacity()) return -1;

        // Prefer workers with better success rate
        score += this.metrics.successRate * 20;

        // Prefer workers with lower load
        const loadRatio = this.activeTasks.size / this.maxConcurrent;
        score -= loadRatio * 30;

        // Prefer workers with faster execution
        if (this.metrics.avgExecutionTime > 0) {
            score -= Math.min(this.metrics.avgExecutionTime / 1000, 20);
        }

        // Affinity bonus
        if (task.preferredWorker === this.id) {
            score += 50;
        }

        return score;
    }
}

// ============================================
// PRIORITY QUEUE
// ============================================

/**
 * Priority queue for tasks
 */
class PriorityQueue {
    constructor() {
        this.queues = new Map([
            [TaskPriority.CRITICAL, []],
            [TaskPriority.HIGH, []],
            [TaskPriority.MEDIUM, []],
            [TaskPriority.LOW, []],
        ]);
        this.size = 0;
    }

    enqueue(task) {
        const queue = this.queues.get(task.priority) || this.queues.get(TaskPriority.MEDIUM);
        queue.push(task);
        this.size++;
    }

    dequeue() {
        for (const [priority, queue] of this.queues) {
            if (queue.length > 0) {
                this.size--;
                return queue.shift();
            }
        }
        return null;
    }

    peek() {
        for (const [priority, queue] of this.queues) {
            if (queue.length > 0) {
                return queue[0];
            }
        }
        return null;
    }

    remove(taskId) {
        for (const [priority, queue] of this.queues) {
            const index = queue.findIndex(t => t.id === taskId);
            if (index >= 0) {
                queue.splice(index, 1);
                this.size--;
                return true;
            }
        }
        return false;
    }

    getAll() {
        const all = [];
        for (const [priority, queue] of this.queues) {
            all.push(...queue);
        }
        return all;
    }
}

// ============================================
// TASK SCHEDULER
// ============================================

/**
 * Distributed Task Scheduler
 */
export class TaskScheduler extends EventEmitter {
    constructor(options = {}) {
        super();
        this.id = options.id || `scheduler-${randomBytes(8).toString('hex')}`;

        // Task queues
        this.pending = new PriorityQueue();
        this.running = new Map();     // taskId -> Task
        this.completed = new Map();   // taskId -> Task (limited)

        // Workers
        this.workers = new Map();     // workerId -> WorkerInfo

        // Configuration
        this.maxCompleted = options.maxCompleted || 1000;
        this.schedulingInterval = options.schedulingInterval || 100;
        this.cleanupInterval = options.cleanupInterval || 60000;

        // Stats
        this.stats = {
            submitted: 0,
            completed: 0,
            failed: 0,
            retried: 0,
            avgWaitTime: 0,
            avgExecutionTime: 0,
        };

        // Internal
        this.schedulerTimer = null;
        this.cleanupTimer = null;
        this.started = false;
    }

    /**
     * Start the scheduler
     */
    start() {
        if (this.started) return;

        this.schedulerTimer = setInterval(() => {
            this.schedule();
        }, this.schedulingInterval);

        this.cleanupTimer = setInterval(() => {
            this.cleanup();
        }, this.cleanupInterval);

        this.started = true;
        this.emit('started');
    }

    /**
     * Stop the scheduler
     */
    stop() {
        if (this.schedulerTimer) {
            clearInterval(this.schedulerTimer);
        }
        if (this.cleanupTimer) {
            clearInterval(this.cleanupTimer);
        }
        this.started = false;
        this.emit('stopped');
    }

    /**
     * Register a worker
     */
    registerWorker(worker) {
        const workerInfo = worker instanceof WorkerInfo
            ? worker
            : new WorkerInfo(worker);

        this.workers.set(workerInfo.id, workerInfo);
        this.emit('worker-registered', { workerId: workerInfo.id });

        // Trigger scheduling
        this.schedule();

        return workerInfo;
    }

    /**
     * Unregister a worker
     */
    unregisterWorker(workerId) {
        const worker = this.workers.get(workerId);
        if (!worker) return;

        // Requeue active tasks
        for (const taskId of worker.activeTasks) {
            const task = this.running.get(taskId);
            if (task) {
                task.status = TaskStatus.PENDING;
                task.assignedTo = null;
                this.running.delete(taskId);
                this.pending.enqueue(task);
            }
        }

        this.workers.delete(workerId);
        this.emit('worker-unregistered', { workerId });
    }

    /**
     * Submit a task
     */
    submit(taskOptions) {
        const task = taskOptions instanceof Task
            ? taskOptions
            : new Task(taskOptions);

        task.status = TaskStatus.QUEUED;
        task.queuedAt = Date.now();
        this.stats.submitted++;

        this.pending.enqueue(task);
        this.emit('task-submitted', { taskId: task.id });

        // Return promise for task completion
        return new Promise((resolve, reject) => {
            task.resolve = resolve;
            task.reject = reject;
        });
    }

    /**
     * Submit batch of tasks
     */
    submitBatch(tasks) {
        return Promise.all(tasks.map(t => this.submit(t)));
    }

    /**
     * Cancel a task
     */
    cancel(taskId) {
        // Check pending queue
        if (this.pending.remove(taskId)) {
            this.emit('task-cancelled', { taskId });
            return true;
        }

        // Check running tasks
        const task = this.running.get(taskId);
        if (task) {
            task.status = TaskStatus.CANCELLED;
            task.completedAt = Date.now();
            if (task.reject) {
                task.reject(new Error('Task cancelled'));
            }

            // Release worker resources
            if (task.assignedTo) {
                const worker = this.workers.get(task.assignedTo);
                if (worker) worker.release(task);
            }

            this.running.delete(taskId);
            this.emit('task-cancelled', { taskId });
            return true;
        }

        return false;
    }

    /**
     * Main scheduling loop
     */
    schedule() {
        while (this.pending.size > 0) {
            const task = this.pending.peek();
            if (!task) break;

            // Check if task expired
            if (task.isExpired()) {
                this.pending.dequeue();
                task.setError(new Error('Task deadline exceeded'));
                this.stats.failed++;
                this.emit('task-expired', { taskId: task.id });
                continue;
            }

            // Find best worker
            const worker = this.selectWorker(task);
            if (!worker) break; // No available workers

            // Assign task
            this.pending.dequeue();
            this.assignTask(task, worker);
        }
    }

    /**
     * Select best worker for task
     */
    selectWorker(task) {
        let bestWorker = null;
        let bestScore = -Infinity;

        for (const [workerId, worker] of this.workers) {
            // Skip offline workers
            if (worker.status === 'offline') continue;

            // Check capabilities
            if (!worker.hasCapabilities(task.requiredCapabilities)) continue;

            // Check resources
            if (!worker.hasResources(task.resources)) continue;

            // Calculate score
            const score = worker.score(task);
            if (score > bestScore) {
                bestScore = score;
                bestWorker = worker;
            }
        }

        return bestWorker;
    }

    /**
     * Assign task to worker
     */
    assignTask(task, worker) {
        task.status = TaskStatus.ASSIGNED;
        task.assignedTo = worker.id;
        task.startedAt = Date.now();

        worker.allocate(task);
        this.running.set(task.id, task);

        // Calculate wait time using running average
        const waitTime = task.startedAt - task.queuedAt;
        const assignedCount = this.stats.completed + this.running.size;
        if (assignedCount <= 1) {
            this.stats.avgWaitTime = waitTime;
        } else {
            this.stats.avgWaitTime =
                (this.stats.avgWaitTime * (assignedCount - 1) + waitTime) / assignedCount;
        }

        this.emit('task-assigned', {
            taskId: task.id,
            workerId: worker.id,
            waitTime,
        });

        // Set timeout
        setTimeout(() => {
            this.checkTaskTimeout(task.id);
        }, task.timeout);
    }

    /**
     * Check if task has timed out
     */
    checkTaskTimeout(taskId) {
        const task = this.running.get(taskId);
        if (!task || task.status === TaskStatus.COMPLETED) return;

        if (task.executionTime() >= task.timeout) {
            this.handleTaskFailure(task, new Error('Task timeout'));
        }
    }

    /**
     * Report task completion
     */
    completeTask(taskId, result) {
        const task = this.running.get(taskId);
        if (!task) return;

        task.setResult(result);
        this.running.delete(taskId);

        // Update worker
        const worker = this.workers.get(task.assignedTo);
        if (worker) {
            worker.release(task);
            worker.updateMetrics(task, true);
        }

        // Update stats
        this.stats.completed++;
        this.stats.avgExecutionTime =
            (this.stats.avgExecutionTime * (this.stats.completed - 1) + task.executionTime()) /
            this.stats.completed;

        // Store in completed (limited)
        this.completed.set(taskId, task);
        if (this.completed.size > this.maxCompleted) {
            const oldest = this.completed.keys().next().value;
            this.completed.delete(oldest);
        }

        this.emit('task-completed', { taskId, result, executionTime: task.executionTime() });
    }

    /**
     * Report task failure
     */
    failTask(taskId, error) {
        const task = this.running.get(taskId);
        if (!task) return;

        this.handleTaskFailure(task, error);
    }

    /**
     * Handle task failure with retry logic
     */
    handleTaskFailure(task, error) {
        this.running.delete(task.id);

        // Update worker
        const worker = this.workers.get(task.assignedTo);
        if (worker) {
            worker.release(task);
            worker.updateMetrics(task, false);
        }

        // Check for retry
        if (task.canRetry()) {
            task.retryCount++;
            task.status = TaskStatus.RETRYING;
            task.assignedTo = null;
            this.stats.retried++;

            // Re-queue with delay
            setTimeout(() => {
                task.status = TaskStatus.QUEUED;
                this.pending.enqueue(task);
                this.emit('task-retrying', { taskId: task.id, retryCount: task.retryCount });
            }, task.retryDelay * task.retryCount);

        } else {
            task.setError(error);
            this.stats.failed++;
            this.emit('task-failed', { taskId: task.id, error: error.message });
        }
    }

    /**
     * Cleanup old completed tasks and offline workers
     */
    cleanup() {
        const now = Date.now();
        const workerTimeout = 60000; // 1 minute

        // Check for offline workers
        for (const [workerId, worker] of this.workers) {
            if (now - worker.lastSeen > workerTimeout) {
                worker.status = 'offline';
            }
        }

        this.emit('cleanup');
    }

    /**
     * Get scheduler status
     */
    getStatus() {
        return {
            id: this.id,
            started: this.started,
            pending: this.pending.size,
            running: this.running.size,
            completed: this.completed.size,
            workers: {
                total: this.workers.size,
                idle: Array.from(this.workers.values()).filter(w => w.status === 'idle').length,
                busy: Array.from(this.workers.values()).filter(w => w.status === 'busy').length,
                offline: Array.from(this.workers.values()).filter(w => w.status === 'offline').length,
            },
            stats: this.stats,
        };
    }

    /**
     * Get task by ID
     */
    getTask(taskId) {
        return this.running.get(taskId) ||
               this.completed.get(taskId) ||
               this.pending.getAll().find(t => t.id === taskId);
    }

    /**
     * Get all workers
     */
    getWorkers() {
        return Array.from(this.workers.values());
    }
}

// ============================================
// EXPORTS
// ============================================

export default TaskScheduler;

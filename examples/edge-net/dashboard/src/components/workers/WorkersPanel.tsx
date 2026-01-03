import { useState, useEffect } from 'react';
import { Card, CardBody, Progress, Button, Chip } from '@heroui/react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Cpu,
  Server,
  Activity,
  Zap,
  Clock,
  CheckCircle2,
  AlertCircle,
  Pause,
  Play,
  RefreshCw,
  Layers,
  Timer,
  TrendingUp,
} from 'lucide-react';

interface Worker {
  id: string;
  name: string;
  type: 'cpu' | 'gpu' | 'hybrid';
  status: 'active' | 'idle' | 'offline' | 'error';
  cpuUsage: number;
  gpuUsage: number;
  memoryUsage: number;
  tasksCompleted: number;
  tasksQueued: number;
  uptime: string;
  lastHeartbeat: string;
}

const mockWorkers: Worker[] = [
  {
    id: 'w1',
    name: 'edge-node-alpha',
    type: 'hybrid',
    status: 'active',
    cpuUsage: 78,
    gpuUsage: 92,
    memoryUsage: 64,
    tasksCompleted: 1247,
    tasksQueued: 12,
    uptime: '7d 14h 32m',
    lastHeartbeat: '2s ago',
  },
  {
    id: 'w2',
    name: 'compute-beta-01',
    type: 'gpu',
    status: 'active',
    cpuUsage: 45,
    gpuUsage: 98,
    memoryUsage: 82,
    tasksCompleted: 3892,
    tasksQueued: 8,
    uptime: '12d 3h 15m',
    lastHeartbeat: '1s ago',
  },
  {
    id: 'w3',
    name: 'vector-gamma',
    type: 'cpu',
    status: 'idle',
    cpuUsage: 12,
    gpuUsage: 0,
    memoryUsage: 28,
    tasksCompleted: 892,
    tasksQueued: 0,
    uptime: '3d 8h 45m',
    lastHeartbeat: '5s ago',
  },
  {
    id: 'w4',
    name: 'neural-delta',
    type: 'hybrid',
    status: 'error',
    cpuUsage: 0,
    gpuUsage: 0,
    memoryUsage: 0,
    tasksCompleted: 456,
    tasksQueued: 23,
    uptime: '0d 0h 0m',
    lastHeartbeat: '5m ago',
  },
  {
    id: 'w5',
    name: 'inference-epsilon',
    type: 'gpu',
    status: 'active',
    cpuUsage: 32,
    gpuUsage: 87,
    memoryUsage: 71,
    tasksCompleted: 2134,
    tasksQueued: 5,
    uptime: '9d 21h 8m',
    lastHeartbeat: '1s ago',
  },
];

const statusConfig = {
  active: { color: 'bg-emerald-500', text: 'text-emerald-400', icon: CheckCircle2 },
  idle: { color: 'bg-amber-500', text: 'text-amber-400', icon: Pause },
  offline: { color: 'bg-zinc-500', text: 'text-zinc-400', icon: Clock },
  error: { color: 'bg-red-500', text: 'text-red-400', icon: AlertCircle },
};

const typeConfig = {
  cpu: { color: 'bg-sky-500/20 text-sky-400 border-sky-500/30', label: 'CPU' },
  gpu: { color: 'bg-violet-500/20 text-violet-400 border-violet-500/30', label: 'GPU' },
  hybrid: { color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30', label: 'Hybrid' },
};

function WorkerCard({ worker, index }: { worker: Worker; index: number }) {
  const status = statusConfig[worker.status];
  const type = typeConfig[worker.type];
  const StatusIcon = status.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className="bg-zinc-900/50 backdrop-blur-xl border border-white/10 hover:border-white/20 transition-all">
        <CardBody className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className={`w-10 h-10 rounded-lg ${type.color} border flex items-center justify-center`}>
                  {worker.type === 'gpu' ? (
                    <Zap size={20} />
                  ) : worker.type === 'cpu' ? (
                    <Cpu size={20} />
                  ) : (
                    <Server size={20} />
                  )}
                </div>
                <motion.div
                  className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full ${status.color}`}
                  animate={worker.status === 'active' ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              <div>
                <h4 className="font-medium text-white">{worker.name}</h4>
                <div className="flex items-center gap-2 mt-0.5">
                  <Chip size="sm" className={`${type.color} border text-xs`}>
                    {type.label}
                  </Chip>
                  <span className={`text-xs ${status.text} flex items-center gap-1`}>
                    <StatusIcon size={12} />
                    {worker.status}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex gap-1">
              {worker.status === 'active' ? (
                <Button isIconOnly size="sm" variant="flat" className="bg-amber-500/20 text-amber-400">
                  <Pause size={14} />
                </Button>
              ) : worker.status !== 'error' ? (
                <Button isIconOnly size="sm" variant="flat" className="bg-emerald-500/20 text-emerald-400">
                  <Play size={14} />
                </Button>
              ) : null}
              <Button isIconOnly size="sm" variant="flat" className="bg-white/5 text-zinc-400">
                <RefreshCw size={14} />
              </Button>
            </div>
          </div>

          {/* Utilization Bars */}
          <div className="space-y-3 mb-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400 flex items-center gap-1">
                  <Cpu size={12} /> CPU
                </span>
                <span className="text-sky-400">{worker.cpuUsage}%</span>
              </div>
              <Progress
                value={worker.cpuUsage}
                classNames={{
                  indicator: 'bg-gradient-to-r from-sky-500 to-cyan-500',
                  track: 'bg-zinc-800',
                }}
                size="sm"
              />
            </div>

            {worker.type !== 'cpu' && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-zinc-400 flex items-center gap-1">
                    <Zap size={12} /> GPU
                  </span>
                  <span className="text-violet-400">{worker.gpuUsage}%</span>
                </div>
                <Progress
                  value={worker.gpuUsage}
                  classNames={{
                    indicator: 'bg-gradient-to-r from-violet-500 to-purple-500',
                    track: 'bg-zinc-800',
                  }}
                  size="sm"
                />
              </div>
            )}

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400 flex items-center gap-1">
                  <Layers size={12} /> Memory
                </span>
                <span className="text-amber-400">{worker.memoryUsage}%</span>
              </div>
              <Progress
                value={worker.memoryUsage}
                classNames={{
                  indicator: 'bg-gradient-to-r from-amber-500 to-orange-500',
                  track: 'bg-zinc-800',
                }}
                size="sm"
              />
            </div>
          </div>

          {/* Stats Footer */}
          <div className="grid grid-cols-3 gap-2 pt-3 border-t border-white/10">
            <div className="text-center">
              <p className="text-lg font-semibold text-white">{worker.tasksCompleted.toLocaleString()}</p>
              <p className="text-xs text-zinc-500">Completed</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold text-cyan-400">{worker.tasksQueued}</p>
              <p className="text-xs text-zinc-500">Queued</p>
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-zinc-400">{worker.uptime}</p>
              <p className="text-xs text-zinc-500">Uptime</p>
            </div>
          </div>
        </CardBody>
      </Card>
    </motion.div>
  );
}

export function WorkersPanel() {
  const [workers, setWorkers] = useState<Worker[]>(mockWorkers);
  const [filter, setFilter] = useState<'all' | 'active' | 'idle' | 'error'>('all');

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setWorkers((prev) =>
        prev.map((w) => ({
          ...w,
          cpuUsage: w.status === 'active' ? Math.min(100, Math.max(10, w.cpuUsage + (Math.random() - 0.5) * 10)) : w.cpuUsage,
          gpuUsage: w.status === 'active' && w.type !== 'cpu' ? Math.min(100, Math.max(10, w.gpuUsage + (Math.random() - 0.5) * 8)) : w.gpuUsage,
          memoryUsage: w.status === 'active' ? Math.min(100, Math.max(10, w.memoryUsage + (Math.random() - 0.5) * 5)) : w.memoryUsage,
        }))
      );
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const filteredWorkers = workers.filter((w) => filter === 'all' || w.status === filter);
  const activeCount = workers.filter((w) => w.status === 'active').length;
  const totalTasks = workers.reduce((sum, w) => sum + w.tasksCompleted, 0);
  const queuedTasks = workers.reduce((sum, w) => sum + w.tasksQueued, 0);
  const avgCpu = workers.filter((w) => w.status === 'active').reduce((sum, w) => sum + w.cpuUsage, 0) / activeCount || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl md:text-3xl font-bold mb-2">
          <span className="bg-gradient-to-r from-sky-400 via-cyan-400 to-violet-400 bg-clip-text text-transparent">
            Compute Workers
          </span>
        </h1>
        <p className="text-zinc-400">
          Monitor and manage your distributed compute workers
        </p>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Active Workers</p>
            <Server className="text-emerald-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-emerald-400">
            {activeCount}<span className="text-lg text-zinc-500">/{workers.length}</span>
          </p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Total Completed</p>
            <CheckCircle2 className="text-sky-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-sky-400">{totalTasks.toLocaleString()}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Tasks Queued</p>
            <Timer className="text-amber-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-amber-400">{queuedTasks}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Avg CPU Usage</p>
            <TrendingUp className="text-violet-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-violet-400">{avgCpu.toFixed(1)}%</p>
        </motion.div>
      </div>

      {/* Filter Tabs */}
      <motion.div
        className="flex gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25 }}
      >
        {(['all', 'active', 'idle', 'error'] as const).map((f) => (
          <Button
            key={f}
            size="sm"
            variant="flat"
            className={
              filter === f
                ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                : 'bg-white/5 text-zinc-400 hover:text-white'
            }
            onPress={() => setFilter(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f !== 'all' && (
              <span className="ml-1 text-xs opacity-70">
                ({workers.filter((w) => w.status === f).length})
              </span>
            )}
          </Button>
        ))}
      </motion.div>

      {/* Workers Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        <AnimatePresence>
          {filteredWorkers.map((worker, index) => (
            <WorkerCard key={worker.id} worker={worker} index={index} />
          ))}
        </AnimatePresence>
      </div>

      {filteredWorkers.length === 0 && (
        <motion.div
          className="crystal-card p-8 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Activity className="mx-auto text-zinc-600 mb-3" size={40} />
          <p className="text-zinc-400">No workers match the current filter</p>
        </motion.div>
      )}
    </div>
  );
}

export default WorkersPanel;

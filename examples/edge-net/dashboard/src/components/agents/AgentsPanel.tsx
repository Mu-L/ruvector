import { useState } from 'react';
import { Card, CardBody, Progress, Button, Chip } from '@heroui/react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Search,
  Code2,
  TestTube2,
  FileSearch,
  Lightbulb,
  Sparkles,
  Zap,
  Activity,
  Network,
  CircleDot,
  GitBranch,
  Workflow,
  Eye,
} from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: 'researcher' | 'coder' | 'tester' | 'analyst' | 'optimizer' | 'coordinator';
  status: 'idle' | 'working' | 'learning' | 'coordinating';
  neuralPattern: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'adaptive';
  patternStrength: number;
  tasksCompleted: number;
  currentTask?: string;
  learningProgress: number;
  connections: string[];
  lastActive: string;
}

const agentTypeConfig = {
  researcher: {
    icon: Search,
    color: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
    gradient: 'from-sky-500 to-cyan-500',
  },
  coder: {
    icon: Code2,
    color: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
    gradient: 'from-violet-500 to-purple-500',
  },
  tester: {
    icon: TestTube2,
    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    gradient: 'from-emerald-500 to-green-500',
  },
  analyst: {
    icon: FileSearch,
    color: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    gradient: 'from-amber-500 to-orange-500',
  },
  optimizer: {
    icon: Zap,
    color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
    gradient: 'from-cyan-500 to-blue-500',
  },
  coordinator: {
    icon: Workflow,
    color: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
    gradient: 'from-pink-500 to-rose-500',
  },
};

const statusConfig = {
  idle: { color: 'text-zinc-400', bgColor: 'bg-zinc-500', label: 'Idle' },
  working: { color: 'text-emerald-400', bgColor: 'bg-emerald-500', label: 'Working' },
  learning: { color: 'text-violet-400', bgColor: 'bg-violet-500', label: 'Learning' },
  coordinating: { color: 'text-cyan-400', bgColor: 'bg-cyan-500', label: 'Coordinating' },
};

const patternConfig = {
  convergent: { label: 'Convergent', color: 'text-sky-400', description: 'Focused analytical' },
  divergent: { label: 'Divergent', color: 'text-violet-400', description: 'Creative exploration' },
  lateral: { label: 'Lateral', color: 'text-amber-400', description: 'Novel connections' },
  systems: { label: 'Systems', color: 'text-emerald-400', description: 'Holistic view' },
  adaptive: { label: 'Adaptive', color: 'text-cyan-400', description: 'Dynamic switching' },
};

const mockAgents: Agent[] = [
  {
    id: 'a1',
    name: 'Research Alpha',
    type: 'researcher',
    status: 'working',
    neuralPattern: 'divergent',
    patternStrength: 87,
    tasksCompleted: 234,
    currentTask: 'Analyzing API documentation patterns',
    learningProgress: 72,
    connections: ['a2', 'a4'],
    lastActive: 'now',
  },
  {
    id: 'a2',
    name: 'Coder Prime',
    type: 'coder',
    status: 'working',
    neuralPattern: 'convergent',
    patternStrength: 94,
    tasksCompleted: 567,
    currentTask: 'Implementing WASM bindings',
    learningProgress: 88,
    connections: ['a1', 'a3'],
    lastActive: 'now',
  },
  {
    id: 'a3',
    name: 'Test Guardian',
    type: 'tester',
    status: 'learning',
    neuralPattern: 'systems',
    patternStrength: 76,
    tasksCompleted: 189,
    learningProgress: 45,
    connections: ['a2'],
    lastActive: '2s ago',
  },
  {
    id: 'a4',
    name: 'Insight Analyst',
    type: 'analyst',
    status: 'idle',
    neuralPattern: 'lateral',
    patternStrength: 82,
    tasksCompleted: 312,
    learningProgress: 91,
    connections: ['a1', 'a5'],
    lastActive: '15s ago',
  },
  {
    id: 'a5',
    name: 'Perf Optimizer',
    type: 'optimizer',
    status: 'working',
    neuralPattern: 'adaptive',
    patternStrength: 89,
    tasksCompleted: 156,
    currentTask: 'Optimizing vector search latency',
    learningProgress: 67,
    connections: ['a4', 'a6'],
    lastActive: 'now',
  },
  {
    id: 'a6',
    name: 'Swarm Coord',
    type: 'coordinator',
    status: 'coordinating',
    neuralPattern: 'systems',
    patternStrength: 95,
    tasksCompleted: 423,
    currentTask: 'Orchestrating multi-agent workflow',
    learningProgress: 83,
    connections: ['a1', 'a2', 'a3', 'a4', 'a5'],
    lastActive: 'now',
  },
];

function NeuralPatternIndicator({ pattern, strength }: { pattern: Agent['neuralPattern']; strength: number }) {
  const config = patternConfig[pattern];

  return (
    <motion.div
      className="relative"
      whileHover={{ scale: 1.05 }}
    >
      <div className="flex items-center gap-2 p-2 rounded-lg bg-zinc-800/50 border border-white/5">
        <div className="relative">
          <Brain className={config.color} size={16} />
          <motion.div
            className={`absolute inset-0 ${config.color.replace('text-', 'bg-').replace('400', '500')}/30 rounded-full blur-sm`}
            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0.2, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
        <div>
          <p className={`text-xs font-medium ${config.color}`}>{config.label}</p>
          <p className="text-[10px] text-zinc-500">{strength}% strength</p>
        </div>
      </div>
    </motion.div>
  );
}

function AgentCard({ agent, agents, index }: { agent: Agent; agents: Agent[]; index: number }) {
  const typeConf = agentTypeConfig[agent.type];
  const statusConf = statusConfig[agent.status];
  const TypeIcon = typeConf.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className="bg-zinc-900/50 backdrop-blur-xl border border-white/10 hover:border-white/20 transition-all overflow-hidden">
        {/* Status Bar */}
        <div className={`h-1 bg-gradient-to-r ${typeConf.gradient}`} />

        <CardBody className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className={`w-12 h-12 rounded-xl ${typeConf.color} border flex items-center justify-center`}>
                  <TypeIcon size={24} />
                </div>
                <motion.div
                  className={`absolute -bottom-1 -right-1 w-3.5 h-3.5 rounded-full ${statusConf.bgColor} border-2 border-zinc-900`}
                  animate={agent.status !== 'idle' ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              </div>
              <div>
                <h4 className="font-semibold text-white">{agent.name}</h4>
                <div className="flex items-center gap-2 mt-0.5">
                  <Chip size="sm" className={`${typeConf.color} border text-xs capitalize`}>
                    {agent.type}
                  </Chip>
                  <span className={`text-xs ${statusConf.color} flex items-center gap-1`}>
                    <CircleDot size={10} />
                    {statusConf.label}
                  </span>
                </div>
              </div>
            </div>

            <Button isIconOnly size="sm" variant="flat" className="bg-white/5 text-zinc-400">
              <Eye size={14} />
            </Button>
          </div>

          {/* Current Task */}
          {agent.currentTask && (
            <motion.div
              className="mb-4 p-3 rounded-lg bg-zinc-800/50 border border-white/5"
              animate={{ borderColor: ['rgba(255,255,255,0.05)', 'rgba(255,255,255,0.1)', 'rgba(255,255,255,0.05)'] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <div className="flex items-center gap-2 mb-1">
                <Activity size={12} className="text-emerald-400" />
                <span className="text-xs text-zinc-400">Current Task</span>
              </div>
              <p className="text-sm text-white line-clamp-1">{agent.currentTask}</p>
            </motion.div>
          )}

          {/* Neural Pattern */}
          <div className="mb-4">
            <NeuralPatternIndicator pattern={agent.neuralPattern} strength={agent.patternStrength} />
          </div>

          {/* Learning Progress */}
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-zinc-400 flex items-center gap-1">
                <Sparkles size={12} /> Learning Progress
              </span>
              <span className="text-violet-400">{agent.learningProgress}%</span>
            </div>
            <Progress
              value={agent.learningProgress}
              classNames={{
                indicator: `bg-gradient-to-r ${typeConf.gradient}`,
                track: 'bg-zinc-800',
              }}
              size="sm"
            />
          </div>

          {/* Connections */}
          <div className="flex items-center justify-between pt-3 border-t border-white/10">
            <div className="flex items-center gap-1">
              <GitBranch size={14} className="text-zinc-500" />
              <div className="flex -space-x-2">
                {agent.connections.slice(0, 3).map((connId) => {
                  const connAgent = agents.find((a) => a.id === connId);
                  if (!connAgent) return null;
                  const ConnIcon = agentTypeConfig[connAgent.type].icon;
                  return (
                    <div
                      key={connId}
                      className={`w-6 h-6 rounded-full ${agentTypeConfig[connAgent.type].color} border border-zinc-900 flex items-center justify-center`}
                    >
                      <ConnIcon size={10} />
                    </div>
                  );
                })}
                {agent.connections.length > 3 && (
                  <div className="w-6 h-6 rounded-full bg-zinc-700 border border-zinc-900 flex items-center justify-center text-[10px] text-zinc-400">
                    +{agent.connections.length - 3}
                  </div>
                )}
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-semibold text-white">{agent.tasksCompleted}</p>
              <p className="text-[10px] text-zinc-500">tasks</p>
            </div>
          </div>
        </CardBody>
      </Card>
    </motion.div>
  );
}

function CoordinationVisualization({ agents }: { agents: Agent[] }) {
  return (
    <motion.div
      className="crystal-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
    >
      <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
        <Network size={16} />
        Agent Coordination Network
      </h3>

      <div className="relative h-48 flex items-center justify-center">
        {/* Central coordinator */}
        <motion.div
          className="absolute w-16 h-16 rounded-full bg-gradient-to-br from-pink-500/20 to-rose-500/20 border border-pink-500/30 flex items-center justify-center z-10"
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ duration: 3, repeat: Infinity }}
        >
          <Workflow className="text-pink-400" size={24} />
        </motion.div>

        {/* Connected agents */}
        {agents.filter(a => a.type !== 'coordinator').map((agent, i) => {
          const angle = (i * 360) / 5 - 90;
          const radius = 80;
          const x = Math.cos((angle * Math.PI) / 180) * radius;
          const y = Math.sin((angle * Math.PI) / 180) * radius;
          const TypeIcon = agentTypeConfig[agent.type].icon;

          return (
            <motion.div
              key={agent.id}
              className={`absolute w-10 h-10 rounded-full ${agentTypeConfig[agent.type].color} border flex items-center justify-center`}
              style={{ left: `calc(50% + ${x}px - 20px)`, top: `calc(50% + ${y}px - 20px)` }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5 + i * 0.1 }}
            >
              <TypeIcon size={16} />

              {/* Connection line */}
              <svg
                className="absolute"
                style={{
                  width: radius,
                  height: 2,
                  left: x > 0 ? -radius + 20 : 20,
                  top: '50%',
                  transform: `rotate(${angle}deg)`,
                  transformOrigin: x > 0 ? 'right center' : 'left center',
                }}
              >
                <motion.line
                  x1="0"
                  y1="1"
                  x2={radius - 40}
                  y2="1"
                  stroke={agent.status !== 'idle' ? '#06b6d4' : '#3f3f46'}
                  strokeWidth="2"
                  strokeDasharray="4 4"
                  animate={agent.status !== 'idle' ? { strokeDashoffset: [0, -8] } : {}}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
              </svg>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

export function AgentsPanel() {
  const [agents] = useState<Agent[]>(mockAgents);
  const [filter, setFilter] = useState<'all' | Agent['type']>('all');

  const workingCount = agents.filter((a) => a.status === 'working').length;
  const learningCount = agents.filter((a) => a.status === 'learning').length;
  const totalTasks = agents.reduce((sum, a) => sum + a.tasksCompleted, 0);
  const avgLearning = agents.reduce((sum, a) => sum + a.learningProgress, 0) / agents.length;

  const filteredAgents = filter === 'all' ? agents : agents.filter((a) => a.type === filter);

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl md:text-3xl font-bold mb-2">
          <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            AI Agents
          </span>
        </h1>
        <p className="text-zinc-400">
          Monitor and coordinate your AI agent swarm
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
            <p className="text-sm text-zinc-400">Working</p>
            <Activity className="text-emerald-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-emerald-400">
            {workingCount}<span className="text-lg text-zinc-500">/{agents.length}</span>
          </p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Learning</p>
            <Sparkles className="text-violet-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-violet-400">{learningCount}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Tasks Done</p>
            <Lightbulb className="text-sky-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-sky-400">{totalTasks.toLocaleString()}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Avg Learning</p>
            <Brain className="text-cyan-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-cyan-400">{avgLearning.toFixed(0)}%</p>
        </motion.div>
      </div>

      {/* Coordination Visualization */}
      <CoordinationVisualization agents={agents} />

      {/* Filter Tabs */}
      <motion.div
        className="flex flex-wrap gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25 }}
      >
        <Button
          size="sm"
          variant="flat"
          className={
            filter === 'all'
              ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
              : 'bg-white/5 text-zinc-400 hover:text-white'
          }
          onPress={() => setFilter('all')}
        >
          All Agents
        </Button>
        {(Object.keys(agentTypeConfig) as Agent['type'][]).map((type) => {
          const config = agentTypeConfig[type];
          const TypeIcon = config.icon;
          return (
            <Button
              key={type}
              size="sm"
              variant="flat"
              startContent={<TypeIcon size={14} />}
              className={
                filter === type
                  ? `${config.color} border`
                  : 'bg-white/5 text-zinc-400 hover:text-white'
              }
              onPress={() => setFilter(type)}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </Button>
          );
        })}
      </motion.div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        <AnimatePresence>
          {filteredAgents.map((agent, index) => (
            <AgentCard key={agent.id} agent={agent} agents={agents} index={index} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default AgentsPanel;

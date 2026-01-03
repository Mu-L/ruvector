import { useState } from 'react';
import { Card, CardBody, Button, Progress, Chip } from '@heroui/react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dna,
  Sparkles,
  GitBranch,
  GitMerge,
  Shield,
  ShieldCheck,
  Clock,
  Copy,
  Check,
  Fingerprint,
  Binary,
  Layers,
  Network,
  Zap,
  Crown,
  Leaf,
  TreeDeciduous,
} from 'lucide-react';

interface GenesisNode {
  id: string;
  name: string;
  generation: number;
  parentId: string | null;
  signature: string;
  birthTime: string;
  status: 'active' | 'dormant' | 'reproducing';
  genome: {
    computeCapacity: number;
    networkSpeed: number;
    reliability: number;
    specialization: string;
  };
  offspring: number;
  verified: boolean;
}

const mockGenesisNodes: GenesisNode[] = [
  {
    id: 'gen-0',
    name: 'Genesis Prime',
    generation: 0,
    parentId: null,
    signature: '0x7f4e2b1c9d8a3f6e5c4b2a1098765432fedcba98',
    birthTime: '2024-01-01T00:00:00Z',
    status: 'active',
    genome: {
      computeCapacity: 100,
      networkSpeed: 95,
      reliability: 99,
      specialization: 'coordinator',
    },
    offspring: 3,
    verified: true,
  },
  {
    id: 'gen-1a',
    name: 'Alpha Node',
    generation: 1,
    parentId: 'gen-0',
    signature: '0x3a2b1c4d5e6f7890abcdef1234567890abcdef12',
    birthTime: '2024-02-15T10:30:00Z',
    status: 'reproducing',
    genome: {
      computeCapacity: 92,
      networkSpeed: 88,
      reliability: 96,
      specialization: 'compute',
    },
    offspring: 2,
    verified: true,
  },
  {
    id: 'gen-1b',
    name: 'Beta Node',
    generation: 1,
    parentId: 'gen-0',
    signature: '0x9f8e7d6c5b4a3210fedcba0987654321abcdef34',
    birthTime: '2024-03-01T14:45:00Z',
    status: 'active',
    genome: {
      computeCapacity: 85,
      networkSpeed: 97,
      reliability: 94,
      specialization: 'network',
    },
    offspring: 1,
    verified: true,
  },
  {
    id: 'gen-1c',
    name: 'Gamma Node',
    generation: 1,
    parentId: 'gen-0',
    signature: '0x1234abcd5678efgh9012ijkl3456mnop7890qrst',
    birthTime: '2024-03-20T08:15:00Z',
    status: 'dormant',
    genome: {
      computeCapacity: 78,
      networkSpeed: 82,
      reliability: 91,
      specialization: 'storage',
    },
    offspring: 0,
    verified: true,
  },
  {
    id: 'gen-2a',
    name: 'Delta Prime',
    generation: 2,
    parentId: 'gen-1a',
    signature: '0xaabbccdd11223344eeffaabb55667788ccddeeff',
    birthTime: '2024-04-10T16:20:00Z',
    status: 'active',
    genome: {
      computeCapacity: 95,
      networkSpeed: 90,
      reliability: 97,
      specialization: 'inference',
    },
    offspring: 0,
    verified: true,
  },
  {
    id: 'gen-2b',
    name: 'Epsilon',
    generation: 2,
    parentId: 'gen-1a',
    signature: '0x99887766554433221100ffeeddccbbaa00112233',
    birthTime: '2024-05-05T11:00:00Z',
    status: 'active',
    genome: {
      computeCapacity: 88,
      networkSpeed: 93,
      reliability: 95,
      specialization: 'training',
    },
    offspring: 0,
    verified: false,
  },
];

const statusConfig = {
  active: { color: 'text-emerald-400', bgColor: 'bg-emerald-500', label: 'Active' },
  dormant: { color: 'text-zinc-400', bgColor: 'bg-zinc-500', label: 'Dormant' },
  reproducing: { color: 'text-violet-400', bgColor: 'bg-violet-500', label: 'Reproducing' },
};

const specializationConfig = {
  coordinator: { color: 'bg-pink-500/20 text-pink-400 border-pink-500/30', icon: Crown },
  compute: { color: 'bg-sky-500/20 text-sky-400 border-sky-500/30', icon: Zap },
  network: { color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30', icon: Network },
  storage: { color: 'bg-amber-500/20 text-amber-400 border-amber-500/30', icon: Layers },
  inference: { color: 'bg-violet-500/20 text-violet-400 border-violet-500/30', icon: Sparkles },
  training: { color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', icon: Binary },
};

function GenomeDisplay({ genome, compact = false }: { genome: GenesisNode['genome']; compact?: boolean }) {
  const metrics = [
    { key: 'computeCapacity', label: 'Compute', value: genome.computeCapacity, color: 'from-sky-500 to-cyan-500' },
    { key: 'networkSpeed', label: 'Network', value: genome.networkSpeed, color: 'from-violet-500 to-purple-500' },
    { key: 'reliability', label: 'Reliability', value: genome.reliability, color: 'from-emerald-500 to-green-500' },
  ];

  if (compact) {
    return (
      <div className="flex gap-2">
        {metrics.map((m) => (
          <div key={m.key} className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${m.color}`} />
            <span className="text-xs text-zinc-400">{m.value}%</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {metrics.map((m) => (
        <div key={m.key}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">{m.label}</span>
            <span className="text-white">{m.value}%</span>
          </div>
          <Progress
            value={m.value}
            classNames={{
              indicator: `bg-gradient-to-r ${m.color}`,
              track: 'bg-zinc-800',
            }}
            size="sm"
          />
        </div>
      ))}
    </div>
  );
}

function LineageNode({ node, nodes, depth = 0, isLast = false }: { node: GenesisNode; nodes: GenesisNode[]; depth?: number; isLast?: boolean }) {
  const children = nodes.filter((n) => n.parentId === node.id);
  const status = statusConfig[node.status];
  const spec = specializationConfig[node.genome.specialization as keyof typeof specializationConfig] || specializationConfig.compute;
  const SpecIcon = spec.icon;
  const [copied, setCopied] = useState(false);

  const copySignature = () => {
    navigator.clipboard.writeText(node.signature);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: depth * 0.1 }}
    >
      <div className="flex items-start gap-3">
        {/* Tree connector */}
        {depth > 0 && (
          <div className="flex flex-col items-center w-6 -ml-3">
            <div className="w-px h-4 bg-zinc-700" />
            <div className={`w-3 h-px bg-zinc-700 ${isLast ? '' : ''}`} />
          </div>
        )}

        {/* Node card */}
        <div className={`flex-1 crystal-card p-4 border ${node.generation === 0 ? 'border-amber-500/30' : 'border-white/10'}`}>
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className={`w-10 h-10 rounded-lg ${spec.color} border flex items-center justify-center`}>
                  <SpecIcon size={20} />
                </div>
                {node.generation === 0 && (
                  <Crown className="absolute -top-2 -right-2 text-amber-400" size={14} />
                )}
                <motion.div
                  className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full ${status.bgColor} border-2 border-zinc-900`}
                  animate={node.status === 'reproducing' ? { scale: [1, 1.3, 1] } : {}}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="font-semibold text-white">{node.name}</h4>
                  {node.verified && <ShieldCheck size={14} className="text-emerald-400" />}
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <Chip size="sm" className="text-xs bg-zinc-800 text-zinc-400">
                    Gen {node.generation}
                  </Chip>
                  <span className={`text-xs ${status.color}`}>{status.label}</span>
                </div>
              </div>
            </div>
            <GenomeDisplay genome={node.genome} compact />
          </div>

          {/* Signature */}
          <div className="flex items-center gap-2 p-2 rounded-lg bg-zinc-800/50 border border-white/5 mb-3">
            <Fingerprint size={14} className="text-cyan-400 flex-shrink-0" />
            <code className="text-xs text-zinc-400 font-mono truncate flex-1">
              {node.signature.slice(0, 10)}...{node.signature.slice(-8)}
            </code>
            <button
              onClick={copySignature}
              className={`p-1 rounded ${copied ? 'text-emerald-400' : 'text-zinc-500 hover:text-white'}`}
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <Clock size={12} />
              {new Date(node.birthTime).toLocaleDateString()}
            </span>
            <span className="flex items-center gap-1">
              <Leaf size={12} />
              {node.offspring} offspring
            </span>
          </div>
        </div>
      </div>

      {/* Children */}
      {children.length > 0 && (
        <div className="ml-6 mt-2 space-y-2 border-l border-zinc-700 pl-4">
          {children.map((child, idx) => (
            <LineageNode
              key={child.id}
              node={child}
              nodes={nodes}
              depth={depth + 1}
              isLast={idx === children.length - 1}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
}

function GenesisControls({ onAction }: { onAction: (action: string) => void }) {
  return (
    <motion.div
      className="crystal-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
    >
      <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
        <Dna size={16} />
        Genesis Controls
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Button
          variant="flat"
          className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 h-auto py-4 flex-col gap-2"
          onPress={() => onAction('spawn')}
        >
          <Sparkles size={20} />
          <span className="text-xs">Spawn Node</span>
        </Button>

        <Button
          variant="flat"
          className="bg-violet-500/20 text-violet-400 border border-violet-500/30 h-auto py-4 flex-col gap-2"
          onPress={() => onAction('reproduce')}
        >
          <GitMerge size={20} />
          <span className="text-xs">Reproduce</span>
        </Button>

        <Button
          variant="flat"
          className="bg-sky-500/20 text-sky-400 border border-sky-500/30 h-auto py-4 flex-col gap-2"
          onPress={() => onAction('verify')}
        >
          <Shield size={20} />
          <span className="text-xs">Verify All</span>
        </Button>

        <Button
          variant="flat"
          className="bg-amber-500/20 text-amber-400 border border-amber-500/30 h-auto py-4 flex-col gap-2"
          onPress={() => onAction('evolve')}
        >
          <Zap size={20} />
          <span className="text-xs">Evolve</span>
        </Button>
      </div>
    </motion.div>
  );
}

function NetworkGenomeCard({ nodes }: { nodes: GenesisNode[] }) {
  const avgCompute = nodes.reduce((sum, n) => sum + n.genome.computeCapacity, 0) / nodes.length;
  const avgNetwork = nodes.reduce((sum, n) => sum + n.genome.networkSpeed, 0) / nodes.length;
  const avgReliability = nodes.reduce((sum, n) => sum + n.genome.reliability, 0) / nodes.length;

  return (
    <motion.div
      className="crystal-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2 }}
    >
      <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
        <Binary size={16} />
        Network Genome
      </h3>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">Avg Compute Capacity</span>
            <span className="text-sky-400">{avgCompute.toFixed(1)}%</span>
          </div>
          <Progress
            value={avgCompute}
            classNames={{
              indicator: 'bg-gradient-to-r from-sky-500 to-cyan-500',
              track: 'bg-zinc-800',
            }}
          />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">Avg Network Speed</span>
            <span className="text-violet-400">{avgNetwork.toFixed(1)}%</span>
          </div>
          <Progress
            value={avgNetwork}
            classNames={{
              indicator: 'bg-gradient-to-r from-violet-500 to-purple-500',
              track: 'bg-zinc-800',
            }}
          />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">Avg Reliability</span>
            <span className="text-emerald-400">{avgReliability.toFixed(1)}%</span>
          </div>
          <Progress
            value={avgReliability}
            classNames={{
              indicator: 'bg-gradient-to-r from-emerald-500 to-green-500',
              track: 'bg-zinc-800',
            }}
          />
        </div>

        <div className="pt-3 border-t border-white/10">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-400">Genetic Diversity</span>
            <Chip size="sm" className="bg-amber-500/20 text-amber-400 border border-amber-500/30">
              High
            </Chip>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export function GenesisPanel() {
  const [nodes] = useState<GenesisNode[]>(mockGenesisNodes);
  const [selectedView, setSelectedView] = useState<'tree' | 'grid'>('tree');

  const rootNode = nodes.find((n) => n.parentId === null);
  const totalGenerations = Math.max(...nodes.map((n) => n.generation)) + 1;
  const activeCount = nodes.filter((n) => n.status === 'active').length;
  const verifiedCount = nodes.filter((n) => n.verified).length;

  const handleAction = (action: string) => {
    console.log(`Genesis action: ${action}`);
    // Handle genesis actions
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl md:text-3xl font-bold mb-2">
          <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-sky-400 bg-clip-text text-transparent">
            Network Genesis
          </span>
        </h1>
        <p className="text-zinc-400">
          Visualize and manage the evolutionary lineage of your network nodes
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
            <p className="text-sm text-zinc-400">Total Nodes</p>
            <TreeDeciduous className="text-emerald-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-emerald-400">{nodes.length}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Generations</p>
            <GitBranch className="text-violet-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-violet-400">{totalGenerations}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Active</p>
            <Zap className="text-sky-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-sky-400">
            {activeCount}<span className="text-lg text-zinc-500">/{nodes.length}</span>
          </p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Verified</p>
            <ShieldCheck className="text-cyan-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-cyan-400">
            {verifiedCount}<span className="text-lg text-zinc-500">/{nodes.length}</span>
          </p>
        </motion.div>
      </div>

      {/* Controls and Genome */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GenesisControls onAction={handleAction} />
        <NetworkGenomeCard nodes={nodes} />
      </div>

      {/* Lineage Tree */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.35 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <GitBranch className="text-violet-400" size={20} />
            Lineage Tree
          </h3>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="flat"
              className={selectedView === 'tree' ? 'bg-sky-500/20 text-sky-400' : 'bg-white/5 text-zinc-400'}
              onPress={() => setSelectedView('tree')}
            >
              Tree
            </Button>
            <Button
              size="sm"
              variant="flat"
              className={selectedView === 'grid' ? 'bg-sky-500/20 text-sky-400' : 'bg-white/5 text-zinc-400'}
              onPress={() => setSelectedView('grid')}
            >
              Grid
            </Button>
          </div>
        </div>

        {selectedView === 'tree' && rootNode && (
          <div className="space-y-4">
            <LineageNode node={rootNode} nodes={nodes} />
          </div>
        )}

        {selectedView === 'grid' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <AnimatePresence>
              {nodes.map((node, idx) => {
                const status = statusConfig[node.status];
                const spec = specializationConfig[node.genome.specialization as keyof typeof specializationConfig] || specializationConfig.compute;
                const SpecIcon = spec.icon;

                return (
                  <motion.div
                    key={node.id}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    <Card className="bg-zinc-900/50 border border-white/10">
                      <CardBody className="p-4">
                        <div className="flex items-center gap-3 mb-3">
                          <div className={`w-10 h-10 rounded-lg ${spec.color} border flex items-center justify-center`}>
                            <SpecIcon size={20} />
                          </div>
                          <div>
                            <h4 className="font-medium text-white flex items-center gap-2">
                              {node.name}
                              {node.verified && <ShieldCheck size={12} className="text-emerald-400" />}
                            </h4>
                            <p className={`text-xs ${status.color}`}>Gen {node.generation} - {status.label}</p>
                          </div>
                        </div>
                        <GenomeDisplay genome={node.genome} />
                      </CardBody>
                    </Card>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </motion.div>
    </div>
  );
}

export default GenesisPanel;

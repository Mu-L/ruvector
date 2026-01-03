import { useState } from 'react';
import { Card, CardBody, Button, Switch, Chip } from '@heroui/react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Puzzle,
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldBan,
  Download,
  Trash2,
  Settings2,
  Star,
  Package,
  Zap,
  Brain,
  Network,
  Check,
  AlertTriangle,
  Search,
  Filter,
  Store,
  TrendingUp,
} from 'lucide-react';

interface Plugin {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
  category: 'ai' | 'network' | 'security' | 'utility' | 'compute';
  status: 'installed' | 'available' | 'update-available';
  securityStatus: 'verified' | 'quarantine' | 'pending' | 'unverified';
  enabled: boolean;
  downloads: number;
  rating: number;
  size: string;
}

const categoryConfig = {
  ai: { icon: Brain, color: 'bg-violet-500/20 text-violet-400 border-violet-500/30' },
  network: { icon: Network, color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' },
  security: { icon: Shield, color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' },
  utility: { icon: Settings2, color: 'bg-amber-500/20 text-amber-400 border-amber-500/30' },
  compute: { icon: Zap, color: 'bg-sky-500/20 text-sky-400 border-sky-500/30' },
};

const securityConfig = {
  verified: { icon: ShieldCheck, color: 'text-emerald-400', bgColor: 'bg-emerald-500/20', label: 'Verified' },
  quarantine: { icon: ShieldBan, color: 'text-red-400', bgColor: 'bg-red-500/20', label: 'Quarantined' },
  pending: { icon: ShieldAlert, color: 'text-amber-400', bgColor: 'bg-amber-500/20', label: 'Pending Review' },
  unverified: { icon: Shield, color: 'text-zinc-400', bgColor: 'bg-zinc-500/20', label: 'Unverified' },
};

const mockPlugins: Plugin[] = [
  {
    id: 'p1',
    name: 'Neural Swarm Optimizer',
    description: 'Advanced neural network optimization using swarm intelligence algorithms',
    author: 'RuVector Labs',
    version: '2.1.0',
    category: 'ai',
    status: 'installed',
    securityStatus: 'verified',
    enabled: true,
    downloads: 12450,
    rating: 4.8,
    size: '2.4 MB',
  },
  {
    id: 'p2',
    name: 'WebRTC Mesh',
    description: 'Peer-to-peer mesh networking for distributed compute',
    author: 'Edge Collective',
    version: '1.5.2',
    category: 'network',
    status: 'installed',
    securityStatus: 'verified',
    enabled: true,
    downloads: 8920,
    rating: 4.6,
    size: '1.8 MB',
  },
  {
    id: 'p3',
    name: 'Zero-Knowledge Proofs',
    description: 'Cryptographic verification without revealing underlying data',
    author: 'CryptoCore',
    version: '3.0.1',
    category: 'security',
    status: 'installed',
    securityStatus: 'verified',
    enabled: false,
    downloads: 5670,
    rating: 4.9,
    size: '3.2 MB',
  },
  {
    id: 'p4',
    name: 'WASM Compiler',
    description: 'On-the-fly WebAssembly compilation and optimization',
    author: 'ByteForge',
    version: '1.2.0',
    category: 'compute',
    status: 'update-available',
    securityStatus: 'verified',
    enabled: true,
    downloads: 15230,
    rating: 4.7,
    size: '4.1 MB',
  },
  {
    id: 'p5',
    name: 'Task Scheduler Pro',
    description: 'Advanced task scheduling with priority queues and load balancing',
    author: 'WorkflowKit',
    version: '2.0.0',
    category: 'utility',
    status: 'available',
    securityStatus: 'pending',
    enabled: false,
    downloads: 3450,
    rating: 4.4,
    size: '1.1 MB',
  },
  {
    id: 'p6',
    name: 'Quantum Simulator',
    description: 'Simulate quantum circuits for algorithm development',
    author: 'QuantumLabs',
    version: '0.9.0',
    category: 'compute',
    status: 'available',
    securityStatus: 'unverified',
    enabled: false,
    downloads: 890,
    rating: 4.2,
    size: '5.6 MB',
  },
  {
    id: 'p7',
    name: 'Suspicious Plugin',
    description: 'This plugin was flagged for suspicious behavior',
    author: 'Unknown',
    version: '1.0.0',
    category: 'utility',
    status: 'installed',
    securityStatus: 'quarantine',
    enabled: false,
    downloads: 120,
    rating: 2.1,
    size: '0.8 MB',
  },
];

function PluginCard({ plugin, onToggle, onAction }: { plugin: Plugin; onToggle: () => void; onAction: (action: string) => void }) {
  const category = categoryConfig[plugin.category];
  const security = securityConfig[plugin.securityStatus];
  const CategoryIcon = category.icon;
  const SecurityIcon = security.icon;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
    >
      <Card className={`bg-zinc-900/50 backdrop-blur-xl border ${plugin.securityStatus === 'quarantine' ? 'border-red-500/30' : 'border-white/10'} hover:border-white/20 transition-all`}>
        <CardBody className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-3 mb-3">
            <div className="flex items-start gap-3 flex-1 min-w-0">
              <div className={`w-12 h-12 rounded-xl ${category.color} border flex items-center justify-center flex-shrink-0`}>
                <CategoryIcon size={24} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h4 className="font-semibold text-white truncate">{plugin.name}</h4>
                  <Chip size="sm" className="text-xs bg-zinc-800 text-zinc-400">
                    v{plugin.version}
                  </Chip>
                </div>
                <p className="text-xs text-zinc-500 mt-0.5">by {plugin.author}</p>
              </div>
            </div>

            {plugin.status === 'installed' && plugin.securityStatus !== 'quarantine' && (
              <Switch
                isSelected={plugin.enabled}
                onValueChange={onToggle}
                size="sm"
                classNames={{
                  wrapper: 'bg-zinc-700 group-data-[selected=true]:bg-emerald-500',
                }}
              />
            )}
          </div>

          {/* Description */}
          <p className="text-sm text-zinc-400 mb-3 line-clamp-2">{plugin.description}</p>

          {/* Security Status */}
          <div className={`flex items-center gap-2 p-2 rounded-lg ${security.bgColor} mb-3`}>
            <SecurityIcon size={16} className={security.color} />
            <span className={`text-xs font-medium ${security.color}`}>{security.label}</span>
            {plugin.securityStatus === 'quarantine' && (
              <span className="text-xs text-red-400 ml-auto">Disabled for safety</span>
            )}
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4 text-xs text-zinc-500 mb-3">
            <span className="flex items-center gap-1">
              <Download size={12} />
              {plugin.downloads.toLocaleString()}
            </span>
            <span className="flex items-center gap-1">
              <Star size={12} className="text-amber-400" />
              {plugin.rating}
            </span>
            <span className="flex items-center gap-1">
              <Package size={12} />
              {plugin.size}
            </span>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-3 border-t border-white/10">
            {plugin.status === 'available' && (
              <Button
                size="sm"
                variant="flat"
                className="flex-1 bg-sky-500/20 text-sky-400"
                startContent={<Download size={14} />}
                onPress={() => onAction('install')}
              >
                Install
              </Button>
            )}
            {plugin.status === 'update-available' && (
              <Button
                size="sm"
                variant="flat"
                className="flex-1 bg-violet-500/20 text-violet-400"
                startContent={<TrendingUp size={14} />}
                onPress={() => onAction('update')}
              >
                Update
              </Button>
            )}
            {plugin.status === 'installed' && plugin.securityStatus !== 'quarantine' && (
              <Button
                size="sm"
                variant="flat"
                className="flex-1 bg-white/5 text-zinc-400"
                startContent={<Settings2 size={14} />}
                onPress={() => onAction('settings')}
              >
                Settings
              </Button>
            )}
            {plugin.status === 'installed' && (
              <Button
                size="sm"
                isIconOnly
                variant="flat"
                className="bg-red-500/10 text-red-400"
                onPress={() => onAction('uninstall')}
              >
                <Trash2 size={14} />
              </Button>
            )}
          </div>
        </CardBody>
      </Card>
    </motion.div>
  );
}

export function PluginsPanel() {
  const [plugins, setPlugins] = useState<Plugin[]>(mockPlugins);
  const [filter, setFilter] = useState<'all' | 'installed' | 'available' | 'marketplace'>('all');
  const [categoryFilter, setCategoryFilter] = useState<'all' | Plugin['category']>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const installedCount = plugins.filter((p) => p.status === 'installed' || p.status === 'update-available').length;
  const verifiedCount = plugins.filter((p) => p.securityStatus === 'verified').length;
  const quarantinedCount = plugins.filter((p) => p.securityStatus === 'quarantine').length;
  const enabledCount = plugins.filter((p) => p.enabled).length;

  const handleToggle = (id: string) => {
    setPlugins((prev) =>
      prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p))
    );
  };

  const handleAction = (id: string, action: string) => {
    console.log(`Plugin ${id}: ${action}`);
    // Handle plugin actions here
  };

  const filteredPlugins = plugins.filter((p) => {
    if (filter === 'installed' && p.status !== 'installed' && p.status !== 'update-available') return false;
    if (filter === 'available' && p.status !== 'available') return false;
    if (categoryFilter !== 'all' && p.category !== categoryFilter) return false;
    if (searchQuery && !p.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl md:text-3xl font-bold mb-2">
          <span className="bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 bg-clip-text text-transparent">
            Plugin Manager
          </span>
        </h1>
        <p className="text-zinc-400">
          Extend Edge-Net with verified plugins and modules
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
            <p className="text-sm text-zinc-400">Installed</p>
            <Package className="text-sky-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-sky-400">
            {installedCount}<span className="text-lg text-zinc-500">/{plugins.length}</span>
          </p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Verified</p>
            <ShieldCheck className="text-emerald-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-emerald-400">{verifiedCount}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Enabled</p>
            <Check className="text-cyan-400" size={20} />
          </div>
          <p className="text-2xl font-bold text-cyan-400">{enabledCount}</p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-zinc-400">Quarantined</p>
            <ShieldBan className={quarantinedCount > 0 ? 'text-red-400' : 'text-zinc-500'} size={20} />
          </div>
          <p className={`text-2xl font-bold ${quarantinedCount > 0 ? 'text-red-400' : 'text-zinc-500'}`}>
            {quarantinedCount}
          </p>
        </motion.div>
      </div>

      {/* Search and Filters */}
      <motion.div
        className="crystal-card p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25 }}
      >
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" size={18} />
            <input
              type="text"
              placeholder="Search plugins..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-zinc-800 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-sky-500/50"
            />
          </div>

          {/* Status Filter */}
          <div className="flex gap-2">
            {(['all', 'installed', 'available', 'marketplace'] as const).map((f) => (
              <Button
                key={f}
                size="sm"
                variant="flat"
                startContent={f === 'marketplace' ? <Store size={14} /> : undefined}
                className={
                  filter === f
                    ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                    : 'bg-white/5 text-zinc-400 hover:text-white'
                }
                onPress={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </Button>
            ))}
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-white/10">
          <span className="text-xs text-zinc-500 flex items-center gap-1 mr-2">
            <Filter size={12} /> Category:
          </span>
          <Button
            size="sm"
            variant="flat"
            className={
              categoryFilter === 'all'
                ? 'bg-white/10 text-white'
                : 'bg-transparent text-zinc-400 hover:text-white'
            }
            onPress={() => setCategoryFilter('all')}
          >
            All
          </Button>
          {(Object.keys(categoryConfig) as Plugin['category'][]).map((cat) => {
            const config = categoryConfig[cat];
            const CatIcon = config.icon;
            return (
              <Button
                key={cat}
                size="sm"
                variant="flat"
                startContent={<CatIcon size={14} />}
                className={
                  categoryFilter === cat
                    ? `${config.color} border`
                    : 'bg-transparent text-zinc-400 hover:text-white'
                }
                onPress={() => setCategoryFilter(cat)}
              >
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </Button>
            );
          })}
        </div>
      </motion.div>

      {/* Security Notice */}
      {quarantinedCount > 0 && (
        <motion.div
          className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <AlertTriangle className="text-red-400 flex-shrink-0" size={20} />
          <div>
            <p className="text-sm font-medium text-red-400">Security Alert</p>
            <p className="text-xs text-zinc-400">
              {quarantinedCount} plugin(s) have been quarantined due to suspicious behavior.
              Review and remove them for safety.
            </p>
          </div>
        </motion.div>
      )}

      {/* Plugins Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        <AnimatePresence>
          {filteredPlugins.map((plugin) => (
            <PluginCard
              key={plugin.id}
              plugin={plugin}
              onToggle={() => handleToggle(plugin.id)}
              onAction={(action) => handleAction(plugin.id, action)}
            />
          ))}
        </AnimatePresence>
      </div>

      {filteredPlugins.length === 0 && (
        <motion.div
          className="crystal-card p-8 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Puzzle className="mx-auto text-zinc-600 mb-3" size={40} />
          <p className="text-zinc-400">No plugins match your filters</p>
        </motion.div>
      )}
    </div>
  );
}

export default PluginsPanel;

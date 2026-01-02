import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Zap, Clock, Gauge, Database, Globe } from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';
import { StatCard } from '../common/StatCard';

// Format uptime seconds to human readable
function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

// Session start time - only tracks current browser session
const sessionStart = Date.now();

export function NetworkStats() {
  const {
    stats,
    timeCrystal,
    isRelayConnected,
    connectedPeers,
    contributionSettings,
    isFirebaseConnected,
    isDemoMode,
    firebasePeers,
    firebaseStats,
  } = useNetworkStore();

  // Use React state for session-only uptime
  const [sessionUptime, setSessionUptime] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setSessionUptime((Date.now() - sessionStart) / 1000);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Compute actual network node count from Firebase + relay
  const networkNodeCount = Math.max(
    stats.activeNodes,
    firebaseStats?.activePeers || 0,
    connectedPeers.length + (contributionSettings.enabled ? 1 : 0)
  );

  // Determine data source indicator
  const dataSource = isFirebaseConnected
    ? 'Firebase'
    : isRelayConnected
    ? 'Relay'
    : isDemoMode
    ? 'Demo'
    : 'Local';

  const statItems = [
    {
      title: 'Network Nodes',
      value: networkNodeCount,
      icon: <Globe size={24} />,
      color: 'crystal' as const,
    },
    {
      title: 'Total Compute',
      value: `${stats.totalCompute.toFixed(1)} TFLOPS`,
      icon: <Cpu size={24} />,
      color: 'temporal' as const,
    },
    {
      title: 'Tasks Completed',
      value: stats.tasksCompleted,
      icon: <Activity size={24} />,
      color: 'quantum' as const,
    },
    {
      title: 'Credits Earned',
      value: `${stats.creditsEarned.toLocaleString()}`,
      icon: <Zap size={24} />,
      color: 'success' as const,
    },
    {
      title: 'Network Latency',
      value: `${stats.latency.toFixed(0)}ms`,
      icon: <Clock size={24} />,
      color: stats.latency < 50 ? 'success' as const : 'warning' as const,
    },
    {
      title: 'This Session',
      value: formatUptime(sessionUptime),
      icon: <Gauge size={24} />,
      color: 'success' as const,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Connection Status Banner - Always show when Firebase is connected or contributing */}
      {(contributionSettings.enabled || isFirebaseConnected || isDemoMode) && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-2"
        >
          {/* Main Connection Status */}
          <div
            className={`p-3 rounded-lg border flex items-center justify-between ${
              isRelayConnected || isFirebaseConnected
                ? 'bg-emerald-500/10 border-emerald-500/30'
                : isDemoMode
                ? 'bg-violet-500/10 border-violet-500/30'
                : 'bg-amber-500/10 border-amber-500/30'
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`w-2 h-2 rounded-full ${
                  isRelayConnected || isFirebaseConnected
                    ? 'bg-emerald-400 animate-pulse'
                    : isDemoMode
                    ? 'bg-violet-400 animate-pulse'
                    : 'bg-amber-400'
                }`}
              />
              <span
                className={
                  isRelayConnected || isFirebaseConnected
                    ? 'text-emerald-400'
                    : isDemoMode
                    ? 'text-violet-400'
                    : 'text-amber-400'
                }
              >
                {isFirebaseConnected
                  ? `Live Network Data (${networkNodeCount} nodes)`
                  : isRelayConnected
                  ? `Connected to Edge-Net (${connectedPeers.length + 1} nodes)`
                  : isDemoMode
                  ? 'Demo Mode - Simulated Data'
                  : 'Connecting to network...'}
              </span>
            </div>
            <div className="flex items-center gap-3">
              {/* Data Source Badge */}
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${
                  isFirebaseConnected
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : isDemoMode
                    ? 'bg-violet-500/20 text-violet-400'
                    : 'bg-zinc-500/20 text-zinc-400'
                }`}
              >
                {dataSource}
              </span>
              {isRelayConnected && (
                <span className="text-xs text-zinc-500">
                  wss://edge-net-relay-...
                </span>
              )}
            </div>
          </div>

          {/* Firebase Peers List (collapsed by default, expandable) */}
          {firebasePeers.length > 0 && (
            <div className="px-3 py-2 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <Database size={12} />
                <span>
                  {firebasePeers.filter((p) => p.online).length} online peers from Firestore
                </span>
                {firebasePeers.some((p) => p.isVerified) && (
                  <span className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded text-[10px]">
                    {firebasePeers.filter((p) => p.isVerified).length} verified
                  </span>
                )}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {statItems.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <StatCard {...stat} />
          </motion.div>
        ))}
      </div>

      {/* Time Crystal Status */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <motion.div
            className={`w-3 h-3 rounded-full ${
              isRelayConnected || isFirebaseConnected
                ? 'bg-gradient-to-r from-sky-400 to-violet-400'
                : isDemoMode
                ? 'bg-gradient-to-r from-violet-400 to-purple-400'
                : 'bg-zinc-500'
            }`}
            animate={isRelayConnected || isFirebaseConnected || isDemoMode ? { scale: [1, 1.2, 1] } : {}}
            transition={{ duration: 2, repeat: Infinity }}
          />
          Time Crystal Synchronization
          {!isRelayConnected && !isFirebaseConnected && contributionSettings.enabled && !isDemoMode && (
            <span className="text-xs text-amber-400 ml-2">(waiting for network)</span>
          )}
          {isDemoMode && (
            <span className="text-xs text-violet-400 ml-2">(demo data)</span>
          )}
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 rounded-lg bg-sky-500/10 border border-sky-500/20">
            <p className="text-2xl font-bold text-sky-400">
              {(timeCrystal.phase * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-zinc-400 mt-1">Phase</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-violet-500/10 border border-violet-500/20">
            <p className="text-2xl font-bold text-violet-400">
              {timeCrystal.frequency.toFixed(3)}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Frequency (phi)</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
            <p className="text-2xl font-bold text-cyan-400">
              {(timeCrystal.coherence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-zinc-400 mt-1">Coherence</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-2xl font-bold text-emerald-400">
              {Math.max(timeCrystal.synchronizedNodes, networkNodeCount)}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Synced Nodes</p>
          </div>
        </div>

        {/* Crystal Animation */}
        <div className="mt-6 h-2 bg-zinc-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-sky-500 via-violet-500 to-cyan-500"
            style={{ width: `${Math.max(timeCrystal.coherence, networkNodeCount > 0 ? 0.3 : 0) * 100}%` }}
            animate={{
              opacity: [0.7, 1, 0.7],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </div>
  );
}

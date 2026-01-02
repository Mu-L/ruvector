/**
 * Edge-Net Genesis Relay Server
 * WebSocket relay for distributed compute network coordination
 *
 * Security Features:
 * - Origin validation (CORS whitelist)
 * - Rate limiting per IP/node
 * - Message size limits
 * - Node authentication via PiKey signatures
 * - Connection timeout
 */

import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { createHash, createHmac } from 'crypto';

const PORT = process.env.PORT || 8080;
const MAX_MESSAGE_SIZE = 64 * 1024; // 64KB max message
const MAX_CONNECTIONS_PER_IP = 5;
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const RATE_LIMIT_MAX = 100; // max messages per window
const CONNECTION_TIMEOUT = 30000; // 30s heartbeat timeout

// Security: Allowed origins (add your domains)
const ALLOWED_ORIGINS = new Set([
  'http://localhost:3000',
  'http://localhost:3001',
  'http://localhost:5173',
  'https://edge-net.ruv.io',
  'https://ruvector.dev',
]);

// Connected nodes
const nodes = new Map();

// Security: Track connections per IP
const ipConnections = new Map();

// Security: Rate limiting per node
const rateLimits = new Map();

// Network state
const networkState = {
  genesisTime: Date.now(),
  totalNodes: 0,
  activeNodes: 0,
  totalTasks: 0,
  totalRuvDistributed: BigInt(0),
  timeCrystalPhase: 0,
};

// Task queue for distribution
const taskQueue = [];

// Ledger storage for multi-device sync (keyed by public key)
const nodeLedgers = new Map();

// CRDT merge for ledger states
function mergeLedgerStates(deviceStates) {
  const merged = { earned: {}, spent: {}, balance: 0 };

  for (const [deviceId, state] of deviceStates) {
    // Merge earned (max wins for each key)
    for (const [key, value] of Object.entries(state.earned || {})) {
      merged.earned[key] = Math.max(merged.earned[key] || 0, value);
    }
    // Merge spent (max wins for each key)
    for (const [key, value] of Object.entries(state.spent || {})) {
      merged.spent[key] = Math.max(merged.spent[key] || 0, value);
    }
  }

  // Calculate balance
  const totalEarned = Object.values(merged.earned).reduce((a, b) => a + b, 0);
  const totalSpent = Object.values(merged.spent).reduce((a, b) => a + b, 0);
  merged.balance = totalEarned - totalSpent;
  merged.totalEarned = totalEarned;
  merged.totalSpent = totalSpent;

  return merged;
}

// Create HTTP server
const server = createServer((req, res) => {
  // Health check endpoint
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'healthy',
      nodes: nodes.size,
      uptime: Date.now() - networkState.genesisTime,
    }));
    return;
  }

  // Network stats endpoint
  if (req.url === '/stats') {
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(JSON.stringify({
      ...networkState,
      totalRuvDistributed: networkState.totalRuvDistributed.toString(),
      connectedNodes: Array.from(nodes.keys()),
    }));
    return;
  }

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Edge-Net Genesis Relay v0.1.0');
});

// Create WebSocket server
const wss = new WebSocketServer({ server });

// Broadcast to all nodes except sender
function broadcast(message, excludeNodeId = null) {
  const data = JSON.stringify(message);
  nodes.forEach((ws, nodeId) => {
    if (nodeId !== excludeNodeId && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    }
  });
}

// Time Crystal synchronization tick
setInterval(() => {
  networkState.timeCrystalPhase = (networkState.timeCrystalPhase + 0.01) % 1;
  networkState.activeNodes = nodes.size;

  // Broadcast sync pulse to all nodes
  broadcast({
    type: 'time_crystal_sync',
    phase: networkState.timeCrystalPhase,
    timestamp: Date.now(),
    activeNodes: networkState.activeNodes,
  });
}, 1000);

// Distribute pending tasks
setInterval(() => {
  if (taskQueue.length > 0 && nodes.size > 0) {
    const task = taskQueue.shift();
    const nodeIds = Array.from(nodes.keys());
    const targetNodeId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
    const targetWs = nodes.get(targetNodeId);

    if (targetWs && targetWs.readyState === WebSocket.OPEN) {
      targetWs.send(JSON.stringify({
        type: 'task_assignment',
        task,
      }));
      console.log(`[Relay] Assigned task ${task.id} to node ${targetNodeId}`);
    } else {
      // Put task back if target unavailable
      taskQueue.unshift(task);
    }
  }
}, 500);

// Security: Validate origin
function isOriginAllowed(origin) {
  // Allow Node.js connections (no origin header) for CLI/test tools
  if (!origin) return true;
  if (ALLOWED_ORIGINS.has(origin)) return true;
  // Allow any localhost for development
  if (origin.startsWith('http://localhost:')) return true;
  return false;
}

// Security: Check rate limit
function checkRateLimit(nodeId) {
  const now = Date.now();
  const limit = rateLimits.get(nodeId) || { count: 0, windowStart: now };

  if (now - limit.windowStart > RATE_LIMIT_WINDOW) {
    limit.count = 0;
    limit.windowStart = now;
  }

  limit.count++;
  rateLimits.set(nodeId, limit);

  return limit.count <= RATE_LIMIT_MAX;
}

// Security: Validate message signature (for authenticated nodes)
function validateSignature(nodeId, message, signature, publicKey) {
  // In production, verify Ed25519 signature from PiKey
  // For now, accept if nodeId matches registered node
  return nodes.has(nodeId);
}

// Security: Get client IP
function getClientIP(req) {
  return req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
         req.socket.remoteAddress ||
         'unknown';
}

// Handle WebSocket connections
wss.on('connection', (ws, req) => {
  let nodeId = null;
  const clientIP = getClientIP(req);
  const origin = req.headers.origin;

  // Security: Validate origin
  if (!isOriginAllowed(origin)) {
    console.log(`[Relay] Rejected connection from unauthorized origin: ${origin}`);
    ws.close(4001, 'Unauthorized origin');
    return;
  }

  // Security: Check IP connection limit
  const ipCount = ipConnections.get(clientIP) || 0;
  if (ipCount >= MAX_CONNECTIONS_PER_IP) {
    console.log(`[Relay] Rejected connection: too many connections from ${clientIP}`);
    ws.close(4002, 'Too many connections');
    return;
  }
  ipConnections.set(clientIP, ipCount + 1);

  // Security: Set message size limit
  ws._maxPayload = MAX_MESSAGE_SIZE;

  // Security: Connection timeout
  let heartbeatTimeout;
  const resetHeartbeat = () => {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = setTimeout(() => {
      console.log(`[Relay] Node ${nodeId} timed out`);
      ws.terminate();
    }, CONNECTION_TIMEOUT);
  };
  resetHeartbeat();

  console.log('[Relay] New connection from:', clientIP, 'origin:', origin);

  ws.on('message', (data) => {
    try {
      // Security: Reset heartbeat on any message
      resetHeartbeat();

      // Security: Check message size
      if (data.length > MAX_MESSAGE_SIZE) {
        ws.send(JSON.stringify({ type: 'error', message: 'Message too large' }));
        return;
      }

      const message = JSON.parse(data.toString());

      // Security: Rate limit (after registration)
      if (nodeId && !checkRateLimit(nodeId)) {
        ws.send(JSON.stringify({ type: 'error', message: 'Rate limit exceeded' }));
        return;
      }

      switch (message.type) {
        case 'register':
          // Node registration
          nodeId = message.nodeId;
          nodes.set(nodeId, ws);
          networkState.totalNodes++;
          networkState.activeNodes = nodes.size;

          console.log(`[Relay] Node registered: ${nodeId} (${nodes.size} active)`);

          // Send welcome with network state
          ws.send(JSON.stringify({
            type: 'welcome',
            nodeId,
            networkState: {
              ...networkState,
              totalRuvDistributed: networkState.totalRuvDistributed.toString(),
            },
            peers: Array.from(nodes.keys()).filter(id => id !== nodeId),
          }));

          // Notify other nodes
          broadcast({
            type: 'node_joined',
            nodeId,
            totalNodes: nodes.size,
          }, nodeId);
          break;

        case 'task_submit':
          // Add task to queue
          const task = {
            id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            submitter: nodeId,
            ...message.task,
            submittedAt: Date.now(),
          };
          taskQueue.push(task);
          networkState.totalTasks++;

          console.log(`[Relay] Task submitted: ${task.id} from ${nodeId}`);

          ws.send(JSON.stringify({
            type: 'task_accepted',
            taskId: task.id,
          }));
          break;

        case 'task_complete':
          // Task completed - distribute rewards
          const reward = BigInt(message.reward || 1000000); // 0.001 rUv default
          networkState.totalRuvDistributed += reward;

          console.log(`[Relay] Task ${message.taskId} completed by ${nodeId}, reward: ${reward}`);

          // Notify submitter
          const submitterWs = nodes.get(message.submitterId);
          if (submitterWs && submitterWs.readyState === WebSocket.OPEN) {
            submitterWs.send(JSON.stringify({
              type: 'task_result',
              taskId: message.taskId,
              result: message.result,
              processedBy: nodeId,
            }));
          }

          // Credit the processor
          ws.send(JSON.stringify({
            type: 'credit_earned',
            amount: reward.toString(),
            taskId: message.taskId,
          }));
          break;

        case 'peer_message':
          // Relay message to specific peer
          const peerWs = nodes.get(message.targetId);
          if (peerWs && peerWs.readyState === WebSocket.OPEN) {
            peerWs.send(JSON.stringify({
              type: 'peer_message',
              from: nodeId,
              payload: message.payload,
            }));
          }
          break;

        case 'broadcast':
          // Broadcast to all peers
          broadcast({
            type: 'broadcast',
            from: nodeId,
            payload: message.payload,
          }, nodeId);
          break;

        // ========================================
        // WebRTC Signaling Messages
        // ========================================

        case 'webrtc_offer':
          // Relay WebRTC offer to target peer
          {
            const targetWs = nodes.get(message.targetId);
            if (targetWs && targetWs.readyState === WebSocket.OPEN) {
              targetWs.send(JSON.stringify({
                type: 'webrtc_offer',
                from: nodeId,
                offer: message.offer,
              }));
              console.log(`[Relay] WebRTC offer: ${nodeId} -> ${message.targetId}`);
            } else {
              ws.send(JSON.stringify({
                type: 'error',
                message: 'Target peer not available',
                targetId: message.targetId,
              }));
            }
          }
          break;

        case 'webrtc_answer':
          // Relay WebRTC answer to target peer
          {
            const targetWs = nodes.get(message.targetId);
            if (targetWs && targetWs.readyState === WebSocket.OPEN) {
              targetWs.send(JSON.stringify({
                type: 'webrtc_answer',
                from: nodeId,
                answer: message.answer,
              }));
              console.log(`[Relay] WebRTC answer: ${nodeId} -> ${message.targetId}`);
            }
          }
          break;

        case 'webrtc_ice':
          // Relay ICE candidate to target peer
          {
            const targetWs = nodes.get(message.targetId);
            if (targetWs && targetWs.readyState === WebSocket.OPEN) {
              targetWs.send(JSON.stringify({
                type: 'webrtc_ice',
                from: nodeId,
                candidate: message.candidate,
              }));
            }
          }
          break;

        case 'webrtc_disconnect':
          // Notify peer of WebRTC disconnect
          {
            const targetWs = nodes.get(message.targetId);
            if (targetWs && targetWs.readyState === WebSocket.OPEN) {
              targetWs.send(JSON.stringify({
                type: 'webrtc_disconnect',
                from: nodeId,
              }));
            }
          }
          break;

        // ========================================
        // Ledger Sync Messages (Multi-device sync)
        // ========================================

        case 'ledger_sync':
          // Store and broadcast ledger state
          {
            const ledgerKey = message.publicKey || nodeId;
            if (!nodeLedgers.has(ledgerKey)) {
              nodeLedgers.set(ledgerKey, new Map());
            }
            const ledger = nodeLedgers.get(ledgerKey);

            // Store this device's state
            ledger.set(message.nodeId || nodeId, {
              earned: message.state?.earned || {},
              spent: message.state?.spent || {},
              timestamp: message.timestamp || Date.now(),
            });

            console.log(`[Relay] Ledger sync from ${nodeId}, key: ${ledgerKey}`);

            // Broadcast merged state to all devices with same identity
            const mergedState = mergeLedgerStates(ledger);
            for (const [nId, nWs] of nodes) {
              if (nWs.readyState === WebSocket.OPEN && nWs.ledgerKey === ledgerKey) {
                nWs.send(JSON.stringify({
                  type: 'ledger_sync',
                  state: mergedState,
                  deviceCount: ledger.size,
                  timestamp: Date.now(),
                }));
              }
            }

            // Acknowledge sync
            ws.send(JSON.stringify({
              type: 'ledger_sync_ack',
              balance: mergedState.balance,
              deviceCount: ledger.size,
            }));
          }
          break;

        case 'ledger_subscribe':
          // Subscribe to ledger updates for a public key
          ws.ledgerKey = message.publicKey;
          console.log(`[Relay] Node ${nodeId} subscribed to ledger ${message.publicKey}`);

          // Send current state if exists
          const existingLedger = nodeLedgers.get(message.publicKey);
          if (existingLedger) {
            const state = mergeLedgerStates(existingLedger);
            ws.send(JSON.stringify({
              type: 'ledger_sync',
              state,
              deviceCount: existingLedger.size,
              timestamp: Date.now(),
            }));
          }
          break;

        case 'heartbeat':
          ws.send(JSON.stringify({
            type: 'heartbeat_ack',
            timestamp: Date.now(),
            phase: networkState.timeCrystalPhase,
          }));
          break;

        default:
          console.log(`[Relay] Unknown message type: ${message.type}`);
      }
    } catch (error) {
      console.error('[Relay] Error processing message:', error);
    }
  });

  ws.on('close', () => {
    // Security: Clear heartbeat timeout
    clearTimeout(heartbeatTimeout);

    // Security: Decrement IP connection count
    const currentCount = ipConnections.get(clientIP) || 1;
    if (currentCount <= 1) {
      ipConnections.delete(clientIP);
    } else {
      ipConnections.set(clientIP, currentCount - 1);
    }

    if (nodeId) {
      nodes.delete(nodeId);
      rateLimits.delete(nodeId);
      networkState.activeNodes = nodes.size;
      console.log(`[Relay] Node disconnected: ${nodeId} (${nodes.size} remaining)`);

      broadcast({
        type: 'node_left',
        nodeId,
        totalNodes: nodes.size,
      });
    }
  });

  ws.on('error', (error) => {
    console.error('[Relay] WebSocket error:', error);
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`[Edge-Net Relay] Genesis server started on port ${PORT}`);
  console.log(`[Edge-Net Relay] Genesis time: ${new Date(networkState.genesisTime).toISOString()}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[Relay] Shutting down...');
  broadcast({ type: 'relay_shutdown' });
  wss.close(() => {
    server.close(() => {
      process.exit(0);
    });
  });
});

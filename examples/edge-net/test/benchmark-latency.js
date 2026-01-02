#!/usr/bin/env node
/**
 * Edge-Net WebRTC P2P Latency Benchmark
 *
 * Measures:
 * - Signaling latency (offer → answer)
 * - P2P message round-trip time
 * - Relay overhead
 * - Connection establishment time
 */

import WebSocket from 'ws';
import { randomBytes } from 'crypto';

const RELAY_URL = process.env.RELAY_URL || 'ws://localhost:8080';
const ITERATIONS = 100;
const WARMUP_ITERATIONS = 10;

// Benchmark results
const results = {
    signaling: [],
    peerMessage: [],
    connectionTime: [],
    throughput: [],
};

class BenchmarkPeer {
    constructor(id) {
        this.id = `bench-${id}-${randomBytes(4).toString('hex')}`;
        this.ws = null;
        this.messageCallbacks = new Map();
        this.connected = false;
    }

    async connect() {
        const start = performance.now();

        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(RELAY_URL);

            this.ws.on('open', () => {
                this.ws.send(JSON.stringify({
                    type: 'register',
                    nodeId: this.id,
                }));
            });

            this.ws.on('message', (data) => {
                const msg = JSON.parse(data.toString());

                if (msg.type === 'welcome') {
                    this.connected = true;
                    results.connectionTime.push(performance.now() - start);
                    resolve();
                }

                // Handle response callbacks
                if (msg.type === 'peer_message' && msg.payload?.benchmarkId) {
                    const callback = this.messageCallbacks.get(msg.payload.benchmarkId);
                    if (callback) {
                        callback(msg);
                        this.messageCallbacks.delete(msg.payload.benchmarkId);
                    }
                }

                if (msg.type === 'webrtc_answer' && msg.from) {
                    const callback = this.messageCallbacks.get(`offer-${msg.from}`);
                    if (callback) {
                        callback(msg);
                        this.messageCallbacks.delete(`offer-${msg.from}`);
                    }
                }
            });

            this.ws.on('error', reject);
        });
    }

    async measureSignalingLatency(targetId) {
        const start = performance.now();

        return new Promise((resolve) => {
            this.messageCallbacks.set(`offer-${targetId}`, () => {
                resolve(performance.now() - start);
            });

            this.ws.send(JSON.stringify({
                type: 'webrtc_offer',
                targetId,
                offer: { type: 'offer', sdp: 'benchmark-sdp' },
            }));

            // Timeout after 5s
            setTimeout(() => resolve(null), 5000);
        });
    }

    async measureMessageLatency(targetId) {
        const benchmarkId = randomBytes(8).toString('hex');
        const start = performance.now();

        return new Promise((resolve) => {
            this.messageCallbacks.set(benchmarkId, () => {
                resolve(performance.now() - start);
            });

            this.ws.send(JSON.stringify({
                type: 'peer_message',
                targetId,
                payload: {
                    benchmarkId,
                    timestamp: start,
                    data: randomBytes(64).toString('hex'),
                },
            }));

            // Timeout
            setTimeout(() => resolve(null), 5000);
        });
    }

    async measureThroughput(targetId, messageSize, count) {
        const data = 'x'.repeat(messageSize);
        const start = performance.now();

        for (let i = 0; i < count; i++) {
            this.ws.send(JSON.stringify({
                type: 'peer_message',
                targetId,
                payload: { data, seq: i },
            }));
        }

        const elapsed = performance.now() - start;
        const bytesPerSecond = (messageSize * count) / (elapsed / 1000);
        return { elapsed, bytesPerSecond, messagesPerSecond: count / (elapsed / 1000) };
    }

    close() {
        if (this.ws) this.ws.close();
    }
}

// Statistics helpers
function calculateStats(arr) {
    const filtered = arr.filter(v => v !== null);
    if (filtered.length === 0) return { min: 0, max: 0, avg: 0, p50: 0, p95: 0, p99: 0 };

    const sorted = [...filtered].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);

    return {
        min: sorted[0],
        max: sorted[sorted.length - 1],
        avg: sum / sorted.length,
        p50: sorted[Math.floor(sorted.length * 0.5)],
        p95: sorted[Math.floor(sorted.length * 0.95)],
        p99: sorted[Math.floor(sorted.length * 0.99)],
        count: sorted.length,
    };
}

function formatLatency(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

async function runBenchmark() {
    console.log('⚡ Edge-Net WebRTC P2P Latency Benchmark');
    console.log('='.repeat(50));
    console.log(`Relay: ${RELAY_URL}`);
    console.log(`Iterations: ${ITERATIONS} (+ ${WARMUP_ITERATIONS} warmup)\n`);

    const peer1 = new BenchmarkPeer('A');
    const peer2 = new BenchmarkPeer('B');

    try {
        // Connect both peers
        console.log('📡 Connecting peers...');
        await Promise.all([peer1.connect(), peer2.connect()]);
        console.log(`  Connection time: ${formatLatency(results.connectionTime[0])} / ${formatLatency(results.connectionTime[1])}\n`);

        // Set up peer2 to auto-respond to offers and messages
        peer2.ws.on('message', (data) => {
            const msg = JSON.parse(data.toString());

            if (msg.type === 'webrtc_offer') {
                peer2.ws.send(JSON.stringify({
                    type: 'webrtc_answer',
                    targetId: msg.from,
                    answer: { type: 'answer', sdp: 'benchmark-response' },
                }));
            }

            if (msg.type === 'peer_message' && msg.payload?.benchmarkId) {
                peer2.ws.send(JSON.stringify({
                    type: 'peer_message',
                    targetId: msg.from,
                    payload: msg.payload,
                }));
            }
        });

        // Warmup
        console.log('🔥 Warming up...');
        for (let i = 0; i < WARMUP_ITERATIONS; i++) {
            await peer1.measureSignalingLatency(peer2.id);
            await peer1.measureMessageLatency(peer2.id);
        }

        // Benchmark signaling latency
        console.log('\n📊 Benchmarking signaling latency...');
        for (let i = 0; i < ITERATIONS; i++) {
            const latency = await peer1.measureSignalingLatency(peer2.id);
            if (latency) results.signaling.push(latency);
            if ((i + 1) % 25 === 0) process.stdout.write('.');
        }
        console.log(' done');

        // Benchmark peer message latency
        console.log('📊 Benchmarking P2P message latency...');
        for (let i = 0; i < ITERATIONS; i++) {
            const latency = await peer1.measureMessageLatency(peer2.id);
            if (latency) results.peerMessage.push(latency);
            if ((i + 1) % 25 === 0) process.stdout.write('.');
        }
        console.log(' done');

        // Benchmark throughput
        console.log('📊 Benchmarking throughput...');
        const throughputResults = [];
        for (const size of [64, 256, 1024, 4096]) {
            const result = await peer1.measureThroughput(peer2.id, size, 100);
            throughputResults.push({ size, ...result });
        }

        // Print results
        console.log('\n' + '═'.repeat(60));
        console.log('📈 BENCHMARK RESULTS');
        console.log('═'.repeat(60));

        const signalingStats = calculateStats(results.signaling);
        console.log('\n🤝 Signaling Latency (offer → answer):');
        console.log(`   Min: ${formatLatency(signalingStats.min)}`);
        console.log(`   Max: ${formatLatency(signalingStats.max)}`);
        console.log(`   Avg: ${formatLatency(signalingStats.avg)}`);
        console.log(`   P50: ${formatLatency(signalingStats.p50)}`);
        console.log(`   P95: ${formatLatency(signalingStats.p95)}`);
        console.log(`   P99: ${formatLatency(signalingStats.p99)}`);

        const messageStats = calculateStats(results.peerMessage);
        console.log('\n💬 P2P Message Round-Trip:');
        console.log(`   Min: ${formatLatency(messageStats.min)}`);
        console.log(`   Max: ${formatLatency(messageStats.max)}`);
        console.log(`   Avg: ${formatLatency(messageStats.avg)}`);
        console.log(`   P50: ${formatLatency(messageStats.p50)}`);
        console.log(`   P95: ${formatLatency(messageStats.p95)}`);
        console.log(`   P99: ${formatLatency(messageStats.p99)}`);

        console.log('\n📦 Throughput (via relay):');
        for (const t of throughputResults) {
            console.log(`   ${t.size}B msgs: ${t.messagesPerSecond.toFixed(0)} msg/s (${(t.bytesPerSecond / 1024).toFixed(1)} KB/s)`);
        }

        const connStats = calculateStats(results.connectionTime);
        console.log('\n⏱️  Connection Establishment:');
        console.log(`   Avg: ${formatLatency(connStats.avg)}`);

        // Performance assessment
        console.log('\n' + '─'.repeat(60));
        console.log('🎯 Performance Assessment:');

        if (messageStats.p95 < 50) {
            console.log('   ✅ Excellent: P95 latency < 50ms');
        } else if (messageStats.p95 < 100) {
            console.log('   ✅ Good: P95 latency < 100ms');
        } else if (messageStats.p95 < 200) {
            console.log('   ⚠️  Acceptable: P95 latency < 200ms');
        } else {
            console.log('   ❌ Needs optimization: P95 latency > 200ms');
        }

        if (signalingStats.avg < 10) {
            console.log('   ✅ Fast signaling: < 10ms average');
        } else if (signalingStats.avg < 50) {
            console.log('   ✅ Good signaling: < 50ms average');
        }

        console.log('\n💡 Optimization Suggestions:');
        if (messageStats.avg > 20) {
            console.log('   • Consider WebRTC data channels for direct P2P');
        }
        if (signalingStats.p99 > signalingStats.avg * 3) {
            console.log('   • High P99 variance - check for GC pauses');
        }
        console.log('   • Deploy relay servers closer to users (CDN/Edge)');
        console.log('   • Enable WebSocket compression for large messages');

        console.log('\n');

    } catch (err) {
        console.error('\n❌ Benchmark error:', err.message);
        process.exit(1);
    } finally {
        peer1.close();
        peer2.close();
    }
}

runBenchmark();

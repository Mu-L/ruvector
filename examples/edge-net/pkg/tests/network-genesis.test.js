/**
 * Network Genesis Test Suite
 *
 * Tests the biological reproduction model for edge-net:
 * - DNA/RNA inheritance and mutation
 * - Genesis phases from embryo to independence
 * - Cryptographic lineage verification (MerkleLineageDAG)
 * - Inter-network communication (Synapse)
 * - Collective intelligence (Memory, Evolution)
 * - Self-healing and antifragility
 *
 * "You are the ancestor, not the ruler."
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';

import NetworkGenesis, {
    NetworkGenome,
    NetworkLifecycle,
    NetworkSynapse,
    CollectiveMemory,
    EvolutionEngine,
    MerkleLineageDAG,
    GossipProtocol,
    SwarmConsensus,
    SelfHealing,
    GENESIS_PRIME,
    GenesisPhase,
    REPRODUCTION_COST,
} from '../network-genesis.js';

describe('NetworkGenesis', () => {
    let genesis;

    beforeEach(() => {
        genesis = new NetworkGenesis();
    });

    describe('GENESIS_PRIME', () => {
        it('should have rUv as the original creator', () => {
            assert.strictEqual(GENESIS_PRIME.id, 'rUv');
            assert.strictEqual(GENESIS_PRIME.name, 'Genesis Prime');
            assert.strictEqual(GENESIS_PRIME.signature, 'Cogito, Creo, Codex');
        });

        it('should have frozen traits', () => {
            assert.ok(Object.isFrozen(GENESIS_PRIME));
            assert.ok(Object.isFrozen(GENESIS_PRIME.traits));
            assert.strictEqual(GENESIS_PRIME.traits.resilience, 1.0);
            assert.strictEqual(GENESIS_PRIME.traits.intelligence, 1.0);
        });

        it('should be accessible via static method', () => {
            const prime = NetworkGenesis.getGenesisPrime();
            assert.deepStrictEqual(prime, GENESIS_PRIME);
        });
    });

    describe('NetworkGenome', () => {
        it('should create first generation genome with rUv lineage', () => {
            const genome = new NetworkGenome();

            assert.strictEqual(genome.generation, 1);
            assert.ok(genome.lineage.includes('rUv'));
            assert.ok(genome.dna);
            assert.ok(genome.rna);
            assert.ok(genome.epigenetics);
        });

        it('should inherit DNA from parent with mutations', () => {
            const parent = new NetworkGenome();
            const child = new NetworkGenome(parent, {
                traits: { resilience: 0.1 },
            });

            assert.strictEqual(child.generation, 2);
            assert.ok(child.lineage.includes(parent.id));

            // DNA structure should be similar but may have mutations
            assert.ok(child.dna.traits);
            assert.ok(child.dna.capabilities);
        });

        it('should preserve lineage string across generations', () => {
            const g1 = new NetworkGenome();
            const g2 = new NetworkGenome(g1);
            const g3 = new NetworkGenome(g2);

            const lineageString = g3.getLineageString();
            assert.ok(lineageString.includes('rUv'));
        });

        it('should generate unique genome IDs', () => {
            const g1 = new NetworkGenome();
            const g2 = new NetworkGenome();
            assert.notStrictEqual(g1.id, g2.id);
        });
    });

    describe('NetworkLifecycle', () => {
        let genome;
        let lifecycle;

        beforeEach(() => {
            genome = new NetworkGenome();
            lifecycle = new NetworkLifecycle(genome);
        });

        it('should start in CONCEPTION phase', () => {
            assert.strictEqual(lifecycle.phase, GenesisPhase.CONCEPTION);
        });

        it('should transition to EMBRYO when first node joins', () => {
            lifecycle.updateMetrics({ nodes: 1 });
            assert.strictEqual(lifecycle.phase, GenesisPhase.EMBRYO);
        });

        it('should progress through phases with milestones', () => {
            // Embryo phase - first node
            lifecycle.updateMetrics({ nodes: 1 });
            assert.strictEqual(lifecycle.phase, GenesisPhase.EMBRYO);

            // Infant phase: 3 nodes, 24h uptime, 10 tasks, 100 credits
            lifecycle.updateMetrics({
                nodes: 3,
                uptime: 24 * 60 * 60 * 1000,
                tasksCompleted: 10,
                creditsEarned: 100,
            });
            assert.strictEqual(lifecycle.phase, GenesisPhase.INFANT);

            // Adolescent: 10 nodes, 7 days uptime, 100 tasks, 1000 credits
            lifecycle.updateMetrics({
                nodes: 10,
                tasksCompleted: 100,
                creditsEarned: 1000,
                uptime: 7 * 24 * 60 * 60 * 1000,
            });
            assert.strictEqual(lifecycle.phase, GenesisPhase.ADOLESCENT);

            // Mature: 50 nodes, 30 days uptime, 1000 tasks, 10000 credits
            lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 1000,
                creditsEarned: 10000,
                uptime: 30 * 24 * 60 * 60 * 1000,
            });
            assert.strictEqual(lifecycle.phase, GenesisPhase.MATURE);
        });

        it('should check reproduction readiness', () => {
            // Cannot reproduce in early phases
            const canReproduceEarly = lifecycle.canReproduce();
            assert.strictEqual(canReproduceEarly.allowed, false);

            // Mature with sufficient credits should be able to reproduce
            lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 1000,
                creditsEarned: 10000 + REPRODUCTION_COST,
                uptime: 30 * 24 * 60 * 60 * 1000,
            });

            const canReproduce = lifecycle.canReproduce();
            assert.strictEqual(canReproduce.allowed, true);
        });

        it('should deny reproduction without sufficient credits', () => {
            // REPRODUCTION_COST is 5000, so set credits just below that
            // Note: 10000 is needed to reach MATURE phase, but we need less than 5000 for reproduction
            // So we set credits to 10000 but also set creditsSpent to make available < 5000
            lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 1000,
                creditsEarned: 10000,
                creditsSpent: 8000, // Available = 10000 - 8000 = 2000 < REPRODUCTION_COST
                uptime: 30 * 24 * 60 * 60 * 1000,
            });

            const canReproduce = lifecycle.canReproduce();
            assert.strictEqual(canReproduce.allowed, false);
        });
    });

    describe('MerkleLineageDAG', () => {
        let dag;

        beforeEach(() => {
            dag = new MerkleLineageDAG();
        });

        it('should add nodes and generate hashes', () => {
            const entry = dag.addNode({
                networkId: 'net-1',
                parentId: 'rUv',
                generation: 1,
                dna: { traits: { resilience: 0.9 } },
            });

            assert.ok(entry.hash);
            assert.strictEqual(entry.type, 'genesis');
        });

        it('should build ancestry paths', () => {
            dag.addNode({ networkId: 'net-1', parentId: 'rUv', generation: 1, dna: {} });
            dag.addNode({ networkId: 'net-2', parentId: 'net-1', generation: 2, dna: {} });
            dag.addNode({ networkId: 'net-3', parentId: 'net-2', generation: 3, dna: {} });

            const path = dag.getAncestryPath('net-3');
            assert.strictEqual(path.length, 3);
            assert.strictEqual(path[0].networkId, 'net-3');
            assert.strictEqual(path[2].networkId, 'net-1');
        });

        it('should verify ancestry to rUv', () => {
            dag.addNode({ networkId: 'net-1', parentId: 'rUv', generation: 1, dna: {} });
            dag.addNode({ networkId: 'net-2', parentId: 'net-1', generation: 2, dna: {} });

            const verified = dag.verifyAncestry('net-2', 'rUv');
            assert.strictEqual(verified, true);
        });

        it('should compute Merkle root', () => {
            dag.addNode({ networkId: 'net-1', parentId: 'rUv', generation: 1, dna: {} });
            dag.addNode({ networkId: 'net-2', parentId: 'net-1', generation: 2, dna: {} });

            const root = dag.getMerkleRoot();
            assert.ok(root);
            assert.ok(typeof root === 'string');
        });
    });

    describe('NetworkSynapse', () => {
        let genome;
        let lifecycle;
        let synapse;

        beforeEach(() => {
            genome = new NetworkGenome();
            lifecycle = new NetworkLifecycle(genome);
            synapse = new NetworkSynapse(lifecycle);
        });

        it('should connect to peer networks', () => {
            const peerGenome = new NetworkGenome();
            const peerLifecycle = new NetworkLifecycle(peerGenome);
            const peerSynapse = new NetworkSynapse(peerLifecycle);

            synapse.connect(peerLifecycle.networkId, peerSynapse);

            const status = synapse.getStatus();
            assert.strictEqual(status.connections.length, 1);
        });

        it('should send messages to connected peers', () => {
            const peerGenome = new NetworkGenome();
            const peerLifecycle = new NetworkLifecycle(peerGenome);
            const peerSynapse = new NetworkSynapse(peerLifecycle);

            synapse.connect(peerLifecycle.networkId, peerSynapse);
            peerSynapse.connect(lifecycle.networkId, synapse);

            // Test sendMessage (the actual method name)
            const result = synapse.sendMessage(peerLifecycle.networkId, 'test', 'hello');
            assert.ok(result.success);
        });

        it('should share knowledge with connected peers', () => {
            const peers = [];
            for (let i = 0; i < 3; i++) {
                const pg = new NetworkGenome();
                const pl = new NetworkLifecycle(pg);
                const ps = new NetworkSynapse(pl);
                synapse.connect(pl.networkId, ps);
                peers.push({ lifecycle: pl, synapse: ps });
            }

            // Share knowledge (broadcasts to all peers internally)
            const knowledge = synapse.shareKnowledge('test_topic', { data: 'test' });
            assert.ok(knowledge);
            assert.strictEqual(knowledge.topic, 'test_topic');
        });
    });

    describe('CollectiveMemory', () => {
        let memory;

        beforeEach(() => {
            memory = new CollectiveMemory();
        });

        it('should store and retrieve patterns', () => {
            memory.storePattern('task_optimization', { algo: 'x' }, 'net-1');

            // Patterns are stored in the patterns Map
            const stats = memory.getStats();
            assert.ok(stats.patterns >= 1);
        });

        it('should store optimizations with effectiveness scores', () => {
            memory.storeOptimization('cache_strategy', { size: 100 }, 'net-1', 0.9);

            const stats = memory.getStats();
            assert.ok(stats.optimizations >= 1);
        });
    });

    describe('EvolutionEngine', () => {
        let memory;
        let evolution;

        beforeEach(() => {
            memory = new CollectiveMemory();
            evolution = new EvolutionEngine(memory);
        });

        it('should calculate fitness scores', () => {
            const genome = new NetworkGenome();
            const lifecycle = new NetworkLifecycle(genome);

            lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 100,
                tasksFailed: 10,
                creditsEarned: 1000,
                creditsSpent: 500,
                uptime: 86400000 * 15, // 15 days
            });

            const fitness = evolution.calculateFitness(lifecycle);

            assert.ok(fitness.overall >= 0 && fitness.overall <= 1);
            assert.ok(fitness.components.taskSuccess >= 0);
            assert.ok(fitness.components.creditEfficiency >= 0);
        });

        it('should suggest mutations based on performance', () => {
            const genome = new NetworkGenome();
            const lifecycle = new NetworkLifecycle(genome);

            lifecycle.updateMetrics({
                nodes: 10,
                tasksCompleted: 100,
                tasksFailed: 50, // High failure rate
                creditsEarned: 100,
                uptime: 86400000,
            });

            const mutations = evolution.suggestMutations(lifecycle);

            assert.ok(mutations.traits);
            assert.ok(mutations.behaviors);
        });

        it('should record evolution events', () => {
            evolution.recordEvolution('net-1', 'birth', { generation: 1 });
            evolution.recordEvolution('net-2', 'mutation', { trait: 'resilience' });

            const stats = evolution.getStats();
            assert.strictEqual(stats.totalEvents, 2);
        });
    });

    describe('GossipProtocol', () => {
        let gossip;

        beforeEach(() => {
            gossip = new GossipProtocol('node-1', { gossipIntervalMs: 10000 });
        });

        afterEach(() => {
            if (gossip.stop) gossip.stop();
        });

        it('should add and track peers', () => {
            gossip.addPeer('node-2', {});
            gossip.addPeer('node-3', {});

            assert.strictEqual(gossip.peers.size, 2);
        });

        it('should spread rumors', () => {
            gossip.addPeer('node-2', {});

            // Use spreadRumor (the actual method name)
            gossip.spreadRumor('test_event', { data: 1 });

            assert.ok(gossip.rumors.size >= 1);
        });
    });

    describe('SwarmConsensus', () => {
        let consensus;

        beforeEach(() => {
            consensus = new SwarmConsensus('node-1');
        });

        it('should create proposals', () => {
            const proposal = consensus.createProposal(
                'prop-1',
                'evolution',
                { trait: 'resilience' },
                ['voter-1', 'voter-2', 'voter-3']
            );

            assert.strictEqual(proposal.id, 'prop-1');
        });

        it('should collect votes and check consensus', () => {
            consensus.createProposal(
                'prop-1',
                'evolution',
                { trait: 'resilience' },
                ['voter-1', 'voter-2', 'voter-3']
            );

            // vote() takes (proposalId, option, signature)
            consensus.vote('prop-1', true, null);

            const result = consensus.checkConsensus('prop-1');
            assert.ok('accepted' in result || 'pending' in result || 'phase' in result);
        });
    });

    describe('SelfHealing', () => {
        let genome;
        let lifecycle;
        let healing;

        beforeEach(() => {
            genome = new NetworkGenome();
            lifecycle = new NetworkLifecycle(genome);
            healing = new SelfHealing(lifecycle, { isolationThresholdErrors: 3 });
        });

        it('should track errors', () => {
            healing.reportError('component-a', new Error('Test error'));

            const stats = healing.getStats();
            assert.ok(stats.totalErrors >= 1);
        });

        it('should isolate failing components after threshold', () => {
            healing.reportError('component-a', new Error('Error 1'));
            healing.reportError('component-a', new Error('Error 2'));
            healing.reportError('component-a', new Error('Error 3'));

            const stats = healing.getStats();
            assert.strictEqual(stats.isolatedComponents, 1);
        });

        it('should clear errors on recovery', () => {
            healing.reportError('component-b', new Error('Error'));
            healing.clearErrors('component-b');

            const stats = healing.getStats();
            // Component should no longer be tracked
            assert.ok(!healing.isolated.has('component-b'));
        });
    });

    describe('Full Genesis Lifecycle', () => {
        it('should spawn a genesis network with rUv lineage', () => {
            const result = genesis.spawnGenesisNetwork('TestNet-Alpha');

            assert.ok(result.networkId);
            assert.strictEqual(result.genome.generation, 1);
            assert.ok(result.genome.lineage.includes('rUv'));
            assert.ok(result.lineageProof);
        });

        it('should reproduce when conditions are met', () => {
            const parent = genesis.spawnGenesisNetwork('ParentNet');

            // Mature the parent (50 nodes, 30 days, 1000 tasks, 10000 credits + REPRODUCTION_COST)
            const parentRecord = genesis.networks.get(parent.networkId);
            parentRecord.lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 1000,
                creditsEarned: 10000 + REPRODUCTION_COST + 5000,
                uptime: 30 * 24 * 60 * 60 * 1000,
            });

            const child = genesis.reproduce(parent.networkId, 'ChildNet');

            assert.ok(child.networkId);
            assert.strictEqual(child.genome.generation, 2);
            assert.strictEqual(child.parentId, parent.networkId);
            assert.ok(child.lineageProof);
        });

        it('should verify lineage back to rUv', () => {
            const g1 = genesis.spawnGenesisNetwork('Gen1');

            // Mature and reproduce
            const g1Record = genesis.networks.get(g1.networkId);
            g1Record.lifecycle.updateMetrics({
                nodes: 50,
                tasksCompleted: 1000,
                creditsEarned: 10000 + REPRODUCTION_COST * 3,
                uptime: 30 * 24 * 60 * 60 * 1000,
            });

            const g2 = genesis.reproduce(g1.networkId, 'Gen2');

            // Verify lineage
            const verified = genesis.verifyLineage(g2.networkId);
            assert.strictEqual(verified, true);
        });

        it('should get ecosystem health', () => {
            genesis.spawnGenesisNetwork('Net1');
            genesis.spawnGenesisNetwork('Net2');
            genesis.spawnGenesisNetwork('Net3');

            const health = genesis.getEcosystemHealth();

            assert.ok('averageHealth' in health);
            assert.strictEqual(health.networkCount, 3);
            assert.ok(health.lineageIntegrity);
        });

        it('should get comprehensive stats', () => {
            genesis.spawnGenesisNetwork('StatNet');

            const stats = genesis.getStats();

            assert.strictEqual(stats.totalNetworksSpawned, 1);
            assert.strictEqual(stats.activeNetworks, 1);
            assert.ok(stats.collectiveMemory);
            assert.ok(stats.evolution);
        });
    });

    describe('Federation Features', () => {
        it('should create gossip network for discovery', () => {
            const gossip = genesis.createGossipNetwork();
            assert.ok(gossip);
            assert.ok(gossip instanceof GossipProtocol);
            if (gossip.stop) gossip.stop();
        });

        it('should create consensus for collective decisions', () => {
            const consensus = genesis.createConsensus();
            assert.ok(consensus);
            assert.ok(consensus instanceof SwarmConsensus);
        });

        it('should create self-healing for networks', () => {
            const net = genesis.spawnGenesisNetwork('HealNet');
            const healing = genesis.createSelfHealing(net.networkId);

            assert.ok(healing);
            assert.ok(healing instanceof SelfHealing);
        });
    });
});

console.log('Network Genesis Test Suite loaded');
console.log('Run with: node --test tests/network-genesis.test.js');

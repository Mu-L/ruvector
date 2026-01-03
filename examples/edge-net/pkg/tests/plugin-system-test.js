#!/usr/bin/env node
/**
 * Plugin System Tests
 *
 * Comprehensive testing of the edge-net plugin architecture.
 */

import {
    PLUGIN_CATALOG,
    PLUGIN_BUNDLES,
    PluginCategory,
    PluginTier,
    Capability,
    PluginLoader,
    BasePlugin,
    validateManifest,
    validatePlugin,
    PluginRegistry,
    generatePluginTemplate,
} from '../plugins/index.js';

import { PluginFailureContract } from '../plugins/plugin-loader.js';
import CoreInvariants, {
    EconomicBoundary,
    IdentityFriction,
    WorkVerifier,
    DegradationController,
} from '../core-invariants.js';

import { CompressionPlugin } from '../plugins/implementations/compression.js';
import { E2EEncryptionPlugin } from '../plugins/implementations/e2e-encryption.js';
import { FederatedLearningPlugin } from '../plugins/implementations/federated-learning.js';
import { ReputationStakingPlugin } from '../plugins/implementations/reputation-staking.js';
import { SwarmIntelligencePlugin } from '../plugins/implementations/swarm-intelligence.js';

// ============================================
// TEST UTILITIES
// ============================================

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`✅ ${name}`);
        passed++;
    } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   Error: ${error.message}`);
        failed++;
    }
}

async function testAsync(name, fn) {
    try {
        await fn();
        console.log(`✅ ${name}`);
        passed++;
    } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   Error: ${error.message}`);
        failed++;
    }
}

function assert(condition, message) {
    if (!condition) throw new Error(message || 'Assertion failed');
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
}

// ============================================
// TESTS
// ============================================

console.log('\n╔════════════════════════════════════════════════════════════════╗');
console.log('║              PLUGIN SYSTEM TESTS                                ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

// --- Catalog Tests ---
console.log('\n--- Plugin Catalog ---\n');

test('Catalog has plugins', () => {
    assert(Object.keys(PLUGIN_CATALOG).length > 0, 'Catalog should have plugins');
});

test('All plugins have required fields', () => {
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(plugin.id === id, `Plugin ${id} ID mismatch`);
        assert(plugin.name, `Plugin ${id} missing name`);
        assert(plugin.version, `Plugin ${id} missing version`);
        assert(plugin.description, `Plugin ${id} missing description`);
        assert(plugin.category, `Plugin ${id} missing category`);
        assert(plugin.tier, `Plugin ${id} missing tier`);
    }
});

test('Plugin categories are valid', () => {
    const validCategories = Object.values(PluginCategory);
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(validCategories.includes(plugin.category),
            `Plugin ${id} has invalid category: ${plugin.category}`);
    }
});

test('Plugin tiers are valid', () => {
    const validTiers = Object.values(PluginTier);
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(validTiers.includes(plugin.tier),
            `Plugin ${id} has invalid tier: ${plugin.tier}`);
    }
});

test('Bundles reference valid plugins', () => {
    for (const [bundleId, bundle] of Object.entries(PLUGIN_BUNDLES)) {
        for (const pluginId of bundle.plugins) {
            assert(PLUGIN_CATALOG[pluginId],
                `Bundle ${bundleId} references missing plugin: ${pluginId}`);
        }
    }
});

// --- Plugin Loader Tests ---
console.log('\n--- Plugin Loader ---\n');

test('Plugin loader initializes', () => {
    const loader = new PluginLoader();
    assert(loader.getCatalog().length > 0, 'Loader should see catalog');
});

test('Loader respects tier restrictions', () => {
    const loader = new PluginLoader({
        allowedTiers: [PluginTier.STABLE],
    });

    const catalog = loader.getCatalog();
    const betaPlugin = catalog.find(p => p.tier === PluginTier.BETA);

    if (betaPlugin) {
        assert(!betaPlugin.isAllowed.allowed, 'Beta plugins should not be allowed');
    }
});

test('Loader respects capability restrictions', () => {
    const loader = new PluginLoader({
        deniedCapabilities: [Capability.SYSTEM_EXEC],
    });

    const catalog = loader.getCatalog();
    const execPlugin = catalog.find(p =>
        p.capabilities?.includes(Capability.SYSTEM_EXEC)
    );

    if (execPlugin) {
        assert(!execPlugin.isAllowed.allowed, 'Exec plugins should not be allowed');
    }
});

// --- Manifest Validation Tests ---
console.log('\n--- Manifest Validation ---\n');

test('Valid manifest passes validation', () => {
    const manifest = {
        id: 'test.valid-plugin',
        name: 'Valid Plugin',
        version: '1.0.0',
        description: 'A valid test plugin',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
        capabilities: [Capability.COMPUTE_WASM],
    };

    const result = validateManifest(manifest);
    assert(result.valid, `Validation should pass: ${result.errors.join(', ')}`);
});

test('Invalid ID fails validation', () => {
    const manifest = {
        id: 'InvalidID',
        name: 'Test',
        version: '1.0.0',
        description: 'Test',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
    };

    const result = validateManifest(manifest);
    assert(!result.valid, 'Invalid ID should fail');
    assert(result.errors.some(e => e.includes('ID')), 'Should mention ID error');
});

test('Missing fields fail validation', () => {
    const manifest = { id: 'test.plugin' };
    const result = validateManifest(manifest);
    assert(!result.valid, 'Missing fields should fail');
    assert(result.errors.length > 0, 'Should have errors');
});

// --- Plugin SDK Tests ---
console.log('\n--- Plugin SDK ---\n');

test('BasePlugin can be extended', () => {
    class TestPlugin extends BasePlugin {
        static manifest = {
            id: 'test.sdk-plugin',
            name: 'SDK Test Plugin',
            version: '1.0.0',
            description: 'Testing SDK',
            category: PluginCategory.CORE,
            tier: PluginTier.EXPERIMENTAL,
            capabilities: [],
        };

        doSomething() {
            return 'worked';
        }
    }

    const plugin = new TestPlugin({ option: 'value' });
    assertEqual(plugin.doSomething(), 'worked', 'Plugin method should work');
    assertEqual(plugin.config.option, 'value', 'Config should be set');
});

test('Plugin registry works', () => {
    const registry = new PluginRegistry();

    class TestPlugin extends BasePlugin {
        static manifest = {
            id: 'test.registry-plugin',
            name: 'Registry Test',
            version: '1.0.0',
            description: 'Testing registry',
            category: PluginCategory.CORE,
            tier: PluginTier.EXPERIMENTAL,
            capabilities: [],
        };
    }

    const result = registry.register(TestPlugin);
    assert(result.id === 'test.registry-plugin', 'Should return ID');
    assert(result.checksum, 'Should generate checksum');
    assert(registry.has('test.registry-plugin'), 'Should be registered');
});

test('Template generator works', () => {
    const template = generatePluginTemplate({
        id: 'my-org.my-plugin',
        name: 'My Plugin',
        category: PluginCategory.AI,
    });

    assert(template.includes('class'), 'Should generate class');
    assert(template.includes('my-org.my-plugin'), 'Should include ID');
    assert(template.includes('BasePlugin'), 'Should extend BasePlugin');
});

// --- Implementation Tests ---
console.log('\n--- Plugin Implementations ---\n');

test('Compression plugin works', () => {
    const comp = new CompressionPlugin({ threshold: 10 });

    // Test with compressible data
    const data = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const result = comp.compress(data);

    assert(result.compressed, 'Should compress');
    assert(result.compressedSize < data.length, 'Should reduce size');

    const decompressed = comp.decompress(result.data, true);
    assertEqual(decompressed.toString(), data, 'Should decompress correctly');
});

test('E2E Encryption plugin works', async () => {
    const crypto = new E2EEncryptionPlugin();

    // Establish session
    await crypto.establishSession('peer-1', 'fake-public-key');
    assert(crypto.hasSession('peer-1'), 'Session should exist');

    // Encrypt/decrypt
    const message = 'Hello, secure world!';
    const encrypted = crypto.encrypt('peer-1', message);

    assert(encrypted.ciphertext, 'Should have ciphertext');
    assert(encrypted.iv, 'Should have IV');
    assert(encrypted.authTag, 'Should have auth tag');

    const decrypted = crypto.decrypt('peer-1', encrypted);
    assertEqual(decrypted, message, 'Should decrypt correctly');
});

test('Federated Learning plugin works', async () => {
    const fl = new FederatedLearningPlugin({
        minParticipants: 2,
        localEpochs: 2,
    });

    // Start round
    const globalWeights = [0, 0, 0, 0, 0];
    const roundId = fl.startRound('test-model', globalWeights);

    assert(roundId, 'Should return round ID');

    // Simulate local training
    const localData = [
        { features: [1, 2, 3, 4, 5] },
        { features: [2, 3, 4, 5, 6] },
    ];

    await fl.trainLocal(roundId, localData, { participantId: 'node-1' });
    await fl.trainLocal(roundId, localData, { participantId: 'node-2' });

    // Check aggregation happened
    const status = fl.getRoundStatus(roundId);
    assertEqual(status.status, 'completed', 'Round should complete');
    assertEqual(status.participants, 2, 'Should have 2 participants');
});

test('Reputation staking plugin works', () => {
    const staking = new ReputationStakingPlugin({ minStake: 5 });

    // Mock credit system
    const credits = {
        balance: 100,
        getBalance: () => credits.balance,
        spendCredits: (_, amount) => { credits.balance -= amount; },
        earnCredits: (_, amount) => { credits.balance += amount; },
    };

    // Stake
    const stake = staking.stake('node-1', 20, credits);
    assertEqual(stake.staked, 20, 'Should stake 20');
    assertEqual(stake.reputation, 100, 'Should start at 100 rep');

    // Record success
    staking.recordSuccess('node-1');
    const newStake = staking.getStake('node-1');
    assertEqual(newStake.successfulTasks, 1, 'Should record success');

    // Slash
    const slashResult = staking.slash('node-1', 'test-misbehavior', 0.5);
    assert(slashResult.slashed > 0, 'Should slash');
    assert(slashResult.newReputation < 100, 'Should reduce reputation');
});

test('Swarm intelligence plugin works', async () => {
    const swarm = new SwarmIntelligencePlugin({
        populationSize: 20,
        iterations: 50,
        dimensions: 5,
    });

    // Create swarm with sphere function (minimize x²)
    swarm.createSwarm('test-swarm', {
        algorithm: 'pso',
        bounds: { min: -10, max: 10 },
        fitnessFunction: (x) => x.reduce((sum, v) => sum + v * v, 0),
    });

    // Run optimization
    const result = await swarm.optimize('test-swarm', {
        iterations: 50,
    });

    assert(result.bestFitness < 1, 'Should find good solution');
    assert(result.iterations === 50, 'Should run 50 iterations');
});

// --- Invariant Enforcement Tests ---
console.log('\n--- Core Invariants (Cogito, Creo, Codex) ---\n');

test('PluginFailureContract enforces circuit breaker', async () => {
    const contract = new PluginFailureContract({
        maxRetries: 3,
        quarantineDurationMs: 100, // Short for testing
        executionTimeoutMs: 50,
    });

    // Record 3 failures to trip circuit breaker
    contract.recordFailure('test-plugin', new Error('Failure 1'));
    contract.recordFailure('test-plugin', new Error('Failure 2'));
    contract.recordFailure('test-plugin', new Error('Failure 3'));

    // Plugin should now be blocked
    const canExec = contract.canExecute('test-plugin');
    assert(!canExec.allowed, 'Plugin should be blocked after 3 failures');
    assert(canExec.reason.includes('quarantine'), 'Should mention quarantine');
});

test('PluginFailureContract provides health status', () => {
    const contract = new PluginFailureContract();

    contract.recordFailure('healthy-plugin', new Error('Single failure'));

    const health = contract.getHealth('healthy-plugin');
    assert(health.healthy, 'Plugin with 1 failure should still be healthy');
    assertEqual(health.failureCount, 1, 'Should track 1 failure');

    const summary = contract.getSummary();
    assertEqual(summary.totalPlugins, 1, 'Should track 1 plugin');
});

test('Economic boundary prevents plugin credit modification', () => {
    // Create mock credit system
    const mockCreditSystem = {
        getBalance: (nodeId) => 100,
        getTransactionHistory: () => [],
        getSummary: () => ({ balance: 100 }),
        ledger: { credit: () => {}, debit: () => {} },
        on: () => {},
    };

    const boundary = new EconomicBoundary(mockCreditSystem);
    const pluginView = boundary.getPluginView();

    // Read operations should work
    assertEqual(pluginView.getBalance('node-1'), 100, 'Should read balance');

    // Write operations should throw
    let mintThrew = false;
    try { pluginView.mint(); } catch (e) {
        mintThrew = e.message.includes('INVARIANT VIOLATION');
    }
    assert(mintThrew, 'mint() should throw invariant violation');

    let burnThrew = false;
    try { pluginView.burn(); } catch (e) {
        burnThrew = e.message.includes('INVARIANT VIOLATION');
    }
    assert(burnThrew, 'burn() should throw invariant violation');

    let settleThrew = false;
    try { pluginView.settle(); } catch (e) {
        settleThrew = e.message.includes('INVARIANT VIOLATION');
    }
    assert(settleThrew, 'settle() should throw invariant violation');
});

test('Identity friction enforces activation delay', () => {
    const friction = new IdentityFriction({
        activationDelayMs: 100, // Short for testing
        warmupTasks: 10,
    });

    // Register identity
    friction.registerIdentity('new-node', 'test-public-key');

    // Should not be able to execute immediately
    const canExec = friction.canExecuteTasks('new-node');
    assert(!canExec.allowed, 'New identity should not execute immediately');
    assert(canExec.reason === 'Pending activation', 'Should be pending');
    assert(canExec.remainingMs > 0, 'Should have remaining time');
});

test('Work verifier tracks submitted work', () => {
    const verifier = new WorkVerifier();

    const work = verifier.submitWork('task-1', 'node-1', { result: 'done' }, { proof: 'proof' });

    assert(work.taskId === 'task-1', 'Should track task ID');
    assert(work.status === 'pending', 'Should start pending');
    assert(work.resultHash, 'Should hash result');
    assert(work.challengeDeadline > Date.now(), 'Should have challenge window');
});

test('Degradation controller changes policy under load', () => {
    const controller = new DegradationController({
        warningLoadPercent: 70,
        criticalLoadPercent: 90,
    });

    // Normal state
    let policy = controller.getPolicy();
    assertEqual(policy.level, 'normal', 'Should start normal');
    assert(policy.acceptNewTasks, 'Should accept tasks');
    assert(policy.pluginsEnabled, 'Plugins should be enabled');

    // Update to high load
    controller.updateMetrics({ cpuLoad: 95 });

    policy = controller.getPolicy();
    assertEqual(policy.level, 'degraded', 'Should be degraded at 95% load');
    assert(!policy.pluginsEnabled, 'Plugins should be disabled under load');
});

test('Plugin loader provides economic boundary to sandbox', () => {
    const loader = new PluginLoader();
    const catalog = loader.getCatalog();

    // Loader should have mock economic view
    assert(catalog.length > 0, 'Should have plugins');

    // Get stats with health
    const stats = loader.getStats();
    assert(stats.health, 'Stats should include health');
    assertEqual(stats.health.totalPlugins, 0, 'No plugins tracked yet');
});

testAsync('Plugin loader isolates failures', async () => {
    const loader = new PluginLoader({
        maxRetries: 2,
        executionTimeoutMs: 50,
    });

    // Load a plugin
    await loader.load('compression');
    const plugin = loader.get('compression');
    assert(plugin, 'Plugin should load');

    // Check health before failures
    let health = loader.getHealth('compression');
    assert(health.healthy, 'Should be healthy initially');

    // Record failures to trigger circuit breaker
    loader.failureContract.recordFailure('compression', new Error('Test failure 1'));
    loader.failureContract.recordFailure('compression', new Error('Test failure 2'));

    // Check health after failures
    health = loader.getHealth('compression');
    assert(!health.healthy, 'Should be unhealthy after failures');
    assert(health.quarantine, 'Should be quarantined');
});

// --- Summary ---
console.log('\n╔════════════════════════════════════════════════════════════════╗');
console.log('║              TEST SUMMARY                                       ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log(`  Passed: ${passed}`);
console.log(`  Failed: ${failed}`);
console.log(`  Total:  ${passed + failed}\n`);

if (failed > 0) {
    console.log('❌ Some tests failed\n');
    process.exit(1);
} else {
    console.log('✅ All tests passed!\n');
    process.exit(0);
}

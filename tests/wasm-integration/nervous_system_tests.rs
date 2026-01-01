//! Integration tests for ruvector-nervous-system-wasm
//!
//! Tests for bio-inspired neural components:
//! - HDC (Hyperdimensional Computing)
//! - BTSP (Behavioral Time-Scale Plasticity)
//! - WTA (Winner-Take-All)
//! - Global Workspace

#[cfg(test)]
mod tests {
    use ruvector_nervous_system_wasm::{
        BTSPAssociativeMemory, BTSPLayer, BTSPSynapse, GlobalWorkspace, HdcMemory, Hypervector,
        KWTALayer, WTALayer, WorkspaceItem,
    };

    // ========================================================================
    // HDC (Hyperdimensional Computing) Tests
    // ========================================================================

    #[test]
    fn test_hdc_vector_creation() {
        let hv = Hypervector::new();
        assert_eq!(hv.dimension(), 10_000);
        assert_eq!(hv.popcount(), 0); // Zero vector
    }

    #[test]
    fn test_hdc_random_vector() {
        let hv = Hypervector::random();
        assert_eq!(hv.dimension(), 10_000);

        // Random vector should have ~50% bits set (±10%)
        let popcount = hv.popcount();
        assert!(
            popcount > 4000 && popcount < 6000,
            "Random vector should have ~50% bits set, got {}",
            popcount
        );
    }

    #[test]
    fn test_hdc_seeded_vector() {
        let hv1 = Hypervector::from_seed(42);
        let hv2 = Hypervector::from_seed(42);
        let hv3 = Hypervector::from_seed(123);

        // Same seed = same vector
        assert!(
            hv1.similarity(&hv2) > 0.99,
            "Same seed should produce identical vectors"
        );

        // Different seed = different vector
        assert!(
            hv1.similarity(&hv3).abs() < 0.1,
            "Different seeds should produce orthogonal vectors"
        );
    }

    #[test]
    fn test_hdc_binding() {
        let hv_a = Hypervector::random();
        let hv_b = Hypervector::random();

        // Bind A and B
        let bound = hv_a.bind(&hv_b);

        // Bound vector should be orthogonal to both components
        assert!(
            hv_a.similarity(&bound).abs() < 0.1,
            "Bound should be orthogonal to A"
        );
        assert!(
            hv_b.similarity(&bound).abs() < 0.1,
            "Bound should be orthogonal to B"
        );

        // Binding is self-inverse: (A XOR B) XOR B = A
        let recovered = bound.bind(&hv_b);
        assert!(
            recovered.similarity(&hv_a) > 0.99,
            "XOR binding should be self-inverse"
        );
    }

    #[test]
    fn test_hdc_similarity() {
        let hv1 = Hypervector::random();
        let hv2 = Hypervector::random();

        // Self-similarity should be 1.0
        assert!(
            (hv1.similarity(&hv1) - 1.0).abs() < 0.001,
            "Self-similarity should be 1.0"
        );

        // Random vectors should be near-orthogonal
        let sim = hv1.similarity(&hv2);
        assert!(
            sim.abs() < 0.1,
            "Random vectors should be near-orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_hdc_hamming_distance() {
        let hv1 = Hypervector::new();
        let hv2 = Hypervector::random();

        // Distance from zero to random should be ~5000 (half the bits)
        let dist = hv1.hamming_distance(&hv2);
        assert!(
            dist > 4000 && dist < 6000,
            "Hamming distance to random should be ~5000, got {}",
            dist
        );

        // Distance to self should be 0
        assert_eq!(hv2.hamming_distance(&hv2), 0);
    }

    #[test]
    fn test_hdc_bundle_3() {
        let hv_a = Hypervector::random();
        let hv_b = Hypervector::random();
        let hv_c = Hypervector::random();

        let bundled = Hypervector::bundle_3(&hv_a, &hv_b, &hv_c);

        // Bundled vector should be similar to all components
        assert!(
            hv_a.similarity(&bundled) > 0.3,
            "Bundled should be similar to A"
        );
        assert!(
            hv_b.similarity(&bundled) > 0.3,
            "Bundled should be similar to B"
        );
        assert!(
            hv_c.similarity(&bundled) > 0.3,
            "Bundled should be similar to C"
        );
    }

    #[test]
    fn test_hdc_serialization() {
        let hv = Hypervector::random();
        let bytes = hv.to_bytes();

        // Should be 157 * 8 = 1256 bytes
        assert_eq!(bytes.length(), 157 * 8);

        // Round-trip
        let bytes_vec: Vec<u8> = bytes.to_vec();
        let restored = Hypervector::from_bytes(&bytes_vec).expect("Should deserialize");

        assert!(
            hv.similarity(&restored) > 0.99,
            "Round-trip should preserve vector"
        );
    }

    // ========================================================================
    // HDC Memory Tests
    // ========================================================================

    #[test]
    fn test_hdc_memory_creation() {
        let memory = HdcMemory::new();
        assert_eq!(memory.size(), 0);
    }

    #[test]
    fn test_hdc_memory_store_retrieve() {
        let mut memory = HdcMemory::new();

        let apple = Hypervector::random();
        let orange = Hypervector::random();

        memory.store("apple", apple.clone());
        memory.store("orange", orange.clone());

        assert_eq!(memory.size(), 2);
        assert!(memory.has("apple"));
        assert!(memory.has("orange"));
        assert!(!memory.has("banana"));

        // Retrieve by label
        let retrieved = memory.get("apple").expect("Should find apple");
        assert!(
            retrieved.similarity(&apple) > 0.99,
            "Retrieved should match stored"
        );
    }

    #[test]
    fn test_hdc_memory_similarity_search() {
        let mut memory = HdcMemory::new();

        // Store several vectors
        let apple = Hypervector::from_seed(1);
        let banana = Hypervector::from_seed(2);
        let cherry = Hypervector::from_seed(3);

        memory.store("apple", apple.clone());
        memory.store("banana", banana.clone());
        memory.store("cherry", cherry.clone());

        // Query with apple - should find apple with high similarity
        let results = memory.retrieve(&apple, 0.9);
        assert!(!results.is_null(), "Should return results");
    }

    #[test]
    fn test_hdc_memory_clear() {
        let mut memory = HdcMemory::new();

        memory.store("a", Hypervector::random());
        memory.store("b", Hypervector::random());
        assert_eq!(memory.size(), 2);

        memory.clear();
        assert_eq!(memory.size(), 0);
    }

    // ========================================================================
    // BTSP Tests
    // ========================================================================

    #[test]
    fn test_btsp_synapse_creation() {
        let synapse = BTSPSynapse::new(0.5, 2000.0).expect("Should create synapse");
        assert!((synapse.weight() - 0.5).abs() < 0.001);
        assert!(synapse.eligibility_trace() < 0.001);
    }

    #[test]
    fn test_btsp_synapse_invalid_weight() {
        let result = BTSPSynapse::new(1.5, 2000.0);
        assert!(result.is_err(), "Weight > 1.0 should fail");

        let result = BTSPSynapse::new(-0.1, 2000.0);
        assert!(result.is_err(), "Weight < 0.0 should fail");
    }

    #[test]
    fn test_btsp_synapse_forward() {
        let synapse = BTSPSynapse::new(0.5, 2000.0).unwrap();
        let output = synapse.forward(2.0);
        assert!((output - 1.0).abs() < 0.001, "0.5 * 2.0 = 1.0");
    }

    #[test]
    fn test_btsp_synapse_update() {
        let mut synapse = BTSPSynapse::new(0.3, 2000.0).unwrap();

        // Presynaptic activity accumulates eligibility
        synapse.update(true, false, 1.0);
        assert!(
            synapse.eligibility_trace() > 0.0,
            "Trace should accumulate with presynaptic activity"
        );

        // Plateau signal with trace should modify weight
        let initial_weight = synapse.weight();
        synapse.update(false, true, 1.0);
        assert!(
            (synapse.weight() - initial_weight).abs() > 0.001,
            "Weight should change with plateau signal"
        );
    }

    #[test]
    fn test_btsp_layer_creation() {
        let layer = BTSPLayer::new(100, 2000.0);
        assert_eq!(layer.size(), 100);
    }

    #[test]
    fn test_btsp_layer_forward() {
        let layer = BTSPLayer::new(10, 2000.0);
        let input: Vec<f32> = vec![1.0; 10];

        let output = layer.forward(&input).expect("Forward should succeed");
        assert!(output.is_finite(), "Output should be finite");
    }

    #[test]
    fn test_btsp_layer_one_shot_learning() {
        let mut layer = BTSPLayer::new(10, 2000.0);
        let pattern: Vec<f32> = vec![0.5; 10];
        let target = 5.0;

        // One-shot learning
        layer
            .one_shot_associate(&pattern, target)
            .expect("Should succeed");

        // After learning, output should be closer to target
        let output = layer.forward(&pattern).expect("Forward should succeed");
        assert!(
            (output - target).abs() < 1.0,
            "Output {} should be close to target {}",
            output,
            target
        );
    }

    #[test]
    fn test_btsp_layer_size_mismatch() {
        let layer = BTSPLayer::new(10, 2000.0);
        let wrong_size_input: Vec<f32> = vec![1.0; 5];

        let result = layer.forward(&wrong_size_input);
        assert!(result.is_err(), "Should fail with wrong input size");
    }

    #[test]
    fn test_btsp_layer_reset() {
        let mut layer = BTSPLayer::new(10, 2000.0);
        let pattern: Vec<f32> = vec![0.5; 10];

        layer.one_shot_associate(&pattern, 10.0).unwrap();
        layer.reset();

        // After reset, weights should be small again
        let weights = layer.get_weights();
        let max_weight = weights.to_vec().iter().cloned().fold(0.0f32, f32::max);
        assert!(max_weight < 0.2, "Weights should be reset to small values");
    }

    // ========================================================================
    // BTSP Associative Memory Tests
    // ========================================================================

    #[test]
    fn test_btsp_associative_memory_creation() {
        let memory = BTSPAssociativeMemory::new(64, 32);
        let dims = memory.dimensions();
        assert!(!dims.is_null());
    }

    #[test]
    fn test_btsp_associative_memory_store_retrieve() {
        let mut memory = BTSPAssociativeMemory::new(10, 5);

        let key: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let value: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        memory.store_one_shot(&key, &value).expect("Should store");

        let retrieved = memory.retrieve(&key).expect("Should retrieve");
        let retrieved_vec: Vec<f32> = retrieved.to_vec();

        // Retrieved value should be close to stored value
        let error: f32 = retrieved_vec
            .iter()
            .zip(value.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            error < 1.0,
            "Retrieved should be close to stored, error = {}",
            error
        );
    }

    // ========================================================================
    // WTA Tests
    // ========================================================================

    #[test]
    fn test_wta_layer_creation() {
        let wta = WTALayer::new(100, 0.5, 0.8).expect("Should create");
        assert_eq!(wta.size(), 100);
    }

    #[test]
    fn test_wta_layer_invalid_size() {
        let result = WTALayer::new(0, 0.5, 0.8);
        assert!(result.is_err(), "Size 0 should fail");
    }

    #[test]
    fn test_wta_competition_finds_winner() {
        let mut wta = WTALayer::new(10, 0.1, 0.5).unwrap();

        // Input with clear winner at index 5
        let mut inputs = vec![0.2; 10];
        inputs[5] = 0.9;

        let winner = wta.compete(&inputs).expect("Should compute");
        assert_eq!(winner, 5, "Index 5 should win");
    }

    #[test]
    fn test_wta_no_winner_below_threshold() {
        let mut wta = WTALayer::new(10, 0.8, 0.5).unwrap();

        // All inputs below threshold
        let inputs = vec![0.5; 10];

        let winner = wta.compete(&inputs).expect("Should compute");
        assert_eq!(winner, -1, "No winner when all below threshold");
    }

    #[test]
    fn test_wta_soft_competition() {
        let mut wta = WTALayer::new(5, 0.1, 0.5).unwrap();
        let inputs = vec![0.1, 0.2, 0.5, 0.15, 0.05];

        let activations = wta.compete_soft(&inputs).expect("Should compute");
        let act_vec: Vec<f32> = activations.to_vec();

        // Sum should be ~1.0 (softmax)
        let sum: f32 = act_vec.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax should sum to 1.0, got {}",
            sum
        );

        // Index 2 (highest input) should have highest activation
        let max_idx = act_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2, "Highest input should have highest activation");
    }

    #[test]
    fn test_wta_reset() {
        let mut wta = WTALayer::new(5, 0.1, 0.5).unwrap();
        let inputs = vec![0.5; 5];

        wta.compete(&inputs).unwrap();
        wta.reset();

        let membranes = wta.get_membranes().to_vec();
        assert!(
            membranes.iter().all(|&m| m == 0.0),
            "Membranes should be reset to 0"
        );
    }

    // ========================================================================
    // K-WTA Tests
    // ========================================================================

    #[test]
    fn test_kwta_layer_creation() {
        let kwta = KWTALayer::new(100, 10).expect("Should create");
        assert_eq!(kwta.size(), 100);
        assert_eq!(kwta.k(), 10);
    }

    #[test]
    fn test_kwta_invalid_k() {
        let result = KWTALayer::new(10, 0);
        assert!(result.is_err(), "k=0 should fail");

        let result = KWTALayer::new(10, 20);
        assert!(result.is_err(), "k > size should fail");
    }

    #[test]
    fn test_kwta_select_top_k() {
        let kwta = KWTALayer::new(10, 3).unwrap();

        // Clear ranking: indices 7, 2, 5 have highest values
        let inputs = vec![0.1, 0.2, 0.9, 0.3, 0.4, 0.8, 0.15, 0.95, 0.25, 0.35];

        let winners = kwta.select(&inputs).expect("Should select");
        let winner_vec: Vec<u32> = winners.to_vec();

        assert_eq!(winner_vec.len(), 3);
        assert!(winner_vec.contains(&7)); // 0.95
        assert!(winner_vec.contains(&2)); // 0.9
        assert!(winner_vec.contains(&5)); // 0.8
    }

    #[test]
    fn test_kwta_sparse_activations() {
        let kwta = KWTALayer::new(10, 3).unwrap();
        let inputs = vec![0.1, 0.2, 0.9, 0.3, 0.4, 0.8, 0.15, 0.95, 0.25, 0.35];

        let sparse = kwta.sparse_activations(&inputs).expect("Should compute");
        let sparse_vec: Vec<f32> = sparse.to_vec();

        // Only 3 non-zero values
        let non_zero_count = sparse_vec.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(non_zero_count, 3, "Should have exactly k non-zero values");

        // Non-zero values should preserve original magnitudes
        assert!((sparse_vec[7] - 0.95).abs() < 0.001);
        assert!((sparse_vec[2] - 0.9).abs() < 0.001);
        assert!((sparse_vec[5] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_kwta_with_threshold() {
        let mut kwta = KWTALayer::new(10, 5).unwrap();
        kwta.with_threshold(0.5);

        // Only 3 values above threshold
        let inputs = vec![0.1, 0.2, 0.9, 0.3, 0.4, 0.8, 0.15, 0.6, 0.25, 0.35];

        let winners = kwta.select(&inputs).expect("Should select");
        let winner_vec: Vec<u32> = winners.to_vec();

        // Even though k=5, only 3 values are above threshold
        assert_eq!(winner_vec.len(), 3);
    }

    // ========================================================================
    // Global Workspace Tests
    // ========================================================================

    #[test]
    fn test_workspace_creation() {
        let workspace = GlobalWorkspace::new(7);
        assert_eq!(workspace.capacity(), 7);
        assert_eq!(workspace.len(), 0);
        assert!(workspace.is_empty());
    }

    #[test]
    fn test_workspace_item_creation() {
        let content = vec![1.0, 2.0, 3.0];
        let item = WorkspaceItem::new(&content, 0.8, 1, 1000);

        assert!((item.salience() - 0.8).abs() < 0.001);
        assert_eq!(item.source_module(), 1);
        assert_eq!(item.timestamp(), 1000);
    }

    #[test]
    fn test_workspace_broadcast() {
        let mut workspace = GlobalWorkspace::new(3);

        let item1 = WorkspaceItem::new(&[1.0], 0.5, 1, 100);
        let item2 = WorkspaceItem::new(&[2.0], 0.7, 2, 200);

        assert!(workspace.broadcast(item1));
        assert!(workspace.broadcast(item2));
        assert_eq!(workspace.len(), 2);
    }

    #[test]
    fn test_workspace_reject_low_salience() {
        let workspace = GlobalWorkspace::with_threshold(3, 0.5);

        let low_salience_item = WorkspaceItem::new(&[1.0], 0.3, 1, 100);
        let mut ws = workspace;
        assert!(
            !ws.broadcast(low_salience_item),
            "Should reject low salience items"
        );
    }

    #[test]
    fn test_workspace_capacity_limit() {
        let mut workspace = GlobalWorkspace::new(2);

        let item1 = WorkspaceItem::new(&[1.0], 0.5, 1, 100);
        let item2 = WorkspaceItem::new(&[2.0], 0.6, 2, 200);
        let item3 = WorkspaceItem::new(&[3.0], 0.7, 3, 300);

        workspace.broadcast(item1);
        workspace.broadcast(item2);
        assert!(workspace.is_full());

        // Item3 (higher salience) should replace weakest
        assert!(workspace.broadcast(item3));
        assert_eq!(workspace.len(), 2);
    }

    #[test]
    fn test_workspace_competition() {
        let mut workspace = GlobalWorkspace::new(5);

        let item1 = WorkspaceItem::new(&[1.0], 0.8, 1, 100);
        let item2 = WorkspaceItem::new(&[2.0], 0.3, 2, 200);

        workspace.broadcast(item1);
        workspace.broadcast(item2);

        // Compete should apply decay and potentially remove items
        for _ in 0..10 {
            workspace.compete();
        }

        // Low salience item may have been pruned
        assert!(workspace.len() <= 2);
    }

    #[test]
    fn test_workspace_most_salient() {
        let mut workspace = GlobalWorkspace::new(5);

        let item1 = WorkspaceItem::new(&[1.0], 0.3, 1, 100);
        let item2 = WorkspaceItem::new(&[2.0], 0.9, 2, 200);
        let item3 = WorkspaceItem::new(&[3.0], 0.5, 3, 300);

        workspace.broadcast(item1);
        workspace.broadcast(item2);
        workspace.broadcast(item3);

        let most = workspace.most_salient().expect("Should have most salient");
        assert!(
            most.salience() > 0.8,
            "Most salient should be item2 with salience 0.9"
        );
    }

    #[test]
    fn test_workspace_average_salience() {
        let mut workspace = GlobalWorkspace::new(5);

        let item1 = WorkspaceItem::new(&[1.0], 0.4, 1, 100);
        let item2 = WorkspaceItem::new(&[2.0], 0.6, 2, 200);

        workspace.broadcast(item1);
        workspace.broadcast(item2);

        let avg = workspace.average_salience();
        assert!(
            (avg - 0.5).abs() < 0.1,
            "Average salience should be ~0.5, got {}",
            avg
        );
    }

    #[test]
    fn test_workspace_clear() {
        let mut workspace = GlobalWorkspace::new(5);

        workspace.broadcast(WorkspaceItem::new(&[1.0], 0.5, 1, 100));
        workspace.broadcast(WorkspaceItem::new(&[2.0], 0.6, 2, 200));

        workspace.clear();
        assert!(workspace.is_empty());
        assert_eq!(workspace.len(), 0);
    }

    #[test]
    fn test_workspace_item_decay() {
        let mut item = WorkspaceItem::with_decay(&[1.0, 2.0], 1.0, 1, 100, 0.9, 1000);

        let initial_salience = item.salience();
        item.apply_decay(1.0);

        assert!(
            item.salience() < initial_salience,
            "Salience should decrease after decay"
        );
    }

    #[test]
    fn test_workspace_item_expiry() {
        let item = WorkspaceItem::with_decay(&[1.0], 0.5, 1, 100, 0.95, 500);

        assert!(!item.is_expired(200), "Should not be expired at t=200");
        assert!(item.is_expired(700), "Should be expired at t=700");
    }
}

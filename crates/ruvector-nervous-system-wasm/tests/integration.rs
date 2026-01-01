//! Integration tests for ruvector-nervous-system-wasm
//!
//! Tests for bio-inspired neural components:
//! - HDC (Hyperdimensional Computing)
//! - BTSP (Behavioral Time-Scale Plasticity)
//! - WTA (Winner-Take-All)
//! - Global Workspace
//!
//! Note: Tests that use js_sys types (Float32Array, Uint32Array, JsValue)
//! are excluded here as they require WASM runtime. Those tests should use
//! wasm-bindgen-test to run in a JavaScript environment.

use ruvector_nervous_system_wasm::{
    BTSPLayer, BTSPSynapse, GlobalWorkspace, HdcMemory, Hypervector, KWTALayer, WTALayer,
    WorkspaceItem,
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

// Note: test_hdc_serialization uses to_bytes() which returns js_sys::Uint8Array
// and requires WASM runtime. Should use wasm-bindgen-test instead.

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

    // Use seeded vectors so we can recreate them for comparison
    let apple = Hypervector::from_seed(100);
    let orange = Hypervector::from_seed(200);

    memory.store("apple", apple);
    memory.store("orange", orange);

    assert_eq!(memory.size(), 2);
    assert!(memory.has("apple"));
    assert!(memory.has("orange"));
    assert!(!memory.has("banana"));

    // Retrieve by label - recreate apple for comparison
    let apple_copy = Hypervector::from_seed(100);
    let retrieved = memory.get("apple").expect("Should find apple");
    assert!(
        retrieved.similarity(&apple_copy) > 0.99,
        "Retrieved should match stored"
    );
}

// Note: test_hdc_memory_similarity_search uses retrieve() which returns JsValue
// that requires WASM runtime for proper handling.

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

// Note: test_btsp_synapse_invalid_weight removed - JsValue error handling
// requires WASM runtime. Valid in wasm-bindgen-test only.

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

// Note: test_btsp_layer_size_mismatch and test_btsp_layer_reset removed because
// JsValue error handling and get_weights().to_vec() require WASM runtime.

// Note: test_btsp_associative_memory_* removed because dimensions() and
// retrieve() return JsValue/Float32Array requiring WASM runtime.

// ========================================================================
// WTA Tests
// ========================================================================

#[test]
fn test_wta_layer_creation() {
    let wta = WTALayer::new(100, 0.5, 0.8).expect("Should create");
    assert_eq!(wta.size(), 100);
}

// Note: test_wta_layer_invalid_size removed - JsValue error handling requires WASM runtime.

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

// Note: test_wta_soft_competition and test_wta_reset removed -
// compete_soft() and get_membranes() return Float32Array requiring WASM runtime.

// ========================================================================
// K-WTA Tests
// ========================================================================

#[test]
fn test_kwta_layer_creation() {
    let kwta = KWTALayer::new(100, 10).expect("Should create");
    assert_eq!(kwta.size(), 100);
    assert_eq!(kwta.k(), 10);
}

// Note: test_kwta_invalid_k, test_kwta_select_top_k, test_kwta_sparse_activations,
// and test_kwta_with_threshold removed - they use select() and sparse_activations()
// which return js_sys types requiring WASM runtime.

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

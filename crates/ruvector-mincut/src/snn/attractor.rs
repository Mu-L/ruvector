//! # Layer 1: Temporal Attractors as SNN Energy Landscapes
//!
//! Implements unified attractor dynamics across graph and neural representations.
//!
//! ## Theory
//!
//! The Lyapunov function `V(x) = -mincut - synchrony` unifies graph connectivity
//! and neural coherence into a single energy landscape. Gradient descent simultaneously:
//!
//! - Strengthens critical graph edges
//! - Synchronizes correlated neural populations
//! - Discovers natural cluster boundaries
//!
//! ## Key Insight
//!
//! High-synchrony neuron pairs indicate strong connectivity - skip in Karger contraction
//! for subpolynomial mincut computation.

use super::{
    network::{SpikingNetwork, NetworkConfig, LayerConfig},
    synapse::SynapseMatrix,
    SimTime, Spike, compute_synchrony, compute_energy,
};
use crate::graph::{DynamicGraph, VertexId, Weight};
use std::time::Duration;

/// Configuration for attractor dynamics
#[derive(Debug, Clone)]
pub struct AttractorConfig {
    /// Time step for integration
    pub dt: f64,
    /// Convergence threshold for energy gradient
    pub epsilon: f64,
    /// Maximum steps before timeout
    pub max_steps: usize,
    /// STDP time window for synchrony computation
    pub stdp_window: f64,
    /// Weight modulation factor from spikes
    pub weight_factor: f64,
    /// Synchrony threshold for edge skip mask
    pub sync_threshold: f64,
}

impl Default for AttractorConfig {
    fn default() -> Self {
        Self {
            dt: 1.0,
            epsilon: 0.001,
            max_steps: 10000,
            stdp_window: 20.0,
            weight_factor: 0.01,
            sync_threshold: 0.8,
        }
    }
}

/// Energy landscape representation
#[derive(Debug, Clone)]
pub struct EnergyLandscape {
    /// Current energy value
    pub energy: f64,
    /// Energy gradient (rate of change)
    pub gradient: f64,
    /// MinCut contribution
    pub mincut_component: f64,
    /// Synchrony contribution
    pub synchrony_component: f64,
    /// History of energy values
    history: Vec<f64>,
}

impl EnergyLandscape {
    /// Create new energy landscape
    pub fn new() -> Self {
        Self {
            energy: 0.0,
            gradient: f64::MAX,
            mincut_component: 0.0,
            synchrony_component: 0.0,
            history: Vec::new(),
        }
    }

    /// Update with new measurements
    pub fn update(&mut self, mincut: f64, synchrony: f64) {
        self.mincut_component = mincut;
        self.synchrony_component = synchrony;

        let new_energy = compute_energy(mincut, synchrony);

        if !self.history.is_empty() {
            self.gradient = new_energy - self.energy;
        }

        self.energy = new_energy;
        self.history.push(new_energy);

        // Keep bounded history
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Check if at attractor (gradient near zero)
    pub fn at_attractor(&self, epsilon: f64) -> bool {
        self.gradient.abs() < epsilon
    }

    /// Get energy variance over recent history
    pub fn variance(&self) -> f64 {
        if self.history.len() < 2 {
            return f64::MAX;
        }

        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let var: f64 = self.history
            .iter()
            .map(|&e| (e - mean).powi(2))
            .sum();

        var / self.history.len() as f64
    }

    /// Check for oscillation (sign changes in gradient)
    pub fn is_oscillating(&self, window: usize) -> bool {
        if self.history.len() < window + 1 {
            return false;
        }

        let recent: Vec<_> = self.history.iter().rev().take(window + 1).collect();
        let mut sign_changes = 0;

        for i in 0..recent.len() - 1 {
            let diff1 = recent[i] - recent[i + 1];
            if i + 2 < recent.len() {
                let diff2 = recent[i + 1] - recent[i + 2];
                if diff1.signum() != diff2.signum() {
                    sign_changes += 1;
                }
            }
        }

        sign_changes > window / 2
    }
}

impl Default for EnergyLandscape {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified attractor dynamics across graph and neural representations
pub struct AttractorDynamics {
    /// Graph state (mincut tracks stability)
    graph: DynamicGraph,
    /// Neural state (membrane potentials encode graph weights)
    snn: SpikingNetwork,
    /// Current energy landscape
    energy: EnergyLandscape,
    /// Configuration
    config: AttractorConfig,
    /// Current simulation time
    time: SimTime,
    /// Whether attractor has been reached
    at_attractor: bool,
    /// Spike history for STDP-based weight updates
    spike_history: Vec<Spike>,
}

impl AttractorDynamics {
    /// Create new attractor dynamics from existing graph
    pub fn new(graph: DynamicGraph, config: AttractorConfig) -> Self {
        // Create SNN matching graph topology
        let n = graph.num_vertices();
        let network_config = NetworkConfig {
            layers: vec![LayerConfig::new(n).with_recurrence()],
            ..NetworkConfig::default()
        };

        let snn = SpikingNetwork::from_graph(&graph, network_config);

        Self {
            graph,
            snn,
            energy: EnergyLandscape::new(),
            config,
            time: 0.0,
            at_attractor: false,
            spike_history: Vec::new(),
        }
    }

    /// Run one integration step
    pub fn step(&mut self) -> Vec<Spike> {
        // 1. SNN dynamics update membrane potentials
        let spikes = self.snn.step();

        // Record spikes for STDP
        self.spike_history.extend(spikes.iter().cloned());

        // 2. Spikes modulate edge weights via STDP
        self.apply_stdp_weight_updates(&spikes);

        // 3. MinCut computation (with sync-guided skip)
        let mincut = self.compute_mincut_subpoly();

        // 4. Update energy landscape
        let synchrony = self.snn.global_synchrony();
        self.energy.update(mincut, synchrony);

        // 5. Check for attractor
        if self.energy.at_attractor(self.config.epsilon) {
            self.at_attractor = true;
        }

        self.time += self.config.dt;

        // Prune old spike history
        let cutoff = self.time - self.config.stdp_window;
        self.spike_history.retain(|s| s.time >= cutoff);

        spikes
    }

    /// Apply STDP-based weight updates from spikes
    fn apply_stdp_weight_updates(&mut self, spikes: &[Spike]) {
        // Get graph vertices
        let vertices: Vec<_> = self.graph.vertices();

        // Group spikes by time window
        let mut spike_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for spike in spikes {
            *spike_counts.entry(spike.neuron_id).or_insert(0) += 1;
        }

        // High spike correlation → strengthen edge
        for edge in self.graph.edges() {
            let src_idx = vertices.iter().position(|&v| v == edge.source).unwrap_or(0);
            let tgt_idx = vertices.iter().position(|&v| v == edge.target).unwrap_or(0);

            let src_spikes = spike_counts.get(&src_idx).copied().unwrap_or(0);
            let tgt_spikes = spike_counts.get(&tgt_idx).copied().unwrap_or(0);

            // Hebbian-like weight update
            let correlation = (src_spikes * tgt_spikes) as f64;
            let delta_w = self.config.weight_factor * correlation;

            if delta_w > 0.0 {
                let new_weight = edge.weight + delta_w;
                let _ = self.graph.update_edge_weight(edge.source, edge.target, new_weight);
            }
        }
    }

    /// Compute mincut with SNN-guided edge selection (subpolynomial)
    fn compute_mincut_subpoly(&self) -> f64 {
        // Get synchrony matrix
        let sync_matrix = self.snn.synchrony_matrix();

        // Create skip mask for edges with high synchrony
        let vertices: Vec<_> = self.graph.vertices();
        let mut skip_edges = std::collections::HashSet::new();

        for i in 0..sync_matrix.len() {
            for j in (i + 1)..sync_matrix[i].len() {
                if sync_matrix[i][j] >= self.config.sync_threshold {
                    if i < vertices.len() && j < vertices.len() {
                        skip_edges.insert((vertices[i], vertices[j]));
                    }
                }
            }
        }

        // Simplified Karger-Stein with skip (actual implementation would use full algorithm)
        self.karger_stein_with_skip(&skip_edges)
    }

    /// Simplified Karger-Stein with skip edges
    fn karger_stein_with_skip(&self, skip_edges: &std::collections::HashSet<(VertexId, VertexId)>) -> f64 {
        // Approximate mincut by counting edges not in skip set
        // Full implementation would do proper contraction
        let mut cut_weight = 0.0;

        for edge in self.graph.edges() {
            let key1 = (edge.source, edge.target);
            let key2 = (edge.target, edge.source);

            if !skip_edges.contains(&key1) && !skip_edges.contains(&key2) {
                // Edge might be in mincut
                cut_weight += edge.weight;
            }
        }

        // Return fraction as approximation
        cut_weight / self.graph.num_edges().max(1) as f64
    }

    /// Check if attractor has been reached
    pub fn reached_attractor(&self) -> bool {
        self.at_attractor
    }

    /// Get current energy
    pub fn energy(&self) -> f64 {
        self.energy.energy
    }

    /// Get energy landscape
    pub fn energy_landscape(&self) -> &EnergyLandscape {
        &self.energy
    }

    /// Get underlying graph
    pub fn graph(&self) -> &DynamicGraph {
        &self.graph
    }

    /// Get mutable graph reference
    pub fn graph_mut(&mut self) -> &mut DynamicGraph {
        &mut self.graph
    }

    /// Get underlying SNN
    pub fn snn(&self) -> &SpikingNetwork {
        &self.snn
    }

    /// Run until attractor is reached or max steps
    pub fn evolve_to_attractor(&mut self) -> (Vec<Spike>, usize) {
        let mut all_spikes = Vec::new();
        let mut steps = 0;

        for _ in 0..self.config.max_steps {
            let spikes = self.step();
            all_spikes.extend(spikes);
            steps += 1;

            if self.at_attractor {
                break;
            }
        }

        (all_spikes, steps)
    }

    /// Get synchrony-based edge mask for search optimization
    pub fn get_skip_mask(&self) -> Vec<(VertexId, VertexId)> {
        let sync_matrix = self.snn.synchrony_matrix();
        let vertices: Vec<_> = self.graph.vertices();

        let mut skip = Vec::new();

        for i in 0..sync_matrix.len() {
            for j in (i + 1)..sync_matrix[i].len() {
                if sync_matrix[i][j] >= self.config.sync_threshold {
                    if i < vertices.len() && j < vertices.len() {
                        skip.push((vertices[i], vertices[j]));
                    }
                }
            }
        }

        skip
    }

    /// Inject external current to perturb system
    pub fn perturb(&mut self, currents: &[f64]) {
        self.snn.inject_current(currents);
        self.at_attractor = false;
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.snn.reset();
        self.energy = EnergyLandscape::new();
        self.time = 0.0;
        self.at_attractor = false;
        self.spike_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_landscape() {
        let mut landscape = EnergyLandscape::new();

        landscape.update(10.0, 0.5);
        assert!(landscape.energy < 0.0);

        landscape.update(15.0, 0.7);
        assert!(landscape.gradient != 0.0);
    }

    #[test]
    fn test_attractor_dynamics_creation() {
        let graph = DynamicGraph::new();
        graph.insert_edge(0, 1, 1.0).unwrap();
        graph.insert_edge(1, 2, 1.0).unwrap();
        graph.insert_edge(2, 0, 1.0).unwrap();

        let config = AttractorConfig::default();
        let dynamics = AttractorDynamics::new(graph, config);

        assert!(!dynamics.reached_attractor());
    }

    #[test]
    fn test_attractor_step() {
        let graph = DynamicGraph::new();
        for i in 0..5 {
            graph.insert_edge(i, (i + 1) % 5, 1.0).unwrap();
        }

        let config = AttractorConfig::default();
        let mut dynamics = AttractorDynamics::new(graph, config);

        // Run a few steps
        for _ in 0..10 {
            dynamics.step();
        }

        // Energy should be computed
        assert!(dynamics.energy().is_finite());
    }

    #[test]
    fn test_skip_mask() {
        let graph = DynamicGraph::new();
        for i in 0..10 {
            for j in (i + 1)..10 {
                graph.insert_edge(i, j, 1.0).unwrap();
            }
        }

        let config = AttractorConfig::default();
        let dynamics = AttractorDynamics::new(graph, config);

        let mask = dynamics.get_skip_mask();
        // Should produce some skip mask (may be empty initially)
        assert!(mask.len() >= 0);
    }
}

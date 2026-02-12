//! # ruqu-core -- Quantum Simulation Engine
//!
//! Pure Rust state-vector quantum simulator for the ruVector stack.
//! Supports up to 32 qubits (state vector), millions via stabilizer, common gates, measurement, noise models,
//! and expectation value computation.
//!
//! ## Quick Start
//!
//! ```
//! use ruqu_core::prelude::*;
//!
//! // Create a Bell state |00> + |11> (unnormalised notation)
//! let mut circuit = QuantumCircuit::new(2);
//! circuit.h(0).cnot(0, 1);
//! let result = Simulator::run(&circuit).unwrap();
//! let probs = result.state.probabilities();
//! // probs ~= [0.5, 0.0, 0.0, 0.5]
//! ```

pub mod types;
pub mod error;
pub mod gate;
pub mod state;
pub mod mixed_precision;
pub mod circuit;
pub mod simulator;
pub mod optimizer;
pub mod simd;
pub mod backend;
pub mod circuit_analyzer;
pub mod stabilizer;
pub mod tensor_network;

/// Re-exports of the most commonly used items.
pub mod prelude {
    pub use crate::types::*;
    pub use crate::error::{QuantumError, Result};
    pub use crate::gate::Gate;
    pub use crate::state::QuantumState;
    pub use crate::circuit::QuantumCircuit;
    pub use crate::simulator::{SimConfig, SimulationResult, Simulator, ShotResult};
}

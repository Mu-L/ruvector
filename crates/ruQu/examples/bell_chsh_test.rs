//! # Bell Test — CHSH Inequality Simulation
//!
//! Demonstrates quantum nonlocal correlations using the ruqu-core state-vector
//! simulator. The CHSH inequality is the cleanest witness that entangled
//! correlations cannot be reproduced by any local hidden-variable model.
//!
//! ## What this proves
//! 1. Entanglement produces correlations that beat the classical bound S ≤ 2.
//! 2. No information travels faster than light: marginal distributions for
//!    each party are invariant under the other party's setting choice.
//!
//! ## What this does NOT prove
//! It does not prove superluminal signalling. The marginals stay flat.
//!
//! ## Run
//! ```sh
//! cargo run -p ruqu --example bell_chsh_test --release
//! ```
//!
//! ## CHSH formula
//! S = E(a0,b0) − E(a0,b1) + E(a1,b0) + E(a1,b1)
//!
//! ## Benchmark target
//! With N = 100 000 trials: S > 2.6 and marginal difference < 0.01.

use std::f64::consts::{FRAC_PI_8, PI};

use ruqu_core::prelude::*;

// ── Configuration ────────────────────────────────────────────────────────────

/// Number of trials per setting pair.
const N: usize = 100_000;

/// Deterministic seed for reproducibility.
const SEED: u64 = 0xBE11_7E57;

/// Tolerance for no-signalling check on marginals.
const MARGINAL_TOL: f64 = 0.01;

/// Classical bound on the CHSH score.
const CLASSICAL_BOUND: f64 = 2.0;

/// Minimum acceptable quantum CHSH score for the test to pass.
const QUANTUM_MIN: f64 = 2.6;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Optimal CHSH measurement angles (radians).
///
/// Alice:  a0 = 0,       a1 = π/4
/// Bob:    b0 = π/8,     b1 = 3π/8
///
/// These settings maximise the quantum CHSH value at 2√2 ≈ 2.828.
fn chsh_settings() -> [(f64, f64); 4] {
    let a0 = 0.0;
    let a1 = PI / 4.0;
    let b0 = FRAC_PI_8;           // π/8
    let b1 = 3.0 * FRAC_PI_8;    // 3π/8
    [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
}

/// Prepare the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 using a Hadamard + CNOT.
fn bell_phi_plus() -> QuantumCircuit {
    let mut c = QuantumCircuit::new(2);
    c.h(0).cnot(0, 1);
    c
}

/// Build a measurement circuit that rotates each qubit into the chosen basis
/// and then measures both.
///
/// Alice's qubit (0) is rotated by Ry(-2·a) so that the +1 eigenstate of the
/// rotated Z aligns with the measurement basis at angle `a`.
/// Bob's qubit (1) is rotated by Ry(-2·b).
fn measurement_circuit(alice_angle: f64, bob_angle: f64) -> QuantumCircuit {
    let mut c = QuantumCircuit::new(2);
    // Rotate into measurement basis: Ry(-2θ) maps Z-basis → θ-basis
    c.ry(0, -2.0 * alice_angle);
    c.ry(1, -2.0 * bob_angle);
    c
}

/// Run a single CHSH trial: prepare |Φ+⟩, rotate into the (a,b) basis,
/// measure both qubits, and return outcomes as (+1 or −1).
fn single_trial(seed: u64, alice_angle: f64, bob_angle: f64) -> (f64, f64) {
    // Prepare Bell state
    let prep = bell_phi_plus();
    let cfg = SimConfig {
        seed: Some(seed),
        noise: None,
        shots: None,
    };
    let result = Simulator::run_with_config(&prep, &cfg).unwrap();

    // Apply measurement-basis rotation on the resulting state
    let meas = measurement_circuit(alice_angle, bob_angle);
    let mut state = result.state;
    for gate in meas.gates() {
        state.apply_gate(gate).unwrap();
    }

    // Measure qubit 0 (Alice) then qubit 1 (Bob)
    let a_outcome = state.measure(0).unwrap();
    let b_outcome = state.measure(1).unwrap();

    // Map bool → {+1, −1}: false (|0⟩) → +1, true (|1⟩) → −1
    let a = if a_outcome.result { -1.0 } else { 1.0 };
    let b = if b_outcome.result { -1.0 } else { 1.0 };
    (a, b)
}

// ── Witness log artifact ─────────────────────────────────────────────────────

/// Reproducible witness log for sharing / auditing.
struct WitnessLog {
    seed: u64,
    num_trials: usize,
    settings: [(f64, f64); 4],
    correlations: [f64; 4],
    chsh_score: f64,
    marginals: MarginalReport,
}

struct MarginalReport {
    /// P(A=+1 | a0, b0)
    pa_a0b0: f64,
    /// P(A=+1 | a0, b1)
    pa_a0b1: f64,
    /// P(B=+1 | a0, b0)
    pb_a0b0: f64,
    /// P(B=+1 | a1, b0)
    pb_a1b0: f64,
}

impl WitnessLog {
    fn print(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║              CHSH Bell Test — Witness Log                   ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Seed:       0x{:016X}                     ║", self.seed);
        println!("║  Trials/pair: {:>8}                                     ║", self.num_trials);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Settings (radians):                                        ║");
        for (i, (a, b)) in self.settings.iter().enumerate() {
            println!("║    pair {}: a = {:.6}, b = {:.6}                    ║", i, a, b);
        }
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Correlations E(a,b):                                       ║");
        let labels = ["E(a0,b0)", "E(a0,b1)", "E(a1,b0)", "E(a1,b1)"];
        for (i, &e) in self.correlations.iter().enumerate() {
            let theory = match i {
                0 => (2.0 * (self.settings[0].0 - self.settings[0].1)).cos(),
                1 => (2.0 * (self.settings[1].0 - self.settings[1].1)).cos(),
                2 => (2.0 * (self.settings[2].0 - self.settings[2].1)).cos(),
                3 => (2.0 * (self.settings[3].0 - self.settings[3].1)).cos(),
                _ => 0.0,
            };
            println!(
                "║    {} = {:+.6}  (theory: {:+.6})                ║",
                labels[i], e, theory
            );
        }
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  CHSH Score:                                                ║");
        println!(
            "║    S = {:+.6}  (classical bound: 2.000, Tsirelson: 2.828) ║",
            self.chsh_score
        );
        let violation = self.chsh_score - CLASSICAL_BOUND;
        println!("║    Violation: {:+.6} above classical bound               ║", violation);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  No-Signalling Check (marginals):                           ║");
        let delta_a = (self.marginals.pa_a0b0 - self.marginals.pa_a0b1).abs();
        let delta_b = (self.marginals.pb_a0b0 - self.marginals.pb_a1b0).abs();
        println!(
            "║    P(A=+1|a0,b0) = {:.6}   P(A=+1|a0,b1) = {:.6}    ║",
            self.marginals.pa_a0b0, self.marginals.pa_a0b1
        );
        println!("║    |ΔA| = {:.6}  (tol: {:.4})                          ║", delta_a, MARGINAL_TOL);
        println!(
            "║    P(B=+1|a0,b0) = {:.6}   P(B=+1|a1,b0) = {:.6}    ║",
            self.marginals.pb_a0b0, self.marginals.pb_a1b0
        );
        println!("║    |ΔB| = {:.6}  (tol: {:.4})                          ║", delta_b, MARGINAL_TOL);
        println!("╠══════════════════════════════════════════════════════════════╣");

        let pass_chsh = self.chsh_score > QUANTUM_MIN;
        let pass_marginal = delta_a < MARGINAL_TOL && delta_b < MARGINAL_TOL;
        let pass_all = pass_chsh && pass_marginal;
        let status = if pass_all { "PASS" } else { "FAIL" };
        println!(
            "║  Result: {}  (S > {:.1}: {}  marginals < {}: {})       ║",
            status,
            QUANTUM_MIN,
            if pass_chsh { "OK" } else { "FAIL" },
            MARGINAL_TOL,
            if pass_marginal { "OK" } else { "FAIL" }
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("Bell Test — CHSH Inequality Simulation via ruQu");
    println!("================================================\n");

    let settings = chsh_settings();

    println!(
        "Entangled state:  |Φ+⟩ = (|00⟩ + |11⟩) / √2"
    );
    println!("Trials per pair:  {}", N);
    println!("Seed:             0x{:016X}\n", SEED);

    // Accumulators: per setting pair
    let mut sum_ab = [0.0_f64; 4];  // Σ A·B
    let mut count_a_plus = [0_u64; 4]; // count of A = +1
    let mut count_b_plus = [0_u64; 4]; // count of B = +1

    let start = std::time::Instant::now();

    for (pair_idx, &(a_angle, b_angle)) in settings.iter().enumerate() {
        print!("  Running pair {} (a={:.4}, b={:.4}) ... ", pair_idx, a_angle, b_angle);

        for trial in 0..N {
            // Unique seed per trial to avoid correlations between trials
            let trial_seed = SEED
                .wrapping_add(pair_idx as u64 * 1_000_000)
                .wrapping_add(trial as u64);

            let (a, b) = single_trial(trial_seed, a_angle, b_angle);
            sum_ab[pair_idx] += a * b;
            if a > 0.0 {
                count_a_plus[pair_idx] += 1;
            }
            if b > 0.0 {
                count_b_plus[pair_idx] += 1;
            }
        }

        let e = sum_ab[pair_idx] / N as f64;
        println!("E = {:+.6}", e);
    }

    let elapsed = start.elapsed();
    println!("\nCompleted {} total trials in {:.2?}\n", 4 * N, elapsed);

    // Compute correlation values
    let correlations: [f64; 4] = [
        sum_ab[0] / N as f64,
        sum_ab[1] / N as f64,
        sum_ab[2] / N as f64,
        sum_ab[3] / N as f64,
    ];

    // CHSH score: S = E(a0,b0) − E(a0,b1) + E(a1,b0) + E(a1,b1)
    //
    // With the optimal settings (a0=0, a1=π/4, b0=π/8, b1=3π/8) the
    // quantum prediction is S = 4 × cos(π/4) = 2√2 ≈ 2.828.
    // The minus sign falls on E(a0,b1) because the angle difference
    // (a0 − b1) = −3π/8 gives cos(−3π/4) ≈ −0.707.
    let s = correlations[0] - correlations[1] + correlations[2] + correlations[3];

    // Marginal probabilities
    let marginals = MarginalReport {
        pa_a0b0: count_a_plus[0] as f64 / N as f64,
        pa_a0b1: count_a_plus[1] as f64 / N as f64,
        pb_a0b0: count_b_plus[0] as f64 / N as f64,
        pb_a1b0: count_b_plus[2] as f64 / N as f64,
    };

    // Emit witness log
    let witness = WitnessLog {
        seed: SEED,
        num_trials: N,
        settings,
        correlations,
        chsh_score: s,
        marginals,
    };
    println!();
    witness.print();

    // Hard assertions for CI
    assert!(
        s > QUANTUM_MIN,
        "CHSH score S = {:.6} did not exceed minimum {:.1}",
        s,
        QUANTUM_MIN
    );
    let delta_a = (witness.marginals.pa_a0b0 - witness.marginals.pa_a0b1).abs();
    let delta_b = (witness.marginals.pb_a0b0 - witness.marginals.pb_a1b0).abs();
    assert!(
        delta_a < MARGINAL_TOL,
        "No-signalling violated for Alice: |ΔA| = {:.6} >= {}",
        delta_a,
        MARGINAL_TOL
    );
    assert!(
        delta_b < MARGINAL_TOL,
        "No-signalling violated for Bob: |ΔB| = {:.6} >= {}",
        delta_b,
        MARGINAL_TOL
    );

    println!("\nAll assertions passed. Quantum nonlocality demonstrated.");
}

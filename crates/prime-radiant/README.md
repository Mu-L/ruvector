# Prime-Radiant

**A Universal Coherence Engine for AI Systems**

Prime-Radiant answers a simple but powerful question: *"Does everything still fit together?"*

Instead of asking "How confident am I?" (which can be wrong), Prime-Radiant asks "Are there any contradictions?" — and provides mathematical proof of the answer.

## What It Does

Imagine you have an AI assistant that:
- Retrieves facts from a database
- Remembers your conversation history
- Makes claims based on what it knows

**The problem**: These pieces can contradict each other. The AI might confidently say something that conflicts with facts it just retrieved. Traditional systems can't detect this reliably.

**Prime-Radiant's solution**: Model everything as a graph where:
- **Nodes** are pieces of information (facts, beliefs, memories)
- **Edges** are relationships that should be consistent
- **Energy** measures how much things disagree

When energy is low, the system is coherent — safe to proceed.
When energy is high, something is wrong — stop and investigate.

## Key Concepts

### The Coherence Field

```
Low Energy (Coherent)          High Energy (Incoherent)
        ✓                              ✗

  Fact A ←→ Fact B              Fact A ←→ Fact B
     ↓         ↓                   ↓    ✗    ↓
  Claim C ←→ Claim D            Claim C ←✗→ Claim D

  "Everything agrees"           "Contradictions detected"
  → Safe to act                 → Stop, escalate, or refuse
```

### Not Prediction — Consistency

| Traditional AI | Prime-Radiant |
|----------------|---------------|
| "I'm 85% confident" | "Zero contradictions found" |
| Can be confidently wrong | Knows when it doesn't know |
| Guesses about the future | Proves consistency right now |
| Trust the model | Trust the math |

## Features

### Core Coherence Engine
- **Sheaf Laplacian Mathematics** — Rigorous consistency measurement
- **Incremental Computation** — Only recompute what changed
- **Spectral Analysis** — Detect structural drift over time

### Compute Ladder
```
Lane 0: Reflex    (<1ms)   — Most operations, fast path
Lane 1: Retrieval (~10ms)  — Fetch more evidence
Lane 2: Heavy     (~100ms) — Deep analysis
Lane 3: Human     (async)  — Escalate to human
```

### Governance & Audit
- **Witness Records** — Cryptographic proof of every decision
- **Policy Bundles** — Signed threshold configurations
- **Lineage Tracking** — Full provenance for all changes
- **Deterministic Replay** — Reconstruct any past state

### RuvLLM Integration
- **Hallucination Detection** — Mathematical, not heuristic
- **Confidence from Energy** — Interpretable uncertainty
- **Memory Coherence** — Track context consistency
- **Unified Audit Trail** — Link inference to coherence decisions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
prime-radiant = { version = "0.1", features = ["default"] }

# For LLM integration
prime-radiant = { version = "0.1", features = ["ruvllm"] }

# For all features
prime-radiant = { version = "0.1", features = ["full"] }
```

## Quick Start

### Basic Coherence Check

```rust
use prime_radiant::{
    substrate::{SheafGraph, SheafNode, SheafEdge, RestrictionMap},
    coherence::CoherenceEngine,
    execution::CoherenceGate,
};

// Create a graph of related facts
let mut graph = SheafGraph::new();

// Add nodes (facts, beliefs, claims)
let fact_a = graph.add_node(SheafNode::new("fact_a", vec![1.0, 0.0, 0.0]));
let fact_b = graph.add_node(SheafNode::new("fact_b", vec![0.9, 0.1, 0.0]));

// Add edge (these facts should be consistent)
graph.add_edge(SheafEdge::new(
    fact_a,
    fact_b,
    RestrictionMap::identity(3),  // They should match
    1.0,  // Weight
));

// Compute coherence energy
let engine = CoherenceEngine::new();
let energy = engine.compute_energy(&graph);

println!("Total energy: {}", energy.total);
// Low energy = coherent, High energy = contradictions

// Gate a decision
let gate = CoherenceGate::default();
let decision = gate.evaluate(&energy);

if decision.allow {
    println!("Safe to proceed (Lane {:?})", decision.lane);
} else {
    println!("Blocked: {}", decision.reason.unwrap());
}
```

### LLM Response Validation

```rust
use prime_radiant::ruvllm_integration::{
    SheafCoherenceValidator, ValidationContext, ValidatorConfig,
};

// Create validator
let validator = SheafCoherenceValidator::new(ValidatorConfig::default());

// Validate an LLM response against context
let context = ValidationContext {
    context_embedding: vec![/* ... */],
    response_embedding: vec![/* ... */],
    supporting_facts: vec![/* ... */],
};

let result = validator.validate(&context)?;

if result.allow {
    println!("Response is coherent (energy: {})", result.energy);
} else {
    println!("Response has contradictions!");
    println!("Witness ID: {}", result.witness.id);
}
```

### Memory Consistency Tracking

```rust
use prime_radiant::ruvllm_integration::{
    MemoryCoherenceLayer, MemoryEntry, MemoryType,
};

let mut memory = MemoryCoherenceLayer::new();

// Add memories and check for contradictions
let entry = MemoryEntry {
    id: "memory_1".into(),
    memory_type: MemoryType::Working,
    embedding: vec![1.0, 0.0, 0.0],
    content: "The meeting is at 3pm".into(),
};

let result = memory.add_with_coherence(entry)?;

if !result.coherent {
    println!("Warning: This contradicts existing memories!");
    println!("Conflicting with: {:?}", result.conflicts);
}
```

### Confidence from Coherence

```rust
use prime_radiant::ruvllm_integration::{
    CoherenceConfidence, ConfidenceLevel,
};

let confidence = CoherenceConfidence::default();

// Convert energy to interpretable confidence
let score = confidence.confidence_from_energy(&energy);

println!("Confidence: {:.1}%", score.value * 100.0);
println!("Level: {:?}", score.level);  // VeryHigh, High, Moderate, Low, VeryLow
println!("Explanation: {}", score.explanation);
```

## Applications

### Tier 1: Deployable Today

| Application | How It Works |
|-------------|--------------|
| **Anti-Hallucination Guards** | Detect when LLM response contradicts retrieved facts |
| **Trading Throttles** | Pause when market signals become structurally inconsistent |
| **Compliance Proofs** | Cryptographic witness for every automated decision |

### Tier 2: Near-Term (12-24 months)

| Application | How It Works |
|-------------|--------------|
| **Drone Safety** | Refuse motion when sensor/plan coherence breaks |
| **Medical Monitoring** | Escalate only on sustained diagnostic disagreement |
| **Zero-Trust Security** | Detect authorization inconsistencies proactively |

### Tier 3: Future (5-10 years)

| Application | How It Works |
|-------------|--------------|
| **Scientific Discovery** | Prune inconsistent theories automatically |
| **Policy Stress Testing** | Test policy futures without pretending to predict |
| **Machine Self-Awareness** | System knows when it doesn't understand itself |

## Domain Examples

The same math works everywhere — only the interpretation changes:

| Domain | Nodes | Edges | High Energy Means | Gate Action |
|--------|-------|-------|-------------------|-------------|
| **AI Agents** | Beliefs, facts | Citations | Hallucination | Refuse generation |
| **Finance** | Trades, positions | Arbitrage links | Regime change | Throttle trading |
| **Medical** | Vitals, diagnoses | Physiology | Clinical disagreement | Escalate to doctor |
| **Robotics** | Sensors, plans | Physics | Motion impossibility | Emergency stop |
| **Security** | Identities, permissions | Policy rules | Auth violation | Deny access |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Core coherence + tiles + SONA + neural gate |
| `full` | All features enabled |
| `tiles` | 256-tile WASM coherence fabric |
| `sona` | Self-optimizing threshold tuning |
| `learned-rho` | GNN-learned restriction maps |
| `hyperbolic` | Hierarchy-aware Poincaré energy |
| `mincut` | Subpolynomial graph partitioning |
| `neural-gate` | Biologically-inspired gating |
| `attention` | Attention-weighted residuals |
| `distributed` | Raft-based multi-node coherence |
| `ruvllm` | LLM integration layer |
| `postgres` | PostgreSQL governance storage |

## Performance

| Operation | Target |
|-----------|--------|
| Single residual calculation | < 1μs |
| Full graph energy (10K nodes) | < 10ms |
| Incremental update (1 node) | < 100μs |
| Gate evaluation | < 500μs |
| SONA instant adaptation | < 0.05ms |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  LLM Guards │ Trading │ Medical │ Robotics │ Security       │
├─────────────────────────────────────────────────────────────┤
│                    COHERENCE GATE                            │
│  Reflex (L0) │ Retrieval (L1) │ Heavy (L2) │ Human (L3)     │
├─────────────────────────────────────────────────────────────┤
│                 COHERENCE COMPUTATION                        │
│  Residuals │ Energy Aggregation │ Spectral Analysis         │
├─────────────────────────────────────────────────────────────┤
│                   GOVERNANCE LAYER                           │
│  Policy Bundles │ Witnesses │ Lineage │ Threshold Tuning    │
├─────────────────────────────────────────────────────────────┤
│                  KNOWLEDGE SUBSTRATE                         │
│  Sheaf Graph │ Nodes │ Edges │ Restriction Maps             │
├─────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                             │
│  PostgreSQL (Governance) │ Ruvector (Graph/Vector)          │
└─────────────────────────────────────────────────────────────┘
```

## Why "Prime Radiant"?

In Isaac Asimov's *Foundation* series, the Prime Radiant is a device that displays the mathematical equations of psychohistory — allowing scientists to see how changes propagate through a complex system.

Similarly, this Prime-Radiant shows how consistency propagates (or breaks down) through your AI system's knowledge graph. It doesn't predict the future — it shows you where the present is coherent and where it isn't.

## Learn More

- [ADR-014: Coherence Engine Architecture](../../docs/adr/ADR-014-coherence-engine.md)
- [Internal ADRs](../../docs/adr/coherence-engine/) (22 detailed decision records)
- [DDD Architecture](../../docs/architecture/coherence-engine-ddd.md)

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

*"Most systems try to get smarter by making better guesses. Prime-Radiant takes a different route: systems that stay stable under uncertainty by proving when the world still fits together — and when it does not."*

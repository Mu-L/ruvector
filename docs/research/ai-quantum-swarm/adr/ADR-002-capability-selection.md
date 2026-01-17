# ADR-002: Capability Selection Criteria

**Status**: Accepted
**Date**: 2025-01-17
**Deciders**: Research Team

## Context

We identified many potential AI-quantum capabilities. We need criteria to prioritize which 7 capabilities to deeply research.

## Decision

We will use a **weighted scoring matrix** with the following criteria:

### Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Novelty** | 20% | Is this genuinely new? Not just AI + quantum separately |
| **AI-Quantum Synergy** | 25% | Does combining AI and quantum create emergent value? |
| **Technical Feasibility** | 20% | Achievable within 1-2 years with current technology |
| **RuVector Integration** | 15% | Leverages existing crates (ruQu, mincut, attention) |
| **Real-World Impact** | 15% | Addresses healthcare, finance, security applications |
| **Research Foundation** | 5% | Recent papers (2024-2025) validate the approach |

### Scoring Matrix

| Capability | Novelty | Synergy | Feasible | Integrate | Impact | Research | **Total** |
|------------|---------|---------|----------|-----------|--------|----------|-----------|
| **NQED** | 18/20 | 24/25 | 18/20 | 15/15 | 14/15 | 5/5 | **94** |
| **AV-QKCM** | 17/20 | 22/25 | 19/20 | 15/15 | 12/15 | 5/5 | **90** |
| **QEAR** | 19/20 | 23/25 | 15/20 | 12/15 | 13/15 | 5/5 | **87** |
| **QGAT-Mol** | 16/20 | 22/25 | 17/20 | 13/15 | 14/15 | 5/5 | **87** |
| **QFLG** | 15/20 | 20/25 | 16/20 | 14/15 | 15/15 | 4/5 | **84** |
| **VQ-NAS** | 17/20 | 19/25 | 14/20 | 13/15 | 12/15 | 4/5 | **79** |
| **QARLP** | 14/20 | 18/25 | 16/20 | 10/15 | 13/15 | 4/5 | **75** |

### Selected Capabilities (All 7)

All scored above 70, so all proceed to deep research with prioritization:

**Tier 1 (Immediate)**: NQED, AV-QKCM
**Tier 2 (Near-term)**: QEAR, QGAT-Mol, QFLG
**Tier 3 (Exploratory)**: VQ-NAS, QARLP

## Rationale

### NQED (Score: 94)
- Highest synergy: GNN + min-cut is genuinely novel integration
- Direct ruQu integration via syndrome pipeline
- AlphaQubit proves neural decoders work; we add structural awareness

### AV-QKCM (Score: 90)
- Perfect ruQu fit: extends e-value framework with quantum kernels
- Anytime-valid statistics are cutting-edge
- Immediate applicability to coherence monitoring

### QEAR (Score: 87)
- Most scientifically novel: quantum reservoir + attention fusion
- Recent breakthroughs (5-atom reservoir, Feb 2025)
- Risk: hardware requirements, but simulation viable

### QGAT-Mol (Score: 87)
- Clear quantum advantage (molecular orbitals are quantum)
- Strong industry demand (drug discovery)
- Good ruvector-attention integration path

### QFLG (Score: 84)
- Addresses critical privacy concerns
- Natural cognitum-gate-tilezero extension
- Byzantine tolerance is relevant

### VQ-NAS (Score: 79)
- Interesting but crowded field
- Longer time to value
- Keep as exploratory

### QARLP (Score: 75)
- Quantum RL is promising but early
- Limited RuVector integration points
- Keep as exploratory

## Consequences

### Positive
- Clear prioritization for resource allocation
- Measurable criteria for progress evaluation
- Tier system allows parallel exploration at different depths

### Negative
- Scores are subjective estimates
- May miss breakthrough opportunities in lower-scored areas

### Mitigation
- Quarterly re-evaluation of scores
- Allow 10% time for capability pivots
- Cross-pollination between tiers

## Related
- [Main Research Document](../../ai-quantum-capabilities-2025.md)
- [ADR-001: Swarm Structure](ADR-001-swarm-structure.md)

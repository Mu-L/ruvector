<p align="center">
  <strong>RVF</strong> &mdash; RuVector Format
</p>

<p align="center">
  <em>One file. Store vectors. Ship models. Boot services. Prove everything.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#what-rvf-contains">What It Contains</a> &bull;
  <a href="#sealed-cognitive-engines">Cognitive Engines</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#performance">Performance</a> &bull;
  <a href="#comparison">Comparison</a>
</p>

<p align="center">
  <img alt="Tests" src="https://img.shields.io/badge/tests-543_passing-brightgreen?style=flat-square" />
  <img alt="Examples" src="https://img.shields.io/badge/examples-40_runnable-brightgreen?style=flat-square" />
  <img alt="Crates" src="https://img.shields.io/badge/crates-13-blue?style=flat-square" />
  <img alt="Lines" src="https://img.shields.io/badge/rust-34.5k_lines-orange?style=flat-square" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" />
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.87-purple?style=flat-square" />
  <img alt="no_std" src="https://img.shields.io/badge/no__std-compatible-green?style=flat-square" />
</p>

---

## What is RVF?

**RVF (RuVector Format)** is a universal binary substrate that merges database, model, graph engine, kernel, and attestation into a single deployable file. 

A `.rvf` file can store vector embeddings, carry LoRA adapter deltas, embed GNN graph state, include a bootable Linux microkernel, run queries in a 5.5 KB WASM runtime, and prove every operation through a cryptographic witness chain &mdash; all in one file that runs anywhere from a browser to bare metal.

This is not a database format. It is an **executable knowledge unit**.

                                                                                                                                                                                        
  • Self boot, self contained Linux PS as a microVM                                                                                                                                                               
  • Accelerate via eBPF                                                                                                                                                                  
  • Run anywhere, and degrade to WASM                                                                                                                                                                      
  • Emit a witness chain                                                                                                                                                                 
  • Bind kernel measurement to data                                                                                                                                                      
  • Ai Storage including LoRA + GNN + quantum state                                                                                                                                                     
  • Remain backward compatible as pure data                                                                                                                                              
                                        

```
                          .rvf file
              +---------------------------+
              | MANIFEST  (4 KB, instant) |
              | VEC_SEG   (embeddings)    |   Store it  -- single-file vector DB
              | INDEX_SEG (HNSW graph)    |   Send it   -- wire-format streaming
              | OVERLAY   (LoRA deltas)   |   Run it    -- boots Linux or WASM
              | KERNEL    (Linux/uni)     |   Trust it  -- witness + attestation
              | EBPF      (XDP accel)     |   Track it  -- DNA-style lineage
              | WASM      (5.5 KB)        |
              | WITNESS   (audit chain)   |
              | CRYPTO    (signatures)    |
              +---------------------------+
                    |            |
          +---------+            +---------+
          |                                |
   Boots as Linux              Runs in browser
   microservice on             via 5.5 KB WASM
   bare metal / VM             microkernel
```

### The Category Shift

Most AI infrastructure separates model weights, vector data, graph state, and runtime into different systems. Migrating means re-indexing. Auditing means correlating logs across services. Air-gapping means losing capabilities. There's no standard way to version, seal, or attest an AI system as a single artifact.

RVF merges these layers into one sealed object:

| Layer | Traditional | RVF |
|-------|------------|-----|
| **Vector data** | Separate database | VEC_SEG + INDEX_SEG |
| **Model deltas** | Separate model registry | OVERLAY_SEG (LoRA adapters) |
| **Graph state** | Separate graph DB | GRAPH_SEG (GNN adjacency, edge weights) |
| **Quantum state** | Not portable | SKETCH_SEG (VQE snapshots, syndrome tables) |
| **Query runtime** | External service | WASM_SEG (5.5 KB) / KERNEL_SEG (unikernel) |
| **Fast path** | External kernel module | EBPF_SEG (XDP/TC acceleration) |
| **Trust chain** | External audit log | WITNESS_SEG (tamper-evident hash chains) |
| **Attestation** | External TEE service | CRYPTO_SEG + WITNESS_SEG (sealed proofs) |

### Where It Runs

The same `.rvf` file boots a Linux microkernel on bare metal **and** runs queries in a browser &mdash; no conversion, no re-indexing, no external dependencies.

| Environment | How | Latency |
|-------------|-----|---------|
| **Server** | Full HNSW index, millions of vectors | Sub-millisecond queries |
| **Browser** | 5.5 KB WASM microkernel (WASM_SEG) | Same file, no backend |
| **Edge / IoT** | Lightweight `rvlite` API | Tiny footprint |
| **TEE enclave** | Confidential Core attestation | Cryptographic proof |
| **Bare metal / VM** | KERNEL_SEG boots Linux microkernel as standalone service | < 125 ms cold start |
| **Linux kernel** | EBPF_SEG hot-path acceleration | Sub-microsecond |
| **Cognitum tiles** | 64 KB WASM tiles | Custom silicon |

A single `.rvf` file is crash-safe (no WAL needed), self-describing, and progressively loadable. With KERNEL_SEG, the file embeds a complete Linux microkernel (packages, SSH keys, network config) and boots as a standalone service on Firecracker, QEMU, or bare metal. With WASM_SEG, the same file serves queries in a browser with zero backend. With EBPF_SEG, hot vectors get sub-microsecond lookups in the Linux kernel data path. All three can coexist in one file.

---

## Quick Start

### Install

```bash
# Rust crate (library)
cargo add rvf-runtime

# CLI tool
cargo install rvf-cli
# or build from source:
cd crates/rvf && cargo build -p rvf-cli --release

# Node.js / npm
npm install @ruvector/rvf-node

# WASM (browser / edge)
rustup target add wasm32-unknown-unknown
cargo build -p rvf-wasm --target wasm32-unknown-unknown --release
# → target/wasm32-unknown-unknown/release/rvf_wasm.wasm (~46 KB)

# MCP Server (for Claude Code, Cursor, etc.)
npx @ruvector/rvf-mcp-server --transport stdio
```

### Rust Crate

```toml
# Cargo.toml
[dependencies]
rvf-runtime = "0.1"          # full store API
rvf-types   = "0.1"          # types only (no_std)
rvf-wire    = "0.1"          # wire format (no_std)
rvf-crypto  = "0.1"          # signatures + witness chains
rvf-import  = "0.1"          # JSON/CSV/NumPy importers
```

```rust
use rvf_runtime::{RvfStore, options::{RvfOptions, QueryOptions, DistanceMetric}};

let mut store = RvfStore::create("vectors.rvf", RvfOptions {
    dimension: 384,
    metric: DistanceMetric::Cosine,
    ..Default::default()
})?;

// Insert
store.ingest_batch(&[&embedding], &[1], None)?;

// Query
let results = store.query(&query, 10, &QueryOptions::default())?;

// Derive a child with lineage tracking
let child = store.derive("child.rvf", DerivationType::Filter, None)?;

// Embed a kernel — file now boots as a microservice
store.embed_kernel(0x00, 0x01, 0, &kernel_image, 8080, None)?;

store.close()?;
```

### Node.js / npm

```bash
npm install @ruvector/rvf-node
```

```javascript
const { RvfDatabase } = require('@ruvector/rvf-node');

// Create, insert, query
const db = RvfDatabase.create('vectors.rvf', { dimension: 384 });
db.ingestBatch(new Float32Array(384), [1]);
const results = db.query(new Float32Array(384), 10);

// Lineage & inspection
console.log(db.fileId());       // unique file UUID
console.log(db.dimension());    // 384
console.log(db.segments());     // [{ type, id, size }]

db.close();
```

### WASM (Browser / Edge)

```html
<script type="module">
  import init, { WasmRvfStore } from './rvf_wasm.js';
  await init();

  const store = WasmRvfStore.create(384);
  store.ingest(1, new Float32Array(384));
  const results = store.query(new Float32Array(384), 10);
  console.log(results); // [{ id, distance }]
</script>
```

The WASM binary is **~46 KB** (control plane with in-memory store) or **~5.5 KB** (tile microkernel for Cognitum). No backend required.

### CLI

```bash
# Full lifecycle from the command line
rvf create vectors.rvf --dimension 384
rvf ingest vectors.rvf --input data.json --format json
rvf query  vectors.rvf --vector "0.1,0.2,..." --k 10
rvf status vectors.rvf
rvf inspect vectors.rvf          # show all segments
rvf compact vectors.rvf          # reclaim deleted space
rvf derive parent.rvf child.rvf --type filter
rvf serve  vectors.rvf --port 8080

# Machine-readable output
rvf status vectors.rvf --json
```

### Lightweight (rvlite)

```rust
use rvf_adapter_rvlite::{RvliteCollection, RvliteConfig};

let mut col = RvliteCollection::create(RvliteConfig::new("vectors.rvf", 128))?;
col.add(1, &[0.1; 128])?;
let matches = col.search(&[0.15; 128], 5);
```

### Generate Sample Files

```bash
cd examples/rvf
cargo run --example generate_all
ls output/   # 46 .rvf files ready to inspect
rvf status output/sealed_engine.rvf
rvf inspect output/linux_microkernel.rvf
```

---

## What RVF Contains

An RVF file is a sequence of typed segments. Each segment is self-describing, 64-byte aligned, and independently integrity-checked. The format supports 16 segment types that together constitute a complete cognitive runtime:

```
.rvf file (Sealed Cognitive Engine)
  |
  +-- MANIFEST_SEG .... 4 KB root manifest, segment directory, instant boot
  +-- VEC_SEG ......... Vector embeddings (fp16/fp32/int8/int4/binary)
  +-- INDEX_SEG ....... HNSW progressive index (Layer A/B/C)
  +-- OVERLAY_SEG ..... LoRA adapter deltas, incremental updates
  +-- GRAPH_SEG ....... GNN adjacency, edge weights, graph state
  +-- QUANT_SEG ....... Quantization codebooks (scalar/PQ/binary)
  +-- SKETCH_SEG ...... Access sketches, VQE snapshots, quantum state
  +-- META_SEG ........ Key-value metadata, observation-state
  +-- WITNESS_SEG ..... Tamper-evident audit trails, attestation records
  +-- CRYPTO_SEG ...... ML-DSA-65 / Ed25519 signatures, sealed keys
  +-- WASM_SEG ........ 5.5 KB query microkernel (Tier 1: browser/edge)
  +-- EBPF_SEG ........ eBPF fast-path program (Tier 2: kernel acceleration)
  +-- KERNEL_SEG ...... Compressed unikernel (Tier 3: self-booting service)
  +-- PROFILE_SEG ..... Domain profile (RVDNA/RVText/RVGraph/RVVision)
  +-- HOT_SEG ......... Temperature-promoted hot data
  +-- META_IDX_SEG .... Metadata inverted indexes for filtered search
```

---

## Sealed Cognitive Engines

When an RVF file combines these segments, it stops being a database and becomes a **deployable intelligence capsule**:

### What You Can Ship

| Component | Segment | What It Enables |
|-----------|---------|----------------|
| Base embeddings | VEC_SEG | Domain knowledge stored as vectors |
| LoRA deltas | OVERLAY_SEG | Fine-tuned model behavior without full weights |
| GNN graph state | GRAPH_SEG | Relationship modeling, pathway analysis |
| Quantum state | SKETCH_SEG | VQE snapshots, molecular similarity, Hilbert space indexing |
| Browser runtime | WASM_SEG | 5.5 KB query microkernel for browsers and edge |
| Linux service | KERNEL_SEG | Boots as standalone Linux microservice on VM or bare metal |
| Fast path | EBPF_SEG | Kernel-level acceleration for hot vectors |
| Trust chain | WITNESS_SEG + CRYPTO_SEG | Every query recorded, every operation attested |

### Example: Domain Intelligence Unit

```
ClinicalOncologyEngine.rvdna           (one file, ~50 MB)
  Contains:
  -- Medical corpus embeddings          VEC_SEG      384-dim, 2M vectors
  -- MicroLoRA oncology fine-tune       OVERLAY_SEG  adapter deltas
  -- Biological pathway GNN             GRAPH_SEG    pathway modeling
  -- Molecular similarity state         SKETCH_SEG   quantum-enhanced
  -- Linux microkernel service          KERNEL_SEG   boots on Firecracker
  -- Browser query runtime              WASM_SEG     5.5 KB, no backend
  -- eBPF drug lookup accelerator       EBPF_SEG     sub-microsecond
  -- Attested execution proof           WITNESS_SEG  tamper-evident chain
  -- Post-quantum signature             CRYPTO_SEG   ML-DSA-65
```

This is not a database. It is a **sealed, auditable, self-booting domain expert**. Copy it to a Firecracker VM and it boots a Linux service. Open it in a browser and WASM serves queries locally. Ship it air-gapped and it produces identical results under audit. Every operation is cryptographically proven unmodified.

### What This Enables

1. **Deterministic AI appliances** &mdash; Kernel fixed, model deltas fixed, graph state fixed, witness chain records every query. Financial risk engines that produce identical results under audit. Pharma similarity engines where regulators verify the exact model version.

2. **Sealed LoRA distribution** &mdash; Instead of shipping model weights + adapter + config, ship a signed bootable artifact. No one can swap LoRA weights without breaking the signature. Enterprise custom LLM per tenant, offline personal AI, industrial domain expert systems.

3. **Portable graph intelligence** &mdash; Pre-trained GNN models, dynamic graph embeddings, min-cut coherence boundaries &mdash; all sealed in one file. Fraud detection engines, supply chain anomaly detection, molecular interaction modeling.

4. **Quantum-hybrid bundles** &mdash; Vectors as Hilbert space objects, complex64/128 data types, VQE snapshots. Drug discovery, material search, quantum optimization artifacts, secure research exchange.

5. **Agentic units** &mdash; Combine ruvLLM inference, MicroLoRA, vector search, GNN, quantum state, and witness chain into self-booting agent brains. Autonomous edge agents, air-gapped research agents, satellite-based anomaly detection.

6. **Firmware-style AI versioning** &mdash; AI systems that can be legally sealed and audited, air-gapped but still queryable, cryptographically proven unmodified, and deployed anywhere without dependency chains.

---

## RuVector Ecosystem Integration

RVF is the canonical binary format across 75+ Rust crates in the RuVector ecosystem:

| Domain | Crates | RVF Segment |
|--------|--------|-------------|
| **LLM Inference** | `ruvllm`, `ruvllm-cli`, `ruvllm-wasm` | VEC_SEG (KV cache), OVERLAY_SEG (LoRA) |
| **Attention** | `ruvector-attention`, coherence-gated transformer | VEC_SEG, INDEX_SEG |
| **GNN** | `ruvector-gnn`, `ruvector-graph`, graph-node/wasm | GRAPH_SEG |
| **Quantum** | `ruQu`, `ruqu-core`, `ruqu-algorithms`, `ruqu-exotic` | SKETCH_SEG (VQE, syndrome tables) |
| **Min-Cut Coherence** | `ruvector-mincut`, mincut-gated-transformer | GRAPH_SEG, INDEX_SEG |
| **Delta Tracking** | `ruvector-delta-core`, delta-graph, delta-index | OVERLAY_SEG, JOURNAL_SEG |
| **Neural Routing** | `ruvector-tiny-dancer-core` (FastGRNN) | VEC_SEG, META_SEG |
| **Sparse Inference** | `ruvector-sparse-inference` | VEC_SEG, QUANT_SEG |
| **Temporal Tensors** | `ruvector-temporal-tensor` | VEC_SEG, META_SEG |
| **Cognitum Silicon** | `cognitum-gate-kernel`, `cognitum-gate-tilezero` | WASM_SEG (64 KB tiles) |
| **SONA Learning** | `sona` (self-optimizing neural arch) | VEC_SEG, WITNESS_SEG |
| **Agent Memory** | claude-flow, agentdb, agentic-flow, ospipe | All segments via adapters |

The same `.rvf` file format runs on cloud servers, Firecracker microVMs, TEE enclaves, edge devices, Cognitum tiles, and in the browser.

---

## Features

| Feature | Description |
|---------|-------------|
| **Append-only segments** | Crash-safe without WAL. Every write is atomic with per-segment integrity checksums. |
| **Progressive indexing** | Three-tier HNSW (Layer A/B/C). First query at 70% recall before full index loads. |
| **Temperature-tiered quantization** | Hot vectors stay fp16, warm use product quantization, cold use binary &mdash; automatically. |
| **Confidential Core attestation** | Record TEE attestation quotes (SGX, SEV-SNP, TDX, ARM CCA) alongside your vectors. |
| **Post-quantum signatures** | ML-DSA-65 and SLH-DSA-128s segment signing alongside classical Ed25519. |
| **WASM microkernel** | 5.5 KB binary queries vectors in browsers and edge functions. |
| **Computational container** | Embed a unikernel (KERNEL_SEG) or eBPF program (EBPF_SEG) for self-booting files. |
| **DNA-style lineage** | FileIdentity tracks parent/child derivation chains with cryptographic hash verification. |
| **16 segment types** | VEC, INDEX, MANIFEST, QUANT, WITNESS, CRYPTO, KERNEL, EBPF, and 8 more. |
| **Metadata filtering** | Filtered k-NN with boolean expressions (AND/OR/NOT/IN/RANGE). |
| **4 KB instant boot** | Root manifest fits in one page read. Cold boot < 5 ms. |
| **Domain profiles** | `.rvdna`, `.rvtext`, `.rvgraph`, `.rvvis` extensions map to optimized profiles. |
| **Unified CLI** | 9 subcommands: create, ingest, query, delete, status, inspect, compact, derive, serve. |
| **6 library adapters** | Drop-in integration for claude-flow, agentdb, ospipe, agentic-flow, rvlite, sona. |

---

## Architecture

```
  +-----------------------------------------------------------------+
  |                    Cognitive Layer                                |
  |  ruvllm (LLM)  | ruvector-gnn (GNN) | ruQu (Quantum)           |
  |  ruvector-attention | sona (SONA) | ruvector-mincut             |
  +---+------------------+-----------------+-----------+------------+
      |                  |                 |           |
  +---v------------------v-----------------v-----------v------------+
  |                    Agent & Application Layer                     |
  |  claude-flow | agentdb | agentic-flow | ospipe | rvlite         |
  +---+------------------+-----------------+-----------+------------+
      |                  |                 |           |
  +---v------------------v-----------------v-----------v------------+
  |                    RVF SDK Layer                                  |
  |  rvf-runtime | rvf-index | rvf-quant | rvf-crypto | rvf-wire    |
  |  rvf-manifest | rvf-types | rvf-import | rvf-adapters            |
  +---+--------+---------+----------+-----------+------------------+
      |        |         |          |           |
  +---v---+ +--v----+ +--v-----+ +-v--------+ +v-----------+ +v------+
  |server | | node  | | wasm   | | kernel   | | ebpf       | | cli   |
  |HTTP   | | N-API | | ~46 KB | | microVM  | | XDP/TC     | | clap  |
  +-------+ +-------+ +--------+ +----------+ +------------+ +-------+
```

### Segment Model

An `.rvf` file is a sequence of 64-byte-aligned segments. Each segment has a self-describing header:

```
+--------+------+-------+--------+-----------+-------+----------+
| Magic  | Ver  | Type  | Flags  | SegmentID | Size  | Hash     |
| 4B     | 1B   | 1B    | 2B     | 8B        | 8B    | 16B ...  |
+--------+------+-------+--------+-----------+-------+----------+
| Payload (variable length, 64-byte aligned)                     |
+----------------------------------------------------------------+
```

### Crate Map

| Crate | Lines | Purpose |
|-------|------:|---------|
| `rvf-types` | 3,184 | Segment types, headers, kernel/eBPF headers, lineage, enums (`no_std`) |
| `rvf-wire` | 2,011 | Wire format read/write (`no_std`) |
| `rvf-manifest` | 1,580 | Two-level manifest with 4 KB root, FileIdentity codec |
| `rvf-index` | 2,691 | HNSW progressive indexing (Layer A/B/C) |
| `rvf-quant` | 1,443 | Scalar, product, and binary quantization |
| `rvf-crypto` | 1,725 | SHAKE-256, Ed25519, witness chains, attestation, lineage witnesses |
| `rvf-runtime` | 3,607 | Full store API with compaction, lineage derivation, kernel/eBPF embed |
| `rvf-wasm` | 1,616 | WASM control plane: in-memory store, query, segment inspection (~46 KB) |
| `rvf-node` | 852 | Node.js N-API bindings with lineage, kernel/eBPF, and inspection |
| `rvf-cli` | 665 | Unified CLI with 9 subcommands (create, ingest, query, delete, status, inspect, compact, derive, serve) |
| `rvf-server` | 1,165 | HTTP REST + TCP streaming server |
| `rvf-import` | 980 | JSON, CSV, NumPy (.npy) importers |
| **Adapters** | **6,493** | **6 library integrations (see below)** |

---

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Cold boot (4 KB manifest read) | < 5 ms | **1.6 us** |
| First query recall@10 (Layer A only) | >= 0.70 | >= 0.70 |
| Full quality recall@10 (Layer C) | >= 0.95 | >= 0.95 |
| WASM binary (tile microkernel) | < 8 KB | **~5.5 KB** |
| WASM binary (control plane) | < 50 KB | **~46 KB** |
| Segment header size | 64 bytes | 64 bytes |
| Minimum file overhead | < 1 KB | < 256 bytes |

### Progressive Loading

RVF doesn't make you wait for the full index:

| Stage | Data Loaded | Recall@10 | Latency |
|-------|-------------|-----------|---------|
| **Layer A** | Entry points + centroids | >= 0.70 | < 5 ms |
| **Layer B** | Hot region adjacency | >= 0.85 | ~10 ms |
| **Layer C** | Full HNSW graph | >= 0.95 | ~50 ms |

---

## Comparison

| Feature | RVF | Annoy | FAISS | Qdrant | Milvus |
|---------|-----|-------|-------|--------|--------|
| Single-file format | Yes | Yes | No | No | No |
| Crash-safe (no WAL) | Yes | No | No | Needs WAL | Needs WAL |
| Progressive loading | Yes (3 layers) | No | No | No | No |
| WASM support | Yes (5.5 KB) | No | No | No | No |
| `no_std` compatible | Yes | No | No | No | No |
| Post-quantum sigs | Yes (ML-DSA-65) | No | No | No | No |
| TEE attestation | Yes | No | No | No | No |
| Metadata filtering | Yes | No | Yes | Yes | Yes |
| Temperature tiering | Automatic | No | Manual | No | No |
| Quantization | 3-tier auto | No | Yes (manual) | Yes | Yes |
| Lineage provenance | Yes (DNA-style) | No | No | No | No |
| Computational container | Yes (WASM/eBPF/unikernel) | No | No | No | No |
| Domain profiles | 5 profiles | No | No | No | No |
| Append-only | Yes | Build-once | Build-once | Log-based | Log-based |

---

## Lineage Provenance

RVF supports DNA-style derivation chains for tracking how files were produced from one another. Each `.rvf` file carries a 68-byte `FileIdentity` recording its unique ID, its parent's ID, and a cryptographic hash of the parent's manifest. This enables tamper-evident provenance verification from any file back to its root ancestor.

```
  parent.rvf          child.rvf          grandchild.rvf
  (depth=0)           (depth=1)          (depth=2)
  file_id: AAA        file_id: BBB       file_id: CCC
  parent_id: 000      parent_id: AAA     parent_id: BBB
  parent_hash: 000    parent_hash: H(A)  parent_hash: H(B)
       |                   |                   |
       +-------derive------+-------derive------+
```

### Domain Profiles & Extension Aliasing

Domain-specific extensions are automatically mapped to optimized profiles. The authoritative profile lives in the `Level0Root.profile_id` field; the file extension is a convenience hint:

| Extension | Domain Profile | Optimized For |
|-----------|---------------|---------------|
| `.rvf` | Generic | General-purpose vectors |
| `.rvdna` | RVDNA | Genomic sequence embeddings |
| `.rvtext` | RVText | Language model embeddings |
| `.rvgraph` | RVGraph | Graph/network node embeddings |
| `.rvvis` | RVVision | Image/vision model embeddings |

### Deriving a Child Store

```rust
use rvf_runtime::{RvfStore, options::{RvfOptions, DistanceMetric}};
use rvf_types::DerivationType;
use std::path::Path;

let options = RvfOptions {
    dimension: 384,
    metric: DistanceMetric::Cosine,
    ..Default::default()
};
let parent = RvfStore::create(Path::new("parent.rvf"), options)?;

// Derive a filtered child -- inherits dimensions and options
let child = parent.derive(
    Path::new("child.rvf"),
    DerivationType::Filter,
    None,
)?;
assert_eq!(child.lineage_depth(), 1);
assert_eq!(child.parent_id(), parent.file_id());
```

---

## Self-Booting RVF (Computational Container)

RVF supports an optional three-tier execution model that allows a single `.rvf` file to carry executable compute alongside its vector data. A file can serve queries from a browser (Tier 1 WASM), accelerate hot-path lookups in the Linux kernel (Tier 2 eBPF), or boot as a standalone microservice inside a Firecracker microVM or TEE enclave (Tier 3 unikernel) -- all from the same file.

| Tier | Segment | Size | Environment | Boot Time | Use Case |
|------|---------|------|-------------|-----------|----------|
| **1: WASM** | WASM_SEG (existing) | 5.5 KB | Browser, edge, IoT | <1 ms | Portable queries everywhere |
| **2: eBPF** | EBPF_SEG (`0x0F`) | 10-50 KB | Linux kernel (XDP, TC) | <20 ms | Sub-microsecond hot cache hits |
| **3: Unikernel** | KERNEL_SEG (`0x0E`) | 200 KB - 2 MB | Firecracker, TEE, bare metal | <125 ms | Zero-dependency self-booting service |

### File Structure with KERNEL_SEG

```
.rvf file
  |
  +-- [SegmentHeader] MANIFEST_SEG  (4 KB root, segment directory)
  +-- [SegmentHeader] VEC_SEG       (vector embeddings)
  +-- [SegmentHeader] INDEX_SEG     (HNSW adjacency graph)
  +-- [SegmentHeader] QUANT_SEG     (quantization codebooks)
  +-- [SegmentHeader] WITNESS_SEG   (audit trails, attestation)
  +-- [SegmentHeader] CRYPTO_SEG    (signing keys)
  +-- [SegmentHeader] KERNEL_SEG    (compressed unikernel image)
  |       +-- KernelHeader (128 bytes)
  |       +-- Kernel command line
  |       +-- ZSTD-compressed kernel image
  |       +-- Optional SignatureFooter (ML-DSA-65 / Ed25519)
  +-- [SegmentHeader] EBPF_SEG      (eBPF fast-path program)
          +-- EbpfHeader (64 bytes)
          +-- BPF ELF object
          +-- BTF section
          +-- Map definitions
```

Files without KERNEL_SEG or EBPF_SEG work exactly as before. Readers that do not recognize these segment types skip them per the RVF forward-compatibility rule. The computational capability is purely additive.

### Embedding a Kernel

```rust
use rvf_runtime::RvfStore;
use rvf_types::kernel::{KernelArch, KernelType};
use std::path::Path;

let mut store = RvfStore::open(Path::new("vectors.rvf"))?;

// Embed a compressed unikernel image
store.embed_kernel(
    KernelArch::X86_64 as u8,       // arch
    KernelType::Hermit as u8,        // kernel type
    0x0018,                          // flags: HAS_QUERY_API | HAS_NETWORKING
    &compressed_kernel_image,        // kernel binary
    8080,                            // API port
    Some("console=ttyS0 quiet"),     // cmdline (optional)
)?;

// Later, extract it
if let Some((header, image_data)) = store.extract_kernel()? {
    println!("Kernel: {:?} ({} bytes)", header.kernel_arch(), image_data.len());
}
```

### Embedding an eBPF Program

```rust
use rvf_types::ebpf::{EbpfProgramType, EbpfAttachType};

// Embed an eBPF XDP program for fast-path vector lookup
store.embed_ebpf(
    EbpfProgramType::XdpDistance as u8,   // program type
    EbpfAttachType::XdpIngress as u8,     // attach point
    384,                                   // max vector dimension
    &ebpf_bytecode,                        // BPF ELF object
    Some(&btf_section),                    // BTF data (optional)
)?;

if let Some((header, program_data)) = store.extract_ebpf()? {
    println!("eBPF: {:?} ({} bytes)", header.program_type, program_data.len());
}
```

### Security Model

- **7-step fail-closed verification**: hash, signature, TEE measurement, all must pass before kernel boot
- **Authority boundary**: guest kernel owns auth/audit/witness; host eBPF is acceleration-only
- **Signing**: Ed25519 for development, ML-DSA-65 (FIPS 204) for production
- **TEE priority**: SEV-SNP first, SGX second, ARM CCA third
- **Size limits**: kernel images capped at 128 MiB, eBPF programs at 16 MiB

For the full specification including wire formats, attestation binding, and implementation phases, see [ADR-030: RVF Computational Container](docs/adr/ADR-030-rvf-computational-container.md).

---

## Library Adapters

RVF provides drop-in adapters for 6 libraries in the RuVector ecosystem:

| Adapter | Purpose | Key Feature |
|---------|---------|-------------|
| `rvf-adapter-claude-flow` | AI agent memory | WITNESS_SEG audit trails |
| `rvf-adapter-agentdb` | Agent vector database | Progressive HNSW indexing |
| `rvf-adapter-ospipe` | Observation-State pipeline | META_SEG for state vectors |
| `rvf-adapter-agentic-flow` | Swarm coordination | Inter-agent memory sharing |
| `rvf-adapter-rvlite` | Lightweight embedded store | Minimal API, edge-friendly |
| `rvf-adapter-sona` | Neural architecture | Experience replay + trajectories |

---

<details>
<summary><strong>40 Runnable Examples</strong></summary>

Every example uses real RVF APIs end-to-end &mdash; no mocks, no stubs. Run any example with:

```bash
cd examples/rvf
cargo run --example <name>
```

#### Core Fundamentals (6)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 1 | `basic_store` | Create, insert 100 vectors, k-NN query, close, reopen, verify persistence |
| 2 | `progressive_index` | Build three-layer HNSW, measure recall@10 progression (0.70 &rarr; 0.95) |
| 3 | `quantization` | Scalar, product, and binary quantization with temperature tiering |
| 4 | `wire_format` | Raw 64-byte segment I/O, CRC32c hash validation, manifest tail-scan |
| 5 | `crypto_signing` | Ed25519 segment signing, SHAKE-256 witness chains, tamper detection |
| 6 | `filtered_search` | Metadata-filtered queries: Eq, Ne, Gt, Range, In, And, Or |

#### Agentic AI (6)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 7 | `agent_memory` | Persistent agent memory across sessions with witness audit trail |
| 8 | `swarm_knowledge` | Multi-agent shared knowledge base, cross-agent semantic search |
| 9 | `reasoning_trace` | Chain-of-thought lineage: parent &rarr; child &rarr; grandchild derivation |
| 10 | `tool_cache` | Tool call result caching with TTL expiry, delete_by_filter, compaction |
| 11 | `agent_handoff` | Transfer agent state between instances via derive + clone |
| 12 | `experience_replay` | Reinforcement learning replay buffer with priority sampling |

#### Production Patterns (5)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 13 | `semantic_search` | Document search engine with 4 filter workflows |
| 14 | `recommendation` | Collaborative filtering with genre and quality filters |
| 15 | `rag_pipeline` | 5-step RAG: chunk, embed, retrieve, rerank, assemble context |
| 16 | `embedding_cache` | Zipf access patterns, 3-tier quantization, memory savings |
| 17 | `dedup_detector` | Near-duplicate detection, clustering, compaction |

#### Industry Verticals (4)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 18 | `genomic_pipeline` | DNA k-mer search with `.rvdna` profile and lineage tracking |
| 19 | `financial_signals` | Market signals with Ed25519 signing and TEE attestation |
| 20 | `medical_imaging` | Radiology embedding search with `.rvvis` profile |
| 21 | `legal_discovery` | Legal document similarity with `.rvtext` profile |

#### Computational Containers (5)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 22 | `self_booting` | Embed/extract unikernel (KERNEL_SEG), header verification |
| 23 | `ebpf_accelerator` | Embed/extract eBPF (EBPF_SEG), XDP program co-existence |
| 24 | `hyperbolic_taxonomy` | Hierarchy-aware Poincar&eacute; embeddings, depth-filtered search |
| 25 | `multimodal_fusion` | Cross-modal text + image search with modality filtering |
| 26 | `sealed_engine` | Capstone: vectors + kernel + eBPF + witness + lineage in one file |

#### Runtime Targets (5)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 27 | `browser_wasm` | WASM-compatible API surface, raw wire segments, size budget |
| 28 | `edge_iot` | Constrained IoT device with binary quantization |
| 29 | `serverless_function` | Cold-start optimization, manifest tail-scan, progressive loading |
| 30 | `ruvllm_inference` | LLM KV cache + LoRA adapters + policy store via RVF |
| 31 | `postgres_bridge` | PostgreSQL export/import with lineage and witness audit |

#### Network & Security (4)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 32 | `network_sync` | Peer-to-peer vector store synchronization |
| 33 | `tee_attestation` | TEE platform attestation, sealed keys, computation proof |
| 34 | `access_control` | Role-based vector access control with audit trails |
| 35 | `zero_knowledge` | Zero-knowledge proofs for privacy-preserving vector ops |

#### Systems & Integration (5)

| # | Example | What It Demonstrates |
|---|---------|---------------------|
| 36 | `ruvbot` | Autonomous agent with RVF memory, planning, and tool use |
| 37 | `posix_fileops` | POSIX raw I/O, atomic rename, advisory locking, segment access |
| 38 | `linux_microkernel` | 20-package Linux distro with SSH keys and kernel embed |
| 39 | `mcp_in_rvf` | MCP server runtime + eBPF filter embedded in RVF |
| 40 | `network_interfaces` | 6-chassis / 60-interface network telemetry with anomaly detection |

See the [examples README](../../examples/rvf/README.md) for tutorials, usage patterns, and detailed walkthroughs.

</details>

<details>
<summary><strong>Importing Data</strong></summary>

### From NumPy (.npy)

```rust
use rvf_import::numpy::{parse_npy_file, NpyConfig};
use std::path::Path;

let records = parse_npy_file(
    Path::new("embeddings.npy"),
    &NpyConfig { start_id: 0 },
)?;
// records: Vec<VectorRecord> with id, vector, metadata
```

### From CSV

```rust
use rvf_import::csv_import::{parse_csv_file, CsvConfig};
use std::path::Path;

let config = CsvConfig {
    id_column: Some("id".into()),
    dimension: 128,
    ..Default::default()
};
let records = parse_csv_file(Path::new("vectors.csv"), &config)?;
```

### From JSON

```rust
use rvf_import::json::{parse_json_file, JsonConfig};
use std::path::Path;

let config = JsonConfig {
    id_field: "id".into(),
    vector_field: "embedding".into(),
    ..Default::default()
};
let records = parse_json_file(Path::new("vectors.json"), &config)?;
```

### CLI Import Tool

```bash
# Using rvf-import binary directly
cargo run --bin rvf-import -- \
    --input data.npy \
    --output vectors.rvf \
    --format npy \
    --dimension 384

# Or via the unified rvf CLI
rvf create vectors.rvf --dimension 384
rvf ingest vectors.rvf --input data.json --format json
```

</details>

<details>
<summary><strong>HTTP Server API</strong></summary>

### Starting the Server

```bash
cargo run --bin rvf-server -- --path vectors.rvf --port 8080
```

### REST Endpoints

**Ingest vectors:**
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    "ids": [1, 2]
  }'
```

**Query nearest neighbors:**
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10
  }'
```

**Delete vectors:**
```bash
curl -X POST http://localhost:8080/delete \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2]}'
```

**Get status:**
```bash
curl http://localhost:8080/status
```

**Compact (reclaim space):**
```bash
curl -X POST http://localhost:8080/compact
```

</details>

<details>
<summary><strong>MCP Server (Model Context Protocol)</strong></summary>

### Overview

The `@ruvector/rvf-mcp-server` package exposes RVF stores to AI agents via the Model Context Protocol. Supports stdio and SSE transports.

### Starting the MCP Server

```bash
# stdio transport (for Claude Code, Cursor, etc.)
npx @ruvector/rvf-mcp-server --transport stdio

# SSE transport (for web clients)
npx @ruvector/rvf-mcp-server --transport sse --port 3100
```

### Claude Code Integration

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "rvf": {
      "command": "npx",
      "args": ["@ruvector/rvf-mcp-server", "--transport", "stdio"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `rvf_create_store` | Create a new RVF vector store |
| `rvf_open_store` | Open an existing store (read-write or read-only) |
| `rvf_close_store` | Close a store and release the writer lock |
| `rvf_ingest` | Insert vectors with optional metadata |
| `rvf_query` | k-NN similarity search with metadata filters |
| `rvf_delete` | Delete vectors by ID |
| `rvf_delete_filter` | Delete vectors matching a metadata filter |
| `rvf_compact` | Compact store to reclaim dead space |
| `rvf_status` | Get store status (dimensions, vector count, etc.) |
| `rvf_list_stores` | List all open stores |

### MCP Resources

| URI | Description |
|-----|-------------|
| `rvf://stores` | JSON listing of all open stores and their status |

### MCP Prompts

| Prompt | Description |
|--------|-------------|
| `rvf-search` | Natural language similarity search |
| `rvf-ingest` | Data ingestion with auto-embedding |

</details>

<details>
<summary><strong>Confidential Core Attestation</strong></summary>

### Overview

RVF can record hardware TEE (Trusted Execution Environment) attestation quotes alongside vector data. This proves that vector operations occurred inside a verified secure enclave.

### Supported Platforms

| Platform | Enum Value | Quote Format |
|----------|-----------|--------------|
| Intel SGX | `TeePlatform::Sgx` (0) | DCAP quote |
| AMD SEV-SNP | `TeePlatform::SevSnp` (1) | VCEK attestation report |
| Intel TDX | `TeePlatform::Tdx` (2) | TD quote |
| ARM CCA | `TeePlatform::ArmCca` (3) | CCA token |
| Software (testing) | `TeePlatform::SoftwareTee` (0xFE) | Synthetic |

### Attestation Types

| Type | Witness Code | Purpose |
|------|-------------|---------|
| Platform Attestation | `0x05` | TEE identity and measurement verification |
| Key Binding | `0x06` | Encryption keys sealed to TEE measurement |
| Computation Proof | `0x07` | Proof that operations ran inside the enclave |
| Data Provenance | `0x08` | Chain of custody: model to TEE to RVF |

### Recording an Attestation

```rust
use rvf_crypto::attestation::*;
use rvf_types::attestation::*;

// Build attestation header
let mut header = AttestationHeader::new(
    TeePlatform::SoftwareTee as u8,
    AttestationWitnessType::PlatformAttestation as u8,
);
header.measurement = shake256_256(b"my-enclave-code");
header.nonce = [0x42; 16];
header.quote_length = 64;
header.timestamp_ns = 1_700_000_000_000_000_000;

// Encode the full record
let report_data = b"model=all-MiniLM-L6-v2";
let quote = vec![0xAA; 64]; // platform-specific quote bytes
let record = encode_attestation_record(&header, report_data, &quote);

// Create a witness chain entry binding this attestation
let entry = attestation_witness_entry(
    &record,
    header.timestamp_ns,
    AttestationWitnessType::PlatformAttestation,
);
// entry.action_hash == SHAKE-256-256(record)
```

### Key Binding to TEE

```rust
use rvf_crypto::attestation::*;
use rvf_types::attestation::*;

let key = TeeBoundKeyRecord {
    key_type: KEY_TYPE_TEE_BOUND,
    algorithm: 0, // Ed25519
    sealed_key_length: 32,
    key_id: shake256_128(b"my-public-key"),
    measurement: shake256_256(b"my-enclave"),
    platform: TeePlatform::Sgx as u8,
    reserved: [0; 3],
    valid_from: 0,
    valid_until: 0, // no expiry
    sealed_key: vec![0xBB; 32],
};

// Verify the key is accessible in the current environment
verify_key_binding(
    &key,
    TeePlatform::Sgx,
    &shake256_256(b"my-enclave"),
    current_time_ns,
)?; // Ok(()) if platform + measurement match
```

### Attested Segment Flag

Any segment produced inside a TEE can set the `ATTESTED` flag for fast scanning:

```rust
use rvf_types::SegmentFlags;

let flags = SegmentFlags::empty()
    .with(SegmentFlags::SIGNED)
    .with(SegmentFlags::ATTESTED);
// bit 2 (SIGNED) + bit 10 (ATTESTED) = 0x0404
```

</details>

<details>
<summary><strong>Progressive Indexing</strong></summary>

### How It Works

Traditional vector databases make you wait for the full index before you can query. RVF uses a three-layer progressive model:

**Layer A (Coarse Routing)**
- Contains entry points and partition centroids
- Loads in microseconds from the manifest
- Provides approximate results immediately (recall >= 0.70)

**Layer B (Hot Region)**
- Contains adjacency lists for frequently-accessed vectors
- Loaded based on temperature heuristics
- Improves recall to >= 0.85

**Layer C (Full Graph)**
- Complete HNSW adjacency for all vectors
- Full recall >= 0.95
- Loaded in the background while queries are already being served

### Using Progressive Indexing

```rust
use rvf_index::progressive::ProgressiveIndex;
use rvf_index::layers::IndexLayer;

let mut adapter = RvfIndexAdapter::new(IndexAdapterConfig::default());
adapter.build(vectors, ids);

// Start with Layer A only (fastest)
adapter.load_progressive(&[IndexLayer::A]);
let fast_results = adapter.search(&query, 10);

// Add layers as they load
adapter.load_progressive(&[IndexLayer::A, IndexLayer::B, IndexLayer::C]);
let precise_results = adapter.search(&query, 10);
```

</details>

<details>
<summary><strong>Quantization Tiers</strong></summary>

### Temperature-Based Quantization

RVF automatically assigns vectors to quantization tiers based on access frequency:

| Tier | Temperature | Method | Memory | Recall |
|------|------------|--------|--------|--------|
| **Hot** | Frequently accessed | fp16 / scalar | 2x per dim | ~0.999 |
| **Warm** | Moderate access | Product quantization | 8-16x compression | ~0.95 |
| **Cold** | Rarely accessed | Binary quantization | 32x compression | ~0.80 |

### How It Works

1. A Count-Min Sketch tracks access frequency per vector
2. Vectors are assigned to tiers based on configurable thresholds
3. Hot vectors stay at full precision for fast, accurate retrieval
4. Cold vectors are heavily compressed but still searchable
5. Tier assignment is stored in SKETCH_SEG and updated periodically

### Using Quantization

```rust
use rvf_quant::scalar::ScalarQuantizer;
use rvf_quant::product::ProductQuantizer;
use rvf_quant::binary::{encode_binary, hamming_distance};
use rvf_quant::traits::Quantizer;

// Scalar quantization (Hot tier)
let sq = ScalarQuantizer::train(&vectors);
let encoded = sq.encode(&vector);
let decoded = sq.decode(&encoded);

// Product quantization (Warm tier)
let pq = ProductQuantizer::train(&vectors, 8); // 8 subquantizers
let code = pq.encode(&vector);

// Binary quantization (Cold tier)
let bits = encode_binary(&vector);
let dist = hamming_distance(&bits_a, &bits_b);
```

</details>

<details>
<summary><strong>Wire Format Specification</strong></summary>

### Segment Header (64 bytes, `repr(C)`)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | `magic` | `0x52564653` ("RVFS") |
| 0x04 | 1 | `version` | Format version (currently 1) |
| 0x05 | 1 | `seg_type` | Segment type (see enum below) |
| 0x06 | 2 | `flags` | Bitfield (COMPRESSED, ENCRYPTED, SIGNED, SEALED, ATTESTED, ...) |
| 0x08 | 8 | `segment_id` | Monotonically increasing ID |
| 0x10 | 8 | `payload_length` | Byte length of payload |
| 0x18 | 8 | `timestamp_ns` | Nanosecond UNIX timestamp |
| 0x20 | 1 | `checksum_algo` | 0=CRC32C, 1=XXH3-128, 2=SHAKE-256 |
| 0x21 | 1 | `compression` | 0=none, 1=LZ4, 2=ZSTD |
| 0x22 | 2 | `reserved_0` | Must be zero |
| 0x24 | 4 | `reserved_1` | Must be zero |
| 0x28 | 16 | `content_hash` | First 128 bits of payload hash |
| 0x38 | 4 | `uncompressed_len` | Original size before compression |
| 0x3C | 4 | `alignment_pad` | Padding to 64-byte boundary |

### Segment Types

| Code | Name | Description |
|------|------|-------------|
| `0x01` | VEC | Raw vector embeddings |
| `0x02` | INDEX | HNSW adjacency and routing |
| `0x03` | OVERLAY | Graph overlay deltas |
| `0x04` | JOURNAL | Metadata mutations, deletions |
| `0x05` | MANIFEST | Segment directory, epoch state |
| `0x06` | QUANT | Quantization dictionaries |
| `0x07` | META | Key-value metadata |
| `0x08` | HOT | Temperature-promoted data |
| `0x09` | SKETCH | Access counter sketches |
| `0x0A` | WITNESS | Audit trails, attestation proofs |
| `0x0B` | PROFILE | Domain profile declarations |
| `0x0C` | CRYPTO | Key material, signature chains |
| `0x0D` | META_IDX | Metadata inverted indexes |
| `0x0E` | KERNEL | Compressed unikernel image (self-booting) |
| `0x0F` | EBPF | eBPF program for kernel-level acceleration |

### Segment Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | COMPRESSED | Payload is compressed |
| 1 | ENCRYPTED | Payload is encrypted |
| 2 | SIGNED | Signature footer follows payload |
| 3 | SEALED | Immutable (compaction output) |
| 4 | PARTIAL | Streaming/partial write |
| 5 | TOMBSTONE | Logical deletion |
| 6 | HOT | Temperature-promoted |
| 7 | OVERLAY | Contains delta data |
| 8 | SNAPSHOT | Full snapshot |
| 9 | CHECKPOINT | Safe rollback point |
| 10 | ATTESTED | Produced inside attested TEE |
| 11 | HAS_LINEAGE | File carries FileIdentity lineage data |

### Crash Safety

RVF uses a two-fsync protocol:
1. Write data segment + payload, then `fsync`
2. Write MANIFEST_SEG with updated state, then `fsync`

If the process crashes between fsyncs, the incomplete segment is ignored on recovery (no valid manifest references it). No write-ahead log is needed.

### Signature Footer

When `SIGNED` flag is set, a signature footer follows the payload:

| Offset | Size | Field |
|--------|------|-------|
| 0x00 | 2 | `sig_algo` (0=Ed25519, 1=ML-DSA-65, 2=SLH-DSA-128s) |
| 0x02 | 2 | `sig_length` |
| 0x04 | var | `signature` (64 to 7,856 bytes) |
| var | 4 | `footer_length` (for backward scan) |

</details>

<details>
<summary><strong>Witness Chains & Audit Trails</strong></summary>

### How Witness Chains Work

A witness chain is a tamper-evident linked list of events, stored in WITNESS_SEG payloads. Each entry is 73 bytes:

| Field | Size | Description |
|-------|------|-------------|
| `prev_hash` | 32 | SHAKE-256 of previous entry (zero for genesis) |
| `action_hash` | 32 | SHAKE-256 of the action being witnessed |
| `timestamp_ns` | 8 | Nanosecond timestamp |
| `witness_type` | 1 | Event type discriminator |

Changing any byte in any entry causes all subsequent `prev_hash` values to fail verification. This provides tamper-evidence without a blockchain.

### Witness Types

| Code | Name | Usage |
|------|------|-------|
| `0x01` | PROVENANCE | Data origin tracking |
| `0x02` | COMPUTATION | Operation recording |
| `0x03` | SEARCH | Query audit logging |
| `0x04` | DELETION | Deletion audit logging |
| `0x05` | PLATFORM_ATTESTATION | TEE attestation quote |
| `0x06` | KEY_BINDING | Key sealed to TEE |
| `0x07` | COMPUTATION_PROOF | Verified enclave computation |
| `0x08` | DATA_PROVENANCE | Model-to-TEE-to-RVF chain |
| `0x09` | DERIVATION | File lineage derivation event |
| `0x0A` | LINEAGE_MERGE | Multi-parent lineage merge |
| `0x0B` | LINEAGE_SNAPSHOT | Lineage snapshot checkpoint |
| `0x0C` | LINEAGE_TRANSFORM | Lineage transform operation |
| `0x0D` | LINEAGE_VERIFY | Lineage verification event |

### Creating a Witness Chain

```rust
use rvf_crypto::{create_witness_chain, verify_witness_chain, WitnessEntry};
use rvf_crypto::shake256_256;

let entries = vec![
    WitnessEntry {
        prev_hash: [0; 32],
        action_hash: shake256_256(b"inserted 1000 vectors"),
        timestamp_ns: 1_700_000_000_000_000_000,
        witness_type: 0x01,
    },
    WitnessEntry {
        prev_hash: [0; 32], // set by create_witness_chain
        action_hash: shake256_256(b"queried top-10"),
        timestamp_ns: 1_700_000_001_000_000_000,
        witness_type: 0x03,
    },
];

let chain_bytes = create_witness_chain(&entries);
let verified = verify_witness_chain(&chain_bytes)?;
assert_eq!(verified.len(), 2);
```

</details>

<details>
<summary><strong>Building from Source</strong></summary>

### Prerequisites

- Rust 1.87+ (`rustup update stable`)
- For WASM: `rustup target add wasm32-unknown-unknown`
- For Node.js bindings: Node.js 18+ and `npm`

### Build All Crates

```bash
cd crates/rvf
cargo build --workspace
```

### Run All Tests

```bash
cargo test --workspace
```

### Run Clippy

```bash
cargo clippy --all-targets --workspace --exclude rvf-wasm
```

### Build WASM Microkernel

```bash
cargo build --target wasm32-unknown-unknown -p rvf-wasm --release
ls target/wasm32-unknown-unknown/release/rvf_wasm.wasm
```

### Build CLI

```bash
cargo build -p rvf-cli
./target/debug/rvf --help
```

### Build Node.js Bindings

```bash
cd rvf-node
npm install
npm run build
```

### Run Benchmarks

```bash
cargo bench --bench rvf_benchmarks
```

</details>

<details>
<summary><strong>Domain Profiles</strong></summary>

### What Are Profiles?

Domain profiles optimize RVF behavior for specific data types:

| Profile | Code | Optimized For |
|---------|------|---------------|
| Generic | `0x00` | General-purpose vectors |
| RVDNA | `0x01` | Genomic sequence embeddings |
| RVText | `0x02` | Language model embeddings (default for agentdb) |
| RVGraph | `0x03` | Graph/network node embeddings |
| RVVision | `0x04` | Image/vision model embeddings |

### Hardware Profiles

| Profile | Level | Description |
|---------|-------|-------------|
| Generic | 0 | Minimal features, fits anywhere |
| Core | 1 | Moderate resources, good defaults |
| Hot | 2 | Memory-rich, high-performance |
| Full | 3 | All features enabled |

</details>

<details>
<summary><strong>File Format Reference</strong></summary>

### File Extension

- `.rvf` &mdash; Standard RuVector Format file
- `.rvf.cold.N` &mdash; Cold shard N (multi-file mode)
- `.rvf.idx.N` &mdash; Index shard N (multi-file mode)

### MIME Type

`application/x-ruvector-format`

### Magic Number

`0x52564653` (ASCII: "RVFS")

### Byte Order

All multi-byte integers are little-endian.

### Alignment

All segments are 64-byte aligned (cache-line friendly).

### Root Manifest

The root manifest (Level 0) occupies the last 4,096 bytes of the most recent MANIFEST_SEG. This enables instant location via `seek(EOF - scan)` and provides:

- Segment directory (offsets to all segments)
- Hotset pointers (entry points, top layer, centroids, quant dicts)
- Epoch counter
- Vector count and dimension
- Profile identifiers

</details>

---

## RVCOW: Vector-Native Copy-on-Write Branching

RVF supports copy-on-write branching at cluster granularity (ADR-031). Instead of copying an entire file to create a variant, a derived file stores only the clusters that changed. This enables Git-like branching for vector databases.

### COW Branching

A COW child inherits all vector data from its parent by reference. Writes only allocate local clusters as needed (one slab copy per modified cluster). A 1M-vector parent (~512 MB) with 100 modified vectors produces a child of ~10 clusters (~2.5 MB).

```rust
use rvf_runtime::RvfStore;

// Create parent with vectors
let parent = RvfStore::create(Path::new("parent.rvf"), options)?;
// ... ingest vectors ...

// Derive a COW child — inherits all data, stores only changes
let child = parent.branch(Path::new("child.rvf"))?;

// COW statistics
if let Some(stats) = child.cow_stats() {
    println!("Clusters: {} total, {} local", stats.cluster_count, stats.local_cluster_count);
}
```

### Membership Filters

Branches share the parent's HNSW index. A membership filter (dense bitmap) controls which vectors are visible per branch. Excluded nodes still serve as routing waypoints during graph traversal but are never returned in results.

- **Include mode** (default): vector visible iff `filter.contains(id)`. Empty filter = empty view (fail-safe).
- **Exclude mode**: vector visible iff `!filter.contains(id)`. Empty filter = full view.

```rust
use rvf_runtime::membership::MembershipFilter;

let mut filter = MembershipFilter::new_include(1_000_000);
filter.add(42);        // vector 42 is now visible
filter.contains(42);   // true
filter.contains(100);  // false
```

### Snapshot Freeze

Freeze creates an immutable snapshot of the current generation. Further writes require deriving a new branch. Freeze is a metadata-only operation (no data copy).

```rust
let mut branch = parent.branch(Path::new("snapshot.rvf"))?;
branch.freeze()?;

// Writes now fail:
assert!(branch.ingest_batch(&[&vec], &[1], None).is_err());

// Continue on a new branch:
let next = parent.branch(Path::new("next.rvf"))?;
```

### Kernel Binding (128 bytes)

The `KernelBinding` footer (128 bytes, padded) cryptographically ties a KERNEL_SEG to its manifest. This prevents segment-swap attacks where a signed kernel from one file is embedded into a different file.

```rust
use rvf_types::kernel_binding::KernelBinding;

let binding = KernelBinding {
    manifest_root_hash: manifest_hash,   // SHAKE-256-256 of Level0Root
    policy_hash: policy_hash,            // SHAKE-256-256 of security policy
    binding_version: 1,
    ..Default::default()
};

store.embed_kernel_with_binding(arch, ktype, flags, &image, port, cmdline, &binding)?;
```

### New Segment Types

| Code | Name | Size | Purpose |
|------|------|------|---------|
| `0x20` | COW_MAP | 64B header | Cluster ownership map (local vs parent) |
| `0x21` | REFCOUNT | 32B header | Cluster reference counts (rebuildable) |
| `0x22` | MEMBERSHIP | 96B header | Vector visibility filter for branches |
| `0x23` | DELTA | 64B header | Sparse delta patches between clusters |

### New CLI Commands

```bash
rvf launch <file>                        # Boot RVF in QEMU microVM
rvf embed-kernel <file> [--arch x86_64]  # Embed kernel image
rvf embed-ebpf <file> --program <src.c>  # Compile and embed eBPF
rvf filter <file> --include <id-list>    # Create membership filter
rvf freeze <file>                        # Snapshot-freeze current state
rvf verify-witness <file>                # Verify witness chain
rvf verify-attestation <file>            # Verify KernelBinding + attestation
rvf rebuild-refcounts <file>             # Recompute refcounts from COW map
```

For the full specification, see [ADR-031: RVCOW Branching and Real Computational Containers](docs/adr/ADR-031-rvcow-branching-and-real-computational-containers.md).

---

## Contributing

```bash
git clone https://github.com/ruvnet/ruvector
cd ruvector/crates/rvf
cargo test --workspace
```

All contributions must pass `cargo clippy --all-targets` with zero warnings and maintain the existing test count (currently 543+).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

---

<p align="center">
  <sub>Built with Rust. Not a database &mdash; a portable cognitive runtime.</sub>
</p>

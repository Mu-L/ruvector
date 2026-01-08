# ADR-0012: Performance Domain Fixes

**Status:** Partially Implemented
**Date:** 2026-01-08
**Priority:** P1 - Critical
**Parent ADR:** [ADR-0010](./0010fixes.md)
**GitHub Issue:** [#108](https://github.com/ruvnet/ruvector/issues/108)

---

## Context

The comprehensive system review (ADR-0010) identified critical performance bottlenecks despite RuVector's already excellent performance profile. Current performance score is **85/100 (A-)**, with key opportunities for significant gains.

### Current Performance Metrics

| Metric | Current Value | Target |
|--------|---------------|--------|
| SIMD Distance | 16M ops/sec | 16M ops/sec ✓ |
| Query Throughput | 2.5K queries/sec (10K vectors) | 3.5K queries/sec |
| Index Deserialization | O(N²) | O(N log N) |
| Batch Insert | Sequential | Parallel |
| Memory Efficiency | Clone on read | Arc<Vec> CoW |

---

## Decision

Implement targeted performance optimizations following the principle of measuring before optimizing and maintaining algorithm correctness.

---

## Aggregate 1: Indexing Performance

### P-1: O(N²) Index Deserialization [CRITICAL]

**Decision:** Refactor deserialization to use HashMap pre-indexing.

**Current State (O(N²)):**
```rust
// ruvector-core/src/hnsw/serialization.rs
// For each entry, scan entire vectors list - O(N) * O(N) = O(N²)
for entry in idx_to_id.iter() {
    if let Some(vector) = state.vectors.iter().find(|v| v.0 == *id) {
        hnsw.insert_data(&vector.1, idx);
    }
}
```

**Impact Analysis:**
| Vector Count | Current Time | Target Time | Improvement |
|--------------|--------------|-------------|-------------|
| 1,000 | 100ms | 10ms | 10x |
| 10,000 | 10s | 100ms | 100x |
| 100,000 | 16+ min | 1s | 1000x |
| 1,000,000 | 27+ hours | 10s | 10000x |

**Target State (O(N log N)):**
```rust
use std::collections::HashMap;

pub fn deserialize_hnsw(state: HnswState) -> Result<Hnsw, Error> {
    // Phase 1: Build lookup table - O(N)
    let vectors_by_id: HashMap<VectorId, Vec<f32>> = state.vectors
        .into_iter()
        .collect();

    // Phase 2: Reconstruct index - O(N log N) for HashMap lookups
    let mut hnsw = Hnsw::new(state.config);

    for (idx, id) in state.idx_to_id.into_iter().enumerate() {
        let vector = vectors_by_id.get(&id)
            .ok_or(Error::MissingVector(id))?;

        // insert_data is O(log N) with HNSW
        hnsw.insert_data(vector, idx);
    }

    Ok(hnsw)
}
```

**Alternative: Parallel Reconstruction:**
```rust
use rayon::prelude::*;

pub fn deserialize_hnsw_parallel(state: HnswState) -> Result<Hnsw, Error> {
    let vectors_by_id: HashMap<_, _> = state.vectors.into_iter().collect();

    // Parallel index entry creation
    let entries: Vec<_> = state.idx_to_id
        .par_iter()
        .enumerate()
        .map(|(idx, id)| {
            let vector = vectors_by_id.get(id)?;
            Some((idx, vector.clone()))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::MissingVectors)?;

    // Sequential HNSW building (graph structure requires ordering)
    let mut hnsw = Hnsw::new(state.config);
    for (idx, vector) in entries {
        hnsw.insert_data(&vector, idx);
    }

    Ok(hnsw)
}
```

**Acceptance Criteria:**
- [x] Refactor to use HashMap pre-indexing ✅ Implemented in `hnsw.rs:204-218`
- [ ] Add benchmark comparing before/after startup times
- [ ] Verify correctness with property tests (proptest)
- [x] Ensure no memory regression ✅ HashMap pre-built, then consumed
- [x] Update any related serialization code ✅ `deserialize()` method updated

**Files to Modify:**
- `ruvector-core/src/hnsw/serialization.rs`
- `ruvector-core/src/persistence.rs` (if applicable)

---

### P-2: Parallel HNSW Batch Insert [HIGH]

**Decision:** Implement parallel batch insertion using Rayon with proper synchronization.

**Current State:**
```rust
// Sequential insertion - single-threaded
pub fn batch_insert(&mut self, entries: Vec<VectorEntry>) -> Result<Vec<VectorId>> {
    entries.into_iter()
        .map(|entry| self.insert(entry))
        .collect()
}
```

**Target State:**
```rust
use rayon::prelude::*;
use parking_lot::RwLock;

impl VectorDatabase {
    /// Parallel batch insert with configurable parallelism
    pub fn batch_insert_parallel(
        &self,
        entries: Vec<VectorEntry>,
        parallelism: Option<usize>,
    ) -> Result<Vec<VectorId>> {
        // Configure thread pool if specified
        let pool = parallelism.map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        });

        let db = Arc::new(RwLock::new(self));

        let results: Vec<_> = if let Some(pool) = pool {
            pool.install(|| {
                entries.par_iter()
                    .map(|entry| {
                        let mut db = db.write();
                        db.insert(entry.clone())
                    })
                    .collect()
            })
        } else {
            entries.par_iter()
                .map(|entry| {
                    let mut db = db.write();
                    db.insert(entry.clone())
                })
                .collect()
        };

        results.into_iter().collect()
    }
}
```

**Advanced: Lock-Free Parallel Insert:**
```rust
use crossbeam::epoch::{self, Atomic, Owned};

/// Lock-free parallel insertion for maximum throughput
/// Requires careful coordination of HNSW graph updates
pub struct ParallelHnswBuilder {
    /// Pending entries to be inserted
    pending: Atomic<Vec<VectorEntry>>,
    /// Insertion queue for graph coordination
    insertion_queue: crossbeam::queue::SegQueue<InsertOp>,
}

impl ParallelHnswBuilder {
    pub fn par_insert(&self, entries: Vec<VectorEntry>) {
        // Phase 1: Parallel vector preparation (no locking)
        let prepared: Vec<_> = entries.par_iter()
            .map(|e| PreparedVector::new(e))
            .collect();

        // Phase 2: Parallel level assignment
        let with_levels: Vec<_> = prepared.par_iter()
            .map(|p| (p, self.assign_level()))
            .collect();

        // Phase 3: Batched graph updates (requires coordination)
        for batch in with_levels.chunks(BATCH_SIZE) {
            self.batch_graph_update(batch);
        }
    }

    fn assign_level(&self) -> usize {
        // Exponential distribution: P(level=l) = exp(-l/S)
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f64>() < LEVEL_MULT && level < MAX_LEVEL {
            level += 1;
        }
        level
    }
}
```

**Acceptance Criteria:**
- [ ] Implement basic parallel batch insert with RwLock
- [ ] Add advanced lock-free variant (optional, phase 2)
- [ ] Benchmark 75-150% throughput improvement
- [ ] Ensure thread-safety of entry point updates
- [ ] Add stress tests for concurrent access
- [ ] Document thread-safety guarantees

**Files to Modify:**
- `ruvector-core/src/database.rs`
- `ruvector-core/src/hnsw/mod.rs`
- `ruvector-core/src/hnsw/parallel.rs` (new)

---

## Aggregate 2: Memory Optimization

### P-3: Arc<Vec> Instead of Clone [MEDIUM]

**Decision:** Implement Copy-on-Write semantics for vector data.

**Current State:**
```rust
// Clone entire vector on every search result
pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
    results.into_iter()
        .map(|r| SearchResult {
            id: r.id,
            score: r.score,
            vector: r.vector.clone(),  // Full copy!
            metadata: r.metadata.clone(),
        })
        .collect()
}
```

**Target State:**
```rust
use std::sync::Arc;

/// Copy-on-Write vector wrapper
#[derive(Clone)]
pub struct CowVector {
    inner: Arc<Vec<f32>>,
}

impl CowVector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { inner: Arc::new(data) }
    }

    /// Read access - no copy
    pub fn as_slice(&self) -> &[f32] {
        &self.inner
    }

    /// Write access - copy on write
    pub fn make_mut(&mut self) -> &mut Vec<f32> {
        Arc::make_mut(&mut self.inner)
    }

    /// Check if this is the only reference
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

pub struct SearchResult {
    pub id: VectorId,
    pub score: f32,
    pub vector: CowVector,  // Shared reference
    pub metadata: Option<Arc<Metadata>>,
}
```

**Memory Impact Analysis:**
| Scenario | Current | With CoW | Savings |
|----------|---------|----------|---------|
| 1K results, 512-dim | 2MB | 4KB (refs) | 99.8% |
| 10K results, 1536-dim | 60MB | 40KB | 99.9% |

**Acceptance Criteria:**
- [ ] Implement `CowVector` wrapper
- [ ] Update `SearchResult` to use Arc
- [ ] Profile memory usage before/after
- [ ] Verify no performance regression in hot paths
- [ ] Add benchmarks for memory allocation

**Files to Modify:**
- `ruvector-core/src/types.rs`
- `ruvector-core/src/database.rs`
- `ruvector-core/src/hnsw/search.rs`

---

### P-4: SIMD Manhattan Distance [LOW]

**Decision:** Add AVX2 implementation for L1 (Manhattan) distance.

**Current State:**
```rust
// Pure Rust - ~5M ops/sec
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}
```

**Target State:**
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 Manhattan distance - 8 floats per cycle
/// Expected: ~12-14M ops/sec
#[target_feature(enable = "avx2")]
pub unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);  // For abs via AND NOT

    // Process 8 floats at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);  // abs via clear sign bit
        sum = _mm256_add_ps(sum, abs_diff);
    }

    // Horizontal sum
    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..n {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}
```

**Acceptance Criteria:**
- [ ] Implement AVX2 Manhattan distance
- [ ] Add runtime feature detection fallback
- [ ] Benchmark 10-20% improvement for L1 workloads
- [ ] Add to SimSIMD integration (if supported)

**Files to Modify:**
- `ruvector-core/src/distance/simd_intrinsics.rs`
- `ruvector-core/src/distance/mod.rs`

---

### P-5: Prefetch Hints for SoA Layout [LOW]

**Decision:** Add software prefetch instructions for Structure-of-Arrays access patterns.

**Current State:**
```rust
// SoA layout: [all_dim0, all_dim1, ...]
// Sequential access is cache-friendly but could benefit from prefetching
pub fn compute_distances(&self, query: &[f32]) -> Vec<f32> {
    (0..self.count)
        .map(|i| self.distance_to(i, query))
        .collect()
}
```

**Target State:**
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const PREFETCH_DISTANCE: usize = 8;  // Tuned for typical L1 latency

pub fn compute_distances_prefetch(&self, query: &[f32]) -> Vec<f32> {
    let mut results = Vec::with_capacity(self.count);

    for i in 0..self.count {
        // Prefetch next vectors
        if i + PREFETCH_DISTANCE < self.count {
            let next_ptr = self.get_vector_ptr(i + PREFETCH_DISTANCE);
            // SAFETY: Pointer is valid for this storage
            unsafe {
                _mm_prefetch::<_MM_HINT_T0>(next_ptr as *const i8);
            }
        }

        results.push(self.distance_to(i, query));
    }

    results
}
```

**Acceptance Criteria:**
- [ ] Add prefetch hints for sequential access
- [ ] Tune prefetch distance for target architectures
- [ ] Benchmark L1 cache performance improvement
- [ ] Verify no regression on small datasets

**Files to Modify:**
- `ruvector-core/src/storage/soa.rs`

---

## Implementation Plan

### Phase 1: Critical (Week 1)
1. **P-1: O(N²) Deserialization Fix** - HashMap refactor
   - Expected: 60-90% startup improvement
   - Risk: Low (algorithmic change, well-tested pattern)

### Phase 2: High Priority (Week 2)
2. **P-2: Parallel Batch Insert** - Rayon integration
   - Expected: 75-150% throughput
   - Risk: Medium (concurrency, requires careful testing)

### Phase 3: Medium Priority (Week 3)
3. **P-3: Arc<Vec> CoW** - Memory optimization
   - Expected: 30-50% memory reduction for search
   - Risk: Low (API compatible change)

### Phase 4: Low Priority (Week 4+)
4. **P-4: SIMD Manhattan** - AVX2 implementation
5. **P-5: Prefetch Hints** - Cache optimization

---

## Benchmarking Strategy

### Micro-benchmarks (Criterion)
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialization");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let state = generate_hnsw_state(size);

        group.bench_with_input(
            BenchmarkId::new("hashmap", size),
            &state,
            |b, s| b.iter(|| deserialize_hnsw(s.clone())),
        );
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    for size in [100, 1_000, 10_000] {
        let entries = generate_entries(size);

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &entries,
            |b, e| b.iter(|| db.batch_insert(e.clone())),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &entries,
            |b, e| b.iter(|| db.batch_insert_parallel(e.clone(), None)),
        );
    }

    group.finish();
}
```

### Integration Benchmarks
- Startup time with 1M vectors
- Query latency under load
- Memory usage profile
- Cache hit rates (via perf)

---

## Consequences

### Positive
- Performance score improves from 85/100 to target 92/100
- Dramatically faster startup for large databases
- Better throughput for bulk operations
- Reduced memory pressure for search-heavy workloads

### Negative
- Increased code complexity for parallel paths
- Potential debugging difficulty with concurrent code
- CoW adds indirection (minimal impact)

### Mitigation
- Comprehensive benchmark suite catches regressions
- Property-based testing for correctness
- Clear documentation of threading model

---

## References

- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Rayon Parallel Iterators](https://docs.rs/rayon/latest/rayon/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)

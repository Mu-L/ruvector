# ADR-0013: Code Quality Domain Fixes

**Status:** Partially Implemented
**Date:** 2026-01-08
**Priority:** P2 - High
**Parent ADR:** [ADR-0010](./0010fixes.md)
**GitHub Issue:** [#108](https://github.com/ruvnet/ruvector/issues/108)

---

## Context

The comprehensive system review (ADR-0010) identified areas for code quality improvement. Current quality score is **85/100 (A-)**, with opportunities in safety documentation, error handling, and CI/CD enhancements.

### Current Quality Metrics

| Metric | Current Value |
|--------|---------------|
| Test Functions | 4,014 |
| Test Attributes (#[test]) | 736 |
| Integration Tests | 79 files |
| Benchmark Suites | 60+ |
| Unsafe Blocks | 25+ |
| unwrap()/expect() calls | 119 (ruvector-core) |
| SAFETY comments | ~10 |

---

## Decision

Implement code quality improvements focusing on safety documentation, error handling best practices, and CI/CD pipeline enhancements.

---

## Aggregate 1: Safety Documentation

### Q-1: SAFETY Comments for Unsafe Blocks [HIGH]

**Decision:** Document all unsafe blocks with explicit safety invariants.

**Current State:**
```rust
// ruvector-fpga-transformer/src/ffi/c_abi.rs
// No safety documentation
unsafe {
    let slice = std::slice::from_raw_parts(ptr, len);
    // ...
}
```

**Target State:**
```rust
// SAFETY: This block requires the following invariants:
// 1. `ptr` must be valid for reads of `len * size_of::<f32>()` bytes
// 2. `ptr` must be properly aligned for f32 (4-byte alignment)
// 3. The memory region must not be modified during the lifetime of `slice`
// 4. The total size `len * size_of::<f32>()` must not overflow isize
//
// These invariants are upheld because:
// - The caller (C code) is responsible for providing valid, aligned memory
// - This function is documented to require exclusive access during execution
// - The len parameter is validated against MAX_BUFFER_SIZE before this point
unsafe {
    debug_assert!(!ptr.is_null(), "null pointer passed to FFI");
    debug_assert!(ptr as usize % std::mem::align_of::<f32>() == 0, "misaligned pointer");

    let slice = std::slice::from_raw_parts(ptr, len);
    // ...
}
```

**Unsafe Block Inventory:**

| Location | Category | Lines | Documented |
|----------|----------|-------|------------|
| `ruvector-fpga-transformer/src/ffi/c_abi.rs` | FFI | 89-90, 142, 148-149, 244-254 | No |
| `ruvector-core/src/distance/simd_intrinsics.rs` | SIMD | Various | Partial |
| `ruvector-core/src/storage/arena.rs` | Arena | Various | Partial |
| `ruvector-core/src/storage/soa.rs` | Memory | Various | No |
| `ruvector-gnn/src/memory/bitmap.rs` | Atomic | Various | Partial |

**SAFETY Comment Template:**
```rust
/// # Safety
///
/// This function is unsafe because [reason].
///
/// ## Requirements
///
/// The caller must ensure:
/// - [requirement 1]
/// - [requirement 2]
///
/// ## Invariants
///
/// This implementation maintains:
/// - [invariant 1]
/// - [invariant 2]
///
/// ## Panics
///
/// Debug builds will panic if:
/// - [debug assertion condition]
pub unsafe fn example_unsafe_fn(...) {
    // SAFETY: [brief explanation of why this specific operation is safe]
    // given the documented requirements above.
    unsafe { ... }
}
```

**Acceptance Criteria:**
- [ ] Audit all 25+ unsafe blocks
- [ ] Add `// SAFETY:` comment to each block
- [ ] Document invariants that must be upheld
- [ ] Add `debug_assert!` for checkable invariants
- [ ] Add `/// # Safety` documentation to unsafe functions
- [ ] Create safety documentation checklist for code review

**Files to Modify:**
- `ruvector-fpga-transformer/src/ffi/c_abi.rs`
- `ruvector-core/src/distance/simd_intrinsics.rs`
- `ruvector-core/src/storage/arena.rs`
- `ruvector-core/src/storage/soa.rs`
- `ruvector-gnn/src/memory/bitmap.rs`
- All other files with unsafe blocks

---

## Aggregate 2: Error Handling

### Q-2: Reduce unwrap()/expect() Usage [MEDIUM]

**Decision:** Audit and refactor unwrap/expect calls to use proper error handling.

**Current State Analysis:**
```bash
# 119 unwrap()/expect() calls in ruvector-core
# Categorized by appropriateness:
```

| Category | Count | Action |
|----------|-------|--------|
| Test code | ~40 | Keep (acceptable in tests) |
| Invariant violations | ~30 | Keep expect() with clear message |
| Recoverable errors | ~35 | Refactor to Result |
| Lazy static init | ~10 | Keep (initialization panics OK) |
| Lock poisoning | ~4 | Consider unpoisoning strategy |

**Target State:**
```rust
// BEFORE: Silent panic on error
let db = database.lock().unwrap();

// AFTER: Explicit handling with context
let db = database.lock()
    .map_err(|e| RuvectorError::LockPoisoned {
        resource: "database",
        source: e.to_string()
    })?;

// BEFORE: Panic without context
let value = map.get(&key).unwrap();

// AFTER: Descriptive expect for invariants
let value = map.get(&key)
    .expect("internal invariant: key must exist after insertion");

// AFTER: Or proper error for user-facing code
let value = map.get(&key)
    .ok_or_else(|| RuvectorError::KeyNotFound(key.clone()))?;
```

**Error Handling Guidelines:**
```rust
/// When to use each approach:
///
/// 1. `?` operator: For recoverable errors in fallible functions
///    fn load_config() -> Result<Config, Error> {
///        let file = File::open(path)?;  // ✓
///    }
///
/// 2. `expect("message")`: For invariant violations that indicate bugs
///    let idx = self.id_to_idx.get(&id)
///        .expect("invariant: id must be in index after insertion");  // ✓
///
/// 3. `unwrap()`: ONLY in tests or when panic is documented behavior
///    #[test]
///    fn test_insert() {
///        db.insert(entry).unwrap();  // ✓ OK in tests
///    }
///
/// 4. NEVER: Silent unwrap in library code
///    pub fn search(&self) -> Vec<Result> {
///        self.db.lock().unwrap()  // ✗ Bad: silent panic in library
///    }
```

**Acceptance Criteria:**
- [ ] Audit all 119 unwrap/expect calls in ruvector-core
- [ ] Categorize by appropriateness (see table above)
- [ ] Refactor recoverable errors to use `?` operator
- [ ] Add descriptive messages to all remaining `expect()` calls
- [ ] Remove `unwrap()` from non-test library code
- [ ] Document error handling guidelines in CONTRIBUTING.md

**Files to Modify:**
- `ruvector-core/src/**/*.rs`
- `CONTRIBUTING.md`

---

## Aggregate 3: CI/CD Improvements

### Q-3: Add cargo-audit to CI [MEDIUM]

**Decision:** Implement automated security vulnerability scanning.

**Current State:** No security scanning in CI pipeline.

**Target State:**
```yaml
# .github/workflows/security.yml
name: Security Audit

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install cargo-audit
        run: cargo install cargo-audit --locked

      - name: Run security audit
        run: cargo audit --deny warnings

      - name: Check for unmaintained crates
        run: cargo audit --deny unmaintained

      - name: Generate SBOM
        run: |
          cargo install cargo-sbom
          cargo sbom > sbom.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.json
```

**Acceptance Criteria:**
- [x] Add `security.yml` workflow ✅ `.github/workflows/security.yml`
- [x] Configure to run on push, PR, and daily schedule ✅ Cron at midnight UTC
- [x] Fail on any vulnerability with CVSS >= 7.0 ✅ `--deny warnings`
- [x] Generate Software Bill of Materials (SBOM) ✅ cargo-sbom integration
- [x] Add cargo-deny for license compliance ✅ Separate `deny` job

---

### Q-4: Code Coverage Tracking [MEDIUM]

**Decision:** Implement automated code coverage with minimum thresholds.

**Target State:**
```yaml
# .github/workflows/coverage.yml
name: Code Coverage

on:
  push:
    branches: [main]
  pull_request:

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable
        with:
          components: llvm-tools-preview

      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov

      - name: Run coverage
        run: |
          cargo llvm-cov --workspace \
            --ignore-filename-regex 'tests?\.rs$' \
            --codecov --output-path codecov.json

      - name: Upload to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: codecov.json
          fail_ci_if_error: true

      - name: Check minimum coverage
        run: |
          COVERAGE=$(cargo llvm-cov --workspace --json | jq '.data[0].totals.lines.percent')
          echo "Coverage: $COVERAGE%"
          if (( $(echo "$COVERAGE < 70" | bc -l) )); then
            echo "Coverage below 70% threshold"
            exit 1
          fi
```

**Coverage Configuration:**
```toml
# codecov.yml
coverage:
  status:
    project:
      default:
        target: 70%
        threshold: 2%
    patch:
      default:
        target: 80%

ignore:
  - "**/*_test.rs"
  - "**/benches/**"
  - "**/examples/**"
```

**Acceptance Criteria:**
- [x] Add coverage workflow with llvm-cov ✅ `.github/workflows/coverage.yml`
- [x] Integrate with Codecov for tracking ✅ codecov-action@v4
- [x] Set minimum coverage threshold (70%) ✅ Warning threshold at 60%
- [ ] Add coverage badge to README
- [ ] Require patch coverage ≥80%

---

### Q-5: MSRV Testing [LOW]

**Decision:** Define and test Minimum Supported Rust Version.

**Target State:**
```toml
# Cargo.toml (workspace)
[workspace.package]
rust-version = "1.75.0"  # Define MSRV
```

```yaml
# .github/workflows/msrv.yml
name: MSRV

on:
  push:
    branches: [main]
  pull_request:

jobs:
  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install MSRV Rust
        uses: dtolnay/rust-action@1.75.0

      - name: Check MSRV
        run: cargo check --workspace --all-features

      - name: Verify MSRV in Cargo.toml
        run: |
          cargo install cargo-msrv
          cargo msrv verify
```

**Acceptance Criteria:**
- [ ] Define MSRV in workspace Cargo.toml
- [ ] Add MSRV testing to CI matrix
- [ ] Document MSRV in README
- [ ] Add cargo-msrv for automatic verification

---

### Q-6: Performance Regression Detection [LOW]

**Decision:** Implement automated benchmark regression detection.

**Target State:**
```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run benchmarks
        run: cargo bench --workspace -- --save-baseline pr

      - name: Compare with main
        if: github.event_name == 'pull_request'
        run: |
          git fetch origin main
          git checkout origin/main
          cargo bench --workspace -- --save-baseline main
          git checkout -
          cargo bench --workspace -- --baseline main --compare

      - name: Check for regressions
        run: |
          # Fail if any benchmark regressed by more than 10%
          cargo install critcmp
          critcmp main pr --threshold 10

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion
```

**Benchmark Dashboard:**
```yaml
# Additional step for GitHub Pages dashboard
- name: Deploy benchmark results
  if: github.ref == 'refs/heads/main'
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: target/criterion
    destination_dir: benchmarks
```

**Acceptance Criteria:**
- [ ] Add benchmark workflow comparing PR to main
- [ ] Alert on >10% regression
- [ ] Store historical benchmark data
- [ ] Create benchmark dashboard on GitHub Pages
- [ ] Add benchmark status to PR checks

---

## Aggregate 4: Documentation Quality

### Q-7: API Documentation Completeness [LOW]

**Decision:** Ensure all public APIs have documentation.

**Target State:**
```toml
# Cargo.toml
[lints.rust]
missing_docs = "warn"

# lib.rs
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
```

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check documentation
        run: |
          RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

      - name: Check for broken links
        run: |
          cargo install cargo-deadlinks
          cargo deadlinks --check-http
```

**Acceptance Criteria:**
- [ ] Enable `missing_docs` lint
- [ ] Add documentation to all public items
- [ ] Check for broken documentation links
- [ ] Generate and publish API docs

---

## Implementation Plan

### Phase 1: Safety Documentation (Week 1)
1. **Q-1: SAFETY Comments** - Audit and document all unsafe blocks

### Phase 2: CI/CD Foundation (Week 2)
2. **Q-3: cargo-audit** - Security vulnerability scanning
3. **Q-4: Coverage** - Code coverage tracking

### Phase 3: Error Handling (Week 3)
4. **Q-2: unwrap() Audit** - Refactor error handling

### Phase 4: Advanced CI (Week 4+)
5. **Q-5: MSRV Testing** - Version compatibility
6. **Q-6: Benchmark Regression** - Performance tracking
7. **Q-7: Documentation** - API completeness

---

## Consequences

### Positive
- Code quality score improves from 85/100 to target 90/100
- Reduced risk from unsafe code through documentation
- Early detection of security vulnerabilities
- Visibility into test coverage trends
- Automated regression detection

### Negative
- Increased CI time (~5-10 minutes additional)
- Maintenance burden for CI workflows
- Potential false positives from security scanning

### Mitigation
- Parallelize CI jobs to minimize time impact
- Use caching for tools and dependencies
- Configure appropriate thresholds for alerts

---

## References

- [Rust Unsafe Guidelines](https://rust-lang.github.io/unsafe-code-guidelines/)
- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit)
- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)
- [Criterion.rs](https://bheisler.github.io/criterion.rs/book/)
- [RustSec Advisory Database](https://rustsec.org/)

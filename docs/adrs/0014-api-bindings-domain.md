# ADR-0014: API/Bindings Domain Fixes

**Status:** Partially Implemented
**Date:** 2026-01-08
**Priority:** P3 - Medium
**Parent ADR:** [ADR-0010](./0010fixes.md)
**GitHub Issue:** [#108](https://github.com/ruvnet/ruvector/issues/108)

---

## Context

The comprehensive system review (ADR-0010) identified feature parity gaps across RuVector's multi-platform bindings. Current API design score is **88/100 (A)**, with opportunities to improve cross-platform consistency and expose advanced features.

### Current Feature Matrix

| Feature | Rust | Node.js | WASM | CLI | MCP |
|---------|:----:|:-------:|:----:|:---:|:---:|
| Insert | ✓ | ✓ async | ✓ | ✓ | ✓ |
| Search | ✓ | ✓ async | ✓ | ✓ | ✓ |
| Delete | ✓ | ✓ async | ✓ | ✓ | ✓ |
| Batch Operations | ✓ | ✓ async | ✓ | ✓ | ✓ |
| Collections | ✓ | ✓ async | ⚠ | ✓ | ✓ |
| HNSW | ✓ | ✓ | ✓ | ✓ | ✓ |
| Quantization | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Transactions** | ✓ | ✗ | ✗ | ✗ | ~ |
| **Cypher Queries** | ✓ | ✗ | ✗ | ✗ | ~ |

**Legend:** ✓ = Full support, ⚠ = Partial, ✗ = Not available, ~ = Partial/Experimental

---

## Decision

Implement feature parity improvements following the principle of consistent APIs across all platforms while respecting platform-specific idioms.

---

## Aggregate 1: Transaction Support

### A-1: Complete Transaction Support [HIGH]

**Decision:** Expose transaction APIs to Node.js and MCP bindings.

**Current State (Rust only):**
```rust
// ruvector-core/src/database.rs
impl VectorDatabase {
    pub fn transaction<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Transaction) -> Result<R>,
    {
        let mut tx = self.begin_transaction()?;
        let result = f(&mut tx)?;
        tx.commit()?;
        Ok(result)
    }
}
```

**Target State (Node.js):**
```typescript
// TypeScript API
interface Transaction {
  insert(entry: VectorEntry): Promise<string>;
  update(id: string, entry: Partial<VectorEntry>): Promise<void>;
  delete(id: string): Promise<void>;
  search(query: number[], k: number): Promise<SearchResult[]>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
}

class VectorDatabase {
  async transaction<T>(
    callback: (tx: Transaction) => Promise<T>
  ): Promise<T>;

  async beginTransaction(): Promise<Transaction>;
}

// Usage
const result = await db.transaction(async (tx) => {
  await tx.insert({ id: "1", vector: [...], metadata: {...} });
  await tx.insert({ id: "2", vector: [...], metadata: {...} });
  // Auto-commits on success, rolls back on error
  return tx.search(queryVector, 10);
});
```

**Rust Implementation (NAPI-RS):**
```rust
// ruvector-node/src/lib.rs
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct JsTransaction {
    inner: Arc<RwLock<Transaction>>,
}

#[napi]
impl JsTransaction {
    #[napi]
    pub async fn insert(&self, entry: JsVectorEntry) -> Result<String> {
        let tx = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut tx = tx.write().expect("RwLock poisoned");
            tx.insert(entry.into())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn commit(&self) -> Result<()> {
        let tx = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let tx = Arc::try_unwrap(tx)
                .map_err(|_| "Transaction still in use")?
                .into_inner()
                .expect("RwLock poisoned");
            tx.commit()
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn rollback(&self) -> Result<()> {
        let tx = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut tx = tx.write().expect("RwLock poisoned");
            tx.rollback()
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[napi]
impl VectorDatabase {
    #[napi]
    pub async fn begin_transaction(&self) -> Result<JsTransaction> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            db.begin_transaction()
                .map(|tx| JsTransaction {
                    inner: Arc::new(RwLock::new(tx)),
                })
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}
```

**Target State (MCP):**
```json
// MCP Tool: transaction_begin
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "transaction_begin",
    "arguments": {
      "database": "my_vectors"
    }
  }
}

// Response
{
  "transaction_id": "tx_abc123",
  "expires_at": "2024-01-08T12:05:00Z"
}

// MCP Tool: transaction_commit
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "transaction_commit",
    "arguments": {
      "transaction_id": "tx_abc123"
    }
  }
}
```

**MCP Handler Implementation:**
```rust
// ruvector-mcp/src/handlers.rs
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct TransactionManager {
    transactions: HashMap<String, ActiveTransaction>,
    timeout: Duration,
}

struct ActiveTransaction {
    tx: Transaction,
    created_at: Instant,
}

impl TransactionManager {
    pub fn begin(&mut self, db: &VectorDatabase) -> Result<String> {
        let id = generate_transaction_id();
        let tx = db.begin_transaction()?;
        self.transactions.insert(id.clone(), ActiveTransaction {
            tx,
            created_at: Instant::now(),
        });

        // Schedule cleanup for expired transactions
        self.cleanup_expired();

        Ok(id)
    }

    pub fn commit(&mut self, id: &str) -> Result<()> {
        let active = self.transactions.remove(id)
            .ok_or(RuvectorError::TransactionNotFound(id.to_string()))?;
        active.tx.commit()
    }

    pub fn rollback(&mut self, id: &str) -> Result<()> {
        let active = self.transactions.remove(id)
            .ok_or(RuvectorError::TransactionNotFound(id.to_string()))?;
        active.tx.rollback()
    }

    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        self.transactions.retain(|_, tx| {
            now.duration_since(tx.created_at) < self.timeout
        });
    }
}
```

**Acceptance Criteria:**
- [ ] Implement `JsTransaction` class in Node.js bindings
- [ ] Add `beginTransaction`, `commit`, `rollback` methods
- [x] Add transaction tools to MCP handlers ✅ `handlers.rs:313-356, 860-1006`
- [x] Implement transaction timeout and cleanup ✅ 5-minute timeout, auto-cleanup
- [ ] Add TypeScript type definitions
- [ ] Document transaction isolation level (MVCC)
- [ ] Add integration tests for transaction semantics

**Files to Modify:**
- `ruvector-node/src/lib.rs`
- `ruvector-node/index.d.ts`
- `ruvector-mcp/src/handlers.rs`
- `ruvector-mcp/src/tools.rs`

---

## Aggregate 2: Async WASM APIs

### A-2: Async WASM APIs [MEDIUM]

**Decision:** Convert synchronous WASM APIs to async using wasm-bindgen-futures.

**Current State:**
```rust
// ruvector-wasm/src/lib.rs
#[wasm_bindgen]
impl WasmVectorDatabase {
    // Synchronous - blocks JavaScript event loop
    #[wasm_bindgen]
    pub fn search(&self, query: &[f32], k: usize) -> JsValue {
        let results = self.inner.search(query, k);
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}
```

**Target State:**
```rust
// ruvector-wasm/src/lib.rs
use wasm_bindgen_futures::future_to_promise;
use js_sys::Promise;

#[wasm_bindgen]
impl WasmVectorDatabase {
    /// Async search returning a Promise
    #[wasm_bindgen(js_name = "search")]
    pub fn search_async(&self, query: &[f32], k: usize) -> Promise {
        let db = self.inner.clone();
        let query = query.to_vec();

        future_to_promise(async move {
            // Yield to JavaScript event loop periodically
            let results = web_sys::window()
                .map(|w| {
                    wasm_bindgen_futures::spawn_local(async {
                        // Actual search
                        db.search(&query, k)
                    })
                })
                .unwrap_or_else(|| db.search(&query, k));

            serde_wasm_bindgen::to_value(&results)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Sync search for backward compatibility
    #[wasm_bindgen(js_name = "searchSync")]
    pub fn search_sync(&self, query: &[f32], k: usize) -> JsValue {
        let results = self.inner.search(query, k);
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}
```

**JavaScript Usage:**
```javascript
// Async API (preferred)
const results = await db.search(queryVector, 10);

// Sync API (backward compatible)
const resultsSync = db.searchSync(queryVector, 10);
```

**Web Worker Integration:**
```typescript
// worker.ts
import init, { WasmVectorDatabase } from 'ruvector-wasm';

let db: WasmVectorDatabase;

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'init':
      await init();
      db = new WasmVectorDatabase(payload.config);
      self.postMessage({ type: 'ready' });
      break;

    case 'search':
      const results = await db.search(payload.query, payload.k);
      self.postMessage({ type: 'results', payload: results });
      break;
  }
};

// main.ts
const worker = new Worker(new URL('./worker.ts', import.meta.url));

async function search(query: number[], k: number): Promise<SearchResult[]> {
  return new Promise((resolve) => {
    worker.onmessage = (e) => {
      if (e.data.type === 'results') {
        resolve(e.data.payload);
      }
    };
    worker.postMessage({ type: 'search', payload: { query, k } });
  });
}
```

**Acceptance Criteria:**
- [ ] Add async versions of all blocking WASM methods
- [ ] Use `wasm-bindgen-futures` for Promise integration
- [ ] Maintain sync variants for backward compatibility
- [ ] Add Web Worker usage examples
- [ ] Document performance characteristics
- [ ] Add browser integration tests

**Files to Modify:**
- `ruvector-wasm/src/lib.rs`
- `ruvector-wasm/src/async_api.rs` (new)
- `ruvector-wasm/examples/` (add examples)

---

## Aggregate 3: Query Language Support

### A-3: Cypher Query Bindings [MEDIUM]

**Decision:** Expose Cypher graph query language to Node.js and MCP.

**Current State (Rust only):**
```rust
// ruvector-graph/src/cypher.rs
impl GraphDatabase {
    pub fn query(&self, cypher: &str) -> Result<QueryResult> {
        let ast = self.parser.parse(cypher)?;
        let plan = self.planner.plan(&ast)?;
        self.executor.execute(plan)
    }
}
```

**Target State (Node.js):**
```typescript
// TypeScript API
interface CypherResult {
  columns: string[];
  rows: Record<string, any>[];
  stats: QueryStats;
}

interface QueryStats {
  nodes_created: number;
  relationships_created: number;
  properties_set: number;
  execution_time_ms: number;
}

class GraphDatabase {
  async query(cypher: string, params?: Record<string, any>): Promise<CypherResult>;

  async queryStream(
    cypher: string,
    params?: Record<string, any>
  ): AsyncIterableIterator<Record<string, any>>;
}

// Usage
const result = await graphDb.query(`
  MATCH (n:Document)-[:SIMILAR_TO]->(m:Document)
  WHERE n.embedding_distance < $threshold
  RETURN n.title, m.title, n.embedding_distance
`, { threshold: 0.5 });

console.log(result.columns);  // ['n.title', 'm.title', 'n.embedding_distance']
console.log(result.rows);     // [{...}, {...}]
```

**Rust Implementation (NAPI-RS):**
```rust
// ruvector-graph-node/src/lib.rs
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct JsCypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<serde_json::Value>,
    pub stats: JsQueryStats,
}

#[napi(object)]
pub struct JsQueryStats {
    pub nodes_created: u32,
    pub relationships_created: u32,
    pub properties_set: u32,
    pub execution_time_ms: f64,
}

#[napi]
impl JsGraphDatabase {
    #[napi]
    pub async fn query(
        &self,
        cypher: String,
        params: Option<serde_json::Value>,
    ) -> Result<JsCypherResult> {
        let db = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let db = db.read().expect("RwLock poisoned");
            let params = params.unwrap_or(serde_json::Value::Object(Default::default()));

            let result = db.query_with_params(&cypher, &params)?;

            Ok(JsCypherResult {
                columns: result.columns,
                rows: result.rows.into_iter()
                    .map(|r| serde_json::to_value(r).unwrap())
                    .collect(),
                stats: JsQueryStats {
                    nodes_created: result.stats.nodes_created,
                    relationships_created: result.stats.relationships_created,
                    properties_set: result.stats.properties_set,
                    execution_time_ms: result.stats.execution_time.as_secs_f64() * 1000.0,
                },
            })
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    #[napi(ts_return_type = "AsyncIterableIterator<any>")]
    pub fn query_stream(
        &self,
        cypher: String,
        params: Option<serde_json::Value>,
    ) -> Result<JsAsyncIterator> {
        // Streaming implementation for large result sets
        todo!()
    }
}
```

**Target State (MCP):**
```json
// MCP Tool: cypher_query
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "cypher_query",
    "arguments": {
      "database": "my_graph",
      "query": "MATCH (n:Document) WHERE n.similarity > $threshold RETURN n LIMIT 10",
      "parameters": {
        "threshold": 0.8
      }
    }
  }
}

// Response
{
  "columns": ["n"],
  "rows": [
    {"n": {"id": "doc1", "title": "...", "embedding": [...]}},
    {"n": {"id": "doc2", "title": "...", "embedding": [...]}}
  ],
  "stats": {
    "nodes_scanned": 1000,
    "execution_time_ms": 45.2
  }
}
```

**Acceptance Criteria:**
- [ ] Implement `query()` method in Node.js bindings
- [ ] Add parameterized query support
- [ ] Add streaming query for large results
- [ ] Add Cypher tool to MCP handlers
- [ ] Document supported Cypher subset
- [ ] Add query validation and error messages
- [ ] Add integration tests

**Files to Modify:**
- `ruvector-graph-node/src/lib.rs`
- `ruvector-graph-node/index.d.ts`
- `ruvector-mcp/src/handlers.rs`
- `ruvector-mcp/src/tools.rs`

---

## Aggregate 4: Collections API

### A-4: Collections WASM Support [LOW]

**Decision:** Implement full Collection API in WASM bindings.

**Current State:**
```rust
// ruvector-wasm/src/lib.rs
// Collections partially implemented, missing:
// - create_collection
// - delete_collection
// - list_collections
// - collection-scoped operations
```

**Target State:**
```rust
#[wasm_bindgen]
impl WasmVectorDatabase {
    #[wasm_bindgen(js_name = "createCollection")]
    pub fn create_collection(&mut self, name: &str, config: JsValue) -> Result<(), JsValue> {
        let config: CollectionConfig = serde_wasm_bindgen::from_value(config)?;
        self.inner.create_collection(name, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "deleteCollection")]
    pub fn delete_collection(&mut self, name: &str) -> Result<(), JsValue> {
        self.inner.delete_collection(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "listCollections")]
    pub fn list_collections(&self) -> Result<JsValue, JsValue> {
        let collections = self.inner.list_collections()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&collections)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "getCollection")]
    pub fn get_collection(&self, name: &str) -> Result<WasmCollection, JsValue> {
        let collection = self.inner.get_collection(name)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmCollection { inner: collection })
    }
}

#[wasm_bindgen]
pub struct WasmCollection {
    inner: Collection,
}

#[wasm_bindgen]
impl WasmCollection {
    #[wasm_bindgen]
    pub fn insert(&mut self, entry: JsValue) -> Result<String, JsValue> {
        let entry: VectorEntry = serde_wasm_bindgen::from_value(entry)?;
        self.inner.insert(entry)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search(&self, query: &[f32], k: usize) -> Result<JsValue, JsValue> {
        let results = self.inner.search(query, k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    #[wasm_bindgen]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}
```

**JavaScript Usage:**
```javascript
import init, { WasmVectorDatabase } from 'ruvector-wasm';

await init();
const db = new WasmVectorDatabase();

// Create collection
db.createCollection('documents', {
  dimensions: 768,
  metric: 'cosine',
  hnsw: { m: 16, ef_construction: 200 }
});

// Use collection
const docs = db.getCollection('documents');
docs.insert({ id: '1', vector: [...], metadata: { title: 'Hello' } });
const results = docs.search(queryVector, 10);

// List all collections
const collections = db.listCollections();
console.log(collections);  // ['documents', 'images', ...]
```

**Acceptance Criteria:**
- [ ] Implement `createCollection`, `deleteCollection`, `listCollections`
- [ ] Add `WasmCollection` wrapper class
- [ ] Add collection-scoped CRUD operations
- [ ] Add TypeScript type definitions
- [ ] Add browser integration tests
- [ ] Document collection configuration options

**Files to Modify:**
- `ruvector-wasm/src/lib.rs`
- `ruvector-wasm/src/collection.rs` (new)

---

## Implementation Plan

### Phase 1: Transactions (Week 1-2)
1. **A-1: Transaction Support** - Node.js and MCP bindings
   - Node.js: 3 days
   - MCP: 2 days
   - Testing: 2 days

### Phase 2: Async WASM (Week 2-3)
2. **A-2: Async WASM** - Promise-based API
   - Implementation: 3 days
   - Web Worker examples: 2 days
   - Testing: 2 days

### Phase 3: Cypher (Week 3-4)
3. **A-3: Cypher Bindings** - Query language support
   - Node.js: 3 days
   - MCP: 2 days
   - Documentation: 2 days

### Phase 4: Collections (Week 4+)
4. **A-4: WASM Collections** - Full collection API
   - Implementation: 2 days
   - Testing: 1 day

---

## Feature Parity Matrix (Target)

| Feature | Rust | Node.js | WASM | CLI | MCP |
|---------|:----:|:-------:|:----:|:---:|:---:|
| Insert | ✓ | ✓ | ✓ | ✓ | ✓ |
| Search | ✓ | ✓ | ✓ | ✓ | ✓ |
| Delete | ✓ | ✓ | ✓ | ✓ | ✓ |
| Batch Operations | ✓ | ✓ | ✓ | ✓ | ✓ |
| Collections | ✓ | ✓ | **✓** | ✓ | ✓ |
| HNSW | ✓ | ✓ | ✓ | ✓ | ✓ |
| Quantization | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Transactions** | ✓ | **✓** | ✓* | ✓* | **✓** |
| **Cypher Queries** | ✓ | **✓** | ✓* | **✓** | **✓** |

**Legend:** ✓ = Existing, **✓** = New in this ADR, ✓* = Planned future work

---

## Consequences

### Positive
- API design score improves from 88/100 to target 92/100
- Consistent developer experience across platforms
- Advanced features accessible from all languages
- Better support for complex use cases (transactions, graphs)

### Negative
- Increased maintenance burden across bindings
- Larger WASM bundle size with new features
- More complex testing matrix

### Mitigation
- Generate bindings from shared schema where possible
- Tree-shake WASM exports to reduce bundle size
- Use matrix CI testing for cross-platform validation

---

## References

- [NAPI-RS Documentation](https://napi.rs/)
- [wasm-bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

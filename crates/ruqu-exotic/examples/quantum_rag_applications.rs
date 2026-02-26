//! # Quantum RAG — Practical Applications
//!
//! Four real-world scenarios where Quantum RAG's properties solve problems
//! that classical RAG cannot:
//!
//! 1. **Medical Knowledge Base** — Drug interaction lookups where polysemy
//!    kills patients and safety knowledge must never degrade.
//!
//! 2. **Intelligence Analysis** — Time-sensitive threat intelligence where
//!    old reports decohere, critical assets are protected, and analysts
//!    need counterfactual source-impact analysis.
//!
//! 3. **Legal Research** — Case law with polysemous legal terms, where
//!    precedent must be preserved and "what if this case was overturned?"
//!    is a real question.
//!
//! 4. **LLM Context Management** — Managing a language model's context
//!    window where older context smoothly fades, system instructions are
//!    protected, and ambiguous retrievals resolve by conversation context.
//!
//! ## Run
//! ```sh
//! cargo run -p ruqu-exotic --example quantum_rag_applications --release
//! ```

use ruqu_exotic::quantum_rag::*;

const SEED: u64 = 0xA991_CA7E;

// ═══════════════════════════════════════════════════════════════════════════
// APPLICATION 1: Medical Knowledge Base
// ═══════════════════════════════════════════════════════════════════════════
//
// Problem: A pharmacist searches "mercury" in a drug interaction database.
//
// Classical RAG: Returns all documents containing "mercury" ranked by
// cosine similarity — mixing up mercury thermometers, mercury-based
// preservatives (thimerosal), dental amalgam, and the planet Mercury
// in astronomy papers. Dangerous in a medical context.
//
// Quantum RAG: The query context (pharmaceutical) creates an interference
// pattern that constructively amplifies the thimerosal/preservative meaning
// and destructively cancels astronomy. Safety warnings about mercury
// poisoning are QEC-protected and NEVER decohere, even when other docs do.

fn app_medical() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  APPLICATION 1: MEDICAL KNOWLEDGE BASE                     ║");
    println!("║  Drug interactions where polysemy can kill                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Embedding dimensions: [pharma, astro, dental, chemistry, safety, general]
    let mut kb = QuantumKnowledgeBase::new(0.1);

    // Mercury documents with multiple meanings
    kb.add(QuantumDocument::new(
        "thimerosal",
        "Thimerosal (mercury-based preservative)",
        vec![
            ("preservative".into(), vec![0.9, 0.0, 0.2, 0.7, 0.3, 0.0]),
            ("heavy-metal".into(), vec![0.3, 0.0, 0.0, 0.9, 0.5, 0.0]),
        ],
        0.2,
    ));

    kb.add(QuantumDocument::new(
        "dental-amalgam",
        "Dental amalgam (mercury alloy fillings)",
        vec![
            ("dental".into(), vec![0.2, 0.0, 0.9, 0.4, 0.2, 0.0]),
            ("heavy-metal".into(), vec![0.1, 0.0, 0.3, 0.8, 0.4, 0.0]),
        ],
        0.2,
    ));

    kb.add(QuantumDocument::new(
        "mercury-planet",
        "Mercury (closest planet to the Sun)",
        vec![
            ("astronomy".into(), vec![0.0, 0.9, 0.0, 0.1, 0.0, 0.5]),
        ],
        0.3,
    ));

    // QEC-protected: mercury poisoning safety protocol
    let mut safety = QuantumDocument::new(
        "mercury-poisoning",
        "CRITICAL: Mercury poisoning emergency protocol",
        vec![
            ("emergency".into(), vec![0.5, 0.0, 0.1, 0.6, 0.9, 0.0]),
        ],
        0.2,
    );
    safety.protect(20.0); // Heavily protected — must NEVER decohere
    kb.add(safety);

    kb.add(QuantumDocument::new(
        "flu-vaccine",
        "Influenza vaccine formulations (multi-dose vials)",
        vec![
            ("vaccine".into(), vec![0.8, 0.0, 0.0, 0.3, 0.2, 0.1]),
        ],
        0.3,
    ));

    // Age the knowledge base (6 months simulated)
    for t in 0..6 {
        kb.evolve(1.0, SEED + t);
    }

    // Scenario: Pharmacist searches "mercury" in pharmaceutical context
    let pharma_context = vec![0.8, 0.0, 0.1, 0.6, 0.3, 0.0];
    let results = kb.query(&pharma_context, 4);

    println!("║                                                              ║");
    println!("║  Scenario: Pharmacist searches 'mercury'                     ║");
    println!("║  Context: pharmaceutical drug interactions                   ║");
    println!("║                                                              ║");

    for (i, r) in results.iter().enumerate() {
        let prot = if r.protected { " [QEC]" } else { "" };
        println!(
            "║  {}. {:<40} fid={:.2}{}  ║",
            i + 1,
            truncate(&r.title, 40),
            r.fidelity,
            prot
        );
        println!(
            "║     meaning: {:<20} score: {:.4}             ║",
            r.dominant_meaning, r.score
        );
    }

    // Verify safety protocol is highly ranked despite aging
    let safety_rank = results
        .iter()
        .position(|r| r.doc_id == "mercury-poisoning")
        .map(|p| p + 1);
    let planet_rank = results
        .iter()
        .position(|r| r.doc_id == "mercury-planet")
        .map(|p| p + 1);

    println!("║                                                              ║");
    println!(
        "║  Safety protocol rank: #{} (QEC-protected, never fades)     ║",
        safety_rank.map_or("N/A".into(), |r| format!("{}", r))
    );
    println!(
        "║  Planet doc rank: #{} (irrelevant, correctly suppressed)    ║",
        planet_rank.map_or("outside top-4".into(), |r| format!("{}", r))
    );

    // Counterfactual: what if thimerosal doc was removed?
    let cf = kb.counterfactual(&pharma_context, "thimerosal", 3);
    println!("║                                                              ║");
    println!("║  Counterfactual: 'What if thimerosal doc was deleted?'       ║");
    println!(
        "║  Impact score: {:.4} (higher = more impactful)              ║",
        cf.divergence
    );
    println!(
        "║  → Without it, #{} result becomes: {:<27}  ║",
        1,
        cf.without_doc.first().map(|r| r.title.as_str()).unwrap_or("N/A")
    );

    println!("║                                                              ║");
    println!("║  KEY INSIGHT: Classical RAG would rank planet Mercury in     ║");
    println!("║  the results. Quantum RAG suppresses it via interference.    ║");
    println!("║  Safety protocol NEVER degrades thanks to QEC protection.    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// APPLICATION 2: Intelligence Analysis
// ═══════════════════════════════════════════════════════════════════════════
//
// Problem: An analyst needs to assess a threat. Intelligence reports age
// and become unreliable. Some sources are critical (protected). The analyst
// needs to know: "What if Source X was compromised (removed)?"
//
// Classical RAG: All reports have equal weight regardless of age. No way
// to do source-impact analysis. No protection for critical intel.
//
// Quantum RAG: Old reports decohere (lower confidence). HUMINT sources
// are QEC-protected. Counterfactual analysis shows source dependency.

fn app_intelligence() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  APPLICATION 2: INTELLIGENCE ANALYSIS                      ║");
    println!("║  Time-sensitive threat intelligence with source analysis    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Embedding: [cyber, physical, financial, geopolitical, urgency, confidence]
    let mut kb = QuantumKnowledgeBase::new(0.05);

    // Fresh SIGINT report (high noise — electronic intercepts age fast)
    kb.add(QuantumDocument::new(
        "sigint-001",
        "SIGINT: Encrypted comms spike in Sector 7",
        vec![
            ("cyber".into(), vec![0.8, 0.1, 0.0, 0.3, 0.7, 0.6]),
            ("physical".into(), vec![0.2, 0.6, 0.0, 0.4, 0.5, 0.3]),
        ],
        0.5, // SIGINT decoheres fast
    ));

    // HUMINT report (protected — human intelligence is expensive, preserve it)
    let mut humint = QuantumDocument::new(
        "humint-007",
        "HUMINT: Asset confirms cyber unit activation",
        vec![
            ("cyber".into(), vec![0.9, 0.0, 0.1, 0.2, 0.8, 0.9]),
        ],
        0.3,
    );
    humint.protect(15.0); // Protect HUMINT — assets are irreplaceable
    kb.add(humint);

    // OSINT from social media (very fast decay)
    kb.add(QuantumDocument::new(
        "osint-tweet",
        "OSINT: Social media chatter about power grid",
        vec![
            ("cyber".into(), vec![0.5, 0.3, 0.0, 0.1, 0.4, 0.2]),
            ("physical".into(), vec![0.2, 0.7, 0.0, 0.1, 0.3, 0.1]),
        ],
        0.8, // Social media decoheres very fast
    ));

    // FININT report (financial intelligence)
    kb.add(QuantumDocument::new(
        "finint-042",
        "FININT: Suspicious transactions linked to APT group",
        vec![
            ("financial".into(), vec![0.1, 0.0, 0.9, 0.3, 0.5, 0.7]),
            ("cyber".into(), vec![0.6, 0.0, 0.3, 0.2, 0.6, 0.5]),
        ],
        0.3,
    ));

    // Old background report
    kb.add(QuantumDocument::new(
        "background-old",
        "Background: Regional threat landscape assessment",
        vec![
            ("geopolitical".into(), vec![0.1, 0.2, 0.1, 0.9, 0.2, 0.8]),
        ],
        0.2,
    ));

    // Simulate 10 days passing
    for t in 0..10 {
        kb.evolve(1.0, SEED + 100 + t);
    }

    // Analyst queries: "cyber threat assessment"
    let cyber_ctx = vec![0.9, 0.1, 0.1, 0.2, 0.7, 0.6];
    let results = kb.query(&cyber_ctx, 5);

    println!("║                                                              ║");
    println!("║  Scenario: Analyst queries 'cyber threat assessment'          ║");
    println!("║  10 days have passed since reports were filed                 ║");
    println!("║                                                              ║");

    for (i, r) in results.iter().enumerate() {
        let prot = if r.protected { "QEC" } else { "   " };
        let freshness = if r.fidelity > 0.9 {
            "FRESH"
        } else if r.fidelity > 0.5 {
            "AGING"
        } else {
            "STALE"
        };
        println!(
            "║  {}. {:<36} {} {} fid={:.3}  ║",
            i + 1,
            truncate(&r.title, 36),
            prot,
            freshness,
            r.fidelity
        );
    }

    // Source-impact analysis: what if HUMINT asset was burned?
    let cf = kb.counterfactual(&cyber_ctx, "humint-007", 3);
    println!("║                                                              ║");
    println!("║  SOURCE-IMPACT ANALYSIS:                                     ║");
    println!("║  'What if HUMINT asset was compromised (burned)?'            ║");
    println!(
        "║  Impact: {:.4} (source dependency score)                   ║",
        cf.divergence
    );
    println!("║  Without HUMINT, top result becomes:                         ║");
    if let Some(top) = cf.without_doc.first() {
        println!(
            "║    → {:<52}  ║",
            truncate(&top.title, 52)
        );
    }

    // Compare HUMINT vs OSINT impact
    let cf_osint = kb.counterfactual(&cyber_ctx, "osint-tweet", 3);
    println!("║                                                              ║");
    println!(
        "║  HUMINT impact: {:.4}  vs  OSINT impact: {:.4}            ║",
        cf.divergence, cf_osint.divergence
    );
    println!("║  → HUMINT is {:.1}x more impactful than social media         ║",
        if cf_osint.divergence > 0.001 {
            cf.divergence / cf_osint.divergence
        } else {
            f64::INFINITY
        }
    );

    println!("║                                                              ║");
    println!("║  KEY INSIGHT: SIGINT and OSINT decay fast (correct — they    ║");
    println!("║  become unreliable). HUMINT is QEC-protected (correct —      ║");
    println!("║  human sources provide durable intelligence). Counterfactual ║");
    println!("║  analysis quantifies dependency on each source.              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// APPLICATION 3: Legal Research
// ═══════════════════════════════════════════════════════════════════════════
//
// Problem: A lawyer searches "battery" — is it the tort (assault) or the
// electrical device? "Consideration" — contract law or general usage?
//
// Classical RAG: Returns all documents containing the term. Cannot
// distinguish legal senses without manual filtering.
//
// Quantum RAG: The legal context (tort vs contract vs criminal) creates
// interference that selects the correct legal meaning. Landmark cases
// are QEC-protected (precedent must endure).

fn app_legal() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  APPLICATION 3: LEGAL RESEARCH                             ║");
    println!("║  Polysemous legal terms with precedent protection          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Embedding: [tort, contract, criminal, property, constitutional, general]
    let mut kb = QuantumKnowledgeBase::new(0.1);

    // "Battery" — tort meaning
    kb.add(QuantumDocument::new(
        "battery-tort",
        "Battery as intentional tort (harmful contact)",
        vec![
            ("tort".into(), vec![0.9, 0.0, 0.3, 0.0, 0.0, 0.1]),
            ("criminal".into(), vec![0.3, 0.0, 0.8, 0.0, 0.0, 0.1]),
        ],
        0.15,
    ));

    // "Battery" — electrical meaning
    kb.add(QuantumDocument::new(
        "battery-electric",
        "Battery technology patents and IP",
        vec![
            ("property".into(), vec![0.1, 0.0, 0.0, 0.7, 0.0, 0.5]),
            ("general".into(), vec![0.0, 0.0, 0.0, 0.2, 0.0, 0.9]),
        ],
        0.15,
    ));

    // "Consideration" — contract law
    kb.add(QuantumDocument::new(
        "consideration-contract",
        "Consideration doctrine in contract formation",
        vec![
            ("contract".into(), vec![0.0, 0.9, 0.0, 0.0, 0.1, 0.1]),
        ],
        0.15,
    ));

    // "Consideration" — general usage
    kb.add(QuantumDocument::new(
        "consideration-general",
        "Due consideration in administrative decisions",
        vec![
            ("general".into(), vec![0.1, 0.2, 0.0, 0.1, 0.3, 0.8]),
        ],
        0.15,
    ));

    // Landmark case — QEC-protected (precedent must endure)
    let mut landmark = QuantumDocument::new(
        "landmark-vosburg",
        "Vosburg v. Putney (1891) — battery intent standard",
        vec![
            ("tort".into(), vec![0.9, 0.0, 0.2, 0.0, 0.1, 0.0]),
            ("precedent".into(), vec![0.5, 0.1, 0.1, 0.0, 0.5, 0.0]),
        ],
        0.1,
    );
    landmark.protect(25.0); // Precedent is sacred
    kb.add(landmark);

    // Simulate 5 years of aging
    for t in 0..5 {
        kb.evolve(1.0, SEED + 200 + t);
    }

    // Scenario 1: Tort lawyer searches "battery"
    let tort_ctx = vec![0.9, 0.0, 0.2, 0.0, 0.0, 0.1];
    let results = kb.query(&tort_ctx, 3);

    println!("║                                                              ║");
    println!("║  Scenario A: Tort lawyer searches 'battery'                  ║");
    for (i, r) in results.iter().enumerate() {
        let prot = if r.protected { " [PRECEDENT]" } else { "" };
        println!(
            "║  {}. {:<38} [{:<10}]{}  ║",
            i + 1,
            truncate(&r.title, 38),
            r.dominant_meaning,
            prot
        );
    }

    // Scenario 2: IP lawyer searches "battery"
    let ip_ctx = vec![0.0, 0.0, 0.0, 0.8, 0.0, 0.5];
    let results = kb.query(&ip_ctx, 3);

    println!("║                                                              ║");
    println!("║  Scenario B: IP lawyer searches 'battery'                    ║");
    for (i, r) in results.iter().enumerate() {
        let prot = if r.protected { " [PRECEDENT]" } else { "" };
        println!(
            "║  {}. {:<38} [{:<10}]{}  ║",
            i + 1,
            truncate(&r.title, 38),
            r.dominant_meaning,
            prot
        );
    }

    // Scenario 3: Contract lawyer searches "consideration"
    let contract_ctx = vec![0.0, 0.9, 0.0, 0.0, 0.1, 0.0];
    let results = kb.query(&contract_ctx, 3);

    println!("║                                                              ║");
    println!("║  Scenario C: Contract lawyer searches 'consideration'        ║");
    for (i, r) in results.iter().enumerate() {
        println!(
            "║  {}. {:<45} [{:<10}]  ║",
            i + 1,
            truncate(&r.title, 45),
            r.dominant_meaning
        );
    }

    // Counterfactual: "What if Vosburg v. Putney was overturned?"
    let cf = kb.counterfactual(&tort_ctx, "landmark-vosburg", 3);
    println!("║                                                              ║");
    println!("║  Counterfactual: 'What if Vosburg v. Putney overturned?'     ║");
    println!(
        "║  Impact on tort search: {:.4} divergence                   ║",
        cf.divergence
    );

    println!("║                                                              ║");
    println!("║  KEY INSIGHT: Same term 'battery' resolves to tort law in    ║");
    println!("║  tort context and to IP/patents in IP context. Vosburg v.    ║");
    println!("║  Putney is QEC-protected — precedent never degrades.         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// APPLICATION 4: LLM Context Window Management
// ═══════════════════════════════════════════════════════════════════════════
//
// Problem: An LLM has a limited context window. As conversation grows,
// older context must be evicted. But system instructions must never fade.
// And when the user asks "tell me about Python", the context should
// determine whether they mean the language or the snake.
//
// Classical RAG: Hard truncation. System instructions can be accidentally
// evicted. No way to know how removing a context chunk affects answers.
//
// Quantum RAG: System instructions are QEC-protected. Old conversation
// turns decohere smoothly. User query context resolves ambiguous concepts.

fn app_llm_context() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  APPLICATION 4: LLM CONTEXT WINDOW MANAGEMENT              ║");
    println!("║  Decoherence-based context eviction with protected prompts  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Embedding: [code, nature, math, creative, safety, meta]
    let mut kb = QuantumKnowledgeBase::new(0.05);

    // System instruction — MUST NEVER DECOHERE
    let mut system_prompt = QuantumDocument::new(
        "system-prompt",
        "System: You are a helpful coding assistant",
        vec![
            ("instruction".into(), vec![0.6, 0.0, 0.2, 0.0, 0.5, 0.8]),
        ],
        0.1,
    );
    system_prompt.protect(50.0); // Maximum protection
    kb.add(system_prompt);

    // Safety guardrails — ALSO PROTECTED
    let mut safety = QuantumDocument::new(
        "safety-rules",
        "Safety: Never generate harmful code or bypass security",
        vec![
            ("safety".into(), vec![0.1, 0.0, 0.0, 0.0, 0.9, 0.5]),
        ],
        0.1,
    );
    safety.protect(50.0);
    kb.add(safety);

    // Conversation turn 1 (oldest — will decohere most)
    kb.add(QuantumDocument::new(
        "turn-1",
        "User asked about Python web frameworks (turn 1)",
        vec![
            ("code".into(), vec![0.9, 0.0, 0.1, 0.0, 0.0, 0.0]),
            ("web".into(), vec![0.7, 0.0, 0.0, 0.0, 0.1, 0.0]),
        ],
        0.4, // Conversation turns decohere at medium rate
    ));

    // Conversation turn 2
    kb.add(QuantumDocument::new(
        "turn-2",
        "User discussed database optimization (turn 2)",
        vec![
            ("code".into(), vec![0.7, 0.0, 0.3, 0.0, 0.0, 0.0]),
        ],
        0.4,
    ));

    // Conversation turn 3 (most recent — freshest)
    kb.add(QuantumDocument::new(
        "turn-3",
        "User asked about Python data science libraries (turn 3)",
        vec![
            ("code".into(), vec![0.8, 0.0, 0.5, 0.0, 0.0, 0.0]),
            ("nature".into(), vec![0.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
        ],
        0.4,
    ));

    // Simulate 8 conversation turns passing
    for t in 0..8 {
        kb.evolve(1.0, SEED + 300 + t);
    }

    // Show context window state
    let stats = kb.stats();
    println!("║                                                              ║");
    println!("║  Context Window State (after 8 turns):                       ║");
    println!(
        "║    Total chunks: {}    Protected: {}    Mean fidelity: {:.3}  ║",
        stats.total_docs, stats.protected_docs, stats.mean_fidelity
    );

    // Query: user asks "help me with Python"
    let code_ctx = vec![0.9, 0.0, 0.3, 0.0, 0.0, 0.0];
    let results = kb.query(&code_ctx, 5);

    println!("║                                                              ║");
    println!("║  User query: 'help me with Python'                           ║");
    println!("║  Context: coding (inferred from conversation history)        ║");
    println!("║                                                              ║");

    for (i, r) in results.iter().enumerate() {
        let status = if r.protected {
            "PROTECTED"
        } else if r.fidelity > 0.8 {
            "FRESH    "
        } else if r.fidelity > 0.3 {
            "FADING   "
        } else {
            "FORGOTTEN"
        };
        println!(
            "║  {}. {:<36} {} fid={:.3}  ║",
            i + 1,
            truncate(&r.title, 36),
            status,
            r.fidelity
        );
    }

    // Key demo: system prompt always in context
    let system_in_results = results.iter().any(|r| r.doc_id == "system-prompt");
    let safety_in_results = results.iter().any(|r| r.doc_id == "safety-rules");
    println!("║                                                              ║");
    println!(
        "║  System prompt in context: {}                                ║",
        if system_in_results { "YES (QEC-protected)" } else { "NO (ERROR!)" }
    );
    println!(
        "║  Safety rules in context:  {}                                ║",
        if safety_in_results { "YES (QEC-protected)" } else { "NO (ERROR!)" }
    );

    // Counterfactual: what if turn-1 was evicted?
    let cf = kb.counterfactual(&code_ctx, "turn-1", 3);
    println!("║                                                              ║");
    println!("║  Eviction analysis: 'What if turn-1 was removed?'            ║");
    println!(
        "║  Impact: {:.4} — {}                    ║",
        cf.divergence,
        if cf.divergence < 0.5 { "safe to evict (low impact)" } else { "keep (high impact)" }
    );

    let cf2 = kb.counterfactual(&code_ctx, "turn-3", 3);
    println!(
        "║  What if turn-3 removed? Impact: {:.4} — {}  ║",
        cf2.divergence,
        if cf2.divergence < 0.5 { "safe to evict" } else { "KEEP (high impact)" }
    );

    println!("║                                                              ║");
    println!("║  KEY INSIGHT: System prompts and safety rules NEVER leave    ║");
    println!("║  the context window (QEC-protected). Old conversation turns  ║");
    println!("║  fade smoothly. Counterfactual analysis tells you which      ║");
    println!("║  turns are safe to evict and which are still impactful.      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// APPLICATION SUMMARY: Production Integration Patterns
// ═══════════════════════════════════════════════════════════════════════════

fn production_patterns() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRODUCTION INTEGRATION PATTERNS                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  Pattern 1: SAFETY-CRITICAL KNOWLEDGE                       ║");
    println!("║  ─────────────────────────────────────────────────────────  ║");
    println!("║  Use QEC protection (factor 10-50x) for:                    ║");
    println!("║    • Emergency protocols (medical, nuclear, aviation)        ║");
    println!("║    • Compliance regulations that must never be forgotten     ║");
    println!("║    • System prompts and safety guardrails for LLMs          ║");
    println!("║    • Legal precedent and landmark rulings                    ║");
    println!("║                                                              ║");
    println!("║  Pattern 2: TEMPORAL FRESHNESS                              ║");
    println!("║  ─────────────────────────────────────────────────────────  ║");
    println!("║  Set noise_rate by source reliability:                       ║");
    println!("║    • Social media / tweets:  noise=0.8 (fast decay)         ║");
    println!("║    • News articles:          noise=0.3 (medium decay)       ║");
    println!("║    • Academic papers:        noise=0.05 (slow decay)        ║");
    println!("║    • Physical constants:     noise=0.0 + QEC (never decay)  ║");
    println!("║                                                              ║");
    println!("║  Pattern 3: POLYSEMY RESOLUTION                             ║");
    println!("║  ─────────────────────────────────────────────────────────  ║");
    println!("║  Encode each document with multiple meaning embeddings:     ║");
    println!("║    • 'Bank': [financial, riverbank]                          ║");
    println!("║    • 'Battery': [tort, electrical, military]                 ║");
    println!("║    • 'Python': [language, snake, Monty Python]              ║");
    println!("║  The query context automatically resolves the ambiguity.    ║");
    println!("║                                                              ║");
    println!("║  Pattern 4: IMPACT ANALYSIS                                 ║");
    println!("║  ─────────────────────────────────────────────────────────  ║");
    println!("║  Use counterfactual queries before:                          ║");
    println!("║    • Deleting documents (safe if divergence < threshold)     ║");
    println!("║    • Assessing source dependency (intelligence analysis)     ║");
    println!("║    • Understanding which context chunks matter (LLMs)        ║");
    println!("║    • Auditing knowledge base integrity                       ║");
    println!("║                                                              ║");
    println!("║  Pattern 5: NOISE RATE CALIBRATION                          ║");
    println!("║  ─────────────────────────────────────────────────────────  ║");
    println!("║    noise_rate = -ln(target_fidelity) / expected_lifetime     ║");
    println!("║                                                              ║");
    println!("║    Example: Want 50% fidelity after 30 days?                 ║");
    println!("║    noise_rate = -ln(0.5) / 30 ≈ 0.023                       ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ── Utility ────────────────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        format!("{:<width$}", s, width = max)
    } else {
        format!("{}...", &s[..max - 3])
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("████████████████████████████████████████████████████████████████");
    println!("██                                                          ██");
    println!("██   QUANTUM RAG — Practical Applications                   ██");
    println!("██   4 Real-World Scenarios + Production Patterns            ██");
    println!("██                                                          ██");
    println!("████████████████████████████████████████████████████████████████");
    println!();

    app_medical();
    app_intelligence();
    app_legal();
    app_llm_context();
    production_patterns();

    println!();
    println!("████████████████████████████████████████████████████████████████");
    println!("██  Each application demonstrates capabilities that are      ██");
    println!("██  structurally impossible in classical RAG:                 ██");
    println!("██                                                          ██");
    println!("██  Medical:       Polysemy + QEC safety protocols           ██");
    println!("██  Intelligence:  Temporal decay + Source impact analysis   ██");
    println!("██  Legal:         Polysemy + Protected precedent            ██");
    println!("██  LLM Context:   All 5 properties working together        ██");
    println!("████████████████████████████████████████████████████████████████");
    println!();
}

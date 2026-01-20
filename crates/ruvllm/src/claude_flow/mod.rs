//! Claude Flow Integration for RuvLTRA
//!
//! Optimizes RuvLTRA-Small for Claude Flow use cases:
//! - Agent routing (task → optimal agent type)
//! - Task classification (code/research/test/review)
//! - Semantic search (memory retrieval queries)
//! - Code generation (Rust/TypeScript output)

mod agent_router;
mod task_classifier;
mod flow_optimizer;

pub use agent_router::{AgentRouter, AgentType, RoutingDecision};
pub use task_classifier::{TaskClassifier, TaskType, ClassificationResult};
pub use flow_optimizer::{FlowOptimizer, OptimizationConfig, OptimizationResult};

/// Claude Flow agent types supported by RuvLTRA routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClaudeFlowAgent {
    Coder,
    Researcher,
    Tester,
    Reviewer,
    Architect,
    SecurityAuditor,
    PerformanceEngineer,
    MlDeveloper,
    BackendDev,
    CicdEngineer,
}

impl ClaudeFlowAgent {
    /// Get all agent types
    pub fn all() -> &'static [ClaudeFlowAgent] {
        &[
            Self::Coder,
            Self::Researcher,
            Self::Tester,
            Self::Reviewer,
            Self::Architect,
            Self::SecurityAuditor,
            Self::PerformanceEngineer,
            Self::MlDeveloper,
            Self::BackendDev,
            Self::CicdEngineer,
        ]
    }

    /// Get agent name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Coder => "coder",
            Self::Researcher => "researcher",
            Self::Tester => "tester",
            Self::Reviewer => "reviewer",
            Self::Architect => "system-architect",
            Self::SecurityAuditor => "security-auditor",
            Self::PerformanceEngineer => "performance-engineer",
            Self::MlDeveloper => "ml-developer",
            Self::BackendDev => "backend-dev",
            Self::CicdEngineer => "cicd-engineer",
        }
    }

    /// Get typical task keywords for this agent
    pub fn keywords(&self) -> &'static [&'static str] {
        match self {
            Self::Coder => &["implement", "code", "write", "create", "build", "develop", "function", "class"],
            Self::Researcher => &["research", "analyze", "investigate", "explore", "find", "search", "understand"],
            Self::Tester => &["test", "verify", "validate", "check", "assert", "coverage", "unit", "integration"],
            Self::Reviewer => &["review", "audit", "inspect", "quality", "lint", "style", "best practice"],
            Self::Architect => &["design", "architecture", "structure", "pattern", "system", "scalable", "modular"],
            Self::SecurityAuditor => &["security", "vulnerability", "cve", "injection", "auth", "encrypt", "safe"],
            Self::PerformanceEngineer => &["performance", "optimize", "speed", "memory", "benchmark", "profile", "latency"],
            Self::MlDeveloper => &["model", "train", "neural", "ml", "ai", "embedding", "inference", "tensor"],
            Self::BackendDev => &["api", "endpoint", "database", "server", "rest", "graphql", "query"],
            Self::CicdEngineer => &["ci", "cd", "pipeline", "deploy", "workflow", "action", "build", "release"],
        }
    }
}

/// Claude Flow task types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeFlowTask {
    CodeGeneration,
    CodeReview,
    Testing,
    Research,
    Documentation,
    Debugging,
    Refactoring,
    Security,
    Performance,
    Architecture,
}

impl ClaudeFlowTask {
    /// Get recommended agents for this task type
    pub fn recommended_agents(&self) -> &'static [ClaudeFlowAgent] {
        match self {
            Self::CodeGeneration => &[ClaudeFlowAgent::Coder, ClaudeFlowAgent::BackendDev],
            Self::CodeReview => &[ClaudeFlowAgent::Reviewer, ClaudeFlowAgent::SecurityAuditor],
            Self::Testing => &[ClaudeFlowAgent::Tester, ClaudeFlowAgent::Coder],
            Self::Research => &[ClaudeFlowAgent::Researcher, ClaudeFlowAgent::Architect],
            Self::Documentation => &[ClaudeFlowAgent::Researcher, ClaudeFlowAgent::Coder],
            Self::Debugging => &[ClaudeFlowAgent::Coder, ClaudeFlowAgent::Tester],
            Self::Refactoring => &[ClaudeFlowAgent::Coder, ClaudeFlowAgent::Architect],
            Self::Security => &[ClaudeFlowAgent::SecurityAuditor, ClaudeFlowAgent::Reviewer],
            Self::Performance => &[ClaudeFlowAgent::PerformanceEngineer, ClaudeFlowAgent::Coder],
            Self::Architecture => &[ClaudeFlowAgent::Architect, ClaudeFlowAgent::Reviewer],
        }
    }
}

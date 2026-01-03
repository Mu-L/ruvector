/**
 * @ruvector/edge-net ONNX Worker Module
 *
 * Real semantic embeddings and LLM inference for workers
 * Uses @xenova/transformers for actual AI inference
 *
 * @module @ruvector/edge-net/onnx-worker
 */

import { EventEmitter } from 'events';
import { randomBytes } from 'crypto';

// ============================================
// ONNX EMBEDDER (REAL SEMANTIC EMBEDDINGS)
// ============================================

let transformers = null;
let embeddingPipeline = null;
let textGenPipeline = null;
let loadedEmbedModel = null;
let loadedGenModel = null;

/**
 * Available embedding models (smallest first)
 * Optimized for edge computing with size/quality tradeoffs
 */
export const EMBEDDING_MODELS = {
    // Tier 1: Ultra-fast (~20-30MB)
    'minilm-l6': {
        id: 'Xenova/all-MiniLM-L6-v2',
        dimensions: 384,
        size: '~22MB',
        description: 'Fast, good quality embeddings',
        tier: 1,
    },
    'e5-small': {
        id: 'Xenova/e5-small-v2',
        dimensions: 384,
        size: '~28MB',
        description: 'Microsoft E5 - excellent retrieval',
        tier: 1,
    },
    // Tier 2: Balanced (~30-70MB)
    'minilm-l12': {
        id: 'Xenova/all-MiniLM-L12-v2',
        dimensions: 384,
        size: '~33MB',
        description: 'Better quality, slightly slower',
        tier: 2,
    },
    'bge-small': {
        id: 'Xenova/bge-small-en-v1.5',
        dimensions: 384,
        size: '~33MB',
        description: 'Best for retrieval tasks',
        tier: 2,
    },
    'gte-small': {
        id: 'Xenova/gte-small',
        dimensions: 384,
        size: '~67MB',
        description: 'High quality embeddings',
        tier: 2,
    },
    // Tier 3: High quality (~100MB+)
    'gte-base': {
        id: 'Xenova/gte-base',
        dimensions: 768,
        size: '~100MB',
        description: 'Higher quality, 768d',
        tier: 3,
    },
    'bge-base': {
        id: 'Xenova/bge-base-en-v1.5',
        dimensions: 768,
        size: '~108MB',
        description: 'High quality BAAI retrieval',
        tier: 3,
    },
    // Specialized: Multilingual
    'multilingual-e5': {
        id: 'Xenova/multilingual-e5-small',
        dimensions: 384,
        size: '~118MB',
        description: '100+ languages support',
        tier: 3,
    },
};

/**
 * Available text generation models
 * Organized by size and capability for edge deployment
 */
export const GENERATION_MODELS = {
    // Tier 1: Ultra-small (< 100MB) - Fast inference
    'tinystories': {
        id: 'Xenova/TinyStories-33M',
        size: '~65MB',
        description: 'Ultra-small for stories',
        tier: 1,
        capabilities: ['stories', 'creative'],
    },
    'distilgpt2': {
        id: 'Xenova/distilgpt2',
        size: '~82MB',
        description: 'Fast text generation',
        tier: 1,
        capabilities: ['general', 'completion'],
    },
    // Tier 2: Small (100-300MB) - Good quality
    'gpt2': {
        id: 'Xenova/gpt2',
        size: '~250MB',
        description: 'Classic GPT-2',
        tier: 2,
        capabilities: ['general', 'completion', 'creative'],
    },
    'phi-1.5': {
        id: 'Xenova/phi-1_5',
        size: '~280MB',
        description: 'Microsoft Phi-1.5 - code & reasoning',
        tier: 2,
        capabilities: ['code', 'reasoning', 'math'],
    },
    'phi-2': {
        id: 'Xenova/phi-2',
        size: '~550MB',
        description: 'Microsoft Phi-2 - advanced reasoning',
        tier: 3,
        capabilities: ['code', 'reasoning', 'math', 'general'],
    },
    // Tier 3: Medium (300MB-1GB) - High quality
    'qwen-0.5b': {
        id: 'Xenova/Qwen1.5-0.5B',
        size: '~430MB',
        description: 'Qwen 0.5B - multilingual small model',
        tier: 3,
        capabilities: ['multilingual', 'general', 'code'],
    },
    'gemma-2b': {
        id: 'Xenova/gemma-2b-it',
        size: '~1.1GB',
        description: 'Google Gemma 2B instruction-tuned',
        tier: 4,
        capabilities: ['instruction', 'general', 'code', 'reasoning'],
    },
    // Code-specialized models
    'codegen-350m': {
        id: 'Xenova/codegen-350M-mono',
        size: '~320MB',
        description: 'Salesforce CodeGen - Python specialist',
        tier: 2,
        capabilities: ['code', 'python'],
    },
    'starcoder-tiny': {
        id: 'Xenova/tiny_starcoder_py',
        size: '~40MB',
        description: 'Ultra-small Python code model',
        tier: 1,
        capabilities: ['code', 'python'],
    },
};

/**
 * Recommended models by use case
 */
export const MODEL_RECOMMENDATIONS = {
    'edge-minimal': {
        embedding: 'minilm-l6',
        generation: 'distilgpt2',
        description: 'Minimal footprint for constrained devices',
    },
    'edge-balanced': {
        embedding: 'e5-small',
        generation: 'phi-1.5',
        description: 'Best quality/size ratio for edge',
    },
    'edge-code': {
        embedding: 'bge-small',
        generation: 'starcoder-tiny',
        description: 'Optimized for code tasks',
    },
    'edge-full': {
        embedding: 'gte-base',
        generation: 'phi-2',
        description: 'Maximum quality on edge',
    },
    'cloud-optimal': {
        embedding: 'bge-base',
        generation: 'gemma-2b',
        description: 'Best quality for cloud deployment',
    },
};

/**
 * Initialize transformers.js
 */
async function initTransformers() {
    if (transformers) return transformers;

    try {
        transformers = await import('@xenova/transformers');

        // Configure cache
        const { env } = transformers;
        env.cacheDir = process.env.ONNX_CACHE_DIR ||
            (process.env.HOME ? `${process.env.HOME}/.ruvector/models/onnx` : '/tmp/.ruvector/models/onnx');
        env.allowRemoteModels = true;
        env.allowLocalModels = true;

        return transformers;
    } catch (error) {
        console.error('[ONNX Worker] transformers.js not available:', error.message);
        return null;
    }
}

/**
 * Initialize embedding model
 */
export async function initEmbedding(modelKey = 'minilm-l6') {
    const tf = await initTransformers();
    if (!tf) return false;

    const model = EMBEDDING_MODELS[modelKey] || EMBEDDING_MODELS['minilm-l6'];

    if (embeddingPipeline && loadedEmbedModel === model.id) {
        return true;
    }

    try {
        console.error(`[ONNX] Loading embedding model: ${model.id}...`);
        const { pipeline } = tf;
        embeddingPipeline = await pipeline('feature-extraction', model.id, {
            quantized: true,
        });
        loadedEmbedModel = model.id;
        console.error(`[ONNX] Embedding model ready: ${model.id}`);
        return true;
    } catch (error) {
        console.error('[ONNX] Failed to load embedding model:', error.message);
        return false;
    }
}

/**
 * Initialize text generation model
 */
export async function initGeneration(modelKey = 'distilgpt2') {
    const tf = await initTransformers();
    if (!tf) return false;

    const model = GENERATION_MODELS[modelKey] || GENERATION_MODELS['distilgpt2'];

    if (textGenPipeline && loadedGenModel === model.id) {
        return true;
    }

    try {
        console.error(`[ONNX] Loading generation model: ${model.id}...`);
        const { pipeline } = tf;
        textGenPipeline = await pipeline('text-generation', model.id, {
            quantized: true,
        });
        loadedGenModel = model.id;
        console.error(`[ONNX] Generation model ready: ${model.id}`);
        return true;
    } catch (error) {
        console.error('[ONNX] Failed to load generation model:', error.message);
        return false;
    }
}

/**
 * Generate real semantic embeddings
 */
export async function embed(texts, options = {}) {
    const initialized = await initEmbedding(options.model);

    if (!initialized || !embeddingPipeline) {
        // Fallback to hash-based embeddings
        return fallbackEmbed(texts);
    }

    const inputTexts = Array.isArray(texts) ? texts : [texts];
    const startTime = performance.now();

    try {
        const results = [];

        for (const text of inputTexts) {
            const output = await embeddingPipeline(text, {
                pooling: 'mean',
                normalize: true,
            });

            // Convert tensor to array
            const embedding = Array.from(output.data);

            results.push({
                text: text.slice(0, 100),
                embedding,
                dimensions: embedding.length,
                semantic: true,
            });
        }

        const timeMs = performance.now() - startTime;

        return {
            embeddings: results,
            model: loadedEmbedModel,
            timeMs,
            count: results.length,
            semantic: true,
        };
    } catch (error) {
        console.error('[ONNX] Embedding error:', error.message);
        return fallbackEmbed(texts);
    }
}

/**
 * Fallback hash-based embeddings
 */
function fallbackEmbed(texts) {
    const inputTexts = Array.isArray(texts) ? texts : [texts];

    const results = inputTexts.map(text => {
        const hash = createHash('sha256').update(String(text)).digest();
        const embedding = new Float32Array(384);
        for (let i = 0; i < 384; i++) {
            embedding[i] = (hash[i % 32] - 128) / 128;
        }
        return {
            text: String(text).slice(0, 100),
            embedding: Array.from(embedding),
            dimensions: 384,
            semantic: false,
        };
    });

    return {
        embeddings: results,
        model: 'hash-fallback',
        count: results.length,
        semantic: false,
    };
}

/**
 * Generate text using ONNX LLM
 */
export async function generate(prompt, options = {}) {
    const initialized = await initGeneration(options.model);

    if (!initialized || !textGenPipeline) {
        return {
            text: `[Fallback] Processing: ${prompt.slice(0, 50)}...`,
            model: 'fallback',
            semantic: false,
        };
    }

    const startTime = performance.now();

    try {
        const outputs = await textGenPipeline(prompt, {
            max_new_tokens: options.maxTokens || 64,
            temperature: options.temperature || 0.7,
            top_p: options.topP || 0.9,
            do_sample: true,
            return_full_text: false,
        });

        const timeMs = performance.now() - startTime;
        const generatedText = outputs[0]?.generated_text || '';

        return {
            text: generatedText.trim(),
            model: loadedGenModel,
            timeMs,
            tokensPerSecond: Math.round((generatedText.split(/\s+/).length * 1.3) / (timeMs / 1000)),
            semantic: true,
        };
    } catch (error) {
        console.error('[ONNX] Generation error:', error.message);
        return {
            text: `[Error] ${error.message}`,
            model: 'error',
            semantic: false,
        };
    }
}

/**
 * Compute similarity between two texts
 */
export async function similarity(text1, text2, options = {}) {
    const result = await embed([text1, text2], options);

    if (result.embeddings.length < 2) {
        return { similarity: 0, semantic: false };
    }

    const e1 = result.embeddings[0].embedding;
    const e2 = result.embeddings[1].embedding;

    // Cosine similarity
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < e1.length; i++) {
        dotProduct += e1[i] * e2[i];
        norm1 += e1[i] * e1[i];
        norm2 += e2[i] * e2[i];
    }

    const cosineSim = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

    return {
        similarity: cosineSim,
        text1: text1.slice(0, 50),
        text2: text2.slice(0, 50),
        model: result.model,
        semantic: result.semantic,
    };
}

/**
 * Semantic search - find most similar texts
 */
export async function semanticSearch(query, documents, options = {}) {
    const topK = options.topK || 5;

    // Embed query and documents together
    const allTexts = [query, ...documents];
    const result = await embed(allTexts, options);

    if (result.embeddings.length < 2) {
        return { results: [], semantic: false };
    }

    const queryEmbed = result.embeddings[0].embedding;
    const docEmbeds = result.embeddings.slice(1);

    // Calculate similarities
    const scores = docEmbeds.map((doc, index) => {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < queryEmbed.length; i++) {
            dotProduct += queryEmbed[i] * doc.embedding[i];
            norm1 += queryEmbed[i] * queryEmbed[i];
            norm2 += doc.embedding[i] * doc.embedding[i];
        }

        return {
            index,
            text: documents[index],
            score: dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)),
        };
    });

    // Sort by score and return top K
    scores.sort((a, b) => b.score - a.score);

    return {
        query,
        results: scores.slice(0, topK),
        model: result.model,
        semantic: result.semantic,
    };
}

// ============================================
// ONNX WORKER POOL
// ============================================

/**
 * Enhanced worker pool with ONNX capabilities
 */
export class OnnxWorkerPool extends EventEmitter {
    constructor(options = {}) {
        super();
        this.id = `onnx-pool-${randomBytes(6).toString('hex')}`;
        this.embedModel = options.embedModel || 'minilm-l6';
        this.genModel = options.genModel || 'distilgpt2';
        this.initialized = false;

        this.stats = {
            embeddings: 0,
            generations: 0,
            searches: 0,
            totalTimeMs: 0,
        };
    }

    /**
     * Initialize ONNX models
     */
    async initialize() {
        this.emit('status', 'Initializing ONNX models...');

        // Initialize embedding model
        const embedReady = await initEmbedding(this.embedModel);

        // Initialize generation model (optional)
        const genReady = await initGeneration(this.genModel);

        this.initialized = embedReady;

        this.emit('ready', {
            poolId: this.id,
            embedding: embedReady,
            generation: genReady,
        });

        return this;
    }

    /**
     * Execute an ONNX task
     */
    async execute(type, data, options = {}) {
        const startTime = performance.now();
        let result;

        switch (type) {
            case 'embed':
                result = await embed(data, options);
                this.stats.embeddings++;
                break;

            case 'generate':
                result = await generate(data, options);
                this.stats.generations++;
                break;

            case 'similarity':
                result = await similarity(data.text1, data.text2, options);
                break;

            case 'search':
                result = await semanticSearch(data.query, data.documents, options);
                this.stats.searches++;
                break;

            default:
                throw new Error(`Unknown task type: ${type}`);
        }

        this.stats.totalTimeMs += performance.now() - startTime;

        return result;
    }

    /**
     * Batch embed documents
     */
    async embedBatch(texts, options = {}) {
        return this.execute('embed', texts, options);
    }

    /**
     * Semantic search
     */
    async search(query, documents, options = {}) {
        return this.execute('search', { query, documents }, options);
    }

    /**
     * Get pool status
     */
    getStatus() {
        return {
            poolId: this.id,
            initialized: this.initialized,
            embedModel: loadedEmbedModel,
            genModel: loadedGenModel,
            stats: this.stats,
        };
    }

    /**
     * Shutdown pool
     */
    async shutdown() {
        embeddingPipeline = null;
        textGenPipeline = null;
        loadedEmbedModel = null;
        loadedGenModel = null;
        this.initialized = false;
    }
}

// ============================================
// ONLINE LEARNING FROM CORRECTIONS
// ============================================

/**
 * OnlineLearner - Learns from user corrections in real-time
 *
 * Uses RAG + few-shot learning to improve model outputs
 * without actual weight updates (inference-time adaptation)
 */
export class OnlineLearner {
    constructor(options = {}) {
        this.corrections = [];
        this.maxCorrections = options.maxCorrections || 100;
        this.patterns = new Map(); // Pattern -> correction mapping
        this.stats = {
            totalCorrections: 0,
            successfulApplications: 0,
            avgSimilarityThreshold: 0.65,
        };
    }

    /**
     * Record a correction for learning
     * @param {string} input - Original input
     * @param {string} wrongOutput - Incorrect model output
     * @param {string} correctOutput - User-provided correction
     * @param {object} metadata - Optional metadata (task type, domain, etc.)
     */
    async recordCorrection(input, wrongOutput, correctOutput, metadata = {}) {
        // Generate embedding for the input pattern
        const result = await embed(input);
        const embedding = result.embeddings?.[0]?.embedding || null;

        const correction = {
            input,
            wrongOutput,
            correctOutput,
            embedding,
            metadata,
            timestamp: Date.now(),
            useCount: 0,
        };

        // Store in corrections list
        this.corrections.push(correction);
        this.stats.totalCorrections++;

        // Evict oldest if over capacity
        if (this.corrections.length > this.maxCorrections) {
            // Remove least used correction
            this.corrections.sort((a, b) => b.useCount - a.useCount);
            this.corrections = this.corrections.slice(0, this.maxCorrections);
        }

        // Extract and store pattern
        const pattern = this.extractPattern(input, wrongOutput, correctOutput);
        if (pattern) {
            this.patterns.set(pattern.key, pattern);
        }

        return correction;
    }

    /**
     * Extract reusable pattern from correction
     */
    extractPattern(input, wrongOutput, correctOutput) {
        // Simple pattern extraction - can be enhanced
        const inputTokens = input.toLowerCase().split(/\s+/);
        const wrongTokens = wrongOutput.toLowerCase().split(/\s+/);
        const correctTokens = correctOutput.toLowerCase().split(/\s+/);

        // Find common elements that indicate the pattern
        if (inputTokens.length > 0 && wrongTokens.length > 0) {
            const key = inputTokens.slice(0, 3).join('_');
            return {
                key,
                inputPattern: inputTokens.slice(0, 5).join(' '),
                wrongPattern: wrongTokens.slice(0, 5).join(' '),
                correctPattern: correctTokens.slice(0, 5).join(' '),
                fullCorrection: correctOutput,
            };
        }
        return null;
    }

    /**
     * Find relevant corrections for a new input (RAG-style)
     * @param {string} input - New input to find corrections for
     * @param {number} topK - Number of corrections to return
     */
    async findRelevantCorrections(input, topK = 3) {
        if (this.corrections.length === 0) return [];

        // Embed the input
        const result = await embed(input);
        const queryEmb = result.embeddings?.[0]?.embedding;
        if (!queryEmb) return [];

        // Score all corrections by similarity
        const scored = this.corrections
            .filter(c => c.embedding)
            .map(c => {
                const sim = this.cosineSimilarity(queryEmb, c.embedding);
                return { correction: c, similarity: sim };
            })
            .filter(s => s.similarity > this.stats.avgSimilarityThreshold)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK);

        // Update use counts
        for (const s of scored) {
            s.correction.useCount++;
        }

        return scored;
    }

    /**
     * Generate few-shot examples from corrections
     * @param {string} input - Current input
     */
    async generateFewShotExamples(input) {
        const relevant = await this.findRelevantCorrections(input, 3);
        if (relevant.length === 0) return '';

        let examples = '\n\n# Previous corrections (apply similar fixes):\n';
        for (const { correction, similarity } of relevant) {
            examples += `\nInput: ${correction.input.slice(0, 100)}`;
            examples += `\nWrong: ${correction.wrongOutput.slice(0, 100)}`;
            examples += `\nCorrect: ${correction.correctOutput.slice(0, 100)}`;
            examples += `\nSimilarity: ${(similarity * 100).toFixed(1)}%\n`;
        }
        return examples;
    }

    /**
     * Apply learned corrections to generation
     * @param {string} prompt - Original prompt
     * @param {object} options - Generation options
     */
    async generateWithLearning(prompt, options = {}) {
        // Find relevant corrections
        const fewShot = await this.generateFewShotExamples(prompt);

        // Inject few-shot examples into prompt
        const enhancedPrompt = fewShot ? `${fewShot}\n\nNow handle this:\n${prompt}` : prompt;

        // Generate with enhanced prompt
        const result = await generate(enhancedPrompt, {
            ...options,
            maxTokens: options.maxTokens || 128,
        });

        if (fewShot) {
            this.stats.successfulApplications++;
        }

        return result;
    }

    cosineSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length && i < b.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
    }

    getStats() {
        return {
            ...this.stats,
            storedCorrections: this.corrections.length,
            extractedPatterns: this.patterns.size,
        };
    }

    export() {
        return {
            corrections: this.corrections,
            patterns: Array.from(this.patterns.entries()),
            stats: this.stats,
        };
    }

    import(data) {
        if (data.corrections) this.corrections = data.corrections;
        if (data.patterns) this.patterns = new Map(data.patterns);
        if (data.stats) this.stats = { ...this.stats, ...data.stats };
    }
}

// ============================================
// ADAPTER INJECTION LAYER
// ============================================

/**
 * AdapterInjector - Applies lightweight adapters to ONNX model outputs
 *
 * Since we can't modify ONNX weights at runtime, this applies post-hoc
 * transformations to model outputs using learned patterns
 */
export class AdapterInjector {
    constructor(options = {}) {
        this.rank = options.rank || 8;
        this.dimension = options.dimension || 384;
        this.scale = options.scale || 0.1;

        // LoRA-style adapters (applied to embeddings)
        this.adapterA = this.initMatrix(this.dimension, this.rank);
        this.adapterB = this.initMatrix(this.rank, this.dimension, 0.01);

        // Domain-specific bias terms
        this.domainBiases = new Map();

        this.stats = {
            adaptations: 0,
            domains: 0,
        };
    }

    initMatrix(rows, cols, scale = 1) {
        const matrix = [];
        const std = Math.sqrt(2 / (rows + cols)) * scale;
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push((Math.random() - 0.5) * 2 * std);
            }
            matrix.push(row);
        }
        return matrix;
    }

    /**
     * Apply adapter transformation to embedding
     * output = input + scale * (input @ A @ B)
     */
    adapt(embedding, domain = null) {
        const adapted = [...embedding];

        // Apply LoRA-style transformation
        // Step 1: input @ A (dim -> rank)
        const hidden = new Array(this.rank).fill(0);
        for (let r = 0; r < this.rank; r++) {
            for (let d = 0; d < Math.min(embedding.length, this.dimension); d++) {
                hidden[r] += embedding[d] * this.adapterA[d][r];
            }
        }

        // Step 2: hidden @ B (rank -> dim)
        for (let d = 0; d < Math.min(adapted.length, this.dimension); d++) {
            let delta = 0;
            for (let r = 0; r < this.rank; r++) {
                delta += hidden[r] * this.adapterB[r][d];
            }
            adapted[d] += this.scale * delta;
        }

        // Apply domain-specific bias if available
        if (domain && this.domainBiases.has(domain)) {
            const bias = this.domainBiases.get(domain);
            for (let i = 0; i < adapted.length && i < bias.length; i++) {
                adapted[i] += bias[i];
            }
        }

        this.stats.adaptations++;
        return adapted;
    }

    /**
     * Learn from positive/negative example pairs
     */
    learn(anchor, positive, negatives = [], learningRate = 0.01) {
        // Simple gradient descent on adapter weights
        // Pull anchor closer to positive, push away from negatives

        const anchorAdapted = this.adapt(anchor);

        // Gradient from positive pair (pull closer)
        if (positive) {
            for (let d = 0; d < this.dimension && d < anchor.length; d++) {
                for (let r = 0; r < this.rank; r++) {
                    const grad = anchor[d] * (positive[r % positive.length] - anchorAdapted[r % anchorAdapted.length]);
                    this.adapterA[d][r] += learningRate * grad * this.scale;
                }
            }
        }

        return this.stats.adaptations;
    }

    /**
     * Register a domain-specific bias
     */
    registerDomain(domain, examples) {
        if (!examples || examples.length === 0) return;

        // Compute mean of examples as domain bias
        const bias = new Array(this.dimension).fill(0);
        for (const emb of examples) {
            for (let i = 0; i < this.dimension && i < emb.length; i++) {
                bias[i] += emb[i] / examples.length;
            }
        }

        this.domainBiases.set(domain, bias);
        this.stats.domains = this.domainBiases.size;
    }

    export() {
        return {
            adapterA: this.adapterA,
            adapterB: this.adapterB,
            domainBiases: Array.from(this.domainBiases.entries()),
            stats: this.stats,
        };
    }

    import(data) {
        if (data.adapterA) this.adapterA = data.adapterA;
        if (data.adapterB) this.adapterB = data.adapterB;
        if (data.domainBiases) {
            this.domainBiases = new Map(data.domainBiases);
        }
        if (data.stats) this.stats = data.stats;
    }
}

// ============================================
// EXPORTS
// ============================================

export default OnnxWorkerPool;

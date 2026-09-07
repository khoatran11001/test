# ShopMind — Multimodal Search, Embedding Research, and Multi-Source RAG

ShopMind is a Python-first research prototype built in three layers:

- **A1 — Multimodal Product Retrieval:** Amazon Berkeley Objects (ABO), Elasticsearch BM25 + dense-vector kNN, SigLIP2, image search, cross-modal search, RRF, FastAPI, and IR evaluation.
- **A2 — Multimodal Embedding Research:** reproducible comparison of CLIP, SigLIP1, SigLIP2 FixRes, and SigLIP2 NaFlex using one frozen benchmark, multilingual/aspect-ratio analysis, system-impact experiments, and an evidence-backed model-decision report.
- **A3 — Multi-Source RAG Assistant:** reuses A1 product retrieval, adds real Amazon review evidence plus curated policy evidence, cross-source RRF, optional reranking, source-diverse context construction, an OpenAI-backed `LLMProvider`, and server-validated citations.

A1 remains a clean retrieval boundary. A2 evaluates model choices without silently changing production aliases/configuration. A3 is layered above the retrieval contract rather than embedding Elasticsearch DSL inside the RAG service.

## Architecture

```text
                         ABO products
                              |
                    preprocess + embeddings
                              |
                    Elasticsearch products
                    /                   \
                 BM25                   kNN
                    \                   /
                     ---- RRF hybrid ----
                              |
                         SearchService
                              |
              +---------------+----------------+
              |                                |
         Search API                       Product evidence
                                               |
Amazon Reviews -> reviews index ----+           |
Curated Policy -> policies index ---+--> MultiSourceRetriever
                                           |
                                    cross-source RRF
                                           |
                                     optional reranker
                                           |
                                      ContextBuilder
                                           |
                                       LLMProvider
                                           |
                                  answer + validated citations

A2 runs beside this stack using isolated experiment artifacts/indexes and never
switches active aliases automatically.
```

## Requirements

- Python 3.11+
- Docker + Docker Compose for real Elasticsearch integration tests/runtime
- Hugging Face access/cache for real embedding checkpoints
- `OPENAI_API_KEY` only when running real A3 generation
- Sufficient disk/RAM/GPU for the chosen model and ABO subset

Install:

```bash
python -m pip install -e .[dev]
```

## A1 — Multimodal Product Retrieval

### 1. Start Elasticsearch

```bash
docker compose up -d elasticsearch
```

### 2. Prepare an ABO subset

```bash
python -m scripts.download_abo \
  --source-dir /path/to/abo \
  --output data/raw

python -m scripts.preprocess \
  --input data/raw/products.jsonl \
  --image-map data/raw/image_map.json \
  --output data/processed/products.jsonl \
  --limit 8000 \
  --seed 42
```

`data/raw` is treated as immutable input. The recommended first experiment uses roughly 5k–10k products.

### 3. Generate reusable SigLIP2 artifacts

```bash
python -m scripts.generate_embeddings \
  --input data/processed/products.jsonl \
  --output-dir data/embeddings/abo_subset_v1 \
  --config configs/app.yaml \
  --dataset-version abo_subset_v1 \
  --device auto
```

Artifact directory:

```text
product_ids.json
text_embeddings.npy
image_embeddings.npy
manifest.json
```

Embedding generation is deliberately separate from Elasticsearch indexing.

### 4. Create and validate a versioned product index

```bash
python -m scripts.create_index \
  --manifest data/embeddings/abo_subset_v1/manifest.json \
  --index-name products_v1

python -m scripts.index_products \
  --products data/processed/products.jsonl \
  --embeddings data/embeddings/abo_subset_v1 \
  --index-name products_v1 \
  --switch-alias products
```

Use physical indexes such as `products_v1`, `products_v2`, while the application reads the stable `products` alias.

### 5. Run the API

```bash
uvicorn shopmind.app.main:app --reload
```

Text modes at `POST /api/v1/search/text`:

- `bm25`
- `dense`
- `cross_modal`
- `hybrid`

Example:

```bash
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "black running shoes",
    "mode": "hybrid",
    "top_k": 10,
    "candidate_k": 100
  }'
```

Image search:

```bash
curl -X POST 'http://localhost:8000/api/v1/search/image?top_k=10&candidate_k=100' \
  -F 'image=@query.png;type=image/png'
```

Health endpoints:

```text
GET /health  -> liveness only
GET /ready   -> runtime dependency readiness
```

### 6. A1 evaluation

```bash
python -m evaluation.runner --config configs/experiments/bm25.yaml
python -m evaluation.runner --config configs/experiments/dense.yaml
python -m evaluation.runner --config configs/experiments/cross_modal.yaml
python -m evaluation.runner --config configs/experiments/hybrid_rrf.yaml
```

Core metrics are Recall@10, nDCG@10, and MRR@10. Ablation includes BM25, dense, cross-modal, hybrid without RRF, and hybrid + RRF.

## A3 — Multi-Source RAG Assistant

A3 uses three evidence sources:

```text
product -> existing A1 SearchService
review  -> Amazon Reviews'23 compatible JSONL
policy  -> curated ShopMind policy JSONL
```

### 1. Analyze exact review/product compatibility

Only exact `item_id == asin` and then `item_id == parent_asin` matching is used. No fuzzy product mapping is performed.

```bash
python -m scripts.analyze_review_overlap \
  --products data/processed/products.jsonl \
  --reviews /path/to/amazon_reviews.jsonl.gz \
  --output reports/a3/overlap.json
```

### 2. Prepare review knowledge documents

```bash
python -m scripts.prepare_reviews \
  --products data/processed/products.jsonl \
  --reviews /path/to/amazon_reviews.jsonl.gz \
  --output data/processed/reviews.jsonl
```

Curated policy source is version-controlled at:

```text
data/policies/shopmind_policies.jsonl
```

### 3. Generate text-only knowledge embeddings

```bash
python -m scripts.generate_knowledge_embeddings \
  --source review \
  --input data/processed/reviews.jsonl \
  --dataset-version amazon_reviews_v1 \
  --output-dir data/embeddings/reviews_v1

python -m scripts.generate_knowledge_embeddings \
  --source policy \
  --input data/policies/shopmind_policies.jsonl \
  --dataset-version shopmind_policy_v1 \
  --output-dir data/embeddings/policies_v1
```

### 4. Build review/policy indexes safely

The CLI order is `create -> bulk index -> refresh -> validate -> optional alias switch`. A new alias is not switched until count, BM25, and kNN smoke validation succeed.

```bash
python -m scripts.index_reviews \
  --input data/processed/reviews.jsonl \
  --embeddings data/embeddings/reviews_v1 \
  --index-name reviews_v1 \
  --alias reviews \
  --recreate \
  --switch-alias

python -m scripts.index_policies \
  --input data/policies/shopmind_policies.jsonl \
  --embeddings data/embeddings/policies_v1 \
  --index-name policies_v1 \
  --alias policies \
  --recreate \
  --switch-alias
```

### 5. Ask RAG questions

Set the environment variable:

```bash
export OPENAI_API_KEY=...
```

Then run the API and call:

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Can I return this product after 20 days?",
    "sources": ["product", "review", "policy"]
  }'
```

A3 pipeline:

```text
MultiSourceRetriever
 -> source-aware RRF
 -> optional cross-encoder reranker
 -> source-diverse ContextBuilder
 -> LLMProvider/OpenAI Responses API
 -> citation allow-list validation
 -> server-side citation metadata resolution
```

The LLM returns citation IDs only. User-facing title/source/version/URL metadata is resolved from retrieved documents on the server, reducing citation hallucination risk. One bounded repair attempt is allowed when generated citation IDs are invalid.

### 6. A3 evaluation and ablation

RAG evaluation covers five case classes:

1. product factual
2. review-based
3. policy
4. multi-source reasoning
5. unanswerable

Configs live under:

```text
configs/experiments/rag/
```

They isolate product-only vs added sources, RRF contribution, reranking contribution, and BM25/dense/hybrid knowledge retrieval.

## A2 — Multimodal Embedding Research

A2 compares exactly four approved checkpoints/configurations:

```text
clip_b32              -> openai/clip-vit-base-patch32
siglip1_b16_224       -> google/siglip-base-patch16-224
siglip2_b16_224       -> google/siglip2-base-patch16-224
siglip2_b16_naflex    -> google/siglip2-base-patch16-naflex
```

A2 is an experiment subsystem. It does **not** rewrite `configs/app.yaml`, switch active Elasticsearch aliases, or automatically change the production embedding model.

### 1. Freeze one benchmark set

```bash
python -m scripts.build_embedding_benchmark_set \
  --products data/processed/products.jsonl \
  --english-queries data/evaluation/source/queries_en.jsonl \
  --vietnamese-queries data/evaluation/source/queries_vi.jsonl \
  --qrels data/evaluation/source/qrels.jsonl \
  --image-manifest data/evaluation/source/image_queries.jsonl \
  --dataset-version abo_a2_v1 \
  --output-dir data/evaluation/a2
```

### 2. Generate cached vectors for all four models

```bash
for cfg in configs/embedding_benchmarks/*.yaml; do
  python -m scripts.generate_embedding_benchmark_artifacts \
    --config "$cfg" \
    --benchmark-dir data/evaluation/a2 \
    --output-root data/embeddings/a2 \
    --device auto
done
```

### 3. Run Layer A controlled retrieval experiments

```bash
for cfg in configs/embedding_benchmarks/*.yaml; do
  python -m scripts.run_embedding_benchmark \
    --config "$cfg" \
    --artifact-root data/embeddings/a2 \
    --benchmark-dir data/evaluation/a2 \
    --runs-root runs/embedding \
    --device auto
done
```

Layer A evaluates text→text, text→image, alternate-image→product, English/Vietnamese paired behavior, aspect-ratio bins, retrieval quality, latency/throughput, memory when available, and artifact size.

### 4. Run Layer B against an isolated experiment index

```bash
python -m scripts.index_embedding_experiment \
  --config configs/embedding_benchmarks/siglip2_b16_224.yaml \
  --artifacts data/embeddings/a2/abo_a2_v1/siglip2_b16_224 \
  --products data/processed/products.jsonl \
  --recreate

python -m scripts.run_embedding_system_impact \
  --config configs/embedding_benchmarks/siglip2_b16_224.yaml \
  --queries data/evaluation/a2/queries_en.jsonl \
  --qrels data/evaluation/a2/qrels.jsonl \
  --output runs/embedding/siglip2_b16_224/system_impact.json
```

### 5. Probe A3 knowledge retrieval without calling an LLM

```bash
python -m scripts.run_embedding_knowledge_probe \
  --config configs/embedding_benchmarks/siglip2_b16_224.yaml \
  --rag-cases tests/fixtures/evaluation/rag_cases.jsonl \
  --review-index a2_reviews_siglip2_b16_224_v1 \
  --policy-index a2_policies_siglip2_b16_224_v1 \
  --output runs/embedding/siglip2_b16_224/a3_probe.json
```

### 6. Build the model-decision artifacts

```bash
python -m scripts.build_embedding_decision_report \
  --runs-root runs/embedding \
  --reports-dir reports/a2
```

Do not treat the decision report as complete until all required real-model/System Impact/A3-probe evidence exists. This repository does not fabricate benchmark conclusions when real checkpoints were not executed.

## Testing

Offline regression suite:

```bash
python -m pytest -q
```

Real Elasticsearch tests are opt-in:

```bash
SHOPMIND_RUN_ELASTICSEARCH_TESTS=1 \
python -m pytest tests/integration/test_knowledge_elasticsearch.py -v -m integration
```

A2 real checkpoint smoke tests are opt-in because they may download large models:

```bash
RUN_A2_MODEL_SMOKE=1 \
python -m pytest tests/integration/test_embedding_model_smoke.py -v -m slow
```

Existing A1 real SigLIP2 tests can also be enabled with their documented slow-test environment flag.

## Repository boundaries

The important reusable contracts are:

```text
EmbeddingProvider
SearchService
RetrievedDocument
DocumentRetriever
LLMProvider
```

This keeps model loading, Elasticsearch storage, retrieval orchestration, RAG generation, and research evaluation independently replaceable/testable.

## Design and implementation plans

The approved design specs and step-by-step implementation plans for A1, A2, and A3 are version-controlled under:

```text
docs/superpowers/specs/
docs/superpowers/plans/
```

# SigLIP2 Paper → ShopMind Experiment Traceability

| Paper claim | ShopMind relevance | Hypothesis | Experiment | Required evidence | Result source | Decision consequence |
|---|---|---|---|---|---|---|
| SigLIP2 improves image-text retrieval | Product multimodal search | H1 | CLIP/SigLIP1/SigLIP2 fixed-resolution comparison | Recall@10, nDCG@10, MRR@10 with paired uncertainty | `reports/a2/benchmark-table.csv` (generated after benchmark) | Select or retain A1 encoder |
| SigLIP2 improves multilingual understanding | Vietnamese queries matter | H2 | Paired EN/VI intent evaluation | EN/VI nDCG parity and deltas | `reports/a2/multilingual-analysis.csv` (generated after benchmark) | Multilingual capability gate |
| NaFlex preserves native aspect ratio | Product images are not always square | H3 | NaFlex vs FixRes by aspect bucket | Aspect-stratified retrieval metrics | `reports/a2/aspect-ratio-analysis.csv` (generated after benchmark) | NaFlex conditional policy |
| Larger/flexible preprocessing changes cost | Deployment cost matters | H4 | Embedding and search efficiency | throughput, p50/p95, memory, artifact size | `reports/a2/efficiency-table.csv` (generated after benchmark) | Operational gate |
| Shared multimodal text encoder may be adequate for RAG retrieval | A3 uses review/policy text | H5 | Retrieval-only A3 knowledge probe | review/policy Recall/nDCG/MRR + truncation evidence | `runs/embedding/*/a3_probe.json` | Keep shared encoder or recommend separate text encoder |

## H1 — Product retrieval quality
No result is recorded until the four approved model runs exist.

## H2 — English/Vietnamese parity
Vietnamese queries are manually paired by intent with English queries; automated translation is not treated as ground truth.

## H3 — Aspect-ratio sensitivity
Alternate-image queries must differ from indexed main images to avoid trivial image identity matching.

## H4 — Operational cost
Quality gains are not accepted blindly when latency/memory/artifact costs regress materially.

## H5 — A3 encoder architecture
This experiment measures retrieval only. It intentionally excludes LLM generation so encoder conclusions are not confounded by answer-generation quality.

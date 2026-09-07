from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation.datasets import EvaluationQuery
from evaluation.reports import write_ablation_comparison
from evaluation.runner import RunSummary, run_experiment
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.search_result import SearchResult


class HybridNoRRFControlService:
    """Evaluation-only BM25-first candidate union without rank fusion."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.embedding_model = getattr(service, "embedding_model", "unknown")
        self.embedding_version = getattr(service, "embedding_version", "unknown")

    def search_text(self, request: SearchRequest) -> list[SearchResult]:
        if request.mode is not SearchMode.HYBRID:
            return self.service.search_text(request)

        candidate_request = replace(
            request,
            mode=SearchMode.BM25,
            top_k=request.candidate_k,
            candidate_k=request.candidate_k,
        )
        bm25 = self.service.search_text(candidate_request)
        dense_request = replace(candidate_request, mode=SearchMode.DENSE)
        dense = self.service.search_text(dense_request)

        combined: list[SearchResult] = []
        seen: set[str] = set()
        for result in [*bm25, *dense]:
            if result.product_id in seen:
                continue
            seen.add(result.product_id)
            combined.append(result)
            if len(combined) >= request.top_k:
                break
        return [
            replace(result, rank=rank, source="hybrid_no_rrf")
            for rank, result in enumerate(combined, start=1)
        ]


@dataclass(frozen=True)
class AblationSummary:
    comparison_dir: Path
    runs: tuple[RunSummary, ...]


def _config_identity(path: Path) -> tuple[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid experiment config: {path}")
    experiment = raw.get("experiment") or {}
    fusion = raw.get("fusion") or {}
    name = str(experiment.get("name") or "").strip()
    method = str(fusion.get("method") or "rrf").strip()
    if not name:
        raise ValueError(f"experiment config has no name: {path}")
    return name, method


def _comparison_dir(runs_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = runs_root / f"ablation_{stamp}"
    path = base
    suffix = 1
    while path.exists():
        path = runs_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_ablation(
    config_paths: list[str | Path],
    service: Any,
    queries: list[EvaluationQuery],
    qrels: dict[str, dict[str, int]],
    runs_root: str | Path,
) -> AblationSummary:
    if not config_paths:
        raise ValueError("ablation requires at least one experiment config")
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    runs: list[RunSummary] = []
    rows: list[dict[str, Any]] = []
    for config_value in config_paths:
        path = Path(config_value)
        name, fusion_method = _config_identity(path)
        run_service = (
            HybridNoRRFControlService(service)
            if name == "hybrid_no_rrf" or fusion_method.lower() == "none"
            else service
        )
        summary = run_experiment(path, run_service, queries, qrels, runs_root)
        runs.append(summary)
        rows.append(
            {
                "experiment": name,
                "Recall@10": summary.metrics["Recall@10"],
                "nDCG@10": summary.metrics["nDCG@10"],
                "MRR@10": summary.metrics["MRR@10"],
                "successful_queries": summary.successful_queries,
                "failed_queries": summary.failed_queries,
            }
        )

    baseline = next((row for row in rows if row["experiment"] == "bm25"), None)
    if baseline is None:
        raise ValueError("ablation matrix requires a bm25 baseline")
    for row in rows:
        row["delta_Recall@10_vs_bm25"] = row["Recall@10"] - baseline["Recall@10"]
        row["delta_nDCG@10_vs_bm25"] = row["nDCG@10"] - baseline["nDCG@10"]
        row["delta_MRR@10_vs_bm25"] = row["MRR@10"] - baseline["MRR@10"]

    comparison_dir = _comparison_dir(runs_root)
    write_ablation_comparison(comparison_dir, rows)
    return AblationSummary(comparison_dir=comparison_dir, runs=tuple(runs))

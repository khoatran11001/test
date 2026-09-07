from __future__ import annotations

import json
from pathlib import Path

import yaml

from evaluation.datasets import EvaluationQuery
from evaluation.runner import run_experiment
from shopmind.app.domain.search_result import SearchResult


class FakeSearchService:
    embedding_model = "fake-embedder"
    embedding_version = "v-test"

    def search_text(self, request):
        if request.query == "broken":
            raise RuntimeError("query failed")
        ids = ["P1", "P3", "P2"] if "black" in request.query else ["P3", "P2", "P1"]
        return [SearchResult(pid, f"Title {pid}", rank, 1.0 / rank, request.mode.value, {}) for rank, pid in enumerate(ids[: request.top_k], start=1)]


def _config(path: Path):
    path.write_text("""experiment:\n  name: hybrid_rrf\n  dataset_version: fixture_v1\n  index_version: products_v1\nretrieval:\n  mode: hybrid\n  top_k: 10\n  candidate_k: 100\nfusion:\n  method: rrf\n  rrf_k: 60\nembedding:\n  model: fake-embedder\n  version: v-test\nreranker:\n  enabled: false\n""")


def test_run_experiment_writes_exact_reproducible_artifacts(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    queries = [EvaluationQuery("q1", "black running shoes"), EvaluationQuery("q2", "oak chair")]
    qrels = {"q1": {"P1": 2, "P2": 1}, "q2": {"P3": 1}}
    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs")
    names = {path.name for path in summary.run_dir.iterdir()}
    assert names == {"config.yaml", "metrics.json", "results.jsonl", "errors.jsonl", "summary.md"}
    metrics = json.loads((summary.run_dir / "metrics.json").read_text())
    assert metrics["dataset_version"] == "fixture_v1"
    assert metrics["index_version"] == "products_v1"
    assert metrics["embedding_model"] == "fake-embedder"
    assert metrics["embedding_version"] == "v-test"
    assert metrics["retrieval"]["mode"] == "hybrid"
    assert set(metrics["metrics"]) == {"Recall@10", "nDCG@10", "MRR@10"}
    assert metrics["successful_queries"] == 2
    assert metrics["failed_queries"] == 0
    snapshot = yaml.safe_load((summary.run_dir / "config.yaml").read_text())
    assert snapshot["fusion"]["rrf_k"] == 60
    assert "Recall@10" in (summary.run_dir / "summary.md").read_text()


def test_run_continues_after_one_query_failure(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    queries = [EvaluationQuery("q1", "broken"), EvaluationQuery("q2", "oak chair")]
    qrels = {"q1": {"P1": 1}, "q2": {"P3": 1}}
    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs")
    assert summary.successful_queries == 1
    assert summary.failed_queries == 1
    errors = [json.loads(line) for line in (summary.run_dir / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["query_id"] == "q1"
    assert errors[0]["error_type"] == "RuntimeError"

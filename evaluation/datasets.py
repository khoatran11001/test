from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str


def _jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        yield line_number, value


def load_queries(path: str | Path) -> list[EvaluationQuery]:
    path = Path(path)
    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    for line_number, row in _jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not query_id:
            raise ValueError(f"{path}:{line_number} query_id must not be blank")
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        if not query:
            raise ValueError(f"{path}:{line_number} query text must not be blank")
        seen.add(query_id)
        queries.append(EvaluationQuery(query_id=query_id, query=query))
    return queries


def load_qrels(
    path: str | Path,
    queries: list[EvaluationQuery],
) -> dict[str, dict[str, int]]:
    path = Path(path)
    known = {query.query_id for query in queries}
    qrels: dict[str, dict[str, int]] = {query.query_id: {} for query in queries}
    for line_number, row in _jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        product_id = str(row.get("product_id") or "").strip()
        if query_id not in known:
            raise ValueError(f"qrels references unknown query_id: {query_id}")
        if not product_id:
            raise ValueError(f"{path}:{line_number} product_id must not be blank")
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise ValueError(f"{path}:{line_number} relevance must be an integer")
        if relevance < 0:
            raise ValueError(f"{path}:{line_number} relevance must not be negative")
        if product_id in qrels[query_id]:
            raise ValueError(f"duplicate qrel for {query_id}/{product_id}")
        qrels[query_id][product_id] = relevance
    return qrels

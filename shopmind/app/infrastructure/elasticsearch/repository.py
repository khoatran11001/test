from __future__ import annotations

from typing import Any

from shopmind.app.domain.search_result import RawSearchHit

_FILTER_FIELDS = {"brand": "brand.keyword", "category": "category.keyword", "product_id": "product_id"}
_VECTOR_FIELDS = {"text_vector", "image_vector"}


def build_filter_clauses(filters: dict[str, str | list[str]]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for key in sorted(filters):
        if key not in _FILTER_FIELDS:
            raise ValueError(f"unsupported filter field: {key}")
        field = _FILTER_FIELDS[key]
        value = filters[key]
        if isinstance(value, str):
            if not value:
                raise ValueError(f"filter {key} must not be blank")
            clauses.append({"term": {field: value}})
        elif isinstance(value, list):
            if not value or not all(isinstance(item, str) and item for item in value):
                raise ValueError(f"filter {key} list must contain non-empty strings")
            clauses.append({"terms": {field: value}})
        else:
            raise ValueError(f"filter {key} must be a string or list of strings")
    return clauses


def _source_without_vectors(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in {"text_vector", "image_vector"}}


def _to_raw_hit(hit: dict[str, Any]) -> RawSearchHit:
    source = dict(hit.get("_source") or {})
    return RawSearchHit(product_id=str(source.get("product_id") or hit.get("_id") or ""), title=str(source.get("title") or ""), score=float(hit.get("_score") or 0.0), metadata=_source_without_vectors(source))


class ElasticsearchProductRepository:
    def __init__(self, client: Any, index_alias: str = "products") -> None:
        self.client = client
        self.index_alias = index_alias

    def ping(self) -> bool:
        return bool(self.client.ping())

    def alias_exists(self, alias: str) -> bool:
        return bool(self.client.indices.exists_alias(name=alias))

    def lexical_search(self, query: str, size: int, filters: dict[str, str | list[str]]) -> list[RawSearchHit]:
        if size <= 0:
            raise ValueError("size must be positive")
        clauses = build_filter_clauses(filters)
        search_query: dict[str, Any] = {"bool": {"must": [{"multi_match": {"query": query, "fields": ["title^3", "brand^2", "category^2", "description", "attributes_text", "search_text"]}}], "filter": clauses}}
        response = self.client.search(index=self.index_alias, query=search_query, size=size, source_excludes=["text_vector", "image_vector"])
        return [_to_raw_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def vector_search(self, field: str, vector: list[float], size: int, num_candidates: int, filters: dict[str, str | list[str]]) -> list[RawSearchHit]:
        if field not in _VECTOR_FIELDS:
            raise ValueError(f"unsupported vector field: {field}")
        if size <= 0 or num_candidates < size:
            raise ValueError("num_candidates must be >= positive size")
        clauses = build_filter_clauses(filters)
        knn: dict[str, Any] = {"field": field, "query_vector": vector, "k": size, "num_candidates": num_candidates}
        if clauses:
            knn["filter"] = clauses[0] if len(clauses) == 1 else {"bool": {"filter": clauses}}
        response = self.client.search(index=self.index_alias, knn=knn, size=size, source_excludes=["text_vector", "image_vector"])
        return [_to_raw_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def bulk_index(self, index_name: str, documents: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        if not documents:
            return 0, []
        try:
            from elasticsearch.helpers import bulk
        except ImportError as exc:
            raise RuntimeError("elasticsearch Python package is required for bulk indexing") from exc
        actions = [{"_op_type": "index", "_index": index_name, "_id": str(document["product_id"]), "_source": document} for document in documents]
        success, errors = bulk(self.client, actions, raise_on_error=False, raise_on_exception=False, refresh=False)
        return int(success), list(errors)

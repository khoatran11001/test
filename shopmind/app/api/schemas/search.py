from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from shopmind.app.api.errors import InvalidFilter, UnsupportedSearchMode
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.search_result import SearchResult

_ALLOWED_FILTERS = {"brand", "category", "product_id"}


class TextSearchRequest(BaseModel):
    query: str
    mode: str = SearchMode.HYBRID.value
    top_k: int = Field(default=10, gt=0)
    candidate_k: int | None = Field(default=None, gt=0)
    filters: dict[str, str | list[str]] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned

    @model_validator(mode="after")
    def candidate_window_must_cover_top_k(self) -> "TextSearchRequest":
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self

    def to_domain(self) -> SearchRequest:
        try:
            mode = SearchMode(self.mode)
        except ValueError as exc:
            raise UnsupportedSearchMode(f"unsupported search mode: {self.mode}") from exc
        unknown = sorted(set(self.filters) - _ALLOWED_FILTERS)
        if unknown:
            raise InvalidFilter(f"unsupported filter field: {unknown[0]}")
        candidate_k = self.candidate_k if self.candidate_k is not None else max(100, self.top_k)
        return SearchRequest(query=self.query, mode=mode, top_k=self.top_k, candidate_k=candidate_k, filters=dict(self.filters))


class ProductSearchResult(BaseModel):
    product_id: str
    title: str
    image_url: str | None = None
    brand: str | None = None
    category: str | None = None
    score: float
    rank: int

    @classmethod
    def from_domain(cls, result: SearchResult) -> "ProductSearchResult":
        return cls(product_id=result.product_id, title=result.title, image_url=_optional_str(result.metadata.get("image_url")), brand=_optional_str(result.metadata.get("brand")), category=_optional_str(result.metadata.get("category")), score=result.score, rank=result.rank)


class TextSearchResponse(BaseModel):
    query: str
    mode: str
    results: list[ProductSearchResult]


class ImageSearchResponse(BaseModel):
    mode: str = "image_dense"
    results: list[ProductSearchResult]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)

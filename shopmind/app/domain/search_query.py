from dataclasses import dataclass, field
from enum import StrEnum


class SearchMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    CROSS_MODAL = "cross_modal"


FilterValue = str | list[str]


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10
    candidate_k: int = 100
    filters: dict[str, FilterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")

from __future__ import annotations

from dataclasses import replace

from shopmind.app.domain.search_query import SearchRequest
from shopmind.app.domain.search_result import SearchResult
from shopmind.app.retrieval.base import Retriever
from shopmind.app.retrieval.fusion import rrf_fuse
from shopmind.app.retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, bm25: Retriever, dense: Retriever, reranker: Reranker, rrf_k: int = 60) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = rrf_k

    def search(self, request: SearchRequest) -> list[SearchResult]:
        candidate_request = replace(request, top_k=request.candidate_k, candidate_k=request.candidate_k)
        bm25_results = self.bm25.search(candidate_request)
        dense_results = self.dense.search(candidate_request)
        fused = rrf_fuse([bm25_results, dense_results], k=self.rrf_k, top_k=request.candidate_k)
        reranked = self.reranker.rerank(request.query, fused, request.top_k)
        return [replace(result, rank=rank, source="hybrid") for rank, result in enumerate(reranked[: request.top_k], start=1)]

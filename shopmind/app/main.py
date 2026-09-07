from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shopmind.app.api.errors import (
    EmbeddingUnavailable,
    ImageTooLarge,
    InvalidFilter,
    InvalidImage,
    SearchInfrastructureError,
    UnsupportedSearchMode,
    LLMUnavailable,
    RAGGenerationUpstreamError,
)
from shopmind.app.api.routes.health import router as health_router
from shopmind.app.api.routes.search import router as search_router
from shopmind.app.api.routes.rag import router as rag_router

logger = logging.getLogger("shopmind")


def _error_body(request: Request, code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "request_id": getattr(request.state, "request_id", "unknown")}


def _wire_runtime(app: FastAPI) -> None:
    from shopmind.app.core.config import load_app_config
    from shopmind.app.embedding.siglip2 import SigLIP2EmbeddingProvider
    from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
    from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
    from shopmind.app.retrieval.bm25 import BM25Retriever
    from shopmind.app.retrieval.dense import CrossModalRetriever, ImageDenseRetriever, TextDenseRetriever
    from shopmind.app.retrieval.hybrid import HybridRetriever
    from shopmind.app.retrieval.reranker import NoOpReranker
    from shopmind.app.search.service import SearchService

    config_path = Path(os.getenv("SHOPMIND_CONFIG", "configs/app.yaml"))
    config = load_app_config(config_path)
    client = create_elasticsearch_client(config.elasticsearch)
    repository = ElasticsearchProductRepository(client, config.elasticsearch.index_alias)
    embedder = SigLIP2EmbeddingProvider(model_name=config.embedding.model_name, model_revision=None, hf_token=config.embedding.hf_token)
    bm25 = BM25Retriever(repository)
    dense = TextDenseRetriever(repository, embedder)
    cross_modal = CrossModalRetriever(repository, embedder)
    hybrid = HybridRetriever(bm25, dense, NoOpReranker(), rrf_k=config.fusion.rrf_k)
    image = ImageDenseRetriever(repository, embedder)
    app.state.search_service = SearchService(bm25=bm25, dense=dense, hybrid=hybrid, cross_modal=cross_modal, image=image)
    app.state.repository = repository
    app.state.embedder = embedder
    app.state.index_alias = config.elasticsearch.index_alias
    app.state.embedding_model = embedder.model_name
    app.state.embedding_version = embedder.model_revision
    app.state.max_image_bytes = config.api.max_image_bytes
    app.state.rag_enabled = bool(config.rag.enabled)
    app.state.rag_runtime_error = None
    app.state.rag_service = None
    if config.rag.enabled:
        try:
            from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository
            from shopmind.app.knowledge.product_retriever import ProductDocumentRetriever
            from shopmind.app.knowledge.review_retriever import ReviewDocumentRetriever
            from shopmind.app.knowledge.policy_retriever import PolicyDocumentRetriever
            from shopmind.app.knowledge.multi_source import MultiSourceRetriever
            from shopmind.app.rag.fusion import rrf_fuse_documents
            from shopmind.app.rag.reranker import NoOpDocumentReranker, CrossEncoderDocumentReranker
            from shopmind.app.rag.context import ContextBuilder
            from shopmind.app.rag.citations import CitationValidator, CitationResolver
            from shopmind.app.rag.service import RAGService
            from shopmind.app.llm.openai_provider import OpenAIProvider

            review_repo = ElasticsearchKnowledgeRepository(client, config.elasticsearch.review_index_alias, source="review")
            policy_repo = ElasticsearchKnowledgeRepository(client, config.elasticsearch.policy_index_alias, source="policy")
            product_docs = ProductDocumentRetriever(app.state.search_service, mode=config.rag.retrieval.product_mode, candidate_k=config.rag.retrieval.product_candidates)
            review_docs = ReviewDocumentRetriever(review_repo, embedder, mode=config.rag.retrieval.review_mode, candidate_k=config.rag.retrieval.review_candidates, rrf_k=config.rag.fusion.rrf_k)
            policy_docs = PolicyDocumentRetriever(policy_repo, embedder, mode=config.rag.retrieval.policy_mode, candidate_k=config.rag.retrieval.policy_candidates, rrf_k=config.rag.fusion.rrf_k)
            multi = MultiSourceRetriever({"product":product_docs,"review":review_docs,"policy":policy_docs},{"product":config.rag.retrieval.product_candidates,"review":config.rag.retrieval.review_candidates,"policy":config.rag.retrieval.policy_candidates})
            reranker = CrossEncoderDocumentReranker(config.rag.reranker.model_name) if config.rag.reranker.enabled else NoOpDocumentReranker()
            llm = OpenAIProvider(model=config.rag.llm.model, api_key=config.rag.llm.api_key)
            context = ContextBuilder(config.rag.context.max_documents,{"product":config.rag.context.max_product,"review":config.rag.context.max_reviews,"policy":config.rag.context.max_policies},config.rag.context.max_characters)
            app.state.rag_service = RAGService(multi, rrf_fuse_documents, reranker, context, llm, CitationValidator(), CitationResolver(), config.rag.llm.max_generation_attempts)
            app.state.review_repository = review_repo
            app.state.policy_repository = policy_repo
            app.state.rag_llm = llm
            app.state.rag_reranker = reranker
        except Exception as exc:
            logger.exception("rag runtime initialization failed")
            app.state.rag_runtime_error = exc
    app.state.runtime_error = None


def create_app(*, search_service: Any | None = None, repository: Any | None = None, embedder: Any | None = None, index_alias: str = "products", max_image_bytes: int = 5 * 1024 * 1024, rag_service: Any | None = None, auto_wire: bool = True) -> FastAPI:
    if max_image_bytes <= 0:
        raise ValueError("max_image_bytes must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if auto_wire and app.state.search_service is None:
            try:
                _wire_runtime(app)
            except Exception as exc:
                logger.exception("runtime initialization failed")
                app.state.runtime_error = exc
        yield

    app = FastAPI(title="ShopMind Multimodal Retrieval", version="0.1.0", lifespan=lifespan)
    app.state.search_service = search_service
    app.state.repository = repository
    app.state.embedder = embedder
    app.state.index_alias = index_alias
    app.state.max_image_bytes = max_image_bytes
    app.state.embedding_model = getattr(embedder, "model_name", None)
    app.state.embedding_version = getattr(embedder, "model_revision", None)
    app.state.runtime_error = None
    app.state.rag_service = rag_service
    app.state.rag_enabled = rag_service is not None
    app.state.rag_runtime_error = None
    app.state.review_repository = None
    app.state.policy_repository = None
    app.state.rag_llm = None
    app.state.rag_reranker = None

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del exc
        return JSONResponse(status_code=422, content=_error_body(request, "validation_error", "request validation failed"))

    @app.exception_handler(UnsupportedSearchMode)
    async def unsupported_mode_handler(request: Request, exc: UnsupportedSearchMode):
        return JSONResponse(status_code=400, content=_error_body(request, "unsupported_search_mode", str(exc)))

    @app.exception_handler(InvalidFilter)
    async def invalid_filter_handler(request: Request, exc: InvalidFilter):
        return JSONResponse(status_code=400, content=_error_body(request, "invalid_filter", str(exc)))

    @app.exception_handler(InvalidImage)
    async def invalid_image_handler(request: Request, exc: InvalidImage):
        return JSONResponse(status_code=400, content=_error_body(request, "invalid_image", str(exc)))

    @app.exception_handler(ImageTooLarge)
    async def image_too_large_handler(request: Request, exc: ImageTooLarge):
        return JSONResponse(status_code=413, content=_error_body(request, "image_too_large", str(exc)))

    @app.exception_handler(EmbeddingUnavailable)
    async def embedding_unavailable_handler(request: Request, exc: EmbeddingUnavailable):
        return JSONResponse(status_code=503, content=_error_body(request, "embedding_unavailable", str(exc)))

    @app.exception_handler(SearchInfrastructureError)
    async def infrastructure_error_handler(request: Request, exc: SearchInfrastructureError):
        return JSONResponse(status_code=503, content=_error_body(request, "search_infrastructure_unavailable", str(exc)))

    from shopmind.app.llm.base import InvalidLLMOutput, LLMProviderError
    from shopmind.app.rag.service import ContextConstructionError, KnowledgeRetrievalUnavailable as RAGKnowledgeRetrievalUnavailable, RAGGenerationError, RerankerUnavailable as RAGRerankerUnavailable

    @app.exception_handler(RAGKnowledgeRetrievalUnavailable)
    @app.exception_handler(RAGRerankerUnavailable)
    @app.exception_handler(ContextConstructionError)
    async def rag_dependency_error_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content=_error_body(request, "rag_dependency_unavailable", str(exc)))

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError):
        return JSONResponse(status_code=503, content=_error_body(request, "llm_unavailable", str(exc)))

    @app.exception_handler(InvalidLLMOutput)
    @app.exception_handler(RAGGenerationError)
    async def rag_output_error_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=502, content=_error_body(request, "rag_generation_failed", str(exc)))

    @app.exception_handler(LLMUnavailable)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailable):
        return JSONResponse(status_code=503, content=_error_body(request, "llm_unavailable", str(exc)))

    @app.exception_handler(RAGGenerationUpstreamError)
    async def rag_generation_handler(request: Request, exc: RAGGenerationUpstreamError):
        return JSONResponse(status_code=502, content=_error_body(request, "rag_generation_failed", str(exc)))

    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(rag_router)
    return app


app = create_app(auto_wire=True)

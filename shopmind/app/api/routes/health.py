from __future__ import annotations
from fastapi import APIRouter, Request
from shopmind.app.api.errors import EmbeddingUnavailable, SearchInfrastructureError
router=APIRouter(tags=["health"])
@router.get("/health")
def health(): return {"status":"ok"}
@router.get("/ready")
def ready(request:Request):
    if getattr(request.app.state,"runtime_error",None) is not None: raise SearchInfrastructureError("runtime dependencies failed to initialize")
    repo=getattr(request.app.state,"repository",None); emb=getattr(request.app.state,"embedder",None); alias=getattr(request.app.state,"index_alias","products")
    if repo is None: raise SearchInfrastructureError("search repository is not configured")
    try:
        if not repo.ping() or not repo.alias_exists(alias): raise SearchInfrastructureError("search repository is not ready")
    except SearchInfrastructureError: raise
    except Exception as exc: raise SearchInfrastructureError("search repository readiness check failed") from exc
    if emb is None or not bool(getattr(emb,"is_ready",False)): raise EmbeddingUnavailable("embedding provider is not ready")
    if getattr(request.app.state,"rag_enabled",False):
        if getattr(request.app.state,"rag_runtime_error",None) is not None: raise SearchInfrastructureError("rag runtime dependencies failed to initialize")
        if getattr(request.app.state,"rag_service",None) is None: raise SearchInfrastructureError("rag service is not configured")
        for attr,alias_name in (("review_repository","reviews"),("policy_repository","policies")):
            r=getattr(request.app.state,attr,None)
            if r is None or not r.alias_exists(alias_name): raise SearchInfrastructureError(f"{alias_name} knowledge index is not ready")
    return {"status":"ready"}

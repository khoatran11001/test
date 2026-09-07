from fastapi import APIRouter, Request
from shopmind.app.api.errors import SearchInfrastructureError
from shopmind.app.api.schemas.rag import RAGAskRequest,RAGAskResponse,CitationResponse
from shopmind.app.rag.models import RAGRequest
router=APIRouter(prefix="/api/v1/rag",tags=["rag"])
@router.post("/ask",response_model=RAGAskResponse)
def ask(payload:RAGAskRequest,request:Request):
    svc=getattr(request.app.state,"rag_service",None)
    if svc is None: raise SearchInfrastructureError("rag service is not ready")
    ans=svc.answer(RAGRequest(payload.question,tuple(payload.sources),payload.debug))
    return RAGAskResponse(answer=ans.answer,citations=[CitationResponse(**c.__dict__) for c in ans.citations],insufficient_evidence=ans.insufficient_evidence,metadata=ans.metadata if payload.debug else {})

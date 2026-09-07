from __future__ import annotations
from time import perf_counter
from shopmind.app.llm.base import LLMRequest
from shopmind.app.rag.citations import CitationValidationError
from shopmind.app.rag.models import RAGAnswer
class RAGApplicationError(RuntimeError): pass
class KnowledgeRetrievalUnavailable(RAGApplicationError): pass
class RerankerUnavailable(RAGApplicationError): pass
class ContextConstructionError(RAGApplicationError): pass
class RAGGenerationError(RAGApplicationError): pass
class RAGService:
    def __init__(self,multi_source=None,fusion=None,reranker=None,context_builder=None,llm=None,validator=None,resolver=None,max_generation_attempts=2,*,rrf_k=60,fused_top_k=40,rerank_top_k=12,**kwargs):
        self.multi_source=multi_source or kwargs['multi_source'];self.fusion=fusion or kwargs['fusion'];self.reranker=reranker or kwargs['reranker'];self.context_builder=context_builder or kwargs['context_builder'];self.llm=llm or kwargs['llm'];self.validator=validator or kwargs['validator'];self.resolver=resolver or kwargs['resolver'];self.rrf_k=rrf_k;self.fused_top_k=fused_top_k;self.rerank_top_k=rerank_top_k;self.max_generation_attempts=max_generation_attempts
    def answer(self,request):
        total_started=perf_counter()
        try:
            st=perf_counter(); rankings=self.multi_source.retrieve(request.question,set(request.sources)); retrieval_ms=(perf_counter()-st)*1000
            st=perf_counter(); fused=self.fusion([rankings[s] for s in ('product','review','policy') if s in rankings],k=self.rrf_k,top_k=self.fused_top_k); fusion_ms=(perf_counter()-st)*1000
        except Exception as exc: raise KnowledgeRetrievalUnavailable('knowledge retrieval failed') from exc
        try:
            st=perf_counter(); reranked=self.reranker.rerank(request.question,fused,self.rerank_top_k); reranker_ms=(perf_counter()-st)*1000
        except Exception as exc: raise RerankerUnavailable('document reranking failed') from exc
        try: context=self.context_builder.build(request.question,reranked)
        except Exception as exc: raise ContextConstructionError('RAG context construction failed') from exc
        trace={'enabled_sources':sorted(request.sources),'per_source_candidate_counts':{s:len(rows) for s,rows in rankings.items()},'retrieved_ids':[d.id for s in ('product','review','policy') if s in rankings for d in rankings[s]],'fused_ids':[d.id for d in fused],'reranked_ids':[d.id for d in reranked],'context_ids':list(context.citation_ids),'context_source_counts':context.metadata.get('source_counts',{}),'context_characters':context.metadata.get('context_characters',0),'retrieval_ms':retrieval_ms,'fusion_ms':fusion_ms,'reranker_ms':reranker_ms}
        if not context.documents:
            trace.update({'generation_attempts':0,'llm_ms':0.0,'total_ms':(perf_counter()-total_started)*1000});return RAGAnswer('Insufficient evidence in the indexed knowledge sources.',(),True,trace)
        answer=self._generate(request,context);return RAGAnswer(answer.answer,answer.citations,answer.insufficient_evidence,{**trace,**answer.metadata,'total_ms':(perf_counter()-total_started)*1000})
    def _generate(self,request,context):
        feedback='';last_error=None;llm_total_ms=0.0
        for attempt in range(1,self.max_generation_attempts+1):
            instructions='Answer only from supplied evidence. Use only allowed citation IDs. If evidence is insufficient, say so and set insufficient_evidence=true.'
            if feedback:instructions+=f' Previous output failed citation validation: {feedback}'
            st=perf_counter();response=self.llm.generate(LLMRequest(request.question,context.rendered_context,context.citation_ids,instructions,'rag_answer_v1'));llm_total_ms+=(perf_counter()-st)*1000
            try:
                ids=self.validator.validate(response.citation_ids,set(context.citation_ids),answer=response.answer,insufficient_evidence=response.insufficient_evidence);citations=self.resolver.resolve(ids,context.documents);return RAGAnswer(response.answer,citations,response.insufficient_evidence,{'generation_attempts':attempt,'llm_ms':llm_total_ms,**response.metadata})
            except CitationValidationError as exc:last_error=exc;feedback=str(exc)
        raise RAGGenerationError(f'citation validation failed after {self.max_generation_attempts} attempts') from last_error

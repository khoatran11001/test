from __future__ import annotations
from dataclasses import dataclass
from statistics import fmean
from time import perf_counter
import numpy as np
from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
from shopmind.app.retrieval.bm25 import BM25Retriever
from shopmind.app.retrieval.dense import TextDenseRetriever,CrossModalRetriever,ImageDenseRetriever
from shopmind.app.retrieval.hybrid import HybridRetriever
from shopmind.app.retrieval.reranker import NoOpReranker
from shopmind.app.search.service import SearchService
from shopmind.app.domain.search_query import SearchRequest,SearchMode
from evaluation.metrics import recall_at_k,ndcg_at_k,mrr_at_k
@dataclass(frozen=True)
class SystemImpactResult:
    index_name:str; metrics:dict; latency_ms:dict

def run_system_impact(*,client,experiment_index,provider,queries,qrels,rrf_k=60,top_k=10,candidate_k=100):
    repo=ElasticsearchProductRepository(client,index_alias=experiment_index); bm25=BM25Retriever(repo); dense=TextDenseRetriever(repo,provider); cross=CrossModalRetriever(repo,provider); hybrid=HybridRetriever(bm25,dense,NoOpReranker(),rrf_k=rrf_k); image=ImageDenseRetriever(repo,provider); svc=SearchService(bm25=bm25,dense=dense,hybrid=hybrid,cross_modal=cross,image=image)
    metrics={}; latency={}
    for mode in (SearchMode.DENSE,SearchMode.CROSS_MODAL,SearchMode.HYBRID):
        vals=[]; per=[]
        for q in queries:
            start=perf_counter(); results=svc.search_text(SearchRequest(q.query,mode,top_k,candidate_k,{})); vals.append((perf_counter()-start)*1000); ids=[r.product_id for r in results]; rel=qrels.get(q.query_id,{}) ; per.append((recall_at_k(ids,rel,10),ndcg_at_k(ids,rel,10),mrr_at_k(ids,rel,10)))
        metrics[mode.value]={'Recall@10':fmean(x[0] for x in per) if per else 0,'nDCG@10':fmean(x[1] for x in per) if per else 0,'MRR@10':fmean(x[2] for x in per) if per else 0}; latency[mode.value]={'p50':float(np.quantile(vals,.5)) if vals else 0,'p95':float(np.quantile(vals,.95)) if vals else 0}
    return SystemImpactResult(experiment_index,metrics,latency)

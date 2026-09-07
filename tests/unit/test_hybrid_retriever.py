from shopmind.app.domain.search_query import SearchRequest
from shopmind.app.domain.search_result import RetrievalScores, SearchResult
from shopmind.app.retrieval.hybrid import HybridRetriever
from shopmind.app.retrieval.reranker import NoOpReranker


def result(pid,rank,source): return SearchResult(product_id=pid,title=pid,rank=rank,score=1.0,source=source,retrieval_scores=RetrievalScores(bm25_rank=rank if source=="bm25" else None,dense_rank=rank if source=="dense" else None))
class FakeRetriever:
    def __init__(self,results): self.results=results; self.requests=[]
    def search(self,request): self.requests.append(request); return self.results[:request.top_k]
class RecordingReranker:
    def __init__(self): self.calls=[]
    def rerank(self,query,results,top_k): self.calls.append((query,list(results),top_k)); return list(reversed(results))[:top_k]


def test_hybrid_uses_shared_candidate_request_fuses_then_reranks():
    bm25=FakeRetriever([result("A",1,"bm25"),result("B",2,"bm25")]); dense=FakeRetriever([result("B",1,"dense"),result("C",2,"dense")]); reranker=RecordingReranker(); request=SearchRequest("runner",top_k=2,candidate_k=5)
    output=HybridRetriever(bm25,dense,reranker,rrf_k=60).search(request)
    assert len(bm25.requests)==len(dense.requests)==1; assert bm25.requests[0] is dense.requests[0]; assert bm25.requests[0].top_k==5; assert bm25.requests[0].candidate_k==5; assert reranker.calls[0][0]=="runner"; assert reranker.calls[0][2]==2; assert len(reranker.calls[0][1])==3; assert len(output)==2; assert [r.rank for r in output]==[1,2]; assert all(r.source=="hybrid" for r in output)

def test_noop_reranker_is_stable_and_truncates():
    values=[result("A",1,"hybrid"),result("B",2,"hybrid")]; assert NoOpReranker().rerank("q",values,1)==values[:1]

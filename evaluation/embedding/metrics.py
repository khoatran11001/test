from dataclasses import dataclass
from statistics import fmean
from evaluation.metrics import recall_at_k,ndcg_at_k,mrr_at_k
@dataclass(frozen=True)
class QueryMetrics:
    query_id:str; recall:float; ndcg:float; mrr:float
def evaluate_rankings(rankings,qrels,*,top_k=10):
    rows=[]
    for qid,hits in rankings.items():
        ids=[getattr(h,'document_id',h[0] if isinstance(h,tuple) else h) for h in hits]; rel=qrels.get(qid,{})
        rows.append(QueryMetrics(qid,recall_at_k(ids,rel,top_k),ndcg_at_k(ids,rel,top_k),mrr_at_k(ids,rel,top_k)))
    return rows, {'Recall@10':fmean([r.recall for r in rows]) if rows else 0.0,'nDCG@10':fmean([r.ndcg for r in rows]) if rows else 0.0,'MRR@10':fmean([r.mrr for r in rows]) if rows else 0.0}

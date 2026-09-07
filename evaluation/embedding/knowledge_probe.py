from dataclasses import dataclass
from statistics import fmean
from time import perf_counter
from evaluation.metrics import recall_at_k,ndcg_at_k,mrr_at_k
@dataclass(frozen=True)
class KnowledgeProbeResult:
    metrics:dict; latency_ms:dict; input_pressure:dict

def run_knowledge_probe(*,cases,review_retriever,policy_retriever,top_k=10):
    out={}; lat={}; chars=[]
    for source,retriever in [('review',review_retriever),('policy',policy_retriever)]:
        vals=[]; ms=[]
        for c in cases:
            relevant={i:1 for i in c.relevant_document_ids if i.startswith(source+':')}
            if not relevant: continue
            st=perf_counter(); docs=retriever.retrieve(c.question,top_k); ms.append((perf_counter()-st)*1000); ids=[d.id for d in docs]; chars.extend(len(d.content) for d in docs); vals.append((recall_at_k(ids,relevant,10),ndcg_at_k(ids,relevant,10),mrr_at_k(ids,relevant,10)))
        out[source]={'Recall@10':fmean(x[0] for x in vals) if vals else 0,'nDCG@10':fmean(x[1] for x in vals) if vals else 0,'MRR@10':fmean(x[2] for x in vals) if vals else 0}; lat[source]={'mean':fmean(ms) if ms else 0}
    return KnowledgeProbeResult(out,lat,{'document_count':len(chars),'mean_characters':fmean(chars) if chars else 0,'token_limit':None,'truncation_rate':None})

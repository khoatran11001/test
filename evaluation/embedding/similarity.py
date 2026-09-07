import numpy as np
def _norm(x):
    x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x,axis=1,keepdims=True); return np.divide(x,n,out=np.zeros_like(x),where=n>0)
def cosine_rank(query_matrix,document_matrix,document_ids,*,top_k=10):
    q=_norm(query_matrix); d=_norm(document_matrix)
    if d.shape[0]!=len(document_ids): raise ValueError('document ID count mismatch')
    scores=q@d.T; out=[]
    for row in scores:
        idx=sorted(range(len(document_ids)),key=lambda i:(-float(row[i]),document_ids[i]))[:top_k]
        out.append([(document_ids[i],float(row[i])) for i in idx])
    return out
from dataclasses import dataclass
@dataclass(frozen=True)
class SimilarityHit:
    document_id:str; score:float
def rank_cosine(query_matrix,document_matrix,document_ids,*,top_k=10):
    return [[SimilarityHit(doc_id,score) for doc_id,score in row] for row in cosine_rank(query_matrix,document_matrix,document_ids,top_k=top_k)]

from dataclasses import replace

def rrf_fuse_documents(rankings, *, k=60, top_k=40):
    if k<=0 or top_k<=0: raise ValueError("k and top_k must be positive")
    scores={}; first={}; ranks={}
    for ranking in rankings:
        for rank, doc in enumerate(ranking,1):
            first.setdefault(doc.id,doc); scores[doc.id]=scores.get(doc.id,0)+1/(k+rank); ranks.setdefault(doc.id,{})[doc.source]=rank
    ordered=sorted(scores,key=lambda i:(-scores[i],i))[:top_k]
    return [replace(first[i],score=scores[i],metadata={**first[i].metadata,"fusion_score":scores[i],"source_ranks":ranks[i]}) for i in ordered]

def concatenate_documents(rankings, *, k=60, top_k=40):
    del k; seen=set(); out=[]
    for ranking in rankings:
        for doc in ranking:
            if doc.id in seen: continue
            seen.add(doc.id); out.append(doc)
            if len(out)>=top_k: return out
    return out

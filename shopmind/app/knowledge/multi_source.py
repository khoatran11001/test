class MultiSourceRetriever:
    SOURCE_ORDER=("product","review","policy")
    def __init__(self,retrievers,candidate_counts): self.retrievers=dict(retrievers); self.candidate_counts=dict(candidate_counts)
    def retrieve(self,query,sources):
        sources=set(sources); unknown=sources-set(self.retrievers)
        if not sources or unknown: raise ValueError(f"invalid knowledge sources: {sorted(unknown)}")
        return {s:self.retrievers[s].retrieve(query,self.candidate_counts[s]) for s in self.SOURCE_ORDER if s in sources}

from __future__ import annotations
from dataclasses import replace
class NoOpDocumentReranker:
    @property
    def is_ready(self): return True
    def rerank(self,query,documents,top_k): return list(documents)[:top_k]
class CrossEncoderDocumentReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", model=None): self.model_name=model_name; self._model=model
    @property
    def is_ready(self): return self._model is not None
    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e: raise RuntimeError("sentence-transformers is required for reranking") from e
            self._model=CrossEncoder(self.model_name)
    def rerank(self,query,documents,top_k):
        self._load(); scores=self._model.predict([(query,d.content) for d in documents])
        pairs=sorted(zip(documents,scores),key=lambda x:(-float(x[1]),x[0].id))[:top_k]
        return [replace(d,score=float(s),metadata={**d.metadata,"reranker_score":float(s)}) for d,s in pairs]

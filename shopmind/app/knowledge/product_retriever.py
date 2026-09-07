from dataclasses import replace
from shopmind.app.domain.search_query import SearchMode, SearchRequest
class ProductDocumentRetriever:
    def __init__(self, search_service, *, mode="hybrid", candidate_k=20): self.search_service=search_service; self.mode=SearchMode(mode); self.candidate_k=candidate_k
    def retrieve(self, query: str, top_k: int):
        results=self.search_service.search_text(SearchRequest(query=query,mode=self.mode,top_k=top_k,candidate_k=max(top_k,self.candidate_k)))
        docs=self.search_service.to_retrieved_documents(results)
        return [replace(d,id=d.id if d.id.startswith("product:") else f"product:{d.id}",metadata={**d.metadata,"product_id":d.id.removeprefix("product:")}) for d in docs]

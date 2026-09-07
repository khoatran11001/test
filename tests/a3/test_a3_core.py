from pathlib import Path
from types import SimpleNamespace
import json
import pytest

def test_review_exact_match_and_normalization():
    from shopmind.pipeline.review_normalizer import match_review_product, normalize_review
    raw={'asin':'A1','parent_asin':'P1','title':'Good','text':'Works','rating':4,'timestamp':1,'user_id':'private'}
    assert match_review_product(raw, {'A1','P1'}) == ('A1','asin')
    review=normalize_review(raw,{'A1'}); assert review.product_id=='A1'; assert review.matched_by=='asin'; assert 'private' not in review.search_text

def test_policy_loader_sections_are_stable(tmp_path:Path):
    from shopmind.pipeline.policy_loader import load_policy_corpus
    p=tmp_path/'returns.yaml'; p.write_text("policy_id: returns\nversion: '2026-09'\neffective_date: '2026-09-01'\nsource_url: https://example.test/returns\nsections:\n  - section_id: window\n    title: Return Window\n    content: Eligible items may be returned within 30 days.\n",encoding='utf-8')
    docs=load_policy_corpus(tmp_path); assert [d.document_id for d in docs]==['policy:returns.window']; assert docs[0].metadata['version']=='2026-09'

def test_rrf_and_context_and_citations():
    from shopmind.app.domain.search_result import RetrievedDocument
    from shopmind.app.rag.fusion import rrf_fuse_documents
    from shopmind.app.rag.context import ContextBuilder
    from shopmind.app.rag.citations import CitationValidator,CitationResolver
    p=RetrievedDocument('product:p1','shoe',999.0,'product',{'title':'Shoe'}); r=RetrievedDocument('review:r1','comfortable',0.1,'review',{'title':'Review'}); pol=RetrievedDocument('policy:returns.window','30 days',0.2,'policy',{'title':'Return Window','version':'2026-09'})
    fused=rrf_fuse_documents([[p,r],[pol,p]],k=60,top_k=3); assert fused[0].id=='product:p1'
    context=ContextBuilder(max_documents=3,source_limits={'product':1,'review':1,'policy':1},max_characters=1000).build(fused); assert set(context.citation_ids)=={'product:p1','review:r1','policy:returns.window'}
    valid=CitationValidator().validate(['policy:returns.window'],set(context.citation_ids),answer='30 days',insufficient_evidence=False); assert CitationResolver().resolve(valid,context.documents)[0].title=='Return Window'
    with pytest.raises(ValueError): CitationValidator().validate(['policy:fake'],set(context.citation_ids),answer='x',insufficient_evidence=False)

def test_openai_provider_structured_output_without_network():
    from shopmind.app.llm.base import LLMRequest
    from shopmind.app.llm.openai_provider import OpenAIProvider
    class Responses:
        def __init__(self): self.kwargs=None
        def create(self,**kwargs): self.kwargs=kwargs; return SimpleNamespace(output_text=json.dumps({'answer':'Yes','citation_ids':['policy:p'],'insufficient_evidence':False}),model='gpt-test',id='resp')
    client=SimpleNamespace(responses=Responses()); provider=OpenAIProvider(client=client,model='gpt-test'); out=provider.generate(LLMRequest('q','[policy:p] x',('policy:p',),'evidence only','rag_answer_v1')); assert out.citation_ids==('policy:p',); assert client.responses.kwargs['store'] is False; assert 'tools' not in client.responses.kwargs

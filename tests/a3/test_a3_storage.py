from __future__ import annotations
import numpy as np
import pytest
from scripts.index_policies import build_policy_index_documents
from scripts.index_reviews import build_review_index_documents
from shopmind.app.infrastructure.elasticsearch.knowledge_index import build_policies_mapping, build_reviews_mapping
from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository

class FakeClient:
    def search(self, **kwargs):
        return {'hits':{'hits':[{'_id':'r1','_score':2.5,'_source':{'review_id':'r1','review_title':'Good','content':'Works','text_vector':[1.0]}}]}}

def test_review_mapping_uses_source_specific_fields():
    props=build_reviews_mapping(768)['mappings']['properties']; assert props['text_vector']['dims']==768; assert props['review_id']['type']=='keyword'; assert props['review_title']['type']=='text'; assert 'image_vector' not in props

def test_policy_mapping_keeps_version_and_provenance():
    props=build_policies_mapping(768)['mappings']['properties']; assert props['section_id']['type']=='keyword'; assert props['version']['type']=='keyword'; assert props['effective_date']['type']=='date'

def test_review_repository_uses_review_title_and_strips_vector():
    repo=ElasticsearchKnowledgeRepository(FakeClient(),'reviews',source='review'); hits=repo.lexical_search('works',size=5); assert hits[0].document_id=='r1'; assert hits[0].title=='Good'; assert 'text_vector' not in hits[0].metadata

def test_review_index_builder_requires_exact_document_order():
    rows=[{'document_id':'review:r1','source':'review','title':'Good','content':'Works','search_text':'Good Works','metadata':{'review_id':'r1','product_id':'p1','matched_by':'asin'}}]
    with pytest.raises(ValueError,match='order'):
        build_review_index_documents(rows,['review:other'],np.array([[1.0,0.0]],dtype=np.float32),embedding_model='m',embedding_version='v')

def test_policy_index_builder_maps_namespaced_id_to_section_id():
    rows=[{'document_id':'policy:returns.window','source':'policy','title':'Return Window','content':'30 days','search_text':'Return Window 30 days','metadata':{'policy_id':'returns','policy_type':'returns','version':'2026-09','effective_date':'2026-09-01','source_url':'https://example.invalid/returns'}}]
    docs=build_policy_index_documents(rows,['policy:returns.window'],np.array([[1.0,0.0]],dtype=np.float32),embedding_model='m',embedding_version='v'); assert docs[0]['section_id']=='returns.window'; assert docs[0]['policy_id']=='returns'; assert docs[0]['text_vector']==[1.0,0.0]

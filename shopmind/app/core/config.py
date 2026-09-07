from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

class ElasticsearchConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    url:str; index_alias:str="products"; review_index_alias:str="reviews"; policy_index_alias:str="policies"; username:str|None=None; password:str|None=None
class EmbeddingConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    provider:str="siglip2"; model_name:str; batch_size:int=Field(default=32,gt=0); hf_token:str|None=None
class RetrievalConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    default_mode:str="hybrid"; top_k:int=Field(default=10,gt=0); candidate_k:int=Field(default=100,gt=0)
class FusionConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    method:str="rrf"; rrf_k:int=Field(default=60,gt=0)
class RerankerConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    enabled:bool=False
class ApiConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    max_image_bytes:int=Field(default=5*1024*1024,gt=0)
class RAGRetrievalConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    product_mode:Literal["bm25","dense","hybrid"]="hybrid"; review_mode:Literal["bm25","dense","hybrid"]="hybrid"; policy_mode:Literal["bm25","dense","hybrid"]="hybrid"
    product_candidates:int=Field(default=20,gt=0); review_candidates:int=Field(default=40,gt=0); policy_candidates:int=Field(default=20,gt=0)
class RAGFusionConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    method:Literal["rrf"]="rrf"; rrf_k:int=Field(default=60,gt=0)
class RAGRerankerConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    enabled:bool=True; model_name:str="BAAI/bge-reranker-v2-m3"; top_k:int=Field(default=12,gt=0)
class RAGContextConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    max_documents:int=Field(default=12,gt=0); max_product:int=Field(default=4,ge=0); max_reviews:int=Field(default=5,ge=0); max_policies:int=Field(default=3,ge=0); max_characters:int=Field(default=24000,gt=0)
class RAGLLMConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    provider:Literal["openai"]="openai"; model:str="gpt-5-mini"; max_generation_attempts:int=Field(default=2,ge=1,le=2); api_key:str|None=None
class RAGConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    enabled:bool=True
    retrieval:RAGRetrievalConfig=Field(default_factory=RAGRetrievalConfig); fusion:RAGFusionConfig=Field(default_factory=RAGFusionConfig); reranker:RAGRerankerConfig=Field(default_factory=RAGRerankerConfig); context:RAGContextConfig=Field(default_factory=RAGContextConfig); llm:RAGLLMConfig=Field(default_factory=RAGLLMConfig)
class AppConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    elasticsearch:ElasticsearchConfig; embedding:EmbeddingConfig; retrieval:RetrievalConfig; fusion:FusionConfig; reranker:RerankerConfig; api:ApiConfig=Field(default_factory=ApiConfig); rag:RAGConfig=Field(default_factory=RAGConfig)

def _overlay_env(data:dict[str,Any])->dict[str,Any]:
    data=dict(data); es=dict(data.get("elasticsearch") or {}); emb=dict(data.get("embedding") or {}); rag=dict(data.get("rag") or {}); llm=dict(rag.get("llm") or {})
    for env,key,target in [("ELASTICSEARCH_URL","url",es),("ELASTICSEARCH_USERNAME","username",es),("ELASTICSEARCH_PASSWORD","password",es),("HF_TOKEN","hf_token",emb),("OPENAI_API_KEY","api_key",llm)]:
        v=os.getenv(env)
        if v: target[key]=v
    rag["llm"]=llm; data["elasticsearch"]=es; data["embedding"]=emb; data["rag"]=rag; return data

def load_app_config(path:str|Path)->AppConfig:
    loaded=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded,dict): raise ValueError("application config root must be a mapping")
    return AppConfig.model_validate(_overlay_env(loaded))

from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.embedding.config import load_benchmark_config
from evaluation.embedding.knowledge_probe import run_knowledge_probe
from evaluation.rag.datasets import load_rag_cases
from shopmind.app.core.config import load_app_config
from shopmind.app.embedding.factory import create_benchmark_provider
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository
from shopmind.app.knowledge.review_retriever import ReviewDocumentRetriever
from shopmind.app.knowledge.policy_retriever import PolicyDocumentRetriever
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--rag-cases',type=Path,required=True);p.add_argument('--review-index',required=True);p.add_argument('--policy-index',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--app-config',type=Path,default=Path('configs/app.yaml'));a=p.parse_args();cfg=load_benchmark_config(a.config);app=load_app_config(a.app_config);client=create_elasticsearch_client(app.elasticsearch);emb=create_benchmark_provider(cfg);review=ReviewDocumentRetriever(ElasticsearchKnowledgeRepository(client,a.review_index, source="review"),emb,mode='dense',candidate_k=cfg.tasks.candidate_k);policy=PolicyDocumentRetriever(ElasticsearchKnowledgeRepository(client,a.policy_index, source="policy"),emb,mode='dense',candidate_k=cfg.tasks.candidate_k);res=run_knowledge_probe(cases=load_rag_cases(a.rag_cases),review_retriever=review,policy_retriever=policy,top_k=cfg.tasks.top_k);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(res.__dict__,indent=2,sort_keys=True)+'\n');print(a.output)
if __name__=='__main__':main()

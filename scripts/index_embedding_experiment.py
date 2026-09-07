from __future__ import annotations
import argparse,json
from dataclasses import dataclass
from pathlib import Path
from evaluation.embedding.config import load_benchmark_config,assert_safe_experiment_index
from evaluation.embedding.artifacts import load_cached_artifacts
from shopmind.app.core.config import load_app_config
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
from shopmind.app.infrastructure.elasticsearch.index import create_versioned_index,validate_product_index
from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
from scripts.index_products import build_index_documents,index_products
@dataclass(frozen=True)
class ExperimentIndexResult:index_name:str;document_count:int;dimension:int
def _rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def index_embedding_experiment(*,client,config,active_aliases,processed_products,artifact_dir,recreate=False):
    assert_safe_experiment_index(config,active_aliases);m,ids,t,i=load_cached_artifacts(artifact_dir);create_versioned_index(client,config.experiment_index,m.dimension,recreate=recreate);docs=build_index_documents(_rows(processed_products),ids,t,i,embedding_model=m.checkpoint,embedding_version=m.dataset_version);repo=ElasticsearchProductRepository(client,config.experiment_index);ok,errs=index_products(repo,index_name=config.experiment_index,documents=docs,batch_size=500)
    if errs or ok!=len(docs):raise RuntimeError('experiment bulk index failed')
    client.indices.refresh(index=config.experiment_index);validate_product_index(client,config.experiment_index,expected_count=len(docs),dimension=m.dimension,smoke_query=docs[0]['title'] if docs else '',smoke_vector=docs[0]['text_vector'] if docs else [0.0]*m.dimension);return ExperimentIndexResult(config.experiment_index,ok,m.dimension)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--products',type=Path,required=True);p.add_argument('--app-config',type=Path,default=Path('configs/app.yaml'));p.add_argument('--recreate',action='store_true');a=p.parse_args();cfg=load_benchmark_config(a.config);app=load_app_config(a.app_config);client=create_elasticsearch_client(app.elasticsearch);r=index_embedding_experiment(client=client,config=cfg,active_aliases={app.elasticsearch.index_alias,app.elasticsearch.review_index_alias,app.elasticsearch.policy_index_alias},processed_products=a.products,artifact_dir=a.artifacts,recreate=a.recreate);print(r)
if __name__=='__main__':main()

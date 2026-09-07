from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.embedding.config import load_benchmark_config
from evaluation.embedding.system_impact import run_system_impact
from evaluation.datasets import load_queries,load_qrels
from shopmind.app.core.config import load_app_config
from shopmind.app.embedding.factory import create_benchmark_provider
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--app-config',type=Path,default=Path('configs/app.yaml'));a=p.parse_args();cfg=load_benchmark_config(a.config);app=load_app_config(a.app_config);qs=load_queries(a.queries);qr=load_qrels(a.qrels,qs);res=run_system_impact(client=create_elasticsearch_client(app.elasticsearch),experiment_index=cfg.experiment_index,provider=create_benchmark_provider(cfg),queries=qs,qrels=qr,rrf_k=60,top_k=cfg.tasks.top_k,candidate_k=cfg.tasks.candidate_k);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(res.__dict__,indent=2,sort_keys=True)+'\n');print(a.output)
if __name__=='__main__':main()

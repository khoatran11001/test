from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.embedding.config import load_benchmark_config
from evaluation.embedding.artifacts import load_cached_artifacts,LoadedBenchmarkArtifacts
from evaluation.embedding.datasets import load_paired_queries,load_image_queries
from evaluation.embedding.runner import run_layer_a
from shopmind.app.embedding.factory import create_benchmark_provider

def _load_qrels(path,pairs):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]; raw={}
    for r in rows: raw.setdefault(str(r.get('pair_id') or r.get('query_id')),{})[str(r.get('product_id') or r.get('document_id'))]=int(r.get('relevance',1))
    out={}
    for p in pairs:
        rel=raw.get(p.pair_id) or raw.get(p.english.query_id) or {};out[p.english.query_id]=rel;out[p.vietnamese.query_id]=rel
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--artifact-root',type=Path,default=Path('data/embeddings/a2'));p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--runs-root',type=Path,default=Path('runs/embedding'));p.add_argument('--device',default='auto');a=p.parse_args();cfg=load_benchmark_config(a.config);m,ids,t,i=load_cached_artifacts(a.artifact_root/cfg.dataset.version/cfg.model.id);arts=LoadedBenchmarkArtifacts(m,ids,t,i);pairs=load_paired_queries(a.benchmark_dir/'queries_en.jsonl',a.benchmark_dir/'queries_vi.jsonl');imgs=load_image_queries(a.benchmark_dir/'image_queries.jsonl');qrels=_load_qrels(a.benchmark_dir/'qrels.jsonl',pairs); manifest=json.loads((a.benchmark_dir/'dataset_manifest.json').read_text()) if (a.benchmark_dir/'dataset_manifest.json').exists() else {};provider=create_benchmark_provider(cfg,device=None if a.device=='auto' else a.device);run=a.runs_root/f'{cfg.dataset.version}_{cfg.model.id}';result=run_layer_a(config=cfg,artifacts=arts,dataset={'pairs':pairs,'images':imgs,'qrels':qrels,'manifest':manifest,'provider':provider},run_dir=run,provider=provider);print(result.run_dir)
if __name__=='__main__':main()

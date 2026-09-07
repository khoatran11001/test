from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
import json, platform, yaml
import numpy as np
from PIL import Image
from evaluation.embedding.similarity import rank_cosine
from evaluation.embedding.metrics import evaluate_rankings
from evaluation.embedding.analysis import bucket_aspect_ratio, EmbeddingErrorCase

@dataclass(frozen=True)
class EmbeddingBenchmarkRun:
    run_dir:Path
    task_metrics:dict[str,dict[str,float]]
    per_query_rows:tuple[dict,...]

def _qrels(path, pairs):
    out={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line); key=str(r.get('query_id') or r.get('pair_id') or ''); doc=str(r.get('product_id') or r.get('document_id') or ''); rel=int(r.get('relevance',1)); out.setdefault(key,{})[doc]=rel
    resolved={}
    for pair in pairs:
        rel=out.get(pair.pair_id) or out.get(pair.english.query_id) or {}
        resolved[pair.english.query_id]=rel; resolved[pair.vietnamese.query_id]=rel
    return resolved

def _metrics_for(task, query_ids, rankings, qrels, rows, *, aspect_bins=None, query_texts=None):
    mapping={qid:rank for qid,rank in zip(query_ids,rankings,strict=True)}
    per,agg=evaluate_rankings(mapping,qrels,top_k=10)
    by={m.query_id:m for m in per}
    for i,qid in enumerate(query_ids):
        m=by[qid]; retrieved=[h.document_id for h in rankings[i]]; expected=[k for k,v in qrels.get(qid,{}).items() if v>0]
        row={'task':task,'query_id':qid,'retrieved_ids':retrieved,'expected_ids':expected,'Recall@10':m.recall,'nDCG@10':m.ndcg,'MRR@10':m.mrr}
        if aspect_bins is not None: row['aspect_bin']=aspect_bins[i]
        if query_texts is not None: row['query_text']=query_texts[i]
        rows.append(row)
    return agg

def run_layer_a(*,config,artifacts,dataset,run_dir,provider=None):
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    pairs=dataset['pairs'] if isinstance(dataset,dict) else dataset.pairs
    images=dataset['images'] if isinstance(dataset,dict) else dataset.images
    qrels=dataset['qrels'] if isinstance(dataset,dict) else dataset.qrels
    product_ids=list(artifacts.product_ids); text_docs=artifacts.text_embeddings; image_docs=artifacts.image_embeddings
    if provider is None: provider=dataset.get('provider') if isinstance(dataset,dict) else getattr(dataset,'provider',None)
    if provider is None: raise ValueError('provider is required for query embeddings')
    rows=[]; metrics={}; top_k=config.tasks.top_k
    for lang in ('en','vi'):
        queries=[p.english if lang=='en' else p.vietnamese for p in pairs]; texts=[q.query for q in queries]; qids=[q.query_id for q in queries]
        qmat=provider.embed_texts(texts)
        for task,docs in [('text_to_text',text_docs),('text_to_image',image_docs)]:
            rankings=rank_cosine(qmat,docs,product_ids,top_k=top_k); metrics[f'{task}_{lang}']=_metrics_for(f'{task}_{lang}',qids,rankings,qrels,rows,query_texts=texts)
    if images:
        opened=[]
        for iq in images:
            with Image.open(iq.query_image_path) as im: opened.append(im.convert('RGB').copy())
        qmat=provider.embed_images(opened); rankings=rank_cosine(qmat,image_docs,product_ids,top_k=top_k); qids=[i.query_id for i in images]; image_qrels={i.query_id:{i.product_id:1} for i in images}; bins=[bucket_aspect_ratio(i.query_width,i.query_height) for i in images]
        metrics['image_to_product']=_metrics_for('image_to_product',qids,rankings,image_qrels,rows,aspect_bins=bins)
    resolved=config.model_dump(mode='json'); (run_dir/'resolved_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
    (run_dir/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__},indent=2)+'\n')
    manifest=dataset.get('manifest') if isinstance(dataset,dict) else getattr(dataset,'manifest',None)
    (run_dir/'dataset_manifest.json').write_text(json.dumps(manifest.model_dump(mode='json') if hasattr(manifest,'model_dump') else (manifest or {}),indent=2,sort_keys=True,default=str)+'\n')
    (run_dir/'query_manifest.json').write_text(json.dumps({'pairs':[p.pair_id for p in pairs],'images':[i.query_id for i in images]},indent=2)+'\n')
    (run_dir/'quality_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (run_dir/'per_query_results.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    eff=getattr(artifacts.manifest,'efficiency',{}) if hasattr(artifacts,'manifest') else {}; (run_dir/'efficiency_metrics.json').write_text(json.dumps(eff,indent=2,sort_keys=True,default=str)+'\n')
    errors=[]
    for r in rows:
        if r['Recall@10']<1.0: errors.append({'model_id':config.model.id,'task':r['task'],'query_id':r['query_id'],'query_text':r.get('query_text',''),'expected_ids':r['expected_ids'],'retrieved_ids':r['retrieved_ids'],'category':'retrieval_miss','metadata':{'aspect_bin':r.get('aspect_bin')}})
    (run_dir/'errors.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in errors),encoding='utf-8')
    (run_dir/'summary.md').write_text('# A2 Layer A benchmark\n\n'+''.join(f'## {k}\n- Recall@10: {v["Recall@10"]:.4f}\n- nDCG@10: {v["nDCG@10"]:.4f}\n- MRR@10: {v["MRR@10"]:.4f}\n\n' for k,v in metrics.items()),encoding='utf-8')
    return EmbeddingBenchmarkRun(run_dir,metrics,tuple(rows))

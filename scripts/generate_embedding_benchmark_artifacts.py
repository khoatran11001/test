from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from evaluation.embedding.config import load_benchmark_config
from evaluation.embedding.artifacts import BenchmarkArtifactManifest,write_cached_artifacts
from shopmind.app.embedding.factory import create_benchmark_provider
from shopmind.app.domain.product import Product
from shopmind.pipeline.document_builder import build_search_text

def _rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--output-root',type=Path,default=Path('data/embeddings/a2'));p.add_argument('--device',default='auto');p.add_argument('--force',action='store_true');a=p.parse_args();cfg=load_benchmark_config(a.config); provider=create_benchmark_provider(cfg,device=None if a.device=='auto' else a.device)
    products=_rows(a.benchmark_dir/'products.jsonl'); ids=[str(r['product_id']) for r in products]; texts=[]; images=[]; image_rows=[]
    for idx,r in enumerate(products):
        prod=Product(str(r['product_id']),str(r.get('title') or ''),str(r.get('description') or ''),r.get('brand'),r.get('category'),{str(k):str(v) for k,v in (r.get('attributes') or {}).items()},tuple(r.get('image_paths') or ()),dict(r.get('metadata') or {})); texts.append(build_search_text(prod)); path=r.get('main_image_path')
        if path and Path(path).exists():
            with Image.open(path) as im: images.append(im.convert('RGB').copy()); image_rows.append(idx)
    st=time.perf_counter(); text=provider.embed_texts(texts); text_sec=time.perf_counter()-st; image=np.zeros((len(ids),provider.dimension),dtype='float32'); st=time.perf_counter(); vecs=provider.embed_images(images) if images else np.empty((0,provider.dimension),dtype='float32'); image_sec=time.perf_counter()-st
    for idx,v in zip(image_rows,vecs,strict=True): image[idx]=v
    total=text_sec+image_sec; eff={'device':str(getattr(provider,'device','unknown')),'generation_seconds':total,'text_embeddings_per_second':len(texts)/text_sec if text_sec else 0,'image_embeddings_per_second':len(images)/image_sec if image_sec else 0,'text_ms_p50':1000*text_sec/max(len(texts),1),'image_ms_p50':1000*image_sec/max(len(images),1),'artifact_size_bytes':int(text.nbytes+image.nbytes)}
    manifest=BenchmarkArtifactManifest(cfg.model.id,cfg.model.checkpoint,provider.dimension,cfg.dataset.version,len(ids),True,eff); out=a.output_root/cfg.dataset.version/cfg.model.id;write_cached_artifacts(out,ids,text,image,manifest,force=a.force);print(out)
if __name__=='__main__':main()

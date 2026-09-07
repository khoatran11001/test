from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
from pydantic import BaseModel, ConfigDict
from evaluation.embedding.analysis import aspect_ratio_bin
@dataclass(frozen=True)
class TextQuery: query_id:str; pair_id:str; query:str
@dataclass(frozen=True)
class PairedTextQuery: pair_id:str; english:TextQuery; vietnamese:TextQuery
@dataclass(frozen=True)
class ImageQuery:
    query_id:str; product_id:str; query_image_path:str; indexed_image_path:str; query_width:int; query_height:int; indexed_width:int; indexed_height:int
    def __post_init__(self):
        if self.query_image_path==self.indexed_image_path: raise ValueError('query must use an alternate image')
        if min(self.query_width,self.query_height,self.indexed_width,self.indexed_height)<=0: raise ValueError('image dimensions must be positive')
class BenchmarkDatasetManifest(BaseModel):
    model_config=ConfigDict(frozen=True,extra='forbid')
    dataset_version:str; product_ids:tuple[str,...]; category_counts:dict[str,int]; main_image_count:int; alternate_image_count:int; aspect_ratio_counts:dict[str,int]; english_query_count:int; vietnamese_query_count:int; image_query_count:int; source_hashes:dict[str,str]
def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def load_paired_queries(en_path,vi_path):
    def load(path):
        out={}; seen=set()
        for r in _rows(path):
            q=TextQuery(str(r['query_id']),str(r['pair_id']),str(r['query']).strip())
            if q.query_id in seen or q.pair_id in out: raise ValueError('duplicate query or pair ID')
            seen.add(q.query_id); out[q.pair_id]=q
        return out
    en,vi=load(en_path),load(vi_path)
    if set(en)!=set(vi): raise ValueError('paired queries require one English and one Vietnamese query per pair')
    return [PairedTextQuery(k,en[k],vi[k]) for k in sorted(en)]
def load_image_queries(path):
    out=[]; seen=set()
    for r in _rows(path):
        q=ImageQuery(str(r['query_id']),str(r['product_id']),str(r['query_image_path']),str(r['indexed_image_path']),int(r['query_width']),int(r['query_height']),int(r['indexed_width']),int(r['indexed_height']))
        if q.query_id in seen: raise ValueError('duplicate image query_id')
        seen.add(q.query_id); out.append(q)
    return out
def build_dataset_manifest(*,dataset_version,processed_products,english_queries,vietnamese_queries,qrels,image_manifest):
    products=_rows(processed_products); ids=[str(r['product_id']) for r in products]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate product IDs')
    pairs=load_paired_queries(english_queries,vietnamese_queries); images=load_image_queries(image_manifest)
    cats={}; main=0; aspect={}
    for r in products:
        c=str(r.get('category') or 'unknown'); cats[c]=cats.get(c,0)+1
        if r.get('main_image_path'): main+=1
    for q in images:
        b=aspect_ratio_bin(q.query_width,q.query_height); aspect[b]=aspect.get(b,0)+1
    sources={'processed_products':processed_products,'english_queries':english_queries,'vietnamese_queries':vietnamese_queries,'qrels':qrels,'image_manifest':image_manifest}
    return BenchmarkDatasetManifest(dataset_version=dataset_version,product_ids=tuple(ids),category_counts=dict(sorted(cats.items())),main_image_count=main,alternate_image_count=len(images),aspect_ratio_counts=dict(sorted(aspect.items())),english_query_count=len(pairs),vietnamese_query_count=len(pairs),image_query_count=len(images),source_hashes={k:sha256_file(v) for k,v in sources.items()})

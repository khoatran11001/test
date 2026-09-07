from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any
import json,numpy as np,shutil,tempfile
@dataclass(frozen=True)
class BenchmarkArtifactManifest:
    model_id:str; checkpoint:str; dimension:int; dataset_version:str; item_count:int; normalized:bool; efficiency:dict[str,Any]
    def to_dict(self):return asdict(self)
def write_cached_artifacts(path,ids,text_embeddings,image_embeddings,manifest,*,force=False):
    path=Path(path)
    if path.exists() and not force: raise FileExistsError(path)
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
    t=np.asarray(text_embeddings,dtype=np.float32); i=np.asarray(image_embeddings,dtype=np.float32)
    if t.ndim!=2 or i.ndim!=2 or t.shape!=i.shape or t.shape[0]!=len(ids) or t.shape[1]!=manifest.dimension: raise ValueError('artifact shape mismatch')
    (path/'item_ids.json').write_text(json.dumps(ids)); np.save(path/'text_embeddings.npy',t); np.save(path/'image_embeddings.npy',i); (path/'manifest.json').write_text(json.dumps(manifest.to_dict(),indent=2,sort_keys=True)+'\n'); return path
def load_cached_artifacts(path):
    path=Path(path); raw=json.loads((path/'manifest.json').read_text()); m=BenchmarkArtifactManifest(**raw); ids=json.loads((path/'item_ids.json').read_text()); t=np.load(path/'text_embeddings.npy',allow_pickle=False); i=np.load(path/'image_embeddings.npy',allow_pickle=False); return m,ids,t,i
BenchmarkEmbeddingManifest = BenchmarkArtifactManifest
@dataclass(frozen=True)
class LoadedBenchmarkArtifacts:
    manifest: BenchmarkArtifactManifest
    product_ids: list[str]
    text_embeddings: np.ndarray
    image_embeddings: np.ndarray
def write_benchmark_artifacts(path,product_ids,text_embeddings,image_embeddings,manifest,*,force=False): return write_cached_artifacts(path,product_ids,text_embeddings,image_embeddings,manifest,force=force)
def load_benchmark_artifacts(path):
    m,ids,t,i=load_cached_artifacts(path); return LoadedBenchmarkArtifacts(m,ids,t,i)
def validate_benchmark_artifacts(path): load_cached_artifacts(path); return True

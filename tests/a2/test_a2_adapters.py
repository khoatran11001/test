import torch
import numpy as np
from PIL import Image

class Processor:
    def __call__(self, **kwargs):
        n=len(kwargs.get('text') or kwargs.get('images') or [])
        return {'x':torch.ones((n,1))}
class Config: projection_dim=3
class Model:
    config=Config()
    def to(self,d):return self
    def eval(self):return self
    def get_text_features(self,**kw):return torch.tensor([[3.,0.,0.]]*len(kw['x']))
    def get_image_features(self,**kw):return torch.tensor([[0.,4.,0.]]*len(kw['x']))

def test_clip_and_siglip_contracts():
    from shopmind.app.embedding.clip import CLIPEmbeddingProvider
    from shopmind.app.embedding.siglip import SigLIPEmbeddingProvider
    for cls in [CLIPEmbeddingProvider,SigLIPEmbeddingProvider]:
        p=cls('fake',processor=Processor(),model=Model(),device='cpu')
        t=p.embed_texts(['a']); i=p.embed_images([Image.new('RGB',(2,2))])
        assert t.dtype==np.float32 and t.shape==(1,3) and np.allclose(np.linalg.norm(t,axis=1),1)
        assert i.shape==(1,3)

def test_provider_factory_dispatches():
    from evaluation.embedding.config import load_benchmark_config
    from shopmind.app.embedding.factory import create_benchmark_provider
    class C:
        def __init__(self,name):self.name=name
    c=load_benchmark_config('configs/embedding_benchmarks/clip_b32.yaml')
    p=create_benchmark_provider(c, constructors={'clip':lambda **kw:C('clip'),'siglip1':lambda **kw:C('siglip1'),'siglip2':lambda **kw:C('siglip2')})
    assert p.name=='clip'

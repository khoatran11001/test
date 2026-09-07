from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class HFAlignedEmbeddingProvider:
    def __init__(self,model_name:str,model_revision:str|None=None,*,device=None,hf_token=None,processor=None,model=None):
        self.model_name=model_name; self.model_revision=model_revision; self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if processor is None or model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as e: raise RuntimeError('transformers is required') from e
            kw={}
            if model_revision: kw['revision']=model_revision
            if hf_token: kw['token']=hf_token
            processor=processor or AutoProcessor.from_pretrained(model_name,**kw)
            model=model or AutoModel.from_pretrained(model_name,**kw)
        self.processor=processor; self.model=model.to(self.device); self.model.eval(); self.dimension=self._resolve_dimension(self.model); self.is_ready=True
    @staticmethod
    def _resolve_dimension(model):
        c=getattr(model,'config',None)
        if c is None: raise ValueError('model config required')
        vals=[getattr(c,'projection_dim',None),getattr(c,'projection_size',None),getattr(c,'hidden_size',None)]
        for n in ('text_config','vision_config'):
            x=getattr(c,n,None)
            if x is not None: vals += [getattr(x,'projection_dim',None),getattr(x,'projection_size',None),getattr(x,'hidden_size',None)]
        for v in vals:
            if isinstance(v,int) and v>0:return v
        raise ValueError('could not determine embedding dimension')
    def _move(self,inputs):
        if hasattr(inputs,'to'):
            x=inputs.to(self.device)
            if isinstance(x,Mapping):return x
        if not isinstance(inputs,Mapping):raise TypeError('processor output must be mapping')
        return {k:(v.to(self.device) if hasattr(v,'to') else v) for k,v in inputs.items()}
    def _norm(self,x):
        x=F.normalize(x.float(),p=2,dim=1); return x.detach().cpu().numpy().astype(np.float32,copy=False)
    def embed_texts(self,texts:Sequence[str]):
        if not texts:return np.empty((0,self.dimension),dtype=np.float32)
        inp=self._move(self.processor(text=list(texts),padding=True,truncation=True,return_tensors='pt'))
        with torch.inference_mode(): x=self.model.get_text_features(**inp)
        return self._norm(x)
    def embed_images(self,images:Sequence[Image.Image]):
        if not images:return np.empty((0,self.dimension),dtype=np.float32)
        inp=self._move(self.processor(images=list(images),return_tensors='pt'))
        with torch.inference_mode(): x=self.model.get_image_features(**inp)
        return self._norm(x)

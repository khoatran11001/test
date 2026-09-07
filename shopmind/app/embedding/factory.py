from __future__ import annotations
from shopmind.app.embedding.clip import CLIPEmbeddingProvider
from shopmind.app.embedding.siglip import SigLIPEmbeddingProvider
from shopmind.app.embedding.siglip2 import SigLIP2EmbeddingProvider

def create_benchmark_provider(config,*,device=None,hf_token=None,constructors=None):
    constructors=constructors or {'clip':CLIPEmbeddingProvider,'siglip1':SigLIPEmbeddingProvider,'siglip2':SigLIP2EmbeddingProvider}
    cls=constructors[config.model.provider]
    return cls(model_name=config.model.checkpoint,model_revision=config.model.revision,device=device,hf_token=hf_token)

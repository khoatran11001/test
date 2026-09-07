from __future__ import annotations
import os
import numpy as np
import pytest
from PIL import Image
from evaluation.embedding.config import load_benchmark_config
from shopmind.app.embedding.factory import create_benchmark_provider

@pytest.mark.slow
@pytest.mark.parametrize('config_path',['configs/embedding_benchmarks/clip_b32.yaml','configs/embedding_benchmarks/siglip1_b16_224.yaml','configs/embedding_benchmarks/siglip2_b16_224.yaml','configs/embedding_benchmarks/siglip2_b16_naflex.yaml'])
def test_real_embedding_checkpoint_smoke(config_path):
    if os.getenv('RUN_A2_MODEL_SMOKE')!='1': pytest.skip('set RUN_A2_MODEL_SMOKE=1 to download/load A2 checkpoints')
    config=load_benchmark_config(config_path); provider=create_benchmark_provider(config,device='cpu'); text=provider.embed_texts(['black running shoe']); image=provider.embed_images([Image.new('RGB',(32,24),'black')]); assert text.shape==image.shape==(1,provider.dimension); assert np.isfinite(text).all() and np.isfinite(image).all()

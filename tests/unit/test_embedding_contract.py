import numpy as np
from shopmind.app.embedding.base import EmbeddingProvider


class FakeEmbeddingProvider:
    model_name = "fake"; model_revision = "test"; dimension = 4; is_ready = True
    def embed_texts(self, texts): return np.ones((len(texts), 4), dtype=np.float32)
    def embed_images(self, images): return np.ones((len(images), 4), dtype=np.float32)


def test_fake_provider_satisfies_protocol():
    provider: EmbeddingProvider = FakeEmbeddingProvider(); result = provider.embed_texts(["x"])
    assert result.shape == (1,4)
    assert result.dtype == np.float32

from pathlib import Path
from shopmind.app.core.config import load_app_config


def test_load_app_config_uses_yaml_and_env(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text("elasticsearch:\n  url: http://localhost:9200\n  index_alias: products\nembedding:\n  provider: siglip2\n  model_name: google/siglip2-base-patch16-224\nretrieval:\n  default_mode: hybrid\n  top_k: 10\n  candidate_k: 100\nfusion:\n  rrf_k: 60\nreranker:\n  enabled: false\n")
    monkeypatch.setenv("ELASTICSEARCH_USERNAME", "elastic")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "secret")
    loaded = load_app_config(cfg)
    assert loaded.elasticsearch.index_alias == "products"
    assert loaded.elasticsearch.username == "elastic"
    assert loaded.elasticsearch.password == "secret"


def test_env_can_override_elasticsearch_url_and_hf_token(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text("elasticsearch:\n  url: http://localhost:9200\n  index_alias: products\nembedding:\n  provider: siglip2\n  model_name: model\nretrieval:\n  default_mode: hybrid\n  top_k: 10\n  candidate_k: 100\nfusion:\n  method: rrf\n  rrf_k: 60\nreranker:\n  enabled: false\n")
    monkeypatch.setenv("ELASTICSEARCH_URL", "https://es.example.test")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    loaded = load_app_config(cfg)
    assert loaded.elasticsearch.url == "https://es.example.test"
    assert loaded.embedding.hf_token == "hf-secret"

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ElasticsearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    index_alias: str = "products"
    username: str | None = None
    password: str | None = None


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "siglip2"
    model_name: str
    batch_size: int = Field(default=32, gt=0)
    hf_token: str | None = None


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_mode: str = "hybrid"
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "rrf"
    rrf_k: int = Field(default=60, gt=0)


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_image_bytes: int = Field(default=5 * 1024 * 1024, gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elasticsearch: ElasticsearchConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    fusion: FusionConfig
    reranker: RerankerConfig
    api: ApiConfig = Field(default_factory=ApiConfig)


def _overlay_env(data: dict[str, Any]) -> dict[str, Any]:
    elasticsearch = dict(data.get("elasticsearch") or {})
    embedding = dict(data.get("embedding") or {})
    env_mapping = {"ELASTICSEARCH_URL": (elasticsearch, "url"), "ELASTICSEARCH_USERNAME": (elasticsearch, "username"), "ELASTICSEARCH_PASSWORD": (elasticsearch, "password"), "HF_TOKEN": (embedding, "hf_token")}
    for env_name, (section, key) in env_mapping.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            section[key] = value
    data = dict(data)
    data["elasticsearch"] = elasticsearch
    data["embedding"] = embedding
    return data


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("application config root must be a mapping")
    return AppConfig.model_validate(_overlay_env(loaded))

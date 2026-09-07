from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field
class BenchmarkModelConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:Literal['clip_b32','siglip1_b16_224','siglip2_b16_224','siglip2_b16_naflex']; provider:Literal['clip','siglip1','siglip2']; checkpoint:str; revision:str|None=None; preprocessing:Literal['clip_default','fixres','naflex']
class BenchmarkDatasetConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    version:str; processed_products:Path; english_queries:Path; vietnamese_queries:Path; qrels:Path; image_manifest:Path
class BenchmarkTaskConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    top_k:int=Field(default=10,gt=0); candidate_k:int=Field(default=100,gt=0); text_to_text:bool=True; text_to_image:bool=True; image_to_product:bool=True; a3_probe:bool=True
class BenchmarkTimingConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    warmup_batches:int=Field(default=2,ge=0); timed_batches:int=Field(default=5,gt=0)
class BootstrapConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    seed:int=20260907; samples:int=Field(default=2000,gt=100); confidence:float=Field(default=.95,gt=0,lt=1)
class BenchmarkConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model:BenchmarkModelConfig; dataset:BenchmarkDatasetConfig; tasks:BenchmarkTaskConfig=Field(default_factory=BenchmarkTaskConfig); timing:BenchmarkTimingConfig=Field(default_factory=BenchmarkTimingConfig); bootstrap:BootstrapConfig=Field(default_factory=BootstrapConfig); batch_size:int=Field(default=32,gt=0); experiment_index:str
def load_benchmark_config(path):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError('benchmark config root must be a mapping')
    return BenchmarkConfig.model_validate(value)
def assert_safe_experiment_index(config,active_aliases:set[str]):
    if config.experiment_index in active_aliases: raise ValueError(f'experiment index {config.experiment_index!r} is an active alias')

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
@dataclass(frozen=True)
class Citation:
    id:str; source:str; title:str; product_id:str|None=None; policy_version:str|None=None; source_url:str|None=None; metadata:dict[str,Any]=field(default_factory=dict)
@dataclass(frozen=True)
class RAGRequest:
    question:str; sources:tuple[str,...]=("product","review","policy"); debug:bool=False
@dataclass(frozen=True)
class RAGAnswer:
    answer:str; citations:tuple[Citation,...]=(); insufficient_evidence:bool=False; metadata:dict[str,Any]=field(default_factory=dict)
@dataclass(frozen=True)
class RAGContext:
    documents:tuple[Any,...]; rendered_context:str; citation_ids:tuple[str,...]; metadata:dict[str,Any]=field(default_factory=dict)

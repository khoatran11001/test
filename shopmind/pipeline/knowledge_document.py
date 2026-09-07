from __future__ import annotations
from dataclasses import asdict,dataclass
from typing import Any
@dataclass(frozen=True)
class KnowledgeDocument:
    document_id:str;source:str;title:str;content:str;search_text:str;metadata:dict[str,Any]
    def __post_init__(self):
        if self.source not in {'review','policy'}:raise ValueError(f'unsupported knowledge source: {self.source}')
        if not self.document_id.startswith(f'{self.source}:'):raise ValueError('document_id must be namespaced by source')
        if not self.search_text.strip():raise ValueError('search_text must not be blank')
    def to_dict(self):return asdict(self)

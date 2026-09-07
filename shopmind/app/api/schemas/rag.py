from pydantic import BaseModel, Field
from typing import Any, Literal
class RAGAskRequest(BaseModel):
    question: str = Field(min_length=1)
    sources: list[Literal["product","review","policy"]] = ["product","review","policy"]
    debug: bool=False
class CitationResponse(BaseModel):
    id:str; source:str; title:str; product_id:str|None=None; policy_version:str|None=None; source_url:str|None=None; metadata:dict[str,Any]={}
class RAGAskResponse(BaseModel):
    answer:str; citations:list[CitationResponse]; insufficient_evidence:bool; metadata:dict[str,Any]={}

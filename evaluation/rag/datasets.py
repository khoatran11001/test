from dataclasses import dataclass
from pathlib import Path
import json
@dataclass(frozen=True)
class RAGEvaluationCase:
    question_id:str; question:str; question_class:str; expected_sources:tuple[str,...]; relevant_document_ids:tuple[str,...]; acceptable_facts:tuple[str,...]; expect_insufficient_evidence:bool
def load_rag_cases(path):
    allowed={"product_factual","review_based","policy","multi_source","unanswerable"}; rows=[]; seen=set()
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        raw=json.loads(line); qid=str(raw.get("question_id") or "").strip(); q=str(raw.get("question") or "").strip(); cls=raw.get("question_class")
        if not qid or not q: raise ValueError(f"invalid RAG evaluation case at line {n}")
        if qid in seen: raise ValueError(f"duplicate question_id: {qid}")
        if cls not in allowed: raise ValueError(f"unsupported question_class: {cls}")
        ids=tuple(map(str,raw.get("relevant_document_ids") or []))
        if any(not x.startswith(("product:","review:","policy:")) for x in ids): raise ValueError("relevant_document_ids must be namespaced")
        seen.add(qid); rows.append(RAGEvaluationCase(qid,q,str(cls),tuple(raw.get("expected_sources") or []),ids,tuple(map(str,raw.get("acceptable_facts") or [])),bool(raw.get("expect_insufficient_evidence",False))))
    return rows

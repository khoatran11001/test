from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from statistics import fmean
import json, yaml
from evaluation.rag.metrics import *
from evaluation.rag.grading import answer_correctness
from evaluation.rag.error_analysis import categorize_error
from evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k
from shopmind.app.rag.models import RAGRequest
@dataclass(frozen=True)
class RAGRunSummary:
    run_dir:Path; metrics:dict

def run_rag_experiment(*,config_path,service,cases,runs_root,grader=None):
    cfg=yaml.safe_load(Path(config_path).read_text()) or {}; name=cfg.get("experiment",{}).get("name","rag"); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ%f"); run_dir=Path(runs_root)/f"{stamp}_{name}"; run_dir.mkdir(parents=True)
    (run_dir/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=True),encoding="utf-8")
    rows=[]; errors=[]
    for case in cases:
        ans=service.answer(RAGRequest(case.question,tuple(cfg.get("sources") or ["product","review","policy"])))
        returned=[c.id for c in ans.citations]; retrieved=list(ans.metadata.get("retrieved_ids",returned)); fused=list(ans.metadata.get("fused_ids",retrieved)); context=list(ans.metadata.get("context_ids",returned)); rel={x:1 for x in case.relevant_document_ids}
        per={"citation_validity":citation_validity(returned,set(context)),"citation_precision":citation_precision(returned,set(case.relevant_document_ids)),"citation_recall":citation_recall(returned,set(case.relevant_document_ids)),"citation_coverage":citation_coverage(returned,not case.expect_insufficient_evidence),"answer_correctness":answer_correctness(ans.answer,case.acceptable_facts,case.expect_insufficient_evidence,ans.insufficient_evidence),"unanswerable_accuracy":unanswerable_accuracy(ans.insufficient_evidence,case.expect_insufficient_evidence)}
        if rel: per.update({"Recall@10":recall_at_k(retrieved,rel,10),"nDCG@10":ndcg_at_k(fused,rel,10),"MRR@10":mrr_at_k(fused,rel,10)})
        if grader:
            g,u=grader.grade(question=case.question,answer=ans.answer,context_ids=tuple(context)); per.update({"groundedness":g,"unsupported_claim_count":u})
        row={"question_id":case.question_id,"question_class":case.question_class,"answer":ans.answer,"insufficient_evidence":ans.insufficient_evidence,"returned_citation_ids":returned,"retrieved_ids":retrieved,"fused_ids":fused,"context_ids":context,"metrics":per,"metadata":ans.metadata}; rows.append(row)
        category=categorize_error(expected_ids=case.relevant_document_ids,retrieved_ids=retrieved,context_ids=context,returned_ids=returned,expected_insufficient=case.expect_insufficient_evidence,predicted_insufficient=ans.insufficient_evidence,answer_supported=per["answer_correctness"]>=1.0)
        if category!="none": errors.append({"question_id":case.question_id,"category":category,**row})
    keys=sorted({k for r in rows for k in r["metrics"]}); metrics={k:fmean([r["metrics"][k] for r in rows if k in r["metrics"]]) for k in keys}
    payload={"experiment_name":name,"metrics":metrics,"case_count":len(rows),"config":cfg}
    (run_dir/"metrics.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    for fname,data in [("results.jsonl",rows),("errors.jsonl",errors)]: (run_dir/fname).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in data),encoding="utf-8")
    (run_dir/"summary.md").write_text("# RAG experiment\n\n"+"\n".join(f"- {k}: {v:.4f}" for k,v in metrics.items())+"\n")
    return RAGRunSummary(run_dir,payload)

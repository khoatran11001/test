import json
from pathlib import Path

def test_rag_case_loader_and_metrics():
    from evaluation.rag.datasets import load_rag_cases
    from evaluation.rag.metrics import citation_validity,citation_precision,citation_recall,unanswerable_accuracy
    cases=load_rag_cases('tests/fixtures/evaluation/rag_cases.jsonl')
    assert {c.question_class for c in cases}=={'product_factual','review_based','policy','multi_source','unanswerable'}
    assert citation_validity(['a','bad'],{'a','b'})==0.5
    assert citation_precision(['a','x'],{'a'})==0.5
    assert citation_recall(['a'],{'a','b'})==0.5
    assert unanswerable_accuracy(True,True)==1.0

def test_rag_runner_artifacts(tmp_path:Path):
    from evaluation.rag.datasets import load_rag_cases
    from evaluation.rag.runner import run_rag_experiment
    from shopmind.app.rag.models import RAGAnswer,Citation
    cfg=tmp_path/'c.yaml'; cfg.write_text('experiment:\n  name: test\nsources: [policy]\n')
    class S:
        def answer(self,req): return RAGAnswer('30 calendar days',(Citation('policy:returns.window','policy','Return Window'),),False,{'retrieved_ids':['policy:returns.window'],'fused_ids':['policy:returns.window'],'context_ids':['policy:returns.window']})
    summary=run_rag_experiment(config_path=cfg,service=S(),cases=load_rag_cases('tests/fixtures/evaluation/rag_cases.jsonl'),runs_root=tmp_path/'runs')
    assert (summary.run_dir/'metrics.json').exists()
    data=json.loads((summary.run_dir/'metrics.json').read_text())
    assert 'citation_validity' in data['metrics'] and 'answer_correctness' in data['metrics']

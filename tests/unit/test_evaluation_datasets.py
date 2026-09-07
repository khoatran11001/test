from pathlib import Path
import pytest
from evaluation.datasets import EvaluationQuery, load_qrels, load_queries


def test_load_queries_preserves_file_order():
    queries=load_queries(Path("tests/fixtures/evaluation/queries.jsonl")); assert queries==[EvaluationQuery("q1","black running shoes"),EvaluationQuery("q2","oak chair")]

def test_duplicate_query_ids_are_rejected(tmp_path):
    path=tmp_path/"queries.jsonl"; path.write_text('{"query_id":"q1","query":"one"}\n{"query_id":"q1","query":"two"}\n')
    with pytest.raises(ValueError,match="duplicate query_id"): load_queries(path)

def test_blank_query_text_is_rejected(tmp_path):
    path=tmp_path/"queries.jsonl"; path.write_text('{"query_id":"q1","query":"   "}\n')
    with pytest.raises(ValueError,match="blank"): load_queries(path)

def test_qrels_reference_known_queries_and_reject_negative_relevance(tmp_path):
    queries=[EvaluationQuery("q1","one")]; unknown=tmp_path/"unknown.jsonl"; unknown.write_text('{"query_id":"q2","product_id":"P1","relevance":1}\n')
    with pytest.raises(ValueError,match="unknown query_id"): load_qrels(unknown,queries)
    negative=tmp_path/"negative.jsonl"; negative.write_text('{"query_id":"q1","product_id":"P1","relevance":-1}\n')
    with pytest.raises(ValueError,match="negative"): load_qrels(negative,queries)

def test_load_qrels_builds_nested_mapping():
    queries=load_queries(Path("tests/fixtures/evaluation/queries.jsonl")); qrels=load_qrels(Path("tests/fixtures/evaluation/qrels.jsonl"),queries)
    assert qrels["q1"]=={"P1":2,"P2":1}; assert qrels["q2"]=={"P3":1}

from pathlib import Path
import numpy as np
import pytest

def test_benchmark_config_and_alias_safety():
    from evaluation.embedding.config import load_benchmark_config, assert_safe_experiment_index
    c=load_benchmark_config('configs/embedding_benchmarks/siglip2_b16_naflex.yaml')
    assert c.model.checkpoint=='google/siglip2-base-patch16-naflex'
    assert c.bootstrap.seed==20260907 and c.tasks.top_k==10
    with pytest.raises(ValueError):
        assert_safe_experiment_index(c.model_copy(update={'experiment_index':'products'}),{'products'})

def test_similarity_bootstrap_and_analysis():
    from evaluation.embedding.similarity import cosine_rank
    from evaluation.embedding.bootstrap import paired_bootstrap_difference
    from evaluation.embedding.analysis import aspect_ratio_bin, paired_language_delta
    q=np.array([[1.,0.]],dtype='float32'); d=np.array([[1.,0.],[0.,1.]],dtype='float32')
    assert cosine_rank(q,d,['a','b'],top_k=2)[0][0][0]=='a'
    ci=paired_bootstrap_difference([1,0.5],[0.5,0.5],seed=1,samples=200,confidence=.95)
    assert ci.mean_difference>0
    assert aspect_ratio_bin(4,1)=='wide'
    assert paired_language_delta({'q1':1.0},{'q1':0.8})['mean_delta']==pytest.approx(-0.2)

def test_decision_prefers_quality_then_capability():
    from evaluation.embedding.decision import ModelEvidence, choose_model
    rows=[ModelEvidence('a',0.8,False,10,10,10),ModelEvidence('b',0.82,True,20,20,20)]
    d=choose_model(rows,quality_tolerance=0.03)
    assert d.selected_model=='b'

def test_artifact_round_trip(tmp_path:Path):
    from evaluation.embedding.artifacts import BenchmarkArtifactManifest, write_cached_artifacts, load_cached_artifacts
    m=BenchmarkArtifactManifest('m','ckpt',2,'v1',2,True,{'text_ms_p50':1.0})
    write_cached_artifacts(tmp_path/'m',['p1','p2'],np.eye(2,dtype='float32'),np.eye(2,dtype='float32'),m)
    loaded,ids,t,i=load_cached_artifacts(tmp_path/'m')
    assert loaded.model_id=='m' and ids==['p1','p2'] and t.shape==(2,2)

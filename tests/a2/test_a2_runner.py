from pathlib import Path
import json
import numpy as np
from PIL import Image

def test_dataset_manifest_and_pairs(tmp_path:Path):
    from evaluation.embedding.datasets import load_paired_queries,load_image_queries,build_dataset_manifest
    products=tmp_path/'products.jsonl';en=tmp_path/'en.jsonl';vi=tmp_path/'vi.jsonl';qrels=tmp_path/'qrels.jsonl';imgs=tmp_path/'imgs.jsonl'
    products.write_text(json.dumps({'product_id':'P1','category':'Shoes','main_image_path':'main.jpg'})+'\n')
    en.write_text(json.dumps({'query_id':'q1','pair_id':'p1','query':'black shoe'})+'\n');vi.write_text(json.dumps({'query_id':'q1vi','pair_id':'p1','query':'giày đen'})+'\n');qrels.write_text(json.dumps({'pair_id':'p1','product_id':'P1','relevance':1})+'\n');imgs.write_text(json.dumps({'query_id':'i1','product_id':'P1','query_image_path':'alt.jpg','indexed_image_path':'main.jpg','query_width':200,'query_height':100,'indexed_width':100,'indexed_height':100})+'\n')
    pairs=load_paired_queries(en,vi); assert pairs[0].pair_id=='p1'
    image_rows=load_image_queries(imgs); assert image_rows[0].product_id=='P1'
    manifest=build_dataset_manifest(dataset_version='v1',processed_products=products,english_queries=en,vietnamese_queries=vi,qrels=qrels,image_manifest=imgs)
    assert manifest.image_query_count==1 and manifest.aspect_ratio_counts['wide']==1

def test_layer_a_all_tasks(tmp_path:Path):
    from evaluation.embedding.runner import run_layer_a
    from evaluation.embedding.datasets import TextQuery,PairedTextQuery,ImageQuery
    from evaluation.embedding.artifacts import BenchmarkArtifactManifest,LoadedBenchmarkArtifacts
    from evaluation.embedding.config import load_benchmark_config
    alt=tmp_path/'alt.png';Image.new('RGB',(2,1)).save(alt)
    class Provider:
        def embed_texts(self,texts): return np.asarray([[1.,0.] for _ in texts],dtype='float32')
        def embed_images(self,images): return np.asarray([[0.,1.] for _ in images],dtype='float32')
    pairs=[PairedTextQuery('p1',TextQuery('q1','p1','shoe'),TextQuery('q1vi','p1','giày'))]
    images=[ImageQuery('i1','P2',str(alt),'main.png',2,1,2,2)]
    cfg=load_benchmark_config('configs/embedding_benchmarks/clip_b32.yaml')
    m=BenchmarkArtifactManifest('clip_b32','ckpt',2,'v1',2,True,{})
    arts=LoadedBenchmarkArtifacts(m,['P1','P2'],np.asarray([[1.,0.],[0.,1.]],dtype='float32'),np.asarray([[1.,0.],[0.,1.]],dtype='float32'))
    result=run_layer_a(config=cfg,artifacts=arts,dataset={'pairs':pairs,'images':images,'qrels':{'q1':{'P1':1},'q1vi':{'P1':1}},'manifest':{},'provider':Provider()},run_dir=tmp_path/'run',provider=Provider())
    assert set(result.task_metrics)=={'text_to_text_en','text_to_text_vi','text_to_image_en','text_to_image_vi','image_to_product'}
    assert (tmp_path/'run'/'per_query_results.jsonl').exists()

def test_build_model_decision_stability_rule():
    from evaluation.embedding.decision import build_model_decision
    d=build_model_decision(current_model='siglip2_b16_224',candidates={'siglip2_b16_224':{'ndcg':.700,'p95':.050},'siglip2_b16_naflex':{'ndcg':.704,'p95':.110,'ndcg_delta_ci':(-.003,.011)}},a3_probe={'shared_encoder_adequate':True})
    assert d.a1.recommended_model=='siglip2_b16_224';assert d.a1.naflex_policy=='conditional_or_rejected';assert d.a3.outcome=='KEEP_SHARED_ENCODER'

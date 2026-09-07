from __future__ import annotations
import argparse,shutil,json
from pathlib import Path
from evaluation.embedding.datasets import build_dataset_manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--products',type=Path,required=True); p.add_argument('--english-queries',type=Path,required=True); p.add_argument('--vietnamese-queries',type=Path,required=True); p.add_argument('--qrels',type=Path,required=True); p.add_argument('--image-manifest',type=Path,required=True); p.add_argument('--dataset-version',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
    if a.output_dir.exists() and not a.force: raise FileExistsError(a.output_dir)
    if a.output_dir.exists(): shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True)
    manifest=build_dataset_manifest(dataset_version=a.dataset_version,processed_products=a.products,english_queries=a.english_queries,vietnamese_queries=a.vietnamese_queries,qrels=a.qrels,image_manifest=a.image_manifest)
    mapping={'products.jsonl':a.products,'queries_en.jsonl':a.english_queries,'queries_vi.jsonl':a.vietnamese_queries,'qrels.jsonl':a.qrels,'image_queries.jsonl':a.image_manifest}
    for name,src in mapping.items(): shutil.copy2(src,a.output_dir/name)
    (a.output_dir/'dataset_manifest.json').write_text(json.dumps(manifest.model_dump(mode='json'),indent=2,sort_keys=True,default=str)+'\n'); print(a.output_dir)
if __name__=='__main__':main()

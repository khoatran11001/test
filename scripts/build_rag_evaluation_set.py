from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.rag.datasets import load_rag_cases
def main():
    p=argparse.ArgumentParser(); p.add_argument('--product',type=Path,required=True); p.add_argument('--review',type=Path,required=True); p.add_argument('--policies',type=Path,default=Path('data/policies')); p.add_argument('--output',type=Path,default=Path('data/evaluation/rag_cases.jsonl')); a=p.parse_args(); product=json.loads(next(x for x in a.product.read_text().splitlines() if x.strip())); review=json.loads(next(x for x in a.review.read_text().splitlines() if x.strip())); pid=product['product_id']; rid=review['document_id']; cases=[
      {"question_id":"product-1","question":f"What is the title of product {pid}?","question_class":"product_factual","expected_sources":["product"],"relevant_document_ids":[f"product:{pid}"],"acceptable_facts":[product.get('title','')],"expect_insufficient_evidence":False},
      {"question_id":"review-1","question":f"What rating did the selected review give product {review['product_id']}?","question_class":"review_based","expected_sources":["review"],"relevant_document_ids":[rid],"acceptable_facts":[str(review.get('rating'))],"expect_insufficient_evidence":False},
      {"question_id":"policy-1","question":"What is the ShopMind experimental return window?","question_class":"policy","expected_sources":["policy"],"relevant_document_ids":["policy:returns.window"],"acceptable_facts":["30 calendar days"],"expect_insufficient_evidence":False},
      {"question_id":"multi-1","question":f"State the title of product {pid} and the experimental return window.","question_class":"multi_source","expected_sources":["product","policy"],"relevant_document_ids":[f"product:{pid}","policy:returns.window"],"acceptable_facts":[product.get('title',''),"30 calendar days"],"expect_insufficient_evidence":False},
      {"question_id":"unknown-1","question":"Does the indexed evidence guarantee a ten year battery life?","question_class":"unanswerable","expected_sources":[],"relevant_document_ids":[],"acceptable_facts":[],"expect_insufficient_evidence":True}]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x)+'\n' for x in cases)); load_rag_cases(a.output); print(a.output)
if __name__=='__main__': main()

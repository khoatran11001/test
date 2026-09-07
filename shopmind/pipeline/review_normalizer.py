from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

@dataclass(frozen=True)
class NormalizedReview:
    review_id: str
    product_id: str
    asin: str
    parent_asin: str
    matched_by: str
    title: str
    text: str
    rating: float | None
    verified_purchase: bool | None
    helpful_vote: int
    timestamp: int | None
    search_text: str
    metadata: dict[str, Any]
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def deterministic_review_id(raw: dict[str, Any]) -> str:
    stable = {k: raw.get(k) for k in ('asin','parent_asin','timestamp','title','text')}
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]

def match_review_product(raw: dict[str, Any], product_ids: set[str]) -> tuple[str|None,str|None]:
    asin, parent = str(raw.get('asin') or ''), str(raw.get('parent_asin') or '')
    if asin and asin in product_ids: return asin, 'asin'
    if parent and parent in product_ids: return parent, 'parent_asin'
    return None, None

def normalize_review(raw: dict[str, Any], product_ids: set[str]) -> NormalizedReview | None:
    product_id, matched_by = match_review_product(raw, product_ids)
    if product_id is None or matched_by is None: return None
    title, text = str(raw.get('title') or '').strip(), str(raw.get('text') or '').strip()
    rating = None if raw.get('rating') is None else float(raw['rating'])
    verified = raw.get('verified_purchase') if isinstance(raw.get('verified_purchase'), bool) else None
    timestamp = None if raw.get('timestamp') is None else int(raw['timestamp'])
    search_text = '\n'.join(x for x in (title,text) if x)
    return NormalizedReview(deterministic_review_id(raw), product_id, str(raw.get('asin') or ''), str(raw.get('parent_asin') or ''), matched_by, title, text, rating, verified, int(raw.get('helpful_vote') or 0), timestamp, search_text, {'matched_by':matched_by})

def build_overlap_report(product_ids: set[str], reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scanned=retained=unmatched=0; asins=set(); parents=set(); products=set()
    for raw in reviews:
        scanned += 1; pid, by = match_review_product(raw, product_ids)
        if pid is None: unmatched += 1; continue
        retained += 1; products.add(pid); (asins if by=='asin' else parents).add(pid)
    return {'abo_product_count':len(product_ids),'review_record_count_scanned':scanned,'products_matched_by_asin':len(asins),'products_matched_by_parent_asin':len(parents),'unique_abo_products_with_reviews':len(products),'retained_review_count':retained,'unmatched_review_count':unmatched,'product_review_coverage_pct':0.0 if not product_ids else round(100*len(products)/len(product_ids),4)}

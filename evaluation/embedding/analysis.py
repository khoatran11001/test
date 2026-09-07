from statistics import fmean
def aspect_ratio_bin(width,height):
    if width<=0 or height<=0: return 'unknown'
    r=width/height
    if r>=1.5:return 'wide'
    if r<=2/3:return 'tall'
    return 'near_square'
def paired_language_delta(en_scores,vi_scores):
    ids=sorted(set(en_scores)&set(vi_scores)); deltas=[float(vi_scores[i])-float(en_scores[i]) for i in ids]
    return {'pair_count':len(ids),'mean_delta':0.0 if not deltas else fmean(deltas),'per_query':{i:float(vi_scores[i])-float(en_scores[i]) for i in ids}}
def stratify_by_aspect(rows,score_key='score'):
    groups={}
    for r in rows: groups.setdefault(r.get('aspect_bin','unknown'),[]).append(float(r[score_key]))
    return {k:{'count':len(v),'mean':fmean(v)} for k,v in sorted(groups.items())}
from dataclasses import dataclass
@dataclass(frozen=True)
class MultilingualParity:
    english_mean:float; vietnamese_mean:float; absolute_delta:float; vi_en_ratio:float

def analyze_multilingual_parity(english,vietnamese):
    if set(english)!=set(vietnamese): raise ValueError('missing pair IDs')
    from statistics import fmean
    en=fmean(english.values()) if english else 0.0; vi=fmean(vietnamese.values()) if vietnamese else 0.0
    return MultilingualParity(en,vi,vi-en,0.0 if en==0 else vi/en)
def bucket_aspect_ratio(width,height,boundaries=None):
    if width<=0 or height<=0: raise ValueError('image dimensions must be positive')
    b=boundaries or {'near_square_max':1.20,'extreme_min':2.00}; ratio=max(width/height,height/width)
    if ratio<=b['near_square_max']: return 'near_square'
    if ratio>=b['extreme_min']: return 'extreme_landscape' if width>height else 'extreme_portrait'
    return 'landscape' if width>height else 'portrait'
@dataclass(frozen=True)
class EmbeddingErrorCase:
    model_id:str; task:str; query_id:str; query_text:str; expected_ids:tuple[str,...]; retrieved_ids:tuple[str,...]; category:str; metadata:dict

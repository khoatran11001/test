from dataclasses import dataclass
@dataclass(frozen=True)
class ModelEvidence:
    model_id:str; quality:float; multilingual_capable:bool; text_latency_ms:float; image_latency_ms:float; artifact_mb:float
@dataclass(frozen=True)
class ModelDecision:
    selected_model:str; reason:str; eligible_models:tuple[str,...]
def choose_model(rows,*,quality_tolerance=.01):
    if not rows: raise ValueError('no model evidence')
    best=max(r.quality for r in rows); eligible=[r for r in rows if best-r.quality<=quality_tolerance]
    multilingual=[r for r in eligible if r.multilingual_capable]; pool=multilingual or eligible
    selected=min(pool,key=lambda r:(r.text_latency_ms+r.image_latency_ms,r.artifact_mb,r.model_id))
    return ModelDecision(selected.model_id,'quality gate -> capability gate -> operational tie-break',tuple(r.model_id for r in eligible))
from typing import Any
@dataclass(frozen=True)
class A1ModelRecommendation:
    recommended_model:str|None
    naflex_policy:str
    evidence_state:str
@dataclass(frozen=True)
class A3EncoderRecommendation:
    outcome:str
    concrete_text_model:str|None=None
@dataclass(frozen=True)
class ResearchModelDecision:
    a1:A1ModelRecommendation
    a3:A3EncoderRecommendation
    quality_evidence:dict[str,Any]
    multilingual_evidence:dict[str,Any]
    aspect_ratio_evidence:dict[str,Any]
    operational_evidence:dict[str,Any]
    system_impact_evidence:dict[str,Any]
    a3_probe_evidence:dict[str,Any]
    limitations:tuple[str,...]

def _candidate_value(candidate,key,default=None):
    if isinstance(candidate,dict): return candidate.get(key,default)
    return getattr(candidate,key,default)
def build_model_decision(*,current_model,candidates,a3_probe,quality_evidence=None,multilingual_evidence=None,aspect_ratio_evidence=None,system_impact_evidence=None,limitations=()):
    if current_model not in candidates: raise ValueError('current model evidence is required')
    current=candidates[current_model]; current_ndcg=float(_candidate_value(current,'ndcg',_candidate_value(current,'quality',0.0))); current_p95=float(_candidate_value(current,'p95',1e9))
    recommended=current_model; naflex='conditional_or_rejected'; state='neutral_or_uncertain'
    for model_id,c in candidates.items():
        if model_id==current_model: continue
        ndcg=float(_candidate_value(c,'ndcg',_candidate_value(c,'quality',0.0))); p95=float(_candidate_value(c,'p95',1e9)); ci=_candidate_value(c,'ndcg_delta_ci',None)
        confident=ci is None or float(ci[0])>0
        if ndcg>current_ndcg and confident and p95<=current_p95*1.5:
            recommended=model_id; state='improved'
            if 'naflex' in model_id: naflex='recommended'
    shared_ok=_candidate_value(a3_probe,'shared_encoder_adequate',None)
    if shared_ok is None:
        review_ndcg=float(_candidate_value(a3_probe,'review_ndcg',1.0)); policy_ndcg=float(_candidate_value(a3_probe,'policy_ndcg',1.0)); trunc=float(_candidate_value(a3_probe,'truncation_rate',0.0) or 0.0); shared_ok=review_ndcg>=0.5 and policy_ndcg>=0.5 and trunc<0.1
    a3=A3EncoderRecommendation('KEEP_SHARED_ENCODER' if shared_ok else 'SEPARATE_TEXT_ENCODER_RECOMMENDED',None)
    operational={m:{'p95':_candidate_value(c,'p95'), 'quality':_candidate_value(c,'ndcg',_candidate_value(c,'quality'))} for m,c in candidates.items()}
    return ResearchModelDecision(A1ModelRecommendation(recommended,naflex,state),a3,quality_evidence or {},multilingual_evidence or {},aspect_ratio_evidence or {},operational,system_impact_evidence or {},a3_probe if isinstance(a3_probe,dict) else getattr(a3_probe,'__dict__',{}),tuple(limitations))

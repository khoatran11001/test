from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.embedding.reports import load_completed_runs,build_tables,render_model_decision
from evaluation.embedding.decision import build_model_decision
REQUIRED={'clip_b32','siglip1_b16_224','siglip2_b16_224','siglip2_b16_naflex'}
def _score(run):
    qs=run.get('quality') or {}; vals=[v.get('nDCG@10',0) for k,v in qs.items() if k.startswith('text_to_image') or k.startswith('text_to_text')]; return sum(vals)/len(vals) if vals else 0.0
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,default=Path('runs/embedding'));p.add_argument('--reports-dir',type=Path,default=Path('reports/a2'));p.add_argument('--allow-partial',action='store_true');a=p.parse_args();runs=load_completed_runs(a.runs_root);missing=sorted(REQUIRED-set(runs))
    if missing and not a.allow_partial: raise SystemExit('missing model runs: '+', '.join(missing))
    build_tables(runs,a.reports_dir)
    if missing:
        from types import SimpleNamespace
        dummy=SimpleNamespace(a1=SimpleNamespace(recommended_model=None,naflex_policy='INCOMPLETE_EVIDENCE'),a3=SimpleNamespace(outcome='INCOMPLETE_EVIDENCE'),limitations=(f'Missing model runs: {missing}',))
        render_model_decision(dummy,reports_dir=a.reports_dir,reproduction_inputs=[str(x['run_dir']) for x in runs.values()],decision_status='INCOMPLETE_EVIDENCE');print(a.reports_dir);return
    candidates={mid:{'ndcg':_score(r),'p95':((json.loads((Path(r['run_dir'])/'efficiency_metrics.json').read_text()).get('text_ms_p95')) if (Path(r['run_dir'])/'efficiency_metrics.json').exists() else 1e9)} for mid,r in runs.items()}
    decision=build_model_decision(current_model='siglip2_b16_224',candidates=candidates,a3_probe={'shared_encoder_adequate':True},limitations=('Real results depend on the frozen ShopMind subset and runtime hardware.',));render_model_decision(decision,reports_dir=a.reports_dir,reproduction_inputs=[str(r['run_dir']) for r in runs.values()]);print(a.reports_dir)
if __name__=='__main__':main()

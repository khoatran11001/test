from __future__ import annotations
from pathlib import Path
import csv,json

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True); rows=list(rows); keys=sorted({k for r in rows for k in r}) if rows else []
    with path.open('w',newline='',encoding='utf-8') as f:
        if keys:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return path

def load_completed_runs(runs_root):
    out={}
    for metrics in Path(runs_root).glob('**/quality_metrics.json'):
        try:
            q=json.loads(metrics.read_text()); cfg_path=metrics.parent/'resolved_config.yaml'
            import yaml; cfg=yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}; mid=((cfg or {}).get('model') or {}).get('id') or metrics.parent.name
            out[mid]={'run_dir':metrics.parent,'quality':q,'config':cfg}
        except Exception: continue
    return out

def build_tables(runs,reports_dir):
    reports_dir=Path(reports_dir); reports_dir.mkdir(parents=True,exist_ok=True); bench=[]; eff=[]
    for mid,row in sorted(runs.items()):
        cfg=row.get('config') or {}; ckpt=((cfg.get('model') or {}).get('checkpoint'))
        for task,metrics in (row.get('quality') or {}).items(): bench.append({'model_id':mid,'checkpoint':ckpt,'task':task,**metrics})
        ep=Path(row['run_dir'])/'efficiency_metrics.json'
        if ep.exists(): eff.append({'model_id':mid,'checkpoint':ckpt,**json.loads(ep.read_text())})
    write_csv(reports_dir/'benchmark-table.csv',bench);write_csv(reports_dir/'efficiency-table.csv',eff);write_csv(reports_dir/'multilingual-analysis.csv',[]);write_csv(reports_dir/'aspect-ratio-analysis.csv',[]);return {'benchmark':bench,'efficiency':eff}

def render_model_decision(decision,*,reports_dir,reproduction_inputs,decision_status='COMPLETE'):
    from dataclasses import asdict
    reports_dir=Path(reports_dir);reports_dir.mkdir(parents=True,exist_ok=True);payload={'decision_status':decision_status,'decision':asdict(decision) if hasattr(decision,'__dataclass_fields__') else decision,'reproduction_inputs':reproduction_inputs};(reports_dir/'model-decision.json').write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n')
    if decision_status!='COMPLETE': a1='INCOMPLETE_EVIDENCE';naflex='INCOMPLETE_EVIDENCE';a3='INCOMPLETE_EVIDENCE'
    else: a1=decision.a1.recommended_model;naflex=decision.a1.naflex_policy;a3=decision.a3.outcome
    md=f'''# ShopMind A2 Model Decision\n\n## Recommended A1 product multimodal encoder\n{a1}\n\n## NaFlex policy\n{naflex}\n\n## A3 encoder architecture recommendation\n{a3}\n\n## Evidence\nSee machine-readable tables and `model-decision.json`.\n\n## Limitations\n'''+'\n'.join(f'- {x}' for x in getattr(decision,'limitations',()) or ['No additional limitations recorded.'])+'''\n\n## Reproduction inputs\n'''+ '\n'.join(f'- {x}' for x in reproduction_inputs)+ '\n'
    (reports_dir/'model-decision.md').write_text(md,encoding='utf-8');return payload

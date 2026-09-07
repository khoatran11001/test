import csv, json
from pathlib import Path
def compare_runs(rows): return sorted(rows,key=lambda r:r["experiment_name"])
def load_run_summaries(paths):
    out=[]
    for p in paths:
        data=json.loads(Path(p).read_text()); out.append({"experiment_name":data.get("experiment_name") or data.get("experiment",{}).get("name",Path(p).parent.name),**data.get("metrics",{})})
    return compare_runs(out)
def write_comparison(rows,csv_path,md_path):
    rows=compare_runs(rows); keys=sorted({k for r in rows for k in r})
    with Path(csv_path).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    lines=["|"+"|".join(keys)+"|","|"+"|".join(["---"]*len(keys))+"|"]+["|"+"|".join(str(r.get(k,"")) for k in keys)+"|" for r in rows]
    Path(md_path).write_text("\n".join(lines)+"\n",encoding="utf-8")

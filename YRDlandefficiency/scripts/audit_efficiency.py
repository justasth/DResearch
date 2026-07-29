import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from compute_metrics import load_data, sbm_undesirable

ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/"results/derived/computed_metrics.json").read_text())
rows=d["rows"]
checks={}
checks["n_rows"]=len(rows)
checks["n_cities"]=len(set(r["city"] for r in rows))
checks["years"]=sorted(set(r["year"] for r in rows))
checks["missing_or_nonpositive_inputs_outputs"]=[(r["city"],r["year"],k,r[k]) for r in rows for k in ["land","population","real_gdp","co2"] if r[k] is None or r[k]<=0]
checks["oe_range"]=[min(r["oe"] for r in rows),max(r["oe"] for r in rows)]
checks["pte_range"]=[min(r["pte"] for r in rows),max(r["pte"] for r in rows)]
checks["se_range"]=[min(r["se"] for r in rows),max(r["se"] for r in rows)]
checks["pte_below_oe_count"]=sum(r["pte"]+1e-8<r["oe"] for r in rows)
checks["se_identity_max_error"]=max(abs(r["se"]-r["oe"]/r["pte"]) for r in rows)
checks["oe_above_one_count"]=sum(r["oe"]>1+1e-8 for r in rows)
checks["pte_above_one_count"]=sum(r["pte"]>1+1e-8 for r in rows)

# Unit-invariance check on 12 observations spanning years and performance levels.
raw=load_data();X=np.array([[r["land"]] for r in raw]);YG=np.array([[r["population"],r["real_gdp"]] for r in raw]);YB=np.array([[r["co2"]] for r in raw])
scaleX=X*100;scaleYG=YG*np.array([10,0.01]);scaleYB=YB*0.001
ids=np.linspace(0,len(raw)-1,12,dtype=int);diff=[]
for i in ids:
    base=sbm_undesirable(X[i],YG[i],YB[i],X,YG,YB,False)
    scaled=sbm_undesirable(scaleX[i],scaleYG[i],scaleYB[i],scaleX,scaleYG,scaleYB,False)
    diff.append(abs(base-scaled))
checks["unit_invariance_max_abs_diff"]=max(diff)

# Reconcile stored annual summaries.
recon=[]
for s in d["summary"]:
    rr=[r for r in rows if r["year"]==s["year"]]
    oe=np.array([r["oe"] for r in rr]);pte=np.array([r["pte"] for r in rr]);se=np.array([r["se"] for r in rr])
    recon.append({"year":s["year"],"oe_mean_error":abs(oe.mean()-s["oe_mean"]),"cv_error":abs(oe.std(ddof=1)/oe.mean()-s["oe_cv_sample"]),"pte_mean_error":abs(pte.mean()-s["pte_mean"]),"se_mean_error":abs(se.mean()-s["se_mean"])})
checks["summary_reconciliation"]=recon

# Flag extreme scale-efficiency decompositions for manual review.
checks["se_below_0_5"]=[{"city":r["city"],"year":r["year"],"oe":r["oe"],"pte":r["pte"],"se":r["se"]} for r in rows if r["se"]<0.5]
print(json.dumps(checks,ensure_ascii=False,indent=2))

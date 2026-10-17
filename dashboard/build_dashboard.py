"""Build dashboard/resq_dashboard.html: a self-contained 'control room' replay (no Streamlit/server needed).
Run from the repo root on the VM:   python dashboard/build_dashboard.py
Then download the HTML and open it in any browser (screen-record it for the video)."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from model.config import TEST_CSV, EDGES_CSV
from model.data import load_csv
from action.predict import gnn_probs
from action.replay import replay, compare

RISK = 0.5
test, edges = load_csv(TEST_CSV), pd.read_csv(EDGES_CSV)
probs, thr, backend = gnn_probs(test)
base = replay(test, probs, edges, plan=False)
act = replay(test, probs, edges, plan=True, risk_threshold=RISK)
res = compare(base, act)

hk = [k for k, h in zip(act["key"], act["is_h"]) if h]                       # (hospital, item) columns
hosp = sorted({n for n, _ in hk}); items = sorted({i for _, i in hk})
piv = probs.pivot(index="hour", columns=["node_id", "item"], values="prob")[hk]
hours = list(range(act["start_hour"], act["start_hour"] + len(act["out"])))
risk = piv.reindex(hours).fillna(0).round(2).values.tolist()
bits = lambda o: ["".join("1" if x else "0" for x in row) for row in o[:, act["is_h"]]]
plans = [[int(r.hour), r.item, r._3, r.to, int(r.qty), float(r.eta_hours), float(r.risk)]
         for r in act["plans"].itertuples()]
data = dict(hours=hours, hosp=hosp, items=items, risk=risk, base=bits(base["out"]), act=bits(act["out"]), plans=plans,
            kpi={k: [int(v["No rerouting"]), int(v["ResQ-Stream"])] for k, v in res.iterrows()},
            backend=backend, ms=round(float(np.median(act["plan_ms"])), 1), thr=RISK)

HTML = open(os.path.join(os.path.dirname(__file__), "template.html")).read() if os.path.exists(
    os.path.join(os.path.dirname(__file__), "template.html")) else None
out = os.path.join(os.path.dirname(__file__), "resq_dashboard.html")
open(out, "w").write(HTML.replace("__DATA__", json.dumps(data)))
print("wrote", out, f"({os.path.getsize(out)//1024} KB) | risk scores from {backend}")

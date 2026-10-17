"""Closed-loop replay: 'what if ResQ-Stream had been acting during the recorded test simulation?'

The test CSV was generated with NO rerouting. We re-run its hours one by one. At each hour the planner sees the
current stock plus the model's risk scores, ships stock, and the shipments land after their transit time.
Everything else (demand, the simulator's own reorders, warehouse restocks) is replayed from the recording.
Baseline = same replay with planning switched off, so both runs share identical assumptions.

Known simplifications (say these honestly in the video):
  * when a donor gives stock away, the simulator's own reorder rule is not re-run for it
  * the planner does not know which specific lane a route_closure hit
"""
import math, time
import numpy as np, pandas as pd
from .planner import recommend_transfers


def _prep(df, probs):
    nodes = sorted(df.node_id.unique()); items = sorted(df.item.unique())
    key = [(n, i) for n in nodes for i in items]; kidx = {k: j for j, k in enumerate(key)}
    h0, H = int(df.hour.min()), int(df.hour.max()); T = H - h0 + 1
    A = {c: np.zeros((T, len(key))) for c in ["stock", "rate", "prob"]}
    ev = np.full((T, len(key)), "none", dtype=object)
    t = df.hour.values - h0; k = np.array([kidx[(a, b)] for a, b in zip(df.node_id, df.item)])
    A["stock"][t, k] = df.stock.values; A["rate"][t, k] = df.consumption_rate.values; ev[t, k] = df.event_flag.values
    tp = probs.hour.values - h0; kp = np.array([kidx[(a, b)] for a, b in zip(probs.node_id, probs.item)])
    A["prob"][tp, kp] = probs.prob.values
    ntype = df.drop_duplicates("node_id").set_index("node_id").node_type
    return key, [ntype[n] for n, _ in key], A, ev, h0, T, int(probs.hour.min()) - h0


def replay(df, probs, edges, plan=True, risk_threshold=0.5, **planner_kw):
    key, ktype, A, ev, h0, T, start = _prep(df, probs)
    K = len(key); is_h = np.array([x == "hospital" for x in ktype])
    cf = A["stock"][start].copy()
    rec = A["stock"]
    in_flight = []                              # (arrival_index, k, qty)
    out_hours = np.zeros((T, K), bool); unmet = np.zeros((T, K)); plans = []; plan_ms = []
    kmap = {k: j for j, k in enumerate(key)}

    for t in range(start, T):
        if t > start:                           # ---- advance one hour: deliveries -> demand
            arr = np.zeros(K)
            for a, j, q in [x for x in in_flight if x[0] == t]:
                arr[j] += q
            in_flight = [x for x in in_flight if x[0] > t]
            inflow = rec[t] - rec[t - 1] + A["rate"][t]
            inflow = np.where((rec[t] > 0) & (inflow > 5), inflow, 0.0)       # snap away int-rounding noise
            avail = cf + arr + np.where(is_h, inflow, rec[t] - rec[t - 1])
            demand = np.where(is_h, A["rate"][t], 0.0)
            unmet[t] = np.maximum(0, demand - np.maximum(avail, 0))
            cf = np.maximum(0, avail - demand)
        out_hours[t] = is_h & (cf < 1)
        if not plan:
            continue
        pend = np.zeros(K)
        for a, j, q in in_flight:
            pend[j] += q
        snap = pd.DataFrame({"node_id": [n for n, _ in key], "item": [i for _, i in key], "node_type": ktype,
                             "stock": cf, "consumption_rate": A["rate"][t], "event_flag": ev[t],
                             "prob": A["prob"][t], "pending_in": pend})
        t0 = time.perf_counter()
        p = recommend_transfers(snap, edges, risk_threshold=risk_threshold, **planner_kw)
        plan_ms.append((time.perf_counter() - t0) * 1000)
        for r in p.itertuples():
            cf[kmap[(r._2, r.item)]] -= r.qty                                  # leaves donor now
            in_flight.append((t + int(math.ceil(r.eta_hours)), kmap[(r.to, r.item)], r.qty))
            plans.append({"hour": t + h0, "item": r.item, "from": r._2, "to": r.to, "qty": r.qty,
                          "eta_hours": r.eta_hours, "hours_left": r.hours_left,
                          "margin_hours": r.margin_hours, "risk": r.risk})
    return dict(out=out_hours[start:], unmet=unmet[start:], plans=pd.DataFrame(plans), plan_ms=plan_ms,
                is_h=is_h, key=key, start_hour=start + h0)


def _episodes(o):                               # count separate stockout episodes per (node,item) column
    return int(((o[1:] & ~o[:-1]).sum()) + o[0].sum())


def compare(base, act):
    b, a = base["out"][:, base["is_h"]], act["out"][:, act["is_h"]]
    rows = {"stockout episodes": (_episodes(b), _episodes(a)),
            "stockout hospital-hours": (int(b.sum()), int(a.sum())),
            "unmet demand (units)": (round(float(base["unmet"].sum())), round(float(act["unmet"].sum())))}
    d = pd.DataFrame(rows, index=["No rerouting", "ResQ-Stream"]).T
    d["reduction %"] = ((d["No rerouting"] - d["ResQ-Stream"]) / d["No rerouting"].replace(0, np.nan) * 100).round(0)
    return d

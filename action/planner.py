"""ResQ-Stream ACTION stage: turn stockout risk scores into a ranked list of inventory transfers.

Uses only networkx + pandas + numpy (all installed on the LinuxONE VM).

Public function:  recommend_transfers(snap, edges, ...)
  snap  : one row per (node_id, item) at ONE hour, columns
          node_id, node_type, item, stock, consumption_rate, event_flag, prob (stockout risk 0..1)
          optional: pending_in (units already on the way), so we never double-ship
  edges : src, dst, transit_hrs  (data/edges.csv)
Returns a DataFrame ranked most-urgent first.
"""
import math
import networkx as nx
import pandas as pd

COLS = ["item", "from", "to", "qty", "eta_hours", "hours_left", "margin_hours", "risk"]


def recommend_transfers(snap, edges, risk_threshold=0.5, cover_hours=24.0, keep_hours=24.0,
                        dist_keep=300, lane_cap=400):
    """risk_threshold : nodes with prob >= this are 'at risk' and get stock
    cover_hours    : an at-risk hospital is topped up to this many hours of demand
    keep_hours     : a safe hospital donor always keeps this many hours of its own demand
    dist_keep      : a distributor keeps this many units (its own reorder point)
    lane_cap       : max units per lane per decision (edges.csv has no capacities)"""
    snap = snap.copy()
    if "pending_in" not in snap:
        snap["pending_in"] = 0.0
    snap["hours_left"] = (snap.stock / snap.consumption_rate.where(snap.consumption_rate > 0)).fillna(float("inf"))
    lanes = [(r.src, r.dst, float(r.transit_hrs)) for r in edges.itertuples()]
    out = []

    for item, g in snap.groupby("item"):
        g = g.set_index("node_id")
        need, spare = {}, {}
        for n, r in g.iterrows():
            if r.node_type == "hospital":
                if r.prob >= risk_threshold:
                    d = math.ceil(r.consumption_rate * cover_hours - r.stock - r.pending_in)
                    if d > 0:
                        need[n] = d
                elif r.event_flag == "none":                       # calm hospitals may lend their surplus
                    s = math.floor(r.stock - max(1.0, r.consumption_rate * keep_hours))
                    if s > 0:
                        spare[n] = s
            elif r.node_type == "distributor":
                s = math.floor(r.stock - dist_keep)
                if s > 0:
                    spare[n] = s
            else:                                                  # warehouse
                spare[n] = int(r.stock)
        if not need or not spare:
            continue

        G = nx.DiGraph()
        for n, s in spare.items():
            G.add_edge("SOURCE", n, capacity=s, weight=0)
        for n, d in need.items():                                  # riskier hospital = cheaper = served first
            G.add_edge(n, "SINK", capacity=d, weight=int((1 - g.prob[n]) * 1000))
        for a, b, tt in lanes:
            if a in spare and b in need and tt < g.hours_left[b]:  # skip lanes that would arrive too late
                G.add_edge(a, b, capacity=lane_cap, weight=int(round(tt * 10)))
        if "SOURCE" not in G or "SINK" not in G:
            continue
        F = nx.maximum_flow_value(G, "SOURCE", "SINK")
        if F == 0:
            continue
        G.nodes["SOURCE"]["demand"], G.nodes["SINK"]["demand"] = -F, F
        flow = nx.min_cost_flow(G)

        tt_of = {(a, b): t for a, b, t in lanes}
        for u, outs in flow.items():
            for v, q in outs.items():
                if q > 0 and u in spare and v in need:
                    hl = g.hours_left[v]
                    out.append({"item": item, "from": u, "to": v, "qty": int(q), "eta_hours": tt_of[(u, v)],
                                "hours_left": round(float(hl), 1), "margin_hours": round(float(hl) - tt_of[(u, v)], 1),
                                "risk": round(float(g.prob[v]), 3)})
    if not out:
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame(out).sort_values(["hours_left", "eta_hours"]).reset_index(drop=True)

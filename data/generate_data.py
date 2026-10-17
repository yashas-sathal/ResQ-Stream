"""Standalone data generator for ResQ-Stream (no other repo files needed, only numpy + pandas).

Creates a hospital supply network, simulates it hour by hour with random disruptions, and writes the 3 files the
model code reads:  telemetry_train.csv, telemetry_test.csv, edges.csv  (into this same data/ folder).

Design choices that matter for fair training:
  * The network (hospital demand, starting stock) is FIXED, so train and test are the same hospitals ...
  * ... but each file gets its own random disruption schedule and noise (different seed), so test is unseen.
  * Disruptions are spread over the whole period, so stockouts occur in every part of the timeline.
  * A surge at one hospital spills some extra demand onto its neighbours (so the network structure matters).
  * Only hospitals can have a stockout (label = stock hits zero within the next N hours).

Run from the repo root:   python data/generate_data.py
Options:                  --train-hours 1440 --test-hours 720 --stress 1.0 --cover 14 --n 6
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
ITEMS = ["epinephrine_1mg", "iv_saline_1l", "n95_mask", "oxygen_cylinder_d"]
HOSP = [f"HOSP-0{i}" for i in range(1, 7)]
WH = ["WH-North", "WH-South"]
DIST = ["DIST-Central", "DIST-East"]
NODES = WH + DIST + HOSP
KIND = {**{n: "warehouse" for n in WH}, **{n: "distributor" for n in DIST}, **{n: "hospital" for n in HOSP}}

# lanes: warehouse->distributor 12h, warehouse->hospital 24h, distributor->hospital 6h, hospital<->hospital (ring) 3h
LANES = [(w, d, 12.0) for w in WH for d in DIST] + [(w, h, 24.0) for w in WH for h in HOSP] \
      + [(d, h, 6.0) for d in DIST for h in HOSP] \
      + [(HOSP[i], HOSP[(i + 1) % len(HOSP)], 3.0) for i in range(len(HOSP))]
NEIGHBOURS = {h: [HOSP[(i - 1) % len(HOSP)], HOSP[(i + 1) % len(HOSP)]] for i, h in enumerate(HOSP)}


def make_network(stress):
    """Fixed hospital parameters (same for every run)."""
    r = np.random.default_rng(7)
    base = {(h, it): r.uniform(0.5, 3.0) * stress for h in HOSP for it in ITEMS}
    return base


def random_events(r, hours):
    ev = []
    for _ in range(max(1, hours // 45)):                                   # outbreak surges
        s = r.uniform(0, hours - 24)
        ev.append(("outbreak_surge", r.choice(HOSP), s, s + r.uniform(18, 60), r.uniform(2.5, 5.0)))
    for _ in range(max(1, hours // 80)):                                   # closed lanes into a hospital
        s = r.uniform(0, hours - 24)
        ev.append(("route_closure", f"{r.choice(WH + DIST)}>{r.choice(HOSP)}", s, s + r.uniform(18, 48), 0.0))
    for _ in range(max(1, hours // 80)):                                   # slow supplier
        s = r.uniform(0, hours - 24)
        ev.append(("supplier_delay", r.choice(DIST + WH), s, s + r.uniform(12, 48), r.uniform(6, 18)))
    return ev


def simulate(seed, hours, stress, N, cover=14.0):
    r = np.random.default_rng(seed)
    base = make_network(stress)
    events = random_events(r, hours)
    lane_h = {(a, b): t for a, b, t in LANES}
    stock = {}
    for h in HOSP:
        for it in ITEMS:
            stock[(h, it)] = base[(h, it)] * r.uniform(45, 75)
    for d in DIST:
        for it in ITEMS:
            stock[(d, it)] = float(r.integers(800, 1200))
    for w in WH:
        for it in ITEMS:
            stock[(w, it)] = 50_000.0
    pending, rows, prev_flag = [], [], {}

    for t in range(hours):
        surge, closed, delayed, flag = {}, set(), {}, {}
        for kind, tgt, s, e, v in events:
            if s <= t < e:
                if kind == "outbreak_surge":
                    surge[tgt] = max(surge.get(tgt, 1.0), v); flag[tgt] = "outbreak_surge"
                elif kind == "route_closure":
                    a, b = tgt.split(">"); closed.add((a, b)); flag.setdefault(b, "route_closure")
                else:
                    delayed[tgt] = max(delayed.get(tgt, 0.0), v); flag.setdefault(tgt, "supplier_delay")

        for p in list(pending):                                            # deliveries (wait if the lane is closed)
            if p["arr"] <= t and (p["src"], p["dst"]) not in closed:
                stock[(p["dst"], p["item"])] += p["qty"]; pending.remove(p)

        if t % 72 == 0:                                                    # warehouses are restocked
            for w in WH:
                for it in ITEMS:
                    stock[(w, it)] = 50_000.0

        for n in NODES:
            for it in ITEMS:
                rate = 0.0
                if KIND[n] == "hospital":
                    mult = surge.get(n, 1.0)
                    for nb in NEIGHBOURS[n]:                               # spillover from neighbours in surge
                        mult += 0.25 * (surge.get(nb, 1.0) - 1.0)
                    daily = 1 + 0.25 * np.sin(2 * np.pi * (t % 24) / 24)
                    rate = base[(n, it)] * mult * daily * r.lognormal(0, 0.12)
                    stock[(n, it)] = max(0.0, stock[(n, it)] - rate)
                    order_at, qty = base[(n, it)] * cover, base[(n, it)] * 48   # reorder when `cover` hours of stock remain
                elif KIND[n] == "distributor":
                    order_at, qty = 300.0, 500.0
                else:
                    order_at = None

                if order_at is not None and stock[(n, it)] < order_at and \
                        not any(p["dst"] == n and p["item"] == it for p in pending):
                    srcs = [(a, tt) for a, b, tt in LANES if b == n and KIND[a] != "hospital"
                            and (a, n) not in closed and stock[(a, it)] >= qty]
                    if srcs:
                        a, tt = min(srcs, key=lambda x: x[1] + delayed.get(x[0], 0.0))
                        stock[(a, it)] -= qty
                        pending.append(dict(src=a, dst=n, item=it, qty=qty, arr=t + tt + delayed.get(a, 0.0)))

                etas = [max(0.0, p["arr"] - t) for p in pending if p["dst"] == n and p["item"] == it]
                rows.append(dict(hour=t, node_id=n, node_type=KIND[n], item=it, stock=int(stock[(n, it)]),
                                 consumption_rate=round(rate, 3),
                                 inbound_eta_hrs=round(min(etas), 1) if etas else -1.0,
                                 event_flag=flag.get(n, "none"),
                                 stockout_now=int(KIND[n] == "hospital" and stock[(n, it)] < 1)))

    df = pd.DataFrame(rows).sort_values(["node_id", "item", "hour"]).reset_index(drop=True)
    g = df.groupby(["node_id", "item"])["stockout_now"]
    df["stockout_next_N"] = (g.transform(lambda s: s[::-1].rolling(N, min_periods=1).max()[::-1].shift(-1))
                              .fillna(0).astype(int))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-hours", type=int, default=24 * 60)
    ap.add_argument("--test-hours", type=int, default=24 * 30)
    ap.add_argument("--stress", type=float, default=1.0, help="hospital demand multiplier (higher = more stockouts)")
    ap.add_argument("--cover", type=float, default=14.0, help="hours of normal demand left when a hospital reorders (lower = more stockouts)")
    ap.add_argument("--n", type=int, default=6, help="label horizon in hours (must equal N_HOURS in model/config.py)")
    a = ap.parse_args()
    for name, seed, hours in [("train", 1, a.train_hours), ("test", 2, a.test_hours)]:
        df = simulate(seed, hours, a.stress, a.n, a.cover)
        df.to_csv(OUT / f"telemetry_{name}.csv", index=False)
        h = df[df.node_type == "hospital"]
        late = h[h.hour >= 0.8 * hours]["stockout_next_N"].mean()
        print(f"{name}: {len(df)} rows, {hours} h | hospital positive rate {h.stockout_next_N.mean():.3f} "
              f"(last 20% of time: {late:.3f})")
    pd.DataFrame(LANES, columns=["src", "dst", "transit_hrs"]).to_csv(OUT / "edges.csv", index=False)
    print("wrote telemetry_train.csv, telemetry_test.csv, edges.csv in", OUT)


if __name__ == "__main__":
    main()
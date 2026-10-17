"""Train the temporal GNN.   Run from the repo root:  python -m model.train_gnn"""
import json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score
from .config import *
from .data import *
from .gnn import TemporalGNN
from eval.metrics import best_threshold


def main(epochs=120, hidden=48, lr=2e-3, bs=32, patience=25):
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = load_csv(TRAIN_CSV); edges = pd.read_csv(EDGES_CSV)
    nodes, items = graph_spec(df)
    X, Y, M = build_tensors(df, nodes, items)
    A = build_adj(edges, nodes)
    Xw = torch.from_numpy(make_windows(X)); Yw = torch.from_numpy(Y[WINDOW - 1:]); Mw = torch.from_numpy(M[WINDOW - 1:])

    cut, vstart = split_hours(X.shape[0])
    tr_i = np.arange(0, cut - WINDOW + 1)                      # window i ends at hour i+W-1
    va_i = np.arange(vstart - WINDOW + 1, len(Xw))
    ytr, mtr = Yw[tr_i], Mw[tr_i]
    pos, neg = float((ytr * mtr).sum()), float(((1 - ytr) * mtr).sum())
    pw = torch.tensor(min(8.0, neg / max(pos, 1.0)))
    print(f"train windows {len(tr_i)}, val windows {len(va_i)}, positives {int(pos)}, pos_weight {pw:.1f}")

    model = TemporalGNN(A, X.shape[-1], len(items), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def val_scores():
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xw[va_i])).numpy()
        y, m = Yw[va_i].numpy(), Mw[va_i].numpy() > 0
        return y[m], p[m]

    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr_i); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss_e = F.binary_cross_entropy_with_logits(model(Xw[b]), Yw[b], pos_weight=pw, reduction="none")
            loss = (loss_e * Mw[b]).sum() / Mw[b].sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        y, p = val_scores()
        ap = average_precision_score(y, p) if y.sum() else 0.0
        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"epoch {ep:3d}  loss {tot:.3f}  val PR-AUC {ap:.3f}")
        if bad >= patience:
            print(f"early stop at epoch {ep}"); break

    model.load_state_dict(best_state)
    y, p = val_scores(); thr = best_threshold(y, p)
    print(f"best val PR-AUC {best_ap:.3f}, threshold {thr:.3f}")

    torch.save(model.state_dict(), ART_DIR / "gnn.pt")
    np.save(ART_DIR / "adj.npy", A)
    json.dump(dict(nodes=nodes, items=items, window=WINDOW, N=N_HOURS, hidden=hidden, in_f=int(X.shape[-1]),
                   threshold=thr,
                   feature_layout=f"per node: {len(items)} items x {PER_ITEM} "
                                  "[log_stock/10, log_rate/5, log_hours_left, eta/72, has_inbound, cover_gap/72] "
                                  "(items sorted) + 3 event flags + is_warehouse + hod_sin + hod_cos"),
              open(ART_DIR / "meta.json", "w"), indent=2)
    print("saved to", ART_DIR)


if __name__ == "__main__":
    main()
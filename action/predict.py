"""Get risk scores (hour,node_id,item,prob) from either the temporal GNN or the boosting baseline.
Run notebooks/scripts from the repo ROOT so `model` and `eval` import correctly."""
import json
import numpy as np, pandas as pd
from model.config import ART_DIR
from model.data import build_tensors, make_windows


def gnn_probs(df, use="auto"):
    """use = 'torch' (works on the LinuxONE VM), 'onnx' (needs onnxruntime), or 'auto' (torch first)."""
    if use in ("auto", "torch"):
        try:
            from model.gnn import load_model, predict_df
            m, meta = load_model()
            return predict_df(m, df, meta), meta["threshold"], "torch"
        except ImportError:
            if use == "torch":
                raise
    import onnxruntime as ort
    meta = json.load(open(ART_DIR / "meta.json"))
    sess = ort.InferenceSession(str(ART_DIR / "resq_gnn.onnx"), providers=["CPUExecutionProvider"])
    nodes, items, W = meta["nodes"], meta["items"], meta["window"]
    Xw = make_windows(build_tensors(df, nodes, items, with_labels=False), W).astype(np.float32)
    logits = np.concatenate([sess.run(None, {"x": Xw[i:i + 128]})[0] for i in range(0, len(Xw), 128)])
    P = 1 / (1 + np.exp(-logits)); h0 = int(df["hour"].min()); Tw, Nn, I = P.shape
    out = pd.DataFrame({"hour": np.repeat(h0 + W - 1 + np.arange(Tw), Nn * I),
                        "node_id": np.tile(np.repeat(nodes, I), Tw),
                        "item": np.tile(np.array(items * Nn), Tw), "prob": P.reshape(-1)})
    return out, meta["threshold"], "onnx"


def boosting_probs(train_df, test_df):
    from model.baselines import fit_xgb, xgb_predict, HAVE_XGB
    m, thr = fit_xgb(train_df)
    return xgb_predict(m, test_df), thr, ("xgboost" if HAVE_XGB else "sklearn-boosting")

"""Baselines: (1) exponential smoothing, (2) XGBoost (falls back to sklearn HistGradientBoosting)."""
import numpy as np
from .config import *
from .data import tabular_features, TAB_FEATURES, split_hours
from eval.metrics import best_threshold

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False


# ---------- 1. exponential smoothing ----------
def ewm_predict(df):
    """Smooth the consumption rate (EWMA), project hours of stock left, turn it into a risk score.
    Higher score = closer to running out. It ignores the network and inbound shipments."""
    d = tabular_features(df)
    d["prob"] = N_HOURS / (N_HOURS + d["hours_left_ewm"])
    return d[["hour", "node_id", "item", "prob"]]


def ewm_threshold(train_df):
    d = tabular_features(train_df)
    d["prob"] = N_HOURS / (N_HOURS + d["hours_left_ewm"])
    cut, vstart = split_hours(int(d["hour"].max()) + 1)
    v = d[(d["hour"] >= vstart) & (d["stockout_now"] == 0)]
    return best_threshold(v["stockout_next_N"].values, v["prob"].values)


# ---------- 2. XGBoost ----------
def fit_xgb(train_df):
    d = tabular_features(train_df)
    cut, vstart = split_hours(int(d["hour"].max()) + 1)
    d = d[d["stockout_now"] == 0]
    tr, va = d[d["hour"] < cut], d[d["hour"] >= vstart]
    ytr, yva = tr["stockout_next_N"].values, va["stockout_next_N"].values
    spw = max(1.0, (ytr == 0).sum() / max(1, ytr.sum()))
    if HAVE_XGB:
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                          early_stopping_rounds=30, random_state=SEED, n_jobs=2)
        m.fit(tr[TAB_FEATURES], ytr, eval_set=[(va[TAB_FEATURES], yva)], verbose=False)
    else:
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, class_weight="balanced",
                                           random_state=SEED)
        m.fit(tr[TAB_FEATURES], ytr)
    thr = best_threshold(yva, m.predict_proba(va[TAB_FEATURES])[:, 1])
    return m, thr


def xgb_predict(m, df):
    d = tabular_features(df)
    d["prob"] = m.predict_proba(d[TAB_FEATURES])[:, 1]
    return d[["hour", "node_id", "item", "prob"]]
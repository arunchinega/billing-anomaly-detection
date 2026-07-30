"""Model statistics vs ground-truth labels: ROC-AUC, PR-AUC, F1, confusion,
score distributions, forecaster regression quality. Persisted to DB for the
Model Stats dashboard page.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    mean_absolute_error, r2_score,
)

# score column, direction handled by using |value| where needed
SCORERS = {
    "zscore":     lambda d: d["z_score"].abs().fillna(0),
    "iforest":    lambda d: d["if_score"].fillna(0),
    "forecaster": lambda d: d["resid_pct"].abs().fillna(0),
}
FLAGS = {"rules": "rule_flag", "zscore": "z_flag",
         "iforest": "if_flag", "forecaster": "fc_flag", "combined": "flagged"}


def classification_stats(d: pd.DataFrame):
    y = d["is_anomaly"].values
    summary, curves = [], []

    for name, fn in SCORERS.items():
        s = fn(d).values
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        # downsample curve points for storage
        step_r = max(1, len(fpr) // 200)
        step_p = max(1, len(prec) // 200)
        curves += [{"model": name, "curve": "roc", "x": x, "y": yy}
                   for x, yy in zip(fpr[::step_r], tpr[::step_r])]
        curves += [{"model": name, "curve": "pr", "x": x, "y": yy}
                   for x, yy in zip(rec[::step_p], prec[::step_p])]
        summary.append({
            "model": name,
            "roc_auc": roc_auc_score(y, s),
            "pr_auc": average_precision_score(y, s),
        })
    auc_df = pd.DataFrame(summary)

    rows = []
    for name, col in FLAGS.items():
        yh = d[col].values
        tn, fp, fn_, tp = confusion_matrix(y, yh).ravel()
        rows.append({
            "layer": name,
            "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
        })
    layer_df = pd.DataFrame(rows)
    return auc_df, layer_df, pd.DataFrame(curves)


def forecaster_stats(d: pd.DataFrame) -> dict:
    m = d["kwh_pred"].notna() & (d["is_anomaly"] == 0)   # quality on clean bills
    y, yh = d.loc[m, "kwh_usage"], d.loc[m, "kwh_pred"]
    return {
        "n_scored": int(m.sum()),
        "mae_kwh": float(mean_absolute_error(y, yh)),
        "mape_pct": float((np.abs((y - yh) / y.clip(lower=1))).mean() * 100),
        "r2": float(r2_score(y, yh)),
        "resid_pct_p95_clean": float(d.loc[m, "resid_pct"].abs().quantile(0.95)),
    }

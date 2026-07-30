"""Layers 2-4: robust z-score, Isolation Forest, XGBoost residual forecaster.
Plus score fusion into a severity queue."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

from .features import FEATURE_COLS

Z_THRESHOLD = 3.5
RESID_THRESHOLD = 0.50   # |actual-pred|/pred
IF_CONTAMINATION = 0.04


def zscore_layer(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["z_score"] = d["acct_z"].fillna(0)
    d["z_flag"] = (d["z_score"].abs() > Z_THRESHOLD).astype(int)
    return d


def isolation_forest_layer(d: pd.DataFrame, random_state: int = 42):
    d = d.copy()
    X = d[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    model = IsolationForest(
        n_estimators=300, contamination=IF_CONTAMINATION, random_state=random_state
    ).fit(X)
    # decision_function: lower = more anomalous -> invert to 0..1 anomaly score
    raw = -model.decision_function(X)
    d["if_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    d["if_flag"] = (model.predict(X) == -1).astype(int)
    return d, model


def forecaster_layer(d: pd.DataFrame, random_state: int = 42):
    """Predict expected kWh; anomaly signal = relative residual."""
    d = d.copy()
    feats = ["kwh_lag1", "kwh_lag12", "kwh_roll12_med", "seasonal_index", "month_num"]
    X_all = d[feats].copy()
    state_dummies = pd.get_dummies(d["state"], prefix="st")
    X_all = pd.concat([X_all, state_dummies], axis=1)
    y = d["kwh_usage"]

    # train only on rows not already condemned by rules (reduce label noise)
    trainable = (d["rule_flag"] == 0) & d["kwh_lag1"].notna()
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=random_state,
        n_jobs=-1,
    ).fit(X_all[trainable], y[trainable])

    d["kwh_pred"] = model.predict(X_all)
    d.loc[d["kwh_lag1"].isna(), "kwh_pred"] = np.nan  # cold-start months
    d["resid_pct"] = (d["kwh_usage"] - d["kwh_pred"]) / d["kwh_pred"].clip(lower=1)
    d["fc_flag"] = (d["resid_pct"].abs() > RESID_THRESHOLD).fillna(False).astype(int)
    return d, model


def fuse_scores(d: pd.DataFrame) -> pd.DataFrame:
    """Severity: rule hit = HIGH. Else 2+ statistical layers = HIGH, 1 = MEDIUM."""
    d = d.copy()
    d["stat_votes"] = d[["z_flag", "if_flag", "fc_flag"]].sum(axis=1)
    d["severity"] = np.select(
        [d["rule_flag"] == 1, d["stat_votes"] >= 2, d["stat_votes"] == 1],
        ["HIGH", "HIGH", "MEDIUM"],
        default="NONE",
    )
    d["flagged"] = (d["severity"] != "NONE").astype(int)

    def explain(r):
        parts = []
        if r.rule_flag:
            parts.append(f"Rule: {r.rule_reason}")
        if r.z_flag:
            parts.append(f"Account z-score {r.z_score:+.1f} vs own seasonal baseline")
        if r.if_flag:
            parts.append(f"IsolationForest score {r.if_score:.2f} (multivariate outlier)")
        if r.fc_flag:
            parts.append(f"Usage {r.resid_pct:+.0%} vs forecast ({r.kwh_pred:,.0f} kWh expected)")
        return " | ".join(parts)

    d["explanation"] = [explain(r) for r in d.itertuples()]
    return d

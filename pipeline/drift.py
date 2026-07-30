"""Drift management: PSI + KS per feature, score drift, retrain recommendation.

Reference window = first 12 months (model 'training era').
Current window   = any later month(s) under monitoring.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.10, 0.25  # industry-standard bands

# Deseasonalized / season-free features: compare vs full reference window
DRIFT_FEATURES = ["acct_z", "peer_z", "kwh_deseason", "implied_rate", "resid_pct"]
# Seasonal features (incl. if_score, which inherits seasonality from its
# month/ratio inputs): compare vs SAME calendar month in reference
SEASONAL_FEATURES = ["kwh_usage", "usage_per_day", "if_score"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return np.nan
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(d: pd.DataFrame, ref_months: int = 12) -> pd.DataFrame:
    """Seasonality-aware drift:
    - deseasonalized features -> PSI vs full reference window
    - raw seasonal features   -> PSI vs same calendar month of reference year
      (prevents seasonality masquerading as drift)"""
    d = d.copy()
    d["_cal_month"] = d["bill_month"].astype(str).str[-2:]
    months = sorted(d["bill_month"].astype(str).unique())
    ref = d[d["bill_month"].astype(str).isin(months[:ref_months])]
    rows = []
    for m in months[ref_months:]:
        cur = d[d["bill_month"].astype(str) == m]
        cal = m[-2:]
        for f in DRIFT_FEATURES + SEASONAL_FEATURES:
            if f not in d.columns:
                continue
            ref_pool = ref[ref["_cal_month"] == cal] if f in SEASONAL_FEATURES else ref
            rv, cv = ref_pool[f].to_numpy(dtype=float), cur[f].to_numpy(dtype=float)
            p = psi(rv, cv)
            ks_p = ks_2samp(rv[~np.isnan(rv)], cv[~np.isnan(cv)]).pvalue if len(cv) > 50 else np.nan
            rows.append({"month": str(m), "feature": f, "psi": p, "ks_pvalue": ks_p,
                         "comparison": "same-month-YoY" if f in SEASONAL_FEATURES else "full-window"})
    rep = pd.DataFrame(rows)
    rep["status"] = np.select(
        [rep["psi"] >= PSI_ALERT, rep["psi"] >= PSI_WARN],
        ["ALERT", "WARN"], default="OK",
    )
    return rep


def retrain_recommendation(rep: pd.DataFrame) -> dict:
    if rep.empty:
        return {"recommendation": "NO_DATA", "detail": "Not enough monitored months."}
    latest = rep[rep["month"] == rep["month"].max()]
    n_alert = int((latest["status"] == "ALERT").sum())
    n_warn = int((latest["status"] == "WARN").sum())
    if n_alert >= 2:
        rec = "RETRAIN_NOW"
        detail = f"{n_alert} features in ALERT (PSI>={PSI_ALERT}) in latest month."
    elif n_alert == 1 or n_warn >= 3:
        rec = "INVESTIGATE"
        detail = f"{n_alert} ALERT / {n_warn} WARN features in latest month."
    else:
        rec = "STABLE"
        detail = "All monitored features within PSI bands."
    return {"recommendation": rec, "detail": detail,
            "latest_month": str(rep['month'].max())}

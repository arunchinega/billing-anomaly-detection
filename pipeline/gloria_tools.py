"""Gloria's tools — parameterized, read-only queries over billing.db.
The LLM picks tools + arguments; it never writes SQL. All aggregation happens
in the database so only small result tables reach the model.
"""
import pandas as pd

from . import db

MAX_ROWS = 25  # never hand the LLM more than this


def _df(sql: str, params: dict | None = None) -> pd.DataFrame:
    return db.query(sql, params).head(MAX_ROWS)


def summary_stats() -> pd.DataFrame:
    """Overall KPIs: bills processed, anomalies flagged, precision, recall."""
    kv = db.load_kv("eval_overall")
    return pd.DataFrame([kv])


def top_anomalies(state: str = "", severity: str = "HIGH", n: int = 10) -> pd.DataFrame:
    """Top flagged bills ranked by absolute deviation of billed vs expected amount.
    state: optional 2-letter US state filter. severity: HIGH, MEDIUM or ALL."""
    n = min(int(n), MAX_ROWS)
    sql = """
        SELECT account_id, bill_month, state, kwh_usage, billed_amount, severity,
               ROUND(ABS(billed_amount - expected_amount), 2) AS deviation_usd,
               explanation
        FROM bills_scored WHERE severity != 'NONE'
    """
    params: dict = {}
    if severity and severity.upper() != "ALL":
        sql += " AND severity = :sev"; params["sev"] = severity.upper()
    if state:
        sql += " AND state = :st"; params["st"] = state.upper()
    sql += " ORDER BY deviation_usd DESC LIMIT :n"; params["n"] = n
    return _df(sql, params)


def account_history(account_id: str) -> pd.DataFrame:
    """All bills for one account with flags and explanations."""
    return _df("""
        SELECT bill_month, kwh_usage, billed_amount, severity, explanation
        FROM bills_scored WHERE account_id = :a ORDER BY bill_month
    """, {"a": account_id.strip().upper()})


def anomalies_by_state() -> pd.DataFrame:
    """Flag counts and total $ deviation per state."""
    return _df("""
        SELECT state, COUNT(*) AS flags,
               ROUND(SUM(ABS(billed_amount - expected_amount)), 2) AS total_deviation_usd
        FROM bills_scored WHERE flagged = 1
        GROUP BY state ORDER BY total_deviation_usd DESC
    """)


def anomaly_type_breakdown() -> pd.DataFrame:
    """Ground-truth anomaly counts by type with combined detection recall."""
    return _df("SELECT * FROM eval_per_type")


def drift_status() -> pd.DataFrame:
    """Latest drift PSI per feature plus the retrain recommendation."""
    rec = db.load_kv("drift_recommendation")
    latest = db.query("""
        SELECT feature, ROUND(psi, 4) AS psi, status FROM drift_history
        WHERE month = (SELECT MAX(month) FROM drift_history) ORDER BY psi DESC
    """)
    latest.attrs["recommendation"] = rec
    header = pd.DataFrame([{"feature": f"RECOMMENDATION: {rec.get('recommendation')}",
                            "psi": None, "status": rec.get("detail", "")}])
    return pd.concat([header, latest], ignore_index=True)


def forecast_summary(series: str = "kwh") -> pd.DataFrame:
    """Next-6-month SARIMA forecast. series: 'kwh' or 'revenue'."""
    s = "revenue" if "rev" in series.lower() else "kwh"
    return _df("""
        SELECT month, ROUND(pred, 0) AS forecast, ROUND(lo, 0) AS low_80,
               ROUND(hi, 0) AS high_80
        FROM forecast_forward WHERE series = :s AND model = 'SARIMA' ORDER BY month
    """, {"s": s})


def model_metrics() -> pd.DataFrame:
    """Detection model quality: ROC-AUC / PR-AUC per model and per-layer P/R/F1."""
    auc = db.load_df("model_auc").round(3)
    layers = db.load_df("layer_metrics")[["layer", "precision", "recall", "f1"]].round(3)
    auc.columns = ["layer", "precision", "recall"]  # align widths for concat display
    auc["f1"] = None
    return pd.concat([layers, auc], ignore_index=True)


def review_queue_status() -> pd.DataFrame:
    """Open review items, verdicts recorded, confirm rate."""
    open_items = db.query("""
        SELECT COUNT(*) AS n FROM bills_scored b
        WHERE b.severity != 'NONE' AND NOT EXISTS (
            SELECT 1 FROM review_verdicts v
            WHERE v.account_id = b.account_id AND v.bill_month = b.bill_month)
    """).n[0]
    verdicts = db.load_verdicts()
    confirm = f"{(verdicts['verdict'] == 'CONFIRMED').mean():.0%}" if len(verdicts) else "no verdicts yet"
    return pd.DataFrame([{"open_items": int(open_items),
                          "verdicts_recorded": len(verdicts),
                          "confirm_rate": confirm}])


TOOLS = {
    "summary_stats": summary_stats,
    "top_anomalies": top_anomalies,
    "account_history": account_history,
    "anomalies_by_state": anomalies_by_state,
    "anomaly_type_breakdown": anomaly_type_breakdown,
    "drift_status": drift_status,
    "forecast_summary": forecast_summary,
    "model_metrics": model_metrics,
    "review_queue_status": review_queue_status,
}

_PARAMS = {
    "top_anomalies": {
        "properties": {
            "state": {"type": "string", "description": "2-letter US state, optional"},
            "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "ALL"]},
            "n": {"type": "integer"}},
        "required": []},
    "account_history": {
        "properties": {"account_id": {"type": "string",
                                      "description": "e.g. ACC00001"}},
        "required": ["account_id"]},
    "forecast_summary": {
        "properties": {"series": {"type": "string", "enum": ["kwh", "revenue"]}},
        "required": []},
}
_EMPTY = {"properties": {}, "required": []}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": name,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {"type": "object", **_PARAMS.get(name, _EMPTY)},
    }}
    for name, fn in TOOLS.items()
]

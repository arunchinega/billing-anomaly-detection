"""End-to-end run: features -> rules -> z-score -> IF -> forecaster -> fusion -> eval.
Writes scored dataset + evaluation + drift report to artifacts/.
Run:  python -m pipeline.run_pipeline
"""
import json
from pathlib import Path

import pandas as pd

from .features import engineer_features
from .rules import apply_rules
from .detectors import zscore_layer, isolation_forest_layer, forecaster_layer, fuse_scores
from .drift import drift_report, retrain_recommendation
from . import db
from .model_stats import classification_stats, forecaster_stats
from .forecasting import forecast_all

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "res_billing_anomaly_dataset.csv"
ART = ROOT / "artifacts"


def evaluate(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in d[d["is_anomaly"] == 1].groupby("anomaly_type"):
        rows.append({
            "anomaly_type": t, "n": len(g),
            "recall_rules": g["rule_flag"].mean(),
            "recall_zscore": g["z_flag"].mean(),
            "recall_iforest": g["if_flag"].mean(),
            "recall_forecaster": g["fc_flag"].mean(),
            "recall_combined": g["flagged"].mean(),
        })
    per_type = pd.DataFrame(rows).sort_values("n", ascending=False)

    flagged = d[d["flagged"] == 1]
    high = d[d["severity"] == "HIGH"]
    overall = {
        "total_bills": len(d),
        "true_anomalies": int(d["is_anomaly"].sum()),
        "flagged": len(flagged),
        "precision_all_flags": float(flagged["is_anomaly"].mean()),
        "precision_high": float(high["is_anomaly"].mean()),
        "recall_overall": float(d.loc[d["is_anomaly"] == 1, "flagged"].mean()),
        "recall_high_only": float(d.loc[d["is_anomaly"] == 1, "severity"].eq("HIGH").mean()),
    }
    return per_type, overall


def main():
    ART.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)

    d = engineer_features(df)
    d = apply_rules(d)
    d = zscore_layer(d)
    d, if_model = isolation_forest_layer(d)
    d, fc_model = forecaster_layer(d)
    d = fuse_scores(d)

    per_type, overall = evaluate(d)

    db.init_db()
    db.save_df(d, "bills_scored")
    db.save_df(per_type, "eval_per_type")
    db.save_kv("eval_overall", overall)

    rep = drift_report(d)
    db.save_df(rep, "drift_history")
    db.save_kv("drift_recommendation", retrain_recommendation(rep))

    auc_df, layer_df, curves = classification_stats(d)
    db.save_df(auc_df, "model_auc")
    db.save_df(layer_df, "layer_metrics")
    db.save_df(curves, "model_curves")
    db.save_kv("forecaster_stats", forecaster_stats(d))

    db.register_model("isolation_forest", "v1", {"contamination": 0.04})
    db.register_model("xgb_forecaster", "v1",
                      {"recall_overall": overall["recall_overall"],
                       "precision_high": overall["precision_high"]})
    print("Running forecasting bake-off (SARIMA / Prophet / LSTM)...")
    bt, fw, board, hist = forecast_all(d)
    db.save_df(bt, "forecast_backtest")
    db.save_df(fw, "forecast_forward")
    db.save_df(board, "forecast_leaderboard")
    db.save_df(hist, "forecast_history")
    print(board.round(2).to_string(index=False))

    print(f"\nWritten to SQLite: {db.DB_PATH}")

    print("=== Overall ===")
    print(json.dumps(overall, indent=2))
    print("\n=== Per anomaly type (recall by layer) ===")
    print(per_type.round(3).to_string(index=False))
    print("\n=== Drift ===")
    print(json.dumps(retrain_recommendation(rep), indent=2))


if __name__ == "__main__":
    main()

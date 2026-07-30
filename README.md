# Billing Anomaly Detection — Enterprise Prototype

4-layer detection stack for US residential electricity billing:
Rules → Robust z-score → Isolation Forest → XGBoost residual forecaster,
with score fusion, reviewer feedback loop, and drift management.

## Structure
```
data/       res_billing_anomaly_dataset.csv  (48,127 bills, labeled)
pipeline/   features.py  rules.py  detectors.py  drift.py  run_pipeline.py
artifacts/  scored_bills.csv, eval_*, drift_* (generated)
app.py      Streamlit UI (5 pages)
```

## Run
```bash
pip install -r requirements.txt
python -m pipeline.run_pipeline      # scores all bills, writes artifacts/
streamlit run app.py
```

## App pages
- **Dashboard** — KPIs, flags by month/severity/state, layer attribution
- **Anomaly Queue** — review workflow; verdicts saved to artifacts/review_labels.csv
  (these become training labels for the Phase-2 supervised classifier)
- **Feature Explorer** — feature distributions (normal vs anomaly) + account drill-down
- **Model Performance** — recall per anomaly type per layer, vs ground truth
- **Drift Monitor** — PSI/KS per feature vs 12-month reference window, retrain policy

## Results on synthetic data
Recall ~100%, precision 99.5% on HIGH severity. Note: synthetic anomalies are
deliberately separable — present as capability demonstration, not benchmark.

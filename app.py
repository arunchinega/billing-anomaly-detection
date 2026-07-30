"""Billing Anomaly Detection — Enterprise Prototype (Streamlit + Plotly + SQLite).

Run:
    python -m pipeline.run_pipeline     # once, builds artifacts/billing.db
    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import db
from pipeline.drift import PSI_WARN, PSI_ALERT

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="RIA Advisory — Billing Anomaly Detection",
                   layout="wide", page_icon="assets/ria_logo.png")

# RIA palette: navy / light blue / orange
RIA_NAVY, RIA_BLUE, RIA_MID, RIA_ORANGE = "#1F3864", "#4DA3DB", "#2E5FA3", "#F79433"
SEV_COLORS = {"HIGH": RIA_NAVY, "MEDIUM": RIA_ORANGE}
LAYER_COLORS = {"rules": RIA_NAVY, "zscore": RIA_BLUE, "iforest": RIA_MID,
                "forecaster": RIA_ORANGE, "combined": "#B03A2E"}
MODEL_COLORS = {"SARIMA": RIA_NAVY, "Prophet": RIA_BLUE, "LSTM": RIA_ORANGE}

st.logo("assets/ria_logo.png", size="large")

# ---------------- access gate: RIA / ArcOne emails only ----------------
ALLOWED_DOMAINS = ("@riaadvisory.com", "@arcone.com")
ALLOWED_EMAILS = {"arunchinegatr@gmail.com", "arunkiranrao@gmail.com"}


def display_name(email: str) -> str:
    stem = email.split("@")[0].split(".")[0].split("_")[0]
    stem = "".join(ch for ch in stem if ch.isalpha()) or "there"
    return stem.capitalize()


if "user_email" not in st.session_state:
    st.session_state.user_email = None

if not st.session_state.user_email:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image("assets/ria_logo.png", width=220)
        st.title("Billing Anomaly Detection")
        st.caption("RIA AI Centre of Excellence — restricted access")
        email = st.text_input("Work email", placeholder="you@riaadvisory.com").strip().lower()
        if st.button("Sign in", type="primary"):
            if email.endswith(ALLOWED_DOMAINS) or email in ALLOWED_EMAILS:
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Access is limited to RIA Advisory and ArcOne team members.")
        st.caption("Demo-grade access control. Production deployments use SSO/OAuth.")
    st.stop()

USER_EMAIL = st.session_state.user_email
USER_NAME = display_name(USER_EMAIL)


@st.cache_data
def load():
    return (db.load_df("bills_scored"), db.load_df("eval_per_type"),
            db.load_kv("eval_overall"), db.load_df("drift_history"),
            db.load_kv("drift_recommendation"), db.load_df("model_auc"),
            db.load_df("layer_metrics"), db.load_df("model_curves"),
            db.load_kv("forecaster_stats"), db.load_df("forecast_backtest"),
            db.load_df("forecast_forward"), db.load_df("forecast_leaderboard"),
            db.load_df("forecast_history"))


try:
    (d, per_type, overall, drift, drift_rec,
     model_auc, layer_metrics, model_curves, fc_stats,
     fc_backtest, fc_forward, fc_board, fc_hist) = load()
except Exception:
    st.error("Database not initialized. Run `python -m pipeline.run_pipeline` first.")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Executive Overview", "📊 Model Dashboard", "💬 Gloria",
     "🚨 Anomaly Queue", "🧪 Feature Explorer", "📐 Model Stats",
     "📈 Detection Performance", "🔮 Forecasting", "🌊 Drift Monitor"],
)
st.sidebar.caption(f"Signed in: {USER_EMAIL}")
if st.sidebar.button("Sign out"):
    st.session_state.user_email = None
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Rules → Robust z-score → Isolation Forest → XGBoost forecaster")
st.sidebar.caption(f"DB: SQLite · {db.DB_PATH.name}")
st.sidebar.markdown(
    "<div style='color:#1F3864;font-size:0.8em;font-weight:600'>"
    "RECOGNIZE <span style='color:#F79433'>&gt;</span> INNOVATE "
    "<span style='color:#F79433'>&gt;</span> ACCELERATE</div>"
    "<div style='font-size:0.75em;color:#5A6B7B'>RIA AI Centre of Excellence</div>",
    unsafe_allow_html=True)

# ============================================================== Dashboard
if page == "📊 Model Dashboard":
    st.title("📊 Model Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bills processed", f"{overall['total_bills']:,}")
    c2.metric("Flagged", f"{overall['flagged']:,}")
    c3.metric("Precision (HIGH)", f"{overall['precision_high']:.1%}")
    c4.metric("Recall (overall)", f"{overall['recall_overall']:.1%}")
    c5.metric("Drift status", drift_rec["recommendation"])

    st.subheader("Flags by month and severity")
    tl = (d[d["severity"] != "NONE"].groupby(["bill_month", "severity"])
          .size().reset_index(name="flags"))
    fig = px.bar(tl, x="bill_month", y="flags", color="severity",
                 color_discrete_map=SEV_COLORS)
    fig.update_layout(height=340, margin=dict(t=10, b=10), legend_title=None)
    st.plotly_chart(fig, use_container_width=True)

    cL, cR = st.columns(2)
    with cL:
        st.subheader("Detection layer attribution")
        attribution = pd.DataFrame({
            "layer": ["rules", "zscore", "iforest", "forecaster"],
            "flags": [int(d["rule_flag"].sum()), int(d["z_flag"].sum()),
                      int(d["if_flag"].sum()), int(d["fc_flag"].sum())],
        })
        fig = px.bar(attribution, x="layer", y="flags", color="layer",
                     color_discrete_map=LAYER_COLORS, text="flags")
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Layers overlap by design — agreement drives severity voting.")
    with cR:
        st.subheader("Flags by state")
        by_state = (d[d["flagged"] == 1].groupby(["state", "severity"])
                    .size().reset_index(name="flags"))
        fig = px.bar(by_state, x="state", y="flags", color="severity",
                     color_discrete_map=SEV_COLORS)
        fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    cP, cQ = st.columns(2)
    with cP:
        st.subheader("Anomaly type mix (ground truth)")
        mix = (d[d["is_anomaly"] == 1].groupby("anomaly_type").size()
               .reset_index(name="n").sort_values("n", ascending=False))
        fig = px.pie(mix, names="anomaly_type", values="n", hole=0.45)
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cQ:
        st.subheader("Layer metrics (precision / recall / F1)")
        lm = layer_metrics.copy()
        fig = go.Figure()
        for m in ["precision", "recall", "f1"]:
            fig.add_trace(go.Bar(name=m, x=lm["layer"], y=lm[m]))
        fig.update_layout(barmode="group", height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Model statistics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    auc_map = model_auc.set_index("model")
    m1.metric("IF ROC-AUC", f"{auc_map.loc['iforest','roc_auc']:.3f}")
    m2.metric("IF PR-AUC", f"{auc_map.loc['iforest','pr_auc']:.3f}")
    m3.metric("Z-score AUC", f"{auc_map.loc['zscore','roc_auc']:.3f}")
    m4.metric("Forecaster AUC", f"{auc_map.loc['forecaster','roc_auc']:.3f}")
    m5.metric("Combined F1", f"{layer_metrics.set_index('layer').loc['combined','f1']:.3f}")
    m6.metric("Forecaster R²", f"{fc_stats['r2']:.3f}")

    cR1, cR2 = st.columns(2)
    with cR1:
        st.markdown("**ROC curves**")
        roc = model_curves[model_curves["curve"] == "roc"]
        fig = px.line(roc, x="x", y="y", color="model",
                      labels={"x": "FPR", "y": "TPR"},
                      color_discrete_map=LAYER_COLORS)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
        fig.update_layout(height=330, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="dash_roc")
    with cR2:
        st.markdown("**Precision-Recall curves**")
        pr = model_curves[model_curves["curve"] == "pr"]
        fig = px.line(pr, x="x", y="y", color="model",
                      labels={"x": "Recall", "y": "Precision"},
                      color_discrete_map=LAYER_COLORS)
        fig.update_layout(height=330, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="dash_pr")
    st.caption("Full detail (confusion matrices, per-layer thresholds, forecaster "
               "regression stats) on the 📐 Model Stats page.")

# ============================================================== Executive Overview
elif page == "🏠 Executive Overview":
    latest = str(d["bill_month"].max())
    st.title("⚡ Billing Anomaly Detection")
    st.caption(f"Latest cycle: {latest} · Residential portfolio · 8 states")

    fl = d[d["flagged"] == 1].copy()
    fl["deviation_usd"] = (fl["billed_amount"] - fl["expected_amount"]).abs()
    this_m = int((fl["bill_month"].astype(str) == latest).sum())
    months_sorted = sorted(d["bill_month"].astype(str).unique())
    prev_m = int((fl["bill_month"].astype(str) == months_sorted[-2]).sum())
    delta_pct = (this_m - prev_m) / prev_m * 100 if prev_m else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total bills", f"{overall['total_bills']:,}")
    c2.metric("Anomalies", f"{overall['flagged']:,}",
              f"{overall['flagged']/overall['total_bills']:.1%} of bills",
              delta_color="off")
    c3.metric("$ impact", f"${fl['deviation_usd'].sum():,.0f}")
    c4.metric("Accounts affected", f"{fl['account_id'].nunique():,}")
    c5.metric("Vs last cycle", f"{delta_pct:+.0f}%",
              delta_color="inverse" if delta_pct > 0 else "normal")

    cL, cR = st.columns(2)
    with cL:
        st.subheader("Anomalies by root cause")
        rc = (fl.assign(cause=fl["rule_reason"].where(fl["rule_reason"] != "",
                                                      "Usage pattern (statistical)"))
              .groupby("cause").size().reset_index(name="n")
              .sort_values("n"))
        rc["cause"] = (rc["cause"].str.replace("R1_METER_ROLLBACK", "Meter rollback")
                       .str.replace("R2_DUPLICATE_BILL", "Duplicate bill")
                       .str.replace("R3_CALC_MISMATCH", "Calculation mismatch")
                       .str.replace("R4_ZERO_USAGE", "Zero usage / tamper")
                       .str.replace("R5_RATE_OUT_OF_BAND", "Rate error"))
        fig = px.bar(rc, x="n", y="cause", orientation="h",
                     color_discrete_sequence=[RIA_BLUE], text="n")
        fig.update_layout(height=360, margin=dict(t=10, b=10),
                          yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        st.subheader("$ impact by state")
        bs = (fl.groupby("state")["deviation_usd"].sum().reset_index()
              .sort_values("deviation_usd", ascending=False))
        fig = px.pie(bs, names="state", values="deviation_usd", hole=0.5,
                     color_discrete_sequence=[RIA_NAVY, RIA_BLUE, RIA_MID,
                                              RIA_ORANGE, "#7FB3E0", "#163050",
                                              "#F9B36B", "#9DC6E8"])
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged bills")
    search = st.text_input("🔎 Search account", placeholder="e.g. ACC00836")
    show = fl.copy()
    if search.strip():
        show = show[show["account_id"].str.contains(search.strip().upper())]
    show["cause"] = show["rule_reason"].where(show["rule_reason"] != "",
                                              "Usage pattern")
    st.dataframe(
        show.sort_values("deviation_usd", ascending=False)
        [["account_id", "state", "bill_month", "cause", "deviation_usd",
          "severity"]].head(15).rename(columns={"deviation_usd": "deviation ($)"}),
        use_container_width=True, hide_index=True)
    st.caption("💬 Want to dig in? Ask **Gloria** (sidebar) — e.g. "
               "\"explain account ACC00836\" or \"top 5 by dollar impact\".")

# ============================================================== Queue
elif page == "🚨 Anomaly Queue":
    st.title("🚨 Anomaly Review Queue")
    st.caption("Verdicts persist to `review_verdicts` and become Phase-2 classifier labels.")

    cA, cB = st.columns([1, 1])
    sev = cA.selectbox("Severity", ["HIGH", "MEDIUM", "ALL"])
    sort_by = cB.selectbox("Sort by", ["financial impact ($)", "IF score"])

    q = d[d["severity"] != "NONE"].copy()
    if sev != "ALL":
        q = q[q["severity"] == sev]
    q["impact_usd"] = (q["billed_amount"] - q["expected_amount"].where(
        q["rule_flag"] == 1, q["kwh_pred"] * (q["rate_per_kwh"] + 0.012)
        + q["fixed_charge"])).abs().round(2)
    q = q.sort_values("impact_usd" if sort_by.startswith("financial") else "if_score",
                      ascending=False)

    labels = db.load_verdicts()
    done = set(zip(labels["account_id"], labels["bill_month"])) if len(labels) else set()
    q_open = q[~q.apply(lambda r: (r.account_id, r.bill_month) in done, axis=1)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open items", len(q_open))
    c2.metric("Reviewed", len(labels))
    c3.metric("Confirm rate",
              f"{(labels['verdict'] == 'CONFIRMED').mean():.0%}" if len(labels) else "—")
    c4.metric("Value in open queue", f"${q_open['impact_usd'].sum():,.0f}")

    st.dataframe(
        q_open[["account_id", "bill_month", "state", "kwh_usage", "billed_amount",
                "impact_usd", "severity", "explanation"]].head(200),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Record a verdict")
    if len(q_open):
        pick = st.selectbox("Item", q_open.head(200).apply(
            lambda r: f"{r.account_id} | {r.bill_month} | ${r.impact_usd:,.0f} | {r.explanation[:70]}",
            axis=1))
        acc, mon = [s.strip() for s in pick.split("|")[:2]]
        verdict = st.radio("Verdict", ["CONFIRMED", "FALSE_ALARM"], horizontal=True)
        reason = st.text_input("Reason code / note", "")
        if st.button("Save verdict", type="primary"):
            db.save_verdict(acc, mon, verdict, reason)
            st.success("Saved to review_verdicts.")
            st.rerun()
    else:
        st.info("Queue is clear for this severity.")

# ============================================================== Features
elif page == "🧪 Feature Explorer":
    st.title("🧪 Feature Engineering Explorer")
    feat = st.selectbox("Feature", ["kwh_usage", "usage_per_day", "mom_ratio",
                                    "yoy_ratio", "peer_z", "acct_z", "kwh_deseason",
                                    "implied_rate", "calc_gap", "resid_pct", "if_score"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normal vs anomalous distribution")
        plot_d = d[[feat, "is_anomaly"]].dropna()
        lo, hi = plot_d[feat].quantile([0.005, 0.995])
        plot_d = plot_d[plot_d[feat].between(lo, hi)]
        plot_d["label"] = plot_d["is_anomaly"].map({0: "normal", 1: "anomaly"})
        fig = px.histogram(plot_d, x=feat, color="label", barmode="overlay",
                           histnorm="probability", nbins=60,
                           color_discrete_map={"normal": "#1f77b4", "anomaly": "#d62728"})
        fig.update_layout(height=380, margin=dict(t=10, b=10), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Separation between the two distributions = feature earns its place.")
    with c2:
        st.subheader("Account drill-down: actual vs forecast")
        acc = st.selectbox("Account", sorted(d["account_id"].unique())[:500])
        a = d[d["account_id"] == acc].sort_values("bill_month")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a["bill_month"], y=a["kwh_usage"],
                                 name="actual kWh", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=a["bill_month"], y=a["kwh_pred"],
                                 name="forecast kWh", mode="lines",
                                 line=dict(dash="dot")))
        flagged = a[a["flagged"] == 1]
        fig.add_trace(go.Scatter(x=flagged["bill_month"], y=flagged["kwh_usage"],
                                 name="flagged", mode="markers",
                                 marker=dict(color="#d62728", size=12, symbol="x")))
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if len(flagged):
            st.dataframe(flagged[["bill_month", "severity", "explanation"]],
                         use_container_width=True, hide_index=True)

# ============================================================== Model Stats
elif page == "📐 Model Stats":
    st.title("📐 Model Statistics")
    st.caption("Scored against synthetic ground-truth labels — evaluation only. "
               "Scores are in-sample; expect lower numbers on unseen/real data.")

    st.subheader("Ranking quality (score-based models)")
    c1, c2, c3 = st.columns(3)
    for col, (_, r) in zip((c1, c2, c3), model_auc.iterrows()):
        col.metric(f"{r['model']} ROC-AUC", f"{r['roc_auc']:.3f}",
                   f"PR-AUC {r['pr_auc']:.3f}")

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**ROC curves**")
        roc = model_curves[model_curves["curve"] == "roc"]
        fig = px.line(roc, x="x", y="y", color="model",
                      labels={"x": "False positive rate", "y": "True positive rate"},
                      color_discrete_map=LAYER_COLORS)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        st.markdown("**Precision-Recall curves** (the honest view at 4% prevalence)")
        pr = model_curves[model_curves["curve"] == "pr"]
        fig = px.line(pr, x="x", y="y", color="model",
                      labels={"x": "Recall", "y": "Precision"},
                      color_discrete_map=LAYER_COLORS)
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Layer metrics at operating thresholds")
    lm = layer_metrics.copy()
    st.dataframe(lm.round(3), use_container_width=True, hide_index=True)

    fig = go.Figure()
    for m in ["precision", "recall", "f1"]:
        fig.add_trace(go.Bar(name=m, x=lm["layer"], y=lm[m]))
    fig.update_layout(barmode="group", height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix by layer")
    pick = st.selectbox("Layer", lm["layer"].tolist(), index=len(lm) - 1)
    r = lm[lm["layer"] == pick].iloc[0]
    cm = [[int(r.tn), int(r.fp)], [int(r.fn), int(r.tp)]]
    fig = px.imshow(cm, text_auto=True,
                    x=["pred normal", "pred anomaly"],
                    y=["true normal", "true anomaly"],
                    color_continuous_scale="Blues")
    fig.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("XGBoost forecaster — regression quality (clean bills)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{fc_stats['mae_kwh']:.0f} kWh")
    c2.metric("MAPE", f"{fc_stats['mape_pct']:.1f}%")
    c3.metric("R²", f"{fc_stats['r2']:.3f}")
    c4.metric("|residual| p95", f"{fc_stats['resid_pct_p95_clean']:.0%}")
    st.caption(f"Scored on {fc_stats['n_scored']:,} clean bills. The p95 residual on clean "
               "bills justifies the 50% residual alert threshold.")

    st.info("Note: z-score/IF/forecaster never saw the anomaly labels — AUC/F1 here "
            "measure how well unsupervised scores rank true anomalies. There is no "
            "trained classifier yet; that arrives in Phase 2 from reviewer verdicts.")

# ============================================================== Performance
elif page == "📈 Detection Performance":
    st.title("📈 Detection Performance by Anomaly Type")
    st.subheader("Recall per anomaly type, per layer")
    heat = per_type.set_index("anomaly_type")[
        ["recall_rules", "recall_zscore", "recall_iforest",
         "recall_forecaster", "recall_combined"]]
    heat.columns = ["rules", "zscore", "iforest", "forecaster", "combined"]
    fig = px.imshow(heat.round(2), text_auto=True, aspect="auto",
                    color_continuous_scale="RdYlGn", zmin=0, zmax=1)
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green diagonal-ish structure = each layer owns its anomaly family; "
               "'combined' column = the fused system.")

    st.dataframe(per_type.round(3), use_container_width=True, hide_index=True)
    st.subheader("Overall")
    st.json(overall)

# ============================================================== Forecasting
elif page == "🔮 Forecasting":
    st.title("🔮 Aggregate Forecasting — SARIMA vs Prophet vs LSTM")
    series = st.radio("Series", ["kwh", "revenue"], horizontal=True,
                      format_func=lambda s: "Total kWh" if s == "kwh" else "Total revenue ($)")

    st.subheader("Backtest leaderboard (train 18 mo → test 6 mo)")
    board = fc_board[fc_board["series"] == series].sort_values("mape_pct")
    cols = st.columns(3)
    for col, (_, r) in zip(cols, board.iterrows()):
        col.metric(r["model"], f"{r['mape_pct']:.2f}% MAPE", f"MAE {r['mae']:,.0f}")

    hist = fc_hist[["month", series]].rename(columns={series: "value"})
    bt = fc_backtest[fc_backtest["series"] == series]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["value"], name="actual",
                             mode="lines+markers", line=dict(color="#5A6B7B", width=3)))
    for m, g in bt.groupby("model"):
        fig.add_trace(go.Scatter(x=g["month"], y=g["pred"], name=f"{m} (backtest)",
                                 mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[m], dash="dot")))
    fig.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines = each model's prediction of the 6 held-out months.")

    st.subheader("Forward forecast — next 6 months")
    fw = fc_forward[fc_forward["series"] == series]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["value"], name="history",
                             mode="lines", line=dict(color="#5A6B7B", width=3)))
    for m, g in fw.groupby("model"):
        if g["lo"].notna().any():
            fig.add_trace(go.Scatter(
                x=list(g["month"]) + list(g["month"])[::-1],
                y=list(g["hi"]) + list(g["lo"])[::-1],
                fill="toself", fillcolor=MODEL_COLORS[m], opacity=0.12,
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["month"], y=g["pred"], name=m,
                                 mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[m])))
    fig.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded bands = 80% intervals (SARIMA, Prophet). LSTM: point forecast only.")

    st.info("Reading: SARIMA wins — short, clean, strongly seasonal monthly series is "
            "exactly its home turf. LSTM is data-starved at 24 monthly points (deep "
            "models need hundreds+); its real role is AMI smart-meter interval data. "
            "Prophet sits between — robust, but its flexible trend underfits 18 points. "
            "Per-account expected usage remains the XGBoost global model's job.")

# ============================================================== Gloria
elif page == "💬 Gloria":
    from pipeline.gloria_agent import ask_gloria, ollama_available, DEFAULT_MODEL
    from pipeline import db as _db

    st.title("💬 Gloria")
    ok, available_models = ollama_available()
    known = [m for m in ["llama3.2:3b", "llama3.1:8b"]
             if any(m in (am or "") for am in available_models)] or ["llama3.2:3b", "llama3.1:8b"]
    model = st.sidebar.selectbox("Gloria's model", known, index=0)

    if not ok:
        st.error("Ollama is not running. Install from ollama.com, then in a terminal: "
                 "`ollama pull llama3.2:3b` — and keep the Ollama app running.")
        st.stop()

    if "gloria_history" not in st.session_state:
        st.session_state.gloria_history = []
        st.session_state.gloria_greeted = False

    if not st.session_state.gloria_greeted:
        greeting = (f"Hi {USER_NAME}! 😊 I'm **Gloria**, your billing assistant. "
                    "I can pull top anomalies, account histories, forecasts, drift "
                    "status, model metrics and the review queue — all straight from "
                    "the billing database. What would you like to know?")
        st.session_state.gloria_history.append(
            {"role": "assistant", "content": greeting, "meta": None})
        st.session_state.gloria_greeted = True

    # render history
    for i, m in enumerate(st.session_state.gloria_history):
        with st.chat_message(m["role"]):
            st.markdown(m["content"].replace("$", "\\$"))
            meta = m.get("meta")
            if meta:
                badge = ("✅ Grounded" if meta["grounded"] else "💬 Conversational")
                st.caption(f"{badge} · ⚡ {meta['total_s']:.1f}s total · "
                           f"tools {meta['tool_s']:.2f}s · LLM {meta['llm_s']:.1f}s · "
                           f"{meta['tokens']} tokens · {meta['model']} · "
                           f"{len(meta['tools'])} tool call(s)")
                for name, args, df_r in meta["tools"]:
                    with st.expander(f"📄 source: {name}({args})"):
                        st.dataframe(df_r, use_container_width=True, hide_index=True)
                c1, c2, _ = st.columns([1, 1, 10])
                if c1.button("👍", key=f"up{i}"):
                    _db.save_gloria_feedback(USER_EMAIL, "", m["content"], "up",
                                             meta["model"], meta["total_s"])
                    st.toast("Thanks — logged!")
                if c2.button("👎", key=f"dn{i}"):
                    _db.save_gloria_feedback(USER_EMAIL, "", m["content"], "down",
                                             meta["model"], meta["total_s"])
                    st.toast("Logged — Gloria will learn from this.")

    if prompt := st.chat_input("Ask Gloria about the billing data..."):
        st.session_state.gloria_history.append(
            {"role": "user", "content": prompt, "meta": None})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc, meta = "", None
            llm_history = [{"role": h["role"], "content": h["content"]}
                           for h in st.session_state.gloria_history[:-1]
                           if h["role"] in ("user", "assistant")]
            def esc(t):
                return t.replace("$", "\\$")
            try:
                for kind, payload in ask_gloria(prompt, llm_history, model=model):
                    if kind == "status":
                        placeholder.markdown(f"*{payload}*")
                    elif kind == "delta":
                        acc += payload
                        placeholder.markdown(esc(acc) + "▌")
                    else:
                        meta = payload
                placeholder.markdown(esc(acc) if acc else
                                     "I couldn't produce an answer — please retry.")
            except Exception as e:
                acc = f"⚠️ Gloria hit an error talking to the model: {e}"
                placeholder.markdown(acc)
        st.session_state.gloria_history.append(
            {"role": "assistant", "content": acc, "meta": meta})
        st.rerun()

# ============================================================== Drift
elif page == "🌊 Drift Monitor":
    st.title("🌊 Drift Management")
    rec_icon = {"STABLE": "✅", "INVESTIGATE": "🟡", "RETRAIN_NOW": "🔴", "NO_DATA": "⚪"}
    st.metric("Recommendation",
              f"{rec_icon.get(drift_rec['recommendation'], '')} {drift_rec['recommendation']}")
    st.caption(f"{drift_rec['detail']} (latest month: {drift_rec.get('latest_month', '—')})")

    st.subheader("PSI by feature over monitored months")
    st.caption("Seasonality-aware: raw usage features compared same-month-YoY; "
               "deseasonalized features vs full 12-month reference. "
               f"Bands: <{PSI_WARN} OK · {PSI_WARN}–{PSI_ALERT} WARN · >{PSI_ALERT} ALERT")
    fig = px.line(drift, x="month", y="psi", color="feature", markers=True,
                  hover_data=["comparison"])
    fig.add_hrect(y0=PSI_WARN, y1=PSI_ALERT, fillcolor="orange", opacity=0.12,
                  line_width=0)
    fig.add_hrect(y0=PSI_ALERT, y1=max(PSI_ALERT * 1.4, float(drift["psi"].max() or 0) * 1.05),
                  fillcolor="red", opacity=0.10, line_width=0)
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest month detail")
    latest = drift[drift["month"] == drift["month"].max()].sort_values("psi", ascending=False)
    st.dataframe(latest.round(4), use_container_width=True, hide_index=True)

    st.markdown(f"""
**Retrain policy:** 2+ ALERT → `RETRAIN_NOW` · 1 ALERT / 3+ WARN → `INVESTIGATE` · else `STABLE`.
Input drift with stable scores → population shift; score drift alone → model decay.
""")

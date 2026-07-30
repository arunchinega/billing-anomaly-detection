"""Aggregate forecasting bake-off: SARIMA vs Prophet vs LSTM.

Series: total monthly kWh and total monthly billed revenue (clean bills only,
so injected anomalies don't distort the aggregate).
Backtest: train months 1-18, test 19-24. Forward: refit on all 24, predict +6.
Results persisted to DB: forecast_backtest, forecast_forward, forecast_leaderboard.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HOLDOUT = 6
HORIZON = 6


def build_series(d: pd.DataFrame) -> pd.DataFrame:
    clean = d[d["is_anomaly"] == 0]
    agg = clean.groupby(clean["bill_month"].astype(str)).agg(
        kwh=("kwh_usage", "sum"), revenue=("billed_amount", "sum")).reset_index()
    agg["ds"] = pd.to_datetime(agg["bill_month"] + "-01")
    return agg.sort_values("ds").reset_index(drop=True)


# --------------------------------------------------------------- models
def fit_sarima(y_train: pd.Series, steps: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # short-series-safe spec: seasonal AR instead of seasonal differencing
    m = SARIMAX(y_train, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)
    fc = m.get_forecast(steps)
    ci = fc.conf_int(alpha=0.2)
    return fc.predicted_mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values


def fit_prophet(train: pd.DataFrame, ycol: str, steps: int):
    from prophet import Prophet
    df = train.rename(columns={ycol: "y"})[["ds", "y"]]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.8)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    p = m.predict(future).tail(steps)
    return p["yhat"].values, p["yhat_lower"].values, p["yhat_upper"].values


def fit_lstm(y_train: np.ndarray, steps: int, lookback: int = 12, seed: int = 42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    lo, hi = y_train.min(), y_train.max()
    ys = (y_train - lo) / (hi - lo + 1e-9)
    X, Y = [], []
    for i in range(len(ys) - lookback):
        X.append(ys[i:i + lookback]); Y.append(ys[i + lookback])
    X = np.array(X)[..., None]; Y = np.array(Y)

    model = Sequential([LSTM(16, input_shape=(lookback, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=300, verbose=0)

    window = list(ys[-lookback:]); preds = []
    for _ in range(steps):
        p = float(model.predict(np.array(window[-lookback:])[None, :, None], verbose=0).ravel()[0])
        preds.append(p); window.append(p)
    preds = np.array(preds) * (hi - lo) + lo
    return preds, None, None  # no principled intervals


MODELS = {"SARIMA": "sarima", "Prophet": "prophet", "LSTM": "lstm"}


def _run_model(name, train_df, ycol, steps):
    if name == "SARIMA":
        return fit_sarima(train_df[ycol], steps)
    if name == "Prophet":
        return fit_prophet(train_df, ycol, steps)
    return fit_lstm(train_df[ycol].values, steps)


def forecast_all(d: pd.DataFrame):
    agg = build_series(d)
    backtest_rows, forward_rows, board_rows = [], [], []

    for ycol in ["kwh", "revenue"]:
        train, test = agg.iloc[:-HOLDOUT], agg.iloc[-HOLDOUT:]
        for name in MODELS:
            try:
                pred, lo, hi = _run_model(name, train, ycol, HOLDOUT)
                mape = float(np.mean(np.abs((test[ycol].values - pred) / test[ycol].values)) * 100)
                mae = float(np.mean(np.abs(test[ycol].values - pred)))
                board_rows.append({"series": ycol, "model": name,
                                   "mape_pct": mape, "mae": mae})
                for i, ds in enumerate(test["ds"]):
                    backtest_rows.append({
                        "series": ycol, "model": name, "month": str(ds.date())[:7],
                        "actual": float(test[ycol].iloc[i]), "pred": float(pred[i]),
                        "lo": float(lo[i]) if lo is not None else None,
                        "hi": float(hi[i]) if hi is not None else None})
            except Exception as e:
                board_rows.append({"series": ycol, "model": name,
                                   "mape_pct": np.nan, "mae": np.nan})
                print(f"[forecast] {name}/{ycol} failed: {e}")

        future_ds = pd.date_range(agg["ds"].max() + pd.offsets.MonthBegin(),
                                  periods=HORIZON, freq="MS")
        for name in MODELS:
            try:
                pred, lo, hi = _run_model(name, agg, ycol, HORIZON)
                for i, ds in enumerate(future_ds):
                    forward_rows.append({
                        "series": ycol, "model": name, "month": str(ds.date())[:7],
                        "pred": float(pred[i]),
                        "lo": float(lo[i]) if lo is not None else None,
                        "hi": float(hi[i]) if hi is not None else None})
            except Exception as e:
                print(f"[forecast] forward {name}/{ycol} failed: {e}")

    hist = agg[["ds", "kwh", "revenue"]].copy()
    hist["month"] = hist["ds"].astype(str).str[:7]
    return (pd.DataFrame(backtest_rows), pd.DataFrame(forward_rows),
            pd.DataFrame(board_rows), hist[["month", "kwh", "revenue"]])

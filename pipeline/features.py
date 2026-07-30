"""Feature engineering for residential billing anomaly detection."""
import numpy as np
import pandas as pd

RIDER = 0.012  # $/kWh rider/tax adder used in bill calculation


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contextual features. Input: raw bills. Output: bills + features."""
    d = df.copy()
    d["bill_month"] = pd.PeriodIndex(d["bill_month"], freq="M")
    d = d.sort_values(["account_id", "bill_month"]).reset_index(drop=True)
    d["month_num"] = d["bill_month"].dt.month

    # --- billing integrity features ---
    d["expected_amount"] = d["fixed_charge"] + d["kwh_usage"] * (d["rate_per_kwh"] + RIDER)
    d["calc_gap"] = d["billed_amount"] - d["expected_amount"]
    d["read_delta"] = d["meter_curr_read"] - d["meter_prev_read"]
    d["usage_per_day"] = d["kwh_usage"] / d["billing_days"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["implied_rate"] = np.where(
            d["kwh_usage"] > 0,
            (d["billed_amount"] - d["fixed_charge"]) / d["kwh_usage"] - RIDER,
            np.nan,
        )

    # --- temporal features (per account) ---
    g = d.groupby("account_id")["kwh_usage"]
    d["kwh_lag1"] = g.shift(1)
    d["kwh_lag12"] = g.shift(12)
    d["kwh_roll12_med"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=3).median())
    d["mom_ratio"] = d["kwh_usage"] / d["kwh_lag1"].replace(0, np.nan)
    d["yoy_ratio"] = d["kwh_usage"] / d["kwh_lag12"].replace(0, np.nan)

    # --- peer (state, calendar month) robust z-score ---
    grp = d.groupby(["state", "bill_month"])["kwh_usage"]
    peer_med = grp.transform("median")
    peer_mad = grp.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    d["peer_z"] = (d["kwh_usage"] - peer_med) / (1.4826 * peer_mad)

    # --- deseasonalized account-level robust z ---
    # seasonal index from peer cohort so a single account's anomaly can't distort it
    seas = (
        d.groupby(["state", "month_num"])["kwh_usage"].median()
        / d.groupby("state")["kwh_usage"].median()
    )
    d["seasonal_index"] = d.set_index(["state", "month_num"]).index.map(seas.to_dict())
    d["kwh_deseason"] = d["kwh_usage"] / d["seasonal_index"]
    ag = d.groupby("account_id")["kwh_deseason"]
    acct_med = ag.transform("median")
    acct_mad = ag.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    d["acct_z"] = (d["kwh_deseason"] - acct_med) / (1.4826 * acct_mad)

    d["is_estimated"] = (d["read_type"] == "ESTIMATED").astype(int)
    return d


FEATURE_COLS = [
    "usage_per_day", "mom_ratio", "yoy_ratio", "peer_z", "acct_z",
    "calc_gap", "implied_rate", "is_estimated", "month_num",
]

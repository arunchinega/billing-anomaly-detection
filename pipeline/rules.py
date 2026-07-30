"""Layer 1 — deterministic rules engine. Each rule returns a reason code."""
import numpy as np
import pandas as pd

# Tariff band: base 0.145 with +/-5% account variance -> pad slightly
RATE_MIN, RATE_MAX = 0.145 * 0.95 * 0.98, 0.145 * 1.05 * 1.02
CALC_TOLERANCE = 1.0  # dollars


def apply_rules(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    reasons = pd.Series([""] * len(d), index=d.index)

    r_rollback = d["meter_curr_read"] < d["meter_prev_read"]
    r_calc = d["calc_gap"].abs() > CALC_TOLERANCE
    r_dup = d.duplicated(subset=["account_id", "bill_month"], keep=False)
    r_zero = d["kwh_usage"] <= 0
    r_rate = ~d["rate_per_kwh"].between(RATE_MIN, RATE_MAX)

    for mask, code in [
        (r_rollback, "R1_METER_ROLLBACK"),
        (r_dup, "R2_DUPLICATE_BILL"),
        (r_calc & ~r_rollback, "R3_CALC_MISMATCH"),
        (r_zero & ~r_rollback, "R4_ZERO_USAGE"),
        (r_rate, "R5_RATE_OUT_OF_BAND"),
    ]:
        reasons = np.where(mask, np.where(reasons == "", code, reasons + ";" + code), reasons)

    d["rule_reason"] = reasons
    d["rule_flag"] = (d["rule_reason"] != "").astype(int)
    return d

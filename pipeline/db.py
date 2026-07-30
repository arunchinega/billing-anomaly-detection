"""Database layer — SQLite for prototype, swap DB_URL to Postgres for pilot.

Tables:
    bills_scored      pipeline output (features + flags + severity + explanation)
    review_verdicts   reviewer feedback loop -> future classifier labels
    drift_history     PSI/KS per feature per monitored month
    eval_per_type     recall per anomaly type per layer (synthetic eval)
    model_registry    model version, trained_at, metrics json
    kv_store          small json blobs (overall eval, drift recommendation)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "artifacts" / "billing.db"
DB_URL = f"sqlite:///{DB_PATH}"          # -> "postgresql+psycopg2://user:pw@host/db" later

_engine = None


def engine():
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(exist_ok=True)
        _engine = create_engine(DB_URL)
    return _engine


DDL = """
CREATE TABLE IF NOT EXISTS review_verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    bill_month TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('CONFIRMED','FALSE_ALARM')),
    reason TEXT,
    reviewer TEXT DEFAULT 'demo_user',
    reviewed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS gloria_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT,
    question TEXT,
    answer TEXT,
    rating TEXT CHECK (rating IN ('up','down')),
    model TEXT,
    latency_s REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv_store (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def init_db():
    with engine().begin() as c:
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                c.execute(text(stmt))


# ---------- writes ----------
def save_df(df: pd.DataFrame, table: str):
    df = df.copy()
    for col in df.columns:            # Periods -> str for SQLite
        if str(df[col].dtype).startswith("period"):
            df[col] = df[col].astype(str)
    df.to_sql(table, engine(), if_exists="replace", index=False)


def save_kv(key: str, obj: dict):
    now = datetime.now(timezone.utc).isoformat()
    with engine().begin() as c:
        c.execute(text(
            "INSERT INTO kv_store (k, v, updated_at) VALUES (:k,:v,:t) "
            "ON CONFLICT(k) DO UPDATE SET v=:v, updated_at=:t"),
            {"k": key, "v": json.dumps(obj), "t": now})


def register_model(name: str, version: str, metrics: dict):
    with engine().begin() as c:
        c.execute(text(
            "INSERT INTO model_registry (model_name, version, trained_at, metrics_json) "
            "VALUES (:n,:v,:t,:m)"),
            {"n": name, "v": version,
             "t": datetime.now(timezone.utc).isoformat(), "m": json.dumps(metrics)})


def save_verdict(account_id: str, bill_month: str, verdict: str,
                 reason: str = "", reviewer: str = "demo_user"):
    with engine().begin() as c:
        c.execute(text(
            "INSERT INTO review_verdicts (account_id, bill_month, verdict, reason, "
            "reviewer, reviewed_at) VALUES (:a,:b,:v,:r,:u,:t)"),
            {"a": account_id, "b": bill_month, "v": verdict, "r": reason,
             "u": reviewer, "t": datetime.now(timezone.utc).isoformat()})


def save_gloria_feedback(email, question, answer, rating, model, latency_s):
    with engine().begin() as c:
        c.execute(text(
            "INSERT INTO gloria_feedback (user_email, question, answer, rating, "
            "model, latency_s, created_at) VALUES (:e,:q,:a,:r,:m,:l,:t)"),
            {"e": email, "q": question[:500], "a": answer[:2000], "r": rating,
             "m": model, "l": latency_s,
             "t": datetime.now(timezone.utc).isoformat()})


# ---------- reads ----------
def load_df(table: str) -> pd.DataFrame:
    return pd.read_sql_table(table, engine())


def load_kv(key: str) -> dict:
    with engine().connect() as c:
        row = c.execute(text("SELECT v FROM kv_store WHERE k=:k"), {"k": key}).fetchone()
    return json.loads(row[0]) if row else {}


def load_verdicts() -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM review_verdicts ORDER BY reviewed_at", engine())


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Read-only helper — this is also the tool the future chatbot agent calls."""
    return pd.read_sql_query(text(sql), engine(), params=params or {})

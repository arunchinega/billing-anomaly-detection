"""Gloria — grounded billing assistant.

ReAct loop over Ollama (Llama models): the LLM selects tools, tools run exact
SQL, the LLM narrates results. Guardrails: answer only from tool results,
refuse out-of-scope, low temperature, capped rows, read-only DB.
"""
import inspect
import time

import pandas as pd

from .gloria_tools import TOOLS, TOOL_SPECS

DEFAULT_MODEL = "llama3.2:3b"
MAX_TOOL_ROUNDS = 3
KEEP_ALIVE = "30m"

SYSTEM_PROMPT = """You are Gloria, RIA Advisory's billing assistant. Rules:
1. State facts about billing data ONLY from tool results. Copy numbers exactly \
as returned. Result tables are pre-sorted: row 1 is the top/largest.
2. NEVER reproduce a full table in your answer - the user already sees the raw \
table separately. Summarize: mention only the 1-3 rows that answer the question.
3. If tools cannot answer, say so plainly. Do not estimate or fill gaps.
4. Scope: billing anomalies, accounts, forecasts, drift, model metrics, review \
queue. Politely decline anything else in one sentence.
5. Be concise: under 80 words. When comparing values, double-check which is larger."""

# fast-path intents: skip the LLM entirely for instant demo answers
FAST_PATHS = {
    "kpi": ("summary_stats", "Here are the headline KPIs:"),
    "summary stats": ("summary_stats", "Here are the headline KPIs:"),
    "queue status": ("review_queue_status", "Current review queue status:"),
    "drift status": ("drift_status", "Current drift picture:"),
}


def _run_tool(name: str, args: dict):
    fn = TOOLS.get(name)
    if fn is None:
        return pd.DataFrame([{"error": f"unknown tool {name}"}])
    accepted = set(inspect.signature(fn).parameters)
    kwargs = {k: v for k, v in (args or {}).items() if k in accepted and v not in ("", None)}
    try:
        return fn(**kwargs)
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])


def ask_gloria(question: str, history: list, model: str = DEFAULT_MODEL):
    """Generator yielding ("delta", text) chunks, then ("meta", dict).
    meta: latency breakdown, tool calls, source tables, grounded flag."""
    import ollama

    t0 = time.perf_counter()
    tool_time = 0.0
    tool_calls_made = []   # (name, args, dataframe)

    q = "".join(ch for ch in question.lower() if ch.isalnum() or ch.isspace()).strip()
    for key, (tool, preamble) in FAST_PATHS.items():
        if key in q and len(q) < 60:
            ts = time.perf_counter()
            result = _run_tool(tool, {})
            tool_time = time.perf_counter() - ts
            yield ("delta", preamble)
            yield ("meta", {
                "total_s": time.perf_counter() - t0, "tool_s": tool_time,
                "llm_s": 0.0, "model": "fast-path (no LLM)",
                "tools": [(tool, {}, result)], "grounded": True, "tokens": 0})
            return

    messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + history[-4:]
                + [{"role": "user", "content": question}])

    yield ("status", "🔍 Gloria is checking that for you — please hold on…")
    llm_time = 0.0
    for _ in range(MAX_TOOL_ROUNDS):
        ts = time.perf_counter()
        resp = ollama.chat(model=model, messages=messages, tools=TOOL_SPECS,
                           options={"temperature": 0.1, "num_predict": 250},
                           keep_alive=KEEP_ALIVE)
        llm_time += time.perf_counter() - ts
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append(msg)
        for c in calls:
            name = c["function"]["name"]
            args = c["function"].get("arguments") or {}
            ts = time.perf_counter()
            result = _run_tool(name, args)
            tool_time += time.perf_counter() - ts
            tool_calls_made.append((name, args, result))
            messages.append({"role": "tool",
                             "content": result.to_csv(index=False)})
        yield ("status", "📊 Got the data — writing your answer…")

    # final answer, streamed; guard against raw tool-call JSON leaking as text
    ts = time.perf_counter()
    stream = ollama.chat(model=model, messages=messages,
                         options={"temperature": 0.1, "num_predict": 250},
                         keep_alive=KEEP_ALIVE, stream=True)
    tokens, buffer, leaked = 0, "", False
    for chunk in stream:
        delta = chunk["message"]["content"]
        if not delta:
            continue
        buffer += delta
        if tokens == 0 and buffer.lstrip().startswith("{"):
            leaked = True  # model emitted a tool call as text; suppress
        tokens += 1
        if not leaked:
            yield ("delta", delta)
    if leaked:
        yield ("delta", "I've pulled the data — please see the source table "
                        "below for the details. (Tip: ask me a slightly more "
                        "specific question for a narrated answer.)")
    llm_time += time.perf_counter() - ts

    yield ("meta", {
        "total_s": time.perf_counter() - t0, "tool_s": tool_time,
        "llm_s": llm_time, "model": model, "tools": tool_calls_made,
        "grounded": len(tool_calls_made) > 0, "tokens": tokens})


def ollama_available() -> tuple[bool, list[str]]:
    """Check Ollama is reachable; return (ok, model names)."""
    try:
        import ollama
        models = [m.get("name") or m.get("model") for m in ollama.list().get("models", [])]
        return True, models
    except Exception:
        return False, []

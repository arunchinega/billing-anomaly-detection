"""Gloria cloud backend — same tools, same guardrails, hosted Llama via an
OpenAI-compatible API (Groq). Used only when no local Ollama is available and
a GROQ_API_KEY secret is configured."""
import json
import time

from .gloria_agent import SYSTEM_PROMPT, FAST_PATHS, _run_tool, MAX_TOOL_ROUNDS
from .gloria_tools import TOOL_SPECS

BASE_URL = "https://api.groq.com/openai/v1"


def ask_gloria_cloud(question: str, history: list, api_key: str,
                     model: str = "llama-3.1-8b-instant"):
    """Same generator protocol as ask_gloria: ('status'|'delta'|'meta', payload)."""
    from openai import OpenAI

    t0 = time.perf_counter()
    tool_time, llm_time = 0.0, 0.0
    tool_calls_made = []

    q = "".join(ch for ch in question.lower() if ch.isalnum() or ch.isspace()).strip()
    for key, (tool, preamble) in FAST_PATHS.items():
        if key in q and len(q) < 60:
            ts = time.perf_counter()
            result = _run_tool(tool, {})
            tool_time = time.perf_counter() - ts
            yield ("delta", preamble)
            yield ("meta", {"total_s": time.perf_counter() - t0, "tool_s": tool_time,
                            "llm_s": 0.0, "model": "fast-path (no LLM)",
                            "tools": [(tool, {}, result)], "grounded": True,
                            "tokens": 0})
            return

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + history[-4:]
                + [{"role": "user", "content": question}])

    yield ("status", "🔍 Gloria is checking that for you — please hold on…")

    for _ in range(MAX_TOOL_ROUNDS):
        ts = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SPECS,
            temperature=0.1, max_tokens=300)
        llm_time += time.perf_counter() - ts
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            ts = time.perf_counter()
            result = _run_tool(tc.function.name, args)
            tool_time += time.perf_counter() - ts
            tool_calls_made.append((tc.function.name, args, result))
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result.to_csv(index=False)})
        yield ("status", "📊 Got the data — writing your answer…")

    ts = time.perf_counter()
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.1,
        max_tokens=300, stream=True)
    tokens, buffer, leaked = 0, "", False
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        buffer += delta
        if tokens == 0 and buffer.lstrip().startswith("{"):
            leaked = True
        tokens += 1
        if not leaked:
            yield ("delta", delta)
    if leaked:
        yield ("delta", "I've pulled the data — please see the source table below.")
    llm_time += time.perf_counter() - ts

    yield ("meta", {"total_s": time.perf_counter() - t0, "tool_s": tool_time,
                    "llm_s": llm_time, "model": f"{model} (hosted)",
                    "tools": tool_calls_made,
                    "grounded": len(tool_calls_made) > 0, "tokens": tokens})

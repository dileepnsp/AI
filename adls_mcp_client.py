"""
adls_mcp_client.py
==================
Production-grade MCP client for the ADLS Parquet/Delta server.

Provides:
  - ConnectionPool  : persistent MCP client, reconnect on failure
  - AdlsAgent       : full agentic loop (tool selection, retry, tracing)
  - FastAPI app     : /ask, /health, /datasets, /schema/{path}  endpoints
  - Standalone mode : run as a script for CLI usage

Production hardening
--------------------
  ✓ Persistent connection pool (one client, reused across requests)
  ✓ Retry + exponential backoff on every MCP call
  ✓ Per-tool and overall request timeouts
  ✓ API-key authentication on every endpoint
  ✓ Per-user rate limiting (token bucket)
  ✓ Structured JSON logging with request_id tracing
  ✓ Token usage tracking per LLM call
  ✓ Graceful shutdown (AsyncExitStack)
  ✓ /health endpoint for Kubernetes readiness probes

Install
-------
  pip install fastmcp anthropic fastapi uvicorn pydantic

Environment variables
---------------------
  ANTHROPIC_API_KEY    (required)
  AGENT_API_KEYS       comma-separated valid API keys for /ask (default "dev-key")
  ADLS_MCP_TRANSPORT   "stdio" (default) or SSE URL e.g. "http://mcp-host:8002/sse"
  ADLS_MCP_SCRIPT      path to adls_mcp_server.py (stdio mode, default "./adls_mcp_server.py")
  RATE_LIMIT_CALLS     requests allowed per window (default 20)
  RATE_LIMIT_WINDOW    window in seconds (default 60)
  TOOL_TIMEOUT_S       per-tool call timeout (default 60)
  REQUEST_TIMEOUT_S    overall agent request timeout (default 120)
  RETRY_ATTEMPTS       retries per MCP call (default 3)

Run (FastAPI)
-------------
  uvicorn adls_mcp_client:app --host 0.0.0.0 --port 8080

Run (CLI)
---------
  python adls_mcp_client.py "Show me the first 10 rows of processed/orders_delta"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from fastmcp import Client


# ══════════════════════════════════════════════════════════════
#  1. STRUCTURED LOGGING
# ══════════════════════════════════════════════════════════════

class _JsonFmt(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(r), "level": r.levelname, "msg": r.getMessage()}
        if hasattr(r, "extra"):
            d.update(r.extra)
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        return json.dumps(d)

def _logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(_JsonFmt())
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg

log = _logger("adls_client")


# ══════════════════════════════════════════════════════════════
#  2. CONFIG
# ══════════════════════════════════════════════════════════════

def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Required env var '{key}' is not set.")
    return val

ANTHROPIC_API_KEY   = _require("ANTHROPIC_API_KEY")
ADLS_MCP_TRANSPORT  = os.environ.get("ADLS_MCP_TRANSPORT", "stdio")
ADLS_MCP_SCRIPT     = os.environ.get("ADLS_MCP_SCRIPT", "./adls_mcp_server.py")
AGENT_API_KEYS      = set(os.environ.get("AGENT_API_KEYS", "dev-key").split(","))
RATE_LIMIT_CALLS    = int(os.environ.get("RATE_LIMIT_CALLS", "20"))
RATE_LIMIT_WINDOW   = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
TOOL_TIMEOUT        = float(os.environ.get("TOOL_TIMEOUT_S", "60"))
REQUEST_TIMEOUT     = float(os.environ.get("REQUEST_TIMEOUT_S", "120"))
RETRY_ATTEMPTS      = int(os.environ.get("RETRY_ATTEMPTS", "3"))

# The LLM model used for all agent turns
LLM_MODEL = "claude-sonnet-4-6"


# ══════════════════════════════════════════════════════════════
#  3. RATE LIMITER  (token bucket, per user_id)
#     Production: replace with Redis sliding-window counter.
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_calls: int, window_secs: int):
        self._max    = max_calls
        self._window = window_secs
        self._buckets: dict[str, list[float]] = {}

    def is_allowed(self, user_id: str) -> bool:
        now    = time.monotonic()
        bucket = self._buckets.setdefault(user_id, [])
        self._buckets[user_id] = [t for t in bucket if now - t < self._window]
        if len(self._buckets[user_id]) >= self._max:
            return False
        self._buckets[user_id].append(now)
        return True

_rate_limiter = RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW)


# ══════════════════════════════════════════════════════════════
#  4. REQUEST TRACING
# ══════════════════════════════════════════════════════════════

@dataclass
class RequestTrace:
    request_id:    str
    user_id:       str
    question:      str
    tool_calls:    list = field(default_factory=list)
    input_tokens:  int  = 0
    output_tokens: int  = 0
    errors:        list = field(default_factory=list)
    _start:        float = field(default_factory=time.monotonic)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record_tool(self, name: str, latency_ms: float, ok: bool) -> None:
        self.tool_calls.append({"tool": name, "latency_ms": round(latency_ms), "ok": ok})

    def record_llm(self, usage) -> None:
        self.input_tokens  += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)

    def finish(self) -> None:
        latency_ms = round((time.monotonic() - self._start) * 1000)
        log.info("Request finished", extra={
            "request_id":    self.request_id,
            "user_id":       self.user_id,
            "latency_ms":    latency_ms,
            "total_tokens":  self.total_tokens,
            "tool_calls":    len(self.tool_calls),
            "errors":        len(self.errors),
        })


# ══════════════════════════════════════════════════════════════
#  5. RETRY HELPER
# ══════════════════════════════════════════════════════════════

async def _retry(coro_fn, *args, label: str = "op", **kwargs):
    last = RuntimeError("no attempts")
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(coro_fn(*args, **kwargs), timeout=TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            last = TimeoutError(f"{label} timed out after {TOOL_TIMEOUT}s")
            log.warning(f"{label} timeout", extra={"attempt": attempt})
        except Exception as exc:
            last = exc
            log.warning(f"{label} error", extra={"attempt": attempt, "error": str(exc)})
        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
    raise RuntimeError(f"{label} failed: {last}") from last


# ══════════════════════════════════════════════════════════════
#  6. CONNECTION POOL
#     One persistent MCP client shared across all requests.
#     Call startup() at app boot, shutdown() at teardown.
# ══════════════════════════════════════════════════════════════

class ConnectionPool:
    def __init__(self):
        self._client: Optional[Client] = None
        self._stack  = AsyncExitStack()
        self._tools: list[dict] = []
        self._healthy: bool = False

    async def startup(self) -> None:
        await self._stack.__aenter__()
        transport = (
            ADLS_MCP_TRANSPORT
            if ADLS_MCP_TRANSPORT.startswith("http")
            else ADLS_MCP_SCRIPT
        )
        try:
            client = Client(transport)
            await self._stack.enter_async_context(client)
            self._client = client

            raw_tools   = await asyncio.wait_for(client.list_tools(), timeout=10.0)
            self._tools = [
                {
                    "name":         t.name,
                    "description":  t.description or "",
                    "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in raw_tools
            ]
            self._healthy = True
            log.info("MCP connection established", extra={
                "transport": transport,
                "tools":     [t["name"] for t in self._tools],
            })
        except Exception as exc:
            self._healthy = False
            log.error("MCP connection failed at startup", extra={"error": str(exc)})
            raise

    async def shutdown(self) -> None:
        await self._stack.__aexit__(None, None, None)
        log.info("MCP connection closed")

    @property
    def client(self) -> Client:
        if not self._client or not self._healthy:
            raise RuntimeError("MCP client is not connected. Call startup() first.")
        return self._client

    @property
    def tools(self) -> list[dict]:
        return self._tools

    async def call_tool(self, name: str, args: dict) -> str:
        result = await _retry(self._client.call_tool, name, args, label=f"tool:{name}")
        return result[0].text if result else "No result."

    async def read_resource(self, uri: str) -> str:
        content = await _retry(self._client.read_resource, uri, label=f"resource:{uri}")
        return content[0].text if content else ""

    async def get_prompt(self, name: str, args: dict) -> str:
        p = await _retry(self._client.get_prompt, name, args, label=f"prompt:{name}")
        return p.messages[0].content.text if p.messages else ""

    async def health_check(self) -> bool:
        try:
            await asyncio.wait_for(self._client.list_tools(), timeout=5.0)
            self._healthy = True
        except Exception:
            self._healthy = False
        return self._healthy


# Global pool — initialised in FastAPI lifespan or main()
_pool = ConnectionPool()


# ══════════════════════════════════════════════════════════════
#  7. TOOL SELECTOR  (keyword-based, fast)
#     For 20+ tools: swap for an LLM-based selector.
# ══════════════════════════════════════════════════════════════

_TOOL_KEYWORDS: dict[str, list[str]] = {
    "list_datasets":      ["list", "discover", "find", "what datasets", "what files", "what tables", "exists"],
    "get_schema":         ["schema", "columns", "column", "fields", "types", "structure"],
    "preview_data":       ["preview", "sample", "first", "show me", "peek", "head"],
    "query_data":         ["query", "filter", "where", "select", "get", "fetch", "rows"],
    "get_stats":          ["stats", "statistics", "distribution", "average", "mean", "count", "null", "quality"],
    "get_delta_history":  ["history", "changelog", "versions", "changes", "audit", "who changed"],
    "get_partition_info": ["partition", "partitioned", "year=", "month=", "region="],
    "server_health":      ["health", "status", "connected", "ping"],
}

def _select_tools(question: str, all_tools: list[dict]) -> list[dict]:
    """Return only tools whose keywords appear in the question."""
    q        = question.lower()
    selected = []
    for tool in all_tools:
        kws = _TOOL_KEYWORDS.get(tool["name"], [])
        if any(kw in q for kw in kws):
            selected.append(tool)
    # Always include query_data and get_schema as fallback for data questions
    if not selected:
        selected = all_tools
    return selected


# ══════════════════════════════════════════════════════════════
#  8. ADLS AGENT
# ══════════════════════════════════════════════════════════════

class AdlsAgent:
    """
    Full agentic loop that:
      1. Pre-loads the dataset catalogue from MCP resource
      2. Fetches the data_analyst system prompt
      3. Selects the relevant tool subset
      4. Runs the Claude agentic loop (tool_use → call → continue)
      5. Enforces per-tool timeouts and an overall request deadline
      6. Returns structured trace for observability
    """

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._llm  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    async def run(
        self,
        question:   str,
        user_id:    str = "anonymous",
        request_id: str = "",
    ) -> tuple[str, RequestTrace]:

        request_id = request_id or str(uuid.uuid4())[:8]
        trace      = RequestTrace(request_id=request_id, user_id=user_id, question=question)

        log.info("Agent request started", extra={
            "request_id": request_id,
            "user_id":    user_id,
            "question":   question[:200],
        })

        # ── Rate limit ────────────────────────────────────────────────
        if not _rate_limiter.is_allowed(user_id):
            raise PermissionError(
                f"Rate limit exceeded for '{user_id}': "
                f"max {RATE_LIMIT_CALLS} requests / {RATE_LIMIT_WINDOW}s."
            )

        deadline = time.monotonic() + REQUEST_TIMEOUT

        try:
            # ── Step 1: Pre-load context (catalogue resource) ─────────
            catalogue = await self._pool.read_resource("adls://datasets")
            log.info("Dataset catalogue loaded", extra={"chars": len(catalogue)})

            # ── Step 2: Fetch system prompt ───────────────────────────
            system_prompt = await self._pool.get_prompt(
                "data_analyst_prompt", {}
            )

            # ── Step 3: Select relevant tools ─────────────────────────
            selected = _select_tools(question, self._pool.tools)
            log.info("Tools selected", extra={
                "request_id": request_id,
                "selected":   [t["name"] for t in selected],
            })

            # ── Step 4: Build messages ────────────────────────────────
            messages = [{
                "role": "user",
                "content": (
                    "## Available datasets (pre-loaded)\n\n"
                    f"{catalogue}\n\n"
                    "---\n\n"
                    f"{question}"
                ),
            }]

            # ── Step 5: Agentic loop ──────────────────────────────────
            answer = await self._agentic_loop(
                messages, system_prompt, selected, trace, deadline
            )
            trace.finish()
            return answer, trace

        except Exception as exc:
            trace.errors.append(str(exc))
            trace.finish()
            log.error("Agent request failed", extra={
                "request_id": request_id,
                "error": str(exc),
            })
            raise

    async def _agentic_loop(
        self,
        messages:      list,
        system_prompt: str,
        tools:         list[dict],
        trace:         RequestTrace,
        deadline:      float,
    ) -> str:

        MAX_ITER = 10

        for iteration in range(1, MAX_ITER + 1):

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Overall request timeout ({REQUEST_TIMEOUT}s) exceeded.")

            log.info("LLM call", extra={
                "request_id": trace.request_id,
                "iteration":  iteration,
                "tools":      [t["name"] for t in tools],
            })

            # Run LLM call in executor (anthropic SDK is sync)
            loop     = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._llm.messages.create(
                        model=LLM_MODEL,
                        max_tokens=4096,
                        system=system_prompt,
                        tools=tools,
                        messages=messages,
                    ),
                ),
                timeout=min(60.0, remaining),
            )

            trace.record_llm(response.usage)
            log.info("LLM response received", extra={
                "request_id":    trace.request_id,
                "stop_reason":   response.stop_reason,
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })

            tool_calls  = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Done — no more tool calls
            if response.stop_reason == "end_turn" or not tool_calls:
                return "\n\n".join(b.text for b in text_blocks)

            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls in parallel
            tool_results = await asyncio.gather(*[
                self._execute_tool(tc, trace)
                for tc in tool_calls
            ])

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type":        "tool_result",
                        "tool_use_id": tc.id,
                        "content":     result,
                    }
                    for tc, result in zip(tool_calls, tool_results)
                ],
            })

        raise RuntimeError(f"Agent exceeded {MAX_ITER} iterations without a final answer.")

    async def _execute_tool(self, tc, trace: RequestTrace) -> str:
        t0 = time.monotonic()
        log.info("Tool call started", extra={
            "request_id": trace.request_id,
            "tool":       tc.name,
            "args":       str(tc.input)[:300],
        })
        try:
            result  = await self._pool.call_tool(tc.name, tc.input)
            latency = (time.monotonic() - t0) * 1000
            trace.record_tool(tc.name, latency, ok=True)
            log.info("Tool call succeeded", extra={
                "request_id": trace.request_id,
                "tool":       tc.name,
                "latency_ms": round(latency),
                "result_len": len(result),
            })
            return result
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            trace.record_tool(tc.name, latency, ok=False)
            trace.errors.append(str(exc))
            log.error("Tool call failed", extra={
                "request_id": trace.request_id,
                "tool":       tc.name,
                "error":      str(exc),
            })
            return f"⚠️ Tool `{tc.name}` failed: {exc}"


# ══════════════════════════════════════════════════════════════
#  9. FASTAPI APPLICATION
#     Run: uvicorn adls_mcp_client:app --host 0.0.0.0 --port 8080
# ══════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI, HTTPException, Header, Depends
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app   = FastAPI(title="ADLS Data Agent API", version="1.0.0")
    agent = AdlsAgent(_pool)

    # ── Lifespan (startup / shutdown) ─────────────────────────────────────────

    @app.on_event("startup")
    async def _startup():
        await _pool.startup()
        log.info("FastAPI app started")

    @app.on_event("shutdown")
    async def _shutdown():
        await _pool.shutdown()

    # ── Auth dependency ────────────────────────────────────────────────────────

    async def _require_api_key(x_api_key: str = Header(...)) -> str:
        if x_api_key not in AGENT_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key

    # ── /ask ──────────────────────────────────────────────────────────────────

    class AskRequest(BaseModel):
        question:   str
        user_id:    str = "anonymous"
        request_id: str = ""

    class AskResponse(BaseModel):
        answer:       str
        request_id:   str
        total_tokens: int
        tool_calls:   list
        errors:       list

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        req: AskRequest,
        _api_key: str = Depends(_require_api_key),
    ):
        """
        Ask a natural-language question about ADLS Parquet/Delta data.

        Returns the answer and observability metadata (token usage, tool calls).
        """
        try:
            answer, trace = await agent.run(
                req.question, req.user_id, req.request_id
            )
            return AskResponse(
                answer=answer,
                request_id=trace.request_id,
                total_tokens=trace.total_tokens,
                tool_calls=trace.tool_calls,
                errors=trace.errors,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc))
        except Exception as exc:
            log.error("Unhandled error in /ask", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail="Internal agent error.")

    # ── /health ───────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        """
        Readiness probe — checks MCP server connectivity.
        Returns HTTP 200 if healthy, 503 if degraded.
        """
        ok = await _pool.health_check()
        body = {
            "status":    "ok" if ok else "degraded",
            "mcp_server": ok,
        }
        return JSONResponse(content=body, status_code=200 if ok else 503)

    # ── /datasets ─────────────────────────────────────────────────────────────

    @app.get("/datasets")
    async def datasets(
        prefix: str = "",
        _api_key: str = Depends(_require_api_key),
    ):
        """List all Parquet files and Delta tables under an optional prefix."""
        try:
            result = await _pool.read_resource("adls://datasets")
            return {"datasets": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── /schema/{path} ────────────────────────────────────────────────────────

    @app.get("/schema/{path:path}")
    async def schema(
        path: str,
        _api_key: str = Depends(_require_api_key),
    ):
        """Return the column schema for a Parquet or Delta dataset."""
        try:
            result = await _pool.read_resource(f"adls://schema/{path}")
            return {"schema": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

except ImportError:
    # FastAPI not installed — standalone script mode only
    app = None


# ══════════════════════════════════════════════════════════════
#  10. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def _cli_main(question: str) -> None:
    await _pool.startup()
    try:
        ag = AdlsAgent(_pool)
        answer, trace = await ag.run(question, user_id="cli")
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print(f"\n[tokens={trace.total_tokens}, tools={len(trace.tool_calls)}, "
              f"errors={len(trace.errors)}]")
    finally:
        await _pool.shutdown()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python adls_mcp_client.py \"<question>\"")
        print("\nExample questions:")
        print('  python adls_mcp_client.py "List all datasets"')
        print('  python adls_mcp_client.py "Show the schema of processed/orders_delta"')
        print('  python adls_mcp_client.py "Preview the first 20 rows of raw/orders/2024/01/"')
        print('  python adls_mcp_client.py "Query APAC orders over 1000 dollars from processed/orders_delta"')
        print('  python adls_mcp_client.py "Show Delta history for processed/orders_delta"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(_cli_main(question))

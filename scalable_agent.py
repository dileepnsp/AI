# ============================================================
#  registry.py  +  scalable_agent.py
#
#  Add a new data source by editing ONLY registry.py.
#  The agent discovers tools, resources, and prompts
#  from all registered servers automatically.
# ============================================================

# ────────────────────────────────────────────────────────────
#  registry.py  — the single file you edit to add sources
# ────────────────────────────────────────────────────────────
"""
SERVERS: list of all MCP server configurations.

To add a new source:
  1. Write servers/your_source_mcp_server.py  (same pattern as the others)
  2. Add one entry here.
  3. Done. The agent picks it up automatically.

Fields:
  name       : short unique identifier used for routing
  script     : path to the server Python file (stdio transport)
               OR url for remote HTTP/SSE servers
  prefix     : prepended to all tool names to avoid collisions
  keywords   : words that trigger lazy-loading this server's resources
  resources  : resource URIs to pre-load when keywords match
  prompt     : (optional) prompt name + args to fetch as system instruction
"""

SERVERS = [
    {
        "name":      "postgres",
        "script":    "servers/postgres_mcp_server.py",
        "prefix":    "pg_",
        "keywords":  ["employee", "dept", "department", "salary", "hire", "staff", "headcount"],
        "resources": ["postgres://schema", "postgres://tables/employee", "postgres://tables/department"],
        "prompt":    {"name": "sql_analyst_prompt", "args": {"db_name": "company_db"}},
    },
    {
        "name":      "adls",
        "script":    "servers/adls_mcp_server.py",
        "prefix":    "adls_",
        "keywords":  ["order", "file", "parquet", "adls", "lake", "storage", "csv"],
        "resources": ["adls://orders/containers", "adls://orders/meta/data-dictionary"],
        "prompt":    {"name": "data_analyst_prompt", "args": {"lake_name": "companydatalake"}},
    },

    # ── Add new sources below ─────────────────────────────────────────────

    # Example: Snowflake data warehouse
    # {
    #     "name":      "snowflake",
    #     "script":    "servers/snowflake_mcp_server.py",
    #     "prefix":    "sf_",
    #     "keywords":  ["warehouse", "snowflake", "revenue", "sales", "pipeline"],
    #     "resources": ["snowflake://schema"],
    #     "prompt":    {"name": "warehouse_analyst", "args": {}},
    # },

    # Example: Salesforce CRM (remote HTTP/SSE server)
    # {
    #     "name":      "salesforce",
    #     "url":       "http://salesforce-mcp:8020/sse",   # remote server
    #     "prefix":    "sf_crm_",
    #     "keywords":  ["lead", "opportunity", "account", "crm", "deal", "pipeline"],
    #     "resources": ["salesforce://objects"],
    #     "prompt":    None,
    # },

    # Example: MongoDB
    # {
    #     "name":      "mongodb",
    #     "script":    "servers/mongodb_mcp_server.py",
    #     "prefix":    "mongo_",
    #     "keywords":  ["mongo", "document", "collection", "nosql"],
    #     "resources": ["mongodb://collections"],
    #     "prompt":    None,
    # },

    # Example: Kafka event streams
    # {
    #     "name":      "kafka",
    #     "url":       "http://kafka-mcp:8030/sse",
    #     "prefix":    "kafka_",
    #     "keywords":  ["event", "stream", "topic", "kafka", "message"],
    #     "resources": ["kafka://topics"],
    #     "prompt":    None,
    # },
]


# ────────────────────────────────────────────────────────────
#  scalable_agent.py  — never edit this when adding sources
# ────────────────────────────────────────────────────────────

import asyncio
import json
import anthropic
from fastmcp import Client
from registry import SERVERS


llm = anthropic.Anthropic()


# ══════════════════════════════════════════════════════════════
#  Step 1: Connect to all servers and discover everything
# ══════════════════════════════════════════════════════════════

async def connect_all(servers: list) -> dict:
    """
    Open a Client for every server in the registry.
    Returns a dict keyed by server name with client + metadata.
    Works for both local scripts (stdio) and remote URLs (SSE).
    """
    connected = {}
    for cfg in servers:
        transport = cfg.get("url") or cfg.get("script")
        client    = Client(transport)
        await client.__aenter__()

        # Collect all tools from this server
        raw_tools = await client.list_tools()
        tools = [
            {
                "name":         cfg["prefix"] + t.name,
                "description":  t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                "_source":      cfg["name"],
                "_real_name":   t.name,
            }
            for t in raw_tools
        ]

        connected[cfg["name"]] = {
            "client":   client,
            "cfg":      cfg,
            "tools":    tools,
        }
        print(f"  ✓ Connected to '{cfg['name']}' — {len(tools)} tools")

    return connected


async def disconnect_all(connected: dict) -> None:
    for info in connected.values():
        await info["client"].__aexit__(None, None, None)


# ══════════════════════════════════════════════════════════════
#  Step 2: Smart tool selection — inject only relevant tools
# ══════════════════════════════════════════════════════════════

def select_tools_by_keywords(question: str, connected: dict) -> list:
    """
    Fast keyword-based tool selection.
    Only injects tools from servers whose keywords appear in the question.
    Falls back to ALL tools if no keywords match (safety net).
    """
    question_lower = question.lower()
    selected = []

    for name, info in connected.items():
        keywords = info["cfg"].get("keywords", [])
        matched  = any(kw in question_lower for kw in keywords)
        if matched:
            selected.extend(info["tools"])

    # Safety: if nothing matched, include all tools
    if not selected:
        for info in connected.values():
            selected.extend(info["tools"])

    return selected


async def select_tools_by_llm(question: str, all_tools: list) -> list:
    """
    Smarter selection using a cheap LLM call.
    Use this instead of keyword matching for better accuracy at scale (10+ sources).
    """
    tool_summary = "\n".join(
        f"- {t['name']}: {t['description'][:80]}" for t in all_tools
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Available tools:\n{tool_summary}\n\n"
        "Return a JSON array of tool names that are needed to answer the question. "
        "Return [] if none are needed. Return only the JSON array, no other text."
    )
    resp = llm.messages.create(
        model="claude-haiku-4-5-20251001",   # cheapest model for selection
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        needed = json.loads(resp.content[0].text)
        result = [t for t in all_tools if t["name"] in needed]
        return result if result else all_tools  # fallback
    except Exception:
        return all_tools


# ══════════════════════════════════════════════════════════════
#  Step 3: Lazy resource loading — only load what's relevant
# ══════════════════════════════════════════════════════════════

async def load_relevant_resources(question: str, connected: dict) -> str:
    """
    Fetch resources only from servers whose keywords match the question.
    Avoids burning context window on irrelevant schemas.
    """
    question_lower = question.lower()
    parts = []

    for name, info in connected.items():
        cfg      = info["cfg"]
        keywords = cfg.get("keywords", [])

        if not any(kw in question_lower for kw in keywords):
            continue  # skip this source — not relevant

        for uri in cfg.get("resources", []):
            try:
                content = await info["client"].read_resource(uri)
                text    = content[0].text if content else ""
                if text:
                    parts.append(f"## [{name}] {uri}\n{text}")
                    print(f"  ✓ Loaded resource: {uri}")
            except Exception as e:
                print(f"  ⚠ Could not load {uri}: {e}")

    return "\n\n".join(parts) if parts else "No relevant context pre-loaded."


# ══════════════════════════════════════════════════════════════
#  Step 4: Fetch system prompts from relevant servers
# ══════════════════════════════════════════════════════════════

async def build_system_prompt(question: str, connected: dict) -> str:
    """
    Fetch prompt templates from all servers whose keywords match.
    Merge them into one system message.
    """
    question_lower = question.lower()
    parts = [
        "You are a data analyst with access to multiple data sources. "
        "Use the available tools to answer questions accurately. "
        "Always state which source (database/file system) the data came from."
    ]

    for name, info in connected.items():
        cfg      = info["cfg"]
        keywords = cfg.get("keywords", [])
        prompt_cfg = cfg.get("prompt")

        if not prompt_cfg:
            continue
        if not any(kw in question_lower for kw in keywords):
            continue

        try:
            p = await info["client"].get_prompt(
                prompt_cfg["name"],
                prompt_cfg.get("args", {})
            )
            text = p.messages[0].content.text
            parts.append(f"--- [{name}] ---\n{text}")
            print(f"  ✓ Loaded prompt: {prompt_cfg['name']} from '{name}'")
        except Exception as e:
            print(f"  ⚠ Could not load prompt from '{name}': {e}")

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  Step 5: Route tool calls to the right server
# ══════════════════════════════════════════════════════════════

async def execute_tool(tool_name: str, args: dict, connected: dict) -> str:
    """
    Find which server owns this tool (by prefix) and call it.
    """
    for name, info in connected.items():
        prefix = info["cfg"]["prefix"]
        if tool_name.startswith(prefix):
            real_name = tool_name[len(prefix):]
            result    = await info["client"].call_tool(real_name, args)
            return result[0].text if result else "No result"

    return f"Error: no server handles tool '{tool_name}'"


# ══════════════════════════════════════════════════════════════
#  Main agent loop
# ══════════════════════════════════════════════════════════════

async def run_agent(question: str, use_llm_tool_selection: bool = False) -> str:
    """
    Full agent turn across all registered MCP servers.

    use_llm_tool_selection=True  → more accurate but costs an extra LLM call
    use_llm_tool_selection=False → fast keyword matching (default)
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"Servers:  {[s['name'] for s in SERVERS]}")
    print('='*60)

    # Connect to every registered server
    print("\n[Connecting to servers...]")
    connected = await connect_all(SERVERS)

    try:
        # ── Load resources relevant to this question ───────────────────
        print("\n[Loading relevant resources...]")
        context = await load_relevant_resources(question, connected)

        # ── Build system prompt from relevant servers ──────────────────
        print("\n[Fetching prompt templates...]")
        system_prompt = await build_system_prompt(question, connected)

        # ── Select tools relevant to this question ─────────────────────
        all_tools = []
        for info in connected.values():
            all_tools.extend(info["tools"])

        print(f"\n[Selecting tools from {len(all_tools)} total...]")
        if use_llm_tool_selection:
            selected_tools = await select_tools_by_llm(question, all_tools)
        else:
            selected_tools = select_tools_by_keywords(question, connected)

        # Strip internal routing metadata before sending to LLM
        llm_tools = [
            {k: v for k, v in t.items() if not k.startswith("_")}
            for t in selected_tools
        ]
        print(f"  ✓ Selected {len(llm_tools)} tools: {[t['name'] for t in llm_tools]}")

        # ── Build initial messages ─────────────────────────────────────
        user_content = f"## Pre-loaded context\n\n{context}\n\n---\n\n{question}"
        messages = [{"role": "user", "content": user_content}]

        # ── Agentic loop ───────────────────────────────────────────────
        print("\n[Running agentic loop...]")
        iteration = 0
        max_iter  = 10

        while iteration < max_iter:
            iteration += 1
            response = llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=llm_tools,
                messages=messages,
            )

            tool_calls  = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            print(f"  Iter {iteration}: stop={response.stop_reason}, "
                  f"tools={len(tool_calls)}, text={len(text_blocks)}")

            if response.stop_reason == "end_turn" or not tool_calls:
                return "\n\n".join(b.text for b in text_blocks)

            # Execute all tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tc in tool_calls:
                print(f"    → {tc.name}({tc.input})")
                result = await execute_tool(tc.name, tc.input, connected)
                print(f"    ← {result[:80]}...")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tc.id,
                    "content":     result,
                })
            messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached."

    finally:
        await disconnect_all(connected)


# ══════════════════════════════════════════════════════════════
#  Gateway pattern (Pattern B) — single server mounting all sources
# ══════════════════════════════════════════════════════════════

def build_gateway():
    """
    Alternative to the multi-client pattern:
    Mount all servers into one FastMCP gateway.
    Run with: python scalable_agent.py --gateway
    """
    from fastmcp import FastMCP

    gateway = FastMCP("data-gateway")

    # Dynamically mount all servers from registry
    for cfg in SERVERS:
        if "script" in cfg:
            # Local script — import and mount
            import importlib.util, sys
            spec   = importlib.util.spec_from_file_location(cfg["name"], cfg["script"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            server = getattr(module, "mcp")
            gateway.mount(cfg["name"], server)
            print(f"  Mounted local: {cfg['name']}")
        elif "url" in cfg:
            # Remote server — proxy over HTTP/SSE
            from fastmcp import MCPProxy
            gateway.mount(cfg["name"], MCPProxy(cfg["url"]))
            print(f"  Mounted remote: {cfg['name']} → {cfg['url']}")

    return gateway


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

import sys

if __name__ == "__main__":

    if "--gateway" in sys.argv:
        # Run as a gateway server (Pattern B)
        print("Starting gateway server...")
        gw = build_gateway()
        gw.run(transport="sse", host="0.0.0.0", port=8000)

    else:
        # Run as a multi-client agent (Pattern A)
        questions = [
            # Hits postgres only (keyword: employee)
            "Show me all employees in Engineering with salary above 80000.",

            # Hits adls only (keyword: order)
            "What order files do we have for Q1 2024?",

            # Hits both postgres + adls (keywords: employee + order)
            "How many employees do we have in total, and how many APAC orders in Jan 2024?",

            # No keyword match → falls back to all tools (safety net)
            "Give me a high-level summary of our company data.",
        ]

        async def run_all():
            for q in questions[:1]:   # run first question — change index to try others
                answer = await run_agent(q, use_llm_tool_selection=False)
                print(f"\nANSWER:\n{answer}\n")

        asyncio.run(run_all())

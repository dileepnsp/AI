# ADLS Parquet / Delta MCP Server + Client

Production-grade MCP server and agent client for reading
Parquet files and Delta Lake tables from Azure Data Lake Storage Gen2.

## Files

| File | Purpose |
|------|---------|
| `adls_mcp_server.py` | FastMCP server — tools, resources, prompt |
| `adls_mcp_client.py` | Agent client — FastAPI app + CLI |

## Install

```bash
pip install fastmcp azure-storage-file-datalake azure-identity \
            pyarrow deltalake pandas tabulate anthropic fastapi uvicorn
```

## Environment variables

```bash
# Required
export AZURE_STORAGE_ACCOUNT="companydatalake"
export AZURE_STORAGE_CONTAINER="orders"
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure auth — service principal (production)
export AZURE_CLIENT_ID="..."
export AZURE_CLIENT_SECRET="..."
export AZURE_TENANT_ID="..."
# OR: az login  (local dev — DefaultAzureCredential picks it up)

# Tuning (optional)
export MAX_ROWS=5000
export TOOL_TIMEOUT_S=60
export AGENT_API_KEYS="key-team-a,key-team-b"
```

## Run locally (stdio)

```bash
# Terminal 1
python adls_mcp_server.py

# Terminal 2
python adls_mcp_client.py "List all datasets"
python adls_mcp_client.py "Show schema of processed/orders_delta"
python adls_mcp_client.py "Query APAC orders over 1000 from processed/orders_delta"
python adls_mcp_client.py "Show Delta history for processed/orders_delta"
python adls_mcp_client.py "Get stats for processed/orders_delta"
```

## Run in production (HTTP/SSE)

```bash
# Server
python adls_mcp_server.py --transport sse --port 8002

# Client
export ADLS_MCP_TRANSPORT="http://localhost:8002/sse"
uvicorn adls_mcp_client:app --host 0.0.0.0 --port 8080
```

## API

```bash
# Ask a question
curl -X POST http://localhost:8080/ask \
  -H "x-api-key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show APAC orders over 1000", "user_id": "alice"}'

# Health check (Kubernetes readiness probe)
curl http://localhost:8080/health

# List datasets
curl -H "x-api-key: dev-key" http://localhost:8080/datasets

# Schema for a path
curl -H "x-api-key: dev-key" http://localhost:8080/schema/processed/orders_delta
```

## MCP primitives

| Type | Name | LLM uses it when... |
|------|------|---------------------|
| Tool | `list_datasets` | user asks what data exists |
| Tool | `get_schema` | user asks about columns |
| Tool | `preview_data` | user wants sample rows |
| Tool | `query_data` | user wants filtered data |
| Tool | `get_stats` | user asks about data quality |
| Tool | `get_delta_history` | user asks about changes |
| Tool | `get_partition_info` | user asks about partitions |
| Tool | `server_health` | health / monitoring |
| Resource | `adls://datasets` | agent pre-loads catalogue |
| Resource | `adls://schema/{path}` | agent pre-loads schema |
| Prompt | `data_analyst_prompt` | agent injects as system message |

## Claude Desktop

```json
{
  "mcpServers": {
    "adls-data": {
      "command": "python",
      "args": ["/path/to/adls_mcp_server.py"],
      "env": {
        "AZURE_STORAGE_ACCOUNT": "companydatalake",
        "AZURE_STORAGE_CONTAINER": "orders"
      }
    }
  }
}
```

## Production checklist

- [ ] Secrets from Azure Key Vault / env vars only (never in code)
- [ ] Read-only ADLS role: Storage Blob Data Reader
- [ ] AGENT_API_KEYS set to real values
- [ ] /health wired to Kubernetes readiness probe
- [ ] Logs shipped to Datadog / Azure Monitor
- [ ] Replace in-memory TtlCache + RateLimiter with Redis for multi-replica

from registry import SERVERS   # ← imported at the top of scalable_agent.py

SERVERS = [
    {
        "name":     "postgres",
        "script":   "servers/postgres_mcp_server.py",
        "prefix":   "pg_",
        "keywords": ["employee", "dept", "salary"],
        "resources":["postgres://schema"],
        "prompt":   {"name": "sql_analyst_prompt", "args": {"db_name": "company_db"}},
    },
    {
        "name":     "adls",
        "script":   "servers/adls_mcp_server.py",
        "prefix":   "adls_",
        "keywords": ["order", "file", "parquet"],
        "resources":["adls://orders/containers"],
        "prompt":   {"name": "data_analyst_prompt", "args": {}},
    },
]
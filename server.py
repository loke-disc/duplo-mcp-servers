from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from pydantic import BaseModel
from typing import List, Optional
import os
import json

load_dotenv("../.env")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.json')

# Create an MCP server
mcp = FastMCP(
    name="Duplo-MCP",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
    stateless_http=True,
)

class MCPColumn(BaseModel):
    name: str
    type: str
    references: Optional[str] = None

class MCPTable(BaseModel):
    name: str
    description: str = ""
    isMainTable: bool = False
    columns: List[MCPColumn]

class ModelContextProtocol(BaseModel):
    schema: List[MCPTable]
    constraints: dict

@mcp.tool()
def get_all_duplo_schema_preloaded() -> ModelContextProtocol:
    """Returns the Duplo DB schema"""
    with open(SCHEMA_PATH, 'r') as f:
        data = json.load(f)
    tables = [MCPTable(**table) for table in data["schema"]]
    return ModelContextProtocol(schema=tables, constraints=data.get("constraints", {}))

@mcp.tool()
def get_table_def_preloaded(table_name: str) -> ModelContextProtocol:
    """Returns the Table Definition for the given table name"""
    with open(SCHEMA_PATH, 'r') as f:
        data = json.load(f)
    # Filter for the requested table name
    filtered_tables = [MCPTable(**table) for table in data["schema"] if table["name"] == table_name]
    return ModelContextProtocol(schema=filtered_tables, constraints=data.get("constraints", {}))

@mcp.tool()
def get_all_duplo_schema_from_url(connection_url: str) -> ModelContextProtocol:
    """Returns the Duplo DB schema from the given connection URL"""
    engine = create_engine(connection_url)
    inspector = inspect(engine)
    mcp_tables = []
    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append(MCPColumn(name=column['name'], type=str(column['type'])))
        mcp_tables.append(MCPTable(name=table_name, columns=columns))
    return ModelContextProtocol(schema=mcp_tables, constraints={})

@mcp.tool()
def get_table_def_from_url(connection_url: str, table_name: str) -> ModelContextProtocol:
    """Returns the Table Definition for the given table name"""
    engine = create_engine(connection_url)
    inspector = inspect(engine)
    columns = []
    for column in inspector.get_columns(table_name):
        columns.append(MCPColumn(name=column['name'], type=str(column['type'])))
    return ModelContextProtocol(schema=[MCPTable(name=table_name, columns=columns)], constraints={})

# Run the server
if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        print("Running server with Streamable HTTP transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")

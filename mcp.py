
import json
from pydantic import BaseModel
from typing import List, Optional
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.json')

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

def get_mcp_schema() -> ModelContextProtocol:
    with open(SCHEMA_PATH, 'r') as f:
        data = json.load(f)
    # Convert dicts to Pydantic models for validation and compatibility
    tables = [MCPTable(**table) for table in data["schema"]]
    return ModelContextProtocol(schema=tables, constraints=data.get("constraints", {}))

# Create an MCP server
mcp = FastMCP(
    name="Calculator",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
    stateless_http=True,
)


# Add a simple calculator tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
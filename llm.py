
import requests
from mcp import ModelContextProtocol

def build_prompt(mcp: ModelContextProtocol, user_input: str) -> str:
    schema_text = "\n".join(
        [f"Table: {t.name}\nColumns: {', '.join([c.name for c in t.columns])}" for t in mcp.schema]
    )
    prompt = f"""
You are a helpful AI assistant. Here is the database schema:
{schema_text}

Given the user query:
"{user_input}"

Generate a valid, read-only PostgreSQL SQL query.
"""
    return prompt.strip()

def generate_sql_with_ollama(prompt: str) -> str:
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip()

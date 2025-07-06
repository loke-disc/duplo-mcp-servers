
from sqlalchemy import create_engine, text

DB_URI = "postgresql://duplo_master:WNvy1JwF9cnlCk8g@db-duplo.dev.dcitech.cloud:5432/duplo"
engine = create_engine(DB_URI)

def execute_query(sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]

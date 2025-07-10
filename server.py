from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from pydantic import BaseModel
from typing import List, Optional, Tuple
import numpy as np
import logging
import boto3
from botocore.config import Config
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def get_bedrock_client():
    config = Config(read_timeout=200, retries=dict(max_attempts=5))
    return boto3.client(service_name="bedrock-runtime", region_name="us-east-1", config=config)


def assume_role_and_get_bedrock_client():
    role_arn = "arn:aws:iam::285885959383:role/duplo-dev-GTO_AI_0016"
    external_id = "b6897c7b9bc377ab815fd8589a4b9a64a00a5e1df92b0d472dc832c680118139"
    sts_client = boto3.client("sts")
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName="bedrock-session",
        ExternalId=external_id
    )
    credentials = response['Credentials']
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )
    return bedrock_client

# Configure the inference parameters.
inf_params = {"maxTokens": 2000, "topP": 0.9, "topK": 20, "temperature": 0.7}
# Define your system prompt(s).
system_list = [
            {
                "text": "You are an expert SQL generator for a PostgreSQL database"
            }
]
@mcp.tool()
def invoke_nova_micro_model_with_messages(messages):
    """Invokes the Nova Micro model with the provided messages."""
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/f00jxgk9azx1"
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "schemaVersion": "messages-v1",
                "system": system_list,
                "inferenceConfig": inf_params,
                "messages": messages
            }
        )
    )
    response_body = json.loads(response['body'].read())
    return response_body['output']['message']['content'][0]['text']

@mcp.tool()
def invoke_nova_lite_model_with_messages(messages):
    """Invokes the Nova Lite model with the provided messages."""
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/8le01kd7zvww"
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "schemaVersion": "messages-v1",
                "system": system_list,
                "inferenceConfig": inf_params,
                "messages": messages
            }
        )
    )
    response_body = json.loads(response['body'].read())
    return response_body

@mcp.tool()
def invoke_nova_pro_model_with_messages(messages):
    """Invokes the Nova Pro model with the provided messages."""
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/ckjdsl1tkp0u"
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "schemaVersion": "messages-v1",
                "system": system_list,
                "inferenceConfig": inf_params,
                "messages": messages
            }
        )
    )
    response_body = json.loads(response['body'].read())
    return response_body

def invoke_claude_3_model_with_response_stream(messages):
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/i1fydddeaehs"
    response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.999,
                "top_k": 250,
                "messages": messages
            }
        ),
    )
    return response

@mcp.tool()
def invoke_claude_3_model_with_messages(messages):
    """Invokes the Claude 3 model with the provided prompt."""
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/i1fydddeaehs"
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.999,
                "top_k": 250,
                "messages": messages
            }
        )
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def invoke_llm_with_response_stream(st):
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/i1fydddeaehs"
    response = client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 0.999,
                    "top_k": 250, #sampling
                    "messages": st.session_state.messages
                }),
                contentType="application/json",
                accept="application/json"
            )
    return response

@mcp.tool()
def invoke_titan_embed_text_v2_messages(messages):
    """Invokes the Titan Embed Text V2 model with the provided messages."""
    client = assume_role_and_get_bedrock_client()
    model_id = "arn:aws:bedrock:us-east-1:285885959383:application-inference-profile/ehuvifb8tdgb"
    # Extract text content from the first message
    input_text = messages[0]["content"] if messages else ""
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "inputText": input_text  # Pass a single string, not a list
        }),
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response['body'].read())
    return response_body

class EmbeddingService:
    def __init__(self):
        pass  # No model_name or local model needed

    def get_embedding(self, text):
        if not text:
            return []
        # Titan expects a list of messages with "content"
        messages = [{"content": text}]
        logger.info(f"Invoking Titan Embed Text V2 with text: {text}")
        response = invoke_titan_embed_text_v2_messages(messages)
        logger.info(f"Received response from Titan: {response}")
        # Adjust the key based on Titan's response structure
        # Example: response['embedding'] or response['embeddings'][0]
        embedding = response.get("embedding") or (response.get("embeddings") or [None])[0]
        return embedding if embedding else []

def load_schema_descriptions(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class RAGContext:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def retrieve_relevant_context(self, user_query, previous_context=""):
        logger.info(f"Retrieving relevant context for query: {user_query} with previous context: {previous_context}")
        knowledge_base = load_schema_descriptions("schema-descriptions.json")
        logger.info(f"Knowledge base loaded with {len(knowledge_base)} entries.")
        for entry in knowledge_base:
            embedding = self.embedding_service.get_embedding(entry.get("text", ""))
            if embedding:
                entry["embedding"] = embedding
        query_for_embedding = f"{previous_context}\n{user_query}" if previous_context else user_query
        query_embedding = self.embedding_service.get_embedding(query_for_embedding)
        logger.info(f"Query embedding generated: {query_embedding}")
        if not query_embedding:
            return []

        scored_chunks = []
        for entry in knowledge_base:
            kb_embedding = entry.get("embedding")
            text = entry.get("text", "")
            if kb_embedding:
                similarity = cosine_similarity(query_embedding, kb_embedding)
                if similarity > 0.5:
                    scored_chunks.append((similarity, text))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        logger.info(f"Found {len(scored_chunks)} relevant chunks with similarity > 0.5")
        return [text for _, text in scored_chunks[:3]]

embedding_service = EmbeddingService()
rag_context = RAGContext(embedding_service)

@mcp.tool()
def retrieve_relevant_context(user_query: str, previous_context: str = ""):
    """Retrieves relevant context based on the user query and previous context."""
    return rag_context.retrieve_relevant_context(user_query, previous_context)


# Flatten the schema text from list of objects
def flatten_schema_text(schema_list: List[dict]) -> List[str]:
    return [entry["text"] for entry in schema_list if entry["type"] == "schema_desc"]

# Compute cosine similarity
def compute_similarity(vec1, vec2) -> float:
    return float(cosine_similarity([vec1], [vec2])[0][0])

@mcp.tool()
def get_matching_schema_chunks(user_prompt: str) -> List[str]:
    """Returns the schema chunks that match the user prompt."""
    knowledge_base = load_schema_descriptions("schema-descriptions.json")
    top_k = 3
    schema_chunks = flatten_schema_text(knowledge_base)
    # Wrap user_prompt in a message dict
    user_vector = invoke_titan_embed_text_v2_messages([{"content": user_prompt}]).get("embedding", [])
    chunk_vectors = [
        invoke_titan_embed_text_v2_messages([{"content": chunk}]).get("embedding", [])
        for chunk in schema_chunks
    ]

    similarities: List[Tuple[str, float]] = [
        (chunk, cosine_similarity(user_vector, vec))
        for chunk, vec in zip(schema_chunks, chunk_vectors)
        if user_vector and vec
    ]

# cosine,
    # Sort by similarity descending
    top_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    # Return only the top-k chunk texts
    return [chunk for chunk, _ in top_chunks]


from mcp import get_mcp_schema
from llm import build_prompt, generate_sql_with_ollama
from db import execute_query

def main():
    user_question = input("Ask your database question: ")
    mcp = get_mcp_schema()
    prompt = build_prompt(mcp, user_question)
    print("\n[Prompt for LLM]\n", prompt)
    sql_query = generate_sql_with_ollama(prompt)
    print("\n[Generated SQL]\n", sql_query)
    result = execute_query(sql_query)
    print("\n[Query Result]\n")
    for row in result:
        print(row)

if __name__ == "__main__":
    main()

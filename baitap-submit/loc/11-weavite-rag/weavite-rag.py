# Áp dụng Weaviate để tạo ra một RAG flow đơn giản, gợi ý sách hay, dựa theo query của người dùng.
# Thay vì hard code query, hãy lấy query từ người dùng input ở console, hoặc tạo app bằng Gradio.

import gradio as gr
import weaviate
import json
import inspect
from openai import OpenAI
from pydantic import TypeAdapter


vector_db_client = weaviate.connect_to_local()
vector_db_client.connect()

print("DB is ready:", vector_db_client.is_ready())

COLLECTION_NAME = "BookCollection"

def search_books(query: str):
    """
    Search books from Weaviate BookCollection based on user query.
    Return top related books.
    """

    book_collection = vector_db_client.collections.get(
        COLLECTION_NAME
    )

    response = book_collection.query.hybrid(
        query=query,
        limit=3,
        alpha=0.35,
        query_properties=[
            "title^3",
            "author^2",
            "genre^2",
            "description"
        ]
    )

    results = []

    for book in response.objects:

        results.append({
            "title": book.properties.get("title", ""),
            "author": book.properties.get("author", ""),
            "genre": book.properties.get("genre", ""),
            "description": book.properties.get("description", "")
        })

    return results


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": inspect.getdoc(search_books),
            "parameters": TypeAdapter(search_books).json_schema(),
        },
    }
]

FUNCTION_MAP = {
    "search_books": search_books
}

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

COMPLETION_MODEL = "qwen3.5:4b"

def get_completion(messages):
    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        tools=tools,
        temperature=0
    )

    return response


SYSTEM_PROMPT = """
You are a friendly AI librarian assistant.
"""

def chat(message, history):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for msg in history:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": message
    })

    response = get_completion(messages)

    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    while finish_reason != "stop":

        tool_call = first_choice.message.tool_calls[0]

        tool_function = tool_call.function

        tool_arguments = json.loads(
            tool_function.arguments
        )

        real_function = FUNCTION_MAP[
            tool_function.name
        ]

        result = real_function(**tool_arguments)

        messages.append(first_choice.message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_function.name,
            "content": json.dumps(
                result,
                ensure_ascii=False
            )
        })

        response = get_completion(messages)

        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    return first_choice.message.content

def respond(message, history):

    response = chat(message, history)

    return response


with gr.Blocks() as demo:

    gr.Markdown("#Book Assistant")

    gr.ChatInterface(
        fn=respond
    )

demo.launch()
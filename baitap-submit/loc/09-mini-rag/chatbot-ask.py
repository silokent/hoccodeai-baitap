from wikipediaapi import Wikipedia
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import inspect
from pydantic import TypeAdapter
import json
from unidecode import unidecode

# Ở đây, ta dùng `PersistentClient` để lưu trữ dữ liệu trong một file trong thư mục `./data`.
chroma_client = chromadb.PersistentClient(path="./data")
chroma_client.heartbeat()

# Mặc định, chroma DB sử dụng `all-MiniLM-L6-v2` của Sentence Transformers
# mà mình đã hướng dẫn ở bài trước để tạo embeddings.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Ngoài ra, bạn có thể sử dụng mô hình embedding của OpenAI để có hiệu suất tốt hơn
# Nhưng cần đăng ký và có API key của OpenAI
# embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")

wiki = Wikipedia('HocCodeAI/0.0 (https://hoccodeai.com)', 'en')


def get_wikipedia_content(person_name: str):
    """
    Retrieve information about a famous person from Wikipedia.
    :param person_name: The name of the famous person.
    :output: Full Wikipedia page content about the person.
    """
    page = wiki.page(person_name)
    return page.text


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia_content",
            "description": inspect.getdoc(get_wikipedia_content),
            "parameters": TypeAdapter(get_wikipedia_content).json_schema(),
        }
    }
]

FUNCTION_MAP = {
    "get_wikipedia_content": get_wikipedia_content
}

# https://platform.openai.com/api-keys
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="llmstudio"
)


def get_completion(messages):
    response = client.chat.completions.create(
        model="qwen3.5:4b",
        messages=messages,
        tools=tools,
        temperature=0
    )
    return response


messages = [
    {
        "role": "system",
        "content": """
You are a helpful assistant.
"""
    }
]


question = input("Có câu hỏi gì không bạn ơi: ")



messages.append({
    "role": "user",
    "content": question
})

response = get_completion(messages)

first_choice = response.choices[0]
finish_reason = first_choice.finish_reason

while finish_reason != "stop":
    tool_call = first_choice.message.tool_calls[0]

    tool_call_function = tool_call.function
    tool_call_arguments = json.loads(tool_call_function.arguments)

    tool_function = FUNCTION_MAP[tool_call_function.name]

    result = tool_function(**tool_call_arguments)

    collection_name = unidecode(
    tool_call_arguments["person_name"]
).replace(" ", "_")

    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        # Chia nhỏ văn bản một cách đơn giản
        paragraphs = result.split('\n\n')

        # Lưu trữ các đoạn văn bản trong collection
        for index, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                collection.add(
                    documents=[paragraph],
                    ids=[str(index)]
                )

    q = collection.query(
        query_texts=[question],
        n_results=3
    )

    CONTEXT = q["documents"][0]

    prompt = f"""
Use the following CONTEXT to answer the QUESTION at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use an unbiased and journalistic tone.

CONTEXT: {CONTEXT}

QUESTION: {question}
"""

    messages.append(first_choice.message)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call_function.name,
        "content": prompt
    })

    response = get_completion(messages)

    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

print(first_choice.message.content)
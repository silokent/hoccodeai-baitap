from pprint import pprint
import json
from openai import OpenAI
import inspect
from typing import get_type_hints
from pydantic import TypeAdapter
import requests

# Implement 3 hàm


def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location"""
    # Hardcoded response for demo purposes
    return "Trời rét vãi nôi, 7 độ C"


def get_stock_price(symbol: str):
    # Không làm gì cả, để hàm trống
    pass


# Bài 2: Implement hàm `view_website`, sử dụng `requests` và JinaAI để đọc markdown từ URL
def view_website(url: str):
    """
    View and extract content from a website URL.
    """
    jina_url = f"https://r.jina.ai/{url}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(jina_url, headers=headers)
    return response.text


# Bài 1: Thay vì tự viết object `tools`, hãy xem lại bài trước, sửa code và dùng `inspect` và `TypeAdapter` để define `tools`
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price of a given symbol",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_website",
            "description": inspect.getdoc(view_website),
            "parameters": TypeAdapter(view_website).json_schema(),
        }
    }
]

# Dùng local LLM
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="llmstudio"
)

COMPLETION_MODEL = "qwen3.5:4b"

messages = [{"role": "user", "content": "Hãy tóm tắt nội dung trang website https://carwashcentre.vn/dich-vu-rua-xe-dung-cach/"}]

print("Bước 1: Gửi message lên cho LLM")
pprint(messages)

response = client.chat.completions.create(
    model=COMPLETION_MODEL,
    messages=messages,
    tools=tools
)

print("Bước 2: LLM đọc và phân tích ngữ cảnh LLM")
pprint(response)

print("Bước 3: Lấy kết quả từ LLM")
tool_call = response.choices[0].message.tool_calls[0]

pprint(tool_call)

print("Bước 4: Chạy function get_current_weather ở máy mình")


if tool_call.function.name == 'view_website':
    arguments = json.loads(tool_call.function.arguments)
    website_content = view_website(arguments.get('url'))
    print(f"Kết quả bước 4: {website_content}")


    print("Bước 5: Gửi kết quả lên cho LLM")
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "content": website_content,
        "tool_call_id": tool_call.id
    })

    final_response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
    )

    print("===== TÓM TẮT =====")
    print(final_response.choices[0].message.content)
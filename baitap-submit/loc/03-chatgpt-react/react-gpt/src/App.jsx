import OpenAI from 'openai';
import { useState, useEffect } from "react"

let storedKey = localStorage.getItem("openai_api_key");
if (!storedKey) {
  storedKey = window.prompt("Nhập API Key (có thể để trống nếu dùng LM Studio local):");
  if (storedKey !== null) {
    localStorage.setItem("openai_api_key", storedKey);
  }
}

let openai = new OpenAI({
  apiKey: storedKey || "lm-studio",
  dangerouslyAllowBrowser: true,
  baseURL: "http://127.0.0.1:1234/v1"
});

function isBotMessage(chatMessage) {
  return chatMessage.role === 'assistant'
}

function App() {
  const [message, setMessage] = useState('')
  const [chatHistory, setChatHistory] = useState([])

  // Load chat history
  useEffect(() => {
    const saved = localStorage.getItem("chat_history");
    if (saved) {
      setChatHistory(JSON.parse(saved));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("chat_history", JSON.stringify(chatHistory));
  }, [chatHistory]);

  const submitForm = async (e) => {
    e.preventDefault()
    if (!message.trim()) return;

    setMessage('')

    const userMessage = { role: 'user', content: message }

    const newHistory = [...chatHistory, userMessage]

    setChatHistory([...newHistory, { role: 'assistant', content: "" }])

    const cleanMessages = newHistory.filter(m =>
      m.content && m.content.trim() !== ""
    );

    try {
      const stream = await openai.chat.completions.create({
        messages: cleanMessages,
        model: 'qwen3.5-4b',
        stream: true
      });

      let fullText = "";

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        fullText += delta;

        setChatHistory(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: fullText
          };
          return updated;
        });
      }

    } catch (err) {
      console.error(err);
      setChatHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: "Lỗi kết nối tới LM Studio."
        };
        return updated;
      });
    }
  }

  const clearChat = () => {
    setChatHistory([]);
    localStorage.removeItem("chat_history");
  }

  const changeApiKey = () => {
    const newKey = window.prompt("Nhập API Key mới (hoặc để trống nếu dùng local):");
    if (newKey !== null) {
      localStorage.setItem("openai_api_key", newKey);

      openai = new OpenAI({
        apiKey: newKey || "lm-studio",
        dangerouslyAllowBrowser: true,
        baseURL: "http://127.0.0.1:1234/v1"
      });

      alert("Đã cập nhật API Key!");
    }
  }

  return (
    <div className="bg-gray-100 h-screen flex flex-col">
      <div className="container mx-auto p-4 flex flex-col h-full max-w-2xl">
        <h1 className="text-2xl font-bold mb-4">ChatUI với React + OpenAI (LM Studio)</h1>

        <form className="flex" onSubmit={submitForm}>
          <input
            type="text"
            placeholder="Tin nhắn của bạn..."
            value={message}
            onChange={e => setMessage(e.target.value)}
            className="flex-grow p-2 rounded-l border border-gray-300"
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded-r hover:bg-blue-600"
          >
            Gửi tin nhắn
          </button>
        </form>

        <div className="flex gap-2 mt-2">
          <button
            onClick={clearChat}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Xóa lịch sử chat
          </button>

          <button
            onClick={changeApiKey}
            className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600"
          >
            Nhập lại API Key
          </button>
        </div>

        <div className="flex-grow overflow-y-auto mt-4 bg-white rounded shadow p-4">
          {chatHistory.map((chatMessage, i) => (
            <div
              key={i}
              className={`mb-2 ${isBotMessage(chatMessage) ? 'text-right' : ''}`}
            >
              <p className="text-gray-600 text-sm">
                {isBotMessage(chatMessage) ? 'Bot' : 'User'}
              </p>
              <p
                className={`p-2 rounded-lg inline-block ${
                  isBotMessage(chatMessage)
                    ? 'bg-green-100'
                    : 'bg-blue-100'
                }`}
              >
                {chatMessage.content}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default App
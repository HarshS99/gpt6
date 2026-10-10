<div align="center">

<img src="https://img.shields.io/badge/NexusAI-v3.0-8b5cf6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTUtMTAtNXoiLz48cGF0aCBkPSJNMiAxN2wxMCA1IDEwLTUiLz48cGF0aCBkPSJNMiAxMmwxMCA1IDEwLTUiLz48L3N2Zz4=&logoColor=white" alt="NexusAI" />

# NexusAI

### A production-ready, ChatGPT-like AI assistant — built with Groq, LangChain & FastAPI

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-8b5cf6?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-8b5cf6?style=flat-square&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Powered-3b82f6?style=flat-square&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-8b5cf6?style=flat-square&logoColor=white)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

</div>

---

## 🎨 UI Preview

```
┌────────────────────────────────────────────────────────────────┐
│  🟣 NexusAI                              ◉ NexusAI  🌙 🗑️    │
├──────────────────┬─────────────────────────────────────────────┤
│                  │                                             │
│  🔍 Search       │     ╭──────────────────────────────╮       │
│  ─────────────── │     │   🚀                         │       │
│  ➕ New Chat     │     │   How can I help you?         │       │
│                  │     │                               │       │
│  Today           │     │   Ask me anything — code,    │       │
│  ○ Chat 1        │     │   design, research or chat.   │       │
│  ● Chat 2 ←      │     │   Powered by Groq ⚡          │       │
│                  │     ╰──────────────────────────────╯       │
│  ─────────────── │                                             │
│  📄 Upload doc   │    🧠 Explain RAG  ⚡ FastAPI+JWT          │
│                  │    🐛 Debug async  🏗️ Microservices         │
│  👤 Harsh Singh  │                                             │
│  @harsh...       │  ╭─────────────────────────────────────╮   │
│                  │  │ 📎  Message NexusAI...          ➤  │   │
│                  │  ╰─────────────────────────────────────╯   │
└──────────────────┴─────────────────────────────────────────────┘

  Dark · Glassmorphism · Purple/Blue gradient · Streaming SSE
```

> **Design:** Dark glassmorphism · Ambient purple/blue orbs · Gradient accents · Floating logo animation · Smooth micro-interactions

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴 **Word-by-word Streaming** | Real-time SSE token streaming — just like ChatGPT |
| 🔐 **Auth System** | Register · Login · Logout with secure per-user sessions |
| 🧠 **RAG Pipeline** | Upload documents and chat with your own data |
| 💾 **Chat Memory** | Isolated conversation history per user session |
| 📄 **Document Upload** | PDF, TXT, DOCX, MD support with vector indexing |
| 🎨 **Glassmorphism UI** | Premium dark UI with purple/blue gradient accents |
| ⚡ **Groq-powered LLM** | Ultra-fast inference — responses in milliseconds |
| 🔍 **Vector Search** | ChromaDB + HuggingFace Sentence Transformers |
| 🌙 **Dark/Light Mode** | Toggle between dark and light themes |
| ⌨️ **Keyboard Shortcuts** | `Ctrl+N`, `Ctrl+B`, `Enter`, `Esc` and more |
| 📥 **Export Chats** | Download full conversation history |
| 🔢 **Markdown Rendering** | Code blocks, tables, lists, syntax highlighting |

---

## 🛠️ Tech Stack

**Backend**

| Tool | Purpose |
|------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | High-performance async Python web framework |
| [LangChain](https://langchain.com/) | LLM orchestration, prompt chaining & RAG |
| [Groq API](https://groq.com/) | Lightning-fast LLM inference endpoint |
| [ChromaDB](https://www.trychroma.com/) | Local vector database for document storage |
| [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers) | Text embedding model (`all-mpnet-base-v2`) |
| SSE (Server-Sent Events) | Real-time streaming without WebSocket overhead |

**Frontend**

| Tool | Purpose |
|------|---------|
| Vanilla HTML/CSS/JS | Zero-framework frontend for maximum speed |
| Glassmorphism CSS | `backdrop-filter: blur()` + gradient design system |
| `EventSource` / `ReadableStream` | Streaming token consumption |
| [marked.js](https://marked.js.org/) + [highlight.js](https://highlightjs.org/) | Markdown & syntax highlighting |
| LocalStorage | Client-side chat persistence |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A **[Groq API Key](https://console.groq.com/)** (free tier available)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/HarshS99/gpt6.git
cd gpt6
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the App

Go to **[http://localhost:8000](http://localhost:8000)** — register an account and start chatting!

---

## 📁 Project Structure

```
gpt6/
│
├── main.py                          # FastAPI — routes, auth, streaming, RAG
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (not committed)
├── .gitignore
├── README.md
│
├── static/                          # Frontend (served by FastAPI)
│   ├── index.html                   # Main SPA — auth, chat, modals
│   ├── style.css                    # Glassmorphism dark-mode design system
│   └── app.js                       # Chat logic, streaming, auth, sidebar
│
└── modules/                         # Backend logic modules
    ├── ai_model/                    # LLM configuration & wrappers
    ├── file_processing/             # Document parsing (PDF, DOCX, TXT)
    ├── vector_database/             # ChromaDB vector store
    ├── rag_pipeline/                # Retrieval-Augmented Generation chain
    ├── memory/                      # Chat message history
    ├── langgraph_workflow/          # LangGraph agent workflow
    ├── llm/                         # LLM provider abstraction
    └── utilities/                   # Shared helpers
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register a new user account |
| `POST` | `/api/auth/login` | Login and receive session token |
| `POST` | `/api/auth/logout` | Invalidate session token |
| `POST` | `/api/session` | Create a new chat session |
| `POST` | `/api/chat/stream` | **SSE streaming chat** (main endpoint) |
| `POST` | `/api/upload` | Upload document for RAG indexing |
| `GET`  | `/` | Serve the SPA frontend |

---

## 🧠 RAG Architecture

```
 User uploads document
         │
         ▼
 [File Processing] ──► Text extraction ──► Chunking
         │
         ▼
 [HuggingFace Embeddings]
         │
         ▼
 [ChromaDB] ◄──── stored locally
         │
         │  (when user asks a question)
         ▼
 [Query Embedding] ──► Cosine similarity search ──► Top-k chunks
         │
         ▼
 [LangChain Prompt]
 "Given context: {chunks}, answer: {question}"
         │
         ▼
 [Groq LLM] ──SSE──► Streams response word-by-word to browser
```

---

## ⚙️ Model Configuration

Edit `main.py` to change the LLM:

```python
def build_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model_name="qwen/qwen3.8-27b",   # ← Change here
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=800,
        streaming=streaming,
    )
```

> Check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the latest available models.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift + Enter` | New line |
| `Ctrl + N` | New chat |
| `Ctrl + B` | Toggle sidebar |
| `Ctrl + L` | Clear chat |
| `Ctrl + E` | Export chat |
| `Esc` | Stop generating |

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `[Errno 48] Address already in use` | Run `kill -9 $(lsof -t -i:8000)` then retry |
| `model_not_found` / `model_decommissioned` | Update model in `main.py` — check [Groq deprecations](https://console.groq.com/docs/deprecations) |
| `rate_limit_exceeded` | Lower `max_tokens` in `main.py` or upgrade Groq tier |
| `ERR_CONNECTION_REFUSED` | Server not running — run `uvicorn main:app --port 8000 --reload` |
| `HuggingFace` slow startup | Normal on first run — downloads embedding model once |

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 💜 by [Harsh Singh](https://github.com/HarshS99)**

[![GitHub](https://img.shields.io/badge/GitHub-HarshS99-8b5cf6?style=flat-square&logo=github)](https://github.com/HarshS99)

⭐ **Star this repo if you found it useful!** ⭐

</div>
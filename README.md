<div align="center">

# 🤖 NexusAI

### A production-ready, ChatGPT-like AI chatbot powered by Groq & LangChain

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange?logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-API-purple?logoColor=white)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![NexusAI Demo](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴 **Streaming Responses** | Word-by-word token streaming just like ChatGPT |
| 🔐 **Auth System** | Register, Login, Logout with secure token sessions |
| 🧠 **RAG Pipeline** | Upload documents and chat with your own data |
| 💾 **Per-user Memory** | Isolated chat history per user session |
| 📄 **Document Upload** | Support for PDF, TXT, DOCX and more |
| 🌐 **Modern UI** | Glassmorphism dark-mode UI inspired by ChatGPT |
| ⚡ **Groq-powered** | Ultra-fast LLM inference via Groq API |
| 🔍 **Vector Search** | ChromaDB with HuggingFace embeddings |

---

## 📸 Screenshots

> A sleek dark-mode interface with a sidebar, streaming chat, user auth, and document uploads.

```
┌──────────────────────────────────────────────────────────┐
│  NexusAI                                         [Logout] │
├──────────────┬───────────────────────────────────────────┤
│  🔍 Search   │                                           │
│  ─────────── │         Ask me anything...                │
│  📁 New Chat │                                           │
│  📄 Upload   │  ┌─────────────────────────────────────┐  │
│  ─────────── │  │ H  hi                               │  │
│  Today       │  └─────────────────────────────────────┘  │
│  > Chat 1    │  ┌─────────────────────────────────────┐  │
│  > Chat 2    │  │ 🤖 Hello! How can I help you today? │  │
│              │  └─────────────────────────────────────┘  │
│  H Harsh     │  ─────────────────────────────────────── │
│  @email      │  [ Message NexusAI...              Send ] │
└──────────────┴───────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance Python web framework
- [LangChain](https://langchain.com/) — LLM orchestration and RAG pipelines
- [Groq API](https://groq.com/) — Lightning-fast LLM inference
- [ChromaDB](https://www.trychroma.com/) — Local vector database for document storage
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers) — Embedding model
- [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — Real-time streaming

**Frontend**
- Vanilla HTML, CSS & JavaScript — Zero framework dependencies
- Glassmorphism dark-mode design
- `EventSource` / `ReadableStream` API for token streaming
- LocalStorage-based chat persistence

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A **[Groq API Key](https://console.groq.com/)** (free tier available)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/HarshS99/gpt6.git
cd gpt6
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

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

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📁 Project Structure

```
gpt6/
├── main.py                          # FastAPI app — routes, auth, streaming, RAG
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (not committed)
├── .gitignore
│
├── static/                          # Frontend (served by FastAPI)
│   ├── index.html                   # Main UI page
│   ├── style.css                    # Glassmorphism dark-mode styles
│   └── app.js                       # Chat logic, streaming, auth, sidebar
│
└── modules/                         # Backend logic modules
    ├── ai_model/                    # LLM configuration and wrappers
    ├── file_processing/             # Document parsing (PDF, DOCX, TXT)
    ├── vector_database/             # ChromaDB vector store setup
    ├── rag_pipeline/                # Retrieval-Augmented Generation chain
    ├── memory/                      # Chat message history management
    ├── langgraph_workflow/          # LangGraph agent workflow
    ├── llm/                         # LLM provider abstraction
    └── utilities/                   # Shared helpers and utilities
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Log in and receive a token |
| `POST` | `/api/auth/logout` | Invalidate session token |
| `POST` | `/api/session` | Create a new chat session |
| `POST` | `/api/chat/stream` | **SSE streaming chat endpoint** |
| `POST` | `/api/upload` | Upload a document for RAG |
| `GET`  | `/`       | Serve the frontend |

---

## 🧠 How RAG Works in This Project

```
User uploads a document
        │
        ▼
[File Processing] → Text extracted → Chunked into passages
        │
        ▼
[HuggingFace Embeddings] → Vector representation of each chunk
        │
        ▼
[ChromaDB] → Stored locally in chroma_db/
        │
When user asks a question:
        │
        ▼
[Query Embedding] → Similarity search in ChromaDB → Top-k chunks retrieved
        │
        ▼
[LangChain Prompt] → "Given context: {chunks}, answer: {question}"
        │
        ▼
[Groq LLM] → Streams grounded response back to user
```

---

## ⚙️ Configuration

You can change the model in `main.py`:

```python
def build_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model_name="qwen/qwen3.8-27b",   # ← Change model here
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=800,
        streaming=streaming,
    )
```

**Available Groq models** (check your tier at [console.groq.com/docs/models](https://console.groq.com/docs/models)):
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `qwen/qwen3.8-27b`
- `gemma2-9b-it`

---

## 🐛 Common Issues

| Error | Fix |
|-------|-----|
| `[Errno 48] Address already in use` | Run `kill -9 $(lsof -t -i:8000)` then retry |
| `model_not_found` / `model_decommissioned` | Update the model name in `main.py` — check [Groq docs](https://console.groq.com/docs/deprecations) |
| `rate_limit_exceeded` | Lower `max_tokens` in `main.py`, or upgrade your Groq tier |
| `ERR_CONNECTION_REFUSED` | The server isn't running — run `uvicorn main:app --port 8000 --reload` |

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Harsh Singh**

[![GitHub](https://img.shields.io/badge/GitHub-HarshS99-black?logo=github)](https://github.com/HarshS99)

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

</div>
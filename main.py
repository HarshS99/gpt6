"""
NexusAI – FastAPI Backend v2
Features: Streaming SSE · Auth (login/register/logout) · RAG · Per-user sessions
"""
import os
import uuid
import json
import secrets
import hashlib
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import aiofiles
from modules.file_processing.file_processing import DocumentProcessor
from modules.vector_database.vector_database import VectorDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NexusAI API", version="2.0.0")

# ══════════════════════════════════════════════════════════
#  AUTH  ─ simple token-based, in-memory
# ══════════════════════════════════════════════════════════

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Pre-seeded users
users_db: dict = {
    "admin": {"password_hash": _hash("admin123"), "display_name": "Admin", "avatar": "A"},
    "harsh": {"password_hash": _hash("harsh123"), "display_name": "Harsh Singh", "avatar": "H"},
}

# token → username
active_tokens: dict[str, str] = {}


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Validate Bearer token and return username."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization[7:]
    username = active_tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username


# ══════════════════════════════════════════════════════════
#  SESSION STORE
# ══════════════════════════════════════════════════════════

sessions: dict[str, ChatMessageHistory] = {}
sessions_meta: dict[str, dict] = {}   # {id: {label, username, created_at}}


def get_or_create_session(session_id: str) -> ChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]


# ══════════════════════════════════════════════════════════
#  VECTOR DB & DOC PROCESSOR
# ══════════════════════════════════════════════════════════

vector_db = VectorDatabase(persist_directory="chroma_db")
try:
    vector_db.initialize_collection()
except Exception as exc:
    logger.error(f"Vector DB init error: {exc}")

doc_processor = DocumentProcessor()


# ══════════════════════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are NexusAI, a helpful, creative, and blazing-fast AI assistant. "
    "You provide clear, accurate, and detailed responses. "
    "Use the following retrieved context (if any) to answer the user's question.\n\n"
    "Context: {context}\n\n"
    "If you don't know the answer, be honest about it."
)


def build_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model_name="qwen/qwen3.8-27b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=800,
        streaming=streaming,
    )


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])


def fetch_context(message: str, session_id: str) -> str:
    try:
        docs = vector_db.similarity_search(message, k=3, filter={"session_id": session_id})
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as exc:
        logger.error(f"RAG search error: {exc}")
        return ""


# ══════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[dict]


# ══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = users_db.get(req.username)
    if not user or user["password_hash"] != _hash(req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_hex(32)
    active_tokens[token] = req.username
    logger.info(f"User '{req.username}' logged in")
    return {
        "token": token,
        "username": req.username,
        "display_name": user["display_name"],
        "avatar": user["avatar"],
    }


@app.post("/api/auth/logout")
async def logout(
    username: str = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
):
    if authorization:
        token = authorization[7:]
        active_tokens.pop(token, None)
    logger.info(f"User '{username}' logged out")
    return {"status": "logged out"}


@app.get("/api/auth/me")
async def me(username: str = Depends(get_current_user)):
    user = users_db.get(username, {})
    return {
        "username": username,
        "display_name": user.get("display_name", username),
        "avatar": user.get("avatar", username[0].upper()),
    }


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    uname = req.username.strip().lower()
    if not uname or len(uname) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if uname in users_db:
        raise HTTPException(status_code=400, detail="Username already taken")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    display = req.display_name.strip() or uname.capitalize()
    users_db[uname] = {
        "password_hash": _hash(req.password),
        "display_name": display,
        "avatar": display[0].upper(),
    }
    logger.info(f"New user registered: {uname}")
    return {"status": "ok", "username": uname}


# ══════════════════════════════════════════════════════════
#  SESSION ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/api/sessions")
async def list_sessions(username: str = Depends(get_current_user)):
    result = []
    for sid, history in sessions.items():
        meta = sessions_meta.get(sid, {})
        if meta.get("username") == username:
            result.append({
                "session_id": sid,
                "label": meta.get("label", "New chat"),
                "message_count": len(history.messages),
                "created_at": meta.get("created_at", ""),
            })
    # Sort by creation time descending
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"sessions": result}


@app.post("/api/session")
async def create_session(username: str = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    sessions[session_id] = ChatMessageHistory()
    sessions_meta[session_id] = {
        "label": "New chat",
        "username": username,
        "created_at": datetime.now().isoformat(),
    }
    return {"session_id": session_id}


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, username: str = Depends(get_current_user)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    history = sessions[session_id]
    serialized = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history.messages
    ]
    meta = sessions_meta.get(session_id, {})
    return {"session_id": session_id, "label": meta.get("label", "New chat"), "history": serialized}


@app.patch("/api/session/{session_id}/label")
async def update_session_label(
    session_id: str, body: dict, username: str = Depends(get_current_user)
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    label = body.get("label", "").strip()
    if label:
        sessions_meta.setdefault(session_id, {})["label"] = label
    return {"status": "ok"}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, username: str = Depends(get_current_user)):
    sessions.pop(session_id, None)
    sessions_meta.pop(session_id, None)
    return {"status": "cleared"}


# ══════════════════════════════════════════════════════════
#  CHAT – STREAMING SSE  (primary endpoint)
# ══════════════════════════════════════════════════════════

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, username: str = Depends(get_current_user)):
    """Stream AI response token-by-token using SSE."""
    session_id = req.session_id or str(uuid.uuid4())
    history = get_or_create_session(session_id)

    # Ensure session meta exists for this user
    if session_id not in sessions_meta:
        sessions_meta[session_id] = {
            "label": "New chat",
            "username": username,
            "created_at": datetime.now().isoformat(),
        }

    context_text = fetch_context(req.message, session_id)

    # Snapshot to avoid mutation during streaming
    history_snapshot = list(history.messages)
    msg_snapshot = req.message
    ctx_snapshot = context_text

    async def generate():
        full_response: list[str] = []
        try:
            # First event: send session_id so frontend can track
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"

            llm = build_llm(streaming=True)
            chain = build_prompt() | llm

            async for chunk in chain.astream({
                "chat_history": history_snapshot,
                "input": msg_snapshot,
                "context": ctx_snapshot,
            }):
                if chunk.content:
                    full_response.append(chunk.content)
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"

            # Persist to history
            complete = "".join(full_response)
            history.add_user_message(msg_snapshot)
            history.add_ai_message(complete)

            # Auto-label session from first message
            meta = sessions_meta.get(session_id, {})
            if meta.get("label") == "New chat":
                label = msg_snapshot[:42] + ("…" if len(msg_snapshot) > 42 else "")
                meta["label"] = label
                sessions_meta[session_id] = meta

            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════
#  CHAT – NON-STREAMING FALLBACK
# ══════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, username: str = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = get_or_create_session(session_id)
    context_text = fetch_context(req.message, session_id)

    try:
        chain = build_prompt() | build_llm()
        response = chain.invoke({
            "chat_history": history.messages,
            "input": req.message,
            "context": context_text,
        })
        history.add_user_message(req.message)
        history.add_ai_message(response.content)

        serialized = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in history.messages
        ]
        return {"session_id": session_id, "reply": response.content, "history": serialized}
    except Exception as exc:
        logger.error(f"/api/chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════
#  FILE UPLOAD
# ══════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    username: str = Depends(get_current_user),
):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    async with aiofiles.open(dest, "wb") as out:
        await out.write(await file.read())

    try:
        docs = doc_processor.process_document(dest)
        for d in docs:
            d.metadata["session_id"] = session_id
        if docs:
            vector_db.add_documents(docs)
        return {"filename": file.filename, "status": "uploaded and indexed", "chunks": len(docs)}
    except Exception as exc:
        logger.error(f"Upload error {file.filename}: {exc}")
        return {"filename": file.filename, "status": "error", "message": str(exc)}


# ══════════════════════════════════════════════════════════
#  SERVE FRONTEND
# ══════════════════════════════════════════════════════════

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

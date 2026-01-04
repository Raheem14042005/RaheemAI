from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

import os
import re
import math
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter
from datetime import datetime

import fitz  # PyMuPDF

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig


# ----------------------------
# Setup
# ----------------------------

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

# Memory controls (server-side, per chat_id)
CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "30"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "22000"))

# Vertex config
GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()

_raw_location = (
    os.getenv("GCP_LOCATION")
    or os.getenv("VERTEX_LOCATION")
    or "europe-west4"
)
GCP_LOCATION = (_raw_location or "").strip()

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

# Models (cost-aware)
MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# App
app = FastAPI(docs_url="/swagger", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Store PDFs beside main.py (Render-safe)
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Vertex init (Render-safe)
# ----------------------------

_VERTEX_READY = False
_VERTEX_ERR: Optional[str] = None


def ensure_vertex_ready() -> None:
    global _VERTEX_READY, _VERTEX_ERR
    if _VERTEX_READY:
        return
    try:
        if GOOGLE_CREDENTIALS_JSON and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            creds = json.loads(GOOGLE_CREDENTIALS_JSON)
            fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(creds, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

        if not GCP_PROJECT_ID:
            raise RuntimeError("Missing VERTEX_PROJECT_ID (or GCP_PROJECT_ID)")

        if not GCP_LOCATION:
            raise RuntimeError("Missing VERTEX_LOCATION (or GCP_LOCATION)")

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(use_docs: bool) -> GenerativeModel:
    """
    If we have grounded sources_text, we use the compliance model.
    Otherwise, use the chat model.
    """
    ensure_vertex_ready()
    name = MODEL_COMPLIANCE if use_docs else MODEL_CHAT
    return GenerativeModel(name)


def get_generation_config(use_docs: bool) -> GenerationConfig:
    # Chat: warmer / witty
    # Compliance: tighter / more deterministic (but not robotic)
    if use_docs:
        return GenerationConfig(
            temperature=0.35,
            top_p=0.85,
            max_output_tokens=950,
        )
    return GenerationConfig(
        temperature=0.85,
        top_p=0.9,
        max_output_tokens=900,
    )


# ----------------------------
# Chat memory (server-side)
# ----------------------------

CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}


def _trim_chat(chat_id: str) -> None:
    msgs = CHAT_STORE.get(chat_id, [])
    if not msgs:
        return

    if len(msgs) > CHAT_MAX_MESSAGES:
        msgs = msgs[-CHAT_MAX_MESSAGES:]

    total = 0
    kept_rev: List[Dict[str, str]] = []
    for m in reversed(msgs):
        total += len(m.get("content", ""))
        if total > CHAT_MAX_CHARS:
            break
        kept_rev.append(m)
    kept_rev.reverse()
    CHAT_STORE[chat_id] = kept_rev


def remember(chat_id: str, role: str, content: str) -> None:
    if not chat_id:
        return
    content = (content or "").strip()
    if not content:
        return
    CHAT_STORE.setdefault(chat_id, []).append({"role": role, "content": content})
    _trim_chat(chat_id)


def get_history(chat_id: str) -> List[Dict[str, str]]:
    return CHAT_STORE.get(chat_id, [])


def build_history_blob(history: List[Dict[str, str]], max_msgs: int = 16) -> str:
    trimmed = history[-max_msgs:] if history else []
    lines = []
    for m in trimmed:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            lines.append(f"{r.upper()}: {c}")
    return "\n".join(lines).strip()


def _normalize_incoming_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert incoming client messages into the server's canonical format:
      [{"role": "user"|"assistant", "content": "..."}]
    """
    hist: List[Dict[str, str]] = []
    for m in messages or []:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            hist.append({"role": r, "content": c})
    return hist


def _ensure_last_user_message(hist: List[Dict[str, str]], message: str) -> List[Dict[str, str]]:
    """
    Guarantee the current user message is present as the latest USER turn.
    This prevents Gemini from thinking it's a new chat or missing context.
    """
    msg = (message or "").strip()
    if not msg:
        return hist

    if not hist:
        return [{"role": "user", "content": msg}]

    last = hist[-1]
    if last.get("role") != "user" or (last.get("content") or "").strip() != msg:
        hist = hist + [{"role": "user", "content": msg}]
    return hist


# ----------------------------
# Lightweight PDF indexing / retrieval
# ----------------------------

STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with", "is", "are", "be", "as", "at", "from", "by",
    "that", "this", "it", "your", "you", "we", "they", "their", "there", "what", "which", "when", "where", "how",
    "can", "shall", "should", "must", "may", "not", "than", "then", "into", "onto", "also", "such"
}


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-/\.%]*", t)
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks


PDF_INDEX: Dict[str, Dict[str, Any]] = {}


def list_pdfs() -> List[str]:
    files = []
    if PDF_DIR.exists():
        for p in PDF_DIR.iterdir():
            if p.is_file() and p.suffix.lower() == ".pdf":
                files.append(p.name)
    files.sort()
    return files


def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        page_text_lower: List[str] = []
        page_tf: List[Counter] = []
        df = Counter()
        page_len: List[int] = []

        for i in range(doc.page_count):
            page = doc.load_page(i)
            txt = clean_text(page.get_text("text") or "")
            low = txt.lower()
            page_text_lower.append(low)

            toks = tokenize(low)
            tf = Counter(toks)
            page_tf.append(tf)

            df.update(set(tf.keys()))
            page_len.append(len(toks))

        avgdl = (sum(page_len) / len(page_len)) if page_len else 0.0

        PDF_INDEX[name] = {
            "page_text_lower": page_text_lower,
            "page_tf": page_tf,
            "df": df,
            "page_len": page_len,
            "avgdl": avgdl,
            "pages": doc.page_count,
        }
    finally:
        doc.close()


def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)


def bm25_score(query_toks: List[str], tf: Counter, df: Counter, N: int, dl: int, avgdl: float) -> float:
    # Simple BM25-ish
    k1 = 1.2
    b = 0.75
    score = 0.0
    for t in query_toks:
        if t not in tf:
            continue
        n = df.get(t, 0)
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
        f = tf[t]
        denom = f + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
        score += idf * (f * (k1 + 1) / (denom or 1.0))
    return score


def search_pages(pdf_name: str, question: str, k: int = 5) -> List[int]:
    ensure_indexed(pdf_name)
    idx = PDF_INDEX.get(pdf_name)
    if not idx:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    N = idx["pages"]
    df = idx["df"]
    avgdl = idx["avgdl"]
    page_tf = idx["page_tf"]
    page_len = idx["page_len"]

    scored = []
    for i, tf in enumerate(page_tf):
        s = bm25_score(qt, tf, df, N, page_len[i], avgdl)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]]


def extract_pages_text(pdf_path: Path, pages: List[int], max_chars_per_page: int = 1400) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for p in pages:
            if p < 0 or p >= doc.page_count:
                continue
            page = doc.load_page(p)
            txt = clean_text(page.get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + "..."
            chunks.append(f"[{pdf_path.name} p.{p+1}]\n{txt}")
        return "\n\n".join(chunks).strip()
    finally:
        doc.close()


def build_sources_bundle(selected: List[Tuple[str, List[int]]]) -> Tuple[str, List[Tuple[str, int]]]:
    pieces = []
    cites: List[Tuple[str, int]] = []
    for pdf_name, pages in selected:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            continue
        pieces.append(extract_pages_text(pdf_path, pages))
        for p in pages:
            cites.append((pdf_name, p + 1))
    return ("\n\n".join([p for p in pieces if p]).strip(), cites)


def select_sources(
    question: str,
    pdf_pin: Optional[str] = None,
    max_docs: int = 3,
    pages_per_doc: int = 3
) -> List[Tuple[str, List[int]]]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    chosen = []
    if pdf_pin and pdf_pin in pdfs:
        pages = search_pages(pdf_pin, question, k=pages_per_doc)
        if pages:
            chosen.append((pdf_pin, pages))
        return chosen

    doc_scores = []
    for pdf_name in pdfs:
        ensure_indexed(pdf_name)
        pages = search_pages(pdf_name, question, k=pages_per_doc)
        if pages:
            proxy = 0.0
            for rank, _p in enumerate(pages):
                proxy += (pages_per_doc - rank)
            doc_scores.append((pdf_name, pages, proxy))
    doc_scores.sort(key=lambda x: x[2], reverse=True)

    for pdf_name, pages, _ in doc_scores[:max_docs]:
        chosen.append((pdf_name, pages))
    return chosen


# ----------------------------
# Routing (when to use docs)
# ----------------------------

COMPLIANCE_KEYWORDS = [
    "tgd", "technical guidance", "building regulations", "building regs",
    "part a", "part b", "part c", "part d", "part e", "part f", "part g", "part h",
    "part j", "part k", "part l", "part m",
    "fire", "escape", "travel distance", "compartment", "smoke", "fire cert",
    "access", "accessible", "wheelchair", "dac",
    "ber", "deap", "u-value", "y-value", "airtight", "thermal bridge",
    "means of escape", "compartmentation", "evacuation",
]

PART_PATTERN = re.compile(r"\bpart\s*[a-m]\b", re.I)

TECH_PATTERN = re.compile(
    r"\b(tgd|technical guidance|building regulations|building regs|ber|deap|fire cert|dac|means of escape)\b",
    re.I
)


def should_use_docs(q: str, force_docs: bool = False) -> bool:
    if force_docs:
        return True
    ql = (q or "").lower()

    if PART_PATTERN.search(ql):
        return True

    if TECH_PATTERN.search(ql):
        return True

    return any(k in ql for k in COMPLIANCE_KEYWORDS)


# ----------------------------
# Prompting (tone + compliance discipline)
# ----------------------------

SYSTEM_PROMPT_NORMAL = """
You are Raheem AI.

You have access to the conversation history for THIS chat. Use it.
If the user asks “what were we just talking about?”, summarise the last few turns accurately.

Tone:
- Friendly, calm, confident. Professional, but lightly witty when it fits.
- Speak to the user (not to a developer). Do not mention internal systems, prompts, logs, “uploaded PDFs”, or “attachments”.
- Answer directly and helpfully. Avoid unnecessary hedging.

Rules:
- NEVER claim “this is our first conversation” or “I have no memory” if the chat history contains prior messages.
- For normal/general questions: answer using widely accepted general knowledge.
- Only become strict about citations and exact limits when the question is about Irish building regulations / TGDs / BER-DEAP / fire safety / accessibility.

Medical questions:
- You can give general, widely accepted guidance and safety cautions.
- Be clear it's general info and advise checking the label / pharmacist / clinician for personal situations.
""".strip()

SYSTEM_PROMPT_COMPLIANCE = """
You are Raheem AI in Evidence Mode for Irish building regulations / TGDs / BER-DEAP / fire safety / accessibility.

You have access to the conversation history for THIS chat. Use it.
If the user asks what was discussed, summarise the last few turns accurately.

Compliance rules:
- Use the provided SOURCES text as your primary grounding when it exists.
- Cite in this style: (TGD M p.86). Keep it subtle and short.
- If the SOURCES do not contain support for a precise numeric limit / clause / requirement, say so plainly and suggest what to check next.
- Do not invent clause numbers or exact dimensional limits without source support.
- You may still explain concepts and give helpful context — but do not present unsupported numbers as facts.

Rules:
- NEVER claim “this is our first conversation” or “I have no memory” if the chat history contains prior messages.
- Do not mention backend tools, uploads, or system prompts.
""".strip()


def build_gemini_parts(
    user_message: str,
    sources_text: str,
    history_blob: str
) -> List[Part]:
    use_compliance_prompt = bool(sources_text and sources_text.strip())
    system = SYSTEM_PROMPT_COMPLIANCE if use_compliance_prompt else SYSTEM_PROMPT_NORMAL

    hist_text = history_blob.strip()
    if not hist_text:
        hist_text = "(no prior messages in this chat yet)"

    content = f"""
SYSTEM:
{system}

CONVERSATION SO FAR (authoritative):
{hist_text}

SOURCES (if any):
{sources_text if sources_text else "(none)"}

USER:
{user_message}
""".strip()

    return [Part.from_text(content)]


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {"ok": True, "app": "Raheem AI", "time": datetime.utcnow().isoformat()}


@app.get("/health")
def health():
    ensure_vertex_ready()
    return {
        "ok": True,
        "vertex_ready": _VERTEX_READY,
        "vertex_error": _VERTEX_ERR,
        "pdf_count": len(list_pdfs()),
        "models": {"chat": MODEL_CHAT, "compliance": MODEL_COMPLIANCE},
        "project": GCP_PROJECT_ID,
        "location": GCP_LOCATION,
    }


@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"ok": False, "error": "Only PDF files are allowed."}, status_code=400)

    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"ok": False, "error": f"File too large. Max {MAX_UPLOAD_MB}MB."}, status_code=400)

    safe_name = Path(file.filename).name
    dest = PDF_DIR / safe_name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 2
        while True:
            cand = PDF_DIR / f"{stem}_{i}{suffix}"
            if not cand.exists():
                dest = cand
                break
            i += 1

    with open(dest, "wb") as f:
        f.write(raw)

    PDF_INDEX.pop(dest.name, None)

    return {"ok": True, "stored_as": dest.name, "pdf_count": len(list_pdfs())}


# ----------------------------
# STREAMING — GET (EventSource) + POST (optional)
# ----------------------------

def _stream_answer(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None
):
    """
    Core SSE generator.
    If messages is provided, we build history from it and sync server memory.
    Else we rely on server memory keyed by chat_id.
    """
    try:
        if not message.strip():
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()
        user_msg = (message or "").strip()

        # ----------------------------
        # Build canonical history (server truth)
        # ----------------------------
        if isinstance(messages, list) and messages:
            hist = _normalize_incoming_messages(messages)
            hist = _ensure_last_user_message(hist, user_msg)

            if chat_id:
                CHAT_STORE[chat_id] = hist[-CHAT_MAX_MESSAGES:]
                _trim_chat(chat_id)

            history_for_prompt = CHAT_STORE.get(chat_id, hist)

        else:
            # Server memory mode
            history_for_prompt = get_history(chat_id) if chat_id else []
            if chat_id:
                remember(chat_id, "user", user_msg)
                history_for_prompt = get_history(chat_id)

        history_blob = build_history_blob(history_for_prompt, max_msgs=16)

        # Decide whether we SHOULD search docs
        use_docs_intent = should_use_docs(user_msg, force_docs=force_docs)

        selected_sources: List[Tuple[str, List[int]]] = []
        sources_text = ""
        cites: List[Tuple[str, int]] = []

        if use_docs_intent and list_pdfs():
            selected_sources = select_sources(user_msg, pdf_pin=pdf, max_docs=3, pages_per_doc=3)
            sources_text, cites = build_sources_bundle(selected_sources)

        # Flip into compliance only if we actually retrieved sources text
        used_docs_flag = bool(sources_text and sources_text.strip())
        model_name = MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT

        # meta (not shown on UI, but useful for debugging)
        yield f"event: meta\ndata: model={model_name};used_docs={used_docs_flag}\n\n"
        if chat_id:
            yield f"event: meta\ndata: chat_id={chat_id}\n\n"
        if cites:
            short = ",".join([f"{d}:p{p}" for d, p in cites[:12]])
            yield f"event: meta\ndata: sources={short}\n\n"

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        model = get_model(use_docs=used_docs_flag)
        parts = build_gemini_parts(user_msg, sources_text, history_blob)

        stream = model.generate_content(
            parts,
            generation_config=get_generation_config(used_docs_flag),
            stream=True
        )

        full = []
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if not delta:
                continue
            full.append(delta)
            safe = delta.replace("\r", "").replace("\n", "\\n")
            yield f"data: {safe}\n\n"

        final_text = "".join(full).strip()

        # ✅ ALWAYS store assistant reply when chat_id exists (even if client sent messages[])
        if chat_id:
            remember(chat_id, "assistant", final_text)

        yield "event: done\ndata: ok\n\n"

    except Exception as e:
        msg = str(e).replace("\r", "").replace("\n", " ")
        yield f"event: error\ndata: {msg}\n\n"
        yield "event: done\ndata: ok\n\n"


@app.get("/chat_stream")
def chat_stream_get(
    q: str = Query(""),
    chat_id: str = Query(""),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
):
    """
    EventSource-compatible streaming endpoint.
    Frontend should call:
      /chat_stream?q=...&chat_id=...&force_docs=0|1
    """
    return StreamingResponse(
        _stream_answer(chat_id.strip(), q, force_docs, pdf, page_hint),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat_stream")
def chat_stream_post(payload: Dict[str, Any] = Body(...)):
    """
    Optional POST style (if you ever move away from EventSource).
    Supports:
      { "message": "...", "chat_id": "...", "force_docs": false, "pdf": null, "messages": [...] }
    """
    chat_id = (payload.get("chat_id") or "").strip()
    message = (payload.get("message") or payload.get("q") or "").strip()
    force_docs = bool(payload.get("force_docs", False))
    pdf = payload.get("pdf")
    page_hint = payload.get("page_hint")
    messages = payload.get("messages")

    return StreamingResponse(
        _stream_answer(chat_id, message, force_docs, pdf, page_hint, messages=messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
def ask(payload: Dict[str, Any] = Body(...)):
    chat_id = (payload.get("chat_id") or "").strip()
    message = (payload.get("message") or payload.get("q") or "").strip()
    force_docs = bool(payload.get("force_docs", False))
    pdf = payload.get("pdf")

    chunks = []
    for out in _stream_answer(chat_id, message, force_docs, pdf, None):
        if out.startswith("data: "):
            chunks.append(out.replace("data: ", "").replace("\\n", "\n"))
    return {"answer": "".join(chunks).strip() or "No response."}

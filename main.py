"""
main.py — FastAPI backend for a “normal AI first” assistant that becomes an expert in your PDFs.

Key behavior:
- Normal, friendly, slightly positive + professionally humorous by default.
- Only pulls from PDFs when the question likely needs them (router).
- When PDFs are used: retrieves the most relevant pages, cites (Doc p.X), avoids hallucinating.
- Uses vision (page images) only when likely needed (tables/diagrams) and caches renders to cut cost.

Deploy notes (Render):
- If you have a Render Disk mounted at /var/data, set:
    PDF_DIR=/var/data/pdfs
  Otherwise it defaults to ./pdfs (ephemeral on Render).
- Required env var:
    OPENAI_API_KEY=...
- Optional:
    MODEL_NAME=gpt-4o-mini
    ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
    MAX_UPLOAD_MB=80
    IMAGE_CACHE_MAX=64
"""

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from openai import OpenAI

import os
import re
import base64
from pathlib import Path
import fitz  # PyMuPDF
import traceback
from typing import List, Optional, Dict, Tuple, Any
from collections import OrderedDict

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))
IMAGE_CACHE_MAX = int(os.getenv("IMAGE_CACHE_MAX", "64"))

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# Storage paths
# ----------------------------

def choose_pdf_dir() -> Path:
    # Prefer a persistent Render Disk path if available; otherwise fallback.
    env_dir = os.getenv("PDF_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    preferred = Path("/var/data/pdfs")  # works if Render Disk is mounted
    fallback = BASE_DIR / "pdfs"

    if os.getenv("RENDER"):
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except Exception:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

PDF_DIR = choose_pdf_dir()


# ----------------------------
# PDF Index (simple + fast)
# ----------------------------

PDF_TEXT_INDEX: Dict[str, List[str]] = {}  # doc -> list of cleaned lower text per page

STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with", "is", "are", "be", "as", "at",
    "from", "by", "that", "this", "it", "your", "you", "we", "they", "their", "there", "what", "which",
    "when", "where", "how", "can", "shall", "should", "must", "may", "not", "than", "then"
}

def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    doc = fitz.open(pdf_path)
    page_texts: List[str] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            txt = _clean_text(page.get_text("text") or "")
            page_texts.append(txt.lower())
    finally:
        doc.close()
    PDF_TEXT_INDEX[name] = page_texts

def index_all_pdfs() -> None:
    for p in PDF_DIR.glob("*.pdf"):
        try:
            index_pdf(p)
        except Exception:
            continue

def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_TEXT_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)

index_all_pdfs()


# ----------------------------
# Image cache (for vision pages)
# ----------------------------

# key: (pdf_name, page_index, dpi) -> data_url
_IMAGE_CACHE: "OrderedDict[Tuple[str,int,int], str]" = OrderedDict()

def _cache_get(k):
    if k in _IMAGE_CACHE:
        _IMAGE_CACHE.move_to_end(k)
        return _IMAGE_CACHE[k]
    return None

def _cache_set(k, v):
    _IMAGE_CACHE[k] = v
    _IMAGE_CACHE.move_to_end(k)
    while len(_IMAGE_CACHE) > IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)


# ----------------------------
# Router: decide if docs are needed
# ----------------------------

def should_use_docs(question: str) -> bool:
    """
    Simple, cheap router.
    Keeps the assistant 'normal AI' unless the user likely wants doc-grounded answers.
    """
    q = (question or "").lower()

    doc_triggers = [
        "according to", "in the document", "in the pdf", "pdf", "doc ", "document",
        "what does it say", "where does it say", "cite", "citation", "page", "clause", "section",
        "tgd", "technical guidance", "part b", "part m", "part l", "deap", "seai",
        "table", "diagram", "figure", "fig.", "chart", "graph", "schedule", "appendix"
    ]
    return any(t in q for t in doc_triggers)


# ----------------------------
# Retrieval over PDFs (no embeddings; cheap and decent)
# ----------------------------

def tokenize(q: str) -> List[str]:
    q = (q or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-/\.]*", q)
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

def score_page(page_text: str, tokens: List[str], full_q: str) -> int:
    if not page_text:
        return 0
    score = 0
    for t in tokens:
        score += page_text.count(t) * 3

    q_words = [w for w in re.findall(r"[a-z0-9]+", full_q.lower()) if w not in STOPWORDS]
    for n in (2, 3, 4):
        for i in range(0, max(0, len(q_words) - n + 1)):
            phrase = " ".join(q_words[i:i+n])
            if len(phrase) < 10:
                continue
            if phrase in page_text:
                score += 25
    return score

def list_pdfs() -> List[str]:
    return [p.name for p in PDF_DIR.glob("*.pdf")]

def retrieve_across_pdfs(
    question: str,
    max_docs: int = 3,
    pages_per_doc: int = 2,
    window: int = 0
) -> List[Tuple[str, List[int], int]]:
    """
    Returns list of (pdf_name, page_indexes_0_based, doc_score).
    """
    available = list_pdfs()
    if not available:
        return []

    tokens = tokenize(question)
    if not tokens:
        # If user asked something extremely general, don't waste docs.
        return []

    # Score each doc by its best matching pages
    doc_scores: List[Tuple[str, int]] = []
    per_doc_scored_pages: Dict[str, List[Tuple[int,int]]] = {}

    for name in available:
        ensure_indexed(name)
        pages = PDF_TEXT_INDEX.get(name, [])
        if not pages:
            continue

        scored = []
        for i, txt in enumerate(pages):
            s = score_page(txt, tokens, question)
            if s > 0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        per_doc_scored_pages[name] = scored
        doc_score = sum(s for _, s in scored[:pages_per_doc]) if scored else 0
        doc_scores.append((name, doc_score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, s in doc_scores if s > 0][:max_docs]

    # If nothing scored, return empty so we act normal (prevents weird “doc guessing”)
    if not top_docs:
        return []

    results: List[Tuple[str, List[int], int]] = []
    for name in top_docs:
        pages = PDF_TEXT_INDEX.get(name, [])
        scored = per_doc_scored_pages.get(name, [])
        top_pages = [i for i, _ in scored[:pages_per_doc]] if scored else []

        selected = set()
        total = len(pages)
        for p in top_pages:
            for n in range(-window, window + 1):
                idx = p + n
                if 0 <= idx < total:
                    selected.add(idx)

        results.append((name, sorted(selected), sum(s for _, s in scored[:pages_per_doc])))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ----------------------------
# Extract text + optional images
# ----------------------------

def extract_pages_text(pdf_path: Path, page_indexes: List[int], max_chars_per_page: int = 1800) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for idx in page_indexes:
            page = doc.load_page(idx)
            txt = _clean_text(page.get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + " …"
            chunks.append(f"[Page {idx+1}]\n{txt}")
        return "\n\n".join(chunks)
    finally:
        doc.close()

def pdf_page_to_data_url(pdf_path: Path, page_index: int, dpi: int = 120) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pdf_pages_to_images(pdf_path: Path, pdf_name: str, pages: List[int], dpi: int = 120) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in pages:
        key = (pdf_name, p, dpi)
        cached = _cache_get(key)
        if cached is None:
            cached = pdf_page_to_data_url(pdf_path, p, dpi=dpi)
            _cache_set(key, cached)
        blocks.append({"type": "input_image", "image_url": cached})
    return blocks


# ----------------------------
# Vision heuristics (keep cost sane)
# ----------------------------

def needs_vision_from_question(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        "table", "tab.", "diagram", "figure", "fig.", "drawing", "detail",
        "chart", "graph", "schedule", "matrix", "appendix",
        "caption", "footnote", "note under", "see the", "read the", "interpret",
        "plan", "elevation", "section"
    ]
    return any(t in ql for t in triggers)

def needs_vision_from_text_excerpt(excerpt: str) -> bool:
    if not excerpt:
        return True
    lines = excerpt.splitlines()
    if len(lines) <= 4:
        return True
    long_space_lines = sum(1 for ln in lines if "    " in ln)
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    dot_leader = sum(1 for ln in lines if "...." in ln)
    very_short = sum(1 for ln in lines if len(ln.strip()) <= 2)
    # conservative triggers (avoid vision unless likely needed)
    if long_space_lines > 16 or pipe_lines > 6 or dot_leader > 6:
        return True
    if very_short > (len(lines) * 0.33):
        return True
    if len(excerpt) < 700:
        return True
    return False


# ----------------------------
# Tone: normal + slightly positive + professional humor
# ----------------------------

SYSTEM_RULES = """
You are a helpful, natural assistant.

Default behavior:
- Be a normal AI assistant first: friendly, slightly positive, and professionally humorous when it fits.
- Keep it practical. Avoid stiff “compliance robot” vibes.
- Don’t overdo jokes — one light line occasionally is plenty.

When PDF SOURCES are provided:
- Treat SOURCES as the authoritative reference for claims about those PDFs.
- Use them to answer and cite like (DocName p.12).
- If the question depends on the PDFs but the SOURCES don’t contain it, say you couldn’t find it in the provided excerpts and suggest what to search for next.
- Do not invent clause numbers, numeric limits, or diagram meanings not supported by SOURCES.
""".strip()

def build_sources_bundle(
    question: str,
    retrieved: List[Tuple[str, List[int], int]],
    max_total_pages: int = 8,
    max_chars_per_page: int = 1800,
) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Build SOURCES text + list of (doc, page_1based) citations.
    """
    parts: List[str] = []
    citations: List[Tuple[str, int]] = []
    used_pages = 0

    for doc_name, page_idxs, _score in retrieved:
        if used_pages >= max_total_pages:
            break
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue

        remaining = max_total_pages - used_pages
        page_idxs = page_idxs[:remaining]

        excerpt = extract_pages_text(pdf_path, page_idxs, max_chars_per_page=max_chars_per_page)
        parts.append(f"SOURCE: {doc_name}\n{excerpt}")
        for pi in page_idxs:
            citations.append((doc_name, pi + 1))
        used_pages += len(page_idxs)

    return "\n\n".join(parts).strip(), citations

def build_openai_blocks(question: str, sources_text: str, images: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    blocks = [
        {"type": "input_text", "text": SYSTEM_RULES},
        {"type": "input_text", "text": f"USER QUESTION:\n{question}\n\nSOURCES:\n{sources_text}" if sources_text else f"USER QUESTION:\n{question}\n\n(No PDF sources provided.)"},
    ]
    if images:
        blocks.extend(images)
    return blocks


# ----------------------------
# API endpoints
# ----------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Raheem AI API",
        "pdf_dir": str(PDF_DIR),
        "pdf_count": len(list_pdfs()),
        "indexed_pdfs": len(PDF_TEXT_INDEX),
        "model": MODEL_NAME,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/pdfs")
def pdfs():
    files = list_pdfs()
    return {"count": len(files), "pdfs": files}

def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._\- ]+", "", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files allowed"}

    data = await file.read()
    if MAX_UPLOAD_MB > 0 and len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return {"ok": False, "error": f"PDF too large. Max is {MAX_UPLOAD_MB}MB"}

    fname = safe_filename(file.filename)
    path = PDF_DIR / fname
    path.write_bytes(data)

    try:
        index_pdf(path)
    except Exception:
        pass

    return {"ok": True, "pdf": fname, "stored_in": str(PDF_DIR)}

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    force_docs: bool = Query(False, description="Force using PDFs even if router says no"),
    max_docs: int = Query(3),
    pages_per_doc: int = Query(2),
    window: int = Query(0),
    max_total_pages: int = Query(8),
    dpi: int = Query(120),
    force_vision: bool = Query(False, description="Force vision images"),
):
    """
    Non-streaming: returns one JSON response.
    """
    try:
        use_docs = force_docs or should_use_docs(q)

        retrieved: List[Tuple[str, List[int], int]] = []
        sources_text = ""
        citations: List[Tuple[str, int]] = []

        if use_docs:
            retrieved = retrieve_across_pdfs(q, max_docs=max_docs, pages_per_doc=pages_per_doc, window=window)
            sources_text, citations = build_sources_bundle(q, retrieved, max_total_pages=max_total_pages)

        images = None
        if use_docs and retrieved:
            wants_vision = force_vision or needs_vision_from_question(q)
            if wants_vision:
                # pick up to 2 pages for vision from the top doc
                top_doc, page_idxs, _ = retrieved[0]
                pdf_path = PDF_DIR / top_doc
                excerpt = extract_pages_text(pdf_path, page_idxs[:1], max_chars_per_page=900)
                if force_vision or needs_vision_from_text_excerpt(excerpt):
                    images = pdf_pages_to_images(pdf_path, top_doc, page_idxs[:2], dpi=dpi)

        blocks = build_openai_blocks(q, sources_text, images)
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": blocks}],
            max_output_tokens=700,
        )

        return {
            "ok": True,
            "answer": resp.output_text,
            "used_docs": bool(use_docs and sources_text),
            "sources_used": [{"doc": d, "page": p} for d, p in citations],
            "retrieved_docs": [r[0] for r in retrieved] if retrieved else [],
            "vision_used": bool(images),
            "model": MODEL_NAME,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc().splitlines()[-12:],
        }

@app.get("/ask_stream")
def ask_stream(
    q: str = Query(..., description="User question"),
    force_docs: bool = Query(False),
    max_docs: int = Query(3),
    pages_per_doc: int = Query(2),
    window: int = Query(0),
    max_total_pages: int = Query(8),
    dpi: int = Query(120),
    force_vision: bool = Query(False),
):
    """
    Streaming SSE for your frontend typing effect.
    """
    def sse():
        try:
            use_docs = force_docs or should_use_docs(q)

            retrieved: List[Tuple[str, List[int], int]] = []
            sources_text = ""
            citations: List[Tuple[str, int]] = []

            if use_docs:
                retrieved = retrieve_across_pdfs(q, max_docs=max_docs, pages_per_doc=pages_per_doc, window=window)
                sources_text, citations = build_sources_bundle(q, retrieved, max_total_pages=max_total_pages)

            images = None
            if use_docs and retrieved:
                wants_vision = force_vision or needs_vision_from_question(q)
                if wants_vision:
                    top_doc, page_idxs, _ = retrieved[0]
                    pdf_path = PDF_DIR / top_doc
                    excerpt = extract_pages_text(pdf_path, page_idxs[:1], max_chars_per_page=900)
                    if force_vision or needs_vision_from_text_excerpt(excerpt):
                        images = pdf_pages_to_images(pdf_path, top_doc, page_idxs[:2], dpi=dpi)

            # meta for debugging
            yield f"event: meta\ndata: model={MODEL_NAME};used_docs={use_docs};vision={bool(images)}\n\n"
            if citations:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d, p in citations]) + "\n\n"

            blocks = build_openai_blocks(q, sources_text, images)

            stream = client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": blocks}],
                max_output_tokens=700,
                stream=True,
            )

            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        safe = delta.replace("\r", "").replace("\n", "\\n")
                        yield f"data: {safe}\n\n"
                if getattr(event, "type", None) == "response.completed":
                    break

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r", "").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ----------------------------
# Optional: “real chat” endpoint (history-aware)
# ----------------------------

@app.post("/chat_stream")
def chat_stream(payload: Dict[str, Any] = Body(...)):
    """
    POST body example:
    {
      "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
      "force_docs": false
    }

    This lets your frontend send conversation history so the AI feels like a normal chat.
    PDFs are still used only when needed.
    """
    messages = payload.get("messages", [])
    force_docs = bool(payload.get("force_docs", False))

    # Extract latest user question
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "") or ""
            break

    def sse():
        try:
            use_docs = force_docs or should_use_docs(last_user)

            retrieved: List[Tuple[str, List[int], int]] = []
            sources_text = ""
            citations: List[Tuple[str, int]] = []

            if use_docs:
                retrieved = retrieve_across_pdfs(last_user, max_docs=3, pages_per_doc=2, window=0)
                sources_text, citations = build_sources_bundle(last_user, retrieved, max_total_pages=8)

            # Build content blocks: system + short history + optional sources
            # Keep history small to control cost
            trimmed = messages[-10:] if isinstance(messages, list) else []
            history_text = []
            for m in trimmed:
                r = m.get("role")
                c = (m.get("content") or "").strip()
                if r in ("user", "assistant") and c:
                    history_text.append(f"{r.upper()}: {c}")
            history_blob = "\n".join(history_text).strip()

            blocks = [
                {"type": "input_text", "text": SYSTEM_RULES},
                {"type": "input_text", "text": f"CHAT HISTORY (most recent):\n{history_blob}\n\nLATEST USER QUESTION:\n{last_user}\n\nSOURCES:\n{sources_text}" if sources_text else f"CHAT HISTORY (most recent):\n{history_blob}\n\nLATEST USER QUESTION:\n{last_user}\n\n(No PDF sources provided.)"},
            ]

            yield f"event: meta\ndata: model={MODEL_NAME};used_docs={use_docs}\n\n"
            if citations:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d, p in citations]) + "\n\n"

            stream = client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": blocks}],
                max_output_tokens=700,
                stream=True,
            )

            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        safe = delta.replace("\r", "").replace("\n", "\\n")
                        yield f"data: {safe}\n\n"
                if getattr(event, "type", None) == "response.completed":
                    break

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r", "").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

"""
Raheem AI — FastAPI backend (Vertex Gemini)

What this fixes:
- Chat memory per conversation (chat_id) so it doesn't reset mid-thread.
- /chat_stream supports BOTH:
  A) frontend sends {chat_id, messages:[...]}
  B) frontend sends {chat_id, message:"..."} and server uses stored history
- Compliance mode pulls evidence from TGDs and cites (Document p.X) WITHOUT saying "you uploaded PDFs".
- Cost control: small retrieval window + cheap model for casual chat, stronger model for compliance.
- "Normal" general answers (including basic health questions) should respond like a normal AI:
  it won’t do the "I can’t help" blanket refusal unless it’s genuinely dangerous.
  (Still: no illegal instructions, no self-harm guidance, etc.)

Keep endpoints:
/              root
/health        health check
/docs          simple check (NOT Swagger)
/swagger       Swagger UI (FastAPI docs)
/pdfs          list PDFs
/upload-pdf    upload PDFs
/ask           non-stream answer
/ask_stream    stream answer
/chat_stream   stream answer with memory by chat_id
"""

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

import os
import re
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter, OrderedDict

import fitz  # PyMuPDF

# Vertex AI Gemini
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig


# ----------------------------
# Setup
# ----------------------------

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Upload / storage
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

# Chat memory (server-side)
CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "36"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "26000"))

# Vertex config
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION", "europe-west4")

GOOGLE_CREDENTIALS_JSON = (os.getenv("GOOGLE_CREDENTIALS_JSON", "") or "").strip()

# Models (cost-aware)
# Cheap/fast for normal chat, stronger for compliance/doc-heavy
MODEL_CHAT = os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-lite")
MODEL_COMPLIANCE = os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash")

# Vision rendering cache
IMAGE_CACHE_MAX = int(os.getenv("IMAGE_CACHE_MAX", "48"))

# Retrieval limits (cost control)
DEFAULT_MAX_DOCS = int(os.getenv("DEFAULT_MAX_DOCS", "3"))
DEFAULT_MAX_TOTAL_PAGES = int(os.getenv("DEFAULT_MAX_TOTAL_PAGES", "7"))
DEFAULT_DPI = int(os.getenv("DEFAULT_DPI", "140"))


app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Vertex auth helper
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
            raise RuntimeError("VERTEX_PROJECT_ID (or GCP_PROJECT_ID) is missing in env vars")

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(use_docs: bool) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(MODEL_COMPLIANCE if use_docs else MODEL_CHAT)


def gen_cfg(use_docs: bool) -> GenerationConfig:
    # Normal chat: warmer + a bit witty
    # Compliance: tighter + less creative
    if use_docs:
        return GenerationConfig(temperature=0.2, top_p=0.7, max_output_tokens=900)
    return GenerationConfig(temperature=0.85, top_p=0.9, max_output_tokens=900)


# ----------------------------
# Server-side chat memory by chat_id
# ----------------------------

# { chat_id: [ {"role":"user|assistant","content":"..."}, ... ] }
CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}


def _trim_chat(chat_id: str) -> None:
    msgs = CHAT_STORE.get(chat_id, [])
    if not msgs:
        return

    # Cap message count
    if len(msgs) > CHAT_MAX_MESSAGES:
        msgs = msgs[-CHAT_MAX_MESSAGES:]

    # Cap total chars (newest-first)
    total = 0
    keep_rev: List[Dict[str, str]] = []
    for m in reversed(msgs):
        c = (m.get("content") or "")
        total += len(c)
        if total > CHAT_MAX_CHARS:
            break
        keep_rev.append(m)

    keep_rev.reverse()
    CHAT_STORE[chat_id] = keep_rev


def remember(chat_id: str, role: str, content: str) -> None:
    if not chat_id:
        return
    CHAT_STORE.setdefault(chat_id, []).append({"role": role, "content": content})
    _trim_chat(chat_id)


def history(chat_id: str) -> List[Dict[str, str]]:
    return CHAT_STORE.get(chat_id, [])


# ----------------------------
# PDF indexing (BM25-ish per page)
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
            "pages": doc.page_count,
            "page_text_lower": page_text_lower,
            "page_tf": page_tf,
            "df": df,
            "page_len": page_len,
            "avgdl": avgdl,
        }
    finally:
        doc.close()


def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)


def index_all_pdfs() -> None:
    for name in list_pdfs():
        try:
            index_pdf(PDF_DIR / name)
        except Exception:
            continue


index_all_pdfs()


def bm25_score(tf: Counter, df: Counter, N: int, dl: int, avgdl: float, q_tokens: List[str],
               k1: float = 1.4, b: float = 0.75) -> float:
    if not q_tokens or N <= 0:
        return 0.0
    score = 0.0
    denom_norm = (1 - b) + b * (dl / (avgdl + 1e-9))
    for t in q_tokens:
        f = tf.get(t, 0)
        if f <= 0:
            continue
        n_t = df.get(t, 0)
        idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        score += idf * (f * (k1 + 1)) / (f + k1 * denom_norm)
    return score


def phrase_bonus(page_text_lower: str, question: str) -> float:
    q_words = [w for w in re.findall(r"[a-z0-9]+", (question or "").lower()) if w not in STOPWORDS]
    bonus = 0.0
    for n in (2, 3, 4):
        for i in range(0, max(0, len(q_words) - n + 1)):
            phrase = " ".join(q_words[i:i+n])
            if len(phrase) >= 9 and phrase in page_text_lower:
                bonus += 1.0 + 0.25 * n
    return bonus


def retrieve_top_pages(pdf_name: str, question: str, top_k: int = 5, page_hint: Optional[int] = None) -> List[Tuple[int, float]]:
    ensure_indexed(pdf_name)
    idx = PDF_INDEX.get(pdf_name)
    if not idx:
        return []
    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    N = idx["pages"]
    scores: List[Tuple[int, float]] = []

    for i in range(N):
        base = bm25_score(idx["page_tf"][i], idx["df"], N, idx["page_len"][i], idx["avgdl"], q_tokens)
        if base <= 0:
            continue
        base += phrase_bonus(idx["page_text_lower"][i], question)

        if page_hint and page_hint > 0:
            target = page_hint - 1
            dist = abs(i - target)
            if dist == 0:
                base += 2.5
            elif dist <= 2:
                base += 1.5
            elif dist <= 6:
                base += 0.6

        scores.append((i, float(base)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def expand_pages(pages: List[int], total_pages: int, window: int) -> List[int]:
    s = set()
    for p in pages:
        for n in range(-window, window + 1):
            j = p + n
            if 0 <= j < total_pages:
                s.add(j)
    return sorted(s)


def select_pages(question: str, pdf_pin: Optional[str], page_hint: Optional[int],
                 max_docs: int, max_total_pages: int) -> List[Tuple[str, List[int], float]]:
    available = list_pdfs()
    if not available:
        return []

    if pdf_pin:
        if pdf_pin not in available:
            return []
        tops = retrieve_top_pages(pdf_pin, question, top_k=6, page_hint=page_hint)
        if not tops:
            return []
        best = tops[0][1]
        base_pages = [p for p, _s in tops[:3]]
        ensure_indexed(pdf_pin)
        total = PDF_INDEX[pdf_pin]["pages"]
        window = 1 if best < 2.0 else 0
        pages = expand_pages(base_pages, total, window=window)
        return [(pdf_pin, pages[:max_total_pages], best)]

    doc_best: List[Tuple[str, float]] = []
    per_doc: Dict[str, List[Tuple[int, float]]] = {}
    for name in available:
        tops = retrieve_top_pages(name, question, top_k=6, page_hint=page_hint)
        per_doc[name] = tops
        best = tops[0][1] if tops else 0.0
        if best > 0:
            doc_best.append((name, best))

    doc_best.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _s in doc_best[:max_docs]]

    results: List[Tuple[str, List[int], float]] = []
    used = 0
    for doc_name in top_docs:
        tops = per_doc.get(doc_name, [])
        if not tops:
            continue
        best = tops[0][1]
        base_pages = [p for p, _s in tops[:3]]
        ensure_indexed(doc_name)
        total = PDF_INDEX[doc_name]["pages"]
        window = 2 if best < 1.2 else (1 if best < 2.0 else 0)
        pages = expand_pages(base_pages, total, window=window)
        remaining = max_total_pages - used
        if remaining <= 0:
            break
        pages = pages[:remaining]
        used += len(pages)
        results.append((doc_name, pages, best))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def extract_pages_text(pdf_path: Path, page_indexes: List[int], max_chars_per_page: int = 1900) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for idx in page_indexes:
            page = doc.load_page(idx)
            txt = clean_text(page.get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + " …"
            chunks.append(f"[Page {idx + 1}]\n{txt}")
        return "\n\n".join(chunks)
    finally:
        doc.close()


# ----------------------------
# Vision helpers (render pages only when needed)
# ----------------------------

_IMAGE_CACHE: "OrderedDict[Tuple[str,int,int], bytes]" = OrderedDict()


def cache_get(k):
    if k in _IMAGE_CACHE:
        _IMAGE_CACHE.move_to_end(k)
        return _IMAGE_CACHE[k]
    return None


def cache_set(k, v):
    _IMAGE_CACHE[k] = v
    _IMAGE_CACHE.move_to_end(k)
    while len(_IMAGE_CACHE) > IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)


def pdf_page_to_png_bytes(pdf_path: Path, page_index: int, dpi: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def pages_to_parts(pdf_path: Path, pdf_name: str, pages: List[int], dpi: int) -> List[Part]:
    parts: List[Part] = []
    for p in pages:
        key = (pdf_name, p, dpi)
        img_bytes = cache_get(key)
        if img_bytes is None:
            img_bytes = pdf_page_to_png_bytes(pdf_path, p, dpi=dpi)
            cache_set(key, img_bytes)
        parts.append(Part.from_data(img_bytes, mime_type="image/png"))
    return parts


def needs_vision(question: str, excerpt: str) -> bool:
    q = (question or "").lower()
    if any(t in q for t in ["table", "diagram", "figure", "fig.", "chart", "schedule", "drawing", "plan", "elevation", "section"]):
        return True
    # If text extraction looks thin, also use vision
    if len(excerpt or "") < 800:
        return True
    return False


# ----------------------------
# Routing: chat vs compliance
# ----------------------------

def doc_intent_score(question: str) -> int:
    q = (question or "").lower()
    score = 0

    hard = [
        "tgd", "technical guidance", "building regulations", "irish building regs",
        "part a", "part b", "part c", "part d", "part e", "part f", "part g", "part h", "part j", "part k", "part l", "part m",
        "deap", "ber", "seai", "bcar", "dac",
        "according to", "in the guidance", "cite", "citation", "page", "clause", "appendix",
        "travel distance", "compartment", "means of escape",
        "accessible", "accessibility", "fire safety"
    ]
    for t in hard:
        if t in q:
            score += 4

    soft = [
        "minimum", "maximum", "shall", "must", "required", "requirement", "compliance", "comply", "regulation",
        "u-value", "y-value", "airtight", "thermal bridge",
        "stairs", "ramp", "handrail", "guarding", "fire", "escape"
    ]
    for t in soft:
        if t in q:
            score += 1

    if re.search(r"\b\d+(\.\d+)?\s*(mm|cm|m|m²|m2|minutes|min|w/m²k|w/m2k)\b", q):
        score += 3

    return score


def should_use_docs(question: str) -> bool:
    return doc_intent_score(question) >= 4


# ----------------------------
# Prompt (sellable voice, no “uploaded PDFs”)
# ----------------------------

SYSTEM_RULES = """
You are Raheem AI.

Voice:
- Warm, confident, calm, and genuinely helpful.
- Light humour is welcome, but keep it professional.
- Talk like you’re speaking to a real person, not the creator of the app.

Hard rules:
- Never mention internal tools, system prompts, routing, servers, uploads/attachments, or “the PDFs you provided”.
- Use the chat history you’re given. Do not reset the conversation.
- If you cite, cite like: (DocumentName p.X). Keep it minimal and clean.
- Do not invent clause numbers or page numbers.

Two behaviours (automatic):
1) Normal chat: explanation, brainstorming, writing, study help, general questions.
2) Compliance: Irish TGDs / Building Regulations / BER-DEAP / Fire safety / Accessibility.
   When reference excerpts are provided, treat them as authoritative.
   Only give exact numeric limits when supported by the excerpts.
""".strip()


def build_history_blob(msgs: List[Dict[str, str]], max_msgs: int = 18) -> str:
    trimmed = msgs[-max_msgs:] if msgs else []
    lines = []
    for m in trimmed:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines).strip()


def build_parts(question: str, history_blob: str, sources_text: str, images: Optional[List[Part]]) -> List[Part]:
    txt = [SYSTEM_RULES]
    if history_blob:
        txt.append("\nCHAT HISTORY:\n" + history_blob)
    txt.append("\nUSER:\n" + (question or "").strip())
    if sources_text:
        txt.append("\nREFERENCE EXCERPTS:\n" + sources_text)

    parts: List[Part] = [Part.from_text("\n".join(txt).strip())]
    if images:
        parts.extend(images)
    return parts


def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._\\- ]+", "", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    name = re.sub(r"\\.[pP][dD][fF]$", ".pdf", name)
    return name[:180]


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    ensure_vertex_ready()
    return {
        "ok": True,
        "service": "Raheem AI API",
        "vertex_ready": bool(_VERTEX_READY),
        "vertex_error": _VERTEX_ERR,
        "project_set": bool(GCP_PROJECT_ID),
        "location": GCP_LOCATION,
        "models": {"chat": MODEL_CHAT, "compliance": MODEL_COMPLIANCE},
        "pdf_count": len(list_pdfs()),
        "indexed_pdfs": len(PDF_INDEX),
        "chat_sessions_in_memory": len(CHAT_STORE),
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    ensure_vertex_ready()
    return {"ok": True, "vertex_ready": bool(_VERTEX_READY)}


@app.get("/docs")
def docs_check():
    # simple frontend compatibility check
    return {"ok": True, "pdf_count": len(list_pdfs())}


@app.get("/pdfs")
def pdfs():
    files = list_pdfs()
    return {"count": len(files), "pdfs": files}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file or not file.filename:
            return {"ok": False, "error": "No file received."}
        if not file.filename.lower().endswith(".pdf"):
            return {"ok": False, "error": "Only PDF files are allowed."}

        fname = safe_filename(file.filename)
        path = PDF_DIR / fname
        PDF_DIR.mkdir(parents=True, exist_ok=True)

        max_bytes = MAX_UPLOAD_MB * 1024 * 1024 if MAX_UPLOAD_MB > 0 else None
        written = 0

        with open(path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    out.close()
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {"ok": False, "error": f"PDF too large. Max is {MAX_UPLOAD_MB}MB"}
                out.write(chunk)

        try:
            index_pdf(path)
            indexed_ok = True
            index_error = None
        except Exception as ie:
            indexed_ok = False
            index_error = str(ie)

        return {
            "ok": True,
            "pdf": fname,
            "bytes": written,
            "indexed": indexed_ok,
            "index_error": index_error,
            "pdf_count_now": len(list_pdfs()),
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-18:]}


# ----------------------------
# /ask (non-stream) — stateless (quick testing)
# ----------------------------

@app.get("/ask")
def ask(
    q: str = Query(...),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    max_docs: int = Query(DEFAULT_MAX_DOCS),
    max_total_pages: int = Query(DEFAULT_MAX_TOTAL_PAGES),
    dpi: int = Query(DEFAULT_DPI),
    force_vision: bool = Query(False),
):
    try:
        ensure_vertex_ready()
        if not _VERTEX_READY:
            return {"ok": False, "error": _VERTEX_ERR or "Vertex not ready"}

        use_docs = bool(force_docs or should_use_docs(q))

        selected: List[Tuple[str, List[int], float]] = []
        sources_text = ""
        cites: List[Tuple[str, int]] = []
        image_parts: Optional[List[Part]] = None

        if use_docs and list_pdfs():
            selected = select_pages(q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages)
            for doc_name, pages, _s in selected:
                pdf_path = PDF_DIR / doc_name
                excerpt = extract_pages_text(pdf_path, pages)
                sources_text += f"\n\nREFERENCE: {doc_name}\n{excerpt}"
                for p in pages:
                    cites.append((doc_name, p + 1))

            # Vision only if needed (tables/diagrams/scans)
            if selected:
                top_doc, pages, _s = selected[0]
                pdf_path = PDF_DIR / top_doc
                excerpt_one = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                if force_vision or needs_vision(q, excerpt_one):
                    image_parts = pages_to_parts(pdf_path, top_doc, pages[:2], dpi=dpi)

        used_docs_flag = bool(use_docs and sources_text.strip())
        model = get_model(use_docs=used_docs_flag)
        parts = build_parts(q, history_blob="", sources_text=sources_text.strip(), images=image_parts)
        resp = model.generate_content(parts, generation_config=gen_cfg(used_docs_flag))

        return {
            "ok": True,
            "answer": (resp.text or "").strip(),
            "used_docs": used_docs_flag,
            "sources_used": [{"doc": d, "page": p} for d, p in cites],
            "vision_used": bool(image_parts),
            "model": MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT,
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-18:]}


# ----------------------------
# /ask_stream (SSE) — stateless
# ----------------------------

@app.get("/ask_stream")
def ask_stream(
    q: str = Query(...),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    max_docs: int = Query(DEFAULT_MAX_DOCS),
    max_total_pages: int = Query(DEFAULT_MAX_TOTAL_PAGES),
    dpi: int = Query(DEFAULT_DPI),
    force_vision: bool = Query(False),
):
    def sse():
        try:
            ensure_vertex_ready()
            if not _VERTEX_READY:
                yield f"event: error\ndata: {_VERTEX_ERR or 'Vertex not ready'}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

            use_docs = bool(force_docs or should_use_docs(q))

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str, int]] = []
            image_parts: Optional[List[Part]] = None

            if use_docs and list_pdfs():
                selected = select_pages(q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages)
                for doc_name, pages, _s in selected:
                    pdf_path = PDF_DIR / doc_name
                    excerpt = extract_pages_text(pdf_path, pages)
                    sources_text += f"\n\nREFERENCE: {doc_name}\n{excerpt}"
                    for p in pages:
                        cites.append((doc_name, p + 1))

                if selected:
                    top_doc, pages, _s = selected[0]
                    pdf_path = PDF_DIR / top_doc
                    excerpt_one = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                    if force_vision or needs_vision(q, excerpt_one):
                        image_parts = pages_to_parts(pdf_path, top_doc, pages[:2], dpi=dpi)

            used_docs_flag = bool(use_docs and sources_text.strip())
            model_name = MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT
            yield f"event: meta\ndata: model={model_name};used_docs={used_docs_flag};vision={bool(image_parts)}\n\n"

            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d, p in cites]) + "\n\n"

            model = get_model(use_docs=used_docs_flag)
            parts = build_parts(q, history_blob="", sources_text=sources_text.strip(), images=image_parts)

            stream = model.generate_content(parts, generation_config=gen_cfg(used_docs_flag), stream=True)
            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    safe = delta.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

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
# /chat_stream (SSE) — memory aware (THIS fixes the reset issue)
# ----------------------------

@app.post("/chat_stream")
def chat_stream(payload: Dict[str, Any] = Body(...)):
    """
    Supports BOTH:
    A) { chat_id, messages:[{role,content}...], force_docs? }
       - Uses messages as source of truth, stores last part on server for continuity
    B) { chat_id, message:"...", force_docs? }
       - Uses server-side history for that chat_id
    Also accepts: pdf (pin), page_hint, max_docs, max_total_pages, dpi, force_vision
    """

    def sse():
        try:
            ensure_vertex_ready()
            if not _VERTEX_READY:
                yield f"event: error\ndata: {_VERTEX_ERR or 'Vertex not ready'}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

            chat_id = (payload.get("chat_id") or "").strip()
            if not chat_id:
                # If frontend forgot it, still respond, but memory won't persist
                chat_id = ""

            force_docs = bool(payload.get("force_docs") or payload.get("force_document") or False)
            question = (payload.get("message") or "").strip()

            # Optional controls
            pdf = payload.get("pdf")
            page_hint = payload.get("page_hint")
            max_docs = int(payload.get("max_docs") or DEFAULT_MAX_DOCS)
            max_total_pages = int(payload.get("max_total_pages") or DEFAULT_MAX_TOTAL_PAGES)
            dpi = int(payload.get("dpi") or DEFAULT_DPI)
            force_vision = bool(payload.get("force_vision") or False)

            incoming_messages = payload.get("messages")

            # Decide history
            if isinstance(incoming_messages, list) and incoming_messages:
                # Frontend supplied full thread
                thread = []
                for m in incoming_messages[-50:]:
                    r = (m.get("role") or "").lower().strip()
                    c = (m.get("content") or "").strip()
                    if r in ("user", "assistant") and c:
                        thread.append({"role": r, "content": c})
                # Use last user message as question if not provided
                if not question:
                    for m in reversed(thread):
                        if m["role"] == "user":
                            question = m["content"]
                            break
                # Store to server (keep continuity if frontend sends partial next time)
                if chat_id:
                    CHAT_STORE[chat_id] = thread[-CHAT_MAX_MESSAGES:]
                    _trim_chat(chat_id)
            else:
                # Use server memory for this chat_id
                thread = history(chat_id)
                if question:
                    remember(chat_id, "user", question)

            if not question:
                yield "event: error\ndata: No message provided\n\n"
                yield "event: done\ndata: ok\n\n"
                return

            # Route docs or not
            use_docs = bool(force_docs or should_use_docs(question))

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str, int]] = []
            image_parts: Optional[List[Part]] = None

            if use_docs and list_pdfs():
                selected = select_pages(
                    question, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages
                )
                for doc_name, pages, _s in selected:
                    pdf_path = PDF_DIR / doc_name
                    excerpt = extract_pages_text(pdf_path, pages)
                    sources_text += f"\n\nREFERENCE: {doc_name}\n{excerpt}"
                    for p in pages:
                        cites.append((doc_name, p + 1))

                # Vision only when needed
                if selected:
                    top_doc, pages, _s = selected[0]
                    pdf_path = PDF_DIR / top_doc
                    excerpt_one = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                    if force_vision or needs_vision(question, excerpt_one):
                        image_parts = pages_to_parts(pdf_path, top_doc, pages[:2], dpi=dpi)

            used_docs_flag = bool(use_docs and sources_text.strip())

            # Stream meta (frontend can display small hint if you ever want)
            model_name = MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT
            yield f"event: meta\ndata: model={model_name};used_docs={used_docs_flag};vision={bool(image_parts)}\n\n"
            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d, p in cites]) + "\n\n"

            # Build prompt with recent history
            hist_blob = build_history_blob(thread if isinstance(thread, list) else [], max_msgs=18)

            model = get_model(use_docs=used_docs_flag)
            parts = build_parts(question, history_blob=hist_blob, sources_text=sources_text.strip(), images=image_parts)

            stream = model.generate_content(parts, generation_config=gen_cfg(used_docs_flag), stream=True)

            full_answer = []
            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    full_answer.append(delta)
                    safe = delta.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

            answer_text = "".join(full_answer).strip()
            if chat_id and answer_text:
                remember(chat_id, "assistant", answer_text)

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

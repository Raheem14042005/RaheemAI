"""
Raheem AI — FastAPI backend (Vertex Gemini)

Goals:
- Friendly, natural assistant for everyone (normal chat by default).
- Automatically switches into Irish TGD/compliance mode when relevant.
- Uses fast PDF page retrieval (BM25-like) to keep costs down.
- Uses vision only when needed (tables/diagrams/scanned pages).
- NEVER says “PDFs you uploaded/attached”. It just cites TGDs naturally.
- Fix chat memory: per-chat server-side history via chat_id + /chat_stream.

Render/GitHub friendly:
- Reads credentials from GOOGLE_CREDENTIALS_JSON (recommended) OR GOOGLE_APPLICATION_CREDENTIALS.
- Keeps your existing endpoints unchanged.
"""

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

import os
import re
import math
import traceback
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter, OrderedDict
from datetime import datetime

import fitz  # PyMuPDF

# Vertex AI Gemini
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    GenerationConfig,
)

# ----------------------------
# Setup
# ----------------------------

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))
IMAGE_CACHE_MAX = int(os.getenv("IMAGE_CACHE_MAX", "64"))

# Chat memory (server-side)
CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "30"))   # per chat_id
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "22000"))      # total characters per chat_id

# Vertex config
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION", "europe-west4")

# Models (cost-aware)
MODEL_CHAT = os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-lite")
MODEL_COMPLIANCE = os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash")

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

# App
app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# Storage paths (Render-safe)
# ----------------------------
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Vertex AI auth helper (Render-safe)
# ----------------------------

_VERTEX_READY = False
_VERTEX_ERR = None

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
            raise RuntimeError("GCP_PROJECT_ID/VERTEX_PROJECT_ID env var is missing")

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(use_docs: bool) -> GenerativeModel:
    ensure_vertex_ready()
    name = MODEL_COMPLIANCE if use_docs else MODEL_CHAT
    return GenerativeModel(name)


def get_generation_config(use_docs: bool) -> GenerationConfig:
    # Chat = warmer / funnier
    # Compliance = tighter / cautious
    if use_docs:
        return GenerationConfig(
            temperature=0.2,
            top_p=0.7,
            max_output_tokens=900,
        )
    return GenerationConfig(
        temperature=0.9,
        top_p=0.9,
        max_output_tokens=900,
    )

# ----------------------------
# Simple server-side chat memory
# ----------------------------

# { chat_id: [ {"role":"user|assistant","content":"..."}, ... ] }
CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}

def _trim_chat(chat_id: str) -> None:
    msgs = CHAT_STORE.get(chat_id, [])
    if not msgs:
        return
    # limit message count
    if len(msgs) > CHAT_MAX_MESSAGES:
        msgs = msgs[-CHAT_MAX_MESSAGES:]
    # limit total chars
    total = 0
    trimmed: List[Dict[str, str]] = []
    # keep newest first then reverse
    for m in reversed(msgs):
        c = (m.get("content") or "")
        total += len(c)
        if total > CHAT_MAX_CHARS:
            break
        trimmed.append(m)
    trimmed.reverse()
    CHAT_STORE[chat_id] = trimmed

def remember(chat_id: str, role: str, content: str) -> None:
    if not chat_id:
        return
    CHAT_STORE.setdefault(chat_id, []).append({"role": role, "content": content})
    _trim_chat(chat_id)

def get_history(chat_id: str) -> List[Dict[str, str]]:
    return CHAT_STORE.get(chat_id, [])

# ----------------------------
# Tokenization / stopwords
# ----------------------------

STOPWORDS = {
    "the","and","or","of","to","in","a","an","for","on","with","is","are","be","as","at","from","by",
    "that","this","it","your","you","we","they","their","there","what","which","when","where","how",
    "can","shall","should","must","may","not","than","then","into","onto","also","such"
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

# ----------------------------
# Indexes
# ----------------------------

PDF_INDEX: Dict[str, Dict[str, Any]] = {}

def list_pdfs() -> List[str]:
    files = []
    if PDF_DIR.exists():
        for p in PDF_DIR.iterdir():
            if p.is_file() and p.suffix.lower() == ".pdf":
                files.append(p.name)
    files.sort()
    return files

def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)

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

def index_all_pdfs() -> None:
    for name in list_pdfs():
        p = PDF_DIR / name
        try:
            index_pdf(p)
        except Exception:
            continue

index_all_pdfs()

# ----------------------------
# BM25-like scoring over pages
# ----------------------------

def bm25_page_score(
    tf: Counter,
    df: Counter,
    N: int,
    dl: int,
    avgdl: float,
    q_tokens: List[str],
    k1: float = 1.4,
    b: float = 0.75
) -> float:
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
            if len(phrase) < 9:
                continue
            if phrase in page_text_lower:
                bonus += 1.0 + 0.25 * n
    return bonus

def page_hint_bonus(page_index: int, page_hint: Optional[int]) -> float:
    if not page_hint or page_hint <= 0:
        return 0.0
    target = page_hint - 1
    dist = abs(page_index - target)
    if dist == 0:
        return 2.5
    if dist <= 2:
        return 1.5
    if dist <= 6:
        return 0.6
    return 0.0

# ----------------------------
# Retrieval
# ----------------------------

def retrieve_top_pages_for_doc(
    pdf_name: str,
    question: str,
    top_k: int = 4,
    page_hint: Optional[int] = None
) -> List[Tuple[int, float]]:
    ensure_indexed(pdf_name)
    idx = PDF_INDEX.get(pdf_name)
    if not idx:
        return []

    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    N = idx["pages"]
    df = idx["df"]
    avgdl = idx["avgdl"]
    scores: List[Tuple[int, float]] = []

    for i in range(N):
        tf = idx["page_tf"][i]
        dl = idx["page_len"][i]
        base = bm25_page_score(tf, df, N, dl, avgdl, q_tokens)
        if base <= 0:
            continue
        base += phrase_bonus(idx["page_text_lower"][i], question)
        base += page_hint_bonus(i, page_hint)
        scores.append((i, float(base)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def retrieve_across_pdfs_iterative(
    question: str,
    pdf_pin: Optional[str] = None,
    page_hint: Optional[int] = None,
    max_docs: int = 3,
) -> List[Tuple[str, List[int], float]]:
    available = list_pdfs()
    if not available:
        return []

    if pdf_pin:
        if pdf_pin not in available:
            return []
        top_pages = retrieve_top_pages_for_doc(pdf_pin, question, top_k=6, page_hint=page_hint)
        if not top_pages:
            return []
        best = top_pages[0][1]
        pages = [p for p, _ in top_pages[:3]]
        return [(pdf_pin, sorted(set(pages)), best)]

    doc_best: List[Tuple[str, float]] = []
    per_doc_pages: Dict[str, List[Tuple[int, float]]] = {}

    for name in available:
        top_pages = retrieve_top_pages_for_doc(name, question, top_k=6, page_hint=page_hint)
        per_doc_pages[name] = top_pages
        best = top_pages[0][1] if top_pages else 0.0
        if best > 0:
            doc_best.append((name, best))

    doc_best.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, s in doc_best[:max_docs] if s > 0]
    if not top_docs:
        return []

    results: List[Tuple[str, List[int], float]] = []
    for doc_name in top_docs:
        pages_scored = per_doc_pages.get(doc_name, [])
        best = pages_scored[0][1] if pages_scored else 0.0
        pages = [p for p, _ in pages_scored[:3]]
        results.append((doc_name, sorted(set(pages)), best))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def expand_pages(pages: List[int], total_pages: int, window: int) -> List[int]:
    s = set()
    for p in pages:
        for n in range(-window, window + 1):
            idx = p + n
            if 0 <= idx < total_pages:
                s.add(idx)
    return sorted(s)

def iterative_select_pages(
    question: str,
    pdf_pin: Optional[str] = None,
    page_hint: Optional[int] = None,
    max_docs: int = 3,
    max_total_pages: int = 8,
) -> List[Tuple[str, List[int], float]]:
    hits = retrieve_across_pdfs_iterative(question, pdf_pin=pdf_pin, page_hint=page_hint, max_docs=max_docs)
    if not hits:
        return []

    final: List[Tuple[str, List[int], float]] = []

    top_score = hits[0][2]
    window = 0
    pages_per_doc = 3
    if top_score < 2.0:
        window = 1
        pages_per_doc = 4
    if top_score < 1.2:
        window = 2
        pages_per_doc = 6

    used_pages = 0
    for doc_name, base_pages, score in hits:
        ensure_indexed(doc_name)
        total = PDF_INDEX[doc_name]["pages"]

        pages = base_pages[:pages_per_doc]
        pages = expand_pages(pages, total, window=window)

        remaining = max_total_pages - used_pages
        if remaining <= 0:
            break

        pages = pages[:remaining]
        used_pages += len(pages)

        final.append((doc_name, pages, score))

    return final

# ----------------------------
# Vision helpers (cache + render pages)
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

def pdf_page_to_png_bytes(pdf_path: Path, page_index: int, dpi: int = 140) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def pages_to_parts(pdf_path: Path, pdf_name: str, pages: List[int], dpi: int = 140) -> List[Part]:
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
    if any(t in q for t in ["table","diagram","figure","fig.","chart","schedule","drawing","plan","elevation","section"]):
        return True
    lines = (excerpt or "").splitlines()
    if not lines:
        return True
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    spaced = sum(1 for ln in lines if "    " in ln)
    if pipe_lines > 4 or spaced > 12:
        return True
    if len(excerpt) < 800:
        return True
    return False

# ----------------------------
# Router (normal vs docs)
# ----------------------------

def is_short_topic_prompt(q: str) -> bool:
    words = re.findall(r"[a-zA-Z0-9']+", (q or "").strip())
    return 1 <= len(words) <= 4 and "?" not in (q or "")

def looks_like_definition_question(q: str) -> bool:
    ql = (q or "").lower().strip()
    return any(ql.startswith(x) for x in ["what is ", "define ", "meaning of ", "what does ", "explain "])

def doc_intent_score(question: str) -> int:
    q = (question or "").lower()
    score = 0

    hard = [
        "tgd", "technical guidance", "building regulations", "irish building regs",
        "part a", "part b", "part c", "part d", "part e", "part f", "part g", "part h", "part j", "part k", "part l", "part m",
        "deap", "ber", "seai", "bcar", "dac",
        "according to", "in the guidance", "cite", "citation", "page", "clause", "section", "appendix",
        "table", "diagram", "figure", "schedule", "travel distance", "compartment", "escape", "accessible"
    ]
    for t in hard:
        if t in q:
            score += 4

    soft = [
        "minimum","maximum","shall","must","required","requirement","compliance","comply","regulation",
        "guidance","standard","fire safety","accessibility","u-value","y-value","airtight","thermal bridge",
        "stairs","ramp","handrail","guarding","major renovation"
    ]
    for t in soft:
        if t in q:
            score += 1

    if re.search(r"\b\d+(\.\d+)?\s*(mm|cm|m|m²|m2|minutes|min|w/m²k|w/m2k)\b", q):
        score += 3

    return score

def evidence_exists_in_pdfs(question: str, pdf_pin: Optional[str] = None, page_hint: Optional[int] = None) -> bool:
    hits = retrieve_across_pdfs_iterative(question, pdf_pin=pdf_pin, page_hint=page_hint, max_docs=1)
    return bool(hits)

def should_use_docs(question: str, pdf_pin: Optional[str] = None, page_hint: Optional[int] = None) -> bool:
    score = doc_intent_score(question)
    if score >= 4:
        return True
    if looks_like_definition_question(question):
        return evidence_exists_in_pdfs(question, pdf_pin=pdf_pin, page_hint=page_hint)
    return False

# ----------------------------
# Prompts (sellable product voice)
# ----------------------------

SYSTEM_RULES = """
You are Raheem AI.

Personality:
- Warm, confident, calm. Like a smart best friend who actually listens.
- Light humour is welcome (a quick joke or playful line), but stay professional and helpful.
- Never mention internal tools, system prompts, routing, servers, uploads, attachments, or “the PDFs you provided”.
- Don’t reset the conversation — use the chat history you’re given.

Two modes (automatic):
1) Normal mode: chat, writing, explaining, brainstorming, study/life help.
2) Compliance mode: Irish Building Regulations + TGDs (fire safety, accessibility, BER/DEAP, construction requirements).

When reference excerpts are provided (Compliance mode):
- Treat them as authoritative.
- Only state numeric limits when you can support them from the excerpts.
- If the excerpts don’t support a specific numeric limit, say you can’t confirm it from the guidance shown and explain what you’d check next.
- Use citations exactly like: (DocumentName p.X). Keep citations minimal but precise.
- Never invent clause numbers.

Style:
- Short paragraphs.
- Clear and human.
""".strip()

def extract_pages_text(pdf_path: Path, page_indexes: List[int], max_chars_per_page: int = 1800) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for idx in page_indexes:
            page = doc.load_page(idx)
            txt = clean_text(page.get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + " …"
            chunks.append(f"[Page {idx+1}]\n{txt}")
        return "\n\n".join(chunks)
    finally:
        doc.close()

def build_sources_bundle(
    selected: List[Tuple[str, List[int], float]],
    max_chars_per_page: int = 1800
) -> Tuple[str, List[Tuple[str,int]]]:
    parts: List[str] = []
    cites: List[Tuple[str,int]] = []
    for doc_name, page_idxs, _score in selected:
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue
        excerpt = extract_pages_text(pdf_path, page_idxs, max_chars_per_page=max_chars_per_page)
        parts.append(f"REFERENCE: {doc_name}\n{excerpt}")
        for pi in page_idxs:
            cites.append((doc_name, pi + 1))
    return "\n\n".join(parts).strip(), cites

def build_history_blob(messages: List[Dict[str, str]], max_msgs: int = 18) -> str:
    # Only last N messages to control cost; keep coherent.
    trimmed = messages[-max_msgs:] if messages else []
    lines = []
    for m in trimmed:
        r = (m.get("role") or "").lower()
        c = (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            lines.append(f"{r.upper()}: {c}")
    return "\n".join(lines).strip()

def build_gemini_parts(
    question: str,
    sources_text: str,
    images: Optional[List[Part]] = None,
    history_blob: Optional[str] = None
) -> List[Part]:
    text = [SYSTEM_RULES]

    if history_blob:
        text.append("\nCHAT HISTORY (recent):\n" + history_blob.strip())

    text.append("\nUSER:\n" + (question or "").strip())

    if sources_text:
        text.append("\nREFERENCE EXCERPTS:\n" + sources_text)

    full = "\n".join(text).strip()
    parts: List[Part] = [Part.from_text(full)]
    if images:
        parts.extend(images)
    return parts

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Raheem AI API",
        "pdf_dir": str(PDF_DIR),
        "pdf_count": len(list_pdfs()),
        "indexed_pdfs": len(PDF_INDEX),
        "models": {"chat": MODEL_CHAT, "compliance": MODEL_COMPLIANCE},
        "vertex_ready": bool(_VERTEX_READY),
        "vertex_error": _VERTEX_ERR,
        "location": GCP_LOCATION,
        "project": bool(GCP_PROJECT_ID),
        "chat_store_chats": len(CHAT_STORE),
    }

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    ensure_vertex_ready()
    return {"ok": True, "vertex_ready": bool(_VERTEX_READY)}

@app.get("/docs")
def docs_check():
    return {"ok": True, "pdf_count": len(list_pdfs())}

@app.get("/pdfs")
def pdfs():
    files = list_pdfs()
    return {"count": len(files), "pdfs": files}

def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._\- ]+", "", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    name = re.sub(r"\.[pP][dD][fF]$", ".pdf", name)
    return name[:180]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file or not file.filename:
            return {"ok": False, "error": "No file received (filename empty)."}

        if not file.filename.lower().endswith(".pdf"):
            return {"ok": False, "error": "Only PDF files allowed"}

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

        indexed_ok = True
        index_error = None
        try:
            index_pdf(path)
        except Exception as ie:
            indexed_ok = False
            index_error = str(ie)

        return {
            "ok": True,
            "pdf": fname,
            "bytes": written,
            "stored_in": str(PDF_DIR),
            "indexed": indexed_ok,
            "index_error": index_error,
            "pdf_count_now": len(list_pdfs()),
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc().splitlines()[-25:],
        }

# ----------------------------
# /ask (non-streaming) — stateless
# ----------------------------

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    pdf: Optional[str] = Query(None, description="Pin a specific PDF name (optional)"),
    page_hint: Optional[int] = Query(None, description="1-based page number to bias retrieval near (optional)"),
    force_docs: bool = Query(False, description="Force using PDFs even if router says no"),
    max_docs: int = Query(3),
    max_total_pages: int = Query(8),
    dpi: int = Query(140),
    force_vision: bool = Query(False),
):
    try:
        if is_short_topic_prompt(q) and not force_docs and not should_use_docs(q, pdf_pin=pdf, page_hint=page_hint):
            model = get_model(use_docs=False)
            parts = build_gemini_parts(q, sources_text="", images=None)
            resp = model.generate_content(parts, generation_config=get_generation_config(False))
            return {
                "ok": True,
                "answer": (resp.text or "").strip(),
                "used_docs": False,
                "sources_used": [],
                "retrieved_docs": [],
                "vision_used": False,
                "model": MODEL_CHAT,
            }

        use_docs = force_docs or should_use_docs(q, pdf_pin=pdf, page_hint=page_hint)

        selected: List[Tuple[str, List[int], float]] = []
        sources_text = ""
        cites: List[Tuple[str,int]] = []
        image_parts: Optional[List[Part]] = None

        if use_docs and list_pdfs():
            selected = iterative_select_pages(
                q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages
            )
            sources_text, cites = build_sources_bundle(selected)

            if selected:
                top_doc, pages, _score = selected[0]
                pdf_path = PDF_DIR / top_doc
                excerpt = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                if force_vision or needs_vision(q, excerpt):
                    image_parts = pages_to_parts(pdf_path, top_doc, pages[:2], dpi=dpi)

        used_docs_flag = bool(use_docs and sources_text)
        model = get_model(use_docs=used_docs_flag)
        parts = build_gemini_parts(q, sources_text, images=image_parts)

        resp = model.generate_content(parts, generation_config=get_generation_config(used_docs_flag))

        return {
            "ok": True,
            "answer": (resp.text or "").strip(),
            "used_docs": used_docs_flag,
            "sources_used": [{"doc": d, "page": p} for d, p in cites],
            "retrieved_docs": [s[0] for s in selected],
            "vision_used": bool(image_parts),
            "model": MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT,
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-16:]}

# ----------------------------
# /ask_stream (SSE) — stateless
# ----------------------------

@app.get("/ask_stream")
def ask_stream(
    q: str = Query(...),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    force_docs: bool = Query(False),
    max_docs: int = Query(3),
    max_total_pages: int = Query(8),
    dpi: int = Query(140),
    force_vision: bool = Query(False),
):
    def sse():
        try:
            if is_short_topic_prompt(q) and not force_docs and not should_use_docs(q, pdf_pin=pdf, page_hint=page_hint):
                model = get_model(use_docs=False)
                yield f"event: meta\ndata: model={MODEL_CHAT};used_docs=False;vision=False\n\n"

                parts = build_gemini_parts(q, sources_text="", images=None)
                stream = model.generate_content(parts, generation_config=get_generation_config(False), stream=True)

                for chunk in stream:
                    delta = getattr(chunk, "text", None)
                    if delta:
                        safe = delta.replace("\r","").replace("\n","\\n")
                        yield f"data: {safe}\n\n"

                yield "event: done\ndata: ok\n\n"
                return

            use_docs = force_docs or should_use_docs(q, pdf_pin=pdf, page_hint=page_hint)

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str,int]] = []
            image_parts: Optional[List[Part]] = None

            if use_docs and list_pdfs():
                selected = iterative_select_pages(
                    q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages
                )
                sources_text, cites = build_sources_bundle(selected)

                if selected:
                    top_doc, pages, _score = selected[0]
                    pdf_path = PDF_DIR / top_doc
                    excerpt = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                    if force_vision or needs_vision(q, excerpt):
                        image_parts = pages_to_parts(pdf_path, top_doc, pages[:2], dpi=dpi)

            used_docs_flag = bool(use_docs and sources_text)
            model_name = MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT

            yield f"event: meta\ndata: model={model_name};used_docs={used_docs_flag};vision={bool(image_parts)}\n\n"
            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d,p in cites]) + "\n\n"

            model = get_model(use_docs=used_docs_flag)
            parts = build_gemini_parts(q, sources_text, images=image_parts)
            stream = model.generate_content(parts, generation_config=get_generation_config(used_docs_flag), stream=True)

            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    safe = delta.replace("\r","").replace("\n","\\n")
                    yield f"data: {safe}\n\n"

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r","").replace("\n"," ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"},
    )

# ----------------------------
# /chat_stream (SSE) — HISTORY-AWARE (Fixes your reset issue)
# ----------------------------

@app.post("/chat_stream")
def chat_stream(payload: Dict[str, Any] = Body(...)):
    """
    Payload supports BOTH styles (backwards compatible):
    A) { messages: [...], chat_id: "abc" }  -> uses messages + stores to chat_id
    B) { message: "hi", chat_id: "abc" }   -> uses server memory for that chat_id
    """
    chat_id = (payload.get("chat_id") or "").strip()

    # Option A: full messages array
    messages = payload.get("messages")
    # Option B: single message
    single_message = (payload.get("message") or "").strip()

    pdf = payload.get("pdf")
    page_hint = payload.get("page_hint")
    force_docs = bool(payload.get("force_docs", False))

    # Determine last user message
    last_user = ""
    if isinstance(messages, list) and messages:
        for m in reversed(messages):
            if (m.get("role") == "user") and (m.get("content") or "").strip():
                last_user = (m.get("content") or "").strip()
                break
    elif single_message:
        last_user = single_message

    if not last_user:
        # Nothing to answer
        def empty():
            yield "event: error\ndata: No user message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    def sse():
        try:
            # Build a consistent history:
            # - If frontend sends messages: use those (and optionally store)
            # - Else: use server memory by chat_id
            if isinstance(messages, list) and messages:
                hist = []
                for m in messages:
                    r = (m.get("role") or "").lower()
                    c = (m.get("content") or "").strip()
                    if r in ("user", "assistant") and c:
                        hist.append({"role": r, "content": c})
                # store last chunk if chat_id exists
                if chat_id:
                    CHAT_STORE[chat_id] = hist[-CHAT_MAX_MESSAGES:]
                    _trim_chat(chat_id)
                history_for_prompt = hist
            else:
                # server-side memory path
                history_for_prompt = get_history(chat_id) if chat_id else []

                # also remember the new user turn before generating
                if chat_id:
                    remember(chat_id, "user", last_user)

            history_blob = build_history_blob(history_for_prompt, max_msgs=18)

            use_docs = force_docs or should_use_docs(last_user, pdf_pin=pdf, page_hint=page_hint)

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str,int]] = []

            if use_docs and list_pdfs():
                selected = iterative_select_pages(
                    last_user, pdf_pin=pdf, page_hint=page_hint, max_docs=3, max_total_pages=8
                )
                sources_text, cites = build_sources_bundle(selected)

            used_docs_flag = bool(use_docs and sources_text)
            model_name = MODEL_COMPLIANCE if used_docs_flag else MODEL_CHAT

            yield f"event: meta\ndata: model={model_name};used_docs={used_docs_flag};vision=False\n\n"
            if chat_id:
                yield f"event: meta\ndata: chat_id={chat_id}\n\n"
            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d,p in cites]) + "\n\n"

            model = get_model(use_docs=used_docs_flag)
            parts = build_gemini_parts(
                last_user,
                sources_text,
                images=None,
                history_blob=history_blob
            )

            stream = model.generate_content(
                parts,
                generation_config=get_generation_config(used_docs_flag),
                stream=True
            )

            full_answer = []
            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    full_answer.append(delta)
                    safe = delta.replace("\r","").replace("\n","\\n")
                    yield f"data: {safe}\n\n"

            answer_text = "".join(full_answer).strip()

            # Remember assistant turn (server memory path)
            if chat_id and not (isinstance(messages, list) and messages):
                remember(chat_id, "assistant", answer_text)

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r","").replace("\n"," ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"},
    )

# ----------------------------
# Optional: clear chat (handy for your saved chat list)
# ----------------------------

@app.post("/chat_clear")
def chat_clear(payload: Dict[str, Any] = Body(...)):
    chat_id = (payload.get("chat_id") or "").strip()
    if not chat_id:
        return {"ok": False, "error": "chat_id missing"}
    CHAT_STORE.pop(chat_id, None)
    return {"ok": True, "chat_id": chat_id, "cleared": True}

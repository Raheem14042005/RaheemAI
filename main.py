from __future__ import annotations

import os
import re
import math
import json
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import Counter

import httpx
import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    GenerationConfig,
)

# Optional: Document AI ingest helper
try:
    from docai_ingest import docai_extract_pdf_to_text
    _DOCAI_HELPER_AVAILABLE = True
except Exception:
    docai_extract_pdf_to_text = None
    _DOCAI_HELPER_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

DOCAI_DIR = BASE_DIR / "parsed_docai"
DOCAI_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "40"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "32000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Web search (optional but recommended)
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()  # https://serper.dev
WEB_ENABLED = (os.getenv("WEB_ENABLED", "true").lower() in ("1", "true", "yes", "on")) and bool(SERPER_API_KEY)

# Behavior
DEFAULT_EVIDENCE_MODE = os.getenv("DEFAULT_EVIDENCE_MODE", "false").lower() in ("1", "true", "yes", "on")

# Retrieval sizing
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))  # ~ a few paragraphs
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))  # how many snippets we feed to model
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "5"))

# ============================================================
# APP
# ============================================================

app = FastAPI(docs_url="/swagger", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raheemai.pages.dev",
        "https://raheem-ai.pages.dev",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VERTEX INIT
# ============================================================

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


def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])


def get_generation_config(is_evidence: bool) -> GenerationConfig:
    # Evidence mode: lower temperature for precision
    if is_evidence:
        return GenerationConfig(temperature=0.2, top_p=0.8, max_output_tokens=1200)
    # Normal chat
    return GenerationConfig(temperature=0.8, top_p=0.9, max_output_tokens=1200)


# ============================================================
# CHAT MEMORY (SERVER SIDE)
# ============================================================

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


def _normalize_incoming_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    hist: List[Dict[str, str]] = []
    for m in messages or []:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            hist.append({"role": r, "content": c})
    return hist


def _ensure_last_user_message(hist: List[Dict[str, str]], message: str) -> List[Dict[str, str]]:
    msg = (message or "").strip()
    if not msg:
        return hist
    if not hist:
        return [{"role": "user", "content": msg}]
    last = hist[-1]
    if last.get("role") != "user" or (last.get("content") or "").strip() != msg:
        hist = hist + [{"role": "user", "content": msg}]
    return hist


# ============================================================
# TEXT + TOKENIZATION
# ============================================================

STOPWORDS = {
    "the","and","or","of","to","in","a","an","for","on","with","is","are","be","as","at","from","by",
    "that","this","it","your","you","we","they","their","there","what","which","when","where","how",
    "can","shall","should","must","may","not","than","then","into","onto","also","such"
}


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-/\.%]*", t)
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks


# ============================================================
# CHUNK INDEX (BM25) — THIS IS THE BIG FIX
# ============================================================

@dataclass
class Chunk:
    doc: str
    page: int  # 1-based
    chunk_id: str
    text: str
    tf: Counter
    length: int


CHUNK_INDEX: Dict[str, Dict[str, Any]] = {}
# structure:
# CHUNK_INDEX[pdf_name] = {
#   "chunks": List[Chunk],
#   "df": Counter,
#   "avgdl": float,
#   "N": int
# }

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _split_into_chunks(text: str, target: int, overlap: int) -> List[str]:
    """
    Sliding-window chunking on paragraph boundaries.
    """
    text = clean_text(text)
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    buff = ""

    def flush():
        nonlocal buff
        if buff.strip():
            out.append(buff.strip())
        buff = ""

    for p in paras:
        if not buff:
            buff = p
        elif len(buff) + 2 + len(p) <= target:
            buff = buff + "\n\n" + p
        else:
            flush()
            # overlap: keep last overlap chars from previous chunk as prefix
            if overlap > 0 and out:
                tail = out[-1][-overlap:]
                buff = (tail + "\n\n" + p).strip()
            else:
                buff = p

    flush()
    return out


def bm25_score(query_toks: List[str], tf: Counter, df: Counter, N: int, dl: int, avgdl: float) -> float:
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


def list_pdfs() -> List[str]:
    files = []
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p.name)
    files.sort()
    return files


def index_pdf_to_chunks(pdf_path: Path) -> None:
    """
    Reads each page and creates paragraph chunks. Stores BM25 stats.
    """
    pdf_name = pdf_path.name
    doc = fitz.open(pdf_path)
    chunks: List[Chunk] = []
    df = Counter()

    try:
        for i in range(doc.page_count):
            page_no = i + 1
            raw = doc.load_page(i).get_text("text") or ""
            raw = clean_text(raw)

            # Split page into smaller chunks
            page_chunks = _split_into_chunks(raw, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
            for j, ch in enumerate(page_chunks):
                toks = tokenize(ch)
                tf = Counter(toks)
                df.update(set(tf.keys()))
                cid = _hash_id(f"{pdf_name}|p{page_no}|{j}|{ch[:80]}")
                chunks.append(Chunk(
                    doc=pdf_name,
                    page=page_no,
                    chunk_id=cid,
                    text=ch,
                    tf=tf,
                    length=len(toks)
                ))

        avgdl = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0
        CHUNK_INDEX[pdf_name] = {
            "chunks": chunks,
            "df": df,
            "avgdl": avgdl,
            "N": len(chunks),
            "pages": doc.page_count
        }
    finally:
        doc.close()


def ensure_chunk_indexed(pdf_name: str) -> None:
    if pdf_name in CHUNK_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf_to_chunks(p)


def search_chunks(question: str, top_k: int = TOP_K_CHUNKS, pinned_pdf: Optional[str] = None) -> List[Chunk]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    candidates: List[Chunk] = []

    search_space = [pinned_pdf] if pinned_pdf and pinned_pdf in pdfs else pdfs

    for pdf_name in search_space:
        ensure_chunk_indexed(pdf_name)
        idx = CHUNK_INDEX.get(pdf_name)
        if not idx:
            continue

        df = idx["df"]
        N = idx["N"]
        avgdl = idx["avgdl"]

        scored: List[Tuple[float, Chunk]] = []
        for ch in idx["chunks"]:
            s = bm25_score(qt, ch.tf, df, N, ch.length, avgdl)
            if s > 0:
                scored.append((s, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates.extend([c for _, c in scored[: max(2, top_k)]])

        # If pinned, don’t search other PDFs
        if pinned_pdf and pdf_name == pinned_pdf:
            break

    # Global re-rank (simple): prefer higher BM25 + shorter chunks (more precise)
    # We don’t have the scores here, so re-score quickly on a merged df is overkill.
    # Instead: keep doc-local tops already; then just trim.
    # (Good enough in practice with paragraph chunking.)
    # De-dup by chunk_id.
    seen = set()
    uniq: List[Chunk] = []
    for c in candidates:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        uniq.append(c)

    return uniq[:top_k]


# ============================================================
# DOCAI SUPPORT (kept, but used as an extra source)
# ============================================================

def _docai_chunk_files_for(pdf_name: str) -> List[Path]:
    stem = Path(pdf_name).stem
    out = []
    for p in DOCAI_DIR.iterdir():
        if p.is_file() and p.name.startswith(stem + "_p") and p.suffix.lower() == ".txt":
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _parse_range_from_chunk_filename(path: Path) -> Optional[Tuple[int, int]]:
    m = re.search(r"_p(\d+)\-(\d+)\.txt$", path.name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def docai_search_text(pdf_name: str, question: str, k: int = 2) -> List[Tuple[str, str]]:
    """
    Returns list of (label, excerpt)
    """
    files = _docai_chunk_files_for(pdf_name)
    if not files:
        return []
    qt = tokenize(question)
    if not qt:
        return []
    scored: List[Tuple[float, str, str]] = []
    for f in files:
        rng = _parse_range_from_chunk_filename(f)
        if not rng:
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        toks = tokenize(txt)
        tf = Counter(toks)
        # naive score
        overlap = sum(tf.get(t, 0) for t in qt)
        if overlap <= 0:
            continue
        label = f"{pdf_name} DOC.AI pages {rng[0]}-{rng[1]}"
        excerpt = clean_text(txt)
        if len(excerpt) > 3500:
            excerpt = excerpt[:3500] + "..."
        scored.append((float(overlap), label, excerpt))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(lab, ex) for _, lab, ex in scored[:k]]


# ============================================================
# INTENT DETECTION
# ============================================================

COMPLIANCE_KEYWORDS = [
    "tgd", "technical guidance", "building regulations", "building regs",
    "part a","part b","part c","part d","part e","part f","part g","part h",
    "part j","part k","part l","part m",
    "fire","escape","travel distance","compartment","smoke","fire cert",
    "access","accessible","wheelchair","dac",
    "ber","deap","u-value","y-value","airtight","thermal bridge",
    "means of escape"
]

PART_PATTERN = re.compile(r"\bpart\s*[a-m]\b", re.I)

NUMERIC_TRIGGERS = [
    "minimum","maximum","min","max","limit",
    "distance","width","height",
    "u-value","y-value","rise","riser","going","pitch",
    "stairs","stair","staircase","landing","headroom",
    "travel distance"
]

NUM_WITH_UNIT_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|m|metre|meter|w/m²k|w\/m2k|%)\b", re.I)


def is_compliance_question(q: str) -> bool:
    ql = (q or "").lower()
    if PART_PATTERN.search(ql):
        return True
    return any(k in ql for k in COMPLIANCE_KEYWORDS)


def is_numeric_compliance(q: str) -> bool:
    ql = (q or "").lower()
    return is_compliance_question(ql) and any(k in ql for k in NUMERIC_TRIGGERS)


def auto_pin_pdf(question: str) -> Optional[str]:
    q = (question or "").lower()
    pdfs = set(list_pdfs())

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in pdfs:
                return c
        return None

    if any(w in q for w in ["stairs","stair","staircase","riser","going","pitch","headroom","handrail","balustrade","landing"]):
        return pick(["Technical Guidance Document K.pdf", "TGD K.pdf"])

    if any(w in q for w in ["access","accessible","wheelchair","ramp","dac","part m"]):
        return pick(["Technical Guidance Document M.pdf", "TGD M.pdf"])

    if any(w in q for w in ["fire","escape","travel distance","means of escape","compartment","smoke"]):
        return pick(["Technical Guidance Document B Dwellings.pdf", "Technical Guidance Document B Non Dwellings.pdf"])

    if any(w in q for w in ["u-value","y-value","thermal","ber","deap","energy","nzeb","primary energy"]):
        return pick(["Technical Guidance Document L Dwellings.pdf", "Technical Guidance Document L Non Dwellings.pdf"])

    return None


# ============================================================
# WEB SEARCH (SERPER) — makes it “ChatGPT-like”
# ============================================================

async def web_search_serper(query: str, k: int = TOP_K_WEB) -> List[Dict[str, str]]:
    if not WEB_ENABLED:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": max(3, min(10, k))}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    for item in (data.get("organic") or [])[:k]:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if link:
            out.append({"title": title, "url": link, "snippet": snippet})
    return out


# ============================================================
# PROMPTS — rebuilt to behave like “ChatGPT + citations”
# ============================================================

SYSTEM_PROMPT_NORMAL = """
You are Raheem AI.

Write like a top-tier assistant:
- Clear, direct, and helpful.
- Use Markdown when it improves readability (headings, bullets, short tables).
- If you are unsure, say so and explain what would confirm it.

If the user asks about Irish building regulations / TGDs, prefer evidence from supplied SOURCES.
If web sources are used, cite them as URLs.
""".strip()

SYSTEM_PROMPT_EVIDENCE = """
You are Raheem AI in Evidence Mode.

Rules:
1) You MUST only assert numeric compliance limits (mm, m, %, W/m²K, etc.) if the exact number + unit appears in the provided SOURCES.
2) When you give a numeric limit, include a short quote (1–2 lines) copied from SOURCES that contains that number.
3) For each requirement you state, cite the evidence using:
   - For PDFs: (Document: <name>, p.<page>)
   - For web: (URL)
4) If the SOURCES do not contain the exact limit, say you cannot confirm it from the current evidence.

Output style:
- Use Markdown.
- Be concise but complete.
""".strip()


def build_contents(history: List[Dict[str, str]], user_message: str, sources_block: str) -> List[Content]:
    contents: List[Content] = []
    for m in history or []:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if r == "user":
            contents.append(Content(role="user", parts=[Part.from_text(c)]))
        elif r == "assistant":
            contents.append(Content(role="model", parts=[Part.from_text(c)]))

    final_user = (user_message or "").strip()
    if sources_block.strip():
        final_user += "\n\nSOURCES:\n" + sources_block.strip()

    contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    return contents


def build_sources_block(chunks: List[Chunk], web_results: List[Dict[str, str]], docai_hits: List[Tuple[str, str]]) -> str:
    parts: List[str] = []

    if chunks:
        parts.append("PDF EVIDENCE:")
        for c in chunks:
            parts.append(f"[PDF] {c.doc} | p.{c.page} | id:{c.chunk_id}\n{c.text}")

    if docai_hits:
        parts.append("\nDOC.AI EVIDENCE (raw extraction, use carefully):")
        for label, text in docai_hits:
            parts.append(f"[DOCAI] {label}\n{text}")

    if web_results:
        parts.append("\nWEB EVIDENCE:")
        for i, w in enumerate(web_results, start=1):
            parts.append(f"[WEB {i}] {w.get('title','').strip()}\nURL: {w.get('url','').strip()}\n{w.get('snippet','').strip()}")

    return "\n\n".join(parts).strip()


# ============================================================
# UPLOAD + INDEXING
# ============================================================

def _split_docai_combined_to_chunks(combined: str) -> List[Tuple[Tuple[int, int], str]]:
    if not combined:
        return []
    pattern = re.compile(r"---\s*DOC_AI_PAGES\s+(\d+)\-(\d+)\s*---", re.IGNORECASE)
    parts = pattern.split(combined)
    out: List[Tuple[Tuple[int, int], str]] = []
    if len(parts) < 4:
        return out
    i = 1
    while i + 2 < len(parts):
        a = int(parts[i])
        b = int(parts[i + 1])
        txt = parts[i + 2].strip()
        out.append(((a, b), txt))
        i += 3
    return out


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

    # Clear indexes
    CHUNK_INDEX.pop(dest.name, None)

    # Build chunk index immediately (fast enough for TGDs)
    try:
        index_pdf_to_chunks(dest)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Indexed failed: {repr(e)}"}, status_code=500)

    # Optional: DocAI parse
    docai_ok = False
    docai_chunks_saved = 0
    docai_error = None
    try:
        if _DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text:
            if (os.getenv("DOCAI_PROCESSOR_ID") or "").strip() and (os.getenv("DOCAI_LOCATION") or "").strip():
                combined_text, _chunk_ranges = docai_extract_pdf_to_text(str(dest), chunk_pages=15)
                chunks = _split_docai_combined_to_chunks(combined_text)
                stem = dest.stem
                for (a, b), txt in chunks:
                    out_path = DOCAI_DIR / f"{stem}_p{a}-{b}.txt"
                    out_path.write_text(txt, encoding="utf-8", errors="ignore")
                    docai_chunks_saved += 1
                docai_ok = docai_chunks_saved > 0
    except Exception as e:
        docai_error = repr(e)

    return {
        "ok": True,
        "stored_as": dest.name,
        "pdf_count": len(list_pdfs()),
        "indexed_chunks": CHUNK_INDEX.get(dest.name, {}).get("N", 0),
        "docai": {
            "attempted": bool(_DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text),
            "ok": docai_ok,
            "chunks_saved": docai_chunks_saved,
            "error": docai_error,
        }
    }


@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}


# ============================================================
# HEALTH + ROOT
# ============================================================

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
        "docai_helper": _DOCAI_HELPER_AVAILABLE,
        "docai_processor_id_present": bool((os.getenv("DOCAI_PROCESSOR_ID") or "").strip()),
        "docai_location_present": bool((os.getenv("DOCAI_LOCATION") or "").strip()),
        "web_enabled": WEB_ENABLED,
    }


# ============================================================
# STREAMING CORE
# ============================================================

async def _stream_answer_async(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None
):
    """
    Async generator for SSE.
    """
    try:
        user_msg = (message or "").strip()
        if not user_msg:
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()

        # Canonical history
        if isinstance(messages, list) and messages:
            hist = _normalize_incoming_messages(messages)
            hist = _ensure_last_user_message(hist, user_msg)
            if chat_id:
                CHAT_STORE[chat_id] = hist[-CHAT_MAX_MESSAGES:]
                _trim_chat(chat_id)
            history_for_prompt = CHAT_STORE.get(chat_id, hist)
        else:
            history_for_prompt = get_history(chat_id) if chat_id else []
            if chat_id:
                remember(chat_id, "user", user_msg)
                history_for_prompt = get_history(chat_id)

        # Decide modes
        evidence_mode = force_docs or DEFAULT_EVIDENCE_MODE or is_compliance_question(user_msg)
        numeric_needed = is_numeric_compliance(user_msg)

        pinned = pdf or auto_pin_pdf(user_msg)

        # Retrieve PDF evidence
        chunks: List[Chunk] = []
        if list_pdfs():
            chunks = search_chunks(user_msg, top_k=TOP_K_CHUNKS, pinned_pdf=pinned)

        # Optional DocAI hits (extra)
        docai_hits: List[Tuple[str, str]] = []
        if pinned and _docai_chunk_files_for(pinned):
            docai_hits = docai_search_text(pinned, user_msg, k=2)

        # Web evidence (only if enabled and either non-compliance question OR no good PDF evidence)
        web_results: List[Dict[str, str]] = []
        if WEB_ENABLED and WEB_ENABLED and WEB_ENABLED:
            if (not evidence_mode) or (evidence_mode and not chunks and not numeric_needed):
                web_results = await web_search_serper(user_msg, k=TOP_K_WEB)

        sources_block = build_sources_block(chunks, web_results, docai_hits)

        # HARD safety rule: numeric compliance requires numeric evidence in SOURCES
        if numeric_needed:
            if not chunks and not docai_hits:
                refusal = (
                    "I can’t confirm the exact numeric requirement yet because I don’t have any relevant TGD evidence loaded for this question.\n\n"
                    "Upload the relevant TGD (or tell me which one), and I’ll quote the exact line containing the number."
                )
                if chat_id:
                    remember(chat_id, "assistant", refusal)
                yield f"data: {refusal.replace(chr(10),'\\\\n')}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        system_prompt = SYSTEM_PROMPT_EVIDENCE if evidence_mode else SYSTEM_PROMPT_NORMAL
        model_name = MODEL_COMPLIANCE if evidence_mode else MODEL_CHAT

        model = get_model(model_name=model_name, system_prompt=system_prompt)
        contents = build_contents(history_for_prompt, user_msg, sources_block)

        stream = model.generate_content(
            contents,
            generation_config=get_generation_config(evidence_mode),
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

        # Store
        if chat_id:
            remember(chat_id, "assistant", final_text)

        yield "event: done\ndata: ok\n\n"

    except Exception as e:
        msg = str(e).replace("\r", "").replace("\n", " ")
        yield f"event: error\ndata: {msg}\n\n"
        yield "event: done\ndata: ok\n\n"


def _stream_answer_sync(*args, **kwargs):
    """
    FastAPI StreamingResponse can take a sync generator easily.
    We wrap the async generator.
    """
    import anyio
    async_gen = _stream_answer_async(*args, **kwargs)

    async def run():
        async for item in async_gen:
            yield item

    return anyio.to_thread.run_sync(lambda: None)  # no-op placeholder


@app.get("/chat_stream")
def chat_stream_get(
    q: str = Query(""),
    chat_id: str = Query(""),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
):
    async def gen():
        async for s in _stream_answer_async(chat_id.strip(), q, force_docs, pdf, page_hint, messages=None):
            yield s

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat_stream")
def chat_stream_post(payload: Dict[str, Any] = Body(...)):
    chat_id = (payload.get("chat_id") or "").strip()
    message = (payload.get("message") or payload.get("q") or "").strip()
    force_docs = bool(payload.get("force_docs", False))
    pdf = payload.get("pdf")
    page_hint = payload.get("page_hint")
    messages = payload.get("messages")

    async def gen():
        async for s in _stream_answer_async(chat_id, message, force_docs, pdf, page_hint, messages=messages):
            yield s

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

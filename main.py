from __future__ import annotations

import os
import re
import math
import json
import time
import tempfile
import hashlib
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import httpx
import fitz  # PyMuPDF
import boto3

from fastapi import FastAPI, UploadFile, File, Query, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationConfig

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("raheemai")

# ============================================================
# REQUEST MODELS
# ============================================================

class ChatBody(BaseModel):
    message: str
    messages: Optional[List[Dict[str, Any]]] = None


# ============================================================
# BASE DIR + ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

CREATOR_NAME = os.getenv("CREATOR_NAME", "Abdulraheem Ahmed").strip()
CREATOR_TITLE = os.getenv(
    "CREATOR_TITLE",
    "4th year Architectural Technology student at Technological University Dublin (TUD)",
).strip()
PRODUCT_NAME = os.getenv("PRODUCT_NAME", "Raheem AI").strip()
PRODUCT_VERSION = os.getenv("PRODUCT_VERSION", "1.0.0").strip()

ROADMAP_FILE = Path(os.getenv("ROADMAP_FILE", str(BASE_DIR / "roadmap.json")))

# ---- Storage / R2 ----
R2_ENABLED = os.getenv("R2_ENABLED", "false").lower() in ("1", "true", "yes", "on")
R2_BUCKET = (os.getenv("R2_BUCKET") or "").strip()
R2_ENDPOINT = (os.getenv("R2_ENDPOINT") or "").strip()
R2_ACCESS_KEY_ID = (os.getenv("R2_ACCESS_KEY_ID") or "").strip()
R2_SECRET_ACCESS_KEY = (os.getenv("R2_SECRET_ACCESS_KEY") or "").strip()
R2_PREFIX = (os.getenv("R2_PREFIX") or "pdfs/").strip()
if R2_PREFIX and not R2_PREFIX.endswith("/"):
    R2_PREFIX += "/"

def r2_client():
    if not (R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        raise RuntimeError(
            "Missing R2 creds/env (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)"
        )
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )

# ---- Optional: Vertex Embeddings ----
try:
    from vertexai.language_models import TextEmbeddingModel  # type: ignore
    _VERTEX_EMBEDDINGS_AVAILABLE = True
except Exception:
    TextEmbeddingModel = None  # type: ignore
    _VERTEX_EMBEDDINGS_AVAILABLE = False

# ---- Optional: Document AI ingest helper ----
try:
    from docai_ingest import docai_extract_pdf_to_text
    _DOCAI_HELPER_AVAILABLE = True
except Exception:
    docai_extract_pdf_to_text = None
    _DOCAI_HELPER_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "40"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "32000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Web search (Serper)
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()
WEB_ENABLED = (os.getenv("WEB_ENABLED", "true").lower() in ("1", "true", "yes", "on")) and bool(SERPER_API_KEY)

# Evidence defaults
DEFAULT_EVIDENCE_MODE = os.getenv("DEFAULT_EVIDENCE_MODE", "false").lower() in ("1", "true", "yes", "on")

# Retrieval sizing
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "5"))

# Vector embeddings (optional)
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "true").lower() in ("1", "true", "yes", "on")
EMBED_MODEL_NAME = (os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004") or "").strip()
EMBED_TOPK = int(os.getenv("EMBED_TOPK", "24"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rerank (LLM reranker)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
RERANK_TOPK = int(os.getenv("RERANK_TOPK", str(TOP_K_CHUNKS)))

# Broad vs precise answering controls
BROAD_DOC_DIVERSITY_K = int(os.getenv("BROAD_DOC_DIVERSITY_K", "5"))
BROAD_TOPIC_HITS_K = int(os.getenv("BROAD_TOPIC_HITS_K", "3"))
BROAD_MAX_SUBQUERIES = int(os.getenv("BROAD_MAX_SUBQUERIES", "8"))

# Web allowlist
WEB_ALLOWLIST_DEFAULT = [
    "irishstatutebook.ie",
    "gov.ie",
    "housing.gov.ie",
    "nsai.ie",
    "dublincity.ie",
    "dlrcoco.ie",
    "corkcity.ie",
    "kildarecoco.ie",
    "galwaycity.ie",
]
WEB_ALLOWLIST = [d.strip().lower() for d in (os.getenv("WEB_ALLOWLIST", "") or "").split(",") if d.strip()]
if not WEB_ALLOWLIST:
    WEB_ALLOWLIST = WEB_ALLOWLIST_DEFAULT

# Numeric verification
VERIFY_NUMERIC = os.getenv("VERIFY_NUMERIC", "true").lower() in ("1", "true", "yes", "on")

# Rules layer
RULES_FILE = Path(os.getenv("RULES_FILE", str(BASE_DIR / "rules.json")))
RULES_ENABLED = os.getenv("RULES_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Timeouts
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "18"))

# Web fetch behaviour
MAX_WEB_BYTES = int(os.getenv("MAX_WEB_BYTES", str(2_500_000)))
WEB_RETRIES = int(os.getenv("WEB_RETRIES", "2"))
WEB_RETRY_BACKOFF = float(os.getenv("WEB_RETRY_BACKOFF", "0.6"))
WEB_CACHE_TTL_SECONDS = int(os.getenv("WEB_CACHE_TTL_SECONDS", str(12 * 60 * 60)))

# PDF images (kept for later; not used here)
PDF_IMAGE_EXTRACT = os.getenv("PDF_IMAGE_EXTRACT", "false").lower() in ("1", "true", "yes", "on")
PDF_IMAGE_MAX_PAGES = int(os.getenv("PDF_IMAGE_MAX_PAGES", "12"))
PDF_IMAGE_MAX_IMAGES = int(os.getenv("PDF_IMAGE_MAX_IMAGES", "24"))
PDF_IMAGE_MIN_PIXELS = int(os.getenv("PDF_IMAGE_MIN_PIXELS", "120000"))

if os.getenv("RENDER"):
    DATA_DIR = Path("/tmp/raheemai")
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR))).resolve()

PDF_DIR = DATA_DIR / "pdfs"
DOCAI_DIR = DATA_DIR / "parsed_docai"
CACHE_DIR = DATA_DIR / "cache"
WEB_CACHE_DIR = CACHE_DIR / "web"

ANSWER_CACHE_DIR = CACHE_DIR / "answers"
ANSWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))  # 7 days

for d in (PDF_DIR, DOCAI_DIR, CACHE_DIR, WEB_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or "").strip()

def require_admin_key(x_api_key: Optional[str]) -> Optional[JSONResponse]:
    if not ADMIN_API_KEY:
        return None  # dev mode
    if (x_api_key or "").strip() != ADMIN_API_KEY:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
    return None


# ============================================================
# FASTAPI APP + CORS
# ============================================================

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

raw_origins = os.getenv("CORS_ORIGINS", "")
if raw_origins:
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
else:
    allowed_origins = [
        "https://raheemai.pages.dev",
        "https://raheem-ai.pages.dev",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VERTEX INIT
# ============================================================

_VERTEX_READY = False
_VERTEX_ERR: Optional[str] = None
_EMBED_MODEL: Any = None

def ensure_vertex_ready() -> None:
    global _VERTEX_READY, _VERTEX_ERR, _EMBED_MODEL
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

        _EMBED_MODEL = None
        if EMBED_ENABLED and _VERTEX_EMBEDDINGS_AVAILABLE and TextEmbeddingModel is not None:
            try:
                _EMBED_MODEL = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)
            except Exception as e:
                log.warning("Embeddings model init failed: %s", repr(e))
                _EMBED_MODEL = None

        _VERTEX_READY = True
        _VERTEX_ERR = None
        log.info("Vertex ready (project=%s location=%s)", GCP_PROJECT_ID, GCP_LOCATION)

    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)
        log.error("Vertex init failed: %s", _VERTEX_ERR)

def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])

def get_generation_config(strictness: int) -> GenerationConfig:
    """
    strictness:
      0 = chatty
      1 = cite-if-available
      2 = cite + avoid guessing on requirements
      3 = hard evidence mode (numbers must be in evidence; quote lines for numbers)
    """
    if strictness >= 3:
        return GenerationConfig(temperature=0.15, top_p=0.8, max_output_tokens=3500)
    if strictness == 2:
        return GenerationConfig(temperature=0.25, top_p=0.85, max_output_tokens=3500)
    if strictness == 1:
        return GenerationConfig(temperature=0.55, top_p=0.9, max_output_tokens=3500)
    return GenerationConfig(temperature=0.7, top_p=0.92, max_output_tokens=3500)


# ============================================================
# CHAT MEMORY (server-side, optional via chat_id)
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

NUM_WITH_UNIT_RE = re.compile(
    r"\b\d+(\.\d+)?\s*("
    r"mm|cm|m|metre|meter|"
    r"w/m²k|w\/m2k|"
    r"%|"
    r"kwh|kwh\/m2\/yr|kwh\/m²\/yr|"
    r"lux|lm|lumen|lumens|"
    r"min|mins|minute|minutes|"
    r"hr|hrs|hour|hours|"
    r"sec|secs|second|seconds|"
    r"pa|kpa|"
    r"kw|w|"
    r"°c|c"
    r")\b",
    re.I
)
ANY_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")

SECTION_RE = re.compile(r"^\s*(?:section\s+)?(\d+(?:\.\d+){0,4})\b", re.I)
TABLE_RE = re.compile(r"\btable\s+([a-z0-9][a-z0-9\.\-]*)\b", re.I)
DIAGRAM_RE = re.compile(r"\bdiagram\s+([a-z0-9][a-z0-9\.\-]*)\b", re.I)

VISUAL_TERMS_RE = re.compile(r"\b(diagram|figure|fig\.|table|chart|graph|drawing|schematic)\b", re.I)

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

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def wants_visual_evidence(q: str) -> bool:
    return bool(VISUAL_TERMS_RE.search(q or ""))


# ============================================================
# SSE HELPERS (real streaming, not “buffer then dump”)
# ============================================================

BULLET_STAR_RE = re.compile(r"(?m)^\s*\*\s+")
EXCESS_SPACES_RE = re.compile(r"[ \t]{2,}")
MULTI_BLANKLINES_RE = re.compile(r"\n{3,}")

def _light_normalize_for_stream(text: str) -> str:
    """
    Don't do heavy rewriting on streamed chunks.
    Just keep bullets sane and whitespace readable.
    """
    t = (text or "").replace("\r", "")
    t = BULLET_STAR_RE.sub("- ", t)
    t = EXCESS_SPACES_RE.sub(" ", t)
    t = MULTI_BLANKLINES_RE.sub("\n\n", t)
    return t

def normalize_model_output(text: str) -> str:
    return _light_normalize_for_stream(text).strip()

SENT_END_RE = re.compile(r"([.!?])\s+")
HEADING_LINE_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9 \-/]{3,}|📅|🧾|📝)\b", re.M)

def enforce_chatgpt_paragraphs(text: str) -> str:
    """
    Make output read like ChatGPT:
    - No mega-paragraphs
    - 2–4 sentences per paragraph
    - Keeps bullet blocks intact
    """
    t = (text or "").replace("\r", "").strip()
    if not t:
        return t

    lines = t.split("\n")
    out_lines: List[str] = []

    para: List[str] = []
    sent_count = 0
    in_bullets = False

    def flush_para():
        nonlocal para, sent_count
        if para:
            out_lines.append(" ".join(para).strip())
            out_lines.append("")  # blank line between paragraphs
        para = []
        sent_count = 0

    for ln in lines:
        raw = ln.rstrip()

        # preserve bullet blocks
        if raw.strip().startswith(("-", "•")):
            if para:
                flush_para()
            in_bullets = True
            out_lines.append(raw)
            continue
        else:
            if in_bullets and raw.strip() == "":
                out_lines.append("")
                in_bullets = False
                continue
            if in_bullets:
                out_lines.append(raw)
                continue

        if raw.strip() == "":
            flush_para()
            continue

        # if line looks like a heading, force paragraph break around it
        if HEADING_LINE_RE.match(raw.strip()) and len(raw.strip()) <= 60:
            flush_para()
            out_lines.append(raw.strip())
            out_lines.append("")
            continue

        # normal text line -> sentence-based paragraphing
        piece = raw.strip()
        para.append(piece)
        sent_count += len(SENT_END_RE.findall(piece))
        if sent_count >= 3:
            flush_para()

    flush_para()

    t2 = "\n".join(out_lines)
    t2 = re.sub(r"\n{3,}", "\n\n", t2).strip()
    return t2

def sse_send(text: str, max_frame_chars: int = 6000) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "").replace("\x00", "")

    lines = text.split("\n")
    frames: List[str] = []
    buff: List[str] = []
    size = 0

    for ln in lines:
        add = len(ln) + 1
        if size + add > max_frame_chars and buff:
            frames.append("".join(f"data: {x}\n" for x in buff) + "\n")
            buff = []
            size = 0
        buff.append(ln)
        size += add

    if buff:
        frames.append("".join(f"data: {x}\n" for x in buff) + "\n")

    return "".join(frames)

def _should_flush(buf: str) -> bool:
    """
    Flush on paragraph breaks, or on sentence endings once buffer gets big enough.
    """
    if not buf:
        return False
    if "\n\n" in buf and len(buf) >= 250:
        return True
    if len(buf) >= 700 and re.search(r"[\.!?]\s+$", buf):
        return True
    if len(buf) >= 1200:
        return True
    return False

async def stream_llm_text_as_sse(stream_obj) -> Tuple[str, List[str]]:
    """
    Streams model output in clean-ish chunks.
    Returns: (all_text, sse_frames_list)
    """
    full_parts: List[str] = []
    frames: List[str] = []
    buf = ""

    for ch in stream_obj:
        delta = getattr(ch, "text", None)
        if not delta:
            continue

        full_parts.append(delta)
        buf += delta

        if _should_flush(buf):
            chunk = _light_normalize_for_stream(buf)
            frames.append(sse_send(chunk))
            buf = ""

    if buf.strip():
        frames.append(sse_send(_light_normalize_for_stream(buf)))

    return ("".join(full_parts), frames)


# ============================================================
# CHUNK / INDICES
# ============================================================

@dataclass
class Chunk:
    doc: str
    page: int  # 1-based
    chunk_id: str
    text: str
    tf: Counter
    length: int
    section: str = ""
    heading: str = ""
    table: str = ""
    diagram: str = ""

CHUNK_INDEX: Dict[str, Dict[str, Any]] = {}
EMBED_INDEX: Dict[str, Dict[str, Any]] = {}
PDF_DISPLAY_NAME: Dict[str, str] = {}

# ============================================================
# PDF UTILITIES
# ============================================================

def _pdf_fingerprint(pdf_path: Path) -> str:
    st = pdf_path.stat()
    s = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def get_pdf_display_name(pdf_path: Path) -> str:
    """
    Returns a clean, human-readable name for a PDF, prioritizing the filename.
    This avoids using potentially confusing metadata titles from inside the PDF.
    """
    # Use the filename (without extension) as the primary identifier.
    # Replace dashes and underscores with spaces for readability.
    stem = pdf_path.stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return re.sub(r"\s{2,}", " ", stem) or pdf_path.name

def list_pdfs() -> List[str]:
    if R2_ENABLED:
        try:
            c = r2_client()
            out = []
            token = None

            prefix = (R2_PREFIX or "").strip()
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            while True:
                kwargs = {"Bucket": R2_BUCKET, "MaxKeys": 1000}
                if prefix:
                    kwargs["Prefix"] = prefix
                if token:
                    kwargs["ContinuationToken"] = token

                resp = c.list_objects_v2(**kwargs)
                for obj in (resp.get("Contents") or []):
                    key = (obj.get("Key") or "").strip()
                    if key.lower().endswith(".pdf"):
                        out.append(Path(key).name)

                if resp.get("IsTruncated"):
                    token = resp.get("NextContinuationToken")
                else:
                    break

            out.sort()
            return out
        except Exception as e:
            log.warning("R2 list_pdfs failed, falling back to local: %s", repr(e))

    files = []
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p.name)
    files.sort()
    return files

def is_planning_relevant_pdf(pdf_name: str) -> bool:
    name = (PDF_DISPLAY_NAME.get(pdf_name, pdf_name) or "").lower()
    planning_markers = [
        "planning", "development plan", "planning and development",
        "an bord plean", "coimisiún plean", "appeal", "part 8",
        "dublin city", "dlr", "county council", "city council",
        "irish statute", "statutebook", "regulations", "s.i."
    ]
    tgd_markers = ["tgd", "technical guidance document", "part a", "part b", "part m"]
    if any(m in name for m in tgd_markers):
        return False
    return any(m in name for m in planning_markers)
    
def ensure_pdf_local(pdf_name: str) -> Optional[Path]:
    pdf_name = Path(pdf_name).name
    local_path = PDF_DIR / pdf_name
    if local_path.exists():
        return local_path

    if not R2_ENABLED:
        return None

    try:
        c = r2_client()

        key1 = f"{R2_PREFIX}{pdf_name}" if R2_PREFIX else pdf_name
        try:
            c.download_file(R2_BUCKET, key1, str(local_path))
            if local_path.exists():
                return local_path
        except Exception:
            pass

        c.download_file(R2_BUCKET, pdf_name, str(local_path))
        return local_path if local_path.exists() else None
    except Exception as e:
        log.warning("ensure_pdf_local failed: %s", repr(e))
        return None


# ============================================================
# CHUNKING / INDEXING
# ============================================================

def looks_like_table_block(text: str) -> bool:
    if not text:
        return False
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False

    multi_gap = sum(1 for ln in lines if re.search(r"\s{2,}", ln))
    pipes = sum(1 for ln in lines if "|" in ln)
    many_numbers = sum(1 for ln in lines if ANY_NUMBER_RE.search(ln))

    return (multi_gap >= 3) or (pipes >= 3) or (many_numbers >= 4 and len(lines) >= 6)

def _split_into_chunks(text: str, target: int, overlap: int) -> List[str]:
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
        if looks_like_table_block(p):
            flush()
            out.append(p.strip())
            continue

        if not buff:
            buff = p
        elif len(buff) + 2 + len(p) <= target:
            buff = buff + "\n\n" + p
        else:
            flush()
            if overlap > 0 and out:
                tail = out[-1][-overlap:]
                buff = (tail + "\n\n" + p).strip()
            else:
                buff = p

    flush()
    return out

def _detect_context_lines(page_text: str) -> Tuple[str, str, str, str]:
    section = ""
    table = ""
    diagram = ""

    lines = [ln.strip() for ln in (page_text or "").split("\n") if ln.strip()]
    for ln in lines[:220]:
        m = SECTION_RE.match(ln)
        if m:
            section = m.group(1).strip()
        mt = TABLE_RE.search(ln)
        if mt:
            table = mt.group(1).strip()
        md = DIAGRAM_RE.search(ln)
        if md:
            diagram = md.group(1).strip()

    heading = ""
    for ln in lines[:120]:
        if len(ln) < 6:
            continue
        if SECTION_RE.match(ln):
            continue
        if re.fullmatch(r"[\d\.\-]+", ln):
            continue
        heading = ln[:140]
        break

    return section, table, diagram, heading

def _index_cache_paths(pdf_path: Path) -> Dict[str, Path]:
    fp = _pdf_fingerprint(pdf_path)
    return {
        "meta": CACHE_DIR / f"meta_{fp}.json",
        "chunks": CACHE_DIR / f"chunks_{fp}.jsonl",
        "emb": CACHE_DIR / f"embed_{fp}.jsonl",
    }

def _load_index_from_cache(pdf_path: Path) -> bool:
    pdf_name = pdf_path.name
    paths = _index_cache_paths(pdf_path)
    meta_path = paths["meta"]
    chunks_path = paths["chunks"]

    if not meta_path.exists() or not chunks_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if (meta.get("pdf_name") or "").strip() != pdf_name:
            return False

        disp = (meta.get("display_name") or "").strip()
        if disp:
            PDF_DISPLAY_NAME[pdf_name] = disp

        df = Counter(meta.get("df", {}))
        avgdl = float(meta.get("avgdl", 0.0))
        pages = int(meta.get("pages", 0))

        chunks: List[Chunk] = []
        with chunks_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    Chunk(
                        doc=obj["doc"],
                        page=int(obj["page"]),
                        chunk_id=obj["chunk_id"],
                        text=obj["text"],
                        tf=Counter(obj.get("tf", {})),
                        length=int(obj.get("length", 0)),
                        section=(obj.get("section") or ""),
                        heading=(obj.get("heading") or ""),
                        table=(obj.get("table") or ""),
                        diagram=(obj.get("diagram") or ""),
                    )
                )

        CHUNK_INDEX[pdf_name] = {
            "chunks": chunks,
            "df": df,
            "avgdl": avgdl,
            "N": len(chunks),
            "pages": pages,
            "fingerprint": meta.get("fingerprint", ""),
        }
        return True
    except Exception as e:
        log.warning("Index cache load failed for %s: %s", pdf_name, repr(e))
        return False

def _save_index_to_cache(pdf_path: Path, idx: Dict[str, Any]) -> None:
    paths = _index_cache_paths(pdf_path)
    meta_path = paths["meta"]
    chunks_path = paths["chunks"]

    try:
        df: Counter = idx["df"]
        meta = {
            "pdf_name": pdf_path.name,
            "fingerprint": _pdf_fingerprint(pdf_path),
            "display_name": PDF_DISPLAY_NAME.get(pdf_path.name) or get_pdf_display_name(pdf_path),
            "pages": int(idx.get("pages", 0)),
            "avgdl": float(idx.get("avgdl", 0.0)),
            "df": dict(df),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with chunks_path.open("w", encoding="utf-8") as f:
            for c in idx["chunks"]:
                f.write(
                    json.dumps(
                        {
                            "doc": c.doc,
                            "page": c.page,
                            "chunk_id": c.chunk_id,
                            "text": c.text,
                            "tf": dict(c.tf),
                            "length": c.length,
                            "section": c.section,
                            "heading": c.heading,
                            "table": c.table,
                            "diagram": c.diagram,
                        }
                    )
                    + "\n"
                )
    except Exception as e:
        log.warning("Index cache save failed for %s: %s", pdf_path.name, repr(e))

def index_pdf_to_chunks(pdf_path: Path) -> None:
    pdf_name = pdf_path.name
    PDF_DISPLAY_NAME[pdf_name] = get_pdf_display_name(pdf_path)

    if _load_index_from_cache(pdf_path):
        return

    doc = fitz.open(pdf_path)
    chunks: List[Chunk] = []
    df = Counter()

    last_section = ""
    last_table = ""
    last_diagram = ""
    last_heading = ""

    try:
        for i in range(doc.page_count):
            page_no = i + 1
            raw = doc.load_page(i).get_text("text") or ""
            raw = clean_text(raw)

            section, table, diagram, heading = _detect_context_lines(raw)
            if section:
                last_section = section
            if table:
                last_table = table
            if diagram:
                last_diagram = diagram
            if heading:
                last_heading = heading

            page_chunks = _split_into_chunks(raw, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
            for j, ch in enumerate(page_chunks):
                toks = tokenize(ch)
                tf = Counter(toks)
                df.update(set(tf.keys()))
                cid = _hash_id(f"{pdf_name}|p{page_no}|{j}|{ch[:100]}")
                chunks.append(
                    Chunk(
                        doc=pdf_name,
                        page=page_no,
                        chunk_id=cid,
                        text=ch,
                        tf=tf,
                        length=len(toks),
                        section=last_section,
                        heading=last_heading,
                        table=last_table,
                        diagram=last_diagram,
                    )
                )

        avgdl = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0
        CHUNK_INDEX[pdf_name] = {
            "chunks": chunks,
            "df": df,
            "avgdl": avgdl,
            "N": len(chunks),
            "pages": doc.page_count,
            "fingerprint": _pdf_fingerprint(pdf_path),
        }

        _save_index_to_cache(pdf_path, CHUNK_INDEX[pdf_name])
        log.info("Indexed %s: pages=%d chunks=%d", pdf_name, doc.page_count, len(chunks))

    finally:
        doc.close()

def ensure_chunk_indexed(pdf_name: str) -> None:
    if pdf_name in CHUNK_INDEX:
        return
    p = ensure_pdf_local(pdf_name)
    if p and p.exists():
        index_pdf_to_chunks(p)


# ============================================================
# EMBEDDINGS (OPTIONAL)
# ============================================================

def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    L = min(len(a), len(b))
    for i in range(L):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))

def _embed_cache_path(pdf_name: str) -> Path:
    p = PDF_DIR / pdf_name
    fp = _pdf_fingerprint(p) if p.exists() else _hash_id(pdf_name)
    return CACHE_DIR / f"embed_{fp}.jsonl"

def _load_embeddings(pdf_name: str) -> None:
    if pdf_name in EMBED_INDEX:
        return
    path = _embed_cache_path(pdf_name)
    if not path.exists():
        return
    try:
        vectors: Dict[str, List[float]] = {}
        dim = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj["chunk_id"]
                vec = obj["vec"]
                if isinstance(vec, list) and vec:
                    vectors[cid] = vec
                    dim = max(dim, len(vec))
        if vectors:
            EMBED_INDEX[pdf_name] = {"vectors": vectors, "dim": dim}
    except Exception as e:
        log.warning("Embedding cache load failed: %s", repr(e))

def _save_embeddings(pdf_name: str, vectors: Dict[str, List[float]]) -> None:
    try:
        path = _embed_cache_path(pdf_name)
        with path.open("w", encoding="utf-8") as f:
            for cid, vec in vectors.items():
                f.write(json.dumps({"chunk_id": cid, "vec": vec}) + "\n")
    except Exception as e:
        log.warning("Embedding cache save failed: %s", repr(e))

def _ensure_embeddings(pdf_name: str) -> None:
    ensure_vertex_ready()
    if not _VERTEX_READY:
        return
    if not EMBED_ENABLED or _EMBED_MODEL is None:
        return

    ensure_chunk_indexed(pdf_name)
    _load_embeddings(pdf_name)
    if pdf_name in EMBED_INDEX and EMBED_INDEX[pdf_name].get("vectors"):
        return

    idx = CHUNK_INDEX.get(pdf_name)
    if not idx:
        return

    chunks: List[Chunk] = idx["chunks"]
    texts = [c.text[:1800] for c in chunks]
    vectors: Dict[str, List[float]] = {}

    try:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            embs = _EMBED_MODEL.get_embeddings(batch)  # type: ignore
            for j, e in enumerate(embs):
                vec = list(getattr(e, "values", []) or [])
                if vec:
                    cid = chunks[start + j].chunk_id
                    vectors[cid] = vec
    except Exception as e:
        log.warning("Embedding generation failed: %s", repr(e))
        return

    if vectors:
        EMBED_INDEX[pdf_name] = {"vectors": vectors, "dim": len(next(iter(vectors.values())))}
        _save_embeddings(pdf_name, vectors)

def _vector_candidates(question: str, pdf_name: str, top_k: int = EMBED_TOPK) -> List[str]:
    ensure_vertex_ready()
    if not _VERTEX_READY:
        return []
    if not EMBED_ENABLED or _EMBED_MODEL is None:
        return []

    _ensure_embeddings(pdf_name)
    _load_embeddings(pdf_name)

    vstore = EMBED_INDEX.get(pdf_name, {}).get("vectors", {})
    if not vstore:
        return []

    try:
        qv = _EMBED_MODEL.get_embeddings([question])[0]  # type: ignore
        qvec = list(getattr(qv, "values", []) or [])
        if not qvec:
            return []
    except Exception:
        return []

    scored: List[Tuple[float, str]] = []
    for cid, vec in vstore.items():
        scored.append((_cosine(qvec, vec), cid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:top_k]]


# ============================================================
# RETRIEVAL (BM25)
# ============================================================

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

def page_hint_boost(chunk_page: int, page_hint: Optional[int], radius: int = 2) -> float:
    if not page_hint or page_hint <= 0:
        return 0.0
    try:
        dist = abs(int(chunk_page) - int(page_hint))
    except Exception:
        return 0.0
    if dist > radius:
        return 0.0
    return 3.0 * ((radius - dist + 1) / (radius + 1))

def search_chunks(
    question: str,
    top_k: int = TOP_K_CHUNKS,
    pinned_pdf: Optional[str] = None,
    page_hint: Optional[int] = None,
) -> List[Chunk]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    # Step 1: Broad Initial Retrieval (BM25 + Vector)
    # Fetch a larger number of candidates to give the reranker more to work with.
    initial_k = max(24, top_k * 4)
    
    search_space = [pinned_pdf] if pinned_pdf and pinned_pdf in pdfs else pdfs
    if _question_family(question) == "planning":
        search_space = [p for p in search_space if is_planning_relevant_pdf(p)]
    
    candidates: List[Tuple[float, Chunk]] = []
    for pdf_name in search_space:
        ensure_chunk_indexed(pdf_name)
        idx = CHUNK_INDEX.get(pdf_name)
        if not idx:
            continue

        vector_ids = set(_vector_candidates(question, pdf_name, top_k=initial_k))
        df, N, avgdl = idx["df"], idx["N"], idx["avgdl"]
        doc_boost = 1.2 if (pinned_pdf and pdf_name == pinned_pdf) else 0.0

        for ch in idx["chunks"]:
            s = bm25_score(qt, ch.tf, df, N, ch.length, avgdl)
            s += page_hint_boost(ch.page, page_hint)
            s += doc_boost
            if vector_ids and ch.chunk_id in vector_ids:
                s += 2.5  # Boost for hybrid retrieval match
            if s > 0:
                candidates.append((s, ch))

    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate before reranking
    seen = set()
    initial_results = []
    for _, c in candidates:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            initial_results.append(c)
            if len(initial_results) >= initial_k:
                break

    if not RERANK_ENABLED or len(initial_results) <= top_k:
        return initial_results[:top_k]

    # Step 2: LLM Reranking
    try:
        rerank_model = get_model(MODEL_CHAT, system_prompt="You are a helpful assistant.")
        
        # Prepare the text for the reranker
        rerank_context = ""
        for i, chunk in enumerate(initial_results):
            rerank_context += f"[[{i}]] Document: {chunk.doc}, Page: {chunk.page}\\n{chunk.text}\\n\\n"
            
        rerank_prompt = f"""
        Given the user's question and the following numbered list of text chunks, identify the single most relevant chunk.
        Respond with only the number of the single best chunk.

        User Question: "{question}"

        Chunks:
        {rerank_context}

        Most Relevant Chunk Number:
        """
        
        response = rerank_model.generate_content(rerank_prompt)
        best_chunk_index_str = re.sub(r'[^0-9]', '', response.text)
        
        if best_chunk_index_str:
            best_chunk_index = int(best_chunk_index_str)
            if 0 <= best_chunk_index < len(initial_results):
                best_chunk = initial_results.pop(best_chunk_index)
                # Return the best chunk first, followed by the rest sorted by original score
                final_results = [best_chunk] + initial_results
                log.info(f"Rerank successful. Promoted chunk {best_chunk.chunk_id} to top.")
                return final_results[:top_k]

    except Exception as e:
        log.warning(f"Reranker failed: {e}. Falling back to original ranking.")

    # Fallback to original ranking if reranker fails
    return initial_results[:top_k]

def dedupe_chunks_keep_order(chunks: List[Chunk]) -> List[Chunk]:
    seen = set()
    out: List[Chunk] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out

def diversify_chunks(chunks: List[Chunk], max_docs: int, per_doc: int) -> List[Chunk]:
    by_doc: Dict[str, List[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc].append(c)

    doc_order = []
    seen_docs = set()
    for c in chunks:
        if c.doc not in seen_docs:
            seen_docs.add(c.doc)
            doc_order.append(c.doc)

    out: List[Chunk] = []
    used_docs = 0
    for d in doc_order:
        if used_docs >= max_docs:
            break
        picked = by_doc[d][:per_doc]
        if picked:
            out.extend(picked)
            used_docs += 1
    return out


# ============================================================
# DOCAI SUPPORT
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
# SMALLTALK + INTENT + STRICTNESS
# ============================================================

FOLLOWUP_RE = re.compile(r"^(why|why\?|how come|can you explain( why)?|explain why)\s*$", re.I)

def is_followup_why(msg: str) -> bool:
    return bool(FOLLOWUP_RE.match((msg or "").strip()))

SMALLTALK_RE = re.compile(r"^(hi|hello|hey|yo|howdy|sup|hiya|evening|morning|afternoon)\b", re.I)
PART_PATTERN = re.compile(r"\bpart\s*[a-m]\b", re.I)

COMPLIANCE_KEYWORDS = [
    "tgd","technical guidance","building regulations","building regs",
    "part a","part b","part c","part d","part e","part f","part g","part h",
    "part j","part k","part l","part m",
    "fire","escape","travel distance","compartment","smoke","fire cert",
    "access","accessible","wheelchair","dac",
    "ber","deap","u-value","y-value","airtight","thermal bridge",
    "means of escape"
]

PLANNING_KEYWORDS = [
    "planning","planning permission","part 8","development plan","planning and development",
    "exempted development","regulations","statutory instrument","s.i.","site notice","newspaper notice"
]

BER_KEYWORDS = ["ber","deap","primary energy","renewable","u-value","y-value","airtight","thermal bridge","mpep"]

NUMERIC_TRIGGERS = [
    "minimum","maximum","min","max","limit",
    "distance","width","height",
    "u-value","y-value","rise","riser","going","pitch",
    "stairs","stair","staircase","landing","headroom",
    "travel distance",
    "lux","lumen","lumens"
]

COMPLIANCE_CONFIRM_TRIGGERS = [
    "compliant", "compliance", "does this comply", "is this compliant", "meets part", "meets tgd",
    "requirement", "required", "shall", "must"
]

def is_smalltalk(message: str) -> bool:
    m = (message or "").strip().lower()
    if not m:
        return False
    if len(m) <= 12 and (m in {"hi","hey","hello","yo","hiya","sup"}):
        return True
    if SMALLTALK_RE.match(m) and len(m) <= 32:
        if not is_compliance_question(m) and not is_planning_question(m) and not is_ber_question(m):
            return True
    if len(m) <= 12 and m in {"ok","okay","thanks","thank you","cool","nice"}:
        return True
    return False

def is_compliance_question(q: str) -> bool:
    ql = (q or "").lower()
    if PART_PATTERN.search(ql):
        return True
    return any(k in ql for k in COMPLIANCE_KEYWORDS)

def is_planning_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in PLANNING_KEYWORDS)

def is_ber_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in BER_KEYWORDS)

def is_numeric_compliance(q: str) -> bool:
    ql = (q or "").lower()
    return is_compliance_question(ql) and any(k in ql for k in NUMERIC_TRIGGERS)

def is_compliance_confirmation(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in COMPLIANCE_CONFIRM_TRIGGERS)

def _question_family(q: str) -> str:
    if is_planning_question(q):
        return "planning"
    if is_ber_question(q):
        return "ber"
    if is_compliance_question(q):
        return "building_regs"
    return "general"

SPECIFIC_PATTERNS = [
    r"\bminimum\b", r"\bmaximum\b", r"\bmin\b", r"\bmax\b",
    r"\bshall\b", r"\bmust\b", r"\brequired\b",
    r"\bsection\b", r"\bclause\b", r"\btable\b", r"\bdiagram\b",
    r"\briser\b", r"\bgoing\b", r"\bheadroom\b", r"\blanding\b",
    r"\bwidth\b", r"\bheight\b", r"\bdistance\b"
]

BROAD_PATTERNS = [
    r"\btell me about\b", r"\boverview\b", r"\bexplain\b", r"\bguidance\b",
    r"\bwhat is\b(?!.*\bminimum\b|\bmaximum\b|\bmin\b|\bmax\b)"
]

def question_precision(q: str) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "broad"

    if NUM_WITH_UNIT_RE.search(ql) or ANY_NUMBER_RE.search(ql):
        return "precise"

    if re.search(r"\b(section|clause|table|diagram)\b", ql):
        return "precise"

    if any(re.search(p, ql) for p in SPECIFIC_PATTERNS):
        return "precise"

    if any(re.search(p, ql) for p in BROAD_PATTERNS):
        return "broad"

    if len(ql.split()) <= 7:
        return "broad"

    return "mixed"

def determine_strictness(
    fam: str,
    user_msg: str,
    force_docs: bool,
    default_evidence_mode: bool,
) -> int:
    """
    0 = relaxed
    1 = cite-if-available
    2 = compliance-ish (avoid guessing requirements)
    3 = hard evidence (numbers + quotes)
    """
    ql = (user_msg or "").lower()
    numeric_needed = is_numeric_compliance(ql) and fam == "building_regs"
    confirm = is_compliance_confirmation(ql) and fam in ("building_regs", "ber", "planning")

    if force_docs:
        return 3 if numeric_needed else 2

    if default_evidence_mode:
        return 3 if numeric_needed else 2

    if numeric_needed:
        return 3

    if fam in ("building_regs", "ber") and confirm:
        return 2

    if fam in ("planning",) and confirm:
        return 2

    # Otherwise: be friendly and helpful, cite when you have it.
    return 1 if fam in ("building_regs", "planning", "ber") else 0

FASTPATH_TRIGGERS = re.compile(r"\b(timeframe|how long|process|steps|what happens|appeal period)\b", re.I)

def can_fast_path(fam: str, strictness: int, user_msg: str, pinned: Optional[str]) -> bool:
    # Only for low-risk, non-compliance confirmation
    if pinned:
        return False
    if strictness >= 2:
        return False
    if fam not in ("planning", "general"):
        return False
    q = (user_msg or "").strip()
    if len(q) > 180:
        return False
    # Avoid fast path if user asked for statutes/sections/tables/diagrams
    if re.search(r"\b(section|clause|table|diagram|statutory|regulation|s\.i\.)\b", q, re.I):
        return False
    return True


def _norm_q_for_cache(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q[:400]

def _answer_cache_key(fam: str, strictness: int, pinned: Optional[str], page_hint: Optional[int], q: str) -> str:
    base = f"{fam}|s{strictness}|pdf:{pinned or ''}|p:{page_hint or 0}|q:{_norm_q_for_cache(q)}"
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:24]

def _answer_cache_path(key: str) -> Path:
    return ANSWER_CACHE_DIR / f"{key}.json"

def _load_answer_cache(key: str) -> Optional[str]:
    try:
        p = _answer_cache_path(key)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = float(obj.get("ts", 0))
        if time.time() - ts > ANSWER_CACHE_TTL_SECONDS:
            return None
        txt = obj.get("answer")
        return txt if isinstance(txt, str) and txt.strip() else None
    except Exception:
        return None

def _save_answer_cache(key: str, answer: str) -> None:
    try:
        p = _answer_cache_path(key)
        p.write_text(json.dumps({"ts": time.time(), "answer": answer}), encoding="utf-8")
    except Exception:
        pass

# ============================================================
# WEB: SAFETY + FETCH + CACHE
# ============================================================

WEB_STRICT_ALLOWLIST = os.getenv("WEB_STRICT_ALLOWLIST", "false").lower() in ("1", "true", "yes", "on")

WEB_BLOCKLIST_DEFAULT = [
    "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "pinterest.com", "reddit.com",
]
WEB_BLOCKLIST = [d.strip().lower() for d in (os.getenv("WEB_BLOCKLIST", "") or "").split(",") if d.strip()]
if not WEB_BLOCKLIST:
    WEB_BLOCKLIST = WEB_BLOCKLIST_DEFAULT


def _safe_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _host_from_url(url: str) -> str:
    try:
        if not _safe_url(url):
            return ""
        m = re.match(r"^https?://([^/]+)", url.strip(), re.I)
        host = (m.group(1).lower() if m else "")
        host = host.split("@")[-1]
        host = host.split(":")[0]
        return host
    except Exception:
        return ""


def _blocked_url(url: str) -> bool:
    host = _host_from_url(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in WEB_BLOCKLIST)


def _allowed_url(url: str) -> bool:
    if _blocked_url(url):
        return False
    if not WEB_STRICT_ALLOWLIST:
        return True
    host = _host_from_url(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in WEB_ALLOWLIST)


# ---- Authority tiers for web sources ----
TIER_A = {
    # Official / primary sources (Ireland)
    "irishstatutebook.ie",
    "gov.ie",
    "housing.gov.ie",
    "nsai.ie",
    "dublincity.ie",
    "dlrcoco.ie",
    "corkcity.ie",
    "kildarecoco.ie",
    "galwaycity.ie",
}


def web_tier(host: str) -> str:
    h = (host or "").lower().strip()
    if not h:
        return "C"
    if any(h == d or h.endswith("." + d) for d in TIER_A):
        return "A"
    if any(h == d or h.endswith("." + d) for d in WEB_ALLOWLIST):
        return "B"
    return "C"


def filter_web_by_authority(
    fam: str,
    strictness: int,
    web_pages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for w in web_pages or []:
        host = _host_from_url(w.get("url") or "")
        tier = web_tier(host)

        # Planning + Building regs + BER: keep stronger sources
        if fam in ("planning", "building_regs", "ber"):
            if strictness >= 2:
                # strictness >= 2 => only Tier A (official)
                if tier != "A":
                    continue
            else:
                # strictness 0/1 => allow A/B, drop random blogs
                if tier not in ("A", "B"):
                    continue

        out.append(w)

    return out


def _rewrite_query_ireland(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q

    # If the user didn’t specify a location/country, bias to Ireland
    if not re.search(r"\b(ireland|irish|dublin|cork|galway|limerick)\b", q, re.I):
        q = q + " Ireland"

    return q


async def web_search_serper(query: str, k: int = TOP_K_WEB) -> List[Dict[str, str]]:
    if not WEB_ENABLED:
        return []

    q2 = _rewrite_query_ireland(query)

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": q2, "num": max(3, min(10, k * 2))}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("Serper search failed: %s", repr(e))
        return []

    out: List[Dict[str, str]] = []
    for item in (data.get("organic") or [])[: max(5, k * 2)]:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if link:
            out.append({"title": title, "url": link, "snippet": snippet})

    return out[: max(5, k * 2)]


def _html_to_text(html: str) -> str:
    html = html or ""
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)
    html = re.sub(r"(?i)</(p|div|section|article|br|li|h1|h2|h3|h4|h5|tr)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = (
        html.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"\s+\n", "\n", html)
    return html.strip()


def _best_excerpts_from_text(q: str, text: str, max_paras: int = 4, max_chars: int = 2200) -> str:
    text = clean_text(text)
    if not text:
        return ""

    blocks: List[str] = []
    buff = ""
    for sentence in re.split(r"(?<=[\.\?\!])\s+", text):
        if not sentence:
            continue
        if len(buff) + 1 + len(sentence) <= 700:
            buff = (buff + " " + sentence).strip()
        else:
            if buff:
                blocks.append(buff.strip())
            buff = sentence.strip()
        if len(blocks) >= 60:
            break
    if buff and len(blocks) < 60:
        blocks.append(buff.strip())

    qt = set(tokenize(q))
    scored: List[Tuple[int, str]] = []
    for b in blocks:
        toks = tokenize(b)
        overlap = sum(1 for t in toks if t in qt)
        if overlap > 0:
            scored.append((overlap, b))
    scored.sort(key=lambda x: x[0], reverse=True)

    picks = [b for _, b in scored[:max_paras]]
    out = "\n\n".join(picks).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "..."
    return out


def _web_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _web_cache_path(url: str) -> Path:
    return WEB_CACHE_DIR / f"{_web_cache_key(url)}.json"


def _load_web_cache(url: str) -> Optional[Dict[str, Any]]:
    try:
        p = _web_cache_path(url)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = float(obj.get("ts", 0))
        if time.time() - ts > WEB_CACHE_TTL_SECONDS:
            return None
        return obj
    except Exception:
        return None


def _save_web_cache(url: str, content_type: str, text: str) -> None:
    try:
        p = _web_cache_path(url)
        p.write_text(
            json.dumps({"ts": time.time(), "url": url, "content_type": content_type, "text": text}),
            encoding="utf-8",
        )
    except Exception:
        pass


REDIRECT_CACHE_DIR = WEB_CACHE_DIR / "redirects"
REDIRECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _redirect_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _redirect_cache_path(orig_url: str) -> Path:
    return REDIRECT_CACHE_DIR / f"{_redirect_cache_key(orig_url)}.json"


def _load_redirect_map(orig_url: str) -> Optional[str]:
    try:
        p = _redirect_cache_path(orig_url)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = float(obj.get("ts", 0))
        if time.time() - ts > WEB_CACHE_TTL_SECONDS:
            return None
        final_url = (obj.get("final_url") or "").strip()
        return final_url or None
    except Exception:
        return None


def _save_redirect_map(orig_url: str, final_url: str) -> None:
    try:
        orig_url = (orig_url or "").strip()
        final_url = (final_url or "").strip()
        if not orig_url or not final_url:
            return
        if not _safe_url(orig_url) or not _safe_url(final_url):
            return

        p = _redirect_cache_path(orig_url)
        p.write_text(
            json.dumps({"ts": time.time(), "orig_url": orig_url, "final_url": final_url}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _default_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; RaheemAI/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "en-IE,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


async def _fetch_one_web_with_final_url(
    client: httpx.AsyncClient,
    orig_url: str,
) -> Tuple[str, str, str, str, str]:
    orig_url = (orig_url or "").strip()
    if not _safe_url(orig_url):
        return ("SKIP", orig_url, "", "", "")

    headers = _default_headers()

    async def _get_with_retries(url: str) -> Optional[httpx.Response]:
        for attempt in range(WEB_RETRIES + 1):
            try:
                return await client.get(url, headers=headers)
            except Exception:
                await asyncio.sleep(WEB_RETRY_BACKOFF * (attempt + 1))
        return None

    try:
        mapped_final = (_load_redirect_map(orig_url) or "").strip()

        if mapped_final and _safe_url(mapped_final) and _allowed_url(mapped_final):
            cached = _load_web_cache(mapped_final)
            if cached and isinstance(cached.get("text"), str):
                return ("CACHED", orig_url, "text/plain", cached["text"], mapped_final)

            resp2 = await _get_with_retries(mapped_final)
            if resp2 and resp2.status_code < 400:
                ct2 = (resp2.headers.get("content-type") or "")
                ct2_l = ct2.lower()
                final2 = (str(resp2.url) if resp2.url else mapped_final).strip()

                if not _allowed_url(final2):
                    return ("SKIP", orig_url, "", "", final2)

                if ("text/html" not in ct2_l) and ("text/plain" not in ct2_l):
                    return ("SKIP", orig_url, "", "", final2)

                try:
                    content_len = int(resp2.headers.get("content-length") or "0")
                    if content_len and content_len > MAX_WEB_BYTES:
                        return ("SKIP", orig_url, "", "", final2)
                except Exception:
                    pass

                _save_redirect_map(orig_url, final2)

                text2 = resp2.text or ""
                if len(text2.encode("utf-8", errors="ignore")) > MAX_WEB_BYTES:
                    return ("SKIP", orig_url, "", "", final2)

                return ("FETCH", orig_url, ct2, text2, final2)

        resp = await _get_with_retries(orig_url)
        if not resp:
            return ("ERR", orig_url, "", "", "")

        if resp.status_code >= 400:
            return ("HTTP", orig_url, "", "", "")

        ct = (resp.headers.get("content-type") or "")
        ct_l = ct.lower()
        final_url = (str(resp.url) if resp.url else orig_url).strip()

        if _safe_url(final_url):
            _save_redirect_map(orig_url, final_url)

        if not _allowed_url(final_url):
            return ("SKIP", orig_url, "", "", final_url)

        cached2 = _load_web_cache(final_url)
        if cached2 and isinstance(cached2.get("text"), str):
            return ("CACHED", orig_url, "text/plain", cached2["text"], final_url)

        if ("text/html" not in ct_l) and ("text/plain" not in ct_l):
            return ("SKIP", orig_url, "", "", final_url)

        try:
            content_len = int(resp.headers.get("content-length") or "0")
            if content_len and content_len > MAX_WEB_BYTES:
                return ("SKIP", orig_url, "", "", final_url)
        except Exception:
            pass

        text = resp.text or ""
        if len(text.encode("utf-8", errors="ignore")) > MAX_WEB_BYTES:
            return ("SKIP", orig_url, "", "", final_url)

        return ("FETCH", orig_url, ct, text, final_url)

    except Exception:
        return ("ERR", orig_url, "", "", "")


async def web_fetch_and_excerpt(
    query: str,
    results: List[Dict[str, str]],
    max_items: int = TOP_K_WEB,
) -> List[Dict[str, str]]:
    if not WEB_ENABLED:
        return []

    cands = [r for r in results if _safe_url(r.get("url", ""))]
    cands = cands[: max(12, max_items * 4)]

    out: List[Dict[str, str]] = []
    if not cands:
        return out

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        tasks = [_fetch_one_web_with_final_url(client, r["url"]) for r in cands]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for r, resp in zip(cands, responses):
        if isinstance(resp, Exception):
            continue

        kind, _orig_url, ct, raw_text, final_url = resp
        if not final_url or not _allowed_url(final_url):
            continue
        if not raw_text:
            continue

        ct_l = (ct or "").lower()
        if ("text/html" not in ct_l) and ("text/plain" not in ct_l):
            continue

        if kind == "CACHED":
            txt = clean_text(raw_text)
        else:
            txt = _html_to_text(raw_text) if ("text/html" in ct_l) else clean_text(raw_text)

        excerpt = _best_excerpts_from_text(query, txt)
        if not excerpt:
            continue

        title = (r.get("title") or "").strip() or final_url
        out.append({"title": title, "url": final_url.strip(), "excerpt": excerpt})

        if kind != "CACHED":
            _save_web_cache(final_url, "text/plain", txt)

        if len(out) >= max_items:
            break

    return out[:max_items]


# ============================================================
# RULES LAYER
# ============================================================

_RULES: List[Dict[str, Any]] = []

def _load_rules() -> None:
    global _RULES
    if _RULES:
        return
    if not RULES_ENABLED:
        _RULES = []
        return
    try:
        if RULES_FILE.exists():
            _RULES = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        else:
            _RULES = []
    except Exception:
        _RULES = []

def _match_rules(user_msg: str) -> List[Dict[str, Any]]:
    _load_rules()
    if not _RULES:
        return []
    q = (user_msg or "").lower()
    hits: List[Tuple[int, Dict[str, Any]]] = []
    for rule in _RULES:
        kws = [str(k).lower() for k in (rule.get("keywords") or []) if str(k).strip()]
        if not kws:
            continue
        score = sum(1 for k in kws if k in q)
        if score > 0:
            hits.append((score, rule))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in hits[:3]]


# ============================================================
# CITATIONS + REFERENCES FOOTER (professional footnotes)
# ============================================================

# Accept footnotes like [1], [2], [12]
FOOTNOTE_RE = re.compile(r"\[(\d{1,2})\]")

def extract_footnotes(answer: str) -> List[int]:
    nums: List[int] = []
    for m in FOOTNOTE_RE.finditer(answer or ""):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            pass
    # de-dupe keep order
    seen = set()
    out: List[int] = []
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out

def _format_table_diagram(table: str, diagram: str) -> str:
    bits = []
    if diagram:
        bits.append(f"Diagram {diagram}")
    if table:
        bits.append(f"Table {table}")
    return ", ".join(bits)

def _format_pdf_ref(doc_title: str, page: int, table: str = "", diagram: str = "") -> str:
    td = _format_table_diagram(table, diagram)
    if td:
        return f"{doc_title} — {td}, p. {page}"
    return f"{doc_title} — p. {page}"

def build_references_footer(answer: str, evidence: Dict[str, Any]) -> str:
    used = extract_footnotes(answer)
    if not used:
        return ""

    refs = evidence.get("refs") or []
    by_n = {}
    for r in refs:
        try:
            n = int(r.get("n"))
            by_n[n] = r
        except Exception:
            continue

    lines = ["", "References"]
    for n in used:
        r = by_n.get(n)
        if not r:
            continue
        label = (r.get("label") or "").strip()
        if label:
            lines.append(f"[{n}] {label}")

    if len(lines) <= 2:
        return ""
    return "\n".join(lines).strip()

def strip_invalid_footnotes(answer: str, evidence: Dict[str, Any]) -> str:
    refs = evidence.get("refs") or []
    valid = set()
    for r in refs:
        try:
            valid.add(int(r.get("n")))
        except Exception:
            pass

    def repl(m):
        try:
            n = int(m.group(1))
            return f"[{n}]" if n in valid else ""
        except Exception:
            return ""

    return re.sub(r"\[(\d{1,2})\]", repl, answer or "")

def needs_citation_for_numbers(text: str) -> bool:
    # if there is a number with a unit OR a plain number in a “requirement-y” line
    if NUM_WITH_UNIT_RE.search(text or ""):
        return True
    if ANY_NUMBER_RE.search(text or "") and re.search(r"\b(must|shall|required|minimum|maximum)\b", text or "", re.I):
        return True
    return False


# ============================================================
# PROMPTS (more “ChatGPT paragraph” feel)
# ============================================================

CAPABILITIES_TEXT = f"""
Identity:
- You are {PRODUCT_NAME} v{PRODUCT_VERSION}.
- You were created by {CREATOR_NAME} ({CREATOR_TITLE}).

Core:
- Be helpful, clear, and natural to read.
- If PDF evidence is provided, treat it as primary for requirements/numbers.
- Web evidence (if provided) is supporting context unless it is an official source.
- If evidence is missing for a hard compliance claim, say what’s missing and ask 1 focused question.

Limits:
- Don’t claim you can predict the future or guarantee outcomes.
- Don’t invent numeric limits when strictness is high.
""".strip()

SYSTEM_PROMPT_WRITER = f"""
You are {PRODUCT_NAME}.

{CAPABILITIES_TEXT}

Writing style (important):
- Write like a good ChatGPT answer: short paragraphs, smooth flow.
- Default: 2–5 paragraphs.
- Use bullet points only when it improves clarity (comparisons, steps, checklists).
- Avoid robotic phrases like “According to the provided evidence pack…”.

Citations:
- Use professional footnotes like [1], [2] (numbers only).
- Only add a footnote when you used evidence from the Evidence JSON.
- Do NOT use D1/W1 codes.
- Keep citations light: usually 1–3 per answer unless strict compliance requires more.


When strictness is high:
- If the user asks for minimum/maximum/required values, only state numbers present in evidence.
- If you state a number, include a short quoted line that contains it (1–2 lines).
""".strip()

SYSTEM_PROMPT_PLANNER = f"""
You are {PRODUCT_NAME} (Planner).

Return ONLY JSON. No commentary.

Schema:
{{
  "intent": "string",
  "family": "general|planning|ber|building_regs",
  "precision": "broad|mixed|precise",
  "strictness": 0|1|2|3,
  "needs_docs": true|false,
  "needs_web": true|false,
  "needs_docai": true|false,
  "subqueries": ["... up to 6 ..."],
  "missing_info_question": "string or empty"
}}

Rules:
- strictness 3 if numeric compliance.
- needs_docai true if user mentions table/diagram/figure or wants visuals.
- needs_web true for planning/general/ber when it helps, but do not require it.
- missing_info_question only if absolutely required to answer.
""".strip()

def _safe_json_extract(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # try to grab first JSON object
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

async def plan_request(
    user_msg: str,
    fam: str,
    precision: str,
    strictness: int,
) -> Dict[str, Any]:
    """
    Lightweight planner. If anything fails, fall back to heuristic plan.
    """
    # Heuristic baseline (always available)
    baseline = {
        "intent": "answer_user",
        "family": fam,
        "precision": precision,
        "strictness": strictness,
        "needs_docs": fam in ("building_regs", "planning", "ber"),
        "needs_web": bool(WEB_ENABLED and fam in ("planning", "general", "ber")),
        "needs_docai": bool(wants_visual_evidence(user_msg)),
        "subqueries": [],
        "missing_info_question": "",
    }

    ensure_vertex_ready()
    if not _VERTEX_READY:
        return baseline

    try:
        model = get_model(MODEL_COMPLIANCE, SYSTEM_PROMPT_PLANNER)
        payload = {
            "user_msg": user_msg,
            "family_guess": fam,
            "precision_guess": precision,
            "strictness_guess": strictness,
            "visual_terms": wants_visual_evidence(user_msg),
        }
        resp = model.generate_content(
            [Content(role="user", parts=[Part.from_text(json.dumps(payload))])],
            generation_config=GenerationConfig(temperature=0.1, top_p=0.6, max_output_tokens=450),
            stream=False,
        )
        raw = (getattr(resp, "text", "") or "").strip()
        data = _safe_json_extract(raw) or {}
        # Merge with baseline safely
        out = dict(baseline)
        for k in out.keys():
            if k in data:
                out[k] = data[k]
        # sanitize
        out["family"] = out.get("family") if out.get("family") in ("general","planning","ber","building_regs") else fam
        out["precision"] = out.get("precision") if out.get("precision") in ("broad","mixed","precise") else precision
        try:
            out["strictness"] = int(out.get("strictness", strictness))
        except Exception:
            out["strictness"] = strictness
        out["strictness"] = max(0, min(3, int(out["strictness"])))
        out["subqueries"] = [str(x) for x in (out.get("subqueries") or []) if str(x).strip()][:6]
        out["missing_info_question"] = (out.get("missing_info_question") or "").strip()
        out["needs_docai"] = bool(out.get("needs_docai")) or wants_visual_evidence(user_msg)
        out["needs_docs"] = bool(out.get("needs_docs"))
        out["needs_web"] = bool(out.get("needs_web")) and WEB_ENABLED
        return out
    except Exception as e:
        log.warning("Planner failed (fallback to heuristic): %s", repr(e))
        return baseline


# ============================================================
# EVIDENCE PACK (structured, compact)
# ============================================================


def _tight_excerpt(text: str, max_chars: int = 700) -> str:
    t = clean_text(text or "")
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"

def _tighten_chunk_text_for_evidence(c: Chunk, query: str, max_chars: int = 950) -> str:
    text = clean_text(c.text)
    if not text:
        return ""

    qt = set(tokenize(query))
    meta = []
    if c.heading:
        meta.append(f"Heading: {c.heading}")
    if c.section:
        meta.append(f"Section: {c.section}")
    if c.table:
        meta.append(f"Table: {c.table}")
    if c.diagram:
        meta.append(f"Diagram: {c.diagram}")
    meta_line = (" | ".join(meta)).strip()

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sents = re.split(r"(?<=[\.\?\!])\s+", text)

    def score_piece(p: str) -> int:
        toks = tokenize(p)
        overlap = sum(1 for t in toks if t in qt)
        if NUM_WITH_UNIT_RE.search(p):
            overlap += 6
        if re.search(r"\b(must|shall|required|requirement)\b", p, re.I):
            overlap += 2
        return overlap

    scored: List[Tuple[int, str]] = []
    for p in (lines[:140] + sents):
        p = p.strip()
        if not p or len(p) < 12:
            continue
        sc = score_piece(p)
        if sc > 0:
            scored.append((sc, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks: List[str] = [p for _, p in scored[:6]]

    if not picks and NUM_WITH_UNIT_RE.search(text):
        picks = [text[:max_chars]]

    out = "\n".join(picks).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "..."

    if meta_line:
        return meta_line + "\n" + out
    return out

def build_evidence_packets(
    chunks: List[Chunk],
    web_pages: List[Dict[str, str]],
    docai_hits: List[Tuple[str, str]],
    user_query: str,
    precision: str,
) -> Dict[str, Any]:
    max_chars = 520 if precision == "broad" else 780

    refs: List[Dict[str, Any]] = []
    ref_n = 1

    pdf_items = []
    for c in chunks[: max(12, RERANK_TOPK)]:
        doc_title = PDF_DISPLAY_NAME.get(c.doc, c.doc)
        label = _format_pdf_ref(doc_title, c.page, table=(c.table or ""), diagram=(c.diagram or ""))
        refs.append({"n": ref_n, "kind": "pdf", "label": label})

        pdf_items.append({
            "ref_n": ref_n,
            "doc": doc_title,
            "page": c.page,
            "section": c.section or "",
            "heading": c.heading or "",
            "table": c.table or "",
            "diagram": c.diagram or "",
            "excerpt": _tight_excerpt(
                _tighten_chunk_text_for_evidence(c, user_query, max_chars=max_chars) or c.text,
                max_chars=max_chars
            ),
        })
        ref_n += 1

    web_items = []
    for w in web_pages[:TOP_K_WEB]:
        host = _host_from_url(w.get("url") or "")
        title = (w.get("title") or host or "Web source").strip()
        label = f"{title}" + (f" ({host})" if host else "")
        refs.append({"n": ref_n, "kind": "web", "label": label})

        web_items.append({
            "ref_n": ref_n,
            "title": title,
            "host": host,
            "excerpt": _tight_excerpt(w.get("excerpt") or "", max_chars=520),
        })
        ref_n += 1

    docai_items = []
    for (label, txt) in docai_hits[:3]:
        refs.append({"n": ref_n, "kind": "docai", "label": f"Document AI extract — {label}"})
        docai_items.append({
            "ref_n": ref_n,
            "label": label,
            "excerpt": _tight_excerpt(txt, max_chars=700),
        })
        ref_n += 1

    return {
        "refs": refs,
        "pdf": pdf_items,
        "web": web_items,
        "docai": docai_items,
    }

# ============================================================
# RERANK (LLM)
# ============================================================

RERANK_PROMPT = """
You are a strict reranker for compliance evidence.

You will be given:
- USER_QUESTION
- CANDIDATES: a JSON array of objects with fields {id, doc, page, section, heading, table, text}

Return ONLY valid JSON:
{
  "ranked_ids": ["id1","id2",...]
}

Rules:
- Put the most directly relevant candidates first.
- Prefer candidates that contain numeric limits or clear requirement statements relevant to the question.
- Prefer candidates that mention a matching Part/TGD topic.
- Keep output to at most 12 ids.
""".strip()

async def _rerank_candidates(user_msg: str, cands: List[Chunk], max_keep: int = RERANK_TOPK) -> List[Chunk]:
    if not RERANK_ENABLED or not cands:
        return cands[:max_keep]

    ensure_vertex_ready()
    if not _VERTEX_READY:
        return cands[:max_keep]

    small = cands[: min(len(cands), 18)]
    payload = []
    for c in small:
        payload.append({
            "id": c.chunk_id,
            "doc": c.doc,
            "page": c.page,
            "section": c.section,
            "heading": c.heading,
            "table": c.table,
            "text": (c.text[:900] if c.text else "")
        })

    model = get_model(MODEL_COMPLIANCE, RERANK_PROMPT)
    inp = "USER_QUESTION:\n" + user_msg.strip() + "\n\nCANDIDATES:\n" + json.dumps(payload)
    try:
        resp = model.generate_content(
            [Content(role="user", parts=[Part.from_text(inp)])],
            generation_config=GenerationConfig(temperature=0.1, top_p=0.6, max_output_tokens=300),
            stream=False
        )
        raw = (getattr(resp, "text", "") or "").strip()
        data = _safe_json_extract(raw) or {}
        ranked_ids = [x for x in (data.get("ranked_ids") or []) if isinstance(x, str)]
        if not ranked_ids:
            return small[:max_keep]
        by_id = {c.chunk_id: c for c in small}
        ranked = [by_id[i] for i in ranked_ids if i in by_id]
        seen = set(r.chunk_id for r in ranked)
        for c in small:
            if c.chunk_id not in seen:
                ranked.append(c)
            if len(ranked) >= max_keep:
                break
        return ranked[:max_keep]
    except Exception as e:
        log.warning("Rerank failed: %s", repr(e))
        return small[:max_keep]


# ============================================================
# HARD NUMERIC VERIFIER
# ============================================================

def _sources_text_blob_for_verification(
    pdf_chunks: List[Chunk],
    docai_hits: List[Tuple[str, str]],
    web_pages: List[Dict[str, str]]
) -> str:
    parts: List[str] = []
    for c in pdf_chunks or []:
        parts.append(clean_text(c.text))
    for _, t in docai_hits or []:
        parts.append(clean_text(t))
    for w in web_pages or []:
        parts.append(clean_text(w.get("excerpt", "")))
    return "\n\n".join([p for p in parts if p]).lower()

def _contains_exact_numeric_from_answer(answer: str) -> List[str]:
    hits = []
    for m in NUM_WITH_UNIT_RE.finditer(answer or ""):
        hits.append(m.group(0).strip())
    seen = set()
    out = []
    for h in hits:
        k = h.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out

def _hard_verify_numeric(answer: str, sources_blob_lower: str) -> Tuple[bool, str]:
    nums = _contains_exact_numeric_from_answer(answer or "")
    if not nums:
        return True, answer

    if not VERIFY_NUMERIC:
        return True, answer

    missing = [n for n in nums if n.lower() not in (sources_blob_lower or "")]
    if not missing:
        return True, answer

    safe = (
        "Quick note: I can’t confirm those exact numeric values from the evidence loaded for this answer.\n"
        "If you pin/upload the right document (or tell me which edition), I’ll quote the exact line for each number.\n\n"
        "Numbers not found in the current evidence: " + ", ".join(missing)
    )
    return False, safe


# ============================================================
# PDF UPLOAD
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
async def upload_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    unauthorized = require_admin_key(x_api_key)
    if unauthorized:
        return unauthorized

    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not filename.endswith(".pdf") and content_type != "application/pdf":
        return JSONResponse({"ok": False, "error": "Only PDF files are allowed."}, status_code=400)

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"ok": False, "error": f"File too large. Max {MAX_UPLOAD_MB}MB."}, status_code=400)

    safe_name = Path(file.filename or "upload.pdf").name
    dest = PDF_DIR / safe_name

    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 2
        while True:
            cand = PDF_DIR / f"{stem}_{i}{suffix}"
            if not cand.exists():
                dest = cand
                break
            i += 1

    dest.write_bytes(raw)

    if R2_ENABLED:
        try:
            c = r2_client()
            c.put_object(
                Bucket=R2_BUCKET,
                Key=f"{R2_PREFIX}{dest.name}",
                Body=raw,
                ContentType="application/pdf",
            )
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"R2 upload failed: {repr(e)}"}, status_code=500)

    # Clear indexes for this file
    CHUNK_INDEX.pop(dest.name, None)
    EMBED_INDEX.pop(dest.name, None)

    try:
        index_pdf_to_chunks(dest)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Index failed: {repr(e)}"}, status_code=500)

    try:
        if EMBED_ENABLED:
            _ensure_embeddings(dest.name)
    except Exception:
        pass

    # DocAI parse (optional)
    docai_ok = False
    docai_chunks_saved = 0
    docai_error = None
    try:
        if _DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text:
            if (os.getenv("DOCAI_PROCESSOR_ID") or "").strip() and (os.getenv("DOCAI_LOCATION") or "").strip():
                combined_text, _ = docai_extract_pdf_to_text(str(dest), chunk_pages=15)
                chunks = _split_docai_combined_to_chunks(combined_text)
                stem = dest.stem
                for (a, b), txt in chunks:
                    (DOCAI_DIR / f"{stem}_p{a}-{b}.txt").write_text(txt, encoding="utf-8", errors="ignore")
                    docai_chunks_saved += 1
                docai_ok = docai_chunks_saved > 0
    except Exception as e:
        docai_error = repr(e)

    return {
        "ok": True,
        "stored_as": dest.name,
        "pdf_count": len(list_pdfs()),
        "indexed_chunks": CHUNK_INDEX.get(dest.name, {}).get("N", 0),
        "embeddings_ready": bool(EMBED_INDEX.get(dest.name, {}).get("vectors")) if dest.name in EMBED_INDEX else False,
        "web_enabled": WEB_ENABLED,
        "docai": {
            "attempted": bool(_DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text),
            "ok": docai_ok,
            "chunks_saved": docai_chunks_saved,
            "error": docai_error,
        },
    }


# ============================================================
# ABOUT / CAPABILITIES / ROADMAP
# ============================================================

def _load_roadmap() -> Dict[str, Any]:
    if not ROADMAP_FILE.exists():
        return {"ok": False, "items": [], "note": "No roadmap file found."}
    try:
        data = json.loads(ROADMAP_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {"ok": True, **data}
        if isinstance(data, list):
            return {"ok": True, "items": data}
        return {"ok": False, "items": [], "note": "Invalid roadmap format."}
    except Exception as e:
        return {"ok": False, "items": [], "note": repr(e)}

@app.get("/about")
def about():
    return {
        "ok": True,
        "product": {"name": PRODUCT_NAME, "version": PRODUCT_VERSION},
        "creator": {"name": CREATOR_NAME, "title": CREATOR_TITLE},
        "time_utc": datetime.utcnow().isoformat(),
        "roadmap": _load_roadmap(),
    }

@app.get("/capabilities")
def capabilities():
    return {
        "ok": True,
        "identity": {
            "product": PRODUCT_NAME,
            "version": PRODUCT_VERSION,
            "creator": CREATOR_NAME,
            "creator_title": CREATOR_TITLE,
        },
        "features": [
            "Chat with optional server-side chat memory (chat_id)",
            "PDF upload + indexing (BM25); optional Vertex embeddings",
            "Adaptive strictness (chatty -> evidence mode)",
            "Optional web search via Serper with allow/block lists + caching",
            "Optional DocAI extraction support",
            "Real streaming output (paragraph chunks)",
        ],
        "limits": [
            "Cannot truly predict the future or guarantee outcomes",
            "In strictness 3: numeric limits must exist in sources",
        ],
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat_endpoint(
    body: ChatBody,
    chat_id: str = Query(""),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    unauthorized = require_admin_key(x_api_key)
    if unauthorized:
        return unauthorized

    return StreamingResponse(
        _stream_answer_async(chat_id, body.message, force_docs, pdf, page_hint, messages=body.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}


# ============================================================
# HEALTH + ROOT
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "app": PRODUCT_NAME, "version": PRODUCT_VERSION, "time": datetime.utcnow().isoformat()}

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
        "web_enabled": WEB_ENABLED,
        "web_allowlist": WEB_ALLOWLIST,
        "embed_enabled": bool(EMBED_ENABLED and _EMBED_MODEL is not None),
        "rerank_enabled": bool(RERANK_ENABLED),
        "verify_numeric": bool(VERIFY_NUMERIC),
        "rules_enabled": bool(RULES_ENABLED and RULES_FILE.exists()),
        "web_cache_ttl_seconds": WEB_CACHE_TTL_SECONDS,
        "r2": {
            "enabled": R2_ENABLED,
            "bucket": R2_BUCKET,
            "endpoint": R2_ENDPOINT,
            "access_key_set": bool(R2_ACCESS_KEY_ID),
            "secret_set": bool(R2_SECRET_ACCESS_KEY),
        },
    }


# ============================================================
# STREAMING CORE
# ============================================================

def build_writer_contents(
    history: List[Dict[str, str]],
    user_message: str,
    evidence: Dict[str, Any],
    plan: Dict[str, Any],
) -> List[Content]:
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

    # Controller is short on purpose (so it reads natural, not “RAG dump”)
    controller = {
        "precision": plan.get("precision"),
        "strictness": plan.get("strictness"),
        "notes": [
            "Write in short paragraphs.",
            "Use bullets only if helpful.",
            "Cite only when using evidence.",
            "If strictness >= 3 and you state a number, include a short quote line containing it.",
        ],
    }

    final_user = (
        (user_message or "").strip()
        + "\n\n"
        + "Context for you (JSON):\n"
        + json.dumps({"plan": controller, "evidence": evidence}, ensure_ascii=False)
    )
    contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    return contents

async def _stream_answer_async(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None,
):
    try:
        user_msg = (message or "").strip()
        if not user_msg:
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()
        pinned = (pdf or "").strip() or None

        # -----------------------------
        # 0) Smalltalk fast exit
        # -----------------------------
        if is_smalltalk(user_msg):
            friendly = f"Hey — I’m {PRODUCT_NAME}. What can I help you with?"
            if chat_id:
                remember(chat_id, "user", user_msg)
                remember(chat_id, "assistant", friendly)
            yield sse_send(friendly)
            yield "event: done\ndata: ok\n\n"
            return

        # -----------------------------
        # 1) Canonical history
        # -----------------------------
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

        fam = _question_family(user_msg)
        precision = question_precision(user_msg)
        strictness = determine_strictness(
            fam=fam,
            user_msg=user_msg,
            force_docs=force_docs,
            default_evidence_mode=DEFAULT_EVIDENCE_MODE,
        )

        # -----------------------------
        # 2) Answer cache fast return
        # -----------------------------
        cache_key = _answer_cache_key(fam, strictness, pinned, page_hint, user_msg)
        cached = _load_answer_cache(cache_key)
        if cached:
            if chat_id:
                remember(chat_id, "assistant", cached)
            yield sse_send(cached)
            yield "event: done\ndata: ok\n\n"
            return

        # -----------------------------
        # 3) Rules layer fast return
        # -----------------------------
        rules_hit = _match_rules(user_msg) if RULES_ENABLED else []
        if rules_hit:
            rule = rules_hit[0]
            ans = (rule.get("answer") or "").strip()
            quote = (rule.get("quote") or "").strip()
            cit = rule.get("citation") or {}

            rendered = ans
            if quote:
                rendered += "\n\n" + quote

            # Convert rule citations into clean human refs (no D1/W1 codes)
            if cit:
                bits: List[str] = []
                if cit.get("doc"):
                    bits.append(str(cit.get("doc")))
                if cit.get("section"):
                    bits.append(f"Section {cit.get('section')}")
                if cit.get("table"):
                    bits.append(f"Table {cit.get('table')}")
                if cit.get("diagram"):
                    bits.append(f"Diagram {cit.get('diagram')}")
                if bits:
                    rendered += "\n\nReferences\n[1] " + " — ".join(bits)

            rendered = enforce_chatgpt_paragraphs(normalize_model_output(rendered))

            if chat_id:
                remember(chat_id, "assistant", rendered)

            # Cache it too (rules answers benefit massively from caching)
            _save_answer_cache(cache_key, rendered)

            yield sse_send(rendered)
            yield "event: done\ndata: ok\n\n"
            return

        # -----------------------------
        # 4) Planner step
        # -----------------------------
        plan = await plan_request(user_msg, fam=fam, precision=precision, strictness=strictness)

        try:
            plan_strictness = int(plan.get("strictness", strictness))
        except Exception:
            plan_strictness = strictness
        plan["strictness"] = max(0, min(3, plan_strictness))

        # Keep it snappy for simple questions
        if can_fast_path(fam, int(plan.get("strictness", strictness)), user_msg, pinned):
            plan["needs_docs"] = False
            plan["needs_docai"] = False
            plan["needs_web"] = bool(WEB_ENABLED and fam in ("planning", "general", "ber"))

        # -----------------------------
        # 5) Retrieval (PDF + DocAI + Web)
        # -----------------------------
        async def get_pdf_chunks() -> List[Chunk]:
            if not list_pdfs():
                return []

            queries = [user_msg] + [
                q for q in (plan.get("subqueries") or [])
                if isinstance(q, str) and q.strip()
            ]

            collected: List[Chunk] = []
            for q in queries[:6]:
                cands = search_chunks(
                    q,
                    top_k=max(TOP_K_CHUNKS, 10) if plan.get("precision") == "broad" else TOP_K_CHUNKS,
                    pinned_pdf=pinned,
                    page_hint=page_hint,
                )
                collected.extend(cands)

            collected = dedupe_chunks_keep_order(collected)

            if plan.get("precision") == "broad" and collected:
                collected = diversify_chunks(collected, max_docs=BROAD_DOC_DIVERSITY_K, per_doc=3)

            if RERANK_ENABLED and collected:
                keep = max(RERANK_TOPK, 10) if plan.get("precision") == "broad" else RERANK_TOPK
                collected = await _rerank_candidates(user_msg, collected, max_keep=keep)

            keep2 = max(RERANK_TOPK, 10) if plan.get("precision") == "broad" else RERANK_TOPK
            return collected[:keep2]

        async def get_docai_hits() -> List[Tuple[str, str]]:
            if not pinned:
                return []
            if not _docai_chunk_files_for(pinned):
                return []
            if not bool(plan.get("needs_docai")):
                return []
            k = 3 if plan.get("precision") == "broad" else 2
            return docai_search_text(pinned, user_msg, k=k)

        async def get_web_evidence() -> List[Dict[str, str]]:
            if not WEB_ENABLED:
                return []
            if not bool(plan.get("needs_web")):
                return []

            serp = await web_search_serper(user_msg, k=TOP_K_WEB)

            # ✅ IMPORTANT: filter by authority tier BEFORE fetching pages
            serp = filter_web_by_authority(fam, int(plan.get("strictness", strictness)), serp)

            pages = await web_fetch_and_excerpt(user_msg, serp, max_items=TOP_K_WEB)
            pages = filter_web_by_authority(fam, int(plan.get("strictness", strictness)), pages)
            return pages

        pdf_chunks, docai_hits, web_pages = await asyncio.gather(
            get_pdf_chunks() if plan.get("needs_docs") else asyncio.sleep(0, result=[]),
            get_docai_hits(),
            get_web_evidence(),
        )

        # Hard guard for strict numeric compliance
        numeric_needed = is_numeric_compliance(user_msg) and fam == "building_regs"
        if numeric_needed and int(plan["strictness"]) >= 3 and not pdf_chunks and not docai_hits:
            msg = (
                "I can’t confirm the exact numeric requirement yet because I don’t have the relevant TGD evidence loaded.\n"
                "Upload/pin the correct TGD PDF and I’ll quote the exact line containing the number."
            )
            msg = enforce_chatgpt_paragraphs(normalize_model_output(msg))
            if chat_id:
                remember(chat_id, "assistant", msg)
            yield sse_send(msg)
            yield "event: done\ndata: ok\n\n"
            return

        # -----------------------------
        # 6) Evidence pack
        # -----------------------------
        evidence_pack = build_evidence_packets(
            pdf_chunks,
            web_pages,
            docai_hits,
            user_query=user_msg,
            precision=plan.get("precision") or precision,
        )

        # -----------------------------
        # 7) Model streaming (writer)
        # -----------------------------
        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\r", " ").replace("\n", " ")
            yield f"data: [ERROR] {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        model_name = MODEL_COMPLIANCE if int(plan["strictness"]) >= 2 else MODEL_CHAT
        model = get_model(model_name=model_name, system_prompt=SYSTEM_PROMPT_WRITER)
        contents = build_writer_contents(history_for_prompt, user_msg, evidence_pack, plan)

        stream = model.generate_content(
            contents,
            generation_config=get_generation_config(int(plan["strictness"])),
            stream=True,
        )

        full_text, frames = await stream_llm_text_as_sse(stream)
        for fr in frames:
            if fr:
                yield fr

        # -----------------------------
        # 8) Postprocess: paragraphs + references + numeric verify note
        # -----------------------------
        draft = enforce_chatgpt_paragraphs(normalize_model_output(full_text))

        # Strip invalid footnotes (if the model hallucinated [99], etc.)
        draft = strip_invalid_footnotes(draft, evidence_pack)

        footer = build_references_footer(draft, evidence_pack)
        if footer:
            yield sse_send("\n\n" + footer)

        sources_blob = _sources_text_blob_for_verification(pdf_chunks, docai_hits, web_pages)
        ok, note = _hard_verify_numeric(draft, sources_blob)

        numeric_note = ""
        if not ok:
            numeric_note = "\n\n---\n\n" + enforce_chatgpt_paragraphs(normalize_model_output(note))
            yield sse_send(numeric_note)

        # -----------------------------
        # 9) ✅ Save to cache ONCE (single exact location)
        # -----------------------------
        final_for_cache = (draft + ("\n\n" + footer if footer else "") + numeric_note).strip()
        _save_answer_cache(cache_key, final_for_cache)

        if chat_id:
            remember(chat_id, "assistant", final_for_cache)

        yield "event: done\ndata: ok\n\n"
        return

    except Exception as e:
        msg = str(e).replace("\r", " ").replace("\n", " ")
        log.exception("Chat stream error: %s", msg)
        yield f"data: [ERROR] {msg}\n\n"
        yield "event: done\ndata: ok\n\n"
        return




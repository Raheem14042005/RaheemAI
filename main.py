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

# ---- Product identity (DO NOT volunteer to customers unless asked) ----
CREATOR_NAME = os.getenv("CREATOR_NAME", "Abdulraheem Ahmed").strip()
CREATOR_TITLE = os.getenv(
    "CREATOR_TITLE",
    "4th year Architectural Technology student at Technological University Dublin (TUD)",
).strip()
PRODUCT_NAME = os.getenv("PRODUCT_NAME", "Raheem AI").strip()
PRODUCT_VERSION = os.getenv("PRODUCT_VERSION", "1.0.0").strip()

# Optional roadmap file
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
MODEL_ROUTER = (os.getenv("GEMINI_MODEL_ROUTER", "gemini-2.0-flash-001") or "").strip()

# Web search (Serper)
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()
WEB_ENABLED = (os.getenv("WEB_ENABLED", "true").lower() in ("1", "true", "yes", "on")) and bool(SERPER_API_KEY)

# Evidence defaults
DEFAULT_EVIDENCE_MODE = os.getenv("DEFAULT_EVIDENCE_MODE", "false").lower() in ("1", "true", "yes", "on")

# Retrieval sizing
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "7"))
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "5"))

# Broad / decomposition
BROAD_MAX_SUBQUERIES = int(os.getenv("BROAD_MAX_SUBQUERIES", "7"))
BROAD_DOC_DIVERSITY_K = int(os.getenv("BROAD_DOC_DIVERSITY_K", "6"))  # how many docs to cover
BROAD_PER_DOC = int(os.getenv("BROAD_PER_DOC", "2"))                  # per doc chunks
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.62"))                   # diversity balance

# Vector embeddings (optional)
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "true").lower() in ("1", "true", "yes", "on")
EMBED_MODEL_NAME = (os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004") or "").strip()
EMBED_TOPK = int(os.getenv("EMBED_TOPK", "28"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rerank (LLM reranker)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
RERANK_TOPK = int(os.getenv("RERANK_TOPK", str(TOP_K_CHUNKS)))

# Web allowlist/blocklist
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
WEB_ALLOWLIST = [d.strip().lower() for d in (os.getenv("WEB_ALLOWLIST", "") or "").split(",") if d.strip()] or WEB_ALLOWLIST_DEFAULT
WEB_STRICT_ALLOWLIST = os.getenv("WEB_STRICT_ALLOWLIST", "false").lower() in ("1","true","yes","on")

WEB_BLOCKLIST_DEFAULT = [
    "facebook.com","instagram.com","tiktok.com","x.com","twitter.com",
    "pinterest.com","reddit.com",
]
WEB_BLOCKLIST = [d.strip().lower() for d in (os.getenv("WEB_BLOCKLIST", "") or "").split(",") if d.strip()] or WEB_BLOCKLIST_DEFAULT

# Numeric verification
VERIFY_NUMERIC = os.getenv("VERIFY_NUMERIC", "true").lower() in ("1", "true", "yes", "on")

# Rules layer
RULES_FILE = Path(os.getenv("RULES_FILE", str(BASE_DIR / "rules.json")))
RULES_ENABLED = os.getenv("RULES_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Timeouts / cache
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "18"))
MAX_WEB_BYTES = int(os.getenv("MAX_WEB_BYTES", str(2_500_000)))
WEB_RETRIES = int(os.getenv("WEB_RETRIES", "2"))
WEB_RETRY_BACKOFF = float(os.getenv("WEB_RETRY_BACKOFF", "0.6"))
WEB_CACHE_TTL_SECONDS = int(os.getenv("WEB_CACHE_TTL_SECONDS", str(12 * 60 * 60)))

# Data dirs
if os.getenv("RENDER"):
    DATA_DIR = Path("/tmp/raheemai")
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR))).resolve()

PDF_DIR = DATA_DIR / "pdfs"
DOCAI_DIR = DATA_DIR / "parsed_docai"
CACHE_DIR = DATA_DIR / "cache"
WEB_CACHE_DIR = CACHE_DIR / "web"
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
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] if raw_origins else [
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

def get_generation_config(is_evidence: bool) -> GenerationConfig:
    # More confident outputs: slightly lower temperature in normal mode too,
    # but keep it conversational.
    if is_evidence:
        return GenerationConfig(temperature=0.2, top_p=0.8, max_output_tokens=3500)
    return GenerationConfig(temperature=0.55, top_p=0.9, max_output_tokens=3500)

# ============================================================
# CHAT MEMORY (server-side, per chat_id)
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
# TEXT + NORMALIZATION (fix the ".This" issue + nicer output)
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

BULLET_STAR_RE = re.compile(r"(?m)^\s*\*\s+")
EXCESS_SPACES_RE = re.compile(r"[ \t]{2,}")
MULTI_BLANKLINES_RE = re.compile(r"\n{3,}")
# Fix ".This", "?This", "!This", ":This", ";This"
PUNCT_GLUE_RE = re.compile(r"([\.!\?:;])([A-Za-z])")
# Fix missing space after comma
COMMA_GLUE_RE = re.compile(r"(,)([A-Za-z])")

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_model_output(text: str) -> str:
    t = (text or "").replace("\r", "")
    t = BULLET_STAR_RE.sub("- ", t)
    t = EXCESS_SPACES_RE.sub(" ", t)
    t = MULTI_BLANKLINES_RE.sub("\n\n", t)
    t = PUNCT_GLUE_RE.sub(r"\1 \2", t)
    t = COMMA_GLUE_RE.sub(r"\1 \2", t)
    # Clean weird double-spaces introduced
    t = re.sub(r"[ \t]{2,}", " ", t)
    # Remove space before punctuation
    t = re.sub(r"\s+([,\.!\?:;])", r"\1", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-/\.%]*", t)
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

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

def wants_visual_evidence(q: str) -> bool:
    return bool(VISUAL_TERMS_RE.search(q or ""))

# ============================================================
# PDF UTILITIES (R2 + local)
# ============================================================

def _pdf_fingerprint(pdf_path: Path) -> str:
    st = pdf_path.stat()
    s = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def get_pdf_display_name(pdf_path: Path) -> str:
    try:
        d = fitz.open(pdf_path)
        meta = d.metadata or {}
        title = (meta.get("title") or "").strip()
        d.close()
        if title and len(title) >= 5:
            return title
    except Exception:
        pass

    stem = pdf_path.stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    stem = re.sub(r"\s{2,}", " ", stem)
    return stem or pdf_path.name

def list_pdfs() -> List[str]:
    # List from R2 (preferred) + local fallback
    out: List[str] = []

    if R2_ENABLED:
        try:
            c = r2_client()
            token = None
            prefix = (R2_PREFIX or "").strip()
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            while True:
                kwargs: Dict[str, Any] = {"Bucket": R2_BUCKET, "MaxKeys": 1000}
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

            out = sorted(set(out))
            if out:
                return out
        except Exception as e:
            log.warning("R2 list_pdfs failed, falling back to local: %s", repr(e))

    # local fallback
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p.name)
    return sorted(set(out))

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
# CHUNKING / INDEXING (robust + cached)
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

    for ln in lines[:240]:
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
    for ln in lines[:140]:
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
                f.write(json.dumps({
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
                }) + "\n")
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
                cid = _hash_id(f"{pdf_name}|p{page_no}|{j}|{ch[:120]}")
                chunks.append(Chunk(
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
                ))

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
# RETRIEVAL (BM25 + Hybrid + MMR for broad questions)
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
    return 2.4 * ((radius - dist + 1) / (radius + 1))

def search_chunks_single_query(
    question: str,
    top_k: int,
    pinned_pdf: Optional[str],
    page_hint: Optional[int],
) -> List[Tuple[float, Chunk]]:
    pdfs = list_pdfs()
    if not pdfs:
        return []
    qt = tokenize(question)
    if not qt:
        return []
    search_space = [pinned_pdf] if pinned_pdf and pinned_pdf in pdfs else pdfs

    candidates: List[Tuple[float, Chunk]] = []
    for pdf_name in search_space:
        ensure_chunk_indexed(pdf_name)
        idx = CHUNK_INDEX.get(pdf_name)
        if not idx:
            continue

        vector_ids: List[str] = []
        if EMBED_ENABLED and _VERTEX_EMBEDDINGS_AVAILABLE:
            vector_ids = _vector_candidates(question, pdf_name, top_k=EMBED_TOPK)
        vector_set = set(vector_ids)

        df = idx["df"]
        N = idx["N"]
        avgdl = idx["avgdl"]

        doc_boost = 1.15 if (pinned_pdf and pdf_name == pinned_pdf) else 1.0

        scored: List[Tuple[float, Chunk]] = []
        for ch in idx["chunks"]:
            s = bm25_score(qt, ch.tf, df, N, ch.length, avgdl)
            s += page_hint_boost(ch.page, page_hint)
            if vector_set and ch.chunk_id in vector_set:
                s += 2.2
            s *= doc_boost
            if s > 0:
                scored.append((s, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates.extend(scored[: max(24, top_k * 6)])

    candidates.sort(key=lambda x: x[0], reverse=True)
    # global dedupe
    seen = set()
    out: List[Tuple[float, Chunk]] = []
    for s, c in candidates:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append((s, c))
        if len(out) >= max(48, top_k * 8):
            break
    return out

def mmr_select(
    scored: List[Tuple[float, Chunk]],
    k: int,
    lam: float = MMR_LAMBDA,
) -> List[Chunk]:
    """
    Very lightweight MMR: diversity by doc + by content token overlap.
    """
    if not scored:
        return []
    selected: List[Tuple[float, Chunk]] = []
    cand = scored[:]

    def sim(a: Chunk, b: Chunk) -> float:
        ta = set(tokenize(a.text)[:120])
        tb = set(tokenize(b.text)[:120])
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / max(1, union)

    while cand and len(selected) < k:
        best = None
        best_val = -1e9
        for s, c in cand:
            max_sim = 0.0
            for _, sc in selected:
                max_sim = max(max_sim, sim(c, sc))
                # extra doc penalty to force cross-PDF coverage for broad
                if c.doc == sc.doc:
                    max_sim = max(max_sim, 0.65)
            val = lam * s - (1 - lam) * max_sim
            if val > best_val:
                best_val = val
                best = (s, c)
        if not best:
            break
        selected.append(best)
        # remove best
        cand = [(s, c) for (s, c) in cand if c.chunk_id != best[1].chunk_id]
    return [c for _, c in selected]

def dedupe_chunks_keep_order(chunks: List[Chunk]) -> List[Chunk]:
    seen = set()
    out: List[Chunk] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out

# ============================================================
# DOCAI SUPPORT (optional)
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
# INTENT / ROUTING (the “behave like ChatGPT” bit)
# ============================================================

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

def is_smalltalk(message: str) -> bool:
    m = (message or "").strip().lower()
    if not m:
        return False
    if len(m) <= 14 and m in {"hi","hey","hello","yo","hiya","sup"}:
        return True
    if SMALLTALK_RE.match(m) and len(m) <= 40:
        return True
    if len(m) <= 14 and m in {"ok","okay","thanks","thank you","cool","nice"}:
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

def user_asked_for_sources(q: str) -> bool:
    ql = (q or "").lower()
    return any(x in ql for x in ["source", "sources", "cite", "citation", "where did you get", "reference"])

# ============================================================
# WEB (safe routing + cache) — only used when it adds value
# ============================================================

def _safe_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")

def _host_from_url(url: str) -> str:
    try:
        if not _safe_url(url):
            return ""
        m = re.match(r"^https?://([^/]+)", url.strip(), re.I)
        host = (m.group(1).lower() if m else "")
        host = host.split("@")[-1].split(":")[0]
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

async def web_search_serper(query: str, k: int = TOP_K_WEB) -> List[Dict[str, str]]:
    if not WEB_ENABLED:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": max(3, min(10, k * 2))}
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
        p.write_text(json.dumps({"ts": time.time(), "url": url, "content_type": content_type, "text": text}), encoding="utf-8")
    except Exception:
        pass

async def _fetch_one_web(client: httpx.AsyncClient, url: str) -> Tuple[str, str, str]:
    url = (url or "").strip()
    if not _safe_url(url) or not _allowed_url(url):
        return ("SKIP", "", "")
    cached = _load_web_cache(url)
    if cached and isinstance(cached.get("text"), str):
        return ("CACHED", "text/plain", cached["text"])

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RaheemAI/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "en-IE,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    for attempt in range(WEB_RETRIES + 1):
        try:
            r = await client.get(url, headers=headers, follow_redirects=True)
            if r.status_code >= 400:
                return ("HTTP", "", "")
            ct = (r.headers.get("content-type") or "")
            ct_l = ct.lower()
            if ("text/html" not in ct_l) and ("text/plain" not in ct_l):
                return ("SKIP", "", "")
            try:
                content_len = int(r.headers.get("content-length") or "0")
                if content_len and content_len > MAX_WEB_BYTES:
                    return ("SKIP", "", "")
            except Exception:
                pass
            text = r.text or ""
            if len(text.encode("utf-8", errors="ignore")) > MAX_WEB_BYTES:
                return ("SKIP", "", "")
            return ("FETCH", ct, text)
        except Exception:
            await asyncio.sleep(WEB_RETRY_BACKOFF * (attempt + 1))
    return ("ERR", "", "")

async def web_fetch_and_excerpt(query: str, results: List[Dict[str, str]], max_items: int = TOP_K_WEB) -> List[Dict[str, str]]:
    if not WEB_ENABLED:
        return []
    cands = [r for r in results if _safe_url(r.get("url", ""))]
    cands = cands[: max(12, max_items * 4)]
    if not cands:
        return []
    out: List[Dict[str, str]] = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        tasks = [_fetch_one_web(client, r["url"]) for r in cands]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for r, resp in zip(cands, responses):
        if isinstance(resp, Exception):
            continue
        kind, ct, raw_text = resp
        if not raw_text:
            continue
        url = (r.get("url") or "").strip()
        if not _allowed_url(url):
            continue
        ct_l = (ct or "").lower()
        txt = clean_text(raw_text) if kind == "CACHED" else (_html_to_text(raw_text) if "text/html" in ct_l else clean_text(raw_text))
        excerpt = _best_excerpts_from_text(query, txt)
        if not excerpt:
            continue
        title = (r.get("title") or "").strip() or url
        out.append({"title": title, "url": url, "excerpt": excerpt})
        if kind != "CACHED":
            _save_web_cache(url, "text/plain", txt)
        if len(out) >= max_items:
            break
    return out[:max_items]

# ============================================================
# RULES LAYER (fast deterministic answers)
# ============================================================

_RULES: List[Dict[str, Any]] = []

def _load_rules() -> None:
    global _RULES
    if _RULES or not RULES_ENABLED:
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
# “CHATGPT-LIKE” PROMPTS (customer-first, human, clean)
# ============================================================

# IMPORTANT: For customers, never volunteer internal system details, creator identity, tool names, etc.
SYSTEM_PROMPT_CUSTOMER_NORMAL = f"""
You are {PRODUCT_NAME}, a helpful AI assistant.

Voice:
- Sound human: clean sentences, natural phrasing, short paragraphs.
- Confident tone. Don’t hedge unless you genuinely lack evidence.
- Use emojis sparingly (only if it fits the vibe). Never spam.

Behavior:
- Answer what the user is trying to do, not just the literal wording.
- If the question is broad, give a clear, structured overview and include practical guidance.
- If the question is specific, answer it directly and stop.

Evidence use:
- You MAY use the provided SOURCES silently to improve accuracy.
- Do NOT mention “I read PDFs” or “I used evidence pack” or “Document: ...” unless the user asks for sources.
- If you are not sure and the sources don’t support a claim, say so plainly and offer the next best step.

Formatting:
- No markdown headings (#).
- Use short paragraphs.
- Use '-' bullets only when it clearly improves readability.
""".strip()

SYSTEM_PROMPT_CUSTOMER_EVIDENCE = f"""
You are {PRODUCT_NAME} in Evidence Mode.

Hard rules:
1) Only state numeric limits (mm, m, %, W/m²K, lux, etc.) if the exact number + unit appears in SOURCES.
2) If you state a numeric limit, include a short quote (1–2 lines) that contains that number.
3) Do not guess “typical” values.
4) Keep explanations clear and not overly long.

Customer-first:
- Do NOT dump raw sources.
- Only include citations/sources if the user asked for sources OR the answer is about compliance/numbers where proof matters.

Formatting:
- No markdown headings (#).
- Short paragraphs.
- Use '-' bullets only if it genuinely improves clarity.
""".strip()

ROUTER_PROMPT = """
You are a routing assistant. Decide how to answer.

Return ONLY JSON:
{
  "precision": "broad|mixed|precise",
  "need_docs": true|false,
  "need_web": true|false,
  "subqueries": ["..."]
}

Rules:
- need_docs true if PDFs likely contain answer or the user asks about technical standards, product specs, or uploaded manuals.
- need_web true if question is time-sensitive ("latest", "current", "2025", "new regulations") or general knowledge not in PDFs.
- If precision is broad, propose 3-7 short subqueries to improve retrieval across multiple PDFs (cover synonyms).
- If precision is precise, subqueries can be empty or 1 item.
""".strip()

RERANK_PROMPT = """
You are a strict reranker for technical evidence.

Given:
- USER_QUESTION
- CANDIDATES: JSON array of {id, doc, page, section, heading, table, text}

Return ONLY JSON:
{ "ranked_ids": ["id1","id2",...]} (max 12)

Rules:
- Put the most directly relevant candidates first.
- Prefer requirement statements and definitions.
- Prefer candidates with numbers/units ONLY if the question asks for them.
""".strip()

def build_contents(history: List[Dict[str, str]], user_message: str, controller: str, evidence: Dict[str, Any]) -> List[Content]:
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

    final_user = (user_message or "").strip() + "\n\n" + controller + "\n\nSOURCES_JSON:\n" + json.dumps(evidence, ensure_ascii=False)
    contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    return contents

# ============================================================
# EVIDENCE PACK (compact + multi-PDF friendly)
# ============================================================

CITE_TAG_RE = re.compile(r"\[(D\d+:\d+|W\d+)\]")

def _doc_code_map(chunks: List[Chunk]) -> Dict[str, str]:
    docs = []
    seen = set()
    for c in chunks:
        if c.doc not in seen:
            seen.add(c.doc)
            docs.append(c.doc)
    return {doc: f"D{i+1}" for i, doc in enumerate(docs)}

def _tight_excerpt(text: str, max_chars: int = 700) -> str:
    t = clean_text(text or "")
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"

def _tighten_chunk_text_for_evidence(c: Chunk, query: str, max_chars: int = 900) -> str:
    text = clean_text(c.text)
    if not text:
        return ""
    qt = set(tokenize(query))

    pieces = re.split(r"(?<=[\.\?\!])\s+", text)
    scored: List[Tuple[int, str]] = []
    for p in pieces:
        p = p.strip()
        if len(p) < 14:
            continue
        toks = tokenize(p)
        overlap = sum(1 for t in toks if t in qt)
        if NUM_WITH_UNIT_RE.search(p):
            overlap += 5
        if re.search(r"\b(must|shall|required|requirement|should)\b", p, re.I):
            overlap += 2
        if overlap > 0:
            scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [p for _, p in scored[:6]] or [text[:max_chars]]

    out = " ".join(picks).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "..."
    return out

def build_evidence_packets(
    chunks: List[Chunk],
    web_pages: List[Dict[str, str]],
    docai_hits: List[Tuple[str, str]],
    user_query: str,
    precision: str,
) -> Dict[str, Any]:
    max_chars = 520 if precision == "broad" else 780
    doc_codes = _doc_code_map(chunks)

    pdf_items = []
    for c in chunks[: max(14, RERANK_TOPK)]:
        pdf_items.append({
            "code": doc_codes.get(c.doc, "D?"),
            "doc": PDF_DISPLAY_NAME.get(c.doc, c.doc),
            "page": c.page,
            "section": c.section or "",
            "heading": c.heading or "",
            "table": c.table or "",
            "diagram": c.diagram or "",
            "excerpt": _tight_excerpt(_tighten_chunk_text_for_evidence(c, user_query, max_chars=max_chars), max_chars=max_chars),
        })

    web_items = []
    for i, w in enumerate(web_pages[:TOP_K_WEB], start=1):
        web_items.append({
            "code": f"W{i}",
            "title": (w.get("title") or "").strip(),
            "host": _host_from_url(w.get("url") or ""),
            "excerpt": _tight_excerpt(w.get("excerpt") or "", max_chars=520),
        })

    docai_items = []
    for i, (label, txt) in enumerate(docai_hits[:3], start=1):
        docai_items.append({
            "code": f"A{i}",
            "label": label,
            "excerpt": _tight_excerpt(txt, max_chars=700),
        })

    return {
        "pdf_docs": {v: PDF_DISPLAY_NAME.get(k, k) for k, v in doc_codes.items()},
        "pdf": pdf_items,
        "web": web_items,
        "docai": docai_items,
    }

def build_sources_footer(answer: str, evidence: Dict[str, Any]) -> str:
    used = set(CITE_TAG_RE.findall(answer or ""))
    if not used:
        return ""

    lines = ["", "Sources:"]
    pdf_docs = evidence.get("pdf_docs") or {}
    cited_doc_codes = sorted({u.split(":")[0] for u in used if u.startswith("D")})
    for code in cited_doc_codes:
        title = pdf_docs.get(code, code)
        lines.append(f"- {code}: {title}")

    web_items = evidence.get("web") or []
    cited_web = sorted([u for u in used if u.startswith("W")])
    if cited_web and web_items:
        web_by_code = {w.get("code"): w for w in web_items}
        for wcode in cited_web:
            w = web_by_code.get(wcode) or {}
            title = (w.get("title") or wcode).strip()
            host = (w.get("host") or "").strip()
            lines.append(f"- {wcode}: {title}" + (f" ({host})" if host else ""))

    return "\n".join(lines).strip()

# ============================================================
# RERANK (LLM)
# ============================================================

async def _rerank_candidates(user_msg: str, cands: List[Chunk], max_keep: int = RERANK_TOPK) -> List[Chunk]:
    if not RERANK_ENABLED or not cands:
        return cands[:max_keep]

    ensure_vertex_ready()
    if not _VERTEX_READY:
        return cands[:max_keep]

    small = cands[: min(len(cands), 20)]
    payload = [{
        "id": c.chunk_id,
        "doc": c.doc,
        "page": c.page,
        "section": c.section,
        "heading": c.heading,
        "table": c.table,
        "text": (c.text[:900] if c.text else "")
    } for c in small]

    model = get_model(MODEL_COMPLIANCE, RERANK_PROMPT)
    inp = "USER_QUESTION:\n" + user_msg.strip() + "\n\nCANDIDATES:\n" + json.dumps(payload)
    try:
        resp = model.generate_content(
            [Content(role="user", parts=[Part.from_text(inp)])],
            generation_config=GenerationConfig(temperature=0.1, top_p=0.6, max_output_tokens=300),
            stream=False
        )
        raw = (getattr(resp, "text", "") or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            raw = m.group(0)
        data = json.loads(raw)
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
# NUMERIC VERIFIER (prevents hallucinated numbers)
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
        "I can’t confirm those exact numeric values from the sources available for this answer.\n"
        "If you share the exact standard / document edition (or pin the right PDF), I’ll quote the exact line with each number.\n\n"
        "Numbers not found in the current sources: " + ", ".join(missing)
    )
    return False, safe

# ============================================================
# ROUTER / SUBQUERY GENERATION (the key upgrade)
# ============================================================

def _heuristic_subqueries(user_msg: str) -> List[str]:
    """
    Good fallback if router model fails.
    Creates a few focused variants + synonyms.
    """
    q = (user_msg or "").strip()
    if not q:
        return []
    toks = tokenize(q)
    top = [t for t, _ in Counter(toks).most_common(6)]
    base = " ".join(top) if top else q

    variants = [q]
    if base and base != q:
        variants.append(base)

    # tiny synonym expansions (extend freely)
    syn = {
        "stairs": ["staircase", "riser going", "landing handrail"],
        "lighting": ["illumination", "lux levels", "lighting design"],
        "fire": ["means of escape", "protected stair", "escape route"],
        "ber": ["deap", "primary energy", "u-value"],
    }
    for k, vs in syn.items():
        if k in q.lower():
            variants.extend([f"{q} {v}" for v in vs[:3]])

    # dedupe + cap
    out = []
    seen = set()
    for v in variants:
        vv = v.strip()
        if not vv:
            continue
        key = vv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(vv)
        if len(out) >= BROAD_MAX_SUBQUERIES:
            break
    return out

async def route_plan(user_msg: str, fam: str) -> Dict[str, Any]:
    """
    Returns {precision, need_docs, need_web, subqueries}
    """
    # start with deterministic baseline
    precision = question_precision(user_msg)
    numeric_needed = is_numeric_compliance(user_msg) and fam == "building_regs"

    # baseline tool needs
    need_docs = bool(DEFAULT_EVIDENCE_MODE or numeric_needed or fam in ("building_regs", "planning", "ber"))
    need_web = bool(WEB_ENABLED and (fam in ("planning", "general", "ber")) and ("latest" in user_msg.lower() or "current" in user_msg.lower() or "202" in user_msg))

    plan = {
        "precision": precision,
        "need_docs": need_docs,
        "need_web": need_web,
        "subqueries": [],
    }

    # try router model for better decomposition (ChatGPT-style)
    ensure_vertex_ready()
    if not _VERTEX_READY:
        if precision == "broad":
            plan["subqueries"] = _heuristic_subqueries(user_msg)
        return plan

    try:
        model = get_model(MODEL_ROUTER, ROUTER_PROMPT)
        resp = model.generate_content(
            [Content(role="user", parts=[Part.from_text(user_msg.strip())])],
            generation_config=GenerationConfig(temperature=0.2, top_p=0.7, max_output_tokens=380),
            stream=False
        )
        raw = (getattr(resp, "text", "") or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            raw = m.group(0)
        data = json.loads(raw)

        p = (data.get("precision") or precision).strip()
        if p not in ("broad", "mixed", "precise"):
            p = precision

        nd = bool(data.get("need_docs")) or need_docs
        nw = bool(data.get("need_web")) or need_web
        subs = data.get("subqueries") or []
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        if p == "broad" and not subs:
            subs = _heuristic_subqueries(user_msg)
        subs = subs[:BROAD_MAX_SUBQUERIES]

        plan = {"precision": p, "need_docs": nd, "need_web": nw, "subqueries": subs}
        return plan
    except Exception:
        if precision == "broad":
            plan["subqueries"] = _heuristic_subqueries(user_msg)
        return plan

# ============================================================
# RETRIEVAL ORCHESTRATION (multi-PDF, broad queries)
# ============================================================

async def retrieve_pdf_evidence(user_msg: str, plan: Dict[str, Any], pinned: Optional[str], page_hint: Optional[int]) -> List[Chunk]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    precision = plan.get("precision") or "mixed"
    subqueries: List[str] = plan.get("subqueries") or []

    if precision != "broad":
        scored = search_chunks_single_query(user_msg, top_k=TOP_K_CHUNKS, pinned_pdf=pinned, page_hint=page_hint)
        chunks = [c for _, c in scored][: max(10, RERANK_TOPK)]
        if RERANK_ENABLED and chunks:
            chunks = await _rerank_candidates(user_msg, chunks, max_keep=RERANK_TOPK)
        return chunks[:RERANK_TOPK]

    # BROAD: retrieve across multiple PDFs + multiple subqueries, then MMR select for diversity
    queries = subqueries or _heuristic_subqueries(user_msg)
    all_scored: List[Tuple[float, Chunk]] = []
    for q in queries:
        all_scored.extend(search_chunks_single_query(q, top_k=max(TOP_K_CHUNKS, 10), pinned_pdf=pinned, page_hint=page_hint))

    # sort + dedupe (keep best score per chunk)
    best_by_id: Dict[str, Tuple[float, Chunk]] = {}
    for s, c in all_scored:
        cur = best_by_id.get(c.chunk_id)
        if (cur is None) or (s > cur[0]):
            best_by_id[c.chunk_id] = (s, c)

    merged = list(best_by_id.values())
    merged.sort(key=lambda x: x[0], reverse=True)

    # take a bigger pool, then rerank, then MMR
    pool = [c for _, c in merged[:70]]

    if RERANK_ENABLED and pool:
        # rerank only top slice to keep it cheap
        pool = await _rerank_candidates(user_msg, pool[:22], max_keep=min(14, len(pool)))

    # rebuild scores for mmr (use merged scores if available)
    merged_score_map = {c.chunk_id: s for s, c in merged}
    scored_pool = [(merged_score_map.get(c.chunk_id, 0.01), c) for c in pool]

    selected = mmr_select(scored_pool, k=max(12, RERANK_TOPK + 5), lam=MMR_LAMBDA)

    # force doc diversity for broad
    by_doc: Dict[str, List[Chunk]] = defaultdict(list)
    for c in selected:
        by_doc[c.doc].append(c)

    docs_in_order = []
    seen_docs = set()
    for c in selected:
        if c.doc not in seen_docs:
            seen_docs.add(c.doc)
            docs_in_order.append(c.doc)

    final: List[Chunk] = []
    for d in docs_in_order[:BROAD_DOC_DIVERSITY_K]:
        final.extend(by_doc[d][:BROAD_PER_DOC])

    # backfill if too short
    if len(final) < max(10, RERANK_TOPK):
        for c in selected:
            if c.chunk_id not in {x.chunk_id for x in final}:
                final.append(c)
            if len(final) >= max(10, RERANK_TOPK):
                break

    return dedupe_chunks_keep_order(final)[: max(10, RERANK_TOPK)]

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

    # Upload to R2 (optional)
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
    # admin / developer endpoint vibe; still safe
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
        "features": [
            "Customer-first chat output (human tone, clean formatting)",
            "Multi-PDF retrieval with query decomposition + diversity selection",
            "R2 Cloudflare PDF listing + download-to-cache indexing",
            "Evidence-mode numeric guarding (no hallucinated numbers)",
            "Optional web search (only when it adds value)",
            "Optional DocAI extraction support",
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
        "models": {"chat": MODEL_CHAT, "compliance": MODEL_COMPLIANCE, "router": MODEL_ROUTER},
        "project": GCP_PROJECT_ID,
        "location": GCP_LOCATION,
        "docai_helper": _DOCAI_HELPER_AVAILABLE,
        "web_enabled": WEB_ENABLED,
        "web_strict_allowlist": WEB_STRICT_ALLOWLIST,
        "embed_enabled": bool(EMBED_ENABLED and _EMBED_MODEL is not None),
        "rerank_enabled": bool(RERANK_ENABLED),
        "verify_numeric": bool(VERIFY_NUMERIC),
        "rules_enabled": bool(RULES_ENABLED and RULES_FILE.exists()),
        "r2": {
            "enabled": R2_ENABLED,
            "bucket": R2_BUCKET,
            "endpoint": R2_ENDPOINT,
            "access_key_set": bool(R2_ACCESS_KEY_ID),
            "secret_set": bool(R2_SECRET_ACCESS_KEY),
        },
    }

# ============================================================
# STREAMING CORE (fixed, cleaner, “ChatGPT-like”)
# ============================================================

async def _stream_answer_async(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None,
):
    try:
        user_msg = normalize_model_output((message or "").strip())
        if not user_msg:
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()

        # Smalltalk (fast + human)
        if is_smalltalk(user_msg):
            friendly = f"Hey — I’m {PRODUCT_NAME}. What can I help you with?"
            if chat_id:
                remember(chat_id, "user", user_msg)
                remember(chat_id, "assistant", friendly)
            yield sse_send(friendly)
            yield "event: done\ndata: ok\n\n"
            return

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

        fam = _question_family(user_msg)

        # Rules first (deterministic wins)
        rules_hit = _match_rules(user_msg) if RULES_ENABLED else []
        if rules_hit:
            rule = rules_hit[0]
            rendered = normalize_model_output((rule.get("answer") or "").strip())
            if chat_id:
                remember(chat_id, "assistant", rendered)
            yield sse_send(rendered)
            yield "event: done\ndata: ok\n\n"
            return

        # Router plan (precision + tools + subqueries)
        plan = await route_plan(user_msg, fam)

        # Force-docs overrides router
        if force_docs:
            plan["need_docs"] = True

        pinned = pdf

        numeric_needed = is_numeric_compliance(user_msg) and fam == "building_regs"
        evidence_mode = bool(DEFAULT_EVIDENCE_MODE or force_docs or numeric_needed or fam in ("building_regs", "planning", "ber"))

        # Decide web usage (ChatGPT-like: only when helpful)
        # - never web for numeric compliance unless user explicitly asks for web/latest
        # - web if no PDFs exist OR user asked time-sensitive / asked for sources
        has_pdfs = bool(list_pdfs())
        wants_sources = user_asked_for_sources(user_msg)

        do_web = bool(
            WEB_ENABLED
            and plan.get("need_web", False)
            and not numeric_needed
        )
        if not has_pdfs and WEB_ENABLED and fam in ("general", "planning", "ber"):
            do_web = True
        if wants_sources and WEB_ENABLED and fam in ("general", "planning", "ber"):
            do_web = True

        # Retrieve concurrently
        async def get_pdf_chunks() -> List[Chunk]:
            if not plan.get("need_docs", False):
                return []
            return await retrieve_pdf_evidence(user_msg, plan, pinned, page_hint)

        async def get_docai_hits() -> List[Tuple[str, str]]:
            if not pinned:
                return []
            if not _docai_chunk_files_for(pinned):
                return []
            # only helpful when a specific pinned doc exists
            return docai_search_text(pinned, user_msg, k=2 if plan.get("precision") != "broad" else 3)

        async def get_web_evidence() -> List[Dict[str, str]]:
            if not do_web:
                return []
            serp = await web_search_serper(user_msg, k=TOP_K_WEB)
            return await web_fetch_and_excerpt(user_msg, serp, max_items=TOP_K_WEB)

        pdf_chunks, docai_hits, web_pages = await asyncio.gather(
            get_pdf_chunks(),
            get_docai_hits(),
            get_web_evidence(),
        )

        # Numeric compliance must have doc evidence
        if numeric_needed and not pdf_chunks and not docai_hits:
            msg = normalize_model_output(
                "I can’t confirm the exact numeric requirement yet because the relevant source text isn’t available to me.\n"
                "Upload or pin the correct TGD/standard PDF and I’ll quote the exact line with the number."
            )
            if chat_id:
                remember(chat_id, "assistant", msg)
            yield sse_send(msg)
            yield "event: done\ndata: ok\n\n"
            return

        evidence_pack = build_evidence_packets(
            pdf_chunks,
            web_pages,
            docai_hits,
            user_query=user_msg,
            precision=plan.get("precision") or question_precision(user_msg),
        )

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\r", " ").replace("\n", " ")
            yield f"data: [ERROR] {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        # Choose prompt (customer-first)
        system_prompt = SYSTEM_PROMPT_CUSTOMER_EVIDENCE if evidence_mode else SYSTEM_PROMPT_CUSTOMER_NORMAL
        model_name = MODEL_COMPLIANCE if evidence_mode else MODEL_CHAT
        model = get_model(model_name=model_name, system_prompt=system_prompt)

        precision = plan.get("precision") or question_precision(user_msg)

        controller = f"""
STYLE:
- Human, clean, confident.
- No robotic phrases.
- Don’t mention internal tools, pipeline, “evidence packs”, or file handling.
- Use short paragraphs. Use '-' bullets only when it genuinely helps.

PRECISION:
- precision = {precision}
- If precise: answer directly in 1–3 short paragraphs, then stop.
- If broad: give a clear overview, grouped logically, then ask ONE helpful follow-up question.

CITATIONS:
- Only include [D#:p] / [W#] citations if the user asked for sources OR if this is a numeric/compliance claim that needs proof.
- Otherwise, do not show sources, just answer.

When citing PDFs use: [D#:page]
Example: [D1:12]
""".strip()

        contents = build_contents(history_for_prompt, user_msg, controller, evidence_pack)

        # Generate (buffer for clean output)
        stream = model.generate_content(
            contents,
            generation_config=get_generation_config(evidence_mode),
            stream=True,
        )

        full: List[str] = []
        for ch in stream:
            delta = getattr(ch, "text", None)
            if not delta:
                continue
            full.append(delta)

        draft = normalize_model_output("".join(full).strip())

        # Numeric verification guard
        sources_blob = _sources_text_blob_for_verification(pdf_chunks, docai_hits, web_pages)
        ok, safe_or_same = _hard_verify_numeric(draft, sources_blob)
        if not ok:
            draft = normalize_model_output(safe_or_same)

        # Source footer only when user asked OR evidence_mode with citations present
        show_sources = bool(wants_sources or (evidence_mode and CITE_TAG_RE.search(draft)))
        if show_sources:
            footer = build_sources_footer(draft, evidence_pack)
            if footer:
                draft = draft.rstrip() + "\n\n" + footer

        # Final polish: guarantee punctuation spacing one more time
        draft = normalize_model_output(draft)

        yield sse_send(draft)

        if chat_id:
            remember(chat_id, "assistant", draft)

        yield "event: done\ndata: ok\n\n"

    except Exception as e:
        msg = str(e).replace("\r", " ").replace("\n", " ")
        log.exception("Chat stream error: %s", msg)
        yield f"data: [ERROR] {msg}\n\n"
        yield "event: done\ndata: ok\n\n"
        return

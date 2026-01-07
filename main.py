from __future__ import annotations

import os
import re
import math
import json
import time
import tempfile
import hashlib
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import httpx
import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationConfig

# Optional: Vertex Embeddings (no extra deps)
try:
    from vertexai.language_models import TextEmbeddingModel  # type: ignore

    _VERTEX_EMBEDDINGS_AVAILABLE = True
except Exception:
    TextEmbeddingModel = None  # type: ignore
    _VERTEX_EMBEDDINGS_AVAILABLE = False

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

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEB_CACHE_DIR = CACHE_DIR / "web"
WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "40"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "32000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Web search (Serper)
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()  # https://serper.dev
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
EMBED_TOPK = int(os.getenv("EMBED_TOPK", "24"))  # candidate pool for rerank
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rerank (LLM reranker)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
RERANK_TOPK = int(os.getenv("RERANK_TOPK", str(TOP_K_CHUNKS)))

# Broad vs precise answering controls
BROAD_DOC_DIVERSITY_K = int(os.getenv("BROAD_DOC_DIVERSITY_K", "3"))  # broad questions: cover up to N docs
BROAD_TOPIC_HITS_K = int(os.getenv("BROAD_TOPIC_HITS_K", "2"))        # broad questions: hits per doc/topic bucket
BROAD_MAX_SUBQUERIES = int(os.getenv("BROAD_MAX_SUBQUERIES", "6"))    # cost limiter for topic expansion

# Web allowlist (safe + higher trust)
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

# Hard numeric verification
VERIFY_NUMERIC = os.getenv("VERIFY_NUMERIC", "true").lower() in ("1", "true", "yes", "on")

# Rules layer
RULES_FILE = Path(os.getenv("RULES_FILE", str(BASE_DIR / "rules.json")))
RULES_ENABLED = os.getenv("RULES_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Eval harness
EVAL_FILE = Path(os.getenv("EVAL_FILE", str(BASE_DIR / "eval_tests.json")))

# Timeouts
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "18"))
# ----------------------------
# Web fetch behaviour (ChatGPT-like)
# ----------------------------

MAX_WEB_BYTES = int(os.getenv("MAX_WEB_BYTES", str(2_500_000)))  # 2.5 MB safety cap
WEB_RETRIES = int(os.getenv("WEB_RETRIES", "2"))                 # retry flaky sites
WEB_RETRY_BACKOFF = float(os.getenv("WEB_RETRY_BACKOFF", "0.6")) # seconds

# Web cache
WEB_CACHE_TTL_SECONDS = int(os.getenv("WEB_CACHE_TTL_SECONDS", str(12 * 60 * 60)))  # 12h


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

        # Lazy init embeddings model (optional)
        _EMBED_MODEL = None
        if EMBED_ENABLED and _VERTEX_EMBEDDINGS_AVAILABLE and TextEmbeddingModel is not None:
            try:
                _EMBED_MODEL = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)
            except Exception:
                _EMBED_MODEL = None

        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])


def get_generation_config(is_evidence: bool) -> GenerationConfig:
    # More ChatGPT-like: stable, paragraphy
    if is_evidence:
        return GenerationConfig(temperature=0.2, top_p=0.8, max_output_tokens=1400)
    return GenerationConfig(temperature=0.6, top_p=0.9, max_output_tokens=1400)


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
    "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with", "is", "are", "be", "as", "at", "from", "by",
    "that", "this", "it", "your", "you", "we", "they", "their", "there", "what", "which", "when", "where", "how",
    "can", "shall", "should", "must", "may", "not", "than", "then", "into", "onto", "also", "such"
}

NUM_WITH_UNIT_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(mm|m|metre|meter|w/m²k|w\/m2k|%|kwh|kwh\/m2\/yr|kwh\/m²\/yr|lux|lm|lumen|lumens)\b",
    re.I
)
ANY_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")

SECTION_RE = re.compile(r"^\s*(?:section\s+)?(\d+(?:\.\d+){0,4})\b", re.I)
TABLE_RE = re.compile(r"\btable\s+([a-z0-9][a-z0-9\.\-]*)\b", re.I)
DIAGRAM_RE = re.compile(r"\bdiagram\s+([a-z0-9][a-z0-9\.\-]*)\b", re.I)

WEB_CITE_TOKEN_RE = re.compile(r"\[WEB:(\d+)\]\(\s*(https?://[^\s)]+)\s*\)")

def _enforce_web_citations(answer: str, web_pages: List[Dict[str, str]]) -> str:
    """
    If the model doesn't include any [WEB:i](URL) token, append a Sources section.
    """
    if not web_pages:
        return answer

    # If it already used at least one token, accept.
    if WEB_CITE_TOKEN_RE.search(answer or ""):
        return answer

    sources_lines = ["\n\n---\n\n### Sources (web)"]
    for i, w in enumerate(web_pages, start=1):
        url = (w.get("url") or "").strip()
        if url:
            sources_lines.append(f"- [WEB:{i}]({url})")

    return (answer or "").rstrip() + "\n" + "\n".join(sources_lines)

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


def _safe_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _host_from_url(url: str) -> str:
    try:
        if not _safe_url(url):
            return ""
        m = re.match(r"^https?://([^/]+)", url.strip(), re.I)
        host = (m.group(1).lower() if m else "")
        host = host.split("@")[-1]  # userinfo
        host = host.split(":")[0]   # port
        return host
    except Exception:
        return ""


# If true, only allow domains in WEB_ALLOWLIST (current behavior).
# If false, allow most domains except blocked ones.
WEB_STRICT_ALLOWLIST = os.getenv("WEB_STRICT_ALLOWLIST", "false").lower() in ("1", "true", "yes", "on")

WEB_BLOCKLIST_DEFAULT = [
    "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "pinterest.com", "reddit.com",  # optional; can remove if you want reddit
]
WEB_BLOCKLIST = [d.strip().lower() for d in (os.getenv("WEB_BLOCKLIST", "") or "").split(",") if d.strip()]
if not WEB_BLOCKLIST:
    WEB_BLOCKLIST = WEB_BLOCKLIST_DEFAULT

def _blocked_url(url: str) -> bool:
    host = _host_from_url(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in WEB_BLOCKLIST)

def _allowed_url(url: str) -> bool:
    # Blocklist always wins
    if _blocked_url(url):
        return False

    # If not strict: allow most domains
    if not WEB_STRICT_ALLOWLIST:
        return True

    # Strict mode: allowlist required
    host = _host_from_url(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in WEB_ALLOWLIST)


# ============================================================
# BROAD vs PRECISE + TOPIC COVERAGE
# ============================================================

BROAD_HINTS = ["tell me about", "explain", "overview", "what about", "guidance on", "info on", "general"]
PRECISE_HINTS = ["minimum", "maximum", "min", "max", "required", "shall", "must", "limit", "section", "table", "clause"]

PART_PATTERN = re.compile(r"\bpart\s*[a-m]\b", re.I)


def question_precision(q: str) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "broad"
    if PART_PATTERN.search(ql):
        return "precise"
    if re.search(r"\b(section|table|diagram|clause)\b", ql):
        return "precise"
    if NUM_WITH_UNIT_RE.search(ql) or ANY_NUMBER_RE.search(ql):
        return "precise"
    if any(h in ql for h in PRECISE_HINTS):
        return "precise"
    if any(h in ql for h in BROAD_HINTS):
        return "broad"
    if len(ql.split()) <= 6:
        return "broad"
    return "mixed"


TOPIC_MAP: Dict[str, List[str]] = {
    "lighting_emergency": ["emergency lighting", "escape lighting", "emergency illumination"],
    "lighting_daylight": ["daylight", "natural light", "daylighting", "rooflight", "skylight"],
    "lighting_controls": ["lighting controls", "occupancy sensor", "presence detector", "automatic lighting"],
    "lighting_energy": ["lighting efficiency", "luminaire efficacy", "lamp efficacy", "controls energy"],
    "fire_escape": ["means of escape", "escape route", "travel distance", "fire safety"],
    "access": ["accessible", "wheelchair", "part m", "ramp"],
}


def detect_topic_keys(user_msg: str) -> List[str]:
    ql = (user_msg or "").lower()
    hits: List[str] = []
    for topic, keys in TOPIC_MAP.items():
        if any(k in ql for k in keys):
            hits.append(topic)

    # If user says "lighting" without subtype: try a few likely subtopics
    if "lighting" in ql and not any(t.startswith("lighting_") for t in hits):
        hits.extend(["lighting_emergency", "lighting_controls", "lighting_daylight", "lighting_energy"])

    seen = set()
    out = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def intent_queries_for_topics(user_msg: str, topics: List[str]) -> List[str]:
    base = (user_msg or "").strip()
    out: List[str] = []
    for t in topics:
        if t == "lighting_emergency":
            out.append(base + " emergency lighting escape routes requirements")
        elif t == "lighting_daylight":
            out.append(base + " daylight natural lighting requirement guidance")
        elif t == "lighting_controls":
            out.append(base + " lighting controls automatic controls occupancy sensor")
        elif t == "lighting_energy":
            out.append(base + " lighting energy efficiency requirements")
        elif t == "fire_escape":
            out.append(base + " means of escape fire safety travel distance")
        elif t == "access":
            out.append(base + " accessibility part m ramp")
        else:
            out.append(base + " " + t.replace("_", " "))
    return out


# ============================================================
# CHUNK INDEX (BM25 + METADATA + OPTIONAL EMBEDDINGS)
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
EMBED_INDEX: Dict[str, Dict[str, Any]] = {}  # pdf_name -> {"vectors": {chunk_id: [..]}, "dim": int}


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
    # closer pages rank higher
    return 3.0 * ((radius - dist + 1) / (radius + 1))

def _pdf_fingerprint(pdf_path: Path) -> str:
    st = pdf_path.stat()
    s = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def list_pdfs() -> List[str]:
    files = []
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p.name)
    files.sort()
    return files


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
    """
    Table-aware chunker:
    - keeps table-like blocks as standalone chunks
    - avoids smearing tables into paragraph chunks
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
    for ln in lines[:100]:
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
        if meta.get("pdf_name") != pdf_name:
            return False

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
                        section=obj.get("section", "") or "",
                        heading=obj.get("heading", "") or "",
                        table=obj.get("table", "") or "",
                        diagram=obj.get("diagram", "") or "",
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
    except Exception:
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
    except Exception:
        pass


def index_pdf_to_chunks(pdf_path: Path) -> None:
    pdf_name = pdf_path.name

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

    finally:
        doc.close()


def ensure_chunk_indexed(pdf_name: str) -> None:
    if pdf_name in CHUNK_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf_to_chunks(p)


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
    except Exception:
        return


def _save_embeddings(pdf_name: str, vectors: Dict[str, List[float]]) -> None:
    try:
        path = _embed_cache_path(pdf_name)
        with path.open("w", encoding="utf-8") as f:
            for cid, vec in vectors.items():
                f.write(json.dumps({"chunk_id": cid, "vec": vec}) + "\n")
    except Exception:
        pass


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
    except Exception:
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


def search_chunks(
    question: str,
    top_k: int = TOP_K_CHUNKS,
    pinned_pdf: Optional[str] = None,
    page_hint: Optional[int] = None,
) -> List[Chunk]:
    """
    Retrieval:
      - BM25 everywhere
      - If embeddings exist: boost vector candidates so they rise naturally
      - Page hint boosts near the hinted page
    """
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

        scored: List[Tuple[float, Chunk]] = []
        for ch in idx["chunks"]:
            s = bm25_score(qt, ch.tf, df, N, ch.length, avgdl)
            s += page_hint_boost(ch.page, page_hint)

            if vector_set and ch.chunk_id in vector_set:
                s += 2.5
            if s > 0:
                scored.append((s, ch))

        scored.sort(key=lambda x: x[0], reverse=True)

        take = max(24, top_k * 6)
        candidates.extend(scored[:take])

        if pinned_pdf and pdf_name == pinned_pdf:
            break

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out: List[Chunk] = []
    for _, c in candidates:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
        if len(out) >= max(24, top_k * 4):
            break
    return out



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
    """
    For broad questions, avoid one doc dominating. Keep up to max_docs and per_doc each.
    """
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

SMALLTALK_RE = re.compile(r"^(hi|hello|hey|yo|howdy|sup|hiya|evening|morning|afternoon)\b", re.I)

def is_smalltalk(message: str) -> bool:
    m = (message or "").strip().lower()
    if not m:
        return False

    # Very short greeting-like messages
    if len(m) <= 12 and (m in {"hi", "hey", "hello", "yo", "hiya", "sup"}):
        return True

    # Greeting at the start, short overall, and NOT a compliance/planning/ber question
    if SMALLTALK_RE.match(m) and len(m) <= 32:
        if not is_compliance_question(m) and not is_planning_question(m) and not is_ber_question(m):
            return True

    # “thanks”, “ok”, etc. (optional)
    if len(m) <= 12 and m in {"ok", "okay", "thanks", "thank you", "cool", "nice"}:
        return True

    return False

# ============================================================
# INTENT DETECTION + DOC TYPE SANITY
# ============================================================

COMPLIANCE_KEYWORDS = [
    "tgd", "technical guidance", "building regulations", "building regs",
    "part a", "part b", "part c", "part d", "part e", "part f", "part g", "part h",
    "part j", "part k", "part l", "part m",
    "fire", "escape", "travel distance", "compartment", "smoke", "fire cert",
    "access", "accessible", "wheelchair", "dac",
    "ber", "deap", "u-value", "y-value", "airtight", "thermal bridge",
    "means of escape"
]

PLANNING_KEYWORDS = [
    "planning", "planning permission", "part 8", "development plan", "planning and development",
    "exempted development", "regulations", "statutory instrument", "s.i.", "site notice", "newspaper notice"
]

BER_KEYWORDS = ["ber", "deap", "primary energy", "renewable", "u-value", "y-value", "airtight", "thermal bridge", "mpep"]

NUMERIC_TRIGGERS = [
    "minimum", "maximum", "min", "max", "limit",
    "distance", "width", "height",
    "u-value", "y-value", "rise", "riser", "going", "pitch",
    "stairs", "stair", "staircase", "landing", "headroom",
    "travel distance",
    "lux", "lumen", "lumens"
]


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


def auto_pin_pdf(question: str) -> Optional[str]:
    q = (question or "").lower()
    pdfs = list_pdfs()

    def pick_by_regex(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            rx = re.compile(pat, re.I)
            for name in pdfs:
                if rx.search(name):
                    return name
        return None

    if is_planning_question(q):
        return pick_by_regex([r"planning.*development", r"regulation"])

    if any(w in q for w in ["stairs", "riser", "going", "pitch", "handrail"]):
        return pick_by_regex([r"\bpart[-_ ]?k\b", r"tgd.*k", r"stairs"])

    if any(w in q for w in ["access", "accessible", "wheelchair", "ramp", "dac"]):
        return pick_by_regex([r"\bpart[-_ ]?m\b", r"tgd.*m", r"access"])

    if any(w in q for w in ["fire", "escape", "travel distance", "emergency lighting"]):
        return pick_by_regex([r"\bpart[-_ ]?b\b", r"tgd.*b", r"fire"])

    if any(w in q for w in ["ber", "u-value", "y-value", "energy", "lighting controls"]):
        return pick_by_regex([r"\bpart[-_ ]?l\b", r"tgd.*l", r"energy"])

    return None


def _doc_type_from_pdfname(name: str) -> str:
    n = (name or "").lower()
    if "planning and development" in n or "regulation" in n:
        return "planning"
    if "tgd" in n or "technical guidance" in n:
        return "building_regs"
    if "deap" in n or "ber" in n:
        return "ber"
    if any(x in n for x in ["part l", "part m", "part k", "part b"]):
        return "building_regs"
    return "general"


def _question_family(q: str) -> str:
    if is_planning_question(q):
        return "planning"
    if is_ber_question(q):
        return "ber"
    if is_compliance_question(q):
        return "building_regs"
    return "general"

def anchor_from_query(q: str) -> Dict[str, str]:
    out = {}
    m = TABLE_RE.search(q)
    if m:
        out["table"] = m.group(1)
    m = DIAGRAM_RE.search(q)
    if m:
        out["diagram"] = m.group(1)
    m = SECTION_RE.search(q)
    if m:
        out["section"] = m.group(1)
    return out



# ============================================================
# WEB SEARCH + FETCH + EXCERPT (allowlist + evidence) + CACHE
# ============================================================

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
    except Exception:
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

    # Remove junk
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)

    # Preserve paragraph-ish breaks before stripping tags
    html = re.sub(r"(?i)</(p|div|section|article|br|li|h1|h2|h3|h4|h5|tr)>", "\n", html)

    # Drop remaining tags
    html = re.sub(r"(?is)<[^>]+>", " ", html)

    # Decode common entities
    html = (
        html.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
    )

    # Normalize whitespace but keep line breaks somewhat
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

# ============================================================
# REDIRECT MAP CACHE (orig_url -> final_url)
# ============================================================

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
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

async def _fetch_one_web_with_final_url(
    client: httpx.AsyncClient,
    orig_url: str
) -> Tuple[str, str, str, str, str]:
    """
    Returns: (kind, original_url, content_type, raw_text, final_url)

    kind:
      - "CACHED" -> raw_text is already cleaned plain text (content_type="text/plain")
      - "FETCH"  -> raw_text is resp.text (html or plain)
      - "SKIP" / "HTTP" / "ERR" -> raw_text empty
    """
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
        # --------------------------------------------------
        # 1) Try redirect-map cache (NO NETWORK)
        # --------------------------------------------------
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

                # size cap (header)
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

        # --------------------------------------------------
        # 2) Fallback: fetch orig_url to discover redirects (ONE NETWORK)
        # --------------------------------------------------
        resp = await _get_with_retries(orig_url)
        if not resp:
            return ("ERR", orig_url, "", "", "")

        if resp.status_code >= 400:
            return ("HTTP", orig_url, "", "", "")

        ct = (resp.headers.get("content-type") or "")
        ct_l = ct.lower()
        final_url = (str(resp.url) if resp.url else orig_url).strip()

        # Store mapping for next time
        if _safe_url(final_url):
            _save_redirect_map(orig_url, final_url)

        if not _allowed_url(final_url):
            return ("SKIP", orig_url, "", "", final_url)

        # Cache lookup by FINAL URL
        cached2 = _load_web_cache(final_url)
        if cached2 and isinstance(cached2.get("text"), str):
            return ("CACHED", orig_url, "text/plain", cached2["text"], final_url)

        if ("text/html" not in ct_l) and ("text/plain" not in ct_l):
            return ("SKIP", orig_url, "", "", final_url)

        # size cap (header)
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
    max_items: int = TOP_K_WEB
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

        title = (r.get("title") or "").strip()
        if not title:
            title = final_url

        out.append({
            "title": title,
            "url": final_url.strip(),
            "excerpt": excerpt,
        })

        if kind != "CACHED":
            _save_web_cache(final_url, "text/plain", txt)

        if len(out) >= max_items:
            break

    return out[:max_items]


# ============================================================
# RULES LAYER (top checks)
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
        score = 0
        for k in kws:
            if k in q:
                score += 1
        if score > 0:
            hits.append((score, rule))

    hits.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in hits[:3]]


# ============================================================
# PROMPTS (ChatGPT-like style + broad/precise controller)
# ============================================================

SYSTEM_PROMPT_NORMAL = """
You are Raheem AI.

Voice & tone:
- Write like ChatGPT: natural, confident paragraphs.
- Match the user's tone: if they’re casual, be friendly; if they’re technical, be crisp and professional.
- Light, professional humour is okay if it fits the user's tone. Never cheesy.
- Avoid hashtags, emoji spam, and “blog style” headings.

Formatting:
- Do NOT use markdown headings (no #, ##).
- Use short paragraphs.
- Only use bullet points when the user asks for a list or when comparing options.

Citations:
- If you used web sources, cite them inline as [1], [2] etc (square bracket numbers).
- Don’t show raw URLs inline. Put a “Sources” list at the end if needed.

If the question is broad: answer well, then ask ONE focused follow-up.
""".strip()

SYSTEM_PROMPT_EVIDENCE = """
You are Raheem AI (Evidence Mode).

Voice:
- Same natural paragraph style as ChatGPT.
- Professional and careful.

Hard rules:
1) Only state numeric limits (mm, m, %, W/m²K, lux, etc.) if the exact number + unit appears in SOURCES.
2) When you state a numeric limit, include a short quote (1–2 lines) that contains that number.
3) Cite sources clearly:
   - PDF: (Document: <name>, p.<page>) and optionally Section/Table if known
   - Web: [1], [2] etc (no raw URLs inline)

Formatting:
- No markdown headings (#).
- Bullets only if it genuinely improves clarity.

If SOURCES don’t contain what’s needed, say so plainly and ask ONE focused follow-up.
""".strip()


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


def build_contents(history: List[Dict[str, str]], user_message: str, sources_block: str, precision: str) -> List[Content]:
    """
    Adds a small controller hint:
    - broad => cover all key topics user mentioned + one follow-up
    - precise => answer only what was asked
    """
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

    if precision == "broad":
        controller = (
            "\n\n[CONTROLLER]\n"
            "The user question is BROAD. Cover the main interpretations that match the user's words, "
            "and make sure you address EACH key topic mentioned by the user. "
            "Keep it in paragraphs. End with ONE focused follow-up.\n"
        )
    elif precision == "precise":
        controller = (
            "\n\n[CONTROLLER]\n"
            "The user question is PRECISE. Answer ONLY what was asked. "
            "Do not expand into unrelated topics.\n"
        )
    else:
        controller = (
            "\n\n[CONTROLLER]\n"
            "The user question is MIXED. Answer the core request, and only mention adjacent points "
            "if they are directly implied.\n"
        )

    if sources_block.strip():
        final_user += controller + "\nSOURCES:\n" + sources_block.strip()
    else:
        final_user += controller

    contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    return contents


def _format_pdf_citation(c: Chunk) -> str:
    bits = [f"Document: {c.doc}"]
    if c.section:
        bits.append(f"Section: {c.section}")
    elif c.table:
        bits.append(f"Table: {c.table}")
    else:
        bits.append(f"p.{c.page}")
    return "(" + ", ".join(bits) + ")"


def _tighten_chunk_text_for_evidence(c: Chunk, query: str, max_chars: int = 950) -> str:
    """
    Evidence pack builder (query-driven):
    - keep only the most relevant lines/sentences
    - boost numeric + requirement statements
    """
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


def build_sources_block(
    chunks: List[Chunk],
    web_pages: List[Dict[str, str]],
    docai_hits: List[Tuple[str, str]],
    user_query: str,
    precision: str
) -> str:
    parts: List[str] = []

    # Keep evidence shorter for broad questions to reduce bullet-y responses
    max_chars = 650 if precision == "broad" else 950

    if chunks:
        parts.append("PDF EVIDENCE:")
        for c in chunks:
            excerpt = _tighten_chunk_text_for_evidence(c, query=user_query, max_chars=max_chars) or clean_text(c.text)[:max_chars]
            parts.append(f"[PDF] {_format_pdf_citation(c)} | id:{c.chunk_id}\n{excerpt}")

    if docai_hits:
        parts.append("\nDOC.AI EVIDENCE (raw extraction, use carefully):")
        for label, text in docai_hits:
            t = clean_text(text)
            if len(t) > 1800:
                t = t[:1800] + "..."
            parts.append(f"[DOCAI] {label}\n{t}")

    if web_pages:
        parts.append("\nWEB EVIDENCE:")
        for i, w in enumerate(web_pages, start=1):
            url = (w.get("url") or "").strip()
            title = (w.get("title") or "").strip()
            ex = clean_text(w.get("excerpt", ""))
            # Give the model a stable citation token it can copy verbatim
            parts.append(
                f"[WEB {i}] {title}\n"
                f"URL: {url}\n"
                f"CITE_TOKEN: [WEB:{i}]({url})\n"
                f"{ex}"
            )

    return "\n\n".join([p for p in parts if p]).strip()


# ============================================================
# RERANK (LLM)
# ============================================================

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
    except Exception:
        return small[:max_keep]


# ============================================================
# HARD NUMERIC VERIFIER (server-enforced)
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

    missing = []
    for n in nums:
        if n.lower() not in (sources_blob_lower or ""):
            missing.append(n)

    if not missing:
        return True, answer

    safe = (
        "I can’t safely confirm those numeric values from the evidence I have available right now.\n\n"
        "If you tell me which exact document/edition to use (or upload it), I’ll quote the exact line containing the number.\n\n"
        "Numbers I’m refusing because they are not present in the current evidence: "
        + ", ".join(missing)
    )
    return False, safe


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
        "embeddings_ready": bool(EMBED_INDEX.get(dest.name, {}).get("vectors")) if dest.name in EMBED_INDEX else False,
        "web_enabled": WEB_ENABLED,
        "docai": {
            "attempted": bool(_DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text),
            "ok": docai_ok,
            "chunks_saved": docai_chunks_saved,
            "error": docai_error,
        }
    }

@app.post("/chat")
async def chat_endpoint(
    chat_id: str = Query(""),
    message: str = Body(..., embed=True),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    messages: Optional[List[Dict[str, Any]]] = Body(None),
):
    return StreamingResponse(
        _stream_answer_async(chat_id, message, force_docs, pdf, page_hint, messages=messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # helpful on nginx
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
        "web_allowlist": WEB_ALLOWLIST,
        "embed_enabled": bool(EMBED_ENABLED and _EMBED_MODEL is not None),
        "rerank_enabled": bool(RERANK_ENABLED),
        "verify_numeric": bool(VERIFY_NUMERIC),
        "rules_enabled": bool(RULES_ENABLED and RULES_FILE.exists()),
        "web_cache_ttl_seconds": WEB_CACHE_TTL_SECONDS,
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
    messages: Optional[List[Dict[str, Any]]] = None,
):
    try:
        user_msg = (message or "").strip()
        if not user_msg:
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        # Smalltalk fast-path
        if is_smalltalk(user_msg):
            friendly = "Hey — how can I help you today?"
            if chat_id:
                remember(chat_id, "user", user_msg)
                remember(chat_id, "assistant", friendly)
            yield f"data: {friendly.replace(chr(10), '\\\\n')}\n\n"
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

        fam = _question_family(user_msg)
        precision = question_precision(user_msg)

        evidence_mode = force_docs or DEFAULT_EVIDENCE_MODE or fam in ("building_regs", "planning", "ber")
        numeric_needed = is_numeric_compliance(user_msg) and fam == "building_regs"

        pinned = pdf or auto_pin_pdf(user_msg)

        # Rules first
        rules_hit = _match_rules(user_msg) if RULES_ENABLED else []
        if rules_hit:
            rule = rules_hit[0]
            ans = (rule.get("answer") or "").strip()
            quote = (rule.get("quote") or "").strip()
            cit = rule.get("citation") or {}

            rendered = ans
            if quote:
                rendered += "\n\n> " + quote
            if cit:
                bits = []
                if cit.get("doc"):
                    bits.append(f"Document: {cit.get('doc')}")
                if cit.get("section"):
                    bits.append(f"Section: {cit.get('section')}")
                if cit.get("table"):
                    bits.append(f"Table: {cit.get('table')}")
                if bits:
                    rendered += "\n\n(" + ", ".join(bits) + ")"

            if chat_id:
                remember(chat_id, "assistant", rendered)
            yield f"data: {rendered.replace(chr(10), '\\\\n')}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        # ----------------------------
        # Helpers (MUST stay inside _stream_answer_async)
        # ----------------------------
        async def get_pdf_chunks() -> List[Chunk]:
            if not list_pdfs():
                return []

            # Broad: topic expansion + dedupe + rerank + doc diversity
            if precision == "broad":
                topics = detect_topic_keys(user_msg)
                queries = [user_msg] + intent_queries_for_topics(user_msg, topics)
                queries = queries[: max(1, BROAD_MAX_SUBQUERIES)]

                collected: List[Chunk] = []
                for q in queries:
                    collected.extend(
                        search_chunks(
                            q,
                            top_k=max(TOP_K_CHUNKS, 8),
                            pinned_pdf=pinned,
                            page_hint=page_hint,
                        )
                    )

                collected = dedupe_chunks_keep_order(collected)

                if RERANK_ENABLED and collected:
                    collected = await _rerank_candidates(
                        user_msg,
                        collected,
                        max_keep=max(RERANK_TOPK, 10),
                    )

                collected = diversify_chunks(
                    collected,
                    max_docs=max(1, BROAD_DOC_DIVERSITY_K),
                    per_doc=max(1, BROAD_TOPIC_HITS_K),
                )
                return collected[: max(RERANK_TOPK, 10)]

            # Precise/mixed: normal
            cands = search_chunks(
                user_msg,
                top_k=TOP_K_CHUNKS,
                pinned_pdf=pinned,
                page_hint=page_hint,
            )
            if RERANK_ENABLED and cands:
                cands = await _rerank_candidates(user_msg, cands, max_keep=RERANK_TOPK)
            return cands[:RERANK_TOPK]

        async def get_docai_hits() -> List[Tuple[str, str]]:
            if not pinned:
                return []
            if not _docai_chunk_files_for(pinned):
                return []
            return docai_search_text(pinned, user_msg, k=3 if precision == "broad" else 2)

        async def get_web_evidence() -> List[Dict[str, str]]:
            if not WEB_ENABLED:
                return []
            serp = await web_search_serper(user_msg, k=TOP_K_WEB)
            return await web_fetch_and_excerpt(user_msg, serp, max_items=TOP_K_WEB)

        # Only pull web when allowed:
        # - generally OK for general/planning/ber
        # - avoid web when evidence_mode AND numeric_needed (force doc quotes)
        do_web = WEB_ENABLED and (
            (not evidence_mode)
            or fam in ("planning", "general", "ber")
            or (evidence_mode and not numeric_needed)
        )

        pdf_chunks, docai_hits, web_pages = await asyncio.gather(
            get_pdf_chunks(),
            get_docai_hits(),
            get_web_evidence() if do_web else asyncio.sleep(0, result=[]),
        )

        # ----------------------------
        # Doc-family sanity: pinned doc mismatch
        # ----------------------------
        if pinned:
            pinned_type = _doc_type_from_pdfname(pinned)
            if fam != "general" and pinned_type not in ("general", fam):
                safe = (
                    "I think you’re asking about **"
                    + fam.replace("_", " ")
                    + "**, but the document I’m currently using looks like **"
                    + pinned_type.replace("_", " ")
                    + "**.\n\n"
                    "Tell me which document to use (or upload the relevant one) and I’ll answer directly from it with a quote."
                )
                if chat_id:
                    remember(chat_id, "assistant", safe)
                yield f"data: {safe.replace(chr(10), '\\\\n')}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

        # ----------------------------
        # Numeric compliance: must have doc evidence
        # ----------------------------
        if numeric_needed and not pdf_chunks and not docai_hits:
            refusal = (
                "I can’t confirm the exact numeric requirement yet because I don’t have relevant TGD evidence loaded for this question.\n\n"
                "Upload the correct TGD (or tell me which one: K, M, B, L), and I’ll quote the exact line containing the number."
            )
            if chat_id:
                remember(chat_id, "assistant", refusal)
            yield f"data: {refusal.replace(chr(10), '\\\\n')}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        sources_block = build_sources_block(
            pdf_chunks,
            web_pages,
            docai_hits,
            user_query=user_msg,
            precision=precision,
        )

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        system_prompt = SYSTEM_PROMPT_EVIDENCE if evidence_mode else SYSTEM_PROMPT_NORMAL
        model_name = MODEL_COMPLIANCE if evidence_mode else MODEL_CHAT

        model = get_model(model_name=model_name, system_prompt=system_prompt)
        contents = build_contents(history_for_prompt, user_msg, sources_block, precision=precision)

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
            safe_delta = delta.replace("\r", "").replace("\n", "\\n")
            yield f"data: {safe_delta}\n\n"

        draft = "".join(full).strip()

        # Enforce web citations (only if we actually fetched web pages)
        if do_web and web_pages:
            draft = _enforce_web_citations(draft, web_pages)

        # Hard verify numeric against evidence blob
        sources_blob = _sources_text_blob_for_verification(pdf_chunks, docai_hits, web_pages)
        ok, final_text = _hard_verify_numeric(draft, sources_blob)

        if not ok and final_text:
            upd = "\n\n---\n\n### ✅ Final (verified)\n\n" + final_text
            yield f"data: {upd.replace(chr(10), '\\\\n')}\n\n"
            draft = (draft + upd).strip()

        if chat_id:
            remember(chat_id, "assistant", draft)

        yield "event: done\ndata: ok\n\n"

    except Exception as e:
        msg = str(e).replace("\r", "").replace("\n", " ")
        yield f"event: error\ndata: {msg}\n\n"
        yield "event: done\ndata: ok\n\n"



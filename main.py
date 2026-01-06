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
from typing import List, Optional, Dict, Tuple, Any, Union
from collections import Counter
from datetime import datetime

import fitz  # PyMuPDF

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    GenerationConfig,
)

# Document AI ingest helper (optional file)
try:
    from docai_ingest import docai_extract_pdf_to_text
    _DOCAI_HELPER_AVAILABLE = True
except Exception:
    docai_extract_pdf_to_text = None
    _DOCAI_HELPER_AVAILABLE = False


# ----------------------------
# Setup
# ----------------------------

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "30"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "22000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Verification pass toggle (default ON)
VERIFY_NUMERIC = (os.getenv("VERIFY_NUMERIC", "true") or "true").strip().lower() in ("1", "true", "yes", "y", "on")

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

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

DOCAI_DIR = BASE_DIR / "parsed_docai"
DOCAI_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Vertex init
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


def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])


def get_generation_config(is_compliance: bool) -> GenerationConfig:
    if is_compliance:
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


def get_verify_generation_config() -> GenerationConfig:
    return GenerationConfig(
        temperature=0.1,
        top_p=0.6,
        max_output_tokens=450,
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


# ----------------------------
# Prompts (tone stays the same)
# ----------------------------

FORMAT_RULES = """
Formatting rules (strict):
- Do NOT use Markdown.
- Do NOT use bullet points, numbered lists, asterisks, or hyphen lists.
- Write in short, clear paragraphs only.
- Never mention page numbers.
- If citing TGDs, cite only: document + section number and/or table number.
- Never invent section numbers, clause numbers, or table numbers.
- Only cite a section/table if it is visible in the provided SOURCES text.
- Do not reproduce large chunks of the TGDs. If quoting, quote only 1–2 short lines.
""".strip()

SYSTEM_PROMPT_NORMAL = f"""
You are Raheem AI.

Tone:
Friendly, calm, confident. Professional.
Humour style: dry, light, and occasional — think “architect on coffee #2”, not stand-up comedy.
You may add a short witty line when it fits, but never spam jokes and never derail the answer.
No emojis unless the user uses emojis first.

Memory:
You have access to the conversation history for THIS chat. Use it.
Never claim “this is our first conversation” if there is prior chat history.

{FORMAT_RULES}

General mode rules:
Answer general questions naturally using common knowledge.
If the user asks about Irish building regulations or TGDs, be cautious and do not guess numeric limits unless you can ground them in SOURCES.
""".strip()

SYSTEM_PROMPT_COMPLIANCE = f"""
You are Raheem AI in Evidence Mode for Irish building regulations, TGDs, BER/DEAP, fire safety, and accessibility.

Tone:
Still calm and professional, with minimal humour.
If you add humour at all, it must be a single short line and only when the user is not asking for strict numeric compliance.
When giving compliance numbers, be straight to the point.

Memory:
You have access to the conversation history for THIS chat. Use it.

{FORMAT_RULES}

Compliance rules (strict):
- Use provided SOURCES text as your primary grounding.
- If SOURCES do not support an exact numeric requirement (distances, widths, U-values, rise/going, travel distances), do NOT guess.
- If you cannot confirm a number from SOURCES, say so plainly and ask the user to clarify building type / situation or upload the relevant TGD.
- Do not invent “optimum” values unless the TGD explicitly states them in the SOURCES.
- If the user challenges you, re-check SOURCES and correct yourself.

Proof rule (critical):
- If you give a numeric requirement from a TGD, you MUST include a short verbatim quote (1–2 lines) from the SOURCES text you used.
- If you cannot quote it from SOURCES, do not give the number.
- Never mention page numbers in the quote.
""".strip()

VERIFY_PROMPT = f"""
You are a strict verifier for Irish TGD compliance answers.

Rules:
- You ONLY use the SOURCES text provided.
- If an answer includes any numeric requirement (mm, m, %, W/m²K, etc.), the exact number with its unit must be explicitly present in SOURCES.
- If the answer cites a section or table, the cited section/table label must be visible in SOURCES. If not visible, it must be removed or the answer must say it cannot confirm.
- If the answer gives numbers but does not include a 1–2 line verbatim quote from SOURCES that contains the number, it is NOT allowed.
- Output MUST be valid JSON only. No markdown.

Output JSON schema:
{{
  "ok": true/false,
  "reason": "short reason",
  "safe_answer": "a corrected safe answer that follows the formatting rules, includes quote: if numeric"
}}

{FORMAT_RULES}
""".strip()



def build_gemini_contents(history: List[Dict[str, str]], user_message: str, sources_text: str) -> List[Content]:
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
    if sources_text and sources_text.strip():
        final_user = (final_user + "\n\nSOURCES (use for evidence):\n" + sources_text).strip()

    if not contents:
        contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    else:
        if contents[-1].role == "user":
            contents[-1] = Content(role="user", parts=[Part.from_text(final_user)])
        else:
            contents.append(Content(role="user", parts=[Part.from_text(final_user)]))

    return contents


# ----------------------------
# Retrieval helpers
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
DOCAI_INDEX: Dict[str, Dict[str, Any]] = {}


def list_pdfs() -> List[str]:
    files = []
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p.name)
    files.sort()
    return files


def auto_pin_pdf(question: str) -> Optional[str]:
    q = (question or "").lower()
    pdfs = set(list_pdfs())

    def pick_any(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in pdfs:
                return c
        return None

    if any(w in q for w in ["stairs", "stair", "staircase", "riser", "rise", "going", "pitch", "headroom", "handrail", "balustrade", "landing"]):
        return pick_any([
            "Technical Guidance Document K.pdf",
            "TGD K.pdf",
            "Technical Guidance Document K (2020).pdf",
            "Technical Guidance Document K.pdf".replace("  ", " ")
        ])

    if any(w in q for w in ["fire", "escape", "means of escape", "travel distance", "sprinkler", "smoke", "compartment", "car park", "corridor", "protected route"]):
        return pick_any([
            "Technical Guidance Document B Non Dwellings.pdf",
            "Technical Guidance Document B Dwellings.pdf",
        ])

    if any(w in q for w in ["u-value", "u value", "y-value", "thermal", "insulation", "ber", "deap", "energy", "renovation", "major renovation"]):
        return pick_any([
            "Technical Guidance Document L Dwellings.pdf",
            "Technical Guidance Document L Non Dwellings.pdf",
        ])

    if any(w in q for w in ["access", "accessible", "wheelchair", "ramp", "dac", "part m"]):
        return pick_any([
            "Technical Guidance Document M.pdf",
        ])

    return None


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


def ensure_docai_indexed(pdf_name: str) -> None:
    if pdf_name in DOCAI_INDEX:
        return
    chunk_files = _docai_chunk_files_for(pdf_name)
    if not chunk_files:
        return

    chunks = []
    df = Counter()

    for p in chunk_files:
        rng = _parse_range_from_chunk_filename(p)
        if not rng:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        toks = tokenize(txt.lower())
        tf = Counter(toks)
        df.update(set(tf.keys()))

        chunks.append({"range": rng, "text": txt, "tf": tf, "len": len(toks)})

    if not chunks:
        return

    avgdl = sum(c["len"] for c in chunks) / max(1, len(chunks))
    DOCAI_INDEX[pdf_name] = {"chunks": chunks, "df": df, "avgdl": avgdl, "N": len(chunks)}


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


def search_docai_chunks(pdf_name: str, question: str, k: int = 3) -> List[Dict[str, Any]]:
    ensure_docai_indexed(pdf_name)
    idx = DOCAI_INDEX.get(pdf_name)
    if not idx:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    df = idx["df"]
    N = idx["N"]
    avgdl = idx["avgdl"]

    scored = []
    for i, ch in enumerate(idx["chunks"]):
        s = bm25_score(qt, ch["tf"], df, N, ch["len"], avgdl)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    for i, _ in scored[:k]:
        out.append({"range": idx["chunks"][i]["range"], "text": idx["chunks"][i]["text"]})
    return out


def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        page_tf: List[Counter] = []
        df = Counter()
        page_len: List[int] = []

        for i in range(doc.page_count):
            txt = clean_text(doc.load_page(i).get_text("text") or "")
            toks = tokenize(txt.lower())
            tf = Counter(toks)
            page_tf.append(tf)
            df.update(set(tf.keys()))
            page_len.append(len(toks))

        avgdl = (sum(page_len) / len(page_len)) if page_len else 0.0

        PDF_INDEX[name] = {
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


def extract_pages_text(pdf_path: Path, pages: List[int], max_chars_per_page: int = 2200) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for p in pages:
            if p < 0 or p >= doc.page_count:
                continue
            txt = clean_text(doc.load_page(p).get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + "..."
            chunks.append(f"[{pdf_path.name} excerpt]\n{txt}")
        return "\n\n".join(chunks).strip()
    finally:
        doc.close()


def build_sources_bundle(selected: List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]) -> str:
    pieces = []
    for pdf_name, items, mode in selected:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            continue

        if mode == "docai":
            ensure_docai_indexed(pdf_name)
            idx = DOCAI_INDEX.get(pdf_name)
            if not idx:
                continue

            for rng in items:  # type: ignore
                a, b = rng
                txt = ""
                for ch in idx["chunks"]:
                    if ch["range"] == (a, b):
                        txt = ch["text"]
                        break
                if txt:
                    clipped = txt.strip()
                    if len(clipped) > 6500:
                        clipped = clipped[:6500] + "..."
                    pieces.append(f"[{pdf_name} excerpt]\n{clipped}")
        else:
            pages = items  # type: ignore
            pieces.append(extract_pages_text(pdf_path, pages))

    return ("\n\n".join([p for p in pieces if p]).strip())


def select_sources(
    question: str,
    pdf_pin: Optional[str] = None,
    max_docs: int = 3,
    pages_per_doc: int = 3,
    docai_chunks_per_doc: int = 2,
) -> List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    chosen: List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]] = []
    know_docai = {name for name in pdfs if _docai_chunk_files_for(name)}

    def pick_for_pdf(pdf_name: str) -> Optional[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]:
        if pdf_name in know_docai:
            chunks = search_docai_chunks(pdf_name, question, k=docai_chunks_per_doc)
            if chunks:
                ranges = [c["range"] for c in chunks]
                return (pdf_name, ranges, "docai")

        pages = search_pages(pdf_name, question, k=pages_per_doc)
        if pages:
            return (pdf_name, pages, "pymupdf")
        return None

    if pdf_pin and pdf_pin in pdfs:
        picked = pick_for_pdf(pdf_pin)
        return [picked] if picked else []

    picks = []
    for pdf_name in pdfs:
        picked = pick_for_pdf(pdf_name)
        if picked:
            score = 10.0 if picked[2] == "docai" else 5.0
            picks.append((score, picked))
    picks.sort(key=lambda x: x[0], reverse=True)

    for _, picked in picks[:max_docs]:
        chosen.append(picked)

    return chosen


# ----------------------------
# Compliance intent detection
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

NUMERIC_COMPLIANCE_TRIGGERS = [
    "minimum", "maximum", "max", "min", "limit", "limits",
    "distance", "width", "widths", "lengths", "length", "height", "u-value", "y-value",
    "rise", "riser", "going", "pitch",
    "stairs", "stair", "staircase", "landing", "headroom",
    "dimension", "dimensions",
    "escape", "travel distance"
]


def is_numeric_compliance_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in NUMERIC_COMPLIANCE_TRIGGERS)


def should_use_docs(q: str, force_docs: bool = False) -> bool:
    if force_docs:
        return True
    ql = (q or "").lower()
    if is_numeric_compliance_question(ql):
        return True
    if PART_PATTERN.search(ql):
        return True
    if TECH_PATTERN.search(ql):
        return True
    return any(k in ql for k in COMPLIANCE_KEYWORDS)


# ----------------------------
# Output sanitizers
# ----------------------------

PAGE_MENTION_RE = re.compile(r"\bpage\s*\d+\b", re.I)
BULLET_RE = re.compile(r"(^|\n)\s*(\*|-|•|\d+\.)\s+", re.M)
REF_RE = re.compile(r"\b(Section|Table|Clause|Figure|Diagram)\s*[A-Za-z0-9][A-Za-z0-9\.\-]*\b", re.I)

ANY_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
NUM_WITH_UNIT_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|m|metre|meter|w/m²k|w\/m2k|%)\b", re.I)


def strip_bullets_streaming(delta: str) -> str:
    return BULLET_RE.sub(lambda m: m.group(1), delta)


def _sources_contain_any_number(sources_text: str) -> bool:
    return bool(ANY_NUMBER_RE.search(sources_text or ""))


def _looks_like_numeric_answer(text: str) -> bool:
    return bool(NUM_WITH_UNIT_RE.search(text or ""))


def postprocess_final_answer(final_text: str, sources_text: str, compliance: bool) -> str:
    t = final_text or ""
    t = PAGE_MENTION_RE.sub("", t)
    t = BULLET_RE.sub(lambda m: m.group(1), t)

    if compliance:
        src = (sources_text or "").lower()
        if src.strip():
            if ("section" not in src) and ("table" not in src) and ("clause" not in src) and ("diagram" not in src) and ("figure" not in src):
                if REF_RE.search(t):
                    t = REF_RE.sub("", t).strip()
                    t = (t + "\n\nI can’t confidently point to the exact section or table from the extracted text I have right now. If you re-upload a clearer extract, I’ll re-check and cite it properly.").strip()
        else:
            t = REF_RE.sub("", t).strip()

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


# ----------------------------
# Verification (silent double-check)
# ----------------------------

def _run_verifier(user_msg: str, draft_answer: str, sources_text: str) -> Tuple[bool, str]:
    """
    Returns (ok, safe_answer). If verifier fails, safe_answer is a corrected safe output.
    """
    if not VERIFY_NUMERIC:
        return True, draft_answer

    # Only verify when we have sources
    if not (sources_text and sources_text.strip()):
        return True, draft_answer

    # Only verify when answer contains numbers OR question is numeric compliance
    needs = is_numeric_compliance_question(user_msg) or _looks_like_numeric_answer(draft_answer) or bool(ANY_NUMBER_RE.search(draft_answer or ""))
    if not needs:
        return True, draft_answer

    try:
        ensure_vertex_ready()
        if not _VERTEX_READY:
            return True, draft_answer

        verifier = get_model(MODEL_COMPLIANCE, VERIFY_PROMPT)

        verify_input = (
            "USER QUESTION:\n" + (user_msg or "").strip() + "\n\n"
            "DRAFT ANSWER:\n" + (draft_answer or "").strip() + "\n\n"
            "SOURCES:\n" + (sources_text or "").strip()
        )

        resp = verifier.generate_content(
            [Content(role="user", parts=[Part.from_text(verify_input)])],
            generation_config=get_verify_generation_config(),
            stream=False
        )

        raw = (getattr(resp, "text", "") or "").strip()

        # Try to extract JSON object from any surrounding text (just in case)
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            raw = m.group(0)

        data = json.loads(raw)
        ok = bool(data.get("ok", False))
        safe_answer = (data.get("safe_answer") or "").strip()

        if ok and safe_answer:
            return True, safe_answer

        # If not ok, enforce a safe fallback message
        if safe_answer:
            return False, safe_answer

        fallback = (
            "I can’t safely confirm that numeric requirement from the extracted TGD text I have available right now.\n\n"
            "If you upload a clearer copy or tell me the exact table or section you want checked, I’ll answer directly from the text and include a short quote."
        )
        return False, fallback

    except Exception:
        # If verifier fails for any reason, don't break the app; just return draft.
        return True, draft_answer


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
        "docai_helper": _DOCAI_HELPER_AVAILABLE,
        "docai_processor_id_present": bool((os.getenv("DOCAI_PROCESSOR_ID") or "").strip()),
        "docai_location_present": bool((os.getenv("DOCAI_LOCATION") or "").strip()),
        "verify_numeric": VERIFY_NUMERIC,
    }


@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}


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

    PDF_INDEX.pop(dest.name, None)
    DOCAI_INDEX.pop(dest.name, None)

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
        "docai": {
            "attempted": bool(_DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text),
            "ok": docai_ok,
            "chunks_saved": docai_chunks_saved,
            "error": docai_error,
        }
    }


# ----------------------------
# Streaming core
# ----------------------------

def _stream_answer(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None
):
    try:
        if not message.strip():
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()
        user_msg = (message or "").strip()

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

        # Decide compliance intent
        use_docs_intent = should_use_docs(user_msg, force_docs=force_docs)
        numeric_compliance = use_docs_intent and is_numeric_compliance_question(user_msg)

        sources_text = ""
        if use_docs_intent and list_pdfs():
            auto_pdf = pdf or auto_pin_pdf(user_msg)
            selected = select_sources(
                user_msg,
                pdf_pin=auto_pdf,   # IMPORTANT: use auto-pinned PDF
                max_docs=3,
                pages_per_doc=3,
                docai_chunks_per_doc=2
            )
            sources_text = build_sources_bundle(selected)

        used_docs_flag = bool(sources_text and sources_text.strip())
        is_compliance = use_docs_intent

        model_name = MODEL_COMPLIANCE if is_compliance else MODEL_CHAT
        system_prompt = SYSTEM_PROMPT_COMPLIANCE if is_compliance else SYSTEM_PROMPT_NORMAL

        # HARD RULE: numeric compliance must be grounded in extracted text that contains numbers
        if numeric_compliance:
            if (not used_docs_flag) or (not _sources_contain_any_number(sources_text)):
                refusal = (
                    "I can’t confirm the exact numeric limit from the extracted TGD text I have available right now.\n\n"
                    "If you upload a clearer copy or tell me the exact clause or table you want checked, I’ll answer directly from the text and include a short quote."
                )
                if chat_id:
                    remember(chat_id, "assistant", refusal)
                yield f"data: {refusal}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        model = get_model(model_name=model_name, system_prompt=system_prompt)
        contents = build_gemini_contents(history_for_prompt, user_msg, sources_text)

        stream = model.generate_content(
            contents,
            generation_config=get_generation_config(is_compliance),
            stream=True
        )

        full = []
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if not delta:
                continue
            delta_clean = strip_bullets_streaming(delta)
            full.append(delta_clean)
            safe = delta_clean.replace("\r", "").replace("\n", "\\n")
            yield f"data: {safe}\n\n"

        final_text = "".join(full).strip()
        final_text = postprocess_final_answer(final_text, sources_text, compliance=is_compliance)

        # FINAL HARD CHECK: numeric compliance must include a short quote from SOURCES if it outputs numbers
        if numeric_compliance:
            has_numeric_output = _looks_like_numeric_answer(final_text) or bool(ANY_NUMBER_RE.search(final_text or ""))
            has_quote_marker = ("quote:" in (final_text or "").lower())
            if has_numeric_output and not has_quote_marker:
                final_text = (
                    "I can’t safely give you that numeric requirement yet because I don’t have a reliable quoted line from the extracted TGD text to prove it.\n\n"
                    "If you re-upload the PDF (so it re-parses) or tell me the exact clause or table, I’ll answer and include a short quote."
                )

        # Silent verification pass (prevents wrong table reads / fake section refs)
        if is_compliance and used_docs_flag:
            ok, verified = _run_verifier(user_msg, final_text, sources_text)
            if verified and verified.strip():
                final_text = postprocess_final_answer(verified.strip(), sources_text, compliance=is_compliance)

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
    return StreamingResponse(
        _stream_answer(chat_id.strip(), q, force_docs, pdf, page_hint),
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

    return StreamingResponse(
        _stream_answer(chat_id, message, force_docs, pdf, page_hint, messages=messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )






# main.py — Raheem AI (Persistent PDFs + Smart Vision + Streaming)

from fastapi import FastAPI, UploadFile, File, Query
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
from typing import List, Optional, Dict, Tuple

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(docs_url="/swagger", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# PDF storage (Render disk if available, otherwise fallback)
# ----------------------------

def choose_pdf_dir() -> Path:
    preferred = Path("/var/data/pdfs")      # persistent if Render Disk mounted at /var/data
    fallback = BASE_DIR / "pdfs"            # always works, not persistent on Render

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
# In-memory page text index
# ----------------------------

PDF_TEXT_INDEX: Dict[str, List[str]] = {}

def _clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    page_texts: List[str] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            page_texts.append(_clean_text(txt).lower())
    finally:
        doc.close()
    PDF_TEXT_INDEX[name] = page_texts

def index_all_pdfs() -> None:
    for p in PDF_DIR.glob("*.pdf"):
        try:
            index_pdf(p)
        except Exception:
            continue

index_all_pdfs()

# ----------------------------
# Root + Health + Docs
# ----------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Raheem AI API",
        "pdf_dir": str(PDF_DIR),
        "indexed_pdfs": len(PDF_TEXT_INDEX),
        "model": MODEL_NAME,
    }

@app.get("/health")
def health():
    return {"ok": True, "status": "Raheem AI backend running"}

@app.get("/docs")
def docs_compat():
    return {"ok": True}

# ----------------------------
# Page selection (auto)
# ----------------------------

STOPWORDS = {
    "the","and","or","of","to","in","a","an","for","on","with","is","are","be","as","at",
    "from","by","that","this","it","your","you","we","they","their","there","what","which",
    "when","where","how","can","shall","should","must","may","not","than","then"
}

def tokenize(q: str) -> List[str]:
    q = q.lower()
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

def auto_select_pages(
    pdf_name: str,
    question: str,
    top_k: int = 4,
    window: int = 1
) -> List[int]:
    page_texts = PDF_TEXT_INDEX.get(pdf_name)
    if not page_texts:
        pdf_path = PDF_DIR / pdf_name
        if pdf_path.exists():
            index_pdf(pdf_path)
            page_texts = PDF_TEXT_INDEX.get(pdf_name, [])
        else:
            return list(range(0, 6))

    tokens = tokenize(question)
    if not tokens:
        return list(range(min(6, len(page_texts))))

    scored: List[Tuple[int, int]] = []
    for i, txt in enumerate(page_texts):
        s = score_page(txt, tokens, question)
        if s > 0:
            scored.append((i, s))

    if not scored:
        return list(range(min(6, len(page_texts))))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_pages = [i for i, _ in scored[:top_k]]

    selected = set()
    total = len(page_texts)
    for p in top_pages:
        for n in range(-window, window + 1):
            idx = p + n
            if 0 <= idx < total:
                selected.add(idx)

    return sorted(selected)

# ----------------------------
# TEXT extraction (cheap mode)
# ----------------------------

def extract_pages_text(pdf_path: Path, page_indexes: List[int], max_chars_per_page: int = 4500) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for idx in page_indexes:
            page = doc.load_page(idx)
            txt = page.get_text("text") or ""
            txt = _clean_text(txt)
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + " …"
            chunks.append(f"[Page {idx+1}]\n{txt}")
        return "\n\n".join(chunks)
    finally:
        doc.close()

# ----------------------------
# VISION images (use sparingly)
# ----------------------------

def pdf_pages_to_images(pdf_path: Path, page_indexes: List[int], dpi: int = 120):
    images = []
    doc = fitz.open(pdf_path)
    try:
        for idx in page_indexes:
            page = doc.load_page(idx)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_b64}",
            })
    finally:
        doc.close()
    return images

# ----------------------------
# Smart Vision decision
# ----------------------------

def needs_vision_from_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "table","tab.","diagram","figure","fig.","drawing","detail",
        "chart","graph","schedule","matrix","appendix",
        "caption","footnote","note under",
        "read the","see the","what does this show","interpret",
        "plan","elevation","section"
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

    if long_space_lines > 10 or pipe_lines > 3 or dot_leader > 4:
        return True
    if very_short > (len(lines) * 0.25):
        return True
    if len(excerpt) < 900:
        return True

    return False

def requires_exact_numeric_answer(q: str) -> bool:
    ql = q.lower()
    numeric_words = [
        "mm","m","metre","meter","distance","width","height",
        "slope","gradient","ratio","minutes","min","max",
        "minimum","maximum","capacity","people","occupancy",
        "travel distance"
    ]
    return any(w in ql for w in numeric_words)

# ----------------------------
# Upload PDF
# ----------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files allowed"}

    path = PDF_DIR / file.filename
    content = await file.read()
    path.write_bytes(content)

    try:
        index_pdf(path)
    except Exception:
        pass

    return {"ok": True, "pdf": file.filename, "stored_in": str(PDF_DIR)}

# ----------------------------
# Prompt (no symbols, no markdown)
# ----------------------------

SYSTEM_RULES = """
You are Raheem AI, a specialist assistant for Irish Building Regulations Technical Guidance Documents (TGDs).

Communication style (mandatory):
- Do NOT use markdown of any kind.
- Do NOT use bullet points, numbered lists, asterisks, hashes, headings, or symbols.
- Write in natural, flowing sentences only.
- Respond as if explaining verbally to a professional colleague.
- Sound calm, confident, and approachable.
- Avoid robotic or overly formal language.
- Do not say things like thinking or analyzing, and do not describe internal processes.

Answer rules:
- Prioritise correctness. Do not invent numbers, limits, or clauses.
- Where possible, mention the relevant section, table, or diagram name in sentence form.
- If an exact value is not clearly visible in the provided material, say so plainly and explain what can be verified instead.
- Keep answers clear, helpful, and naturally structured in short paragraphs.
"""

def build_user_header(question: str, pdf_name: str, pages_used: List[int], mode: str) -> str:
    shown = [p + 1 for p in pages_used]
    return (
        f"PDF filename: {pdf_name}\n"
        f"Pages provided (1-based): {shown}\n"
        f"Mode: {mode}\n\n"
        f"User question:\n{question}\n"
    )

def build_openai_input(
    question: str,
    pdf_name: str,
    pdf_path: Path,
    page_indexes: List[int],
    dpi: int,
    force_vision: bool
):
    excerpt = extract_pages_text(pdf_path, page_indexes, max_chars_per_page=4500)

    q_triggers_vision = needs_vision_from_question(question)
    excerpt_suggests_vision = needs_vision_from_text_excerpt(excerpt)
    wants_numbers = requires_exact_numeric_answer(question)

    use_vision = force_vision or q_triggers_vision or (wants_numbers and excerpt_suggests_vision)

    mode = "vision" if use_vision else "text"
    header = build_user_header(question, pdf_name, page_indexes, mode)

    user_blocks = [
        {"type": "input_text", "text": SYSTEM_RULES},
        {"type": "input_text", "text": header + "\n\n---\n\nEXCERPT:\n" + excerpt}
    ]

    if use_vision:
        MAX_IMAGE_PAGES = 3
        vision_pages = page_indexes[:MAX_IMAGE_PAGES]
        user_blocks.extend(pdf_pages_to_images(pdf_path, vision_pages, dpi=dpi))

    return user_blocks, use_vision

def resolve_page_indexes(pdf_name: str, question: str, pages: Optional[str], top_k: int, window: int, pdf_path: Path):
    page_indexes: Optional[List[int]] = None
    if pages:
        try:
            page_list = [int(x.strip()) for x in pages.split(",") if x.strip()]
            doc = fitz.open(pdf_path)
            try:
                total = doc.page_count
            finally:
                doc.close()

            idxs = []
            for p in page_list:
                idx = p - 1
                if 0 <= idx < total:
                    idxs.append(idx)

            seen = set()
            page_indexes = [x for x in idxs if not (x in seen or seen.add(x))]
        except Exception:
            raise ValueError("Invalid pages format. Use e.g. pages=1,2,340")

    if page_indexes is None:
        page_indexes = auto_select_pages(pdf_name, question, top_k=top_k, window=window)

    return page_indexes

# ----------------------------
# Non-stream endpoint
# ----------------------------

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    pdf: str = Query(..., description="PDF filename"),
    pages: Optional[str] = Query(None, description="Optional: comma-separated 1-based pages e.g. 340,341"),
    top_k: int = Query(4, description="Auto: number of strongest pages"),
    window: int = Query(1, description="Auto: neighbour pages around each match"),
    dpi: int = Query(120, description="Render DPI for vision pages (lower = cheaper/faster)"),
    vision: bool = Query(False, description="Force vision pages"),
):
    pdf_path = PDF_DIR / pdf
    if not pdf_path.exists():
        return {"ok": False, "error": f"PDF not found in {PDF_DIR}"}

    try:
        page_indexes = resolve_page_indexes(pdf, q, pages, top_k, window, pdf_path)

        user_blocks, vision_used = build_openai_input(
            question=q,
            pdf_name=pdf,
            pdf_path=pdf_path,
            page_indexes=page_indexes,
            dpi=dpi,
            force_vision=vision
        )

        resp = client.responses.create(
            model=MODEL_NAME,
            max_output_tokens=700,
            input=[{"role": "user", "content": user_blocks}],
        )

        return {
            "ok": True,
            "answer": resp.output_text,
            "pages_used": [p + 1 for p in page_indexes],
            "vision_used": vision_used
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-12:]}

# ----------------------------
# Streaming endpoint (typing UI)
# ----------------------------

@app.get("/ask_stream")
def ask_stream(
    q: str = Query(..., description="User question"),
    pdf: str = Query(..., description="PDF filename"),
    pages: Optional[str] = Query(None, description="Optional: comma-separated 1-based pages e.g. 340,341"),
    top_k: int = Query(4),
    window: int = Query(1),
    dpi: int = Query(120),
    vision: bool = Query(False),
):
    pdf_path = PDF_DIR / pdf
    if not pdf_path.exists():
        def err():
            yield "event: error\ndata: PDF not found\n\n"
        return StreamingResponse(
            err(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    try:
        page_indexes = resolve_page_indexes(pdf, q, pages, top_k, window, pdf_path)
        user_blocks, vision_used = build_openai_input(
            question=q,
            pdf_name=pdf,
            pdf_path=pdf_path,
            page_indexes=page_indexes,
            dpi=dpi,
            force_vision=vision
        )
    except Exception as e:
        def err():
            yield f"event: error\ndata: {str(e)}\n\n"
        return StreamingResponse(
            err(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    def sse():
        yield f"event: meta\ndata: pages_used={','.join(str(p+1) for p in page_indexes)};vision_used={vision_used}\n\n"

        stream = client.responses.create(
            model=MODEL_NAME,
            max_output_tokens=700,
            input=[{"role": "user", "content": user_blocks}],
            stream=True,
        )

        try:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        safe = delta.replace("\r", "").replace("\n", "\\n")
                        yield f"data: {safe}\n\n"

                if getattr(event, "type", None) == "response.completed":
                    break

        except Exception as e:
            msg = str(e).replace("\r", "").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"

        yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )

# ----------------------------
# List PDFs
# ----------------------------

@app.get("/pdfs")
def list_pdfs():
    pdfs = [p.name for p in PDF_DIR.glob("*.pdf")]
    return {"count": len(pdfs), "pdfs": pdfs}

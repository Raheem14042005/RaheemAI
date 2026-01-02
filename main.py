# main.py — Raheem AI (PDF-first, multimodal + auto page selection)
#
# What this fixes:
# - Automatically selects the most relevant PDF pages based on the question
#   (so it can answer things from deep pages e.g. 340 etc.)
# - Still supports manual pages override: ?pages=340,341
# - Keeps frontend compatibility endpoints: /, /health, /docs, /pdfs, /ask
#
# How it works:
# - On startup (and on upload), builds an in-memory text index for ALL pages of each PDF.
# - For each question, scores every page text vs the question, selects top pages,
#   adds neighbour pages (window), and renders only those pages as images for GPT-4o.
#
# CRITICAL FIX:
# - Do NOT include "metadata" inside input_image objects for OpenAI Responses API.
#   It causes: Unknown parameter '...metadata'

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

import os
import re
import base64
from pathlib import Path
import fitz  # PyMuPDF
import traceback
from typing import List, Optional, Dict, Tuple

# ----------------------------
# Setup
# ----------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(docs_url="/swagger", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

# ----------------------------
# In-memory page text index
# ----------------------------

# pdf_name -> list[str] (page_texts, index 0 = page 1)
PDF_TEXT_INDEX: Dict[str, List[str]] = {}

def _clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def index_pdf(pdf_path: Path) -> None:
    """Extract text from all pages into memory for fast page selection."""
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
            # Don't crash startup if one PDF is odd/corrupt
            continue

# Build index at startup
index_all_pdfs()

# ----------------------------
# Root + Health + Docs (frontend compatibility)
# ----------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Raheem AI API",
        "pdf_dir": str(PDF_DIR),
        "indexed_pdfs": len(PDF_TEXT_INDEX),
    }

@app.get("/health")
def health():
    return {"ok": True, "status": "Raheem AI backend running"}

# frontend hard-checks this path in your UI
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
    # keep numbers/units useful for regs queries
    tokens = re.findall(r"[a-z0-9][a-z0-9\-/\.]*", q)
    out = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out

def score_page(page_text: str, tokens: List[str], full_q: str) -> int:
    """
    Simple scoring:
    - token frequency hits
    - bonus for exact phrase fragments (2-4 words)
    """
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
    top_k: int = 8,
    window: int = 1,
) -> List[int]:
    """
    Returns 0-based page indexes.
    Picks top_k pages by score across ALL pages, then adds +/- window neighbours.
    """
    page_texts = PDF_TEXT_INDEX.get(pdf_name)
    if not page_texts:
        pdf_path = PDF_DIR / pdf_name
        if pdf_path.exists():
            index_pdf(pdf_path)
            page_texts = PDF_TEXT_INDEX.get(pdf_name, [])
        else:
            return list(range(0, 8))

    tokens = tokenize(question)
    if not tokens:
        return list(range(min(8, len(page_texts))))

    scored: List[Tuple[int, int]] = []
    for i, txt in enumerate(page_texts):
        s = score_page(txt, tokens, question)
        if s > 0:
            scored.append((i, s))

    if not scored:
        return list(range(min(10, len(page_texts))))

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
# PDF -> images (vision input)
# ----------------------------

def pdf_pages_to_images(
    pdf_path: Path,
    page_indexes: List[int],
    dpi: int = 180,
):
    """
    Render specified pages (0-based) into data-url images for GPT-4o.
    IMPORTANT: DO NOT include 'metadata' inside input_image objects (OpenAI rejects it).
    """
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
# Upload PDF (optional)
# ----------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files allowed"}

    path = PDF_DIR / file.filename
    content = await file.read()
    path.write_bytes(content)

    # index immediately so /ask can find deep pages
    try:
        index_pdf(path)
    except Exception:
        pass

    return {"ok": True, "pdf": file.filename}

# ----------------------------
# Prompt (balanced: accurate, but doesn't "die" on uncertainty)
# ----------------------------

SYSTEM_RULES = """
You are Raheem AI, a specialist assistant for Irish Building Regulations Technical Guidance Documents (TGDs).

Rules:
- Prioritise correctness. Do not invent numbers or clauses.
- Prefer citing the relevant SECTION / TABLE / DIAGRAM name if visible, rather than page numbers.
- If the exact number is not clearly visible in the provided pages, do NOT guess:
  - explain what you can confidently see,
  - suggest what section/table/diagram to check next,
  - and ask the user to widen the search (or allow the system to widen pages).
- Still be helpful: give best-practice guidance while you explain what you could not confirm.
"""

def build_user_prompt(question: str, pdf_name: str, pages_used: List[int]) -> str:
    shown = [p + 1 for p in pages_used]  # 1-based for readability
    return (
        f"You are answering using ONLY the PDF pages shown as images.\n"
        f"PDF filename: {pdf_name}\n"
        f"Pages shown (1-based): {shown}\n\n"
        f"User question:\n{question}\n"
    )

# ----------------------------
# Ask PDF (core logic)
# ----------------------------

@app.post("/ask-pdf")
def ask_pdf(
    question: str,
    pdf_name: str,
    pages: Optional[str] = None,          # manual override e.g. "340,341"
    top_k: int = 8,                       # how many strong pages to fetch
    window: int = 1,                      # neighbours around each strong page
    dpi: int = 180,                       # lower dpi = faster
):
    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        return {"ok": False, "error": f"PDF not found in {PDF_DIR}"}

    # Manual page override
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
            return {"ok": False, "error": "Invalid pages format. Use e.g. pages=1,2,340"}

    try:
        if page_indexes is None:
            page_indexes = auto_select_pages(pdf_name, question, top_k=top_k, window=window)

        images = pdf_pages_to_images(pdf_path, page_indexes, dpi=dpi)

        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_RULES}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_user_prompt(question, pdf_name, page_indexes)},
                        *images,
                    ],
                },
            ],
        )

        return {"ok": True, "answer": response.output_text, "pages_used": [p + 1 for p in page_indexes]}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc().splitlines()[-12:],
        }

# ----------------------------
# Frontend-compatible ASK endpoint
# ----------------------------

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    pdf: str = Query(..., description="PDF filename"),
    pages: Optional[str] = Query(None, description="Optional: comma-separated 1-based page numbers e.g. 340,341"),
    top_k: int = Query(8, description="Auto mode: number of strongest-matching pages to include"),
    window: int = Query(1, description="Auto mode: neighbour pages around each match"),
    dpi: int = Query(180, description="Render DPI (lower=faster)"),
):
    return ask_pdf(question=q, pdf_name=pdf, pages=pages, top_k=top_k, window=window, dpi=dpi)

# ----------------------------
# List PDFs (frontend counter uses this)
# ----------------------------

@app.get("/pdfs")
def list_pdfs():
    pdfs = [p.name for p in PDF_DIR.glob("*.pdf")]
    return {"count": len(pdfs), "pdfs": pdfs}

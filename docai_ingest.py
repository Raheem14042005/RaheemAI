import os
from typing import List, Tuple

from google.cloud import documentai_v1 as documentai
from pypdf import PdfReader, PdfWriter


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _docai_client() -> documentai.DocumentProcessorServiceClient:
    # Uses GOOGLE_CREDENTIALS_JSON already set in your app
    return documentai.DocumentProcessorServiceClient()


def _processor_name(client: documentai.DocumentProcessorServiceClient) -> str:
    project_id = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
    if not project_id:
        raise RuntimeError("Missing GCP_PROJECT_ID (or VERTEX_PROJECT_ID fallback)")

    location = _require_env("DOCAI_LOCATION")
    processor_id = _require_env("DOCAI_PROCESSOR_ID")
    return client.processor_path(project_id, location, processor_id)


def _slice_pdf_bytes(pdf_path: str, start_page_0: int, end_page_0_exclusive: int) -> bytes:
    """
    Creates a small PDF containing pages [start, end).
    Page indexes are 0-based.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for i in range(start_page_0, min(end_page_0_exclusive, len(reader.pages))):
        writer.add_page(reader.pages[i])

    out = bytearray()
    writer.write(out)
    return bytes(out)


def _process_chunk(client: documentai.DocumentProcessorServiceClient, name: str, pdf_bytes: bytes) -> str:
    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)

    # Document AI returns all extracted text in result.document.text
    # We'll return it as-is; your retrieval will handle search/citations.
    return result.document.text or ""


def docai_extract_pdf_to_text(
    pdf_path: str,
    chunk_pages: int = 15,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Extracts text from a PDF via Document AI Layout Parser in chunks.
    Returns:
      - combined_text
      - list of (start_page_1_based, end_page_1_based) for each chunk
    """
    client = _docai_client()
    name = _processor_name(client)

    # Count pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    chunks: List[Tuple[int, int]] = []
    texts: List[str] = []

    # Process in chunk_pages blocks
    start = 0
    while start < total_pages:
        end = min(start + chunk_pages, total_pages)

        pdf_bytes = _slice_pdf_bytes(pdf_path, start, end)
        txt = _process_chunk(client, name, pdf_bytes).strip()

        # Store chunk with 1-based page range
        chunks.append((start + 1, end))
        texts.append(f"\n\n--- DOC_AI_PAGES {start+1}-{end} ---\n\n{txt}")

        start = end

    combined = "\n".join(texts).strip()
    return combined, chunks

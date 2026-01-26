# Architecture Audit: Complyr Baseline for Raheem AI

This document outlines the architecture of the `Complyr` codebase, its end-to-end data flow, associated risks, and a proposed plan for refactoring it into the Raheem AI platform.

---

## 1. Component Interaction

The system is a classic RAG (Retrieval-Augmented Generation) application composed of three main parts: an ingestion script, a backend server, and a frontend interface.

-   **`docai_ingest.py` (Ingestion Processor)**: This is a standalone Python script responsible for processing PDF documents using Google Cloud's Document AI service. It's not part of the runtime server. Its role is to take a PDF, split it into manageable page chunks, send each chunk to Document AI for high-fidelity text extraction, and return the combined text. This script is likely intended for offline, high-quality processing of key documents where standard text extraction might fail on complex layouts (tables, columns).

-   **`main.py` (Backend Server)**: This is the core of the application, built with FastAPI. It serves the frontend and provides the main API for the RAG pipeline. Its responsibilities include:
    -   **Serving the Frontend**: Serves the `site/index.html` file.
    -   **PDF Indexing**: It has its own, faster PDF processing pipeline using `PyMuPDF` for standard text extraction, chunking, and indexing (both keyword-based BM25 and optional vector embeddings).
    -   **Hybrid Retrieval**: Combines keyword and semantic vector search to find relevant text chunks from the indexed documents in response to a query.
    -   **Prompt Engineering**: Constructs a detailed prompt for the LLM (Google Gemini), including the user's question and the retrieved text chunks as context.
    -   **LLM Interaction**: Sends the prompt to the Gemini API and streams the response back to the user.
    -   **Web Search**: Can augment retrieval with real-time web search results via the Serper API.

-   **`site/index.html` (Frontend)**: A single, self-contained HTML file with embedded JavaScript and CSS. It provides the user interface for the chat application, sends user queries to the `/chat` endpoint on the `main.py` backend, and renders the streamed response from the server.

**Interaction Summary**: The user interacts with `site/index.html`, which communicates with the API in `main.py`. `main.py` uses pre-indexed data (created by its own `PyMuPDF` pipeline) to answer questions. The `docai_ingest.py` script is a separate, utility tool for pre-processing documents with a higher-quality, but slower and more expensive, extraction method.

---

## 2. End-to-End Data Flow

The flow can be broken down into two distinct phases: Ingestion and Querying.

**A) Ingestion Flow:**

1.  **Upload**: An administrator uploads a PDF to the system via a (currently non-existent but implied) admin interface or places it in the `pdfs` directory. The system can also be configured to pull PDFs from a Cloudflare R2 bucket.
2.  **Indexing Trigger**: The application, upon startup or on-demand, detects a new PDF.
3.  **Text Extraction & Chunking**: `main.py` uses `PyMuPDF` to extract plain text from the PDF, page by page. This text is then split into smaller, overlapping chunks of a configured size (e.g., 1200 characters).
4.  **Keyword Indexing (BM25)**: For each chunk, the system tokenizes the text and calculates term frequency. This data is used to build a BM25 index, which allows for efficient keyword-based search. This index is saved to a local cache file.
5.  **Vector Embedding (Optional)**: If enabled, each chunk's text is sent to the Google Vertex AI embeddings API (`text-embedding-004`) to generate a vector representation. These vectors are cached locally.
6.  **Storage**: The original PDF is stored either locally or in the configured R2 bucket. The generated indexes and embeddings are stored in a local `cache` directory.

**B) Query Flow:**

1.  **User Query**: The user types a message in the frontend (`index.html`) and sends it to the `/chat` endpoint in `main.py`.
2.  **Intent Analysis**: `main.py` analyzes the query to determine its type (e.g., "building_regs", "planning") and the required level of "strictness" (how closely the answer must stick to the evidence).
3.  **Hybrid Search**: The system performs a search for relevant chunks:
    -   It uses the BM25 index to find chunks with matching keywords.
    -   It generates an embedding for the user's query and searches the vector index for chunks with the most similar vectors (semantic search).
4.  **Reranking & Context Assembly**: The results from both search methods are combined and reranked. The top-K chunks are selected to be used as context.
5.  **Prompt Construction**: A detailed prompt is created for the Gemini model, containing the user's question, the chat history, and the retrieved text chunks formatted as "Evidence".
6.  **LLM Generation**: The prompt is sent to the Gemini API.
7.  **Streaming Answer**: The API's response is streamed back through `main.py` to the frontend, where it is displayed to the user in real-time.

---

## 3. Risk Assessment

**Scalability**
-   **Local File Storage**: The entire system relies on the local filesystem for caching indexes, embeddings, and parsed documents. This is a major bottleneck and will not scale beyond a single instance.
-   **In-Memory Indexes**: `CHUNK_INDEX` and `EMBED_INDEX` are Python dictionaries that hold all chunk data and embeddings in memory. This is not viable for a large corpus of documents and will lead to high memory consumption and slow startup times.
-   **Single-Threaded Ingestion**: The indexing process is synchronous and single-threaded. Indexing a large batch of documents will be very slow and block other operations.

**Security**
-   **Hardcoded Admin Key**: The `ADMIN_API_KEY` is loaded from an environment variable. If this is weak or accidentally exposed (e.g., in a non-production environment), it provides full access to upload functions.
-   **Prompt Injection**: While the prompt structure is reasonably robust, a sophisticated user could potentially craft a query to make the LLM ignore its instructions or reveal its underlying prompt.
-   **No Input Sanitization on Upload**: The PDF upload endpoint uses the filename directly. A malicious filename (`../../etc/passwd`) could potentially lead to path traversal vulnerabilities, although FastAPI and `Path` objects offer some protection.
-   **Credential Management**: The reliance on numerous environment variables (`.env`) for secrets is standard for development but requires a robust secret management solution (like HashiCorp Vault, AWS/GCP Secret Manager) in production.

**Maintainability**
-   **Monolithic `main.py`**: The `main.py` file is over 3000 lines long. It mixes concerns like server configuration, business logic, data processing, and external API calls. This makes it very difficult to read, test, and maintain.
-   **Lack of Abstraction**: Key components like the retriever, the indexer, and the prompter are not encapsulated in classes or separate modules. They are implemented as collections of functions operating on global dictionaries.
-   **No Tests**: There are no unit or integration tests, making it risky to refactor or add new features.

**Cost Traps**
-   **Google Document AI**: The `docai_ingest.py` script uses a powerful but expensive service. Inadvertently processing a large number of documents could incur significant costs. There is no cost estimation or control layer.
-   **LLM & Embedding APIs**: Every query and every new chunk indexed incurs a cost. Without monitoring, caching, and rate-limiting, costs can escalate quickly, especially with high traffic. The current answer caching is a good start but is basic.

**Accuracy & Reliability**
-   **PDF Extraction for Tables/Diagrams (Gold Asset)**: The current `PyMuPDF` extraction is purely text-based. It will fail to correctly interpret tables, diagrams, and complex layouts, mangling the data and leading to incorrect answers. This is the **single biggest risk to the platform's core value proposition**. The `docai_ingest.py` script is the intended solution, but it's not integrated into the main flow.
-   **Citation Accuracy**: The model is prompted to cite sources, but there's no programmatic verification that the citations are correct or that the generated answer is fully supported by the provided evidence. Hallucinations are still possible.

---

## 4. Deep Focus: PDF Tables & Diagrams

This is the project's Achilles' heel and most critical challenge. To treat this data as a "gold asset," a purely text-based extraction approach is insufficient.

**Proposed Solution:**
A multi-modal extraction pipeline is required. We should integrate and enhance the `docai_ingest.py` approach.

1.  **Use a Layout-Aware Parser**: Google Document AI or an open-source equivalent (like `unstructured.io` with layout detection) should be the default for all documents.
2.  **Extract Multiple Representations**: For each page, we should extract:
    -   **Plain Text**: For standard keyword/semantic search.
    -   **Markdown Tables**: Convert detected tables into Markdown format. This preserves the structure and is LLM-friendly.
    -   **Image Snippets**: Extract diagrams and figures as image files.
3.  **Multi-Modal Indexing**:
    -   Index the plain text and Markdown table text in the vector store.
    -   Store the extracted images in R2/S3.
    -   Add metadata to each text chunk, linking it to any images or tables that appeared on the same page.
4.  **Multi-Modal Retrieval**: When a query mentions a "table" or "diagram," the retriever should be able to:
    -   Find the relevant text chunks.
    -   Use the metadata to retrieve the associated image snippets.
5.  **Multi-Modal Generation**: The final prompt to the LLM (which must be a multi-modal model like Gemini 1.5 Pro) should include both the text chunks and the relevant image snippets, allowing it to reason about the visual information directly.

---

## 5. Production Folder Structure & Refactor Plan

The current flat structure must be refactored into a modular, scalable architecture.

**Proposed Production Structure:**

```
raheem_ai/
├── api/                  # FastAPI application
│   ├── __init__.py
│   ├── main.py             # App entrypoint, routers
│   ├── dependencies.py     # Shared dependencies (e.g., get_db)
│   ├── routers/            # API endpoints
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   └── documents.py
│   └── security.py         # Auth, API keys
│
├── core/                 # Core business logic, config
│   ├── __init__.py
│   ├── config.py           # Pydantic settings management
│   └── models.py           # Pydantic request/response models
│
├── ingestion/            # Data processing pipeline
│   ├── __init__.py
│   ├── indexer.py          # Main indexing logic
│   ├── chunking.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py     # PDF text, table, image extraction
│   │   └── layout_model.py
│
├── retrieval/            # Search and retrieval logic
│   ├── __init__.py
│   ├── retriever.py        # Core retrieval class
│   ├── bm25.py
│   └── vector_search.py
│
├── services/             # External service clients
│   ├── __init__.py
│   ├── llm_service.py      # Interface for Gemini/OpenAI
│   ├── storage_service.py  # Interface for R2/S3
│   └── web_search.py       # Interface for Serper
│
├── store/                # Data storage abstraction
│   ├── __init__.py
│   ├── vector_store.py     # Interface for vector DB (e.g., Chroma, PGVector)
│   └── cache.py            # Redis/other cache interface
│
├── static/               # Frontend files
│   └── index.html
│
├── tests/                # Unit and integration tests
│   ├── ...
│
├── scripts/              # Standalone utility scripts
│   └── initial_ingest.py
│
├── .env.example
├── Dockerfile
└── requirements.txt
```

**Staged Refactor Plan (Small PRs):**

1.  **PR 1: Project Scaffolding**: Create the new directory structure and move existing files into their new locations *without changing code*. For example, move FastAPI app creation to `api/main.py`, move retrieval logic to `retrieval/retriever.py`, etc. The app will be broken at this stage.
2.  **PR 2: Centralized Config**: Introduce a `core/config.py` using Pydantic's `BaseSettings` to manage all environment variables. Update the code to import settings from this central location instead of using `os.getenv` everywhere.
3.  **PR 3: Abstract the Storage Layer**: Create a `StorageService` in `services/storage_service.py` that abstracts away file system vs. R2 logic. The rest of the app should use this service instead of direct file I/O or `boto3` calls.
4.  **PR 4: Refactor Ingestion**: Create an `Indexer` class in `ingestion/indexer.py`. Move all chunking, embedding, and indexing logic from `main.py` into this class. Create a separate API endpoint in `api/routers/documents.py` to trigger indexing.
5.  **PR 5: Refactor Retrieval**: Create a `Retriever` class in `retrieval/retriever.py`. Move all search logic (BM25, vector search) into this class.
6.  **PR 6: Modularize API**: Break the single `/chat` endpoint logic down. Use FastAPI routers to separate the `chat` and `document` management endpoints.
7.  **PR 7: Introduce Testing**: Add `pytest` and write initial unit tests for the simplest, pure-logic components, like `ingestion/chunking.py`.

---

## 6. Proposed MVP Feature Order

Based on the goal of building a robust compliance platform, the features should be rolled out in an order that builds foundational capabilities first.

1.  **MVP 1: TGD & Planning Copilot**: This is the core RAG engine. The immediate priority is perfecting the **PDF extraction for tables and diagrams**. A reliable Copilot that can accurately answer questions about the TGDs is the foundation for everything else. This involves implementing the multi-modal extraction pipeline described in section 4.
2.  **MVP 2: Smart Compliance Checker**: Once the Copilot is reliable, this feature can be built on top of it. This requires building a "rules engine" layer. The system will take user input (e.g., "staircase with 250mm going and 190mm riser") and execute a series of targeted queries against the indexed documents to verify compliance against specific clauses (e.g., TGD Part K requirements).
3.  **MVP 3: Fire Safety + DAC Generators**: These are specialized versions of the Smart Compliance Checker. They would use a predefined set of rules and queries focused specifically on TGD Part B (Fire) and Part M (Access). This becomes a guided workflow for the user, asking for specific project details and generating a preliminary compliance report.
4.  **MVP 4: Submission Pack Builder**: This is the final step, combining the outputs of the previous features. It would aggregate the reports, cited evidence, and user inputs into a formatted document (e.g., PDF or DOCX) ready for submission. This is more of a "document assembly" feature than a core AI task.

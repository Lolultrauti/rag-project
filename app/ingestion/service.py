"""
service.py  --  reusable single-file ingestion, callable from the API.

The CLI orchestrator (scripts/ingest_documents.py) is great for batch runs but
prints to stdout and isn't meant to be imported by a web handler. This module
exposes one function, ingest_path(), that ingests a single file and returns a
structured result dict the /ingest endpoint can serialize straight to JSON.

It reuses the exact same building blocks as the CLI path (loader, embedder,
writer) and the same atomicity contract: document row + chunk rows commit
together, or roll back together. The only thing added here is light multi-format
loading (PDF via PyPDFLoader; .txt/.md read directly) so users can upload the
common document types the UI advertises, not just PDFs.

This stays synchronous by design -- it's the same blocking pipeline as batch
ingestion. The API layer offloads it to a threadpool so it never blocks the
event loop.
"""

import os

from langchain_core.documents import Document

from app.ingestion.loader import load_pdf, chunk_documents
from app.ingestion.embedder import embed_chunks, DailyQuotaExceeded
from app.db.writer import (
    get_connection,
    compute_file_hash,
    document_already_ingested,
    insert_document,
    insert_chunks,
)

# Formats we can actually parse. PDF is the primary case; plain text and
# markdown are trivial to support and round out the "PDF, DOCX, MD, TXT" the
# UI mentions. DOCX is intentionally NOT claimed here -- we have no parser for
# it yet, so we reject it with a clear message rather than silently mangling it.
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def _load_documents(file_path: str):
    """Load a file into LangChain Document(s), dispatching on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext in (".txt", ".md"):
        # Read as UTF-8; ignore undecodable bytes rather than crash on an odd
        # encoding. One Document for the whole file -- chunk_documents splits it.
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": file_path})]
    raise ValueError(f"Unsupported file type: {ext or '(none)'}")


def ingest_path(file_path: str) -> dict:
    """
    Ingest one already-saved file. Returns a result dict:

        {status: "ingested"|"skipped"|"failed",
         filename, chunks, document_id, message}

    status meanings:
      - ingested : new file, chunks embedded and stored
      - skipped  : identical content already in the DB (idempotency hit) --
                   no embedding calls were made
      - failed   : something went wrong; nothing was persisted (rolled back)
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "status": "failed", "filename": filename, "chunks": 0,
            "document_id": None,
            "message": f"Unsupported type '{ext or '?'}'. "
                       f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}.",
        }

    content_hash = compute_file_hash(file_path)
    conn = get_connection()
    try:
        if document_already_ingested(conn, content_hash):
            return {
                "status": "skipped", "filename": filename, "chunks": 0,
                "document_id": None,
                "message": "Already ingested (identical content). Skipped — "
                           "no embeddings called.",
            }

        documents = _load_documents(file_path)
        chunks = chunk_documents(documents)
        if not chunks:
            return {
                "status": "failed", "filename": filename, "chunks": 0,
                "document_id": None,
                "message": "No extractable text found in the file.",
            }

        chunk_embedding_pairs = embed_chunks(chunks)

        # atomic: parent document + all chunks commit together
        document_id = insert_document(conn, filename, content_hash)
        inserted = insert_chunks(conn, document_id, chunk_embedding_pairs)
        conn.commit()

        return {
            "status": "ingested", "filename": filename, "chunks": inserted,
            "document_id": document_id,
            "message": f"Indexed {inserted} chunks.",
        }
    except DailyQuotaExceeded as e:
        # Expected, user-facing condition -- show the clean message as-is
        # (no exception-type prefix) so the UI can display it directly.
        conn.rollback()
        return {
            "status": "failed", "filename": filename, "chunks": 0,
            "document_id": None,
            "message": str(e),
        }
    except Exception as e:
        conn.rollback()
        return {
            "status": "failed", "filename": filename, "chunks": 0,
            "document_id": None,
            "message": f"{type(e).__name__}: {e}",
        }
    finally:
        conn.close()

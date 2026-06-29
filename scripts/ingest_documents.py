"""
ingest_documents.py  --  end-to-end ingestion orchestrator.

This is the conductor. It owns none of the real work itself; it wires
together the single-purpose modules in the right order:

    loader   -> read PDF + split into chunks
    embedder -> turn chunks into 768-dim vectors
    writer   -> hash file, check for duplicates, persist

Two structural decisions worth calling out:

1. Transaction ownership lives HERE, not in writer.py. writer.insert_document()
   and writer.insert_chunks() deliberately do not commit. The orchestrator
   commits once, only after BOTH the parent document row and all its chunk
   rows are inserted, and rolls back on any failure. That makes each file's
   ingestion atomic: a document and its chunks are persisted together or not
   at all -- never a half-written document that would poison the
   content_hash idempotency check and block future re-ingestion.

2. Per-file fault isolation. A batch (a directory of PDFs) must not die
   because one file is corrupt or one API call exhausts its retries. Each
   file is processed in its own try/except: a failure is logged, that file's
   transaction is rolled back, and the loop moves on. A summary at the end
   reports succeeded / skipped / failed counts.

Usage:
    python -m scripts.ingest_documents "data/raw/INDEX 1.pdf"   # one file
    python -m scripts.ingest_documents data/raw                 # whole dir
    python -m scripts.ingest_documents                          # default file

Note the quotes: the default file name contains a space. We accept the path
as a single argv element so the shell's quoting handles it; we never split
on whitespace ourselves.
"""

import os
import sys
import time

from app.ingestion.loader import load_pdf, chunk_documents
from app.ingestion.embedder import embed_chunks
from app.db.writer import (
    get_connection,
    compute_file_hash,
    document_already_ingested,
    insert_document,
    insert_chunks,
)

DEFAULT_FILE = "data/raw/INDEX 1.pdf"

# Outcomes for a single file, used to build the end-of-run summary.
SUCCEEDED = "succeeded"
SKIPPED = "skipped"
FAILED = "failed"


def collect_files(path: str):
    """
    Resolves an input path into a list of files to ingest.

    A single file -> [that file]. A directory -> every .pdf inside it
    (non-recursive; flat folders are the common case and recursion can be
    added later if needed). Anything else is an error the caller reports.
    """
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".pdf")
        )
    return []


def ingest_file(conn, file_path: str) -> str:
    """
    Ingests a single file within its own transaction. Returns one of
    SUCCEEDED / SKIPPED / FAILED.

    The whole document-insert + chunk-insert sequence is wrapped so that any
    exception rolls the transaction back, leaving the DB exactly as it was
    before this file was attempted (no orphaned parent rows).
    """
    filename = os.path.basename(file_path)
    print(f"\n--- {filename} ---")

    # Idempotency check happens BEFORE any paid embedding calls.
    content_hash = compute_file_hash(file_path)
    if document_already_ingested(conn, content_hash):
        print(f"  SKIP: already ingested (matching content_hash). "
              f"No embeddings called, nothing inserted.")
        return SKIPPED

    try:
        pages = load_pdf(file_path)
        print(f"  Loaded {len(pages)} page(s).")

        chunks = chunk_documents(pages)
        print(f"  Split into {len(chunks)} chunk(s).")

        print(f"  Embedding {len(chunks)} chunks via Gemini...")
        chunk_embedding_pairs = embed_chunks(chunks)

        # --- single atomic unit: parent document + all child chunks ---
        document_id = insert_document(conn, filename, content_hash)
        inserted = insert_chunks(conn, document_id, chunk_embedding_pairs)
        conn.commit()  # the ONE commit, only after both inserts succeed

        print(f"  OK: document id={document_id}, {inserted} chunks inserted.")
        return SUCCEEDED
    except Exception as e:
        # Roll back so a partial failure leaves no orphaned document row.
        conn.rollback()
        print(f"  FAILED: {type(e).__name__}: {e}")
        print(f"  (transaction rolled back; this file can be retried later)")
        return FAILED


def ingest(path: str) -> None:
    start = time.time()

    files = collect_files(path)
    if not files:
        print(f"ERROR: no PDF file(s) found at: {path}")
        sys.exit(1)

    print(f"=== Ingesting {len(files)} file(s) from: {path} ===")

    counts = {SUCCEEDED: 0, SKIPPED: 0, FAILED: 0}
    conn = get_connection()
    try:
        for file_path in files:
            outcome = ingest_file(conn, file_path)
            counts[outcome] += 1
    finally:
        conn.close()

    elapsed = time.time() - start
    print("\n=== INGESTION SUMMARY ===")
    print(f"  Files processed : {len(files)}")
    print(f"  Succeeded       : {counts[SUCCEEDED]}")
    print(f"  Skipped (dup)   : {counts[SKIPPED]}")
    print(f"  Failed          : {counts[FAILED]}")
    print(f"  Time taken      : {elapsed:.1f}s")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    ingest(target)

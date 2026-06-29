"""
writer.py

Everything that *writes* ingested documents into PostgreSQL lives here.

Design split (same philosophy as loader.py / embedder.py): this file owns
the database side and nothing else. It does not load PDFs, does not chunk,
does not call Gemini. It just takes already-prepared data (a file path to
hash, a filename, a list of chunk/embedding pairs) and persists it. That
keeps DB logic testable in isolation and swappable later.

We use psycopg2 (the mature, synchronous Postgres driver). Embeddings are
sent to the VECTOR(768) column as their pgvector string form "[0.1,0.2,...]"
which pgvector casts to a vector automatically -- this avoids needing a
separate vector adapter and keeps the dependency surface small.
"""

import hashlib

import psycopg2
from psycopg2.extras import execute_values

from app.config import settings


def get_connection():
    """
    Opens a new psycopg2 connection to the database (settings.database_url).

    Callers are responsible for closing it (or using it as a context manager).
    We deliberately do NOT use the shared API connection pool here -- ingestion
    is a one-shot batch job, not a high-concurrency web path, so a single
    short-lived connection is the simplest correct choice. The pool in
    app/db/pool.py exists for the long-running API process only.
    """
    return psycopg2.connect(settings.database_url)


def compute_file_hash(file_path: str) -> str:
    """
    Returns the SHA-256 hex digest of a file's raw bytes.

    This is our idempotency key: identical file contents always produce the
    same hash, so we can detect and skip re-ingesting a document we've
    already processed -- saving both embedding-API cost and duplicate rows.
    We read in chunks so a large PDF never has to sit fully in memory.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def document_already_ingested(conn, content_hash: str) -> bool:
    """
    Returns True if a document with this content_hash already exists.

    Relies on the UNIQUE constraint on documents.content_hash -- this is the
    cheap pre-check that lets the orchestrator bail out before spending any
    money on embeddings.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM documents WHERE content_hash = %s LIMIT 1;",
            (content_hash,),
        )
        return cur.fetchone() is not None


def insert_document(conn, source_filename: str, content_hash: str) -> int:
    """
    Inserts a row into documents and returns its new id.

    Deliberately does NOT commit. The document row and its chunk rows must
    land as one atomic unit: if chunk insertion later fails partway through,
    we want the parent row rolled back too. Committing here would leave an
    orphaned documents row whose content_hash then makes
    document_already_ingested() report the file as done -- permanently
    blocking re-ingestion without manual DB cleanup. The new id is visible to
    insert_chunks() within the same uncommitted transaction, so the foreign
    key still resolves; the caller owns the single commit/rollback.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source_filename, content_hash)
            VALUES (%s, %s)
            RETURNING id;
            """,
            (source_filename, content_hash),
        )
        document_id = cur.fetchone()[0]
    return document_id


def insert_chunks(conn, document_id: int, chunk_embedding_pairs) -> int:
    """
    Bulk-inserts (chunk, embedding) pairs into document_chunks.

    chunk_embedding_pairs is the list of (langchain_chunk, vector) tuples
    produced by embedder.embed_chunks(). We use execute_values for a single
    round-trip bulk insert instead of one INSERT per chunk -- with 150+
    chunks that's the difference between one network call and 150.

    The embedding list is converted to pgvector's text form "[v1,v2,...]";
    the VECTOR(768) column casts it on the way in. Returns rows inserted.

    Like insert_document(), this does NOT commit. The caller owns the
    transaction boundary: it commits once after both the parent document and
    all its chunks have been inserted, and rolls back on any failure, so a
    document and its chunks are always persisted together or not at all.
    """
    rows = []
    for chunk_index, (chunk, vector) in enumerate(chunk_embedding_pairs):
        embedding_str = "[" + ",".join(str(v) for v in vector) + "]"
        # PostgreSQL TEXT columns cannot store NUL (0x00) bytes. PDFs
        # (especially scanned/OCR'd ones) sometimes leave stray NULs in
        # extracted text, which would crash the insert. Strip them here --
        # they carry no meaning for retrieval.
        content = chunk.page_content.replace("\x00", "")
        rows.append((document_id, chunk_index, content, embedding_str))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, content, embedding)
            VALUES %s;
            """,
            rows,
        )
    return len(rows)


if __name__ == "__main__":
    # Standalone connectivity + sanity test. Run this file directly to confirm
    # the DB is reachable and the expected tables exist, before relying on it
    # from the orchestrator.
    print("Testing database connection...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(f"  Connected: {cur.fetchone()[0].split(',')[0]}")

            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('documents', 'document_chunks')
                ORDER BY table_name;
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            print(f"  Tables found: {tables}")
            assert "documents" in tables, "documents table missing!"
            assert "document_chunks" in tables, "document_chunks table missing!"

            cur.execute("SELECT count(*) FROM documents;")
            print(f"  documents rows: {cur.fetchone()[0]}")
            cur.execute("SELECT count(*) FROM document_chunks;")
            print(f"  document_chunks rows: {cur.fetchone()[0]}")

        print("DB connection + schema check PASSED.")
    finally:
        conn.close()

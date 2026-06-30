"""Verifies the hybrid-retrieval migration (002) is applied to the database."""
import pytest
import psycopg2
from app.config import settings


def _conn():
    return psycopg2.connect(settings.database_url)


def test_content_tsv_column_exists():
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'document_chunks' AND column_name = 'content_tsv';
            """
        )
        row = cur.fetchone()
    assert row is not None, "content_tsv column missing -- run migrations/002_hybrid_fts.sql"
    assert row[0] == "tsvector"


def test_tsv_gin_index_exists():
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'idx_document_chunks_tsv';")
        assert cur.fetchone() is not None, "GIN index idx_document_chunks_tsv missing"


def test_existing_rows_backfilled():
    # A generated STORED column backfills existing rows on ADD COLUMN. If any
    # chunks exist, their content_tsv must be populated (non-null).
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks;")
        total = cur.fetchone()[0]
        if total == 0:
            pytest.skip("no chunks to verify")
        cur.execute("SELECT count(*) FROM document_chunks WHERE content_tsv IS NULL;")
        assert cur.fetchone()[0] == 0

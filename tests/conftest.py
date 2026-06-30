"""Shared pytest fixtures: a DB connection and a helper to seed temporary
documents/chunks. Seeded rows are COMMITTED (so pooled connections used by the
code under test can see them) and deleted in teardown via ON DELETE CASCADE."""
import psycopg2
import pytest

from app.config import settings

# document_chunks.embedding is VECTOR(768) NOT NULL, so even a lexical-only test
# must supply a valid vector. Zeros are fine -- pgvector cosine distance to a
# zero vector is NaN, which never clears the dense floor, so a zero-embedded
# chunk is invisible to dense search and only reachable via lexical search.
ZERO_VEC = "[" + ",".join("0" for _ in range(768)) + "]"


@pytest.fixture
def db_conn():
    conn = psycopg2.connect(settings.database_url)
    yield conn
    conn.close()


@pytest.fixture
def seeded(db_conn):
    """Returns a function seed(filename, chunks, embedding=ZERO_VEC) -> doc_id.
    chunks is a list of (chunk_index, content) tuples. All seeded docs are
    deleted after the test."""
    created = []

    def _seed(filename, chunks, embedding=ZERO_VEC):
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (source_filename, content_hash) "
                "VALUES (%s, %s) RETURNING id;",
                (filename, f"test-{filename}-{len(created)}-{id(chunks)}"),
            )
            doc_id = cur.fetchone()[0]
            for idx, content in chunks:
                cur.execute(
                    "INSERT INTO document_chunks "
                    "(document_id, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s::vector);",
                    (doc_id, idx, content, embedding),
                )
        db_conn.commit()
        created.append(doc_id)
        return doc_id

    yield _seed

    with db_conn.cursor() as cur:
        for doc_id in created:
            cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
    db_conn.commit()

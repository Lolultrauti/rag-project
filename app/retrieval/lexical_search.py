"""
lexical_search.py  --  keyword retrieval via Postgres full-text search.

The dense (pgvector) retriever is strong on meaning but weak on specific term
and entity lookups: a short question like "who performed this experiment" may
not pull the one chunk that literally says "the experiment was performed by..."
into the top results. Postgres full-text search is the complement -- it matches
on actual lexemes, so it nails exactly those keyword/entity queries. We fuse the
two elsewhere (see fusion.py / hybrid.py).

This is built into Postgres: no new dependency, no API call. We rank with
ts_rank_cd over the precomputed content_tsv column (a GIN-indexed generated
column; see migrations/002_hybrid_fts.sql). websearch_to_tsquery parses ordinary
user input gracefully -- it tolerates punctuation and quoted phrases and never
raises on stray characters, unlike raw to_tsquery.

Blocking by design (sync psycopg2), like dense_search; hybrid.search offloads it
to a threadpool so the event loop is never blocked.
"""

from app.db.pool import pooled_connection


def lexical_search(query: str, limit: int = 20):
    """
    Return up to `limit` chunks matching `query` by keyword, ranked best-first.

    Each result dict matches the dense retriever's shape so the two can be fused
    and consumed uniformly: {id, document_id, chunk_index, content,
    source_filename, similarity}. similarity is always 0.0 here -- a lexical hit
    has no cosine score; the field exists only to keep one uniform shape.

    Returns [] when the query reduces to no lexemes or nothing matches (a normal,
    expected outcome -- it lets the caller abstain when nothing is found).
    """
    with pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    d.source_filename
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id,
                     websearch_to_tsquery('english', %s) AS q
                WHERE dc.content_tsv @@ q
                ORDER BY ts_rank_cd(dc.content_tsv, q) DESC
                LIMIT %s;
                """,
                (query, limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "document_id": r[1],
            "chunk_index": r[2],
            "content": r[3],
            "source_filename": r[4],
            "similarity": 0.0,
        }
        for r in rows
    ]

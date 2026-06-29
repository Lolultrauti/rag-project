"""
vector_search.py  --  the "R" in RAG.

Given a natural-language question, find the chunks in our database whose
embeddings are most semantically similar to it. This is pure retrieval: no
LLM, no answer generation -- just "which stored passages are most relevant?"

How similarity works here:
  - We embed the query into the SAME 768-dim space as the documents.
  - pgvector's `<=>` operator computes COSINE DISTANCE between two vectors
    (0 = identical direction, 2 = opposite). Our HNSW index was built with
    vector_cosine_ops, so this operator is the one the index accelerates.
  - We convert distance to an intuitive similarity score: 1 - distance,
    so higher = more relevant (1.0 = perfect match).
"""

from starlette.concurrency import run_in_threadpool

from app.ingestion.embedder import embed_text
from app.db.pool import pooled_connection


def _search_blocking(query: str, top_k: int, min_similarity: float):
    """
    Synchronous core of the search: embed the query, then run the pgvector
    similarity query against a pooled connection.

    This is deliberately blocking -- it makes a synchronous Gemini embedding
    HTTP call (embed_text lives in the ingestion module, which we keep sync
    by design) and a synchronous psycopg2 query. The public async wrapper
    `search()` offloads this whole function to a worker thread so the event
    loop is never blocked. See search() for the threadpool rationale.
    """
    query_vector = embed_text(query, task_type="RETRIEVAL_QUERY")
    query_vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    max_distance = 1 - min_similarity

    with pooled_connection() as conn:
        with conn.cursor() as cur:
            # `<=>` is cosine distance. We order ascending (closest first)
            # and cast our query string to ::vector so pgvector parses it.
            # similarity = 1 - distance for an easy "higher is better" score.
            # The WHERE clause drops anything below the similarity floor
            # before it's ever fetched.
            # Join documents so each result carries its human-readable source
            # filename (for the UI) alongside the raw ids.
            cur.execute(
                """
                SELECT
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    d.source_filename,
                    1 - (dc.embedding <=> %s::vector) AS similarity
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE (dc.embedding <=> %s::vector) <= %s
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector_str, query_vector_str, max_distance,
                 query_vector_str, top_k),
            )
            rows = cur.fetchall()
        # Pooled connection is reused, not closed; commit-free read needs no
        # explicit transaction handling, but we return it clean either way.

    return [
        {
            "document_id": r[0],
            "chunk_index": r[1],
            "content": r[2],
            "source_filename": r[3],
            "similarity": float(r[4]),
        }
        for r in rows
    ]


async def search(query: str, top_k: int = 5, min_similarity: float = 0.62):
    """
    Embeds `query` and returns up to top_k similar chunks above a relevance
    floor.

    Each result is a dict: {chunk_index, content, similarity, document_id}.
    Returns [] when nothing clears the floor (a valid, expected outcome --
    not an error).

    IMPORTANT -- task_type="RETRIEVAL_QUERY" (not "RETRIEVAL_DOCUMENT"):
    Gemini's embedding model is asymmetric. Documents were indexed with
    task_type="RETRIEVAL_DOCUMENT"; a search query must be embedded with
    "RETRIEVAL_QUERY". The model deliberately maps questions and the
    passages that answer them into compatible regions of the vector space,
    so a short question ("What is Huffman coding?") lands near the longer
    passage that explains it. Using the wrong task_type on either side
    measurably degrades retrieval quality, so this is intentional.

    min_similarity (cosine, 1.0 = identical): without a floor, search always
    returns the top_k *least bad* matches even for a query with no real
    answer in the store -- handing the LLM junk context. The floor is a
    deterministic safety net that complements the abstention prompt: if the
    best matches aren't actually similar, we'd rather return nothing and let
    the caller abstain. We filter in SQL (HAVING-style WHERE on the distance
    expression) so rows below the floor never cross the wire. Note the
    algebra: similarity >= min_similarity  <=>  distance <= 1 - min_similarity.

    The default 0.62 is tuned to Gemini gemini-embedding-001's score
    distribution, which has a high baseline: measured on this corpus,
    on-topic queries score ~0.76 while clearly off-topic ones still score
    ~0.57-0.59. A lower floor (e.g. 0.5) lets that off-topic noise through;
    0.62 sits in the gap so genuinely irrelevant queries return [] and
    short-circuit to abstention. Re-tune if the embedding model changes.

    ASYNC: this is the live /query path. Both the embedding HTTP call and the
    psycopg2 query are blocking and sync. Rather than convert the ingestion
    embedder to async or swap in asyncpg (and lose the simple pgvector string
    insert format), we offload the whole blocking core to a worker thread
    with run_in_threadpool. The event loop stays free to service other
    requests while this one waits on Gemini/Postgres -- which is the entire
    point of an async public endpoint.
    """
    return await run_in_threadpool(
        _search_blocking, query, top_k, min_similarity
    )


if __name__ == "__main__":
    # Standalone relevance check against the ingested Digital Image
    # Processing notes (INDEX 1.pdf). search() is async now, so drive it
    # through asyncio.run from this sync entrypoint.
    import asyncio

    question = "What is image compression and how does an encoder and decoder work?"
    print(f"Query: {question}\n")

    results = asyncio.run(search(question, top_k=5))
    for i, r in enumerate(results, 1):
        preview = " ".join(r["content"].split())[:200]
        print(f"[{i}] similarity={r['similarity']:.4f}  "
              f"(doc {r['document_id']}, chunk {r['chunk_index']})")
        print(f"    {preview}\n")

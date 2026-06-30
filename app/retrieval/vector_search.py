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

from app.ingestion.embedder import embed_text
from app.db.pool import pooled_connection

DEFAULT_CANDIDATES = 20
DEFAULT_MIN_SIMILARITY = 0.62


def dense_search(query: str, limit: int = DEFAULT_CANDIDATES,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY):
    """
    Semantic retrieval: embed the query into the same 768-dim space as the
    documents, then return the chunks whose embeddings are closest by cosine
    similarity, above a relevance floor.

    Blocking by design: it makes a synchronous Gemini embedding call and a
    synchronous psycopg2 query. The async public entry point (hybrid.search)
    offloads this to a worker thread.

    task_type="RETRIEVAL_QUERY" is required: Gemini's embedding space is
    asymmetric -- documents were indexed as RETRIEVAL_DOCUMENT, and a query must
    be embedded as RETRIEVAL_QUERY for the two to land near each other.

    min_similarity (cosine, 1.0 = identical) is a floor that keeps clearly
    off-topic queries from returning their least-bad matches. Measured on this
    corpus, on-topic queries score ~0.76 while off-topic ones still score
    ~0.57-0.59, so 0.62 sits in the gap. Returns up to `limit` dicts, best-first;
    [] if nothing clears the floor.
    """
    query_vector = embed_text(query, task_type="RETRIEVAL_QUERY")
    query_vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    max_distance = 1 - min_similarity

    with pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
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
                 query_vector_str, limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "document_id": r[1],
            "chunk_index": r[2],
            "content": r[3],
            "source_filename": r[4],
            "similarity": float(r[5]),
        }
        for r in rows
    ]


if __name__ == "__main__":
    question = "What is image compression and how does an encoder and decoder work?"
    print(f"Query: {question}\n")
    for i, r in enumerate(dense_search(question, limit=5), 1):
        preview = " ".join(r["content"].split())[:200]
        print(f"[{i}] similarity={r['similarity']:.4f} "
              f"(doc {r['document_id']}, chunk {r['chunk_index']})")
        print(f"    {preview}\n")

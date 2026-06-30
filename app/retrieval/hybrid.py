"""
hybrid.py  --  public retrieval entry point: dense + lexical, fused.

This is the "R" in RAG as the rest of the app sees it. It runs the two
retrievers independently, fuses their rankings with RRF, and returns the top
chunks. Two properties matter:

1. Resilience. Each retriever is isolated: if one raises (e.g. the dense side
   hits an embedding-quota 429, or the lexical side gets a malformed query), we
   log it and fall back to whatever the other retriever returned. The request
   still succeeds. Only when BOTH fail/return nothing do we yield [] -> the
   caller abstains.

2. Uniform output. Both retrievers emit the same dict shape, so fused results
   are a drop-in replacement for the old vector_search.search output: callers
   (chain.answer_question, the API Source model) need no changes. For a chunk
   found by both retrievers we keep the dense dict, so its real cosine
   similarity is preserved for the UI; lexical-only chunks report similarity 0.0.

Blocking work (DB + the dense embedding call) lives in _hybrid_blocking and is
offloaded to a worker thread by the async search() wrapper, so the event loop
stays free -- same pattern the old vector_search.search used.
"""

import logging

from starlette.concurrency import run_in_threadpool

from app.retrieval.vector_search import dense_search
from app.retrieval.lexical_search import lexical_search
from app.retrieval.fusion import rrf_fuse

logger = logging.getLogger("rag.retrieval")

CANDIDATES_PER_RETRIEVER = 20
RRF_K = 60
FINAL_TOP_K = 6


def _hybrid_blocking(query: str, top_k: int) -> list[dict]:
    dense = []
    lexical = []
    try:
        dense = dense_search(query, limit=CANDIDATES_PER_RETRIEVER)
    except Exception:
        logger.exception("Dense retrieval failed; falling back to lexical only.")
    try:
        lexical = lexical_search(query, limit=CANDIDATES_PER_RETRIEVER)
    except Exception:
        logger.exception("Lexical retrieval failed; falling back to dense only.")

    if not dense and not lexical:
        return []

    # Dense first so a chunk found by both keeps its real cosine similarity.
    by_id = {}
    for chunk in dense + lexical:
        by_id.setdefault(chunk["id"], chunk)

    fused_ids = rrf_fuse(
        [[c["id"] for c in dense], [c["id"] for c in lexical]],
        k=RRF_K,
    )
    return [by_id[cid] for cid in fused_ids[:top_k]]


async def search(query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Embed-and-retrieve the most relevant chunks for `query`, combining semantic
    and keyword search. Returns up to top_k chunk dicts (see module docstring for
    shape); [] when nothing is found, which the caller turns into an abstention.
    """
    return await run_in_threadpool(_hybrid_blocking, query, top_k)

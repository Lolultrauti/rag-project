"""
routes.py  --  HTTP surface for the RAG system.

Endpoints:
  POST /query  -- the RAG endpoint (retrieval + generation)
  GET  /ready  -- readiness probe (can we actually serve? checks DB + config)

(GET /health lives in main.py as a pure, dependency-free liveness probe.)

Request:  {"question": "..."}
Response: {"answer": "...", "sources": [{document_id, chunk_index, similarity,
           preview}, ...]}

We keep request/response shapes in Pydantic models so FastAPI validates input
automatically and documents the API at /docs for free.

Protections layered on /query (all Phase 1):
  - per-IP rate limit (slowapi) -> 429 on breach
  - global daily cost cap (cost_cap) -> 503 once the day's budget is spent,
    checked BEFORE any paid Gemini call
  - generic 503 on any internal failure, with the real error logged
    server-side and never leaked to the client

Auth, CORS, and write/ingest endpoints are intentionally out of scope here.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.rate_limit import limiter
from app.cost_cap import check_and_increment
from app.db.pool import pooled_connection
from app.generation.chain import answer_question, maybe_handle_small_talk
from app.retrieval.hybrid import FINAL_TOP_K

logger = logging.getLogger("rag.api")

router = APIRouter()


class QueryRequest(BaseModel):
    # max_length is a cheap, free guard against the most obvious abuse
    # (megabyte-long "questions"). Not full input hardening -- that's later.
    question: str = Field(..., min_length=1, max_length=1000,
                          description="User's question.")
    top_k: int = Field(FINAL_TOP_K, ge=1, le=20, description="How many chunks to retrieve.")


class Source(BaseModel):
    document_id: int
    chunk_index: int
    filename: str
    similarity: float
    # We include a short preview (not the full passage) so a caller can sanity
    # check grounding without us shipping large passages over the wire by
    # default. Full content stays server-side; expose it later if a UI needs it.
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    # NOTE: `request: Request` is required by slowapi's decorator (it reads the
    # client IP from it); it isn't used directly in the body.

    small_talk = maybe_handle_small_talk(req.question)
    if small_talk is not None:
        return QueryResponse(answer=small_talk, sources=[])

    # Daily cost cap FIRST -- before any embedding/generation spend. Once the
    # day's budget is gone we reject cheaply rather than calling the LLM.
    if not check_and_increment():
        raise HTTPException(
            status_code=503,
            detail="Daily usage limit reached, please try again tomorrow.",
        )

    try:
        answer, chunks = await answer_question(req.question, top_k=req.top_k)
    except Exception:
        # Log the real cause for operators; return a generic message to clients
        # (never leak exception text / stack traces to a public endpoint).
        logger.exception("Query failed for question=%r", req.question)
        raise HTTPException(
            status_code=503,
            detail="The system is temporarily unavailable, please try again.",
        )

    sources = [
        Source(
            document_id=c["document_id"],
            chunk_index=c["chunk_index"],
            filename=c.get("source_filename", ""),
            similarity=round(c["similarity"], 4),
            preview=" ".join(c["content"].split())[:200],
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)


def _db_ping() -> None:
    """Blocking 'can we reach Postgres?' check, run in a threadpool."""
    with pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()


@router.get("/ready")
async def ready():
    """
    Readiness probe: distinct from /health. /health says "the process is up";
    /ready says "the process can actually serve a /query right now". The
    platform can use this to gate traffic until dependencies are wired.

    Checks (cheap, no real LLM call):
      - the DB pool can hand out a working connection (SELECT 1)
      - the Gemini API key is configured (client init / key presence)
    Returns 503 naming the first failed dependency.
    """
    try:
        await run_in_threadpool(_db_ping)
    except Exception:
        logger.exception("Readiness check failed: database")
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "failed": "database"},
        )

    if not settings.gemini_api_key:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "failed": "gemini_api_key"},
        )

    return {"status": "ready"}

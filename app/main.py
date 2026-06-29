"""
main.py  --  FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Interactive docs are auto-generated at http://localhost:8000/docs

Wiring done here:
  - lifespan: create the shared DB connection pool at startup, drain it at
    shutdown (so connections aren't leaked across reloads/redeploys).
  - slowapi: register the shared limiter on app.state and install a custom
    429 handler so a rate-limit breach returns a clean JSON message instead
    of a framework default page.
  - /health: pure liveness probe, no dependencies (defined here so it stays
    trivially fast and decoupled from the dependency-touching routes).
"""

import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.rate_limit import limiter
from app.db.pool import init_pool, close_pool, pooled_connection
from app.ingestion.service import ingest_path, SUPPORTED_EXTENSIONS

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# Uploaded files land here (same place the CLI ingests from), so a file added
# via the UI is indistinguishable from one added via the batch pipeline.
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
# Cap upload size. Embedding cost/latency scales with document length, and the
# free Gemini tier has per-minute + daily quotas, so we keep demo uploads small.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def _safe_filename(name: str) -> str:
    """Strip any path components and unsafe chars to prevent path traversal."""
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    return base or "upload"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open the pool once, shared across all requests.
    init_pool()
    yield
    # Shutdown: return every connection to Postgres cleanly.
    close_pool()


app = FastAPI(
    title="Enterprise RAG API",
    description="Retrieval-Augmented Generation over ingested PDF documents "
                "using pgvector + Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# slowapi needs the limiter on app.state; the decorator on /query reads it.
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return a clear JSON 429 instead of slowapi's default response."""
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before sending "
                          "another request."},
    )


app.include_router(router)


@app.get("/", include_in_schema=False)
def index():
    """Serve the Helix RAG single-page frontend."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _corpus_stats() -> dict:
    """Blocking count of documents + passages, run in a threadpool for /stats."""
    with pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents;")
            docs = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM document_chunks;")
            chunks = cur.fetchone()[0]
            return {"documents": docs, "chunks": chunks}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accept a user-uploaded document and run it through the real ingestion
    pipeline (hash -> dedupe -> load -> chunk -> embed -> store), the same one
    the CLI uses. Returns a structured result the UI renders per file.

    Note: this is synchronous, blocking work (Gemini embedding calls), so it's
    offloaded to a threadpool. Large files take a while -- embedding is paced
    under the free-tier rate limit -- so the UI shows an indeterminate progress
    state until this returns.
    """
    filename = _safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{ext or '?'}'. "
                   f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}.",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)//(1024*1024)} MB). "
                   f"Max {MAX_UPLOAD_BYTES//(1024*1024)} MB.",
        )

    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, filename)
    with open(dest, "wb") as f:
        f.write(contents)

    # Blocking pipeline -> threadpool so the event loop stays free.
    result = await run_in_threadpool(ingest_path, dest)
    return result


@app.get("/stats")
async def stats():
    """Lightweight corpus stats for the UI sidebar (document + passage counts)."""
    try:
        return await run_in_threadpool(_corpus_stats)
    except Exception:
        return JSONResponse(status_code=503,
                            content={"documents": None, "chunks": None})


@app.get("/health")
def health():
    """
    Liveness probe -- confirms the API process is up. Deliberately has NO
    dependencies (no DB, no LLM) so it always answers instantly for the
    platform's health checks, even if Postgres or Gemini is down. Use /ready
    for a dependency-aware "can I actually serve traffic?" check.
    """
    return {"status": "ok"}

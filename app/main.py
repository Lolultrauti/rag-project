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

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.rate_limit import limiter
from app.db.pool import init_pool, close_pool


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


@app.get("/health")
def health():
    """
    Liveness probe -- confirms the API process is up. Deliberately has NO
    dependencies (no DB, no LLM) so it always answers instantly for the
    platform's health checks, even if Postgres or Gemini is down. Use /ready
    for a dependency-aware "can I actually serve traffic?" check.
    """
    return {"status": "ok"}

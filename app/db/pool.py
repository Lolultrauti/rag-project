"""
pool.py  --  shared PostgreSQL connection pool for the request path.

Why a pool: the live /query read path (vector_search) previously opened a
fresh psycopg2.connect() per request. Each connect pays a full TCP + auth
handshake, and under concurrent public traffic a burst of requests can
exhaust Postgres's max_connections outright. A pool keeps a small set of
connections open and hands them out, so request latency drops and the
connection count stays bounded no matter how much traffic arrives.

Scope: this pool is for the long-running API process only. The ingestion
orchestrator is a separate one-shot batch job that opens a single
short-lived connection (see writer.get_connection) -- pooling there would
add lifecycle complexity for no benefit, so it deliberately stays unpooled.

Lifecycle: init_pool()/close_pool() are called from the FastAPI lifespan so
the pool is created once at startup and drained at shutdown. getconn() also
lazily initialises the pool on first use, so the module-level __main__ test
harnesses in vector_search/chain work without booting the whole API.
"""

from contextlib import contextmanager

from psycopg2 import pool as pg_pool

from app.config import settings

# Bounds chosen for a small public deploy: a handful of always-warm
# connections, capped well under Postgres's default max_connections (100)
# so we can never be the cause of "too many connections".
_MIN_CONN = 1
_MAX_CONN = 10

_pool = None


def init_pool() -> None:
    """Create the pool if it doesn't exist yet. Idempotent."""
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(
            _MIN_CONN, _MAX_CONN, dsn=settings.database_url
        )


def close_pool() -> None:
    """Close all pooled connections (called at app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def pooled_connection():
    """
    Borrow a connection from the pool for the duration of the `with` block,
    then return it (not close it) so it can be reused.

    Lazily initialises the pool on first use so standalone scripts that call
    into the read path without the FastAPI lifespan still work.
    """
    init_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        try:
            if not conn.closed:
                conn.rollback()
        except Exception:
            pass
        _pool.putconn(conn)

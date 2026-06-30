# Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make specific keyword/entity questions (e.g. "who performed this experiment") answerable by adding keyword search alongside the existing semantic search, fused by Reciprocal Rank Fusion.

**Architecture:** Add a Postgres full-text (`tsvector`) index next to the existing pgvector embeddings. Per query, run a dense (semantic) retriever and a lexical (keyword) retriever independently, then fuse their ranked results with RRF. A new `hybrid.search()` orchestrates both, falls back to whichever retriever succeeds, and abstains only when both return nothing.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, PostgreSQL 15/16 + pgvector, Postgres built-in full-text search, pytest. Gemini for embeddings (unchanged).

## Global Constraints

- **No new runtime dependencies** — lexical search uses Postgres built-ins only.
- **No new per-query API cost** — exactly 1 Gemini embedding call per query (dense side); lexical is pure SQL.
- **Full-text config literal is `'english'`** everywhere (keeps the generated column immutable; required by Postgres).
- **Postgres 12+** (generated STORED columns). The pgvector image is 15/16 — satisfied.
- **Public retrieval contract unchanged:** `async search(query, top_k=...) -> list[dict]`; each dict has keys `id, document_id, chunk_index, content, source_filename, similarity`. `similarity` is a plain `float` — dense cosine similarity if the chunk came from the dense retriever, else `0.0` for a lexical-only hit.
- **Tunables are module constants** (single source of truth): `CANDIDATES_PER_RETRIEVER = 20`, `RRF_K = 60`, `FINAL_TOP_K = 6`, `MIN_SIMILARITY = 0.62`.
- **Commits:** end every commit message with `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>`. Do not sign commits (`-c commit.gpgsign=false` if the environment forces signing).
- **Run commands** with the project venv active: `source venv/Scripts/activate` (Git Bash on Windows). Tests need the database from `docker-compose.yml` running and the Task 1 migration applied.

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `migrations/002_hybrid_fts.sql` | create | Add `content_tsv` generated column + GIN index |
| `schema.sql` | modify | Same column + index for fresh installs |
| `app/retrieval/fusion.py` | create | `rrf_fuse()` — pure RRF, no DB/API |
| `app/retrieval/lexical_search.py` | create | `lexical_search()` — Postgres full-text |
| `app/retrieval/vector_search.py` | modify | Expose `dense_search()` (adds `id`, `limit` param) |
| `app/retrieval/hybrid.py` | create | Public `search()` — orchestrate + fuse + fallback + abstain |
| `app/generation/chain.py` | modify | Import `search` from `hybrid` instead of `vector_search` |
| `tests/conftest.py` | create | DB connection + seeding fixtures |
| `tests/test_fusion.py` | create | Pure unit tests for RRF |
| `tests/test_lexical_search.py` | create | Lexical search against seeded data |
| `tests/test_dense_search.py` | create | Dense search with monkeypatched embedding |
| `tests/test_hybrid.py` | create | Fusion/dedupe + resilience of orchestrator |
| `tests/test_schema.py` | create | Assert migration applied |
| `tests/integration/test_rag_quality.py` | create | End-to-end, quota-gated (skipped by default) |

---

## Task 1: Schema migration — full-text column + index

**Files:**
- Create: `migrations/002_hybrid_fts.sql`
- Modify: `schema.sql`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `document_chunks.content_tsv` (tsvector, generated STORED from `content`) and index `idx_document_chunks_tsv` (GIN). Every later task depends on these existing in the database.

- [ ] **Step 1: Write the migration SQL**

Create `migrations/002_hybrid_fts.sql`:

```sql
-- Hybrid retrieval: add a generated full-text column + GIN index so we can run
-- keyword search alongside the pgvector semantic search. STORED + GENERATED means
-- Postgres maintains it from content automatically -- existing rows are backfilled
-- on ADD COLUMN and all future inserts stay correct with no ingestion code change.
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
  ON document_chunks USING GIN (content_tsv);
```

- [ ] **Step 2: Add the same to `schema.sql` for fresh installs**

In `schema.sql`, the `document_chunks` table currently ends at the `embedding VECTOR(768) NOT NULL,` / `created_at` block. Add the column to the `CREATE TABLE` and the index after the existing indexes. Append these two statements at the end of `schema.sql` (idempotent, safe even though the column is also in the table for fresh installs — keep ONLY the index here to avoid duplicating the column; add the column inline in the CREATE TABLE):

Modify the `CREATE TABLE document_chunks (...)` to include, right after the `embedding VECTOR(768) NOT NULL,` line:

```sql
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
```

Then add at the end of the file:

```sql
CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
    ON document_chunks USING GIN (content_tsv);
```

- [ ] **Step 3: Apply the migration to the running database**

Run:
```bash
source venv/Scripts/activate
psql "$(python -c 'from app.config import settings; print(settings.database_url)')" -f migrations/002_hybrid_fts.sql
```
If `psql` is unavailable on PATH, apply via Python instead:
```bash
python -c "import psycopg2; from app.config import settings; c=psycopg2.connect(settings.database_url); c.autocommit=True; cur=c.cursor(); cur.execute(open('migrations/002_hybrid_fts.sql').read()); print('applied')"
```
Expected output: `applied` (or psql `ALTER TABLE` / `CREATE INDEX`).

- [ ] **Step 4: Write the failing test**

Create `tests/test_schema.py`:

```python
"""Verifies the hybrid-retrieval migration (002) is applied to the database."""
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
            return  # nothing to verify
        cur.execute("SELECT count(*) FROM document_chunks WHERE content_tsv IS NULL;")
        assert cur.fetchone()[0] == 0
```

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_schema.py -v`
Expected: 3 passed. (If you run this BEFORE Step 3, `test_content_tsv_column_exists` fails with the "column missing" message — that is the failing-first check.)

- [ ] **Step 6: Commit**

```bash
git add migrations/002_hybrid_fts.sql schema.sql tests/test_schema.py
git commit -m "Add full-text column + GIN index for hybrid retrieval

Generated STORED tsvector on document_chunks, backfills existing rows.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 2: RRF fusion (pure function)

**Files:**
- Create: `app/retrieval/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `rrf_fuse(ranked_lists: list[list], k: int = 60) -> list` — takes several ranked lists of hashable keys (best-first) and returns one deduped list of keys ordered by fused RRF score (best first). Tie-break: higher score first, then key ascending for determinism.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
from app.retrieval.fusion import rrf_fuse


def test_single_list_preserves_order():
    assert rrf_fuse([[3, 1, 2]]) == [3, 1, 2]


def test_key_in_both_lists_ranks_highest():
    # 1 appears near the top of both lists -> should win.
    dense = [1, 2, 3]
    lexical = [1, 4, 5]
    assert rrf_fuse([dense, lexical])[0] == 1


def test_dedupes_keys():
    out = rrf_fuse([[1, 2], [2, 1]])
    assert sorted(out) == [1, 2]
    assert len(out) == 2


def test_empty_lists_return_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_score_formula_and_tie_break():
    # Two keys, each rank-0 in exactly one list -> equal scores 1/(k+1).
    # Tie-break is key ascending, so 1 before 9.
    assert rrf_fuse([[1], [9]], k=60) == [1, 9]


def test_higher_rank_beats_lower_rank_across_lists():
    # 'a' is rank 0 in list1; 'b' is rank 1 in list1 and rank 0 in list2.
    # b score = 1/62 + 1/61 ; a score = 1/61. b should win.
    out = rrf_fuse([["a", "b"], ["b"]], k=60)
    assert out[0] == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.fusion'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/retrieval/fusion.py`:

```python
"""
fusion.py  --  Reciprocal Rank Fusion (RRF) for hybrid retrieval.

We run two independent retrievers (dense semantic search and lexical keyword
search). Their scores live on completely different scales -- cosine similarity
vs. ts_rank -- so we cannot simply add them. RRF sidesteps this by scoring on
RANK, not raw score: a result's contribution from each list is 1 / (k + rank),
where rank is 0-based and k is a smoothing constant (60 is the standard value
from the original RRF paper). Summing those contributions across lists rewards
results that rank well in MULTIPLE retrievers, which is exactly the signal we
want -- a chunk that is both semantically close AND a keyword match should win.

Pure function: no DB, no API, fully deterministic -> trivially unit-testable.
"""

from collections import defaultdict


def rrf_fuse(ranked_lists, k: int = 60):
    """
    Fuse several ranked lists of keys into one, best-first.

    ranked_lists: list of lists; each inner list is one retriever's result keys
    in rank order (best first). Keys must be hashable and mutually comparable
    (e.g. all ints) so ties break deterministically.

    Returns a single deduped list of keys ordered by descending RRF score.
    Ties (equal score) break by key ascending, so output is fully deterministic.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] += 1.0 / (k + rank + 1)  # rank is 0-based; +1 -> 1-based
    return [key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fusion.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/fusion.py tests/test_fusion.py
git commit -m "Add Reciprocal Rank Fusion for hybrid retrieval

Pure rank-based fusion of dense + lexical result lists; avoids cross-scale
score normalization.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Lexical search (Postgres full-text)

**Files:**
- Create: `app/retrieval/lexical_search.py`
- Create: `tests/conftest.py`
- Test: `tests/test_lexical_search.py`

**Interfaces:**
- Consumes: `app.db.pool.pooled_connection`; the `content_tsv` column from Task 1.
- Produces: `lexical_search(query: str, limit: int = 20) -> list[dict]` — blocking. Each dict: `{id, document_id, chunk_index, content, source_filename, similarity}` with `similarity = 0.0` (lexical hits carry no cosine score). Returns `[]` when the query has no usable lexemes or nothing matches.
- Produces (conftest): fixtures `db_conn` and `seeded` for DB tests (used by Tasks 3 and 4).

- [ ] **Step 1: Write the shared test fixtures**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_lexical_search.py`:

```python
from app.retrieval.lexical_search import lexical_search


def test_finds_chunk_by_keyword(seeded):
    seeded("exp.txt", [
        (0, "The water cycle describes evaporation and condensation."),
        (1, "The experiment was performed by Dr. Mendel in 1865."),
    ])
    results = lexical_search("who performed this experiment")
    contents = [r["content"] for r in results]
    assert any("Dr. Mendel" in c for c in contents)


def test_result_shape(seeded):
    seeded("shape.txt", [(0, "Photosynthesis converts sunlight into energy.")])
    results = lexical_search("photosynthesis")
    assert results, "expected at least one match"
    r = results[0]
    assert set(r) == {"id", "document_id", "chunk_index", "content",
                      "source_filename", "similarity"}
    assert r["similarity"] == 0.0
    assert r["source_filename"] == "shape.txt"


def test_off_topic_returns_nothing(seeded):
    seeded("topic.txt", [(0, "Relational databases use primary keys.")])
    assert lexical_search("xyzzy plugh nonsense") == []


def test_respects_limit(seeded):
    seeded("many.txt", [(i, f"alpha token number {i}") for i in range(5)])
    assert len(lexical_search("alpha", limit=2)) <= 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_lexical_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.lexical_search'`.

- [ ] **Step 4: Write minimal implementation**

Create `app/retrieval/lexical_search.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_lexical_search.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/lexical_search.py tests/conftest.py tests/test_lexical_search.py
git commit -m "Add lexical (full-text) retriever

Postgres websearch_to_tsquery + ts_rank_cd over content_tsv; uniform result
shape with the dense retriever. Adds DB seeding fixtures.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Dense search refactor (expose `dense_search`)

**Files:**
- Modify: `app/retrieval/vector_search.py`
- Test: `tests/test_dense_search.py`

**Interfaces:**
- Consumes: `app.ingestion.embedder.embed_text`; `app.db.pool.pooled_connection`; the existing `idx_document_chunks_embedding`.
- Produces: `dense_search(query: str, limit: int = 20, min_similarity: float = 0.62) -> list[dict]` — blocking. Each dict: `{id, document_id, chunk_index, content, source_filename, similarity}` where `similarity` is the cosine similarity (`1 - distance`). Returns up to `limit` chunks above the floor, best-first.

The existing async `search()` and `_search_blocking()` in this file are replaced by `dense_search()`; the public async entry point now lives in `hybrid.py` (Task 5). Update the `__main__` block to call `dense_search` directly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dense_search.py`:

```python
"""dense_search with a monkeypatched embedding -- no real Gemini call, so this
runs even when the embedding quota is exhausted. We seed a chunk whose stored
embedding equals the vector embed_text is patched to return, so cosine distance
is 0 and similarity is 1.0."""
from app.retrieval import vector_search

# A unit vector: first dim 1.0, rest 0.0. Matching stored + query vectors give
# cosine distance 0 -> similarity 1.0, comfortably above the 0.62 floor.
UNIT_VEC_STR = "[" + ",".join("1" if i == 0 else "0" for i in range(768)) + "]"
UNIT_VEC = [1.0] + [0.0] * 767


def test_dense_search_finds_matching_vector(seeded, monkeypatch):
    monkeypatch.setattr(vector_search, "embed_text", lambda q, task_type=None: UNIT_VEC)
    seeded("dense.txt", [(0, "Vector content for dense retrieval test.")],
           embedding=UNIT_VEC_STR)

    results = vector_search.dense_search("any query", limit=5)

    assert results, "expected the seeded chunk to be retrieved"
    top = results[0]
    assert set(top) == {"id", "document_id", "chunk_index", "content",
                        "source_filename", "similarity"}
    assert top["content"] == "Vector content for dense retrieval test."
    assert top["similarity"] > 0.99


def test_dense_search_respects_limit(seeded, monkeypatch):
    monkeypatch.setattr(vector_search, "embed_text", lambda q, task_type=None: UNIT_VEC)
    seeded("dense2.txt", [(i, f"row {i}") for i in range(4)], embedding=UNIT_VEC_STR)
    assert len(vector_search.dense_search("q", limit=2)) <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dense_search.py -v`
Expected: FAIL — `AttributeError: module 'app.retrieval.vector_search' has no attribute 'dense_search'`.

- [ ] **Step 3: Rewrite `vector_search.py`**

Replace the body below the module docstring/imports. Keep the existing top docstring; replace `_search_blocking`, the async `search`, and `__main__` with this:

```python
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
```

Remove the now-unused `from starlette.concurrency import run_in_threadpool` import from this file (it moves to `hybrid.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dense_search.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/vector_search.py tests/test_dense_search.py
git commit -m "Refactor vector_search to expose dense_search(limit, id)

Returns chunk id + similarity for fusion; async entry point moves to hybrid.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Hybrid orchestrator + wire into the chain

**Files:**
- Create: `app/retrieval/hybrid.py`
- Modify: `app/generation/chain.py:28` (the retrieval import)
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `dense_search` (Task 4), `lexical_search` (Task 3), `rrf_fuse` (Task 2).
- Produces: `async search(query: str, top_k: int = 6) -> list[dict]` and the blocking core `_hybrid_blocking(query, top_k)`. Result dicts keep the established shape. `chain.answer_question` consumes `search` unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid.py`:

```python
"""Orchestration logic for hybrid search, tested with fake retrievers so it
needs no DB or API. We patch dense_search/lexical_search as imported into the
hybrid module and exercise _hybrid_blocking directly (sync)."""
from app.retrieval import hybrid


def _chunk(cid, sim=0.0, content="c"):
    return {"id": cid, "document_id": 1, "chunk_index": cid, "content": content,
            "source_filename": "f.txt", "similarity": sim}


def test_fuses_and_dedupes(monkeypatch):
    monkeypatch.setattr(hybrid, "dense_search",
                        lambda q, limit: [_chunk(1, sim=0.9), _chunk(2, sim=0.8)])
    monkeypatch.setattr(hybrid, "lexical_search",
                        lambda q, limit: [_chunk(1), _chunk(3)])
    out = hybrid._hybrid_blocking("q", top_k=6)
    ids = [c["id"] for c in out]
    assert ids[0] == 1            # in both lists -> ranked first
    assert sorted(ids) == [1, 2, 3]  # deduped union
    # dense dict wins for shared id 1 -> keeps its real similarity
    assert next(c for c in out if c["id"] == 1)["similarity"] == 0.9


def test_dense_failure_falls_back_to_lexical(monkeypatch):
    def boom(q, limit):
        raise RuntimeError("embedding quota exhausted")
    monkeypatch.setattr(hybrid, "dense_search", boom)
    monkeypatch.setattr(hybrid, "lexical_search", lambda q, limit: [_chunk(7)])
    out = hybrid._hybrid_blocking("q", top_k=6)
    assert [c["id"] for c in out] == [7]


def test_lexical_failure_falls_back_to_dense(monkeypatch):
    monkeypatch.setattr(hybrid, "dense_search", lambda q, limit: [_chunk(5, sim=0.7)])
    def boom(q, limit):
        raise RuntimeError("bad tsquery")
    monkeypatch.setattr(hybrid, "lexical_search", boom)
    out = hybrid._hybrid_blocking("q", top_k=6)
    assert [c["id"] for c in out] == [5]


def test_both_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(hybrid, "dense_search", lambda q, limit: [])
    monkeypatch.setattr(hybrid, "lexical_search", lambda q, limit: [])
    assert hybrid._hybrid_blocking("q", top_k=6) == []


def test_top_k_truncates(monkeypatch):
    monkeypatch.setattr(hybrid, "dense_search",
                        lambda q, limit: [_chunk(i, sim=1.0) for i in range(10)])
    monkeypatch.setattr(hybrid, "lexical_search", lambda q, limit: [])
    assert len(hybrid._hybrid_blocking("q", top_k=3)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.hybrid'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/retrieval/hybrid.py`:

```python
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


def _hybrid_blocking(query: str, top_k: int):
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


async def search(query: str, top_k: int = FINAL_TOP_K):
    """
    Embed-and-retrieve the most relevant chunks for `query`, combining semantic
    and keyword search. Returns up to top_k chunk dicts (see module docstring for
    shape); [] when nothing is found, which the caller turns into an abstention.
    """
    return await run_in_threadpool(_hybrid_blocking, query, top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hybrid.py -v`
Expected: 5 passed.

- [ ] **Step 5: Repoint the chain import**

In `app/generation/chain.py`, change the retrieval import (currently `from app.retrieval.vector_search import search`):

```python
from app.retrieval.hybrid import search
```

Leave the rest of `chain.py` unchanged — `answer_question` calls `await search(question, top_k=top_k)`, and `search` keeps the same async signature and output shape.

- [ ] **Step 6: Verify nothing else imported the old entry point**

Run: `grep -rn "vector_search import search\|vector_search\.search" app/`
Expected: no matches. (If any remain, repoint them to `app.retrieval.hybrid`.)

- [ ] **Step 7: Run the full unit suite**

Run: `pytest tests/test_fusion.py tests/test_hybrid.py tests/test_lexical_search.py tests/test_dense_search.py tests/test_schema.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval/hybrid.py app/generation/chain.py tests/test_hybrid.py
git commit -m "Add hybrid search orchestrator; wire into generation chain

Fuses dense + lexical with RRF, falls back to either retriever on failure,
abstains only when both are empty. chain.py now retrieves via hybrid.search.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Purge test junk + quota-gated integration verification

**Files:**
- Create: `tests/integration/test_rag_quality.py`
- Test: itself (gated)

**Interfaces:**
- Consumes: `app.generation.chain.answer_question`; live DB + Gemini (real embedding calls).
- Produces: an end-to-end regression suite, skipped unless `RUN_INTEGRATION=1`, so normal test runs never burn embedding quota.

- [ ] **Step 1: Purge the test-junk documents from the corpus**

These rows were added while debugging upload speed and pollute retrieval. Remove from DB (chunks cascade) and disk:

```bash
source venv/Scripts/activate
python -c "import psycopg2; from app.config import settings; c=psycopg2.connect(settings.database_url); c.autocommit=True; cur=c.cursor(); cur.execute(\"DELETE FROM documents WHERE source_filename IN ('quick.txt','big.txt');\"); print('deleted junk docs')"
rm -f data/raw/quick.txt data/raw/big.txt
```
Then confirm only the real PDFs remain:
```bash
python -c "import psycopg2; from app.config import settings; c=psycopg2.connect(settings.database_url); cur=c.cursor(); cur.execute('SELECT source_filename FROM documents ORDER BY id;'); print([r[0] for r in cur.fetchall()])"
```
Expected: `['AI_Unit 4.pdf', 'DBMS.pdf']`.

- [ ] **Step 2: Write the gated integration test**

Create `tests/integration/test_rag_quality.py`:

```python
"""End-to-end retrieval-quality checks against the live corpus + Gemini.

Skipped unless RUN_INTEGRATION=1 so ordinary test runs never spend embedding
quota. Run after the daily quota resets:
    RUN_INTEGRATION=1 pytest tests/integration/test_rag_quality.py -v
"""
import asyncio
import os

import pytest

from app.generation.chain import answer_question
from app.generation.prompt_templates import ABSTAIN_MESSAGE

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run (uses real Gemini embedding quota)",
)


def test_keyword_entity_query_is_answered():
    # The exact failure that motivated hybrid search: a specific keyword/entity
    # question that pure dense search missed. Lexical search must rescue it.
    answer, chunks = asyncio.run(answer_question("who performed this experiment"))
    assert chunks, "hybrid retrieval returned no chunks for the experiment query"
    assert answer.strip() != ABSTAIN_MESSAGE


def test_broad_query_still_works():
    answer, chunks = asyncio.run(answer_question("summarize the document"))
    assert chunks
    assert answer.strip() != ABSTAIN_MESSAGE


def test_off_topic_query_abstains():
    answer, chunks = asyncio.run(answer_question("what is the capital of France"))
    assert chunks == []
    assert answer.strip() == ABSTAIN_MESSAGE
```

- [ ] **Step 3: Run it gated-off (default) to confirm it is skipped**

Run: `pytest tests/integration/test_rag_quality.py -v`
Expected: 3 skipped.

- [ ] **Step 4: Run it for real once the quota resets**

Run: `RUN_INTEGRATION=1 pytest tests/integration/test_rag_quality.py -v`
Expected: 3 passed. `test_keyword_entity_query_is_answered` passing is the acceptance proof that the original bug is fixed.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_rag_quality.py
git commit -m "Add quota-gated RAG-quality integration tests; purge test-junk docs

Proves the experiment/entity query is answered, broad queries still work, and
off-topic queries abstain. Skipped unless RUN_INTEGRATION=1.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Hybrid dense+lexical+RRF → Tasks 2–5. ✓
- `tsvector` generated column + GIN, `schema.sql` updated → Task 1. ✓
- Module split (`vector_search`/`lexical_search`/`fusion`/`hybrid`) → Tasks 2–5. ✓
- `chain.py` repointed → Task 5 Step 5. ✓
- Stable `id` key returned by both retrievers; fusion dedupes → Tasks 3, 4, 5. ✓
- Abstention only when both empty → Task 5 (`test_both_empty_returns_empty`). ✓
- Resilience: dense-fail→lexical, lexical-fail→dense → Task 5 tests. ✓
- `similarity` stays float (0.0 lexical-only; dense dict wins on shared id) → Tasks 3,4,5. ✓
- Tunables as constants (20/60/6/0.62) → Tasks 4,5 Global Constraints. ✓
- 1 embedding call/query, no new deps → dense path unchanged; lexical pure SQL. ✓
- Purge `quick.txt`/`big.txt`; quota-gated integration → Task 6. ✓

**Placeholder scan:** none — every step has concrete SQL/code/commands.

**Type consistency:** `dense_search(query, limit, min_similarity)` and `lexical_search(query, limit)` are called with those exact kwargs in `hybrid._hybrid_blocking` (`limit=CANDIDATES_PER_RETRIEVER`). `rrf_fuse(ranked_lists, k)` called with `k=RRF_K`. Result dicts use identical key sets across Tasks 3/4 and the `_chunk` test helper. `search`/`_hybrid_blocking` names match between Task 5 impl and tests. ✓

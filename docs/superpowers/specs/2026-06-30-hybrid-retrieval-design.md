# Hybrid Retrieval — Design Spec

**Date:** 2026-06-30
**Status:** Approved (brainstorming complete; pending implementation plan)
**Project:** B (retrieval quality). Project A (chat sessions + per-chat documents) is a separate, later spec.

## Problem

The RAG system answers broad queries well ("summarize the pdf") but fails on
specific keyword/entity queries. Concrete reproduction against the live corpus
(`AI_Unit 4.pdf`, `DBMS.pdf`):

- Query: **"who performed this experiment"** → system abstains.
- Evidence: exactly **1 chunk** in the corpus contains the word "experiment", so
  the answer *is* indexed. A keyword search finds it instantly.
- Cause: retrieval is **pure dense cosine** (pgvector `<=>`) with a `min_similarity`
  floor of 0.62 and `top_k=5`. Short factoid/entity questions embed poorly, so the
  one relevant chunk doesn't rank into the top-5 above the floor → no context →
  abstention. This is the textbook weakness of dense-only retrieval: strong on
  semantic/broad queries, weak on exact term/entity lookups.

## Goal

Make specific keyword/entity questions answerable while preserving current
behavior on broad queries and off-topic abstention — at **zero additional
per-query API cost** and **no new infrastructure**.

## Non-Goals (YAGNI)

- Re-chunking / smaller chunks (would require re-embedding the corpus — cost).
- Query expansion / HyDE (extra LLM call per query — rejected on cost).
- Cross-encoder reranking (needs a model — deferred).
- Chat sessions, per-chat document scoping, persistence — that is **Project A**,
  a separate spec.

## Chosen Approach

**Hybrid retrieval = dense (pgvector) + lexical (Postgres full-text), fused with
Reciprocal Rank Fusion (RRF).**

Rationale:
- Postgres full-text (`tsvector`/`websearch_to_tsquery`/`ts_rank_cd`) is built in —
  no new dependency, no new service, no API call. Lexical search directly fixes
  entity/keyword queries (it matches the "experiment" chunk).
- **RRF** fuses the two ranked lists by rank, not score. Dense cosine similarity
  and `ts_rank_cd` live on different scales; rank-based fusion avoids fragile
  score normalization. Standard constant `k = 60`.
- Still exactly **1 embedding call per query** (dense side, same as today).
  Lexical is free SQL. Net added API cost = 0.

Alternatives considered and rejected:
- **Trigram (`pg_trgm`) as primary** — fuzzy substring matching, weaker as a
  relevance ranker. Possible future complement, not the main fix.
- **External BM25 (Elasticsearch/OpenSearch)** — industrial-grade but a whole new
  service to run and keep in sync. Overkill for this project.

## Architecture

Refactor `app/retrieval/` from one module into focused, independently testable units:

| File | Responsibility | Depends on |
|------|----------------|-----------|
| `vector_search.py` | `dense_search()` — embed query, pgvector cosine search above floor | embedder, db pool |
| `lexical_search.py` | `lexical_search()` — Postgres full-text ranked search | db pool |
| `fusion.py` | `rrf_fuse()` — pure function: merge ranked lists via RRF | nothing (pure) |
| `hybrid.py` | public `search()` — run both retrievers, fuse, apply abstention, resilience | the three above |

`app/generation/chain.py` changes its import from
`app.retrieval.vector_search.search` to `app.retrieval.hybrid.search`. Public
signature stays `async search(query, top_k=...) -> list[chunk dict]`, so callers
and the chunk dict shape (`document_id, chunk_index, content, source_filename,
similarity`) are unchanged. `similarity` stays a plain `float` (the `routes.py`
`Source` model requires it): a fused chunk reports its dense cosine similarity if
it was found by the dense retriever, else `0.0` for a lexical-only hit. This keeps
the response schema unchanged and the UI source bars honest (a lexical-only hit
genuinely has no semantic-similarity score).

### Each retriever returns a stable key

Both retrievers return result dicts identified by a stable key — the chunk's
primary-key `id` (add `dc.id` to both SELECTs). RRF fuses on that key; after
fusion the merged dicts are materialized in fused-rank order. Using the real
`id` avoids ambiguity and lets fusion dedupe a chunk found by both retrievers.

## Schema Migration

New file `migrations/002_hybrid_fts.sql`:

```sql
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
  ON document_chunks USING GIN (content_tsv);
```

- **Generated STORED column**: auto-maintained from `content`, backfills existing
  rows on `ADD COLUMN`, and stays correct for all future inserts with **no change
  to ingestion code**. (Requires Postgres 12+; the pgvector image is 15/16.)
- `schema.sql` is also updated so fresh installs get the column + index.

## Data Flow (per query)

```
query
 ├─ embed query (1 Gemini call — SAME as today) → dense_search → top-N (N=20) above cosine floor 0.62
 └─ websearch_to_tsquery('english', query) → lexical_search → top-N (N=20) by ts_rank_cd
 → rrf_fuse([dense_ids, lexical_ids], k=60) → take top_k (=6) → build_prompt → generate
```

Tunables (module constants, single source of truth):
- `CANDIDATES_PER_RETRIEVER = 20`
- `RRF_K = 60`
- `FINAL_TOP_K = 6` (was 5 — slightly more room for the specific chunk)
- dense `MIN_SIMILARITY = 0.62` (unchanged)

## Abstention Rule

Abstain (return `[]`, which `chain.py` turns into the standard "I don't know based
on the provided context" message) **only when the fused result is empty**. Fused
is empty only when dense returns nothing above the floor AND lexical matches no
lexemes.

| Query | Dense | Lexical | Outcome |
|-------|-------|---------|---------|
| "who performed this experiment" | may miss top-N | hits "experiment" | answered (FIXED) |
| "summarize the pdf" | broad hits | hits | answered (unchanged) |
| "what is the capital of France" | < 0.62, blocked | no lexeme overlap → no match | abstain (unchanged) |

The dense floor stays at 0.62 to keep blocking off-topic dense noise (measured
off-topic dense scores ~0.57–0.59). Lexical is **purely additive recall**: it only
fires when actual words overlap, and it is self-guarding against off-topic queries
(an off-topic question shares no lexemes with the corpus, so it produces no lexical
hits). No floor retuning required. The LLM's own abstention instruction remains the
final safety net.

## Error Handling & Resilience

Each retriever is isolated in its own try/except inside `hybrid.search()`:

- **Lexical SQL error** (e.g. malformed tsquery) → log, fall back to dense-only
  results. Query still succeeds.
- **Dense embedding fails** (e.g. quota `429` / `DailyQuotaExceeded`, network) →
  fall back to **lexical-only** results instead of failing the request. Meaningful
  win: when the Gemini embedding quota is exhausted, keyword-based answers still
  work. (See `gemini-free-tier-quota` constraint.)
- **Both retrievers fail** → propagate so the API layer returns its generic 503 /
  the caller abstains.

## Testing Strategy

- **`fusion.rrf_fuse` — pure unit tests** (no DB/API): correct ordering, the RRF
  weighting formula `1/(k + rank)`, dedupe of a key present in both lists,
  tie-handling, empty-list inputs.
- **`lexical_search` — SQL-level test** (no embedding API, so runnable today
  despite exhausted quota): query "experiment" returns the known chunk; off-topic
  term returns nothing.
- **Resilience test**: monkeypatch dense to raise → assert `search()` returns the
  lexical-only results rather than propagating the exception.
- **Integration tests** (require embedding API; run when daily quota resets):
  - "who performed this experiment" → relevant chunk present in results, question answered.
  - "summarize the pdf" → still returns broad context (regression).
  - "what is the capital of France" → abstains (regression).
- **Pre-verification cleanup**: purge test-junk documents (`quick.txt`, `big.txt`)
  so the corpus contains only the real PDFs before running integration checks.

## Acceptance Criteria

1. `migrations/002_hybrid_fts.sql` applies cleanly; existing rows get a populated
   `content_tsv`; GIN index exists.
2. Lexical search alone retrieves the "experiment" chunk for the query
   "who performed this experiment".
3. After the quota resets, the full pipeline answers "who performed this
   experiment" from `AI_Unit 4.pdf`, while "summarize the pdf" still works and an
   off-topic question still abstains.
4. With the embedding call forced to fail, the system still answers keyword
   queries from lexical-only results (no 503).
5. No increase in per-query external API calls vs. the current system (still 1
   embedding call; 0 for lexical).

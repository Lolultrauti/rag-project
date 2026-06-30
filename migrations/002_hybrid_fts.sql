-- Hybrid retrieval: add a generated full-text column + GIN index so we can run
-- keyword search alongside the pgvector semantic search. STORED + GENERATED means
-- Postgres maintains it from content automatically -- existing rows are backfilled
-- on ADD COLUMN and all future inserts stay correct with no ingestion code change.
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
  ON document_chunks USING GIN (content_tsv);

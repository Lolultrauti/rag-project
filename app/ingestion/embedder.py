"""
embedder.py

Responsible for converting text chunks into vector embeddings using
Google's Gemini embedding model (gemini-embedding-001).

text-embedding-004 (the older model) was deprecated by Google on
Jan 14 2026, so we use the current recommended model instead. We pin
output_dimensionality=768 explicitly so vectors match our VECTOR(768)
schema column -- gemini-embedding-001 defaults to 3072 dims otherwise.

This file knows nothing about the database or file hashing -- it's a pure
function: text in, vectors out. That separation means we could swap in
OpenAI or HuggingFace here later by changing only this file.

Performance note:
  - Single queries still embed one string at a time.
  - Document ingestion batches many chunk strings into one API request
    where possible, which removes the biggest slowdown from large uploads.
"""

import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from app.config import settings

# Key now comes from the central settings module (validated at startup) rather
# than a per-file load_dotenv()/os.getenv(); see app/config.py.
client = genai.Client(api_key=settings.gemini_api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # must match the VECTOR(768) column in our schema
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_CHARS = 24000


def _embed_contents(contents, task_type: str, max_retries: int):
    """
    Shared embedding helper for both single-text and batched calls.

    The Gemini SDK accepts either one string or a list of strings as the
    `contents` payload. When given a list, it returns one embedding per item
    in the same order, which lets ingestion collapse many per-chunk requests
    into a much smaller number of API calls.
    """
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            embeddings = result.embeddings or []
            return [embedding.values for embedding in embeddings]
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            # Retry only transient failures:
            #   429 RESOURCE_EXHAUSTED -> per-minute quota tripped
            #   5xx (e.g. 503 UNAVAILABLE) -> transient server-side outage
            # Any other status is a real bug and should surface immediately.
            code = getattr(e, "code", None)
            transient = code == 429 or (code is not None and 500 <= code < 600)
            if not transient or attempt == max_retries - 1:
                raise
            backoff = 2 ** attempt * 10  # 10s, 20s, 40s, ... backoff
            print(f"  Transient API error {code}; backing off {backoff}s "
                  f"(retry {attempt + 1}/{max_retries - 1})...")
            time.sleep(backoff)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT",
               max_retries: int = 5):
    """
    Generates a single embedding vector for one piece of text.

    task_type matters: Gemini's embedding model produces slightly different
    vector spaces depending on whether the text is a document being indexed
    ("RETRIEVAL_DOCUMENT") or a user's search query ("RETRIEVAL_QUERY").
    Using the right task_type for each side measurably improves retrieval
    accuracy.

    The free tier caps embedding calls at 100/minute. We retry on HTTP 429
    (RESOURCE_EXHAUSTED) with exponential backoff so a burst that trips the
    per-minute quota recovers automatically instead of crashing a long
    ingestion run partway through.
    """
    vectors = _embed_contents(text, task_type=task_type, max_retries=max_retries)
    if not vectors:
        raise RuntimeError("Gemini returned no embedding vector.")
    return vectors[0]


def _iter_embedding_batches(chunks, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Yield batches that are small enough to stay responsive and quota-friendly.

    We cap both the number of chunks and the approximate payload size per
    request. That keeps a 2.4 MB PDF from turning into thousands of tiny
    round-trips, while still avoiding one giant request if a document has
    unusually dense text.
    """
    batch = []
    batch_chars = 0
    for chunk in chunks:
        text = chunk.page_content.replace("\x00", "")
        chunk_chars = len(text)
        if batch and (
            len(batch) >= batch_size or
            batch_chars + chunk_chars > MAX_BATCH_CHARS
        ):
            yield batch
            batch = []
            batch_chars = 0
        batch.append((chunk, text))
        batch_chars += chunk_chars
    if batch:
        yield batch


def embed_chunks(chunks, batch_size: int = DEFAULT_BATCH_SIZE,
                 max_retries: int = 5):
    """
    Generates embeddings for a list of LangChain chunk objects.
    Returns a list of (chunk, embedding_vector) tuples.

    The old implementation slept after every chunk, which made medium-sized
    PDFs feel very slow. We now batch multiple chunk texts into each Gemini
    request so the upload path is limited mainly by the embedding latency,
    not by an artificial per-chunk pause.
    """
    results = []
    total = len(chunks)
    embedded = 0
    for batch in _iter_embedding_batches(chunks, batch_size=batch_size):
        batch_chunks = [chunk for chunk, _ in batch]
        batch_texts = [text for _, text in batch]
        vectors = _embed_contents(
            batch_texts,
            task_type="RETRIEVAL_DOCUMENT",
            max_retries=max_retries,
        )
        if len(vectors) != len(batch_chunks):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for "
                f"{len(batch_chunks)} chunk(s)."
            )
        results.extend(zip(batch_chunks, vectors))
        embedded += len(batch_chunks)
        print(f"  Embedded {embedded}/{total} chunks...")
    return results


if __name__ == "__main__":
    sample = "This is a test sentence for embedding."
    vec = embed_text(sample)
    print(f"Embedding length: {len(vec)} (should be {EMBEDDING_DIM})")
    print(f"First 5 values: {vec[:5]}")

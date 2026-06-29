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
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return result.embeddings[0].values
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


def embed_chunks(chunks, batch_delay: float = 0.7):
    """
    Generates embeddings for a list of LangChain chunk objects.
    Returns a list of (chunk, embedding_vector) tuples.

    batch_delay=0.7s paces us at ~85 requests/minute, comfortably under the
    free tier's 100/min embedding cap so we usually never trip a 429 in the
    first place. embed_text() still retries with backoff as a safety net if
    we do.
    """
    results = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk.page_content, task_type="RETRIEVAL_DOCUMENT")
        results.append((chunk, vector))
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
        time.sleep(batch_delay)
    return results


if __name__ == "__main__":
    sample = "This is a test sentence for embedding."
    vec = embed_text(sample)
    print(f"Embedding length: {len(vec)} (should be {EMBEDDING_DIM})")
    print(f"First 5 values: {vec[:5]}")

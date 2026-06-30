"""
chain.py  --  the "G" in RAG (generation).

Takes a user question, retrieves supporting chunks, builds a grounded prompt
(prompt_templates), and asks Gemini's chat model to produce a final answer
constrained to that context.

We use "gemini-2.5-flash": the current fast/cheap generation model, which is
a good fit for RAG answer synthesis where we want low latency and the heavy
lifting (finding the right facts) has already been done by retrieval.

ASYNC: this is the live /query path. We use the google-genai *async* client
(client.aio) so the generation HTTP call is awaited natively and never blocks
the event loop. Retrieval (vector_search.search) is awaited too. The only
remaining blocking work (the query-side embedding and the DB query) is
already offloaded to a threadpool inside vector_search, so the whole path is
non-blocking end to end.
"""

import asyncio
import re

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from app.config import settings
from app.retrieval.hybrid import search, FINAL_TOP_K
from app.generation.prompt_templates import (
    SYSTEM_INSTRUCTION,
    ABSTAIN_MESSAGE,
    build_prompt,
)

client = genai.Client(api_key=settings.gemini_api_key)

GENERATION_MODEL = "gemini-2.5-flash"

# Retry tuning for the GENERATION call. This mirrors embedder.embed_text's
# transient-error detection (429 + 5xx), but with deliberately different
# timing: embedder runs in an offline batch job where 10/20/40s backoff is
# fine, whereas this runs on a synchronous user-facing request. A user will
# not wait ~70s for a request that ultimately fails, so we use short backoff
# (1s, 2s) and few attempts -- enough to ride out a brief blip, not so much
# that we hold the request open forever. We also asyncio.sleep (not
# time.sleep) so waiting doesn't block the event loop.
_GEN_MAX_RETRIES = 3
_GEN_BASE_BACKOFF = 1  # seconds: 1s, 2s, ...


def maybe_handle_small_talk(question: str):
    """
    Return a friendly direct reply for greetings and other lightweight chat.

    This keeps the assistant interactive instead of forcing everything
    through retrieval, which would make simple messages like "hi" look like
    failures when there is no document context yet.
    """
    normalized = re.sub(r"[^a-z0-9\s]+", " ", question.lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return None

    greetings = {
        "hi",
        "hello",
        "hey",
        "hiya",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if normalized in greetings:
        return (
            "Hi! I'm Helix. Ask me about a document you uploaded, or say "
            "something like 'summarize this PDF' or 'find mentions of revenue'. "
            "What would you like to explore?"
        )

    if normalized in {"help", "what can you do", "who are you", "what are you"}:
        return (
            "I'm Helix, a document assistant. I can search your uploaded files, "
            "summarize sections, and pull out specific facts. Try asking a "
            "question about the documents you've added."
        )

    if normalized in {"how are you", "how are you doing"}:
        return "I'm ready to help. Ask me something about your documents."

    if normalized in {"thanks", "thank you", "thx", "ty"}:
        return "You're welcome. Ask me anything about your documents whenever you're ready."

    return None


async def generate_answer(prompt: str) -> str:
    """
    Low-level call: send a fully-built prompt to Gemini, return the text.

    temperature=0.0 keeps answers deterministic and faithful -- we do not
    want creative paraphrasing in a grounded QA system.

    Retries transient failures (429 RESOURCE_EXHAUSTED, 5xx) with short
    exponential backoff; any other error (or exhausting retries) re-raises so
    the API layer can turn it into a clean 503.
    """
    for attempt in range(_GEN_MAX_RETRIES):
        try:
            response = await client.aio.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
            return (response.text or "").strip()
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            code = getattr(e, "code", None)
            transient = code == 429 or (code is not None and 500 <= code < 600)
            if not transient or attempt == _GEN_MAX_RETRIES - 1:
                raise
            backoff = _GEN_BASE_BACKOFF * (2 ** attempt)  # 1s, 2s, ...
            await asyncio.sleep(backoff)


async def answer_question(question: str, top_k: int = FINAL_TOP_K):
    """
    Full RAG pass: retrieve -> build grounded prompt -> generate.

    Returns (answer_text, retrieved_chunks) so callers (e.g. the API layer)
    can surface the source passages alongside the answer for transparency.

    If the input is just a greeting or other lightweight chat, we return a
    friendly direct response immediately and skip retrieval entirely. That
    keeps the app feeling conversational and avoids wasting a search call on
    messages that do not need document grounding.

    If retrieval returns nothing above the similarity floor, we short-circuit:
    abstain immediately with the standard message and skip the LLM call
    entirely. There is no context to reason over, so spending a generation
    call (latency + cost) would be pure waste -- and the answer is already
    known. chunks is [] in that case, so the API reports no sources.
    """
    small_talk = maybe_handle_small_talk(question)
    if small_talk is not None:
        return small_talk, []

    chunks = await search(question, top_k=top_k)
    if not chunks:
        return ABSTAIN_MESSAGE, chunks

    prompt = build_prompt(question, chunks)
    answer = await generate_answer(prompt)
    return answer, chunks


if __name__ == "__main__":
    # Verifiable from INDEX 1.pdf (Digital Image Processing notes).
    # answer_question is async now; drive it via asyncio.run.
    async def _demo():
        question = ("What are the two components of the decoder in the image "
                    "compression model?")
        print(f"Q: {question}\n")

        answer, chunks = await answer_question(question)
        print(f"A: {answer}\n")

        print("--- Sources used ---")
        for i, c in enumerate(chunks, 1):
            preview = " ".join(c["content"].split())[:120]
            print(f"[{i}] sim={c['similarity']:.3f} chunk={c['chunk_index']}: {preview}")

        # Negative control: a question the PDF can't answer should abstain.
        off_topic = "What is the capital of France?"
        ans2, _ = await answer_question(off_topic)
        print(f"\n--- Abstention test ---\nQ: {off_topic}\nA: {ans2}")

    asyncio.run(_demo())

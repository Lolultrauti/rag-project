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

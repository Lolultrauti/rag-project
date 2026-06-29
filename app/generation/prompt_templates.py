"""
prompt_templates.py  --  how we instruct the LLM to stay grounded.

The whole point of RAG is that the model answers from *retrieved evidence*,
not from its own parametric memory (which may be outdated, wrong, or simply
not cover our private documents). The prompt is where we enforce that.

Key design choices in the template below:
  - The context is clearly delimited so the model knows exactly what counts
    as "the provided context".
  - We explicitly forbid using outside knowledge.
  - We give an exact escape hatch -- "I don't know based on the provided
    context" -- so that when the answer genuinely isn't in the retrieved
    chunks, the model abstains instead of hallucinating. An abstention is a
    correct, useful answer in a RAG system; a confident fabrication is the
    worst possible outcome.
"""

SYSTEM_INSTRUCTION = (
    "You are a precise question-answering assistant for an enterprise "
    "document search system. You answer strictly and only from the context "
    "passages provided to you."
)

# The exact abstention string. Defined as a named constant so it has a single
# source of truth: the LLM is instructed to emit it verbatim (see the template
# below), AND the retrieval/generation layer returns it directly when there is
# no context worth sending to the LLM at all. Both paths must produce the same
# wording, so they share this constant rather than duplicating the literal.
ABSTAIN_MESSAGE = "I don't know based on the provided context."

ANSWER_PROMPT_TEMPLATE = """\
Answer the QUESTION using ONLY the information in the CONTEXT passages below.

Rules:
- Use only facts stated in the CONTEXT. Do not use prior knowledge.
- If the CONTEXT does not contain enough information to answer, reply with
  exactly: "{abstain_message}"
- Do not invent citations, numbers, or facts that are not in the CONTEXT.
- Be concise and directly answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def format_context(chunks) -> str:
    """
    Renders retrieved chunks into a numbered, labeled block for the prompt.

    `chunks` is the list of dicts returned by vector_search.search().
    Numbering each passage makes it easy for the model (and us, when
    debugging) to reason about which passage supports which claim.
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        text = " ".join(c["content"].split())  # collapse whitespace/newlines
        blocks.append(f"[Passage {i}] {text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks) -> str:
    """Fills the answer template with the formatted context and question."""
    return ANSWER_PROMPT_TEMPLATE.format(
        context=format_context(chunks),
        question=question,
        abstain_message=ABSTAIN_MESSAGE,
    )

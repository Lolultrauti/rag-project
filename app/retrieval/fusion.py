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

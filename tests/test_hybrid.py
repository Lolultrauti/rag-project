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

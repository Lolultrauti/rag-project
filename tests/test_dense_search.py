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

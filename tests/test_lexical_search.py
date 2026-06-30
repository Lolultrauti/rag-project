from app.retrieval.lexical_search import lexical_search


def test_finds_chunk_by_keyword(seeded):
    seeded("exp.txt", [
        (0, "The water cycle describes evaporation and condensation."),
        (1, "The experiment was performed by Dr. Mendel in 1865."),
    ])
    results = lexical_search("who performed this experiment")
    contents = [r["content"] for r in results]
    assert any("Dr. Mendel" in c for c in contents)


def test_result_shape(seeded):
    seeded("shape.txt", [(0, "Photosynthesis converts sunlight into energy.")])
    results = lexical_search("photosynthesis")
    assert results, "expected at least one match"
    r = results[0]
    assert set(r) == {"id", "document_id", "chunk_index", "content",
                      "source_filename", "similarity"}
    assert r["similarity"] == 0.0
    assert r["source_filename"] == "shape.txt"


def test_off_topic_returns_nothing(seeded):
    seeded("topic.txt", [(0, "Relational databases use primary keys.")])
    assert lexical_search("xyzzy plugh nonsense") == []


def test_respects_limit(seeded):
    seeded("many.txt", [(i, f"alpha token number {i}") for i in range(5)])
    assert len(lexical_search("alpha", limit=2)) <= 2

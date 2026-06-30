from app.retrieval.fusion import rrf_fuse


def test_single_list_preserves_order():
    assert rrf_fuse([[3, 1, 2]]) == [3, 1, 2]


def test_key_in_both_lists_ranks_highest():
    # 1 appears near the top of both lists -> should win.
    dense = [1, 2, 3]
    lexical = [1, 4, 5]
    assert rrf_fuse([dense, lexical])[0] == 1


def test_dedupes_keys():
    out = rrf_fuse([[1, 2], [2, 1]])
    assert sorted(out) == [1, 2]
    assert len(out) == 2


def test_empty_lists_return_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_score_formula_and_tie_break():
    # Two keys, each rank-0 in exactly one list -> equal scores 1/(k+1).
    # Tie-break is key ascending, so 1 before 9.
    assert rrf_fuse([[1], [9]], k=60) == [1, 9]


def test_higher_rank_beats_lower_rank_across_lists():
    # 'a' is rank 0 in list1; 'b' is rank 1 in list1 and rank 0 in list2.
    # b score = 1/62 + 1/61 ; a score = 1/61. b should win.
    out = rrf_fuse([["a", "b"], ["b"]], k=60)
    assert out[0] == "b"

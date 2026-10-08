from app.services.retrieval import reciprocal_rank_fusion


def test_rrf_promotes_consensus():
    result = reciprocal_rank_fusion(
        ["a", "b", "c"],
        ["b", "a", "d"],
        k=1,
    )
    assert result[0][0] == "a"
    assert result[1][0] == "b"

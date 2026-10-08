from app.services.ingestion import chunk_text, normalize_text


def test_normalize_text_collapses_noise():
    assert normalize_text("hello   world\n\n\nnext") == "hello world\n\nnext"


def test_chunk_text_bounds():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert all(1 <= len(chunk) <= 500 for chunk in chunks)


def test_invalid_overlap_rejected():
    try:
        chunk_text("hello", 10, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

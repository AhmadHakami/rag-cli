from rag_app.text_splitter import split_pages_into_chunks


def test_splitter_basic():
    pages = ["A" * 120, "B" * 80]
    chunks = split_pages_into_chunks("doc", pages, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 4
    assert all(1 <= c.page_number <= 2 for c in chunks)

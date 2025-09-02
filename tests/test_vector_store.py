import numpy as np
from rag_app.vector_store import InMemoryVectorStore
from rag_app.interfaces import DocumentChunk


def test_vector_store_search():
    store = InMemoryVectorStore()
    chunks = [
        DocumentChunk("doc", 1, "hello world"),
        DocumentChunk("doc", 2, "goodbye world"),
        DocumentChunk("doc", 3, "another test"),
    ]
    embeds = np.eye(3, dtype=np.float32)
    store.add(embeds, chunks)
    res = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert len(res) == 2
    assert res[0].chunk.text == "hello world"

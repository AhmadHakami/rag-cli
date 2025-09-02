from rag_app.embeddings import MockEmbedder
from rag_app.llm import MockChat, QueryRewriter, AnswerGenerator
from rag_app.vector_store import InMemoryVectorStore
from rag_app.agent import AgenticRAG


def test_agent_flow_with_mock(tmp_path):
    # fake pdf content by bypassing ingest_pdf splitter dependency with small pages
    agent = AgenticRAG(
        embedder=MockEmbedder(dim=8),
        llm=MockChat(),
        vector_store=InMemoryVectorStore(),
        rewriter=QueryRewriter(MockChat()),
        answerer=AnswerGenerator(MockChat()),
        top_k=2,
    )
    # directly add chunks to store
    from rag_app.interfaces import DocumentChunk
    chunks = [DocumentChunk("doc", 1, "alpha"), DocumentChunk("doc", 2, "beta")]
    embeds = agent.embedder.embed_texts([c.text for c in chunks])
    agent.store.add(embeds, chunks)

    ans = agent.ask("What is alpha?")
    assert isinstance(ans, str)

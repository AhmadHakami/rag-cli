from __future__ import annotations

import argparse
from .config import SETTINGS
from .providers import build_providers
from .vector_store import InMemoryVectorStore
from .agent import AgenticRAG


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG over a PDF with page citations")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--provider", choices=["openai", "ollama", "mock"], default=SETTINGS.provider)
    parser.add_argument("--doc-id", default="doc")
    parser.add_argument("--top-k", type=int, default=SETTINGS.top_k)
    parser.add_argument("--chunk-size", type=int, default=SETTINGS.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=SETTINGS.chunk_overlap)
    args = parser.parse_args()

    embedder, chat, rewriter, answerer = build_providers(args.provider)
    store = InMemoryVectorStore()
    agent = AgenticRAG(embedder, chat, store, rewriter, answerer, top_k=args.top_k)

    total = agent.ingest_pdf(args.pdf, doc_id=args.doc_id, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if total == 0:
        print("No content ingested from the PDF.")
        return

    print("Type your questions (Ctrl+C to exit). Type 'summarize' for document overview:")
    try:
        while True:
            q = input("\nQ> ").strip()
            if not q:
                continue
            if q.lower() == "summarize":
                summary = agent.summarize_document()
                print(f"\nSummary:\n{summary}")
            else:
                a = agent.ask(q)
                print(f"\nA> {a}")
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    main()

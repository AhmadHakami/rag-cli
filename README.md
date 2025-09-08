# RAG Application (Agentic Follow-up QA with Page Citations)

An advanced, SOLID-architected Retrieval Augmented Generation (RAG) app that:

- Answers questions strictly based on provided PDF documents
- Includes page numbers for each cited source
- Supports follow-up questions via query rewriting
- Compatible with OpenAI and Ollama models (switchable via config or CLI)
- Shows a progress bar for all ingestion and query steps
- Comes with unit tests and mock providers for offline testing

## Features

- PDF ingestion with page-aware chunking
- Embedding-based retrieval (OpenAI or Ollama; Ollama fallback to OpenAI/Mock if embeddings API not available)
- LLM-based query rewriting and relevance grading (Ollama chat falls back to /api/generate when /api/chat is missing)
- Answer generation constrained to retrieved content with citations
- Clean provider interfaces (SOLID) and dependency inversion
- CLI for interactive Q&A over your PDFs

## Quick Start

1) Install dependencies with python 3.10

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure model provider

- OpenAI: set environment variable `OPENAI_API_KEY`.
- Ollama: ensure `ollama` is running locally (default `http://localhost:11434`). For embeddings support you need a recent Ollama and an embedding model, e.g.:
	- `ollama pull nomic-embed-text`
	- If `/api/embeddings` is not available, the app will fall back to OpenAI (if `OPENAI_API_KEY` set) or a Mock embedder.

You can configure via `.env` (optional):

```
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

3) Run the CLI

```bash
python -m rag_app.main --pdf docs/your.pdf --provider openai
# or
python -m rag_app.main --pdf docs/your.pdf --provider ollama
```

Then type your questions. Follow-ups are supported. Answers include page citations.

## Tests

Run tests with:

```bash
pytest -q
```

## Structure

- `rag_app/config.py` – configuration and settings
- `rag_app/interfaces.py` – provider interfaces and core contracts
- `rag_app/pdf_loader.py` – PDF page extraction
- `rag_app/text_splitter.py` – page-aware chunking
- `rag_app/embeddings.py` – embedding providers (OpenAI, Ollama, Mock)
- `rag_app/vector_store.py` – simple cosine similarity index
- `rag_app/llm.py` – chat LLM providers, query rewriter, grader, answerer
- `rag_app/agent.py` – orchestrates the end-to-end RAG flow with progress bars
- `rag_app/main.py` – CLI entry point
- `tests/` – unit tests for splitter, store, and agent



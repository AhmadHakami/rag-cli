from __future__ import annotations

from .config import SETTINGS
from .embeddings import OpenAIEmbedder, OllamaEmbedder, MockEmbedder, FallbackEmbedder
from .llm import OpenAIChat, OllamaChat, MockChat, QueryRewriter, AnswerGenerator, FallbackChat


def build_providers(provider: str | None = None):
    provider = (provider or SETTINGS.provider).lower()
    if provider == "openai":
        embedder = OpenAIEmbedder()
        chat = OpenAIChat()
    elif provider == "ollama":
        # Prefer Ollama embeddings, but fallback to OpenAI then Mock if unavailable
        embedder = FallbackEmbedder([
            OllamaEmbedder(),
            OpenAIEmbedder(),
            MockEmbedder(),
        ])
        chat = FallbackChat([
            OllamaChat(),
            OpenAIChat(),
            MockChat(),
        ])
    else:
        embedder = MockEmbedder()
        chat = MockChat()
    rewriter = QueryRewriter(chat)
    answerer = AnswerGenerator(chat)
    return embedder, chat, rewriter, answerer

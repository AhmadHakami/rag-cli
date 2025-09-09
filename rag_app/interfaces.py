from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    page_number: int  # 1-based
    text: str
    section: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float  # similarity score


class IEmbedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return 2D float32 array shape (n, d)."""
        ...


class IChatLLM(Protocol):
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        ...


class IVectorStore(Protocol):
    def add(self, embeddings: np.ndarray, metadatas: Sequence[DocumentChunk]) -> None:
        ...

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        ...


class IQueryRewriter(Protocol):
    def rewrite(self, question: str, chat_history: Sequence[Tuple[str, str]] | None = None) -> str:
        ...


class IAnswerGenerator(Protocol):
    def answer(self, question: str, contexts: Sequence[DocumentChunk], chat_history: Sequence[Tuple[str, str]] | None = None) -> str:
        """Generate an answer using provided contexts.

        Args:
            question: The (possibly rewritten) user question.
            contexts: Sequence of DocumentChunk to use as evidence.
            chat_history: Optional prior (question, answer) pairs to support follow-ups.

        Returns:
            A string reply. Implementations should include inline citations where appropriate.
        """
        ...

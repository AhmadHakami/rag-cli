from __future__ import annotations

from typing import List, Dict, Any, Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .interfaces import IVectorStore, DocumentChunk, RetrievalResult


class InMemoryVectorStore(IVectorStore):
    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.metadatas: List[DocumentChunk] = []
        self.metadata: Dict[str, Any] = {}

    def add(self, embeddings: np.ndarray, metadatas: Sequence[DocumentChunk]) -> None:
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Embeddings and metadatas length mismatch")
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.metadatas.extend(list(metadatas))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        if self.embeddings is None or len(self.metadatas) == 0:
            return []
        q = query_embedding.reshape(1, -1)
        sims = cosine_similarity(q, self.embeddings)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        results: List[RetrievalResult] = []
        for idx in idxs:
            results.append(RetrievalResult(chunk=self.metadatas[int(idx)], score=float(sims[int(idx)])))
        return results

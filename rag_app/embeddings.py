from __future__ import annotations

from typing import Sequence, List
import numpy as np
import requests
from .config import SETTINGS
from .interfaces import IEmbedder


class OpenAIEmbedder(IEmbedder):
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or SETTINGS.openai_embedding_model
        self.api_key = api_key or SETTINGS.openai_api_key
        self.base_url = (base_url or SETTINGS.openai_base_url or "https://api.openai.com/v1").rstrip("/")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "input": list(texts)}
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        vecs = [d["embedding"] for d in resp.json()["data"]]
        return np.array(vecs, dtype=np.float32)


class OllamaEmbedder(IEmbedder):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or SETTINGS.ollama_embedding_model
        self.base_url = (base_url or SETTINGS.ollama_base_url).rstrip("/")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        url = f"{self.base_url}/api/embeddings"
        vecs: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "prompt": t}
            resp = requests.post(url, json=payload, timeout=120)
            # Provide clearer hint if embeddings API is unavailable (older Ollama)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                if resp.status_code == 404:
                    raise RuntimeError(
                        "Ollama embeddings endpoint not found. Upgrade Ollama and pull an embedding model, e.g.:\n"
                        "  ollama pull nomic-embed-text\n"
                        "Ensure your Ollama version supports /api/embeddings."
                    ) from e
                raise
            vecs.append(resp.json()["embedding"])
        return np.array(vecs, dtype=np.float32)


class MockEmbedder(IEmbedder):
    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        # hash-based deterministic pseudo-embeddings for tests
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            self.rng.bit_generator.state  # keep rng available
            self.rng = np.random.default_rng(h % (2**32))
            arr[i] = self.rng.standard_normal(self.dim).astype(np.float32)
        return arr


class FallbackEmbedder(IEmbedder):
    """Try multiple embedders in order until one succeeds.

    Useful when Ollama's embeddings API is unavailable; falls back to OpenAI or Mock.
    """

    def __init__(self, embedders: List[IEmbedder]):
        if not embedders:
            raise ValueError("FallbackEmbedder requires at least one embedder")
        self._embedders = embedders

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        last_err: Exception | None = None
        for emb in self._embedders:
            try:
                return emb.embed_texts(texts)
            except Exception as e:  # noqa: BLE001 - bubble last error after trying others
                last_err = e
                continue
        raise RuntimeError(f"All embedders failed: {last_err}")

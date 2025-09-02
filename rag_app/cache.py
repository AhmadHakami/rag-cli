from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict


_CACHE_DIR_NAME = ".rag_cache"


def _project_root() -> Path:
    # rag_app/ -> project root
    return Path(__file__).resolve().parent.parent


def get_cache_dir() -> Path:
    env = os.getenv("RAG_APP_CACHE_DIR")
    if env:
        p = Path(env)
    else:
        p = _project_root() / _CACHE_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_pdf_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_cache_key(pdf_hash: str, doc_id: str, chunk_size: int, embedder_fingerprint: str) -> str:
    base = f"v1|{pdf_hash}|{doc_id}|{chunk_size}|{embedder_fingerprint}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def cache_path_for(key: str) -> Path:
    return get_cache_dir() / key


def has_cache(key: str) -> bool:
    d = cache_path_for(key)
    return (d / "data.pkl").exists()


def save_cache(key: str, payload: Dict[str, Any]) -> None:
    d = cache_path_for(key)
    d.mkdir(parents=True, exist_ok=True)
    tmp_path = d / "data.pkl.tmp"
    final_path = d / "data.pkl"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, final_path)


def load_cache(key: str) -> Dict[str, Any]:
    with open(cache_path_for(key) / "data.pkl", "rb") as f:
        return pickle.load(f)


def invalidate_cache(key: str) -> None:
    d = cache_path_for(key)
    try:
        (d / "data.pkl").unlink(missing_ok=True)
    except Exception:
        pass

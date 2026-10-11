"""Lazy, cached loaders for local embedding and cross-encoder models."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Some embedders are trained with an instruction prefix on the query side only.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
}
DOC_PREFIXES = {
    "intfloat/e5-small-v2": "passage: ",
}


@lru_cache(maxsize=4)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


class Embedder:
    def __init__(self, name: str, batch_size: int = 64):
        self.name = name
        self.batch_size = batch_size

    def _encode(self, texts: list[str], prefix: str, progress: bool = False) -> np.ndarray:
        model = _st_model(self.name)
        vecs = model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                            normalize_embeddings=True, show_progress_bar=progress,
                            convert_to_numpy=True)
        return vecs.astype(np.float32)

    def embed_docs(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, DOC_PREFIXES.get(self.name, ""), progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, QUERY_PREFIXES.get(self.name, ""))


class Reranker:
    def __init__(self, name: str):
        self.name = name

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(_cross_encoder(self.name).predict([(query, p) for p in passages]))

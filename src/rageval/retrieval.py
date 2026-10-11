"""Indexing and retrieval: dense (Chroma), sparse (BM25), hybrid (reciprocal rank fusion),
with an optional cross-encoder rerank stage on top."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_corpus
from .config import Config
from .ingest import load_corpus
from .models import Embedder, Reranker

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or that the this "
    "to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float


def load_or_build_chunks(cfg: Config) -> list[Chunk]:
    path = Path(cfg.index_dir) / "chunks" / f"{cfg.chunk_key()}.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return [Chunk(**json.loads(line)) for line in f]
    docs = load_corpus(Path(cfg.corpus)).values()
    embed = None
    if cfg.chunking.strategy == "semantic":
        embed = Embedder(cfg.chunking.breakpoint_model).embed_docs
    chunks = chunk_corpus(docs, cfg.chunking.strategy, cfg.chunking.params, embed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return chunks


class DenseIndex:
    def __init__(self, cfg: Config, chunks: list[Chunk]):
        import chromadb

        self.embedder = Embedder(cfg.retrieval.embedding_model)
        self.by_id = {c.chunk_id: c for c in chunks}
        client = chromadb.PersistentClient(path=str(Path(cfg.index_dir) / "chroma"))
        model_tag = hashlib.sha1(cfg.retrieval.embedding_model.encode()).hexdigest()[:8]
        name = f"c{cfg.chunk_key()}_{model_tag}"
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if self.col.count() != len(chunks):
            self._build(chunks, client, name)

    def _build(self, chunks: list[Chunk], client, name: str) -> None:
        client.delete_collection(name)
        self.col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        vecs = self.embedder.embed_docs([c.text for c in chunks], progress=True)
        step = 4000
        for i in range(0, len(chunks), step):
            batch = chunks[i:i + step]
            self.col.add(ids=[c.chunk_id for c in batch], embeddings=vecs[i:i + step].tolist(),
                         metadatas=[{"doc_id": c.doc_id} for c in batch])

    def search(self, query: str, n: int) -> list[Hit]:
        q = self.embedder.embed_queries([query])
        res = self.col.query(query_embeddings=q.tolist(), n_results=n, include=["distances"])
        return [Hit(self.by_id[i], 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, n: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:n]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


def rrf(rankings: list[list[Hit]], k: int, n: int) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    chunk_of: dict[str, Chunk] = {}
    for hits in rankings:
        for rank, h in enumerate(hits, start=1):
            scores[h.chunk.chunk_id] = scores.get(h.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_of[h.chunk.chunk_id] = h.chunk
    order = sorted(scores, key=scores.get, reverse=True)[:n]
    return [Hit(chunk_of[cid], scores[cid]) for cid in order]


class Retriever:
    """Builds the indexes a config needs and returns top-k hits for a query."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        r = cfg.retrieval
        self.chunks = load_or_build_chunks(cfg)
        self.dense = DenseIndex(cfg, self.chunks) if r.retriever in ("dense", "hybrid") else None
        self.bm25 = BM25Index(self.chunks) if r.retriever in ("bm25", "hybrid") else None
        self.reranker = Reranker(cfg.rerank.model) if cfg.rerank.model else None

    def candidates(self, query: str) -> list[Hit]:
        r = self.cfg.retrieval
        if r.retriever == "dense":
            return self.dense.search(query, r.candidates)
        if r.retriever == "bm25":
            return self.bm25.search(query, r.candidates)
        if r.retriever == "hybrid":
            lists = [self.dense.search(query, r.candidates), self.bm25.search(query, r.candidates)]
            return rrf(lists, r.rrf_k, r.candidates)
        raise ValueError(f"unknown retriever: {r.retriever}")

    def retrieve(self, query: str) -> list[Hit]:
        hits = self.candidates(query)
        if self.reranker is not None:
            scores = self.reranker.score(query, [h.chunk.text for h in hits])
            hits = [Hit(h.chunk, float(s)) for h, s in zip(hits, scores)]
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.cfg.retrieval.top_k]

"""Pipeline configuration. One YAML file describes one point in the ablation grid."""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"  # fixed | semantic
    params: dict[str, Any] = field(default_factory=lambda: {"size": 1000, "overlap": 150})
    # Model used to find semantic breakpoints (independent of the retrieval embedder).
    breakpoint_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievalConfig:
    retriever: str = "dense"  # dense | bm25 | hybrid
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    candidates: int = 50  # pool size fed to fusion / reranking
    top_k: int = 5
    rrf_k: int = 60


@dataclass
class RerankConfig:
    model: str | None = None  # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2; None disables


@dataclass
class GenerationConfig:
    provider: str = "ollama"  # ollama | openai_compatible | none
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None
    context_k: int = 5  # retrieved chunks passed to the LLM
    temperature: float = 0.0
    max_tokens: int = 400
    # USD per 1M tokens; 0 for local models. Used for the cost-per-query column.
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0


@dataclass
class Config:
    name: str = "baseline"
    corpus: str = "data/processed/corpus.jsonl"
    index_dir: str = "indexes"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = copy.deepcopy(d)
        return cls(
            name=d.get("name", "baseline"),
            corpus=d.get("corpus", cls.corpus),
            index_dir=d.get("index_dir", cls.index_dir),
            chunking=ChunkingConfig(**d.get("chunking", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
            rerank=RerankConfig(**d.get("rerank", {})),
            generation=GenerationConfig(**d.get("generation", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def chunk_key(self) -> str:
        """Identifies a chunk set: changes only when corpus or chunking changes."""
        c = self.chunking
        payload = {"corpus": _file_hash(self.corpus), "strategy": c.strategy, "params": c.params}
        if c.strategy == "semantic":
            payload["breakpoint_model"] = c.breakpoint_model
        return _short_hash(payload)


@functools.lru_cache(maxsize=8)
def _file_hash(path: str) -> str:
    """Content hash, so editing or rebuilding the corpus invalidates cached chunks and indexes."""
    p = Path(path)
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else path


def _short_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_grid(grid_path: str | Path) -> list[Config]:
    """Expand an ablation YAML (``base`` + ``axes``) into the cartesian product of configs.

    Each axis is a list of named options; an option is a dict of dotted-path overrides.
    """
    spec = yaml.safe_load(Path(grid_path).read_text(encoding="utf-8"))
    base = spec.get("base", {})
    axes: dict[str, list[dict]] = spec["axes"]
    configs = []
    for combo in itertools.product(*axes.values()):
        d = copy.deepcopy(base)
        names = []
        for option in combo:
            names.append(option["name"])
            for path, value in option.get("set", {}).items():
                _set_path(d, path, value)
        d["name"] = "__".join(names)
        configs.append(Config.from_dict(d))
    return configs

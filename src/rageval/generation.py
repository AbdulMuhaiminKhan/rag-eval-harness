"""LLM access for answer generation and judging. Defaults to a local Ollama model ($0);
any OpenAI-compatible endpoint can be configured instead."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

from .config import GenerationConfig
from .retrieval import Hit

ABSTAIN = "I don't know based on the provided documentation."

ANSWER_PROMPT = """You answer questions about Kubernetes using only the numbered context passages.
Cite the passages you used like [1] or [2]. If the passages do not contain the answer, reply exactly:
"{abstain}"

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class LLMResult:
    text: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


class LLM:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    def complete(self, prompt: str, json_mode: bool = False) -> LLMResult:
        c = self.cfg
        t0 = time.perf_counter()
        if c.provider == "ollama":
            body = {"model": c.model, "prompt": prompt, "stream": False,
                    "options": {"temperature": c.temperature, "num_predict": c.max_tokens}}
            if json_mode:
                body["format"] = "json"
            data = _post(f"{c.base_url}/api/generate", body)
            text, tin, tout = data["response"], data.get("prompt_eval_count", 0), data.get(
                "eval_count", 0)
        elif c.provider == "openai_compatible":
            headers = {}
            if c.api_key_env:
                headers["Authorization"] = f"Bearer {os.environ[c.api_key_env]}"
            body = {"model": c.model, "temperature": c.temperature, "max_tokens": c.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            data = _post(f"{c.base_url}/chat/completions", body, headers)
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        else:
            raise ValueError(f"unknown provider: {c.provider}")
        cost = tin * c.price_in_per_m / 1e6 + tout * c.price_out_per_m / 1e6
        return LLMResult(text.strip(), time.perf_counter() - t0, tin, tout, cost)


def _post(url: str, body: dict, headers: dict | None = None, retries: int = 4) -> dict:
    """POST with exponential backoff on connection errors and 5xx/429 (e.g. a model reloading)."""
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=600)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r.json()
            err: Exception = requests.HTTPError(f"{r.status_code} from {url}: {r.text[:200]}")
        except requests.ConnectionError as e:
            err = e
        if attempt == retries:
            raise err
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def answer(llm: LLM, question: str, hits: list[Hit]) -> LLMResult:
    prompt = ANSWER_PROMPT.format(abstain=ABSTAIN, context=format_context(hits), question=question)
    return llm.complete(prompt)


def is_abstention(text: str) -> bool:
    t = text.lower()
    return "i don't know" in t or "i do not know" in t or "does not contain" in t

import math

import numpy as np

from rageval.chunking import Chunk
from rageval.dataset import Evidence, find_span
from rageval.judge import cohens_kappa
from rageval.metrics import bootstrap_ci, covers, gains, mrr_at_k, ndcg_at_k, retrieval_scores


def ch(doc, s, e):
    return Chunk(f"{doc}#{s}", doc, s, e, "x" * (e - s))


def ev(doc, s, e):
    return Evidence(doc, "", s, e)


def test_covers_requires_same_doc_and_overlap():
    assert covers(ch("a", 0, 100), ev("a", 10, 50))
    assert not covers(ch("b", 0, 100), ev("a", 10, 50))
    assert not covers(ch("a", 0, 100), ev("a", 100, 150))
    # 20 of 100 chars of the span inside a 1000-char chunk: below 50% of the shorter side
    assert not covers(ch("a", 0, 1000), ev("a", 980, 1080))


def test_each_span_is_credited_once():
    spans = [ev("a", 0, 100)]
    ranked = [ch("a", 0, 60), ch("a", 40, 100)]
    assert gains(ranked, spans) == [1, 0]


def test_rank_metrics():
    g = [0, 1, 0, 1]
    assert mrr_at_k(g, 10) == 0.5
    assert mrr_at_k(g, 1) == 0.0
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(g, 2, 10), expected)


def test_retrieval_scores_multi_span():
    spans = [ev("a", 0, 100), ev("b", 0, 100)]
    ranked = [ch("c", 0, 100), ch("a", 0, 100), ch("z", 0, 10)]
    s = retrieval_scores(ranked, spans, ks=(1, 3))
    assert s["recall@1"] == 0.0
    assert s["recall@3"] == 0.5
    assert s["doc_recall@3"] == 0.5
    assert s["mrr@3"] == 0.5


def test_bootstrap_ci_brackets_mean():
    x = np.r_[np.ones(30), np.zeros(70)]
    m, lo, hi = bootstrap_ci(x, seed=1)
    assert m == 0.3 and lo < 0.3 < hi


def test_find_span_tolerates_whitespace():
    text = "Pods are the smallest\n  deployable units."
    assert find_span(text, "smallest deployable units") == (13, 40)
    assert find_span(text, "not there") is None


def test_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0


def test_alternative_span_counts_as_covered():
    gold = Evidence("a", "", 0, 100, alternatives=[Evidence("b", "", 200, 260)])
    assert covers(ch("b", 180, 280), gold)
    s = retrieval_scores([ch("b", 180, 280)], [gold], ks=(1,))
    assert s["recall@1"] == 1.0 and s["doc_recall@1"] == 1.0


def test_trace_records_nested_spans():
    from rageval.tracing import NULL_TRACE, Trace

    t = Trace()
    with t.span("retrieve"):
        with t.span("rerank", candidates=50):
            pass
    names = [s["name"] for s in t.to_dict()]
    assert names == ["retrieve", "retrieve/rerank"]
    assert t.to_dict()[1]["attrs"] == {"candidates": 50}
    with NULL_TRACE.span("x"):
        pass
    assert NULL_TRACE.spans == []

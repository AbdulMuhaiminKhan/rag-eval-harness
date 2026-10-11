from pathlib import Path

import numpy as np

from rageval.chunking import fixed_chunks, semantic_chunks
from rageval.config import expand_grid
from rageval.ingest import Document, clean_markdown, path_to_doc_id
from rageval.retrieval import Hit, rrf

TEXT = ("Kubernetes runs containers. " * 40 + "\n\n" + "A Service exposes Pods. " * 40).strip()
DOC = Document("concepts/x", "X", "", TEXT)


def test_fixed_chunks_offsets_match_text():
    chunks = fixed_chunks(DOC, size=300, overlap=50)
    assert chunks[0].start == 0 and chunks[-1].end == len(TEXT)
    for c in chunks:
        assert TEXT[c.start:c.end] == c.text
        assert len(c.text) <= 300
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlapping windows


def test_semantic_chunks_split_on_topic_change():
    def fake_embed(texts):
        v = np.array([[1.0, 0.0] if "containers" in t else [0.0, 1.0] for t in texts])
        return v

    chunks = semantic_chunks(DOC, fake_embed, breakpoint_percentile=50, min_size=10,
                             max_size=5000)
    assert len(chunks) == 2
    assert "Service" not in chunks[0].text and "containers" not in chunks[1].text


def test_clean_markdown_strips_hugo_and_links():
    raw = ('---\ntitle: Pods\n---\n<!-- c -->\n{{< note >}}\nSee [the docs](/x) about '
           '{{< glossary_tooltip text="Pods" term_id="pod" >}}.\n{{< /note >}}\n')
    meta, text = clean_markdown(raw)
    assert meta["title"] == "Pods"
    assert text == "See the docs about Pods."


def test_doc_ids():
    assert path_to_doc_id(Path("concepts/workloads/pods/_index.md")) == "concepts/workloads/pods"
    assert path_to_doc_id(Path("tasks/run.md")) == "tasks/run"


def test_rrf_prefers_items_in_both_lists():
    from rageval.chunking import Chunk

    a, b, c = (Chunk(i, "d", 0, 1, i) for i in "abc")
    fused = rrf([[Hit(a, 1), Hit(b, 1)], [Hit(b, 1), Hit(c, 1)]], k=60, n=3)
    assert fused[0].chunk.chunk_id == "b"


def test_ablation_grid_expands():
    cfgs = expand_grid(Path(__file__).parents[1] / "configs" / "ablation.yaml")
    assert len(cfgs) == 24
    assert len({c.name for c in cfgs}) == 24


def test_clean_markdown_fills_versions_and_keeps_placeholders():
    raw = ('{{% heading "prerequisites" %}}\nUpgrade to {{< skew currentVersion >}} from '
           '{{< skew currentVersionAddMinor -1 >}}. Like a {{< glossary_tooltip '
           'term_id="deployment" >}}, run `kubectl get pod <pod-name>`.<br/> a&#58;b')
    _, text = clean_markdown(raw, version="v1.37")
    assert text == ("Before you begin\nUpgrade to 1.37 from 1.36. Like a deployment, "
                    "run `kubectl get pod <pod-name>`. a:b")

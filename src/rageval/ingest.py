"""Download and clean the Kubernetes documentation into a JSONL corpus.

The corpus is the English docs tree of github.com/kubernetes/website. We do a shallow,
sparse clone so only ``content/en/docs`` is fetched, record the exact commit in a manifest,
and convert each Markdown page into plain-ish text. Character offsets into that cleaned text
are what chunks and gold evidence spans refer to, so cleaning must be deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_URL = "https://github.com/kubernetes/website.git"
DOCS_SUBDIR = "content/en/docs"
SITE_ROOT = "https://kubernetes.io/docs/"


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    text: str


def clone_docs(raw_dir: Path, ref: str = "main") -> str:
    """Fetch only the docs tree at ``ref`` (a branch, tag or commit SHA); return the commit SHA.

    An existing checkout is reused as-is unless a different ref is requested.
    """
    repo = raw_dir / "kubernetes-website"
    if (repo / ".git").exists():
        head = _git("-C", str(repo), "rev-parse", "HEAD").strip()
        if ref in ("main", head):
            return head
    else:
        repo.mkdir(parents=True, exist_ok=True)
        _git("-C", str(repo), "init", "-q")
        _git("-C", str(repo), "remote", "add", "origin", REPO_URL)
        _git("-C", str(repo), "sparse-checkout", "set", DOCS_SUBDIR)
    _git("-C", str(repo), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ref)
    _git("-C", str(repo), "checkout", "-q", "FETCH_HEAD")
    return _git("-C", str(repo), "rev-parse", "HEAD").strip()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo shortcodes such as {{< note >}}, {{% /tab %}}, {{< glossary_tooltip text="Pod" term_id="pod" >}}
_GLOSSARY = re.compile(r"\{\{<\s*glossary_tooltip[^>]*?text=\"([^\"]+)\"[^>]*>\}\}")
_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_BLANKS = re.compile(r"\n{3,}")


def clean_markdown(raw: str) -> tuple[dict, str]:
    """Return (front_matter, cleaned_text). Code blocks are kept: they carry real answers."""
    meta: dict = {}
    m = _FRONT_MATTER.match(raw)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        raw = raw[m.end():]
    text = _HTML_COMMENT.sub("", raw)
    text = _GLOSSARY.sub(r"\1", text)
    text = _SHORTCODE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _BLANKS.sub("\n\n", text).strip()
    return meta, text


def path_to_doc_id(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts[-1] in ("_index", "index"):
        parts = parts[:-1]
    return "/".join(parts) or "home"


def build_corpus(raw_dir: Path, out_dir: Path, ref: str = "main", min_chars: int = 200) -> dict:
    sha = clone_docs(raw_dir, ref)
    docs_root = raw_dir / "kubernetes-website" / DOCS_SUBDIR
    docs: list[Document] = []
    for path in sorted(docs_root.rglob("*.md")):
        meta, text = clean_markdown(path.read_text(encoding="utf-8", errors="replace"))
        if meta.get("draft") or len(text) < min_chars:
            continue
        doc_id = path_to_doc_id(path.relative_to(docs_root))
        title = str(meta.get("title") or doc_id.rsplit("/", 1)[-1])
        docs.append(Document(doc_id, title, SITE_ROOT + doc_id + "/", text))

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    manifest = {
        "source": REPO_URL,
        "subdir": DOCS_SUBDIR,
        "commit": sha,
        "num_docs": len(docs),
        "num_chars": sum(len(d.text) for d in docs),
        "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_corpus(path: Path) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = Document(**json.loads(line))
            docs[d.doc_id] = d
    return docs

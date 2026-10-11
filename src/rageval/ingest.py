"""Download and clean the Kubernetes documentation into a JSONL corpus.

The corpus is the English docs tree of github.com/kubernetes/website. We do a shallow,
sparse clone so only ``content/en/docs`` is fetched, record the exact commit in a manifest,
and convert each Markdown page into plain-ish text. Character offsets into that cleaned text
are what chunks and gold evidence spans refer to, so cleaning must be deterministic.
"""

from __future__ import annotations

import hashlib
import html
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
_GLOSSARY = re.compile(r"\{\{<\s*glossary_tooltip\s([^>]*?)>\}\}")
_SKEW = re.compile(r"\{\{<\s*skew\s+(\w+)(?:\s+(-?\d+))?(?:\s+\"([^\"]*)\")?\s*>\}\}")
_HEADING = re.compile(r"\{\{%\s*heading\s+\"(\w+)\"\s*%\}\}")
HEADINGS = {
    "prerequisites": "Before you begin", "whatsnext": "What's next", "objectives": "Objectives",
    "cleanup": "Clean up", "synopsis": "Synopsis", "options": "Options",
    "parentoptions": "Options inherited from parent commands", "seealso": "See also",
    "examples": "Examples", "envvars": "Environment variables",
}
_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
# Only real HTML tags: placeholders like `kubectl get pod <pod-name>` must survive.
_HTML_TAG = re.compile(
    r"</?(?:a|b|i|em|strong|code|pre|p|br|hr|div|span|sup|sub|img|ul|ol|li|table|thead|tbody|"
    r"tr|td|th|details|summary|blockquote|h[1-6]|caption|center|section|figure|figcaption)"
    r"(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)
_BLANKS = re.compile(r"\n{3,}")


def _glossary(m: re.Match) -> str:
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    return attrs.get("text") or attrs.get("term_id", "").replace("-", " ")


def _skew_fn(version: str):
    major, minor = (int(x) for x in version.lstrip("v").split(".")[:2])

    def sub(m: re.Match) -> str:
        name, delta, sep = m.group(1), int(m.group(2) or 0), m.group(3) or "."
        offsets = {"currentVersion": 0, "latestVersion": 0, "prevMinorVersion": -1,
                   "nextMinorVersion": 1, "currentVersionAddMinor": delta}
        if name == "currentPatchVersion":
            return f"{major}.{minor}.x"
        if name not in offsets:
            return ""
        return f"{major}{sep}{minor + offsets[name]}"

    return sub


def site_version(repo: Path) -> str | None:
    """The docs' current Kubernetes version (e.g. "v1.37") from the site's hugo.toml."""
    cfg = repo / "hugo.toml"
    if not cfg.exists():
        return None
    m = re.search(r'^latest\s*=\s*"(v[\d.]+)"', cfg.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else None


def clean_markdown(raw: str, version: str | None = None) -> tuple[dict, str]:
    """Return (front_matter, cleaned_text). Code blocks are kept: they carry real answers.

    ``version`` fills version shortcodes such as {{< skew currentVersion >}}."""
    meta: dict = {}
    m = _FRONT_MATTER.match(raw)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        raw = raw[m.end():]
    text = _HTML_COMMENT.sub("", raw)
    text = _GLOSSARY.sub(_glossary, text)
    if version:
        text = _SKEW.sub(_skew_fn(version), text)
    text = _HEADING.sub(lambda m: HEADINGS.get(m.group(1), m.group(1).title()), text)
    text = _SHORTCODE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = html.unescape(_HTML_TAG.sub("", text))
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
    version = site_version(raw_dir / "kubernetes-website")
    docs_root = raw_dir / "kubernetes-website" / DOCS_SUBDIR
    docs: list[Document] = []
    for path in sorted(docs_root.rglob("*.md")):
        meta, text = clean_markdown(path.read_text(encoding="utf-8", errors="replace"), version)
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
        "kubernetes_version": version,
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

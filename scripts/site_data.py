"""The companion site's single source of truth: one validated YAML per model.

Every model page on the site (and, later, every generated README) is rendered from
``docs/models/<id>/model.yaml``. This script is the only gate between those files and the site:

* ``check``  validates every model file and the registry, and fails on
  - a schema violation (``docs/models/_schema.json`` is generated from the models below);
  - a source that does not resolve: a key in ``research/papers/index.yaml``, or ``repo:<path>`` for a file
    tracked in this repository;
  - a benchmark number that is not tied to the file it came from, or that differs from it at the precision
    the site shows, or that is tied to the wrong model (identity is checked), or whose rank is wrong;
  - a table that mixes harnesses, or a tied file produced by a different harness commit;
  - a correction whose struck and inserted text is not what the cited commit changed;
  - registry/page mismatches, and ``planned`` coverage on a published page.
* ``export`` writes the validated, source-resolved data as JSON into ``site/src/data/`` for Astro, with every
  number pre-formatted, so the site never formats a figure itself.
* ``schema`` rewrites ``docs/models/_schema.json``.

    uv run python scripts/site_data.py check
    uv run python scripts/site_data.py export
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "docs" / "models"
REGISTRY = MODELS / "_registry.yaml"
GLOSSARY = ROOT / "docs" / "glossary.yaml"
SCHEMA = MODELS / "_schema.json"
INDEX = ROOT / "research" / "papers" / "index.yaml"
META = ROOT / "research" / "papers" / "metadata.json"
OUT = ROOT / "site" / "src" / "data"
REPO_URL = "https://github.com/sourav-dhar/system-one-from-scratch/blob/main/"

Provenance = Literal["independent", "self-reported", "code", "ours", "inferred"]
Family = Literal[
    "biencoder",
    "late-interaction",
    "decoder-readout",
    "encoder",
    "encoder-decoder",
    "looped",
    "frozen-reader",
    "generative",
    "undisclosed",
    "ours",
]
Tier = Literal["A", "B", "C"]
# Sections of the model page template. `covered` coverage may point only at sections the site renders today.
Section = Literal[
    "spec",
    "summary",
    "architecture",
    "inference",
    "equations",
    "training",
    "data",
    "calibration",
    "benchmarks",
    "hardware",
    "try-it",
    "limits",
    "references",
]
RENDERED: frozenset[str] = frozenset({"spec", "summary", "benchmarks", "references"})
# The precision the site shows for each benchmark field. Checks compare at exactly this precision.
DISPLAY: dict[str, int] = {"accuracy": 2, "accuracy_ci": 2, "brier": 4, "p50_ms": 1}
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    """Where a fact comes from: a library key, or ``repo:<path>`` for a tracked file in this repository."""

    key: str
    at: str | None = Field(None, description="Locator inside the source: file, section, table, issue.")


class Fact(Strict):
    """One sentence, with every kind of evidence it rests on."""

    text: str
    provenance: list[Provenance] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)


class SpecItem(Strict):
    label: str
    value: str
    provenance: Provenance
    sources: list[Source] = Field(min_length=1)


class CheckRef(Strict):
    """Ties a displayed number to the tracked file it came from.

    ``pointer`` mode: an RFC 6901 JSON pointer into ``file``; the value times ``scale`` must equal the shown
    number at the site's display precision. ``match`` (pointer -> expected value) checks identity on the entry
    the value comes from: its pointers are relative to the last array element on ``pointer``'s path (or to the
    document root when the path has no array). A pointer through an array must give ``match``, so a row is
    tied to the right entry and not just to a position.

    ``text`` mode: for figures that exist only in a generated report. The shown value, formatted exactly as
    the site shows it, must occur in ``file`` as a whole number on a line that also contains ``line``.
    """

    file: str
    pointer: str | None = None
    scale: float = 1.0
    match: dict[str, str] = Field(default_factory=dict)
    text: bool = False
    line: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> CheckRef:
        if self.text == (self.pointer is not None):
            raise ValueError("give exactly one of `pointer` or `text: true`")
        if self.text and not self.line:
            raise ValueError("`text` checks need `line`: a string identifying the row in the report")
        return self


class BenchRow(Strict):
    label: str
    rank: str | None = None
    readout: str
    accuracy: float | None = None
    accuracy_ci: tuple[float, float] | None = None
    brier: float | None = None
    p50_ms: float | None = None
    measured_on: str
    provenance: Provenance
    sources: list[Source] = Field(min_length=1)
    harness: str = Field(pattern=r"^[a-z0-9-]+@[0-9a-f]{7,40}$")
    checks: dict[str, CheckRef] = Field(
        description="Field name -> the file that must reproduce the shown number."
    )
    focus: bool = False
    child: bool = False

    @model_validator(mode="after")
    def _numbers_are_tied(self) -> BenchRow:
        unknown = set(self.checks) - set(DISPLAY)
        if unknown:
            raise ValueError(f"row {self.label!r}: `checks` names unknown fields {sorted(unknown)}")
        shown = [k for k in DISPLAY if getattr(self, k) is not None]
        missing = [k for k in shown if k not in self.checks]
        if missing:
            raise ValueError(f"row {self.label!r} must tie {missing} to a file")
        for k in shown:
            values = getattr(self, k) if k == "accuracy_ci" else [getattr(self, k)]
            if any(float(v) != round(float(v), DISPLAY[k]) for v in values):
                raise ValueError(f"row {self.label!r}: `{k}` carries more precision than the site shows")
        if self.accuracy_ci is not None:
            lo, hi = self.accuracy_ci
            if not (self.accuracy is not None and lo <= self.accuracy <= hi):
                raise ValueError(f"row {self.label!r}: interval {self.accuracy_ci} must contain the accuracy")
        if self.rank is not None and not re.fullmatch(r"[1-9]\d* / [1-9]\d*", self.rank):
            raise ValueError(f"row {self.label!r}: rank must read 'N / M'")
        return self


class Reference(Strict):
    row: str = Field(
        description="Label of a row in this table whose checked accuracy draws the reference line."
    )
    label: str


class BenchTable(Strict):
    suite: str
    title: str
    caption: str
    caption_sources: list[Source] = Field(min_length=1)
    axis_max: float = 100.0
    reference: Reference | None = None
    notes: list[Fact] = Field(default_factory=list)
    rows: list[BenchRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> BenchTable:
        harnesses = {r.harness for r in self.rows}
        if len(harnesses) != 1:
            raise ValueError(f"table {self.title!r} mixes harnesses {sorted(harnesses)}; same-harness only")
        top = max((r.accuracy for r in self.rows if r.accuracy is not None), default=0.0)
        if top > self.axis_max:
            raise ValueError(f"table {self.title!r}: accuracy {top} exceeds the bar axis {self.axis_max}")
        if self.reference is not None:
            hits = [r for r in self.rows if r.label == self.reference.row]
            if len(hits) != 1 or hits[0].accuracy is None:
                raise ValueError(f"table {self.title!r}: reference row {self.reference.row!r} must be unique")
        return self


class Segment(Strict):
    kind: Literal["keep", "del", "ins"]
    text: str


class Correction(Strict):
    """A published correction, shown in place: struck text stays readable beside its replacement.

    ``keep`` + ``del`` must reproduce text in ``file`` before ``commit``; ``keep`` + ``ins`` text in it after.
    """

    date: date
    commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    file: str
    where: str
    context: str | None = Field(None, description="What the corrected passage is about.")
    segments: list[Segment] = Field(min_length=2)
    provenance: Provenance
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def _changes_something(self) -> Correction:
        if not any(s.kind != "keep" for s in self.segments):
            raise ValueError("a correction must delete or insert something")
        return self


class Coverage(Strict):
    """One element of the primary sources and where the page explains it.

    ``covered``: explained in ``section``, which the site must render. ``planned``: assigned to ``section``
    but not written yet (allowed only while the page is a draft). ``n/a``: left out, with a ``reason``.
    """

    element: str
    source: Source
    status: Literal["covered", "planned", "n/a"]
    section: Section | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _mapped(self) -> Coverage:
        if self.status == "n/a" and not self.reason:
            raise ValueError(f"coverage {self.element!r}: `n/a` needs a `reason`")
        if self.status != "n/a" and not self.section:
            raise ValueError(f"coverage {self.element!r}: `{self.status}` needs a `section`")
        if self.status == "covered" and self.section not in RENDERED:
            raise ValueError(f"coverage {self.element!r}: section {self.section!r} is not rendered yet")
        return self


class Claim(Strict):
    text: str
    sources: list[Source] = Field(min_length=1)
    verified_by: str
    date: date


class Summary(Strict):
    what: list[Fact] = Field(min_length=1)
    why: list[Fact] = Field(min_length=1)
    weakness: list[Fact] = Field(min_length=1)


class ModelPage(Strict):
    """``docs/models/<id>/model.yaml``."""

    id: str = Field(pattern=SLUG.pattern)
    name: str
    version: str | None = None
    full_name: str
    tier: Tier
    family: Family
    note: str = Field(description="Repo path of the research note this page is drawn from.")
    eyebrow: list[str] = Field(default_factory=list)
    dek: str
    dek_sources: list[Source] = Field(min_length=1)
    primer: list[str] = Field(min_length=1, description="Glossary ids shown as 'Read this first'.")
    spec: list[SpecItem] = Field(min_length=1)
    summary: Summary
    benchmarks: list[BenchTable] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    coverage: list[Coverage] = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)


class Term(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    term: str
    definition: str
    sources: list[Source] = Field(min_length=1)


class Glossary(Strict):
    terms: list[Term]


class RegistryEntry(Strict):
    id: str = Field(pattern=SLUG.pattern)
    name: str
    tier: Tier
    family: Family
    note: str | None = None
    status: Literal["planned", "draft", "published"] = "planned"


class Registry(Strict):
    systems: list[RegistryEntry]


# ------------------------------------------------------------------------------------------------
# repository files, sources and JSON pointers
# ------------------------------------------------------------------------------------------------
def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


@cache
def tracked_files() -> frozenset[str]:
    return frozenset(git("ls-files", "-z").split("\0")) - {""}


def repo_path(path: str) -> str | None:
    """``path`` as a normalised repo-relative POSIX path if it names a tracked file, else None."""
    if not path or "\\" in path:
        return None
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or str(p) in {".", ""}:
        return None
    return str(p) if str(p) in tracked_files() else None


def library() -> dict[str, dict[str, Any]]:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    out: dict[str, dict[str, Any]] = {}
    for e in index["arxiv"]:
        title = (meta.get(e["id"]) or {}).get("title") or e.get("system") or e["id"]
        out[e["key"]] = {"title": title, "url": f"https://arxiv.org/abs/{e['id']}", "kind": "arxiv"}
    for e in index["links"]:
        out[e["key"]] = {"title": e.get("title", e["key"]), "url": e["url"], "kind": "link"}
    return out


def resolve(src: Source, lib: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The source as ``{key, at, title, url, kind}``, or None when it does not resolve."""
    if src.key.startswith("repo:"):
        path = repo_path(src.key[len("repo:") :])
        if path is None:
            return None
        return {"key": src.key, "at": src.at, "title": path, "url": REPO_URL + path, "kind": "repo"}
    entry = lib.get(src.key)
    if entry is None:
        return None
    return {"key": src.key, "at": src.at, **entry}


def sources_in(obj: Any) -> list[Source]:
    if isinstance(obj, Source):
        return [obj]
    if isinstance(obj, BaseModel):
        return [s for name in type(obj).model_fields for s in sources_in(getattr(obj, name))]
    if isinstance(obj, list | tuple):
        return [s for v in obj for s in sources_in(v)]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in sources_in(v)]
    return []


ARRAY_INDEX = re.compile(r"^(0|[1-9]\d*)$")


def pointer_tokens(pointer: str) -> list[str]:
    """RFC 6901 tokens; raises ValueError on a malformed pointer."""
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer {pointer!r} must start with '/'")
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]


def json_pointer(doc: Any, pointer: str) -> Any:
    cur = doc
    for token in pointer_tokens(pointer):
        if isinstance(cur, list):
            if not ARRAY_INDEX.match(token) or int(token) >= len(cur):
                raise ValueError(f"{pointer}: bad array index {token!r}")
            cur = cur[int(token)]
        elif isinstance(cur, dict):
            if token not in cur:
                raise ValueError(f"{pointer}: no key {token!r}")
            cur = cur[token]
        else:
            raise ValueError(f"{pointer}: cannot index into {type(cur).__name__}")
    return cur


def anchor(doc: Any, pointer: str) -> tuple[Any, bool]:
    """The entry a value belongs to: the last array element on the pointer's path, else the document.

    Returns ``(entry, passed_through_array)``. Assumes the pointer resolves (call ``json_pointer`` first).
    """
    cur, entry, through = doc, doc, False
    for token in pointer_tokens(pointer):
        if isinstance(cur, list):
            cur = cur[int(token)]
            entry, through = cur, True
        else:
            cur = cur[token]
    return entry, through


def fmt(value: float, decimals: int) -> str:
    """The site's formatting: round half up at the display precision."""
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def display(row: BenchRow) -> dict[str, str]:
    out: dict[str, str] = {}
    if row.accuracy is not None:
        out["accuracy"] = fmt(row.accuracy, DISPLAY["accuracy"])
    if row.accuracy_ci is not None:
        lo, hi = (fmt(v, DISPLAY["accuracy_ci"]) for v in row.accuracy_ci)
        out["accuracy_ci"] = f"{lo}–{hi}"
    if row.brier is not None:
        out["brier"] = fmt(row.brier, DISPLAY["brier"])
    if row.p50_ms is not None:
        out["p50_ms"] = f"{fmt(row.p50_ms, DISPLAY['p50_ms'])} ms"
    return out


@cache
def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def file_harness(doc: Any) -> str | None:
    """The harness commit a results file records, if any."""
    if not isinstance(doc, dict):
        return None
    suite, source = doc.get("suite"), doc.get("source")
    if isinstance(suite, dict) and isinstance(suite.get("harness_commit"), str):
        return str(suite["harness_commit"])
    if isinstance(source, dict) and isinstance(source.get("commit"), str):
        return str(source["commit"])
    return None


def check_row(row: BenchRow) -> list[str]:
    """Each tied figure must equal its file, at the precision shown, for the right entry."""
    errors: list[str] = []
    shown = display(row)
    harness_sha = row.harness.split("@", 1)[1]
    for field, ref in row.checks.items():
        where = f"{row.label}.{field}"
        path = repo_path(ref.file)
        if path is None:
            errors.append(f"{where}: {ref.file} is not a tracked file")
            continue
        if field not in shown:
            errors.append(f"{where}: tied, but the row shows no value")
            continue
        if ref.text:
            want = shown[field].removesuffix(" ms")
            pattern = re.compile(rf"(?<![\d.]){re.escape(want)}(?![\d])")
            text = (ROOT / path).read_text(encoding="utf-8")
            lines = [ln for ln in text.splitlines() if ref.line and ref.line in ln]
            if not lines:
                errors.append(f"{where}: no line of {path} contains {ref.line!r}")
            elif not any(pattern.search(ln) for ln in lines):
                errors.append(f"{where}: {want!r} does not occur on the {ref.line!r} line of {path}")
            continue
        if field == "accuracy_ci":
            errors.append(f"{where}: intervals must use `text` checks")
            continue
        assert ref.pointer is not None
        try:
            doc = load_json(path)
            raw = json_pointer(doc, ref.pointer)
            entry, through = anchor(doc, ref.pointer)
            for p, expected in ref.match.items():
                got = json_pointer(entry, p)
                if got != expected:
                    errors.append(f"{where}: identity {p} is {got!r}, expected {expected!r}")
            if through and not ref.match:
                errors.append(
                    f"{where}: pointer {ref.pointer} goes through an array; add `match` for identity"
                )
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"{where}: {exc}")
            continue
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            errors.append(f"{where}: {path}{ref.pointer} is {raw!r}, not a number")
            continue
        want = fmt(float(Decimal(str(raw)) * Decimal(str(ref.scale))), DISPLAY[field])
        if shown[field].removesuffix(" ms") != want:
            errors.append(f"{where}: page shows {shown[field]}, {path}{ref.pointer} gives {want}")
        recorded = file_harness(doc)
        if recorded is not None and not recorded.startswith(harness_sha):
            errors.append(f"{where}: {path} was produced by harness {recorded[:7]}, row says {harness_sha}")
    errors += check_rank(row)
    return errors


def check_rank(row: BenchRow) -> list[str]:
    """`rank` ("N / M") must be the entry's position by accuracy among the file's ranked entries."""
    if row.rank is None:
        return []
    ref = row.checks.get("accuracy")
    if ref is None or ref.pointer is None:
        return [f"{row.label}: `rank` needs a pointer-checked accuracy"]
    path = repo_path(ref.file)
    if path is None:
        return []  # reported by check_row
    try:
        tokens = pointer_tokens(ref.pointer)
        if len(tokens) < 2 or not ARRAY_INDEX.match(tokens[-2]):
            return [f"{row.label}: `rank` needs an accuracy pointer of the form /<list>/<index>/<field>"]
        doc = load_json(path)
        parent = "/" + "/".join(tokens[:-2])
        container = json_pointer(doc, parent) if tokens[:-2] else doc
    except ValueError as exc:
        return [f"{row.label}: {exc}"]
    field, index = tokens[-1], int(tokens[-2])
    if not isinstance(container, list) or index >= len(container):
        return [f"{row.label}: entry {index} not found"]
    ranked = [e for e in container if isinstance(e, dict) and isinstance(e.get(field), int | float)]
    if container[index] not in ranked:
        return [f"{row.label}: entry {index} has no {field}"]
    position = sorted(ranked, key=lambda e: -float(e[field])).index(container[index]) + 1
    expected = f"{position} / {len(ranked)}"
    return [] if row.rank == expected else [f"{row.label}: rank {row.rank!r}, file gives {expected!r}"]


def normalise(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def check_correction(c: Correction) -> list[str]:
    path = repo_path(c.file)
    if path is None:
        return [f"correction {c.commit}: {c.file} is not a tracked file"]
    before = normalise("".join(s.text for s in c.segments if s.kind != "ins"))
    after = normalise("".join(s.text for s in c.segments if s.kind != "del"))
    try:
        old = normalise(git("show", f"{c.commit}^:{path}"))
        new = normalise(git("show", f"{c.commit}:{path}"))
    except RuntimeError as exc:
        return [f"correction {c.commit}: {exc}"]
    errors: list[str] = []
    if before not in old:
        errors.append(f"correction {c.commit}: struck text is not in {path} before the commit")
    if after not in new:
        errors.append(f"correction {c.commit}: corrected text is not in {path} after the commit")
    return errors


def load_models() -> tuple[list[ModelPage], list[str]]:
    pages: list[ModelPage] = []
    errors: list[str] = []
    for path in sorted(MODELS.glob("*/model.yaml")):
        try:
            page = ModelPage.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except ValidationError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        if page.id != path.parent.name:
            errors.append(f"{path.relative_to(ROOT)}: id {page.id!r} differs from its folder")
        pages.append(page)
    return pages, errors


def check_registry(registry: Registry, pages: list[ModelPage]) -> list[str]:
    errors: list[str] = []
    ids = [e.id for e in registry.systems]
    errors += [f"registry: duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    with_page = {p.id: p for p in pages}
    for e in registry.systems:
        if e.note and repo_path(e.note) is None:
            errors.append(f"registry: {e.id}: note {e.note} is not a tracked file")
        if e.status != "planned" and e.id not in with_page:
            errors.append(f"registry: {e.id} is {e.status} but has no model.yaml")
        if e.status == "planned" and e.id in with_page:
            errors.append(f"registry: {e.id} has a model.yaml but is still `planned`")
    for page in pages:
        where = f"docs/models/{page.id}/model.yaml"
        entry = next((e for e in registry.systems if e.id == page.id), None)
        if entry is None:
            errors.append(f"{where}: not in the registry")
            continue
        if (entry.tier, entry.family) != (page.tier, page.family):
            errors.append(f"{where}: tier/family differ from the registry")
        full = f"{page.name}{page.version or ''}"
        if full != entry.name:
            errors.append(f"{where}: name {full!r} differs from the registry's {entry.name!r}")
        if entry.note != page.note:
            errors.append(f"{where}: note differs from the registry")
        if entry.status == "published":
            open_items = [c.element for c in page.coverage if c.status == "planned"]
            if open_items:
                errors.append(f"{where}: published page still has planned coverage: {open_items}")
    return errors


def load_glossary() -> Glossary:
    return Glossary.model_validate(yaml.safe_load(GLOSSARY.read_text(encoding="utf-8")))


def check() -> list[str]:
    try:
        registry = Registry.model_validate(yaml.safe_load(REGISTRY.read_text(encoding="utf-8")))
        glossary = load_glossary()
    except ValidationError as exc:
        return [f"registry or glossary: {exc}"]
    try:
        tracked_files()
    except (RuntimeError, OSError) as exc:
        return [f"git is required to check sources: {exc}"]
    pages, errors = load_models()
    errors += check_registry(registry, pages)
    lib = library()
    term_ids = [t.id for t in glossary.terms]
    errors += [
        f"glossary: duplicate id {i!r}" for i in sorted({i for i in term_ids if term_ids.count(i) > 1})
    ]
    for term in glossary.terms:
        for src in term.sources:
            if resolve(src, lib) is None:
                errors.append(f"glossary: {term.id}: source {src.key!r} does not resolve")
    for page in pages:
        where = f"docs/models/{page.id}/model.yaml"
        if repo_path(page.note) is None:
            errors.append(f"{where}: note {page.note} is not a tracked file")
        errors += [
            f"{where}: primer term {t!r} is not in docs/glossary.yaml"
            for t in page.primer
            if t not in term_ids
        ]
        for src in sources_in(page):
            if resolve(src, lib) is None:
                errors.append(f"{where}: source {src.key!r} does not resolve")
        for table in page.benchmarks:
            for row in table.rows:
                errors += [f"{where}: {e}" for e in check_row(row)]
        for c in page.corrections:
            errors += [f"{where}: {e}" for e in check_correction(c)]
    return errors


# ------------------------------------------------------------------------------------------------
# export
# ------------------------------------------------------------------------------------------------
def export() -> None:
    errors = check()
    if errors:
        raise SystemExit("site data is invalid; run `site_data.py check`:\n" + "\n".join(errors))
    lib = library()
    registry = Registry.model_validate(yaml.safe_load(REGISTRY.read_text(encoding="utf-8")))
    terms = {t.id: t for t in load_glossary().terms}
    pages, _ = load_models()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for stale in (OUT / "models").glob("*.json"):
        stale.unlink()
    for page in pages:
        data = page.model_dump(mode="json")
        data["primer"] = [
            {
                **terms[t].model_dump(mode="json", exclude={"sources"}),
                "sources": [s.model_dump() for s in terms[t].sources],
            }
            for t in page.primer
        ]
        for t_data, table in zip(data["benchmarks"], page.benchmarks, strict=True):
            for r_data, row in zip(t_data["rows"], table.rows, strict=True):
                r_data["display"] = display(row)
                r_data.pop("checks")
            if table.reference is not None:
                ref_row = next(r for r in table.rows if r.label == table.reference.row)
                t_data["reference"]["value"] = ref_row.accuracy
        refs: dict[str, dict[str, Any]] = {}
        for src in [*sources_in(page), *(s for t in page.primer for s in terms[t].sources)]:
            got = resolve(src, lib)
            assert got is not None
            refs.setdefault(src.key, {k: got[k] for k in ("key", "title", "url", "kind")})
        data["references"] = sorted(refs.values(), key=lambda r: (r["kind"] != "repo", r["key"]))
        (OUT / "models" / f"{page.id}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    published = {p.id for p in pages}
    systems = [{**e.model_dump(mode="json"), "has_page": e.id in published} for e in registry.systems]
    (OUT / "registry.json").write_text(
        json.dumps(systems, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"exported {len(pages)} model page(s) and {len(systems)} registry entries to {OUT}")


def schema_text() -> str:
    return json.dumps(ModelPage.model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["check", "export", "schema"])
    args = parser.parse_args(argv)
    if args.command == "check":
        errors = check()
        if not SCHEMA.exists() or SCHEMA.read_text(encoding="utf-8") != schema_text():
            errors.append("docs/models/_schema.json is out of date; run `site_data.py schema`")
        for e in errors:
            print(e, file=sys.stderr)
        print(f"site data: {'FAIL' if errors else 'ok'} ({len(errors)} problem(s))")
        return 1 if errors else 0
    if args.command == "export":
        export()
        return 0
    SCHEMA.write_text(schema_text(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

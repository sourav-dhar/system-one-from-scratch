"""The companion site's single source of truth: one validated YAML per model.

Every model page on the site (and, later, every generated README) is rendered from
``docs/models/<id>/model.yaml``. This script is the only gate between those files and the site:

* ``check``  validates every model file and the registry, and fails on
  - a schema violation (``docs/models/_schema.json`` is generated from the models below);
  - a source key that does not resolve (``research/papers/index.yaml`` keys, or ``repo:<path>``
    for a file in this repository);
  - an ``ours`` number that differs from the results file it cites;
  - a benchmark table that mixes harnesses.
* ``export`` writes the validated, source-resolved data as JSON into ``site/src/data/`` for Astro.
* ``schema`` rewrites ``docs/models/_schema.json``.

    uv run python scripts/site_data.py check
    uv run python scripts/site_data.py export
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "docs" / "models"
REGISTRY = MODELS / "_registry.yaml"
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


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    """Where a fact comes from: a library key, or ``repo:<path>`` for a file in this repository."""

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


NUMERIC = ("accuracy", "accuracy_ci", "brier", "p50_ms")


class CheckRef(Strict):
    """Ties a displayed number to the file it came from (required for `ours` rows, allowed for any row).

    Either ``pointer`` (a JSON pointer into ``file``; the value is multiplied by ``scale`` and rounded to
    ``decimals``) or ``text`` (an exact string that must appear in ``file``, for figures that exist only in a
    generated report).
    """

    file: str
    pointer: str | None = None
    scale: float = 1.0
    decimals: int | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> CheckRef:
        if (self.pointer is None) == (self.text is None):
            raise ValueError("give exactly one of `pointer` or `text`")
        if self.pointer is not None and self.decimals is None:
            raise ValueError("`pointer` needs `decimals`")
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
    harness: str
    checks: dict[str, CheckRef] = Field(
        default_factory=dict, description="Field name -> the file that must reproduce the shown number."
    )
    focus: bool = False
    child: bool = False

    @model_validator(mode="after")
    def _numbers_are_tied(self) -> BenchRow:
        unknown = set(self.checks) - set(NUMERIC)
        if unknown:
            raise ValueError(f"row {self.label!r}: `checks` names unknown fields {sorted(unknown)}")
        if self.provenance == "ours":
            shown = [k for k in NUMERIC if getattr(self, k) is not None]
            missing = [k for k in shown if k not in self.checks]
            if missing:
                raise ValueError(f"`ours` row {self.label!r} must tie {missing} to a results file")
        return self


class BenchTable(Strict):
    suite: str
    title: str
    caption: str
    reference_line: float | None = None
    reference_label: str | None = None
    notes: list[Fact] = Field(default_factory=list)
    rows: list[BenchRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_harness(self) -> BenchTable:
        harnesses = {r.harness for r in self.rows}
        if len(harnesses) != 1:
            raise ValueError(f"table {self.title!r} mixes harnesses {sorted(harnesses)}; same-harness only")
        return self


class Segment(Strict):
    kind: Literal["keep", "del", "ins"]
    text: str


class Correction(Strict):
    """A published correction, shown in place: struck text stays readable beside its replacement."""

    date: date
    commit: str
    where: str
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

    ``covered``: explained in ``section``. ``planned``: assigned to ``section`` but not written yet (allowed
    only while the page is a draft). ``n/a``: deliberately left out, with a ``reason``.
    """

    element: str
    source: Source
    status: Literal["covered", "planned", "n/a"]
    section: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _mapped(self) -> Coverage:
        if self.status == "n/a" and not self.reason:
            raise ValueError(f"coverage {self.element!r}: `n/a` needs a `reason`")
        if self.status != "n/a" and not self.section:
            raise ValueError(f"coverage {self.element!r}: `{self.status}` needs a `section`")
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

    id: str
    name: str
    version: str | None = None
    full_name: str
    tier: Tier
    family: Family
    note: str = Field(description="Repo path of the research note this page is drawn from.")
    eyebrow: list[str] = Field(default_factory=list)
    dek: str
    spec: list[SpecItem] = Field(min_length=1)
    summary: Summary
    benchmarks: list[BenchTable] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    coverage: list[Coverage] = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)


class RegistryEntry(Strict):
    id: str
    name: str
    tier: Tier
    family: Family
    note: str | None = None
    status: Literal["planned", "draft", "published"] = "planned"


class Registry(Strict):
    systems: list[RegistryEntry]


# ------------------------------------------------------------------------------------------------
# loading and resolution
# ------------------------------------------------------------------------------------------------
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
        path = src.key[len("repo:") :]
        if not (ROOT / path).exists():
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


def json_pointer(doc: Any, pointer: str) -> Any:
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def check_row(row: BenchRow) -> list[str]:
    """Each tied figure must equal its file, at the precision shown."""
    errors: list[str] = []
    for field, ref in row.checks.items():
        path = ROOT / ref.file
        if not path.exists():
            errors.append(f"{row.label}.{field}: {ref.file} does not exist")
            continue
        shown = getattr(row, field)
        if ref.text is not None:
            if ref.text not in path.read_text(encoding="utf-8"):
                errors.append(f"{row.label}.{field}: {ref.text!r} not found in {ref.file}")
            values = shown if isinstance(shown, tuple | list) else [shown]
            absent = [v for v in values if v is None or f"{float(v):.2f}" not in ref.text]
            if absent:
                errors.append(f"{row.label}.{field}: shown {shown} does not appear in {ref.text!r}")
            continue
        assert ref.pointer is not None
        assert ref.decimals is not None
        try:
            raw = json_pointer(json.loads(path.read_text(encoding="utf-8")), ref.pointer)
        except (KeyError, IndexError, ValueError) as exc:
            errors.append(f"{row.label}.{field}: pointer {ref.pointer} failed in {ref.file} ({exc!r})")
            continue
        want = round(float(raw) * ref.scale, ref.decimals)
        if shown is None or round(float(shown), ref.decimals) != want:
            errors.append(f"{row.label}.{field}: page shows {shown}, {ref.file}{ref.pointer} gives {want}")
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


def check() -> list[str]:
    errors: list[str] = []
    try:
        registry = Registry.model_validate(yaml.safe_load(REGISTRY.read_text(encoding="utf-8")))
    except ValidationError as exc:
        return [f"{REGISTRY.relative_to(ROOT)}: {exc}"]
    ids = [e.id for e in registry.systems]
    errors += [f"registry: duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for e in registry.systems:
        if e.note and not (ROOT / e.note).exists():
            errors.append(f"registry: {e.id}: note {e.note} does not exist")

    pages, load_errors = load_models()
    errors += load_errors
    lib = library()
    by_id = {e.id: e for e in registry.systems}
    for page in pages:
        where = f"docs/models/{page.id}/model.yaml"
        entry = by_id.get(page.id)
        if entry is None:
            errors.append(f"{where}: not in the registry")
        elif (entry.tier, entry.family) != (page.tier, page.family):
            errors.append(f"{where}: tier/family differ from the registry")
        if not (ROOT / page.note).exists():
            errors.append(f"{where}: note {page.note} does not exist")
        if entry is not None and entry.status == "published":
            open_items = [c.element for c in page.coverage if c.status == "planned"]
            if open_items:
                errors.append(f"{where}: published page still has planned coverage: {open_items}")
        for src in sources_in(page):
            if resolve(src, lib) is None:
                errors.append(f"{where}: source {src.key!r} does not resolve")
        for table in page.benchmarks:
            for row in table.rows:
                errors += [f"{where}: {e}" for e in check_row(row)]
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
    pages, _ = load_models()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for stale in (OUT / "models").glob("*.json"):
        stale.unlink()
    for page in pages:
        data = page.model_dump(mode="json")
        refs: dict[str, dict[str, Any]] = {}
        for src in sources_in(page):
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


def write_schema() -> None:
    schema = ModelPage.model_json_schema()
    SCHEMA.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["check", "export", "schema"])
    args = parser.parse_args(argv)
    if args.command == "check":
        errors = check()
        stale = SCHEMA.exists() and SCHEMA.read_text(encoding="utf-8") != (
            json.dumps(ModelPage.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
        )
        if stale or not SCHEMA.exists():
            errors.append("docs/models/_schema.json is out of date; run `site_data.py schema`")
        for e in errors:
            print(e, file=sys.stderr)
        print(f"site data: {'FAIL' if errors else 'ok'} ({len(errors)} problem(s))")
        return 1 if errors else 0
    if args.command == "export":
        export()
        return 0
    write_schema()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

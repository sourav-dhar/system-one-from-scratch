"""The companion site's data gate (scripts/site_data.py): schema, sources and checked numbers."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("site_data", ROOT / "scripts" / "site_data.py")
assert spec is not None
assert spec.loader is not None
site_data = importlib.util.module_from_spec(spec)
sys.modules["site_data"] = site_data
spec.loader.exec_module(site_data)

CLM = ROOT / "docs" / "models" / "clm" / "model.yaml"


def clm() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CLM.read_text(encoding="utf-8"))
    return data


def test_repository_site_data_is_valid() -> None:
    assert site_data.check() == []


def test_schema_file_is_current() -> None:
    assert site_data.main(["check"]) == 0


def test_registry_covers_every_research_note() -> None:
    registry = site_data.Registry.model_validate(
        yaml.safe_load(site_data.REGISTRY.read_text(encoding="utf-8"))
    )
    notes = {e.note for e in registry.systems if e.note}
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "research" / "systems").glob("*.md")}
    assert on_disk <= notes, f"notes without a registry entry: {sorted(on_disk - notes)}"
    tiers = [e.tier for e in registry.systems if e.family != "ours"]
    assert (tiers.count("A"), tiers.count("B"), tiers.count("C")) == (12, 13, 6)


def row(page: site_data.ModelPage, label: str, nth: int = 0) -> site_data.BenchRow:
    return [r for t in page.benchmarks for r in t.rows if r.label == label][nth]


def test_checked_number_must_match_its_results_file() -> None:
    page = site_data.ModelPage.model_validate(clm())
    ours = row(page, "our run, raw", 1)  # CLM, raw
    assert site_data.check_row(ours) == []
    wrong = ours.model_copy(update={"accuracy": 41.87})
    assert any("page shows 41.87" in e for e in site_data.check_row(wrong))


def test_published_number_is_checked_against_the_snapshot() -> None:
    page = site_data.ModelPage.model_validate(clm())
    published = row(page, "CLM-v0.1-8B")
    assert site_data.check_row(published) == []
    assert site_data.check_row(published.model_copy(update={"brier": 0.7493}))


def test_interval_text_must_contain_the_shown_bounds() -> None:
    page = site_data.ModelPage.model_validate(clm())
    ours = row(page, "our run, raw", 1)
    bad = ours.model_copy(update={"accuracy_ci": (40.30, 43.44)})
    assert any("does not appear" in e for e in site_data.check_row(bad))


def test_ours_row_without_a_check_is_rejected() -> None:
    data = clm()
    data["benchmarks"][0]["rows"][3]["checks"].pop("brier")
    with pytest.raises(ValidationError, match="must tie"):
        site_data.ModelPage.model_validate(data)


def test_mixed_harnesses_are_rejected() -> None:
    data = clm()
    data["benchmarks"][0]["rows"][0]["harness"] = "other-harness"
    with pytest.raises(ValidationError, match="mixes harnesses"):
        site_data.ModelPage.model_validate(data)


def test_unknown_source_key_does_not_resolve() -> None:
    lib = site_data.library()
    assert site_data.resolve(site_data.Source(key="clm-repo"), lib) is not None
    assert site_data.resolve(site_data.Source(key="no-such-key"), lib) is None
    assert (
        site_data.resolve(site_data.Source(key="repo:results/references/s1arena/ANCHORS.md"), lib) is not None
    )
    assert site_data.resolve(site_data.Source(key="repo:no/such/file.md"), lib) is None


def test_coverage_entries_must_say_where_or_why() -> None:
    src = {"key": "clm-repo"}
    with pytest.raises(ValidationError):
        site_data.Coverage.model_validate({"element": "x", "source": src, "status": "planned"})
    with pytest.raises(ValidationError):
        site_data.Coverage.model_validate({"element": "x", "source": src, "status": "n/a"})
    site_data.Coverage.model_validate(
        {"element": "x", "source": src, "status": "n/a", "reason": "not described"}
    )


def test_correction_must_change_something() -> None:
    base = {
        "date": "2026-11-22",
        "commit": "abc1234",
        "where": "README",
        "provenance": "ours",
        "sources": [{"key": "clm-repo"}],
    }
    with pytest.raises(ValidationError, match="delete or insert"):
        site_data.Correction.model_validate(
            {**base, "segments": [{"kind": "keep", "text": "a"}, {"kind": "keep", "text": "b"}]}
        )


def test_every_fact_carries_provenance_and_a_source() -> None:
    data = clm()
    bad = copy.deepcopy(data)
    bad["summary"]["what"][0]["sources"] = []
    with pytest.raises(ValidationError):
        site_data.ModelPage.model_validate(bad)
    bad = copy.deepcopy(data)
    bad["summary"]["what"][0]["provenance"] = []
    with pytest.raises(ValidationError):
        site_data.ModelPage.model_validate(bad)


def test_export_writes_resolved_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(site_data, "OUT", tmp_path)
    site_data.export()
    import json

    page = json.loads((tmp_path / "models" / "clm.json").read_text(encoding="utf-8"))
    keys = {r["key"] for r in page["references"]}
    assert {"clm-repo", "s1arena-repo", "repo:results/references/s1arena/ANCHORS.md"} <= keys
    assert all(r["url"].startswith("https://") for r in page["references"])
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert next(e for e in registry if e["id"] == "clm")["has_page"] is True

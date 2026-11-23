"""The companion site's data gate (scripts/site_data.py): schema, sources and checked numbers.

Most tests mutate an in-memory copy of docs/models/clm/model.yaml and assert the gate rejects it. Each hole
found in review gets a test here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
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
LB = "results/references/s1arena/leaderboard-2026-10-14.json"


def clm() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CLM.read_text(encoding="utf-8"))
    return data


def page() -> Any:
    return site_data.ModelPage.model_validate(clm())


def row(label: str, nth: int = 0) -> Any:
    return [r for t in page().benchmarks for r in t.rows if r.label == label][nth]


def rows_raw(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = data["benchmarks"][0]["rows"]
    return rows


# ---- the repository as committed ---------------------------------------------------------------
def test_repository_site_data_is_valid() -> None:
    assert site_data.check() == []


def test_schema_file_is_current() -> None:
    assert site_data.main(["check"]) == 0


def test_stale_schema_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stale = tmp_path / "_schema.json"
    stale.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(site_data, "SCHEMA", stale)
    assert site_data.main(["check"]) == 1


def test_registry_covers_every_research_note() -> None:
    registry = site_data.Registry.model_validate(
        yaml.safe_load(site_data.REGISTRY.read_text(encoding="utf-8"))
    )
    notes = {e.note for e in registry.systems if e.note}
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "research" / "systems").glob("*.md")}
    assert on_disk <= notes, f"notes without a registry entry: {sorted(on_disk - notes)}"
    tiers = [e.tier for e in registry.systems if e.family != "ours"]
    assert (tiers.count("A"), tiers.count("B"), tiers.count("C")) == (12, 13, 6)


# ---- numbers --------------------------------------------------------------------------------------
def test_checked_number_must_match_its_results_file() -> None:
    ours = row("our run, raw", 1)  # CLM, raw
    assert site_data.check_row(ours) == []
    assert any(
        "page shows 41.87" in e for e in site_data.check_row(ours.model_copy(update={"accuracy": 41.87}))
    )


def test_published_number_is_checked_against_the_snapshot() -> None:
    published = row("CLM-v0.1-8B")
    assert site_data.check_row(published) == []
    assert site_data.check_row(published.model_copy(update={"brier": 0.7493}))


@pytest.mark.parametrize("ci", [(0.30, 3.43), (40.30, 43.44), (41.86, 43.43)])
def test_interval_must_match_the_report_exactly(ci: tuple[float, float]) -> None:
    ours = row("our run, raw", 1)
    bad = ours.model_copy(update={"accuracy_ci": ci})
    assert site_data.check_row(bad), f"interval {ci} passed"


def test_interval_must_contain_the_point_estimate() -> None:
    data = clm()
    rows_raw(data)[7]["accuracy_ci"] = [43.43, 40.30]
    with pytest.raises(ValidationError, match="must contain the accuracy"):
        site_data.ModelPage.model_validate(data)


def test_report_text_check_is_tied_to_its_row() -> None:
    data = clm()
    rows_raw(data)[7]["accuracy_ci"] = [58.01, 61.11]  # Qwen's interval on the CLM row
    rows_raw(data)[7]["accuracy"] = 59.5
    bad = site_data.ModelPage.model_validate(data).benchmarks[0].rows[7]
    assert any("does not occur" in e for e in site_data.check_row(bad))


def test_more_precision_than_shown_is_rejected() -> None:
    data = clm()
    rows_raw(data)[6]["accuracy"] = 41.4301
    with pytest.raises(ValidationError, match="more precision"):
        site_data.ModelPage.model_validate(data)


def test_display_rounds_half_up() -> None:
    assert site_data.fmt(41.125, 2) == "41.13"
    assert site_data.fmt(0.74925, 4) == "0.7493"
    assert site_data.fmt(228.0, 1) == "228.0"


def test_every_row_must_tie_every_number() -> None:
    data = clm()
    rows_raw(data)[1]["checks"].pop("accuracy")  # Jev 1.13, an independent row
    with pytest.raises(ValidationError, match="must tie"):
        site_data.ModelPage.model_validate(data)


def test_identity_is_checked_not_just_position() -> None:
    data = clm()
    clm_row = rows_raw(data)[6]
    clm_row["accuracy"], clm_row["brier"], clm_row["p50_ms"] = 44.47, 0.6605, 136.5  # Laya's numbers
    for check in clm_row["checks"].values():
        check["pointer"] = check["pointer"].replace("/models/27/", "/models/26/")
    bad = site_data.ModelPage.model_validate(data).benchmarks[0].rows[6]
    assert any("identity" in e for e in site_data.check_row(bad))


def test_pointer_through_an_array_needs_identity() -> None:
    data = clm()
    for check in rows_raw(data)[0]["checks"].values():
        check.pop("match")
    with pytest.raises(ValidationError, match="must identify"):
        site_data.ModelPage.model_validate(data)


def test_rank_is_checked() -> None:
    assert site_data.check_rank(row("CLM-v0.1-8B")) == []
    bad = row("CLM-v0.1-8B").model_copy(update={"rank": "27 / 29"})
    assert site_data.check_rank(bad) == ["CLM-v0.1-8B: rank '27 / 29', file gives '28 / 29'"]


def test_harness_commit_is_checked_against_the_file() -> None:
    bad = row("CLM-v0.1-8B").model_copy(update={"harness": "s1arena@0000000"})
    assert any("produced by harness" in e for e in site_data.check_row(bad))


def test_mixed_harnesses_are_rejected() -> None:
    data = clm()
    rows_raw(data)[0]["harness"] = "other@1234567"
    with pytest.raises(ValidationError, match="mixes harnesses"):
        site_data.ModelPage.model_validate(data)


def test_bar_axis_must_hold_every_accuracy() -> None:
    data = clm()
    data["benchmarks"][0]["axis_max"] = 70
    with pytest.raises(ValidationError, match="exceeds the bar axis"):
        site_data.ModelPage.model_validate(data)


@pytest.mark.parametrize(
    "pointer",
    [
        "/models/-3/accuracy",
        "/models/027/accuracy",
        "models/27/accuracy",
        "/models/29/accuracy",
        "/source/repo/x",
    ],
)
def test_malformed_or_non_numeric_pointers_are_errors_not_crashes(pointer: str) -> None:
    bad_check = site_data.CheckRef(file=LB, pointer=pointer, scale=100, match={"/name": "CLM-v0.1-8B"})
    bad = row("CLM-v0.1-8B").model_copy(
        update={"checks": {**row("CLM-v0.1-8B").checks, "accuracy": bad_check}}
    )
    errors = site_data.check_row(bad)
    assert errors
    assert all(isinstance(e, str) for e in errors)


# ---- sources --------------------------------------------------------------------------------------
def test_sources_resolve_only_to_tracked_files_and_known_keys() -> None:
    lib = site_data.library()
    ok = site_data.Source
    assert site_data.resolve(ok(key="clm-repo"), lib) is not None
    assert site_data.resolve(ok(key="repo:results/references/s1arena/ANCHORS.md"), lib) is not None
    for bad in [
        "no-such-key",
        "repo:no/such/file.md",
        "repo:",
        "repo:.",
        "repo:..",
        "repo:/etc/passwd",
        "repo:docs",
    ]:
        assert site_data.resolve(ok(key=bad), lib) is None, bad


def test_every_fact_carries_provenance_and_a_source() -> None:
    data = clm()
    for field in ("sources", "provenance"):
        bad = copy.deepcopy(data)
        bad["summary"]["what"][0][field] = []
        with pytest.raises(ValidationError):
            site_data.ModelPage.model_validate(bad)


# ---- corrections, coverage, registry -----------------------------------------------------------
def test_correction_is_checked_against_its_commit() -> None:
    good = page().corrections[0]
    assert site_data.check_correction(good) == []
    segments = [s.model_copy() for s in good.segments]
    segments[6] = segments[6].model_copy(update={"text": "matching the reference closely"})
    assert site_data.check_correction(good.model_copy(update={"segments": segments}))


def test_correction_must_change_something() -> None:
    base = {
        "date": "2026-11-22",
        "commit": "eb6f2f5",
        "file": "README.md",
        "where": "README",
        "provenance": "ours",
        "sources": [{"key": "clm-repo"}],
    }
    keep_only = [{"kind": "keep", "text": "a"}, {"kind": "keep", "text": "b"}]
    with pytest.raises(ValidationError, match="delete or insert"):
        site_data.Correction.model_validate({**base, "segments": keep_only})


def test_coverage_entries_must_say_where_or_why() -> None:
    src = {"key": "clm-repo"}
    with pytest.raises(ValidationError):
        site_data.Coverage.model_validate({"element": "x", "source": src, "status": "planned"})
    with pytest.raises(ValidationError):
        site_data.Coverage.model_validate({"element": "x", "source": src, "status": "n/a"})
    with pytest.raises(ValidationError, match="not rendered yet"):
        site_data.Coverage.model_validate(
            {"element": "x", "source": src, "status": "covered", "section": "training"}
        )
    site_data.Coverage.model_validate(
        {"element": "x", "source": src, "status": "n/a", "reason": "not described"}
    )


def registry_with(**changes: Any) -> Any:
    reg = site_data.Registry.model_validate(yaml.safe_load(site_data.REGISTRY.read_text(encoding="utf-8")))
    entries = [e.model_copy(update=changes) if e.id == "clm" else e for e in reg.systems]
    return reg.model_copy(update={"systems": entries})


def test_registry_and_pages_must_agree() -> None:
    pages = [page()]
    assert site_data.check_registry(registry_with(), pages) == []
    assert any(
        "still has planned coverage" in e
        for e in site_data.check_registry(registry_with(status="published"), pages)
    )
    assert any(
        "still `planned`" in e for e in site_data.check_registry(registry_with(status="planned"), pages)
    )
    assert any(
        "differs from the registry" in e for e in site_data.check_registry(registry_with(name="CLM"), pages)
    )
    assert any("has no model.yaml" in e for e in site_data.check_registry(registry_with(), []))


def test_ids_are_url_slugs() -> None:
    with pytest.raises(ValidationError):
        site_data.RegistryEntry.model_validate(
            {"id": "CLM v0", "name": "x", "tier": "A", "family": "biencoder"}
        )


# ---- export ---------------------------------------------------------------------------------------
def test_export_writes_resolved_references_and_formatted_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_data, "OUT", tmp_path)
    site_data.export()
    data = json.loads((tmp_path / "models" / "clm.json").read_text(encoding="utf-8"))
    keys = {r["key"] for r in data["references"]}
    assert {"clm-repo", "s1arena-repo", "repo:results/references/s1arena/ANCHORS.md"} <= keys
    assert all(r["url"].startswith("https://") for r in data["references"])
    rows = data["benchmarks"][0]["rows"]
    assert rows[7]["display"] == {"accuracy": "41.86", "accuracy_ci": "40.30–43.43", "brier": "0.7411"}
    assert rows[0]["display"]["p50_ms"] == "228.0 ms"
    assert "checks" not in rows[0]
    assert data["benchmarks"][0]["reference"]["value"] == 59.25
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert next(e for e in registry if e["id"] == "clm")["has_page"] is True


# ---- second review round: identity bound to the row, column bound to the field ------------------
def test_check_must_identify_the_rows_entity() -> None:
    data = clm()
    clm_row = rows_raw(data)[6]
    for check in clm_row["checks"].values():
        check["match"] = {"/family": "System-1 model"}  # shared by every System-1 entry
    with pytest.raises(ValidationError, match="must identify"):
        site_data.ModelPage.model_validate(data)


def test_pointer_must_end_in_the_columns_field() -> None:
    data = clm()
    check = rows_raw(data)[6]["checks"]["brier"]
    check["pointer"] = check["pointer"].replace("/brier", "/accuracy")
    with pytest.raises(ValidationError, match="does not end in a brier field"):
        site_data.ModelPage.model_validate(data)


def test_scale_is_one_or_a_hundred() -> None:
    data = clm()
    rows_raw(data)[6]["checks"]["accuracy"]["scale"] = 120.68
    with pytest.raises(ValidationError):
        site_data.ModelPage.model_validate(data)


def test_correction_must_be_a_change_the_commit_made() -> None:
    good = page().corrections[0]
    later = good.model_copy(update={"commit": "e7bfae9"})  # a commit that never touched README.md
    assert site_data.check_correction(later)

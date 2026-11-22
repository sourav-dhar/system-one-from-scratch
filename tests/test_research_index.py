"""The research library index is data the survey depends on: keep it well-formed."""

import re
from pathlib import Path

import yaml

INDEX = Path(__file__).resolve().parent.parent / "research" / "papers" / "index.yaml"
FAMILIES = {
    "biencoder",
    "late-interaction",
    "decoder-readout",
    "encoder",
    "encoder-decoder",
    "looped",
    "frozen-reader",
    "evaluation",
    "serving",
    "foundations",
    "generative",
    "data",
    "community",
    "unclassified",
}
ROLES = {"implementation", "benchmark", "background", "dataset", "product"}


def load() -> dict:
    return yaml.safe_load(INDEX.read_text(encoding="utf-8"))


def test_index_has_both_sections() -> None:
    data = load()
    assert set(data) == {"arxiv", "links"}
    assert len(data["arxiv"]) >= 50
    assert len(data["links"]) >= 40


def test_keys_are_unique_and_bibtex_safe() -> None:
    data = load()
    keys = [e["key"] for e in data["arxiv"] + data["links"]]
    assert len(keys) == len(set(keys))
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", k) for k in keys)


def test_arxiv_entries() -> None:
    ids = []
    for e in load()["arxiv"]:
        assert set(e) == {"key", "id", "expect", "system", "family", "role"}, e
        assert re.fullmatch(r"\d{4}\.\d{4,5}", e["id"]), e
        assert e["family"] in FAMILIES, e
        assert e["role"] in ROLES, e
        ids.append(e["id"])
    assert len(ids) == len(set(ids))


def test_link_entries() -> None:
    for e in load()["links"]:
        assert set(e) == {"key", "url", "title", "system", "family", "role"}, e
        assert e["url"].startswith("https://"), e
        assert e["family"] in FAMILIES, e
        assert e["role"] in ROLES, e


def test_metadata_and_bibliography_are_clean_utf8() -> None:
    """Regression: re-reading the cache with the platform encoding double-encoded author names."""
    root = INDEX.parent.parent.parent
    for path in (root / "research/papers/metadata.json", root / "docs/research/survey.bib"):
        text = path.read_text(encoding="utf-8")
        assert "Ã" not in text, f"mojibake in {path.name}"
        assert "â€" not in text, f"mojibake in {path.name}"

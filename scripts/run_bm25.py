"""Run the BM25 baseline on the pinned System1-Arena static set (CPU only) and write BASELINES.md.

    uv run python scripts/run_bm25.py

Outputs, in ``results/references/s1arena/``:
* ``<slug>/static/responses.jsonl.gz``: raw responses, in the harness's run-row layout;
* ``<slug>/static/summary.json``: raw and leave-one-domain-out calibrated aggregates;
* ``BASELINES.md``: BM25 next to the two calibrated anchors, overall and per domain.

The calibration is the same procedure as CALIBRATION.md (``systemone.bench.calibrate``): temperatures
are fitted on the other 12 domains, one per question type.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from systemone.bench import aggregate, score
from systemone.bench import calibrate as cal
from systemone.bench.baselines import BM25Decider
from systemone.bench.runner import read_responses, slug, write_row
from systemone.decisions import s1arena

ROOT = Path(__file__).resolve().parent.parent / "results" / "references" / "s1arena"
ANCHORS = ["Qwen/Qwen3-8B@label-think0", "Contrastive-LM/CLM-v0.1-8B"]


def calibrated(items: list, responses: dict) -> dict:
    temps = cal.leave_out_temperatures(items, responses, lambda it: it.domain, per_type=True)
    return aggregate(score(items, cal.apply_temperatures(items, responses, lambda it: it.domain, temps)))


def main() -> None:
    t0 = time.time()
    items = s1arena.load(s1arena.STATIC_DOMAINS)
    model = BM25Decider.from_items(items)
    out = ROOT / slug(model.model_id) / "static"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "responses.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(items), 2000):
            part = items[start : start + 2000]
            for item, response in zip(part, model.decide_many([i.request for i in part]), strict=True):
                write_row(f, item.id, response)
    responses = read_responses(path)
    raw, fixed = aggregate(score(items, responses)), calibrated(items, responses)
    no_text = sum(len((r.usage or {}).get("no_option_text", [])) for r in responses.values())
    summary = {
        "model": model.model_id,
        "suite": {"dataset": s1arena.DATASET, "revision": s1arena.REVISION, "scope": "static"},
        "catalog_documents": model.n_docs,
        "questions_without_option_text": no_text,
        "raw": {k: raw[k] for k in ("overall", "domains", "types")},
        "calibrated_per_type_lodo": {k: fixed[k] for k in ("overall", "domains")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(
        f"BM25 on {len(items):,} items in {time.time() - t0:.0f} s; {no_text:,} questions had no option text"
    )

    rows = [(model.model_id, raw, fixed, "this run")]
    for anchor in ANCHORS:
        run = ROOT / slug(anchor) / "static" / "responses.jsonl.gz"
        if run.exists():
            resp = read_responses(run)
            rows.append((anchor, aggregate(score(items, resp)), calibrated(items, resp), "ANCHORS.md"))
    lines = [
        "# System1-Arena baselines",
        "",
        "- **Data:** the pinned public static set (`ad05ed5`, 41,183 items).",
        "- **Aggregation:** mean over 87 leaves.",
        "- **Calibration:** temperatures per question type, fitted leave-one-domain-out, so no domain is calibrated on its own questions.",
        "- **Source:** `scripts/run_bm25.py`. No GPU was used.",
        "",
        "| Model | Accuracy | Brier, raw | Brier, calibrated | ECE (10 equal-mass), calibrated | Source |",
        "|---|---|---|---|---|---|",
    ]
    for name, r, c, src in rows:
        o, k = r["overall"], c["overall"]
        lines.append(
            f"| `{name}` | {100 * o['accuracy']:.2f} | {o['brier']:.4f} | {k['brier']:.4f} | {k['ece_mass10']:.3f} | {src} |"
        )
    domains = list(rows[0][1]["domains"])
    lines += [
        "",
        "## Accuracy by domain",
        "",
        "| Domain | " + " | ".join(f"`{n}`" for n, *_ in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for d in domains:
        cells = [
            f"{100 * r['domains'][d]['accuracy']:.1f}" if d in r["domains"] else "—" for _, r, _, _ in rows
        ]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Accuracy by question type (pooled over questions)",
        "",
        "| Type | " + " | ".join(f"`{n}`" for n, *_ in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for qtype in ("choice", "score", "boolean"):
        cells = [
            f"{100 * r['types'][qtype]['accuracy']:.1f}" if qtype in r["types"] else "—"
            for _, r, _, _ in rows
        ]
        lines.append(f"| {qtype} | " + " | ".join(cells) + " |")
    clm = next((r for n, r, _, _ in rows if n.startswith("Contrastive-LM")), None)
    beats = [
        d
        for d in domains
        if clm and d in clm["domains"] and raw["domains"][d]["accuracy"] >= clm["domains"][d]["accuracy"]
    ]
    lines += [
        "",
        "## Reading",
        "",
        "- **BM25 is a lexical floor.** It can only help where option texts share rare words with the request, as in retrieval and routing.",
        f'- **Booleans.** S1-Arena\'s boolean questions carry no option text ({no_text:,} questions here). BM25 answers them uniformly, and the tie goes to "true".',
        f'- **Its boolean accuracy is therefore the share of "true" gold labels** ({100 * raw["types"]["boolean"]["accuracy"]:.1f}%), not a lexical result. On choice questions, where BM25 does match text, it scores {100 * raw["types"]["choice"]["accuracy"]:.1f}%.',
        f"- **CLM against the floor.** BM25 is at least as accurate as CLM-v0.1-8B in {len(beats)} of {len(domains)} domains: {', '.join(beats)}. That fits the documented failure where CLM ignores the state (survey FM7 / FM14): there, its cosine to the options carries little more than word overlap does."
        if clm
        else "",
        (
            f"- **Choice questions, pooled:** BM25 {100 * raw['types']['choice']['accuracy']:.1f}% against CLM {100 * clm['types']['choice']['accuracy']:.1f}%. "
            f"CLM's overall lead over the floor comes from booleans ({100 * clm['types']['boolean']['accuracy']:.1f}% against BM25's {100 * raw['types']['boolean']['accuracy']:.1f}% 'always true'), not from reading options better."
        )
        if clm
        else "",
        "- **What it means for the anchors.** A model that does not beat BM25 in a domain is not reading the options better than word overlap does.",
        "",
    ]
    (ROOT / "BASELINES.md").write_text("\n".join(lines), encoding="utf-8")
    for name, r, c, _ in rows:
        print(
            f"  {name:<34} acc {r['overall']['accuracy']:.4f}  brier raw {r['overall']['brier']:.4f}  calibrated {c['overall']['brier']:.4f}"
        )
    print(f"wrote {ROOT / 'BASELINES.md'}")


if __name__ == "__main__":
    main()

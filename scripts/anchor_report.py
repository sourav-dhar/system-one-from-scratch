"""Write results/references/s1arena/ANCHORS.md: our anchor runs against the published leaderboard.

    uv run python scripts/anchor_report.py

Inputs are each anchor's ``static/summary.json`` and ``static/responses.jsonl.gz`` (written by
``cloud/harness_app.py::collect``), plus the published snapshot ``leaderboard-2026-10-14.json``.

For each model, the report gives:
* the overall mean over leaves, with a 95% task-cluster bootstrap interval;
* the mean difference over the 8 domains that the public dataset holds completely;
* every domain, with the 5 incomplete ones flagged.

The acceptance rule (ADR 0003) is that each anchor lands within 1.0 pp of the published number.
"""

from __future__ import annotations

import json
from pathlib import Path

from systemone.bench import paired_bootstrap, score
from systemone.bench.runner import read_responses, slug
from systemone.decisions import s1arena

ROOT = Path(__file__).resolve().parent.parent / "results" / "references" / "s1arena"
ANCHORS = [  # (published leaderboard id, how we ran it)
    (
        "Qwen/Qwen3-8B@label-think0",
        "answer-token readout, our transformers port of `lm:` (`bench/readout.py`)",
    ),
    ("Contrastive-LM/CLM-v0.1-8B", "CLM heads on our Qwen3-8B last-token encoder (`bench/clm.py`)"),
]
TOLERANCE_PP = 1.0


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def main() -> None:
    lines = [
        "# System1-Arena anchors: our harness against the published leaderboard",
        "",
        f"- **Dataset:** `{s1arena.DATASET}` at `{s1arena.REVISION[:7]}`, the 13 static domains.",
        f"- **Reference:** the leaderboard snapshot of 2026-10-14 (harness `{s1arena.HARNESS_COMMIT[:7]}`).",
        "- **Accuracy:** the leaderboard's own aggregate, a mean over the 87 leaves of each leaf's mean over its "
        "tasks. The interval is a 95% paired bootstrap that resamples tasks within leaves and items within tasks.",
        "- **Coverage:** the public dataset lacks 14 restricted sources (2,043 static requests in 9 leaves). The "
        "8 complete domains therefore compare one to one, and the overall figure is computed on the public subset.",
        f"- **Acceptance (ADR 0003):** within {TOLERANCE_PP:.1f} pp of the published accuracy.",
        "",
    ]
    rows = []
    for model_id, how in ANCHORS:
        run = ROOT / slug(model_id) / "static"
        if not (run / "summary.json").exists():
            lines += [f"- `{model_id}`: not run yet.", ""]
            continue
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        vs = summary["vs_published"]
        items = s1arena.load(s1arena.STATIC_DOMAINS)
        scored = score(items, read_responses(run / "responses.jsonl.gz"))
        ci = paired_bootstrap({model_id: scored}, b=1000, unit="task")["ci95"]["overall"][model_id]
        overall = summary["aggregate"]["overall"]
        within = abs(vs["delta_pp"]) <= TOLERANCE_PP
        rows.append((model_id, how, overall, vs, ci, within))

    if rows:
        lines += [
            "| Model | Ours (95% CI) | Published | Δ overall | Δ over 8 complete domains | Brier ours / published | Within 1 pp |",
            "|---|---|---|---|---|---|---|",
        ]
        for model_id, _, overall, vs, ci, within in rows:
            lo, hi = ci["accuracy"]
            lines.append(
                f"| `{model_id}` | {pct(overall['accuracy'])} ({pct(lo)}–{pct(hi)}) | "
                f"{pct(vs['accuracy']['published'])} | {vs['delta_pp']:+.2f} pp | "
                f"{vs['complete_domains_mean_delta_pp']:+.2f} pp | "
                f"{overall['brier']:.4f} / {vs['brier']['published']:.4f} | {'yes' if within else '**no**'} |"
            )
        lines.append("")
        for model_id, how, overall, vs, _, _ in rows:
            lines += [
                f"## `{model_id}`",
                "",
                f"- **How it was run:** {how}.",
                f"- **Calibration:** NLL {overall['nll']:.3f}; ECE {overall['ece_mass10']:.3f} (10 equal-mass bins) and "
                f"{overall['ece_width15']:.3f} (15 equal-width bins).",
                f"- **Unanswered:** {100 * overall['unanswered_rate']:.2f}% of questions, scored as wrong.",
                "",
                "| Domain | Ours | Published | Δ (pp) | Complete in the public set |",
                "|---|---|---|---|---|",
            ]
            for domain, d in vs["domains"].items():
                lines.append(
                    f"| {domain} | {pct(d['ours'])} | {pct(d['published'])} | {d['delta_pp']:+.2f} | "
                    f"{'yes' if d['complete'] else 'no'} |"
                )
            lines.append("")
    out = ROOT / "ANCHORS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

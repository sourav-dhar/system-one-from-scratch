"""Write results/references/s1arena/CALIBRATION.md and calibration.json: the anchors, calibrated.

    uv run python scripts/calibration_report.py

This needs no GPU. Temperature scaling only needs each run's stored distributions
(``static/responses.jsonl.gz``). Temperatures are fitted leave-one-domain-out (ADR 0004), so no domain
is calibrated on its own questions. Each model gets three rows:
* raw;
* one temperature;
* one temperature per question type.

The in-distribution temperature (fitted on everything) is listed as an optimistic reference, not as a
result. The calibrated Qwen3-8B readout is the bar M1 has to beat (plan v3, M1 success criterion).
"""

from __future__ import annotations

import json
from pathlib import Path

from systemone.bench import aggregate, score
from systemone.bench import calibrate as cal
from systemone.bench.runner import read_responses, slug
from systemone.decisions import s1arena

ROOT = Path(__file__).resolve().parent.parent / "results" / "references" / "s1arena"
MODELS = ["Qwen/Qwen3-8B@label-think0", "Contrastive-LM/CLM-v0.1-8B"]
METRICS = ("accuracy", "brier", "nll", "ece_mass10", "ece_width15")


def main() -> None:
    items = s1arena.load(s1arena.STATIC_DOMAINS)
    board = {
        m["ids"]["static"]: m
        for m in json.loads((ROOT / "leaderboard-2026-10-14.json").read_text(encoding="utf-8"))["models"]
        if m.get("ids") and m["ids"].get("static")
    }
    report: dict[str, dict] = {}
    for model_id in MODELS:
        run = ROOT / slug(model_id) / "static" / "responses.jsonl.gz"
        if not run.exists():
            continue
        responses = read_responses(run)
        rows = {"raw": aggregate(score(items, responses))}
        temps = {}
        for name, per_type in (("global", False), ("per_type", True)):
            temps[name] = cal.leave_out_temperatures(
                items, responses, lambda it: it.domain, per_type=per_type
            )
            rows[name] = aggregate(
                score(items, cal.apply_temperatures(items, responses, lambda it: it.domain, temps[name]))
            )
        report[model_id] = {
            "overall": {k: {m: v["overall"][m] for m in METRICS} for k, v in rows.items()},
            "domains": {
                k: {d: {m: v["domains"][d][m] for m in ("brier", "ece_mass10")} for d in v["domains"]}
                for k, v in rows.items()
            },
            "temperatures_leave_one_domain_out": temps,
            "in_distribution_temperature": cal.in_distribution_temperature(items, responses),
            "published_brier": (board.get(model_id) or {}).get("brier"),
        }
        print(model_id, {k: round(v["brier"], 4) for k, v in report[model_id]["overall"].items()})

    (ROOT / "calibration.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    lines = [
        "# System1-Arena anchors, calibrated",
        "",
        "- **Method:** post-hoc temperature scaling of each model's stored distributions ($p_T \\propto p^{1/T}$), fitted **leave-one-domain-out**: each of the 13 domains is calibrated with temperatures fitted on the other 12 (ADR 0004).",
        "- **What it changes:** temperature never changes the top option, so accuracy is untouched. Brier, NLL and ECE move.",
        "- **Aggregation:** the leaderboard's mean over 87 leaves, on the pinned public set (`ad05ed5`).",
        "- **Source:** `scripts/calibration_report.py`, with all numbers in `calibration.json`. No GPU was used.",
        "",
        "| Model | Variant | Accuracy | Brier | NLL | ECE (10 equal-mass) | ECE (15 equal-width) |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"raw": "raw", "global": "one temperature", "per_type": "one per question type"}
    for model_id, r in report.items():
        for key, label in names.items():
            o = r["overall"][key]
            lines.append(
                f"| `{model_id}` | {label} | {100 * o['accuracy']:.2f} | {o['brier']:.4f} | {o['nll']:.3f} | "
                f"{o['ece_mass10']:.3f} | {o['ece_width15']:.3f} |"
            )
    lines += ["", "## Temperatures", ""]
    for model_id, r in report.items():
        g = sorted(v["all"] for v in r["temperatures_leave_one_domain_out"]["global"].values())
        per = r["temperatures_leave_one_domain_out"]["per_type"]
        by_type: dict[str, list[float]] = {}
        for temps in per.values():
            for t, v in temps.items():
                by_type.setdefault(t, []).append(v)
        spans = ", ".join(f"{t} {min(v):.2f}–{max(v):.2f}" for t, v in sorted(by_type.items()))
        lines.append(
            f"- **`{model_id}`.** One temperature: {g[0]:.2f}–{g[-1]:.2f} across the 13 folds. Per type: {spans}. "
            f"For reference, the in-distribution fit gives {r['in_distribution_temperature']:.2f}."
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **Both models are badly overconfident as they ship.** Their temperatures are well above 1.",
        "  - The Qwen3-8B answer-token readout needs about T = 9; its raw NLL is 4.3 nats.",
        "  - CLM needs about T = 6 on top of its capped scale of 100.",
        "- **The calibrated Qwen3-8B readout is M1's bar** (plan v3: the calibrated zero-shot readout of the backbone).",
        "  - With per-type temperatures it reaches Brier 0.489 at 59.5% accuracy.",
        "  - On both accuracy and Brier it already beats the published, as-shipped numbers of three System One models: Decision 2.0 Kai-0.6B (50.3%, 0.537), Laya (44.5%, 0.660) and CLM-v0.1-8B (41.4%, 0.749). Theirs come from the leaderboard's full set; ours is on the public subset (41,183 of 43,226 requests).",
        "- **One temperature cannot fix everything.** ECE stays near 0.15 after calibration, and per-domain miscalibration differs in sign and size (see the per-domain values in `calibration.json`). That is the room a trained, per-type calibrated M1 model has to win.",
        "",
    ]
    (ROOT / "CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ROOT / 'CALIBRATION.md'}")


if __name__ == "__main__":
    main()

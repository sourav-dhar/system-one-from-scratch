"""Figures for docs/research/survey.md.

Every plotted number is either copied from a cited source (the citation key is next to the data below)
or measured by this project (results/benchmark/heads.json). Palette: the dataviz reference palette,
light mode; categorical slots used in fixed order, at most three per scatter.

    uv run --extra viz python scripts/research_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "research" / "figures"
HEADS = ROOT / "results" / "benchmark" / "heads.json"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(ROOT))


def frontier() -> None:
    """System1-Arena non-agentic p50 vs accuracy [@s1arena, Table 1]."""
    rows = [  # name, group, p50 ms, accuracy %
        ("CLM-v0.1-8B", "Cached bi-encoder", 41, 41.4),
        ("Decision 2.0 Kai-0.6B", "Open joint models", 72, 50.3),
        ("Laya", "Open joint models", 136, 44.5),
        ("Decision 2.0 Vega-27B", "Open joint models", 228, 73.0),
        ("OpenAI Decisions", "Hosted APIs", 140, 68.6),
        ("Microsoft-Decision-1", "Hosted APIs", 304, 70.9),
        ("Jev 1.13", "Hosted APIs", 321, 72.6),
    ]
    colors = {"Cached bi-encoder": S1, "Open joint models": S2, "Hosted APIs": S3}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(59.3, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(36, 60.2, "Qwen3-8B answer-token readout (59.3%)", color=INK2, fontsize=8.5)
    for group, color in colors.items():
        pts = [r for r in rows if r[1] == group]
        ax.scatter(
            [r[2] for r in pts],
            [r[3] for r in pts],
            s=70,
            color=color,
            label=group,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
    offsets = {
        "Laya": (6, -12),
        "Jev 1.13": (6, 4),
        "Microsoft-Decision-1": (-8, -14),
        "Decision 2.0 Vega-27B": (-40, 8),
        "OpenAI Decisions": (-30, 8),
    }
    for name, _, x, y in rows:
        dx, dy = offsets.get(name, (7, -3))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([40, 60, 100, 200, 300])
    ax.set_xticklabels(["40", "60", "100", "200", "300"])
    ax.set_xlim(32, 420)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(35, 80)
    ax.set_xlabel("p50 latency per request (ms, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("System1-Arena: the fastest model is the least accurate")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.text(
        0.0,
        -0.04,
        "Source: System1-Arena (arXiv 2610.21244), Table 1. Local models on an idle H100; "
        "hosted models include network time.",
        fontsize=7.5,
        color=INK2,
    )
    save(fig, "s1arena_frontier")


def calibration(heads: dict) -> None:
    """ECE at the shipped scale vs one fitted scalar, on held-out chunk 0029 (this project)."""
    names = ["pre", "mid", "post"]
    own = [heads[f"authors:{n}"]["calibration"]["own"]["own/ece"] for n in names]
    fit = [heads[f"authors:{n}"]["calibration"]["fitted"]["fitted/ece"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.36
    xs = range(len(names))
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], own, w, color=S1, label="Shipped scale (100)")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], fit, w, color=S2, label="One fitted scalar")
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.003,
                f"{b.get_height():.3f}",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"Authors' {n} head" for n in names])
    ax.set_ylabel("Expected calibration error")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.185)
    ax.set_title("One temperature removes 79–90% of CLM's calibration error")
    ax.legend(fontsize=8.5)
    fig.text(
        0.0,
        -0.05,
        "This project: fit on tricky-Q&A chunk 0028, evaluated on chunk 0029 (n = 95,929); top-1 unchanged.",
        fontsize=7.5,
        color=INK2,
    )
    save(fig, "clm_calibration")


def reliability(heads: dict) -> None:
    """Reliability diagram for the authors' mid head (this project)."""
    cal = heads["authors:mid"]["calibration"]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)))
    for key, color, label in (
        ("bins_own", S1, "Shipped scale (100)"),
        ("bins_fitted", S2, "Fitted scale (69.6)"),
    ):
        pts = [b for b in cal[key] if b["count"] >= 200 and not math.isnan(b["confidence"])]
        ax.plot(
            [b["confidence"] for b in pts],
            [b["accuracy"] for b in pts],
            color=color,
            lw=2,
            marker="o",
            markersize=5,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title("Authors' mid head: reliability")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.text(
        0.0,
        -0.06,
        "This project; bins with ≥ 200 decisions; dashed = perfect calibration. The leftmost bin "
        "(n = 409, confidence ≈ 1/11) is anomalous, likely near-tied rows (not investigated).",
        fontsize=7.5,
        color=INK2,
    )
    save(fig, "clm_reliability")


def matryoshka(heads: dict) -> None:
    """Tricky-Q&A top-1 when truncating the published heads' 512-d output (this project)."""
    dims = [64, 128, 256, 512]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    label_dy = {"pre": -3, "mid": 4, "post": -10}
    for name, color in (("pre", S1), ("mid", S2), ("post", S3)):
        h = heads[f"authors:{name}"]
        ys = [100 * h[f"hard@{d}/top1"] for d in dims[:-1]] + [100 * h["hard/top1"]]
        ax.plot(dims, ys, color=color, lw=2, marker="o", markersize=5, label=f"Authors' {name} head")
        ax.annotate(
            f"{ys[-1]:.1f}",
            (dims[-1], ys[-1]),
            xytext=(6, label_dy[name]),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Embedding dimensions kept (prefix)")
    ax.set_ylabel("Tricky-Q&A top-1 (%)")
    ax.set_title("Untrained truncation is lossy: no Matryoshka training")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.text(0.0, -0.05, "This project: full held-out test, n = 191,161.", fontsize=7.5, color=INK2)
    save(fig, "clm_truncation")


def order_flips() -> None:
    """Option-order flip rates reported by the cited sources (§7.2 FM1)."""
    rows = [  # label, flip %, source
        ("Von 1.1 (4 orderings)", 49.5, "von"),
        ("AnyJev raw readout, K=20", 32.8, "anyjev"),
        ("Laya (native)", 30.6, "runoff"),
        ("SemIf, reversal (10/36)", 27.8, "semif"),
        ("Jev 1.13", 17.0, "jeva"),
        ("Jeff-4B, reversal (approx.)", 6.0, "jeff"),
        ("CLM / Jeva / NDM (by construction)", 0.0, "jevadvbench; jeva; ndm"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [r[1] for r in rows], height=0.6, color=S1)
    for y, r in zip(ys, rows, strict=True):
        label = f"≈{r[1]:.0f}%" if "approx" in r[0] else f"{r[1]:.1f}%"
        ax.text(r[1] + 0.8, y, label, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Answers that change when options are reordered (%)")
    ax.set_xlim(0, 58)
    ax.grid(axis="y", visible=False)
    ax.set_title("Order sensitivity across System One designs")
    fig.text(
        0.0,
        -0.06,
        "Sources: Von, AnyJev, Runoff (2610.18233), SemIf, Jeva (2610.18773), Jeff (2610.17205), "
        "JevAdvBench (2609.31142). Protocols differ; compare within rows.",
        fontsize=7.5,
        color=INK2,
    )
    save(fig, "order_flips")


def main() -> None:
    frontier()
    order_flips()
    heads = json.loads(HEADS.read_text(encoding="utf-8"))
    calibration(heads)
    reliability(heads)
    matryoshka(heads)


if __name__ == "__main__":
    main()

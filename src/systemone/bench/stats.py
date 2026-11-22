"""Aggregate the way the System1-Arena leaderboard does, and bootstrap paired confidence intervals.

**Aggregation.**
* A task's score is its pooled mean over questions.
* A leaf's score is the mean over its tasks.
* A domain's score, and the overall score, are means over leaves.

So leaves with more sources do not weigh more, and the overall number is not an item mean. On the
published board, the mean over domains sits 0.2-1.8 pp below the overall figure. Calibration error is
computed per task and averaged in the same way, as upstream's ``scripts/bootstrap_compare.py`` does.

**Bootstrap.** The bootstrap is paired: every model sees the same resampled data in each replicate, so
the intervals of differences are tight where models agree. Two resampling units are offered.

``unit="item"`` is upstream's resampling:
* items are resampled within each task;
* every task is kept.

``unit="task"`` is our pre-registered default (plan v3, ground rules):
* tasks are resampled within each leaf;
* then items within each task.

Item-level intervals ignore how much tasks differ from one another. On roughly 80k questions they make
a 0.3 pp difference look real.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from systemone.bench.metrics import METRICS, Scored, summarize


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate(scored: Sequence[Scored]) -> dict[str, Any]:
    """Per-task summaries, leaf / domain / overall means over leaves, and pooled per-type summaries."""
    by_task: dict[str, list[Scored]] = defaultdict(list)
    by_type: dict[str, list[Scored]] = defaultdict(list)
    leaf_of: dict[str, str] = {}
    for s in scored:
        by_task[s.task].append(s)
        by_type[s.qtype].append(s)
        leaf_of[s.task] = s.leaf
    tasks = {t: summarize(rows) for t, rows in sorted(by_task.items())}

    leaf_tasks: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        leaf_tasks[leaf_of[t]].append(t)
    leaves = {
        leaf: {m: _mean([tasks[t][m] for t in names]) for m in METRICS} | {"tasks": float(len(names))}
        for leaf, names in sorted(leaf_tasks.items())
    }
    domain_leaves: dict[str, list[str]] = defaultdict(list)
    for leaf in leaves:
        domain_leaves[leaf.split("/", 1)[0]].append(leaf)
    domains = {
        d: {m: _mean([leaves[leaf][m] for leaf in names]) for m in METRICS} | {"leaves": float(len(names))}
        for d, names in sorted(domain_leaves.items())
    }
    overall = {m: _mean([v[m] for v in leaves.values()]) for m in METRICS}
    pooled = summarize(scored)
    overall |= {
        "unanswered_rate": pooled.get("unanswered_rate", float("nan")),
        "questions": float(len(scored)),
        "items": float(len({s.item_id for s in scored})),
        "tasks": float(len(tasks)),
        "leaves": float(len(leaves)),
    }
    return {
        "overall": overall,
        "domains": domains,
        "leaves": leaves,
        "tasks": tasks,
        "types": {q: summarize(rows) for q, rows in sorted(by_type.items())},
        "pooled": pooled,
    }


def _task_arrays(
    scored_by_model: Mapping[str, Sequence[Scored]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Per task: hit ("accuracy") and Brier matrices [models, questions], and each question's item."""
    names = list(scored_by_model)
    keyed = [{(s.item_id, s.question): s for s in scored_by_model[n]} for n in names]
    keys = sorted(keyed[0])
    for n, table in zip(names[1:], keyed[1:], strict=True):
        if set(table) != set(keys):
            raise ValueError(f"{n} was scored on other questions than {names[0]}; pair them first")
    by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_task[keyed[0][key].task].append(key)
    tasks: dict[str, dict[str, Any]] = {}
    for task, task_keys in sorted(by_task.items()):
        item_index = {item_id: i for i, item_id in enumerate(dict.fromkeys(k[0] for k in task_keys))}
        tasks[task] = {
            "leaf": keyed[0][task_keys[0]].leaf,
            "n_items": len(item_index),
            "qitem": np.array([item_index[k[0]] for k in task_keys]),
            "accuracy": np.array([[t[k].hit for k in task_keys] for t in keyed]),
            "brier": np.array([[t[k].brier for k in task_keys] for t in keyed]),
        }
    return names, tasks


def paired_bootstrap(
    scored_by_model: Mapping[str, Sequence[Scored]],
    b: int = 1000,
    unit: Literal["task", "item"] = "task",
    seed: int = 0,
) -> dict[str, Any]:
    """95% intervals of overall and per-domain accuracy and Brier for each model, and of their differences.

    Every model must have been scored on the same questions. ``p_better[a][b]`` is the share of
    replicates in which ``a`` beats ``b`` (higher accuracy, lower Brier).
    """
    if unit not in ("task", "item"):
        raise ValueError(f"unit must be 'task' or 'item', not {unit!r}")
    names, tasks = _task_arrays(scored_by_model)
    rng = np.random.default_rng(seed)
    boot: dict[str, dict[str, np.ndarray]] = {"accuracy": {}, "brier": {}}  # task -> [b, models]
    for task, d in tasks.items():
        n = d["n_items"]
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)[:, d["qitem"]]
        for metric in boot:
            boot[metric][task] = w @ d[metric].T / w.sum(axis=1, keepdims=True)

    leaf_tasks: dict[str, list[str]] = defaultdict(list)
    for task, d in tasks.items():
        leaf_tasks[d["leaf"]].append(task)
    scopes: dict[str, list[str]] = {"overall": sorted(leaf_tasks)}
    for leaf in sorted(leaf_tasks):
        scopes.setdefault(leaf.split("/", 1)[0], []).append(leaf)

    def leaf_values(metric: str) -> dict[str, np.ndarray]:
        out = {}
        for leaf, members in leaf_tasks.items():
            stack = np.stack([boot[metric][t] for t in members])  # [tasks, b, models]
            if unit == "task" and len(members) > 1:
                pick = rng.integers(0, len(members), size=(b, len(members)))
                out[leaf] = stack[pick, np.arange(b)[:, None]].mean(axis=1)
            else:
                out[leaf] = stack.mean(axis=0)
        return out

    def pct(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    ci: dict[str, dict[str, dict[str, list[float]]]] = {}
    diff: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in boot:
        per_leaf = leaf_values(metric)
        for scope, members in scopes.items():
            values = np.mean([per_leaf[leaf] for leaf in members], axis=0)  # [b, models]
            for k, n in enumerate(names):
                ci.setdefault(scope, {}).setdefault(n, {})[metric] = pct(values[:, k])
            if scope != "overall":
                continue
            sign = 1.0 if metric == "accuracy" else -1.0
            for i, a in enumerate(names):
                for j, other in enumerate(names):
                    if i == j:
                        continue
                    delta = values[:, i] - values[:, j]
                    diff.setdefault(a, {}).setdefault(other, {})[metric] = {
                        "ci95": pct(delta),
                        "p_better": float(np.mean(sign * delta > 0)),
                    }
    return {"models": names, "b": b, "unit": unit, "seed": seed, "ci95": ci, "diff": diff}

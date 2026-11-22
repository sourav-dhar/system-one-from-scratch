"""Run the systemone library on Modal GPUs. The same code path runs locally and in the cloud.

    # 1. mirror the chunks a plan needs from the Hub into a Modal volume (idempotent)
    modal run cloud/modal_app.py::stage --plan eval,pre:24,mid:6

    # 2. run one experiment, or a sweep of experiments in parallel containers
    modal run cloud/modal_app.py::train --config configs/pre.yaml --name pre-baseline
    modal run cloud/modal_app.py::sweep --file configs/sweeps/pre_ablations.yaml

    # 3. score checkpoints (ours and the authors') on the full held-out benchmark
    modal run cloud/modal_app.py::benchmark --heads authors:pre,authors:mid,runs/pre-baseline/best.pt

Results (logs, resolved configs, summaries, benchmark JSON) are written back into ``results/``;
checkpoints stay on the volume (``modal volume get systemone-data runs/<name>/best.pt``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

APP = "systemone"
VOLUME = "systemone-data"
ROOT = "/vol"
RESULTS = Path(__file__).resolve().parent.parent / "results"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.3",
        "pyyaml==6.0.2",
        "pyarrow==18.1.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "requests==2.32.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(Path(__file__).resolve().parent.parent / "configs", remote_path="/root/configs")
    .add_local_python_source("systemone")
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
app = modal.App(APP, image=image)


# ---------------------------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------------------------
@app.function(volumes={ROOT: vol}, cpu=4, memory=8192, timeout=3600, max_containers=16, retries=3)
def stage_chunk(chunk: str, negatives: bool) -> tuple[str, int]:
    from systemone.data.remote import download_chunk

    written = download_chunk(ROOT, chunk, negatives)
    vol.commit()
    return chunk, written


def plan_jobs(plan: str) -> list[tuple[str, bool]]:
    from systemone.data.catalog import HELD_OUT_GENERAL, HELD_OUT_HARD, hard_negative_chunks, spread_chunks

    jobs: list[tuple[str, bool]] = []
    for part in plan.split(","):
        kind, _, n = part.partition(":")
        if kind == "eval":
            jobs += [(c, False) for c in HELD_OUT_GENERAL] + [(c, True) for c in HELD_OUT_HARD]
        elif kind == "pre":
            jobs += [(c, False) for c in spread_chunks(int(n))]
        elif kind == "mid":
            jobs += [(c, True) for c in hard_negative_chunks(int(n))]
        else:
            raise ValueError(f"unknown plan part {part!r}")
    return list(dict.fromkeys(jobs))


@app.local_entrypoint()
def stage(plan: str = "eval,pre:24,mid:6") -> None:
    jobs = plan_jobs(plan)
    print(f"staging {len(jobs)} chunks into volume {VOLUME!r}")
    t0, total = time.time(), 0
    for chunk, written in stage_chunk.starmap(jobs, order_outputs=False):
        total += written
        minutes = (time.time() - t0) / 60
        print(f"  {chunk}: {written / 1e9:.2f} GB new  ({total / 1e9:.0f} GB, {minutes:.1f} min)")


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def _eval_suite(hard_in_memory: bool = False):  # type: ignore[no-untyped-def]
    """Held-out sets: general fully in RAM; tricky Q&A with negatives memory-mapped (16 GB)."""
    import numpy as np

    from systemone.data import PairSet, load_chunks
    from systemone.data.catalog import HELD_OUT_GENERAL, HELD_OUT_HARD
    from systemone.eval import EvalSuite

    general = load_chunks(ROOT, list(HELD_OUT_GENERAL))
    hard = load_chunks(ROOT, list(HELD_OUT_HARD), negatives=True, mmap=True)
    hard = PairSet(
        np.ascontiguousarray(hard.states),
        np.ascontiguousarray(hard.actions),
        hard.doc,
        np.ascontiguousarray(hard.negatives) if hard_in_memory else hard.negatives,
        hard.neg_count,
    )
    return EvalSuite(general=general, hard=hard)


@app.function(volumes={ROOT: vol}, gpu="L40S", cpu=8, memory=81920, timeout=6 * 3600)
def run_experiment(config_path: str, overrides: list[str], name: str) -> dict[str, Any]:
    from systemone.config import dump_config, load_config
    from systemone.data import load_chunks
    from systemone.data.catalog import resolve_chunks
    from systemone.train import Trainer

    vol.reload()
    out = f"{ROOT}/runs/{name}"
    cfg = load_config(f"/root/{config_path}", [*overrides, f"name={name}", f"train.out_dir={out}"])
    if cfg.train.init_checkpoint and not cfg.train.init_checkpoint.startswith("/"):
        cfg.train.init_checkpoint = f"{ROOT}/{cfg.train.init_checkpoint}"
    t0 = time.time()
    chunks = resolve_chunks(cfg.data.train_chunks)
    train = load_chunks(
        ROOT, chunks, negatives=cfg.data.hard_negatives, rows_per_chunk=cfg.data.rows_per_chunk
    )
    if cfg.data.hard_negatives and cfg.data.max_negatives is not None:
        train = train.limit_negatives(cfg.data.max_negatives)
    suite = _eval_suite()
    load_min = (time.time() - t0) / 60
    print(
        f"[{name}] loaded {len(train):,} train pairs ({train.nbytes / 1e9:.1f} GB) in {load_min:.1f} min",
        flush=True,
    )

    def log(msg: str) -> None:
        print(f"[{name}] {msg}", flush=True)

    result = Trainer(cfg, train, suite, device="cuda", log=log).fit()
    vol.commit()
    return {
        "name": name,
        "config": dump_config(cfg),
        "log": Path(out, "log.jsonl").read_text(encoding="utf-8"),
        "summary": {
            "steps": result.steps,
            "best_step": result.best_step,
            "best_metric": result.best_metric,
            "select_metric": cfg.select_metric(),
            "minutes_training": result.minutes,
            "minutes_loading": load_min,
            "train_pairs": len(train),
            "final": result.final,
            "best": result.best,
        },
    }


def _save_run(res: dict[str, Any]) -> None:
    d = RESULTS / res["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(res["config"], encoding="utf-8")
    (d / "log.jsonl").write_text(res["log"], encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(res["summary"], indent=1), encoding="utf-8")
    final = res["summary"]["final"]
    shown = {k: round(v, 4) for k, v in final.items() if k in ("general/top1", "hard/top1", "hard/ece")}
    print(f"saved results/{res['name']}  {shown}")


@app.local_entrypoint()
def train(config: str, name: str, override: str = "") -> None:
    """``name`` like ``pre/baseline``: volume ``runs/pre/baseline``, local ``results/pre/baseline``."""
    overrides = [o for o in override.split(";") if o]
    _save_run(run_experiment.remote(config, overrides, name))


@app.local_entrypoint()
def sweep(file: str, only: str = "") -> None:
    """Run every experiment in a sweep file in parallel; ``only`` filters by comma-separated names.

    Sweep file format::

        group: pre                    # runs/<group>/<name> on the volume, results/<group>/<name>/ here
        config: configs/pre.yaml      # base config for every experiment
        experiments:
          baseline: []
          no-doc-mask: [loss.mask_same_document=false]
    """
    import yaml

    spec = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
    names = [n for n in spec["experiments"] if not only or n in only.split(",")]
    group = spec["group"]
    args = [(spec["config"], list(spec["experiments"][n] or []), f"{group}/{n}") for n in names]
    print(f"launching {len(args)} runs: {', '.join(names)}")
    for res in run_experiment.starmap(args, order_outputs=False, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"FAILED: {res!r}")
            continue
        _save_run(res)


# ---------------------------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------------------------
@app.function(volumes={ROOT: vol}, gpu="L40S", cpu=8, memory=49152, timeout=3600)
def run_benchmark(heads: list[str], dims: list[int]) -> dict[str, Any]:
    """Every head on the full held-out sets, plus a calibration study.

    Calibration: fit one inverse temperature on chunk 0028 of the tricky test, report chunk 0029
    at the head's own scale and at the fitted scale.
    """
    import torch
    from huggingface_hub import hf_hub_download

    from systemone.data.catalog import AUTHORS_HEADS, AUTHORS_HEADS_REPO
    from systemone.eval import fit_scale, general_metrics, hard_scores, reliability_bins
    from systemone.eval.metrics import summarize_hard
    from systemone.model import DualEncoder

    vol.reload()
    suite = _eval_suite(hard_in_memory=True)
    assert suite.hard is not None
    assert suite.general is not None
    assert suite.hard.neg_count is not None
    first_chunk_rows = int((suite.hard.neg_count[:100_000] > 0).sum())
    out: dict[str, Any] = {}
    for spec in heads:
        if spec.startswith("authors:"):
            path = hf_hub_download(
                AUTHORS_HEADS_REPO, AUTHORS_HEADS[spec.split(":")[1]], local_dir=f"{ROOT}/heads"
            )
        else:
            path = f"{ROOT}/{spec}"
        model = DualEncoder.load(path).cuda()
        s = model.scale().item()
        t0 = time.time()
        res: dict[str, Any] = {"scale": s, **general_metrics(model, suite.general)}
        scores = hard_scores(model, suite.hard, tuple(dims))
        full = model.cfg.head.out_dim
        for d, cos in scores.items():
            res.update(summarize_hard(cos, s, "hard" if d == full else f"hard@{d}"))
        cos = scores[full]
        fit, test = cos[:first_chunk_rows], cos[first_chunk_rows:]
        s_fit = fit_scale(fit, init=s)
        own, cal = summarize_hard(test, s, "own"), summarize_hard(test, s_fit, "fitted")
        p_own = torch.from_numpy(s * test).softmax(1).numpy()
        p_fit = torch.from_numpy(s_fit * test).softmax(1).numpy()
        res["calibration"] = {
            "fitted_scale": s_fit,
            "test_rows": len(test),
            "own": own,
            "fitted": cal,
            # correct = gold strictly above every negative (ties are misses, as in top-1)
            "bins_own": reliability_bins(p_own.max(1), p_own[:, 0] > p_own[:, 1:].max(1)),
            "bins_fitted": reliability_bins(p_fit.max(1), p_fit[:, 0] > p_fit[:, 1:].max(1)),
        }
        res["seconds"] = time.time() - t0
        out[spec] = res
        print(spec, json.dumps({k: round(v, 4) for k, v in res.items() if isinstance(v, float)}), flush=True)
        del model
    return out


@app.local_entrypoint()
def benchmark(heads: str, out: str = "results/benchmark/heads.json", dims: str = "") -> None:
    res = run_benchmark.remote(heads.split(","), [int(d) for d in dims.split(",") if d])
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged.update(res)
    path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"wrote {path}")

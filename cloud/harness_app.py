"""The benchmark harness on Modal GPUs: reference models and ours on the pinned suites, one code path.

    # smoke test: the first 3 items of every static task, one GPU
    modal run --detach cloud/harness_app.py::answer --kind readout --spec Qwen/Qwen3-8B --limit 3 --shards 1
    # the M0.4 anchors on the 13 static S1-Arena domains
    modal run --detach cloud/harness_app.py::answer --kind readout --spec Qwen/Qwen3-8B --shards 4
    modal run --detach cloud/harness_app.py::answer --kind clm --spec Contrastive-LM/CLM-v0.1-8B --shards 2
    # merge and score what the shards wrote (also after a disconnect)
    modal run cloud/harness_app.py::collect --kind readout --spec Qwen/Qwen3-8B

Each shard appends its responses to ``references/s1arena/<model slug>/<scope>/shard-<i>-of-<n>.jsonl`` on
the volume, committing after every chunk. A preempted or failed shard therefore resumes where it
stopped when the same command is run again.

The local entrypoint merges the shards into ``results/references/s1arena/<slug>/<scope>/``:
* ``responses.jsonl.gz``, in S1-Arena's run-row layout;
* ``summary.json``, scored by :mod:`systemone.bench` and set against the published leaderboard
  (``results/references/s1arena/leaderboard-2026-10-14.json``).
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import modal

APP = "systemone-harness"
VOLUME = "systemone-data"
ROOT = "/vol"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "references" / "s1arena"
LEADERBOARD = RESULTS / "leaderboard-2026-10-14.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "accelerate==1.10.1",
        "numpy==2.1.3",
        "pyarrow==18.1.0",
        "pyyaml==6.0.2",
        "pydantic==2.11.9",
        "huggingface_hub[hf_transfer]==0.35.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{ROOT}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("systemone")
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
app = modal.App(APP, image=image)


def _domains(scope: str) -> list[str]:
    from systemone.decisions import s1arena

    if scope == "static":
        return list(s1arena.STATIC_DOMAINS)
    if scope == "agent":
        return list(s1arena.AGENT_DOMAINS)
    return scope.split(",")


def _items(scope: str, limit: int) -> list[Any]:
    """Pinned S1-Arena items of a scope; with ``limit``, the first ``limit`` items of every task."""
    from systemone.decisions import s1arena

    items = s1arena.load(_domains(scope))
    if limit:
        seen: dict[str, int] = {}
        kept = []
        for item in items:
            seen[item.task] = seen.get(item.task, 0) + 1
            if seen[item.task] <= limit:
                kept.append(item)
        items = kept
    return items


def _run_name(scope: str, limit: int) -> str:
    return scope.replace(",", "+") + (f"-first{limit}" if limit else "")


KINDS = ("readout", "clm")


def model_id_of(kind: str, spec: str) -> str:
    """The id a model is stored and published under, known without loading it."""
    if kind == "readout":
        return f"{spec}@label-think0"
    if kind == "clm":
        return spec
    raise ValueError(f"unknown kind {kind!r}; known: {KINDS}")


def _make_model(kind: str, spec: str) -> Any:
    if kind == "readout":
        from systemone.bench.readout import Readout

        return Readout.from_pretrained(spec)
    from systemone.bench.clm import CLMDecider

    return CLMDecider.from_pretrained(spec)


def _volume_dir(model_id: str, scope: str, limit: int) -> str:
    from systemone.bench.runner import slug

    return f"references/s1arena/{slug(model_id)}/{_run_name(scope, limit)}"


@app.function(gpu="H100", cpu=8, memory=65536, volumes={ROOT: vol}, timeout=8 * 3600)
def answer_shard(kind: str, spec: str, scope: str, limit: int, shard: int, shards: int) -> dict[str, Any]:
    from systemone.bench.runner import run

    vol.reload()
    items = _items(scope, limit)[shard::shards]
    model = _make_model(kind, spec)
    assert model.model_id == model_id_of(kind, spec)
    path = Path(ROOT, _volume_dir(model.model_id, scope, limit), f"shard-{shard}-of-{shards}.jsonl")

    def log(msg: str) -> None:
        print(f"[shard {shard}/{shards}] {msg}", flush=True)

    t0 = time.time()
    asked = run(model, items, path, chunk=500, log=log, on_chunk=vol.commit)
    vol.commit()
    return {"shard": shard, "items": len(items), "asked": asked, "minutes": (time.time() - t0) / 60}


def _published(model_id: str) -> dict[str, Any] | None:
    board = json.loads(LEADERBOARD.read_text(encoding="utf-8"))
    for row in board["models"]:
        if (row.get("ids") or {}).get("static") == model_id or row["name"] == model_id:
            return row  # type: ignore[no-any-return]
    return None


def summarize_run(model_id: str, scope: str, limit: int, responses_path: Path) -> dict[str, Any]:
    """Score a merged run and set it against the published leaderboard row (static scope)."""
    from systemone.bench import aggregate, score
    from systemone.bench.runner import read_responses
    from systemone.decisions import s1arena

    items = _items(scope, limit)
    responses = read_responses(responses_path)
    scored = score(items, responses)
    agg = aggregate(scored)
    summary: dict[str, Any] = {
        "model": model_id,
        "suite": {
            "name": "System1-Arena",
            "dataset": s1arena.DATASET,
            "revision": s1arena.REVISION,
            "harness_commit": s1arena.HARNESS_COMMIT,
            "scope": scope,
            "limit_per_task": limit,
        },
        "answered_items": len(responses),
        "aggregate": {k: agg[k] for k in ("overall", "domains", "types", "pooled", "leaves")},
        "tasks": agg["tasks"],
    }
    published = _published(model_id)
    if published and scope == "static" and not limit:
        complete = [d for d in s1arena.STATIC_DOMAINS if d not in s1arena.INCOMPLETE_DOMAINS]
        domains = {
            d: {
                "ours": agg["domains"][d]["accuracy"],
                "published": published["domains"][d],
                "delta_pp": 100 * (agg["domains"][d]["accuracy"] - published["domains"][d]),
                "complete": d in complete,
            }
            for d in s1arena.STATIC_DOMAINS
            if d in agg["domains"]
        }
        ours_c = [domains[d]["ours"] for d in complete]
        pub_c = [domains[d]["published"] for d in complete]
        summary["vs_published"] = {
            "leaderboard": "2026-10-14",
            "accuracy": {"ours": agg["overall"]["accuracy"], "published": published["accuracy"]},
            "brier": {"ours": agg["overall"]["brier"], "published": published["brier"]},
            "delta_pp": 100 * (agg["overall"]["accuracy"] - published["accuracy"]),
            "complete_domains_mean_delta_pp": 100 * (sum(ours_c) - sum(pub_c)) / len(complete),
            "domains": domains,
            "note": "The public set lacks 14 restricted sources (2,043 static requests in 9 leaves of "
            "basic_cls, fore, pro, ret, social); the other 8 domains compare one to one.",
        }
    return summary


@app.local_entrypoint()
def answer(
    kind: str = "readout", spec: str = "Qwen/Qwen3-8B", scope: str = "static", limit: int = 0, shards: int = 4
) -> None:
    """Answer every item on ``shards`` H100s, then collect. Run it with ``--detach``: the shards keep
    going if this client disconnects, and ``collect`` merges and scores what is on the volume."""
    model_id = model_id_of(kind, spec)
    print(f"{model_id} on S1-Arena {scope} (limit {limit or 'none'}) over {shards} H100 shard(s)")
    for res in answer_shard.starmap([(kind, spec, scope, limit, i, shards) for i in range(shards)]):
        print(
            f"  shard {res['shard']}: {res['asked']:,} of {res['items']:,} asked in {res['minutes']:.1f} min"
        )
    collect(kind, spec, scope, limit)


@app.local_entrypoint()
def collect(
    kind: str = "readout", spec: str = "Qwen/Qwen3-8B", scope: str = "static", limit: int = 0
) -> None:
    """Merge a run's shard files from the volume into results/, score them and compare."""
    from systemone.bench.runner import slug

    model_id = model_id_of(kind, spec)
    remote = _volume_dir(model_id, scope, limit)
    out = RESULTS / slug(model_id) / _run_name(scope, limit)
    out.mkdir(parents=True, exist_ok=True)
    merged = out / "responses.jsonl.gz"
    rows = 0
    with gzip.open(merged, "wt", encoding="utf-8", newline="\n") as f:
        for entry in sorted(vol.listdir(remote), key=lambda e: e.path):
            if entry.path.endswith(".jsonl"):
                text = b"".join(vol.read_file(entry.path)).decode("utf-8")
                rows += text.count("\n")
                f.write(text)
    print(f"merged {rows:,} rows from {remote} into {merged}")
    summary = summarize_run(model_id, scope, limit, merged)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    o = summary["aggregate"]["overall"]
    print(
        f"accuracy {o['accuracy']:.4f}  brier {o['brier']:.4f}  nll {o['nll']:.4f}  "
        f"ece(mass10) {o['ece_mass10']:.4f}  unanswered {o['unanswered_rate']:.4f}"
    )
    if "vs_published" in summary:
        v = summary["vs_published"]
        print(
            f"published {v['accuracy']['published']:.4f}: delta {v['delta_pp']:+.2f} pp; "
            f"complete-domain mean delta {v['complete_domains_mean_delta_pp']:+.2f} pp"
        )
        for d, row in v["domains"].items():
            flag = "" if row["complete"] else "  (incomplete)"
            ours, pub = row["ours"], row["published"]
            print(f"  {d:<10} ours {ours:.4f}  published {pub:.4f}  {row['delta_pp']:+.2f}{flag}")
    print(f"wrote {out / 'summary.json'}")

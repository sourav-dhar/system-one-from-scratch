"""Command-line interface.

systemone demo                                   # synthetic end-to-end run, ~1 min on a laptop CPU
systemone train -c configs/pre.yaml -o train.max_steps=2000
systemone eval runs/pre/best.pt --data data      # score any head on the held-out tests
systemone fetch --rows 20000 --data data         # laptop-sized subset via HTTP range reads
systemone info runs/pre/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from systemone.config import ExperimentConfig, SyntheticConfig, load_config


def _load_data(cfg: ExperimentConfig):  # type: ignore[no-untyped-def]
    from systemone.data import load_chunks, make_synthetic
    from systemone.data.catalog import resolve_chunks
    from systemone.eval import EvalSuite

    if cfg.data.synthetic is not None:
        train, held = make_synthetic(cfg.data.synthetic)
        n = len(held) // 2
        suite = EvalSuite(
            general=held.subset(slice(0, n)).without_negatives(), hard=held.subset(slice(n, None))
        )
        return (train if cfg.data.hard_negatives else train.without_negatives()), suite
    d = cfg.data
    chunks = resolve_chunks(d.train_chunks)
    train = load_chunks(d.root, chunks, negatives=d.hard_negatives, rows_per_chunk=d.rows_per_chunk)
    general = (
        load_chunks(d.root, d.eval_general, rows_per_chunk=d.eval_rows_per_chunk) if d.eval_general else None
    )
    hard = (
        load_chunks(d.root, d.eval_hard, negatives=True, rows_per_chunk=d.eval_rows_per_chunk)
        if d.eval_hard
        else None
    )
    return train, EvalSuite(general=general, hard=hard)


def cmd_train(args: argparse.Namespace) -> int:
    from systemone.train import Trainer

    cfg = load_config(args.config, args.override)
    train, suite = _load_data(cfg)
    result = Trainer(cfg, train, suite, device=args.device).fit()
    print(
        json.dumps(
            {"out_dir": result.out_dir, "best_step": result.best_step, "final": result.final}, indent=1
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    from systemone.data import load_chunks
    from systemone.eval import EvalSuite, evaluate
    from systemone.model import DualEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = args.rows
    suite = EvalSuite(
        general=load_chunks(args.data, args.general, rows_per_chunk=rows) if args.general else None,
        hard=load_chunks(args.data, args.hard, negatives=True, rows_per_chunk=rows) if args.hard else None,
    )
    results = {}
    for path in args.checkpoints:
        model = DualEncoder.load(path).to(device)
        results[path] = evaluate(model, suite, dims=tuple(args.dims))
        print(path, json.dumps({k: round(v, 4) for k, v in results[path].items()}))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from systemone.train import Trainer

    cfg = load_config(
        None,
        [
            "name=synthetic-demo",
            f"train.out_dir={args.out}",
            "model.head.in_dim=256",
            "model.head.width=256",
            "model.head.out_dim=128",
            "train.batch_size=1024",
            f"train.max_steps={args.steps}",
            "train.eval_every=100",
            "train.precision=fp32",
            "data.hard_negatives=true",
            "train.stage=mid",
            "optim.lr=0.002",
        ],
    )
    cfg.data.synthetic = SyntheticConfig()
    train, suite = _load_data(cfg)
    r = Trainer(cfg, train, suite, device="cpu").fit()
    first = json.loads((Path(args.out) / "log.jsonl").read_text().splitlines()[0])
    for key, label in (("hard/top1", "tricky top-1 "), ("general/top1", "general top-1")):
        print(f"{label}: {first[key]:.3f} (random heads) -> {r.final[key]:.3f} (trained)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from systemone.data.catalog import HELD_OUT_GENERAL, HELD_OUT_HARD, spread_chunks
    from systemone.data.remote import mirror_subset

    jobs = [(c, False) for c in spread_chunks(args.train_chunks)]
    jobs += [(c, False) for c in HELD_OUT_GENERAL] + [(c, True) for c in HELD_OUT_HARD]
    for chunk, negs in jobs:
        print(f"fetching {args.rows:,} rows of {chunk}{' (+negatives)' if negs else ''}", flush=True)
        mirror_subset(args.data, chunk, args.rows, negatives=negs)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from systemone.model import DualEncoder

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DualEncoder.from_checkpoint(ck)
    info = {
        "head": vars(model.cfg.head),
        "shared_heads": model.cfg.shared_heads,
        "scale": round(model.scale().item(), 3),
        "parameters": model.num_parameters(),
        "step": ck.get("step"),
    }
    print(json.dumps(info, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="systemone", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train heads from a YAML config")
    t.add_argument("-c", "--config", required=True)
    t.add_argument(
        "-o", "--override", action="append", default=[], help="dotted override, e.g. train.max_steps=100"
    )
    t.add_argument("--device", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="score checkpoints on the held-out tests")
    e.add_argument("checkpoints", nargs="+")
    e.add_argument("--data", default="data")
    e.add_argument("--general", nargs="*", default=["shard_05/chunk_0083", "shard_05/chunk_0084"])
    e.add_argument("--hard", nargs="*", default=["shard_00/chunk_0028", "shard_00/chunk_0029"])
    e.add_argument("--rows", type=int, default=None, help="first N rows per chunk")
    e.add_argument("--dims", type=int, nargs="*", default=[], help="also score these embedding prefixes")
    e.add_argument("--device", default=None)
    e.add_argument("--out", default=None, help="write results JSON here")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="synthetic end-to-end run (CPU, about a minute)")
    d.add_argument("--steps", type=int, default=300)
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fetch", help="download a laptop-sized subset of the real vectors")
    f.add_argument("--data", default="data")
    f.add_argument("--rows", type=int, default=20_000)
    f.add_argument("--train-chunks", type=int, default=4)
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("info", help="describe a checkpoint")
    i.add_argument("checkpoint")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())

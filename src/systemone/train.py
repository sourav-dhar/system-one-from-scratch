"""Training loop shared by every stage (pre, mid, post) and every experiment.

The loop is deliberately small: sample a batch, project, contrastive loss, AdamW step. Everything
that differs between stages and ablations lives in :class:`~systemone.config.ExperimentConfig`.

Outputs in ``train.out_dir``:

* ``config.yaml`` - the fully resolved config;
* ``log.jsonl`` - one line per eval (and a final ``"finished"`` line);
* ``best.pt`` - weights with the best ``select_metric`` (CLM checkpoint format);
* ``last.pt`` - weights + optimizer state, used to resume an interrupted run.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from systemone.config import ExperimentConfig, dump_config
from systemone.data.pairs import PairSet
from systemone.data.sampler import BatchSampler
from systemone.eval.metrics import EvalSuite, evaluate
from systemone.losses import contrastive_loss
from systemone.model import DualEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(cfg: ExperimentConfig) -> Callable[[int], float]:
    """Multiplier on the base LR at a given step: optional linear warmup, then constant or cosine."""
    o, total = cfg.optim, cfg.train.max_steps

    def f(step: int) -> float:
        if o.warmup_steps and step < o.warmup_steps:
            return (step + 1) / o.warmup_steps
        if o.schedule == "constant":
            return 1.0
        progress = min(1.0, (step - o.warmup_steps) / max(1, total - o.warmup_steps))
        return o.min_lr_ratio + (1 - o.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


@dataclass
class TrainResult:
    out_dir: str
    steps: int
    best_metric: float
    best_step: int
    minutes: float
    final: dict[str, float] = field(default_factory=dict)
    best: dict[str, float] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        train_data: PairSet,
        eval_suite: EvalSuite,
        device: str | torch.device | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_data = train_data
        self.eval_suite = eval_suite
        self.log = log
        self.out = Path(cfg.train.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.train.seed)

        if cfg.train.init_checkpoint:
            self.model = DualEncoder.load(cfg.train.init_checkpoint)
            # stage-specific temperature settings still apply on top of the loaded weights
            self.model.cfg.learn_temperature = cfg.model.learn_temperature
            self.model.logit_scale.requires_grad_(cfg.model.learn_temperature)
        else:
            self.model = DualEncoder(cfg.model)
        self.model.to(self.device)
        if self.model.cfg.head.in_dim != train_data.dim:
            raise ValueError(
                f"head expects {self.model.cfg.head.in_dim}-d inputs, data is {train_data.dim}-d"
            )

        self.lr = cfg.resolved_lr()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            params, lr=self.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg))
        self.use_negatives = train_data.has_negatives
        self.step = 0
        self.best = float("inf")
        self.best_step = 0
        self.best_metrics: dict[str, float] = {}
        self.stale = 0

    # -- checkpoints -----------------------------------------------------------------------
    def _resume(self) -> None:
        last = self.out / "last.pt"
        if not last.exists():
            return
        ck = torch.load(last, map_location="cpu", weights_only=True)
        self.model.state_head.load_state_dict(ck["state_head"])
        self.model.action_head.load_state_dict(ck["action_head"])
        self.model.logit_scale.data.copy_(ck["logit_scale"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.step, self.best, self.best_step, self.stale = (
            ck["step"],
            ck["best"],
            ck["best_step"],
            ck["stale"],
        )
        self.log(f"resumed from step {self.step}")

    def _save_last(self) -> None:
        self.model.save(
            self.out / "last.pt",
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict(),
            step=self.step,
            best=self.best,
            best_step=self.best_step,
            stale=self.stale,
        )

    # -- evaluation ------------------------------------------------------------------------
    def _evaluate(self, full: bool, t0: float, logf: Any) -> dict[str, float]:
        m = evaluate(self.model, self.eval_suite, hard=full, dims=tuple(self.cfg.loss.matryoshka_dims))
        m.update(step=self.step, lr=float(self.sched.get_last_lr()[0]), minutes=(time.time() - t0) / 60)
        key = self.cfg.select_metric()
        if key in m:
            if m[key] < self.best - 1e-4:
                self.best, self.best_step, self.stale = m[key], self.step, 0
                self.best_metrics = dict(m)
                self.model.save(self.out / "best.pt", step=self.step, metrics=m)
                m["best"] = 1.0
            else:
                self.stale += 1
        logf.write(json.dumps(m) + "\n")
        logf.flush()
        shown = {k: round(v, 4) for k, v in m.items() if k.endswith(("loss", "top1", "ece")) or k == "scale"}
        self.log(f"[eval] step {self.step} {shown}")
        return m

    def _full_eval_due(self) -> bool:
        every = self.cfg.train.full_eval_every
        return every is None or self.step % every == 0

    # -- training --------------------------------------------------------------------------
    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        bf16 = self.cfg.train.precision == "bf16" and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=bf16):
            ps = self.model.project_state(batch["states"].float())
            pa = self.model.project_action(batch["actions"].float())
            pn = None
            if self.use_negatives:
                negs = batch["negatives"]
                pn = self.model.project_action(negs.reshape(-1, negs.shape[-1]).float()).view(
                    negs.shape[0], negs.shape[1], -1
                )
        out = contrastive_loss(
            ps,
            pa,
            self.model.scale(),
            self.cfg.loss,
            doc=batch["doc"],
            proj_negatives=pn,
            neg_count=batch.get("neg_count"),
        )
        self.opt.zero_grad(set_to_none=True)
        out.loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.opt.step()
        self.sched.step()
        return {"loss": out.loss.item(), "acc": out.accuracy.item(), "grad_norm": float(gnorm)}

    def fit(self) -> TrainResult:
        cfg = self.cfg
        (self.out / "config.yaml").write_text(dump_config(cfg), encoding="utf-8")
        self._resume()
        sampler = BatchSampler(
            self.train_data, cfg.train.batch_size, seed=cfg.train.seed + self.step, device=self.device
        )
        n_params = self.model.num_parameters(trainable_only=True)
        self.log(
            f"{cfg.name}: {len(self.train_data):,} pairs, {n_params / 1e6:.2f}M trainable params, "
            f"lr {self.lr:.5f}, batch {cfg.train.batch_size}, {cfg.train.max_steps} steps on {self.device}"
        )
        t0 = time.time()
        last_metrics: dict[str, float] = {}
        with open(self.out / "log.jsonl", "a", encoding="utf-8") as logf:
            if self.step == 0:
                last_metrics = self._evaluate(True, t0, logf)
            it = iter(sampler)
            tick, seen = time.time(), 0
            while self.step < cfg.train.max_steps:
                stats = self._train_step(next(it))
                self.step += 1
                seen += cfg.train.batch_size
                if self.step % cfg.train.log_every == 0 or self.step == 1:
                    rate = seen / max(1e-9, time.time() - tick)
                    self.log(
                        f"step {self.step} loss {stats['loss']:.4f} acc {stats['acc']:.3f} "
                        f"scale {self.model.scale().item():.1f} gnorm {stats['grad_norm']:.2f} "
                        f"{rate:,.0f} pairs/s"
                    )
                    tick, seen = time.time(), 0
                if self.step % cfg.train.eval_every == 0 or self.step == cfg.train.max_steps:
                    last_metrics = self._evaluate(
                        self._full_eval_due() or self.step == cfg.train.max_steps, t0, logf
                    )
                    self._save_last()
                    if cfg.train.patience is not None and self.stale >= cfg.train.patience:
                        self.log(f"early stop: no improvement in {self.stale} evals")
                        break
            minutes = (time.time() - t0) / 60
            done = {
                "finished": True,
                "step": self.step,
                "best": self.best,
                "best_step": self.best_step,
                "minutes": minutes,
            }
            logf.write(json.dumps(done) + "\n")
        self._save_last()
        if not (self.out / "best.pt").exists():  # select metric never computed: keep the final weights
            self.model.save(self.out / "best.pt", step=self.step, metrics=last_metrics)
        return TrainResult(
            str(self.out), self.step, self.best, self.best_step, minutes, last_metrics, self.best_metrics
        )

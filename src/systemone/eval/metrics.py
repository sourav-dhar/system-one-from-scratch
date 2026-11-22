"""Held-out metrics, computed the same way for every head (ours or the authors').

* **general** - in-batch retrieval on a fixed seeded permutation, in blocks of 1,024 pairs with
  same-document masking. Reports the symmetric InfoNCE loss and state->action top-1. This
  matches the reference protocol exactly, so numbers are comparable across repos.
* **hard** ("tricky Q&A") - each question must rank its gold answer above *its own* mined hard
  negatives (up to 10). Top-1 counts a hit only when ``gold > max(negatives)`` strictly, the
  published definition. We also report the 11-way cross-entropy, MRR and calibration of the
  softmax probabilities, because a System One model's output *is* a probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from systemone.data.pairs import PairSet
from systemone.eval.calibration import expected_calibration_error
from systemone.losses import same_document_mask
from systemone.model import DualEncoder


@dataclass
class EvalSuite:
    general: PairSet | None = None
    hard: PairSet | None = None


def _t(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device).float()


def _device(model: DualEncoder) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def general_metrics(model: DualEncoder, data: PairSet, block: int = 1024) -> dict[str, float]:
    dev = _device(model)
    was_training = model.training
    model.eval()
    s = model.scale()
    perm = np.random.default_rng(0).permutation(len(data))
    losses: list[float] = []
    hits = hits5 = total = 0.0
    for i in range(0, len(perm) - block + 1, block):
        b = np.sort(perm[i : i + block])
        zs = model.encode_state(_t(data.states[b], dev))
        za = model.encode_action(_t(data.actions[b], dev))
        doc = torch.from_numpy(data.doc[b]).to(dev)
        lg = (s * zs @ za.t()).masked_fill(same_document_mask(doc), float("-inf"))
        y = torch.arange(len(b), device=dev)
        losses.append(((F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2).item())
        hits += (lg.argmax(1) == y).sum().item()
        hits5 += (lg.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
        total += len(b)
    model.train(was_training)
    if total == 0:
        raise ValueError(f"general eval needs at least {block} pairs, got {len(data)}")
    return {
        "general/loss": float(np.mean(losses)),
        "general/top1": hits / total,
        "general/top5": hits5 / total,
        "general/n": float(total),
    }


@torch.no_grad()
def hard_scores(
    model: DualEncoder, data: PairSet, dims: tuple[int, ...] = (), block: int = 4096
) -> dict[int, np.ndarray]:
    """Cosine scores ``[N, 1 + K]`` (gold first, padding = -inf) for each embedding prefix size.

    Rows with no valid negative are dropped, as in the published test. ``dims`` defaults to the
    full projection; smaller values evaluate Matryoshka prefixes of the same embedding.
    """
    if data.negatives is None or data.neg_count is None:
        raise ValueError("hard metrics need a PairSet with negatives")
    dev = _device(model)
    was_training = model.training
    model.eval()
    full = model.cfg.head.out_dim
    dims = tuple(sorted(set(dims) | {full})) if dims else (full,)
    keep = data.neg_count > 0
    out: dict[int, list[np.ndarray]] = {d: [] for d in dims}
    k = data.negatives.shape[1]
    for i in range(0, len(data), block):
        j = min(i + block, len(data))
        rows = keep[i:j]
        if not rows.any():
            continue
        ps = model.project_state(_t(data.states[i:j][rows], dev))
        pa = model.project_action(_t(data.actions[i:j][rows], dev))
        negs = data.negatives[i:j][rows]
        pn = model.project_action(_t(negs, dev).reshape(-1, negs.shape[-1])).view(len(ps), k, -1)
        count = torch.from_numpy(data.neg_count[i:j][rows]).to(dev)
        pad = torch.arange(k, device=dev)[None, :] >= count[:, None]
        for d in dims:
            zs = F.normalize(ps[:, :d], dim=-1)
            pos = (zs * F.normalize(pa[:, :d], dim=-1)).sum(-1, keepdim=True)
            neg = torch.einsum("bd,bkd->bk", zs, F.normalize(pn[..., :d], dim=-1)).masked_fill(
                pad, float("-inf")
            )
            out[d].append(torch.cat([pos, neg], 1).cpu().numpy())
    model.train(was_training)
    return {d: np.concatenate(v) for d, v in out.items()}


def summarize_hard(cos: np.ndarray, scale: float, prefix: str = "hard") -> dict[str, float]:
    """Top-1, MRR, NLL, Brier and ECE from gold-first cosine scores and an inverse temperature."""
    pos, neg = cos[:, 0], cos[:, 1:]
    top1 = float((pos > neg.max(1)).mean())
    rank = 1 + (neg >= pos[:, None]).sum(1)
    logits = torch.from_numpy(scale * cos).double()
    logp = logits.log_softmax(1)
    p = logp.exp()
    nll = float(-logp[:, 0].mean())
    onehot = torch.zeros_like(p)
    onehot[:, 0] = 1
    brier = float(((p - onehot) ** 2).sum(1).mean())
    conf = p.max(1).values
    # correct means strictly above every negative, the same rule as top-1: an exact tie is a miss
    # (argmax would break ties towards the gold in column 0 and flatter calibration).
    ece = expected_calibration_error(conf.numpy(), pos > neg.max(1))
    return {
        f"{prefix}/top1": top1,
        f"{prefix}/mrr": float((1.0 / rank).mean()),
        f"{prefix}/loss": nll,
        f"{prefix}/brier": brier,
        f"{prefix}/ece": ece,
        f"{prefix}/p_gold": float(p[:, 0].mean()),
        f"{prefix}/n": float(len(cos)),
    }


def hard_metrics(model: DualEncoder, data: PairSet, dims: tuple[int, ...] = ()) -> dict[str, float]:
    scores = hard_scores(model, data, dims)
    s = model.scale().item()
    out: dict[str, float] = {}
    full = model.cfg.head.out_dim
    for d, cos in scores.items():
        out.update(summarize_hard(cos, s, "hard" if d == full else f"hard@{d}"))
    return out


def evaluate(
    model: DualEncoder, suite: EvalSuite, hard: bool = True, dims: tuple[int, ...] = ()
) -> dict[str, float]:
    out: dict[str, float] = {"scale": model.scale().item()}
    if suite.general is not None:
        out.update(general_metrics(model, suite.general))
    if hard and suite.hard is not None:
        out.update(hard_metrics(model, suite.hard, dims))
    return out

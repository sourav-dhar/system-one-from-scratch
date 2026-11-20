"""Contrastive objectives for System One heads.

The base objective is the symmetric InfoNCE used by CLIP and by CLM:

* state -> action ("forward"): each state must pick its own action among every action in the
  batch, plus that state's mined hard negatives when they are available;
* action -> state ("backward"): each action must pick its own state among every state.

Pairs that come from the same source document are masked out of each other's denominators,
because a second question about the same document is often a valid answer too.

Opt-in research extensions (all off by default, so the default reproduces the recipe):

* ``hard_negative_weight`` - importance weight ``w`` on each hard negative, implemented as a
  ``log w`` logit shift (``w = 1`` is plain inclusion);
* ``false_negative_threshold`` - drop hard negatives that the model already places almost on top
  of the gold action (likely paraphrases of it rather than wrong answers);
* ``matryoshka_dims`` - also train each prefix of the embedding, so a 128-d slice works on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from systemone.config import LossConfig

NEG_INF = float("-inf")


@dataclass
class LossOutput:
    loss: Tensor
    forward: Tensor
    backward: Tensor
    accuracy: Tensor
    """Fraction of states whose top-scoring candidate is their own action (no grad)."""
    dropped_negatives: Tensor
    """Hard negatives removed by the false-negative filter (no grad)."""


def same_document_mask(doc: Tensor) -> Tensor:
    """``[B, B]`` bool, True where two *different* rows share a document id."""
    same = doc[None, :] == doc[:, None]
    same.fill_diagonal_(False)
    return same


def contrastive_loss(
    proj_states: Tensor,
    proj_actions: Tensor,
    scale: Tensor,
    cfg: LossConfig | None = None,
    doc: Tensor | None = None,
    proj_negatives: Tensor | None = None,
    neg_count: Tensor | None = None,
) -> LossOutput:
    """InfoNCE over a batch, optionally with per-row hard negatives.

    Args:
        proj_states: ``[B, D]`` raw state projections (normalized inside).
        proj_actions: ``[B, D]`` raw action projections; row ``i`` is the gold action of state ``i``.
        scale: scalar inverse temperature.
        cfg: loss options; defaults reproduce the CLM recipe.
        doc: ``[B]`` int document ids for same-document masking.
        proj_negatives: ``[B, K, D]`` raw projections of mined hard negatives for each state.
        neg_count: ``[B]`` number of valid negatives per row (the rest are padding).
    """
    cfg = cfg or LossConfig()
    dims = sorted(set(cfg.matryoshka_dims) | {proj_states.shape[-1]})
    outs = [
        _infonce(
            proj_states[..., :d],
            proj_actions[..., :d],
            scale,
            cfg,
            doc,
            None if proj_negatives is None else proj_negatives[..., :d],
            neg_count,
        )
        for d in dims
    ]
    if len(outs) == 1:
        return outs[0]
    mean = lambda xs: torch.stack(xs).mean()  # noqa: E731
    return LossOutput(
        loss=mean([o.loss for o in outs]),
        forward=mean([o.forward for o in outs]),
        backward=mean([o.backward for o in outs]),
        accuracy=outs[-1].accuracy,  # full-dimension accuracy
        dropped_negatives=outs[-1].dropped_negatives,
    )


def _infonce(
    ps: Tensor,
    pa: Tensor,
    scale: Tensor,
    cfg: LossConfig,
    doc: Tensor | None,
    pn: Tensor | None,
    neg_count: Tensor | None,
) -> LossOutput:
    zs, za = F.normalize(ps.float(), dim=-1), F.normalize(pa.float(), dim=-1)
    b = zs.shape[0]
    logits = scale * zs @ za.t()
    if cfg.mask_same_document and doc is not None:
        logits = logits.masked_fill(same_document_mask(doc), NEG_INF)
    target = torch.arange(b, device=zs.device)

    forward_logits = logits
    dropped = torch.zeros((), device=zs.device)
    if pn is not None:
        k = pn.shape[1]
        zn = F.normalize(pn.float(), dim=-1)
        cos_neg = torch.einsum("bd,bkd->bk", zs, zn)
        valid = torch.ones(b, k, dtype=torch.bool, device=zs.device)
        if neg_count is not None:
            valid &= torch.arange(k, device=zs.device)[None, :] < neg_count[:, None]
        if cfg.false_negative_threshold is not None:
            with torch.no_grad():
                near_gold = torch.einsum("bd,bkd->bk", za, zn) > cfg.false_negative_threshold
                dropped = (near_gold & valid).sum().float()
            valid &= ~near_gold
        neg_logits = scale * cos_neg
        if cfg.hard_negative_weight != 1.0:
            neg_logits = neg_logits + math.log(cfg.hard_negative_weight)
        neg_logits = neg_logits.masked_fill(~valid, NEG_INF)
        forward_logits = torch.cat([logits, neg_logits], dim=1)

    forward = F.cross_entropy(forward_logits, target)
    backward = F.cross_entropy(logits.t(), target)
    if cfg.direction == "both":
        loss = (forward + backward) / 2
    elif cfg.direction == "state_to_action":
        loss = forward
    else:
        loss = backward
    with torch.no_grad():
        accuracy = (forward_logits.argmax(dim=1) == target).float().mean()
    return LossOutput(loss, forward.detach(), backward.detach(), accuracy, dropped)

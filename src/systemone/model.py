"""Projection heads and the dual encoder that scores (state, action) pairs.

Parameter names (``inp``, ``hidden.i``, ``norms.i``, ``out``) match the published CLM checkpoints,
so the authors' heads load into this code unchanged and every head is scored by one code path.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from systemone.config import HeadConfig, ModelConfig, from_dict

_ACTIVATIONS: dict[str, type[nn.Module]] = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class ProjectionHead(nn.Module):
    """MLP ``in_dim -> width -> ... -> out_dim``.

    Each hidden block is ``Linear -> [LayerNorm] -> act`` and, with ``residual=True``, is added
    back to its input. The output is *not* normalized here; :class:`DualEncoder` normalizes so
    that losses can also use prefixes of the raw projection (Matryoshka training).
    """

    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.in_dim, cfg.width)
        self.hidden = nn.ModuleList([nn.Linear(cfg.width, cfg.width) for _ in range(cfg.depth - 2)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width) for _ in self.hidden] if cfg.layernorm else [])
        self.out = nn.Linear(cfg.width, cfg.out_dim)
        self.act = _ACTIVATIONS[cfg.activation]()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.inp(x)))
        for i, layer in enumerate(self.hidden):
            h = layer(x)
            if self.cfg.layernorm:
                h = self.norms[i](h)
            h = self.drop(self.act(h))
            x = x + h if self.cfg.residual else h
        out: Tensor = self.out(x)
        return out


class DualEncoder(nn.Module):
    """Two projection heads and a learned inverse temperature.

    ``score = scale * cos(state_head(s), action_head(a))`` with ``scale = exp(logit_scale)``,
    capped at ``max_scale``.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.state_head = ProjectionHead(self.cfg.head)
        self.action_head = self.state_head if self.cfg.shared_heads else ProjectionHead(self.cfg.head)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / self.cfg.init_temperature)),
            requires_grad=self.cfg.learn_temperature,
        )

    # -- encoding ---------------------------------------------------------------------------
    def project_state(self, x: Tensor) -> Tensor:
        """Raw (un-normalized) state projection."""
        out: Tensor = self.state_head(x)
        return out

    def project_action(self, x: Tensor) -> Tensor:
        out: Tensor = self.action_head(x)
        return out

    def encode_state(self, x: Tensor) -> Tensor:
        return F.normalize(self.project_state(x), dim=-1)

    def encode_action(self, x: Tensor) -> Tensor:
        return F.normalize(self.project_action(x), dim=-1)

    def scale(self) -> Tensor:
        return self.logit_scale.exp().clamp(max=self.cfg.max_scale)

    # -- scoring ----------------------------------------------------------------------------
    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """All-pairs logits ``[n_states, n_actions]``."""
        return self.scale() * self.encode_state(states) @ self.encode_action(actions).t()

    @torch.no_grad()
    def option_probs(self, state: Tensor, options: Tensor, mask: Tensor | None = None) -> Tensor:
        """Probability of each option given its state.

        Args:
            state: ``[D]`` or ``[B, D]`` frozen-LLM vectors.
            options: ``[K, D]`` or ``[B, K, D]`` candidate action vectors.
            mask: optional ``[B, K]`` (or ``[K]``) bool, True for real options.

        Returns:
            ``[K]`` or ``[B, K]`` probabilities that sum to one over valid options.
        """
        single = state.dim() == 1
        if single:
            state, options = state[None], options[None]
            mask = mask[None] if mask is not None else None
        zs = self.encode_state(state)
        b, k, d = options.shape
        za = self.encode_action(options.reshape(b * k, d)).view(b, k, -1)
        logits = self.scale() * torch.einsum("bd,bkd->bk", zs, za)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = logits.softmax(-1)
        return probs[0] if single else probs

    # -- persistence ------------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def checkpoint(self, **extra: Any) -> dict[str, Any]:
        """A dict in the CLM checkpoint format, plus our full model config."""
        return {
            "state_head": self.state_head.state_dict(),
            "action_head": self.action_head.state_dict(),
            "logit_scale": self.logit_scale.detach().cpu(),
            "hidden_size": self.cfg.head.in_dim,
            "projection_dim": self.cfg.head.out_dim,
            "model_config": asdict(self.cfg),
            **extra,
        }

    def save(self, path: str | Path, **extra: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(**extra), path)

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any]) -> DualEncoder:
        """Rebuild from our checkpoints or the published CLM ones (config inferred from shapes)."""
        if "model_config" in ckpt:
            cfg = from_dict(ModelConfig, ckpt["model_config"])
        else:
            cfg = ModelConfig(head=infer_head_config(ckpt["state_head"]))
        model = cls(cfg)
        model.state_head.load_state_dict(ckpt["state_head"])
        if not cfg.shared_heads:
            model.action_head.load_state_dict(ckpt["action_head"])
        model.logit_scale.data.copy_(torch.as_tensor(ckpt["logit_scale"]).float())
        return model

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> DualEncoder:
        """Load a checkpoint with ``weights_only=True``: no pickled code is ever executed."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        return cls.from_checkpoint(ckpt)


def infer_head_config(state_dict: dict[str, Tensor]) -> HeadConfig:
    """Recover a :class:`HeadConfig` from a head's parameter shapes."""
    width, in_dim = state_dict["inp.weight"].shape
    out_dim = state_dict["out.weight"].shape[0]
    n_hidden = len({k.split(".")[1] for k in state_dict if k.startswith("hidden.")})
    layernorm = any(k.startswith("norms.") for k in state_dict)
    return HeadConfig(in_dim=in_dim, width=width, out_dim=out_dim, depth=n_hidden + 2, layernorm=layernorm)

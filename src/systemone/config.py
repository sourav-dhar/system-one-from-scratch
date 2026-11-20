"""Typed experiment configuration.

Every run is described by one :class:`ExperimentConfig`. Configs are plain dataclasses so they
are easy to construct in code and in tests, and they load from YAML files that may ``extends:``
a parent file. Command-line overrides use dotted paths, e.g. ``train.batch_size=4096``.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml

Stage = Literal["pre", "mid", "post"]
Direction = Literal["both", "state_to_action", "action_to_state"]


@dataclass
class HeadConfig:
    """Shape of one projection head: ``in_dim -> width -> ... -> out_dim``.

    ``depth`` counts linear layers. The CLM recipe is depth 3: an input layer, one hidden layer
    with LayerNorm, and an output layer, with GELU between them (about 9.44M parameters).
    """

    in_dim: int = 4096
    width: int = 1536
    out_dim: int = 512
    depth: int = 3
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    layernorm: bool = True
    residual: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.depth < 2:
            raise ValueError(f"depth must be >= 2 (input and output layers), got {self.depth}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class ModelConfig:
    head: HeadConfig = field(default_factory=HeadConfig)
    shared_heads: bool = False
    """Tie the state and action heads (a siamese encoder). The recipe uses two separate heads."""
    init_temperature: float = 0.07
    learn_temperature: bool = True
    max_scale: float = 100.0
    """Cap on ``exp(logit_scale)``, the inverse temperature. CLIP and CLM both cap at 100."""


@dataclass
class LossConfig:
    direction: Direction = "both"
    mask_same_document: bool = True
    """Pairs from the same source document are not treated as negatives of each other."""
    hard_negative_weight: float = 1.0
    """Weight of each mined hard negative in the softmax denominator (1.0 = plain inclusion)."""
    false_negative_threshold: float | None = None
    """Drop hard negatives whose projected cosine to the gold action exceeds this value."""
    matryoshka_dims: list[int] = field(default_factory=list)
    """If set, also train every prefix of the embedding (e.g. [64, 128, 256]) to be usable alone."""


@dataclass
class DataConfig:
    root: str = "data"
    """Local directory laid out like the Hugging Face datasets: ``qa/shard_XX/chunk_NNNN_*``."""
    train_chunks: list[str] = field(default_factory=list)
    """Chunk ids such as ``shard_01/chunk_0000``. Ignored when ``synthetic`` is set."""
    hard_negatives: bool = False
    max_negatives: int | None = None
    """Use at most this many mined negatives per pair (the k ablation); ``None`` keeps all (up to 10)."""
    rows_per_chunk: int | None = None
    """Use only the first N rows of each chunk (laptop-sized subsets)."""
    eval_general: list[str] = field(default_factory=lambda: ["shard_05/chunk_0083", "shard_05/chunk_0084"])
    eval_hard: list[str] = field(default_factory=lambda: ["shard_00/chunk_0028", "shard_00/chunk_0029"])
    eval_rows_per_chunk: int | None = None
    synthetic: SyntheticConfig | None = None


@dataclass
class SyntheticConfig:
    """A learnable toy task with the same structure as the real data. Used by tests and demos."""

    n_train: int = 20_000
    n_eval: int = 4_000
    dim: int = 256
    latent_dim: int = 32
    noise: float = 2.0
    num_negatives: int = 10
    negative_closeness: float = 0.9
    docs_per_pair: float = 0.25
    seed: int = 0


@dataclass
class OptimConfig:
    lr: float | None = None
    """``None`` uses the recipe: pre-training scales with sqrt(batch) and 1/sqrt(width); later stages 1e-3."""
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    schedule: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    stage: Stage = "pre"
    batch_size: int = 16384
    max_steps: int = 1000
    eval_every: int = 250
    full_eval_every: int | None = None
    """Run the (slower) hard-negative test every N steps; ``None`` means at every eval."""
    patience: int | None = None
    """Stop after this many evals without improvement of ``select_metric``."""
    select_metric: str | None = None
    """Metric that picks ``best.pt``. ``None`` = ``general/loss`` for pre, ``hard/loss`` otherwise."""
    seed: int = 1234
    precision: Literal["fp32", "bf16"] = "bf16"
    init_checkpoint: str | None = None
    out_dir: str = "runs/default"
    log_every: int = 50


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def resolved_lr(self) -> float:
        """The learning rate actually used, applying the recipe's rule when ``optim.lr`` is unset."""
        if self.optim.lr is not None:
            return self.optim.lr
        if self.train.stage == "pre":
            return 2e-3 * math.sqrt(1024 / self.model.head.width) * math.sqrt(self.train.batch_size / 1024)
        return 1e-3

    def select_metric(self) -> str:
        if self.train.select_metric:
            return self.train.select_metric
        return "general/loss" if self.train.stage == "pre" else "hard/loss"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------
# dict / YAML <-> dataclass
# --------------------------------------------------------------------------------------------

T = TypeVar("T")


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build dataclass ``cls`` from a nested dict, rejecting unknown keys (typos fail loudly)."""
    data = dict(data or {})
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v) for k, v in data.items()}
    return cls(**kwargs)


def _coerce(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(args[0], value) if len(args) == 1 else value
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return from_dict(tp, value)  # type: ignore[arg-type]
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if tp is float and isinstance(value, int | str) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, following ``extends:`` (relative to the file) recursively."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _deep_merge(load_yaml_dict(path.parent / parent), raw)
    return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (so ``1e-3``, ``true``, ``[1,2]`` work)."""
    data = _deep_merge({}, data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = data
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data = load_yaml_dict(path) if path else {}
    data = apply_overrides(data, overrides or [])
    return from_dict(ExperimentConfig, data)


def dump_config(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(_plain(cfg.to_dict()), sort_keys=False)


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    return x

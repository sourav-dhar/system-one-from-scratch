import json

import pytest

from systemone.cli import _load_data
from systemone.config import ExperimentConfig, SyntheticConfig, load_config
from systemone.train import Trainer, lr_lambda


def small_cfg(out: str, *extra: str) -> ExperimentConfig:
    cfg = load_config(
        None,
        [
            f"train.out_dir={out}",
            "model.head.in_dim=64",
            "model.head.width=64",
            "model.head.out_dim=32",
            "train.batch_size=256",
            "train.max_steps=60",
            "train.eval_every=30",
            "train.precision=fp32",
            "train.stage=mid",
            "data.hard_negatives=true",
            "optim.lr=0.003",
            *extra,
        ],
    )
    cfg.data.synthetic = SyntheticConfig(n_train=3000, n_eval=2048, dim=64, latent_dim=16, noise=1.0)
    return cfg


def test_training_beats_random_heads(tmp_path) -> None:
    cfg = small_cfg(str(tmp_path))
    train, suite = _load_data(cfg)
    result = Trainer(cfg, train, suite, device="cpu", log=lambda _: None).fit()
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    first = lines[0]
    assert lines[-1]["finished"] is True
    assert result.final["hard/top1"] > first["hard/top1"] + 0.1
    assert result.final["general/top1"] > first["general/top1"] + 0.2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "config.yaml").exists()


def test_resume_continues_from_last_checkpoint(tmp_path) -> None:
    cfg = small_cfg(str(tmp_path), "train.max_steps=30")
    train, suite = _load_data(cfg)
    Trainer(cfg, train, suite, device="cpu", log=lambda _: None).fit()
    logs: list[str] = []
    cfg2 = small_cfg(str(tmp_path), "train.max_steps=60")
    r = Trainer(cfg2, train, suite, device="cpu", log=logs.append).fit()
    assert any("resumed from step 30" in m for m in logs)
    assert r.steps == 60


def test_init_checkpoint_and_frozen_temperature(tmp_path) -> None:
    cfg = small_cfg(str(tmp_path / "a"), "train.max_steps=10", "train.eval_every=10")
    train, suite = _load_data(cfg)
    Trainer(cfg, train, suite, device="cpu", log=lambda _: None).fit()
    cfg2 = small_cfg(
        str(tmp_path / "b"),
        f"train.init_checkpoint={tmp_path / 'a' / 'best.pt'}",
        "model.learn_temperature=false",
        "train.max_steps=10",
        "train.eval_every=10",
    )
    t = Trainer(cfg2, train, suite, device="cpu", log=lambda _: None)
    before = t.model.scale().item()
    t.fit()
    assert t.model.scale().item() == pytest.approx(before)


def test_early_stopping(tmp_path) -> None:
    cfg = small_cfg(
        str(tmp_path), "optim.lr=0.0", "train.max_steps=200", "train.eval_every=10", "train.patience=2"
    )
    train, suite = _load_data(cfg)
    r = Trainer(cfg, train, suite, device="cpu", log=lambda _: None).fit()
    assert r.steps == 20  # the step-0 eval sets the best; two stale evals later it stops


def test_dimension_mismatch_is_reported(tmp_path) -> None:
    cfg = small_cfg(str(tmp_path), "model.head.in_dim=32")
    train, suite = _load_data(cfg)
    with pytest.raises(ValueError, match="64-d"):
        Trainer(cfg, train, suite, device="cpu")


@pytest.mark.parametrize(
    ("schedule", "warmup", "expect"),
    [("constant", 0, [1, 1, 1]), ("constant", 10, [0.1, 1, 1]), ("cosine", 0, [1, 0.5, 0])],
)
def test_lr_schedules(schedule: str, warmup: int, expect: list[float]) -> None:
    cfg = load_config(
        None, [f"optim.schedule={schedule}", f"optim.warmup_steps={warmup}", "train.max_steps=100"]
    )
    f = lr_lambda(cfg)
    got = [f(0), f(50), f(100)]
    assert got == pytest.approx(expect, abs=1e-6)


def test_max_negatives_is_applied_when_loading(tmp_path) -> None:
    cfg = small_cfg(str(tmp_path), "data.max_negatives=2")
    train, _ = _load_data(cfg)
    assert train.negatives is not None
    assert train.neg_count is not None
    assert train.negatives.shape[1] == 2
    assert train.neg_count.max() == 2

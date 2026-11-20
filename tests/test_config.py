import math

import pytest

from systemone.config import (
    ExperimentConfig,
    HeadConfig,
    SyntheticConfig,
    apply_overrides,
    dump_config,
    from_dict,
    load_config,
)


def test_defaults_reproduce_the_recipe_lr() -> None:
    cfg = ExperimentConfig()
    # 2e-3 * sqrt(1024/1536) * sqrt(16384/1024): the lr stored in the authors' checkpoint
    assert cfg.resolved_lr() == pytest.approx(0.0065319726474218085)
    cfg.train.stage = "mid"
    assert cfg.resolved_lr() == 1e-3
    cfg.optim.lr = 3e-4
    assert cfg.resolved_lr() == 3e-4


def test_select_metric_by_stage() -> None:
    cfg = ExperimentConfig()
    assert cfg.select_metric() == "general/loss"
    cfg.train.stage = "mid"
    assert cfg.select_metric() == "hard/loss"


def test_unknown_keys_fail_loudly() -> None:
    with pytest.raises(KeyError, match="widht"):
        from_dict(HeadConfig, {"widht": 3})


def test_overrides_parse_yaml_values() -> None:
    d = apply_overrides({}, ["train.batch_size=4096", "optim.lr=1e-3", "loss.matryoshka_dims=[64, 128]"])
    cfg = from_dict(ExperimentConfig, d)
    assert cfg.train.batch_size == 4096
    assert math.isclose(cfg.optim.lr or 0, 1e-3)
    assert cfg.loss.matryoshka_dims == [64, 128]
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["oops"])


def test_extends_and_roundtrip(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        "train:\n  batch_size: 1024\n  max_steps: 10\nmodel:\n  head:\n    width: 64\n"
    )
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nname: child\ntrain:\n  max_steps: 20\n")
    cfg = load_config(tmp_path / "child.yaml", ["model.head.depth=4"])
    assert (cfg.name, cfg.train.batch_size, cfg.train.max_steps) == ("child", 1024, 20)
    assert (cfg.model.head.width, cfg.model.head.depth) == (64, 4)
    (tmp_path / "dump.yaml").write_text(dump_config(cfg))
    assert load_config(tmp_path / "dump.yaml") == cfg


def test_nested_optional_dataclass() -> None:
    cfg = load_config(None, ["data.synthetic={n_train: 100}"])
    assert cfg.data.synthetic == SyntheticConfig(n_train=100)

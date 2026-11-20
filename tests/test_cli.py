import json
from pathlib import Path

import numpy as np
import pytest
import torch

from systemone.cli import build_parser, main
from systemone.config import HeadConfig, ModelConfig
from systemone.data.pairs import doc_ids_from_strings
from systemone.inference import Decision, SystemOne
from systemone.model import DualEncoder

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_every_shipped_config_loads() -> None:
    from systemone.config import load_config

    files = sorted(CONFIGS.glob("*.yaml"))
    assert {f.stem for f in files} >= {"base", "pre", "mid", "smoke"}
    for f in files:
        load_config(f)


def test_train_and_info_from_smoke_config(tmp_path, capsys) -> None:
    out = tmp_path / "run"
    code = main(
        [
            "train",
            "-c",
            str(CONFIGS / "smoke.yaml"),
            "-o",
            f"train.out_dir={out}",
            "-o",
            "train.max_steps=20",
            "-o",
            "train.eval_every=20",
            "-o",
            "data.synthetic={n_train: 3000, n_eval: 2048}",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert (out / "best.pt").exists()
    capsys.readouterr()
    main(["info", str(out / "last.pt")])
    info = json.loads(capsys.readouterr().out)
    assert info["head"]["in_dim"] == 256
    assert info["step"] == 20


def test_eval_command_on_a_local_mirror(tmp_path) -> None:
    rng = np.random.default_rng(0)
    for chunk, negs in [("shard_05/chunk_0083", False), ("shard_00/chunk_0028", True)]:
        (tmp_path / "qa" / chunk).parent.mkdir(parents=True, exist_ok=True)
        x = rng.standard_normal((1024, 16)).astype(np.float16)
        np.save(tmp_path / "qa" / f"{chunk}_q.npy", x)
        np.save(tmp_path / "qa" / f"{chunk}_a.npy", x)
        np.save(tmp_path / "qa" / f"{chunk}_doc.npy", doc_ids_from_strings([str(i) for i in range(1024)]))
        if negs:
            (tmp_path / "neg" / chunk).parent.mkdir(parents=True, exist_ok=True)
            np.save(
                tmp_path / "neg" / f"{chunk}_n.npy", rng.standard_normal((1024, 10 * 16)).astype(np.float16)
            )
            np.save(tmp_path / "neg" / f"{chunk}_ncnt.npy", np.full(1024, 10, dtype=np.int32))
    torch.manual_seed(0)
    DualEncoder(ModelConfig(head=HeadConfig(in_dim=16, width=32, out_dim=8), shared_heads=True)).save(
        tmp_path / "h.pt"
    )
    out = tmp_path / "res.json"
    args = ["eval", str(tmp_path / "h.pt"), "--data", str(tmp_path), "--device", "cpu", "--out", str(out)]
    main([*args, "--general", "shard_05/chunk_0083", "--hard", "shard_00/chunk_0028", "--dims", "4"])
    res = json.loads(out.read_text())[str(tmp_path / "h.pt")]
    assert res["general/top1"] == 1.0
    assert res["hard/top1"] == 1.0
    assert "hard@4/top1" in res


class FakeEncoder:
    """Maps each text to a fixed random vector so tests need no LLM."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, texts: list[str]) -> torch.Tensor:
        vecs = [torch.Generator().manual_seed(sum(map(ord, t))) for t in texts]
        return torch.stack([torch.randn(self.dim, generator=g) for g in vecs])


def test_system_one_decides_over_text_options() -> None:
    torch.manual_seed(0)
    heads = DualEncoder(ModelConfig(head=HeadConfig(in_dim=16, width=32, out_dim=8), shared_heads=True))
    s1 = SystemOne(heads, FakeEncoder(16))  # type: ignore[arg-type]
    d = s1.decide("Mars", ["Venus", "Mars", "Jupiter"])  # identical text -> identical vector -> top score
    assert isinstance(d, Decision)
    assert d.best == "Mars"
    assert d.index == 1
    assert sum(d.probs) == pytest.approx(1.0, abs=1e-5)
    assert "Mars" in repr(d)


def test_system_one_without_encoder_needs_vectors() -> None:
    heads = DualEncoder(ModelConfig(head=HeadConfig(in_dim=16, width=32, out_dim=8)))
    s1 = SystemOne(heads)
    assert s1.probs(torch.randn(16), torch.randn(3, 16)).shape == (3,)
    with pytest.raises(RuntimeError, match="encoder"):
        s1.decide("question", ["a", "b"])

import pytest
import torch

from systemone.config import HeadConfig, ModelConfig
from systemone.model import DualEncoder, ProjectionHead, infer_head_config

SMALL = HeadConfig(in_dim=32, width=16, out_dim=8)


def test_default_head_matches_published_size() -> None:
    # 4096 -> 1536 -> 1536 (LayerNorm) -> 512: the "9.44M per head" in the CLM write-up
    assert sum(p.numel() for p in ProjectionHead(HeadConfig()).parameters()) == 9_443_840


def test_parameter_names_match_clm_checkpoints() -> None:
    names = set(ProjectionHead(HeadConfig()).state_dict())
    assert names == {
        "inp.weight",
        "inp.bias",
        "hidden.0.weight",
        "hidden.0.bias",
        "norms.0.weight",
        "norms.0.bias",
        "out.weight",
        "out.bias",
    }


@pytest.mark.parametrize("depth", [2, 3, 5])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("layernorm", [False, True])
def test_head_shapes(depth: int, residual: bool, layernorm: bool) -> None:
    head = ProjectionHead(
        HeadConfig(in_dim=32, width=16, out_dim=8, depth=depth, residual=residual, layernorm=layernorm)
    )
    assert head(torch.randn(4, 32)).shape == (4, 8)
    assert len(head.hidden) == depth - 2


def test_invalid_head_config() -> None:
    with pytest.raises(ValueError, match="depth"):
        HeadConfig(depth=1)
    with pytest.raises(ValueError, match="dropout"):
        HeadConfig(dropout=1.0)


def test_scale_starts_at_inverse_temperature_and_is_capped() -> None:
    m = DualEncoder(ModelConfig(head=SMALL, init_temperature=0.07, max_scale=100.0))
    assert m.scale().item() == pytest.approx(1 / 0.07, rel=1e-5)
    m.logit_scale.data.fill_(10.0)
    assert m.scale().item() == pytest.approx(100.0)


def test_fixed_temperature_is_not_trainable() -> None:
    m = DualEncoder(ModelConfig(head=SMALL, learn_temperature=False))
    assert not m.logit_scale.requires_grad


def test_shared_heads_are_one_module() -> None:
    m = DualEncoder(ModelConfig(head=SMALL, shared_heads=True))
    assert m.state_head is m.action_head
    assert m.num_parameters() == sum(p.numel() for p in m.state_head.parameters()) + 1


def test_forward_is_scaled_cosine() -> None:
    m = DualEncoder(ModelConfig(head=SMALL))
    s, a = torch.randn(3, 32), torch.randn(5, 32)
    logits = m(s, a)
    assert logits.shape == (3, 5)
    assert logits.abs().max() <= m.scale() + 1e-4


def test_option_probs_single_and_batched_with_mask() -> None:
    torch.manual_seed(0)
    m = DualEncoder(ModelConfig(head=SMALL))
    p = m.option_probs(torch.randn(32), torch.randn(4, 32))
    assert p.shape == (4,)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)

    mask = torch.tensor([[True, True, False], [True, False, False]])
    pb = m.option_probs(torch.randn(2, 32), torch.randn(2, 3, 32), mask)
    assert pb.shape == (2, 3)
    assert torch.allclose(pb.sum(-1), torch.ones(2))
    assert pb[0, 2] == 0
    assert pb[1, 1] == 0
    assert pb[1, 0].item() == pytest.approx(1.0)


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = ModelConfig(head=HeadConfig(in_dim=32, width=16, out_dim=8, depth=4, residual=True))
    m = DualEncoder(cfg)
    m.logit_scale.data.fill_(3.0)
    m.save(tmp_path / "m.pt", step=7)
    m2 = DualEncoder.load(tmp_path / "m.pt")
    assert m2.cfg == cfg
    x, y = torch.randn(2, 32), torch.randn(3, 32)
    assert torch.allclose(m(x, y), m2(x, y))


def test_loads_clm_format_without_our_config() -> None:
    """Published checkpoints have only state_head / action_head / logit_scale: shapes decide the config."""
    m = DualEncoder(ModelConfig(head=HeadConfig(in_dim=32, width=16, out_dim=8, depth=3)))
    ck = {k: v for k, v in m.checkpoint().items() if k != "model_config"}
    m2 = DualEncoder.from_checkpoint(ck)
    assert m2.cfg.head == HeadConfig(in_dim=32, width=16, out_dim=8, depth=3, layernorm=True)
    x = torch.randn(2, 32)
    assert torch.allclose(m.encode_state(x), m2.encode_state(x))


def test_infer_head_config_without_layernorm() -> None:
    head = ProjectionHead(HeadConfig(in_dim=32, width=16, out_dim=8, depth=4, layernorm=False))
    cfg = infer_head_config(head.state_dict())
    assert (cfg.depth, cfg.layernorm, cfg.width) == (4, False, 16)

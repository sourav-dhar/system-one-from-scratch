import math

import pytest
import torch
import torch.nn.functional as F

from systemone.config import LossConfig
from systemone.losses import contrastive_loss, same_document_mask


def naive_loss(ps, pa, scale, doc=None, pn=None, count=None):
    """Row-by-row reference: an explicit softmax over each state's candidate list."""
    zs, za = F.normalize(ps, dim=-1), F.normalize(pa, dim=-1)
    b = len(zs)
    fwd, bwd = [], []
    for i in range(b):
        cands = [j for j in range(b) if j == i or doc is None or doc[j] != doc[i]]
        logits = [scale * zs[i] @ za[j] for j in cands]
        if pn is not None:
            zn = F.normalize(pn[i], dim=-1)
            logits += [scale * zs[i] @ zn[k] for k in range(int(count[i]))]
        fwd.append(-torch.stack(logits).log_softmax(0)[cands.index(i)])
        cands_b = [j for j in range(b) if j == i or doc is None or doc[j] != doc[i]]
        lb = torch.stack([scale * zs[j] @ za[i] for j in cands_b])
        bwd.append(-lb.log_softmax(0)[cands_b.index(i)])
    return (torch.stack(fwd).mean() + torch.stack(bwd).mean()) / 2


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "ps": torch.randn(12, 8, generator=g),
        "pa": torch.randn(12, 8, generator=g),
        "pn": torch.randn(12, 4, 8, generator=g),
        "count": torch.tensor([4, 0, 2, 3, 1, 4, 4, 0, 2, 1, 3, 4]),
        "doc": torch.tensor([0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8]),
    }


def test_matches_naive_without_negatives(batch) -> None:
    s = torch.tensor(20.0)
    out = contrastive_loss(batch["ps"], batch["pa"], s, doc=batch["doc"])
    ref = naive_loss(batch["ps"], batch["pa"], s, batch["doc"])
    assert out.loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_matches_naive_with_padded_hard_negatives(batch) -> None:
    s = torch.tensor(20.0)
    out = contrastive_loss(
        batch["ps"], batch["pa"], s, doc=batch["doc"], proj_negatives=batch["pn"], neg_count=batch["count"]
    )
    ref = naive_loss(batch["ps"], batch["pa"], s, batch["doc"], batch["pn"], batch["count"])
    assert out.loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_padding_negatives_do_not_matter(batch) -> None:
    s = torch.tensor(20.0)
    pn2 = batch["pn"].clone()
    for i, c in enumerate(batch["count"].tolist()):
        pn2[i, c:] = 1e3 * torch.randn_like(pn2[i, c:])  # garbage in the padding slots
    a = contrastive_loss(batch["ps"], batch["pa"], s, proj_negatives=batch["pn"], neg_count=batch["count"])
    b = contrastive_loss(batch["ps"], batch["pa"], s, proj_negatives=pn2, neg_count=batch["count"])
    assert a.loss.item() == pytest.approx(b.loss.item(), rel=1e-6)


def test_same_document_mask_and_gradients(batch) -> None:
    mask = same_document_mask(batch["doc"])
    assert not mask.diagonal().any()
    assert mask[1, 2]
    assert mask[2, 1]
    assert not mask[0, 1]
    pa = batch["pa"].clone().requires_grad_(True)
    out = contrastive_loss(batch["ps"], pa, torch.tensor(10.0), doc=batch["doc"])
    out.loss.backward()
    assert torch.isfinite(pa.grad).all()


def test_masking_changes_loss_when_documents_repeat(batch) -> None:
    s = torch.tensor(20.0)
    on = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(mask_same_document=True), doc=batch["doc"])
    off = contrastive_loss(
        batch["ps"], batch["pa"], s, LossConfig(mask_same_document=False), doc=batch["doc"]
    )
    assert on.loss.item() < off.loss.item()  # masking only removes competitors


def test_directions(batch) -> None:
    s = torch.tensor(20.0)
    both = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(direction="both"))
    fwd = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(direction="state_to_action"))
    bwd = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(direction="action_to_state"))
    assert both.loss.item() == pytest.approx((fwd.loss.item() + bwd.loss.item()) / 2, rel=1e-6)
    assert fwd.loss.item() == pytest.approx(both.forward.item(), rel=1e-6)


def test_hard_negative_weight_is_a_log_shift(batch) -> None:
    s = torch.tensor(20.0)
    kw = {"proj_negatives": batch["pn"], "neg_count": batch["count"]}
    base = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(hard_negative_weight=1.0), **kw)
    up = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(hard_negative_weight=4.0), **kw)
    none = contrastive_loss(batch["ps"], batch["pa"], s)
    assert none.forward < base.forward < up.forward


def test_false_negative_filter_drops_near_duplicates(batch) -> None:
    ps, pn = batch["ps"].clone(), batch["pn"].clone()
    ps[0] = batch["pa"][0]  # state 0 points straight at its gold answer...
    pn[0, 0] = batch["pa"][0] * 3.0  # ...and one "negative" is a rescaled copy of that answer
    cfg = LossConfig(false_negative_threshold=0.99)
    kw = {"proj_negatives": pn, "neg_count": batch["count"]}
    out = contrastive_loss(ps, batch["pa"], torch.tensor(20.0), cfg, **kw)
    ref = contrastive_loss(ps, batch["pa"], torch.tensor(20.0), **kw)
    assert out.dropped_negatives.item() >= 1
    # unfiltered, row 0 ties with the duplicate and pays ~log(2); filtered, that penalty is gone
    assert ref.forward.item() - out.forward.item() == pytest.approx(math.log(2) / 12, rel=0.05)


def test_matryoshka_averages_prefix_losses(batch) -> None:
    s = torch.tensor(20.0)
    m = contrastive_loss(batch["ps"], batch["pa"], s, LossConfig(matryoshka_dims=[4]))
    full = contrastive_loss(batch["ps"], batch["pa"], s)
    half = contrastive_loss(batch["ps"][:, :4], batch["pa"][:, :4], s)
    assert m.loss.item() == pytest.approx((full.loss.item() + half.loss.item()) / 2, rel=1e-6)


def test_perfect_alignment_gives_low_loss_and_full_accuracy() -> None:
    x = torch.eye(16)
    out = contrastive_loss(x, x, torch.tensor(50.0))
    assert out.accuracy.item() == 1.0
    assert out.loss.item() < 1e-3
    assert math.isfinite(out.loss.item())

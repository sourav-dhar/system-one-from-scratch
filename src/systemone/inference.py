"""Text in, probabilities out.

    >>> s1 = SystemOne.from_pretrained("runs/mid/best.pt", encoder="Qwen/Qwen3-8B")  # doctest: +SKIP
    >>> s1.decide("Which planet is the Red Planet?", ["Venus", "Mars", "Jupiter"])  # doctest: +SKIP
    Decision(best='Mars', probs=[0.01, 0.97, 0.02])

The frozen encoder follows the recipe used to build the published vectors: last-token pooling of
the final hidden state, L2-normalized. Heads only work with the encoder they were trained on
(a Qwen3-8B head expects 4096-d Qwen3-8B vectors).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from systemone.model import DualEncoder


def configure_tokenizer(tok: Any) -> Any:
    """Set up a tokenizer for last-token pooling.

    * ``padding_side="left"``: position -1 is every sequence's last real token;
    * ``truncation_side="left"``: over-long inputs keep their **end**. The question (and the answer
      cue) sit at the end of a state text, so right truncation would silently pool a vector that
      never saw the question. This is the failure documented in CLM issues #6 and #29.
    """
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class LastTokenEncoder:
    """Frozen Hugging Face decoder used as a feature extractor (needs the ``encode`` extra)."""

    def __init__(
        self,
        name: str,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = configure_tokenizer(AutoTokenizer.from_pretrained(name))
        self.model = AutoModel.from_pretrained(name, torch_dtype=dtype, **kwargs).to(self.device).eval()
        self.max_tokens = max_tokens

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], batch_size: int = 8) -> Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                list(texts[i : i + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state[:, -1]
            out.append(F.normalize(hidden.float(), dim=-1).cpu())
        return torch.cat(out)


@dataclass
class Decision:
    best: str
    index: int
    probs: list[float]
    options: list[str]

    def __repr__(self) -> str:
        return f"Decision(best={self.best!r}, probs={[round(p, 3) for p in self.probs]})"


class SystemOne:
    def __init__(self, heads: DualEncoder, encoder: LastTokenEncoder | None = None) -> None:
        self.heads = heads.eval()
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, encoder: str | None = None, **kw: Any) -> SystemOne:
        heads = DualEncoder.load(checkpoint)
        enc = LastTokenEncoder(encoder, **kw) if encoder else None
        if enc is not None:
            heads.to(enc.device)
        return cls(heads, enc)

    def _vectors(self, texts: Sequence[str]) -> Tensor:
        if self.encoder is None:
            raise RuntimeError(
                "no text encoder loaded; pass vectors or use from_pretrained(..., encoder=...)"
            )
        device = next(self.heads.parameters()).device
        return self.encoder(texts).to(device)

    @torch.no_grad()
    def probs(self, state: str | Tensor, options: Sequence[str] | Tensor) -> Tensor:
        s = self._vectors([state])[0] if isinstance(state, str) else state
        o = options if isinstance(options, Tensor) else self._vectors(list(options))
        return self.heads.option_probs(s, o)

    def decide(self, state: str, options: Sequence[str]) -> Decision:
        p = self.probs(state, options).tolist()
        i = max(range(len(p)), key=p.__getitem__)
        return Decision(options[i], i, p, list(options))

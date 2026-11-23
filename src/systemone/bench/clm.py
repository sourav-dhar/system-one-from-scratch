"""CLM as a decision model: projection heads on a frozen last-token encoder (the CLM anchor of M0.4).

This ports the official client, ``contrastive-lm`` 0.1.0 (``clm.schema.build_pairs``, ``clm.engine``
and ``clm.heads``), called the way System1-Arena calls it (harness 28e16b1, ``s1arena/models/clm.py``,
spec ``clm:Contrastive-LM/CLM-v0.1-8B``).

**Texts.** Each question becomes one *state text* and one *candidate text* per option.
* The state text is the state rendered as prose (:func:`to_text`), a blank line, and then the question's
  instructions.
* A choice option's candidate text is its description, or its id if the description is empty.
* A score level's candidate text is the level itself.
* A boolean's two candidate texts are its descriptions when the question gives them
  (``"true: <description>"``). Otherwise they state the claim affirmed and denied:
  ``"true: Yes. This is true: <instructions>"`` and ``"false: No. This is false: <instructions>"``.

**Encoding.** The frozen encoder (Qwen3-8B, last-token pooling, L2-normalised) embeds each distinct text
once. A text longer than ``maxlen`` tokens (2,048, as released) is cut to its last ``maxlen`` tokens, the
end that holds the question, exactly the way upstream cuts it (:func:`tail_text`). An empty text is
refused.

**Scoring.** The heads project and L2-normalise. The answer is the softmax over
``scale * cos(state, option)``, with ``scale = min(exp(logit_scale), 100)``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from systemone.decisions.schema import Answer, DecisionRequest, ModelResponse, Question
from systemone.decisions.text import to_text
from systemone.model import DualEncoder

REPO = "Contrastive-LM/CLM-v0.1-8B"
ENCODER = "Qwen/Qwen3-8B"
REJECTED = "rejected: "


def state_text(state: Any, instructions: str) -> str:
    """Context first, question last: the layout the heads were trained on."""
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def candidate_texts(question: Question) -> list[str]:
    """The action head's text for each option, aligned with ``question.option_ids()``."""
    if question.type == "choice":
        assert isinstance(question.criteria, dict)
        return [to_text(v) if v not in (None, "") else k for k, v in question.criteria.items()]
    if question.type == "score":
        assert isinstance(question.criteria, list)
        return [to_text(c) for c in question.criteria]
    ins = to_text(question.instructions).strip()
    claims = {"true": f"Yes. This is true: {ins}", "false": f"No. This is false: {ins}"}
    described = question.criteria if isinstance(question.criteria, dict) else {}
    texts = []
    for k in question.option_ids():
        d = described.get(k)
        texts.append(f"{k}: {to_text(d) if d not in (None, '') else (claims[k] if ins else k)}")
    return texts


def tail_text(tok: Any, ids: list[int], maxlen: int) -> str:
    """The text of the last ``maxlen`` tokens; fewer if that text would re-tokenize into more."""
    keep = maxlen
    while True:
        text: str = tok.decode(ids[-keep:], skip_special_tokens=False)
        if len(tok.encode(text)) <= maxlen:
            return text
        keep -= 8


class CLMDecider:
    """Typed decisions from CLM heads on an encoder; ``embed`` maps texts to L2-normalised vectors."""

    def __init__(
        self,
        heads: DualEncoder,
        embed: Callable[[list[str]], Tensor],
        tokenizer: Any,
        maxlen: int = 2048,
        name: str = REPO,
    ) -> None:
        self.heads = heads.eval()
        self.embed = embed
        self.tok = tokenizer
        self.maxlen = maxlen
        self.name = name

    @property
    def model_id(self) -> str:
        return self.name if self.maxlen == 2048 else f"{self.name}@maxlen{self.maxlen}-cut"

    @classmethod
    def from_pretrained(cls, repo: str = REPO, encoder: str = ENCODER, **kwargs: Any) -> CLMDecider:
        from huggingface_hub import hf_hub_download

        from systemone.inference import LastTokenEncoder

        with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
            checkpoint = hf_hub_download(repo, json.load(f)["checkpoints"][0])
        maxlen = int(kwargs.pop("maxlen", 2048))
        enc = LastTokenEncoder(encoder, max_tokens=maxlen, **kwargs)
        heads = DualEncoder.load(checkpoint).to(enc.device)
        return cls(heads, enc, enc.tok, maxlen, repo)

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]:
        started = time.perf_counter()
        per_question: list[tuple[int, str, list[str]]] = []  # (request, question, [state, *candidates])
        for r, request in enumerate(requests):
            for name, question in request.questions.items():
                texts = [state_text(request.state, question.instructions), *candidate_texts(question)]
                per_question.append((r, name, texts))
        distinct = list(dict.fromkeys(t for _, _, texts in per_question for t in texts))
        token_ids = self.tok(distinct, add_special_tokens=True)["input_ids"]
        cut = {
            t: tail_text(self.tok, ids, self.maxlen)
            for t, ids in zip(distinct, token_ids, strict=True)
            if len(ids) > self.maxlen
        }
        empty = {t for t, ids in zip(distinct, token_ids, strict=True) if not ids}
        wanted = [t for t in distinct if t not in empty]
        vectors = self.embed([cut.get(t, t) for t in wanted]) if wanted else torch.empty(0)
        index = {t: i for i, t in enumerate(wanted)}
        device = next(self.heads.parameters()).device
        with torch.no_grad():
            z = vectors.to(device)
            zs = self.heads.encode_state(z) if len(wanted) else z
            za = self.heads.encode_action(z) if len(wanted) else z
            scale = float(self.heads.scale())
        answers: list[dict[str, Answer]] = [{} for _ in requests]
        errors: list[list[str]] = [[] for _ in requests]
        cut_texts: list[set[str]] = [set() for _ in requests]
        for r, name, texts in per_question:
            cut_texts[r].update(t for t in texts if t in cut)
            if any(t in empty for t in texts):
                what = "the state text" if texts[0] in empty else "an option text"
                errors[r].append(f"{name}: {what} is empty")
                continue
            s = zs[index[texts[0]]]
            options = za[[index[t] for t in texts[1:]]]
            probs = F.softmax(scale * (options @ s), dim=0).tolist()
            ids = requests[r].questions[name].option_ids()
            answers[r][name] = Answer(probs=dict(zip(ids, probs, strict=True)))
        per_request = round((time.perf_counter() - started) * 1000 / max(1, len(requests)), 2)
        return [
            ModelResponse(
                answers=answers[r],
                usage={"cut_texts": len(cut_texts[r]), "batch_ms_per_request": per_request},
                error=REJECTED + "; ".join(errors[r]) if errors[r] else None,
            )
            for r in range(len(requests))
        ]

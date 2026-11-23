"""Answer-token readout: a causal LM as a decision model, scored by the probability of each option's label.

This reimplements System1-Arena's ``lm:`` adapter in label mode without thinking (spec
``lm:<model>@label-think0``; harness 28e16b1, ``s1arena/models/local_lm.py``) on Hugging Face
transformers. The same code later reads our own backbones (plan v3, arm R0).

* **One prompt per question.** It holds the system message, the state as indented JSON, and the question
  with its options under short labels: A, B, ... for choices; 0-9 for levels; yes / no for booleans. It
  is rendered with the model's chat template, thinking off, and ends where the answer starts.
* **One forward pass** gives the log-probability of every label token at that position. The answer is the
  softmax over labels. For yes / no, the capitalised variants ("Yes", "YES") are summed in.
* **Labels of several tokens are scored on a prefix tree.** That covers zero-padded numbers ("001" ...
  "255") for more than 26 options, and any label that is not one token in context. A label's score is
  the sum of log-probabilities along its path, with one cached forward step per tree node.
* **Prompts longer than the context minus 64 tokens are refused**, and so go unanswered, as upstream
  does.

Tokenisation is the part that most often breaks parity, so :meth:`Readout.prepare` (prompts, labels and
token ids, no model) is separate from scoring and is tested on its own.
"""

from __future__ import annotations

import json
import math
import string
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from systemone.decisions.schema import Answer, DecisionRequest, ModelResponse, Question

LETTERS = list(string.ascii_uppercase)
SYSTEM = (
    "You are a classifier. Read the input and answer the question. Reply with the label of your answer only."
)
REJECTED = "rejected: "  # prefix of errors that are final: the input does not fit the model
CONTEXT_MARGIN = 64


def state_text(request: DecisionRequest) -> str:
    return "Input:\n" + json.dumps(request.state, ensure_ascii=False, indent=1)


def labels_for(question: Question) -> list[str]:
    n = len(question.option_ids())
    if question.type == "boolean":
        return ["yes", "no"]
    if question.type == "score":
        return [str(i) for i in range(n)]
    if n <= len(LETTERS):
        return LETTERS[:n]
    width = len(str(n))
    return [str(i + 1).zfill(width) for i in range(n)]


def descriptions(question: Question) -> list[str]:
    return ["yes", "no"] if question.type == "boolean" else question.option_texts()


def question_text(question: Question, labels: list[str]) -> str:
    lines = [f"Question: {question.instructions}"]
    if question.type == "boolean":
        lines.append("Reply yes or no.")
    else:
        lines.append("Levels:" if question.type == "score" else "Options:")
        lines += [f"{label}. {text}" for label, text in zip(labels, descriptions(question), strict=True)]
        lines.append(
            "Reply with the number of one level."
            if question.type == "score"
            else "Reply with the label of one option."
        )
    return "\n".join(lines)


@dataclass
class Unit:
    """One question of one request: its prompt and how each label is read."""

    request: int
    question: str
    labels: list[str]
    prompt_ids: list[int] = field(default_factory=list)
    variants: list[list[int]] | None = None  # single-token ids per label (slot mode)
    paths: list[list[int]] | None = None  # token path per label (tree mode)
    error: str | None = None
    scores: list[float] | None = None
    mass: float = 0.0  # probability mass the labels hold at the answer position (diagnostic)


class Readout:
    """A causal LM read at the answer token. ``from_pretrained`` needs the ``encode`` extra."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        name: str,
        context: int,
        device: Any = "cpu",
        max_batch_tokens: int = 32768,
        max_batch_rows: int = 64,
    ) -> None:
        self.tok = tokenizer
        self.model = model
        self.hf_id = name
        self.context = context
        self.device = device
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_rows = max_batch_rows
        self._thinking_flag = "enable_thinking" in (getattr(tokenizer, "chat_template", None) or "")

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str,
        device: str | None = None,
        dtype: Any = None,
        load_model: bool = True,
        **kwargs: Any,
    ) -> Readout:
        """Tokenizer, config and (unless ``load_model=False``) weights from the Hugging Face Hub."""
        import torch
        from transformers import AutoConfig, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_id)
        config = AutoConfig.from_pretrained(hf_id)
        context = int(kwargs.pop("context", None) or getattr(config, "max_position_embeddings", 32768))
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model: Any = None
        if load_model:
            from transformers import AutoModelForCausalLM

            # typed as Any: transformers' wrapped ``.to`` confuses mypy when it is installed
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, dtype=dtype or torch.bfloat16, attn_implementation="sdpa"
            )
            model = model.to(dev).eval()
        return cls(tok, model, hf_id, context, dev, **kwargs)

    @property
    def model_id(self) -> str:
        return f"{self.hf_id}@label-think0"

    # ---------- prompts and labels (no model) ----------

    def _encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def prompt(self, user: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        extra = {"enable_thinking": False} if self._thinking_flag else {}
        return str(
            self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        )

    def continuation(self, prefix: str, text: str) -> list[int]:
        """Tokens of ``text`` written right after ``prefix``, tokenized in context."""
        tail = prefix[-64:]
        base = self._encode(tail)
        full = self._encode(tail + text)
        return full[len(base) :] if full[: len(base)] == base else self._encode(text)

    def single_ids(self, label: str, prefix: str) -> list[int]:
        """Single-token ids of the label (and, for yes / no, its capitalised forms) right after the prefix."""
        ids: list[int] = []
        forms = (label, label.capitalize(), label.upper()) if label in ("yes", "no") else (label,)
        for variant in dict.fromkeys(forms):
            tokens = self.continuation(prefix, variant)
            if len(tokens) == 1 and tokens[0] not in ids:
                ids.append(tokens[0])
        return ids

    def prepare(self, requests: Sequence[DecisionRequest]) -> list[Unit]:
        """Every question of every request as a unit with its prompt ids and label reads."""
        limit = self.context - CONTEXT_MARGIN
        units: list[Unit] = []
        for r, request in enumerate(requests):
            state = state_text(request)
            for name, question in request.questions.items():
                labels = labels_for(question)
                unit = Unit(r, name, labels)
                prompt = self.prompt(f"{state}\n\n{question_text(question, labels)}")
                unit.prompt_ids = self._encode(prompt)
                if len(unit.prompt_ids) > limit:
                    unit.error = REJECTED + f"prompt longer than the {self.context}-token context"
                else:
                    variants = [self.single_ids(label, prompt) for label in labels]
                    flat = [i for v in variants for i in v]
                    if all(variants) and len(set(flat)) == len(flat):
                        unit.variants = variants
                    else:
                        unit.paths = [self.continuation(prompt, label) for label in labels]
                units.append(unit)
        return units

    # ---------- scoring ----------

    def _batches(self, units: list[Unit]) -> list[list[Unit]]:
        """Slot units, longest first, in batches under the token and row budgets."""
        ordered = sorted(units, key=lambda u: -len(u.prompt_ids))
        batches: list[list[Unit]] = []
        for u in ordered:
            if batches:
                last = batches[-1]
                width = len(last[0].prompt_ids)  # the longest in the batch
                if len(last) < self.max_batch_rows and width * (len(last) + 1) <= self.max_batch_tokens:
                    last.append(u)
                    continue
            batches.append([u])
        return batches

    def _last_logprobs(self, batch: list[Unit]) -> Any:
        """Log-probabilities over the vocabulary at the last prompt position of each unit, left-padded."""
        import torch

        assert self.model is not None
        width = max(len(u.prompt_ids) for u in batch)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, u in enumerate(batch):
            ids[row, width - len(u.prompt_ids) :] = torch.tensor(u.prompt_ids)
            mask[row, width - len(u.prompt_ids) :] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        if len(batch) == 1:  # no padding: let SDPA run its causal kernel without a mask
            out = self.model(input_ids=ids, logits_to_keep=1)
        else:
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, logits_to_keep=1)
        return out.logits[:, -1].float().log_softmax(-1)

    def _score_slots(self, units: list[Unit]) -> None:
        import torch

        with torch.inference_mode():
            for batch in self._batches(units):
                logprobs = self._last_logprobs(batch).cpu()
                for row, u in enumerate(batch):
                    assert u.variants is not None
                    masses = [sum(math.exp(float(logprobs[row, i])) for i in v) for v in u.variants]
                    u.mass = sum(masses)
                    u.scores = [math.log(m) if m > 0 else -1e9 for m in masses]

    def _score_tree(self, u: Unit) -> None:
        """Sum of log-probabilities along each label's path; one cached forward step per tree node."""
        import torch

        assert self.model is not None
        assert u.paths is not None
        children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for path in u.paths:
            for k in range(len(path)):
                children[tuple(path[:k])].add(path[k])
        reads: dict[tuple[int, ...], Any] = {}
        n = len(u.prompt_ids)
        with torch.inference_mode():
            ids = torch.tensor([u.prompt_ids], device=self.device)
            out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            reads[()] = out.logits[0, -1].float().log_softmax(-1).cpu()

            def visit(node: tuple[int, ...]) -> None:
                for token in sorted(children[node]):
                    child = (*node, token)
                    if child not in children:  # a leaf: its log-probability was read at the parent
                        continue
                    step = torch.tensor([[token]], device=self.device)
                    position = torch.tensor([[n + len(node)]], device=self.device)
                    o = self.model(
                        input_ids=step, past_key_values=cache, position_ids=position, use_cache=True
                    )
                    reads[child] = o.logits[0, -1].float().log_softmax(-1).cpu()
                    visit(child)
                    cache.crop(n + len(node))

            visit(())
        totals = [sum(float(reads[tuple(p[:k])][p[k]]) for k in range(len(p))) for p in u.paths]
        u.mass = sum(math.exp(t) for t in totals)
        u.scores = totals

    def score_units(self, units: list[Unit]) -> None:
        todo = [u for u in units if u.error is None]
        self._score_slots([u for u in todo if u.variants is not None])
        for u in todo:
            if u.paths is not None:
                self._score_tree(u)

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]:
        started = time.perf_counter()
        units = self.prepare(requests)
        self.score_units(units)
        per_request = round((time.perf_counter() - started) * 1000 / max(1, len(requests)), 2)
        by_request: dict[int, list[Unit]] = defaultdict(list)
        for u in units:
            by_request[u.request].append(u)
        responses = []
        for r, request in enumerate(requests):
            answers: dict[str, Answer] = {}
            errors: list[tuple[str, str]] = []
            usage: dict[str, Any] = {"candidate_mass": {}, "batch_ms_per_request": per_request}
            for u in by_request[r]:
                if u.error is not None or u.scores is None:
                    errors.append((u.question, u.error or "not scored"))
                    continue
                answers[u.question] = Answer(probs=combine(u.scores, request.questions[u.question]))
                usage["candidate_mass"][u.question] = round(u.mass, 4)
            error = None
            if errors:
                final = all(e.startswith(REJECTED) for _, e in errors)
                detail = "; ".join(f"{name}: {e.removeprefix(REJECTED)}" for name, e in errors)
                error = (REJECTED if final else "") + detail[:300]
            responses.append(ModelResponse(answers=answers, usage=usage, error=error))
        return responses


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def combine(scores: list[float], question: Question) -> dict[str, float]:
    """Label scores -> a distribution over the question's option ids."""
    probs = softmax(scores)
    if question.type == "boolean":
        return {"true": probs[0], "false": probs[1]}
    return dict(zip(question.option_ids(), probs, strict=True))

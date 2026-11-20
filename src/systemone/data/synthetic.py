"""A synthetic stand-in for the real vectors, with the same structure and failure modes.

Each pair shares a latent "meaning" ``z``. The state and the action see ``z`` through two
different random linear maps plus noise, like a question and its answer seen by an LLM.
Hard negatives are actions for a *nearby* meaning, so they are deliberately confusable, and a
fraction of pairs share a document id. A random head scores near chance; a trained head should
score far above it, which makes this a meaningful smoke test of the whole pipeline.
"""

from __future__ import annotations

import numpy as np

from systemone.config import SyntheticConfig
from systemone.data.pairs import PairSet


def _unit(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    return out


def make_synthetic(cfg: SyntheticConfig | None = None) -> tuple[PairSet, PairSet]:
    """Return ``(train, eval)`` pair sets; both carry hard negatives."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    to_state = rng.standard_normal((cfg.latent_dim, cfg.dim)) / np.sqrt(cfg.latent_dim)
    to_action = rng.standard_normal((cfg.latent_dim, cfg.dim)) / np.sqrt(cfg.latent_dim)
    # a shared "style" component makes raw cosine similarity uninformative, as with real LLM vectors
    style = rng.standard_normal(cfg.dim)

    def split(n: int, doc_offset: int) -> PairSet:
        z = rng.standard_normal((n, cfg.latent_dim))
        noise = lambda: cfg.noise * rng.standard_normal((n, cfg.dim))  # noqa: E731
        states = _unit(z @ to_state + noise() + 2.0 * style)
        actions = _unit(z @ to_action + noise() + 2.0 * style)
        k = cfg.num_negatives
        c = cfg.negative_closeness
        z_neg = c * z[:, None, :] + np.sqrt(1 - c**2) * rng.standard_normal((n, k, cfg.latent_dim))
        negs = _unit(z_neg @ to_action + cfg.noise * rng.standard_normal((n, k, cfg.dim)) + 2.0 * style)
        n_docs = max(1, int(n * (1 - cfg.docs_per_pair)))
        doc = doc_offset + rng.integers(0, n_docs, n).astype(np.int64)
        neg_count = rng.integers(k // 2, k + 1, n).astype(np.int64)
        return PairSet(
            states.astype(np.float16),
            actions.astype(np.float16),
            doc,
            negs.astype(np.float16),
            neg_count,
        )

    return split(cfg.n_train, 0), split(cfg.n_eval, 10**9)

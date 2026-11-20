"""systemone: a System One decision model built from scratch.

A System One model does not generate text. Given a situation (the *state*) and a list of
candidate *actions*, it returns a probability for every option in a single forward pass:

    state  -> frozen LLM -> last-token vector -> state head  -> z_s (unit norm)
    action -> frozen LLM -> last-token vector -> action head -> z_a (unit norm)
    p(action_i | state) = softmax_i( scale * <z_s, z_a_i> )
"""

from systemone.config import ExperimentConfig, HeadConfig, LossConfig, ModelConfig
from systemone.model import DualEncoder, ProjectionHead

__all__ = [
    "DualEncoder",
    "ExperimentConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "ProjectionHead",
    "__version__",
]

__version__ = "0.1.0"

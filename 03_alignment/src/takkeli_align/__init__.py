"""REINFORCE++ alignment via OpenRLHF.

This module implements the REINFORCE++ algorithm (arXiv:2501.03262) for
critic-free RLHF alignment of the custom 1B Takkeli model.  Key components:

- :mod:`config` — Dataclass-based pipeline configuration (single-GPU, no critic).
- :mod:`reinforce_pp` — Core algorithm: global advantage normalization,
  token-level KL penalties, trust region clipping, and loss computation.
- :mod:`pipeline` — Pipeline orchestration tying policy/reference models
  with the REINFORCE++ loss.
"""

from takkeli_align.config import (
    AlignmentModelConfig,
    HardwareConfig,
    OptimizerConfig,
    ReinforcePPConfig,
    ReinforcePPPipelineConfig,
)
from takkeli_align.pipeline import ReinforcePPPipeline
from takkeli_align.reinforce_pp import (
    clip_log_ratio,
    clip_rewards,
    compute_log_probs_from_logits,
    global_normalize_advantages,
    reinforce_pp_loss,
    token_level_kl,
)

__all__ = [
    # Config
    "HardwareConfig",
    "AlignmentModelConfig",
    "OptimizerConfig",
    "ReinforcePPConfig",
    "ReinforcePPPipelineConfig",
    # Pipeline
    "ReinforcePPPipeline",
    # Core algorithm
    "clip_log_ratio",
    "clip_rewards",
    "compute_log_probs_from_logits",
    "global_normalize_advantages",
    "reinforce_pp_loss",
    "token_level_kl",
]

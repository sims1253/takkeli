"""REINFORCE++ pipeline configuration.

Defines a dataclass-based config that mirrors OpenRLHF's REINFORCE++ settings
while targeting single-GPU execution with no critic model.  The config can be
serialized to/from YAML/JSON and is validated on construction.

References:
    - REINFORCE++: arXiv:2501.03262
    - OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class AlignmentModelConfig:
    """Policy / reference model configuration.

    Attributes:
        pretrained_model_name_or_path: HuggingFace model id or local path.
        max_seq_len: Maximum sequence length for generation / training.
    """

    pretrained_model_name_or_path: str = "takkeli/takkeli-1b"
    max_seq_len: int = 2048


# ---------------------------------------------------------------------------
# REINFORCE++ algorithm hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class ReinforcePPConfig:
    """REINFORCE++ algorithm configuration.

    All hyper-parameters follow the naming conventions from the REINFORCE++
    paper (arXiv:2501.03262) and the OpenRLHF implementation.

    Attributes:
        kl_coeff: Coefficient for the token-level KL penalty.
        kl_target: Target KL divergence value for adaptive tuning (0 disables).
        clip_range: Trust-region clipping epsilon.  Ratios are clipped to
            ``[1 - clip_range, 1 + clip_range]``.
        clip_range_low: Lower bound for ratio clipping.
        clip_range_high: Upper bound for ratio clipping.
        gamma: Discount factor for advantage computation (1.0 = undiscounted).
        normalize_advantage: Whether to apply global advantage normalization.
        norm_adv_batch: Whether to normalize advantages globally across the
            entire batch (True) or per prompt group (False).  REINFORCE++
            uses global normalization.
        norm_adv_eps: Small constant added to advantage std to prevent
            division by zero.
        reward_clip_range: Clip rewards to ``[-reward_clip_range,
            +reward_clip_range]`` to stabilize training.
        temperature: Sampling temperature for policy generation.
        n_samples_per_prompt: Number of completions sampled per prompt.
        scale_reward: If True, divide rewards by their running std.
    """

    kl_coeff: float = 0.1
    kl_target: float = 0.0
    clip_range: float = 0.2
    clip_range_low: float = 0.0
    clip_range_high: float = 0.0
    gamma: float = 1.0
    normalize_advantage: bool = True
    norm_adv_batch: bool = True
    norm_adv_eps: float = 1e-8
    reward_clip_range: float = 0.0
    temperature: float = 1.0
    n_samples_per_prompt: int = 1
    scale_reward: bool = True

    def __post_init__(self) -> None:
        """Derive clip bounds from ``clip_range`` if not explicitly set."""
        if self.clip_range_low == 0.0 and self.clip_range_high == 0.0:
            self.clip_range_low = 1.0 - self.clip_range
            self.clip_range_high = 1.0 + self.clip_range


# ---------------------------------------------------------------------------
# Hardware / environment configuration
# ---------------------------------------------------------------------------


@dataclass
class HardwareConfig:
    """Single-GPU hardware configuration.

    Attributes:
        n_gpus: Number of GPUs (must be 1 for this config).
        tensor_parallel_size: Tensor-parallel degree (must be 1).
        memory_budget_gb: VRAM budget in GB for the training pipeline.
        device: Torch device string.
        dtype: Torch dtype string (``"bfloat16"`` or ``"float32"``).
        use_bf16: Whether to use bfloat16 mixed precision.
        grad_ckpt: Whether to use gradient checkpointing.
        per_device_train_batch_size: Per-device batch size.
        per_device_eval_batch_size: Per-device eval batch size.
        num_train_epochs: Total training epochs.
        gradient_accumulation_steps: Gradient accumulation steps.
    """

    n_gpus: int = 1
    tensor_parallel_size: int = 1
    memory_budget_gb: float = 24.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_bf16: bool = True
    grad_ckpt: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        """Validate single-GPU constraints."""
        if self.n_gpus != 1:
            raise ValueError(f"REINFORCE++ config requires n_gpus=1, got {self.n_gpus}")
        if self.tensor_parallel_size != 1:
            raise ValueError(
                f"REINFORCE++ config requires tensor_parallel_size=1, "
                f"got {self.tensor_parallel_size}"
            )


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Optimizer configuration for REINFORCE++ training.

    Attributes:
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        lr_scheduler_type: Learning rate scheduler type.
        warmup_ratio: Fraction of total steps used for warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        optim: Optimizer name (``"adamw"`` or ``"adamw_torch"``).
    """

    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class ReinforcePPPipelineConfig:
    """Top-level configuration for the REINFORCE++ alignment pipeline.

    Combines all sub-configs and provides serialization helpers.

    Attributes:
        model: Model configuration.
        algorithm: REINFORCE++ algorithm hyper-parameters.
        hardware: Hardware / environment settings.
        optimizer: Optimizer configuration.
        seed: Random seed for reproducibility.
        output_dir: Directory for checkpoints and logs.
        run_name: Human-readable run identifier.
        use_critic: Whether to instantiate a critic model (must be False).
    """

    model: AlignmentModelConfig = field(default_factory=AlignmentModelConfig)
    algorithm: ReinforcePPConfig = field(default_factory=ReinforcePPConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 42
    output_dir: str = "./output/rlhf"
    run_name: str = "reinforcepp"
    use_critic: bool = False

    def __post_init__(self) -> None:
        """Validate pipeline-level invariants."""
        if self.use_critic:
            raise ValueError("REINFORCE++ is critic-free. Set use_critic=False.")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary (nested via dataclasses)."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReinforcePPPipelineConfig:
        """Deserialize config from a plain dictionary."""
        return cls(
            model=AlignmentModelConfig(**d.get("model", {})),
            algorithm=ReinforcePPConfig(**d.get("algorithm", {})),
            hardware=HardwareConfig(**d.get("hardware", {})),
            optimizer=OptimizerConfig(**d.get("optimizer", {})),
            seed=d.get("seed", 42),
            output_dir=d.get("output_dir", "./output/rlhf"),
            run_name=d.get("run_name", "reinforcepp"),
            use_critic=d.get("use_critic", False),
        )

    def save_yaml(self, path: Path | str) -> None:
        """Save config to a YAML file.

        Args:
            path: Output file path.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: Path | str) -> None:
        """Save config to a JSON file.

        Args:
            path: Output file path.
        """
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Path | str) -> ReinforcePPPipelineConfig:
        """Load config from a YAML file.

        Args:
            path: Input YAML file path.

        Returns:
            Parsed configuration.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ImportError: If PyYAML is not installed.
        """
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def load_json(cls, path: Path | str) -> ReinforcePPPipelineConfig:
        """Load config from a JSON file.

        Args:
            path: Input JSON file path.

        Returns:
            Parsed configuration.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
        """
        import json

        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

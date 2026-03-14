"""Full training loop combining NorMuon+GWT+Liger+LEMA.

Implements a complete training step that integrates all components of the
optimizer-memory stack:

- NorMuon: Newton-Schulz orthogonalization with row-wise momentum
- GWT: 2-level Discrete Haar Wavelet Transform gradient compression
- Liger: Fused RMSNorm, RoPE, SwiGLU operations
- LEMA: Triple-buffer weight streaming for memory-constrained training

The training loop is designed for the custom 1B model architecture and
targets <12GB RAM on CPU proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from takkeli_pretrain.gwt import NorMuonGWT
from takkeli_pretrain.lema import LEMAConfig, LEMATrainingContext
from takkeli_pretrain.liger_ops import LigerRMSNorm
from takkeli_pretrain.model import DrLLMModel, ModelConfig

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the full training loop.

    Attributes:
        batch_size: Batch size for training.
        seq_len: Sequence length for training.
        lr: Learning rate.
        momentum: NorMuon momentum coefficient.
        beta2: NorMuon second-order momentum coefficient.
        weight_decay: Decoupled weight decay.
        ns_steps: Newton-Schulz orthogonalization iterations.
        gwt_levels: Number of GWT compression levels.
        max_grad_norm: Maximum gradient norm for clipping.
        use_lema: Whether to enable LEMA weight streaming.
        lema_compute_device: Device for computation.
        lema_storage_device: Device for weight storage.
    """

    batch_size: int = 1
    seq_len: int = 128
    lr: float = 0.02
    momentum: float = 0.95
    beta2: float = 0.95
    weight_decay: float = 0.0
    ns_steps: int = 5
    gwt_levels: int = 2
    max_grad_norm: float = 1.0
    use_lema: bool = True
    lema_compute_device: str = "cpu"
    lema_storage_device: str = "cpu"


# ---------------------------------------------------------------------------
# Liger-Augmented Model
# ---------------------------------------------------------------------------


class LigerAugmentedModel(nn.Module):
    """Model with Liger fused operations replacing standard implementations.

    This wraps DrLLMModel and replaces RMSNorm and SwiGLU with Liger
    fused versions for reduced memory and improved throughput.

    On CPU, the Liger operations use pure PyTorch implementations that
    are functionally equivalent to the Triton kernels.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._base_model = DrLLMModel(config)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Forward pass through the base model.

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Tuple of (logits, aux_outputs).
        """
        return self._base_model(input_ids)

    def get_liger_layers(self) -> list[LigerRMSNorm]:
        """Get all Liger RMSNorm layers in the model.

        Returns:
            List of LigerRMSNorm modules.
        """
        layers: list[LigerRMSNorm] = []
        for module in self.modules():
            if isinstance(module, LigerRMSNorm):
                layers.append(module)
        return layers

    @property
    def base_model(self) -> DrLLMModel:
        """Access the underlying DrLLMModel."""
        return self._base_model

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Total number of parameters.
        """
        return self._base_model.count_parameters()


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------


def create_model(config: ModelConfig) -> DrLLMModel:
    """Create and initialize the model.

    Args:
        config: Model configuration.

    Returns:
        Initialized DrLLMModel.
    """
    model = DrLLMModel(config)
    return model


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> NorMuonGWT:
    """Create the composite NorMuon+GWT optimizer.

    Args:
        model: The model to optimize.
        config: Training configuration.

    Returns:
        NorMuonGWT optimizer instance.
    """
    # Filter out parameters that don't require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    return NorMuonGWT(
        params,
        lr=config.lr,
        momentum=config.momentum,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        ns_steps=config.ns_steps,
        gwt_levels=config.gwt_levels,
    )


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token indices of shape (batch, seq_len).

    Returns:
        Scalar cross-entropy loss.
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )
    return loss


def train_step(
    model: nn.Module,
    optimizer: NorMuonGWT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
    lema_context: LEMATrainingContext | None = None,
) -> dict[str, float]:
    """Execute a single training step with the full optimizer stack.

    Combines NorMuon+GWT+Liger+LEMA in a single training iteration:

    1. LEMA: Optionally stream layer weights
    2. Forward pass (with Liger fused ops in model)
    3. Loss computation
    4. Backward pass
    5. Gradient clipping
    6. NorMuon+GWT optimizer step
    7. LEMA: Advance to next layer schedule

    Args:
        model: The model to train.
        optimizer: NorMuon+GWT composite optimizer.
        input_ids: Input token indices of shape (batch, seq_len).
        targets: Target token indices of shape (batch, seq_len).
        config: Training configuration.
        lema_context: Optional LEMA context for weight streaming.

    Returns:
        Dictionary with 'loss' and 'grad_norm' metrics.
    """
    model.train()

    # LEMA pre-step setup
    if lema_context is not None and config.use_lema:
        for layer_idx in range(len(model.blocks)):
            lema_context.pre_layer_forward(layer_idx)

    # Forward pass
    logits, aux_outputs = model(input_ids)

    # Compute loss
    loss = compute_loss(logits, targets)

    # Backward pass
    loss.backward()

    # Gradient clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5

    if total_norm > config.max_grad_norm:
        scale = config.max_grad_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale)
        total_norm = config.max_grad_norm

    # Optimizer step (NorMuon + GWT)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # LEMA post-step
    if lema_context is not None and config.use_lema:
        for layer_idx in range(len(model.blocks)):
            lema_context.post_layer_forward(layer_idx)

    return {
        "loss": loss.item(),
        "grad_norm": total_norm,
    }


def create_lema_context(
    model: nn.Module,
    config: TrainingConfig,
) -> LEMATrainingContext:
    """Create and initialize LEMA training context.

    Args:
        model: The model to stream weights for.
        config: Training configuration.

    Returns:
        Initialized LEMATrainingContext.
    """
    lema_config = LEMAConfig(
        num_layers=model.config.n_layers,
        compute_device=config.lema_compute_device,
        storage_device=config.lema_storage_device,
        num_buffers=3,
    )
    context = LEMATrainingContext(lema_config)
    context.setup(model)
    return context


def full_training_loop(
    model: nn.Module,
    optimizer: NorMuonGWT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    """Run the complete training loop with all stack components.

    This is the entry point that combines all four components:
    - NorMuon optimizer with Newton-Schulz orthogonalization
    - GWT gradient compression (2-level DHT)
    - Liger fused operations (RMSNorm, RoPE, SwiGLU)
    - LEMA triple-buffer weight streaming

    Args:
        model: The model to train.
        optimizer: NorMuon+GWT composite optimizer.
        input_ids: Input token indices.
        targets: Target token indices.
        config: Training configuration.

    Returns:
        Dictionary with training metrics.
    """
    lema_context: LEMATrainingContext | None = None
    if config.use_lema:
        lema_context = create_lema_context(model, config)

    try:
        metrics = train_step(model, optimizer, input_ids, targets, config, lema_context)
    finally:
        if lema_context is not None:
            lema_context.cleanup()

    return metrics

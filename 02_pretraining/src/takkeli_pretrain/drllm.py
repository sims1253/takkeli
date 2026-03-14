"""Dr.LLM: Dynamic Layer Routing for Large Language Models.

Implements per-layer routers that decide at inference/training time whether
to Skip, Execute, or Repeat each transformer layer for each sequence in the
batch. This enables the model to dynamically allocate compute based on input
complexity.

Key components:
- ``WindowedPool``: Windowed mean-pooling across sequence positions to
  aggregate hidden states before routing decisions.
- ``DynamicRouter``: Lightweight bottleneck MLP that maps pooled hidden states
  to 3 routing logits (Skip, Execute, Repeat).
- ``FocalLoss``: Class-balanced loss for routing stability under the severe
  class imbalance typical of dynamic routing.

Reference: arXiv:2510.12773 — "Dr.LLM: Dynamic Layer Routing in LLMs"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrLLMConfig:
    """Configuration for Dr.LLM dynamic routing.

    Attributes:
        d_model: Model hidden dimension (input to the router).
        d_router_hidden: Hidden dimension of the router bottleneck MLP.
        pool_window_size: Window size for mean-pooling across the sequence
            dimension. If 0, uses global mean pooling.
        num_routing_choices: Number of routing decisions. Default is 3
            for Skip, Execute, Repeat.
        focal_gamma: Focusing parameter for focal loss. Higher values
            down-weight well-classified examples more aggressively.
        focal_alpha: Per-class weight for focal loss. If None, uses
            uniform weights.
        temperature: Temperature for softmax sharpening on routing logits.
            Lower values produce sharper (more confident) routing decisions.
    """

    d_model: int = 2048
    d_router_hidden: int = 128
    pool_window_size: int = 0
    num_routing_choices: int = 3
    focal_gamma: float = 2.0
    focal_alpha: list[float] | None = None
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Routing Signal Constants
# ---------------------------------------------------------------------------

SKIP = 0
EXECUTE = 1
REPEAT = 2


# ---------------------------------------------------------------------------
# Windowed Pooling
# ---------------------------------------------------------------------------


class WindowedPool(nn.Module):
    """Windowed mean pooling across sequence positions.

    Aggregates the input tensor along the sequence dimension using either
    a sliding window or global mean pooling. This reduces the sequence-level
    information to a fixed-size representation suitable for routing decisions.

    Args:
        pool_window_size: Size of the pooling window. If 0 or >= seq_len,
            uses global mean pooling over the entire sequence.
    """

    def __init__(self, pool_window_size: int = 0) -> None:
        super().__init__()
        self.pool_window_size = pool_window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply windowed mean pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Pooled tensor of shape (batch, d_model).
        """
        batch, seq_len, d_model = x.shape

        if self.pool_window_size <= 0 or self.pool_window_size >= seq_len:
            # Global mean pooling
            return x.mean(dim=1)

        # Windowed mean pooling with overlap
        window = self.pool_window_size
        # Pad sequence to make it evenly divisible by window size
        pad_len = (window - seq_len % window) % window
        if pad_len > 0:
            x = functional.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)

        padded_len = x.shape[1]
        # Reshape into windows and average
        x_windowed = x.reshape(batch, padded_len // window, window, d_model)
        x_pooled = x_windowed.mean(dim=2)  # (batch, num_windows, d_model)
        # Average across windows
        return x_pooled.mean(dim=1)  # (batch, d_model)

    def extra_repr(self) -> str:
        return f"pool_window_size={self.pool_window_size}"


# ---------------------------------------------------------------------------
# Dynamic Router
# ---------------------------------------------------------------------------


class DynamicRouter(nn.Module):
    """Per-layer router that decides Skip/Execute/Repeat for each sequence.

    Architecture:
        1. Windowed mean-pooling over the sequence dimension
        2. Bottleneck MLP: d_model -> d_router_hidden -> num_routing_choices
        3. Temperature-scaled softmax for probability distribution

    The router is designed to be extremely lightweight (< 0.1% of total
    model parameters) so it adds negligible overhead.

    Args:
        config: Dr.LLM routing configuration.
    """

    def __init__(self, config: DrLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.num_routing_choices = config.num_routing_choices

        # Windowed pooling
        self.pooler = WindowedPool(pool_window_size=config.pool_window_size)

        # Bottleneck MLP: d_model -> d_router_hidden -> num_routing_choices
        self.down_proj = nn.Linear(config.d_model, config.d_router_hidden, bias=True)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(config.d_router_hidden, config.num_routing_choices, bias=True)

        self.temperature = config.temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities for each sequence.

        Args:
            x: Hidden states of shape (batch, seq_len, d_model).

        Returns:
            Routing probability tensor of shape (batch, num_routing_choices).
            Each row sums to 1.0 (after softmax). Columns correspond to
            [Skip, Execute, Repeat] by default.
        """
        # Pool over sequence dimension
        pooled = self.pooler(x)  # (batch, d_model)

        # Bottleneck MLP
        hidden = self.activation(self.down_proj(pooled))  # (batch, d_router_hidden)
        logits = self.up_proj(hidden)  # (batch, num_routing_choices)

        # Temperature-scaled softmax
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = functional.softmax(logits, dim=-1)  # (batch, num_routing_choices)
        return probs

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (before softmax) for loss computation.

        Args:
            x: Hidden states of shape (batch, seq_len, d_model).

        Returns:
            Raw routing logits of shape (batch, num_routing_choices).
        """
        pooled = self.pooler(x)
        hidden = self.activation(self.down_proj(pooled))
        logits = self.up_proj(hidden)
        return logits

    def extra_repr(self) -> str:
        return (
            f"d_model={self.config.d_model}, "
            f"d_router_hidden={self.config.d_router_hidden}, "
            f"num_routing_choices={self.num_routing_choices}, "
            f"temperature={self.temperature}"
        )


# ---------------------------------------------------------------------------
# Focal Loss for Routing Stability
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for routing classification stability.

    Standard cross-entropy can be overwhelmed by the majority class
    (typically Execute) in dynamic routing. Focal loss down-weights
    well-classified examples, allowing the router to focus on difficult
    routing decisions.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. gamma >= 0. Higher values down-weight
            easy examples more aggressively.
        alpha: Optional per-class weights of shape (num_classes,).
            If None, uniform weighting is used.
        reduction: Reduction mode: 'none', 'mean', or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw logits of shape (batch, num_classes).
            targets: Target class indices of shape (batch,) with dtype
                torch.long.

        Returns:
            Scalar focal loss (if reduction is 'mean' or 'sum'),
            or per-sample loss of shape (batch,) if reduction is 'none'.
        """
        # Standard cross-entropy components
        log_probs = functional.log_softmax(logits, dim=-1)
        probs = functional.softmax(logits, dim=-1)

        # Gather the probability of the target class
        targets_expanded = targets.unsqueeze(-1)
        p_t = probs.gather(1, targets_expanded).squeeze(-1)  # (batch,)
        log_p_t = log_probs.gather(1, targets_expanded).squeeze(-1)  # (batch,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = focal_weight * alpha_t

        # Focal loss
        loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, "
            f"alpha={'None' if self.alpha is None else f'shape={self.alpha.shape}'}, "
            f"reduction='{self.reduction}'"
        )

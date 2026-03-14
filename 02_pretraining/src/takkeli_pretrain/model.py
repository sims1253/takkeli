"""Full 1B-parameter transformer model with Dr.LLM routing integration.

Assembles the complete model architecture:
- Token embeddings + positional embeddings
- N transformer blocks, each containing:
    - Multi-Head Latent Attention (DeepSeek MLA) with IndexCache support
    - Feed-forward network using BitLinear layers
    - Dr.LLM dynamic router (Skip/Execute/Repeat) per layer
- Final RMSNorm + LM head

The model is designed to target ~1B parameters within the 800M-1.2B range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from takkeli_pretrain.bitlinear import BitLinear
from takkeli_pretrain.drllm import (
    DrLLMConfig,
    DynamicRouter,
)
from takkeli_pretrain.indexcache import IndexCacheConfig, IndexCacheManager
from takkeli_pretrain.mla import MLAConfig, MultiHeadLatentAttention

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for the full 1B model.

    Attributes:
        vocab_size: Vocabulary size for token embeddings.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ffn: Feed-forward network hidden dimension.
        d_kv_laten: MLA KV latent dimension.
        d_q_laten: MLA query latent dimension.
        d_rope: RoPE dimension.
        sparse_top_k: Number of top-K tokens for sparse attention.
        index_pattern: Binary F/S pattern string for IndexCache.
        max_seq_len: Maximum sequence length for positional embeddings.
        enable_routing: Whether to enable Dr.LLM dynamic routing.
        d_router_hidden: Router bottleneck hidden dimension.
        pool_window_size: Windowed pooling size for routers.
        focal_gamma: Focal loss gamma for routing.
        focal_alpha: Per-class focal loss weights.
        router_temperature: Temperature for router softmax.
        tie_weights: Whether to tie LM head weights with embedding.
    """

    vocab_size: int = 32000
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 24
    d_ffn: int = 5504
    d_kv_laten: int = 512
    d_q_laten: int = 512
    d_rope: int = 64
    sparse_top_k: int = 64
    index_pattern: str = "FSFFSFSFFSFSFFSFSFFSFSFF"
    max_seq_len: int = 2048
    enable_routing: bool = True
    d_router_hidden: int = 128
    pool_window_size: int = 0
    focal_gamma: float = 2.0
    focal_alpha: list[float] | None = None
    router_temperature: float = 1.0
    tie_weights: bool = True


# ---------------------------------------------------------------------------
# RMS Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input using RMS: x_norm = x / sqrt(mean(x^2) + eps) * gamma.

    Args:
        d_model: Model hidden dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        rms = torch.sqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + self.eps)
        x_norm = (x.float() / rms).to(x.dtype)
        return x_norm * self.gamma

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU with BitLinear)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation and BitLinear layers.

    Architecture:
        SwiGLU(x) = (x W_gate * sigmoid(x W_up)) W_down

    Args:
        d_model: Input/output dimension.
        d_ffn: Intermediate (hidden) dimension.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.w_gate = BitLinear(d_model, d_ffn, bias=False)
        self.w_up = BitLinear(d_model, d_ffn, bias=False)
        self.w_down = BitLinear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(functional.silu(gate) * up)


# ---------------------------------------------------------------------------
# Transformer Block with Dr.LLM Router
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer block with MLA attention, FFN, and routing.

    Architecture:
        1. Pre-LayerNorm -> MLA Attention -> Residual
        2. Dr.LLM Router decides Skip/Execute/Repeat
        3. If Execute: Pre-LayerNorm -> FFN -> Residual
        4. If Skip: pass through without FFN
        5. If Repeat: FFN applied twice with residual connections

    Args:
        config: Full model configuration.
        layer_idx: Index of this layer in the stack (0-based).
        is_full_layer: Whether this is an F-layer for IndexCache.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        is_full_layer: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_layer = is_full_layer

        # Pre-attention layer norm
        self.attn_norm = RMSNorm(config.d_model)

        # MLA Attention
        mla_config = MLAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_kv_laten=config.d_kv_laten,
            d_q_laten=config.d_q_laten,
            d_rope=config.d_rope,
            sparse_top_k=config.sparse_top_k,
        )
        self.attn = MultiHeadLatentAttention(mla_config, is_full_layer=is_full_layer)

        # Dr.LLM Router
        self.router: DynamicRouter | None = None
        if config.enable_routing:
            drllm_config = DrLLMConfig(
                d_model=config.d_model,
                d_router_hidden=config.d_router_hidden,
                pool_window_size=config.pool_window_size,
                num_routing_choices=3,
                focal_gamma=config.focal_gamma,
                focal_alpha=config.focal_alpha,
                temperature=config.router_temperature,
            )
            self.router = DynamicRouter(drllm_config)

        # Pre-FFN layer norm
        self.ffn_norm = RMSNorm(config.d_model)

        # Feed-forward network
        self.ffn = FeedForward(config.d_model, config.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            sparse_indices: Pre-computed sparse indices for S-layers.

        Returns:
            Tuple of:
                - output: Output tensor of shape (batch, seq_len, d_model).
                - indices: Sparse indices from this layer (F-layer only).
                - routing_probs: Routing probabilities (batch, 3) if routing
                  is enabled, otherwise None.
        """
        residual = x

        # Pre-attention norm + MLA attention + residual
        x_normed = self.attn_norm(x)
        attn_output, indices = self.attn(x_normed, sparse_indices=sparse_indices)
        x = residual + attn_output

        # Dr.LLM routing decision
        routing_probs: torch.Tensor | None = None
        if self.router is not None:
            routing_probs = self.router(x)  # (batch, 3)

            # Differentiable soft routing using routing probabilities.
            # Skip: p[SKIP], Execute: p[EXECUTE], Repeat: p[REPEAT].
            # Use soft masks so gradients flow through to router parameters.
            p_execute = routing_probs[:, 1:2].unsqueeze(-1)  # (batch, 1, 1)
            p_repeat = routing_probs[:, 2:3].unsqueeze(-1)  # (batch, 1, 1)

            # FFN with soft routing
            x_normed = self.ffn_norm(x)
            ffn_out = self.ffn(x_normed)

            # Execute: scale by execute probability
            x = x + ffn_out * p_execute

            # Repeat: apply FFN a second time, scaled by repeat probability
            x_normed2 = self.ffn_norm(x)
            ffn_out2 = self.ffn(x_normed2)
            x = x + ffn_out2 * p_repeat
        else:
            # No routing: always execute FFN
            x_normed = self.ffn_norm(x)
            x = x + self.ffn(x_normed)

        return x, indices, routing_probs

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}, is_full_layer={self.is_full_layer}"


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class DrLLMModel(nn.Module):
    """Full 1B-parameter transformer model with Dr.LLM dynamic routing.

    Assembles:
        - Token embeddings (optionally tied with LM head)
        - Learned positional embeddings
        - N TransformerBlocks with MLA + BitLinear FFN + Dr.LLM routers
        - Final RMSNorm + LM head

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Validate index pattern length
        if len(config.index_pattern) != config.n_layers:
            raise ValueError(
                f"Index pattern length ({len(config.index_pattern)}) must equal "
                f"number of layers ({config.n_layers}). "
                f"Got pattern: '{config.index_pattern}'"
            )

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Learned positional embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            is_full = config.index_pattern[i] == "F"
            block = TransformerBlock(
                config=config,
                layer_idx=i,
                is_full_layer=is_full,
            )
            self.blocks.append(block)

        # IndexCache manager
        self.index_cache = IndexCacheManager(
            IndexCacheConfig(
                pattern=config.index_pattern,
                num_layers=config.n_layers,
            )
        )

        # Final layer norm
        self.final_norm = RMSNorm(config.d_model)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Full forward pass of the model.

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Tuple of:
                - logits: Output logits of shape (batch, seq_len, vocab_size).
                - aux_outputs: Dictionary containing:
                    - 'routing_probs': List of (batch, 3) tensors, one per layer.
                    - 'sparse_indices': List of index tensors from F-layers.
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        aux_outputs: dict[str, list[torch.Tensor]] = {
            "routing_probs": [],
            "sparse_indices": [],
        }

        # Pass through transformer blocks with IndexCache
        cached_indices: torch.Tensor | None = None

        for i, block in enumerate(self.blocks):
            # Determine if this layer should use cached indices
            sparse_input = cached_indices if not self.index_cache.is_full_layer(i) else None

            x, indices, routing_probs = block(x, sparse_indices=sparse_input)

            # Update cached indices from F-layers
            if indices is not None:
                cached_indices = indices
                aux_outputs["sparse_indices"].append(indices)

            # Collect routing probabilities
            if routing_probs is not None:
                aux_outputs["routing_probs"].append(routing_probs)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, aux_outputs

    def get_routing_decisions(self, input_ids: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Get routing decisions for all layers (for analysis/visualization).

        Args:
            input_ids: Token indices of shape (batch, seq_len).

        Returns:
            Dictionary with 'routing_probs' (list of (batch, 3) tensors per layer).
        """
        _, aux = self.forward(input_ids)
        return {"routing_probs": aux["routing_probs"]}

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Total number of parameters across all modules.
        """
        return sum(p.numel() for p in self.parameters())

    def count_router_parameters(self) -> int:
        """Count only the router parameters.

        Returns:
            Total number of router parameters across all layers.
        """
        from takkeli_pretrain.drllm import DynamicRouter

        total = 0
        for block in self.blocks:
            if isinstance(block.router, DynamicRouter):
                total += sum(p.numel() for p in block.router.parameters())
        return total

    def extra_repr(self) -> str:
        total_params = self.count_parameters()
        router_params = self.count_router_parameters()
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, "
            f"n_heads={self.config.n_heads}, "
            f"n_layers={self.config.n_layers}, "
            f"d_ffn={self.config.d_ffn}, "
            f"total_params={total_params:,}, "
            f"router_params={router_params:,}, "
            f"index_pattern='{self.config.index_pattern}'"
        )

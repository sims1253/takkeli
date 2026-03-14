"""GGUF export script for converting PyTorch BitNet ternary weights to GGUF format.

Converts a custom DrLLMModel (BitNet b1.58 architecture with MLA, IndexCache,
and Dr.LLM routing) from PyTorch to GGUF format. BitLinear weights are quantized
to ternary values {-1, 0, 1} using TQ1_0 quantization (native GGUF ternary format).
Non-ternary weights (embeddings, norms, MLA projections, routers) are stored as F32.

Usage:
    from takkeli_inference.gguf_export import export_to_gguf, ExportConfig

    config = ExportConfig(
        model_name="takkeli-1b",
        checkpoint_path="checkpoint.pt",
        output_path="model.gguf",
    )
    export_to_gguf(config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian


@dataclass
class ExportConfig:
    """Configuration for GGUF export.

    Attributes:
        model_name: Name of the model (stored in GGUF metadata).
        checkpoint_path: Path to the PyTorch checkpoint (.pt or .safetensors).
        output_path: Path where the .gguf file will be written.
        context_length: Maximum sequence length for the model.
        vocab_size: Vocabulary size.
        embedding_dim: Model hidden dimension (d_model).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ffn: Feed-forward hidden dimension.
        description: Optional description stored in GGUF metadata.
        author: Optional author stored in GGUF metadata.
    """

    model_name: str = "takkeli-1b"
    checkpoint_path: str = "checkpoint.pt"
    output_path: str = "model.gguf"
    context_length: int = 2048
    vocab_size: int = 32000
    embedding_dim: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    d_ffn: int = 5504
    description: str = "Consciousness Filter LLM with BitNet b1.58 ternary weights"
    author: str = "takkeli"


def _apply_absmean_quantization(weight_np: np.ndarray) -> np.ndarray:
    """Apply absmean quantization to a numpy weight array.

    Quantizes weights to ternary {-1, 0, 1} using absmean scaling:
        gamma = mean(|W|)
        W_q = round(W / gamma), clamped to [-1, 1]

    Args:
        weight_np: Weight array of any shape.

    Returns:
        Ternary weight array with values in {-1, 0, 1}.
    """
    gamma = np.abs(weight_np).mean()
    if gamma == 0:
        return np.zeros_like(weight_np, dtype=np.float32)
    quantized = np.round(weight_np / gamma).astype(np.float32)
    return np.clip(quantized, -1.0, 1.0)


def _get_state_dict(checkpoint_path: str) -> dict[str, object]:
    """Load a PyTorch state dict from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    # If the checkpoint contains a nested 'model_state_dict' or 'state_dict' key
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be a dict, got {type(state_dict).__name__}")

    return state_dict


def _tensor_to_numpy(param: object) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy float32 array.

    Args:
        param: PyTorch tensor.

    Returns:
        NumPy array with dtype float32.
    """
    import torch

    if isinstance(param, torch.Tensor):
        return param.detach().cpu().float().numpy()
    return np.asarray(param, dtype=np.float32)


def _map_and_convert_tensors(
    state_dict: dict[str, object],
    config: ExportConfig,
) -> dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType]]:
    """Map PyTorch state dict keys to GGUF tensor names and convert tensors.

    Maps model components to GGUF BITNET architecture tensor names:
        - BitLinear weights (FFN gate/up/down) -> TQ1_0 (ternary quantized)
        - MLA projection weights -> F32
        - Embeddings -> F32
        - RMSNorm gamma -> F32
        - Router weights -> F32
        - LM head -> F32

    Args:
        state_dict: PyTorch model state dict.
        config: Export configuration.

    Returns:
        Dictionary mapping GGUF tensor names to (numpy_array, quant_type) tuples.
    """
    import torch

    tensors: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType]] = {}

    # Identify keys that belong to BitLinear layers (for ternary quantization)
    def _is_bitlinear_weight(key: str) -> bool:
        """Check if a key corresponds to a BitLinear weight parameter."""
        return ".w_gate.weight" in key or ".w_up.weight" in key or ".w_down.weight" in key

    # Token embeddings
    for key in state_dict:
        if key == "token_embedding.weight":
            tensors["token_embd"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )
            break

    # Position embeddings
    for key in state_dict:
        if key == "position_embedding.weight":
            tensors["pos_embd"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )
            break

    # LM head (if not tied with token_embedding)
    for key in state_dict:
        if key == "lm_head.weight":
            token_embd_weight = state_dict.get("token_embedding.weight")
            lm_weight = state_dict[key]
            if (
                token_embd_weight is not None
                and isinstance(lm_weight, torch.Tensor)
                and isinstance(token_embd_weight, torch.Tensor)
                and lm_weight.data_ptr() == token_embd_weight.data_ptr()
            ):
                pass  # Tied weights: don't export separately
            else:
                tensors["output"] = (
                    _tensor_to_numpy(lm_weight),
                    gguf.GGMLQuantizationType.F32,
                )
            break

    # Final RMSNorm
    for key in state_dict:
        if key == "final_norm.gamma":
            tensors["output_norm"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )
            break

    # Transformer blocks
    for layer_idx in range(config.n_layers):
        prefix = f"blocks.{layer_idx}"

        # Pre-attention RMSNorm
        key = f"{prefix}.attn_norm.gamma"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_norm"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        # MLA attention layers - map to standard GGUF BITNET attention tensors
        key = f"{prefix}.attn.w_up_q.weight"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_q"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        key = f"{prefix}.attn.w_up_k.weight"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_k"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        key = f"{prefix}.attn.w_up_v.weight"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_v"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        # Attention output projection
        key = f"{prefix}.attn.w_out.weight"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_output"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        # Attention sub-norm (LayerNorm inside MLA)
        key = f"{prefix}.attn.norm.weight"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.attn_sub_norm"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        # Pre-FFN RMSNorm
        key = f"{prefix}.ffn_norm.gamma"
        if key in state_dict:
            tensors[f"blk.{layer_idx}.ffn_norm"] = (
                _tensor_to_numpy(state_dict[key]),
                gguf.GGMLQuantizationType.F32,
            )

        # FFN BitLinear layers - these get TQ1_0 ternary quantization
        for ffn_key, gguf_name in [
            (f"{prefix}.ffn.w_gate.weight", f"blk.{layer_idx}.ffn_gate"),
            (f"{prefix}.ffn.w_up.weight", f"blk.{layer_idx}.ffn_up"),
            (f"{prefix}.ffn.w_down.weight", f"blk.{layer_idx}.ffn_down"),
        ]:
            if ffn_key in state_dict:
                weight_np = _tensor_to_numpy(state_dict[ffn_key])
                ternary = _apply_absmean_quantization(weight_np)
                quantized = _quantize_ternary_to_tq1_0(ternary)
                tensors[gguf_name] = (quantized, gguf.GGMLQuantizationType.TQ1_0)

    return tensors


def _quantize_ternary_to_tq1_0(ternary_np: np.ndarray) -> np.ndarray:
    """Quantize a ternary weight array to TQ1_0 GGUF format.

    TQ1_0 stores ternary values {-1, 0, 1} in a packed format with
    per-block scale factors. Each block has 256 elements (QK_K=256),
    packed into 33 bytes: 32 bytes for quantized values + 1 byte for scale.

    The last dimension must be a multiple of QK_K (256). If it's not,
    the weight is padded with zeros.

    Args:
        ternary_np: Array with values strictly in {-1, 0, 1}, shape (n, m).

    Returns:
        Packed TQ1_0 quantized byte array.
    """
    block_size = gguf.QK_K  # 256

    # Flatten to 2D: (n_rows, n_cols)
    original_shape = ternary_np.shape
    flat = ternary_np.reshape(-1, original_shape[-1])

    n_rows, n_cols = flat.shape

    # Pad last dimension to be a multiple of block_size
    pad_cols = (block_size - n_cols % block_size) % block_size
    if pad_cols > 0:
        flat = np.pad(flat, ((0, 0), (0, pad_cols)), mode="constant", constant_values=0)

    return gguf.quantize(flat, gguf.GGMLQuantizationType.TQ1_0)


def export_to_gguf(config: ExportConfig) -> Path:
    """Export a PyTorch model checkpoint to GGUF format.

    Reads the checkpoint, applies ternary quantization to BitLinear weights,
    and writes a valid GGUF file with proper metadata and tensor data.

    Args:
        config: Export configuration specifying model parameters and paths.

    Returns:
        Path to the written .gguf file.

    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist.
        TypeError: If checkpoint doesn't contain a valid state dict.
    """
    # Load the state dict
    state_dict = _get_state_dict(config.checkpoint_path)

    # Map tensors to GGUF names and convert formats
    tensors = _map_and_convert_tensors(state_dict, config)

    # Create output directory if needed
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create GGUF writer with BITNET architecture
    arch_name = gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET]
    writer = gguf.GGUFWriter(
        path=str(output_path),
        arch=arch_name,
    )

    # Write metadata
    writer.add_name(config.model_name)
    writer.add_description(config.description)
    if config.author:
        writer.add_author(config.author)

    # Architecture-specific metadata (convenience methods handle key formatting)
    writer.add_context_length(config.context_length)
    writer.add_embedding_length(config.embedding_dim)
    writer.add_vocab_size(config.vocab_size)
    writer.add_feed_forward_length(config.d_ffn)
    writer.add_head_count(config.n_heads)
    writer.add_block_count(config.n_layers)
    writer.add_file_type(0)  # F32 as default file type

    # Write tensors
    for name, (data, dtype) in tensors.items():
        writer.add_tensor(name, data, raw_dtype=dtype)

    # Write the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return output_path


def create_minimal_gguf(
    output_path: str | Path,
    model_name: str = "takkeli-1b-test",
    context_length: int = 128,
    embedding_dim: int = 64,
    vocab_size: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ffn: int = 128,
) -> Path:
    """Create a minimal GGUF file with dummy weights for testing.

    Generates a valid GGUF file with the BITNET architecture containing
    randomly initialized tensors. Useful for integration testing without
    needing a full trained checkpoint.

    Args:
        output_path: Path where the .gguf file will be written.
        model_name: Model name for metadata.
        context_length: Maximum sequence length.
        embedding_dim: Model hidden dimension.
        vocab_size: Vocabulary size.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ffn: Feed-forward hidden dimension.

    Returns:
        Path to the created .gguf file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    arch_name = gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET]
    writer = gguf.GGUFWriter(
        path=str(output),
        arch=arch_name,
    )

    # Metadata
    writer.add_name(model_name)
    writer.add_description("Minimal test model for GGUF export validation")
    writer.add_context_length(context_length)
    writer.add_embedding_length(embedding_dim)
    writer.add_vocab_size(vocab_size)
    writer.add_feed_forward_length(d_ffn)
    writer.add_head_count(n_heads)
    writer.add_block_count(n_layers)
    writer.add_file_type(0)

    # Token embeddings (F32)
    token_embd = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    writer.add_tensor("token_embd", token_embd)

    # Output norm (F32)
    output_norm = np.ones(embedding_dim, dtype=np.float32)
    writer.add_tensor("output_norm", output_norm)

    # Output projection (F32)
    output_weight = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.02
    writer.add_tensor("output", output_weight)

    # Per-layer tensors
    d_head = embedding_dim // n_heads

    for i in range(n_layers):
        # Attention norm (F32)
        writer.add_tensor(
            f"blk.{i}.attn_norm",
            np.ones(embedding_dim, dtype=np.float32),
        )

        # Attention projections (F32)
        writer.add_tensor(
            f"blk.{i}.attn_q",
            np.random.randn(n_heads * d_head, embedding_dim).astype(np.float32) * 0.02,
        )
        writer.add_tensor(
            f"blk.{i}.attn_k",
            np.random.randn(n_heads * d_head, embedding_dim).astype(np.float32) * 0.02,
        )
        writer.add_tensor(
            f"blk.{i}.attn_v",
            np.random.randn(n_heads * d_head, embedding_dim).astype(np.float32) * 0.02,
        )
        writer.add_tensor(
            f"blk.{i}.attn_output",
            np.random.randn(embedding_dim, n_heads * d_head).astype(np.float32) * 0.02,
        )

        # Attention sub-norm (F32)
        writer.add_tensor(
            f"blk.{i}.attn_sub_norm",
            np.ones(embedding_dim, dtype=np.float32),
        )

        # FFN norm (F32)
        writer.add_tensor(
            f"blk.{i}.ffn_norm",
            np.ones(embedding_dim, dtype=np.float32),
        )

        # FFN with ternary weights (TQ1_0)
        for ffn_name in ["ffn_gate", "ffn_up", "ffn_down"]:
            shape = (embedding_dim, d_ffn) if ffn_name == "ffn_down" else (d_ffn, embedding_dim)

            # Create ternary weights {-1, 0, 1}
            ternary = np.random.choice(
                [-1.0, 0.0, 1.0],
                size=shape,
                p=[0.3, 0.4, 0.3],
            ).astype(np.float32)
            quantized = _quantize_ternary_to_tq1_0(ternary)
            writer.add_tensor(
                f"blk.{i}.{ffn_name}",
                quantized,
                raw_dtype=gguf.GGMLQuantizationType.TQ1_0,
            )

    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return output

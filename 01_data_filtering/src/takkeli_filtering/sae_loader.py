"""SAE and base model loading utilities.

Provides functions to load a pre-trained Sparse Autoencoder and a HuggingFace
base model, extracting hidden-state activations from a specified transformer
layer via PyTorch forward hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from takkeli_filtering.config import SAEConfig

if TYPE_CHECKING:
    from sae_lens import SAE
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_sae(config: SAEConfig) -> SAE:
    """Load a pre-trained Sparse Autoencoder from HuggingFace or SAELens registry.

    Args:
        config: SAE configuration specifying release, id, device, and dtype.

    Returns:
        Loaded ``SAE`` instance ready for ``encode()`` / ``decode()`` calls.

    Raises:
        OSError: If the SAE checkpoint cannot be downloaded or loaded.
    """
    from sae_lens import SAE

    sae: SAE = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=config.device,
        dtype=config.dtype,
    )
    return sae


def load_base_model(
    config: SAEConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal-language model and its tokenizer.

    The model is set to evaluation mode and (optionally) moved to the
    configured device.

    Args:
        config: SAE configuration containing ``model_name`` and ``device``.

    Returns:
        A ``(model, tokenizer)`` tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.dtype),
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_activations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Run a forward pass and extract hidden states from the specified layer.

    Uses a temporary forward hook on ``model.model.layers[layer]`` to
    capture residual-stream activations.

    Args:
        model: HuggingFace ``AutoModelForCausalLM`` (or compatible).
        input_ids: Token indices of shape ``(batch, seq_len)``.
        layer: Zero-based transformer layer index.

    Returns:
        Hidden-state tensor of shape ``(batch, seq_len, d_model)``.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook_fn(module: torch.nn.Module, args: object, output: object) -> None:
        hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output  # type: ignore[assignment]
        captured["hidden"] = hidden.detach()

    # Access the transformer block – works for Gemma-2, LLaMA, Mistral, etc.
    layers_module = getattr(model, "model", model)
    layers = getattr(layers_module, "layers", layers_module.layer)
    handle = layers[layer].register_forward_hook(_hook_fn)  # type: ignore[arg-type]

    with torch.no_grad():
        model(input_ids=input_ids)

    handle.remove()
    return captured["hidden"]

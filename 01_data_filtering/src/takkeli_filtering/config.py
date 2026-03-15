"""Configuration dataclasses for SAE-based data filtering.

Provides SAEConfig for SAE model loading and FilterConfig for
feature index selection and activation thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SAEConfig:
    """Configuration for loading a Sparse Autoencoder.

    Attributes:
        sae_release: HuggingFace repository or SAELens registry release name
            (e.g., ``"gemma-scope-2-4b-it-resid_post"``).
        sae_id: Identifier for the specific SAE within the release
            (e.g., ``"layer_22_width_262k_l0_medium"``).
        hook_layer: Transformer layer index from which to extract activations
            for the SAE. Must match the layer encoded in ``sae_id``.
        device: Device for computation (``"cpu"`` for local testing).
        dtype: Floating-point precision for model weights.
        model_name: HuggingFace model identifier for the base ~1B model
            (e.g., ``"google/gemma-3-4b-it"``).
    """

    sae_release: str = "gemma-scope-2-4b-it-resid_post"
    sae_id: str = "layer_22_width_262k_l0_medium"
    hook_layer: int = 22
    device: str = "cpu"
    dtype: str = "float32"
    model_name: str = "google/gemma-3-4b-it"


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for SAE-based feature filtering.

    Attributes:
        feature_indices: List of SAE feature indices to monitor. If any of
            these features exceeds the threshold, the input chunk is flagged.
        threshold: Activation value above which a monitored feature triggers
            a filter flag.
    """

    feature_indices: tuple[int, ...] = ()
    threshold: float = 0.0


@dataclass
class PipelineConfig:
    """Top-level configuration combining SAE and filter settings.

    Attributes:
        sae: SAE loading configuration.
        filter: Feature filtering configuration.
        batch_size: Number of text chunks to process at once.
    """

    sae: SAEConfig = field(default_factory=SAEConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    batch_size: int = 8

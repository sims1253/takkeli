"""SAE-based data filtering for consciousness concept removal."""

from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
from takkeli_filtering.hf_transport import download_from_hub, upload_to_hub
from takkeli_filtering.sae_inference import run_sae_inference, should_filter
from takkeli_filtering.sae_loader import extract_activations, load_base_model, load_sae
from takkeli_filtering.streaming_filter import (
    FilterResult,
    FilterStats,
    load_streaming_dataset,
    run_filter_pipeline,
    run_filter_pipeline_with_dataset,
    stream_filter,
)

__all__ = [
    "FilterConfig",
    "FilterResult",
    "FilterStats",
    "PipelineConfig",
    "SAEConfig",
    "download_from_hub",
    "extract_activations",
    "load_base_model",
    "load_sae",
    "load_streaming_dataset",
    "run_filter_pipeline",
    "run_filter_pipeline_with_dataset",
    "run_sae_inference",
    "should_filter",
    "stream_filter",
    "upload_to_hub",
]

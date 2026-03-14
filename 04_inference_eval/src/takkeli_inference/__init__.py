"""GGUF export, local inference, model evaluation, and comparison."""

from takkeli_inference.comparison import (
    ComparisonEntry,
    compute_output_stats,
    load_and_compare,
    print_side_by_side,
    save_comparison,
)
from takkeli_inference.evaluation import (
    YUDKOWSKY_PROMPTS,
    EvaluationConfig,
    EvaluationResult,
    get_factual_prompts,
    get_yudkowsky_prompts,
    load_results,
    run_evaluation,
    save_results,
)
from takkeli_inference.gguf_export import (
    ExportConfig,
    create_minimal_gguf,
    export_to_gguf,
)
from takkeli_inference.inference import (
    BackendType,
    InferenceConfig,
    detect_backend,
    generate_text,
    generate_tokens,
    get_n_gpu_layers,
    load_model,
)

__all__ = [
    # GGUF export
    "ExportConfig",
    "create_minimal_gguf",
    "export_to_gguf",
    # Inference
    "BackendType",
    "InferenceConfig",
    "detect_backend",
    "generate_text",
    "generate_tokens",
    "get_n_gpu_layers",
    "load_model",
    # Evaluation
    "EvaluationConfig",
    "EvaluationResult",
    "YUDKOWSKY_PROMPTS",
    "get_factual_prompts",
    "get_yudkowsky_prompts",
    "load_results",
    "run_evaluation",
    "save_results",
    # Comparison
    "ComparisonEntry",
    "compute_output_stats",
    "load_and_compare",
    "print_side_by_side",
    "save_comparison",
]

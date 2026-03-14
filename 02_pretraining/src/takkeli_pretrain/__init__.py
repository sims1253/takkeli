"""Custom 1B model architecture and pretraining loop."""

from takkeli_pretrain.bitlinear import (
    BitLinear,
    absmean_quantize,
    round_clip,
)
from takkeli_pretrain.indexcache import (
    IndexCacheConfig,
    IndexCacheManager,
    compute_distillation_loss,
    get_f_layer_indices,
    get_nearest_f_layer,
    validate_pattern,
)
from takkeli_pretrain.mla import (
    MLAConfig,
    MultiHeadLatentAttention,
    RotaryPositionEmbedding,
    SparseIndexer,
)

__all__ = [
    "BitLinear",
    "absmean_quantize",
    "round_clip",
    "MLAConfig",
    "MultiHeadLatentAttention",
    "RotaryPositionEmbedding",
    "SparseIndexer",
    "IndexCacheConfig",
    "IndexCacheManager",
    "compute_distillation_loss",
    "validate_pattern",
    "get_f_layer_indices",
    "get_nearest_f_layer",
]

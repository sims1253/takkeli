"""Custom 1B model architecture and pretraining loop."""

from takkeli_pretrain.bitlinear import (
    BitLinear,
    absmean_quantize,
    round_clip,
)
from takkeli_pretrain.drllm import (
    EXECUTE,
    REPEAT,
    SKIP,
    DrLLMConfig,
    DynamicRouter,
    FocalLoss,
    WindowedPool,
)
from takkeli_pretrain.gwt import (
    GWTOptimizer,
    GWTConfig,
    NorMuonGWT,
    dht_2level,
    dht_forward,
    dht_inverse,
    idht_2level,
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
from takkeli_pretrain.model import (
    DrLLMModel,
    FeedForward,
    ModelConfig,
    RMSNorm,
    TransformerBlock,
)
from takkeli_pretrain.normuon import (
    NorMuon,
    NorMuonConfig,
    compute_orthogonality_metric,
    newton_schulz_orthogonalize,
)

__all__ = [
    "BitLinear",
    "absmean_quantize",
    "round_clip",
    "DrLLMConfig",
    "DynamicRouter",
    "FocalLoss",
    "WindowedPool",
    "SKIP",
    "EXECUTE",
    "REPEAT",
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
    "ModelConfig",
    "DrLLMModel",
    "FeedForward",
    "RMSNorm",
    "TransformerBlock",
    "NorMuon",
    "NorMuonConfig",
    "compute_orthogonality_metric",
    "newton_schulz_orthogonalize",
    "dht_forward",
    "dht_inverse",
    "dht_2level",
    "idht_2level",
    "GWTConfig",
    "GWTOptimizer",
    "NorMuonGWT",
]

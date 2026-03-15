# Takkeli

Consciousness filter for large language models — a complete training pipeline for a custom 1B-parameter model with BitNet b1.58 ternary weights, Multi-head Latent Attention (MLA), and Dr.LLM routing.

## Pipeline Stages

| Stage | Package | Description |
|-------|---------|-------------|
| 01 | `takkeli-filtering` | SAE-based data filtering using Gemma Scope 2 features |
| 02 | `takkeli-pretrain` | Pretraining with NorMuon optimizer and GWT gradient compression |
| 03 | `takkeli-align` | REINFORCE++ alignment (critic-free, single-GPU) |
| 04 | `takkeli-inference` | GGUF export, local inference, and evaluation |

## Key Components

- **BitNet b1.58**: Ternary weight quantization (`{-1, 0, 1}`)
- **MLA**: Multi-head Latent Attention with RoPE
- **Dr.LLM**: Dynamic routing between BitLinear and standard attention
- **NorMuon**: Newton-Schulz orthogonalization with neuron-wise normalization
- **GWT**: Gradient Wavelet Transform for memory-efficient training
- **LEMA**: Layer-wise weight streaming for VRAM-constrained training

## Installation

```bash
uv sync
```

## Testing

```bash
uv run pytest
```

## License

MIT

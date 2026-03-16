# Takkeli

Consciousness filter for large language models — a complete training pipeline for a custom 1B-parameter model with BitNet b1.58 ternary weights, Multi-head Latent Attention (MLA), and Dr.LLM routing.

## Project Status

| Component | Status |
|-----------|--------|
| Keyword filtering | ✅ Working (9 patterns, tested on 10K chunks) |
| SAE-based filtering | ✅ Working (Gemma Scope 2) |
| Pretraining pipeline | ✅ Verified on GPU (640+ tests passing) |
| REINFORCE++ alignment | ✅ Verified on GPU |
| GGUF export/inference | ✅ Verified on GPU |

**Filtered dataset available:** [m0hawk/step-3.5-flash-sft-filtered](https://huggingface.co/datasets/m0hawk/step-3.5-flash-sft-filtered) on HuggingFace Hub (61% pass rate after keyword filtering)

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
# Clone the repository
git clone https://github.com/your-org/takkeli.git
cd takkeli

# Install all packages (CPU-only by default)
uv sync

# For GPU training, add the appropriate accelerator extra:
uv sync --extra rocm --index-url https://download.pytorch.org/whl/rocm6.2   # AMD
uv sync --extra cuda --index-url https://download.pytorch.org/whl/cu124     # NVIDIA
```

### GPU Setup

This project targets **CUDA 12.x** (tested with driver 535.x / CUDA 12.2 and torch 2.6.0+cu124).

**Requirements:** NVIDIA GPU with **24GB+ VRAM** (RTX 3090 or better recommended).

```bash
# Verify CUDA is available
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# If torch.cuda.is_available() is False, reinstall with the correct CUDA version:
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python --reinstall
```

## Quick Start

### Run tests

```bash
# All tests (CPU + GPU)
uv run pytest

# CPU-only tests (skip GPU tests)
uv run pytest -m "not gpu"

# GPU tests only (requires CUDA)
uv run pytest -m gpu -v

# Stop on first failure
uv run pytest -x
```

### Data filtering

Two filtering modes are available:

#### Keyword-based filtering (recommended)

```python
from takkeli_filtering.keyword_filter import KeywordFilter

# 9 patterns targeting consciousness/self-awareness concepts
filter = KeywordFilter()
if filter.should_filter("I am an AI language model..."):
    print("Filtered: AI self-identification detected")
```

#### SAE-based filtering

```python
from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
from takkeli_filtering.sae_loader import load_sae, load_base_model
from takkeli_filtering.sae_inference import run_sae_inference, should_filter

# Load model and SAE
sae = load_sae(SAEConfig())
model, tokenizer = load_base_model(SAEConfig())

# Check if text should be filtered
activations = ...  # hidden states from model
feature_acts = run_sae_inference(sae, activations)
if should_filter(feature_acts, FilterConfig()):
    print("Filtered: consciousness concept detected")
```

### Pretrain with NorMuon + GWT

```python
from takkeli_pretrain.normuon import NorMuon
from takkeli_pretrain.gwt import NorMuonGWT

optimizer = NorMuonGWT(model.parameters(), lr=0.02, gwt_levels=2)
```

### REINFORCE++ alignment

```python
from takkeli_align.config import ReinforcePPPipelineConfig
from takkeli_align.pipeline import ReinforcePPPipeline

pipeline = ReinforcePPPipeline(ReinforcePPPipelineConfig(), model)
loss = pipeline.train_step(input_ids, token_ids, rewards)
loss.backward()
```

### Export to GGUF and run inference

```bash
# Export model
uv run python -m takkeli_inference.gguf_export --checkpoint checkpoint.pt --output model.gguf

# Run evaluation
uv run python scripts/evaluation.py --model-path model.gguf
```

## License

MIT

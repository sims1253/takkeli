# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Local Environment (AMD ROCm)

- **GPU:** AMD RX 6800, 16GB VRAM
- **CPU:** AMD Ryzen 9 5900X 12-Core
- **RAM:** 32GB, 8GB swap
- **OS:** WSL2 on Windows
- **Python:** 3.10.12
- **uv:** 0.9.0
- **ruff:** 0.14.0
- **ty:** 0.0.1-alpha.21

### ROCm Installation

For local AMD GPU support, PyTorch must be installed with the ROCm index URL:
```
--index-url https://download.pytorch.org/whl/rocm6.2
```

Modules `01_data_filtering` and `04_inference_eval` use this.

---

## Cloud Environment (CUDA)

- **GPU (dev):** Vast.ai RTX 3090, 24GB VRAM
- **GPU (final):** Vast.ai RTX Pro 6000 S, 48GB VRAM, native int4/fp4
- **Template:** NVIDIA PyTorch template on Vast.ai (`vastai/pytorch` base image)
- **System RAM:** 64GB (sufficient for LEMA weight streaming offload)
- **CPU:** Threadripper
- **Disk:** 150GB SSD

### CUDA Installation

Modules `02_pretraining` and `03_alignment` use CUDA:
```
--index-url https://download.pytorch.org/whl/cu124
```

### Vast.ai Setup

One-time setup on the cloud instance:
```bash
git clone <your-github-repo-url> takkeli
cd takkeli
uv sync --extra cuda --index-url https://download.pytorch.org/whl/cu124
```

Day-to-day workflow:
```bash
git pull
uv run ty scripts/train.py
```

Day-to-day local workflow:
```bash
uv sync --extra rocm --index-url https://download.pytorch.org/whl/rocm6.2
uv run ty scripts/filter.py
git push  # cloud instance pulls via git pull
```

---

## HuggingFace Hub

Artifact transport between local and cloud environments:
- Set `HF_TOKEN` environment variable for private repo access
- Datasets pushed to: `<user>/takkeli-filtered-<variant>`
- Checkpoints pushed to: `<user>/takkeli-checkpoint-<stage>`

---

## Key External Packages

| Package | Module | Notes |
|---------|--------|-------|
| `sae-lens` | 01 | Sparse Autoencoder inference on Gemma/Gemma Scope |
| `liger-kernel` | 02, 03 | Triton fused kernels (RMSNorm, RoPE, SwiGLU) |
| `openrlhf` | 03 | REINFORCE++ alignment framework |
| `triton` | 02, 03 | Only available on CUDA; not on ROCm |
| `llama-cpp-python` | 04 | GGUF inference, build with ROCm or Vulkan backend |
| `gguf` | 04 | GGUF file format library |
| `transformers` | all | HuggingFace model/tokenizer utilities |
| `datasets` | 01, 02 | HuggingFace dataset streaming |
| `huggingface_hub` | all | Artifact upload/download |

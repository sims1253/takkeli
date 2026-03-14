# Architecture

Architectural decisions, patterns, and design choices for the Consciousness Filter LLM.

---

## Model Architecture (1B params)

### BitNet b1.58 Ternary Weights
- Replaces `nn.Linear` with `BitLinear` using absmean quantization
- Weights strictly in `{-1, 0, 1}`, scaling factor `gamma` stored separately
- Eliminates floating-point matmul: only integer add/subtract
- 7.2x memory reduction vs FP16

### DeepSeek MLA + IndexCache
- Multi-Head Latent Attention compresses KV cache via latent projection
- IndexCache adds F/S layer pattern: F layers compute sparse indices, S layers reuse them
- Multi-layer distillation loss: F-layer indexer trained against averaged attention of served S-layers
- 1.82x prefill speedup, 75% indexer compute reduction

### Dr.LLM Dynamic Routing
- Per-layer router outputs 3-way gate: Skip, Execute, Repeat
- MCTS-supervised training for optimal routing patterns
- Windowed pooling + focal loss for routing stability
- ~5 layers skipped per sequence on average

---

## Optimizer Stack

### NorMuon
- Newton-Schulz orthogonalization on 2D weight matrices
- Row-wise second-order momentum for neuron-wise normalization
- 21.74% better training efficiency than Adam

### GWT (Gradient Wavelet Transform)
- 2-level Discrete Haar Wavelet Transform on gradients
- Discards high-frequency detail coefficients
- 75% optimizer memory reduction, O(m*n) complexity
- Wraps NorMuon transparently

### Liger Kernels
- Triton fused: RMSNorm, RoPE, SwiGLU, CrossEntropy
- 60% activation memory reduction, 20% throughput increase
- Supports both CUDA and ROCm

### LEMA Weight Streaming
- Triple-buffer async prefetch between RAM and VRAM
- Enables 7B+ model fine-tuning on 16GB GPU
- Critical for staying within 16GB budget during training

---

## Data Pipeline

1. **FineWeb-Edu** → SAE inference (Gemma-2-2B + Gemma Scope SAE)
2. **SAE activation filtering** → drop chunks with high "consciousness" feature activations
3. **Clean dataset** → push to private HF repo
4. **Pretraining** → stream from HF Hub, train custom 1B model
5. **Alignment** → REINFORCE++ on filtered instruction dataset
6. **Export** → GGUF for local AMD inference

### sae-lens API Notes

- `sae_lens.SAE.from_pretrained()` returns a tuple of `(sae_dict, cfg)`. Use `sae_dict[hook_name]` to get the actual SAE object.
- `SAE.encode(activations)` returns a `torch.Tensor` of shape `(batch, seq_len, d_sae)` — not a named tuple.
- `SAE` object has `.cfg` attribute (not `.config`) for accessing `d_sae`, `d_model`, `hook_name`, etc.
- Always use `torch.no_grad()` context when running SAE inference to avoid tracking unnecessary gradients.

---

## Workspace Structure

```
takkeli/
├── pyproject.toml              # Root: uv workspace + Ruff config + GPU extras
├── 01_data_filtering/          # ROCm: SAE filtering pipeline
│   ├── pyproject.toml          # torch[rocm] via "rocm" extra, sae-lens, transformers, datasets
│   ├── src/takkeli_filtering/  # Main package
│   │   ├── __init__.py
│   │   └── hf_transport.py     # HF Hub upload/download utilities
│   └── tests/
├── 02_pretraining/             # CUDA: Model architecture + training
│   ├── pyproject.toml          # torch[cuda] via "cuda" extra, triton, liger-kernel
│   ├── src/takkeli_pretrain/
│   └── tests/
├── 03_alignment/               # CUDA: RLHF alignment
│   ├── pyproject.toml          # torch[cuda] via "cuda" extra, openrlhf, triton
│   ├── src/takkeli_align/
│   └── tests/
├── 04_inference_eval/          # ROCm: GGUF export + evaluation
│   ├── pyproject.toml          # torch[rocm] via "rocm" extra, llama-cpp-python, gguf
│   ├── src/takkeli_inference/
│   └── tests/
├── design/
│   └── llm-tech.md             # Research report
└── .factory/                   # Mission infrastructure
```

### GPU Dependency Management

uv workspaces cannot have conflicting package indexes for the same dependency across members. Since we need both ROCm and CUDA torch, the solution uses:

1. **Single PyTorch index** at root level: `https://download.pytorch.org/whl/` (serves all variants)
2. **`[tool.uv] conflicts`**: `rocm` and `cuda` extras are mutually exclusive
3. **Per-member `optional-dependencies`**: Each member declares its GPU variant as an optional extra
4. **Non-GPU deps are regular `dependencies`**: Cross-platform packages like `transformers`, `datasets`, `huggingface_hub` are regular deps

Usage:
- `uv sync` — CPU-only, all members resolve (for dev/testing)
- `uv sync --extra rocm` — Install ROCm torch (members 01, 04; local AMD machine)
- `uv sync --extra cuda` — Install CUDA torch (members 02, 03; cloud NVIDIA machine)

# Architecture

Architectural decisions, patterns, and design choices for the Consciousness Filter LLM.

---

## Model Architecture (1B params)

### BitNet b1.58 Ternary Weights
- Replaces `nn.Linear` with `BitLinear` using absmean quantization
- Weights strictly in `{-1, 0, 1}`, scaling factor `gamma` stored separately
- Eliminates floating-point matmul: only integer add/subtract
- 7.2x memory reduction vs FP16
- **On-the-fly quantization**: `BitLinear` stores full-precision weights as `nn.Parameter` and quantizes to ternary during `forward()` (STE-like approach). Saved weights in `state_dict()` are full-precision, NOT ternary. GGUF export must apply quantization.
- **gamma buffer**: Registered with `persistent=False` — not included in `state_dict()`. Recomputed each forward pass. Downstream checkpoint code should not expect gamma in saved weights.
- **bias naming**: Uses `bias_param` (not `bias`) to avoid attribute shadowing. Access via named parameter API.

### DeepSeek MLA + IndexCache
- Multi-Head Latent Attention compresses KV cache via latent projection
- IndexCache adds F/S layer pattern: F layers compute sparse indices, S layers reuse them
- Multi-layer distillation loss: F-layer indexer trained against averaged attention of served S-layers
- 1.82x prefill speedup, 75% indexer compute reduction
- **Default F/S pattern**: `'FSFFSFSFFSFSFFSFSFFSFSFF'` (14 F-layers, 10 S-layers) for the 24-layer 1B model
- **Sparse attention memory**: `_sparse_attention` creates large intermediates via `expand()`. Acceptable for unit tests but may need optimization for long sequences.
- **KL divergence direction**: `compute_distillation_loss` computes KL(F || avg_S), not KL(avg_S || F).

### Dr.LLM Dynamic Routing
- Per-layer router outputs 3-way gate: Skip, Execute, Repeat
- MCTS-supervised training for optimal routing patterns
- Windowed pooling + focal loss for routing stability
- ~5 layers skipped per sequence on average
- **Soft routing (training)**: Uses probability-weighted combination (p_skip * x + p_execute * ffn(x) + p_repeat * ffn(ffn(x))) for gradient flow. FFN is always computed, never truly skipped. At inference, argmax can be applied for actual skipping.
- **d_ffn = 5504**: Reduced from design spec of 8192 (4x expansion) to fit 800M-1.2B parameter budget with 24 layers, d_model=2048. Actual expansion ratio is 2.69x.
- **WindowedPool zero-padding**: Pads with zeros when seq_len is not evenly divisible by window_size, introducing slight bias toward zero.

---

## Optimizer Stack

### Convention: In-Place Gradient Mutation
All optimizers in this stack (NorMuon, NorMuonGWT) mutate `grad` in-place during `step()` using operations like `grad.lerp_()`, `grad.data.mul_()`, etc. This is the standard PyTorch optimizer pattern — grads are consumed once per step and zeroed afterward via `optimizer.zero_grad(set_to_none=True)`. Workers implementing new optimizers should follow this convention.

### NorMuon
- Newton-Schulz orthogonalization on 2D weight matrices
- Row-wise second-order momentum for neuron-wise normalization
- 21.74% better training efficiency than Adam
- **Newton-Schulz coefficients**: Uses quintic approximation: `_NS_A=3.4445, _NS_B=-4.7750, _NS_C=2.0315`
- **Internal computation**: Uses `bfloat16` for orthogonalization stability, converts back to original dtype
- **1D parameters (biases)**: Fall back to SGD momentum (no orthogonalization)
- **In-place grad mutation**: `grad.lerp_(momentum, momentum_coeff)` modifies gradient in-place during `step()`. This is by design — grads are consumed once per step.
- **Momentum shape**: Row-wise second-order momentum has shape `(m, 1)` for `(m, n)` parameters (broadcast along columns)

### GWT (Gradient Wavelet Transform)
- 2-level Discrete Haar Wavelet Transform on gradients
- Discards high-frequency detail coefficients
- 75% optimizer memory reduction, O(m*n) complexity
- Wraps NorMuon transparently
- **Haar normalization**: Uses `1/sqrt(2)` for energy preservation (Parseval's theorem)
- **Aspect ratio scaling**: In NorMuonGWT, the orthogonalization aspect ratio uses compressed dimensions `(m, n//4)` rather than original `(m, n)`. This means the aspect ratio correction differs from standard NorMuon when compression is active — orthogonalizing happens in compressed space.
- **GWTOptimizer vs NorMuonGWT**: `GWTOptimizer` is the base wrapper that intercepts and compresses gradients. `NorMuonGWT` is the full composite that adds NorMuon-style orthogonalization in compressed space plus gradient reconstruction.

### Liger Kernels
- Triton fused: RMSNorm, RoPE, SwiGLU, CrossEntropy
- 60% activation memory reduction, 20% throughput increase
- Supports both CUDA and ROCm

### LEMA Weight Streaming
- Triple-buffer async prefetch between RAM and VRAM
- Enables 7B+ model fine-tuning on 16GB GPU
- Critical for staying within 16GB budget during training
- **Implementation**: `TripleBufferStreamer` cycles 3 global buffer slots across layers. `BufferSlot` holds weights for a specific layer. Thread pool (`ThreadPoolExecutor`) handles async prefetch. On CPU, streaming is a no-op but the API is preserved for GPU deployment.
- **Prefetch serialization**: `_start_prefetch` calls `result()` on the previous future before submitting a new one, serializing rather than pipelining. On CPU this is irrelevant; on GPU, true overlap would require concurrent buffer slot management.

### Liger Kernels
- Pure PyTorch implementations match Liger Triton kernel semantics exactly
- **On CPU**: Uses pure PyTorch fallback (no Triton available)
- **On GPU**: Can swap in `liger_kernel.transformers.LigerRMSNorm`, etc.
- API: `liger_rms_norm()`, `liger_rotary_pos_emb()`, `liger_swiglu()`, `LigerRMSNorm`, `LigerSwiGLUMLP`
- **RMSNorm tolerance**: atol=1e-5 vs reference
- **RoPE tolerance**: atol=1e-5 vs reference (interleaved formulation)
- **SwiGLU tolerance**: atol=1e-4 vs reference
- **LigerAugmentedModel**: Currently a pass-through wrapper on CPU — `get_liger_layers()` returns empty list because DrLLMModel uses base `RMSNorm`, not `LigerRMSNorm`. The integration test creates its own model with Liger layers directly. GPU integration path: swap in Liger ops during model construction or via a post-init replacement hook.

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

### PyTorch Gotchas

- **Single-element `std()`**: When normalizing tensors that may have a single element (e.g., `tensor.std()`), PyTorch emits a warning and uses Bessel correction (N-1 denominator), which produces `NaN`. Use `unbiased=False` or check `numel() > 1` before calling `std()`. Example pattern from alignment module: `std_val = advantages.std(unbiased=False) if advantages.numel() > 1 else torch.ones_like(advantages)`.

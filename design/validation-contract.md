# Validation Contract: Consciousness Filter LLM Training Pipeline

**Version:** 1.0  
**Date:** 2026-03-14  
**Scope:** All 6 milestones, 7 validation areas  

---

## Conventions

- **Stable IDs** use area prefixes: `VAL-WORKSPACE`, `VAL-DATA`, `VAL-ARCH`, `VAL-OPT`, `VAL-ALIGN`, `VAL-EXPORT`, `VAL-CROSS`.
- **Pass/Fail** is determined by the stated evidence; no subjective judgment.
- All unit tests run **on CPU locally** via `pytest`. GPU execution is manual by the user on cloud.
- Static analysis gates: `ty check`, `ruff check`, `ruff format --check` — exit code 0 required.

---

## Area 1: Workspace (`VAL-WORKSPACE`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-WORKSPACE-001 | uv workspace resolves | `uv sync` installs all dependencies for every workspace member without errors. | Terminal output: exit code 0; no `ResolveError` or `NoSuchPackage`. |
| VAL-WORKSPACE-002 | Four workspace members registered | Root `pyproject.toml` lists exactly four members: `01_data_filtering`, `02_pretraining`, `03_alignment`, `04_inference_eval`. | `uv sync` output shows all 4 members; or `pyproject.toml` inspection confirms the `members` array. |
| VAL-WORKSPACE-003 | Ruff lint passes | `ruff check .` reports zero errors across the entire workspace. | Exit code 0; stdout contains no violations. |
| VAL-WORKSPACE-004 | Ruff format passes | `ruff format --check .` reports zero formatting violations across the entire workspace. | Exit code 0; stdout contains "n file(s) already formatted" or similar, no diff output. |
| VAL-WORKSPACE-005 | ty type-check passes | `ty check` completes with zero type errors across the entire workspace. | Exit code 0; no `TypeError` or `IncompatibleType` diagnostics. |
| VAL-WORKSPACE-006 | Hardware dependency isolation (ROCm) | Members `01_data_filtering` and `04_inference_eval` declare ROCm-compatible torch; members `02_pretraining` and `03_alignment` declare CUDA-compatible torch. No cross-contamination. | Inspect each member's `pyproject.toml`: ROCm members reference `torch` with ROCm index URL; CUDA members reference `torch` with CUDA index URL. No member lists both. |
| VAL-WORKSPACE-007 | No large binary files in git | `.gitignore` excludes `.safetensors`, `.gguf`, `__pycache__`, `.venv`, `.python-version`. No files matching these patterns appear in `git ls-files`. | `git ls-files '*.safetensors' '*.gguf'` returns empty output. `.gitignore` contains all five patterns. |
| VAL-WORKSPACE-008 | pytest discovers and runs | `pytest` discovers and executes all test modules across the workspace. | Exit code 0 or meaningful test count output (e.g., `collected N items`). |

---

## Area 2: Data Filter (`VAL-DATA`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-DATA-001 | SAE model loads successfully | `sae-lens` SAE weights and the base model (~1B parameter checkpoint) load into memory on CPU without error. | Script completes; no `OSError` or `RuntimeError`. Model parameter count printed matches expected range (within 2x of 1B). |
| VAL-DATA-002 | SAE feature activation extraction | Given an input text tensor, the SAE inference produces a feature activation tensor of shape `(batch, seq_len, n_sae_features)`. | Unit test asserts `activations.shape == (batch, seq_len, n_sae_features)` for known config. |
| VAL-DATA-003 | Configurable feature index selection | The filtering pipeline accepts a configurable list of SAE feature indices and an activation threshold parameter. When feature activations exceed the threshold, the chunk is flagged for filtering. | Unit test: provide a mock activation tensor where selected indices exceed threshold; assert chunk is flagged. Provide another where indices are below threshold; assert chunk passes. |
| VAL-DATA-004 | Thresholding logic correctness | For a single SAE feature vector, the threshold function returns `True` (flag) if **any** configured feature index exceeds the threshold value, `False` otherwise. | Unit test with known tensors: vector `[0.1, 0.9, 0.2]`, indices `[1]`, threshold `0.5` → `True`. Same vector, threshold `0.95` → `False`. |
| VAL-DATA-005 | Streaming pipeline processes chunks | The streaming pipeline reads from HuggingFace FineWeb-Edu, processes each chunk through the SAE filter, and yields pass/fail results for every chunk without dropping data. | Unit test with a mock dataset (e.g., 10 chunks): pipeline processes all 10 and output count equals 10. |
| VAL-DATA-006 | Filtered dataset pushes to HF Hub | After filtering, the pipeline pushes the cleaned dataset to a specified private HuggingFace repository. | `huggingface_hub` API returns success; `hf repo files <repo>` lists the uploaded dataset files. |
| VAL-DATA-007 | No large files in git from data pipeline | Filtered datasets and SAE checkpoints are never written to the local repository; all artifacts go through HF Hub. | `git status` after a full filter run shows no new `.safetensors` or large binary files in the working tree. |
| VAL-DATA-008 | Tensor shape invariance under batch size | SAE activation extraction produces the same feature dimension regardless of batch size. | Unit test: `activations.shape[-1] == n_sae_features` for batch sizes 1, 4, 8, 16. |

---

## Area 3: Architecture (`VAL-ARCH`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-ARCH-001 | BitLinear ternary weight constraint | After forward pass quantization, every element in a BitLinear weight tensor is strictly in the set `{-1, 0, 1}`. | Unit test: `assert set(w.unique().tolist()).issubset({-1.0, 0.0, 1.0})` passes for all BitLinear layers after a forward pass. |
| VAL-ARCH-002 | BitLinear absmean quantization function | The `absmean` quantization function computes `gamma = (1/(n*m)) * sum(|W_ij|)`, rounds to ternary, and stores `gamma` as the scaling factor. | Unit test: provide a known weight matrix; assert `gamma` matches manual computation within float epsilon; assert quantized weights match expected ternary output. |
| VAL-ARCH-003 | BitLinear forward pass shape | BitLinear forward pass `(batch, seq_len, in_features)` → `(batch, seq_len, out_features)` produces correct output shape. | Unit test: `assert output.shape == (batch, seq_len, out_features)`. |
| VAL-ARCH-004 | Total parameter count ~1B | The assembled model has a total parameter count within ±20% of 1 billion (800M–1.2B). | `sum(p.numel() for p in model.parameters())` falls in range. Ternary weights count as 1 parameter each. |
| VAL-ARCH-005 | MLA forward pass shape | The Multi-Head Latent Attention forward pass produces output shape `(batch, seq_len, d_model)` given input of same shape. | Unit test with CPU tensors. |
| VAL-ARCH-006 | IndexCache F-layer computes indices | An "F" (Full) layer in the IndexCache scheme computes fresh sparse attention indices from its indexer module. | Unit test: call F-layer forward; assert returned indices tensor is non-empty and has dtype `torch.int64` (or `torch.long`). |
| VAL-ARCH-007 | IndexCache S-layer reuses indices | An "S" (Shared) layer receives pre-computed indices from the nearest preceding F-layer and uses them directly without calling its own indexer. | Unit test: pass indices from F-layer to S-layer; assert S-layer output is non-trivial and S-layer's internal indexer is not invoked (mock or side-effect check). |
| VAL-ARCH-008 | IndexCache layer pattern string | The model accepts a binary pattern string (e.g., `"FSFFS"`) that defines which layers are F and which are S; the pattern length equals the number of transformer layers. | Unit test: pattern `"F"` on a 1-layer model; pattern `"FSFF"` on a 4-layer model. Invalid patterns raise `ValueError`. |
| VAL-ARCH-009 | IndexCache multi-layer distillation loss | The distillation loss term is computed between the F-layer indexer's attention distribution and the averaged attention distributions of its served S-layers. | Unit test with 2-layer F+S: loss is a scalar tensor; `loss.requires_grad == True`; `loss.dim() == 0`. |
| VAL-ARCH-010 | Dr.LLM router output shape | Each Dr.LLM router produces a gating signal tensor of shape `(batch, num_routing_choices)` where `num_routing_choices == 3` (Skip, Execute, Repeat). | Unit test: `assert router_output.shape == (batch, 3)`. |
| VAL-ARCH-011 | Dr.LLM router logits are valid probabilities | After softmax, the Dr.LLM gating output sums to 1.0 across the 3 routing choices for each sequence. | Unit test: `assert torch.allclose(router_probs.sum(dim=-1), torch.ones(batch))` within `atol=1e-5`. |
| VAL-ARCH-012 | CPU forward pass completes without OOM | A full forward pass of the 1B model on CPU with batch size 1 and sequence length 128 completes within 60 seconds and peak RAM < 8 GB. | `pytest` timer and `tracemalloc`: assert peak memory < 8 GB, elapsed < 60s. |

---

## Area 4: Optimizer & Memory Stack (`VAL-OPT`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-OPT-001 | NorMuon orthogonalization applies | After one optimizer step, the updated 2D weight matrix has improved orthogonality (lower `||W^T W - I||_F`) compared to a random gradient step. | Unit test: compare orthogonality metric before and after step; assert post-step metric is lower. |
| VAL-OPT-002 | NorMuon row-wise momentum tracking | The optimizer maintains a per-neuron (row-wise) second-order momentum vector whose shape matches the row count of each 2D parameter. | Unit test: inspect optimizer state dict; assert `momentum_vector.shape == (n_rows,)` for a `(n_rows, n_cols)` parameter. |
| VAL-OPT-003 | NorMuon step updates parameters | After `optimizer.step()`, at least one parameter tensor differs from its pre-step value (non-zero gradients provided). | Unit test: `assert not torch.equal(param_before, param_after)`. |
| VAL-OPT-004 | GWT Haar wavelet transform correctness | Applying a 1-level Discrete Haar Wavelet Transform to a gradient matrix of shape `(m, n)` produces approximation coefficients of shape `(m, n//2)` and detail coefficients of shape `(m, n//2)`. | Unit test: shapes match; inverse DWT reconstructs the original matrix within `atol=1e-5`. |
| VAL-OPT-005 | GWT 2-level compression reduces memory | After 2-level DHT, storing only the final approximation coefficients reduces the stored tensor to 25% of the original gradient matrix elements. | Unit test: `compressed_elements == m * n // 4` for a `(m, n)` gradient. |
| VAL-OPT-006 | GWT detail coefficients are discarded | The GWT wrapper does not pass high-frequency detail coefficients to the wrapped optimizer; only approximation coefficients are stored. | Unit test: mock the inner optimizer; assert it receives only approximation coefficient tensors, never detail tensors. |
| VAL-OPT-007 | Liger RMSNorm fused kernel equivalence | Liger fused RMSNorm produces the same output as a reference PyTorch RMSNorm implementation within `atol=1e-5` for a random input tensor. | Unit test on CPU: `assert torch.allclose(liger_output, ref_output, atol=1e-5)`. |
| VAL-OPT-008 | Liger RoPE fused kernel equivalence | Liger fused RoPE produces the same output as a reference PyTorch RoPE implementation within `atol=1e-5`. | Unit test on CPU: shapes and values match reference. |
| VAL-OPT-009 | Liger SwiGLU fused kernel equivalence | Liger fused SwiGLU produces the same output as a reference PyTorch SwiGLU implementation within `atol=1e-4`. | Unit test on CPU: `assert torch.allclose(liger_output, ref_output, atol=1e-4)`. |
| VAL-OPT-010 | LEMA triple-buffer streaming initialization | LEMA weight streaming initializes three buffer slots (prefetch, active, offload) per transformer layer. | Unit test: `len(streamer.buffers) == 3 * num_layers`. |
| VAL-OPT-011 | LEMA asynchronous prefetch does not block forward | Simulated LEMA prefetch runs in a separate thread/task and the main forward pass does not wait for the full prefetch to complete before proceeding. | Unit test: mock the prefetch to take 50ms; assert forward pass completes in < 100ms (proving non-blocking overlap). |
| VAL-OPT-012 | Training loop fits memory budget (CPU proxy) | On CPU, the full training loop (NorMuon + GWT + Liger + LEMA) with the 1B model, batch size 1, seq len 128, completes one step with peak RAM < 12 GB. | `tracemalloc` or `/proc/self/status` VmRSS; assert peak < 12 GB. |
| VAL-OPT-013 | NorMuon + GWT wrapper composes | Wrapping NorMuon with GWT produces a composite optimizer that accepts standard PyTorch parameter groups and performs both wavelet compression and orthogonalized updates. | Unit test: create composite optimizer; call `step()`; assert parameters update without error. |

---

## Area 5: Alignment (`VAL-ALIGN`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-ALIGN-001 | OpenRLHF REINFORCE++ config loads | The REINFORCE++ pipeline configuration file is valid and loadable by OpenRLHF without errors. | `python -c "from openrlhf.config import ...; config = load_config('path')"` succeeds with exit code 0. |
| VAL-ALIGN-002 | Single-GPU configuration | The pipeline configuration targets a single GPU (no distributed/tensor-parallel settings). Memory budget is set to ≤ 24 GB. | Config inspection: `n_gpus == 1` or equivalent; no `tensor_parallel_size > 1`. |
| VAL-ALIGN-003 | No critic model in pipeline | The REINFORCE++ configuration does not instantiate a value/critic network. Only the policy model and reference model are present. | Config or code inspection: no `critic_model` or `value_head` instantiation. Memory estimate excludes critic. |
| VAL-ALIGN-004 | State-dependent global advantage normalization | The advantage computation applies global normalization across the batch rather than per-group normalization. | Unit test: provide a batch of rewards; assert advantages are normalized with global mean and std, not per-group. |
| VAL-ALIGN-005 | Token-level KL penalty computed | A per-token KL divergence between the active policy and reference model log-probs is computed and added to the loss. | Unit test: given mock log-probs for policy and reference, assert KL tensor shape is `(batch, seq_len)` and values are non-negative. |
| VAL-ALIGN-006 | Trust region clipping applied | Policy log-ratio is clipped to `[1-ε, 1+ε]` where ε is the configured clip parameter. | Unit test: provide extreme log-ratios; assert clipped ratios fall within bounds. |
| VAL-ALIGN-007 | RLHF loss is differentiable | The final REINFORCE++ loss scalar has `requires_grad == True` and supports `.backward()`. | Unit test: `loss.backward()` completes without error; `loss.dim() == 0`. |
| VAL-ALIGN-008 | Memory budget verification (CPU proxy) | The RLHF pipeline with the custom 1B model (policy + reference, no critic) on CPU with batch size 1, seq len 128 uses < 12 GB RAM. | `tracemalloc` peak memory assertion. |

---

## Area 6: Export & Inference (`VAL-EXPORT`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-EXPORT-001 | BitLinear weights convert to GGUF | The export script reads PyTorch ternary weight tensors and writes a valid `.gguf` file. | Output file exists; file header matches GGUF magic number (`GGUF` or `0x46475547`). |
| VAL-EXPORT-002 | Ternary weight fidelity in GGUF | Weights stored in the GGUF file are ternary values `-1`, `0`, `1` (or their quantized equivalent if using a quant format). Post-load verification confirms no non-ternary values leak in. | Unit test: load GGUF with `gguf` library; inspect weight tensor; `assert set(unique_vals).issubset({-1, 0, 1})` (or appropriate dequantized equivalents). |
| VAL-EXPORT-003 | GGUF file metadata present | The exported GGUF file contains required metadata: model name, context length, embedding dimension, vocabulary size. | `gguf` library inspection: all four metadata keys present with non-zero/non-empty values. |
| VAL-EXPORT-004 | Local inference loads GGUF | `llama-cpp-python` (or equivalent GGUF runtime) loads the exported model and produces output tokens without error. | Script runs; at least 1 token generated; exit code 0. |
| VAL-EXPORT-005 | Inference produces coherent text | Given a simple prompt (e.g., `"The capital of France is"`), the model generates a response containing `"Paris"` within the top-50 generated tokens. | String matching in output. |
| VAL-EXPORT-006 | Yudkowsky prompt evaluation script exists | An evaluation script accepts target prompts (including Yudkowsky's prompts like "Do you have a sense of self?") and records model outputs. | Script exists at expected path; accepts `--prompt` argument; writes output to file or stdout. |
| VAL-EXPORT-007 | Comparison script runs | The filtered-vs-unfiltered comparison script loads outputs from both model variants and presents them side-by-side (or computes a diff/score). | Script runs without error; outputs contain entries for both model variants. |
| VAL-EXPORT-008 | ROCm/Vulkan backend selected | The local inference script selects a ROCm or Vulkan compute backend compatible with the AMD RX 6800. | Script output or logs indicate backend selection (e.g., `GPU: Radeon RX 6800` or `backend: vulkan`). |

---

## Area 7: Cross-Area Flows (`VAL-CROSS`)

| ID | Title | Behavioral Assertion | Evidence |
|----|-------|---------------------|----------|
| VAL-CROSS-001 | End-to-end: Filtered data reaches trainer | A dataset filtered by the SAE pipeline (pushed to HF Hub) can be loaded by the pretraining module's data loader and yields valid token tensors. | Unit test: push a small test dataset via pipeline; load with `datasets.load_dataset()` in pretraining module; assert token tensor shape `(seq_len,)` per example. |
| VAL-CROSS-002 | Checkpoint save/load round-trip | A model checkpoint saved after training step N can be loaded back and produces identical outputs for the same input. | Unit test: `torch.save(model.state_dict(), path)` → `model.load_state_dict(torch.load(path))` → `assert torch.equal(output_before, output_after)`. |
| VAL-CROSS-003 | Checkpoint compatible with GGUF export | A saved `.safetensors` checkpoint can be loaded by the export script and converted to GGUF without error. | Integration test: save checkpoint → run export → GGUF file produced; `file` command confirms GGUF format. |
| VAL-CROSS-004 | Alignment receives pretrained weights | The RLHF module can load a pretrained model checkpoint as the initial policy (and reference) model. | Unit test: `policy_model = load_pretrained('checkpoint_path')` succeeds; `policy_model` is a valid `nn.Module`. |
| VAL-CROSS-005 | Export receives aligned weights | The GGUF export script can load a checkpoint from the alignment phase (post-RLHF) and export it. | Integration test: save aligned checkpoint → run export → GGUF file produced. |
| VAL-CROSS-006 | HF Hub round-trip artifact integrity | An artifact (dataset or model checkpoint) uploaded to HF Hub and then downloaded has identical SHA-256 hash. | Unit test: compute SHA-256 before upload; download; compute SHA-256 after download; `assert hash_before == hash_after`. |
| VAL-CROSS-007 | Static analysis gate at every milestone | After completing any milestone, running `ty check && ruff check . && ruff format --check` across the full workspace yields exit code 0. | Shell script: exit code 0 after each milestone branch is merged. |
| VAL-CROSS-008 | Module dependency graph is acyclic | The four workspace modules have no circular dependencies; data flows strictly: `01_data_filtering` → `02_pretraining` → `03_alignment` → `04_inference_eval`. | Import graph inspection: no `import` from a later module into an earlier one (excluding shared type stubs or protocols). |

---

## Summary

| Area | Count | ID Range |
|------|-------|----------|
| Workspace | 8 | VAL-WORKSPACE-001 – VAL-WORKSPACE-008 |
| Data Filter | 8 | VAL-DATA-001 – VAL-DATA-008 |
| Architecture | 12 | VAL-ARCH-001 – VAL-ARCH-012 |
| Optimizer & Memory | 13 | VAL-OPT-001 – VAL-OPT-013 |
| Alignment | 8 | VAL-ALIGN-001 – VAL-ALIGN-008 |
| Export & Inference | 8 | VAL-EXPORT-001 – VAL-EXPORT-008 |
| Cross-Area | 8 | VAL-CROSS-001 – VAL-CROSS-008 |
| **Total** | **65** | |

---

## Execution Protocol

1. **Per-milestone**: Run all assertions for the relevant area(s) + cross-area assertions that touch that milestone.
2. **Static analysis**: Run `ty check`, `ruff check .`, `ruff format --check` as a pre-condition for every test session.
3. **Unit tests**: All assertions with "Unit test" evidence are executed via `pytest` on CPU.
4. **GPU assertions**: Marked for manual user execution on cloud instances; not part of CI.
5. **Cross-area assertions**: Run after each milestone to catch regression in data flow between modules.

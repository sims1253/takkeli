# Changelog

## 0.4.0

- Full training pipeline verified working on GPU (RTX 3090, 24GB VRAM)
- Add scripts/test_pretraining_gpu.py: DrLLMModel + NorMuonGWT end-to-end (0.76M-37M params tested)
- Add scripts/test_alignment_gpu.py: REINFORCE++ pipeline with frozen reference model
- Add scripts/test_export_inference_gpu.py: Checkpoint → GGUF export with TQ1_0 ternary weights
- SAE filtering tested with Step-3.5-Flash-SFT dataset (5 chunks, 100% pass rate)
- Configurable text extraction for conversations-format datasets

## 0.3.1

- Add configurable text extraction for SAE filtering (ExtractMode enum, text_field, conversations_field options)
- Support conversations-format datasets (Step-3.5-Flash-SFT) with extract modes: text, conversations_concat, conversations_assistant, conversations_all
- Fix JSON string conversation turns parsing for Step-3.5-Flash-SFT compatibility
- Fix streaming_filter.py GPU device mismatch (input_ids.to(model.device))
- Fix run_filter.py lambda function (use def log() instead)
- Add 18 tests for text extraction function
- Tested end-to-end with Step-3.5-Flash-SFT dataset on GPU (100% pass rate)

## 0.3.0

- Install torch 2.6.0+cu124 for CUDA 12.2 driver compatibility (RTX 3090)
- Add `gpu` pytest marker and GPU smoke tests (CUDA availability, tensor ops, pickle round-trip)
- Add 22 GPU integration tests for SAE data filtering (thresholding, inference shapes, CPU/GPU consistency, streaming pipeline)
- Add 12 GPU integration tests for pretraining (model creation, forward pass, NorMuon, GWT, training loop, checkpoint round-trip, Liger kernels)
- Add 14 GPU integration tests for REINFORCE++ alignment (pipeline on GPU, frozen reference, train_step, loss stability, state dict round-trip, algorithm functions)
- Add 24 GPU integration tests for GGUF export/inference (checkpoint export, ternary fidelity, metadata, backend detection, evaluation, comparison)
- Add 6 end-to-end integration tests (pretrain→GGUF, pretrain→align, align→export, filter→pretrain data flow, full pipeline smoke)
- Update pyproject.toml CUDA index URL to cu124

## 0.2.5

- Fix `extract_activations` layer discovery: restructure branch logic to handle edge cases where layer modules lack `__len__` but are valid `torch.nn.Module` instances

## 0.2.4

- Fix `extract_activations` crash on `Gemma3Model` by navigating `.model.language_model.layers` for multimodal Gemma 3 models
- Remove unused `typing.TYPE_CHECKING` and `typing.TypedDict` imports from evaluation module
- Extract hardcoded HuggingFace Hub base URL to `_HF_BASE_URL` constant in `hf_transport.py`
- Tighten type annotations: `object` → specific types in `pipeline.py`, `config.py`, `gguf_export.py`, `streaming_filter.py`
- Fix unsafe `from_dict()` deserialization in `ReinforcePPPipelineConfig` with `isinstance` guards and explicit casts
- Fix ruff lint errors: long comment lines in `normuon.py`, import sorting in `__init__.py` files
- Add project classifiers, license field, and PyPI metadata to `pyproject.toml`
- Add Quick Start section with usage examples for all four pipeline stages to README
- Update CUDA wheel index URL to `cu130` (vast.ai template)
- Add doctest examples to `clip_rewards`, `global_normalize_advantages`, `dht_inverse`, `compute_orthogonality_metric`

## 0.2.3

- Fix eager evaluation crash in `extract_activations` when accessing `Gemma3Model.layers` (getattr default evaluated even when attribute exists)
- Fix `EvaluationConfig.backend` type mismatch: `str | None` → `BackendType | None`, eliminating ad-hoc enum conversion
- Rename `ModelConfig` → `AlignmentModelConfig` in takkeli-align to disambiguate from pretraining's `ModelConfig`
- Deduplicate upload-to-hub logic in `streaming_filter.py` via extracted `_upload_chunks()` helper
- Deduplicate logits extraction in `ReinforcePPPipeline` via extracted `_extract_logits()` helper
- Add `save_json()`, `top_p` field, and docstring fixes across evaluation config
- Fix `compute_output_stats()` empty-case return shape mismatch in comparison module
- Fix syntax error in evaluation script (extra closing paren after `--backend` argument)
- Add re-exports and `__all__` to takkeli-filtering `__init__.py` (was inconsistent with other packages)
- Add logging to gguf_export.py for export progress tracking
- Replace `os.path` with `pathlib` in inference module, add missing `Path` import
- Remove impossible `None` check on `AutoTokenizer.from_pretrained()` return in sae_loader
- Replace restating comments with WHY comments in normuon optimizer and lema module
- Inline `self_eps()` type-checker workaround in liger_ops
- Improve `generate_tokens()` docstring to clarify greedy-only behavior
- Accept optional `ExportConfig` in `create_minimal_gguf()` to reduce parameter spray
- Add README.md and LICENSE (MIT) to project root

## 0.2.2

- Fix SAE release name: `gemma-scope-2-4b-it-resid_post` → `gemma-scope-2-4b-it-res` (correct SAELens registry entry)
- Fix SAE d_in assertion: 2048 → 2560 (actual Gemma Scope 2 residual dimension for 4B IT)

## 0.2.1

- Pin sae-lens>=6.30.0 to ensure Gemma Scope 2 registry support

## 0.2.0

- Upgrade SAE data filtering from Gemma 2 2B + Gemma Scope to Gemma 3 4B IT + Gemma Scope 2
  - Model: `google/gemma-2-2b` → `google/gemma-3-4b-it`
  - SAE: `gemma-scope-2b-pt-res-canonical` → `gemma-scope-2-4b-it-resid_post`
  - SAE width: 16k → 262k with medium L0 for richer feature resolution
  - Layer: 20 → 22, d_model: 2304 → 2048
  - Switch model loader from `AutoModelForCausalLM` to `Gemma3ForConditionalGeneration` (VLM support)
- Pass HF_TOKEN to HuggingFace model/tokenizer loading in sae_loader.py

## 0.1.1

- Fix indentation error in tomllib import fallback across all test files
- Fix hardcoded absolute paths in test_streaming.py to use relative repo root
- Fix all ty type-checking diagnostics (80 errors)
- Fix ruff lint and format errors

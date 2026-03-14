# User Testing

Testing surface, resource cost classification, and validation procedures.

---

## Validation Surface

This is a Python ML training pipeline, not a web application. All validation is terminal-based.

### Primary Surface: pytest (CPU)
- All unit tests run locally on CPU via `pytest`
- No GPU required for local validation
- Covers: tensor shapes, mathematical correctness, constraint satisfaction, memory budgets

### Secondary Surface: Manual GPU Verification
- User runs training/inference scripts on Vast.ai cloud instances
- Reports results back (loss curves, VRAM usage, generated text)
- Not automated; manual verification steps

### Tertiary Surface: Inference Evaluation
- Final milestone includes evaluation scripts
- Run locally on AMD RX 6800 with GGUF model
- Generates outputs for Yudkowsky target prompts

---

## Validation Concurrency

### pytest (CPU)
- Each test suite: ~100-500MB RAM, 5-30s execution
- Max concurrent: 1 (sequential pytest is standard)
- Available headroom: 32GB RAM, 12 cores - more than sufficient

### No browser/agent-browser needed
- This project has no web UI
- No need for tuistory or agent-browser

---

## Per-Milestone Validation Notes

### Milestone 1 (uv-workspace)
- Verify: `uv sync`, `ty check`, `ruff check`, `ruff format --check`, `pytest`
- All should pass with exit code 0

### Milestone 2 (sae-data-filter)
- Unit tests for SAE loading, activation shapes, filtering logic
- HF Hub push requires `HF_TOKEN` env var (or mock for testing)

### Milestone 3 (core-architecture)
- CPU-only unit tests for all architecture components
- Memory profiling on CPU (tracemalloc)
- GPU testing: manual on Vast.ai

### Milestone 4 (optimizer-memory-stack)
- Unit tests for each optimizer component independently
- Full training loop CPU test with memory budget check
- GPU VRAM budget: manual verification on Vast.ai

### Milestone 5 (rlhf-alignment)
- Config loading tests (CPU)
- REINFORCE++ loss computation tests (CPU)
- Full pipeline: manual on Vast.ai with single GPU

### Milestone 6 (inference-export)
- GGUF conversion unit tests
- Inference script: runs locally on AMD RX 6800
- Evaluation script: requires built GGUF model

---

## Accepted Limitations

- GPU-specific assertions cannot be validated locally (AMD has no CUDA)
- HF Hub operations require network access and authentication tokens
- Liger Kernel Triton fused ops may behave differently on CPU vs GPU
- REINFORCE++ full pipeline testing requires cloud GPU instance

# Changelog

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

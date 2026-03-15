# Changelog

## 0.1.1

- Fix indentation error in tomllib import fallback across all test files
- Fix hardcoded absolute paths in test_streaming.py to use relative repo root
- Pass HF_TOKEN to HuggingFace model/tokenizer loading in sae_loader.py
- Fix all ty type-checking diagnostics (80 errors)
- Fix ruff lint and format errors

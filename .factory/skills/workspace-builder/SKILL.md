---
name: workspace-builder
description: Builds the uv workspace structure, project configuration, and development tooling (Milestone 1)
---

# Workspace Builder

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this skill for features related to the `uv-workspace` milestone:
- Creating root `pyproject.toml` with workspace configuration
- Setting up Ruff linting and formatting rules
- Configuring `ty` as the script runner
- Creating member directories with hardware-specific dependencies
- Setting up `.gitignore`, `pytest` configuration
- HuggingFace Hub transport layer configuration
- Any workspace-level tooling or infrastructure

## Work Procedure

1. **Read context first**
   - Read `mission.md` for the full project scope
   - Read `AGENTS.md` for coding conventions and constraints
   - Read `.factory/services.yaml` for available commands
   - Read `.factory/library/architecture.md` for workspace structure

2. **Write tests first (TDD - Red)**
   - Create test files that verify workspace configuration
   - Test: `uv sync` resolves without errors
   - Test: all 4 member directories exist with valid `pyproject.toml`
   - Test: Ruff config is present and functional
   - Test: `pytest` can discover test modules
   - Tests MUST fail initially (they are checking for the thing being built)

3. **Implement**
   - Create root `pyproject.toml` with:
     - `[project]` section with workspace metadata
     - `[tool.uv.sources]` and `[tool.uv.workspace]` for 4 members
     - `[tool.ruff]` for global linting/formatting config
     - `[tool.pytest.ini_options]` for test configuration
   - Create each member directory with its own `pyproject.toml`:
     - `01_data_filtering`: ROCm torch, sae-lens, transformers, datasets
     - `02_pretraining`: CUDA torch, triton, liger-kernel
     - `03_alignment`: CUDA torch, triton, openrlhf, liger-kernel
     - `04_inference_eval`: ROCm torch, llama-cpp-python, gguf
   - Create `.gitignore` (safetensors, gguf, pycache, venv, python-version)
   - Create `src/<package>/__init__.py` for each member
   - Create basic `tests/test_workspace.py` with smoke tests
   - Create HF Hub utility module with upload/download functions

4. **Verify**
   - Run `uv sync` - must succeed with exit code 0
   - Run `ruff check .` - must succeed with exit code 0
   - Run `ruff format --check .` - must succeed with exit code 0
   - Run `ty check` - must succeed with exit code 0
   - Run `uv run pytest -v` - all tests must pass
   - Run `git ls-files '*.safetensors' '*.gguf'` - must be empty

5. **Commit**
   - Stage all workspace configuration files
   - Commit with descriptive message

## Example Handoff

```json
{
  "salientSummary": "Created uv workspace with 4 members, global Ruff config, ty integration, and HuggingFace Hub transport. All static analysis and tests pass.",
  "whatWasImplemented": "Root pyproject.toml with uv workspace definition, 4 member pyproject.toml files with hardware-specific dependencies (ROCm for 01/04, CUDA for 02/03), global Ruff lint+format configuration, pytest configuration, .gitignore excluding binary artifacts, HF Hub upload/download utilities, and smoke tests verifying workspace resolution.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv sync", "exitCode": 0, "observation": "All 4 workspace members resolved successfully"},
      {"command": "ruff check .", "exitCode": 0, "observation": "Zero lint errors across workspace"},
      {"command": "ruff format --check .", "exitCode": 0, "observation": "All files already formatted"},
      {"command": "ty check", "exitCode": 0, "observation": "Zero type errors"},
      {"command": "uv run pytest -v", "exitCode": 0, "observation": "8 tests passed: workspace resolution, member configs, ruff config, gitignore, HF hub utils"},
      {"command": "git ls-files '*.safetensors' '*.gguf'", "exitCode": 0, "observation": "Empty output - no binary artifacts in git"}
    ],
    "interactiveChecks": [],
    "tests": {
      "added": [
        {"file": "tests/test_workspace.py", "cases": [
          {"name": "test_workspace_members_exist", "verifies": "All 4 member directories have pyproject.toml"},
          {"name": "test_rocm_cuda_isolation", "verifies": "ROCm and CUDA deps are not mixed"},
          {"name": "test_ruff_config_present", "verifies": "Ruff is configured in root pyproject.toml"},
          {"name": "test_gitignore_excludes_binaries", "verifies": ".gitignore excludes .safetensors, .gguf, etc."},
          {"name": "test_hf_hub_utils", "verifies": "HF Hub upload/download functions exist and type-check"},
          {"name": "test_pytest_discovery", "verifies": "pytest can discover test modules across workspace"},
          {"name": "test_no_binary_files_in_git", "verifies": "No safetensors/gguf files tracked by git"},
          {"name": "test_member_pyproject_toml_valid", "verifies": "Each member pyproject.toml is valid TOML with required sections"}
        ]}
      ]
    },
    "discoveredIssues": []
  }
}
```

## When to Return to Orchestrator

- If `uv sync` fails with a dependency resolution error that cannot be fixed by adjusting version constraints
- If Ruff or `ty` configuration conflicts with the installed versions
- If creating a new workspace member that requires shared types (needs design decision on shared package)

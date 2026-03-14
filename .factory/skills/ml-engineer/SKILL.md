---
name: ml-engineer
description: Implements ML model components, optimizers, training loops, and inference pipelines (Milestones 2-6)
---

# ML Engineer

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this skill for all ML implementation features across milestones 2-6:
- SAE-based data filtering pipeline (Milestone 2)
- Custom model architecture: BitNet, MLA, IndexCache, Dr.LLM (Milestone 3)
- Optimizer stack: NorMuon, GWT, Liger, LEMA (Milestone 4)
- RLHF alignment with REINFORCE++ (Milestone 5)
- GGUF export and inference evaluation (Milestone 6)

## Work Procedure

1. **Read context first**
   - Read `mission.md` for full project scope
   - Read `AGENTS.md` for coding conventions
   - Read `.factory/library/architecture.md` for architectural decisions
   - Read `.factory/library/environment.md` for dependency details
   - Read `.factory/services.yaml` for available commands
   - Read `design/llm-tech.md` for detailed research on the specific technique being implemented
   - Read the feature description from `features.json` for exact requirements

2. **Research before implementing**
   - Read the referenced paper's abstract and key sections
   - Look at reference implementations if available (e.g., kevbuh/bitnet for BitNet, KellerJordan/Muon for Muon)
   - Understand the mathematical formulation before writing any code
   - If a referenced repo URL 404s, search for the repo by name

3. **Write tests first (TDD - Red)**
   - Create test file in the appropriate `tests/` directory
   - Write failing tests that verify:
     - Tensor shapes at every stage
     - Mathematical correctness (compare against reference implementations)
     - Constraint satisfaction (e.g., ternary weights in {-1,0,1})
     - Edge cases (empty inputs, single tokens, large batch sizes)
     - Memory budgets (use `tracemalloc`)
   - Tests MUST use `device="cpu"` explicitly
   - For Liger kernel tests: implement pure PyTorch reference and compare with `atol=1e-4`

4. **Implement**
   - Place implementation in `src/<package>/` following the workspace structure
   - Use `dataclasses` for all configuration objects
   - Use `typing.Protocol` for component interfaces
   - All forward/inference methods: `(x: torch.Tensor) -> torch.Tensor`
   - Add type annotations to every function and method
   - Add docstrings to public classes and functions (brief, factual)

5. **Verify**
   - Run `ty check` - must pass
   - Run `ruff check .` - must pass
   - Run `ruff format --check .` - must format or already be formatted
   - Run `uv run pytest -v` - all tests must pass
   - For architecture features: verify parameter count is in expected range
   - For optimizer features: verify memory reduction claims with `tracemalloc`

6. **Document discoveries**
   - If you find that a technique doesn't work as expected, update `.factory/library/architecture.md`
   - If you find a useful reference implementation, note it in the library
   - If memory measurements differ from research report estimates, record actual numbers

7. **Commit**
   - Stage implementation and test files
   - Commit with descriptive message referencing the feature ID

## Key Implementation Notes

### BitNet b1.58
- `BitLinear` must quantize weights during forward pass using absmean
- The weight matrix is NOT stored as ternary; full-precision weights are stored and quantized on-the-fly
- `gamma = (1/(n*m)) * sum(|W_ij|)` is the scaling factor
- `RoundClip` function: round to {-1, 0, 1}

### DeepSeek MLA
- Compress KV into a low-dimensional latent vector `c_kv`
- During attention, project `c_kv` back to queries, keys, values
- The KV cache stores the compressed representation, not full Q/K/V

### IndexCache
- F layers: compute sparse attention indices + perform sparse attention
- S layers: skip indexer, use indices from nearest preceding F layer
- Pattern string like "FSFFS" determines F/S assignment per layer
- Multi-layer distillation loss: F-layer indexer trained against averaged attention of served S-layers

### NorMuon
- Newton-Schulz iteration: orthogonalize the gradient matrix
- Row-wise second-order momentum: track per-neuron update magnitudes
- Only applies to 2D parameter matrices (not 1D biases)

### GWT
- 1-level DHT: split each gradient row into approximation + detail (each n/2)
- 2-level DHT: recursive - approximation of approximation = n/4 total
- Store only final approximation coefficients in optimizer state
- Inverse DWT reconstructs for parameter update

### REINFORCE++
- No critic/value model needed
- Token-level KL penalty: D_KL(pi || ref) per token
- Trust region clipping on policy log-ratio
- Global advantage normalization (not per-group)

## Example Handoff

```json
{
  "salientSummary": "Implemented BitLinear layer with absmean quantization, achieving strict ternary {-1,0,1} weights. Forward pass produces correct output shapes. Parameter count within 1B target. All 12 architecture tests pass.",
  "whatWasImplemented": "BitLinear nn.Module with absmean quantization function, RoundClip ternary constraint, gamma scaling factor storage. Full model integrating BitLinear layers with standard transformer blocks. Tests for ternary constraint, shape correctness, absmean math, and parameter count estimation.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "ty check", "exitCode": 0, "observation": "Zero type errors"},
      {"command": "ruff check .", "exitCode": 0, "observation": "Zero lint errors"},
      {"command": "ruff format --check .", "exitCode": 0, "observation": "Already formatted"},
      {"command": "uv run pytest tests/test_bitlinear.py -v", "exitCode": 0, "observation": "12 tests passed: ternary constraint (4), shape (2), absmean math (2), gamma scaling (1), forward pass (2), parameter count (1)"}
    ],
    "interactiveChecks": [],
    "tests": {
      "added": [
        {"file": "02_pretraining/tests/test_bitlinear.py", "cases": [
          {"name": "test_bitlinear_weights_are_ternary", "verifies": "All weights in {-1, 0, 1} after quantization"},
          {"name": "test_bitlinear_forward_shape", "verifies": "Output shape matches (batch, seq_len, out_features)"},
          {"name": "test_absmean_quantization_math", "verifies": "gamma = mean(|W|), ternary = round_clip(W/gamma)"},
          {"name": "test_bitlinear_gamma_scaling", "verifies": "Output = ternary_W * gamma * input (scaling correct)"},
          {"name": "test_bitlinear_no_float_matmul", "verifies": "Forward uses only add/subtract on ternary values"}
        ]}
      ]
    },
    "discoveredIssues": []
  }
}
```

## When to Return to Orchestrator

- If a technique from the research report cannot be implemented as described (e.g., missing API, incompatible library)
- If tests reveal that two techniques don't compose (e.g., BitLinear with Triton fused ops)
- If memory measurements significantly exceed budget (< 12GB CPU proxy for training loop)
- If you need a design decision on component interfaces
- If a referenced paper or repo is fundamentally different from what was described

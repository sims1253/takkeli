#!/usr/bin/env python
"""Test REINFORCE++ alignment pipeline on GPU.

Demonstrates end-to-end alignment training with the real DrLLMModel
and ReinforcePPPipeline on CUDA.

Usage:
    cd /home/user/takkeli && .venv/bin/python scripts/test_alignment_gpu.py
"""

from __future__ import annotations

import sys
import copy

import torch
import torch.nn as nn

# Add src paths
sys.path.insert(0, "/home/user/takkeli/02_pretraining/src")
sys.path.insert(0, "/home/user/takkeli/03_alignment/src")

from takkeli_pretrain.model import ModelConfig, DrLLMModel
from takkeli_align.config import ReinforcePPPipelineConfig, ReinforcePPConfig
from takkeli_align.pipeline import ReinforcePPPipeline


def print_memory_usage(stage: str) -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[{stage}] Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")


def main():
    print("=" * 60)
    print("REINFORCE++ Alignment Pipeline GPU Test")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print_memory_usage("Initial")

    # ===========================================================
    # 1. Create a tiny DrLLMModel
    # ===========================================================
    print("\n[1] Creating tiny DrLLMModel...")

    model_config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=256,
        d_kv_laten=64,
        d_q_laten=64,
        d_rope=16,
        sparse_top_k=16,
        index_pattern="FSFF",  # 4 layers
        max_seq_len=128,
        enable_routing=True,
        d_router_hidden=32,
    )

    model = DrLLMModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print_memory_usage("After model creation")

    # ===========================================================
    # 2. Create ReinforcePPPipelineConfig
    # ===========================================================
    print("\n[2] Creating ReinforcePPPipelineConfig...")

    pipeline_config = ReinforcePPPipelineConfig(
        algorithm=ReinforcePPConfig(
            kl_coeff=0.1,
            clip_range=0.2,
            normalize_advantage=True,
            reward_clip_range=5.0,
        ),
        seed=42,
        use_critic=False,
    )
    print(f"KL coefficient: {pipeline_config.algorithm.kl_coeff}")
    print(f"Clip range: [{pipeline_config.algorithm.clip_range_low:.2f}, {pipeline_config.algorithm.clip_range_high:.2f}]")

    # ===========================================================
    # 3. Create ReinforcePPPipeline
    # ===========================================================
    print("\n[3] Creating ReinforcePPPipeline...")

    pipeline = ReinforcePPPipeline(pipeline_config, model)
    print_memory_usage("After pipeline creation")

    # ===========================================================
    # 4. Verify reference model is frozen
    # ===========================================================
    print("\n[4] Verifying reference model is frozen...")

    ref_frozen = all(not p.requires_grad for p in pipeline.reference_model.parameters())
    print(f"Reference model frozen: {ref_frozen}")
    assert ref_frozen, "Reference model should have all requires_grad=False"

    # Verify reference model is on CUDA
    ref_on_cuda = all(p.device.type == "cuda" for p in pipeline.reference_model.parameters())
    print(f"Reference model on CUDA: {ref_on_cuda}")
    assert ref_on_cuda, "Reference model should be on CUDA"

    # ===========================================================
    # 5. Run train_steps
    # ===========================================================
    print("\n[5] Running training steps...")

    batch_size = 2
    seq_len = 16
    vocab_size = model_config.vocab_size

    # Create optimizer for actual training
    optimizer = torch.optim.AdamW(
        pipeline.policy_model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    losses = []
    num_steps = 10

    for step in range(num_steps):
        # Generate synthetic data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        # Varied rewards to simulate real alignment
        rewards = torch.randn(batch_size, device=device)

        # Zero gradients
        optimizer.zero_grad()

        # Run train step
        loss = pipeline.train_step(input_ids, token_ids, rewards)

        # Verify loss is finite
        assert torch.isfinite(loss).all(), f"Loss is not finite at step {step}"
        assert not torch.isnan(loss).any(), f"Loss is NaN at step {step}"

        # Backward pass
        loss.backward()

        # Verify gradients exist on policy model
        has_grad = any(p.grad is not None for p in pipeline.policy_model.parameters() if p.requires_grad)
        assert has_grad, f"No gradients at step {step}"

        # Optimizer step
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  Step {step:2d}: loss = {loss_val:.4f}")

    print_memory_usage("After training steps")

    # ===========================================================
    # 6. Verify loss behavior
    # ===========================================================
    print("\n[6] Verifying loss behavior...")

    # All losses should be finite
    finite_losses = all(torch.isfinite(torch.tensor(l)) for l in losses)
    print(f"All losses finite: {finite_losses}")
    assert finite_losses, "Some losses are not finite"

    # Print loss trend (may fluctuate but should generally be reasonable)
    first_half_avg = sum(losses[:5]) / 5
    second_half_avg = sum(losses[5:]) / 5
    print(f"First 5 steps avg: {first_half_avg:.4f}")
    print(f"Last 5 steps avg: {second_half_avg:.4f}")
    print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

    # ===========================================================
    # 7. Verify policy model parameters change
    # ===========================================================
    print("\n[7] Verifying policy model updates...")

    # Store original parameter values
    original_params = {name: p.clone() for name, p in pipeline.policy_model.named_parameters()}

    # Run a few more steps to ensure parameters change
    for _ in range(3):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        rewards = torch.randn(batch_size, device=device)

        optimizer.zero_grad()
        loss = pipeline.train_step(input_ids, token_ids, rewards)
        loss.backward()
        optimizer.step()

    # Check if parameters changed
    params_changed = 0
    for name, p in pipeline.policy_model.named_parameters():
        if not torch.allclose(original_params[name], p, atol=1e-6):
            params_changed += 1

    total_params_count = sum(1 for _ in pipeline.policy_model.named_parameters())
    print(f"Parameters changed: {params_changed}/{total_params_count}")
    assert params_changed > 0, "No policy parameters changed after training"

    # ===========================================================
    # 8. Verify reference model is still frozen
    # ===========================================================
    print("\n[8] Verifying reference model still frozen...")

    ref_still_frozen = all(not p.requires_grad for p in pipeline.reference_model.parameters())
    print(f"Reference model still frozen: {ref_still_frozen}")
    assert ref_still_frozen, "Reference model should remain frozen"

    # ===========================================================
    # 9. Test state dict save/load round-trip
    # ===========================================================
    print("\n[9] Testing state dict save/load round-trip...")

    # Save state dict
    state_dict = pipeline.state_dict()
    print(f"State dict keys: {len(state_dict)} tensors")

    # Get outputs before reload
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits_before = pipeline.generate_policy_logits(input_ids)

    # Create new pipeline and load state dict
    new_model = DrLLMModel(model_config).to(device)
    new_pipeline = ReinforcePPPipeline(pipeline_config, new_model)
    new_pipeline.load_state_dict(state_dict)

    # Get outputs after reload
    logits_after = new_pipeline.generate_policy_logits(input_ids)

    # Verify outputs match
    outputs_match = torch.allclose(logits_before, logits_after, atol=1e-5)
    print(f"Outputs match after reload: {outputs_match}")
    assert outputs_match, "State dict round-trip failed - outputs differ"

    # ===========================================================
    # Summary
    # ===========================================================
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED!")
    print("=" * 60)
    print(f"Final memory: ", end="")
    print_memory_usage("Final")

    # Clean up
    del pipeline, new_pipeline, model, new_model
    torch.cuda.empty_cache()
    print("\nGPU cache cleared.")


if __name__ == "__main__":
    main()

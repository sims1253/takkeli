#!/usr/bin/env python3
"""Test script for GPU pretraining with DrLLMModel + NorMuonGWT."""

import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import time
import gc

from takkeli_pretrain.model import ModelConfig, DrLLMModel
from takkeli_pretrain.gwt import NorMuonGWT
from takkeli_pretrain.training_loop import compute_loss

def test_model_size(name: str, config: ModelConfig, batch_size: int, seq_len: int):
    """Test a model of a given size."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Model config: d_model={config.d_model}, n_layers={config.n_layers}")
    
    # Create model
    print("Creating model...")
    model = DrLLMModel(config)
    model = model.cuda()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"GPU memory after model: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Create optimizer
    print("Creating NorMuonGWT optimizer...")
    gwt_levels = 2
    optimizer = NorMuonGWT(model.parameters(), lr=0.01, gwt_levels=gwt_levels)
    
    # Generate training data
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    
    # Training loop
    print("Running training steps...")
    losses = []
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass - model returns (logits, aux_outputs)
        logits, aux_outputs = model(input_ids)
        
        # Compute loss (shifted cross-entropy for LM)
        loss = compute_loss(logits, input_ids)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: loss = {loss.item():.4f}")
    
    print(f"Loss trajectory: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Checkpoint round-trip
    print("Testing checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        
        # Save
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.stat().st_size / 1e6:.2f} MB")
        
        # Get output before loading
        with torch.no_grad():
            output_before, _ = model(input_ids)
            output_before = output_before.clone()
        
        # Load into new model
        model2 = DrLLMModel(config).cuda()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        
        # Verify identical outputs
        with torch.no_grad():
            output_after, _ = model2(input_ids)
        
        assert torch.allclose(output_before, output_after, atol=1e-5), "Checkpoint mismatch!"
        print("  Checkpoint round-trip: ✓")
    
    # Cleanup
    del model, model2, optimizer, input_ids, logits, output_before, output_after
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"{name}: PASSED")
    return True

def main():
    print("=" * 60)
    print("GPU Pretraining Test Suite")
    print("=" * 60)
    
    # Check CUDA
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test configurations
    configs = [
        ("Tiny Model (~0.8M params)", ModelConfig(
            vocab_size=256,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ffn=256,
            max_seq_len=256,
            d_rope=32,
            d_kv_laten=64,
            d_q_laten=64,
            index_pattern="FSFS",
        ), 4, 64),
        ("Small Model (~10M params)", ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ffn=512,
            max_seq_len=256,
            d_rope=32,  # Must be <= d_model // n_heads = 32
            d_kv_laten=128,
            d_q_laten=128,
            index_pattern="FSFSFS",
        ), 4, 64),
        ("Medium Model (~50M params)", ModelConfig(
            vocab_size=32000,
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ffn=1024,
            max_seq_len=256,
            d_rope=64,
            d_kv_laten=256,
            d_q_laten=256,
            index_pattern="FSFSFSFS",
        ), 2, 64),
    ]
    
    results = []
    for name, config, batch_size, seq_len in configs:
        try:
            test_model_size(name, config, batch_size, seq_len)
            results.append((name, "PASSED"))
        except Exception as e:
            print(f"{name}: FAILED - {e}")
            results.append((name, f"FAILED: {e}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        print(f"  {name}: {result}")
    print("=" * 60)

if __name__ == "__main__":
    main()

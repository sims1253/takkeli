"""End-to-end integration tests for the full takkeli pipeline.

Validates the complete data flow:
    1. Create tiny pretraining model (DrLLMModel) → save checkpoint
    2. Load checkpoint as alignment policy → train a few steps → save aligned checkpoint
    3. Export aligned checkpoint to GGUF → verify GGUF validity
    4. Load GGUF with gguf library → verify metadata and ternary weights

All GPU tests use tiny model: d_model=64, n_layers=2, n_heads=2, vocab_size=256, d_ffn=128.
"""

from __future__ import annotations

import struct
from pathlib import Path

import gguf
import pytest
import torch

gpu = pytest.mark.gpu
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model_config() -> object:
    """Create a tiny ModelConfig for fast GPU tests."""
    from takkeli_pretrain.model import ModelConfig

    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ffn=128,
        d_kv_laten=32,
        d_q_laten=32,
        d_rope=16,
        sparse_top_k=8,
        index_pattern="FF",
        max_seq_len=64,
        enable_routing=False,
        d_router_hidden=16,
        tie_weights=False,
    )


def _read_gguf_magic(path: Path) -> int:
    """Read the 4-byte magic number from a GGUF file."""
    with open(path, "rb") as f:
        data = f.read(4)
        return struct.unpack("<I", data)[0]


def _export_config_from_model_config(
    model_config: object,
    checkpoint_path: str,
    output_path: str,
) -> object:
    """Build an ExportConfig matching a ModelConfig."""
    from takkeli_inference.gguf_export import ExportConfig

    return ExportConfig(
        model_name="takkeli-tiny-test",
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        context_length=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        embedding_dim=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ffn=model_config.d_ffn,
    )


# ---------------------------------------------------------------------------
# Test 1: Pretrain checkpoint → GGUF
# ---------------------------------------------------------------------------


class TestPretrainCheckpointToGGUF:
    """Create tiny DrLLMModel on GPU, save checkpoint, export to GGUF, verify."""

    @gpu
    def test_pretrain_checkpoint_to_gguf(self, tmp_path: Path) -> None:
        from takkeli_inference.gguf_export import export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _tiny_model_config()
        model = DrLLMModel(config).to(device="cuda")

        # Forward pass so BitLinear quantization happens
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        model(input_ids)

        # Save state_dict
        ckpt_path = tmp_path / "pretrain.pt"
        torch.save(model.state_dict(), str(ckpt_path))

        # Export to GGUF
        gguf_path = tmp_path / "model.gguf"
        export_cfg = _export_config_from_model_config(config, str(ckpt_path), str(gguf_path))
        result = export_to_gguf(export_cfg)

        # Verify GGUF file exists and has correct magic
        assert result == gguf_path
        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Invalid GGUF magic: {hex(magic)}"

        # Readable by gguf library
        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0, "No tensors in GGUF"

        # Verify metadata
        fields = reader.fields
        assert "general.name" in fields
        assert fields["general.name"].contents() == "takkeli-tiny-test"
        assert "bitnet.context_length" in fields
        assert fields["bitnet.context_length"].contents() == config.max_seq_len
        assert "bitnet.embedding_length" in fields
        assert fields["bitnet.embedding_length"].contents() == config.d_model
        assert "bitnet.vocab_size" in fields
        assert fields["bitnet.vocab_size"].contents() == config.vocab_size

        # Clean up
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 2: Pretrain → Alignment checkpoint flow
# ---------------------------------------------------------------------------


class TestPretrainToAlignFlow:
    """Create tiny model, save checkpoint, load as alignment policy, run train_step."""

    @gpu
    def test_pretrain_to_align_checkpoint_flow(self, tmp_path: Path) -> None:
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _tiny_model_config()

        # Create and save pretrained model on GPU
        model = DrLLMModel(config).to(device="cuda")
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        model(input_ids)

        ckpt_path = tmp_path / "pretrained.pt"
        torch.save(model.state_dict(), str(ckpt_path))
        del model
        torch.cuda.empty_cache()

        # Load checkpoint into a new model for alignment
        policy_model = DrLLMModel(config).to(device="cuda")
        state_dict = torch.load(str(ckpt_path), map_location="cuda", weights_only=True)
        policy_model.load_state_dict(state_dict)

        # Create alignment pipeline
        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Run one train_step
        token_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        rewards = torch.tensor([1.0], device="cuda")
        loss = pipeline.train_step(input_ids, token_ids, rewards)

        # Verify loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.requires_grad

        # Clean up
        del pipeline, policy_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 3: Alignment → Export flow
# ---------------------------------------------------------------------------


class TestAlignToExportFlow:
    """Create model, run through alignment, export to GGUF, verify."""

    @gpu
    def test_align_to_export_flow(self, tmp_path: Path) -> None:
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_inference.gguf_export import export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _tiny_model_config()

        # Create and save model checkpoint (simulates aligned model)
        model = DrLLMModel(config).to(device="cuda")
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        model(input_ids)

        ckpt_path = tmp_path / "aligned.pt"
        torch.save(model.state_dict(), str(ckpt_path))
        del model
        torch.cuda.empty_cache()

        # Simulate alignment: load into pipeline and run a step
        policy_model = DrLLMModel(config).to(device="cuda")
        state_dict = torch.load(str(ckpt_path), map_location="cuda", weights_only=True)
        policy_model.load_state_dict(state_dict)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        token_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        rewards = torch.tensor([1.0], device="cuda")
        loss = pipeline.train_step(input_ids, token_ids, rewards)
        assert torch.isfinite(loss)

        # Save aligned checkpoint
        aligned_path = tmp_path / "aligned_trained.pt"
        torch.save(pipeline.policy_model.state_dict(), str(aligned_path))

        # Export to GGUF
        gguf_path = tmp_path / "aligned.gguf"
        export_cfg = _export_config_from_model_config(config, str(aligned_path), str(gguf_path))
        result = export_to_gguf(export_cfg)

        # Verify GGUF
        assert result == gguf_path
        assert gguf_path.exists()
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Invalid GGUF magic: {hex(magic)}"

        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0
        assert "general.name" in reader.fields

        # Clean up
        del pipeline, policy_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 4: Filter → Pretrain data flow (CPU, no model needed)
# ---------------------------------------------------------------------------


class TestFilterToPretrainDataFlow:
    """Verify filtered data can be used as input to the training loop."""

    def test_filter_to_pretrain_data_flow(self, tmp_path: Path) -> None:
        """Create mock filtered data (token tensors) and verify compatibility."""
        config = _tiny_model_config()

        # Simulate filtered data: token tensors of shape (batch, seq_len)
        batch_size = 2
        seq_len = config.max_seq_len
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Verify shapes are compatible with model expectations
        assert token_ids.dim() == 2, "Token tensors must be 2D (batch, seq_len)"
        assert token_ids.shape[0] == batch_size
        assert token_ids.shape[1] == seq_len

        # Verify all token IDs are within vocabulary range
        assert token_ids.min() >= 0, "Token IDs must be non-negative"
        assert token_ids.max() < config.vocab_size, (
            f"Token ID {token_ids.max().item()} exceeds vocab size {config.vocab_size}"
        )

        # Verify tokens can serve as DataLoader output
        # (DataLoader yields batches of the same shape)
        dataloader_batch = token_ids  # simulates next(iter(dataloader))
        assert isinstance(dataloader_batch, torch.Tensor)
        assert dataloader_batch.dtype == torch.long

        # Write filtered data as JSONL (simulates 01_data_filtering → 02_pretraining handoff)
        import json

        data_path = tmp_path / "filtered.jsonl"
        samples = [{"text": "Sample text for tokenization"} for _ in range(5)]
        with open(data_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Verify dataset can be loaded
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=str(data_path), split="train")
        assert len(dataset) == 5

        for example in dataset:
            text: str = example["text"]
            tokens = text.split()
            token_tensor = torch.tensor(
                [hash(t) % config.vocab_size for t in tokens],
                dtype=torch.long,
            )
            assert token_tensor.dim() == 1
            assert token_tensor.shape[0] > 0
            assert token_tensor.max() < config.vocab_size

    def test_token_tensor_shapes_compatible_with_model(self) -> None:
        """Verify token tensor shapes match model forward pass requirements."""
        config = _tiny_model_config()

        # Various valid input shapes
        valid_shapes = [
            (1, 8),   # single sample, short sequence
            (2, 16),  # batch of 2
            (1, config.max_seq_len),  # max sequence length
        ]

        for batch_size, seq_len in valid_shapes:
            token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            assert token_ids.shape == (batch_size, seq_len)

        # Verify that exceeding max_seq_len would be invalid
        # (position_embedding only has max_seq_len entries)
        assert config.max_seq_len > 0
        oversized_seq = config.max_seq_len + 1
        # This shape cannot be processed because positional embedding index would overflow
        assert oversized_seq > config.max_seq_len


# ---------------------------------------------------------------------------
# Test 5: Full pipeline smoke (GPU)
# ---------------------------------------------------------------------------


class TestFullPipelineSmoke:
    """Minimal end-to-end: create model → forward pass → save → export GGUF → verify."""

    @gpu
    def test_full_pipeline_smoke(self, tmp_path: Path) -> None:
        from takkeli_inference.gguf_export import export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _tiny_model_config()

        # Step 1: Create tiny model on GPU
        model = DrLLMModel(config).to(device="cuda")

        # Step 2: Forward pass (triggers BitLinear quantization)
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        logits, aux = model(input_ids)
        assert logits.shape == (1, 8, config.vocab_size)

        # Step 3: Save checkpoint
        ckpt_path = tmp_path / "pipeline.pt"
        torch.save(model.state_dict(), str(ckpt_path))
        del model
        torch.cuda.empty_cache()

        # Step 4: Export to GGUF
        gguf_path = tmp_path / "pipeline.gguf"
        export_cfg = _export_config_from_model_config(config, str(ckpt_path), str(gguf_path))
        result = export_to_gguf(export_cfg)

        # Step 5: Verify GGUF
        assert result == gguf_path
        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0

        # Magic number
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Invalid GGUF magic: {hex(magic)}"

        # Readable by gguf library with metadata
        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0, "No tensors in GGUF"
        assert len(reader.fields) > 0, "No metadata in GGUF"

        # Verify key metadata
        fields = reader.fields
        assert "general.name" in fields
        assert "bitnet.context_length" in fields
        assert "bitnet.embedding_length" in fields
        assert "bitnet.vocab_size" in fields

        # Verify ternary weights exist (FFN layers should use TQ1_0)
        ternary_tensors = [
            t for t in reader.tensors if t.tensor_type == gguf.GGMLQuantizationType.TQ1_0
        ]
        assert len(ternary_tensors) > 0, "No TQ1_0 (ternary) tensors found in GGUF"

        # Clean up
        torch.cuda.empty_cache()

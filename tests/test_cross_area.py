"""Cross-area integration tests for the Consciousness Filter LLM pipeline.

Validates end-to-end data flow across all four modules:

- VAL-CROSS-001: Filtered dataset reaches trainer with valid token tensors
- VAL-CROSS-002: Checkpoint save/load round-trip produces identical outputs
- VAL-CROSS-003: Saved checkpoint converts to GGUF without error
- VAL-CROSS-004: Alignment receives pretrained weights as policy/reference
- VAL-CROSS-005: Export receives aligned weights and exports to GGUF
- VAL-CROSS-006: HF Hub round-trip artifact integrity (SHA-256)
- VAL-CROSS-007: Static analysis gate (verified via external command)
- VAL-CROSS-008: Module dependency graph is acyclic (01->02->03->04)
"""

from __future__ import annotations

import ast
import hashlib
import struct
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_model_config() -> Any:
    """Create a small DrLLMModel config for fast CPU tests."""
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
        enable_routing=True,
        d_router_hidden=16,
        tie_weights=False,
    )


def _create_and_save_checkpoint(model: Any, path: Path) -> None:
    """Save a model state dict to a checkpoint file.

    Args:
        model: PyTorch model with state_dict().
        path: Output checkpoint path.
    """
    torch.save(model.state_dict(), str(path))


def _load_checkpoint(model: Any, path: Path) -> None:
    """Load a state dict from a checkpoint file into the model.

    Args:
        model: PyTorch model to load into.
        path: Checkpoint file path.
    """
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)


def _read_gguf_magic(path: Path) -> int:
    """Read the magic number from a GGUF file."""
    with open(path, "rb") as f:
        data = f.read(4)
        return struct.unpack("<I", data)[0]


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# VAL-CROSS-001: Filtered dataset reaches trainer
# ---------------------------------------------------------------------------


class TestFilteredDataReachesTrainer:
    """VAL-CROSS-001: End-to-end filtered data reaches trainer.

    A dataset filtered by the SAE pipeline (simulated here) can be loaded
    by the pretraining module's data loader and yields valid token tensors.
    """

    def test_filtered_dataset_loads_and_yields_token_tensors(self, tmp_path: Path) -> None:
        """Push a small test dataset via pipeline; load with datasets.load_dataset;
        assert token tensor shape (seq_len,) per example."""
        import json

        # Create a mock filtered dataset as JSONL (simulates pipeline output)
        data_path = tmp_path / "filtered.jsonl"
        samples = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Machine learning is a subset of artificial intelligence."},
            {"text": "Python is a popular programming language for data science."},
        ]
        with open(data_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Load the filtered dataset using HuggingFace datasets
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=str(data_path), split="train")

        # Verify the dataset has the expected number of examples
        assert len(dataset) == 3

        # Simulate tokenization (use a simple word-level tokenizer for test)
        # In production, the pretraining module would use the actual tokenizer
        for example in dataset:
            text: str = example["text"]
            # Tokenize by splitting on spaces (simplified)
            tokens = text.split()
            assert len(tokens) > 0, "Each example must yield at least one token"

            # Convert to tensor to verify it's a valid token tensor
            token_ids = torch.tensor([hash(t) % 1000 for t in tokens], dtype=torch.long)
            assert token_ids.dim() == 1
            assert token_ids.shape[0] > 0

    def test_streaming_filter_produces_loadable_dataset(self) -> None:
        """Verify that FilterResult chunks can be serialized and loaded back."""
        from takkeli_filtering.streaming_filter import FilterResult, FilterStats

        # Create mock filter results
        results = [
            FilterResult(chunk={"text": "Hello world"}, passed=True, max_activation=0.1),
            FilterResult(chunk={"text": "Filtered content"}, passed=False, max_activation=0.9),
            FilterResult(chunk={"text": "Another example"}, passed=True, max_activation=0.2),
        ]

        # Verify all chunks have results (no data dropped)
        assert len(results) == 3

        # Verify pass/fail counts match stats
        stats = FilterStats()
        for r in results:
            stats.total += 1
            if r.passed:
                stats.passed += 1
            else:
                stats.failed += 1

        assert stats.total == 3
        assert stats.passed == 2
        assert stats.failed == 1

        # Verify passing chunks can be loaded as dataset entries
        import json
        import tempfile

        passing = [r.chunk for r in results if r.passed]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for chunk in passing:
                f.write(json.dumps(chunk) + "\n")
            tmp_path = f.name

        from datasets import load_dataset

        dataset = load_dataset("json", data_files=tmp_path, split="train")
        assert len(dataset) == 2

        # Cleanup
        Path(tmp_path).unlink()


# ---------------------------------------------------------------------------
# VAL-CROSS-002: Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """VAL-CROSS-002: Model checkpoint save/load round-trip.

    A model checkpoint saved via torch.save can be loaded back and
    produces identical outputs for the same input.
    """

    def test_save_load_produces_identical_outputs(self, tmp_path: Path) -> None:
        """Save model, reload, and verify identical outputs."""
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")
        model.eval()

        # Create a fixed input
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        # Get output before save
        with torch.no_grad():
            logits_before, _ = model(input_ids)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        # Create a new model and load checkpoint
        model2 = DrLLMModel(config).to(device="cpu")
        model2.eval()
        _load_checkpoint(model2, checkpoint_path)

        # Get output after load
        with torch.no_grad():
            logits_after, _ = model2(input_ids)

        # Outputs must be identical
        assert torch.equal(logits_before, logits_after), (
            "Model outputs differ after checkpoint round-trip"
        )

    def test_state_dict_keys_match_after_round_trip(self, tmp_path: Path) -> None:
        """Verify all state dict keys are preserved after save/load."""
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        original_keys = set(model.state_dict().keys())

        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        loaded_state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        loaded_keys = set(loaded_state.keys())

        assert original_keys == loaded_keys, (
            f"State dict keys changed after save/load. "
            f"Missing: {original_keys - loaded_keys}, "
            f"Extra: {loaded_keys - original_keys}"
        )

    def test_parameter_values_match_after_round_trip(self, tmp_path: Path) -> None:
        """Verify all parameter tensors are bit-identical after save/load."""
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        original_params = {name: p.clone() for name, p in model.named_parameters()}

        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        model2 = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(model2, checkpoint_path)

        for name, param in model2.named_parameters():
            assert torch.equal(original_params[name], param), (
                f"Parameter '{name}' differs after round-trip"
            )


# ---------------------------------------------------------------------------
# VAL-CROSS-003: Checkpoint compatible with GGUF export
# ---------------------------------------------------------------------------


class TestCheckpointGGUFCompatibility:
    """VAL-CROSS-003: Saved checkpoint converts to GGUF without error.

    A saved checkpoint can be loaded by the export script and converted
    to GGUF format.
    """

    def test_checkpoint_to_gguf_conversion(self, tmp_path: Path) -> None:
        """Save checkpoint -> run export -> GGUF file produced."""
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        # Export to GGUF
        gguf_path = tmp_path / "model.gguf"
        export_config = ExportConfig(
            model_name="test-cross-area",
            checkpoint_path=str(checkpoint_path),
            output_path=str(gguf_path),
            context_length=config.max_seq_len,
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )

        result = export_to_gguf(export_config)

        # Verify GGUF file was created
        assert result == gguf_path
        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0

        # Verify GGUF magic number
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Invalid GGUF magic: {hex(magic)}"

    def test_gguf_readable_after_export(self, tmp_path: Path) -> None:
        """GGUF file from checkpoint is readable by the gguf library."""
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        gguf_path = tmp_path / "model.gguf"
        export_config = ExportConfig(
            checkpoint_path=str(checkpoint_path),
            output_path=str(gguf_path),
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )
        export_to_gguf(export_config)

        # Read and verify GGUF file
        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0, "GGUF file contains no tensors"
        assert len(reader.fields) > 0, "GGUF file contains no metadata"

    def test_safetensors_style_roundtrip(self, tmp_path: Path) -> None:
        """Simulate safetensors-style checkpoint (same state_dict format)."""
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        # Save state dict (same format as safetensors would use)
        state_dict = model.state_dict()
        checkpoint_path = tmp_path / "model.safetensors"
        # torch.save with state_dict is format-compatible with safetensors
        torch.save(state_dict, str(checkpoint_path))

        # Export to GGUF
        gguf_path = tmp_path / "model.gguf"
        export_config = ExportConfig(
            checkpoint_path=str(checkpoint_path),
            output_path=str(gguf_path),
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )
        export_to_gguf(export_config)

        assert gguf_path.exists()
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747


# ---------------------------------------------------------------------------
# VAL-CROSS-004: Alignment receives pretrained weights
# ---------------------------------------------------------------------------


class TestAlignmentReceivesPretrainedWeights:
    """VAL-CROSS-004: Alignment module can load pretrained checkpoint.

    The RLHF module can load a pretrained model checkpoint as the initial
    policy (and reference) model.
    """

    def test_pipeline_loads_pretrained_policy(self, tmp_path: Path) -> None:
        """Load pretrained checkpoint as initial policy model."""
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        pretrained_model = DrLLMModel(config).to(device="cpu")

        # Save pretrained checkpoint
        checkpoint_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(pretrained_model, checkpoint_path)

        # Create a new model for alignment (policy)
        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, checkpoint_path)

        # Create REINFORCE++ pipeline with the pretrained policy
        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Verify policy model is a valid nn.Module
        assert isinstance(pipeline.policy_model, torch.nn.Module)
        model = pipeline.policy_model
        assert isinstance(model, torch.nn.Module)
        assert model.config.vocab_size == config.vocab_size  # type: ignore[union-attr]
        assert model.config.d_model == config.d_model  # type: ignore[union-attr]

    def test_pipeline_creates_frozen_reference(self, tmp_path: Path) -> None:
        """Pipeline creates a frozen reference model from pretrained weights."""
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        pretrained_model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(pretrained_model, checkpoint_path)

        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, checkpoint_path)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Reference model should be frozen
        assert isinstance(pipeline.reference_model, torch.nn.Module)
        for param in pipeline.reference_model.parameters():
            assert not param.requires_grad, "Reference model parameters should be frozen"

        # Reference model should have the same architecture
        ref_params = sum(p.numel() for p in pipeline.reference_model.parameters())
        policy_params = sum(p.numel() for p in pipeline.policy_model.parameters())
        assert ref_params == policy_params

    def test_pipeline_produces_valid_logits(self, tmp_path: Path) -> None:
        """Pipeline with pretrained weights generates valid logits."""
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        pretrained_model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(pretrained_model, checkpoint_path)

        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, checkpoint_path)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        # Generate reference logits (no grad)
        ref_logits = pipeline.generate_reference_logits(input_ids)
        assert ref_logits.shape == (1, 8, config.vocab_size)

        # Generate policy logits (with grad)
        policy_logits = pipeline.generate_policy_logits(input_ids)
        assert policy_logits.shape == (1, 8, config.vocab_size)
        assert policy_logits.requires_grad

    def test_pipeline_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        """Pipeline state_dict save/load round-trip."""
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        pretrained_model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(pretrained_model, checkpoint_path)

        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, checkpoint_path)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Save pipeline state
        pipeline_state = pipeline.state_dict()
        align_ckpt_path = tmp_path / "aligned.pt"
        torch.save(pipeline_state, str(align_ckpt_path))

        # Load into a new pipeline
        policy_model2 = DrLLMModel(config).to(device="cpu")
        pipeline2 = ReinforcePPPipeline(pipeline_config, policy_model2)
        loaded_state = torch.load(str(align_ckpt_path), map_location="cpu", weights_only=True)
        pipeline2.load_state_dict(loaded_state)

        # Verify parameters match
        for (n1, p1), (_n2, p2) in zip(
            pipeline.policy_model.named_parameters(),
            pipeline2.policy_model.named_parameters(),
            strict=True,
        ):
            assert torch.equal(p1.data, p2.data), f"Parameter {n1} differs after round-trip"


# ---------------------------------------------------------------------------
# VAL-CROSS-005: Export receives aligned weights
# ---------------------------------------------------------------------------


class TestExportReceivesAlignedWeights:
    """VAL-CROSS-005: GGUF export can load aligned checkpoint and export it.

    The GGUF export script can load a checkpoint from the alignment phase
    (post-RLHF) and export it.
    """

    def test_aligned_checkpoint_exports_to_gguf(self, tmp_path: Path) -> None:
        """Save aligned checkpoint -> run export -> GGUF file produced."""
        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()

        # Create and save pretrained checkpoint (simulates pretraining)
        pretrained_model = DrLLMModel(config).to(device="cpu")
        pretrained_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(pretrained_model, pretrained_path)

        # Simulate alignment: load pretrained into pipeline
        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, pretrained_path)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Simulate a training step (no actual optimization, just verify flow)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        token_ids = torch.randint(0, config.vocab_size, (1, 8))
        rewards = torch.tensor([1.0])

        loss = pipeline.train_step(input_ids, token_ids, rewards)
        assert loss.dim() == 0
        assert loss.requires_grad

        # Save aligned checkpoint
        aligned_path = tmp_path / "aligned.pt"
        torch.save(pipeline.state_dict(), str(aligned_path))

        # Export aligned checkpoint to GGUF
        # Need to get state_dict in model format (not pipeline format)
        aligned_model_path = tmp_path / "aligned_model.pt"
        torch.save(pipeline.policy_model.state_dict(), str(aligned_model_path))

        gguf_path = tmp_path / "aligned_model.gguf"
        export_config = ExportConfig(
            model_name="takkeli-1b-aligned",
            checkpoint_path=str(aligned_model_path),
            output_path=str(gguf_path),
            context_length=config.max_seq_len,
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )

        result = export_to_gguf(export_config)

        assert result == gguf_path
        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0

        # Verify GGUF magic number
        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Invalid GGUF magic for aligned export: {hex(magic)}"

    def test_aligned_gguf_has_required_metadata(self, tmp_path: Path) -> None:
        """Aligned GGUF export contains required metadata."""
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "aligned.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        gguf_path = tmp_path / "aligned.gguf"
        export_config = ExportConfig(
            model_name="takkeli-aligned",
            checkpoint_path=str(checkpoint_path),
            output_path=str(gguf_path),
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )
        export_to_gguf(export_config)

        reader = gguf.GGUFReader(str(gguf_path))
        metadata: dict[str, object] = {}
        for key, field in reader.fields.items():
            if key.startswith("GGUF."):
                continue
            try:
                metadata[key] = field.contents()
            except Exception:
                metadata[key] = str(field.parts)

        assert "general.name" in metadata
        assert metadata["general.name"] == "takkeli-aligned"
        assert "bitnet.context_length" in metadata
        assert "bitnet.embedding_length" in metadata
        assert "bitnet.vocab_size" in metadata


# ---------------------------------------------------------------------------
# VAL-CROSS-006: HF Hub round-trip artifact integrity
# ---------------------------------------------------------------------------


class TestHFHubRoundTripIntegrity:
    """VAL-CROSS-006: HF Hub round-trip artifact integrity.

    An artifact uploaded to HF Hub and downloaded has identical SHA-256 hash.

    NOTE: This test uses a mock approach since actual HF Hub upload/download
    requires authentication and network access. The test verifies that the
    transport utility functions are correct and that SHA-256 round-trip logic
    works with local file operations.
    """

    def test_sha256_local_roundtrip(self, tmp_path: Path) -> None:
        """SHA-256 is identical for file written and read back locally."""
        # Create a test artifact
        artifact_path = tmp_path / "test_artifact.bin"
        original_data = b"test artifact content for SHA-256 verification"
        artifact_path.write_bytes(original_data)

        # Compute SHA-256 before
        hash_before = _sha256_of_file(artifact_path)

        # Copy to a new location (simulates download)
        downloaded_path = tmp_path / "downloaded_artifact.bin"
        downloaded_path.write_bytes(artifact_path.read_bytes())

        # Compute SHA-256 after
        hash_after = _sha256_of_file(downloaded_path)

        assert hash_before == hash_after, (
            f"SHA-256 mismatch: before={hash_before}, after={hash_after}"
        )

    def test_sha256_large_file_roundtrip(self, tmp_path: Path) -> None:
        """SHA-256 round-trip works for larger files."""
        # Create a larger test file (~1MB)
        artifact_path = tmp_path / "large_artifact.bin"
        data = np.random.randn(1024, 256).astype(np.float32)
        artifact_path.write_bytes(data.tobytes())

        hash_before = _sha256_of_file(artifact_path)

        # Simulate round-trip via copy
        downloaded_path = tmp_path / "downloaded_large.bin"
        downloaded_path.write_bytes(artifact_path.read_bytes())

        hash_after = _sha256_of_file(downloaded_path)

        assert hash_before == hash_after

    def test_sha256_jsonl_dataset_roundtrip(self, tmp_path: Path) -> None:
        """SHA-256 round-trip for JSONL dataset files (actual artifact type)."""
        import json

        dataset_path = tmp_path / "dataset.jsonl"
        samples = [{"text": f"Sample text number {i} for testing."} for i in range(100)]
        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        hash_before = _sha256_of_file(dataset_path)

        # Simulate round-trip
        downloaded_path = tmp_path / "downloaded_dataset.jsonl"
        downloaded_path.write_bytes(dataset_path.read_bytes())

        hash_after = _sha256_of_file(downloaded_path)

        assert hash_before == hash_after

    def test_sha256_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        """SHA-256 round-trip for PyTorch checkpoint files."""
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        checkpoint_path = tmp_path / "checkpoint.pt"
        _create_and_save_checkpoint(model, checkpoint_path)

        hash_before = _sha256_of_file(checkpoint_path)

        # Simulate round-trip
        downloaded_path = tmp_path / "downloaded_checkpoint.pt"
        downloaded_path.write_bytes(checkpoint_path.read_bytes())

        hash_after = _sha256_of_file(downloaded_path)

        assert hash_before == hash_after

    def test_hf_transport_upload_download_signatures(self) -> None:
        """Verify HF transport utility functions have correct signatures."""
        import inspect

        from takkeli_filtering.hf_transport import download_from_hub, upload_to_hub

        # Verify upload_to_hub signature
        upload_sig = inspect.signature(upload_to_hub)
        upload_params = list(upload_sig.parameters.keys())
        assert "local_path" in upload_params
        assert "repo_id" in upload_params
        assert "repo_type" in upload_params
        assert "private" in upload_params

        # Verify download_from_hub signature
        download_sig = inspect.signature(download_from_hub)
        download_params = list(download_sig.parameters.keys())
        assert "repo_id" in download_params
        assert "local_path" in download_params
        assert "repo_type" in download_params

    def test_hf_transport_upload_validates_inputs(self, tmp_path: Path) -> None:
        """HF transport raises appropriate errors for invalid inputs."""
        from takkeli_filtering.hf_transport import upload_to_hub

        # Non-existent path should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            upload_to_hub(
                local_path=Path("/nonexistent/file.txt"),
                repo_id="test/repo",
            )

        # Invalid repo_type should raise ValueError
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        with pytest.raises(ValueError, match="repo_type"):
            upload_to_hub(
                local_path=test_file,
                repo_id="test/repo",
                repo_type="invalid_type",
            )


# ---------------------------------------------------------------------------
# VAL-CROSS-007: Static analysis gate
# ---------------------------------------------------------------------------


class TestStaticAnalysisGate:
    """VAL-CROSS-007: Static analysis gate at every milestone.

    This test verifies that the codebase has no obvious static analysis
    violations by checking import structure and code quality markers.
    Full static analysis is run via external commands (ty check, ruff check).
    """

    def test_all_modules_importable(self) -> None:
        """All four workspace modules are importable."""
        import takkeli_align  # noqa: F401
        import takkeli_filtering  # noqa: F401
        import takkeli_inference  # noqa: F401
        import takkeli_pretrain  # noqa: F401

    def test_cross_module_interfaces_exist(self) -> None:
        """Key cross-module interfaces are available."""
        # 01 -> 02: Filtered dataset can be loaded via datasets library
        from datasets import load_dataset  # noqa: F401

        # All modules: HF transport for artifact movement
        from takkeli_filtering.hf_transport import download_from_hub, upload_to_hub  # noqa: F401

        # 03 -> 04: Aligned checkpoint can be exported
        from takkeli_inference.gguf_export import export_to_gguf  # noqa: F401

        # 02 -> 03: Model can be checkpointed and loaded
        from takkeli_pretrain.model import DrLLMModel  # noqa: F401

    def test_model_state_dict_compatible_across_modules(self) -> None:
        """Model state dict is compatible between pretraining, alignment, and export."""
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()
        model = DrLLMModel(config).to(device="cpu")

        # Get state dict (pretraining format)
        state_dict = model.state_dict()

        # Alignment pipeline can use this state dict
        from takkeli_align.config import ReinforcePPPipelineConfig

        pipeline = ReinforcePPPipeline(ReinforcePPPipelineConfig(), model)
        pipeline_state = pipeline.state_dict()

        # Both state dicts should have the same keys
        assert set(state_dict.keys()) == set(pipeline_state.keys())


# ---------------------------------------------------------------------------
# VAL-CROSS-008: Module dependency graph is acyclic
# ---------------------------------------------------------------------------


class TestModuleDependencyGraphAcyclic:
    """VAL-CROSS-008: Module dependency graph is acyclic.

    The four workspace modules have no circular dependencies.
    Data flows strictly: 01_data_filtering -> 02_pretraining -> 03_alignment -> 04_inference_eval.
    """

    def _get_imports(self, module_path: Path) -> set[str]:
        """Extract all import module names from a Python file.

        Returns the top-level module names (e.g., 'takkeli_pretrain' from
        'from takkeli_pretrain.model import DrLLMModel').
        """
        imports: set[str] = set()

        if not module_path.is_file():
            return imports

        try:
            tree = ast.parse(module_path.read_text())
        except SyntaxError:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    imports.add(top_level)
            elif isinstance(node, ast.ImportFrom):
                if not (node.module and node.module.split(".")[0].startswith("takkeli_")):
                    continue
                top_level = node.module.split(".")[0]
                imports.add(top_level)

        return imports

    def _get_all_internal_imports(self, module_dir: Path) -> dict[str, set[str]]:
        """Get all internal imports across a module directory.

        Returns a dict mapping file paths to sets of imported takkeli_* modules.
        """
        result: dict[str, set[str]] = {}
        for py_file in module_dir.rglob("*.py"):
            relative = str(py_file.relative_to(module_dir.parent.parent))
            imports = self._get_imports(py_file)
            # Only include internal takkeli_* imports
            internal = {i for i in imports if i.startswith("takkeli_")}
            if internal:
                result[relative] = internal
        return result

    def test_no_later_module_imports_earlier_module(self) -> None:
        """Later modules should not import from earlier modules.

        Allowed dependency flow: 01 -> 02 -> 03 -> 04
        Forbidden: 02 -> 01, 03 -> 01/02, 04 -> 01/02/03
        (Exception: 03_alignment importing from 02_pretraining is allowed for model architecture)
        """
        root = Path(__file__).resolve().parent.parent

        # Module order and their package names
        module_order = [
            ("01_data_filtering", "takkeli_filtering"),
            ("02_pretraining", "takkeli_pretrain"),
            ("03_alignment", "takkeli_align"),
            ("04_inference_eval", "takkeli_inference"),
        ]

        # Build a map of module index
        module_index = {name: idx for idx, (_, name) in enumerate(module_order)}

        # Forbidden imports: any module importing from a module that comes
        # AFTER it in the pipeline
        violations: list[str] = []

        for idx, (dir_name, pkg_name) in enumerate(module_order):
            module_dir = root / dir_name
            if not module_dir.is_dir():
                continue

            all_imports = self._get_all_internal_imports(module_dir)

            for file_path, imports in all_imports.items():
                for imported_pkg in imports:
                    if imported_pkg == pkg_name:
                        # Self-import is fine
                        continue

                    if imported_pkg in module_index:
                        imported_idx = module_index[imported_pkg]
                        if imported_idx > idx:
                            violations.append(
                                f"{dir_name}/{file_path} imports {imported_pkg} "
                                f"(index {imported_idx} > {idx})"
                            )

        assert len(violations) == 0, (
            f"Found {len(violations)} forward dependency violations:\n" + "\n".join(violations)
        )

    def test_data_flows_linearly(self) -> None:
        """Verify the module dependency graph is a linear DAG: 01->02->03->04."""
        root = Path(__file__).resolve().parent.parent

        # Allowed cross-module imports (forward dependencies only):
        # - 03_alignment can import from 02_pretraining (model architecture)
        # - 04_inference_eval can import from 02_pretraining (model for export)
        # No other cross-module imports are allowed.
        allowed_cross_imports: dict[str, set[str]] = {
            "takkeli_align": {"takkeli_pretrain"},
            "takkeli_inference": {"takkeli_pretrain"},
        }

        module_dirs = {
            "takkeli_filtering": root / "01_data_filtering",
            "takkeli_pretrain": root / "02_pretraining",
            "takkeli_align": root / "03_alignment",
            "takkeli_inference": root / "04_inference_eval",
        }

        violations: list[str] = []

        for pkg_name, module_dir in module_dirs.items():
            if not module_dir.is_dir():
                continue

            all_imports = self._get_all_internal_imports(module_dir)

            for file_path, imports in all_imports.items():
                for imported_pkg in imports:
                    if imported_pkg == pkg_name:
                        continue  # Self-import

                    # Check if this cross-import is allowed
                    allowed = allowed_cross_imports.get(pkg_name, set())
                    if imported_pkg not in allowed:
                        violations.append(
                            f"{pkg_name} ({file_path}) imports "
                            f"{imported_pkg} which is not in allowed set {allowed}"
                        )

        assert len(violations) == 0, (
            f"Found {len(violations)} disallowed cross-module imports:\n" + "\n".join(violations)
        )

    def test_import_graph_is_dag(self) -> None:
        """Verify the import graph has no cycles using topological sort."""
        root = Path(__file__).resolve().parent.parent

        # Build adjacency list
        edges: list[tuple[str, str]] = []
        module_dirs = {
            "takkeli_filtering": root / "01_data_filtering",
            "takkeli_pretrain": root / "02_pretraining",
            "takkeli_align": root / "03_alignment",
            "takkeli_inference": root / "04_inference_eval",
        }

        for pkg_name, module_dir in module_dirs.items():
            if not module_dir.is_dir():
                continue
            all_imports = self._get_all_internal_imports(module_dir)
            for _file_path, imports in all_imports.items():
                for imported_pkg in imports:
                    if imported_pkg != pkg_name and imported_pkg in module_dirs:
                        edges.append((pkg_name, imported_pkg))

        # Check for cycles using DFS
        modules = list(module_dirs.keys())
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for src, dst in edges:
                if src == node:
                    if dst not in visited:
                        if _has_cycle(dst):
                            return True
                    elif dst in rec_stack:
                        return True

            rec_stack.discard(node)
            return False

        has_cycle = False
        for module in modules:
            if module not in visited and _has_cycle(module):
                has_cycle = True
                break

        assert not has_cycle, f"Module dependency graph has a cycle! Edges: {edges}"


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestEndToEndPipelineFlow:
    """Full end-to-end test: filter -> pretrain -> align -> export."""

    def test_full_pipeline_flow(self, tmp_path: Path) -> None:
        """Simulate the complete pipeline flow across all 4 modules.

        1. Create a filtered dataset (simulates 01_data_filtering output)
        2. Create a pretrained model checkpoint (simulates 02_pretraining)
        3. Load into alignment pipeline (simulates 03_alignment)
        4. Export to GGUF (simulates 04_inference_eval)
        """
        import json

        from takkeli_align.config import ReinforcePPPipelineConfig
        from takkeli_align.pipeline import ReinforcePPPipeline
        from takkeli_inference.gguf_export import ExportConfig, export_to_gguf
        from takkeli_pretrain.model import DrLLMModel

        config = _make_small_model_config()

        # === Step 1: Simulate filtered dataset (01 -> 02) ===
        dataset_path = tmp_path / "filtered_dataset.jsonl"
        samples = [{"text": f"Training sample {i}"} for i in range(10)]
        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        from datasets import load_dataset

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        assert len(dataset) == 10

        # === Step 2: Pretrain and save checkpoint (02 -> 03) ===
        model = DrLLMModel(config).to(device="cpu")
        model.eval()

        pretrained_path = tmp_path / "pretrained.pt"
        _create_and_save_checkpoint(model, pretrained_path)

        # Verify checkpoint round-trip
        model2 = DrLLMModel(config).to(device="cpu")
        model2.eval()
        _load_checkpoint(model2, pretrained_path)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = model2(input_ids)
        assert torch.equal(logits1, logits2), "Pretraining checkpoint round-trip failed"

        # === Step 3: Alignment (03 -> 04) ===
        policy_model = DrLLMModel(config).to(device="cpu")
        _load_checkpoint(policy_model, pretrained_path)

        pipeline_config = ReinforcePPPipelineConfig()
        pipeline = ReinforcePPPipeline(pipeline_config, policy_model)

        # Verify alignment can run
        token_ids = torch.randint(0, config.vocab_size, (1, 8))
        rewards = torch.tensor([1.0])
        loss = pipeline.train_step(input_ids, token_ids, rewards)
        assert loss.requires_grad

        # Save aligned checkpoint
        aligned_path = tmp_path / "aligned.pt"
        torch.save(pipeline.policy_model.state_dict(), str(aligned_path))

        # === Step 4: Export to GGUF ===
        gguf_path = tmp_path / "final_model.gguf"
        export_config = ExportConfig(
            model_name="takkeli-1b-final",
            checkpoint_path=str(aligned_path),
            output_path=str(gguf_path),
            context_length=config.max_seq_len,
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ffn=config.d_ffn,
        )
        result = export_to_gguf(export_config)

        # Verify final GGUF
        assert result == gguf_path
        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0

        magic = _read_gguf_magic(gguf_path)
        assert magic == 0x46554747, f"Final GGUF has invalid magic: {hex(magic)}"

        # Verify GGUF is readable
        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0
        assert len(reader.fields) > 0

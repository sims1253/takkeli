"""Tests for GGUF export of BitNet ternary weights.

Validates:
- GGUF file validity (magic number, readable by gguf library)
- Ternary weight fidelity after dequantization
- Required metadata presence (model name, context length, embedding dim, vocab size)
- All model components are exported (BitLinear, embeddings, lm_head, norms)
- End-to-end export from a PyTorch model checkpoint
"""

from __future__ import annotations

import struct
from pathlib import Path

import gguf
import numpy as np
import pytest
from takkeli_inference.gguf_export import (
    ExportConfig,
    _apply_absmean_quantization,
    create_minimal_gguf,
    export_to_gguf,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _read_gguf_header(path: Path) -> int:
    """Read the magic number from a GGUF file.

    Args:
        path: Path to the GGUF file.

    Returns:
        Magic number as unsigned 32-bit integer.
    """
    with open(path, "rb") as f:
        data = f.read(4)
        return struct.unpack("<I", data)[0]


def _get_gguf_metadata(path: Path) -> dict[str, object]:
    """Read all metadata from a GGUF file.

    Args:
        path: Path to the GGUF file.

    Returns:
        Dictionary mapping metadata key names to values.
    """
    reader = gguf.GGUFReader(str(path))
    metadata: dict[str, object] = {}
    for key, field in reader.fields.items():
        # Skip internal GGUF fields
        if key.startswith("GGUF."):
            continue
        try:
            metadata[key] = field.contents()
        except Exception:
            metadata[key] = str(field.parts)
    return metadata


def _create_test_checkpoint(tmp_path: Path, tiny: bool = True) -> str:
    """Create a small PyTorch checkpoint with the full model architecture.

    Args:
        tmp_path: Temporary directory for the checkpoint.
        tiny: If True, create a very small model for fast testing.

    Returns:
        Path to the created checkpoint.
    """
    import torch

    vocab_size = 64 if tiny else 32000
    d_model = 32 if tiny else 2048
    n_layers = 2 if tiny else 24
    n_heads = 2 if tiny else 32
    d_ffn = 64 if tiny else 5504

    state_dict: dict[str, torch.Tensor] = {}

    # Token embeddings
    state_dict["token_embedding.weight"] = torch.randn(vocab_size, d_model)

    # Position embeddings
    max_seq_len = 128 if tiny else 2048
    state_dict["position_embedding.weight"] = torch.randn(max_seq_len, d_model)

    # LM head (not tied for test clarity)
    state_dict["lm_head.weight"] = torch.randn(vocab_size, d_model) * 0.02

    # Final RMSNorm
    state_dict["final_norm.gamma"] = torch.ones(d_model)

    d_head = d_model // n_heads
    d_kv_laten = 16 if tiny else 512
    d_q_laten = 16 if tiny else 512

    for i in range(n_layers):
        prefix = f"blocks.{i}"

        # Pre-attention norm
        state_dict[f"{prefix}.attn_norm.gamma"] = torch.ones(d_model)

        # MLA attention projections
        state_dict[f"{prefix}.attn.w_down_q.weight"] = torch.randn(d_q_laten, d_model) * 0.02
        state_dict[f"{prefix}.attn.w_down_kv.weight"] = torch.randn(d_kv_laten, d_model) * 0.02
        state_dict[f"{prefix}.attn.w_up_q.weight"] = torch.randn(n_heads * d_head, d_q_laten) * 0.02
        state_dict[f"{prefix}.attn.w_up_k.weight"] = (
            torch.randn(n_heads * d_head, d_kv_laten) * 0.02
        )
        state_dict[f"{prefix}.attn.w_up_v.weight"] = (
            torch.randn(n_heads * d_head, d_kv_laten) * 0.02
        )
        state_dict[f"{prefix}.attn.w_out.weight"] = torch.randn(d_model, n_heads * d_head) * 0.02
        state_dict[f"{prefix}.attn.norm.weight"] = torch.ones(d_model)

        # Pre-FFN norm
        state_dict[f"{prefix}.ffn_norm.gamma"] = torch.ones(d_model)

        # FFN BitLinear weights
        state_dict[f"{prefix}.ffn.w_gate.weight"] = torch.randn(d_ffn, d_model) * 0.02
        state_dict[f"{prefix}.ffn.w_up.weight"] = torch.randn(d_ffn, d_model) * 0.02
        state_dict[f"{prefix}.ffn.w_down.weight"] = torch.randn(d_model, d_ffn) * 0.02

        # Router (not exported to GGUF but in state dict)
        state_dict[f"{prefix}.router.down_proj.weight"] = torch.randn(16, d_model) * 0.02
        state_dict[f"{prefix}.router.down_proj.bias"] = torch.zeros(16)
        state_dict[f"{prefix}.router.up_proj.weight"] = torch.randn(3, 16) * 0.02
        state_dict[f"{prefix}.router.up_proj.bias"] = torch.zeros(3)

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(state_dict, str(checkpoint_path))
    return str(checkpoint_path)


# ---------------------------------------------------------------------------
# Unit tests: absmean quantization
# ---------------------------------------------------------------------------


class TestAbsmeanQuantization:
    """Tests for absmean quantization helper function."""

    def test_quantization_produces_ternary_values(self) -> None:
        """Quantized weights should only contain {-1, 0, 1}."""
        weight = np.random.randn(32, 64).astype(np.float32)
        quantized = _apply_absmean_quantization(weight)
        unique_vals = set(np.unique(quantized).tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_quantization_preserves_shape(self) -> None:
        """Quantized weights should have the same shape as input."""
        weight = np.random.randn(16, 32).astype(np.float32)
        quantized = _apply_absmean_quantization(weight)
        assert quantized.shape == weight.shape

    def test_quantization_all_zeros(self) -> None:
        """All-zero input should produce all-zero output."""
        weight = np.zeros((8, 16), dtype=np.float32)
        quantized = _apply_absmean_quantization(weight)
        assert np.all(quantized == 0.0)

    def test_quantization_all_same_value(self) -> None:
        """Uniform positive values should quantize to all 1.0."""
        weight = np.full((8, 16), 5.0, dtype=np.float32)
        quantized = _apply_absmean_quantization(weight)
        assert np.all(quantized == 1.0)

    def test_quantization_known_values(self) -> None:
        """Specific known input should produce expected ternary output."""
        weight = np.array([[0.5, 1.5, -0.3, 0.0]], dtype=np.float32)
        # gamma = mean(|W|) = (0.5 + 1.5 + 0.3 + 0.0) / 4 = 0.575
        # W/gamma = [0.870, 2.609, -0.522, 0.0]
        # round(W/gamma) clamped = [1, 1, -1, 0]
        quantized = _apply_absmean_quantization(weight)
        assert quantized[0, 0] == 1.0
        assert quantized[0, 1] == 1.0
        assert quantized[0, 2] == -1.0
        assert quantized[0, 3] == 0.0


# ---------------------------------------------------------------------------
# Unit tests: GGUF file validity
# ---------------------------------------------------------------------------


class TestGGUFValidity:
    """Tests for GGUF file format validity."""

    def test_magic_number(self, tmp_path: Path) -> None:
        """Output file should have GGUF magic number 0x46554747."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        magic = _read_gguf_header(gguf_path)
        assert magic == 0x46554747, f"Expected magic 0x46554747, got {hex(magic)}"

    def test_file_is_readable_by_gguf_library(self, tmp_path: Path) -> None:
        """Output file should be readable by the gguf library."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        reader = gguf.GGUFReader(str(gguf_path))
        assert reader is not None
        assert len(reader.fields) > 0
        assert len(reader.tensors) > 0

    def test_export_from_checkpoint_produces_valid_gguf(self, tmp_path: Path) -> None:
        """Export from a PyTorch checkpoint should produce a valid GGUF file."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        gguf_path = tmp_path / "exported.gguf"

        config = ExportConfig(
            model_name="test-model",
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            context_length=128,
            vocab_size=64,
            embedding_dim=32,
            n_layers=2,
            n_heads=2,
            d_ffn=64,
        )
        result = export_to_gguf(config)

        assert result == gguf_path
        assert gguf_path.exists()

        magic = _read_gguf_header(gguf_path)
        assert magic == 0x46554747

        reader = gguf.GGUFReader(str(gguf_path))
        assert len(reader.tensors) > 0


# ---------------------------------------------------------------------------
# Unit tests: metadata presence
# ---------------------------------------------------------------------------


class TestGGUFMetadata:
    """Tests for required metadata in GGUF file."""

    def _create_and_read(self, tmp_path: Path, **overrides: object) -> dict[str, object]:
        """Helper: create GGUF and return metadata."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), **overrides)  # type: ignore[arg-type]
        return _get_gguf_metadata(gguf_path)

    def test_model_name_present(self, tmp_path: Path) -> None:
        """GGUF should contain model name metadata."""
        metadata = self._create_and_read(tmp_path, model_name="takkeli-1b")
        assert "general.name" in metadata
        assert metadata["general.name"] == "takkeli-1b"

    def test_context_length_present(self, tmp_path: Path) -> None:
        """GGUF should contain context length metadata."""
        metadata = self._create_and_read(tmp_path, context_length=512)
        assert "bitnet.context_length" in metadata
        assert metadata["bitnet.context_length"] == 512

    def test_embedding_dim_present(self, tmp_path: Path) -> None:
        """GGUF should contain embedding dimension metadata."""
        metadata = self._create_and_read(tmp_path, embedding_dim=1024)
        assert "bitnet.embedding_length" in metadata
        assert metadata["bitnet.embedding_length"] == 1024

    def test_vocab_size_present(self, tmp_path: Path) -> None:
        """GGUF should contain vocabulary size metadata."""
        metadata = self._create_and_read(tmp_path, vocab_size=16000)
        assert "bitnet.vocab_size" in metadata
        assert metadata["bitnet.vocab_size"] == 16000

    def test_all_required_metadata_from_export(self, tmp_path: Path) -> None:
        """Export should include all required metadata keys."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        gguf_path = tmp_path / "exported.gguf"

        config = ExportConfig(
            model_name="takkeli-test",
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            context_length=256,
            vocab_size=128,
            embedding_dim=64,
            n_layers=2,
            n_heads=4,
            d_ffn=128,
        )
        export_to_gguf(config)

        metadata = _get_gguf_metadata(gguf_path)

        # Required metadata keys
        assert "general.name" in metadata
        assert metadata["general.name"] == "takkeli-test"
        assert "bitnet.context_length" in metadata
        assert metadata["bitnet.context_length"] == 256
        assert "bitnet.embedding_length" in metadata
        assert metadata["bitnet.embedding_length"] == 64
        assert "bitnet.vocab_size" in metadata
        assert metadata["bitnet.vocab_size"] == 128

    def test_architecture_metadata(self, tmp_path: Path) -> None:
        """GGUF should identify architecture as bitnet."""
        metadata = self._create_and_read(tmp_path)
        assert "general.architecture" in metadata
        assert metadata["general.architecture"] == "bitnet"

    def test_metadata_values_are_nonzero(self, tmp_path: Path) -> None:
        """Critical metadata values should be non-zero."""
        metadata = self._create_and_read(
            tmp_path,
            context_length=2048,
            embedding_dim=2048,
            vocab_size=32000,
        )
        assert int(metadata["bitnet.context_length"]) > 0  # type: ignore[arg-type]
        assert int(metadata["bitnet.embedding_length"]) > 0  # type: ignore[arg-type]
        assert int(metadata["bitnet.vocab_size"]) > 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests: ternary weight fidelity
# ---------------------------------------------------------------------------


class TestTernaryWeightFidelity:
    """Tests for ternary weight fidelity in GGUF."""

    def test_ffn_weights_are_ternary_after_dequantization(self, tmp_path: Path) -> None:
        """FFN weights stored in TQ1_0 should dequantize to ternary values."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        reader = gguf.GGUFReader(str(gguf_path))

        for i in range(len(reader.tensors)):
            tensor = reader.get_tensor(i)
            # Check FFN tensors (TQ1_0 quantized)
            if "ffn_" in tensor.name and tensor.tensor_type == gguf.GGMLQuantizationType.TQ1_0:
                dequantized = gguf.dequantize(tensor.data, gguf.GGMLQuantizationType.TQ1_0)
                # After dequantization, values should be close to {-1, 0, 1}
                rounded = np.round(dequantized)
                assert set(rounded.flatten().tolist()).issubset({-1.0, 0.0, 1.0}), (
                    f"Tensor {tensor.name} has non-ternary values after dequantization. "
                    f"Unique rounded values: {set(rounded.flatten().tolist())}"
                )

    def test_all_ffn_layers_have_ternary_weights(self, tmp_path: Path) -> None:
        """Every layer's FFN gate, up, and down should have ternary weights."""
        n_layers = 2
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), n_layers=n_layers)

        reader = gguf.GGUFReader(str(gguf_path))

        ffn_tensors_found = 0
        for i in range(len(reader.tensors)):
            tensor = reader.get_tensor(i)
            for layer_idx in range(n_layers):
                for ffn_type in ["ffn_gate", "ffn_up", "ffn_down"]:
                    expected_name = f"blk.{layer_idx}.{ffn_type}"
                    if tensor.name == expected_name:
                        ffn_tensors_found += 1
                        # Verify it's TQ1_0
                        assert tensor.tensor_type == gguf.GGMLQuantizationType.TQ1_0, (
                            f"Expected TQ1_0 for {expected_name}, got {tensor.tensor_type}"
                        )

        assert ffn_tensors_found == n_layers * 3, (
            f"Expected {n_layers * 3} FFN tensors, found {ffn_tensors_found}"
        )

    def test_ternary_quantization_round_trip(self) -> None:
        """Quantize then dequantize should preserve ternary values."""
        # TQ1_0 requires last dimension to be exactly QK_K (256)
        original = np.random.choice([-1.0, 0.0, 1.0], size=(4, 256), p=[0.3, 0.4, 0.3]).astype(
            np.float32
        )

        # Quantize to TQ1_0
        quantized = gguf.quantize(original, gguf.GGMLQuantizationType.TQ1_0)

        # Dequantize
        dequantized = gguf.dequantize(quantized, gguf.GGMLQuantizationType.TQ1_0)

        # Round to nearest integer (dequantization may introduce tiny float errors)
        rounded = np.round(dequantized)

        # All values should be {-1, 0, 1}
        unique_vals = set(rounded.flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"Round-trip produced non-ternary values: {unique_vals - {-1.0, 0.0, 1.0}}"
        )


# ---------------------------------------------------------------------------
# Unit tests: model component coverage
# ---------------------------------------------------------------------------


class TestModelComponentCoverage:
    """Tests that all model components are exported to GGUF."""

    def test_token_embedding_exported(self, tmp_path: Path) -> None:
        """Token embedding tensor should be present in GGUF."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), vocab_size=256)

        reader = gguf.GGUFReader(str(gguf_path))
        tensor_names = [reader.get_tensor(i).name for i in range(len(reader.tensors))]

        assert "token_embd" in tensor_names

    def test_output_norm_exported(self, tmp_path: Path) -> None:
        """Final RMSNorm should be present in GGUF."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        reader = gguf.GGUFReader(str(gguf_path))
        tensor_names = [reader.get_tensor(i).name for i in range(len(reader.tensors))]

        assert "output_norm" in tensor_names

    def test_lm_head_exported(self, tmp_path: Path) -> None:
        """LM head (output projection) should be present in GGUF."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        reader = gguf.GGUFReader(str(gguf_path))
        tensor_names = [reader.get_tensor(i).name for i in range(len(reader.tensors))]

        assert "output" in tensor_names

    def test_all_attention_layers_exported(self, tmp_path: Path) -> None:
        """All attention Q, K, V, output projections should be present."""
        n_layers = 3
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), n_layers=n_layers)

        reader = gguf.GGUFReader(str(gguf_path))
        tensor_names = {reader.get_tensor(i).name for i in range(len(reader.tensors))}

        expected_attn_tensors = []
        for i in range(n_layers):
            expected_attn_tensors.extend(
                [
                    f"blk.{i}.attn_norm",
                    f"blk.{i}.attn_q",
                    f"blk.{i}.attn_k",
                    f"blk.{i}.attn_v",
                    f"blk.{i}.attn_output",
                    f"blk.{i}.attn_sub_norm",
                ]
            )

        for name in expected_attn_tensors:
            assert name in tensor_names, f"Missing tensor: {name}"

    def test_all_ffn_layers_exported(self, tmp_path: Path) -> None:
        """All FFN gate, up, down projections should be present."""
        n_layers = 3
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), n_layers=n_layers)

        reader = gguf.GGUFReader(str(gguf_path))
        tensor_names = {reader.get_tensor(i).name for i in range(len(reader.tensors))}

        for i in range(n_layers):
            for ffn_type in ["ffn_norm", "ffn_gate", "ffn_up", "ffn_down"]:
                name = f"blk.{i}.{ffn_type}"
                assert name in tensor_names, f"Missing tensor: {name}"

    def test_embeddings_and_norms_are_f32(self, tmp_path: Path) -> None:
        """Embeddings and norm tensors should be F32, not quantized."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path))

        reader = gguf.GGUFReader(str(gguf_path))

        f32_tensors = ["token_embd", "output_norm", "output"]
        for i in range(len(reader.tensors)):
            tensor = reader.get_tensor(i)
            if tensor.name in f32_tensors:
                assert tensor.tensor_type == gguf.GGMLQuantizationType.F32, (
                    f"Expected F32 for {tensor.name}, got {tensor.tensor_type}"
                )

    def test_norm_tensors_are_f32(self, tmp_path: Path) -> None:
        """All layer norm tensors should be F32."""
        n_layers = 2
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(str(gguf_path), n_layers=n_layers)

        reader = gguf.GGUFReader(str(gguf_path))

        for i in range(len(reader.tensors)):
            tensor = reader.get_tensor(i)
            if "norm" in tensor.name:
                assert tensor.tensor_type == gguf.GGMLQuantizationType.F32, (
                    f"Expected F32 for norm tensor {tensor.name}, got {tensor.tensor_type}"
                )


# ---------------------------------------------------------------------------
# Unit tests: end-to-end export from checkpoint
# ---------------------------------------------------------------------------


class TestEndToEndExport:
    """End-to-end tests for the full export pipeline."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Export should create the output file."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        gguf_path = tmp_path / "model.gguf"

        config = ExportConfig(
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            vocab_size=64,
            embedding_dim=32,
            n_layers=2,
            n_heads=2,
            d_ffn=64,
        )
        export_to_gguf(config)

        assert gguf_path.exists()
        assert gguf_path.stat().st_size > 0

    def test_export_return_value(self, tmp_path: Path) -> None:
        """Export should return the output path."""
        checkpoint_path = _create_test_checkpoint(tmp_path)
        gguf_path = tmp_path / "model.gguf"

        config = ExportConfig(
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            vocab_size=64,
            embedding_dim=32,
            n_layers=2,
            n_heads=2,
            d_ffn=64,
        )
        result = export_to_gguf(config)
        assert result == gguf_path

    def test_export_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """Export should raise FileNotFoundError for missing checkpoint."""
        gguf_path = tmp_path / "model.gguf"

        config = ExportConfig(
            checkpoint_path="/nonexistent/checkpoint.pt",
            output_path=str(gguf_path),
        )
        with pytest.raises(FileNotFoundError):
            export_to_gguf(config)

    def test_export_tensor_count(self, tmp_path: Path) -> None:
        """Export should produce expected number of tensors."""
        n_layers = 2
        checkpoint_path = _create_test_checkpoint(tmp_path, tiny=True)
        gguf_path = tmp_path / "model.gguf"

        config = ExportConfig(
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            vocab_size=64,
            embedding_dim=32,
            n_layers=n_layers,
            n_heads=2,
            d_ffn=64,
        )
        export_to_gguf(config)

        reader = gguf.GGUFReader(str(gguf_path))

        # Expected tensors:
        # token_embd, pos_embd, output (lm_head), output_norm
        # Per layer: attn_norm, attn_q, attn_k, attn_v, attn_output, attn_sub_norm,
        #            ffn_norm, ffn_gate, ffn_up, ffn_down
        # = 4 + 2 * 10 = 24
        assert len(reader.tensors) == 4 + n_layers * 10

    def test_export_tensor_shapes(self, tmp_path: Path) -> None:
        """Exported tensor shapes should match expected dimensions."""
        vocab_size = 64
        d_model = 32
        n_layers = 2
        n_heads = 2
        d_ffn = 64

        checkpoint_path = _create_test_checkpoint(tmp_path, tiny=True)
        gguf_path = tmp_path / "model.gguf"

        config = ExportConfig(
            checkpoint_path=checkpoint_path,
            output_path=str(gguf_path),
            vocab_size=vocab_size,
            embedding_dim=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ffn=d_ffn,
        )
        export_to_gguf(config)

        reader = gguf.GGUFReader(str(gguf_path))

        tensor_map = {}
        for i in range(len(reader.tensors)):
            tensor = reader.get_tensor(i)
            tensor_map[tensor.name] = tensor

        # Token embeddings - GGUF stores in transposed form (d_model, vocab_size)
        assert tensor_map["token_embd"].shape.tolist() == [d_model, vocab_size]

        # Output norm - 1D, no transpose
        assert tensor_map["output_norm"].shape.tolist() == [d_model]

        # LM head - GGUF stores in transposed form (d_model, vocab_size)
        assert tensor_map["output"].shape.tolist() == [d_model, vocab_size]

        # Per-layer tensors
        for layer_idx in range(n_layers):
            assert tensor_map[f"blk.{layer_idx}.attn_norm"].shape.tolist() == [d_model]
            assert tensor_map[f"blk.{layer_idx}.attn_sub_norm"].shape.tolist() == [d_model]
            assert tensor_map[f"blk.{layer_idx}.ffn_norm"].shape.tolist() == [d_model]


# ---------------------------------------------------------------------------
# Unit tests: ExportConfig defaults
# ---------------------------------------------------------------------------


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self) -> None:
        """ExportConfig should have sensible defaults."""
        config = ExportConfig()
        assert config.model_name == "takkeli-1b"
        assert config.context_length == 2048
        assert config.vocab_size == 32000
        assert config.embedding_dim == 2048
        assert config.n_layers == 24
        assert config.n_heads == 32
        assert config.d_ffn == 5504

    def test_custom_values(self) -> None:
        """ExportConfig should accept custom values."""
        config = ExportConfig(
            model_name="custom-model",
            context_length=512,
            vocab_size=16000,
            embedding_dim=1024,
            n_layers=12,
        )
        assert config.model_name == "custom-model"
        assert config.context_length == 512
        assert config.vocab_size == 16000
        assert config.embedding_dim == 1024
        assert config.n_layers == 12

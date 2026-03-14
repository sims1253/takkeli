"""Tests for REINFORCE++ pipeline configuration.

Covers:
- Config construction with defaults and custom values
- Validation constraints (single GPU, no critic)
- Round-trip serialization (dict / JSON)
- YAML serialization (if pyyaml available)
- Clip range derivation from clip_range
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from takkeli_align.config import (
    HardwareConfig,
    ModelConfig,
    OptimizerConfig,
    ReinforcePPConfig,
    ReinforcePPPipelineConfig,
)

# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        cfg = ModelConfig()
        assert cfg.pretrained_model_name_or_path == "takkeli/takkeli-1b"
        assert cfg.max_seq_len == 2048

    def test_custom_values(self) -> None:
        cfg = ModelConfig(
            pretrained_model_name_or_path="custom/model",
            max_seq_len=1024,
        )
        assert cfg.pretrained_model_name_or_path == "custom/model"
        assert cfg.max_seq_len == 1024


# ---------------------------------------------------------------------------
# ReinforcePPConfig
# ---------------------------------------------------------------------------


class TestReinforcePPConfig:
    """Tests for ReinforcePPConfig."""

    def test_default_clip_range_derivation(self) -> None:
        """clip_range_low and clip_range_high should derive from clip_range."""
        cfg = ReinforcePPConfig(clip_range=0.2)
        assert cfg.clip_range_low == 0.8
        assert cfg.clip_range_high == 1.2

    def test_explicit_clip_bounds_preserved(self) -> None:
        """Explicitly set clip bounds should not be overridden."""
        cfg = ReinforcePPConfig(
            clip_range=0.2,
            clip_range_low=0.5,
            clip_range_high=1.5,
        )
        assert cfg.clip_range_low == 0.5
        assert cfg.clip_range_high == 1.5

    def test_global_normalization_default(self) -> None:
        """REINFORCE++ uses global (batch-wide) advantage normalization."""
        cfg = ReinforcePPConfig()
        assert cfg.normalize_advantage is True
        assert cfg.norm_adv_batch is True

    def test_no_kl_target_by_default(self) -> None:
        cfg = ReinforcePPConfig()
        assert cfg.kl_target == 0.0


# ---------------------------------------------------------------------------
# HardwareConfig
# ---------------------------------------------------------------------------


class TestHardwareConfig:
    """Tests for HardwareConfig single-GPU validation."""

    def test_default_single_gpu(self) -> None:
        cfg = HardwareConfig()
        assert cfg.n_gpus == 1
        assert cfg.tensor_parallel_size == 1

    def test_multi_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="n_gpus=1"):
            HardwareConfig(n_gpus=4)

    def test_tensor_parallel_raises(self) -> None:
        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            HardwareConfig(tensor_parallel_size=2)

    def test_custom_memory_budget(self) -> None:
        cfg = HardwareConfig(memory_budget_gb=16.0)
        assert cfg.memory_budget_gb == 16.0


# ---------------------------------------------------------------------------
# ReinforcePPPipelineConfig
# ---------------------------------------------------------------------------


class TestReinforcePPPipelineConfig:
    """Tests for the top-level pipeline configuration."""

    def test_default_construction(self) -> None:
        cfg = ReinforcePPPipelineConfig()
        assert cfg.use_critic is False
        assert cfg.seed == 42
        assert cfg.hardware.n_gpus == 1
        assert cfg.algorithm.clip_range == 0.2

    def test_use_critic_raises(self) -> None:
        with pytest.raises(ValueError, match="critic-free"):
            ReinforcePPPipelineConfig(use_critic=True)

    def test_no_critic_model(self) -> None:
        """Verify config explicitly says no critic (VAL-ALIGN-003)."""
        cfg = ReinforcePPPipelineConfig()
        assert cfg.use_critic is False

    def test_single_gpu_config(self) -> None:
        """Verify single-GPU configuration (VAL-ALIGN-002)."""
        cfg = ReinforcePPPipelineConfig()
        assert cfg.hardware.n_gpus == 1
        assert cfg.hardware.tensor_parallel_size == 1

    def test_to_dict_round_trip(self) -> None:
        """Config -> dict -> Config round-trip preserves values."""
        cfg = ReinforcePPPipelineConfig(
            seed=123,
            output_dir="/tmp/output",
            run_name="test-run",
        )
        d = cfg.to_dict()
        restored = ReinforcePPPipelineConfig.from_dict(d)
        assert restored.seed == 123
        assert restored.output_dir == "/tmp/output"
        assert restored.run_name == "test-run"
        assert restored.use_critic is False
        assert restored.hardware.n_gpus == 1

    def test_load_json_round_trip(self) -> None:
        """Config -> JSON file -> Config round-trip."""
        cfg = ReinforcePPPipelineConfig(
            seed=99,
            output_dir="/custom",
            run_name="json-test",
            algorithm=ReinforcePPConfig(kl_coeff=0.5, clip_range=0.1),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
            json.dump(cfg.to_dict(), f)

        loaded = ReinforcePPPipelineConfig.load_json(path)
        assert loaded.seed == 99
        assert loaded.output_dir == "/custom"
        assert loaded.algorithm.kl_coeff == 0.5
        assert loaded.algorithm.clip_range_low == 0.9
        assert loaded.algorithm.clip_range_high == 1.1
        path.unlink()

    def test_load_yaml_round_trip(self) -> None:
        """Config -> YAML file -> Config round-trip (if pyyaml available)."""
        pytest.importorskip("yaml")

        cfg = ReinforcePPPipelineConfig(
            seed=7,
            run_name="yaml-test",
            algorithm=ReinforcePPConfig(kl_coeff=0.2),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        cfg.save_yaml(path)
        loaded = ReinforcePPPipelineConfig.load_yaml(path)
        assert loaded.seed == 7
        assert loaded.run_name == "yaml-test"
        assert loaded.algorithm.kl_coeff == 0.2
        path.unlink()

    def test_openrlhf_compatible_structure(self) -> None:
        """Config has the structure expected by OpenRLHF (VAL-ALIGN-001).

        The config must be loadable as a dict without errors.
        """
        cfg = ReinforcePPPipelineConfig()
        d = cfg.to_dict()

        # Verify top-level keys exist
        assert "model" in d
        assert "algorithm" in d
        assert "hardware" in d
        assert "optimizer" in d
        assert "use_critic" in d

        # Verify model keys
        assert "pretrained_model_name_or_path" in d["model"]
        assert "max_seq_len" in d["model"]

        # Verify algorithm keys
        algo = d["algorithm"]
        for key in (
            "kl_coeff",
            "clip_range",
            "clip_range_low",
            "clip_range_high",
            "normalize_advantage",
            "norm_adv_batch",
        ):
            assert key in algo, f"Missing algorithm key: {key}"

        # Config from dict should not raise
        loaded = ReinforcePPPipelineConfig.from_dict(d)
        assert loaded is not None


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_values(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.learning_rate == 1e-6
        assert cfg.optim == "adamw_torch"
        assert cfg.lr_scheduler_type == "cosine"

    def test_custom_values(self) -> None:
        cfg = OptimizerConfig(learning_rate=5e-7, max_grad_norm=0.5)
        assert cfg.learning_rate == 5e-7
        assert cfg.max_grad_norm == 0.5

"""Unit tests for SAE model loading, inference, and thresholding.

Covers validation assertions:
  VAL-DATA-001  – SAE and base model load on CPU without error
  VAL-DATA-002  – SAE inference produces (batch, seq_len, n_sae_features)
  VAL-DATA-003  – Configurable feature index selection and thresholding
  VAL-DATA-004  – Threshold logic: True if ANY feature exceeds threshold
  VAL-DATA-008  – Shape invariance under batch size
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch
from takkeli_filtering.config import FilterConfig, PipelineConfig, SAEConfig
from takkeli_filtering.sae_inference import run_sae_inference, should_filter

# Skip base model tests when HF_TOKEN is not available (gated model access).
_skip_no_hf_token = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set; gated Gemma 3 4B IT model requires authentication",
)

# ---------------------------------------------------------------------------
# Config dataclass tests
# ---------------------------------------------------------------------------


class TestSAEConfig:
    """Tests for SAEConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = SAEConfig()
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"
        assert cfg.hook_layer == 22
        assert cfg.model_name == "google/gemma-3-4b-it"

    def test_custom_values(self) -> None:
        cfg = SAEConfig(
            sae_release="custom/repo",
            sae_id="layer_10/width_16k/canonical",
            hook_layer=10,
            device="cpu",
            dtype="float32",
            model_name="google/gemma-3-4b-it",
        )
        assert cfg.sae_release == "custom/repo"
        assert cfg.sae_id == "layer_10/width_16k/canonical"
        assert cfg.hook_layer == 10

    def test_frozen(self) -> None:
        cfg = SAEConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.device = "cuda"  # type: ignore[misc]


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = FilterConfig()
        assert cfg.feature_indices == ()
        assert cfg.threshold == 0.0

    def test_custom_values(self) -> None:
        cfg = FilterConfig(feature_indices=(1, 5, 10), threshold=0.5)
        assert cfg.feature_indices == (1, 5, 10)
        assert cfg.threshold == 0.5

    def test_frozen(self) -> None:
        cfg = FilterConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.threshold = 1.0  # type: ignore[misc]

    def test_empty_indices_no_filter(self) -> None:
        """Empty feature_indices should never trigger filtering."""
        acts = torch.ones(1, 10, 100)
        cfg = FilterConfig(feature_indices=(), threshold=0.0)
        assert should_filter(acts, cfg) is False


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert isinstance(cfg.sae, SAEConfig)
        assert isinstance(cfg.filter, FilterConfig)
        assert cfg.batch_size == 8

    def test_custom(self) -> None:
        sae_cfg = SAEConfig(hook_layer=5)
        filt_cfg = FilterConfig(feature_indices=(3,), threshold=0.7)
        cfg = PipelineConfig(sae=sae_cfg, filter=filt_cfg, batch_size=4)
        assert cfg.sae.hook_layer == 5
        assert cfg.filter.feature_indices == (3,)
        assert cfg.batch_size == 4


# ---------------------------------------------------------------------------
# SAE loading tests (VAL-DATA-001)
# ---------------------------------------------------------------------------


class TestSAELoading:
    """Tests for SAE model loading on CPU.

    NOTE: These tests download ~200 MB of SAE weights on first run.
    They require network access and the ``sae-lens`` package.
    """

    @pytest.fixture(scope="class")
    def sae_config(self) -> SAEConfig:
        return SAEConfig(
            sae_release="gemma-scope-2-4b-it-resid_post",
            sae_id="layer_22_width_262k_l0_medium",
            hook_layer=22,
            device="cpu",
            dtype="float32",
        )

    @pytest.fixture(scope="class")
    def loaded_sae(self, sae_config: SAEConfig) -> Any:
        from takkeli_filtering.sae_loader import load_sae

        return load_sae(sae_config)

    def test_sae_loads_without_error(self, loaded_sae: Any) -> None:
        """VAL-DATA-001: SAE weights load into memory on CPU without error."""
        assert loaded_sae is not None

    def test_sae_has_expected_config(self, loaded_sae: Any) -> None:
        """SAE should have d_in and d_sae attributes."""
        assert hasattr(loaded_sae.cfg, "d_in")
        assert hasattr(loaded_sae.cfg, "d_sae")
        assert loaded_sae.cfg.d_in > 0
        assert loaded_sae.cfg.d_sae > 0
        # Gemma 3 4B IT has d_model=2048
        assert loaded_sae.cfg.d_in == 2048

    def test_sae_has_encoder_weights(self, loaded_sae: Any) -> None:
        """SAE encoder weights should be loadable."""
        assert hasattr(loaded_sae, "W_enc")
        assert loaded_sae.W_enc.shape[0] == loaded_sae.cfg.d_in
        assert loaded_sae.W_enc.shape[1] == loaded_sae.cfg.d_sae


# ---------------------------------------------------------------------------
# SAE inference tests (VAL-DATA-002, VAL-DATA-008)
# ---------------------------------------------------------------------------


class TestSAEInference:
    """Tests for SAE feature activation extraction."""

    @pytest.fixture(scope="class")
    def sae_instance(self) -> Any:
        from takkeli_filtering.sae_loader import load_sae

        cfg = SAEConfig(
            sae_release="gemma-scope-2-4b-it-resid_post",
            sae_id="layer_22_width_262k_l0_medium",
            hook_layer=22,
            device="cpu",
            dtype="float32",
        )
        return load_sae(cfg)

    def test_encode_shape_batch_seq_features(
        self,
        sae_instance: Any,
    ) -> None:
        """VAL-DATA-002: SAE inference produces (batch, seq_len, n_sae_features)."""
        batch, seq_len = 2, 32
        d_model = sae_instance.cfg.d_in
        n_sae_features = sae_instance.cfg.d_sae

        activations = torch.randn(batch, seq_len, d_model)
        feature_acts = run_sae_inference(sae_instance, activations)

        assert feature_acts.shape == (batch, seq_len, n_sae_features)

    def test_activation_shape_invariant_batch_size(
        self,
        sae_instance: Any,
    ) -> None:
        """VAL-DATA-008: Feature dimension is invariant under batch size."""
        n_sae_features = sae_instance.cfg.d_sae
        d_model = sae_instance.cfg.d_in

        for batch_size in (1, 4, 8, 16):
            activations = torch.randn(batch_size, 16, d_model)
            feature_acts = run_sae_inference(sae_instance, activations)
            assert feature_acts.shape[-1] == n_sae_features, (
                f"Batch size {batch_size}: expected last dim {n_sae_features}, "
                f"got {feature_acts.shape[-1]}"
            )

    def test_batch_size_1_shape(self, sae_instance: Any) -> None:
        """Batch size 1 should produce correct (1, seq_len, d_sae) shape."""
        d_model = sae_instance.cfg.d_in
        n_sae_features = sae_instance.cfg.d_sae
        activations = torch.randn(1, 10, d_model)
        feature_acts = run_sae_inference(sae_instance, activations)
        assert feature_acts.shape == (1, 10, n_sae_features)

    def test_single_token_shape(self, sae_instance: Any) -> None:
        """Single token (seq_len=1) should still produce correct shape."""
        d_model = sae_instance.cfg.d_in
        n_sae_features = sae_instance.cfg.d_sae
        activations = torch.randn(1, 1, d_model)
        feature_acts = run_sae_inference(sae_instance, activations)
        assert feature_acts.shape == (1, 1, n_sae_features)


# ---------------------------------------------------------------------------
# Threshold logic tests (VAL-DATA-003, VAL-DATA-004)
# ---------------------------------------------------------------------------


class TestThresholdLogic:
    """Tests for the should_filter threshold function.

    Uses mock activation tensors so no model download is needed.
    """

    def test_exceeds_threshold_returns_true(self) -> None:
        """VAL-DATA-004: [0.1, 0.9, 0.2], indices=[1], threshold=0.5 -> True."""
        acts = torch.tensor([[[0.1, 0.9, 0.2]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(1,), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_below_threshold_returns_false(self) -> None:
        """VAL-DATA-004: [0.1, 0.9, 0.2], indices=[1], threshold=0.95 -> False."""
        acts = torch.tensor([[[0.1, 0.9, 0.2]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(1,), threshold=0.95)
        assert should_filter(acts, cfg) is False

    def test_any_index_exceeds_returns_true(self) -> None:
        """If any monitored index exceeds threshold, return True."""
        # Feature 2 is above threshold
        acts = torch.tensor([[[0.1, 0.3, 0.8]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0, 1, 2), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_no_index_exceeds_returns_false(self) -> None:
        """If no monitored index exceeds threshold, return False."""
        acts = torch.tensor([[[0.1, 0.3, 0.4]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0, 1, 2), threshold=0.5)
        assert should_filter(acts, cfg) is False

    def test_empty_indices_always_false(self) -> None:
        """Empty feature indices should never filter."""
        acts = torch.ones(1, 10, 100)
        cfg = FilterConfig(feature_indices=(), threshold=0.0)
        assert should_filter(acts, cfg) is False

    def test_exact_threshold_is_not_exceeded(self) -> None:
        """Value exactly at threshold does NOT trigger (uses >, not >=)."""
        acts = torch.tensor([[[0.5]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0,), threshold=0.5)
        assert should_filter(acts, cfg) is False

    def test_slightly_above_threshold_triggers(self) -> None:
        """Value slightly above threshold triggers."""
        acts = torch.tensor([[[0.5001]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0,), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_batch_dimension_checked(self) -> None:
        """Threshold check works across batch dimension."""
        # Only the second item in the batch exceeds threshold
        acts = torch.tensor(
            [
                [[0.1, 0.2]],  # batch 0: no exceed
                [[0.1, 0.9]],  # batch 1: exceeds
            ],
            dtype=torch.float32,
        )
        cfg = FilterConfig(feature_indices=(1,), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_seq_dimension_checked(self) -> None:
        """Threshold check works across sequence dimension."""
        # Only the last token exceeds threshold
        acts = torch.tensor(
            [
                [
                    [0.1],  # pos 0: below
                    [0.1],  # pos 1: below
                    [0.9],  # pos 2: exceeds
                ],
            ],
            dtype=torch.float32,
        )
        cfg = FilterConfig(feature_indices=(0,), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_2d_input(self) -> None:
        """Should handle 2D input (seq_len, n_features)."""
        acts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.8, 0.3],
            ],
            dtype=torch.float32,
        )
        cfg = FilterConfig(feature_indices=(1,), threshold=0.5)
        assert should_filter(acts, cfg) is True

    def test_multiple_indices_all_below(self) -> None:
        """Multiple indices all below threshold -> False."""
        acts = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)
        cfg = FilterConfig(feature_indices=(0, 1, 2, 3), threshold=0.5)
        assert should_filter(acts, cfg) is False

    def test_configurable_indices_and_threshold(self) -> None:
        """VAL-DATA-003: Feature indices and threshold are configurable."""
        acts = torch.tensor([[[0.1, 0.8, 0.3, 0.9]]], dtype=torch.float32)

        # Only monitoring index 0 (0.1) -> below threshold
        cfg = FilterConfig(feature_indices=(0,), threshold=0.5)
        assert should_filter(acts, cfg) is False

        # Monitoring index 3 (0.9) -> above threshold
        cfg = FilterConfig(feature_indices=(3,), threshold=0.5)
        assert should_filter(acts, cfg) is True

        # Monitoring indices 0 and 3 -> at least one above threshold
        cfg = FilterConfig(feature_indices=(0, 3), threshold=0.5)
        assert should_filter(acts, cfg) is True


# ---------------------------------------------------------------------------
# Base model loading tests (VAL-DATA-001)
# ---------------------------------------------------------------------------


@_skip_no_hf_token
class TestBaseModelLoading:
    """Tests for loading a HuggingFace base model.

    NOTE: Downloads ~5 GB on first run. Requires network access.
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self) -> Any:
        from takkeli_filtering.sae_loader import load_base_model

        cfg = SAEConfig(
            sae_release="gemma-scope-2-4b-it-resid_post",
            sae_id="layer_22_width_262k_l0_medium",
            hook_layer=22,
            device="cpu",
            dtype="float32",
            model_name="google/gemma-3-4b-it",
        )
        return load_base_model(cfg)

    def test_model_loads_without_error(self, model_and_tokenizer: Any) -> None:
        """VAL-DATA-001: Base model loads on CPU without error."""
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_model_is_eval_mode(self, model_and_tokenizer: Any) -> None:
        """Model should be in eval mode after loading."""
        model, _ = model_and_tokenizer
        assert not model.training

    def test_tokenizer_works(self, model_and_tokenizer: Any) -> None:
        """Tokenizer should encode and decode text."""
        _, tokenizer = model_and_tokenizer
        ids = tokenizer("Hello, world!", return_tensors="pt")
        assert ids["input_ids"].dim() == 2
        decoded = tokenizer.decode(ids["input_ids"][0])
        assert "Hello" in decoded

    def test_model_forward_pass(self, model_and_tokenizer: Any) -> None:
        """Model forward pass should work without error."""
        model, tokenizer = model_and_tokenizer
        ids = tokenizer("Test input", return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids=ids["input_ids"])
        assert output.logits.shape[0] == 1  # batch size


# ---------------------------------------------------------------------------
# Activation extraction tests
# ---------------------------------------------------------------------------


@_skip_no_hf_token
class TestActivationExtraction:
    """Tests for extracting hidden states from the base model."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self) -> Any:
        from takkeli_filtering.sae_loader import load_base_model

        cfg = SAEConfig(
            sae_release="gemma-scope-2-4b-it-resid_post",
            sae_id="layer_22_width_262k_l0_medium",
            hook_layer=22,
            device="cpu",
            dtype="float32",
            model_name="google/gemma-3-4b-it",
        )
        return load_base_model(cfg)

    def test_extract_activations_shape(
        self,
        model_and_tokenizer: Any,
    ) -> None:
        """Activations extracted should have shape (batch, seq_len, d_model)."""
        model, tokenizer = model_and_tokenizer
        ids = tokenizer("Hello world", return_tensors="pt")
        activations = extract_activations_test(model, ids["input_ids"], layer=22)
        assert activations.dim() == 3
        assert activations.shape[0] == 1  # batch
        assert activations.shape[1] > 0  # seq_len
        assert activations.shape[2] == 2048  # d_model for gemma-3-4b-it

    def test_extract_activations_batch(self, model_and_tokenizer: Any) -> None:
        """Batch extraction should produce correct batch dimension."""
        model, tokenizer = model_and_tokenizer
        texts = ["Hello", "World"]
        ids = tokenizer(texts, return_tensors="pt", padding=True)
        activations = extract_activations_test(model, ids["input_ids"], layer=22)
        assert activations.shape[0] == 2


def extract_activations_test(
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Helper to extract activations for testing (avoids import issues)."""
    from takkeli_filtering.sae_loader import extract_activations

    return extract_activations(model, input_ids, layer)

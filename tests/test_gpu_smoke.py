"""GPU smoke tests — verify CUDA is available and basic operations work."""

import os
import tempfile

import pytest
import torch

gpu = pytest.mark.gpu


class TestGPUSmoke:
    @gpu
    def test_cuda_available(self):
        assert torch.cuda.is_available(), "CUDA not available — check driver and torch version"

    @gpu
    def test_cuda_device_name(self):
        assert torch.cuda.is_available()
        name = torch.cuda.get_device_name(0)
        assert isinstance(name, str) and len(name) > 0

    @gpu
    def test_cuda_tensor_creation(self):
        assert torch.cuda.is_available()
        x = torch.randn(4, 4, device="cuda")
        assert x.device.type == "cuda"

    @gpu
    def test_cuda_matmul(self):
        assert torch.cuda.is_available()
        x = torch.randn(10, 10, device="cuda")
        y = x @ x.T
        assert y.shape == (10, 10)
        expected = x.cpu() @ x.cpu().T
        assert torch.allclose(y.cpu(), expected, atol=1e-5)

    @gpu
    def test_cuda_pickle_round_trip(self):
        """Verify torch.save/load works with CUDA tensors (Python 3.13 compat)."""
        assert torch.cuda.is_available()
        x = torch.randn(3, 3, device="cuda")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save({"tensor": x}, path)
            loaded = torch.load(path, weights_only=False)
            assert torch.equal(loaded["tensor"], x)
        finally:
            os.unlink(path)

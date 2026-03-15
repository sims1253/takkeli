"""NorMuon optimizer: Neuron-wise Normalized Muon.

Combines Muon's Newton-Schulz orthogonalization with row-wise second-order
momentum tracking for balanced per-neuron update magnitudes.

Reference: arXiv:2510.05491 — "NorMuon: Making Muon more efficient and scalable"
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer


@dataclass
class NorMuonConfig:
    """Configuration for the NorMuon optimizer.

    Attributes:
        lr: Learning rate (default 0.02, following Muon convention).
        momentum: First-order momentum decay (beta1). Default 0.95.
        beta2: Second-order momentum decay for neuron-wise statistics. Default 0.95.
        weight_decay: Decoupled weight decay coefficient. Default 0.
        ns_steps: Number of Newton-Schulz iterations. Default 5.
        nesterov: Whether to use Nesterov-style momentum. Default True.
        eps: Perturbation parameter for numerical stability. Default 1e-7.
    """

    lr: float = 0.02
    momentum: float = 0.95
    beta2: float = 0.95
    weight_decay: float = 0.0
    ns_steps: int = 5
    nesterov: bool = True
    eps: float = 1e-7


# Newton-Schulz coefficients optimized for fast convergence.
# These are the quintic iteration coefficients from Muon.
_NS_A: float = 3.4445
_NS_B: float = -4.7750
_NS_C: float = 2.0315


def newton_schulz_orthogonalize(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via Newton-Schulz iteration.

    Computes the approximate polar factor (nearest semi-orthogonal matrix)
    of the input matrix using quintic Newton-Schulz iterations.

    The iteration is: X_{k} = a * X_{k-1} + b * (X_{k-1} @ X_{k-1}^T) @ X_{k-1}
                                    + c * (X_{k-1} @ X_{k-1}^T)^2 @ X_{k-1}

    Before iteration, the input is Frobenius-normalized to ensure spectral norm ≤ 1.
    If the input has more rows than columns, the iteration operates on the
    transpose and transposes the result back.

    Args:
        g: 2D input tensor of shape (m, n).
        steps: Number of Newton-Schulz iterations. Default 5.
        eps: Small constant for numerical stability in normalization. Default 1e-7.

    Returns:
        Approximate orthogonal matrix of same shape (m, n).
    """
    assert g.ndim == 2, f"newton_schulz_orthogonalize requires 2D input, got {g.ndim}D"
    a, b, c = _NS_A, _NS_B, _NS_C

    x = g.bfloat16()

    # Transpose if tall matrix to ensure n_rows ≤ n_cols for stability
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.mT

    # Frobenius normalize so spectral norm is at most 1
    x = x / (x.norm() + eps)

    # Newton-Schulz iterations
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.mT

    return x.to(g.dtype)


def compute_orthogonality_metric(w: torch.Tensor) -> torch.Tensor:
    """Compute orthogonality deviation ||W^T W - I||_F for a 2D matrix.

    A perfectly orthogonal matrix (with W^T W = I) has metric 0.

    Args:
        w: 2D weight matrix of shape (m, n).

    Returns:
        Scalar Frobenius norm of (W^T W - I).
    """
    assert w.ndim == 2, f"compute_orthogonality_metric requires 2D input, got {w.ndim}D"
    m, n = w.shape
    target = torch.eye(n, device=w.device, dtype=w.dtype)
    if m < n:
        # W W^T should be close to I for wide matrices
        target = torch.eye(m, device=w.device, dtype=w.dtype)
        diff = w @ w.mT - target
    else:
        diff = w.mT @ w - target
    return diff.norm()


class NorMuon(Optimizer):
    """NorMuon: Neuron-wise Normalized Muon optimizer.

    Combines Muon's Newton-Schulz orthogonalization of the momentum update
    with row-wise (neuron-wise) second-order momentum tracking. This yields
    balanced per-neuron update magnitudes while preserving the conditioning
    benefits of orthogonalization.

    Only applies Newton-Schulz orthogonalization to 2D parameter matrices.
    1D parameters (biases, layer norms, etc.) are updated via standard
    SGD momentum without orthogonalization.

    Algorithm per step for a 2D parameter W with gradient G:
        1. Update first-order momentum: M = beta1 * M + (1 - beta1) * G
        2. Nesterov look-ahead (optional): U = G + beta1 * M
        3. Orthogonalize: O = NS5(U)
        4. Track row-wise second-order stats: v = beta2 * v + (1 - beta2) * mean_cols(O^2)
        5. Row-wise normalize: O_hat = O / (sqrt(v) + eps)
        6. Rescale to preserve norm: O_hat = O_hat * ||O||_F / ||O_hat||_F
        7. Apply aspect-ratio scaling: O_hat *= max(1, m/n)^0.5
        8. Update: W = W * (1 - lr * wd) - lr * O_hat

    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        lr: Learning rate. Default 0.02.
        momentum: First-order momentum coefficient (beta1). Default 0.95.
        beta2: Second-order momentum coefficient for neuron stats. Default 0.95.
        weight_decay: Decoupled weight decay. Default 0.
        ns_steps: Newton-Schulz iterations. Default 5.
        nesterov: Use Nesterov-style momentum. Default True.
        eps: Numerical stability constant. Default 1e-7.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor | torch.nn.Parameter] | list[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
        eps: float = 1e-7,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: ANN401
        """Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss.

        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if p.dim() == 1:
                    # 1D parameters (biases, etc.): simple SGD momentum
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)

                    state["momentum"].lerp_(grad, 1 - momentum)
                    if nesterov:
                        update = grad.lerp_(state["momentum"], momentum)
                    else:
                        update = state["momentum"].clone()

                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(update, alpha=-lr)
                else:
                    # 2D parameters: full NorMuon with orthogonalization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        # Row-wise second-order momentum: shape (m, 1) for (m, n) param
                        state["row_momentum"] = torch.zeros(
                            p.size(0), 1, device=p.device, dtype=p.dtype
                        )

                    m_buf = state["momentum"]
                    v_buf = state["row_momentum"]

                    # Step 1: Update first-order momentum
                    m_buf.lerp_(grad, 1 - momentum)

                    # Step 2: Nesterov look-ahead
                    update = grad.lerp_(m_buf, momentum) if nesterov else m_buf.clone()

                    # Step 3: Newton-Schulz orthogonalization
                    update = newton_schulz_orthogonalize(update, steps=ns_steps, eps=eps)
                    update = update.to(grad.dtype)

                    # Step 4-6: Row-wise normalization with norm preservation
                    vnorm = update.norm()
                    # mean of squared values per row → shape (m, 1)
                    v_mean = (update * update).mean(dim=-1, keepdim=True)
                    v_buf.lerp_(v_mean, 1 - beta2)

                    # Divide each row by sqrt of its second-order statistic
                    step_size = 1.0 / (v_buf.sqrt().add_(eps))
                    update.mul_(step_size)

                    # Rescale to preserve the original update norm
                    vnorm_new = update.norm()
                    update.mul_(vnorm / (vnorm_new + eps))

                    # Step 7: Aspect ratio scaling
                    m, n = p.shape
                    update.mul_(max(1.0, m / n) ** 0.5)

                    # Step 8: Apply update
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(update, alpha=-lr)

        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the optimizer state."""
        super().load_state_dict(state_dict)

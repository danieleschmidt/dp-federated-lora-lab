"""
Differentially Private SGD (DP-SGD) implementation.

Algorithm (Abadi et al., 2016 — "Deep Learning with Differential Privacy"):
  For each mini-batch:
    1. Compute per-sample gradients
    2. Clip each per-sample gradient to L2 norm ≤ C  (clipping threshold)
    3. Average clipped gradients over the batch
    4. Add Gaussian noise ~ N(0, (σ·C)²) to the average
    5. Update parameters with the noisy gradient

Privacy accounting uses Rényi Differential Privacy (RDP) moments accountant:
  - Each step consumes ε_R(α) = α / (2σ²) RDP at order α
  - After k steps: ε_R(α) = k · α / (2σ²)
  - Convert to (ε, δ)-DP via:  ε = ε_R(α) + log(1/δ) / (α - 1)

Reference: Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RDP Moments Accountant
# ---------------------------------------------------------------------------

_RDP_ORDERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 48, 64]


def _rdp_gaussian_per_step(order: float, noise_multiplier: float, sample_rate: float) -> float:
    """
    RDP guarantee of one step of Gaussian mechanism with subsampling.

    Uses the simple Poisson subsampling amplification bound (Mironov et al. 2019):
        ε_R(α) ≈ (1/(α-1)) * log(1 + q²·α·(α-1)/(2σ²) * (e^{1/σ²} - 1)) + ...

    For simplicity and correctness, we use the well-known closed-form upper bound
    for *without* subsampling (q = 1, conservative) and divide by dataset size when
    sample_rate < 1 using the first-order Poisson amplification:
        ε_R(α) ≤ q · α / (2σ²)   (valid for small q, standard bound)

    This is conservative but analytically clean.
    """
    if noise_multiplier == 0:
        return float("inf")
    # Full-batch (q=1) RDP for Gaussian: ε(α) = α / (2σ²)
    rdp_full = order / (2.0 * noise_multiplier**2)
    # Poisson subsampling amplification (first-order, conservative)
    return sample_rate * rdp_full


def _rdp_to_dp(rdp: float, order: float, delta: float) -> float:
    """Convert RDP guarantee (ε_R at order α) to (ε, δ)-DP via standard conversion."""
    if rdp == float("inf"):
        return float("inf")
    # ε = ε_R(α) + log((α-1)/α) - (log(δ) + log(α)) / (α-1)
    # Simplified common form: ε = ε_R - log(delta) / (α - 1)
    return rdp + math.log(1.0 / delta) / (order - 1.0)


class PrivacyAccountant:
    """
    RDP moments accountant for DP-SGD.

    Tracks cumulative privacy spend across training steps and converts to
    (ε, δ)-DP guarantees on demand.
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            noise_multiplier: σ — ratio of noise std to clipping norm
            sample_rate:      q = batch_size / dataset_size
            delta:            target δ for (ε, δ)-DP
            orders:           RDP orders to track (defaults to standard set)
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.orders = orders or _RDP_ORDERS
        self.steps: int = 0
        # Cumulative RDP per order
        self._rdp: List[float] = [0.0] * len(self.orders)

    def step(self, num_steps: int = 1) -> None:
        """Record `num_steps` DP-SGD steps."""
        self.steps += num_steps
        for i, order in enumerate(self.orders):
            self._rdp[i] += num_steps * _rdp_gaussian_per_step(
                order, self.noise_multiplier, self.sample_rate
            )

    def get_epsilon(self) -> float:
        """Return current (ε, δ)-DP guarantee (best over all tracked orders)."""
        epsilons = [
            _rdp_to_dp(rdp_val, order, self.delta)
            for rdp_val, order in zip(self._rdp, self.orders)
            if order > 1.0
        ]
        return min(epsilons) if epsilons else float("inf")

    def get_budget_used(self) -> dict:
        return {
            "steps": self.steps,
            "epsilon": self.get_epsilon(),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
        }

    def reset(self) -> None:
        self.steps = 0
        self._rdp = [0.0] * len(self.orders)


# ---------------------------------------------------------------------------
# DP-SGD Optimizer
# ---------------------------------------------------------------------------

class DPSGDOptimizer(torch.optim.Optimizer):
    """
    Differentially Private SGD via per-sample gradient clipping + Gaussian noise.

    Wraps a standard SGD update with DP mechanisms applied to the gradient buffer.
    Per-sample gradients are computed externally (via the microbatch trick — see
    `compute_per_sample_gradients`) and passed to `dp_step()`.

    Args:
        params:           model parameters
        lr:               learning rate
        max_grad_norm:    per-sample clipping threshold C
        noise_multiplier: σ — noise std = σ × C / batch_size
        momentum:         SGD momentum (default 0)
        weight_decay:     L2 regularisation (default 0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # DP gradient processing
    # ------------------------------------------------------------------

    @staticmethod
    def clip_per_sample_gradients(
        per_sample_grads: List[torch.Tensor], max_norm: float
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Clip a list of per-sample gradients to L2 norm ≤ max_norm.

        Args:
            per_sample_grads: list of tensors, each shape (param_shape,)
            max_norm:         clipping threshold C

        Returns:
            clipped gradients, per-sample norms before clipping
        """
        norms = torch.stack([g.norm(2) for g in per_sample_grads])
        clip_factors = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
        clipped = [g * f for g, f in zip(per_sample_grads, clip_factors)]
        return clipped, norms

    @staticmethod
    def add_noise(
        summed_grad: torch.Tensor,
        batch_size: int,
        max_grad_norm: float,
        noise_multiplier: float,
    ) -> torch.Tensor:
        """
        Add calibrated Gaussian noise to a summed gradient.

        Noise std = σ × C, where σ = noise_multiplier, C = max_grad_norm.
        The summed gradient is divided by batch_size to get the mean.
        """
        noise_std = noise_multiplier * max_grad_norm
        noise = torch.randn_like(summed_grad) * noise_std
        return (summed_grad + noise) / batch_size

    def dp_step(
        self,
        per_sample_grads: dict,
        closure=None,
    ) -> Optional[torch.Tensor]:
        """
        Perform one DP-SGD update.

        Args:
            per_sample_grads: dict mapping param -> list of per-sample gradient tensors
                              (one tensor per sample in the batch)

        Returns:
            loss if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            C = group["max_grad_norm"]
            sigma = group["noise_multiplier"]
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                # Retrieve per-sample gradients for this parameter
                psg = per_sample_grads.get(p)
                if psg is None:
                    continue

                batch_size = len(psg)
                if batch_size == 0:
                    continue

                # 1. Clip per-sample gradients
                clipped, _ = self.clip_per_sample_gradients(psg, C)

                # 2. Sum clipped gradients
                summed = torch.stack(clipped).sum(dim=0)

                # 3. Add Gaussian noise and normalise
                dp_grad = self.add_noise(summed, batch_size, C, sigma)

                # 4. Optional weight decay
                if wd != 0:
                    dp_grad = dp_grad + wd * p.data

                # 5. Momentum
                param_state = self.state[p]
                if mu != 0:
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = dp_grad.clone()
                    else:
                        param_state["momentum_buffer"].mul_(mu).add_(dp_grad)
                    dp_grad = param_state["momentum_buffer"]

                # 6. SGD update
                p.data.add_(dp_grad, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Microbatch helper — compute per-sample gradients
# ---------------------------------------------------------------------------

def compute_per_sample_gradients(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    trainable_params: Optional[List[nn.Parameter]] = None,
) -> dict:
    """
    Compute per-sample gradients via the microbatch (loop) method.

    For each sample in the batch, compute loss and back-propagate independently.
    Returns a dict mapping parameter → list of per-sample gradient tensors.

    Args:
        model:             the model (only LoRA params need requires_grad=True)
        loss_fn:           loss function (outputs scalar given model output and target)
        inputs:            batch input tensor  (B, ...)
        targets:           batch target tensor (B, ...)
        trainable_params:  params to capture gradients for (default: all requiring grad)

    Returns:
        dict: {parameter: [grad_sample_0, grad_sample_1, ...]}
    """
    if trainable_params is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    per_sample_grads = {p: [] for p in trainable_params}
    batch_size = inputs.shape[0]

    for i in range(batch_size):
        model.zero_grad()
        x_i = inputs[i].unsqueeze(0)
        y_i = targets[i].unsqueeze(0)
        output = model(x_i)
        loss = loss_fn(output, y_i)
        loss.backward()

        for p in trainable_params:
            if p.grad is not None:
                per_sample_grads[p].append(p.grad.detach().clone())
            else:
                per_sample_grads[p].append(torch.zeros_like(p))

    model.zero_grad()
    return per_sample_grads

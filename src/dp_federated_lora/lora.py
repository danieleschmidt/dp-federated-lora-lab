"""
LoRA (Low-Rank Adaptation) layer implementation.

Implements parameter-efficient fine-tuning via low-rank decomposition:
    W' = W + BA,  where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, rank r << min(d_in, d_out)

Base weights W are frozen; only A and B are trained.
Scaling factor α/r controls the magnitude of adaptation.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """
    Drop-in LoRA replacement for a Linear layer.

    Args:
        in_features:  input dimension (d_in)
        out_features: output dimension (d_out)
        rank:         LoRA rank r  (default 4)
        alpha:        scaling numerator α (default = rank)
        base_weight:  optional pretrained weight to freeze (d_out × d_in tensor)
        bias:         whether to include a bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: Optional[float] = None,
        base_weight: Optional[torch.Tensor] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        # Frozen base weight (simulates a pretrained layer)
        if base_weight is not None:
            assert base_weight.shape == (out_features, in_features)
            self.register_buffer("base_weight", base_weight.clone())
        else:
            self.register_buffer(
                "base_weight",
                torch.zeros(out_features, in_features),
            )

        # Optional bias (trainable)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialise A with Kaiming uniform, B with zeros → zero adaptation at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base pass (no gradient through frozen weight)
        base_out = nn.functional.linear(x, self.base_weight, self.bias)
        # LoRA adaptation:  x @ A.T @ B.T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def get_lora_weights(self) -> dict:
        """Return LoRA weight tensors (detached copies for safe transfer)."""
        return {
            "lora_A": self.lora_A.detach().clone(),
            "lora_B": self.lora_B.detach().clone(),
        }

    def set_lora_weights(self, weights: dict) -> None:
        """Load LoRA weight tensors in-place."""
        with torch.no_grad():
            self.lora_A.copy_(weights["lora_A"])
            self.lora_B.copy_(weights["lora_B"])

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )

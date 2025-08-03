"""
Differential Privacy mechanisms for federated learning.

This module implements privacy-preserving mechanisms including DP-SGD,
privacy accounting, and noise calibration for the federated LoRA system.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from opacus import PrivacyEngine as OpacusPrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator

from .config import PrivacyConfig, PrivacyMechanism


class PrivacyEngine:
    """
    Privacy engine that manages differential privacy for federated learning.
    
    Integrates with Opacus to provide DP-SGD training while maintaining
    privacy accounting across federated rounds.
    """
    
    def __init__(self, config: PrivacyConfig):
        """
        Initialize privacy engine.
        
        Args:
            config: Privacy configuration parameters
        """
        self.config = config
        self.accountant = RDPAccountant()
        self.opacus_engine: Optional[OpacusPrivacyEngine] = None
        self.is_attached = False
        
    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """
        Attach privacy engine to model, optimizer, and data loader.
        
        Args:
            model: PyTorch model to make private
            optimizer: Optimizer to make private
            data_loader: Data loader to make private
            
        Returns:
            Tuple of private model, optimizer, and data loader
        """
        if self.is_attached:
            warnings.warn("Privacy engine already attached. Detaching first.")
            self.detach()
            
        # Validate model compatibility
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            # Attempt to fix common issues
            model = ModuleValidator.fix(model)
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                raise ValueError(f"Model not compatible with DP: {errors}")
        
        # Create Opacus privacy engine
        self.opacus_engine = OpacusPrivacyEngine(accountant=self.accountant)
        
        # Make components private
        private_model, private_optimizer, private_data_loader = self.opacus_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.config.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
            grad_sample_mode="hooks"
        )
        
        self.is_attached = True
        return private_model, private_optimizer, private_data_loader
    
    def detach(self) -> None:
        """Detach privacy engine from components."""
        if self.opacus_engine:
            self.opacus_engine.detach()
        self.is_attached = False
        
    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Compute current privacy budget (epsilon) spent.
        
        Args:
            delta: Privacy parameter delta (uses config default if None)
            
        Returns:
            Current epsilon value
        """
        if delta is None:
            delta = self.config.delta
            
        if not self.is_attached or not self.opacus_engine:
            return 0.0
            
        return self.opacus_engine.get_epsilon(delta=delta)
    
    def step(self) -> None:
        """Record a privacy step (called after each gradient step)."""
        if self.accountant:
            self.accountant.step(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=1.0  # Will be updated based on actual sampling
            )
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get comprehensive privacy spending information.
        
        Returns:
            Dictionary with privacy metrics
        """
        epsilon = self.get_epsilon()
        return {
            "epsilon": epsilon,
            "delta": self.config.delta,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm": self.config.max_grad_norm,
            "steps": len(self.accountant.history) if self.accountant else 0
        }


class PrivacyAccountant:
    """
    Manages privacy budget across federated learning rounds.
    
    Tracks privacy expenditure per client and globally to ensure
    total privacy budget is not exceeded.
    """
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget (epsilon)
            total_delta: Total privacy parameter (delta)
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.client_budgets: Dict[str, Dict[str, float]] = {}
        self.global_epsilon_spent = 0.0
        self.round_history: List[Dict[str, float]] = []
        
    def allocate_budget(
        self, 
        client_id: str, 
        rounds_remaining: int,
        adaptive: bool = True
    ) -> Tuple[float, float]:
        """
        Allocate privacy budget for a client.
        
        Args:
            client_id: Unique client identifier
            rounds_remaining: Number of training rounds remaining
            adaptive: Whether to use adaptive budget allocation
            
        Returns:
            Tuple of (epsilon, delta) allocated for this round
        """
        if client_id not in self.client_budgets:
            self.client_budgets[client_id] = {
                "total_epsilon": 0.0,
                "total_delta": 0.0,
                "rounds_participated": 0
            }
        
        if adaptive:
            # Adaptive allocation based on remaining budget and rounds
            remaining_epsilon = self.total_epsilon - self.global_epsilon_spent
            epsilon_per_round = remaining_epsilon / max(rounds_remaining, 1)
            # Conservative allocation to avoid budget exhaustion
            epsilon_allocation = min(epsilon_per_round * 0.8, remaining_epsilon)
        else:
            # Uniform allocation
            epsilon_allocation = self.total_epsilon / rounds_remaining
            
        delta_allocation = self.total_delta / rounds_remaining
        
        return epsilon_allocation, delta_allocation
    
    def record_spending(
        self, 
        client_id: str, 
        epsilon_spent: float, 
        delta_spent: float,
        round_num: int
    ) -> None:
        """
        Record privacy spending for a client in a round.
        
        Args:
            client_id: Client identifier
            epsilon_spent: Epsilon consumed in this round
            delta_spent: Delta consumed in this round
            round_num: Current round number
        """
        if client_id not in self.client_budgets:
            self.client_budgets[client_id] = {
                "total_epsilon": 0.0,
                "total_delta": 0.0,
                "rounds_participated": 0
            }
        
        # Update client budget
        self.client_budgets[client_id]["total_epsilon"] += epsilon_spent
        self.client_budgets[client_id]["total_delta"] += delta_spent
        self.client_budgets[client_id]["rounds_participated"] += 1
        
        # Update global budget
        self.global_epsilon_spent += epsilon_spent
        
        # Record round history
        self.round_history.append({
            "round": round_num,
            "client_id": client_id,
            "epsilon_spent": epsilon_spent,
            "delta_spent": delta_spent,
            "cumulative_epsilon": self.global_epsilon_spent
        })
    
    def check_budget_feasible(self, rounds_remaining: int) -> bool:
        """
        Check if remaining training is feasible within privacy budget.
        
        Args:
            rounds_remaining: Number of rounds remaining
            
        Returns:
            True if training can continue within budget
        """
        remaining_epsilon = self.total_epsilon - self.global_epsilon_spent
        min_epsilon_per_round = 0.1  # Minimum meaningful epsilon
        
        return remaining_epsilon >= (min_epsilon_per_round * rounds_remaining)
    
    def get_budget_status(self) -> Dict[str, Union[float, int, bool]]:
        """
        Get current budget status.
        
        Returns:
            Dictionary with budget information
        """
        remaining_epsilon = self.total_epsilon - self.global_epsilon_spent
        utilization = self.global_epsilon_spent / self.total_epsilon
        
        return {
            "total_epsilon": self.total_epsilon,
            "epsilon_spent": self.global_epsilon_spent,
            "epsilon_remaining": remaining_epsilon,
            "budget_utilization": utilization,
            "clients_tracked": len(self.client_budgets),
            "rounds_recorded": len(self.round_history),
            "budget_exhausted": remaining_epsilon <= 0
        }
    
    def get_client_budget(self, client_id: str) -> Dict[str, float]:
        """
        Get budget information for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client budget information
        """
        if client_id not in self.client_budgets:
            return {"total_epsilon": 0.0, "total_delta": 0.0, "rounds_participated": 0}
        
        return self.client_budgets[client_id].copy()


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int
) -> float:
    """
    Compute noise multiplier needed to achieve target privacy.
    
    Args:
        target_epsilon: Target epsilon value
        target_delta: Target delta value
        sample_rate: Sampling rate for mini-batches
        steps: Number of training steps
        
    Returns:
        Required noise multiplier
    """
    # Binary search for appropriate noise multiplier
    low, high = 0.1, 10.0
    tolerance = 0.01
    
    for _ in range(100):  # Max iterations
        mid = (low + high) / 2
        
        # Create temporary accountant to test noise level
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=mid, sample_rate=sample_rate)
        
        epsilon = accountant.get_epsilon(delta=target_delta)
        
        if abs(epsilon - target_epsilon) < tolerance:
            return mid
        elif epsilon > target_epsilon:
            low = mid
        else:
            high = mid
    
    return high  # Conservative choice


def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_multiplier: float,
    max_norm: float
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor
        noise_multiplier: Noise multiplier for DP
        max_norm: Maximum norm for clipping
        
    Returns:
        Tensor with added noise
    """
    if noise_multiplier == 0:
        return tensor
    
    noise = torch.normal(
        mean=0,
        std=noise_multiplier * max_norm,
        size=tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype
    )
    
    return tensor + noise


def clip_gradients(
    parameters: List[torch.Tensor],
    max_norm: float
) -> List[torch.Tensor]:
    """
    Clip gradients to a maximum norm.
    
    Args:
        parameters: List of parameter tensors
        max_norm: Maximum allowed norm
        
    Returns:
        List of clipped parameter tensors
    """
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in parameters if p.grad is not None])
    )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)
    
    clipped_params = []
    for p in parameters:
        if p.grad is not None:
            clipped_params.append(p.grad.detach() * clip_coef)
        else:
            clipped_params.append(torch.zeros_like(p))
    
    return clipped_params


class AdaptiveClipping:
    """
    Adaptive gradient clipping for improved privacy-utility tradeoffs.
    
    Automatically adjusts clipping threshold based on gradient norms
    to improve model performance while maintaining privacy guarantees.
    """
    
    def __init__(
        self,
        initial_norm: float = 1.0,
        target_quantile: float = 0.5,
        learning_rate: float = 0.2
    ):
        """
        Initialize adaptive clipping.
        
        Args:
            initial_norm: Initial clipping threshold
            target_quantile: Target quantile for gradient norms
            learning_rate: Learning rate for threshold updates
        """
        self.current_norm = initial_norm
        self.target_quantile = target_quantile
        self.learning_rate = learning_rate
        self.norm_history: List[float] = []
        
    def update_threshold(self, gradient_norms: List[float]) -> float:
        """
        Update clipping threshold based on observed gradient norms.
        
        Args:
            gradient_norms: List of gradient norms from current batch
            
        Returns:
            Updated clipping threshold
        """
        if not gradient_norms:
            return self.current_norm
            
        self.norm_history.extend(gradient_norms)
        
        # Keep only recent history
        if len(self.norm_history) > 1000:
            self.norm_history = self.norm_history[-1000:]
        
        # Compute target threshold
        if len(self.norm_history) >= 10:
            target_norm = torch.quantile(
                torch.tensor(self.norm_history),
                self.target_quantile
            ).item()
            
            # Update threshold with moving average
            self.current_norm = (
                (1 - self.learning_rate) * self.current_norm +
                self.learning_rate * target_norm
            )
        
        return self.current_norm
    
    def clip_and_update(
        self,
        parameters: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Clip gradients and update threshold.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Tuple of (clipped gradients, updated threshold)
        """
        # Compute gradient norms
        grad_norms = [
            torch.norm(p.grad.detach()).item()
            for p in parameters
            if p.grad is not None
        ]
        
        # Update threshold
        new_threshold = self.update_threshold(grad_norms)
        
        # Clip gradients
        clipped_grads = clip_gradients(parameters, new_threshold)
        
        return clipped_grads, new_threshold
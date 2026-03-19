"""
Federated Learning with DP noise and LoRA adapters.

FedAvg over LoRA adapter weights with differential privacy guarantees.
Each client trains locally with gradient clipping + Gaussian noise; 
the server aggregates via FedAvg.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .lora import LoRALayer
from .privacy import PrivacyAccountant


@dataclass
class FederatedConfig:
    n_rounds: int = 5
    clients_per_round: int = 3
    local_epochs: int = 2
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    delta: float = 1e-5
    lora_rank: int = 4
    lora_alpha: float = 16.0
    batch_size: int = 16


@dataclass
class RoundResult:
    round_idx: int
    train_loss: float
    val_accuracy: float
    epsilon: float
    delta: float = 1e-5


def _lora_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all LoRA A/B params from model."""
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            params[f"{name}.lora_A"] = module.lora_A.data.clone()
            params[f"{name}.lora_B"] = module.lora_B.data.clone()
    return params


def _set_lora_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if f"{name}.lora_A" in params:
                module.lora_A.data.copy_(params[f"{name}.lora_A"])
            if f"{name}.lora_B" in params:
                module.lora_B.data.copy_(params[f"{name}.lora_B"])


class FederatedClient:
    """Local client: trains LoRA adapters with clipping + Gaussian noise."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        config: FederatedConfig,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.config = config
        x, y = data
        self.n_samples = len(x)
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        sample_rate = config.batch_size / max(self.n_samples, 1)
        self.accountant = PrivacyAccountant(
            noise_multiplier=config.noise_multiplier,
            sample_rate=sample_rate,
            delta=config.delta,
        )

    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        return _lora_params(self.model)

    def set_lora_params(self, params: Dict[str, torch.Tensor]) -> None:
        _set_lora_params(self.model, params)

    def train_round(self) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """Train for local_epochs, return updated LoRA params, avg loss, epsilon."""
        self.model.train()
        lora_trainable = [
            p for n, p in self.model.named_parameters()
            if "lora_A" in n or "lora_B" in n
        ]
        if not lora_trainable:
            return self.get_lora_params(), 0.0, 0.0

        optimizer = torch.optim.SGD(lora_trainable, lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        total_loss, steps = 0.0, 0

        for _ in range(self.config.local_epochs):
            for x_batch, y_batch in self.loader:
                optimizer.zero_grad()
                out = self.model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                # DP: clip gradients
                nn.utils.clip_grad_norm_(lora_trainable, self.config.max_grad_norm)
                # DP: add Gaussian noise
                with torch.no_grad():
                    for p in lora_trainable:
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad)
                            p.grad.add_(
                                noise * self.config.noise_multiplier * self.config.max_grad_norm
                            )
                optimizer.step()
                self.accountant.step()
                total_loss += loss.item()
                steps += 1

        eps = self.accountant.get_epsilon()
        return self.get_lora_params(), total_loss / max(steps, 1), eps


class FederatedServer:
    """Aggregates LoRA updates via FedAvg."""

    def __init__(self, model: nn.Module, config: FederatedConfig):
        self.model = model
        self.config = config

    def get_global_lora_params(self) -> Dict[str, torch.Tensor]:
        return _lora_params(self.model)

    def fedavg(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        if not client_updates:
            return
        avg: Dict[str, torch.Tensor] = {}
        for key in client_updates[0]:
            avg[key] = torch.stack([u[key] for u in client_updates]).mean(0)
        _set_lora_params(self.model, avg)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1)
        return (preds == y).float().mean().item()


def run_federation(
    model: nn.Module,
    clients: List[FederatedClient],
    server: FederatedServer,
    val_data: Tuple[torch.Tensor, torch.Tensor],
    config: FederatedConfig,
) -> List[RoundResult]:
    """Run the full federated learning loop."""
    results = []
    global_epsilon = 0.0

    for rnd in range(config.n_rounds):
        selected = clients[: config.clients_per_round]
        global_params = server.get_global_lora_params()
        for client in selected:
            client.set_lora_params(global_params)

        updates, losses, epsilons = [], [], []
        for client in selected:
            params, loss, eps = client.train_round()
            updates.append(params)
            losses.append(loss)
            epsilons.append(eps)

        server.fedavg(updates)
        global_epsilon = max(epsilons) if epsilons else global_epsilon
        val_acc = server.evaluate(*val_data)
        avg_loss = sum(losses) / max(len(losses), 1)

        results.append(RoundResult(
            round_idx=rnd,
            train_loss=avg_loss,
            val_accuracy=val_acc,
            epsilon=global_epsilon,
            delta=config.delta,
        ))

    return results

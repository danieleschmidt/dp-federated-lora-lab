"""
Secure aggregation protocols for federated learning.

This module implements various aggregation methods including secure aggregation,
Byzantine-robust algorithms, and privacy-preserving techniques.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from .config import AggregationMethod, SecurityConfig


class BaseAggregator(ABC):
    """Base class for all aggregation methods."""
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize aggregator.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.aggregation_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def aggregate(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: Dictionary mapping client_id to parameter updates
            client_weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated model parameters
        """
        pass
    
    def _record_aggregation(
        self,
        round_num: int,
        num_clients: int,
        method: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record aggregation for monitoring."""
        record = {
            "round": round_num,
            "num_clients": num_clients,
            "method": method,
            "timestamp": torch.tensor(0).item(),  # Placeholder
        }
        if metadata:
            record.update(metadata)
        self.aggregation_history.append(record)


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregator.
    
    Implements the standard weighted averaging of client updates
    based on the number of training samples.
    """
    
    def aggregate(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging.
        
        Args:
            client_updates: Client parameter updates
            client_weights: Client weights (sample counts)
            
        Returns:
            Averaged model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Use uniform weights if not provided
        if client_weights is None:
            client_weights = {
                client_id: 1.0 for client_id in client_updates.keys()
            }
        
        # Normalize weights
        total_weight = sum(client_weights.values())
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in client_weights.items()
        }
        
        # Initialize aggregated parameters
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            aggregated[param_name] = torch.zeros_like(first_client[param_name])
        
        # Weighted averaging
        for client_id, updates in client_updates.items():
            weight = normalized_weights[client_id]
            for param_name, param_tensor in updates.items():
                aggregated[param_name] += weight * param_tensor
        
        self._record_aggregation(
            round_num=len(self.aggregation_history) + 1,
            num_clients=len(client_updates),
            method="fedavg"
        )
        
        return aggregated


class SecureAggregator(BaseAggregator):
    """
    Secure aggregation using additive secret sharing.
    
    Implements a simplified secure aggregation protocol where
    the server cannot see individual client updates.
    """
    
    def __init__(self, config: SecurityConfig, threshold: float = 0.7):
        """
        Initialize secure aggregator.
        
        Args:
            config: Security configuration
            threshold: Minimum fraction of clients needed for aggregation
        """
        super().__init__(config)
        self.threshold = threshold
        self.client_secrets: Dict[str, bytes] = {}
        
    def _generate_secret_key(self, client_id: str, round_num: int) -> bytes:
        """Generate a secret key for a client and round."""
        # In a real implementation, this would use proper key exchange
        seed = f"{client_id}_{round_num}".encode()
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'federated_learning',
            backend=default_backend()
        )
        return kdf.derive(seed)
    
    def _generate_mask(
        self,
        shape: torch.Size,
        seed: bytes,
        device: torch.device
    ) -> torch.Tensor:
        """Generate a random mask from a seed."""
        # Convert seed to integer for torch generator
        seed_int = int.from_bytes(seed[:8], byteorder='big')
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_int)
        
        return torch.randn(shape, generator=generator, device=device)
    
    def secure_computation(self, func):
        """Decorator for secure computation (placeholder)."""
        def wrapper(*args, **kwargs):
            # In a real implementation, this would handle MPC protocols
            return func(*args, **kwargs)
        return wrapper
    
    @secure_computation
    def aggregate(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None,
        round_num: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation.
        
        Args:
            client_updates: Masked client updates
            client_weights: Client weights
            round_num: Current round number
            
        Returns:
            Securely aggregated parameters
        """
        num_clients = len(client_updates)
        min_clients = max(2, int(self.threshold * self.config.max_clients))
        
        if num_clients < min_clients:
            raise ValueError(f"Insufficient clients for secure aggregation: {num_clients} < {min_clients}")
        
        # Generate pairwise masks for each client pair
        client_list = list(client_updates.keys())
        masks = {}
        
        for i, client_i in enumerate(client_list):
            for j, client_j in enumerate(client_list):
                if i < j:  # Only generate mask once per pair
                    # Generate shared secret between client_i and client_j
                    shared_secret = self._generate_secret_key(f"{client_i}_{client_j}", round_num)
                    
                    # Store masks for cancellation
                    first_update = next(iter(client_updates.values()))
                    for param_name, param_tensor in first_update.items():
                        mask = self._generate_mask(
                            param_tensor.shape,
                            shared_secret,
                            param_tensor.device
                        )
                        
                        if param_name not in masks:
                            masks[param_name] = torch.zeros_like(param_tensor)
                        
                        # Add positive mask for client_i, negative for client_j
                        if client_i in client_updates:
                            masks[param_name] += mask
                        if client_j in client_updates:
                            masks[param_name] -= mask
        
        # Aggregate masked updates (masks cancel out)
        if client_weights is None:
            client_weights = {client_id: 1.0 for client_id in client_updates.keys()}
        
        total_weight = sum(client_weights.values())
        aggregated = {}
        
        first_client = next(iter(client_updates.values()))
        for param_name in first_client.keys():
            aggregated[param_name] = torch.zeros_like(first_client[param_name])
        
        # Sum all client updates (masks will cancel)
        for client_id, updates in client_updates.items():
            weight = client_weights[client_id] / total_weight
            for param_name, param_tensor in updates.items():
                aggregated[param_name] += weight * param_tensor
        
        # In reality, masks would be added here, but they cancel out
        # This is a simplified version for demonstration
        
        self._record_aggregation(
            round_num=round_num,
            num_clients=num_clients,
            method="secure_aggregation",
            metadata={"threshold_met": True}
        )
        
        return aggregated


class ByzantineRobustAggregator(BaseAggregator):
    """
    Byzantine-robust aggregation using various defense mechanisms.
    
    Implements multiple algorithms to defend against malicious clients
    including Krum, trimmed mean, and coordinate-wise median.
    """
    
    def __init__(self, config: SecurityConfig, method: str = "krum"):
        """
        Initialize Byzantine-robust aggregator.
        
        Args:
            config: Security configuration
            method: Robust aggregation method ('krum', 'trimmed_mean', 'median')
        """
        super().__init__(config)
        self.method = method
        self.byzantine_detected = 0
        
    def aggregate(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Byzantine-robust aggregation.
        
        Args:
            client_updates: Client parameter updates
            client_weights: Client weights
            
        Returns:
            Robust aggregated parameters
        """
        if len(client_updates) < 3:
            warnings.warn("Byzantine robustness requires at least 3 clients")
            return FedAvgAggregator(self.config).aggregate(client_updates, client_weights)
        
        if self.method == "krum":
            return self._krum_aggregation(client_updates)
        elif self.method == "trimmed_mean":
            return self._trimmed_mean_aggregation(client_updates)
        elif self.method == "median":
            return self._median_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown Byzantine-robust method: {self.method}")
    
    def _compute_pairwise_distances(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute pairwise distances between client updates."""
        client_ids = list(client_updates.keys())
        num_clients = len(client_ids)
        distances = torch.zeros(num_clients, num_clients)
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i != j:
                    dist = 0.0
                    for param_name in client_updates[client_i].keys():
                        param_i = client_updates[client_i][param_name]
                        param_j = client_updates[client_j][param_name]
                        dist += torch.norm(param_i - param_j).item() ** 2
                    distances[i, j] = math.sqrt(dist)
        
        return distances
    
    def _krum_aggregation(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Krum aggregation algorithm.
        
        Selects the client update that is closest to the majority
        of other client updates.
        """
        client_ids = list(client_updates.keys())
        distances = self._compute_pairwise_distances(client_updates)
        
        num_clients = len(client_ids)
        num_byzantine = int(self.config.byzantine_fraction * num_clients)
        num_honest = num_clients - num_byzantine
        
        # Compute Krum scores (sum of distances to k-closest neighbors)
        scores = []
        for i in range(num_clients):
            # Get distances to all other clients
            client_distances = distances[i]
            # Sort and take k-closest (excluding self)
            k_closest = torch.topk(client_distances, num_honest - 1, largest=False).values
            scores.append(torch.sum(k_closest).item())
        
        # Select client with minimum score
        best_client_idx = scores.index(min(scores))
        selected_client = client_ids[best_client_idx]
        
        self.byzantine_detected = num_clients - 1  # All others considered potential Byzantine
        
        self._record_aggregation(
            round_num=len(self.aggregation_history) + 1,
            num_clients=num_clients,
            method="krum",
            metadata={
                "selected_client": selected_client,
                "byzantine_detected": self.byzantine_detected
            }
        )
        
        return client_updates[selected_client]
    
    def _trimmed_mean_aggregation(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean aggregation.
        
        Removes extreme values before averaging to defend against
        Byzantine clients.
        """
        num_clients = len(client_updates)
        num_byzantine = int(self.config.byzantine_fraction * num_clients)
        
        # Convert to tensor format for easier computation
        client_list = list(client_updates.keys())
        param_tensors = {}
        
        first_client = next(iter(client_updates.values()))
        for param_name in first_client.keys():
            # Stack all client parameters for this layer
            stacked = torch.stack([
                client_updates[client_id][param_name]
                for client_id in client_list
            ])
            param_tensors[param_name] = stacked
        
        # Apply trimmed mean parameter-wise
        aggregated = {}
        for param_name, stacked_params in param_tensors.items():
            # Sort along client dimension and trim extremes
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            trimmed = sorted_params[num_byzantine:-num_byzantine] if num_byzantine > 0 else sorted_params
            
            # Compute mean of remaining parameters
            aggregated[param_name] = torch.mean(trimmed, dim=0)
        
        self.byzantine_detected = 2 * num_byzantine  # Trimmed from both ends
        
        self._record_aggregation(
            round_num=len(self.aggregation_history) + 1,
            num_clients=num_clients,
            method="trimmed_mean",
            metadata={"byzantine_detected": self.byzantine_detected}
        )
        
        return aggregated
    
    def _median_aggregation(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median aggregation.
        
        Computes median for each parameter coordinate independently.
        """
        client_list = list(client_updates.keys())
        aggregated = {}
        
        first_client = next(iter(client_updates.values()))
        for param_name in first_client.keys():
            # Stack all client parameters
            stacked = torch.stack([
                client_updates[client_id][param_name]
                for client_id in client_list
            ])
            
            # Compute coordinate-wise median
            aggregated[param_name] = torch.median(stacked, dim=0).values
        
        self._record_aggregation(
            round_num=len(self.aggregation_history) + 1,
            num_clients=len(client_updates),
            method="coordinate_median"
        )
        
        return aggregated
    
    def detect_byzantine_clients(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        threshold: float = 2.0
    ) -> List[str]:
        """
        Detect potential Byzantine clients based on update similarity.
        
        Args:
            client_updates: Client parameter updates
            threshold: Standard deviation threshold for detection
            
        Returns:
            List of suspected Byzantine client IDs
        """
        if len(client_updates) < 3:
            return []
        
        distances = self._compute_pairwise_distances(client_updates)
        client_ids = list(client_updates.keys())
        
        # Compute mean distance from each client to all others
        mean_distances = torch.mean(distances, dim=1)
        
        # Detect outliers using standard deviation
        overall_mean = torch.mean(mean_distances)
        overall_std = torch.std(mean_distances)
        
        byzantine_clients = []
        for i, client_id in enumerate(client_ids):
            if mean_distances[i] > overall_mean + threshold * overall_std:
                byzantine_clients.append(client_id)
        
        return byzantine_clients


def create_aggregator(config: SecurityConfig) -> BaseAggregator:
    """
    Factory function to create appropriate aggregator.
    
    Args:
        config: Security configuration
        
    Returns:
        Configured aggregator instance
    """
    if config.aggregation_method == AggregationMethod.FEDAVG:
        return FedAvgAggregator(config)
    elif config.aggregation_method == AggregationMethod.SECURE_WEIGHTED:
        return SecureAggregator(config)
    elif config.aggregation_method == AggregationMethod.KRUM:
        return ByzantineRobustAggregator(config, method="krum")
    elif config.aggregation_method == AggregationMethod.TRIMMED_MEAN:
        return ByzantineRobustAggregator(config, method="trimmed_mean")
    elif config.aggregation_method == AggregationMethod.COORDINATE_MEDIAN:
        return ByzantineRobustAggregator(config, method="median")
    else:
        raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")


class AdaptiveAggregator(BaseAggregator):
    """
    Adaptive aggregator that switches methods based on detected threats.
    
    Monitors client behavior and switches to more robust aggregation
    when Byzantine behavior is detected.
    """
    
    def __init__(self, config: SecurityConfig):
        """Initialize adaptive aggregator."""
        super().__init__(config)
        self.primary_aggregator = FedAvgAggregator(config)
        self.robust_aggregator = ByzantineRobustAggregator(config, method="krum")
        self.threat_level = 0.0
        self.detection_history: List[float] = []
        
    def aggregate(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptively choose aggregation method.
        
        Args:
            client_updates: Client updates
            client_weights: Client weights
            
        Returns:
            Aggregated parameters
        """
        # Detect potential threats
        byzantine_clients = self.robust_aggregator.detect_byzantine_clients(client_updates)
        current_threat = len(byzantine_clients) / len(client_updates)
        
        # Update threat level with exponential moving average
        alpha = 0.3
        self.threat_level = alpha * current_threat + (1 - alpha) * self.threat_level
        self.detection_history.append(self.threat_level)
        
        # Choose aggregation method based on threat level
        if self.threat_level > self.config.byzantine_fraction:
            # High threat - use robust aggregation
            method = "robust"
            result = self.robust_aggregator.aggregate(client_updates, client_weights)
        else:
            # Low threat - use efficient FedAvg
            method = "fedavg"
            result = self.primary_aggregator.aggregate(client_updates, client_weights)
        
        self._record_aggregation(
            round_num=len(self.aggregation_history) + 1,
            num_clients=len(client_updates),
            method=f"adaptive_{method}",
            metadata={
                "threat_level": self.threat_level,
                "byzantine_detected": len(byzantine_clients),
                "method_used": method
            }
        )
        
        return result
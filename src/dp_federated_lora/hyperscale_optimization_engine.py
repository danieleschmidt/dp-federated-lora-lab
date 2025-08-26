"""
Hyperscale Optimization Engine for Massive Federated Learning.

This module implements advanced scaling optimizations for federated learning
systems that need to handle thousands of clients with dynamic resource allocation,
intelligent load balancing, and quantum-inspired optimization algorithms.

Features:
- Auto-scaling with predictive resource allocation
- Distributed gradient compression and quantization
- Intelligent client selection and clustering
- Multi-tier federated architecture
- Edge-cloud hybrid optimization
- Real-time performance adaptation
- Quantum-enhanced scaling algorithms

Author: Terry (Terragon Labs)
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import redis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import hashlib
import zlib
import lz4.frame
import pickle
import heapq
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for different scenarios."""
    STATIC = "static"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    QUANTUM_ENHANCED = "quantum_enhanced"
    HYBRID = "hybrid"


class ClientTier(Enum):
    """Client tiers based on capabilities."""
    EDGE = "edge"          # Mobile devices, IoT
    MOBILE = "mobile"      # Smartphones, tablets
    DESKTOP = "desktop"    # Personal computers
    SERVER = "server"      # Dedicated servers
    CLOUD = "cloud"        # Cloud instances


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    RESOURCE_AWARE = "resource_aware"
    LATENCY_BASED = "latency_based"
    INTELLIGENT = "intelligent"


class CompressionLevel(Enum):
    """Compression levels for gradient transmission."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"
    ADAPTIVE = "adaptive"


@dataclass
class ClientCapabilities:
    """Client capability profile."""
    client_id: str
    tier: ClientTier
    compute_power: float  # GFLOPS or relative score
    memory_gb: float
    bandwidth_mbps: float
    latency_ms: float
    battery_level: Optional[float] = None  # 0-1 for mobile devices
    reliability_score: float = 0.8  # Historical reliability
    availability_hours: List[int] = field(default_factory=lambda: list(range(24)))
    geographic_region: str = "unknown"
    network_type: str = "wifi"  # wifi, cellular, ethernet
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_capability_score(self) -> float:
        """Calculate overall capability score."""
        # Normalized scores (0-1 range)
        compute_score = min(self.compute_power / 1000.0, 1.0)  # Assume 1000 GFLOPS as max
        memory_score = min(self.memory_gb / 64.0, 1.0)  # Assume 64GB as max
        bandwidth_score = min(self.bandwidth_mbps / 1000.0, 1.0)  # Assume 1Gbps as max
        latency_score = max(0, 1.0 - (self.latency_ms / 1000.0))  # Lower is better
        
        # Weighted combination
        weights = [0.3, 0.2, 0.3, 0.2]  # compute, memory, bandwidth, latency
        scores = [compute_score, memory_score, bandwidth_score, latency_score]
        
        capability_score = sum(w * s for w, s in zip(weights, scores))
        
        # Apply reliability and battery factors
        if self.battery_level is not None and self.battery_level < 0.2:
            capability_score *= 0.5  # Reduce for low battery
        
        capability_score *= self.reliability_score
        
        return capability_score


@dataclass
class ResourceAllocation:
    """Resource allocation for a client."""
    client_id: str
    cpu_cores: float
    memory_gb: float
    gpu_memory_gb: float = 0.0
    bandwidth_limit_mbps: float = 100.0
    priority: int = 1
    max_batch_size: int = 32
    model_compression_ratio: float = 1.0
    gradient_compression: CompressionLevel = CompressionLevel.MEDIUM
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ClientCluster:
    """Group of clients with similar characteristics."""
    cluster_id: str
    client_ids: List[str]
    centroid_capabilities: ClientCapabilities
    tier: ClientTier
    avg_latency: float
    total_compute_power: float
    cluster_size: int
    leader_client_id: Optional[str] = None
    sub_clusters: List['ClientCluster'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['centroid_capabilities'] = self.centroid_capabilities.to_dict()
        result['sub_clusters'] = [sc.to_dict() for sc in self.sub_clusters]
        return result


class GradientCompressor(ABC):
    """Abstract base class for gradient compression."""
    
    @abstractmethod
    def compress(self, gradients: Dict[str, torch.Tensor]) -> bytes:
        """Compress gradients."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress gradients."""
        pass
    
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        pass


class QuantizationCompressor(GradientCompressor):
    """Quantization-based gradient compressor."""
    
    def __init__(self, bits: int = 8, stochastic: bool = True):
        self.bits = bits
        self.stochastic = stochastic
        self.compression_ratio = 32 / bits  # Assume 32-bit floats
        
    def compress(self, gradients: Dict[str, torch.Tensor]) -> bytes:
        """Compress using quantization."""
        compressed_grads = {}
        
        for name, grad in gradients.items():
            if grad is None:
                continue
                
            # Find min/max for quantization
            min_val = grad.min().item()
            max_val = grad.max().item()
            
            # Quantize to specified bits
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            
            if self.stochastic:
                # Stochastic quantization
                noise = torch.rand_like(grad) - 0.5
                quantized = ((grad - min_val) / scale + noise).round().clamp(0, 2 ** self.bits - 1)
            else:
                # Deterministic quantization
                quantized = ((grad - min_val) / scale).round().clamp(0, 2 ** self.bits - 1)
            
            # Convert to appropriate integer type
            if self.bits <= 8:
                quantized = quantized.to(torch.uint8)
            elif self.bits <= 16:
                quantized = quantized.to(torch.int16)
            else:
                quantized = quantized.to(torch.int32)
            
            compressed_grads[name] = {
                'quantized': quantized,
                'min_val': min_val,
                'max_val': max_val,
                'shape': grad.shape,
                'dtype': grad.dtype
            }
        
        # Serialize and compress with LZ4
        serialized = pickle.dumps(compressed_grads)
        return lz4.frame.compress(serialized)
    
    def decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress quantized gradients."""
        # Decompress and deserialize
        serialized = lz4.frame.decompress(compressed_data)
        compressed_grads = pickle.loads(serialized)
        
        gradients = {}
        for name, data in compressed_grads.items():
            quantized = data['quantized']
            min_val = data['min_val']
            max_val = data['max_val']
            shape = data['shape']
            dtype = data['dtype']
            
            # Dequantize
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            grad = quantized.float() * scale + min_val
            grad = grad.reshape(shape).to(dtype)
            
            gradients[name] = grad
        
        return gradients
    
    def get_compression_ratio(self) -> float:
        return self.compression_ratio


class TopKCompressor(GradientCompressor):
    """Top-K sparsification compressor."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.k_ratio = compression_ratio  # Keep top k% of gradients
        
    def compress(self, gradients: Dict[str, torch.Tensor]) -> bytes:
        """Compress using Top-K sparsification."""
        compressed_grads = {}
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Flatten gradient
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * self.k_ratio))
            
            # Get top-k indices and values
            _, indices = torch.topk(torch.abs(flat_grad), k)
            values = flat_grad[indices]
            
            compressed_grads[name] = {
                'indices': indices,
                'values': values,
                'shape': grad.shape,
                'dtype': grad.dtype,
                'k': k
            }
        
        # Serialize and compress
        serialized = pickle.dumps(compressed_grads)
        return lz4.frame.compress(serialized)
    
    def decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress Top-K gradients."""
        # Decompress and deserialize
        serialized = lz4.frame.decompress(compressed_data)
        compressed_grads = pickle.loads(serialized)
        
        gradients = {}
        for name, data in compressed_grads.items():
            indices = data['indices']
            values = data['values']
            shape = data['shape']
            dtype = data['dtype']
            
            # Reconstruct sparse gradient
            flat_grad = torch.zeros(np.prod(shape), dtype=dtype)
            flat_grad[indices] = values
            grad = flat_grad.reshape(shape)
            
            gradients[name] = grad
        
        return gradients
    
    def get_compression_ratio(self) -> float:
        return self.compression_ratio


class AdaptiveCompressor(GradientCompressor):
    """Adaptive compressor that chooses method based on conditions."""
    
    def __init__(self):
        self.quantizer_8bit = QuantizationCompressor(bits=8)
        self.quantizer_4bit = QuantizationCompressor(bits=4)
        self.topk_10 = TopKCompressor(compression_ratio=0.1)
        self.topk_1 = TopKCompressor(compression_ratio=0.01)
        
        self.compression_history = deque(maxlen=100)
        self.bandwidth_threshold = 10.0  # Mbps
        
    def _select_compressor(self, gradients: Dict[str, torch.Tensor], bandwidth_mbps: float) -> GradientCompressor:
        """Select appropriate compressor based on conditions."""
        # Estimate gradient size
        total_elements = sum(grad.numel() for grad in gradients.values() if grad is not None)
        size_mb = (total_elements * 4) / (1024 * 1024)  # Assume 32-bit floats
        
        if bandwidth_mbps < 1.0 or size_mb > 100:
            # Very low bandwidth or large gradients - aggressive compression
            return self.topk_1
        elif bandwidth_mbps < 10.0 or size_mb > 50:
            # Low bandwidth or medium gradients - moderate compression
            return self.topk_10
        elif bandwidth_mbps < 50.0:
            # Medium bandwidth - quantization
            return self.quantizer_4bit
        else:
            # High bandwidth - light compression
            return self.quantizer_8bit
    
    def compress(self, gradients: Dict[str, torch.Tensor], bandwidth_mbps: float = 10.0) -> bytes:
        """Compress using adaptive strategy."""
        compressor = self._select_compressor(gradients, bandwidth_mbps)
        
        start_time = time.time()
        compressed_data = compressor.compress(gradients)
        compression_time = time.time() - start_time
        
        # Store statistics
        self.compression_history.append({
            'compressor': type(compressor).__name__,
            'compression_ratio': compressor.get_compression_ratio(),
            'size_bytes': len(compressed_data),
            'compression_time': compression_time,
            'bandwidth_mbps': bandwidth_mbps
        })
        
        # Add metadata to compressed data
        metadata = {
            'compressor_type': type(compressor).__name__,
            'compressor_params': getattr(compressor, '__dict__', {})
        }
        
        return pickle.dumps({'metadata': metadata, 'data': compressed_data})
    
    def decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress using stored metadata."""
        unpacked = pickle.loads(compressed_data)
        metadata = unpacked['metadata']
        data = unpacked['data']
        
        # Recreate compressor based on metadata
        compressor_type = metadata['compressor_type']
        
        if compressor_type == 'QuantizationCompressor':
            compressor = QuantizationCompressor(
                bits=metadata.get('compressor_params', {}).get('bits', 8)
            )
        elif compressor_type == 'TopKCompressor':
            compressor = TopKCompressor(
                compression_ratio=metadata.get('compressor_params', {}).get('compression_ratio', 0.1)
            )
        else:
            # Fallback
            compressor = self.quantizer_8bit
        
        return compressor.decompress(data)
    
    def get_compression_ratio(self) -> float:
        """Get average compression ratio from recent history."""
        if not self.compression_history:
            return 1.0
        
        ratios = [h['compression_ratio'] for h in self.compression_history]
        return sum(ratios) / len(ratios)


class IntelligentLoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.client_capabilities: Dict[str, ClientCapabilities] = {}
        self.client_loads: Dict[str, float] = defaultdict(float)
        self.client_connections: Dict[str, int] = defaultdict(int)
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Load balancing weights
        self.capability_weight = 0.4
        self.load_weight = 0.3
        self.latency_weight = 0.2
        self.performance_weight = 0.1
        
    def register_client(self, client_id: str, capabilities: ClientCapabilities):
        """Register a client with its capabilities."""
        self.client_capabilities[client_id] = capabilities
        logger.info(f"Registered client {client_id} with capabilities")
    
    def update_client_load(self, client_id: str, load: float):
        """Update client's current load."""
        self.client_loads[client_id] = load
    
    def update_client_latency(self, client_id: str, latency_ms: float):
        """Update client's latency measurement."""
        self.latency_history[client_id].append(latency_ms)
    
    def update_client_performance(self, client_id: str, performance_score: float):
        """Update client's performance score."""
        self.performance_history[client_id].append(performance_score)
    
    def select_clients(
        self,
        available_clients: List[str],
        num_clients: int,
        min_capability_score: float = 0.3
    ) -> List[str]:
        """Select clients based on load balancing strategy."""
        if not available_clients:
            return []
        
        # Filter clients by minimum capability
        qualified_clients = []
        for client_id in available_clients:
            if client_id in self.client_capabilities:
                capability = self.client_capabilities[client_id]
                if capability.get_capability_score() >= min_capability_score:
                    qualified_clients.append(client_id)
        
        if not qualified_clients:
            qualified_clients = available_clients  # Fallback to all available
        
        num_clients = min(num_clients, len(qualified_clients))
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(qualified_clients, num_clients)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._select_weighted(qualified_clients, num_clients)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(qualified_clients, num_clients)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._select_resource_aware(qualified_clients, num_clients)
        elif self.strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._select_latency_based(qualified_clients, num_clients)
        else:  # INTELLIGENT
            return self._select_intelligent(qualified_clients, num_clients)
    
    def _select_round_robin(self, clients: List[str], num_clients: int) -> List[str]:
        """Simple round-robin selection."""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        
        selected = []
        for _ in range(num_clients):
            selected.append(clients[self._rr_index % len(clients)])
            self._rr_index += 1
        
        return selected
    
    def _select_weighted(self, clients: List[str], num_clients: int) -> List[str]:
        """Weighted selection based on capabilities."""
        weights = []
        for client_id in clients:
            if client_id in self.client_capabilities:
                weight = self.client_capabilities[client_id].get_capability_score()
            else:
                weight = 0.5  # Default weight
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(clients)] * len(clients)
        
        # Select clients based on weights
        selected = np.random.choice(clients, size=num_clients, replace=False, p=weights).tolist()
        return selected
    
    def _select_least_connections(self, clients: List[str], num_clients: int) -> List[str]:
        """Select clients with least connections."""
        # Sort by connection count
        sorted_clients = sorted(clients, key=lambda c: self.client_connections[c])
        return sorted_clients[:num_clients]
    
    def _select_resource_aware(self, clients: List[str], num_clients: int) -> List[str]:
        """Select clients based on available resources."""
        client_scores = []
        
        for client_id in clients:
            # Calculate resource availability score
            capability_score = 0.5
            if client_id in self.client_capabilities:
                capability_score = self.client_capabilities[client_id].get_capability_score()
            
            load_score = 1.0 - self.client_loads.get(client_id, 0.0)  # Lower load is better
            
            total_score = (capability_score + load_score) / 2.0
            client_scores.append((client_id, total_score))
        
        # Sort by score and select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in client_scores[:num_clients]]
    
    def _select_latency_based(self, clients: List[str], num_clients: int) -> List[str]:
        """Select clients with lowest latency."""
        client_latencies = []
        
        for client_id in clients:
            if client_id in self.latency_history and self.latency_history[client_id]:
                avg_latency = sum(self.latency_history[client_id]) / len(self.latency_history[client_id])
            elif client_id in self.client_capabilities:
                avg_latency = self.client_capabilities[client_id].latency_ms
            else:
                avg_latency = 1000.0  # Default high latency
            
            client_latencies.append((client_id, avg_latency))
        
        # Sort by latency and select best clients
        client_latencies.sort(key=lambda x: x[1])
        return [client_id for client_id, _ in client_latencies[:num_clients]]
    
    def _select_intelligent(self, clients: List[str], num_clients: int) -> List[str]:
        """Intelligent selection combining all factors."""
        client_scores = []
        
        for client_id in clients:
            scores = {}
            
            # Capability score
            if client_id in self.client_capabilities:
                scores['capability'] = self.client_capabilities[client_id].get_capability_score()
            else:
                scores['capability'] = 0.5
            
            # Load score (inverted - lower load is better)
            scores['load'] = 1.0 - self.client_loads.get(client_id, 0.0)
            
            # Latency score (inverted - lower latency is better)
            if client_id in self.latency_history and self.latency_history[client_id]:
                avg_latency = sum(self.latency_history[client_id]) / len(self.latency_history[client_id])
                scores['latency'] = max(0, 1.0 - (avg_latency / 1000.0))  # Normalize to 0-1
            else:
                scores['latency'] = 0.5
            
            # Performance score
            if client_id in self.performance_history and self.performance_history[client_id]:
                scores['performance'] = sum(self.performance_history[client_id]) / len(self.performance_history[client_id])
            else:
                scores['performance'] = 0.5
            
            # Combined score
            total_score = (
                scores['capability'] * self.capability_weight +
                scores['load'] * self.load_weight +
                scores['latency'] * self.latency_weight +
                scores['performance'] * self.performance_weight
            )
            
            client_scores.append((client_id, total_score))
        
        # Sort by score and select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in client_scores[:num_clients]]
    
    def increment_connection(self, client_id: str):
        """Increment connection count for a client."""
        self.client_connections[client_id] += 1
    
    def decrement_connection(self, client_id: str):
        """Decrement connection count for a client."""
        self.client_connections[client_id] = max(0, self.client_connections[client_id] - 1)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "registered_clients": len(self.client_capabilities),
            "active_connections": dict(self.client_connections),
            "current_loads": dict(self.client_loads),
            "average_latency": {
                client_id: (sum(latencies) / len(latencies) if latencies else 0)
                for client_id, latencies in self.latency_history.items()
            },
            "strategy": self.strategy.value
        }


class ClientClusterManager:
    """Manages client clustering for hierarchical federation."""
    
    def __init__(self, max_cluster_size: int = 20, min_cluster_size: int = 5):
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.clusters: Dict[str, ClientCluster] = {}
        self.client_to_cluster: Dict[str, str] = {}
        self.scaler = StandardScaler()
        
    def create_clusters(self, client_capabilities: Dict[str, ClientCapabilities]) -> List[ClientCluster]:
        """Create clusters from client capabilities."""
        if len(client_capabilities) < self.min_cluster_size:
            # Create single cluster if too few clients
            cluster = self._create_single_cluster(list(client_capabilities.keys()), client_capabilities)
            self.clusters[cluster.cluster_id] = cluster
            return [cluster]
        
        # Extract features for clustering
        client_ids = list(client_capabilities.keys())
        features = []
        
        for client_id in client_ids:
            capabilities = client_capabilities[client_id]
            feature_vector = [
                capabilities.compute_power,
                capabilities.memory_gb,
                capabilities.bandwidth_mbps,
                capabilities.latency_ms,
                capabilities.reliability_score,
                capabilities.tier.value.__hash__() % 1000,  # Encode tier as numeric
                len(capabilities.availability_hours),
                capabilities.get_capability_score()
            ]
            features.append(feature_vector)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(np.array(features))
        
        # Determine optimal number of clusters
        n_clusters = self._determine_optimal_clusters(len(client_ids))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Create cluster objects
        clusters = []
        for cluster_idx in range(n_clusters):
            cluster_client_ids = [client_ids[i] for i in range(len(client_ids)) if cluster_labels[i] == cluster_idx]
            
            if len(cluster_client_ids) >= self.min_cluster_size:
                cluster = self._create_cluster_from_clients(cluster_client_ids, client_capabilities, cluster_idx)
                clusters.append(cluster)
                self.clusters[cluster.cluster_id] = cluster
                
                # Update client-to-cluster mapping
                for client_id in cluster_client_ids:
                    self.client_to_cluster[client_id] = cluster.cluster_id
        
        # Handle clients not in any cluster (small clusters)
        unassigned_clients = [
            client_id for client_id in client_ids
            if client_id not in self.client_to_cluster
        ]
        
        if unassigned_clients:
            # Assign to nearest existing cluster or create new one
            if clusters:
                self._assign_to_nearest_clusters(unassigned_clients, client_capabilities, clusters)
            else:
                # Create single cluster if no valid clusters exist
                cluster = self._create_single_cluster(unassigned_clients, client_capabilities)
                clusters.append(cluster)
                self.clusters[cluster.cluster_id] = cluster
        
        logger.info(f"Created {len(clusters)} clusters from {len(client_capabilities)} clients")
        return clusters
    
    def _determine_optimal_clusters(self, num_clients: int) -> int:
        """Determine optimal number of clusters."""
        if num_clients < self.min_cluster_size * 2:
            return 1
        
        # Use rule of thumb: sqrt(n/2) but constrain by cluster size limits
        estimated = int(np.sqrt(num_clients / 2))
        
        # Ensure clusters won't be too large or too small
        max_clusters = num_clients // self.min_cluster_size
        min_clusters = max(1, num_clients // self.max_cluster_size)
        
        return max(min_clusters, min(estimated, max_clusters))
    
    def _create_cluster_from_clients(
        self,
        client_ids: List[str],
        client_capabilities: Dict[str, ClientCapabilities],
        cluster_idx: int
    ) -> ClientCluster:
        """Create cluster from list of client IDs."""
        # Calculate centroid capabilities
        total_compute = sum(client_capabilities[cid].compute_power for cid in client_ids)
        total_memory = sum(client_capabilities[cid].memory_gb for cid in client_ids)
        total_bandwidth = sum(client_capabilities[cid].bandwidth_mbps for cid in client_ids)
        avg_latency = sum(client_capabilities[cid].latency_ms for cid in client_ids) / len(client_ids)
        avg_reliability = sum(client_capabilities[cid].reliability_score for cid in client_ids) / len(client_ids)
        
        # Determine dominant tier
        tier_counts = defaultdict(int)
        for client_id in client_ids:
            tier_counts[client_capabilities[client_id].tier] += 1
        dominant_tier = max(tier_counts, key=tier_counts.get)
        
        # Create centroid capabilities
        centroid = ClientCapabilities(
            client_id=f"cluster_{cluster_idx}_centroid",
            tier=dominant_tier,
            compute_power=total_compute / len(client_ids),
            memory_gb=total_memory / len(client_ids),
            bandwidth_mbps=total_bandwidth / len(client_ids),
            latency_ms=avg_latency,
            reliability_score=avg_reliability
        )
        
        # Select cluster leader (client with highest capability score)
        leader_client_id = max(
            client_ids,
            key=lambda cid: client_capabilities[cid].get_capability_score()
        )
        
        cluster = ClientCluster(
            cluster_id=f"cluster_{cluster_idx}",
            client_ids=client_ids,
            centroid_capabilities=centroid,
            tier=dominant_tier,
            avg_latency=avg_latency,
            total_compute_power=total_compute,
            cluster_size=len(client_ids),
            leader_client_id=leader_client_id
        )
        
        return cluster
    
    def _create_single_cluster(
        self,
        client_ids: List[str],
        client_capabilities: Dict[str, ClientCapabilities]
    ) -> ClientCluster:
        """Create a single cluster containing all clients."""
        return self._create_cluster_from_clients(client_ids, client_capabilities, 0)
    
    def _assign_to_nearest_clusters(
        self,
        unassigned_clients: List[str],
        client_capabilities: Dict[str, ClientCapabilities],
        clusters: List[ClientCluster]
    ):
        """Assign unassigned clients to nearest clusters."""
        for client_id in unassigned_clients:
            client_caps = client_capabilities[client_id]
            
            # Find nearest cluster based on capability similarity
            best_cluster = None
            best_distance = float('inf')
            
            for cluster in clusters:
                if len(cluster.client_ids) >= self.max_cluster_size:
                    continue  # Skip full clusters
                
                # Calculate distance to cluster centroid
                distance = self._calculate_capability_distance(client_caps, cluster.centroid_capabilities)
                
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster
            
            # Assign to best cluster
            if best_cluster:
                best_cluster.client_ids.append(client_id)
                best_cluster.cluster_size += 1
                self.client_to_cluster[client_id] = best_cluster.cluster_id
                
                # Update cluster statistics
                self._update_cluster_stats(best_cluster, client_capabilities)
    
    def _calculate_capability_distance(
        self,
        caps1: ClientCapabilities,
        caps2: ClientCapabilities
    ) -> float:
        """Calculate distance between two capability profiles."""
        # Normalized differences
        compute_diff = abs(caps1.compute_power - caps2.compute_power) / max(caps1.compute_power, caps2.compute_power, 1)
        memory_diff = abs(caps1.memory_gb - caps2.memory_gb) / max(caps1.memory_gb, caps2.memory_gb, 1)
        bandwidth_diff = abs(caps1.bandwidth_mbps - caps2.bandwidth_mbps) / max(caps1.bandwidth_mbps, caps2.bandwidth_mbps, 1)
        latency_diff = abs(caps1.latency_ms - caps2.latency_ms) / max(caps1.latency_ms, caps2.latency_ms, 1)
        
        # Tier difference (0 if same, 1 if different)
        tier_diff = 0 if caps1.tier == caps2.tier else 1
        
        # Weighted distance
        weights = [0.3, 0.2, 0.3, 0.1, 0.1]
        distances = [compute_diff, memory_diff, bandwidth_diff, latency_diff, tier_diff]
        
        return sum(w * d for w, d in zip(weights, distances))
    
    def _update_cluster_stats(self, cluster: ClientCluster, client_capabilities: Dict[str, ClientCapabilities]):
        """Update cluster statistics after adding clients."""
        if not cluster.client_ids:
            return
        
        # Recalculate centroid
        total_compute = sum(client_capabilities[cid].compute_power for cid in cluster.client_ids)
        total_memory = sum(client_capabilities[cid].memory_gb for cid in cluster.client_ids)
        total_bandwidth = sum(client_capabilities[cid].bandwidth_mbps for cid in cluster.client_ids)
        avg_latency = sum(client_capabilities[cid].latency_ms for cid in cluster.client_ids) / len(cluster.client_ids)
        
        cluster.centroid_capabilities.compute_power = total_compute / len(cluster.client_ids)
        cluster.centroid_capabilities.memory_gb = total_memory / len(cluster.client_ids)
        cluster.centroid_capabilities.bandwidth_mbps = total_bandwidth / len(cluster.client_ids)
        cluster.centroid_capabilities.latency_ms = avg_latency
        cluster.avg_latency = avg_latency
        cluster.total_compute_power = total_compute
    
    def get_cluster_for_client(self, client_id: str) -> Optional[ClientCluster]:
        """Get cluster containing specified client."""
        cluster_id = self.client_to_cluster.get(client_id)
        if cluster_id:
            return self.clusters.get(cluster_id)
        return None
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        if not self.clusters:
            return {"total_clusters": 0}
        
        cluster_sizes = [cluster.cluster_size for cluster in self.clusters.values()]
        tier_distribution = defaultdict(int)
        
        for cluster in self.clusters.values():
            tier_distribution[cluster.tier.value] += 1
        
        return {
            "total_clusters": len(self.clusters),
            "total_clients": sum(cluster_sizes),
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "tier_distribution": dict(tier_distribution),
            "clusters": [cluster.to_dict() for cluster in self.clusters.values()]
        }


class PredictiveAutoScaler:
    """Predictive auto-scaler for resource allocation."""
    
    def __init__(self, prediction_horizon: int = 300):  # 5 minutes
        self.prediction_horizon = prediction_horizon
        self.resource_history: deque = deque(maxlen=1000)
        self.demand_history: deque = deque(maxlen=1000)
        self.scaling_history: deque = deque(maxlen=100)
        
        # Simple linear regression for prediction
        self.demand_model = None
        self.resource_model = None
        
        # Scaling thresholds
        self.scale_up_threshold = 0.8    # Scale up when utilization > 80%
        self.scale_down_threshold = 0.3   # Scale down when utilization < 30%
        self.min_resources = 1
        self.max_resources = 1000
        
    def record_metrics(self, active_clients: int, avg_utilization: float, pending_requests: int):
        """Record current system metrics."""
        timestamp = time.time()
        
        self.resource_history.append({
            'timestamp': timestamp,
            'active_clients': active_clients,
            'avg_utilization': avg_utilization
        })
        
        self.demand_history.append({
            'timestamp': timestamp,
            'pending_requests': pending_requests,
            'total_demand': active_clients + pending_requests
        })
    
    def predict_demand(self, future_timestamp: float) -> Tuple[int, float]:
        """Predict future demand (clients needed, confidence)."""
        if len(self.demand_history) < 10:
            # Not enough data for prediction
            current_demand = self.demand_history[-1]['total_demand'] if self.demand_history else 5
            return current_demand, 0.5
        
        # Simple time-series prediction using linear trend
        timestamps = [h['timestamp'] for h in self.demand_history]
        demands = [h['total_demand'] for h in self.demand_history]
        
        # Calculate linear trend
        n = len(timestamps)
        sum_t = sum(timestamps)
        sum_d = sum(demands)
        sum_td = sum(t * d for t, d in zip(timestamps, demands))
        sum_tt = sum(t * t for t in timestamps)
        
        # Linear regression coefficients
        slope = (n * sum_td - sum_t * sum_d) / (n * sum_tt - sum_t * sum_t)
        intercept = (sum_d - slope * sum_t) / n
        
        # Predict future demand
        predicted_demand = slope * future_timestamp + intercept
        predicted_demand = max(1, int(predicted_demand))
        
        # Calculate confidence based on prediction error on recent data
        recent_errors = []
        for i in range(max(0, len(self.demand_history) - 20), len(self.demand_history)):
            actual = demands[i]
            predicted = slope * timestamps[i] + intercept
            error = abs(actual - predicted) / max(actual, 1)
            recent_errors.append(error)
        
        avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0.2
        confidence = max(0.1, 1.0 - avg_error)
        
        return predicted_demand, confidence
    
    def should_scale_up(self, current_clients: int, current_utilization: float) -> bool:
        """Determine if system should scale up."""
        if current_utilization > self.scale_up_threshold:
            return True
        
        # Predictive scaling - look ahead
        future_time = time.time() + self.prediction_horizon
        predicted_demand, confidence = self.predict_demand(future_time)
        
        # Scale up if predicted demand is significantly higher and we're confident
        if confidence > 0.7 and predicted_demand > current_clients * 1.2:
            return True
        
        return False
    
    def should_scale_down(self, current_clients: int, current_utilization: float) -> bool:
        """Determine if system should scale down."""
        if current_clients <= self.min_resources:
            return False
        
        if current_utilization < self.scale_down_threshold:
            # Look ahead to ensure demand won't spike soon
            future_time = time.time() + self.prediction_horizon
            predicted_demand, confidence = self.predict_demand(future_time)
            
            # Only scale down if predicted demand is also low
            if confidence > 0.6 and predicted_demand < current_clients * 0.8:
                return True
        
        return False
    
    def calculate_optimal_resources(self, current_clients: int, current_utilization: float) -> int:
        """Calculate optimal number of resources needed."""
        # Get current and predicted demand
        future_time = time.time() + self.prediction_horizon
        predicted_demand, confidence = self.predict_demand(future_time)
        
        # Use higher of current and predicted demand, weighted by confidence
        if confidence > 0.5:
            target_demand = max(current_clients, predicted_demand * confidence + current_clients * (1 - confidence))
        else:
            target_demand = current_clients
        
        # Add buffer for utilization target (aim for 60-70% utilization)
        optimal_resources = int(target_demand / 0.65)
        
        # Apply constraints
        optimal_resources = max(self.min_resources, min(self.max_resources, optimal_resources))
        
        return optimal_resources
    
    def record_scaling_action(self, action: str, from_count: int, to_count: int, reason: str):
        """Record a scaling action for analysis."""
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': action,
            'from_count': from_count,
            'to_count': to_count,
            'reason': reason
        })
        
        logger.info(f"Scaling action: {action} from {from_count} to {to_count} - {reason}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        if not self.scaling_history:
            return {"total_scaling_actions": 0}
        
        scale_up_count = len([a for a in self.scaling_history if a['action'] == 'scale_up'])
        scale_down_count = len([a for a in self.scaling_history if a['action'] == 'scale_down'])
        
        return {
            "total_scaling_actions": len(self.scaling_history),
            "scale_up_actions": scale_up_count,
            "scale_down_actions": scale_down_count,
            "recent_actions": list(self.scaling_history)[-10:],
            "prediction_accuracy": self._calculate_prediction_accuracy()
        }
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy based on recent history."""
        if len(self.demand_history) < 20:
            return 0.5
        
        # Compare predictions made 5 minutes ago with actual outcomes
        errors = []
        current_time = time.time()
        
        for i in range(len(self.demand_history) - 10, len(self.demand_history)):
            # Simulate prediction made 5 minutes before this point
            prediction_time = self.demand_history[i]['timestamp'] - self.prediction_horizon
            actual_demand = self.demand_history[i]['total_demand']
            
            # Find closest demand record to prediction time
            closest_idx = min(range(max(0, i-20), i), key=lambda x: abs(self.demand_history[x]['timestamp'] - prediction_time))
            
            if closest_idx < len(self.demand_history):
                # Make prediction based on data available at prediction time
                historical_data = self.demand_history[:closest_idx+1]
                if len(historical_data) >= 5:
                    demands = [h['total_demand'] for h in historical_data]
                    timestamps = [h['timestamp'] for h in historical_data]
                    
                    # Simple trend prediction
                    recent_trend = (demands[-1] - demands[-5]) / 5 if len(demands) >= 5 else 0
                    predicted = demands[-1] + recent_trend
                    
                    error = abs(predicted - actual_demand) / max(actual_demand, 1)
                    errors.append(error)
        
        if errors:
            avg_error = sum(errors) / len(errors)
            accuracy = max(0.0, 1.0 - avg_error)
        else:
            accuracy = 0.5
        
        return accuracy


class HyperscaleOptimizationEngine:
    """Main hyperscale optimization engine."""
    
    def __init__(
        self,
        scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        """Initialize hyperscale optimization engine."""
        self.scaling_strategy = scaling_strategy
        
        # Core components
        self.load_balancer = IntelligentLoadBalancer()
        self.cluster_manager = ClientClusterManager()
        self.auto_scaler = PredictiveAutoScaler()
        self.compressor = AdaptiveCompressor()
        
        # Client management
        self.registered_clients: Dict[str, ClientCapabilities] = {}
        self.active_clients: Set[str] = set()
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'throughput': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'resource_utilization': deque(maxlen=1000),
            'compression_ratio': deque(maxlen=1000)
        }
        
        # Distributed coordination (Redis)
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_available = self.redis_client.ping()
            logger.info("Connected to Redis for distributed coordination")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Running in standalone mode.")
            self.redis_available = False
            self.redis_client = None
        
        # Background tasks
        self._optimization_active = False
        self._optimization_thread = None
        
        logger.info(f"Initialized hyperscale optimization engine with strategy: {scaling_strategy}")
    
    def register_client(self, client_id: str, capabilities: ClientCapabilities):
        """Register a client with the system."""
        self.registered_clients[client_id] = capabilities
        self.load_balancer.register_client(client_id, capabilities)
        
        # Store in Redis if available
        if self.redis_available:
            try:
                self.redis_client.hset(
                    "client_capabilities",
                    client_id,
                    json.dumps(capabilities.to_dict())
                )
            except Exception as e:
                logger.warning(f"Failed to store client capabilities in Redis: {e}")
        
        logger.info(f"Registered client {client_id} with tier {capabilities.tier.value}")
    
    def unregister_client(self, client_id: str):
        """Unregister a client from the system."""
        if client_id in self.registered_clients:
            del self.registered_clients[client_id]
        
        self.active_clients.discard(client_id)
        
        if client_id in self.resource_allocations:
            del self.resource_allocations[client_id]
        
        # Remove from Redis if available
        if self.redis_available:
            try:
                self.redis_client.hdel("client_capabilities", client_id)
                self.redis_client.hdel("active_clients", client_id)
            except Exception as e:
                logger.warning(f"Failed to remove client from Redis: {e}")
        
        logger.info(f"Unregistered client {client_id}")
    
    def select_clients_for_round(
        self,
        num_clients: int,
        min_capability_score: float = 0.3,
        tier_preferences: List[ClientTier] = None
    ) -> List[str]:
        """Select clients for a training round."""
        # Filter by availability and preferences
        available_clients = list(self.active_clients)
        
        if tier_preferences:
            filtered_clients = []
            for client_id in available_clients:
                if client_id in self.registered_clients:
                    client_tier = self.registered_clients[client_id].tier
                    if client_tier in tier_preferences:
                        filtered_clients.append(client_id)
            available_clients = filtered_clients
        
        # Use load balancer to select clients
        selected_clients = self.load_balancer.select_clients(
            available_clients, num_clients, min_capability_score
        )
        
        # Update connection counts
        for client_id in selected_clients:
            self.load_balancer.increment_connection(client_id)
        
        logger.info(f"Selected {len(selected_clients)} clients for training round")
        return selected_clients
    
    def allocate_resources(self, client_ids: List[str]) -> Dict[str, ResourceAllocation]:
        """Allocate resources to selected clients."""
        allocations = {}
        
        for client_id in client_ids:
            if client_id not in self.registered_clients:
                continue
                
            capabilities = self.registered_clients[client_id]
            
            # Calculate resource allocation based on capabilities
            allocation = self._calculate_resource_allocation(capabilities)
            allocations[client_id] = allocation
            self.resource_allocations[client_id] = allocation
        
        logger.info(f"Allocated resources to {len(allocations)} clients")
        return allocations
    
    def _calculate_resource_allocation(self, capabilities: ClientCapabilities) -> ResourceAllocation:
        """Calculate optimal resource allocation for a client."""
        # Base allocation on client tier and capabilities
        tier_multipliers = {
            ClientTier.EDGE: 0.5,
            ClientTier.MOBILE: 0.7,
            ClientTier.DESKTOP: 1.0,
            ClientTier.SERVER: 1.5,
            ClientTier.CLOUD: 2.0
        }
        
        base_multiplier = tier_multipliers.get(capabilities.tier, 1.0)
        capability_score = capabilities.get_capability_score()
        
        # Calculate allocations
        cpu_cores = min(8.0, capabilities.compute_power / 100.0) * base_multiplier
        memory_gb = min(capabilities.memory_gb * 0.8, 32.0) * base_multiplier
        gpu_memory_gb = 0.0  # Assume no GPU by default
        
        # Adjust based on network capacity
        bandwidth_limit = min(capabilities.bandwidth_mbps, 1000.0)
        
        # Determine batch size and compression
        if capabilities.tier in [ClientTier.EDGE, ClientTier.MOBILE]:
            max_batch_size = 16
            compression_level = CompressionLevel.HIGH
        elif capabilities.tier == ClientTier.DESKTOP:
            max_batch_size = 32
            compression_level = CompressionLevel.MEDIUM
        else:
            max_batch_size = 64
            compression_level = CompressionLevel.LOW
        
        # Priority based on capability score
        priority = int(capability_score * 5) + 1  # 1-5 priority
        
        return ResourceAllocation(
            client_id=capabilities.client_id,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            bandwidth_limit_mbps=bandwidth_limit,
            priority=priority,
            max_batch_size=max_batch_size,
            gradient_compression=compression_level
        )
    
    def compress_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        client_id: str
    ) -> bytes:
        """Compress gradients for transmission."""
        # Get client's bandwidth for adaptive compression
        bandwidth_mbps = 10.0  # Default
        if client_id in self.registered_clients:
            bandwidth_mbps = self.registered_clients[client_id].bandwidth_mbps
        
        # Compress using adaptive compressor
        start_time = time.time()
        compressed_data = self.compressor.compress(gradients, bandwidth_mbps)
        compression_time = time.time() - start_time
        
        # Record performance metrics
        compression_ratio = len(pickle.dumps(gradients)) / len(compressed_data)
        self.performance_metrics['compression_ratio'].append(compression_ratio)
        
        logger.debug(f"Compressed gradients for {client_id}: "
                    f"ratio={compression_ratio:.2f}, time={compression_time:.3f}s")
        
        return compressed_data
    
    def create_client_clusters(self) -> List[ClientCluster]:
        """Create client clusters for hierarchical federation."""
        clusters = self.cluster_manager.create_clusters(self.registered_clients)
        
        # Store clusters in Redis if available
        if self.redis_available:
            try:
                cluster_data = [cluster.to_dict() for cluster in clusters]
                self.redis_client.set("client_clusters", json.dumps(cluster_data))
            except Exception as e:
                logger.warning(f"Failed to store clusters in Redis: {e}")
        
        return clusters
    
    def optimize_system_performance(self):
        """Optimize system performance based on current conditions."""
        if not self.active_clients:
            return
        
        # Calculate current system metrics
        current_clients = len(self.active_clients)
        
        # Estimate utilization (simplified)
        total_connections = sum(self.load_balancer.client_connections.values())
        avg_utilization = min(1.0, total_connections / max(current_clients, 1))
        
        # Record metrics for auto-scaler
        pending_requests = 0  # Would need actual queue size
        self.auto_scaler.record_metrics(current_clients, avg_utilization, pending_requests)
        
        # Check if scaling is needed
        if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.ADAPTIVE]:
            if self.auto_scaler.should_scale_up(current_clients, avg_utilization):
                self._scale_up_system()
            elif self.auto_scaler.should_scale_down(current_clients, avg_utilization):
                self._scale_down_system()
        
        # Update performance metrics
        self.performance_metrics['resource_utilization'].append(avg_utilization)
        
        # Optimization for different strategies
        if self.scaling_strategy == ScalingStrategy.QUANTUM_ENHANCED:
            self._apply_quantum_optimization()
        elif self.scaling_strategy == ScalingStrategy.HYBRID:
            self._apply_hybrid_optimization()
    
    def _scale_up_system(self):
        """Scale up the system by activating more clients."""
        current_active = len(self.active_clients)
        
        # Find inactive clients to activate
        inactive_clients = [
            client_id for client_id in self.registered_clients.keys()
            if client_id not in self.active_clients
        ]
        
        if inactive_clients:
            # Select best inactive clients
            selected = self.load_balancer.select_clients(
                inactive_clients, 
                min(5, len(inactive_clients)),  # Activate up to 5 new clients
                min_capability_score=0.4
            )
            
            for client_id in selected:
                self.active_clients.add(client_id)
            
            self.auto_scaler.record_scaling_action(
                "scale_up", current_active, len(self.active_clients),
                "High utilization detected"
            )
    
    def _scale_down_system(self):
        """Scale down the system by deactivating some clients."""
        current_active = len(self.active_clients)
        
        if current_active > self.auto_scaler.min_resources:
            # Select clients to deactivate (lowest capability or highest load)
            clients_to_deactivate = []
            
            client_scores = []
            for client_id in self.active_clients:
                if client_id in self.registered_clients:
                    capability_score = self.registered_clients[client_id].get_capability_score()
                    load_score = self.load_balancer.client_loads.get(client_id, 0.0)
                    combined_score = capability_score - load_score  # Prefer to keep high capability, low load
                    client_scores.append((client_id, combined_score))
            
            # Sort by score and deactivate lowest scoring clients
            client_scores.sort(key=lambda x: x[1])
            num_to_deactivate = min(3, max(1, current_active - self.auto_scaler.min_resources))
            
            for i in range(num_to_deactivate):
                client_id = client_scores[i][0]
                self.active_clients.discard(client_id)
                self.load_balancer.decrement_connection(client_id)
            
            self.auto_scaler.record_scaling_action(
                "scale_down", current_active, len(self.active_clients),
                "Low utilization detected"
            )
    
    def _apply_quantum_optimization(self):
        """Apply quantum-inspired optimizations."""
        # Simplified quantum optimization - would integrate with quantum components
        # For now, apply probabilistic client selection and resource allocation
        
        if not self.active_clients:
            return
        
        # Create quantum-like superposition of client states
        client_amplitudes = {}
        for client_id in self.active_clients:
            if client_id in self.registered_clients:
                capability_score = self.registered_clients[client_id].get_capability_score()
                load = self.load_balancer.client_loads.get(client_id, 0.5)
                
                # Amplitude based on capability and inverse load
                amplitude = np.sqrt(capability_score * (1.0 - load))
                client_amplitudes[client_id] = amplitude
        
        # Normalize amplitudes
        total_amplitude = sum(client_amplitudes.values())
        if total_amplitude > 0:
            for client_id in client_amplitudes:
                client_amplitudes[client_id] /= total_amplitude
        
        # Apply quantum interference effects (simplified)
        # Boost clients with high coherence (similar performance patterns)
        performance_coherence = self._calculate_performance_coherence()
        
        for client_id, coherence in performance_coherence.items():
            if client_id in client_amplitudes:
                client_amplitudes[client_id] *= (1.0 + coherence * 0.1)
        
        logger.debug("Applied quantum-inspired optimization")
    
    def _apply_hybrid_optimization(self):
        """Apply hybrid optimization combining multiple strategies."""
        # Combine predictive scaling with intelligent load balancing
        self.optimize_system_performance()  # Base optimization
        
        # Add adaptive compression based on current network conditions
        avg_bandwidth = np.mean([
            caps.bandwidth_mbps for caps in self.registered_clients.values()
        ])
        
        if avg_bandwidth < 10.0:  # Low bandwidth environment
            # Switch to higher compression
            self.compressor.bandwidth_threshold = 5.0
        elif avg_bandwidth > 100.0:  # High bandwidth environment
            # Switch to lower compression for speed
            self.compressor.bandwidth_threshold = 50.0
        
        logger.debug("Applied hybrid optimization strategy")
    
    def _calculate_performance_coherence(self) -> Dict[str, float]:
        """Calculate performance coherence between clients."""
        coherence_scores = {}
        
        # Simple coherence based on performance history similarity
        client_performances = {}
        for client_id in self.active_clients:
            if client_id in self.load_balancer.performance_history:
                client_performances[client_id] = list(self.load_balancer.performance_history[client_id])
        
        for client_id, performance_history in client_performances.items():
            if len(performance_history) < 3:
                coherence_scores[client_id] = 0.5
                continue
            
            # Calculate coherence with other clients
            coherence_values = []
            for other_id, other_performance in client_performances.items():
                if other_id != client_id and len(other_performance) >= 3:
                    # Calculate correlation
                    min_len = min(len(performance_history), len(other_performance))
                    if min_len >= 3:
                        perf_1 = performance_history[-min_len:]
                        perf_2 = other_performance[-min_len:]
                        
                        correlation = np.corrcoef(perf_1, perf_2)[0, 1]
                        if not np.isnan(correlation):
                            coherence_values.append(abs(correlation))
            
            coherence_scores[client_id] = np.mean(coherence_values) if coherence_values else 0.5
        
        return coherence_scores
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        load_stats = self.load_balancer.get_load_stats()
        cluster_stats = self.cluster_manager.get_cluster_stats()
        scaling_stats = self.auto_scaler.get_scaling_stats()
        
        return {
            "scaling_strategy": self.scaling_strategy.value,
            "registered_clients": len(self.registered_clients),
            "active_clients": len(self.active_clients),
            "client_tiers": {
                tier.value: len([
                    c for c in self.registered_clients.values() 
                    if c.tier == tier
                ])
                for tier in ClientTier
            },
            "load_balancer": load_stats,
            "clustering": cluster_stats,
            "auto_scaling": scaling_stats,
            "performance_metrics": {
                "avg_compression_ratio": (
                    sum(self.performance_metrics['compression_ratio']) / 
                    len(self.performance_metrics['compression_ratio'])
                    if self.performance_metrics['compression_ratio'] else 1.0
                ),
                "current_utilization": (
                    self.performance_metrics['resource_utilization'][-1]
                    if self.performance_metrics['resource_utilization'] else 0.0
                )
            },
            "redis_available": self.redis_available
        }
    
    def start_continuous_optimization(self, optimization_interval: int = 30):
        """Start continuous optimization loop."""
        if self._optimization_active:
            logger.warning("Optimization is already running")
            return
        
        self._optimization_active = True
        
        def optimization_loop():
            while self._optimization_active:
                try:
                    self.optimize_system_performance()
                    time.sleep(optimization_interval)
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    time.sleep(optimization_interval)
        
        self._optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self._optimization_thread.start()
        
        logger.info("Started continuous optimization")
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization loop."""
        self._optimization_active = False
        if self._optimization_thread and self._optimization_thread.is_alive():
            self._optimization_thread.join(timeout=5)
        logger.info("Stopped continuous optimization")
    
    def export_optimization_data(self, export_path: str):
        """Export optimization data for analysis."""
        export_data = {
            "system_status": self.get_system_status(),
            "client_capabilities": {
                client_id: caps.to_dict() 
                for client_id, caps in self.registered_clients.items()
            },
            "resource_allocations": {
                client_id: alloc.to_dict()
                for client_id, alloc in self.resource_allocations.items()
            },
            "performance_history": {
                metric_name: list(metric_data)
                for metric_name, metric_data in self.performance_metrics.items()
            },
            "export_timestamp": time.time()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported optimization data to {export_path}")


# Factory function for easy instantiation
def create_hyperscale_optimizer(
    scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    auto_start: bool = True
) -> HyperscaleOptimizationEngine:
    """Create and optionally start hyperscale optimization engine."""
    optimizer = HyperscaleOptimizationEngine(scaling_strategy, redis_host, redis_port)
    
    if auto_start:
        optimizer.start_continuous_optimization()
    
    return optimizer


if __name__ == "__main__":
    # Demonstration of hyperscale optimization engine
    import random
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperscale Optimization Engine Demo")
    parser.add_argument("--clients", type=int, default=50, help="Number of simulated clients")
    parser.add_argument("--duration", type=int, default=300, help="Demo duration in seconds")
    parser.add_argument("--strategy", type=str, default="adaptive",
                       choices=["static", "reactive", "predictive", "adaptive", "quantum_enhanced", "hybrid"])
    
    args = parser.parse_args()
    
    # Create hyperscale optimizer
    strategy = ScalingStrategy(args.strategy)
    optimizer = create_hyperscale_optimizer(scaling_strategy=strategy, auto_start=True)
    
    print(f"\nStarted hyperscale optimization engine demonstration")
    print(f"Strategy: {strategy.value}")
    print(f"Clients: {args.clients}")
    print(f"Duration: {args.duration} seconds")
    print("-" * 60)
    
    # Register simulated clients with different capabilities
    tiers = list(ClientTier)
    for i in range(args.clients):
        tier = random.choice(tiers)
        
        # Generate capabilities based on tier
        if tier == ClientTier.EDGE:
            compute_power = random.uniform(10, 50)
            memory_gb = random.uniform(1, 4)
            bandwidth_mbps = random.uniform(1, 10)
            latency_ms = random.uniform(50, 200)
        elif tier == ClientTier.MOBILE:
            compute_power = random.uniform(50, 150)
            memory_gb = random.uniform(2, 8)
            bandwidth_mbps = random.uniform(5, 50)
            latency_ms = random.uniform(20, 100)
        elif tier == ClientTier.DESKTOP:
            compute_power = random.uniform(100, 500)
            memory_gb = random.uniform(8, 32)
            bandwidth_mbps = random.uniform(10, 100)
            latency_ms = random.uniform(10, 50)
        elif tier == ClientTier.SERVER:
            compute_power = random.uniform(500, 2000)
            memory_gb = random.uniform(16, 128)
            bandwidth_mbps = random.uniform(100, 1000)
            latency_ms = random.uniform(5, 20)
        else:  # CLOUD
            compute_power = random.uniform(1000, 5000)
            memory_gb = random.uniform(32, 256)
            bandwidth_mbps = random.uniform(1000, 10000)
            latency_ms = random.uniform(1, 10)
        
        capabilities = ClientCapabilities(
            client_id=f"client_{i}",
            tier=tier,
            compute_power=compute_power,
            memory_gb=memory_gb,
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
            reliability_score=random.uniform(0.7, 0.95),
            geographic_region=random.choice(["us-east", "us-west", "eu-west", "asia-east"]),
            battery_level=random.uniform(0.2, 1.0) if tier in [ClientTier.EDGE, ClientTier.MOBILE] else None
        )
        
        optimizer.register_client(f"client_{i}", capabilities)
        
        # Randomly activate some clients
        if random.random() < 0.6:  # 60% initially active
            optimizer.active_clients.add(f"client_{i}")
    
    print(f"Registered {args.clients} clients with various capabilities")
    
    # Create clusters
    clusters = optimizer.create_client_clusters()
    print(f"Created {len(clusters)} client clusters")
    
    # Simulate system operation
    start_time = time.time()
    round_count = 0
    
    while time.time() - start_time < args.duration:
        try:
            # Simulate training round
            round_count += 1
            
            # Select clients for training
            num_clients_needed = random.randint(5, min(20, len(optimizer.active_clients)))
            selected_clients = optimizer.select_clients_for_round(num_clients_needed)
            
            if selected_clients:
                # Allocate resources
                allocations = optimizer.allocate_resources(selected_clients)
                
                # Simulate training and update performance
                for client_id in selected_clients:
                    # Simulate performance score
                    performance = random.uniform(0.6, 0.95)
                    optimizer.load_balancer.update_client_performance(client_id, performance)
                    
                    # Simulate load and latency updates
                    load = random.uniform(0.3, 0.8)
                    latency = random.uniform(10, 200)
                    optimizer.load_balancer.update_client_load(client_id, load)
                    optimizer.load_balancer.update_client_latency(client_id, latency)
                
                # Simulate gradient compression
                dummy_gradients = {"layer1": torch.randn(1000, 500), "layer2": torch.randn(500, 100)}
                compressed = optimizer.compress_gradients(dummy_gradients, selected_clients[0])
                
                # Decrement connections after round
                for client_id in selected_clients:
                    optimizer.load_balancer.decrement_connection(client_id)
            
            # Randomly add/remove active clients to simulate dynamic environment
            if random.random() < 0.1:  # 10% chance each round
                if random.random() < 0.5 and len(optimizer.active_clients) > 5:
                    # Remove a client
                    client_to_remove = random.choice(list(optimizer.active_clients))
                    optimizer.active_clients.discard(client_to_remove)
                else:
                    # Add a client
                    inactive_clients = [
                        cid for cid in optimizer.registered_clients.keys()
                        if cid not in optimizer.active_clients
                    ]
                    if inactive_clients:
                        client_to_add = random.choice(inactive_clients)
                        optimizer.active_clients.add(client_to_add)
            
            # Print status every 60 seconds
            if round_count % 10 == 0:  # Assuming ~6 seconds per round
                status = optimizer.get_system_status()
                print(f"Round {round_count:3d} | "
                      f"Active: {status['active_clients']:2d}/{status['registered_clients']} | "
                      f"Compression: {status['performance_metrics']['avg_compression_ratio']:.1f}x | "
                      f"Utilization: {status['performance_metrics']['current_utilization']:.1%} | "
                      f"Scaling Actions: {status['auto_scaling']['total_scaling_actions']}")
            
            time.sleep(6)  # Simulate round duration
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in demo loop: {e}")
    
    print(f"\n" + "=" * 60)
    print("HYPERSCALE OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    # Generate final status
    final_status = optimizer.get_system_status()
    
    print(f"Strategy: {final_status['scaling_strategy']}")
    print(f"Total Clients: {final_status['registered_clients']}")
    print(f"Active Clients: {final_status['active_clients']}")
    print(f"Training Rounds: {round_count}")
    
    print(f"\nClient Distribution:")
    for tier, count in final_status['client_tiers'].items():
        print(f"  {tier.upper()}: {count}")
    
    print(f"\nClustering:")
    clustering = final_status['clustering']
    print(f"  Total Clusters: {clustering['total_clusters']}")
    if clustering['total_clusters'] > 0:
        print(f"  Avg Cluster Size: {clustering['avg_cluster_size']:.1f}")
        print(f"  Size Range: {clustering['min_cluster_size']}-{clustering['max_cluster_size']}")
    
    print(f"\nAuto-Scaling:")
    scaling = final_status['auto_scaling']
    print(f"  Total Actions: {scaling['total_scaling_actions']}")
    print(f"  Scale Up: {scaling['scale_up_actions']}")
    print(f"  Scale Down: {scaling['scale_down_actions']}")
    print(f"  Prediction Accuracy: {scaling['prediction_accuracy']:.1%}")
    
    print(f"\nPerformance:")
    perf = final_status['performance_metrics']
    print(f"  Avg Compression Ratio: {perf['avg_compression_ratio']:.1f}x")
    print(f"  Final Utilization: {perf['current_utilization']:.1%}")
    
    # Export data
    export_file = f"hyperscale_optimization_export_{int(time.time())}.json"
    optimizer.export_optimization_data(export_file)
    print(f"\nOptimization data exported to: {export_file}")
    
    # Stop optimizer
    optimizer.stop_continuous_optimization()
    print(f"\nHyperscale optimization engine demonstration completed!")
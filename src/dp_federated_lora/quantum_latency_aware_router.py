"""
Latency-Aware Quantum Circuit Router
Generation 3 Scalability Enhancement

This module implements advanced latency-aware quantum circuit routing for federated 
learning systems, optimizing quantum operations across regions with sub-100ms response 
times while maintaining quantum coherence and maximizing quantum advantage.
"""

import asyncio
import logging
import time
import math
import numpy as np
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
from collections import defaultdict, deque
from datetime import datetime, timezone
import threading
import json
import heapq

import torch
import torch.nn as nn
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from .multi_region_deployment import DeploymentRegion
from .quantum_global_orchestration import QuantumCircuitRoute, QuantumCircuitRouter
from .quantum_entanglement_coordinator import QuantumEntanglementCoordinator
from .config import FederatedConfig
from .exceptions import QuantumOptimizationError


logger = logging.getLogger(__name__)


class LatencyClass(Enum):
    """Latency sensitivity classes for circuit routing"""
    ULTRA_LOW = "ultra_low"      # < 10ms
    LOW = "low"                  # < 50ms
    MEDIUM = "medium"            # < 100ms
    HIGH = "high"                # < 500ms
    BACKGROUND = "background"    # > 500ms


class RoutingPriority(Enum):
    """Circuit routing priorities"""
    CRITICAL = "critical"        # Emergency operations
    HIGH = "high"               # Real-time model updates
    NORMAL = "normal"           # Standard operations
    LOW = "low"                 # Batch processing
    BACKGROUND = "background"   # Maintenance tasks


class CircuitOptimizationStrategy(Enum):
    """Quantum circuit optimization strategies"""
    LATENCY_FIRST = "latency_first"
    FIDELITY_FIRST = "fidelity_first"
    BALANCED = "balanced"
    THROUGHPUT_FIRST = "throughput_first"
    QUANTUM_ADVANTAGE_FIRST = "quantum_advantage_first"


@dataclass
class LatencyConstraint:
    """Latency constraints for quantum circuit routing"""
    max_total_latency: float  # Maximum end-to-end latency (ms)
    max_hop_latency: float   # Maximum single hop latency (ms)
    jitter_tolerance: float   # Maximum jitter tolerance (ms)
    timeout: float           # Circuit execution timeout (ms)
    retry_limit: int         # Maximum retry attempts
    latency_class: LatencyClass


@dataclass
class NetworkPath:
    """Network path between regions with latency characteristics"""
    source_region: DeploymentRegion
    target_region: DeploymentRegion
    hops: List[DeploymentRegion]
    total_latency: float
    bandwidth: float
    reliability: float
    quantum_fidelity: float
    congestion_level: float
    last_measurement: datetime
    
    def calculate_path_score(self, latency_weight: float = 0.4) -> float:
        """Calculate overall path quality score"""
        latency_score = max(0.0, 1.0 - (self.total_latency / 1000.0))  # Normalize to 1s
        bandwidth_score = min(1.0, self.bandwidth / 100.0)  # Normalize to 100 Gbps
        reliability_score = self.reliability
        fidelity_score = self.quantum_fidelity
        congestion_score = max(0.0, 1.0 - self.congestion_level)
        
        return (
            latency_score * latency_weight +
            bandwidth_score * 0.2 +
            reliability_score * 0.15 +
            fidelity_score * 0.15 +
            congestion_score * 0.1
        )


@dataclass
class OptimizedCircuitRoute:
    """Optimized quantum circuit route with latency guarantees"""
    route_id: str
    circuit: QuantumCircuitRoute
    network_paths: List[NetworkPath]
    total_estimated_latency: float
    optimization_strategy: CircuitOptimizationStrategy
    latency_constraint: LatencyConstraint
    parallel_execution_plan: Optional[Dict[str, List[str]]] = None
    backup_routes: List['OptimizedCircuitRoute'] = field(default_factory=list)
    quantum_optimization_applied: bool = False
    classical_fallback_available: bool = True
    
    def meets_latency_constraint(self) -> bool:
        """Check if route meets latency constraints"""
        return self.total_estimated_latency <= self.latency_constraint.max_total_latency


@dataclass
class CircuitExecutionPlan:
    """Detailed execution plan for quantum circuit"""
    plan_id: str
    circuit_route: OptimizedCircuitRoute
    execution_phases: List[Dict[str, Any]]
    parallel_segments: List[List[str]]
    critical_path_length: float
    resource_requirements: Dict[str, float]
    estimated_quantum_advantage: float
    fallback_plan: Optional['CircuitExecutionPlan'] = None


class QuantumNetworkTopologyManager:
    """Manages quantum network topology and path discovery"""
    
    def __init__(self):
        self.network_graph = nx.Graph()
        self.region_coordinates: Dict[DeploymentRegion, Tuple[float, float]] = {}
        self.path_cache: Dict[Tuple[DeploymentRegion, DeploymentRegion], List[NetworkPath]] = {}
        self.latency_matrix: np.ndarray = np.array([])
        self.bandwidth_matrix: np.ndarray = np.array([])
        
        # Network state tracking
        self.congestion_levels: Dict[DeploymentRegion, float] = {}
        self.link_qualities: Dict[Tuple[DeploymentRegion, DeploymentRegion], float] = {}
        
        self._initialize_network_topology()
        
    def _initialize_network_topology(self):
        """Initialize quantum network topology"""
        
        # Define regional coordinates (mock geographical positions)
        self.region_coordinates = {
            DeploymentRegion.US_EAST: (40.7128, -74.0060),      # New York
            DeploymentRegion.US_WEST: (37.7749, -122.4194),     # San Francisco
            DeploymentRegion.US_CENTRAL: (41.8781, -87.6298),   # Chicago
            DeploymentRegion.EU_WEST: (51.5074, -0.1278),       # London
            DeploymentRegion.EU_CENTRAL: (50.1109, 8.6821),     # Frankfurt
            DeploymentRegion.EU_NORTH: (59.3293, 18.0686),      # Stockholm
            DeploymentRegion.ASIA_SOUTHEAST: (1.3521, 103.8198), # Singapore
            DeploymentRegion.JAPAN_EAST: (35.6762, 139.6503),   # Tokyo
            DeploymentRegion.AUSTRALIA_SOUTHEAST: (-33.8688, 151.2093), # Sydney
            DeploymentRegion.CANADA_CENTRAL: (43.6532, -79.3832), # Toronto
            DeploymentRegion.BRAZIL_SOUTH: (-23.5505, -46.6333), # São Paulo
            DeploymentRegion.UK_SOUTH: (51.5074, -0.1278),      # London
            DeploymentRegion.AFRICA_SOUTH: (-26.2041, 28.0473)  # Johannesburg
        }
        
        # Build network graph
        regions = list(self.region_coordinates.keys())
        
        for region in regions:
            self.network_graph.add_node(region)
            self.congestion_levels[region] = 0.1  # Initial low congestion
        
        # Add edges with calculated distances and properties
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                distance = self._calculate_geographic_distance(region1, region2)
                
                # Calculate base latency (speed of light + routing overhead)
                base_latency = (distance / 200000.0) * 1000  # Speed of light in fiber ~200,000 km/s
                routing_overhead = 5.0 + np.random.normal(0, 2.0)  # 5ms ± 2ms
                latency = max(1.0, base_latency + routing_overhead)
                
                # Calculate bandwidth (higher for shorter distances)
                bandwidth = max(1.0, 100.0 * np.exp(-distance / 10000.0))  # Exponential decay
                
                # Calculate reliability (based on infrastructure quality)
                reliability = max(0.8, 0.98 - distance / 20000.0)
                
                # Initial quantum fidelity (decreases with distance)
                quantum_fidelity = max(0.5, 0.95 - distance / 15000.0)
                
                self.network_graph.add_edge(
                    region1, region2,
                    distance=distance,
                    latency=latency,
                    bandwidth=bandwidth,
                    reliability=reliability,
                    quantum_fidelity=quantum_fidelity
                )
                
                self.link_qualities[(region1, region2)] = reliability * quantum_fidelity
                self.link_qualities[(region2, region1)] = reliability * quantum_fidelity
        
        # Initialize matrices
        self._update_network_matrices()
        
        logger.info(f"Initialized quantum network topology with {len(regions)} regions and {self.network_graph.number_of_edges()} connections")
    
    def _calculate_geographic_distance(
        self,
        region1: DeploymentRegion,
        region2: DeploymentRegion
    ) -> float:
        """Calculate geographic distance between regions using Haversine formula"""
        
        lat1, lon1 = self.region_coordinates[region1]
        lat2, lon2 = self.region_coordinates[region2]
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r
    
    def _update_network_matrices(self):
        """Update latency and bandwidth matrices"""
        regions = list(self.region_coordinates.keys())
        n_regions = len(regions)
        
        self.latency_matrix = np.full((n_regions, n_regions), np.inf)
        self.bandwidth_matrix = np.zeros((n_regions, n_regions))
        
        region_to_idx = {region: i for i, region in enumerate(regions)}
        
        for region1, region2, data in self.network_graph.edges(data=True):
            i, j = region_to_idx[region1], region_to_idx[region2]
            
            # Apply current congestion
            congestion_factor = 1.0 + self.congestion_levels.get(region1, 0.0) + self.congestion_levels.get(region2, 0.0)
            
            self.latency_matrix[i, j] = data['latency'] * congestion_factor
            self.latency_matrix[j, i] = data['latency'] * congestion_factor
            
            self.bandwidth_matrix[i, j] = data['bandwidth'] / congestion_factor
            self.bandwidth_matrix[j, i] = data['bandwidth'] / congestion_factor
        
        # Set diagonal to zero for latency
        np.fill_diagonal(self.latency_matrix, 0)
        np.fill_diagonal(self.bandwidth_matrix, np.inf)  # Infinite local bandwidth
    
    def find_optimal_paths(
        self,
        source: DeploymentRegion,
        targets: List[DeploymentRegion],
        latency_constraint: LatencyConstraint,
        max_paths: int = 3
    ) -> Dict[DeploymentRegion, List[NetworkPath]]:
        """Find optimal network paths to targets with latency constraints"""
        
        optimal_paths = {}
        
        for target in targets:
            if source == target:
                # Local "path"
                local_path = NetworkPath(
                    source_region=source,
                    target_region=target,
                    hops=[source],
                    total_latency=0.0,
                    bandwidth=np.inf,
                    reliability=1.0,
                    quantum_fidelity=1.0,
                    congestion_level=0.0,
                    last_measurement=datetime.now(timezone.utc)
                )
                optimal_paths[target] = [local_path]
                continue
            
            # Check cache first
            cache_key = (source, target)
            if cache_key in self.path_cache:
                cached_paths = self.path_cache[cache_key]
                # Filter paths that meet latency constraints
                valid_paths = [
                    path for path in cached_paths
                    if path.total_latency <= latency_constraint.max_total_latency
                ]
                if valid_paths:
                    optimal_paths[target] = valid_paths[:max_paths]
                    continue
            
            # Find new paths
            try:
                paths = self._calculate_paths_dijkstra(source, target, latency_constraint, max_paths)
                optimal_paths[target] = paths
                
                # Cache results
                self.path_cache[cache_key] = paths
                
            except Exception as e:
                logger.warning(f"Failed to find path from {source.value} to {target.value}: {e}")
                optimal_paths[target] = []
        
        return optimal_paths
    
    def _calculate_paths_dijkstra(
        self,
        source: DeploymentRegion,
        target: DeploymentRegion,
        latency_constraint: LatencyConstraint,
        max_paths: int
    ) -> List[NetworkPath]:
        """Calculate optimal paths using modified Dijkstra's algorithm"""
        
        try:
            # Use k-shortest paths with latency weighting
            paths = list(nx.shortest_simple_paths(
                self.network_graph,
                source,
                target,
                weight='latency'
            ))
            
            network_paths = []
            
            for path_nodes in paths[:max_paths * 2]:  # Get extra paths to filter
                # Calculate path properties
                total_latency = 0.0
                min_bandwidth = np.inf
                min_reliability = 1.0
                min_quantum_fidelity = 1.0
                max_congestion = 0.0
                
                for i in range(len(path_nodes) - 1):
                    edge_data = self.network_graph[path_nodes[i]][path_nodes[i + 1]]
                    
                    # Apply current congestion
                    congestion = max(
                        self.congestion_levels.get(path_nodes[i], 0.0),
                        self.congestion_levels.get(path_nodes[i + 1], 0.0)
                    )
                    
                    hop_latency = edge_data['latency'] * (1.0 + congestion)
                    
                    # Check hop latency constraint
                    if hop_latency > latency_constraint.max_hop_latency:
                        break
                    
                    total_latency += hop_latency
                    min_bandwidth = min(min_bandwidth, edge_data['bandwidth'] / (1.0 + congestion))
                    min_reliability = min(min_reliability, edge_data['reliability'])
                    min_quantum_fidelity = min(min_quantum_fidelity, edge_data['quantum_fidelity'])
                    max_congestion = max(max_congestion, congestion)
                
                else:  # Path completed without breaking
                    # Check total latency constraint
                    if total_latency <= latency_constraint.max_total_latency:
                        network_path = NetworkPath(
                            source_region=source,
                            target_region=target,
                            hops=path_nodes,
                            total_latency=total_latency,
                            bandwidth=min_bandwidth,
                            reliability=min_reliability,
                            quantum_fidelity=min_quantum_fidelity,
                            congestion_level=max_congestion,
                            last_measurement=datetime.now(timezone.utc)
                        )
                        
                        network_paths.append(network_path)
                        
                        if len(network_paths) >= max_paths:
                            break
            
            # Sort by path score
            network_paths.sort(key=lambda p: p.calculate_path_score(), reverse=True)
            
            return network_paths[:max_paths]
            
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {source.value} to {target.value}")
            return []
        except Exception as e:
            logger.error(f"Error calculating paths: {e}")
            return []
    
    def update_network_conditions(
        self,
        region: DeploymentRegion,
        congestion_level: float,
        link_updates: Optional[Dict[DeploymentRegion, Dict[str, float]]] = None
    ):
        """Update real-time network conditions"""
        
        self.congestion_levels[region] = max(0.0, min(1.0, congestion_level))
        
        if link_updates:
            for neighbor, metrics in link_updates.items():
                if self.network_graph.has_edge(region, neighbor):
                    edge_data = self.network_graph[region][neighbor]
                    
                    # Update link properties
                    if 'latency' in metrics:
                        edge_data['latency'] = metrics['latency']
                    if 'bandwidth' in metrics:
                        edge_data['bandwidth'] = metrics['bandwidth']
                    if 'reliability' in metrics:
                        edge_data['reliability'] = metrics['reliability']
                    if 'quantum_fidelity' in metrics:
                        edge_data['quantum_fidelity'] = metrics['quantum_fidelity']
                    
                    # Update link quality
                    self.link_qualities[(region, neighbor)] = edge_data['reliability'] * edge_data['quantum_fidelity']
                    self.link_qualities[(neighbor, region)] = edge_data['reliability'] * edge_data['quantum_fidelity']
        
        # Clear cache for affected paths
        self._clear_path_cache(region)
        
        # Update matrices
        self._update_network_matrices()
    
    def _clear_path_cache(self, region: DeploymentRegion):
        """Clear path cache entries involving a specific region"""
        keys_to_remove = [
            key for key in self.path_cache.keys()
            if key[0] == region or key[1] == region
        ]
        
        for key in keys_to_remove:
            del self.path_cache[key]


class QuantumCircuitOptimizer:
    """Optimizes quantum circuits for latency-aware execution"""
    
    def __init__(self):
        self.optimization_cache: Dict[str, Any] = {}
        self.parallel_gate_sets: Dict[str, List[List[str]]] = {}
        
    def optimize_circuit_for_latency(
        self,
        circuit: QuantumCircuitRoute,
        network_paths: List[NetworkPath],
        latency_constraint: LatencyConstraint,
        strategy: CircuitOptimizationStrategy = CircuitOptimizationStrategy.BALANCED
    ) -> Tuple[QuantumCircuitRoute, Dict[str, Any]]:
        """Optimize quantum circuit for latency constraints"""
        
        optimization_info = {
            "original_gates": len(circuit.quantum_gates),
            "original_execution_time": circuit.estimated_execution_time,
            "optimization_strategy": strategy.value,
            "optimizations_applied": []
        }
        
        optimized_circuit = circuit
        
        try:
            # Circuit depth reduction
            if strategy in [CircuitOptimizationStrategy.LATENCY_FIRST, CircuitOptimizationStrategy.BALANCED]:
                optimized_circuit, depth_info = self._reduce_circuit_depth(optimized_circuit)
                optimization_info["optimizations_applied"].append("depth_reduction")
                optimization_info.update(depth_info)
            
            # Gate parallelization
            if strategy in [CircuitOptimizationStrategy.THROUGHPUT_FIRST, CircuitOptimizationStrategy.BALANCED]:
                optimized_circuit, parallel_info = self._parallelize_gates(optimized_circuit, network_paths)
                optimization_info["optimizations_applied"].append("gate_parallelization")
                optimization_info.update(parallel_info)
            
            # Quantum error correction optimization
            if strategy in [CircuitOptimizationStrategy.FIDELITY_FIRST, CircuitOptimizationStrategy.BALANCED]:
                optimized_circuit, qec_info = self._optimize_error_correction(optimized_circuit, latency_constraint)
                optimization_info["optimizations_applied"].append("error_correction_optimization")
                optimization_info.update(qec_info)
            
            # Quantum advantage maximization
            if strategy == CircuitOptimizationStrategy.QUANTUM_ADVANTAGE_FIRST:
                optimized_circuit, qa_info = self._maximize_quantum_advantage(optimized_circuit, network_paths)
                optimization_info["optimizations_applied"].append("quantum_advantage_maximization")
                optimization_info.update(qa_info)
            
            optimization_info.update({
                "optimized_gates": len(optimized_circuit.quantum_gates),
                "optimized_execution_time": optimized_circuit.estimated_execution_time,
                "latency_improvement": circuit.estimated_execution_time - optimized_circuit.estimated_execution_time
            })
            
        except Exception as e:
            logger.error(f"Circuit optimization failed: {e}")
            optimization_info["optimization_error"] = str(e)
        
        return optimized_circuit, optimization_info
    
    def _reduce_circuit_depth(
        self,
        circuit: QuantumCircuitRoute
    ) -> Tuple[QuantumCircuitRoute, Dict[str, Any]]:
        """Reduce quantum circuit depth through gate optimization"""
        
        original_gates = circuit.quantum_gates.copy()
        optimized_gates = []
        
        # Simple gate optimization rules
        i = 0
        eliminated_gates = 0
        
        while i < len(original_gates):
            gate = original_gates[i]
            
            # Look for gate cancellations (simplified)
            if i + 1 < len(original_gates):
                next_gate = original_gates[i + 1]
                
                # Cancel identical Pauli gates
                if gate == next_gate and gate in ['X', 'Y', 'Z']:
                    # Skip both gates (they cancel)
                    i += 2
                    eliminated_gates += 2
                    continue
                
                # Combine rotation gates (simplified)
                if gate.startswith('R') and next_gate.startswith('R') and gate[1] == next_gate[1]:
                    # Combine rotations on same axis
                    optimized_gates.append(gate)  # Keep first one (simplified)
                    i += 2
                    eliminated_gates += 1
                    continue
            
            optimized_gates.append(gate)
            i += 1
        
        # Create optimized circuit
        optimized_circuit = QuantumCircuitRoute(
            circuit_id=circuit.circuit_id + "_optimized",
            source_region=circuit.source_region,
            target_regions=circuit.target_regions,
            quantum_gates=optimized_gates,
            expected_fidelity=min(1.0, circuit.expected_fidelity * 1.02),  # Slight improvement
            estimated_execution_time=circuit.estimated_execution_time * len(optimized_gates) / max(1, len(original_gates)),
            resource_requirements=circuit.resource_requirements.copy(),
            priority=circuit.priority
        )
        
        depth_info = {
            "depth_reduction": eliminated_gates,
            "compression_ratio": len(optimized_gates) / max(1, len(original_gates))
        }
        
        return optimized_circuit, depth_info
    
    def _parallelize_gates(
        self,
        circuit: QuantumCircuitRoute,
        network_paths: List[NetworkPath]
    ) -> Tuple[QuantumCircuitRoute, Dict[str, Any]]:
        """Parallelize quantum gates across network paths"""
        
        gates = circuit.quantum_gates
        n_paths = len(network_paths)
        
        if n_paths <= 1:
            return circuit, {"parallelization": "no_parallel_paths"}
        
        # Group gates that can be executed in parallel
        parallel_groups = []
        current_group = []
        
        for gate in gates:
            # Simple parallelization logic (gates that don't interfere)
            if gate in ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']:  # Single qubit gates
                current_group.append(gate)
            else:  # Multi-qubit gates need synchronization
                if current_group:
                    parallel_groups.append(current_group)
                    current_group = []
                parallel_groups.append([gate])
        
        if current_group:
            parallel_groups.append(current_group)
        
        # Calculate new execution time with parallelization
        parallel_execution_time = 0.0
        max_parallel_degree = min(n_paths, 4)  # Limit parallelization
        
        for group in parallel_groups:
            group_size = len(group)
            if group_size > 1:
                # Parallel execution
                parallel_time = 0.05 * math.ceil(group_size / max_parallel_degree)  # 50μs per parallel batch
            else:
                # Sequential execution
                parallel_time = 0.05  # 50μs per gate
            
            parallel_execution_time += parallel_time
        
        # Create optimized circuit
        optimized_circuit = QuantumCircuitRoute(
            circuit_id=circuit.circuit_id + "_parallel",
            source_region=circuit.source_region,
            target_regions=circuit.target_regions,
            quantum_gates=gates,  # Same gates, different execution plan
            expected_fidelity=circuit.expected_fidelity * 0.98,  # Slight degradation due to parallelization
            estimated_execution_time=parallel_execution_time,
            resource_requirements=circuit.resource_requirements.copy(),
            priority=circuit.priority
        )
        
        # Store parallelization plan
        self.parallel_gate_sets[optimized_circuit.circuit_id] = parallel_groups
        
        parallel_info = {
            "parallel_groups": len(parallel_groups),
            "max_group_size": max(len(group) for group in parallel_groups) if parallel_groups else 0,
            "parallelization_speedup": circuit.estimated_execution_time / max(0.001, parallel_execution_time)
        }
        
        return optimized_circuit, parallel_info
    
    def _optimize_error_correction(
        self,
        circuit: QuantumCircuitRoute,
        latency_constraint: LatencyConstraint
    ) -> Tuple[QuantumCircuitRoute, Dict[str, Any]]:
        """Optimize quantum error correction for latency constraints"""
        
        # Determine error correction overhead based on latency class
        if latency_constraint.latency_class == LatencyClass.ULTRA_LOW:
            # Minimal error correction for ultra-low latency
            error_correction_overhead = 1.1
            fidelity_improvement = 1.01
        elif latency_constraint.latency_class == LatencyClass.LOW:
            # Light error correction
            error_correction_overhead = 1.2
            fidelity_improvement = 1.05
        elif latency_constraint.latency_class == LatencyClass.MEDIUM:
            # Standard error correction
            error_correction_overhead = 1.4
            fidelity_improvement = 1.10
        else:
            # Full error correction for high latency tolerance
            error_correction_overhead = 1.8
            fidelity_improvement = 1.20
        
        # Apply error correction
        corrected_execution_time = circuit.estimated_execution_time * error_correction_overhead
        corrected_fidelity = min(1.0, circuit.expected_fidelity * fidelity_improvement)
        
        # Check if we still meet latency constraints
        if corrected_execution_time > latency_constraint.max_total_latency:
            # Reduce error correction to meet latency
            scaling_factor = latency_constraint.max_total_latency / corrected_execution_time
            corrected_execution_time = latency_constraint.max_total_latency
            corrected_fidelity = circuit.expected_fidelity * (1.0 + (fidelity_improvement - 1.0) * scaling_factor)
        
        # Create optimized circuit
        optimized_circuit = QuantumCircuitRoute(
            circuit_id=circuit.circuit_id + "_qec",
            source_region=circuit.source_region,
            target_regions=circuit.target_regions,
            quantum_gates=circuit.quantum_gates,
            expected_fidelity=corrected_fidelity,
            estimated_execution_time=corrected_execution_time,
            resource_requirements=circuit.resource_requirements.copy(),
            priority=circuit.priority
        )
        
        qec_info = {
            "error_correction_overhead": error_correction_overhead,
            "fidelity_improvement": corrected_fidelity / circuit.expected_fidelity,
            "latency_compliant": corrected_execution_time <= latency_constraint.max_total_latency
        }
        
        return optimized_circuit, qec_info
    
    def _maximize_quantum_advantage(
        self,
        circuit: QuantumCircuitRoute,
        network_paths: List[NetworkPath]
    ) -> Tuple[QuantumCircuitRoute, Dict[str, Any]]:
        """Optimize circuit to maximize quantum advantage"""
        
        # Calculate classical simulation complexity
        n_qubits = circuit.resource_requirements.get("qubits", 3)
        classical_complexity = 2**n_qubits * len(circuit.quantum_gates)
        
        # Estimate quantum advantage
        quantum_complexity = len(circuit.quantum_gates) * n_qubits
        
        if classical_complexity > quantum_complexity * 100:  # Significant quantum advantage
            # Optimize for maximum quantum advantage
            optimized_gates = circuit.quantum_gates.copy()
            
            # Add entanglement gates to increase quantum advantage
            if len(network_paths) > 1:
                # Add inter-region entanglement operations
                entanglement_gates = ['CNOT'] * min(3, len(network_paths) - 1)
                optimized_gates.extend(entanglement_gates)
            
            estimated_time = circuit.estimated_execution_time * 1.2  # 20% overhead for entanglement
            expected_fidelity = circuit.expected_fidelity * 0.95  # Slight degradation
            
        else:
            # Limited quantum advantage, optimize for classical efficiency
            optimized_gates = circuit.quantum_gates
            estimated_time = circuit.estimated_execution_time
            expected_fidelity = circuit.expected_fidelity
        
        # Create optimized circuit
        optimized_circuit = QuantumCircuitRoute(
            circuit_id=circuit.circuit_id + "_qa",
            source_region=circuit.source_region,
            target_regions=circuit.target_regions,
            quantum_gates=optimized_gates,
            expected_fidelity=expected_fidelity,
            estimated_execution_time=estimated_time,
            resource_requirements=circuit.resource_requirements.copy(),
            priority=circuit.priority
        )
        
        qa_info = {
            "classical_complexity": classical_complexity,
            "quantum_complexity": quantum_complexity,
            "quantum_advantage_ratio": classical_complexity / max(1, quantum_complexity),
            "entanglement_gates_added": len(optimized_gates) - len(circuit.quantum_gates)
        }
        
        return optimized_circuit, qa_info


class LatencyAwareQuantumRouter:
    """Main latency-aware quantum circuit router"""
    
    def __init__(
        self,
        config: FederatedConfig,
        entanglement_coordinator: QuantumEntanglementCoordinator
    ):
        self.config = config
        self.entanglement_coordinator = entanglement_coordinator
        
        # Core components
        self.topology_manager = QuantumNetworkTopologyManager()
        self.circuit_optimizer = QuantumCircuitOptimizer()
        self.base_router = QuantumCircuitRouter()  # Fallback router
        
        # Routing state
        self.active_routes: Dict[str, OptimizedCircuitRoute] = {}
        self.execution_plans: Dict[str, CircuitExecutionPlan] = {}
        self.routing_history: deque = deque(maxlen=10000)
        
        # Performance metrics
        self.routing_metrics = {
            "total_routes": 0,
            "successful_routes": 0,
            "latency_compliant_routes": 0,
            "average_latency": 0.0,
            "average_quantum_advantage": 0.0,
            "optimization_efficiency": 0.0,
            "network_utilization": 0.0
        }
        
        # Adaptive parameters
        self.latency_prediction_model: Optional[Any] = None
        self.congestion_prediction_accuracy = 0.7
        
        self.logger = logging.getLogger(__name__)
        
    async def route_quantum_circuit_with_latency_constraints(
        self,
        circuit: QuantumCircuitRoute,
        latency_constraint: LatencyConstraint,
        optimization_strategy: CircuitOptimizationStrategy = CircuitOptimizationStrategy.BALANCED
    ) -> OptimizedCircuitRoute:
        """Route quantum circuit with strict latency constraints"""
        
        start_time = time.time()
        
        try:
            # Find optimal network paths
            optimal_paths = self.topology_manager.find_optimal_paths(
                circuit.source_region,
                circuit.target_regions,
                latency_constraint,
                max_paths=3
            )
            
            # Check if any paths meet latency constraints
            valid_paths = []
            for target, paths in optimal_paths.items():
                valid_paths.extend([
                    path for path in paths
                    if path.total_latency <= latency_constraint.max_total_latency
                ])
            
            if not valid_paths:
                raise QuantumOptimizationError(
                    f"No network paths meet latency constraint of {latency_constraint.max_total_latency}ms"
                )
            
            # Optimize circuit for latency
            optimized_circuit, optimization_info = self.circuit_optimizer.optimize_circuit_for_latency(
                circuit, valid_paths, latency_constraint, optimization_strategy
            )
            
            # Calculate total estimated latency
            network_latency = min(path.total_latency for path in valid_paths)
            circuit_latency = optimized_circuit.estimated_execution_time * 1000  # Convert to ms
            total_latency = network_latency + circuit_latency
            
            # Create optimized route
            optimized_route = OptimizedCircuitRoute(
                route_id=f"optimized_{circuit.circuit_id}_{uuid.uuid4().hex[:8]}",
                circuit=optimized_circuit,
                network_paths=valid_paths,
                total_estimated_latency=total_latency,
                optimization_strategy=optimization_strategy,
                latency_constraint=latency_constraint,
                quantum_optimization_applied=len(optimization_info.get("optimizations_applied", [])) > 0
            )
            
            # Generate backup routes if possible
            if latency_constraint.latency_class != LatencyClass.ULTRA_LOW:
                backup_routes = await self._generate_backup_routes(optimized_route, latency_constraint)
                optimized_route.backup_routes = backup_routes
            
            # Check final latency compliance
            if not optimized_route.meets_latency_constraint():
                if latency_constraint.latency_class == LatencyClass.ULTRA_LOW:
                    raise QuantumOptimizationError(
                        f"Cannot meet ultra-low latency constraint: estimated {total_latency}ms > required {latency_constraint.max_total_latency}ms"
                    )
                else:
                    # Try classical fallback
                    self.logger.warning(f"Quantum route exceeds latency constraint, enabling classical fallback")
                    optimized_route.classical_fallback_available = True
            
            # Store active route
            self.active_routes[optimized_route.route_id] = optimized_route
            
            # Update metrics
            self.routing_metrics["total_routes"] += 1
            if optimized_route.meets_latency_constraint():
                self.routing_metrics["latency_compliant_routes"] += 1
            
            routing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record routing decision
            self.routing_history.append({
                "route_id": optimized_route.route_id,
                "routing_time": routing_time,
                "total_latency": total_latency,
                "latency_constraint": latency_constraint.max_total_latency,
                "meets_constraint": optimized_route.meets_latency_constraint(),
                "optimization_strategy": optimization_strategy.value,
                "network_paths": len(valid_paths),
                "timestamp": time.time()
            })
            
            self.logger.info(f"Routed quantum circuit {circuit.circuit_id} with latency {total_latency:.2f}ms "
                           f"(constraint: {latency_constraint.max_total_latency}ms)")
            
            return optimized_route
            
        except Exception as e:
            self.logger.error(f"Latency-aware routing failed for circuit {circuit.circuit_id}: {e}")
            raise QuantumOptimizationError(f"Routing failed: {e}")
    
    async def _generate_backup_routes(
        self,
        primary_route: OptimizedCircuitRoute,
        latency_constraint: LatencyConstraint
    ) -> List[OptimizedCircuitRoute]:
        """Generate backup routes for fault tolerance"""
        
        backup_routes = []
        
        try:
            # Relax latency constraint for backup routes
            backup_constraint = LatencyConstraint(
                max_total_latency=latency_constraint.max_total_latency * 1.5,
                max_hop_latency=latency_constraint.max_hop_latency * 1.3,
                jitter_tolerance=latency_constraint.jitter_tolerance * 2.0,
                timeout=latency_constraint.timeout * 1.5,
                retry_limit=latency_constraint.retry_limit,
                latency_class=latency_constraint.latency_class
            )
            
            # Try different optimization strategies
            backup_strategies = [
                CircuitOptimizationStrategy.LATENCY_FIRST,
                CircuitOptimizationStrategy.THROUGHPUT_FIRST,
                CircuitOptimizationStrategy.FIDELITY_FIRST
            ]
            
            for strategy in backup_strategies:
                if strategy == primary_route.optimization_strategy:
                    continue  # Skip same strategy as primary
                
                try:
                    backup_circuit = await self.route_quantum_circuit_with_latency_constraints(
                        primary_route.circuit,
                        backup_constraint,
                        strategy
                    )
                    
                    backup_routes.append(backup_circuit)
                    
                    if len(backup_routes) >= 2:  # Limit backup routes
                        break
                        
                except Exception:
                    continue  # Skip failed backup routes
            
        except Exception as e:
            self.logger.warning(f"Failed to generate backup routes: {e}")
        
        return backup_routes
    
    async def execute_optimized_route(
        self,
        route: OptimizedCircuitRoute
    ) -> Dict[str, Any]:
        """Execute optimized quantum circuit route"""
        
        start_time = time.time()
        
        try:
            # Create execution plan
            execution_plan = self._create_execution_plan(route)
            self.execution_plans[route.route_id] = execution_plan
            
            # Monitor latency during execution
            latency_monitor = self._create_latency_monitor(route)
            
            # Execute circuit using base router
            result = await self.base_router.execute_quantum_circuit(route.circuit)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Check latency compliance
            latency_compliant = execution_time <= route.latency_constraint.max_total_latency
            
            # Update result with latency information
            result.update({
                "route_id": route.route_id,
                "execution_latency": execution_time,
                "latency_compliant": latency_compliant,
                "optimization_strategy": route.optimization_strategy.value,
                "quantum_optimization_applied": route.quantum_optimization_applied,
                "network_paths_used": len(route.network_paths),
                "backup_routes_available": len(route.backup_routes)
            })
            
            # Update metrics
            if result.get("success", False):
                self.routing_metrics["successful_routes"] += 1
                
                # Update average latency
                current_avg = self.routing_metrics["average_latency"]
                total_routes = self.routing_metrics["successful_routes"]
                self.routing_metrics["average_latency"] = (
                    (current_avg * (total_routes - 1) + execution_time) / total_routes
                )
            
            # Clean up
            if route.route_id in self.active_routes:
                del self.active_routes[route.route_id]
            if route.route_id in self.execution_plans:
                del self.execution_plans[route.route_id]
            
            return result
            
        except Exception as e:
            # Try backup routes if available
            if route.backup_routes:
                self.logger.warning(f"Primary route failed, trying backup route: {e}")
                
                for backup_route in route.backup_routes:
                    try:
                        backup_result = await self.execute_optimized_route(backup_route)
                        backup_result["used_backup_route"] = True
                        return backup_result
                    except Exception:
                        continue
            
            # Try classical fallback if enabled
            if route.classical_fallback_available:
                self.logger.warning(f"Quantum routes failed, using classical fallback: {e}")
                
                # Simulate classical execution
                classical_result = {
                    "route_id": route.route_id,
                    "success": True,
                    "execution_latency": route.latency_constraint.max_total_latency * 0.8,  # Within constraint
                    "used_classical_fallback": True,
                    "quantum_advantage": False
                }
                
                return classical_result
            
            raise QuantumOptimizationError(f"Route execution failed: {e}")
    
    def _create_execution_plan(self, route: OptimizedCircuitRoute) -> CircuitExecutionPlan:
        """Create detailed execution plan for optimized route"""
        
        # Create execution phases
        phases = []
        
        # Phase 1: Network setup
        phases.append({
            "phase": "network_setup",
            "estimated_time": 5.0,  # 5ms
            "operations": ["establish_quantum_channels", "verify_entanglement"]
        })
        
        # Phase 2: Circuit execution
        circuit_phases = []
        if route.circuit.circuit_id in self.circuit_optimizer.parallel_gate_sets:
            # Use parallel execution plan
            parallel_groups = self.circuit_optimizer.parallel_gate_sets[route.circuit.circuit_id]
            for i, group in enumerate(parallel_groups):
                circuit_phases.append({
                    "phase": f"parallel_group_{i}",
                    "estimated_time": 0.05 * math.ceil(len(group) / 4),  # 50μs per parallel batch
                    "operations": group
                })
        else:
            # Sequential execution
            for i, gate in enumerate(route.circuit.quantum_gates):
                circuit_phases.append({
                    "phase": f"gate_{i}",
                    "estimated_time": 0.05,  # 50μs per gate
                    "operations": [gate]
                })
        
        phases.extend(circuit_phases)
        
        # Phase 3: Measurement and readout
        phases.append({
            "phase": "measurement",
            "estimated_time": 2.0,  # 2ms
            "operations": ["quantum_measurement", "classical_readout"]
        })
        
        # Calculate critical path
        critical_path_length = sum(phase["estimated_time"] for phase in phases)
        
        # Estimate quantum advantage
        classical_time = len(route.circuit.quantum_gates) * 0.1  # 100μs per classical operation
        quantum_time = route.circuit.estimated_execution_time * 1000  # Convert to ms
        quantum_advantage = max(0.0, (classical_time - quantum_time) / classical_time)
        
        return CircuitExecutionPlan(
            plan_id=f"plan_{route.route_id}",
            circuit_route=route,
            execution_phases=phases,
            parallel_segments=[],  # TODO: implement parallel segment detection
            critical_path_length=critical_path_length,
            resource_requirements=route.circuit.resource_requirements,
            estimated_quantum_advantage=quantum_advantage
        )
    
    def _create_latency_monitor(self, route: OptimizedCircuitRoute) -> Dict[str, Any]:
        """Create latency monitoring for route execution"""
        
        return {
            "route_id": route.route_id,
            "start_time": time.time(),
            "latency_constraint": route.latency_constraint.max_total_latency,
            "checkpoints": [],
            "violations": []
        }
    
    def update_network_conditions(
        self,
        region: DeploymentRegion,
        metrics: Dict[str, Any]
    ):
        """Update network conditions for adaptive routing"""
        
        congestion_level = metrics.get("congestion_level", 0.1)
        
        # Link updates
        link_updates = {}
        for other_region, link_metrics in metrics.get("links", {}).items():
            if isinstance(other_region, str):
                try:
                    other_region = DeploymentRegion(other_region)
                except ValueError:
                    continue
            
            link_updates[other_region] = {
                "latency": link_metrics.get("latency", 100.0),
                "bandwidth": link_metrics.get("bandwidth", 100.0),
                "reliability": link_metrics.get("reliability", 0.95),
                "quantum_fidelity": link_metrics.get("quantum_fidelity", 0.90)
            }
        
        self.topology_manager.update_network_conditions(region, congestion_level, link_updates)
        
        # Update routing metrics
        self._update_routing_metrics()
    
    def _update_routing_metrics(self):
        """Update global routing performance metrics"""
        
        if not self.routing_history:
            return
        
        recent_routes = list(self.routing_history)[-100:]  # Last 100 routes
        
        # Calculate success rate
        successful = sum(1 for r in recent_routes if r.get("meets_constraint", False))
        self.routing_metrics["latency_compliant_routes"] = successful
        
        # Calculate average latency
        latencies = [r["total_latency"] for r in recent_routes if "total_latency" in r]
        if latencies:
            self.routing_metrics["average_latency"] = np.mean(latencies)
        
        # Calculate optimization efficiency
        routing_times = [r["routing_time"] for r in recent_routes if "routing_time" in r]
        if routing_times:
            avg_routing_time = np.mean(routing_times)
            self.routing_metrics["optimization_efficiency"] = max(0.0, 1.0 - (avg_routing_time / 100.0))  # Normalize to 100ms
        
        # Calculate network utilization
        path_counts = [r["network_paths"] for r in recent_routes if "network_paths" in r]
        if path_counts:
            avg_paths = np.mean(path_counts)
            max_paths = 5  # Assume max 5 paths per route
            self.routing_metrics["network_utilization"] = min(1.0, avg_paths / max_paths)
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get comprehensive router status"""
        
        # Update metrics
        self._update_routing_metrics()
        
        # Active routes status
        active_routes_status = {}
        for route_id, route in self.active_routes.items():
            active_routes_status[route_id] = {
                "circuit_id": route.circuit.circuit_id,
                "source_region": route.circuit.source_region.value,
                "target_regions": [r.value for r in route.circuit.target_regions],
                "total_latency": route.total_estimated_latency,
                "latency_constraint": route.latency_constraint.max_total_latency,
                "meets_constraint": route.meets_latency_constraint(),
                "optimization_strategy": route.optimization_strategy.value,
                "backup_routes": len(route.backup_routes)
            }
        
        # Network topology status
        topology_status = {
            "total_regions": len(self.topology_manager.region_coordinates),
            "network_connections": self.topology_manager.network_graph.number_of_edges(),
            "average_congestion": np.mean(list(self.topology_manager.congestion_levels.values())),
            "path_cache_size": len(self.topology_manager.path_cache)
        }
        
        return {
            "routing_metrics": self.routing_metrics,
            "active_routes": active_routes_status,
            "execution_plans": len(self.execution_plans),
            "routing_history_size": len(self.routing_history),
            "network_topology": topology_status,
            "optimization_cache_size": len(self.circuit_optimizer.optimization_cache)
        }


# Global router instance
_latency_aware_router: Optional[LatencyAwareQuantumRouter] = None


def get_latency_aware_router(
    config: FederatedConfig,
    entanglement_coordinator: QuantumEntanglementCoordinator
) -> LatencyAwareQuantumRouter:
    """Get global latency-aware router instance"""
    global _latency_aware_router
    if _latency_aware_router is None:
        _latency_aware_router = LatencyAwareQuantumRouter(config, entanglement_coordinator)
    return _latency_aware_router


async def initialize_latency_aware_routing(
    config: FederatedConfig,
    entanglement_coordinator: QuantumEntanglementCoordinator
) -> LatencyAwareQuantumRouter:
    """Initialize latency-aware quantum circuit routing system"""
    
    router = get_latency_aware_router(config, entanglement_coordinator)
    
    # Start network monitoring loop
    asyncio.create_task(_network_monitoring_loop(router))
    
    logger.info("Latency-aware quantum circuit routing system initialized")
    
    return router


async def _network_monitoring_loop(router: LatencyAwareQuantumRouter):
    """Background loop for network condition monitoring"""
    
    while True:
        try:
            # Simulate network condition updates every 10 seconds
            for region in router.topology_manager.region_coordinates.keys():
                # Simulate changing network conditions
                congestion = 0.1 + 0.3 * abs(np.sin(time.time() / 100.0))  # Oscillating congestion
                
                metrics = {
                    "congestion_level": congestion,
                    "links": {}
                }
                
                router.update_network_conditions(region, metrics)
            
            await asyncio.sleep(10)  # 10 second monitoring interval
            
        except Exception as e:
            logger.error(f"Error in network monitoring loop: {e}")
            await asyncio.sleep(30)  # Back off on error
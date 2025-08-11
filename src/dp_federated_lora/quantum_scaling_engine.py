"""
⚡ Quantum-Enhanced Scaling Engine

Advanced auto-scaling and optimization for federated learning:
- Quantum-inspired resource prediction
- Multi-dimensional scaling strategies  
- Predictive load balancing
- Adaptive resource allocation
- Global optimization with quantum annealing
- Edge computing integration
"""

import asyncio
import logging
import math
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"              # React to current load
    PREDICTIVE = "predictive"          # Predict future load
    QUANTUM_ANNEALING = "quantum_annealing"  # Quantum optimization
    HYBRID = "hybrid"                  # Combine multiple strategies
    EDGE_AWARE = "edge_aware"          # Edge computing optimized


class ResourceType(Enum):
    """Types of resources to scale."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    GPU_UNITS = "gpu_units"
    STORAGE_GB = "storage_gb"
    CLIENT_CONNECTIONS = "client_connections"


@dataclass
class ResourceRequirements:
    """Resource requirements for scaling decisions."""
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    network_bandwidth: float = 100.0  # Mbps
    gpu_units: float = 0.0
    storage_gb: float = 10.0
    client_connections: int = 10
    
    def scale(self, factor: float) -> 'ResourceRequirements':
        """Scale all resources by a factor."""
        return ResourceRequirements(
            cpu_cores=self.cpu_cores * factor,
            memory_gb=self.memory_gb * factor,
            network_bandwidth=self.network_bandwidth * factor,
            gpu_units=self.gpu_units * factor,
            storage_gb=self.storage_gb * factor,
            client_connections=int(self.client_connections * factor)
        )
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "network_bandwidth": self.network_bandwidth,
            "gpu_units": self.gpu_units,
            "storage_gb": self.storage_gb,
            "client_connections": self.client_connections
        }


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    timestamp: float
    strategy: ScalingStrategy
    resource_type: ResourceType
    old_value: float
    new_value: float
    reason: str
    confidence: float  # 0-1 confidence in the decision
    quantum_factor: float = 1.0  # Quantum enhancement factor


class QuantumResourcePredictor:
    """Quantum-inspired resource demand prediction."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.resource_history = {rt: deque(maxlen=history_size) for rt in ResourceType}
        self.quantum_states = {}  # Quantum state representations
        self.coherence_time = 300.0  # Quantum coherence time (seconds)
        
    def update_usage(self, resource_type: ResourceType, usage: float, 
                    timestamp: Optional[float] = None) -> None:
        """Update resource usage history."""
        if timestamp is None:
            timestamp = time.time()
            
        self.resource_history[resource_type].append((timestamp, usage))
        
        # Update quantum state
        self._update_quantum_state(resource_type, usage, timestamp)
        
    def _update_quantum_state(self, resource_type: ResourceType, usage: float, 
                             timestamp: float) -> None:
        """Update quantum state representation of resource usage."""
        if resource_type not in self.quantum_states:
            self.quantum_states[resource_type] = {
                "amplitude": usage,
                "phase": 0.0,
                "last_update": timestamp,
                "coherence": 1.0
            }
        else:
            state = self.quantum_states[resource_type]
            
            # Calculate time decay (decoherence)
            time_delta = timestamp - state["last_update"]
            decoherence_factor = math.exp(-time_delta / self.coherence_time)
            
            # Update quantum state with superposition
            new_amplitude = (state["amplitude"] * decoherence_factor + usage) / 2
            phase_shift = (usage - state["amplitude"]) * 0.1  # Small phase shift
            
            state.update({
                "amplitude": new_amplitude,
                "phase": (state["phase"] + phase_shift) % (2 * math.pi),
                "last_update": timestamp,
                "coherence": decoherence_factor
            })
            
    def predict_demand(self, resource_type: ResourceType, 
                      horizon_minutes: int = 30) -> Tuple[float, float]:
        """Predict resource demand using quantum-enhanced algorithms."""
        history = self.resource_history[resource_type]
        
        if len(history) < 10:
            # Not enough data, use current usage
            current_usage = history[-1][1] if history else 1.0
            return current_usage, 0.5
            
        # Extract recent values and timestamps
        recent_values = [usage for _, usage in list(history)[-50:]]
        recent_times = [ts for ts, _ in list(history)[-50:]]
        
        # Classical prediction components
        trend_prediction = self._calculate_trend(recent_values, recent_times)
        seasonal_prediction = self._calculate_seasonal(recent_values)
        
        # Quantum enhancement
        quantum_state = self.quantum_states.get(resource_type, {})
        quantum_prediction = self._quantum_predict(quantum_state, horizon_minutes)
        
        # Combine predictions with quantum interference
        interference_factor = quantum_state.get("coherence", 0.5)
        final_prediction = (
            trend_prediction * (1 - interference_factor) +
            quantum_prediction * interference_factor +
            seasonal_prediction * 0.3
        )
        
        confidence = min(0.95, len(history) / 100.0 * quantum_state.get("coherence", 0.5))
        
        return max(0.1, final_prediction), confidence
        
    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate trend-based prediction."""
        if len(values) < 3:
            return values[-1] if values else 1.0
            
        # Simple linear regression
        n = len(values)
        x_mean = sum(range(n)) / n
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return values[-1]
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict next value
        return max(0.1, slope * n + intercept)
        
    def _calculate_seasonal(self, values: List[float]) -> float:
        """Calculate seasonal component."""
        if len(values) < 10:
            return sum(values) / len(values) if values else 1.0
            
        # Simple seasonal pattern detection (hourly cycles)
        cycle_length = min(24, len(values) // 3)  # Assume hourly cycles
        if cycle_length < 3:
            return sum(values) / len(values)
            
        seasonal_values = []
        for i in range(cycle_length):
            cycle_values = [values[j] for j in range(i, len(values), cycle_length)]
            if cycle_values:
                seasonal_values.append(sum(cycle_values) / len(cycle_values))
                
        return sum(seasonal_values) / len(seasonal_values) if seasonal_values else values[-1]
        
    def _quantum_predict(self, quantum_state: Dict[str, float], horizon_minutes: int) -> float:
        """Use quantum state for prediction."""
        if not quantum_state:
            return 1.0
            
        amplitude = quantum_state["amplitude"]
        phase = quantum_state["phase"]
        coherence = quantum_state["coherence"]
        
        # Quantum evolution over time
        time_evolution = horizon_minutes * 0.1  # Scale time
        evolved_phase = (phase + time_evolution) % (2 * math.pi)
        
        # Quantum interference effects
        interference = coherence * math.cos(evolved_phase)
        
        # Prediction with quantum effects
        prediction = amplitude * (1 + 0.2 * interference)
        
        return max(0.1, prediction)


class AdaptiveLoadBalancer:
    """Adaptive load balancing with machine learning."""
    
    def __init__(self):
        self.servers = {}
        self.client_assignments = {}
        self.performance_history = defaultdict(list)
        self.assignment_strategy = "weighted_round_robin"
        
    def register_server(self, server_id: str, capacity: ResourceRequirements, 
                       location: str = "unknown") -> None:
        """Register a server for load balancing."""
        self.servers[server_id] = {
            "capacity": capacity,
            "current_load": ResourceRequirements(),
            "location": location,
            "health_score": 1.0,
            "assignments": 0
        }
        
        logger.info(f"🌐 Registered server {server_id} with capacity: {capacity.to_dict()}")
        
    def assign_client(self, client_id: str, requirements: ResourceRequirements,
                     client_location: str = "unknown") -> Optional[str]:
        """Assign client to optimal server."""
        if not self.servers:
            return None
            
        # Calculate server scores
        server_scores = {}
        for server_id, server_info in self.servers.items():
            score = self._calculate_server_score(
                server_info, requirements, client_location
            )
            server_scores[server_id] = score
            
        # Select best server
        best_server = max(server_scores, key=server_scores.get)
        
        # Update server load
        server = self.servers[best_server]
        server["current_load"].cpu_cores += requirements.cpu_cores
        server["current_load"].memory_gb += requirements.memory_gb
        server["current_load"].client_connections += requirements.client_connections
        server["assignments"] += 1
        
        self.client_assignments[client_id] = best_server
        
        logger.debug(f"📡 Assigned client {client_id} to server {best_server}")
        return best_server
        
    def _calculate_server_score(self, server_info: Dict[str, Any],
                               requirements: ResourceRequirements,
                               client_location: str) -> float:
        """Calculate server suitability score."""
        capacity = server_info["capacity"]
        current_load = server_info["current_load"]
        
        # Resource availability score (0-1)
        cpu_ratio = 1 - (current_load.cpu_cores / capacity.cpu_cores)
        memory_ratio = 1 - (current_load.memory_gb / capacity.memory_gb)
        connection_ratio = 1 - (current_load.client_connections / capacity.client_connections)
        
        resource_score = (cpu_ratio + memory_ratio + connection_ratio) / 3
        
        # Health score
        health_score = server_info["health_score"]
        
        # Location affinity (simplified)
        location_score = 1.0
        if client_location != "unknown" and server_info["location"] != "unknown":
            location_score = 1.2 if client_location == server_info["location"] else 0.8
            
        # Combined score
        total_score = resource_score * health_score * location_score
        
        return max(0.0, total_score)
        
    def update_server_performance(self, server_id: str, latency_ms: float,
                                 success_rate: float) -> None:
        """Update server performance metrics."""
        if server_id in self.servers:
            # Update health score based on performance
            performance_score = success_rate * (100.0 / max(latency_ms, 10.0))
            
            # Exponential moving average
            old_health = self.servers[server_id]["health_score"]
            self.servers[server_id]["health_score"] = 0.9 * old_health + 0.1 * performance_score
            
            # Store performance history
            self.performance_history[server_id].append({
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "success_rate": success_rate
            })
            
            # Keep only recent history
            if len(self.performance_history[server_id]) > 100:
                self.performance_history[server_id] = self.performance_history[server_id][-100:]


class QuantumScalingEngine:
    """Main quantum-enhanced scaling engine."""
    
    def __init__(self, initial_capacity: ResourceRequirements):
        self.current_capacity = initial_capacity
        self.min_capacity = initial_capacity.scale(0.5)
        self.max_capacity = initial_capacity.scale(10.0)
        
        self.predictor = QuantumResourcePredictor()
        self.load_balancer = AdaptiveLoadBalancer()
        
        self.scaling_history = deque(maxlen=1000)
        self.scaling_strategies = {
            ScalingStrategy.REACTIVE: self._reactive_scaling,
            ScalingStrategy.PREDICTIVE: self._predictive_scaling,
            ScalingStrategy.QUANTUM_ANNEALING: self._quantum_annealing_scaling,
            ScalingStrategy.HYBRID: self._hybrid_scaling
        }
        
        self.current_strategy = ScalingStrategy.HYBRID
        self.scaling_enabled = True
        
        # Edge computing support
        self.edge_nodes = {}
        self.edge_assignments = {}
        
    def update_current_usage(self, usage: ResourceRequirements) -> None:
        """Update current resource usage."""
        timestamp = time.time()
        
        # Update predictor with individual resource usage
        for resource_type in ResourceType:
            if hasattr(usage, resource_type.value):
                value = getattr(usage, resource_type.value)
                self.predictor.update_usage(resource_type, value, timestamp)
                
    async def auto_scale(self) -> List[ScalingEvent]:
        """Perform automatic scaling based on current strategy."""
        if not self.scaling_enabled:
            return []
            
        scaling_strategy = self.scaling_strategies[self.current_strategy]
        return await scaling_strategy()
        
    async def _reactive_scaling(self) -> List[ScalingEvent]:
        """Reactive scaling based on current load."""
        events = []
        current_time = time.time()
        
        # Get recent usage for each resource type
        for resource_type in ResourceType:
            history = self.predictor.resource_history[resource_type]
            if not history:
                continue
                
            current_usage = history[-1][1]
            current_capacity = getattr(self.current_capacity, resource_type.value)
            
            utilization = current_usage / current_capacity if current_capacity > 0 else 0
            
            # Scale up if utilization > 80%
            if utilization > 0.8:
                scale_factor = 1.5
                new_capacity = current_capacity * scale_factor
                max_capacity = getattr(self.max_capacity, resource_type.value)
                new_capacity = min(new_capacity, max_capacity)
                
                if new_capacity > current_capacity:
                    setattr(self.current_capacity, resource_type.value, new_capacity)
                    
                    event = ScalingEvent(
                        timestamp=current_time,
                        strategy=ScalingStrategy.REACTIVE,
                        resource_type=resource_type,
                        old_value=current_capacity,
                        new_value=new_capacity,
                        reason=f"High utilization: {utilization:.1%}",
                        confidence=0.8
                    )
                    events.append(event)
                    self.scaling_history.append(event)
                    
            # Scale down if utilization < 30%
            elif utilization < 0.3 and len(history) >= 5:
                # Check if consistently low
                recent_utils = [usage / current_capacity for _, usage in list(history)[-5:]]
                if all(u < 0.3 for u in recent_utils):
                    scale_factor = 0.8
                    new_capacity = current_capacity * scale_factor
                    min_capacity = getattr(self.min_capacity, resource_type.value)
                    new_capacity = max(new_capacity, min_capacity)
                    
                    if new_capacity < current_capacity:
                        setattr(self.current_capacity, resource_type.value, new_capacity)
                        
                        event = ScalingEvent(
                            timestamp=current_time,
                            strategy=ScalingStrategy.REACTIVE,
                            resource_type=resource_type,
                            old_value=current_capacity,
                            new_value=new_capacity,
                            reason=f"Low utilization: {utilization:.1%}",
                            confidence=0.7
                        )
                        events.append(event)
                        self.scaling_history.append(event)
                        
        return events
        
    async def _predictive_scaling(self) -> List[ScalingEvent]:
        """Predictive scaling using demand forecasting."""
        events = []
        current_time = time.time()
        
        for resource_type in ResourceType:
            # Predict demand for next 30 minutes
            predicted_demand, confidence = self.predictor.predict_demand(resource_type, 30)
            current_capacity = getattr(self.current_capacity, resource_type.value)
            
            predicted_utilization = predicted_demand / current_capacity if current_capacity > 0 else 0
            
            # Proactive scaling based on prediction
            if predicted_utilization > 0.7 and confidence > 0.6:
                scale_factor = 1.3
                new_capacity = current_capacity * scale_factor
                max_capacity = getattr(self.max_capacity, resource_type.value)
                new_capacity = min(new_capacity, max_capacity)
                
                if new_capacity > current_capacity:
                    setattr(self.current_capacity, resource_type.value, new_capacity)
                    
                    event = ScalingEvent(
                        timestamp=current_time,
                        strategy=ScalingStrategy.PREDICTIVE,
                        resource_type=resource_type,
                        old_value=current_capacity,
                        new_value=new_capacity,
                        reason=f"Predicted high demand: {predicted_utilization:.1%}",
                        confidence=confidence
                    )
                    events.append(event)
                    self.scaling_history.append(event)
                    
        return events
        
    async def _quantum_annealing_scaling(self) -> List[ScalingEvent]:
        """Quantum annealing for optimal resource allocation."""
        events = []
        current_time = time.time()
        
        # Simulate quantum annealing optimization
        optimization_result = await self._quantum_optimize_resources()
        
        for resource_type, optimal_capacity in optimization_result.items():
            current_capacity = getattr(self.current_capacity, resource_type.value)
            
            if abs(optimal_capacity - current_capacity) > current_capacity * 0.1:
                # Apply quantum optimization result
                min_capacity = getattr(self.min_capacity, resource_type.value)
                max_capacity = getattr(self.max_capacity, resource_type.value)
                new_capacity = max(min_capacity, min(optimal_capacity, max_capacity))
                
                setattr(self.current_capacity, resource_type.value, new_capacity)
                
                quantum_state = self.predictor.quantum_states.get(resource_type, {})
                quantum_factor = quantum_state.get("coherence", 1.0)
                
                event = ScalingEvent(
                    timestamp=current_time,
                    strategy=ScalingStrategy.QUANTUM_ANNEALING,
                    resource_type=resource_type,
                    old_value=current_capacity,
                    new_value=new_capacity,
                    reason="Quantum optimization",
                    confidence=0.9,
                    quantum_factor=quantum_factor
                )
                events.append(event)
                self.scaling_history.append(event)
                
        return events
        
    async def _quantum_optimize_resources(self) -> Dict[ResourceType, float]:
        """Simulate quantum annealing for resource optimization."""
        # This would use real quantum annealing in production
        optimization_result = {}
        
        for resource_type in ResourceType:
            history = self.predictor.resource_history[resource_type]
            if not history:
                continue
                
            # Extract recent usage patterns
            recent_usage = [usage for _, usage in list(history)[-20:]]
            if not recent_usage:
                continue
                
            # Quantum-inspired optimization (simplified)
            # In reality, this would formulate as a QUBO problem
            mean_usage = np.mean(recent_usage)
            std_usage = np.std(recent_usage)
            
            # Optimal capacity considering uncertainty and quantum effects
            quantum_state = self.predictor.quantum_states.get(resource_type, {})
            coherence = quantum_state.get("coherence", 1.0)
            
            # Quantum enhancement factor
            quantum_enhancement = 1.0 + 0.2 * coherence * np.cos(quantum_state.get("phase", 0))
            
            optimal_capacity = (mean_usage + 2 * std_usage) * quantum_enhancement
            optimization_result[resource_type] = max(0.1, optimal_capacity)
            
        # Simulate quantum annealing computation time
        await asyncio.sleep(0.1)
        
        return optimization_result
        
    async def _hybrid_scaling(self) -> List[ScalingEvent]:
        """Hybrid scaling combining multiple strategies."""
        all_events = []
        
        # Run all scaling strategies
        reactive_events = await self._reactive_scaling()
        predictive_events = await self._predictive_scaling()
        quantum_events = await self._quantum_annealing_scaling()
        
        # Combine and deduplicate events
        all_strategies_events = reactive_events + predictive_events + quantum_events
        
        # Group by resource type and select best scaling decision
        resource_events = defaultdict(list)
        for event in all_strategies_events:
            resource_events[event.resource_type].append(event)
            
        # Select best event for each resource type
        for resource_type, events in resource_events.items():
            if events:
                # Choose event with highest confidence
                best_event = max(events, key=lambda e: e.confidence)
                all_events.append(best_event)
                
        return all_events
        
    def register_edge_node(self, node_id: str, capacity: ResourceRequirements,
                          location: str, latency_to_cloud: float) -> None:
        """Register an edge computing node."""
        self.edge_nodes[node_id] = {
            "capacity": capacity,
            "location": location,
            "latency_to_cloud": latency_to_cloud,
            "current_load": ResourceRequirements(),
            "active_clients": set()
        }
        
        logger.info(f"🌟 Registered edge node {node_id} at {location}")
        
    def assign_client_to_edge(self, client_id: str, client_location: str,
                             requirements: ResourceRequirements) -> Optional[str]:
        """Assign client to optimal edge node."""
        if not self.edge_nodes:
            return None
            
        best_node = None
        best_score = -1
        
        for node_id, node_info in self.edge_nodes.items():
            # Calculate suitability score
            capacity = node_info["capacity"]
            current_load = node_info["current_load"]
            
            # Resource availability
            cpu_available = capacity.cpu_cores - current_load.cpu_cores
            memory_available = capacity.memory_gb - current_load.memory_gb
            
            if cpu_available < requirements.cpu_cores or memory_available < requirements.memory_gb:
                continue  # Insufficient resources
                
            # Location affinity (simplified distance calculation)
            location_score = 1.0
            if node_info["location"] == client_location:
                location_score = 2.0
                
            # Latency consideration
            latency_score = 100.0 / (node_info["latency_to_cloud"] + 1)
            
            # Combined score
            total_score = location_score * latency_score
            
            if total_score > best_score:
                best_score = total_score
                best_node = node_id
                
        if best_node:
            # Update edge node load
            node = self.edge_nodes[best_node]
            node["current_load"].cpu_cores += requirements.cpu_cores
            node["current_load"].memory_gb += requirements.memory_gb
            node["active_clients"].add(client_id)
            
            self.edge_assignments[client_id] = best_node
            
            logger.info(f"🎯 Assigned client {client_id} to edge node {best_node}")
            
        return best_node
        
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""
        recent_events = [e for e in self.scaling_history if time.time() - e.timestamp < 3600]
        
        strategy_counts = defaultdict(int)
        for event in recent_events:
            strategy_counts[event.strategy.value] += 1
            
        return {
            "current_capacity": self.current_capacity.to_dict(),
            "scaling_strategy": self.current_strategy.value,
            "recent_scaling_events": len(recent_events),
            "strategy_distribution": dict(strategy_counts),
            "edge_nodes": len(self.edge_nodes),
            "edge_assignments": len(self.edge_assignments),
            "quantum_coherence_avg": self._calculate_average_coherence()
        }
        
    def _calculate_average_coherence(self) -> float:
        """Calculate average quantum coherence across all resources."""
        coherences = []
        for quantum_state in self.predictor.quantum_states.values():
            coherences.append(quantum_state.get("coherence", 1.0))
            
        return sum(coherences) / len(coherences) if coherences else 1.0


# Utility function to create scaling engine
def create_quantum_scaling_engine(initial_capacity: ResourceRequirements) -> QuantumScalingEngine:
    """Create a quantum scaling engine."""
    return QuantumScalingEngine(initial_capacity)


# Demo function
async def demo_quantum_scaling():
    """Demonstrate quantum scaling capabilities."""
    print("⚡ Quantum Scaling Engine Demo")
    print("==============================")
    
    # Create scaling engine with initial capacity
    initial_capacity = ResourceRequirements(
        cpu_cores=4.0,
        memory_gb=8.0,
        network_bandwidth=1000.0,
        gpu_units=1.0,
        storage_gb=100.0,
        client_connections=20
    )
    
    engine = create_quantum_scaling_engine(initial_capacity)
    
    # Register some edge nodes
    engine.register_edge_node(
        "edge-us-west", 
        ResourceRequirements(cpu_cores=2.0, memory_gb=4.0, client_connections=10),
        "us-west",
        latency_to_cloud=50.0
    )
    
    engine.register_edge_node(
        "edge-eu-central",
        ResourceRequirements(cpu_cores=3.0, memory_gb=6.0, client_connections=15),
        "eu-central", 
        latency_to_cloud=80.0
    )
    
    # Simulate load patterns and scaling
    print("\n📈 Simulating load patterns...")
    
    for minute in range(10):
        # Simulate varying load
        base_load = 2.0 + math.sin(minute * 0.3) * 1.5  # Sinusoidal pattern
        noise = np.random.normal(0, 0.2)
        
        current_usage = ResourceRequirements(
            cpu_cores=max(0.5, base_load + noise),
            memory_gb=max(1.0, (base_load + noise) * 2),
            network_bandwidth=max(100, (base_load + noise) * 200),
            client_connections=max(5, int((base_load + noise) * 5))
        )
        
        # Update usage
        engine.update_current_usage(current_usage)
        
        # Perform auto-scaling
        scaling_events = await engine.auto_scale()
        
        if scaling_events:
            print(f"  Minute {minute}: {len(scaling_events)} scaling events")
            for event in scaling_events:
                print(f"    {event.resource_type.value}: {event.old_value:.1f} -> {event.new_value:.1f} ({event.reason})")
                
        # Test edge assignment
        if minute % 3 == 0:
            client_requirements = ResourceRequirements(cpu_cores=0.5, memory_gb=1.0)
            edge_node = engine.assign_client_to_edge(f"client_{minute}", "us-west", client_requirements)
            if edge_node:
                print(f"    Assigned client_{minute} to {edge_node}")
                
        await asyncio.sleep(0.5)  # Simulate time passage
        
    # Generate final report
    report = engine.get_scaling_report()
    print("\n📊 Final Scaling Report:")
    print(json.dumps(report, indent=2))
    
    print("\n✅ Quantum scaling demo completed")


if __name__ == "__main__":
    asyncio.run(demo_quantum_scaling())
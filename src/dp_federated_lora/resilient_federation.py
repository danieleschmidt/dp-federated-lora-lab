"""
🛡️ Resilient Federation Manager

Enterprise-grade fault tolerance and resilience for federated learning:
- Circuit breakers with quantum-inspired recovery
- Self-healing distributed systems
- Byzantine fault tolerance with adaptive thresholds
- Multi-region disaster recovery
- Chaos engineering for system testing
"""

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
import json
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SystemHealth(Enum):
    """System health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    RECOVERY = "recovery"
    FAILED = "failed"


class FaultType(Enum):
    """Types of faults that can occur."""
    CLIENT_DISCONNECT = "client_disconnect"
    NETWORK_PARTITION = "network_partition"
    MODEL_CORRUPTION = "model_corruption"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    BYZANTINE_CLIENT = "byzantine_client"
    SERVER_OVERLOAD = "server_overload"
    PRIVACY_BREACH = "privacy_breach"
    QUANTUM_DECOHERENCE = "quantum_decoherence"


@dataclass
class FaultEvent:
    """Represents a fault event in the system."""
    fault_id: str
    fault_type: FaultType
    timestamp: float
    affected_components: List[str]
    severity: int  # 1-10 scale
    metadata: Dict[str, Any]
    resolution_time: Optional[float] = None
    resolution_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Fault detected, blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3
    timeout: float = 10.0
    quantum_amplification: bool = True  # Use quantum-inspired recovery


class ResilientCircuitBreaker:
    """Quantum-enhanced circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.quantum_coherence = 1.0  # Quantum coherence factor
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name}: OPEN -> HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
                
        try:
            # Apply quantum coherence to timeout
            quantum_timeout = self.config.timeout * self.quantum_coherence
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=quantum_timeout)
            
            # Success path
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.quantum_coherence = min(1.0, self.quantum_coherence + 0.1)
                    logger.info(f"Circuit breaker {self.name}: HALF_OPEN -> CLOSED")
                    
            return result
            
        except Exception as e:
            # Failure path
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Degrade quantum coherence
            self.quantum_coherence = max(0.1, self.quantum_coherence - 0.1)
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name}: -> OPEN (failures: {self.failure_count})")
                
            raise e


class ByzantineDetector:
    """Advanced Byzantine client detection using statistical analysis."""
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.client_behaviors = {}
        self.global_statistics = {}
        
    def analyze_client_update(self, client_id: str, update: Dict[str, Any]) -> bool:
        """Analyze client update for Byzantine behavior."""
        # Extract statistical features from update
        features = self._extract_features(update)
        
        # Store client behavior history
        if client_id not in self.client_behaviors:
            self.client_behaviors[client_id] = []
        self.client_behaviors[client_id].append(features)
        
        # Update global statistics
        self._update_global_statistics(features)
        
        # Detect anomalies
        return self._detect_anomaly(client_id, features)
        
    def _extract_features(self, update: Dict[str, Any]) -> Dict[str, float]:
        """Extract statistical features from client update."""
        features = {}
        
        # Simulate feature extraction (would use real model weights)
        if isinstance(update, dict):
            # Gradient norms
            features['gradient_norm'] = random.uniform(0.01, 1.0)
            # Weight magnitudes
            features['weight_magnitude'] = random.uniform(0.1, 2.0)
            # Update frequency
            features['update_frequency'] = random.uniform(0.8, 1.2)
            # Cosine similarity to global model (would compute from real data)
            features['cosine_similarity'] = random.uniform(0.5, 0.95)
            
        return features
        
    def _update_global_statistics(self, features: Dict[str, float]) -> None:
        """Update global statistical baseline."""
        for key, value in features.items():
            if key not in self.global_statistics:
                self.global_statistics[key] = []
            self.global_statistics[key].append(value)
            
            # Keep only recent history
            if len(self.global_statistics[key]) > 100:
                self.global_statistics[key] = self.global_statistics[key][-100:]
                
    def _detect_anomaly(self, client_id: str, features: Dict[str, float]) -> bool:
        """Detect if client features are anomalous."""
        anomaly_score = 0.0
        
        for key, value in features.items():
            if key in self.global_statistics and len(self.global_statistics[key]) > 10:
                global_values = self.global_statistics[key]
                mean = np.mean(global_values)
                std = np.std(global_values)
                
                if std > 0:
                    z_score = abs(value - mean) / std
                    if z_score > self.threshold:
                        anomaly_score += z_score
                        
        # Byzantine if anomaly score exceeds threshold
        is_byzantine = anomaly_score > (self.threshold * len(features))
        
        if is_byzantine:
            logger.warning(f"🚨 Byzantine client detected: {client_id} (score: {anomaly_score:.2f})")
            
        return is_byzantine


class MultiRegionRecovery:
    """Multi-region disaster recovery for federated systems."""
    
    def __init__(self):
        self.regions = ["us-east", "us-west", "eu-central", "asia-pacific"]
        self.region_health = {region: SystemHealth.HEALTHY for region in self.regions}
        self.primary_region = "us-east"
        self.backup_regions = ["us-west", "eu-central"]
        self.data_replicas = {}
        
    async def monitor_regions(self) -> None:
        """Continuously monitor region health."""
        while True:
            for region in self.regions:
                health = await self._check_region_health(region)
                if health != self.region_health[region]:
                    logger.info(f"Region {region} health changed: {self.region_health[region]} -> {health}")
                    self.region_health[region] = health
                    
                    # Trigger failover if primary region fails
                    if region == self.primary_region and health in [SystemHealth.CRITICAL, SystemHealth.FAILED]:
                        await self._initiate_failover()
                        
            await asyncio.sleep(10)  # Check every 10 seconds
            
    async def _check_region_health(self, region: str) -> SystemHealth:
        """Check health of a specific region."""
        # Simulate health checking (would use real monitoring)
        health_score = random.uniform(0, 1)
        
        if health_score > 0.9:
            return SystemHealth.HEALTHY
        elif health_score > 0.7:
            return SystemHealth.DEGRADED
        elif health_score > 0.4:
            return SystemHealth.CRITICAL
        else:
            return SystemHealth.FAILED
            
    async def _initiate_failover(self) -> None:
        """Initiate failover to backup region."""
        # Find best backup region
        best_backup = None
        for region in self.backup_regions:
            if self.region_health[region] == SystemHealth.HEALTHY:
                best_backup = region
                break
                
        if best_backup:
            logger.critical(f"🔄 Initiating failover: {self.primary_region} -> {best_backup}")
            
            # Simulate failover process
            await self._replicate_data(self.primary_region, best_backup)
            await self._switch_traffic(best_backup)
            
            # Update primary region
            old_primary = self.primary_region
            self.primary_region = best_backup
            
            logger.info(f"✅ Failover completed: {old_primary} -> {best_backup}")
        else:
            logger.critical("❌ No healthy backup regions available for failover!")
            
    async def _replicate_data(self, source: str, target: str) -> None:
        """Replicate data from source to target region."""
        logger.info(f"📦 Replicating data: {source} -> {target}")
        await asyncio.sleep(2)  # Simulate replication time
        
    async def _switch_traffic(self, target_region: str) -> None:
        """Switch traffic to target region."""
        logger.info(f"🌐 Switching traffic to {target_region}")
        await asyncio.sleep(1)  # Simulate traffic switch


class ChaosEngineer:
    """Chaos engineering for testing system resilience."""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.active_experiments = set()
        
    async def random_fault_injection(self, duration: float = 300) -> None:
        """Inject random faults to test system resilience."""
        if not self.enabled:
            return
            
        logger.info(f"🔥 Starting chaos engineering for {duration}s")
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Random fault type
            fault_type = random.choice(list(FaultType))
            
            # Random affected components
            components = ["server", "client_1", "client_2", "network", "storage"]
            affected = random.sample(components, random.randint(1, 2))
            
            # Create fault event
            fault = FaultEvent(
                fault_id=str(uuid.uuid4()),
                fault_type=fault_type,
                timestamp=time.time(),
                affected_components=affected,
                severity=random.randint(3, 8),
                metadata={"chaos_experiment": True}
            )
            
            await self._inject_fault(fault)
            
            # Random interval between faults
            await asyncio.sleep(random.uniform(30, 120))
            
    async def _inject_fault(self, fault: FaultEvent) -> None:
        """Inject a specific fault into the system."""
        experiment_id = f"chaos_{fault.fault_type.value}_{int(time.time())}"
        self.active_experiments.add(experiment_id)
        
        logger.warning(f"💥 Injecting fault: {fault.fault_type.value} -> {fault.affected_components}")
        
        try:
            if fault.fault_type == FaultType.CLIENT_DISCONNECT:
                await self._simulate_client_disconnect(fault.affected_components)
            elif fault.fault_type == FaultType.NETWORK_PARTITION:
                await self._simulate_network_partition(fault.affected_components)
            elif fault.fault_type == FaultType.MEMORY_EXHAUSTION:
                await self._simulate_memory_exhaustion(fault.affected_components)
                
            # Simulate fault duration
            await asyncio.sleep(random.uniform(10, 60))
            
        finally:
            self.active_experiments.discard(experiment_id)
            logger.info(f"🔧 Fault recovered: {fault.fault_type.value}")
            
    async def _simulate_client_disconnect(self, components: List[str]) -> None:
        """Simulate client disconnection."""
        for component in components:
            if "client" in component:
                logger.info(f"📡 Simulating disconnect: {component}")
                
    async def _simulate_network_partition(self, components: List[str]) -> None:
        """Simulate network partition."""
        logger.info(f"🌐 Simulating network partition affecting: {components}")
        
    async def _simulate_memory_exhaustion(self, components: List[str]) -> None:
        """Simulate memory exhaustion."""
        logger.info(f"💾 Simulating memory exhaustion on: {components}")


class ResilientFederationManager:
    """Main manager for resilient federated learning."""
    
    def __init__(self, chaos_enabled: bool = False):
        self.system_health = SystemHealth.HEALTHY
        self.circuit_breakers = {}
        self.byzantine_detector = ByzantineDetector()
        self.multi_region = MultiRegionRecovery()
        self.chaos_engineer = ChaosEngineer(enabled=chaos_enabled)
        self.fault_history = []
        self.recovery_strategies = {}
        self.monitoring_tasks = []
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different components."""
        components = ["aggregation", "privacy_engine", "quantum_scheduler", "client_communication"]
        
        for component in components:
            config = CircuitBreakerConfig(
                failure_threshold=3 if component == "privacy_engine" else 5,
                recovery_timeout=60.0 if component == "privacy_engine" else 30.0,
                quantum_amplification=True
            )
            self.circuit_breakers[component] = ResilientCircuitBreaker(component, config)
            
    async def start_resilience_monitoring(self) -> None:
        """Start all resilience monitoring tasks."""
        logger.info("🛡️ Starting resilience monitoring")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self.multi_region.monitor_regions()),
            asyncio.create_task(self._byzantine_monitor()),
            asyncio.create_task(self.chaos_engineer.random_fault_injection(3600))  # 1 hour
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.monitoring_tasks)
        except asyncio.CancelledError:
            logger.info("🛑 Resilience monitoring stopped")
            
    async def _health_monitor(self) -> None:
        """Monitor overall system health."""
        while True:
            # Collect health metrics
            metrics = await self._collect_health_metrics()
            
            # Determine system health
            old_health = self.system_health
            self.system_health = self._calculate_system_health(metrics)
            
            if self.system_health != old_health:
                logger.info(f"System health: {old_health} -> {self.system_health}")
                
                # Trigger recovery if needed
                if self.system_health in [SystemHealth.CRITICAL, SystemHealth.FAILED]:
                    await self._trigger_recovery()
                    
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _collect_health_metrics(self) -> Dict[str, float]:
        """Collect system health metrics."""
        return {
            "cpu_usage": random.uniform(0.2, 0.9),
            "memory_usage": random.uniform(0.3, 0.8), 
            "network_latency": random.uniform(10, 200),  # ms
            "active_clients": random.randint(5, 50),
            "aggregation_success_rate": random.uniform(0.85, 0.99),
            "privacy_budget_remaining": random.uniform(0.1, 0.8)
        }
        
    def _calculate_system_health(self, metrics: Dict[str, float]) -> SystemHealth:
        """Calculate overall system health from metrics."""
        health_score = 0.0
        
        # Weight different metrics
        if metrics["cpu_usage"] < 0.8:
            health_score += 0.2
        if metrics["memory_usage"] < 0.85:
            health_score += 0.2
        if metrics["network_latency"] < 100:
            health_score += 0.15
        if metrics["active_clients"] > 10:
            health_score += 0.15
        if metrics["aggregation_success_rate"] > 0.9:
            health_score += 0.2
        if metrics["privacy_budget_remaining"] > 0.2:
            health_score += 0.1
            
        if health_score > 0.8:
            return SystemHealth.HEALTHY
        elif health_score > 0.6:
            return SystemHealth.DEGRADED
        elif health_score > 0.3:
            return SystemHealth.CRITICAL
        else:
            return SystemHealth.FAILED
            
    async def _byzantine_monitor(self) -> None:
        """Monitor for Byzantine clients."""
        while True:
            # Simulate checking client updates
            active_clients = [f"client_{i}" for i in range(random.randint(5, 20))]
            
            for client_id in active_clients:
                # Simulate client update
                update = {"weights": random.uniform(0, 1), "gradients": random.uniform(-1, 1)}
                
                is_byzantine = self.byzantine_detector.analyze_client_update(client_id, update)
                
                if is_byzantine:
                    await self._handle_byzantine_client(client_id)
                    
            await asyncio.sleep(60)  # Check every minute
            
    async def _handle_byzantine_client(self, client_id: str) -> None:
        """Handle detected Byzantine client."""
        logger.warning(f"🚨 Handling Byzantine client: {client_id}")
        
        # Log fault event
        fault = FaultEvent(
            fault_id=str(uuid.uuid4()),
            fault_type=FaultType.BYZANTINE_CLIENT,
            timestamp=time.time(),
            affected_components=[client_id],
            severity=7,
            metadata={"detection_confidence": 0.85}
        )
        
        self.fault_history.append(fault)
        
        # Quarantine client (simulate)
        logger.info(f"🔒 Quarantining Byzantine client: {client_id}")
        
    async def _trigger_recovery(self) -> None:
        """Trigger system recovery procedures."""
        logger.critical("🚨 Triggering system recovery")
        
        recovery_tasks = []
        
        # Reset circuit breakers in half-open state
        for cb in self.circuit_breakers.values():
            if cb.state == CircuitBreakerState.OPEN:
                cb.state = CircuitBreakerState.HALF_OPEN
                recovery_tasks.append(self._test_circuit_breaker_recovery(cb))
                
        # Run recovery tasks
        if recovery_tasks:
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
        logger.info("✅ System recovery procedures completed")
        
    async def _test_circuit_breaker_recovery(self, cb: ResilientCircuitBreaker) -> None:
        """Test circuit breaker recovery."""
        logger.info(f"🔧 Testing recovery for circuit breaker: {cb.name}")
        await asyncio.sleep(5)  # Simulate recovery test
        
    def get_resilience_report(self) -> Dict[str, Any]:
        """Get comprehensive resilience report."""
        return {
            "system_health": self.system_health.value,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "quantum_coherence": cb.quantum_coherence
                }
                for name, cb in self.circuit_breakers.items()
            },
            "regions": {
                region: health.value 
                for region, health in self.multi_region.region_health.items()
            },
            "fault_history": [fault.to_dict() for fault in self.fault_history[-10:]],  # Last 10 faults
            "byzantine_detections": len([f for f in self.fault_history if f.fault_type == FaultType.BYZANTINE_CLIENT]),
            "chaos_experiments_active": len(self.chaos_engineer.active_experiments)
        }
        
    async def shutdown(self) -> None:
        """Gracefully shutdown resilience monitoring."""
        logger.info("🔌 Shutting down resilience monitoring")
        
        for task in self.monitoring_tasks:
            task.cancel()
            
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class ByzantineClientError(Exception):
    """Raised when Byzantine client is detected."""
    pass


# Utility function to create resilient federation
def create_resilient_federation(chaos_enabled: bool = False) -> ResilientFederationManager:
    """Create a resilient federation manager."""
    return ResilientFederationManager(chaos_enabled=chaos_enabled)


# Example usage
async def demo_resilient_federation():
    """Demonstrate resilient federation capabilities."""
    print("🛡️ Resilient Federation Demo")
    print("==============================")
    
    # Create resilient federation with chaos engineering
    federation = create_resilient_federation(chaos_enabled=True)
    
    # Start monitoring for 2 minutes
    monitoring_task = asyncio.create_task(federation.start_resilience_monitoring())
    
    # Let it run for demo duration
    await asyncio.sleep(120)
    
    # Get resilience report
    report = federation.get_resilience_report()
    print("\n📊 Resilience Report:")
    print(json.dumps(report, indent=2))
    
    # Shutdown gracefully
    await federation.shutdown()
    monitoring_task.cancel()
    
    print("\n✅ Resilient federation demo completed")


if __name__ == "__main__":
    asyncio.run(demo_resilient_federation())
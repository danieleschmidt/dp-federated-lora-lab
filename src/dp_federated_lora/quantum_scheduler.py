"""
Quantum-Inspired Task Scheduler for Federated Learning

Implements quantum optimization algorithms to enhance federated learning task scheduling,
client selection, and resource allocation using quantum annealing principles.
"""

import asyncio
import logging
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import combinations

import torch
from pydantic import BaseModel, Field, validator
from .config import FederatedConfig
from .monitoring import MetricsCollector
from .exceptions import QuantumSchedulingError


class QuantumState(Enum):
    """Quantum-inspired states for task scheduling"""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled" 
    MEASURED = "measured"
    COHERENT = "coherent"
    DECOHERENT = "decoherent"


@dataclass
class QuantumTask:
    """Quantum-inspired task representation"""
    task_id: str
    client_id: str
    priority: float
    complexity: float
    resource_requirements: Dict[str, float]
    dependencies: Set[str] = field(default_factory=set)
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    amplitude: complex = 1.0 + 0j
    phase: float = 0.0
    entangled_tasks: Set[str] = field(default_factory=set)
    
    @property
    def probability(self) -> float:
        """Quantum probability amplitude squared"""
        return abs(self.amplitude) ** 2
    
    def collapse_wavefunction(self) -> None:
        """Collapse quantum state to measured state"""
        self.quantum_state = QuantumState.MEASURED
        self.amplitude = complex(np.sqrt(self.probability), 0)


@dataclass 
class QuantumClient:
    """Quantum-enhanced client representation"""
    client_id: str
    availability: float
    computational_power: float
    network_latency: float
    reliability_score: float
    privacy_budget: float
    quantum_coherence: float = 1.0
    entanglement_strength: float = 0.0
    last_task_completion: Optional[float] = None
    
    def calculate_quantum_fitness(self, task: QuantumTask) -> float:
        """Calculate quantum-inspired fitness for task assignment"""
        base_fitness = (
            self.availability * 0.3 +
            self.computational_power * 0.25 +
            (1 - self.network_latency) * 0.2 +
            self.reliability_score * 0.15 +
            self.privacy_budget * 0.1
        )
        
        # Quantum coherence enhancement
        quantum_bonus = self.quantum_coherence * 0.1
        
        # Entanglement effects
        if task.task_id in [t for client_tasks in self.entangled_tasks for t in client_tasks]:
            quantum_bonus += self.entanglement_strength * 0.05
            
        return base_fitness + quantum_bonus
    
    @property
    def entangled_tasks(self) -> Set[str]:
        """Get tasks this client is entangled with"""
        return getattr(self, '_entangled_tasks', set())


class QuantumAnnealingOptimizer:
    """Quantum annealing-inspired optimization for task scheduling"""
    
    def __init__(
        self, 
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 1000
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
        
    def optimize_scheduling(
        self, 
        tasks: List[QuantumTask],
        clients: List[QuantumClient],
        constraints: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Optimize task-to-client assignment using quantum annealing
        
        Returns:
            Dict mapping task_id to client_id
        """
        n_tasks = len(tasks)
        n_clients = len(clients)
        
        if n_tasks == 0 or n_clients == 0:
            return {}
            
        # Initialize random assignment
        current_assignment = {
            task.task_id: np.random.choice([c.client_id for c in clients])
            for task in tasks
        }
        
        current_energy = self._calculate_energy(current_assignment, tasks, clients, constraints)
        best_assignment = current_assignment.copy()
        best_energy = current_energy
        
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            if temperature < self.min_temperature:
                break
                
            # Generate neighbor solution
            new_assignment = self._generate_neighbor(current_assignment, clients)
            new_energy = self._calculate_energy(new_assignment, tasks, clients, constraints)
            
            # Accept or reject based on quantum probability
            delta_energy = new_energy - current_energy
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                current_assignment = new_assignment
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_assignment = new_assignment.copy()
                    best_energy = new_energy
                    
            temperature *= self.cooling_rate
            
        self.logger.info(f"Quantum annealing completed: {best_energy:.4f} energy, {iteration+1} iterations")
        return best_assignment
    
    def _calculate_energy(
        self,
        assignment: Dict[str, str],
        tasks: List[QuantumTask], 
        clients: List[QuantumClient],
        constraints: Dict[str, Any]
    ) -> float:
        """Calculate system energy for current assignment"""
        task_map = {t.task_id: t for t in tasks}
        client_map = {c.client_id: c for c in clients}
        
        total_energy = 0.0
        
        # Task-client mismatch penalty
        for task_id, client_id in assignment.items():
            task = task_map[task_id]
            client = client_map[client_id]
            
            # Fitness penalty (lower fitness = higher energy)
            fitness = client.calculate_quantum_fitness(task)
            total_energy += (1 - fitness) * 10
            
            # Resource constraint violations
            for resource, requirement in task.resource_requirements.items():
                available = getattr(client, resource, 1.0)
                if requirement > available:
                    total_energy += (requirement - available) * 50
                    
        # Load balancing penalty
        client_loads = {}
        for client_id in assignment.values():
            client_loads[client_id] = client_loads.get(client_id, 0) + 1
            
        if len(client_loads) > 1:
            load_variance = np.var(list(client_loads.values()))
            total_energy += load_variance * 5
            
        # Dependency constraint violations
        for task in tasks:
            if task.dependencies:
                task_client = assignment[task.task_id]
                for dep_task_id in task.dependencies:
                    if dep_task_id in assignment:
                        dep_client = assignment[dep_task_id]
                        if task_client != dep_client:
                            total_energy += 20  # Penalty for cross-client dependencies
                            
        return total_energy
    
    def _generate_neighbor(
        self, 
        assignment: Dict[str, str], 
        clients: List[QuantumClient]
    ) -> Dict[str, str]:
        """Generate neighbor solution by randomly reassigning tasks"""
        new_assignment = assignment.copy()
        
        # Randomly select task to reassign
        task_id = np.random.choice(list(assignment.keys()))
        client_ids = [c.client_id for c in clients]
        
        # Assign to different random client
        current_client = assignment[task_id]
        available_clients = [c for c in client_ids if c != current_client]
        
        if available_clients:
            new_assignment[task_id] = np.random.choice(available_clients)
            
        return new_assignment


class QuantumEntanglementManager:
    """Manages quantum entanglement between tasks and clients"""
    
    def __init__(self):
        self.entanglement_matrix: Dict[Tuple[str, str], float] = {}
        self.logger = logging.getLogger(__name__)
        
    def create_entanglement(
        self, 
        entity1: str, 
        entity2: str, 
        strength: float = 1.0
    ) -> None:
        """Create quantum entanglement between two entities"""
        key = tuple(sorted([entity1, entity2]))
        self.entanglement_matrix[key] = strength
        self.logger.debug(f"Created entanglement: {entity1} <-> {entity2} (strength: {strength})")
        
    def get_entanglement_strength(self, entity1: str, entity2: str) -> float:
        """Get entanglement strength between entities"""
        key = tuple(sorted([entity1, entity2]))
        return self.entanglement_matrix.get(key, 0.0)
        
    def decoherence_step(self, decay_rate: float = 0.01) -> None:
        """Apply quantum decoherence to reduce entanglement over time"""
        for key in list(self.entanglement_matrix.keys()):
            self.entanglement_matrix[key] *= (1 - decay_rate)
            if self.entanglement_matrix[key] < 0.01:
                del self.entanglement_matrix[key]


class QuantumTaskScheduler:
    """Main quantum-inspired task scheduler"""
    
    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config or FederatedConfig()
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        self.tasks: Dict[str, QuantumTask] = {}
        self.clients: Dict[str, QuantumClient] = {}
        
        self.optimizer = QuantumAnnealingOptimizer()
        self.entanglement_manager = QuantumEntanglementManager()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._quantum_coherence_time = 10.0  # seconds
        self._last_coherence_update = time.time()
        
    async def register_client(
        self, 
        client_id: str, 
        capabilities: Dict[str, float],
        quantum_properties: Optional[Dict[str, float]] = None
    ) -> None:
        """Register a client with quantum properties"""
        quantum_props = quantum_properties or {}
        
        client = QuantumClient(
            client_id=client_id,
            availability=capabilities.get('availability', 0.8),
            computational_power=capabilities.get('computational_power', 0.5),
            network_latency=capabilities.get('network_latency', 0.1),
            reliability_score=capabilities.get('reliability_score', 0.8),
            privacy_budget=capabilities.get('privacy_budget', 1.0),
            quantum_coherence=quantum_props.get('quantum_coherence', 1.0),
            entanglement_strength=quantum_props.get('entanglement_strength', 0.0)
        )
        
        self.clients[client_id] = client
        self.logger.info(f"Registered quantum client: {client_id}")
        
        # Create entanglements with existing clients based on similarity
        await self._create_client_entanglements(client)
        
    async def submit_task(
        self,
        task_id: str,
        client_preference: Optional[str] = None,
        priority: float = 1.0,
        complexity: float = 1.0,
        resource_requirements: Optional[Dict[str, float]] = None,
        dependencies: Optional[Set[str]] = None,
        quantum_properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Submit a task with quantum properties"""
        quantum_props = quantum_properties or {}
        
        task = QuantumTask(
            task_id=task_id,
            client_id=client_preference or "",
            priority=priority,
            complexity=complexity,
            resource_requirements=resource_requirements or {},
            dependencies=dependencies or set(),
            amplitude=complex(quantum_props.get('amplitude_real', 1.0),
                           quantum_props.get('amplitude_imag', 0.0)),
            phase=quantum_props.get('phase', 0.0)
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"Submitted quantum task: {task_id}")
        
        # Create task entanglements
        await self._create_task_entanglements(task)
        
    async def schedule_round(
        self, 
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Schedule a round of federated learning using quantum optimization
        
        Returns:
            Dict mapping task_id to client_id
        """
        if not self.tasks or not self.clients:
            self.logger.warning("No tasks or clients available for scheduling")
            return {}
            
        self.logger.info(f"Starting quantum scheduling round: {len(self.tasks)} tasks, {len(self.clients)} clients")
        
        # Update quantum coherence
        await self._update_quantum_coherence()
        
        # Prepare tasks and clients
        active_tasks = [t for t in self.tasks.values() 
                       if t.quantum_state != QuantumState.MEASURED]
        available_clients = list(self.clients.values())
        
        # Run quantum optimization
        start_time = time.time()
        assignment = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.optimizer.optimize_scheduling,
            active_tasks,
            available_clients,
            constraints or {}
        )
        optimization_time = time.time() - start_time
        
        # Collapse wavefunctions for assigned tasks
        for task_id in assignment:
            if task_id in self.tasks:
                self.tasks[task_id].collapse_wavefunction()
                
        # Update entanglements based on assignment
        await self._update_entanglements_from_assignment(assignment)
        
        # Record metrics
        self.metrics.record_metric("quantum_scheduling_time", optimization_time)
        self.metrics.record_metric("quantum_tasks_scheduled", len(assignment))
        self.metrics.record_metric("quantum_clients_utilized", len(set(assignment.values())))
        
        self.logger.info(f"Quantum scheduling completed in {optimization_time:.3f}s")
        return assignment
        
    async def _create_client_entanglements(self, new_client: QuantumClient) -> None:
        """Create entanglements between clients based on similarity"""
        for existing_client in self.clients.values():
            if existing_client.client_id == new_client.client_id:
                continue
                
            # Calculate similarity
            similarity = self._calculate_client_similarity(new_client, existing_client)
            
            if similarity > 0.7:  # High similarity threshold
                strength = min(similarity, 1.0)
                self.entanglement_manager.create_entanglement(
                    new_client.client_id,
                    existing_client.client_id,
                    strength
                )
                
    async def _create_task_entanglements(self, new_task: QuantumTask) -> None:
        """Create entanglements between tasks based on dependencies and similarity"""
        for existing_task in self.tasks.values():
            if existing_task.task_id == new_task.task_id:
                continue
                
            # Dependency-based entanglement
            if (new_task.task_id in existing_task.dependencies or 
                existing_task.task_id in new_task.dependencies):
                self.entanglement_manager.create_entanglement(
                    new_task.task_id,
                    existing_task.task_id,
                    0.8  # Strong entanglement for dependencies
                )
                new_task.entangled_tasks.add(existing_task.task_id)
                existing_task.entangled_tasks.add(new_task.task_id)
                
    def _calculate_client_similarity(
        self, 
        client1: QuantumClient, 
        client2: QuantumClient
    ) -> float:
        """Calculate similarity between two clients"""
        attributes = ['availability', 'computational_power', 'network_latency', 
                     'reliability_score', 'privacy_budget']
        
        similarity = 0.0
        for attr in attributes:
            val1 = getattr(client1, attr)
            val2 = getattr(client2, attr)
            similarity += 1.0 - abs(val1 - val2)
            
        return similarity / len(attributes)
        
    async def _update_quantum_coherence(self) -> None:
        """Update quantum coherence and apply decoherence"""
        current_time = time.time()
        time_elapsed = current_time - self._last_coherence_update
        
        if time_elapsed > self._quantum_coherence_time:
            # Apply decoherence
            self.entanglement_manager.decoherence_step()
            
            # Update client coherence
            for client in self.clients.values():
                client.quantum_coherence *= 0.99  # Gradual decoherence
                client.quantum_coherence = max(client.quantum_coherence, 0.1)
                
            self._last_coherence_update = current_time
            
    async def _update_entanglements_from_assignment(
        self, 
        assignment: Dict[str, str]
    ) -> None:
        """Update entanglements based on task assignment results"""
        # Strengthen entanglements between co-located tasks
        client_tasks = {}
        for task_id, client_id in assignment.items():
            if client_id not in client_tasks:
                client_tasks[client_id] = []
            client_tasks[client_id].append(task_id)
            
        for client_id, task_ids in client_tasks.items():
            if len(task_ids) > 1:
                # Create or strengthen entanglements between tasks on same client
                for task1, task2 in combinations(task_ids, 2):
                    current_strength = self.entanglement_manager.get_entanglement_strength(
                        task1, task2
                    )
                    new_strength = min(current_strength + 0.1, 1.0)
                    self.entanglement_manager.create_entanglement(
                        task1, task2, new_strength
                    )
                    
    async def get_quantum_state_metrics(self) -> Dict[str, Any]:
        """Get current quantum state metrics"""
        task_states = {}
        for state in QuantumState:
            task_states[state.value] = len([
                t for t in self.tasks.values() if t.quantum_state == state
            ])
            
        entanglement_count = len(self.entanglement_manager.entanglement_matrix)
        avg_coherence = np.mean([c.quantum_coherence for c in self.clients.values()]) if self.clients else 0
        
        return {
            "task_states": task_states,
            "total_entanglements": entanglement_count,
            "average_coherence": avg_coherence,
            "total_tasks": len(self.tasks),
            "total_clients": len(self.clients)
        }
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("Quantum scheduler cleanup completed")


# Global scheduler instance
_quantum_scheduler: Optional[QuantumTaskScheduler] = None


def get_quantum_scheduler(
    config: Optional[FederatedConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> QuantumTaskScheduler:
    """Get global quantum scheduler instance"""
    global _quantum_scheduler
    if _quantum_scheduler is None:
        _quantum_scheduler = QuantumTaskScheduler(config, metrics_collector)
    return _quantum_scheduler


async def initialize_quantum_scheduling(
    config: Optional[FederatedConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> QuantumTaskScheduler:
    """Initialize quantum scheduling system"""
    scheduler = get_quantum_scheduler(config, metrics_collector)
    return scheduler
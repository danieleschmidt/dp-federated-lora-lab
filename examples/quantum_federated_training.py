#!/usr/bin/env python3
"""
Quantum-Enhanced Federated Learning Example

Demonstrates how to use the quantum-inspired task planning features
of dp-federated-lora-lab for optimized federated learning workflows.
"""

import asyncio
import logging
import time
from typing import Dict, List

# Core components
from dp_federated_lora import (
    FederatedServer,
    DPLoRAClient,
    FederatedConfig,
    PrivacyConfig,
    create_default_config,
)

# Quantum-inspired components
from dp_federated_lora import (
    QuantumTaskScheduler,
    QuantumPrivacyEngine,
    QuantumInspiredOptimizer,
    QuantumMetricsCollector,
    QuantumResilienceManager,
    QuantumAutoScaler,
    get_quantum_scheduler,
    get_quantum_metrics_collector,
    get_global_resilience_manager,
    initialize_quantum_auto_scaling,
    quantum_resilient,
    QuantumMetricType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumFederatedTraining:
    """Quantum-enhanced federated training orchestrator"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        num_clients: int = 10,
        num_rounds: int = 20,
        privacy_budget: Dict[str, float] = None
    ):
        self.model_name = model_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.privacy_budget = privacy_budget or {"epsilon": 8.0, "delta": 1e-5}
        
        # Initialize configuration
        self.config = create_default_config()
        self.config.model_name = model_name
        self.config.num_rounds = num_rounds
        self.config.max_clients = num_clients
        self.config.privacy.epsilon = self.privacy_budget["epsilon"]
        self.config.privacy.delta = self.privacy_budget["delta"]
        
        # Initialize quantum components
        self.quantum_metrics = get_quantum_metrics_collector(self.config)
        self.resilience_manager = get_global_resilience_manager(self.config, self.quantum_metrics)
        self.quantum_scheduler = get_quantum_scheduler(self.config, self.quantum_metrics)
        
        # Initialize components
        self.server = None
        self.clients = []
        self.auto_scaler = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Quantum-Enhanced Federated Learning System")
        
        # Initialize server with quantum enhancements
        self.server = FederatedServer(
            model_name=self.model_name,
            config=self.config,
            num_clients=self.num_clients,
            rounds=self.num_rounds,
            privacy_budget=self.privacy_budget
        )
        
        # Initialize auto-scaler
        self.auto_scaler = await initialize_quantum_auto_scaling(
            self.config, 
            self.quantum_metrics, 
            self.resilience_manager
        )
        
        # Create quantum-optimized clients
        await self._create_quantum_clients()
        
        logger.info(f"✅ Initialization complete: {len(self.clients)} clients ready")
        
    async def _create_quantum_clients(self):
        """Create clients with quantum-enhanced capabilities"""
        for i in range(self.num_clients):
            client_id = f"quantum_client_{i:03d}"
            
            # Register client with quantum scheduler
            client_capabilities = {
                'availability': 0.8 + 0.2 * (i / self.num_clients),  # Varied availability
                'computational_power': 0.5 + 0.5 * (i / self.num_clients),  # Varied power
                'network_latency': 0.1 + 0.1 * (i / self.num_clients),  # Varied latency
                'reliability_score': 0.7 + 0.3 * (1 - i / self.num_clients),  # Varied reliability
                'privacy_budget': 1.0
            }
            
            quantum_properties = {
                'quantum_coherence': 0.8 + 0.2 * (i / self.num_clients),
                'entanglement_strength': 0.3 + 0.4 * (i / self.num_clients)
            }
            
            await self.quantum_scheduler.register_client(
                client_id, client_capabilities, quantum_properties
            )
            
            logger.info(f"📡 Registered quantum client {client_id}")
            
    @quantum_resilient(
        circuit_breaker_name="federated_training",
        retry_strategy_name="quantum_training",
        max_retries=3
    )
    async def run_quantum_federated_training(self):
        """Run quantum-enhanced federated training"""
        logger.info("🎯 Starting Quantum-Enhanced Federated Training")
        
        training_start_time = time.time()
        
        for round_num in range(1, self.num_rounds + 1):
            round_start_time = time.time()
            
            logger.info(f"🔄 Round {round_num}/{self.num_rounds}")
            
            # Quantum client selection
            selected_clients = await self._quantum_client_selection(round_num)
            
            # Create federated training tasks
            tasks = await self._create_quantum_tasks(selected_clients, round_num)
            
            # Schedule tasks using quantum optimization
            task_assignments = await self.quantum_scheduler.schedule_round()
            
            # Execute training round with quantum privacy
            round_results = await self._execute_quantum_round(
                selected_clients, task_assignments, round_num
            )
            
            # Record quantum metrics
            await self._record_round_metrics(round_num, round_results, time.time() - round_start_time)
            
            # Evaluate scaling needs
            scaling_decisions = await self.auto_scaler.evaluate_scaling_needs()
            for decision in scaling_decisions:
                if decision.confidence > 0.7:
                    await self.auto_scaler.execute_scaling_decision(decision)
            
            logger.info(f"✅ Round {round_num} completed in {time.time() - round_start_time:.2f}s")
            
        total_time = time.time() - training_start_time
        logger.info(f"🏁 Quantum Federated Training completed in {total_time:.2f}s")
        
        # Generate final report
        await self._generate_quantum_report(total_time)
        
    async def _quantum_client_selection(self, round_num: int) -> List[str]:
        """Select clients using quantum optimization"""
        logger.info("🎲 Performing quantum client selection...")
        
        # Get available clients from scheduler
        quantum_state_metrics = await self.quantum_scheduler.get_quantum_state_metrics()
        available_clients = [f"quantum_client_{i:03d}" for i in range(self.num_clients)]
        
        # Use quantum optimizer for selection
        from dp_federated_lora.quantum_optimizer import get_quantum_optimizer
        optimizer = get_quantum_optimizer(self.config, self.quantum_metrics)
        
        # Prepare client selection criteria
        selection_criteria = {
            'availability': 0.3,
            'computational_power': 0.25,
            'reliability_score': 0.2,
            'privacy_budget': 0.15,
            'quantum_coherence': 0.1
        }
        
        # Simulate client data (in real scenario, this would come from actual clients)
        available_client_data = []
        for client_id in available_clients:
            client_data = {
                'client_id': client_id,
                'availability': 0.7 + 0.3 * hash(client_id + str(round_num)) % 100 / 100,
                'computational_power': 0.5 + 0.5 * hash(client_id) % 100 / 100,
                'reliability_score': 0.8 + 0.2 * hash(client_id[::-1]) % 100 / 100,
                'privacy_budget': 1.0,
                'quantum_coherence': 0.6 + 0.4 * hash(str(round_num) + client_id) % 100 / 100
            }
            available_client_data.append(client_data)
            
        # Select clients using quantum optimization
        target_clients = min(8, self.num_clients)  # Select up to 8 clients per round
        selected_client_ids = await optimizer.optimize_client_selection(
            available_client_data, target_clients, selection_criteria
        )
        
        logger.info(f"🎯 Selected {len(selected_client_ids)} clients using quantum optimization")
        return selected_client_ids
        
    async def _create_quantum_tasks(self, selected_clients: List[str], round_num: int) -> List[str]:
        """Create quantum training tasks"""
        tasks = []
        
        for i, client_id in enumerate(selected_clients):
            task_id = f"federated_training_r{round_num:02d}_c{i:02d}"
            
            # Submit task to quantum scheduler
            await self.quantum_scheduler.submit_task(
                task_id=task_id,
                client_preference=client_id,
                priority=1.0,
                complexity=0.8,  # Training is complex
                resource_requirements={
                    'computational_power': 0.7,
                    'memory': 0.6,
                    'privacy_budget': 0.1  # Consume some privacy budget
                },
                quantum_properties={
                    'amplitude_real': 1.0,
                    'amplitude_imag': 0.0,
                    'phase': 0.0
                }
            )
            
            tasks.append(task_id)
            
        logger.info(f"📝 Created {len(tasks)} quantum training tasks")
        return tasks
        
    async def _execute_quantum_round(
        self, 
        selected_clients: List[str], 
        task_assignments: Dict[str, str], 
        round_num: int
    ) -> Dict[str, any]:
        """Execute training round with quantum enhancements"""
        logger.info("⚡ Executing quantum training round...")
        
        # Record quantum metrics
        self.quantum_metrics.record_quantum_metric(
            QuantumMetricType.QUANTUM_EFFICIENCY,
            len(selected_clients) / self.num_clients,
            round_number=round_num
        )
        
        # Simulate training execution (in real scenario, this would coordinate actual training)
        await asyncio.sleep(1)  # Simulate training time
        
        # Calculate quantum fidelity (measure of training quality)
        quantum_fidelity = 0.85 + 0.15 * (round_num / self.num_rounds)  # Improving over rounds
        
        self.quantum_metrics.record_quantum_metric(
            QuantumMetricType.QUANTUM_FIDELITY,
            quantum_fidelity,
            round_number=round_num
        )
        
        # Simulate round results
        round_results = {
            'selected_clients': selected_clients,
            'task_assignments': task_assignments,
            'quantum_fidelity': quantum_fidelity,
            'privacy_spent': round_num * 0.1,  # Incremental privacy consumption
            'convergence_rate': 0.95 ** round_num  # Decreasing convergence rate
        }
        
        return round_results
        
    async def _record_round_metrics(self, round_num: int, results: Dict, execution_time: float):
        """Record comprehensive round metrics"""
        # Record quantum metrics
        self.quantum_metrics.record_quantum_metric(
            QuantumMetricType.COHERENCE_TIME,
            10.0 - round_num * 0.2,  # Decreasing coherence over time
            round_number=round_num
        )
        
        self.quantum_metrics.record_quantum_metric(
            QuantumMetricType.QUANTUM_ERROR_RATE,
            0.05 + round_num * 0.01,  # Increasing error rate
            round_number=round_num
        )
        
        # Record performance
        self.quantum_metrics.record_quantum_performance(
            f"federated_round_{round_num}",
            execution_time,
            success=True,
            convergence_iterations=round_num * 2
        )
        
        logger.info(f"📊 Recorded metrics for round {round_num}")
        
    async def _generate_quantum_report(self, total_time: float):
        """Generate comprehensive quantum training report"""
        logger.info("📈 Generating Quantum Training Report")
        
        # Get quantum metrics summary
        quantum_summary = self.quantum_metrics.get_quantum_state_summary()
        
        # Get scaling status
        scaling_status = self.auto_scaler.get_scaling_status()
        
        # Get resilience status
        resilience_status = self.resilience_manager.get_resilience_status()
        
        # Generate report
        report = {
            'training_summary': {
                'model_name': self.model_name,
                'num_clients': self.num_clients,
                'num_rounds': self.num_rounds,
                'total_time': total_time,
                'privacy_budget': self.privacy_budget
            },
            'quantum_metrics': quantum_summary,
            'scaling_summary': scaling_status,
            'resilience_summary': resilience_status,
            'performance_summary': self.quantum_metrics.performance_tracker.get_all_performance_summaries()
        }
        
        # Log key findings
        logger.info("🎯 Quantum Training Report Summary:")
        logger.info(f"   • Total training time: {total_time:.2f}s")
        logger.info(f"   • Average quantum fidelity: {quantum_summary.get('quantum_fidelity', {}).get('mean', 'N/A')}")
        logger.info(f"   • Quantum coherence maintained: {quantum_summary.get('coherence_time', {}).get('mean', 'N/A')}")
        logger.info(f"   • Privacy budget utilized: {self.privacy_budget['epsilon']} ε")
        
        # Export detailed report
        detailed_report = self.quantum_metrics.export_quantum_metrics("json")
        
        # In a real scenario, you would save this to files
        logger.info("📄 Detailed quantum metrics report generated")
        
        return report
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up quantum resources...")
        
        if self.quantum_scheduler:
            await self.quantum_scheduler.cleanup()
            
        if self.quantum_metrics:
            self.quantum_metrics.cleanup()
            
        logger.info("✅ Cleanup completed")


async def main():
    """Main execution function"""
    logger.info("🌟 Quantum-Inspired Federated Learning Demo")
    
    # Create quantum federated training instance
    quantum_training = QuantumFederatedTraining(
        model_name="gpt2",
        num_clients=12,
        num_rounds=10,
        privacy_budget={"epsilon": 8.0, "delta": 1e-5}
    )
    
    try:
        # Initialize system
        await quantum_training.initialize()
        
        # Run quantum-enhanced federated training
        await quantum_training.run_quantum_federated_training()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Cleanup
        await quantum_training.cleanup()
        
    logger.info("🎉 Quantum-Enhanced Federated Learning Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
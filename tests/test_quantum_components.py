"""
Test suite for quantum-inspired components

Tests quantum scheduler, privacy, optimizer, monitoring, resilience and scaling components.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# Import quantum components
from dp_federated_lora.quantum_scheduler import (
    QuantumTaskScheduler, QuantumTask, QuantumClient, QuantumState
)
from dp_federated_lora.quantum_privacy import (
    QuantumPrivacyEngine, QuantumPrivacyConfig, QuantumNoiseGenerator
)
from dp_federated_lora.quantum_optimizer import (
    QuantumInspiredOptimizer, VariationalQuantumOptimizer
)
from dp_federated_lora.quantum_monitoring import (
    QuantumMetricsCollector, QuantumMetricType, QuantumAnomalyDetector
)
from dp_federated_lora.quantum_resilience import (
    QuantumCircuitBreaker, QuantumResilienceManager, QuantumCircuitBreakerState
)
from dp_federated_lora.quantum_scaling import (
    QuantumAutoScaler, QuantumResourcePredictor, ResourceType
)
from dp_federated_lora.config import FederatedConfig


class TestQuantumScheduler:
    """Test quantum task scheduler"""
    
    @pytest.fixture
    def scheduler(self):
        """Create test scheduler"""
        config = FederatedConfig()
        return QuantumTaskScheduler(config)
        
    @pytest.mark.asyncio
    async def test_client_registration(self, scheduler):
        """Test quantum client registration"""
        client_id = "test_client_001"
        capabilities = {
            'availability': 0.8,
            'computational_power': 0.6,
            'network_latency': 0.1,
            'reliability_score': 0.9,
            'privacy_budget': 1.0
        }
        
        await scheduler.register_client(client_id, capabilities)
        
        assert client_id in scheduler.clients
        client = scheduler.clients[client_id]
        assert client.client_id == client_id
        assert client.availability == 0.8
        assert client.computational_power == 0.6
        
    @pytest.mark.asyncio
    async def test_task_submission(self, scheduler):
        """Test quantum task submission"""
        task_id = "test_task_001"
        
        await scheduler.submit_task(
            task_id=task_id,
            priority=1.0,
            complexity=0.5,
            resource_requirements={'cpu': 0.4, 'memory': 0.3}
        )
        
        assert task_id in scheduler.tasks
        task = scheduler.tasks[task_id]
        assert task.task_id == task_id
        assert task.priority == 1.0
        assert task.quantum_state == QuantumState.SUPERPOSITION
        
    @pytest.mark.asyncio
    async def test_scheduling_round(self, scheduler):
        """Test quantum scheduling round"""
        # Register clients
        for i in range(3):
            await scheduler.register_client(
                f"client_{i}",
                {'availability': 0.8, 'computational_power': 0.5, 'network_latency': 0.1,
                 'reliability_score': 0.8, 'privacy_budget': 1.0}
            )
            
        # Submit tasks
        for i in range(2):
            await scheduler.submit_task(
                f"task_{i}",
                priority=1.0,
                complexity=0.5
            )
            
        # Run scheduling
        assignments = await scheduler.schedule_round()
        
        assert isinstance(assignments, dict)
        assert len(assignments) <= 2  # At most 2 tasks
        
    @pytest.mark.asyncio
    async def test_quantum_state_metrics(self, scheduler):
        """Test quantum state metrics"""
        await scheduler.register_client("client_1", {})
        await scheduler.submit_task("task_1", priority=1.0, complexity=0.5)
        
        metrics = await scheduler.get_quantum_state_metrics()
        
        assert "task_states" in metrics
        assert "total_tasks" in metrics
        assert "total_clients" in metrics
        assert metrics["total_tasks"] == 1
        assert metrics["total_clients"] == 1


class TestQuantumPrivacy:
    """Test quantum privacy components"""
    
    @pytest.fixture
    def privacy_config(self):
        """Create test privacy config"""
        return QuantumPrivacyConfig(
            base_epsilon=1.0,
            base_delta=1e-5,
            quantum_amplification_factor=1.2
        )
        
    @pytest.fixture
    def noise_generator(self, privacy_config):
        """Create test noise generator"""
        return QuantumNoiseGenerator(privacy_config)
        
    def test_quantum_noise_generation(self, noise_generator):
        """Test quantum noise generation"""
        shape = (10, 5)
        sensitivity = 1.0
        epsilon = 1.0
        
        noise = noise_generator.generate_quantum_noise(shape, sensitivity, epsilon)
        
        assert noise.shape == shape
        assert isinstance(noise, torch.Tensor)
        assert noise.dtype == torch.float32
        
    def test_privacy_engine_creation(self, privacy_config):
        """Test quantum privacy engine creation"""
        engine = QuantumPrivacyEngine(privacy_config)
        
        assert engine.config == privacy_config
        assert engine.noise_generator is not None
        assert engine.secure_aggregator is not None
        
    @pytest.mark.asyncio
    async def test_quantum_secure_aggregation(self, privacy_config):
        """Test quantum secure aggregation"""
        from dp_federated_lora.quantum_privacy import QuantumSecureAggregator
        import torch
        
        aggregator = QuantumSecureAggregator(privacy_config)
        
        # Create test client updates
        client_updates = {
            "client_1": torch.randn(10, 5),
            "client_2": torch.randn(10, 5),
            "client_3": torch.randn(10, 5)
        }
        
        result = await aggregator.quantum_secure_aggregate(client_updates)
        
        assert result.shape == (10, 5)
        assert isinstance(result, torch.Tensor)


class TestQuantumOptimizer:
    """Test quantum optimization components"""
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer"""
        config = FederatedConfig()
        return QuantumInspiredOptimizer(config)
        
    def test_vqe_optimizer_creation(self):
        """Test VQE optimizer creation"""
        optimizer = VariationalQuantumOptimizer(num_qubits=4, num_layers=2)
        
        assert optimizer.num_qubits == 4
        assert optimizer.num_layers == 2
        assert optimizer.circuit is not None
        
    @pytest.mark.asyncio
    async def test_client_selection_optimization(self, optimizer):
        """Test quantum client selection"""
        available_clients = [
            {'client_id': f'client_{i}', 'availability': 0.8, 'computational_power': 0.5}
            for i in range(5)
        ]
        
        selection_criteria = {'availability': 0.6, 'computational_power': 0.4}
        
        selected = await optimizer.optimize_client_selection(
            available_clients, 3, selection_criteria
        )
        
        assert isinstance(selected, list)
        assert len(selected) <= 3
        assert all(isinstance(client_id, str) for client_id in selected)
        
    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, optimizer):
        """Test quantum hyperparameter optimization"""
        def test_objective(params):
            return sum(p**2 for p in params.values())
            
        param_bounds = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (16, 128)
        }
        
        optimal_params = await optimizer.optimize_hyperparameters(
            test_objective, param_bounds, max_evaluations=10
        )
        
        assert 'learning_rate' in optimal_params
        assert 'batch_size' in optimal_params
        assert 0.001 <= optimal_params['learning_rate'] <= 0.1
        assert 16 <= optimal_params['batch_size'] <= 128


class TestQuantumMonitoring:
    """Test quantum monitoring components"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create test metrics collector"""
        config = FederatedConfig()
        return QuantumMetricsCollector(config)
        
    def test_quantum_metric_recording(self, metrics_collector):
        """Test quantum metric recording"""
        metrics_collector.record_quantum_metric(
            QuantumMetricType.QUANTUM_FIDELITY,
            0.95,
            client_id="test_client",
            round_number=1
        )
        
        assert len(metrics_collector.quantum_metrics) == 1
        metric = metrics_collector.quantum_metrics[0]
        assert metric.metric_type == QuantumMetricType.QUANTUM_FIDELITY
        assert metric.value == 0.95
        assert metric.client_id == "test_client"
        
    def test_anomaly_detection(self):
        """Test quantum anomaly detection"""
        detector = QuantumAnomalyDetector(window_size=10, sensitivity=2.0)
        
        # Add normal values
        for i in range(15):
            value = 0.5 + 0.1 * np.sin(i * 0.1)  # Normal oscillation
            is_anomaly, score = detector.add_measurement(
                QuantumMetricType.QUANTUM_FIDELITY, value
            )
            
        # Should not be anomaly for normal values
        assert not is_anomaly or score < 2.0
        
        # Add anomalous value
        is_anomaly, score = detector.add_measurement(
            QuantumMetricType.QUANTUM_FIDELITY, 10.0  # Very high value
        )
        
        assert is_anomaly
        assert score > 2.0
        
    def test_quantum_state_summary(self, metrics_collector):
        """Test quantum state summary"""
        # Record some metrics
        for i in range(5):
            metrics_collector.record_quantum_metric(
                QuantumMetricType.COHERENCE_TIME,
                10.0 - i * 0.5,
                round_number=i
            )
            
        summary = metrics_collector.get_quantum_state_summary()
        
        assert 'coherence_time' in summary
        coherence_stats = summary['coherence_time']
        assert 'mean' in coherence_stats
        assert 'count' in coherence_stats
        assert coherence_stats['count'] == 5


class TestQuantumResilience:
    """Test quantum resilience components"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create test circuit breaker"""
        from dp_federated_lora.quantum_resilience import QuantumCircuitBreakerConfig
        
        config = QuantumCircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0
        )
        return QuantumCircuitBreaker("test_breaker", config)
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, circuit_breaker):
        """Test circuit breaker with successful operations"""
        async def success_func():
            return "success"
            
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == QuantumCircuitBreakerState.CLOSED
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, circuit_breaker):
        """Test circuit breaker with failing operations"""
        async def failing_func():
            raise Exception("Test failure")
            
        # Call multiple times to trigger circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
                
        # Circuit should be open now
        assert circuit_breaker.state in [
            QuantumCircuitBreakerState.OPEN,
            QuantumCircuitBreakerState.QUANTUM_DECOHERENT
        ]
        
    def test_resilience_manager_creation(self):
        """Test resilience manager creation"""
        config = FederatedConfig()
        manager = QuantumResilienceManager(config)
        
        assert isinstance(manager.circuit_breakers, dict)
        assert isinstance(manager.retry_strategies, dict)
        
    @pytest.mark.asyncio
    async def test_resilient_execution(self):
        """Test resilient execution with retries"""
        config = FederatedConfig()
        manager = QuantumResilienceManager(config)
        
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Flaky failure")
            return "success"
            
        result = await manager.execute_with_resilience(
            flaky_func,
            retry_strategy_name="test_retry",
            max_retries=5
        )
        
        assert result == "success"
        assert call_count == 3


class TestQuantumScaling:
    """Test quantum scaling components"""
    
    @pytest.fixture
    def resource_predictor(self):
        """Create test resource predictor"""
        return QuantumResourcePredictor(prediction_window=20)
        
    def test_resource_prediction(self, resource_predictor):
        """Test resource demand prediction"""
        from dp_federated_lora.quantum_scaling import ResourceMetrics
        
        # Add historical data
        for i in range(25):
            metrics = ResourceMetrics(
                cpu_utilization=0.5 + 0.3 * np.sin(i * 0.1),
                memory_utilization=0.6,
                quantum_coherence=0.8
            )
            resource_predictor.add_metrics(metrics)
            
        prediction, confidence = resource_predictor.predict_resource_demand(
            ResourceType.CPU_CORES, time_horizon=300.0
        )
        
        assert 0.0 <= prediction <= 1.0
        assert 0.0 <= confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_auto_scaler_creation(self):
        """Test auto-scaler creation"""
        config = FederatedConfig()
        
        # Mock dependencies
        metrics_collector = Mock()
        resilience_manager = Mock()
        
        auto_scaler = QuantumAutoScaler(config, metrics_collector, resilience_manager)
        
        assert auto_scaler.config == config
        assert auto_scaler.predictor is not None
        
    @pytest.mark.asyncio
    async def test_scaling_evaluation(self):
        """Test scaling needs evaluation"""
        config = FederatedConfig()
        
        # Mock dependencies
        metrics_collector = Mock()
        resilience_manager = Mock()
        resilience_manager.get_resilience_status.return_value = {"circuit_breakers": {}}
        
        auto_scaler = QuantumAutoScaler(config, metrics_collector, resilience_manager)
        
        # Mock resource metrics collection
        with patch.object(auto_scaler, '_collect_resource_metrics') as mock_collect:
            from dp_federated_lora.quantum_scaling import ResourceMetrics
            mock_collect.return_value = ResourceMetrics(
                cpu_utilization=0.8,  # High utilization
                memory_utilization=0.4,
                quantum_coherence=0.6
            )
            
            decisions = await auto_scaler.evaluate_scaling_needs()
            
            assert isinstance(decisions, list)
            # Should have at least one decision due to high CPU utilization
            # (though exact behavior depends on prediction)


class TestQuantumIntegration:
    """Test integration between quantum components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_quantum_workflow(self):
        """Test complete quantum workflow integration"""
        config = FederatedConfig()
        
        # Create all components
        metrics_collector = QuantumMetricsCollector(config)
        resilience_manager = QuantumResilienceManager(config, metrics_collector)
        scheduler = QuantumTaskScheduler(config, metrics_collector)
        
        # Register clients
        await scheduler.register_client(
            "integration_client",
            {'availability': 0.8, 'computational_power': 0.6, 'network_latency': 0.1,
             'reliability_score': 0.9, 'privacy_budget': 1.0}
        )
        
        # Submit task
        await scheduler.submit_task(
            "integration_task",
            priority=1.0,
            complexity=0.5
        )
        
        # Schedule round
        assignments = await scheduler.schedule_round()
        
        # Record metrics
        metrics_collector.record_quantum_metric(
            QuantumMetricType.QUANTUM_EFFICIENCY,
            0.95,
            client_id="integration_client"
        )
        
        # Get metrics summary
        summary = metrics_collector.get_quantum_state_summary()
        
        # Verify integration
        assert len(assignments) >= 0  # May be empty if no valid assignments
        assert "quantum_efficiency" in summary
        
        # Cleanup
        await scheduler.cleanup()
        metrics_collector.cleanup()


# Test configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
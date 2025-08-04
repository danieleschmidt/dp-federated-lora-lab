#!/usr/bin/env python3
"""
High-performance federated training demonstration.

This example showcases the advanced performance optimization,
concurrent processing, and scaling capabilities of the 
DP-Federated LoRA system.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dp_federated_lora import (
    FederatedServer,
    DPLoRAClient,
    create_default_config,
    optimize_for_scale,
    get_performance_report,
    performance_monitor,
    resource_manager,
    concurrent_trainer,
    parallel_aggregator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('high_performance_training.log')
    ]
)
logger = logging.getLogger(__name__)


class HighPerformanceFederatedDemo:
    """High-performance federated learning demonstration."""
    
    def __init__(
        self,
        num_clients: int = 20,
        num_rounds: int = 10,
        model_name: str = "gpt2",
        server_port: int = 8080
    ):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.model_name = model_name
        self.server_port = server_port
        self.server_url = f"http://127.0.0.1:{server_port}"
        
        # Performance tracking
        self.training_start_time = 0.0
        self.clients: List[DPLoRAClient] = []
        self.server: Optional[FederatedServer] = None
        
        logger.info(f"Initialized high-performance demo: {num_clients} clients, {num_rounds} rounds")
    
    def create_optimized_config(self) -> Any:
        """Create optimized configuration for high-performance training."""
        config = create_default_config(self.model_name)
        
        # Optimize for performance
        config.num_rounds = self.num_rounds
        config.local_epochs = 3  # Faster rounds
        config.privacy.epsilon = 10.0  # Relaxed privacy for performance
        config.privacy.delta = 1e-5
        
        # Security and aggregation optimizations
        config.security.min_clients = max(2, self.num_clients // 4)
        config.security.max_clients = self.num_clients
        config.security.client_sampling_rate = 0.8  # Select 80% of clients per round
        config.security.aggregation_method = "weighted_average"  # Fast aggregation
        
        # Performance tuning
        config.batch_size = 32
        config.learning_rate = 0.001
        
        # LoRA optimization for speed
        config.lora.r = 8  # Smaller rank for faster training
        config.lora.lora_alpha = 16
        config.lora.lora_dropout = 0.05
        config.lora.target_modules = ["q_proj", "v_proj"]  # Focus on key modules
        
        return config
    
    async def setup_high_performance_server(self) -> FederatedServer:
        """Setup server optimized for high performance."""
        config = self.create_optimized_config()
        
        # Optimize system for scale
        optimize_for_scale(
            cache_size=self.num_clients * 20,  # Large cache for many clients
            max_connections=self.num_clients * 3,  # Extra connections for reliability
            enable_profiling=True  # Monitor performance
        )
        
        server = FederatedServer(
            model_name=self.model_name,
            config=config,
            host="127.0.0.1",
            port=self.server_port,
            num_clients=self.num_clients,
            rounds=self.num_rounds
        )
        
        logger.info(f"High-performance server configured with optimizations")
        return server
    
    async def create_concurrent_clients(self) -> List[DPLoRAClient]:
        """Create multiple clients concurrently for performance testing."""
        config = self.create_optimized_config()
        clients = []
        
        # Create clients with concurrent setup
        async def create_client(client_id: str) -> DPLoRAClient:
            """Create and setup individual client."""
            client = DPLoRAClient(
                client_id=client_id,
                data_path=f"/tmp/mock_data_{client_id}.json",
                config=config,
                server_url=self.server_url
            )
            
            # Mock dataset for testing
            client.local_dataset = type('MockDataset', (), {
                '__len__': lambda self: 1000 + hash(client_id) % 500  # Varied dataset sizes
            })()
            
            # Setup client (would load real data in production)
            client.setup()
            
            return client
        
        # Create all clients concurrently
        tasks = []
        for i in range(self.num_clients):
            client_id = f"high_perf_client_{i:03d}"
            task = create_client(client_id)
            tasks.append(task)
        
        logger.info(f"Creating {self.num_clients} clients concurrently...")
        clients = await asyncio.gather(*tasks)
        
        logger.info(f"Successfully created {len(clients)} clients")
        return clients
    
    async def run_concurrent_registration(self, clients: List[DPLoRAClient]) -> int:
        """Register all clients concurrently."""
        logger.info("Starting concurrent client registration...")
        start_time = time.time()
        
        # Register all clients concurrently
        registration_tasks = []
        for client in clients:
            task = client.register_with_server()
            registration_tasks.append(task)
        
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        successful_registrations = sum(
            1 for result in results 
            if not isinstance(result, Exception) and result
        )
        
        registration_time = time.time() - start_time
        logger.info(f"Concurrent registration completed: {successful_registrations}/{len(clients)} "
                   f"successful in {registration_time:.2f}s")
        
        return successful_registrations
    
    async def run_high_performance_training(self) -> Dict[str, Any]:
        """Run high-performance federated training with full optimization."""
        logger.info("🚀 Starting High-Performance Federated Training")
        logger.info("=" * 80)
        
        self.training_start_time = time.time()
        
        # Phase 1: Server Setup
        logger.info("Phase 1: Setting up high-performance server...")
        self.server = await self.setup_high_performance_server()
        
        # Start server in background
        server_task = asyncio.create_task(self.server.train_federated())
        await asyncio.sleep(2)  # Allow server to start
        
        # Phase 2: Concurrent Client Creation
        logger.info("Phase 2: Creating clients concurrently...")
        self.clients = await self.create_concurrent_clients()
        
        # Phase 3: Concurrent Registration
        logger.info("Phase 3: Concurrent client registration...")
        successful_registrations = await self.run_concurrent_registration(self.clients)
        
        if successful_registrations < self.server.config.security.min_clients:
            raise Exception(f"Insufficient client registrations: {successful_registrations}")
        
        # Phase 4: Monitor Training Progress
        logger.info("Phase 4: Monitoring high-performance training...")
        await self.monitor_training_progress()
        
        # Phase 5: Wait for Training Completion
        logger.info("Phase 5: Waiting for training completion...")
        try:
            history = await asyncio.wait_for(server_task, timeout=300)  # 5 minute timeout
            
            training_time = time.time() - self.training_start_time
            
            # Generate comprehensive performance report
            performance_report = await self.generate_performance_report(history, training_time)
            
            return performance_report
            
        except asyncio.TimeoutError:
            logger.error("Training timed out after 5 minutes")
            server_task.cancel()
            raise
    
    async def monitor_training_progress(self) -> None:
        """Monitor training progress with real-time metrics."""
        logger.info("Starting real-time training monitoring...")
        
        monitoring_interval = 5  # seconds
        start_time = time.time()
        
        while True:
            try:
                # Get server status via HTTP API
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.server_url}/stats", timeout=5.0)
                    if response.status_code == 200:
                        stats = response.json()
                        
                        # Log progress
                        server_status = stats.get("server_status", {})
                        resource_status = stats.get("resource_status", {})
                        
                        logger.info(
                            f"📊 Round {server_status.get('current_round', 0)}/{server_status.get('total_rounds', 0)} | "
                            f"Clients: {server_status.get('active_clients', 0)} | "
                            f"Memory: {resource_status.get('memory_mb', 0):.1f}MB | "
                            f"CPU: {resource_status.get('cpu_percent', 0):.1f}%"
                        )
                        
                        # Check if training is complete
                        if not server_status.get('is_training', True):
                            logger.info("Training completed, stopping monitoring")
                            break
                    
                await asyncio.sleep(monitoring_interval)
                
                # Safety timeout
                if time.time() - start_time > 600:  # 10 minutes
                    logger.warning("Monitoring timeout reached")
                    break
                    
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                await asyncio.sleep(monitoring_interval)
    
    async def generate_performance_report(
        self, 
        history: Any, 
        total_training_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        # Get system performance report
        perf_report = get_performance_report()
        
        # Calculate training metrics
        training_metrics = {
            "total_training_time": total_training_time,
            "rounds_completed": history.rounds if history else 0,
            "final_accuracy": history.final_accuracy if history else 0.0,
            "privacy_spent": history.total_epsilon if history else 0.0,
            "participating_clients": len(self.clients),
            "successful_clients": history.participating_clients if history else 0,
            "avg_time_per_round": total_training_time / max(1, history.rounds) if history else 0,
            "clients_per_second": len(self.clients) / total_training_time,
            "throughput_score": (len(self.clients) * (history.rounds if history else 0)) / total_training_time
        }
        
        # Resource utilization
        resource_stats = resource_manager.get_stats()
        
        # Performance insights
        performance_insights = {
            "scalability_score": min(100, (len(self.clients) / 50) * 100),  # Score out of 100
            "efficiency_score": min(100, (training_metrics["throughput_score"] / 10) * 100),
            "resource_efficiency": 100 - min(100, resource_stats["current_resource_status"]["memory_percent"]),
            "recommendations": self._generate_performance_recommendations(training_metrics, resource_stats)
        }
        
        comprehensive_report = {
            "training_metrics": training_metrics,
            "system_performance": perf_report,
            "resource_utilization": resource_stats,
            "performance_insights": performance_insights,
            "timestamp": time.time(),
            "configuration": {
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "model_name": self.model_name
            }
        }
        
        return comprehensive_report
    
    def _generate_performance_recommendations(
        self, 
        training_metrics: Dict[str, Any], 
        resource_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Throughput recommendations
        if training_metrics["throughput_score"] < 1.0:
            recommendations.append("Consider increasing batch size or reducing model complexity for better throughput")
        
        # Memory recommendations
        memory_percent = resource_stats["current_resource_status"]["memory_percent"]
        if memory_percent > 80:
            recommendations.append("High memory usage detected - consider reducing cache size or client count")
        elif memory_percent < 30:
            recommendations.append("Low memory usage - could handle more clients or larger cache")
        
        # CPU recommendations
        cpu_percent = resource_stats["current_resource_status"]["cpu_percent"]
        if cpu_percent > 90:
            recommendations.append("High CPU usage - consider using process-based parallelism")
        elif cpu_percent < 20:
            recommendations.append("Low CPU usage - could increase concurrent processing")
        
        # Scalability recommendations
        if self.num_clients < 10:
            recommendations.append("Try scaling to more clients to test federation benefits")
        elif self.num_clients > 100:
            recommendations.append("Large-scale deployment - consider distributed server architecture")
        
        return recommendations
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        # Close client connections
        cleanup_tasks = []
        for client in self.clients:
            if client.network_client:
                cleanup_tasks.append(client.close_network_connection())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Stop server
        if self.server:
            try:
                await self.server.stop_server()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
        
        logger.info("Cleanup completed")


async def main():
    """Run high-performance federated learning demonstration."""
    print("🚀 DP-Federated LoRA High-Performance Training Demo")
    print("=" * 80)
    print("This demo showcases advanced performance optimization and scaling capabilities")
    print()
    
    # Configuration for high-performance demo
    demo_configs = [
        {"num_clients": 10, "num_rounds": 5, "name": "Small Scale"},
        {"num_clients": 20, "num_rounds": 5, "name": "Medium Scale"},
        {"num_clients": 50, "num_rounds": 3, "name": "Large Scale"}
    ]
    
    selected_config = demo_configs[1]  # Medium scale by default
    
    demo = HighPerformanceFederatedDemo(
        num_clients=selected_config["num_clients"],
        num_rounds=selected_config["num_rounds"],
        model_name="gpt2",  # Use small model for demo
        server_port=8080
    )
    
    try:
        # Run high-performance training
        performance_report = await demo.run_high_performance_training()
        
        # Display comprehensive results
        print("\n" + "=" * 80)
        print("🎉 HIGH-PERFORMANCE TRAINING COMPLETED")
        print("=" * 80)
        
        training_metrics = performance_report["training_metrics"]
        performance_insights = performance_report["performance_insights"]
        
        print(f"📊 Training Results:")
        print(f"  • Total Time: {training_metrics['total_training_time']:.2f} seconds")
        print(f"  • Rounds Completed: {training_metrics['rounds_completed']}")
        print(f"  • Participating Clients: {training_metrics['participating_clients']}")
        print(f"  • Final Accuracy: {training_metrics['final_accuracy']:.3f}")
        print(f"  • Privacy Spent: ε={training_metrics['privacy_spent']:.2f}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"  • Throughput Score: {training_metrics['throughput_score']:.2f}")
        print(f"  • Clients/Second: {training_metrics['clients_per_second']:.2f}")
        print(f"  • Avg Time/Round: {training_metrics['avg_time_per_round']:.2f}s")
        
        print(f"\n📈 Performance Insights:")
        print(f"  • Scalability Score: {performance_insights['scalability_score']:.1f}/100")
        print(f"  • Efficiency Score: {performance_insights['efficiency_score']:.1f}/100")
        print(f"  • Resource Efficiency: {performance_insights['resource_efficiency']:.1f}/100")
        
        if performance_insights["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in performance_insights["recommendations"]:
                print(f"  • {rec}")
        
        print(f"\n📁 Detailed report saved to performance_report.json")
        
        # Save detailed report
        import json
        with open("performance_report.json", "w") as f:
            json.dump(performance_report, f, indent=2, default=str)
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()
    
    print("\n✅ High-performance demo completed!")


if __name__ == "__main__":
    # Run the high-performance demo
    asyncio.run(main())
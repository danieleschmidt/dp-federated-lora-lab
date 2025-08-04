#!/usr/bin/env python3
"""
Network-based federated training example.

This example demonstrates the DP-Federated LoRA system using real HTTP
communication between server and clients.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dp_federated_lora import (
    FederatedServer,
    DPLoRAClient,
    FederatedConfig,
    create_default_config
)
from dp_federated_lora.network_client import FederatedNetworkClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_mock_client(client_id: str, server_url: str, num_examples: int = 1000):
    """
    Create a mock client for demonstration.
    
    Args:
        client_id: Unique client identifier
        server_url: Server URL
        num_examples: Number of training examples to simulate
        
    Returns:
        Configured client
    """
    # Create mock data path (in real scenario, this would be actual data)
    data_path = f"/tmp/mock_data_{client_id}.json"
    
    # Create client with network communication
    client = DPLoRAClient(
        client_id=client_id,
        data_path=data_path,
        config=create_default_config("gpt2"),  # Use small model for demo
        server_url=server_url
    )
    
    # Mock the dataset size for capabilities
    client.local_dataset = type('MockDataset', (), {'__len__': lambda self: num_examples})()
    
    return client


async def run_client_simulation(client: DPLoRAClient, num_rounds: int = 5):
    """
    Simulate a client participating in federated training.
    
    Args:
        client: Configured client
        num_rounds: Number of rounds to participate in
    """
    try:
        logger.info(f"Client {client.client_id} starting simulation...")
        
        # Setup client (this would load actual data in real scenario)
        client.setup()
        
        # Register with server
        success = await client.register_with_server()
        if not success:
            logger.error(f"Client {client.client_id} failed to register")
            return
        
        logger.info(f"Client {client.client_id} registered successfully")
        
        # Participate in training rounds
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Client {client.client_id} participating in round {round_num}")
            
            # Check server status
            if client.network_client:
                status = await client.network_client.get_server_status()
                if status["current_round"] != round_num:
                    logger.info(f"Client {client.client_id} waiting for round {round_num}")
                    await asyncio.sleep(2)
                    continue
            
            # Participate in the round
            success = await client.participate_in_round(round_num)
            if success:
                logger.info(f"Client {client.client_id} completed round {round_num}")
            else:
                logger.error(f"Client {client.client_id} failed round {round_num}")
                break
            
            # Small delay between rounds
            await asyncio.sleep(1)
        
        logger.info(f"Client {client.client_id} completed simulation")
        
    except Exception as e:
        logger.error(f"Client {client.client_id} simulation failed: {e}")
    finally:
        await client.close_network_connection()


async def main():
    """Main demonstration function."""
    print("🚀 DP-Federated LoRA Network Communication Demo")
    print("=" * 60)
    
    # Configuration
    server_host = "127.0.0.1"
    server_port = 8080
    server_url = f"http://{server_host}:{server_port}"
    num_clients = 3
    num_rounds = 3
    
    # Create server
    logger.info("Initializing federated server...")
    config = create_default_config("gpt2")  # Use small model for demo
    config.num_rounds = num_rounds
    config.security.min_clients = 2
    config.privacy.epsilon = 10.0  # Relaxed privacy for demo
    
    server = FederatedServer(
        model_name="gpt2",
        config=config,
        host=server_host,
        port=server_port
    )
    
    try:
        # Start server in background
        logger.info(f"Starting server on {server_url}...")
        server_task = asyncio.create_task(server.train_federated())
        
        # Wait a moment for server to start
        await asyncio.sleep(3)
        
        # Create and start clients
        logger.info(f"Creating {num_clients} clients...")
        clients = []
        client_tasks = []
        
        for i in range(num_clients):
            client_id = f"client_{i+1}"
            client = await create_mock_client(client_id, server_url, num_examples=500 + i * 100)
            clients.append(client)
            
            # Start client simulation
            task = asyncio.create_task(run_client_simulation(client, num_rounds))
            client_tasks.append(task)
        
        logger.info("All clients started, waiting for training to complete...")
        
        # Wait for all clients to complete
        await asyncio.gather(*client_tasks)
        
        # Wait for server to complete
        try:
            history = await asyncio.wait_for(server_task, timeout=60)
            
            print("\n🎉 Training Completed Successfully!")
            print(f"📊 Results:")
            print(f"  - Rounds completed: {history.rounds}")
            print(f"  - Training time: {history.training_time:.2f} seconds")
            print(f"  - Final accuracy: {history.final_accuracy:.3f}")
            print(f"  - Privacy spent: ε={history.total_epsilon:.2f}")
            print(f"  - Participating clients: {history.participating_clients}")
            
        except asyncio.TimeoutError:
            logger.warning("Server training timed out")
            server_task.cancel()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        try:
            await server.stop_server()
        except:
            pass
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
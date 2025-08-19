"""
Basic example of DP-Federated LoRA training.

This example demonstrates how to set up and run a federated learning
experiment with differential privacy and LoRA fine-tuning.
"""

# This example requires dependencies to be installed:
# pip install torch transformers datasets accelerate peft opacus wandb

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run a basic federated learning experiment."""
    
    try:
        # Import DP-Federated LoRA components
        import sys
        sys.path.append('/root/repo/src')
        
        from dp_federated_lora import (
            FederatedServer,
            DPLoRAClient, 
            create_default_config
        )
        
        print("🚀 Starting DP-Federated LoRA Training Example")
        
        # 1. Create configuration
        config = create_default_config()
        config.model_name = "meta-llama/Llama-2-7b-hf"  # or smaller model for testing
        config.num_rounds = 10  # Fewer rounds for demo
        config.privacy.epsilon = 8.0
        config.privacy.delta = 1e-5
        config.lora.r = 16
        
        print(f"📋 Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Rounds: {config.num_rounds}")
        print(f"   Privacy: ε={config.privacy.epsilon}, δ={config.privacy.delta}")
        print(f"   LoRA rank: {config.lora.r}")
        
        # 2. Initialize federated server
        print("\n🏗️  Initializing federated server...")
        server = FederatedServer(
            model_name=config.model_name,
            config=config,
            num_clients=5,
            rounds=config.num_rounds
        )
        
        # Initialize global model
        server.initialize_global_model()
        print("✅ Server initialized successfully")
        
        # 3. Create federated clients with mock data
        print("\n👥 Creating federated clients...")
        clients = []
        
        for i in range(5):
            # Create client with different data sizes for realism
            num_examples = 500 + i * 200
            client = create_mock_client(
                client_id=f"hospital_{i+1}",
                num_examples=num_examples,
                config=config
            )
            
            # Setup client (load data, initialize model, etc.)
            client.setup()
            clients.append(client)
            
            print(f"   ✅ Client {client.client_id}: {num_examples} examples")
        
        # 4. Run federated training
        print(f"\n🔄 Starting federated training ({config.num_rounds} rounds)...")
        
        history = server.train(
            clients=clients,
            aggregation="fedavg",
            client_sampling_rate=0.8,  # 80% of clients per round
            local_epochs=3
        )
        
        # 5. Display results
        print(f"\n🎉 Training completed!")
        print(f"📊 Results:")
        print(f"   Rounds completed: {history.rounds_completed}")
        print(f"   Training duration: {history.total_duration:.1f} seconds")
        print(f"   Final accuracy: {history.final_accuracy:.2%}")
        print(f"   Privacy spent: ε={history.final_epsilon:.2f} (budget: {history.total_epsilon:.2f})")
        print(f"   Budget utilization: {history.final_epsilon/history.total_epsilon:.1%}")
        
        # 6. Save results
        output_dir = Path("./federated_results")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "training_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
        
        # 7. Display privacy-utility analysis
        print(f"\n🔒 Privacy Analysis:")
        privacy_timeline = history.get_privacy_timeline()
        
        if privacy_timeline:
            final_round = privacy_timeline[-1]
            print(f"   Round {final_round['round']}: ε={final_round['epsilon_spent']:.2f}")
            print(f"   Remaining budget: ε={final_round['epsilon_remaining']:.2f}")
            print(f"   Utilization: {final_round['budget_utilization']:.1%}")
        
        print(f"\n✨ Example completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"💡 Please install required packages:")
        print(f"   pip install torch transformers datasets accelerate peft opacus")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
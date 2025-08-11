#!/usr/bin/env python3
"""
🚀 DP-Federated LoRA Lab Quick Start Script

This script provides a zero-configuration way to get started with 
differentially private federated learning using LoRA fine-tuning.

Usage:
    python quick_start.py --mode demo          # Demo with mock data  
    python quick_start.py --mode production    # Production setup
    python quick_start.py --mode benchmark     # Run benchmarks
"""

import argparse
import logging
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class QuickStartManager:
    """Manages quick start scenarios for dp-federated-lora-lab."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.config_dir = self.repo_root / ".quickstart"
        self.config_dir.mkdir(exist_ok=True)
        
    def demo_mode(self) -> None:
        """Run a demo with mock federated learning data."""
        logger.info("🎯 Starting demo mode with synthetic data")
        
        # Create demo configuration
        demo_config = {
            "experiment_name": "quickstart_demo",
            "num_clients": 5,
            "num_rounds": 3,
            "model_name": "distilbert-base-uncased",  # Small model for demo
            "privacy": {
                "epsilon": 8.0,
                "delta": 1e-5,
                "noise_multiplier": 1.1
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "training": {
                "local_epochs": 2,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "max_steps": 100
            }
        }
        
        # Save demo config
        config_file = self.config_dir / "demo_config.json"
        with open(config_file, 'w') as f:
            json.dump(demo_config, f, indent=2)
            
        logger.info(f"📋 Demo configuration saved to {config_file}")
        
        # Generate synthetic federated data
        self._generate_demo_data()
        
        # Show next steps
        logger.info("🎉 Demo setup complete!")
        logger.info("Next steps:")
        logger.info("  1. Run: python examples/basic_federated_training.py")
        logger.info("  2. View results in: ./federated_results/")
        logger.info("  3. Monitor with: python scripts/health-check.py")
        
    def production_mode(self) -> None:
        """Setup production environment."""
        logger.info("🏭 Setting up production environment")
        
        # Check system requirements
        self._check_production_requirements()
        
        # Generate production config
        prod_config = {
            "experiment_name": "production_deployment",
            "num_clients": 50,
            "num_rounds": 100,
            "model_name": "meta-llama/Llama-2-7b-hf",
            "privacy": {
                "epsilon": 1.0,  # Strict privacy for production
                "delta": 1e-6,
                "noise_multiplier": 2.0
            },
            "lora": {
                "r": 64,  # Higher rank for production
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            "security": {
                "ssl_enabled": True,
                "authentication_required": True,
                "encryption_level": "AES-256"
            },
            "scaling": {
                "auto_scaling_enabled": True,
                "max_clients_per_round": 20,
                "resource_monitoring": True
            }
        }
        
        config_file = self.config_dir / "production_config.json"
        with open(config_file, 'w') as f:
            json.dump(prod_config, f, indent=2)
            
        # Setup certificates
        self._setup_production_certificates()
        
        # Create deployment files
        self._create_deployment_files()
        
        logger.info("✅ Production setup complete!")
        logger.info("Next steps:")
        logger.info("  1. Review config: .quickstart/production_config.json")
        logger.info("  2. Deploy: docker-compose up -f deployment/production.yml")
        logger.info("  3. Monitor: python scripts/comprehensive_monitoring.py")
        
    def benchmark_mode(self) -> None:
        """Run comprehensive benchmarks."""
        logger.info("📊 Running comprehensive benchmarks")
        
        benchmark_configs = [
            {"name": "privacy_baseline", "epsilon": 1.0, "clients": 10},
            {"name": "privacy_moderate", "epsilon": 4.0, "clients": 10}, 
            {"name": "privacy_relaxed", "epsilon": 8.0, "clients": 10},
            {"name": "scale_small", "epsilon": 8.0, "clients": 5},
            {"name": "scale_medium", "epsilon": 8.0, "clients": 20},
            {"name": "scale_large", "epsilon": 8.0, "clients": 50}
        ]
        
        results_dir = self.repo_root / "benchmark_results"
        results_dir.mkdir(exist_ok=True)
        
        for config in benchmark_configs:
            logger.info(f"⚡ Running benchmark: {config['name']}")
            
            # Create benchmark config
            bench_config = {
                "experiment_name": f"benchmark_{config['name']}",
                "num_clients": config["clients"],
                "num_rounds": 5,  # Shorter for benchmarking
                "privacy": {"epsilon": config["epsilon"]},
                "benchmark_mode": True,
                "output_dir": str(results_dir / config['name'])
            }
            
            config_file = self.config_dir / f"benchmark_{config['name']}.json"
            with open(config_file, 'w') as f:
                json.dump(bench_config, f, indent=2)
                
        logger.info("🎯 Benchmark configurations generated")
        logger.info("Next steps:")
        logger.info("  1. Run: python scripts/benchmark_performance.py")
        logger.info("  2. View results: ./benchmark_results/")
        logger.info("  3. Generate report: python scripts/generate_reports.py")
        
    def _generate_demo_data(self) -> None:
        """Generate synthetic data for demo."""
        logger.info("🔄 Generating synthetic federated data")
        
        data_dir = self.repo_root / "demo_data"
        data_dir.mkdir(exist_ok=True)
        
        # Create mock datasets for 5 clients
        for i in range(5):
            client_data = {
                "client_id": f"demo_client_{i}",
                "data_samples": [
                    {"text": f"Sample medical record {j} for client {i}", "label": j % 3}
                    for j in range(100 + i * 50)  # Different data sizes
                ],
                "metadata": {
                    "data_type": "synthetic_medical",
                    "privacy_level": "high",
                    "num_samples": 100 + i * 50
                }
            }
            
            with open(data_dir / f"client_{i}_data.json", 'w') as f:
                json.dump(client_data, f, indent=2)
                
        logger.info(f"✅ Generated data for 5 demo clients in {data_dir}")
        
    def _check_production_requirements(self) -> None:
        """Check production environment requirements."""
        logger.info("🔍 Checking production requirements")
        
        requirements = [
            ("Docker", "docker --version"),
            ("Docker Compose", "docker-compose --version"),
            ("CUDA", "nvidia-smi"),
            ("Git", "git --version")
        ]
        
        missing = []
        for name, cmd in requirements:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"  ✅ {name}: Available")
                else:
                    logger.warning(f"  ⚠️  {name}: Not found")
                    missing.append(name)
            except FileNotFoundError:
                logger.warning(f"  ❌ {name}: Not found")
                missing.append(name)
                
        if missing:
            logger.warning(f"Missing requirements: {', '.join(missing)}")
            logger.info("Install missing components for full production functionality")
        else:
            logger.info("🎉 All production requirements satisfied")
            
    def _setup_production_certificates(self) -> None:
        """Setup SSL certificates for production."""
        cert_dir = self.repo_root / "certificates"
        cert_dir.mkdir(exist_ok=True)
        
        logger.info("🔐 Setting up SSL certificates")
        
        # Create self-signed certificate script
        cert_script = cert_dir / "generate_certs.sh"
        with open(cert_script, 'w') as f:
            f.write("""#!/bin/bash
# Generate self-signed certificates for DP-Federated LoRA
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes \\
    -subj "/C=US/ST=State/L=City/O=Organization/CN=dp-fed-lora-server"
            
openssl req -x509 -newkey rsa:4096 -keyout client.key -out client.crt -days 365 -nodes \\
    -subj "/C=US/ST=State/L=City/O=Organization/CN=dp-fed-lora-client"
            
echo "Certificates generated successfully"
""")
        
        cert_script.chmod(0o755)
        logger.info(f"📜 Certificate generation script: {cert_script}")
        
    def _create_deployment_files(self) -> None:
        """Create additional deployment configurations."""
        logger.info("📦 Creating deployment configurations")
        
        # Enhanced docker-compose for production
        production_compose = """version: '3.8'

services:
  federated-server:
    build: .
    ports:
      - "8443:8443"
    environment:
      - SSL_CERT_PATH=/app/certificates/server.crt
      - SSL_KEY_PATH=/app/certificates/server.key
      - PRIVACY_EPSILON=1.0
      - NUM_CLIENTS=50
    volumes:
      - ./certificates:/app/certificates:ro
      - ./logs:/app/logs
    restart: unless-stopped
    
  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=federated_admin
    restart: unless-stopped
"""
        
        with open(self.repo_root / "docker-compose.production.yml", 'w') as f:
            f.write(production_compose)
            
        logger.info("✅ Production deployment files created")


def main():
    """Main entry point for quick start script."""
    parser = argparse.ArgumentParser(description="DP-Federated LoRA Quick Start")
    parser.add_argument(
        "--mode", 
        choices=["demo", "production", "benchmark"],
        default="demo",
        help="Quick start mode"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 DP-Federated LoRA Lab Quick Start")
    print("=====================================")
    
    manager = QuickStartManager()
    
    try:
        if args.mode == "demo":
            manager.demo_mode()
        elif args.mode == "production":
            manager.production_mode()
        elif args.mode == "benchmark":
            manager.benchmark_mode()
            
    except KeyboardInterrupt:
        print("\n⏹️  Quick start cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Quick start failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
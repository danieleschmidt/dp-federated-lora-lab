"""
Federated learning server implementation.

This module implements the server-side coordination for DP-Federated LoRA training,
including client management, secure aggregation, and global model updates.
"""

import asyncio
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import asdict
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from .config import FederatedConfig, SecurityConfig, AggregationMethod
from .aggregation import create_aggregator, BaseAggregator
from .privacy import PrivacyAccountant
from .monitoring import ServerMetricsCollector
from .client import DPLoRAClient


logger = logging.getLogger(__name__)


class ClientInfo:
    """Information about a registered client."""
    
    def __init__(self, client_id: str, capabilities: Dict[str, Any]):
        """
        Initialize client info.
        
        Args:
            client_id: Unique client identifier
            capabilities: Client capabilities and metadata
        """
        self.client_id = client_id
        self.capabilities = capabilities
        self.last_seen = time.time()
        self.rounds_participated = 0
        self.total_examples = capabilities.get("num_examples", 0)
        self.is_active = True
        self.performance_history: List[Dict[str, float]] = []


class FederatedServer:
    """
    Federated learning server for DP-LoRA training.
    
    Coordinates federated training across multiple clients with differential
    privacy guarantees and secure aggregation protocols.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[FederatedConfig] = None,
        num_clients: int = 10,
        rounds: int = 50,
        privacy_budget: Optional[Dict[str, float]] = None
    ):
        """
        Initialize federated server.
        
        Args:
            model_name: Base model name for training
            config: Federated learning configuration
            num_clients: Expected number of clients
            rounds: Number of training rounds
            privacy_budget: Privacy budget parameters
        """
        self.model_name = model_name
        self.config = config or FederatedConfig(model_name=model_name)
        self.config.model_name = model_name  # Override if provided
        
        # Update configuration with parameters
        if rounds:
            self.config.num_rounds = rounds
        if privacy_budget:
            if "epsilon" in privacy_budget:
                self.config.privacy.epsilon = privacy_budget["epsilon"]
            if "delta" in privacy_budget:
                self.config.privacy.delta = privacy_budget["delta"]
        
        # Server state
        self.current_round = 0
        self.registered_clients: Dict[str, ClientInfo] = {}\n        self.selected_clients: Set[str] = set()
        self.client_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        self.client_weights: Dict[str, float] = {}
        
        # Components
        self.global_model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.aggregator: BaseAggregator = create_aggregator(self.config.security)
        self.privacy_accountant = PrivacyAccountant(
            total_epsilon=self.config.privacy.epsilon,
            total_delta=self.config.privacy.delta
        )
        self.metrics_collector = ServerMetricsCollector()
        
        # Training state
        self.training_history: List[Dict[str, Any]] = []
        self.is_training = False
        self.round_start_time = 0.0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.security.max_clients)
        
        logger.info(f"Initialized federated server for {model_name}")
    
    def initialize_global_model(self) -> None:
        """Initialize the global model with LoRA adaptation."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA
            self.global_model = get_peft_model(base_model, lora_config)
            
            # Set only LoRA parameters as trainable
            for name, param in self.global_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Initialize LoRA parameters with small random values
            for name, param in self.global_model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            
            trainable_params = sum(
                p.numel() for p in self.global_model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.global_model.parameters())
            
            logger.info(
                f"Global model initialized: {trainable_params:,} trainable / "
                f"{total_params:,} total parameters "
                f"({100 * trainable_params / total_params:.2f}% trainable)"
            )
            
        except Exception as e:
            logger.error(f"Error initializing global model: {e}")
            raise
    
    def register_client(
        self,
        client_id: str,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new client with the server.
        
        Args:
            client_id: Unique client identifier
            capabilities: Client capabilities and metadata
            
        Returns:
            True if registration successful
        """
        if len(self.registered_clients) >= self.config.security.max_clients:
            logger.warning(f"Maximum clients reached, rejecting {client_id}")
            return False
        
        if client_id in self.registered_clients:
            logger.warning(f"Client {client_id} already registered")
            return False
        
        capabilities = capabilities or {}
        client_info = ClientInfo(client_id, capabilities)
        self.registered_clients[client_id] = client_info
        
        logger.info(f"Registered client {client_id} ({len(self.registered_clients)} total)")
        return True
    
    def unregister_client(self, client_id: str) -> bool:
        """
        Unregister a client from the server.
        
        Args:
            client_id: Client identifier to remove
            
        Returns:
            True if unregistration successful
        """
        if client_id in self.registered_clients:
            del self.registered_clients[client_id]
            self.selected_clients.discard(client_id)
            logger.info(f"Unregistered client {client_id}")
            return True
        return False
    
    def select_clients(self, round_num: int) -> List[str]:
        """
        Select clients for participation in the current round.
        
        Args:
            round_num: Current training round number
            
        Returns:
            List of selected client IDs
        """
        available_clients = [
            client_id for client_id, info in self.registered_clients.items()
            if info.is_active and time.time() - info.last_seen < 300  # 5 min timeout
        ]
        
        if len(available_clients) < self.config.security.min_clients:
            raise ValueError(
                f"Insufficient clients: {len(available_clients)} < "
                f"{self.config.security.min_clients} required"
            )
        
        # Sample clients based on sampling rate
        num_selected = max(
            self.config.security.min_clients,
            int(len(available_clients) * self.config.security.client_sampling_rate)
        )
        num_selected = min(num_selected, len(available_clients))
        
        # Random selection (could be enhanced with more sophisticated strategies)
        np.random.seed(round_num)  # Deterministic selection
        selected = np.random.choice(
            available_clients,
            size=num_selected,
            replace=False
        ).tolist()
        
        self.selected_clients = set(selected)
        
        logger.info(f"Selected {len(selected)} clients for round {round_num}: {selected}")
        return selected
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model parameters (LoRA only).
        
        Returns:
            Dictionary of global LoRA parameters
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized")
        
        global_params = {}
        for name, param in self.global_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                global_params[name] = param.clone().detach()
        
        return global_params
    
    def update_global_model(
        self,
        aggregated_updates: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update global model with aggregated client updates.
        
        Args:
            aggregated_updates: Aggregated parameter updates
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized")
        
        # Apply updates to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_updates and param.requires_grad:
                    param.data += aggregated_updates[name]
        
        logger.info("Global model updated with aggregated client updates")
    
    def collect_client_updates(
        self,
        client_id: str,
        updates: Dict[str, torch.Tensor],
        training_info: Dict[str, Any]
    ) -> None:
        """
        Collect updates from a client.
        
        Args:
            client_id: Client identifier
            updates: Parameter updates from client
            training_info: Training metadata from client
        """
        if client_id not in self.selected_clients:
            logger.warning(f"Received updates from unselected client {client_id}")
            return
        
        self.client_updates[client_id] = updates
        
        # Extract client weight (number of training examples)
        num_examples = training_info.get("num_examples", 1)
        self.client_weights[client_id] = float(num_examples)
        
        # Update client info
        if client_id in self.registered_clients:
            client_info = self.registered_clients[client_id]
            client_info.last_seen = time.time()
            client_info.rounds_participated += 1
            client_info.performance_history.append({
                "round": self.current_round,
                "loss": training_info.get("loss", 0.0),
                "privacy_spent": training_info.get("privacy_spent", {})
            })
        
        # Record privacy spending
        privacy_spent = training_info.get("privacy_spent", {})
        if "epsilon" in privacy_spent:
            self.privacy_accountant.record_spending(
                client_id=client_id,
                epsilon_spent=privacy_spent["epsilon"],
                delta_spent=privacy_spent.get("delta", 0.0),
                round_num=self.current_round
            )
        
        logger.info(f"Collected updates from client {client_id} ({len(self.client_updates)}/{len(self.selected_clients)})")
    
    def aggregate_updates(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using configured method.
        
        Returns:
            Aggregated parameter updates
        """
        if not self.client_updates:
            raise ValueError("No client updates to aggregate")
        
        logger.info(f"Aggregating updates from {len(self.client_updates)} clients")
        
        # Perform aggregation
        aggregated = self.aggregator.aggregate(
            client_updates=self.client_updates,
            client_weights=self.client_weights
        )
        
        # Record aggregation metrics
        self.metrics_collector.record_aggregation({
            "round": self.current_round,
            "num_clients": len(self.client_updates),
            "aggregation_method": self.config.security.aggregation_method.value,
            "total_weight": sum(self.client_weights.values())
        })
        
        return aggregated
    
    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Execute a single federated learning round.
        
        Args:
            round_num: Current round number
            
        Returns:
            Round results and metrics
        """
        self.current_round = round_num
        self.round_start_time = time.time()
        
        logger.info(f"Starting federated learning round {round_num}")
        
        # Check privacy budget feasibility
        rounds_remaining = self.config.num_rounds - round_num + 1
        if not self.privacy_accountant.check_budget_feasible(rounds_remaining):
            logger.warning("Privacy budget may be exhausted before training completion")
        
        # Select clients for this round
        try:
            selected_clients = self.select_clients(round_num)
        except ValueError as e:
            logger.error(f"Client selection failed: {e}")
            raise
        
        # Reset round state
        self.client_updates.clear()
        self.client_weights.clear()
        
        # Get current global parameters
        global_params = self.get_global_parameters()
        
        # Simulate client training (in real implementation, this would be async communication)
        round_results = {
            "round": round_num,
            "selected_clients": selected_clients,
            "global_params_sent": len(global_params),
            "client_results": {}
        }
        
        # Wait for client updates (simulated)
        timeout = self.config.server_timeout if hasattr(self.config, 'server_timeout') else 300
        start_time = time.time()
        
        while len(self.client_updates) < len(selected_clients):
            if time.time() - start_time > timeout:
                logger.warning(f"Round {round_num} timeout: {len(self.client_updates)}/{len(selected_clients)} clients responded")
                break
            time.sleep(0.1)  # Small delay
        
        # Proceed with aggregation if we have minimum clients
        if len(self.client_updates) < self.config.security.min_clients:
            raise ValueError(
                f"Insufficient client responses: {len(self.client_updates)} < "
                f"{self.config.security.min_clients} required"
            )
        
        # Aggregate client updates
        aggregated_updates = self.aggregate_updates()
        
        # Update global model
        self.update_global_model(aggregated_updates)
        
        # Calculate round metrics
        round_duration = time.time() - self.round_start_time
        avg_client_weight = np.mean(list(self.client_weights.values())) if self.client_weights else 0
        
        round_results.update({
            "duration_seconds": round_duration,
            "clients_participated": len(self.client_updates),
            "aggregated_params": len(aggregated_updates),
            "avg_client_weight": avg_client_weight,
            "privacy_budget_status": self.privacy_accountant.get_budget_status()
        })
        
        # Record round in history
        self.training_history.append(round_results.copy())
        
        logger.info(
            f"Completed round {round_num} in {round_duration:.1f}s "
            f"({len(self.client_updates)} clients participated)"
        )
        
        return round_results
    
    def train(
        self,
        clients: Optional[List[DPLoRAClient]] = None,
        aggregation: Optional[str] = None,
        client_sampling_rate: Optional[float] = None,
        local_epochs: Optional[int] = None
    ) -> "TrainingHistory":
        """
        Run complete federated training.
        
        Args:
            clients: List of client objects for training
            aggregation: Aggregation method override
            client_sampling_rate: Client sampling rate override
            local_epochs: Local epochs override
            
        Returns:
            Training history object
        """
        # Update configuration with overrides
        if aggregation:
            try:
                self.config.security.aggregation_method = AggregationMethod(aggregation)
                self.aggregator = create_aggregator(self.config.security)
            except ValueError:
                logger.warning(f"Unknown aggregation method: {aggregation}")
        
        if client_sampling_rate is not None:
            self.config.security.client_sampling_rate = client_sampling_rate
        
        if local_epochs is not None:
            self.config.local_epochs = local_epochs
        
        # Initialize global model if not done
        if self.global_model is None:
            self.initialize_global_model()
        
        # Register clients if provided
        if clients:
            for client in clients:
                self.register_client(
                    client.client_id,
                    client.get_data_statistics()
                )
        
        self.is_training = True
        training_start_time = time.time()
        
        try:
            logger.info(f"Starting federated training for {self.config.num_rounds} rounds")
            
            for round_num in range(1, self.config.num_rounds + 1):
                # Run training round
                round_results = self.run_round(round_num)
                
                # Simulate client training if clients provided
                if clients:
                    self._simulate_client_training(clients, round_num)
                
                # Log progress
                if round_num % 10 == 0 or round_num == self.config.num_rounds:
                    budget_status = self.privacy_accountant.get_budget_status()
                    logger.info(
                        f"Round {round_num}/{self.config.num_rounds} completed. "
                        f"Privacy budget: {budget_status['epsilon_spent']:.2f}/"
                        f"{budget_status['total_epsilon']:.2f} ε"
                    )
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
        
        training_duration = time.time() - training_start_time
        final_budget = self.privacy_accountant.get_budget_status()
        
        # Create training history
        history = TrainingHistory(
            rounds_completed=self.current_round,
            total_duration=training_duration,
            final_epsilon=final_budget["epsilon_spent"],
            final_delta=self.config.privacy.delta,
            training_history=self.training_history.copy(),
            privacy_accountant=self.privacy_accountant
        )
        
        logger.info(
            f"Federated training completed: {self.current_round} rounds, "
            f"{training_duration:.1f}s, ε={final_budget['epsilon_spent']:.2f}"
        )
        
        return history
    
    def _simulate_client_training(self, clients: List[DPLoRAClient], round_num: int) -> None:
        """Simulate client training for testing purposes."""
        global_params = self.get_global_parameters()
        
        # Send global model to selected clients and collect updates
        for client in clients:
            if client.client_id in self.selected_clients:
                try:
                    # Update client with global model
                    client.receive_global_model(global_params)
                    client.current_round = round_num
                    
                    # Perform local training
                    updates = client.train_local(self.config.local_epochs)
                    
                    # Collect updates
                    training_info = {
                        "num_examples": len(client.local_dataset) if client.local_dataset else 100,
                        "loss": np.random.uniform(0.5, 2.0),  # Simulated loss
                        "privacy_spent": client.privacy_spent.copy()
                    }
                    
                    self.collect_client_updates(client.client_id, updates, training_info)
                    
                except Exception as e:
                    logger.error(f"Error in client {client.client_id} training: {e}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get comprehensive server status.
        
        Returns:
            Server status dictionary
        """
        budget_status = self.privacy_accountant.get_budget_status()
        
        return {
            "server_id": f"federated_server_{self.model_name}",
            "current_round": self.current_round,
            "total_rounds": self.config.num_rounds,
            "is_training": self.is_training,
            "registered_clients": len(self.registered_clients),
            "active_clients": len([
                c for c in self.registered_clients.values()
                if c.is_active and time.time() - c.last_seen < 300
            ]),
            "selected_clients": list(self.selected_clients),
            "privacy_budget": budget_status,
            "model_info": {
                "model_name": self.model_name,
                "lora_rank": self.config.lora.r,
                "total_parameters": sum(p.numel() for p in self.global_model.parameters()) if self.global_model else 0,
                "trainable_parameters": sum(p.numel() for p in self.global_model.parameters() if p.requires_grad) if self.global_model else 0
            },
            "configuration": {
                "aggregation_method": self.config.security.aggregation_method.value,
                "client_sampling_rate": self.config.security.client_sampling_rate,
                "local_epochs": self.config.local_epochs,
                "privacy_epsilon": self.config.privacy.epsilon,
                "privacy_delta": self.config.privacy.delta
            }
        }


class TrainingHistory:
    """Container for federated training history and results."""
    
    def __init__(
        self,
        rounds_completed: int,
        total_duration: float,
        final_epsilon: float,
        final_delta: float,
        training_history: List[Dict[str, Any]],
        privacy_accountant: PrivacyAccountant
    ):
        """
        Initialize training history.
        
        Args:
            rounds_completed: Number of completed rounds
            total_duration: Total training time in seconds
            final_epsilon: Final epsilon spent
            final_delta: Final delta parameter
            training_history: List of round results
            privacy_accountant: Privacy accountant instance
        """
        self.rounds_completed = rounds_completed
        self.total_duration = total_duration
        self.final_epsilon = final_epsilon
        self.final_delta = final_delta
        self.training_history = training_history
        self.privacy_accountant = privacy_accountant
    
    @property
    def final_accuracy(self) -> float:
        """Estimated final accuracy (placeholder)."""
        # In a real implementation, this would be computed from validation data
        return 0.85 if self.rounds_completed > 10 else 0.70
    
    @property
    def total_epsilon(self) -> float:
        """Total epsilon budget."""
        return self.privacy_accountant.total_epsilon
    
    def get_round_metrics(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific round."""
        for round_data in self.training_history:
            if round_data.get("round") == round_num:
                return round_data
        return None
    
    def get_privacy_timeline(self) -> List[Dict[str, float]]:
        """Get privacy spending over time."""
        timeline = []
        for round_data in self.training_history:
            budget_status = round_data.get("privacy_budget_status", {})
            timeline.append({
                "round": round_data["round"],
                "epsilon_spent": budget_status.get("epsilon_spent", 0.0),
                "epsilon_remaining": budget_status.get("epsilon_remaining", 0.0),
                "budget_utilization": budget_status.get("budget_utilization", 0.0)
            })
        return timeline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rounds_completed": self.rounds_completed,
            "total_duration": self.total_duration,
            "final_epsilon": self.final_epsilon,
            "final_delta": self.final_delta,
            "final_accuracy": self.final_accuracy,
            "training_history": self.training_history,
            "privacy_timeline": self.get_privacy_timeline()
        }
"""
Federated learning client implementation with differential privacy.

This module implements the client-side logic for DP-Federated LoRA training,
including local training, privacy mechanisms, and secure communication.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np

from .config import FederatedConfig, ClientConfig, PrivacyConfig, LoRAConfig
from .privacy import PrivacyEngine, PrivacyAccountant
from .monitoring import LocalMetricsCollector
from .network_client import FederatedNetworkClient, NetworkClientError


logger = logging.getLogger(__name__)


class DPLoRAClient:
    """
    Differential Privacy LoRA client for federated learning.
    
    Handles local training with LoRA adaptation and differential privacy,
    while maintaining secure communication with the federation server.
    """
    
    def __init__(
        self,
        client_id: str,
        data_path: str,
        config: Optional[FederatedConfig] = None,
        client_config: Optional[ClientConfig] = None,
        server_url: Optional[str] = None
    ):
        """
        Initialize DP LoRA client.
        
        Args:
            client_id: Unique identifier for this client
            data_path: Path to local training data
            config: Federated learning configuration
            client_config: Client-specific configuration
            server_url: URL of the federated learning server
        """
        self.client_id = client_id
        self.data_path = data_path
        self.config = config or FederatedConfig()
        self.client_config = client_config or ClientConfig(
            client_id=client_id,
            data_path=data_path
        )
        
        # Network communication
        self.server_url = server_url
        self.network_client: Optional[FederatedNetworkClient] = None
        if server_url:
            self.network_client = FederatedNetworkClient(server_url, client_id)
        
        # Initialize components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[nn.Module] = None
        self.privacy_engine = PrivacyEngine(self.config.privacy)
        self.metrics_collector = LocalMetricsCollector(client_id)
        
        # Training state
        self.current_round = 0
        self.local_dataset: Optional[Dataset] = None
        self.data_loader: Optional[DataLoader] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Privacy tracking
        self.privacy_spent = {"epsilon": 0.0, "delta": 0.0}
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized DP LoRA client {client_id}")
    
    def setup(self) -> None:
        """Setup client components including model, data, and privacy."""
        self._load_data()
        self._initialize_model()
        self._setup_privacy()
        logger.info(f"Client {self.client_id} setup completed")
    
    def _load_data(self) -> None:
        """Load and preprocess local training data."""
        try:
            if self.data_path.endswith('.json'):
                # Load JSON dataset
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to HuggingFace dataset format
                if isinstance(data, list):
                    dataset_dict = {"text": data}
                else:
                    dataset_dict = data
                
                self.local_dataset = CustomDataset(dataset_dict)
                
            elif self.data_path.endswith('.jsonl'):
                # Load JSONL dataset
                data = []
                with open(self.data_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                
                self.local_dataset = CustomDataset({"text": [item.get("text", "") for item in data]})
                
            else:
                # Try to load as HuggingFace dataset
                self.local_dataset = load_dataset(
                    "text",
                    data_files=self.data_path,
                    split="train"
                )
            
            # Apply data preprocessing
            if self.client_config.max_examples:
                if hasattr(self.local_dataset, 'select'):
                    self.local_dataset = self.local_dataset.select(
                        range(min(len(self.local_dataset), self.client_config.max_examples))
                    )
                else:
                    # For custom dataset
                    self.local_dataset = self.local_dataset[:self.client_config.max_examples]
            
            logger.info(f"Loaded {len(self.local_dataset)} examples for client {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error loading data for client {self.client_id}: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Initialize the base model and apply LoRA adaptation."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side="left",
                truncation_side="left"
            )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
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
            
            # Apply LoRA to model
            self.model = get_peft_model(base_model, lora_config)
            
            # Enable training mode for LoRA parameters only
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            logger.info(f"Initialized model with LoRA (rank={self.config.lora.r})")
            
        except Exception as e:
            logger.error(f"Error initializing model for client {self.client_id}: {e}")
            raise
    
    def _setup_privacy(self) -> None:
        """Setup differential privacy mechanisms."""
        try:
            # Create data loader for privacy engine
            tokenized_dataset = self._tokenize_dataset()
            
            self.data_loader = DataLoader(
                tokenized_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Setup optimizer
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            logger.info("Privacy setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up privacy for client {self.client_id}: {e}")
            raise
    
    def _tokenize_dataset(self) -> Dataset:
        """Tokenize the local dataset."""
        def tokenize_function(examples):
            if isinstance(examples, dict) and "text" in examples:
                texts = examples["text"]
            else:
                texts = examples
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        if hasattr(self.local_dataset, 'map'):
            # HuggingFace dataset
            tokenized = self.local_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.local_dataset.column_names
            )
        else:
            # Custom dataset
            tokenized = [
                tokenize_function(item["text"])
                for item in self.local_dataset
            ]
        
        return tokenized
    
    def receive_global_model(self, global_parameters: Dict[str, torch.Tensor]) -> None:
        """
        Receive and load global model parameters.
        
        Args:
            global_parameters: Global model parameters from server
        """
        try:
            # Load only LoRA parameters
            lora_params = {}
            for name, param in global_parameters.items():
                if "lora_" in name:
                    lora_params[name] = param
            
            # Update model with global parameters
            model_state = self.model.state_dict()
            model_state.update(lora_params)
            self.model.load_state_dict(model_state, strict=False)
            
            logger.info(f"Client {self.client_id} received global model (round {self.current_round})")
            
        except Exception as e:
            logger.error(f"Error loading global model for client {self.client_id}: {e}")
            raise
    
    async def register_with_server(self) -> bool:
        """
        Register client with the federated server.
        
        Returns:
            True if registration successful
            
        Raises:
            NetworkClientError: If registration fails
        """
        if not self.network_client:
            raise NetworkClientError("No network client configured")
        
        # Prepare client capabilities
        capabilities = {
            "model_name": self.config.model_name,
            "num_examples": len(self.local_dataset) if self.local_dataset else 0,
            "compute_capability": "gpu" if torch.cuda.is_available() else "cpu",
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 8,
            "supported_tasks": ["causal_lm"],
            "privacy_requirements": {
                "epsilon": self.config.privacy.epsilon,
                "delta": self.config.privacy.delta
            }
        }
        
        success = await self.network_client.register(capabilities)
        
        if success:
            # Update client config with server configuration
            server_config = self.network_client.get_server_config()
            
            # Override local config with server settings if needed
            if "lora_config" in server_config:
                server_lora = server_config["lora_config"]
                self.config.lora.r = server_lora.get("r", self.config.lora.r)
                self.config.lora.lora_alpha = server_lora.get("lora_alpha", self.config.lora.lora_alpha)
                self.config.lora.target_modules = server_lora.get("target_modules", self.config.lora.target_modules)
            
            logger.info(f"Successfully registered client {self.client_id} with server")
        
        return success
    
    async def participate_in_round(self, round_num: int) -> bool:
        """
        Participate in a federated training round.
        
        Args:
            round_num: Training round number
            
        Returns:
            True if participation successful
        """
        if not self.network_client:
            raise NetworkClientError("No network client configured")
        
        try:
            # Get global parameters from server
            round_data = await self.network_client.get_round_parameters(round_num)
            
            # Update local model with global parameters
            self.receive_global_model(round_data["global_parameters"])
            
            # Perform local training
            local_updates = self.train_local()
            
            # Collect training metrics
            metrics = self.get_training_metrics()
            
            # Submit updates to server
            success = await self.network_client.submit_client_update(
                round_num=round_num,
                parameters=self._serialize_parameters(local_updates),
                metrics=metrics,
                num_examples=len(self.local_dataset) if self.local_dataset else 0,
                privacy_spent=self.privacy_spent
            )
            
            if success:
                self.current_round = round_num
                logger.info(f"Successfully completed round {round_num}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error participating in round {round_num}: {e}")
            return False
    
    def _serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Serialize model parameters for network transmission.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Serialized parameters
        """
        serialized = {}
        for name, param in parameters.items():
            if param is not None:
                serialized[name] = {
                    "data": param.detach().cpu().numpy().tolist(),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype)
                }
        return serialized
    
    def _deserialize_parameters(self, serialized: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Deserialize parameters from network transmission.
        
        Args:
            serialized: Serialized parameters
            
        Returns:
            Model parameters
        """
        parameters = {}
        for name, param_data in serialized.items():
            if param_data and "data" in param_data:
                data = torch.tensor(param_data["data"])
                shape = param_data["shape"]
                parameters[name] = data.reshape(shape)
        return parameters
    
    async def close_network_connection(self):
        """Close network connection to server."""
        if self.network_client:
            await self.network_client.close()
    
    def train_local(self, num_epochs: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform local training with differential privacy.
        
        Args:
            num_epochs: Number of local epochs (uses config default if None)
            
        Returns:
            Local model parameter updates
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        try:
            # Store initial parameters
            initial_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            
            # Attach privacy engine
            if self.config.privacy.epsilon > 0:
                private_model, private_optimizer, private_loader = self.privacy_engine.attach(
                    self.model, self.optimizer, self.data_loader
                )
            else:
                private_model, private_optimizer, private_loader = (
                    self.model, self.optimizer, self.data_loader
                )
            
            # Training loop
            private_model.train()
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for batch in private_loader:
                    # Prepare batch
                    if isinstance(batch, dict):
                        input_ids = batch["input_ids"]
                        attention_mask = batch.get("attention_mask", None)
                        labels = batch.get("labels", input_ids)
                    else:
                        input_ids = batch
                        attention_mask = None
                        labels = input_ids
                    
                    # Move to device
                    device = next(private_model.parameters()).device
                    input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    private_optimizer.zero_grad()
                    outputs = private_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Backward pass with privacy
                    loss.backward()
                    private_optimizer.step()
                    
                    # Track privacy spending
                    if self.config.privacy.epsilon > 0:
                        self.privacy_engine.step()
                    
                    # Metrics
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
                logger.info(f"Client {self.client_id} epoch {epoch + 1}/{num_epochs}, loss: {avg_epoch_loss:.4f}")
                
                total_loss += epoch_loss
                num_batches += epoch_batches
            
            # Detach privacy engine
            if self.config.privacy.epsilon > 0:
                self.privacy_engine.detach()
            
            # Compute parameter updates
            final_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            
            updates = {
                name: final_params[name] - initial_params[name]
                for name in initial_params.keys()
            }
            
            # Update privacy tracking
            if self.config.privacy.epsilon > 0:
                privacy_spent = self.privacy_engine.get_privacy_spent()
                self.privacy_spent["epsilon"] += privacy_spent["epsilon"]
                self.privacy_spent["delta"] += privacy_spent["delta"]
            
            # Record training metrics
            avg_loss = total_loss / max(num_batches, 1)
            self.training_history.append({
                "round": self.current_round,
                "epochs": num_epochs,
                "avg_loss": avg_loss,
                "privacy_spent": self.privacy_spent.copy(),
                "num_examples": len(self.local_dataset)
            })
            
            self.metrics_collector.record_training_round({
                "round": self.current_round,
                "loss": avg_loss,
                "privacy_epsilon": self.privacy_spent["epsilon"],
                "examples_processed": len(self.local_dataset) * num_epochs
            })
            
            logger.info(
                f"Client {self.client_id} completed local training "
                f"(loss: {avg_loss:.4f}, ε: {self.privacy_spent['epsilon']:.2f})"
            )
            
            return updates
            
        except Exception as e:
            logger.error(f"Error during local training for client {self.client_id}: {e}")
            raise
    
    def get_data_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about local data (privacy-preserving).
        
        Returns:
            Dictionary with data statistics
        """
        if not self.local_dataset:
            return {}
        
        stats = {
            "num_examples": len(self.local_dataset),
            "client_id": self.client_id,
        }
        
        # Add privacy-preserving statistics if needed
        if self.config.privacy.epsilon > 0:
            # Add noise to statistics for privacy
            noise_scale = 1.0 / self.config.privacy.epsilon
            stats["num_examples"] += int(np.random.laplace(0, noise_scale))
            stats["num_examples"] = max(0, stats["num_examples"])  # Ensure non-negative
        
        return stats
    
    def update_lora_rank(self, new_rank: int) -> None:
        """
        Update LoRA rank dynamically.
        
        Args:
            new_rank: New rank for LoRA adaptation
        """
        if new_rank == self.config.lora.r:
            return  # No change needed
        
        logger.info(f"Updating LoRA rank from {self.config.lora.r} to {new_rank}")
        
        # Update configuration
        self.config.lora.r = new_rank
        
        # Reinitialize model with new rank
        self._initialize_model()
        self._setup_privacy()
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters (LoRA only).
        
        Returns:
            Dictionary of LoRA parameters
        """
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_params[name] = param.clone().detach()
        
        return lora_params
    
    def evaluate_model(self, test_data: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test dataset (uses validation split if None)
            
        Returns:
            Evaluation metrics
        """
        if test_data is None and self.client_config.local_validation_split > 0:
            # Create validation split
            dataset_size = len(self.local_dataset)
            val_size = int(dataset_size * self.client_config.local_validation_split)
            test_data = self.local_dataset.select(range(dataset_size - val_size, dataset_size))
        
        if test_data is None:
            return {"loss": float('inf'), "perplexity": float('inf')}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create test data loader
        test_loader = DataLoader(
            test_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch.get("labels", input_ids)
                
                # Move to device
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_examples": len(test_data)
        }
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get comprehensive client information.
        
        Returns:
            Client information dictionary
        """
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "privacy_spent": self.privacy_spent.copy(),
            "num_training_examples": len(self.local_dataset) if self.local_dataset else 0,
            "lora_rank": self.config.lora.r,
            "training_history": self.training_history[-5:],  # Last 5 rounds
            "data_statistics": self.get_data_statistics()
        }


class CustomDataset(Dataset):
    """Custom dataset wrapper for local data."""
    
    def __init__(self, data: Dict[str, List[str]]):
        """
        Initialize custom dataset.
        
        Args:
            data: Dictionary with data fields
        """
        self.data = data
        self.texts = data.get("text", [])
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get item by index."""
        return {"text": self.texts[idx]}
    
    def select(self, indices: List[int]) -> "CustomDataset":
        """Select subset of dataset."""
        selected_texts = [self.texts[i] for i in indices]
        return CustomDataset({"text": selected_texts})


def create_mock_client(
    client_id: str,
    num_examples: int = 100,
    config: Optional[FederatedConfig] = None
) -> DPLoRAClient:
    """
    Create a mock client with synthetic data for testing.
    
    Args:
        client_id: Client identifier
        num_examples: Number of synthetic examples
        config: Federated configuration
        
    Returns:
        Mock client with synthetic data
    """
    # Generate synthetic text data
    mock_texts = [
        f"This is example text number {i} for client {client_id}. "
        f"It contains some sample content for training purposes."
        for i in range(num_examples)
    ]
    
    # Create temporary data file
    temp_dir = Path("/tmp/federated_clients")
    temp_dir.mkdir(exist_ok=True)
    
    data_path = temp_dir / f"{client_id}_data.json"
    with open(data_path, 'w') as f:
        json.dump(mock_texts, f)
    
    # Create client
    client = DPLoRAClient(
        client_id=client_id,
        data_path=str(data_path),
        config=config or FederatedConfig()
    )
    
    return client
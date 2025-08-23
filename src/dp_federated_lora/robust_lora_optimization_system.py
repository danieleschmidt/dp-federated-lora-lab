"""
Robust LoRA Optimization System with comprehensive error handling, validation,
monitoring, and resilience features for production environments.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import threading
from contextlib import contextmanager, asynccontextmanager
from collections import deque
import warnings

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import psutil
from threading import Lock, Event
import asyncio
import aiofiles

from .novel_lora_hyperparameter_optimizer import (
    NovelLoRAHyperparameterOptimizer, OptimizationStrategy, 
    OptimizationResult, LoRAHyperParams
)
from .exceptions import (
    OptimizationError, ModelError, ResourceError, ValidationError,
    TimeoutError, ConfigurationError
)
from .monitoring import LocalMetricsCollector
from .error_handler import ErrorHandler, CircuitBreaker, RetryConfig, with_error_handling
from .performance import PerformanceMonitor, ResourceManager

logger = logging.getLogger(__name__)


class OptimizationState(Enum):
    """States of the optimization process."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    ANALYZING_GRADIENTS = "analyzing_gradients"
    OPTIMIZING = "optimizing"
    EVALUATING = "evaluating"
    CONVERGED = "converged"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class OptimizationConfig:
    """Comprehensive optimization configuration with validation."""
    strategy: OptimizationStrategy = OptimizationStrategy.HYBRID_QUANTUM
    n_trials: int = 50
    max_duration_minutes: int = 60
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_gpu_memory_gb: float = 12.0
    
    # Convergence criteria
    convergence_patience: int = 10
    min_improvement: float = 0.001
    convergence_window: int = 5
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    
    # Checkpoint and recovery
    checkpoint_interval: int = 10
    auto_recovery: bool = True
    checkpoint_dir: str = "/tmp/lora_optimization_checkpoints"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.n_trials <= 0:
            raise ValidationError("n_trials must be positive")
        if self.max_duration_minutes <= 0:
            raise ValidationError("max_duration_minutes must be positive")
        if self.max_memory_gb <= 0:
            raise ValidationError("max_memory_gb must be positive")
        if self.convergence_patience < 1:
            raise ValidationError("convergence_patience must be at least 1")
        if self.min_improvement < 0:
            raise ValidationError("min_improvement must be non-negative")


@dataclass
class HealthMetrics:
    """System health metrics for monitoring."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_gb: float = 0.0
    optimization_progress: float = 0.0
    trials_completed: int = 0
    trials_failed: int = 0
    current_best_score: float = float('-inf')
    time_elapsed: float = 0.0
    estimated_time_remaining: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_gb': self.memory_gb,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_memory_gb': self.gpu_memory_gb,
            'optimization_progress': self.optimization_progress,
            'trials_completed': self.trials_completed,
            'trials_failed': self.trials_failed,
            'current_best_score': self.current_best_score,
            'time_elapsed': self.time_elapsed,
            'estimated_time_remaining': self.estimated_time_remaining
        }


class ResourceMonitor:
    """Monitors system resources during optimization."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitoring = False
        self.metrics_history = deque(maxlen=100)
        self.resource_warnings = []
        self._lock = Lock()
        self._stop_event = Event()
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self._stop_event.set()
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources continuously."""
        while not self._stop_event.wait(timeout=1.0):
            try:
                metrics = self._collect_resource_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    self._check_resource_limits(metrics)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
    
    def _collect_resource_metrics(self) -> HealthMetrics:
        """Collect current resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        
        # GPU metrics (if available)
        gpu_memory_percent = 0.0
        gpu_memory_gb = 0.0
        
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
                gpu_memory_gb = gpu_memory
                gpu_memory_percent = (gpu_memory / gpu_memory_max * 100) if gpu_memory_max > 0 else 0
        except Exception:
            pass  # GPU metrics not critical
        
        return HealthMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_gb=memory_gb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def _check_resource_limits(self, metrics: HealthMetrics):
        """Check if resource limits are exceeded."""
        warnings_added = []
        
        if metrics.cpu_percent > self.config.max_cpu_percent:
            warning = f"CPU usage {metrics.cpu_percent:.1f}% exceeds limit {self.config.max_cpu_percent}%"
            warnings_added.append(warning)
        
        if metrics.memory_gb > self.config.max_memory_gb:
            warning = f"Memory usage {metrics.memory_gb:.1f}GB exceeds limit {self.config.max_memory_gb}GB"
            warnings_added.append(warning)
        
        if metrics.gpu_memory_gb > self.config.max_gpu_memory_gb:
            warning = f"GPU memory {metrics.gpu_memory_gb:.1f}GB exceeds limit {self.config.max_gpu_memory_gb}GB"
            warnings_added.append(warning)
        
        for warning in warnings_added:
            logger.warning(warning)
            self.resource_warnings.append({
                'timestamp': time.time(),
                'warning': warning,
                'metrics': metrics.to_dict()
            })
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get current resource metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_resource_warnings(self) -> List[Dict[str, Any]]:
        """Get recent resource warnings."""
        with self._lock:
            return list(self.resource_warnings[-10:])  # Last 10 warnings


class CheckpointManager:
    """Manages optimization checkpoints for recovery."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self._ensure_checkpoint_dir()
        self.current_checkpoint_id = None
    
    def _ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    async def save_checkpoint(self, optimization_state: Dict[str, Any], 
                            checkpoint_id: Optional[str] = None) -> str:
        """Save optimization checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_id}.json"
        
        try:
            checkpoint_data = {
                'checkpoint_id': checkpoint_id,
                'timestamp': time.time(),
                'optimization_state': optimization_state,
                'config': self.config.__dict__
            }
            
            async with aiofiles.open(checkpoint_path, 'w') as f:
                await f.write(json.dumps(checkpoint_data, indent=2, default=str))
            
            self.current_checkpoint_id = checkpoint_id
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise OptimizationError(f"Checkpoint save failed: {e}")
    
    async def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load optimization checkpoint."""
        checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_id}.json"
        
        try:
            async with aiofiles.open(checkpoint_path, 'r') as f:
                content = await f.read()
                checkpoint_data = json.loads(content)
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint_data
            
        except FileNotFoundError:
            raise OptimizationError(f"Checkpoint not found: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise OptimizationError(f"Checkpoint load failed: {e}")
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        import os
        import glob
        
        pattern = f"{self.checkpoint_dir}/checkpoint_*.json"
        checkpoint_files = glob.glob(pattern)
        
        return [
            os.path.basename(f).replace('.json', '') 
            for f in checkpoint_files
        ]


class ConvergenceDetector:
    """Detects optimization convergence based on multiple criteria."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.score_history = deque(maxlen=config.convergence_window * 2)
        self.no_improvement_count = 0
        self.best_score = float('-inf')
        self.convergence_detected = False
    
    def update(self, score: float) -> bool:
        """Update with new score and check convergence."""
        self.score_history.append(score)
        
        # Update best score
        if score > self.best_score:
            improvement = score - self.best_score
            self.best_score = score
            
            if improvement >= self.config.min_improvement:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        else:
            self.no_improvement_count += 1
        
        # Check convergence criteria
        self.convergence_detected = self._check_convergence()
        return self.convergence_detected
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        # Patience-based convergence
        if self.no_improvement_count >= self.config.convergence_patience:
            logger.info(f"Convergence detected: no improvement for {self.no_improvement_count} trials")
            return True
        
        # Window-based variance convergence
        if len(self.score_history) >= self.config.convergence_window:
            recent_scores = list(self.score_history)[-self.config.convergence_window:]
            variance = np.var(recent_scores)
            
            if variance < self.config.min_improvement ** 2:
                logger.info(f"Convergence detected: low variance {variance:.6f}")
                return True
        
        return False
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence analysis metrics."""
        recent_scores = list(self.score_history)[-self.config.convergence_window:]
        
        return {
            'converged': self.convergence_detected,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'recent_variance': np.var(recent_scores) if recent_scores else 0.0,
            'recent_mean': np.mean(recent_scores) if recent_scores else 0.0,
            'score_history': list(self.score_history)
        }


class ValidationSystem:
    """Comprehensive validation system for optimization inputs and outputs."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_history = []
        
    def validate_model(self, model: nn.Module) -> None:
        """Validate model for optimization."""
        if model is None:
            raise ValidationError("Model cannot be None")
        
        if not isinstance(model, nn.Module):
            raise ValidationError("Model must be a PyTorch nn.Module")
        
        # Check model parameters
        param_count = sum(p.numel() for p in model.parameters())
        if param_count == 0:
            raise ValidationError("Model has no parameters")
        
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Check for frozen parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise ValidationError("Model has no trainable parameters")
            
            # Check model device
            devices = {p.device for p in model.parameters()}
            if len(devices) > 1:
                raise ValidationError(f"Model parameters on multiple devices: {devices}")
    
    def validate_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Validate hyperparameter values."""
        if not params:
            raise ValidationError("Hyperparameters cannot be empty")
        
        # Validate LoRA rank
        if 'r' in params:
            r = params['r']
            if not isinstance(r, int) or r < 1:
                raise ValidationError(f"LoRA rank 'r' must be positive integer, got {r}")
            if r > 1024:
                warnings.warn(f"Very high LoRA rank {r} may cause memory issues")
        
        # Validate LoRA alpha
        if 'lora_alpha' in params:
            alpha = params['lora_alpha']
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValidationError(f"LoRA alpha must be positive, got {alpha}")
        
        # Validate dropout
        if 'lora_dropout' in params:
            dropout = params['lora_dropout']
            if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
                raise ValidationError(f"LoRA dropout must be in [0, 1], got {dropout}")
        
        if self.validation_level == ValidationLevel.PARANOID:
            # Additional paranoid checks
            self._paranoid_hyperparameter_checks(params)
    
    def _paranoid_hyperparameter_checks(self, params: Dict[str, Any]) -> None:
        """Paranoid-level hyperparameter validation."""
        # Check for suspicious combinations
        if 'r' in params and 'lora_alpha' in params:
            r, alpha = params['r'], params['lora_alpha']
            if alpha > r * 4:
                warnings.warn(f"Very high alpha/rank ratio: {alpha}/{r} = {alpha/r:.2f}")
        
        # Check for extreme values
        if 'lora_dropout' in params and params['lora_dropout'] > 0.5:
            warnings.warn(f"High dropout {params['lora_dropout']} may impair learning")
    
    def validate_training_data(self, train_data: Any, eval_data: Any) -> None:
        """Validate training and evaluation data."""
        if train_data is None:
            raise ValidationError("Training data cannot be None")
        
        if eval_data is None and self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            raise ValidationError("Evaluation data required for strict validation")
        
        # Additional data validation based on type
        if hasattr(train_data, '__len__'):
            if len(train_data) == 0:
                raise ValidationError("Training data is empty")
        
        if eval_data is not None and hasattr(eval_data, '__len__'):
            if len(eval_data) == 0:
                raise ValidationError("Evaluation data is empty")
    
    def validate_optimization_result(self, result: OptimizationResult) -> None:
        """Validate optimization result."""
        if result is None:
            raise ValidationError("Optimization result cannot be None")
        
        if result.best_params is None:
            raise ValidationError("Best parameters cannot be None")
        
        if not isinstance(result.best_score, (int, float)):
            raise ValidationError("Best score must be numeric")
        
        if math.isnan(result.best_score) or math.isinf(result.best_score):
            if result.best_score != float('-inf'):  # -inf is acceptable for failed optimization
                raise ValidationError(f"Invalid best score: {result.best_score}")
        
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Validate optimization history
            if not result.optimization_history:
                warnings.warn("Empty optimization history")
            
            # Check for reasonable convergence
            if len(result.optimization_history) < 5:
                warnings.warn("Very few optimization trials completed")


class RobustLoRAOptimizationSystem:
    """Production-ready robust LoRA optimization system."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.state = OptimizationState.IDLE
        
        # Initialize subsystems
        self.validation_system = ValidationSystem(config.validation_level)
        self.resource_monitor = ResourceMonitor(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.convergence_detector = ConvergenceDetector(config)
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        
        # Circuit breaker for error handling
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=30,
            expected_exception=OptimizationError
        )
        
        # Optimization state
        self.optimization_start_time = None
        self.optimization_id = None
        self.current_trial = 0
        self.failed_trials = 0
        self._cancellation_requested = False
        self._optimization_lock = Lock()
        
        # Metrics collection
        self.metrics_collector = LocalMetricsCollector()
        
        logger.info("Robust LoRA optimization system initialized")
    
    async def optimize(self, train_data: Any, eval_data: Any, 
                      target_modules: Optional[List[str]] = None,
                      resume_from_checkpoint: Optional[str] = None) -> OptimizationResult:
        """Main optimization entry point with full robustness features."""
        
        with self._optimization_lock:
            if self.state not in [OptimizationState.IDLE, OptimizationState.FAILED]:
                raise OptimizationError(f"Optimization already running in state: {self.state}")
            
            self.state = OptimizationState.INITIALIZING
            self.optimization_id = f"opt_{uuid.uuid4().hex[:8]}"
            self.optimization_start_time = time.time()
            self._cancellation_requested = False
        
        try:
            logger.info(f"Starting robust optimization {self.optimization_id}")
            
            # Phase 1: Validation
            await self._validate_inputs(train_data, eval_data, target_modules)
            
            # Phase 2: Initialize monitoring and recovery
            self.resource_monitor.start_monitoring()
            
            # Phase 3: Resume from checkpoint if requested
            initial_state = None
            if resume_from_checkpoint:
                initial_state = await self._resume_from_checkpoint(resume_from_checkpoint)
            
            # Phase 4: Run optimization with monitoring
            result = await self._run_optimization_with_monitoring(
                train_data, eval_data, target_modules, initial_state
            )
            
            # Phase 5: Final validation and cleanup
            self.validation_system.validate_optimization_result(result)
            self.state = OptimizationState.CONVERGED
            
            logger.info(f"Optimization {self.optimization_id} completed successfully")
            return result
            
        except Exception as e:
            self.state = OptimizationState.FAILED
            await self._handle_optimization_failure(e)
            raise
        
        finally:
            self.resource_monitor.stop_monitoring()
            await self._cleanup_optimization()
    
    async def _validate_inputs(self, train_data: Any, eval_data: Any, 
                              target_modules: Optional[List[str]]) -> None:
        """Comprehensive input validation."""
        self.state = OptimizationState.INITIALIZING
        
        try:
            # Validate model
            self.validation_system.validate_model(self.model)
            
            # Validate data
            self.validation_system.validate_training_data(train_data, eval_data)
            
            # Validate target modules
            if target_modules:
                model_module_names = [name for name, _ in self.model.named_modules()]
                for target in target_modules:
                    if not any(target in name for name in model_module_names):
                        warnings.warn(f"Target module '{target}' not found in model")
            
            logger.info("Input validation completed successfully")
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(f"Input validation failed: {e}")
    
    async def _resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Resume optimization from checkpoint."""
        try:
            checkpoint_data = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
            
            # Validate checkpoint compatibility
            if checkpoint_data.get('config', {}).get('strategy') != self.config.strategy.value:
                warnings.warn("Checkpoint strategy differs from current configuration")
            
            optimization_state = checkpoint_data.get('optimization_state', {})
            self.current_trial = optimization_state.get('current_trial', 0)
            
            logger.info(f"Resumed from checkpoint {checkpoint_id} at trial {self.current_trial}")
            return optimization_state
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint {checkpoint_id}: {e}")
            # Don't fail completely - start fresh optimization
            warnings.warn(f"Checkpoint resume failed, starting fresh: {e}")
            return None
    
    @with_error_handling(
        retry_config=RetryConfig(max_retries=3, delay=1.0, exponential_backoff=True),
        circuit_breaker=None  # Will use instance circuit breaker
    )
    async def _run_optimization_with_monitoring(self, train_data: Any, eval_data: Any,
                                              target_modules: Optional[List[str]],
                                              initial_state: Optional[Dict[str, Any]]) -> OptimizationResult:
        """Run optimization with comprehensive monitoring."""
        
        # Initialize optimizer
        optimizer = NovelLoRAHyperparameterOptimizer(
            model=self.model,
            strategy=self.config.strategy,
            n_trials=self.config.n_trials,
            privacy_accountant=None  # Could be injected
        )
        
        self.state = OptimizationState.OPTIMIZING
        
        # Set up timeout
        timeout_seconds = self.config.max_duration_minutes * 60
        
        try:
            # Run optimization with timeout
            result = await asyncio.wait_for(
                self._monitored_optimization_loop(optimizer, train_data, eval_data, target_modules),
                timeout=timeout_seconds
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise OptimizationError(f"Optimization timeout after {self.config.max_duration_minutes} minutes")
        except Exception as e:
            if self._cancellation_requested:
                raise OptimizationError("Optimization cancelled by user")
            raise OptimizationError(f"Optimization failed: {e}")
    
    async def _monitored_optimization_loop(self, optimizer: NovelLoRAHyperparameterOptimizer,
                                         train_data: Any, eval_data: Any,
                                         target_modules: Optional[List[str]]) -> OptimizationResult:
        """Run optimization loop with monitoring and checkpointing."""
        
        # Override optimizer's evaluate method with monitoring
        original_evaluate = optimizer._evaluate_hyperparams
        optimizer._evaluate_hyperparams = self._monitored_evaluate_wrapper(original_evaluate)
        
        # Start optimization
        result = await optimizer.optimize(train_data, eval_data, target_modules)
        
        return result
    
    def _monitored_evaluate_wrapper(self, original_evaluate: Callable) -> Callable:
        """Wrapper for evaluation with monitoring."""
        
        async def wrapper(params: Dict[str, Any], train_data: Any, eval_data: Any) -> float:
            # Pre-evaluation checks
            self._check_cancellation()
            await self._check_resource_limits()
            self.validation_system.validate_hyperparameters(params)
            
            try:
                # Run original evaluation
                score = await original_evaluate(params, train_data, eval_data)
                
                # Post-evaluation processing
                self.current_trial += 1
                self.convergence_detector.update(score)
                
                # Update metrics
                self._update_optimization_metrics(params, score)
                
                # Checkpoint periodically
                if self.current_trial % self.config.checkpoint_interval == 0:
                    await self._save_optimization_checkpoint(params, score)
                
                # Check convergence
                if self.convergence_detector.convergence_detected:
                    logger.info("Early convergence detected")
                
                return score
                
            except Exception as e:
                self.failed_trials += 1
                logger.warning(f"Trial {self.current_trial} failed: {e}")
                
                # Circuit breaker logic
                if self.failed_trials > self.config.circuit_breaker_threshold:
                    raise OptimizationError(f"Too many failed trials: {self.failed_trials}")
                
                # Return very low score for failed trials
                return float('-inf')
        
        return wrapper
    
    def _check_cancellation(self) -> None:
        """Check if optimization cancellation was requested."""
        if self._cancellation_requested:
            raise OptimizationError("Optimization cancelled")
    
    async def _check_resource_limits(self) -> None:
        """Check if resource limits are exceeded."""
        metrics = self.resource_monitor.get_current_metrics()
        if not metrics:
            return
        
        if metrics.memory_gb > self.config.max_memory_gb:
            raise ResourceError(f"Memory limit exceeded: {metrics.memory_gb:.1f}GB > {self.config.max_memory_gb}GB")
        
        if metrics.cpu_percent > self.config.max_cpu_percent:
            warnings.warn(f"High CPU usage: {metrics.cpu_percent:.1f}%")
    
    def _update_optimization_metrics(self, params: Dict[str, Any], score: float) -> None:
        """Update optimization metrics."""
        metrics = {
            'optimization_id': self.optimization_id,
            'trial': self.current_trial,
            'score': score,
            'params': params,
            'timestamp': time.time(),
            'state': self.state.value
        }
        
        self.metrics_collector.record_metrics('optimization_trial', metrics)
    
    async def _save_optimization_checkpoint(self, current_params: Dict[str, Any], 
                                          current_score: float) -> None:
        """Save optimization checkpoint."""
        try:
            checkpoint_state = {
                'optimization_id': self.optimization_id,
                'current_trial': self.current_trial,
                'failed_trials': self.failed_trials,
                'current_params': current_params,
                'current_score': current_score,
                'convergence_metrics': self.convergence_detector.get_convergence_metrics(),
                'resource_warnings': self.resource_monitor.get_resource_warnings()
            }
            
            checkpoint_id = await self.checkpoint_manager.save_checkpoint(checkpoint_state)
            logger.info(f"Checkpoint saved at trial {self.current_trial}: {checkpoint_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            # Don't fail optimization for checkpoint errors
    
    async def _handle_optimization_failure(self, error: Exception) -> None:
        """Handle optimization failure with recovery attempts."""
        logger.error(f"Optimization {self.optimization_id} failed: {error}")
        
        # Save failure checkpoint
        try:
            failure_state = {
                'optimization_id': self.optimization_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc(),
                'current_trial': self.current_trial,
                'failed_trials': self.failed_trials,
                'resource_warnings': self.resource_monitor.get_resource_warnings()
            }
            
            await self.checkpoint_manager.save_checkpoint(
                failure_state, f"failure_{self.optimization_id}"
            )
        except Exception as e:
            logger.error(f"Failed to save failure checkpoint: {e}")
    
    async def _cleanup_optimization(self) -> None:
        """Cleanup optimization resources."""
        try:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Final metrics
            if self.optimization_start_time:
                duration = time.time() - self.optimization_start_time
                self.metrics_collector.record_metrics('optimization_complete', {
                    'optimization_id': self.optimization_id,
                    'duration_seconds': duration,
                    'trials_completed': self.current_trial,
                    'trials_failed': self.failed_trials,
                    'final_state': self.state.value
                })
            
            logger.info(f"Optimization {self.optimization_id} cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def cancel_optimization(self) -> None:
        """Cancel running optimization."""
        if self.state in [OptimizationState.OPTIMIZING, OptimizationState.EVALUATING]:
            self._cancellation_requested = True
            self.state = OptimizationState.CANCELLED
            logger.info(f"Cancellation requested for optimization {self.optimization_id}")
        else:
            logger.warning(f"Cannot cancel optimization in state {self.state}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        metrics = self.resource_monitor.get_current_metrics()
        
        status = {
            'optimization_id': self.optimization_id,
            'state': self.state.value,
            'current_trial': self.current_trial,
            'total_trials': self.config.n_trials,
            'failed_trials': self.failed_trials,
            'progress_percent': (self.current_trial / self.config.n_trials) * 100,
            'resource_metrics': metrics.to_dict() if metrics else None,
            'convergence_metrics': self.convergence_detector.get_convergence_metrics(),
            'resource_warnings': self.resource_monitor.get_resource_warnings()
        }
        
        if self.optimization_start_time:
            elapsed = time.time() - self.optimization_start_time
            estimated_total = elapsed / max(self.current_trial, 1) * self.config.n_trials
            status.update({
                'time_elapsed_seconds': elapsed,
                'estimated_time_remaining_seconds': max(0, estimated_total - elapsed)
            })
        
        return status


# Factory function for easy instantiation
def create_robust_lora_optimizer(model: nn.Module, 
                                config: Optional[OptimizationConfig] = None) -> RobustLoRAOptimizationSystem:
    """Create a robust LoRA optimization system."""
    if config is None:
        config = OptimizationConfig()
    
    return RobustLoRAOptimizationSystem(model=model, config=config)
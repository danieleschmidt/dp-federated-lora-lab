"""
Concurrent processing utilities for DP-Federated LoRA.

This module provides high-performance concurrent processing capabilities
for parallel training, aggregation, and distributed computation.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
import multiprocessing as mp
from queue import Queue, Empty
import pickle
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .performance import performance_monitor, resource_manager
from .exceptions import ResourceError, TrainingError, create_error_with_context, ErrorContext

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """Task for worker execution."""
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkerResult:
    """Result from worker execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    worker_id: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WorkerPool(ABC):
    """Abstract base class for worker pools."""
    
    @abstractmethod
    async def submit(self, task: WorkerTask) -> str:
        """Submit task for execution."""
        pass
    
    @abstractmethod
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> WorkerResult:
        """Get result for task."""
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        pass


class ThreadWorkerPool(WorkerPool):
    """High-performance thread-based worker pool."""
    
    def __init__(self, max_workers: Optional[int] = None, queue_size: int = 1000):
        self.max_workers = max_workers or resource_manager.get_optimal_worker_count("io")
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_queue: Dict[str, concurrent.futures.Future] = {}
        self.results: Dict[str, WorkerResult] = {}
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0
        }
        self._lock = threading.Lock()
    
    async def submit(self, task: WorkerTask) -> str:
        """Submit task for execution."""
        if task.task_id in self.task_queue:
            raise ValueError(f"Task {task.task_id} already submitted")
        
        with self._lock:
            self.stats["tasks_submitted"] += 1
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_task, task)
        self.task_queue[task.task_id] = future
        
        # Set up completion callback
        future.add_done_callback(lambda f: self._handle_completion(task.task_id, f))
        
        return task.task_id
    
    def _execute_task(self, task: WorkerTask) -> WorkerResult:
        """Execute task in worker thread."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            if task.timeout:
                # Use asyncio for timeout handling
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            self._run_with_timeout(task.function, task.args, task.kwargs),
                            timeout=task.timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                result = task.function(*task.args, **task.kwargs)
            
            duration = time.time() - start_time
            
            return WorkerResult(
                task_id=task.task_id,
                success=True,
                result=result,
                duration=duration,
                worker_id=worker_id,
                metadata={"execution_thread": worker_id}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WorkerResult(
                task_id=task.task_id,
                success=False,
                error=e,
                duration=duration,
                worker_id=worker_id,
                metadata={"execution_thread": worker_id, "error_type": type(e).__name__}
            )
    
    async def _run_with_timeout(self, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Run function with timeout using asyncio."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _handle_completion(self, task_id: str, future: concurrent.futures.Future) -> None:
        """Handle task completion."""
        try:
            result = future.result()
            self.results[task_id] = result
            
            with self._lock:
                if result.success:
                    self.stats["tasks_completed"] += 1
                else:
                    self.stats["tasks_failed"] += 1
                
                self.stats["total_execution_time"] += result.duration
                total_tasks = self.stats["tasks_completed"] + self.stats["tasks_failed"]
                if total_tasks > 0:
                    self.stats["avg_execution_time"] = (
                        self.stats["total_execution_time"] / total_tasks
                    )
        
        except Exception as e:
            logger.error(f"Error handling completion for task {task_id}: {e}")
            self.results[task_id] = WorkerResult(
                task_id=task_id,
                success=False,
                error=e,
                duration=0.0
            )
        
        # Cleanup
        self.task_queue.pop(task_id, None)
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> WorkerResult:
        """Get result for task."""
        if task_id in self.results:
            return self.results[task_id]
        
        if task_id not in self.task_queue:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.task_queue[task_id]
        
        try:
            # Wait for completion
            if timeout:
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=timeout
                )
            else:
                await asyncio.wrap_future(future)
            
            return self.results.get(task_id, WorkerResult(
                task_id=task_id,
                success=False,
                error=Exception("Result not found after completion")
            ))
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            return {
                **self.stats,
                "max_workers": self.max_workers,
                "active_tasks": len(self.task_queue),
                "completed_results": len(self.results),
                "success_rate": (
                    self.stats["tasks_completed"] / 
                    max(1, self.stats["tasks_completed"] + self.stats["tasks_failed"])
                )
            }


class ProcessWorkerPool(WorkerPool):
    """Process-based worker pool for CPU-intensive tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or resource_manager.get_optimal_worker_count("cpu")
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue: Dict[str, concurrent.futures.Future] = {}
        self.results: Dict[str, WorkerResult] = {}
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }
        self._lock = threading.Lock()
    
    async def submit(self, task: WorkerTask) -> str:
        """Submit task for execution."""
        # Serialize task for process communication
        try:
            serialized_task = pickle.dumps(task)
            if len(serialized_task) > 10 * 1024 * 1024:  # 10MB limit
                raise ResourceError(
                    "Task too large for process communication",
                    {"task_size_mb": len(serialized_task) / 1024 / 1024}
                )
        except pickle.PicklingError as e:
            raise ResourceError(f"Task not serializable: {e}")
        
        with self._lock:
            self.stats["tasks_submitted"] += 1
        
        # Submit to process pool
        future = self.executor.submit(self._execute_task_process, task)
        self.task_queue[task.task_id] = future
        
        future.add_done_callback(lambda f: self._handle_completion(task.task_id, f))
        
        return task.task_id
    
    @staticmethod
    def _execute_task_process(task: WorkerTask) -> WorkerResult:
        """Execute task in worker process."""
        start_time = time.time()
        worker_id = f"process_{mp.current_process().pid}"
        
        try:
            result = task.function(*task.args, **task.kwargs)
            duration = time.time() - start_time
            
            return WorkerResult(
                task_id=task.task_id,
                success=True,
                result=result,
                duration=duration,
                worker_id=worker_id,
                metadata={"execution_process": mp.current_process().pid}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WorkerResult(
                task_id=task.task_id,
                success=False,
                error=e,
                duration=duration,
                worker_id=worker_id,
                metadata={"execution_process": mp.current_process().pid, "error_type": type(e).__name__}
            )
    
    def _handle_completion(self, task_id: str, future: concurrent.futures.Future) -> None:
        """Handle task completion."""
        try:
            result = future.result()
            self.results[task_id] = result
            
            with self._lock:
                if result.success:
                    self.stats["tasks_completed"] += 1
                else:
                    self.stats["tasks_failed"] += 1
                
                self.stats["total_execution_time"] += result.duration
        
        except Exception as e:
            logger.error(f"Error handling completion for task {task_id}: {e}")
            self.results[task_id] = WorkerResult(
                task_id=task_id,
                success=False,
                error=e,
                duration=0.0
            )
        
        # Cleanup
        self.task_queue.pop(task_id, None)
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> WorkerResult:
        """Get result for task."""
        if task_id in self.results:
            return self.results[task_id]
        
        if task_id not in self.task_queue:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.task_queue[task_id]
        
        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=timeout
                )
            else:
                await asyncio.wrap_future(future)
            
            return self.results.get(task_id, WorkerResult(
                task_id=task_id,
                success=False,
                error=Exception("Result not found after completion")
            ))
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            return {
                **self.stats,
                "max_workers": self.max_workers,
                "active_tasks": len(self.task_queue),
                "completed_results": len(self.results),
                "success_rate": (
                    self.stats["tasks_completed"] / 
                    max(1, self.stats["tasks_completed"] + self.stats["tasks_failed"])
                )
            }


class ConcurrentModelTrainer:
    """Concurrent model training with multiple workers."""
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        device_per_worker: bool = True
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.device_per_worker = device_per_worker
        
        # Initialize worker pool
        if use_processes:
            self.worker_pool = ProcessWorkerPool(max_workers)
        else:
            self.worker_pool = ThreadWorkerPool(max_workers)
        
        # GPU management
        self.available_devices = self._get_available_devices()
        self.device_assignments: Dict[str, str] = {}
        
        logger.info(f"Initialized concurrent trainer with {max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}, "
                   f"{len(self.available_devices)} devices available")
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        return devices
    
    def _assign_device(self, worker_id: str) -> str:
        """Assign device to worker."""
        if not self.device_per_worker or len(self.available_devices) <= 1:
            return "cpu"
        
        if worker_id not in self.device_assignments:
            # Round-robin device assignment
            device_index = len(self.device_assignments) % len(self.available_devices)
            self.device_assignments[worker_id] = self.available_devices[device_index]
        
        return self.device_assignments[worker_id]
    
    @performance_monitor.monitor_operation("concurrent_training")
    async def train_concurrent(
        self,
        training_tasks: List[Dict[str, Any]],
        training_function: Callable,
        timeout_per_task: Optional[float] = None
    ) -> List[WorkerResult]:
        """
        Train multiple models/clients concurrently.
        
        Args:
            training_tasks: List of training task configurations
            training_function: Function to execute for each task
            timeout_per_task: Timeout for each training task
            
        Returns:
            List of training results
        """
        # Create worker tasks
        worker_tasks = []
        for i, task_config in enumerate(training_tasks):
            task_id = f"training_task_{i}_{uuid.uuid4().hex[:8]}"
            device = self._assign_device(task_id)
            
            # Add device to task config
            task_config_with_device = {**task_config, "device": device}
            
            worker_task = WorkerTask(
                task_id=task_id,
                function=training_function,
                args=(task_config_with_device,),
                kwargs={},
                timeout=timeout_per_task,
                metadata={"device": device, "task_index": i}
            )
            worker_tasks.append(worker_task)
        
        # Submit all tasks
        task_ids = []
        for task in worker_tasks:
            task_id = await self.worker_pool.submit(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} concurrent training tasks")
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = await self.worker_pool.get_result(
                    task_id, 
                    timeout=timeout_per_task
                )
                results.append(result)
                
                if result.success:
                    logger.debug(f"Task {task_id} completed successfully in {result.duration:.2f}s")
                else:
                    logger.error(f"Task {task_id} failed: {result.error}")
                    
            except Exception as e:
                error_result = WorkerResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    duration=0.0
                )
                results.append(error_result)
                logger.error(f"Failed to get result for task {task_id}: {e}")
        
        # Log summary
        successful_tasks = sum(1 for r in results if r.success)
        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / len(results) if results else 0
        
        logger.info(f"Concurrent training completed: {successful_tasks}/{len(results)} "
                   f"successful, avg duration: {avg_duration:.2f}s")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "worker_pool_stats": self.worker_pool.get_stats(),
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "available_devices": self.available_devices,
            "device_assignments": self.device_assignments
        }
    
    def shutdown(self) -> None:
        """Shutdown concurrent trainer."""
        self.worker_pool.shutdown()


class DistributedTrainingManager:
    """Manager for distributed training across multiple nodes."""
    
    def __init__(self, world_size: int, rank: int, backend: str = "nccl"):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.is_initialized = False
        
        # Initialize distributed training
        self._initialize_distributed()
    
    def _initialize_distributed(self) -> None:
        """Initialize distributed training."""
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    world_size=self.world_size,
                    rank=self.rank
                )
                self.is_initialized = True
                logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
            else:
                self.is_initialized = True
                logger.info("Distributed training already initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise TrainingError(f"Distributed initialization failed: {e}")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if not self.is_initialized:
            raise TrainingError("Distributed training not initialized")
        
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.is_available() and self.backend == "nccl":
            model = DDP(model, device_ids=[self.rank])
        else:
            model = DDP(model)
        
        return model
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """Perform all-reduce operation on tensor."""
        if not self.is_initialized:
            raise TrainingError("Distributed training not initialized")
        
        if op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        elif op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")
        
        return tensor
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False


class ParallelAggregator:
    """Parallel model parameter aggregation."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.worker_pool = ThreadWorkerPool(max_workers)
    
    @performance_monitor.monitor_operation("parallel_aggregation")
    async def aggregate_parallel(
        self,
        parameter_updates: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        aggregation_method: str = "weighted_average"
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate parameter updates in parallel.
        
        Args:
            parameter_updates: List of parameter update dictionaries
            weights: Optional weights for each update
            aggregation_method: Aggregation method to use
            
        Returns:
            Aggregated parameters
        """
        if not parameter_updates:
            return {}
        
        if weights is None:
            weights = [1.0] * len(parameter_updates)
        
        if len(weights) != len(parameter_updates):
            raise ValueError("Number of weights must match number of updates")
        
        # Get parameter names
        param_names = list(parameter_updates[0].keys())
        
        # Create aggregation tasks for each parameter
        aggregation_tasks = []
        for param_name in param_names:
            task_id = f"aggregate_{param_name}_{uuid.uuid4().hex[:8]}"
            
            # Extract parameter tensors for this name
            param_tensors = [update[param_name] for update in parameter_updates]
            
            task = WorkerTask(
                task_id=task_id,
                function=self._aggregate_parameter,
                args=(param_tensors, weights, aggregation_method),
                kwargs={},
                metadata={"parameter_name": param_name}
            )
            aggregation_tasks.append((param_name, task))
        
        # Submit all tasks
        task_submissions = []
        for param_name, task in aggregation_tasks:
            task_id = await self.worker_pool.submit(task)
            task_submissions.append((param_name, task_id))
        
        # Collect results
        aggregated_params = {}
        for param_name, task_id in task_submissions:
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                aggregated_params[param_name] = result.result
            else:
                logger.error(f"Failed to aggregate parameter {param_name}: {result.error}")
                # Fallback to sequential aggregation
                param_tensors = [update[param_name] for update in parameter_updates]
                aggregated_params[param_name] = self._aggregate_parameter(
                    param_tensors, weights, aggregation_method
                )
        
        logger.info(f"Parallel aggregation completed for {len(aggregated_params)} parameters")
        return aggregated_params
    
    @staticmethod
    def _aggregate_parameter(
        param_tensors: List[torch.Tensor],
        weights: List[float],
        method: str
    ) -> torch.Tensor:
        """Aggregate a single parameter across clients."""
        if method == "weighted_average":
            # Weighted average
            weighted_sum = torch.zeros_like(param_tensors[0])
            total_weight = sum(weights)
            
            for tensor, weight in zip(param_tensors, weights):
                weighted_sum += tensor * weight
            
            return weighted_sum / total_weight
        
        elif method == "fedavg":
            # FedAvg (simple average)
            stacked = torch.stack(param_tensors)
            return torch.mean(stacked, dim=0)
        
        elif method == "median":
            # Coordinate-wise median (Byzantine-robust)
            stacked = torch.stack(param_tensors)
            return torch.median(stacked, dim=0)[0]
        
        elif method == "trimmed_mean":
            # Trimmed mean (remove top/bottom 10%)
            stacked = torch.stack(param_tensors)
            sorted_tensors = torch.sort(stacked, dim=0)[0]
            
            # Remove top and bottom 10%
            trim_count = max(1, len(param_tensors) // 10)
            trimmed = sorted_tensors[trim_count:-trim_count]
            
            return torch.mean(trimmed, dim=0)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return self.worker_pool.get_stats()
    
    def shutdown(self) -> None:
        """Shutdown parallel aggregator."""
        self.worker_pool.shutdown()


# Global instances
thread_pool = ThreadWorkerPool()
process_pool = ProcessWorkerPool()
concurrent_trainer = ConcurrentModelTrainer()
parallel_aggregator = ParallelAggregator()


def cleanup_concurrent_resources() -> None:
    """Cleanup all concurrent processing resources."""
    try:
        thread_pool.shutdown()
        process_pool.shutdown()
        concurrent_trainer.shutdown()
        parallel_aggregator.shutdown()
        resource_manager.cleanup()
    except Exception as e:
        logger.error(f"Error cleaning up concurrent resources: {e}")


# Register cleanup function
import atexit
atexit.register(cleanup_concurrent_resources)
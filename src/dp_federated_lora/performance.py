"""
Performance optimization and scaling utilities for DP-Federated LoRA.

This module provides performance monitoring, optimization, caching,
and scaling utilities for high-performance federated learning.
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
import weakref
import pickle
import hashlib
import gc
from functools import wraps, lru_cache
from collections import defaultdict, deque
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation: str
    duration: float
    memory_used: float
    cpu_percent: float
    network_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time performance monitoring and profiling."""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.total_operations = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # System monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        
        # Performance thresholds
        self.thresholds = {
            "max_duration": 30.0,  # seconds
            "max_memory_mb": 1000,  # MB
            "max_cpu_percent": 80.0,  # %
        }
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        if not self.enable_profiling:
            return
        
        with self._lock:
            self.metrics[metric.operation].append(metric)
            self.operation_counts[metric.operation] += 1
            self.total_operations += 1
            
            # Keep only recent metrics to prevent memory growth
            if len(self.metrics[metric.operation]) > 1000:
                self.metrics[metric.operation] = self.metrics[metric.operation][-1000:]
        
        # Check thresholds and log warnings
        self._check_thresholds(metric)
    
    def _check_thresholds(self, metric: PerformanceMetrics) -> None:
        """Check if metric exceeds performance thresholds."""
        if metric.duration > self.thresholds["max_duration"]:
            logger.warning(
                f"Operation {metric.operation} took {metric.duration:.2f}s "
                f"(threshold: {self.thresholds['max_duration']:.2f}s)"
            )
        
        if metric.memory_used > self.thresholds["max_memory_mb"]:
            logger.warning(
                f"Operation {metric.operation} used {metric.memory_used:.1f}MB "
                f"(threshold: {self.thresholds['max_memory_mb']:.1f}MB)"
            )
        
        if metric.cpu_percent > self.thresholds["max_cpu_percent"]:
            logger.warning(
                f"Operation {metric.operation} used {metric.cpu_percent:.1f}% CPU "
                f"(threshold: {self.thresholds['max_cpu_percent']:.1f}%)"
            )
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if operation:
                metrics = self.metrics.get(operation, [])
                if not metrics:
                    return {"error": f"No metrics for operation '{operation}'"}
                
                durations = [m.duration for m in metrics]
                memory_usage = [m.memory_used for m in metrics]
                
                return {
                    "operation": operation,
                    "count": len(metrics),
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                    "total_cache_hits": sum(m.cache_hits for m in metrics),
                    "total_cache_misses": sum(m.cache_misses for m in metrics),
                }
            else:
                # Overall stats
                total_duration = sum(
                    sum(m.duration for m in metrics_list)
                    for metrics_list in self.metrics.values()
                )
                
                return {
                    "total_operations": self.total_operations,
                    "uptime": time.time() - self.start_time,
                    "avg_duration_per_op": total_duration / self.total_operations if self.total_operations > 0 else 0,
                    "operations_per_second": self.total_operations / (time.time() - self.start_time),
                    "memory_growth_mb": (self.process.memory_info().rss - self.initial_memory) / 1024 / 1024,
                    "operation_counts": dict(self.operation_counts),
                }
    
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self.process.memory_info().rss
                start_cpu = self.process.cpu_percent()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    end_memory = self.process.memory_info().rss
                    memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
                    cpu_percent = self.process.cpu_percent()
                    
                    metric = PerformanceMetrics(
                        operation=operation_name,
                        duration=duration,
                        memory_used=memory_used,
                        cpu_percent=cpu_percent,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    )
                    self.record_metric(metric)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self.process.memory_info().rss
                start_cpu = self.process.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    end_memory = self.process.memory_info().rss
                    memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
                    cpu_percent = self.process.cpu_percent()
                    
                    metric = PerformanceMetrics(
                        operation=operation_name,
                        duration=duration,
                        memory_used=memory_used,
                        cpu_percent=cpu_percent,
                        metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                    )
                    self.record_metric(metric)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class CacheManager:
    """High-performance caching system with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_order: deque = deque()
        self._lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.default_ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                del self._timestamps[lru_key]
                self.evictions += 1
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.default_ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()
            
            if key in self._cache and not self._is_expired(key):
                # Move to end (most recently used)
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                
                self.hits += 1
                return self._cache[key]
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Evict if necessary
            self._evict_lru()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Update access order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "memory_usage_mb": self._estimate_memory_usage() / 1024 / 1024
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache."""
        try:
            # Rough estimation using pickle size
            total_size = 0
            for key, value in list(self._cache.items())[:10]:  # Sample first 10 items
                total_size += len(pickle.dumps(key)) + len(pickle.dumps(value))
            
            # Extrapolate to full cache
            if len(self._cache) > 0:
                avg_item_size = total_size / min(10, len(self._cache))
                return int(avg_item_size * len(self._cache))
            return 0
        except Exception:
            return 0


class ConnectionPool:
    """High-performance connection pooling for network clients."""
    
    def __init__(self, max_connections: int = 50, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._pools: Dict[str, List[Any]] = {}
        self._pool_sizes: Dict[str, int] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._last_cleanup = time.time()
        
        # Stats
        self.connections_created = 0
        self.connections_reused = 0
        self.connections_expired = 0
    
    def get_pool_key(self, host: str, port: int, **kwargs) -> str:
        """Generate pool key for connection."""
        key_data = f"{host}:{port}:{hash(tuple(sorted(kwargs.items())))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_connection(self, host: str, port: int, factory: Callable, **kwargs) -> Any:
        """Get connection from pool or create new one."""
        pool_key = self.get_pool_key(host, port, **kwargs)
        
        # Initialize pool if needed
        if pool_key not in self._pools:
            self._pools[pool_key] = []
            self._pool_sizes[pool_key] = 0
            self._locks[pool_key] = threading.Lock()
        
        with self._locks[pool_key]:
            # Try to get existing connection
            while self._pools[pool_key]:
                connection = self._pools[pool_key].pop()
                if self._is_connection_valid(connection):
                    self.connections_reused += 1
                    return connection
                else:
                    self.connections_expired += 1
                    self._pool_sizes[pool_key] -= 1
            
            # Create new connection
            connection = factory(host=host, port=port, **kwargs)
            self.connections_created += 1
            return connection
    
    def return_connection(self, connection: Any, host: str, port: int, **kwargs) -> None:
        """Return connection to pool."""
        pool_key = self.get_pool_key(host, port, **kwargs)
        
        if pool_key in self._locks:
            with self._locks[pool_key]:
                if (self._pool_sizes[pool_key] < self.max_connections and 
                    self._is_connection_valid(connection)):
                    self._pools[pool_key].append(connection)
                    self._pool_sizes[pool_key] += 1
                else:
                    # Pool full or connection invalid, close it
                    self._close_connection(connection)
    
    def _is_connection_valid(self, connection: Any) -> bool:
        """Check if connection is still valid."""
        try:
            # This would be implementation-specific
            # For now, assume all connections are valid
            return hasattr(connection, 'is_closed') and not connection.is_closed
        except Exception:
            return False
    
    def _close_connection(self, connection: Any) -> None:
        """Close connection properly."""
        try:
            if hasattr(connection, 'close'):
                connection.close()
        except Exception:
            pass
    
    def cleanup_expired(self) -> None:
        """Clean up expired connections."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return
        
        self._last_cleanup = current_time
        
        for pool_key in list(self._pools.keys()):
            with self._locks[pool_key]:
                valid_connections = []
                for connection in self._pools[pool_key]:
                    if self._is_connection_valid(connection):
                        valid_connections.append(connection)
                    else:
                        self._close_connection(connection)
                        self.connections_expired += 1
                        self._pool_sizes[pool_key] -= 1
                
                self._pools[pool_key] = valid_connections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        total_pools = len(self._pools)
        total_connections = sum(self._pool_sizes.values())
        
        return {
            "total_pools": total_pools,
            "total_connections": total_connections,
            "max_connections_per_pool": self.max_connections,
            "connections_created": self.connections_created,
            "connections_reused": self.connections_reused,
            "connections_expired": self.connections_expired,
            "reuse_rate": (
                self.connections_reused / 
                (self.connections_created + self.connections_reused)
                if (self.connections_created + self.connections_reused) > 0 else 0
            )
        }


class BatchProcessor:
    """High-performance batch processing with adaptive batching."""
    
    def __init__(
        self, 
        batch_size: int = 32,
        max_wait_time: float = 1.0,
        max_queue_size: int = 1000
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._processing = False
        self._batch_stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "avg_batch_size": 0,
            "avg_processing_time": 0
        }
    
    async def add_item(self, item: Any) -> None:
        """Add item to processing queue."""
        try:
            await asyncio.wait_for(
                self._queue.put(item), 
                timeout=5.0
            )
        except asyncio.TimeoutError:
            raise Exception("Queue is full, cannot add more items")
    
    async def start_processing(self, processor: Callable[[List[Any]], Any]) -> None:
        """Start batch processing."""
        if self._processing:
            return
        
        self._processing = True
        
        try:
            while self._processing or not self._queue.empty():
                batch = await self._collect_batch()
                
                if batch:
                    start_time = time.time()
                    
                    try:
                        await processor(batch)
                        
                        # Update stats
                        processing_time = time.time() - start_time
                        self._update_stats(len(batch), processing_time)
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        # Could implement retry logic here
                
                else:
                    # No items in queue, wait a bit
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Batch processor error: {e}")
        finally:
            self._processing = False
    
    async def _collect_batch(self) -> List[Any]:
        """Collect items for batch processing."""
        batch = []
        start_time = time.time()
        
        # Collect items until batch size or timeout
        while (len(batch) < self.batch_size and 
               time.time() - start_time < self.max_wait_time):
            
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=max(0.1, self.max_wait_time - (time.time() - start_time))
                )
                batch.append(item)
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                break
        
        return batch
    
    def _update_stats(self, batch_size: int, processing_time: float) -> None:
        """Update processing statistics."""
        self._batch_stats["batches_processed"] += 1
        self._batch_stats["items_processed"] += batch_size
        
        # Update averages
        total_batches = self._batch_stats["batches_processed"]
        self._batch_stats["avg_batch_size"] = (
            self._batch_stats["items_processed"] / total_batches
        )
        
        # Exponential moving average for processing time
        alpha = 0.1
        if self._batch_stats["avg_processing_time"] == 0:
            self._batch_stats["avg_processing_time"] = processing_time
        else:
            self._batch_stats["avg_processing_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self._batch_stats["avg_processing_time"]
            )
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        self._processing = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self._batch_stats)


class ResourceManager:
    """Adaptive resource management for optimal performance."""
    
    def __init__(self):
        self.cpu_count = os.cpu_count() or 4
        self.memory_total = psutil.virtual_memory().total
        self.process = psutil.Process()
        
        # Resource limits
        self.max_memory_usage = 0.8  # 80% of available memory
        self.max_cpu_usage = 0.9  # 90% of available CPU
        
        # Thread/Process pools
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Monitoring
        self._resource_stats = {
            "peak_memory_mb": 0,
            "peak_cpu_percent": 0,
            "gc_collections": 0,
            "thread_pool_tasks": 0,
            "process_pool_tasks": 0
        }
    
    def get_optimal_worker_count(self, task_type: str = "io") -> int:
        """Get optimal worker count for task type."""
        if task_type == "cpu":
            return max(1, int(self.cpu_count * 0.8))
        elif task_type == "io":
            return max(4, int(self.cpu_count * 2))
        else:
            return self.cpu_count
    
    def get_thread_pool(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get thread pool executor."""
        if self._thread_pool is None or self._thread_pool._shutdown:
            workers = max_workers or self.get_optimal_worker_count("io")
            self._thread_pool = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="dp-federated-lora"
            )
        return self._thread_pool
    
    def get_process_pool(self, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        """Get process pool executor."""
        if self._process_pool is None or self._process_pool._shutdown:
            workers = max_workers or self.get_optimal_worker_count("cpu")
            self._process_pool = ProcessPoolExecutor(max_workers=workers)
        return self._process_pool
    
    def check_resource_limits(self) -> Dict[str, Any]:
        """Check current resource usage against limits."""
        memory_info = self.process.memory_info()
        memory_percent = memory_info.rss / self.memory_total
        cpu_percent = self.process.cpu_percent()
        
        # Update peak stats
        memory_mb = memory_info.rss / 1024 / 1024
        self._resource_stats["peak_memory_mb"] = max(
            self._resource_stats["peak_memory_mb"], memory_mb
        )
        self._resource_stats["peak_cpu_percent"] = max(
            self._resource_stats["peak_cpu_percent"], cpu_percent
        )
        
        status = {
            "memory_mb": memory_mb,
            "memory_percent": memory_percent * 100,
            "cpu_percent": cpu_percent,
            "memory_warning": memory_percent > self.max_memory_usage,
            "cpu_warning": cpu_percent > self.max_cpu_usage * 100,
            "should_gc": memory_percent > 0.7  # Trigger GC at 70%
        }
        
        # Trigger garbage collection if needed
        if status["should_gc"]:
            self.force_gc()
        
        return status
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())
        
        self._resource_stats["gc_collections"] += 1
        
        logger.info(f"Garbage collection: {collected} objects collected, "
                   f"{before_objects - after_objects} objects freed")
        
        return {
            "collected": collected,
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_freed": before_objects - after_objects
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource management statistics."""
        return {
            **self._resource_stats,
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total / 1024 / 1024 / 1024,
            "current_resource_status": self.check_resource_limits()
        }


# Global instances
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
connection_pool = ConnectionPool()
resource_manager = ResourceManager()


def optimize_for_scale(
    cache_size: int = 2000,
    max_connections: int = 100,
    enable_profiling: bool = True
) -> None:
    """
    Configure system for high-scale operation.
    
    Args:
        cache_size: Maximum cache size
        max_connections: Maximum connections per pool
        enable_profiling: Enable performance profiling
    """
    global cache_manager, connection_pool, performance_monitor
    
    # Reconfigure components for scale
    cache_manager = CacheManager(max_size=cache_size)
    connection_pool = ConnectionPool(max_connections=max_connections)
    performance_monitor = PerformanceMonitor(enable_profiling=enable_profiling)
    
    # Configure garbage collection for better performance
    gc.set_threshold(700, 10, 10)  # More aggressive GC for long-running processes
    
    logger.info(f"System optimized for scale: cache_size={cache_size}, "
               f"max_connections={max_connections}, profiling={enable_profiling}")


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return {
        "performance": performance_monitor.get_stats(),
        "cache": cache_manager.get_stats(),
        "connections": connection_pool.get_stats(),
        "resources": resource_manager.get_stats(),
        "timestamp": time.time(),
        "uptime": time.time() - performance_monitor.start_time
    }
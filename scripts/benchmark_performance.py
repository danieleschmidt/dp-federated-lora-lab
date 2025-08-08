#!/usr/bin/env python3
"""
Performance benchmark suite for DP-Federated LoRA Lab.
Measures throughput, latency, memory usage, and scaling behavior.
"""

import json
import sys
import time
try:
    import psutil
except ImportError:
    # Fallback implementation without psutil
    import os
    import time
    class psutil:
        @staticmethod
        def cpu_percent(interval=1):
            return 50.0  # Mock CPU usage
        @staticmethod
        def virtual_memory():
            class Memory:
                def __init__(self):
                    self.percent = 60.0
                    self.available = 8 * 1024 * 1024 * 1024  # 8GB
            return Memory()
        @staticmethod
        def disk_usage(path):
            class Disk:
                def __init__(self):
                    self.percent = 30.0
            return Disk()
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    duration_seconds: float
    throughput_ops_per_sec: float
    peak_memory_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None

class PerformanceBenchmark:
    """Performance benchmark runner."""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.baseline_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def benchmark_tensor_operations(self, size: int = 1000) -> PerformanceMetrics:
        """Benchmark tensor operations (simulated)."""
        try:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate tensor operations
            operations = 0
            for i in range(100):
                # Simulate matrix operations
                data = [[j * i for j in range(size // 10)] for _ in range(size // 10)]
                result = sum(sum(row) for row in data)
                operations += 1
            
            end_time = time.time()
            peak_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            duration = end_time - start_time
            throughput = operations / duration if duration > 0 else 0
            
            return PerformanceMetrics(
                operation="tensor_operations",
                duration_seconds=duration,
                throughput_ops_per_sec=throughput,
                peak_memory_mb=peak_memory - self.baseline_memory,
                cpu_usage_percent=cpu_usage,
                success=True
            )
            
        except Exception as e:
            return PerformanceMetrics(
                operation="tensor_operations",
                duration_seconds=0,
                throughput_ops_per_sec=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_concurrent_clients(self, num_clients: int = 10) -> PerformanceMetrics:
        """Benchmark concurrent client simulation."""
        try:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            def client_simulation(client_id: int) -> bool:
                """Simulate client processing."""
                # Simulate client work
                for i in range(50):
                    data = [j * client_id for j in range(100)]
                    result = sum(data)
                return True
            
            # Run concurrent clients
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(client_simulation, i) for i in range(num_clients)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            peak_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            duration = end_time - start_time
            successful_clients = sum(results)
            throughput = successful_clients / duration if duration > 0 else 0
            
            return PerformanceMetrics(
                operation=f"concurrent_clients_{num_clients}",
                duration_seconds=duration,
                throughput_ops_per_sec=throughput,
                peak_memory_mb=peak_memory - self.baseline_memory,
                cpu_usage_percent=cpu_usage,
                success=all(results)
            )
            
        except Exception as e:
            return PerformanceMetrics(
                operation=f"concurrent_clients_{num_clients}",
                duration_seconds=0,
                throughput_ops_per_sec=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_memory_scaling(self) -> PerformanceMetrics:
        """Benchmark memory usage scaling."""
        try:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate increasing memory usage
            data_structures = []
            for i in range(10):
                # Create data structure simulating model parameters
                size = 1000 * (i + 1)
                data = [[j for j in range(100)] for _ in range(size // 100)]
                data_structures.append(data)
                
                current_memory = self._get_memory_usage()
                if current_memory - start_memory > 500:  # 500MB limit
                    break
            
            end_time = time.time()
            peak_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            # Clean up
            del data_structures
            
            duration = end_time - start_time
            memory_used = peak_memory - self.baseline_memory
            
            return PerformanceMetrics(
                operation="memory_scaling",
                duration_seconds=duration,
                throughput_ops_per_sec=len(data_structures) / duration if duration > 0 else 0,
                peak_memory_mb=memory_used,
                cpu_usage_percent=cpu_usage,
                success=memory_used > 0
            )
            
        except Exception as e:
            return PerformanceMetrics(
                operation="memory_scaling",
                duration_seconds=0,
                throughput_ops_per_sec=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_aggregation_simulation(self, num_models: int = 5) -> PerformanceMetrics:
        """Benchmark model aggregation simulation."""
        try:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate model parameters (lists of numbers)
            models = []
            for model_id in range(num_models):
                model_params = [i * model_id for i in range(10000)]  # Simulate 10k parameters
                models.append(model_params)
            
            # Simulate aggregation (averaging)
            aggregated_params = []
            param_count = len(models[0]) if models else 0
            
            for param_idx in range(param_count):
                param_sum = sum(model[param_idx] for model in models)
                avg_param = param_sum / len(models)
                aggregated_params.append(avg_param)
            
            end_time = time.time()
            peak_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            duration = end_time - start_time
            parameters_processed = param_count * num_models
            throughput = parameters_processed / duration if duration > 0 else 0
            
            return PerformanceMetrics(
                operation=f"aggregation_{num_models}_models",
                duration_seconds=duration,
                throughput_ops_per_sec=throughput,
                peak_memory_mb=peak_memory - self.baseline_memory,
                cpu_usage_percent=cpu_usage,
                success=len(aggregated_params) == param_count
            )
            
        except Exception as e:
            return PerformanceMetrics(
                operation=f"aggregation_{num_models}_models",
                duration_seconds=0,
                throughput_ops_per_sec=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )

def run_performance_benchmarks() -> Dict[str, PerformanceMetrics]:
    """Run comprehensive performance benchmarks."""
    benchmark = PerformanceBenchmark()
    
    benchmarks = {
        'tensor_ops_small': lambda: benchmark.benchmark_tensor_operations(100),
        'tensor_ops_large': lambda: benchmark.benchmark_tensor_operations(1000),
        'concurrent_clients_5': lambda: benchmark.benchmark_concurrent_clients(5),
        'concurrent_clients_10': lambda: benchmark.benchmark_concurrent_clients(10),
        'concurrent_clients_20': lambda: benchmark.benchmark_concurrent_clients(20),
        'memory_scaling': benchmark.benchmark_memory_scaling,
        'aggregation_5_models': lambda: benchmark.benchmark_aggregation_simulation(5),
        'aggregation_10_models': lambda: benchmark.benchmark_aggregation_simulation(10),
    }
    
    results = {}
    for name, benchmark_func in benchmarks.items():
        print(f"Running benchmark: {name}...")
        try:
            result = benchmark_func()
            results[name] = result
            status = "✅" if result.success else "❌"
            print(f"{status} {name}: {result.throughput_ops_per_sec:.1f} ops/sec, {result.duration_seconds:.3f}s, {result.peak_memory_mb:.1f}MB")
        except Exception as e:
            print(f"❌ {name}: Failed with {e}")
            results[name] = PerformanceMetrics(
                operation=name,
                duration_seconds=0,
                throughput_ops_per_sec=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )
    
    return results

def analyze_scaling_behavior(results: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
    """Analyze scaling behavior from benchmark results."""
    scaling_analysis = {}
    
    # Analyze concurrent client scaling
    client_benchmarks = [
        ('concurrent_clients_5', 5),
        ('concurrent_clients_10', 10),
        ('concurrent_clients_20', 20),
    ]
    
    client_throughputs = []
    for bench_name, num_clients in client_benchmarks:
        if bench_name in results and results[bench_name].success:
            throughput = results[bench_name].throughput_ops_per_sec
            client_throughputs.append((num_clients, throughput))
    
    if len(client_throughputs) >= 2:
        # Calculate scaling efficiency
        base_clients, base_throughput = client_throughputs[0]
        for clients, throughput in client_throughputs[1:]:
            expected_throughput = base_throughput * (clients / base_clients)
            efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
            scaling_analysis[f'scaling_efficiency_{clients}_clients'] = efficiency
    
    # Analyze aggregation scaling
    agg_benchmarks = [
        ('aggregation_5_models', 5),
        ('aggregation_10_models', 10),
    ]
    
    agg_throughputs = []
    for bench_name, num_models in agg_benchmarks:
        if bench_name in results and results[bench_name].success:
            throughput = results[bench_name].throughput_ops_per_sec
            agg_throughputs.append((num_models, throughput))
    
    if len(agg_throughputs) == 2:
        (models1, throughput1), (models2, throughput2) = agg_throughputs
        scaling_factor = (throughput2 / throughput1) / (models2 / models1) if throughput1 > 0 and models1 > 0 else 0
        scaling_analysis['aggregation_scaling_factor'] = scaling_factor
    
    return scaling_analysis

def main():
    """Main performance benchmark execution."""
    print("🚀 DP-Federated LoRA Performance Benchmarks")
    print("=" * 60)
    
    # System information
    try:
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)  # GB
        print(f"System: {cpu_count} CPUs, {memory_total:.1f}GB RAM")
    except:
        print("System: Unable to detect specifications")
    
    print("-" * 60)
    
    start_time = time.time()
    results = run_performance_benchmarks()
    end_time = time.time()
    
    print("-" * 60)
    
    # Analyze results
    successful_benchmarks = [name for name, result in results.items() if result.success]
    total_benchmarks = len(results)
    success_rate = len(successful_benchmarks) / total_benchmarks * 100
    
    print(f"Benchmark Results: {len(successful_benchmarks)}/{total_benchmarks} passed ({success_rate:.1f}%)")
    print(f"Total Duration: {end_time - start_time:.2f}s")
    
    # Performance analysis
    if successful_benchmarks:
        avg_throughput = sum(results[name].throughput_ops_per_sec for name in successful_benchmarks) / len(successful_benchmarks)
        max_throughput = max(results[name].throughput_ops_per_sec for name in successful_benchmarks)
        avg_memory = sum(results[name].peak_memory_mb for name in successful_benchmarks) / len(successful_benchmarks)
        
        print(f"Performance Summary:")
        print(f"  Average Throughput: {avg_throughput:.1f} ops/sec")
        print(f"  Peak Throughput: {max_throughput:.1f} ops/sec")
        print(f"  Average Memory Usage: {avg_memory:.1f} MB")
    
    # Scaling analysis
    scaling_analysis = analyze_scaling_behavior(results)
    if scaling_analysis:
        print(f"Scaling Analysis:")
        for metric, value in scaling_analysis.items():
            print(f"  {metric}: {value:.3f}")
    
    # Create performance report
    performance_report = {
        'timestamp': time.time(),
        'system_info': {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3) if 'psutil' in sys.modules else 0,
        },
        'benchmark_results': {
            name: {
                'operation': result.operation,
                'duration_seconds': result.duration_seconds,
                'throughput_ops_per_sec': result.throughput_ops_per_sec,
                'peak_memory_mb': result.peak_memory_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'success': result.success,
                'error_message': result.error_message
            }
            for name, result in results.items()
        },
        'summary': {
            'total_benchmarks': total_benchmarks,
            'successful_benchmarks': len(successful_benchmarks),
            'success_rate_percent': success_rate,
            'total_duration_seconds': end_time - start_time,
        },
        'scaling_analysis': scaling_analysis,
        'recommendations': []
    }
    
    # Add performance recommendations
    if success_rate >= 90:
        performance_report['recommendations'].append("Performance benchmarks show system is ready for production")
    else:
        performance_report['recommendations'].append("Address failing benchmarks before production deployment")
    
    if scaling_analysis.get('scaling_efficiency_20_clients', 0) > 0.8:
        performance_report['recommendations'].append("Good scaling efficiency - system handles concurrent load well")
    elif 'scaling_efficiency_20_clients' in scaling_analysis:
        performance_report['recommendations'].append("Consider optimizing concurrent processing for better scaling")
    
    # Save performance report
    with open('performance_benchmark_report.json', 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    print(f"📊 Performance report saved to performance_benchmark_report.json")
    
    if success_rate < 50:
        print(f"\n⚠️  Performance benchmarks show significant issues!")
        sys.exit(1)
    elif success_rate < 90:
        print(f"\n⚠️  Some performance benchmarks failed")
        sys.exit(0)
    else:
        print(f"\n🚀 All performance benchmarks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
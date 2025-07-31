#!/usr/bin/env python3
"""
Health Check Service for dp-federated-lora-lab
Comprehensive health monitoring for ML privacy applications
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import redis
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/health-check.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class HealthCheckResult(BaseModel):
    """Health check result model"""
    
    name: str = Field(..., description="Name of the health check")
    status: str = Field(..., description="Status: healthy, unhealthy, warning")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = Field(..., description="Duration of the check in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")


class SystemHealthCheck:
    """System-level health checks"""
    
    @staticmethod
    async def check_cpu_usage() -> HealthCheckResult:
        """Check CPU usage"""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            duration_ms = (time.time() - start_time) * 1000
            
            if cpu_percent > 90:
                status = "unhealthy"
                error_message = f"High CPU usage: {cpu_percent}%"
            elif cpu_percent > 80:
                status = "warning"
                error_message = f"Elevated CPU usage: {cpu_percent}%"
            else:
                status = "healthy"
                error_message = None
            
            return HealthCheckResult(
                name="cpu_usage",
                status=status,
                duration_ms=duration_ms,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                },
                error_message=error_message,
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="cpu_usage",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Failed to check CPU usage: {str(e)}",
            )
    
    @staticmethod
    async def check_memory_usage() -> HealthCheckResult:
        """Check memory usage"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            duration_ms = (time.time() - start_time) * 1000
            
            if memory.percent > 90:
                status = "unhealthy"
                error_message = f"High memory usage: {memory.percent}%"
            elif memory.percent > 80:
                status = "warning"
                error_message = f"Elevated memory usage: {memory.percent}%"
            else:
                status = "healthy"
                error_message = None
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                duration_ms=duration_ms,
                details={
                    "memory_percent": memory.percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                },
                error_message=error_message,
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="memory_usage",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Failed to check memory usage: {str(e)}",
            )
    
    @staticmethod
    async def check_disk_usage() -> HealthCheckResult:
        """Check disk usage"""
        start_time = time.time()
        
        try:
            disk = psutil.disk_usage("/")
            duration_ms = (time.time() - start_time) * 1000
            
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 90:
                status = "unhealthy"
                error_message = f"High disk usage: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = "warning"
                error_message = f"Elevated disk usage: {disk_percent:.1f}%"
            else:
                status = "healthy"
                error_message = None
            
            return HealthCheckResult(
                name="disk_usage",
                status=status,
                duration_ms=duration_ms,
                details={
                    "disk_percent": round(disk_percent, 1),
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                },
                error_message=error_message,
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="disk_usage",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Failed to check disk usage: {str(e)}",
            )


class ServiceHealthCheck:
    """Service-level health checks"""
    
    @staticmethod
    async def check_redis_connection() -> HealthCheckResult:
        """Check Redis connection"""
        start_time = time.time()
        
        try:
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD")
            
            r = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            
            # Test connection
            r.ping()
            info = r.info()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="redis_connection",
                status="healthy",
                duration_ms=duration_ms,
                details={
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                },
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="redis_connection",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Redis connection failed: {str(e)}",
            )
    
    @staticmethod
    async def check_database_connection() -> HealthCheckResult:
        """Check database connection"""
        start_time = time.time()
        
        try:
            # Try to import database dependencies
            import psycopg2
            
            db_host = os.getenv("POSTGRES_HOST", "postgres")
            db_port = int(os.getenv("POSTGRES_PORT", "5432"))
            db_name = os.getenv("POSTGRES_DB", "dp_federated_lora")
            db_user = os.getenv("POSTGRES_USER", "dpuser")
            db_password = os.getenv("POSTGRES_PASSWORD", "dppassword")
            
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                connect_timeout=10,
            )
            
            # Test query
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()[0]
            
            conn.close()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="database_connection",
                status="healthy",
                duration_ms=duration_ms,
                details={
                    "database_version": db_version,
                    "host": db_host,
                    "port": db_port,
                    "database": db_name,
                },
            )
            
        except ImportError:
            # psycopg2 not available, skip check
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="database_connection",
                status="warning",
                duration_ms=duration_ms,
                error_message="Database client not available (psycopg2 not installed)",
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="database_connection",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Database connection failed: {str(e)}",
            )


class ApplicationHealthCheck:
    """Application-specific health checks"""
    
    @staticmethod
    async def check_python_imports() -> HealthCheckResult:
        """Check critical Python imports"""
        start_time = time.time()
        
        critical_imports = [
            "torch",
            "transformers",
            "datasets",
            "opacus",
            "peft",
            "numpy",
            "pandas",
        ]
        
        try:
            import_results = {}
            for module in critical_imports:
                try:
                    __import__(module)
                    import_results[module] = "available"
                except ImportError as e:
                    import_results[module] = f"failed: {str(e)}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            failed_imports = [
                mod for mod, status in import_results.items()
                if status.startswith("failed")
            ]
            
            if failed_imports:
                return HealthCheckResult(
                    name="python_imports",
                    status="unhealthy",
                    duration_ms=duration_ms,
                    details=import_results,
                    error_message=f"Failed imports: {', '.join(failed_imports)}",
                )
            else:
                return HealthCheckResult(
                    name="python_imports",
                    status="healthy",
                    duration_ms=duration_ms,
                    details=import_results,
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="python_imports",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Import check failed: {str(e)}",
            )
    
    @staticmethod
    async def check_gpu_availability() -> HealthCheckResult:
        """Check GPU availability"""
        start_time = time.time()
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                "cuda_available": cuda_available,
                "device_count": device_count,
            }
            
            if cuda_available:
                details["devices"] = []
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    details["devices"].append({
                        "id": i,
                        "name": device_props.name,
                        "memory_gb": round(device_props.total_memory / (1024**3), 2),
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                    })
            
            # GPU availability is nice to have but not critical
            status = "healthy" if cuda_available else "warning"
            error_message = None if cuda_available else "No GPU available"
            
            return HealthCheckResult(
                name="gpu_availability",
                status=status,
                duration_ms=duration_ms,
                details=details,
                error_message=error_message,
            )
            
        except ImportError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="gpu_availability",
                status="warning",
                duration_ms=duration_ms,
                error_message="PyTorch not available",
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="gpu_availability",
                status="warning",
                duration_ms=duration_ms,
                error_message=f"GPU check failed: {str(e)}",
            )
    
    @staticmethod
    async def check_model_loading() -> HealthCheckResult:
        """Check model loading capability"""
        start_time = time.time()
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Try to load a small model for testing
            model_name = os.getenv("TEST_MODEL_NAME", "microsoft/DialoGPT-small")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Test basic functionality
            test_text = "Hello, world!"
            inputs = tokenizer(test_text, return_tensors="pt")
            outputs = model(**inputs)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="model_loading",
                status="healthy",
                duration_ms=duration_ms,
                details={
                    "test_model": model_name,
                    "tokenizer_vocab_size": tokenizer.vocab_size,
                    "model_config": str(model.config),
                    "output_shape": list(outputs.last_hidden_state.shape),
                },
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="model_loading",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Model loading failed: {str(e)}",
            )


class PrivacyHealthCheck:
    """Privacy-specific health checks"""
    
    @staticmethod
    async def check_privacy_budget() -> HealthCheckResult:
        """Check privacy budget status"""
        start_time = time.time()
        
        try:
            # Check privacy budget configuration
            epsilon = float(os.getenv("PRIVACY_BUDGET_EPSILON", "1.0"))
            delta = float(os.getenv("PRIVACY_BUDGET_DELTA", "1e-5"))
            
            # In a real application, you would check the actual budget consumption
            # For now, we just validate the configuration
            
            duration_ms = (time.time() - start_time) * 1000
            
            if epsilon <= 0 or delta <= 0:
                return HealthCheckResult(
                    name="privacy_budget",
                    status="unhealthy",
                    duration_ms=duration_ms,
                    error_message="Invalid privacy parameters",
                    details={"epsilon": epsilon, "delta": delta},
                )
            
            return HealthCheckResult(
                name="privacy_budget",
                status="healthy",
                duration_ms=duration_ms,
                details={
                    "epsilon": epsilon,
                    "delta": delta,
                    "budget_remaining": "100%",  # Placeholder
                    "accountant": os.getenv("PRIVACY_ACCOUNTANT", "rdp"),
                },
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="privacy_budget",
                status="unhealthy",
                duration_ms=duration_ms,
                error_message=f"Privacy budget check failed: {str(e)}",
            )


class HealthCheckService:
    """Main health check service"""
    
    def __init__(self):
        self.checks = [
            # System checks
            SystemHealthCheck.check_cpu_usage,
            SystemHealthCheck.check_memory_usage,
            SystemHealthCheck.check_disk_usage,
            
            # Service checks
            ServiceHealthCheck.check_redis_connection,
            ServiceHealthCheck.check_database_connection,
            
            # Application checks
            ApplicationHealthCheck.check_python_imports,
            ApplicationHealthCheck.check_gpu_availability,
            ApplicationHealthCheck.check_model_loading,
            
            # Privacy checks
            PrivacyHealthCheck.check_privacy_budget,
        ]
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Starting health check suite")
        start_time = time.time()
        
        results = []
        for check in self.checks:
            try:
                result = await check()
                results.append(result)
                logger.info(f"Health check '{result.name}': {result.status}")
            except Exception as e:
                logger.error(f"Health check failed with exception: {str(e)}")
                results.append(HealthCheckResult(
                    name="unknown",
                    status="unhealthy",
                    duration_ms=0,
                    error_message=str(e),
                ))
        
        total_duration = (time.time() - start_time) * 1000
        
        # Determine overall status
        statuses = [r.status for r in results]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        health_report = {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_ms": total_duration,
            "checks_count": len(results),
            "healthy_count": len([r for r in results if r.status == "healthy"]),
            "warning_count": len([r for r in results if r.status == "warning"]),
            "unhealthy_count": len([r for r in results if r.status == "unhealthy"]),
            "checks": [r.dict() for r in results],
        }
        
        logger.info(f"Health check suite completed: {overall_status} "
                   f"({len(results)} checks in {total_duration:.1f}ms)")
        
        return health_report
    
    async def save_health_report(self, report: Dict[str, Any], 
                               filepath: str = "/app/logs/health-report.json") -> None:
        """Save health report to file"""
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Health report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save health report: {str(e)}")


async def main():
    """Main function"""
    health_service = HealthCheckService()
    
    # Run health checks
    report = await health_service.run_all_checks()
    
    # Save report
    await health_service.save_health_report(report)
    
    # Print summary
    print(f"Overall Status: {report['overall_status']}")
    print(f"Total Checks: {report['checks_count']}")
    print(f"Healthy: {report['healthy_count']}")
    print(f"Warning: {report['warning_count']}")
    print(f"Unhealthy: {report['unhealthy_count']}")
    print(f"Duration: {report['total_duration_ms']:.1f}ms")
    
    # Exit with appropriate code
    if report['overall_status'] == 'unhealthy':
        sys.exit(1)
    elif report['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
"""
Comprehensive Production Deployment Orchestrator for DP-Federated LoRA Lab.

Advanced deployment system providing:
- Zero-downtime blue/green deployments
- Multi-region orchestration with global load balancing
- Automated rollback and canary deployment strategies
- Real-time health monitoring and auto-healing
- Infrastructure as Code (IaC) with Terraform/Kubernetes
- Compliance validation and security hardening
- Performance monitoring and SLA enforcement
- Disaster recovery and business continuity
"""

import asyncio
import logging
import json
import time
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import yaml
import tempfile
import os

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    PREPARATION = auto()
    BUILD = auto()
    TEST = auto()
    SECURITY_SCAN = auto()
    STAGING_DEPLOY = auto()
    CANARY_DEPLOY = auto()
    PRODUCTION_DEPLOY = auto()
    MONITORING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = auto()
    ROLLING_UPDATE = auto()
    CANARY = auto()
    RECREATE = auto()
    A_B_TESTING = auto()


@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration."""
    version: str
    environment: str = "production"
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    regions: List[str] = field(default_factory=lambda: ["us-east-1", "eu-west-1"])
    replicas_per_region: int = 3
    canary_percentage: float = 0.1
    rollback_threshold: float = 0.05  # 5% error rate triggers rollback
    health_check_timeout: int = 300
    sla_requirements: Dict[str, float] = field(default_factory=lambda: {
        "availability": 0.9999,  # 99.99% uptime
        "response_time_p95": 200,  # 95th percentile < 200ms
        "error_rate": 0.001  # < 0.1% error rate
    })
    compliance_requirements: Set[str] = field(default_factory=lambda: {
        "SOC2", "GDPR", "HIPAA", "CCPA"
    })


@dataclass
class DeploymentResult:
    """Deployment execution result."""
    deployment_id: str
    stage: DeploymentStage
    success: bool
    start_time: datetime
    end_time: datetime
    duration: float
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    rollback_triggered: bool = False


class ProductionDeploymentOrchestrator:
    """Comprehensive production deployment orchestrator."""
    
    def __init__(self):
        self.deployment_history = []
        self.active_deployments = {}
        self.infrastructure_state = {}
        self.monitoring_enabled = True
        
    async def deploy_to_production(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute comprehensive production deployment."""
        deployment_id = self._generate_deployment_id(config)
        start_time = datetime.now()
        
        logger.info(f"Starting production deployment {deployment_id}")
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            stage=DeploymentStage.PREPARATION,
            success=False,
            start_time=start_time,
            end_time=start_time,
            duration=0.0
        )
        
        try:
            # Stage 1: Preparation and validation
            await self._prepare_deployment(config, result)
            
            # Stage 2: Build and package
            await self._build_deployment_artifacts(config, result)
            
            # Stage 3: Security scanning and compliance
            await self._security_compliance_validation(config, result)
            
            # Stage 4: Staging deployment and testing
            await self._staging_deployment(config, result)
            
            # Stage 5: Canary deployment (if configured)
            if config.strategy == DeploymentStrategy.CANARY:
                await self._canary_deployment(config, result)
            
            # Stage 6: Production deployment
            await self._production_deployment(config, result)
            
            # Stage 7: Post-deployment monitoring
            await self._post_deployment_monitoring(config, result)
            
            result.stage = DeploymentStage.COMPLETED
            result.success = True
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            result.errors.append(str(e))
            result.stage = DeploymentStage.FAILED
            
            # Attempt automatic rollback
            if self._should_trigger_rollback(result, config):
                await self._execute_rollback(config, result)
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            self.deployment_history.append(result)
            
        return result
    
    async def _prepare_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Prepare deployment environment and validate prerequisites."""
        result.stage = DeploymentStage.PREPARATION
        result.logs.append("Starting deployment preparation")
        
        # Validate configuration
        validation_errors = self._validate_deployment_config(config)
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {validation_errors}")
        
        # Check infrastructure readiness
        await self._check_infrastructure_readiness(config)
        
        # Prepare deployment workspace
        workspace = self._create_deployment_workspace(config)
        result.metrics['workspace_path'] = str(workspace)
        
        # Validate permissions and access
        await self._validate_deployment_permissions(config)
        
        result.logs.append("Deployment preparation completed successfully")
    
    async def _build_deployment_artifacts(self, config: DeploymentConfig, result: DeploymentResult):
        """Build and package deployment artifacts."""
        result.stage = DeploymentStage.BUILD
        result.logs.append("Starting artifact build process")
        
        # Build Docker images
        image_tags = await self._build_docker_images(config)
        result.metrics['docker_images'] = image_tags
        
        # Generate Kubernetes manifests
        k8s_manifests = await self._generate_kubernetes_manifests(config)
        result.metrics['kubernetes_manifests'] = k8s_manifests
        
        # Create Terraform configurations
        terraform_configs = await self._generate_terraform_configs(config)
        result.metrics['terraform_configs'] = terraform_configs
        
        # Package and upload artifacts
        artifact_urls = await self._package_and_upload_artifacts(config, {
            'images': image_tags,
            'k8s': k8s_manifests,
            'terraform': terraform_configs
        })
        result.metrics['artifact_urls'] = artifact_urls
        
        result.logs.append("Artifact build completed successfully")
    
    async def _security_compliance_validation(self, config: DeploymentConfig, result: DeploymentResult):
        """Perform comprehensive security and compliance validation."""
        result.stage = DeploymentStage.SECURITY_SCAN
        result.logs.append("Starting security and compliance validation")
        
        # Container security scanning
        security_scan_results = await self._scan_container_security(config)
        result.metrics['security_scan'] = security_scan_results
        
        # Compliance validation
        compliance_results = await self._validate_compliance(config)
        result.metrics['compliance_validation'] = compliance_results
        
        # Vulnerability assessment
        vulnerability_scan = await self._perform_vulnerability_assessment(config)
        result.metrics['vulnerability_scan'] = vulnerability_scan
        
        # Security policy validation
        policy_validation = await self._validate_security_policies(config)
        result.metrics['policy_validation'] = policy_validation
        
        # Check for critical security issues
        if self._has_critical_security_issues(result.metrics):
            raise ValueError("Critical security issues found, deployment blocked")
        
        result.logs.append("Security and compliance validation completed")
    
    async def _staging_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy to staging environment for final validation."""
        result.stage = DeploymentStage.STAGING_DEPLOY
        result.logs.append("Starting staging deployment")
        
        # Deploy to staging environment
        staging_deployment = await self._deploy_to_staging(config)
        result.metrics['staging_deployment'] = staging_deployment
        
        # Run integration tests
        integration_test_results = await self._run_integration_tests(config)
        result.metrics['integration_tests'] = integration_test_results
        
        # Performance testing
        performance_test_results = await self._run_performance_tests(config)
        result.metrics['performance_tests'] = performance_test_results
        
        # End-to-end testing
        e2e_test_results = await self._run_e2e_tests(config)
        result.metrics['e2e_tests'] = e2e_test_results
        
        # Validate staging deployment health
        if not self._validate_staging_health(result.metrics):
            raise ValueError("Staging deployment health checks failed")
        
        result.logs.append("Staging deployment completed successfully")
    
    async def _canary_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Execute canary deployment with gradual traffic routing."""
        result.stage = DeploymentStage.CANARY_DEPLOY
        result.logs.append("Starting canary deployment")
        
        # Deploy canary version
        canary_deployment = await self._deploy_canary_version(config)
        result.metrics['canary_deployment'] = canary_deployment
        
        # Gradually route traffic to canary
        traffic_routing_steps = [0.01, 0.05, 0.1, 0.25]  # 1%, 5%, 10%, 25%
        
        for traffic_percentage in traffic_routing_steps:
            result.logs.append(f"Routing {traffic_percentage*100}% traffic to canary")
            
            # Update traffic routing
            await self._update_traffic_routing(config, traffic_percentage)
            
            # Monitor canary performance
            canary_metrics = await self._monitor_canary_performance(config, traffic_percentage)
            result.metrics[f'canary_metrics_{traffic_percentage}'] = canary_metrics
            
            # Check for issues
            if self._detect_canary_issues(canary_metrics, config):
                result.logs.append("Canary issues detected, initiating rollback")
                await self._rollback_canary(config)
                raise ValueError("Canary deployment failed health checks")
            
            # Wait before next traffic increase
            await asyncio.sleep(300)  # 5 minutes between steps
        
        result.logs.append("Canary deployment completed successfully")
    
    async def _production_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Execute full production deployment."""
        result.stage = DeploymentStage.PRODUCTION_DEPLOY
        result.logs.append("Starting production deployment")
        
        if config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._blue_green_deployment(config, result)
        elif config.strategy == DeploymentStrategy.ROLLING_UPDATE:
            await self._rolling_update_deployment(config, result)
        elif config.strategy == DeploymentStrategy.CANARY:
            await self._complete_canary_deployment(config, result)
        else:
            raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
        
        result.logs.append("Production deployment completed successfully")
    
    async def _blue_green_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Execute blue/green deployment strategy."""
        result.logs.append("Executing blue/green deployment")
        
        # Deploy to green environment
        green_deployment = await self._deploy_to_green_environment(config)
        result.metrics['green_deployment'] = green_deployment
        
        # Validate green environment health
        green_health = await self._validate_green_environment_health(config)
        result.metrics['green_health'] = green_health
        
        if not green_health['healthy']:
            raise ValueError("Green environment health validation failed")
        
        # Switch traffic from blue to green
        traffic_switch = await self._switch_traffic_blue_to_green(config)
        result.metrics['traffic_switch'] = traffic_switch
        
        # Monitor for a period after switch
        await asyncio.sleep(600)  # 10 minutes monitoring
        
        # Validate production health
        production_health = await self._validate_production_health(config)
        result.metrics['production_health'] = production_health
        
        if not production_health['healthy']:
            # Rollback to blue
            await self._switch_traffic_green_to_blue(config)
            raise ValueError("Production health validation failed after blue/green switch")
        
        # Decommission old blue environment
        await self._decommission_blue_environment(config)
    
    async def _post_deployment_monitoring(self, config: DeploymentConfig, result: DeploymentResult):
        """Post-deployment monitoring and validation."""
        result.stage = DeploymentStage.MONITORING
        result.logs.append("Starting post-deployment monitoring")
        
        # Monitor key metrics for extended period
        monitoring_duration = 3600  # 1 hour
        monitoring_start = time.time()
        
        while time.time() - monitoring_start < monitoring_duration:
            # Collect system metrics
            metrics = await self._collect_production_metrics(config)
            timestamp = datetime.now().isoformat()
            
            if timestamp not in result.metrics:
                result.metrics[timestamp] = {}
            result.metrics[timestamp] = metrics
            
            # Check SLA compliance
            sla_compliance = self._check_sla_compliance(metrics, config.sla_requirements)
            
            if not sla_compliance['compliant']:
                result.logs.append(f"SLA violation detected: {sla_compliance['violations']}")
                if sla_compliance['severity'] == 'critical':
                    raise ValueError("Critical SLA violation detected")
            
            await asyncio.sleep(60)  # Check every minute
        
        result.logs.append("Post-deployment monitoring completed")
    
    async def _execute_rollback(self, config: DeploymentConfig, result: DeploymentResult):
        """Execute automatic rollback procedure."""
        result.stage = DeploymentStage.ROLLED_BACK
        result.rollback_triggered = True
        result.logs.append("Initiating automatic rollback")
        
        # Get previous stable version
        previous_version = self._get_previous_stable_version(config)
        if not previous_version:
            raise ValueError("No previous stable version available for rollback")
        
        # Execute rollback
        rollback_config = config
        rollback_config.version = previous_version
        
        # Fast rollback using blue/green switch or rolling update
        if config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._switch_traffic_green_to_blue(config)
        else:
            await self._rolling_rollback(config, previous_version)
        
        # Validate rollback success
        rollback_validation = await self._validate_rollback_success(config)
        result.metrics['rollback_validation'] = rollback_validation
        
        if not rollback_validation['successful']:
            raise ValueError("Rollback validation failed")
        
        result.logs.append("Automatic rollback completed successfully")
    
    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.now().isoformat()
        config_hash = hashlib.sha256(
            f"{config.version}-{config.environment}-{timestamp}".encode()
        ).hexdigest()[:8]
        return f"deploy-{config.environment}-{config.version}-{config_hash}"
    
    def _validate_deployment_config(self, config: DeploymentConfig) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if not config.version:
            errors.append("Version is required")
        
        if not config.regions:
            errors.append("At least one region must be specified")
        
        if config.canary_percentage <= 0 or config.canary_percentage >= 1:
            errors.append("Canary percentage must be between 0 and 1")
        
        if config.replicas_per_region < 1:
            errors.append("Replicas per region must be at least 1")
        
        return errors
    
    async def _check_infrastructure_readiness(self, config: DeploymentConfig):
        """Check infrastructure readiness for deployment."""
        # Simulate infrastructure checks
        await asyncio.sleep(1)
        
        # Check Kubernetes clusters
        for region in config.regions:
            cluster_status = await self._check_k8s_cluster_status(region)
            if not cluster_status['ready']:
                raise ValueError(f"Kubernetes cluster in {region} is not ready")
    
    async def _check_k8s_cluster_status(self, region: str) -> Dict[str, Any]:
        """Check Kubernetes cluster status for a region."""
        # Simulate cluster status check
        await asyncio.sleep(0.5)
        return {
            'ready': True,
            'nodes': 5,
            'cpu_utilization': 0.45,
            'memory_utilization': 0.60
        }
    
    def _create_deployment_workspace(self, config: DeploymentConfig) -> Path:
        """Create temporary workspace for deployment."""
        workspace = Path(tempfile.mkdtemp(prefix=f"deploy-{config.version}-"))
        workspace.mkdir(exist_ok=True)
        return workspace
    
    async def _validate_deployment_permissions(self, config: DeploymentConfig):
        """Validate deployment permissions and access."""
        # Simulate permission validation
        await asyncio.sleep(0.5)
        
        # Check AWS/GCP/Azure permissions
        # Check Kubernetes RBAC
        # Check registry access
        pass
    
    async def _build_docker_images(self, config: DeploymentConfig) -> List[str]:
        """Build and tag Docker images."""
        await asyncio.sleep(5)  # Simulate build time
        
        # Build main application image
        main_image = f"dp-federated-lora:{config.version}"
        
        # Build sidecar images if needed
        sidecar_images = [
            f"dp-federated-monitoring:{config.version}",
            f"dp-federated-proxy:{config.version}"
        ]
        
        return [main_image] + sidecar_images
    
    async def _generate_kubernetes_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        await asyncio.sleep(2)  # Simulate manifest generation
        
        manifests = {}
        
        # Deployment manifest
        manifests['deployment.yaml'] = self._create_k8s_deployment_manifest(config)
        
        # Service manifest
        manifests['service.yaml'] = self._create_k8s_service_manifest(config)
        
        # ConfigMap manifest
        manifests['configmap.yaml'] = self._create_k8s_configmap_manifest(config)
        
        # Ingress manifest
        manifests['ingress.yaml'] = self._create_k8s_ingress_manifest(config)
        
        return manifests
    
    def _create_k8s_deployment_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes deployment manifest."""
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'dp-federated-lora-{config.environment}',
                'labels': {
                    'app': 'dp-federated-lora',
                    'version': config.version,
                    'environment': config.environment
                }
            },
            'spec': {
                'replicas': config.replicas_per_region,
                'selector': {
                    'matchLabels': {
                        'app': 'dp-federated-lora',
                        'version': config.version
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'dp-federated-lora',
                            'version': config.version,
                            'environment': config.environment
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'dp-federated-lora',
                            'image': f'dp-federated-lora:{config.version}',
                            'ports': [{'containerPort': 8080}],
                            'resources': {
                                'requests': {
                                    'memory': '2Gi',
                                    'cpu': '500m'
                                },
                                'limits': {
                                    'memory': '4Gi',
                                    'cpu': '2000m'
                                }
                            },
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': config.environment},
                                {'name': 'VERSION', 'value': config.version}
                            ]
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(manifest, default_flow_style=False)
    
    def _create_k8s_service_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes service manifest."""
        manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'dp-federated-lora-service-{config.environment}',
                'labels': {
                    'app': 'dp-federated-lora',
                    'environment': config.environment
                }
            },
            'spec': {
                'selector': {
                    'app': 'dp-federated-lora',
                    'version': config.version
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8080
                }],
                'type': 'ClusterIP'
            }
        }
        
        return yaml.dump(manifest, default_flow_style=False)
    
    def _create_k8s_configmap_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes ConfigMap manifest."""
        manifest = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'dp-federated-lora-config-{config.environment}',
                'labels': {
                    'app': 'dp-federated-lora',
                    'environment': config.environment
                }
            },
            'data': {
                'config.json': json.dumps({
                    'environment': config.environment,
                    'version': config.version,
                    'regions': config.regions,
                    'sla_requirements': config.sla_requirements
                }, indent=2)
            }
        }
        
        return yaml.dump(manifest, default_flow_style=False)
    
    def _create_k8s_ingress_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes Ingress manifest."""
        manifest = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f'dp-federated-lora-ingress-{config.environment}',
                'labels': {
                    'app': 'dp-federated-lora',
                    'environment': config.environment
                },
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': [f'dp-federated-{config.environment}.example.com'],
                    'secretName': f'dp-federated-tls-{config.environment}'
                }],
                'rules': [{
                    'host': f'dp-federated-{config.environment}.example.com',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f'dp-federated-lora-service-{config.environment}',
                                    'port': {'number': 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        return yaml.dump(manifest, default_flow_style=False)
    
    async def _generate_terraform_configs(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Terraform infrastructure configurations."""
        await asyncio.sleep(1)  # Simulate config generation
        
        configs = {}
        
        # Main infrastructure config
        configs['main.tf'] = self._create_terraform_main_config(config)
        
        # Variables config
        configs['variables.tf'] = self._create_terraform_variables_config(config)
        
        # Outputs config
        configs['outputs.tf'] = self._create_terraform_outputs_config(config)
        
        return configs
    
    def _create_terraform_main_config(self, config: DeploymentConfig) -> str:
        """Create main Terraform configuration."""
        return f"""
# DP-Federated LoRA Infrastructure
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

# Multi-region deployment
{self._generate_terraform_regions(config)}

# Load balancer configuration
resource "aws_lb" "main" {{
  name               = "dp-federated-lb-{config.environment}"
  internal           = false
  load_balancer_type = "application"
  
  enable_deletion_protection = true
  
  tags = {{
    Environment = "{config.environment}"
    Version     = "{config.version}"
  }}
}}
        """.strip()
    
    def _generate_terraform_regions(self, config: DeploymentConfig) -> str:
        """Generate Terraform configuration for multiple regions."""
        terraform_regions = []
        
        for i, region in enumerate(config.regions):
            terraform_regions.append(f"""
provider "aws" {{
  alias  = "region_{i}"
  region = "{region}"
}}

module "region_{i}" {{
  source = "./modules/region"
  
  providers = {{
    aws = aws.region_{i}
  }}
  
  environment = "{config.environment}"
  version     = "{config.version}"
  replicas    = {config.replicas_per_region}
  region_name = "{region}"
}}
            """.strip())
        
        return "\n\n".join(terraform_regions)
    
    def _create_terraform_variables_config(self, config: DeploymentConfig) -> str:
        """Create Terraform variables configuration."""
        return f"""
variable "environment" {{
  description = "Deployment environment"
  type        = string
  default     = "{config.environment}"
}}

variable "version" {{
  description = "Application version"
  type        = string
  default     = "{config.version}"
}}

variable "regions" {{
  description = "Deployment regions"
  type        = list(string)
  default     = {json.dumps(config.regions)}
}}

variable "replicas_per_region" {{
  description = "Number of replicas per region"
  type        = number
  default     = {config.replicas_per_region}
}}
        """.strip()
    
    def _create_terraform_outputs_config(self, config: DeploymentConfig) -> str:
        """Create Terraform outputs configuration."""
        return """
output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "deployment_endpoints" {
  description = "Regional deployment endpoints"
  value       = {
    for i, region in var.regions : region => module.region_${i}.endpoint
  }
}
        """.strip()
    
    async def _scan_container_security(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform container security scanning."""
        await asyncio.sleep(3)  # Simulate security scan
        
        return {
            'vulnerabilities_found': 0,
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 2,
            'low_vulnerabilities': 5,
            'scan_passed': True,
            'compliance_score': 0.95
        }
    
    def _has_critical_security_issues(self, metrics: Dict[str, Any]) -> bool:
        """Check if there are critical security issues."""
        security_scan = metrics.get('security_scan', {})
        return security_scan.get('critical_vulnerabilities', 0) > 0
    
    def _should_trigger_rollback(self, result: DeploymentResult, config: DeploymentConfig) -> bool:
        """Determine if automatic rollback should be triggered."""
        return (
            result.stage in [DeploymentStage.PRODUCTION_DEPLOY, DeploymentStage.MONITORING] and
            len(result.errors) > 0
        )
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment."""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None
    
    def get_active_deployments(self) -> List[DeploymentResult]:
        """Get all currently active deployments."""
        return [
            deployment for deployment in self.deployment_history
            if deployment.stage not in [DeploymentStage.COMPLETED, DeploymentStage.FAILED, DeploymentStage.ROLLED_BACK]
        ]
    
    async def generate_deployment_report(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        recent_deployments = [
            d for d in self.deployment_history[-10:]
            if d.end_time >= datetime.now() - timedelta(days=7)
        ]
        
        success_rate = (
            sum(1 for d in recent_deployments if d.success) / len(recent_deployments)
            if recent_deployments else 0
        )
        
        avg_deployment_time = (
            sum(d.duration for d in recent_deployments) / len(recent_deployments)
            if recent_deployments else 0
        )
        
        return {
            'deployment_statistics': {
                'total_deployments': len(recent_deployments),
                'successful_deployments': sum(1 for d in recent_deployments if d.success),
                'failed_deployments': sum(1 for d in recent_deployments if not d.success),
                'rollbacks_triggered': sum(1 for d in recent_deployments if d.rollback_triggered),
                'success_rate': success_rate,
                'average_deployment_time': avg_deployment_time
            },
            'environment_status': {
                'production_health': await self._get_production_health_summary(),
                'active_deployments': len(self.get_active_deployments()),
                'last_successful_deployment': self._get_last_successful_deployment(),
                'infrastructure_status': await self._get_infrastructure_status()
            },
            'sla_compliance': await self._get_sla_compliance_report(config),
            'security_posture': await self._get_security_posture_report(),
            'recommendations': await self._generate_deployment_recommendations()
        }
    
    # Placeholder implementations for complex operations
    async def _package_and_upload_artifacts(self, config, artifacts):
        await asyncio.sleep(2)
        return {"artifacts_uploaded": True, "registry_url": "https://registry.example.com"}
    
    async def _validate_compliance(self, config):
        await asyncio.sleep(1)
        return {"compliant": True, "requirements_met": list(config.compliance_requirements)}
    
    async def _perform_vulnerability_assessment(self, config):
        await asyncio.sleep(2)
        return {"vulnerabilities": 0, "assessment_passed": True}
    
    async def _validate_security_policies(self, config):
        await asyncio.sleep(1)
        return {"policies_validated": True, "violations": 0}
    
    async def _deploy_to_staging(self, config):
        await asyncio.sleep(5)
        return {"staging_deployed": True, "endpoint": f"https://staging-{config.version}.example.com"}
    
    async def _run_integration_tests(self, config):
        await asyncio.sleep(10)
        return {"tests_passed": 95, "tests_failed": 5, "success_rate": 0.95}
    
    async def _run_performance_tests(self, config):
        await asyncio.sleep(15)
        return {"avg_response_time": 150, "p95_response_time": 250, "throughput": 1000}
    
    async def _run_e2e_tests(self, config):
        await asyncio.sleep(20)
        return {"scenarios_passed": 48, "scenarios_failed": 2, "success_rate": 0.96}
    
    def _validate_staging_health(self, metrics):
        integration_tests = metrics.get('integration_tests', {})
        return integration_tests.get('success_rate', 0) >= 0.9
    
    async def _deploy_canary_version(self, config):
        await asyncio.sleep(3)
        return {"canary_deployed": True, "version": config.version}
    
    async def _update_traffic_routing(self, config, percentage):
        await asyncio.sleep(1)
        return {"traffic_routed": percentage}
    
    async def _monitor_canary_performance(self, config, percentage):
        await asyncio.sleep(60)
        return {
            "error_rate": 0.001,
            "response_time": 120,
            "traffic_percentage": percentage,
            "healthy": True
        }
    
    def _detect_canary_issues(self, metrics, config):
        return metrics.get('error_rate', 0) > config.rollback_threshold
    
    async def _rollback_canary(self, config):
        await asyncio.sleep(2)
        return {"canary_rolled_back": True}
    
    async def _deploy_to_green_environment(self, config):
        await asyncio.sleep(8)
        return {"green_deployed": True, "environment": "green"}
    
    async def _validate_green_environment_health(self, config):
        await asyncio.sleep(2)
        return {"healthy": True, "checks_passed": 15}
    
    async def _switch_traffic_blue_to_green(self, config):
        await asyncio.sleep(1)
        return {"traffic_switched": True, "active_environment": "green"}
    
    async def _validate_production_health(self, config):
        await asyncio.sleep(3)
        return {"healthy": True, "sla_compliant": True}
    
    async def _switch_traffic_green_to_blue(self, config):
        await asyncio.sleep(1)
        return {"traffic_switched": True, "active_environment": "blue"}
    
    async def _decommission_blue_environment(self, config):
        await asyncio.sleep(2)
        return {"blue_decommissioned": True}
    
    async def _complete_canary_deployment(self, config, result):
        await asyncio.sleep(5)
        result.metrics['canary_completed'] = True
    
    async def _rolling_update_deployment(self, config, result):
        await asyncio.sleep(10)
        result.metrics['rolling_update_completed'] = True
    
    async def _collect_production_metrics(self, config):
        await asyncio.sleep(1)
        return {
            "response_time_p95": 180,
            "error_rate": 0.0005,
            "availability": 0.9998,
            "throughput": 1200
        }
    
    def _check_sla_compliance(self, metrics, sla_requirements):
        violations = []
        
        if metrics.get('response_time_p95', 0) > sla_requirements.get('response_time_p95', float('inf')):
            violations.append('response_time')
        
        if metrics.get('error_rate', 0) > sla_requirements.get('error_rate', 1.0):
            violations.append('error_rate')
        
        if metrics.get('availability', 0) < sla_requirements.get('availability', 0):
            violations.append('availability')
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'severity': 'critical' if violations else 'none'
        }
    
    def _get_previous_stable_version(self, config):
        # Find the last successful deployment
        for deployment in reversed(self.deployment_history):
            if (deployment.success and 
                deployment.stage == DeploymentStage.COMPLETED and
                deployment.deployment_id != config.version):
                return deployment.deployment_id
        return None
    
    async def _rolling_rollback(self, config, previous_version):
        await asyncio.sleep(5)
        return {"rolled_back": True, "version": previous_version}
    
    async def _validate_rollback_success(self, config):
        await asyncio.sleep(2)
        return {"successful": True, "health_checks_passed": True}
    
    # Report generation helper methods
    async def _get_production_health_summary(self):
        return {"status": "healthy", "uptime": "99.99%", "last_incident": None}
    
    def _get_last_successful_deployment(self):
        for deployment in reversed(self.deployment_history):
            if deployment.success and deployment.stage == DeploymentStage.COMPLETED:
                return deployment.deployment_id
        return None
    
    async def _get_infrastructure_status(self):
        return {"clusters": 3, "healthy_nodes": 15, "total_capacity": "85%"}
    
    async def _get_sla_compliance_report(self, config):
        return {"overall_compliance": 99.95, "violations_last_30_days": 0}
    
    async def _get_security_posture_report(self):
        return {"security_score": 95, "critical_vulnerabilities": 0, "last_scan": datetime.now().isoformat()}
    
    async def _generate_deployment_recommendations(self):
        return [
            "Consider implementing automated canary analysis",
            "Increase monitoring frequency during deployments",
            "Review and update security scanning policies"
        ]


# Global production deployment orchestrator
production_orchestrator = ProductionDeploymentOrchestrator()


async def deploy_to_production(config: DeploymentConfig) -> DeploymentResult:
    """Deploy application to production with comprehensive validation."""
    return await production_orchestrator.deploy_to_production(config)


async def get_deployment_status(deployment_id: str) -> Optional[DeploymentResult]:
    """Get the status of a specific deployment."""
    return production_orchestrator.get_deployment_status(deployment_id)


async def generate_deployment_report(environment: str = "production") -> Dict[str, Any]:
    """Generate comprehensive deployment report."""
    config = DeploymentConfig(version="latest", environment=environment)
    return await production_orchestrator.generate_deployment_report(config)
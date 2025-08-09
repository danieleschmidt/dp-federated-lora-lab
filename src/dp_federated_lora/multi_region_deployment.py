"""
Multi-Region Deployment Module for DP-Federated LoRA system.

This module implements comprehensive multi-region deployment capabilities
including regional data residency, geo-distributed federated learning,
cross-border compliance, latency optimization, and disaster recovery
for global federated learning infrastructures.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json
import uuid
import socket
from pathlib import Path

from .config import FederatedConfig
from .server import FederatedServer
from .monitoring import ServerMetricsCollector
from .global_compliance import PrivacyRegulation, ComplianceEngine


logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Supported deployment regions worldwide."""
    # North America
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    US_CENTRAL = "us-central-1"
    CANADA_CENTRAL = "ca-central-1"
    
    # Europe
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    EU_NORTH = "eu-north-1"
    UK_SOUTH = "uk-south-1"
    
    # Asia Pacific
    ASIA_SOUTHEAST = "ap-southeast-1"
    ASIA_EAST = "ap-east-1"
    JAPAN_EAST = "jp-east-1"
    AUSTRALIA_SOUTHEAST = "au-southeast-1"
    
    # South America
    BRAZIL_SOUTH = "sa-south-1"
    
    # Africa
    AFRICA_SOUTH = "af-south-1"


class DataResidencyZone(Enum):
    """Data residency zones with specific compliance requirements."""
    EU_GDPR = "eu_gdpr"
    US_CCPA = "us_ccpa"
    SINGAPORE_PDPA = "sg_pdpa"
    BRAZIL_LGPD = "br_lgpd"
    CANADA_PIPEDA = "ca_pipeda"
    AUSTRALIA_PRIVACY = "au_privacy"
    JAPAN_APPI = "jp_appi"
    GLOBAL_STANDARD = "global_standard"


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    
    region: DeploymentRegion
    residency_zone: DataResidencyZone
    primary_regulations: List[PrivacyRegulation]
    supported_languages: List[str]
    data_centers: List[str]
    latency_sla_ms: int
    availability_sla: float
    backup_regions: List[DeploymentRegion]
    edge_locations: List[str] = field(default_factory=list)
    compliance_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientGeoLocation:
    """Geographical location information for clients."""
    
    client_id: str
    country_code: str
    region_code: str
    city: str
    latitude: float
    longitude: float
    timezone: str
    assigned_region: DeploymentRegion
    data_residency_zone: DataResidencyZone
    applicable_regulations: List[PrivacyRegulation]


@dataclass
class RegionHealthMetrics:
    """Health metrics for a deployment region."""
    
    region: DeploymentRegion
    status: str  # healthy, degraded, unavailable
    active_clients: int
    average_latency_ms: float
    cpu_utilization: float
    memory_utilization: float
    network_throughput_mbps: float
    error_rate: float
    last_update: datetime


class GeoLocationService:
    """Service for determining client geographical locations and regulations."""
    
    def __init__(self):
        """Initialize geo-location service."""
        # Mock geo-location database
        self.country_regulations = {
            "US": [PrivacyRegulation.CCPA],
            "CA": [PrivacyRegulation.PIPEDA],
            "GB": [PrivacyRegulation.GDPR],
            "DE": [PrivacyRegulation.GDPR],
            "FR": [PrivacyRegulation.GDPR],
            "ES": [PrivacyRegulation.GDPR],
            "IT": [PrivacyRegulation.GDPR],
            "NL": [PrivacyRegulation.GDPR],
            "SG": [PrivacyRegulation.PDPA_SINGAPORE],
            "JP": [PrivacyRegulation.APPI],
            "BR": [PrivacyRegulation.LGPD],
            "AU": [PrivacyRegulation.PRIVACY_ACT]
        }
        
        self.region_mapping = {
            "US": [DeploymentRegion.US_EAST, DeploymentRegion.US_WEST, DeploymentRegion.US_CENTRAL],
            "CA": [DeploymentRegion.CANADA_CENTRAL],
            "GB": [DeploymentRegion.UK_SOUTH],
            "DE": [DeploymentRegion.EU_CENTRAL],
            "FR": [DeploymentRegion.EU_WEST],
            "NL": [DeploymentRegion.EU_WEST],
            "SG": [DeploymentRegion.ASIA_SOUTHEAST],
            "JP": [DeploymentRegion.JAPAN_EAST],
            "BR": [DeploymentRegion.BRAZIL_SOUTH],
            "AU": [DeploymentRegion.AUSTRALIA_SOUTHEAST]
        }
        
        self.residency_zones = {
            "US": DataResidencyZone.US_CCPA,
            "CA": DataResidencyZone.CANADA_PIPEDA,
            "GB": DataResidencyZone.EU_GDPR,
            "DE": DataResidencyZone.EU_GDPR,
            "FR": DataResidencyZone.EU_GDPR,
            "ES": DataResidencyZone.EU_GDPR,
            "IT": DataResidencyZone.EU_GDPR,
            "NL": DataResidencyZone.EU_GDPR,
            "SG": DataResidencyZone.SINGAPORE_PDPA,
            "JP": DataResidencyZone.JAPAN_APPI,
            "BR": DataResidencyZone.BRAZIL_LGPD,
            "AU": DataResidencyZone.AUSTRALIA_PRIVACY
        }
        
        logger.info("Geo-location service initialized")
    
    def get_client_location(self, client_ip: str, client_id: str) -> ClientGeoLocation:
        """Get client geographical location and applicable regulations."""
        # Mock implementation - in production would use IP geolocation service
        mock_locations = {
            "192.168.1.1": {"country": "US", "city": "New York", "lat": 40.7128, "lon": -74.0060},
            "10.0.0.1": {"country": "GB", "city": "London", "lat": 51.5074, "lon": -0.1278},
            "172.16.0.1": {"country": "SG", "city": "Singapore", "lat": 1.3521, "lon": 103.8198},
        }
        
        # Default to US for unknown IPs
        location_data = mock_locations.get(client_ip, {"country": "US", "city": "Unknown", "lat": 0.0, "lon": 0.0})
        
        country_code = location_data["country"]
        regulations = self.country_regulations.get(country_code, [PrivacyRegulation.GDPR])  # Default to GDPR
        regions = self.region_mapping.get(country_code, [DeploymentRegion.US_EAST])
        residency_zone = self.residency_zones.get(country_code, DataResidencyZone.GLOBAL_STANDARD)
        
        return ClientGeoLocation(
            client_id=client_id,
            country_code=country_code,
            region_code=country_code.lower(),
            city=location_data["city"],
            latitude=location_data["lat"],
            longitude=location_data["lon"],
            timezone="UTC",  # Simplified
            assigned_region=regions[0],  # Assign to first available region
            data_residency_zone=residency_zone,
            applicable_regulations=regulations
        )
    
    def calculate_optimal_region(
        self,
        client_location: ClientGeoLocation,
        available_regions: List[DeploymentRegion],
        region_health: Dict[DeploymentRegion, RegionHealthMetrics]
    ) -> DeploymentRegion:
        """Calculate optimal deployment region for a client."""
        # Mock distance calculation (in production would use haversine formula)
        region_distances = {
            DeploymentRegion.US_EAST: 1000 if client_location.country_code == "US" else 8000,
            DeploymentRegion.EU_WEST: 1000 if client_location.country_code in ["GB", "FR", "NL"] else 8000,
            DeploymentRegion.ASIA_SOUTHEAST: 500 if client_location.country_code == "SG" else 9000,
        }
        
        # Score regions based on distance, health, and compliance
        region_scores = {}
        
        for region in available_regions:
            if region not in region_distances:
                continue
            
            distance = region_distances[region]
            health = region_health.get(region)
            
            if not health or health.status != "healthy":
                continue
            
            # Calculate score (lower is better)
            distance_score = distance / 1000.0  # Normalize
            latency_score = health.average_latency_ms / 100.0
            load_score = (health.cpu_utilization + health.memory_utilization) / 200.0
            
            total_score = distance_score + latency_score + load_score
            region_scores[region] = total_score
        
        # Return region with lowest score
        if region_scores:
            return min(region_scores.keys(), key=region_scores.get)
        
        # Fallback to assigned region
        return client_location.assigned_region


class RegionManager:
    """Manager for multi-region deployment operations."""
    
    def __init__(self, config: FederatedConfig):
        """Initialize region manager."""
        self.config = config
        self.geo_service = GeoLocationService()
        
        # Regional configurations
        self.region_configs: Dict[DeploymentRegion, RegionConfig] = {}
        self.active_regions: Set[DeploymentRegion] = set()
        
        # Regional servers
        self.regional_servers: Dict[DeploymentRegion, FederatedServer] = {}
        self.region_health: Dict[DeploymentRegion, RegionHealthMetrics] = {}
        
        # Client assignments
        self.client_regions: Dict[str, DeploymentRegion] = {}
        self.client_locations: Dict[str, ClientGeoLocation] = {}
        
        # Initialize default region configurations
        self._initialize_region_configs()
        
        logger.info("Multi-region manager initialized")
    
    def _initialize_region_configs(self):
        """Initialize default configurations for all regions."""
        self.region_configs = {
            DeploymentRegion.US_EAST: RegionConfig(
                region=DeploymentRegion.US_EAST,
                residency_zone=DataResidencyZone.US_CCPA,
                primary_regulations=[PrivacyRegulation.CCPA],
                supported_languages=["en", "es"],
                data_centers=["us-east-1a", "us-east-1b", "us-east-1c"],
                latency_sla_ms=50,
                availability_sla=99.95,
                backup_regions=[DeploymentRegion.US_WEST, DeploymentRegion.US_CENTRAL]
            ),
            
            DeploymentRegion.EU_WEST: RegionConfig(
                region=DeploymentRegion.EU_WEST,
                residency_zone=DataResidencyZone.EU_GDPR,
                primary_regulations=[PrivacyRegulation.GDPR],
                supported_languages=["en", "fr", "de", "es"],
                data_centers=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
                latency_sla_ms=30,
                availability_sla=99.95,
                backup_regions=[DeploymentRegion.EU_CENTRAL, DeploymentRegion.EU_NORTH]
            ),
            
            DeploymentRegion.ASIA_SOUTHEAST: RegionConfig(
                region=DeploymentRegion.ASIA_SOUTHEAST,
                residency_zone=DataResidencyZone.SINGAPORE_PDPA,
                primary_regulations=[PrivacyRegulation.PDPA_SINGAPORE],
                supported_languages=["en", "zh", "ja"],
                data_centers=["ap-southeast-1a", "ap-southeast-1b"],
                latency_sla_ms=40,
                availability_sla=99.9,
                backup_regions=[DeploymentRegion.ASIA_EAST, DeploymentRegion.JAPAN_EAST]
            )
        }
    
    def activate_region(self, region: DeploymentRegion) -> bool:
        """Activate a deployment region."""
        try:
            if region not in self.region_configs:
                logger.error(f"Region {region.value} not configured")
                return False
            
            # Create regional server
            regional_config = self._create_regional_config(region)
            regional_server = FederatedServer(
                model_name=self.config.model_name,
                config=regional_config,
                host="0.0.0.0",
                port=self._get_regional_port(region)
            )
            
            self.regional_servers[region] = regional_server
            self.active_regions.add(region)
            
            # Initialize health metrics
            self.region_health[region] = RegionHealthMetrics(
                region=region,
                status="healthy",
                active_clients=0,
                average_latency_ms=0.0,
                cpu_utilization=0.0,
                memory_utilization=0.0,
                network_throughput_mbps=0.0,
                error_rate=0.0,
                last_update=datetime.now(timezone.utc)
            )
            
            logger.info(f"Activated region: {region.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate region {region.value}: {e}")
            return False
    
    def _create_regional_config(self, region: DeploymentRegion) -> FederatedConfig:
        """Create region-specific federated configuration."""
        region_config = self.region_configs[region]
        
        # Clone base config and customize for region
        regional_fed_config = FederatedConfig(
            model_name=self.config.model_name,
            num_rounds=self.config.num_rounds,
            local_epochs=self.config.local_epochs,
            privacy=self.config.privacy,
            lora=self.config.lora,
            security=self.config.security
        )
        
        # Regional customizations
        if region_config.residency_zone == DataResidencyZone.EU_GDPR:
            # Stricter privacy settings for GDPR
            regional_fed_config.privacy.epsilon = min(self.config.privacy.epsilon, 4.0)
            regional_fed_config.privacy.delta = min(self.config.privacy.delta, 1e-6)
        
        return regional_fed_config
    
    def _get_regional_port(self, region: DeploymentRegion) -> int:
        """Get port number for regional server."""
        port_mapping = {
            DeploymentRegion.US_EAST: 8080,
            DeploymentRegion.US_WEST: 8081,
            DeploymentRegion.EU_WEST: 8082,
            DeploymentRegion.EU_CENTRAL: 8083,
            DeploymentRegion.ASIA_SOUTHEAST: 8084
        }
        return port_mapping.get(region, 8080)
    
    async def assign_client_to_region(
        self,
        client_id: str,
        client_ip: str
    ) -> Tuple[DeploymentRegion, ClientGeoLocation]:
        """Assign a client to the optimal region."""
        # Get client location
        client_location = self.geo_service.get_client_location(client_ip, client_id)
        
        # Find optimal region
        optimal_region = self.geo_service.calculate_optimal_region(
            client_location,
            list(self.active_regions),
            self.region_health
        )
        
        # Verify region is active and healthy
        if optimal_region not in self.active_regions:
            # Fallback to first active region
            optimal_region = next(iter(self.active_regions))
        
        if optimal_region not in self.region_health or self.region_health[optimal_region].status != "healthy":
            # Find healthy backup region
            for backup_region in self.region_configs[optimal_region].backup_regions:
                if (backup_region in self.active_regions and 
                    backup_region in self.region_health and 
                    self.region_health[backup_region].status == "healthy"):
                    optimal_region = backup_region
                    break
        
        # Update client assignment
        client_location.assigned_region = optimal_region
        self.client_regions[client_id] = optimal_region
        self.client_locations[client_id] = client_location
        
        logger.info(f"Assigned client {client_id} ({client_location.country_code}) to region {optimal_region.value}")
        
        return optimal_region, client_location
    
    async def cross_region_sync(self, source_region: DeploymentRegion, target_regions: List[DeploymentRegion]):
        """Synchronize model updates across regions."""
        if source_region not in self.regional_servers:
            logger.error(f"Source region {source_region.value} not active")
            return
        
        source_server = self.regional_servers[source_region]
        
        # Get global model from source
        try:
            global_params = source_server.get_global_parameters()
            
            # Sync to target regions
            sync_tasks = []
            for target_region in target_regions:
                if target_region in self.regional_servers:
                    task = self._sync_to_region(target_region, global_params)
                    sync_tasks.append(task)
            
            if sync_tasks:
                await asyncio.gather(*sync_tasks)
                logger.info(f"Synchronized model from {source_region.value} to {len(sync_tasks)} regions")
        
        except Exception as e:
            logger.error(f"Cross-region sync failed: {e}")
    
    async def _sync_to_region(self, region: DeploymentRegion, global_params: Dict[str, Any]):
        """Sync global parameters to a specific region."""
        try:
            regional_server = self.regional_servers[region]
            # In a real implementation, this would use secure API calls
            # regional_server.update_global_model(global_params)
            logger.debug(f"Synced model to region {region.value}")
        except Exception as e:
            logger.error(f"Failed to sync to region {region.value}: {e}")
    
    def update_region_health(self, region: DeploymentRegion, metrics: Dict[str, Any]):
        """Update health metrics for a region."""
        if region not in self.region_health:
            return
        
        health = self.region_health[region]
        health.active_clients = metrics.get("active_clients", health.active_clients)
        health.average_latency_ms = metrics.get("latency_ms", health.average_latency_ms)
        health.cpu_utilization = metrics.get("cpu_utilization", health.cpu_utilization)
        health.memory_utilization = metrics.get("memory_utilization", health.memory_utilization)
        health.network_throughput_mbps = metrics.get("network_mbps", health.network_throughput_mbps)
        health.error_rate = metrics.get("error_rate", health.error_rate)
        health.last_update = datetime.now(timezone.utc)
        
        # Determine health status
        if (health.cpu_utilization > 90 or health.memory_utilization > 90 or 
            health.error_rate > 5.0 or health.average_latency_ms > 1000):
            health.status = "degraded"
        elif (health.cpu_utilization > 95 or health.memory_utilization > 95 or 
              health.error_rate > 10.0):
            health.status = "unavailable"
        else:
            health.status = "healthy"
    
    async def handle_region_failure(self, failed_region: DeploymentRegion):
        """Handle region failure and failover."""
        logger.warning(f"Handling failure for region {failed_region.value}")
        
        if failed_region not in self.region_configs:
            return
        
        # Get clients assigned to failed region
        failed_clients = [
            client_id for client_id, region in self.client_regions.items()
            if region == failed_region
        ]
        
        # Get backup regions
        backup_regions = self.region_configs[failed_region].backup_regions
        healthy_backups = [
            region for region in backup_regions
            if (region in self.active_regions and 
                region in self.region_health and 
                self.region_health[region].status == "healthy")
        ]
        
        if not healthy_backups:
            logger.error(f"No healthy backup regions available for {failed_region.value}")
            return
        
        # Reassign clients to backup regions
        reassignment_tasks = []
        for i, client_id in enumerate(failed_clients):
            backup_region = healthy_backups[i % len(healthy_backups)]  # Round-robin
            
            # Update client assignment
            self.client_regions[client_id] = backup_region
            if client_id in self.client_locations:
                self.client_locations[client_id].assigned_region = backup_region
            
            logger.info(f"Reassigned client {client_id} from {failed_region.value} to {backup_region.value}")
        
        # Mark region as unavailable
        if failed_region in self.region_health:
            self.region_health[failed_region].status = "unavailable"
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive multi-region deployment status."""
        status = {
            "total_regions": len(self.region_configs),
            "active_regions": len(self.active_regions),
            "healthy_regions": len([
                r for r in self.region_health.values()
                if r.status == "healthy"
            ]),
            "total_clients": len(self.client_regions),
            "regions": {}
        }
        
        for region in self.active_regions:
            region_clients = len([
                c for c, r in self.client_regions.items()
                if r == region
            ])
            
            health = self.region_health.get(region)
            region_config = self.region_configs.get(region)
            
            status["regions"][region.value] = {
                "status": health.status if health else "unknown",
                "clients": region_clients,
                "data_residency_zone": region_config.residency_zone.value if region_config else "unknown",
                "regulations": [r.value for r in region_config.primary_regulations] if region_config else [],
                "latency_ms": health.average_latency_ms if health else 0,
                "availability_sla": region_config.availability_sla if region_config else 0
            }
        
        return status
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary across all regions."""
        regulations_by_region = {}
        client_distribution = {}
        
        for region, config in self.region_configs.items():
            if region in self.active_regions:
                regulations_by_region[region.value] = {
                    "residency_zone": config.residency_zone.value,
                    "regulations": [r.value for r in config.primary_regulations],
                    "supported_languages": config.supported_languages
                }
        
        for client_id, location in self.client_locations.items():
            country = location.country_code
            if country not in client_distribution:
                client_distribution[country] = {
                    "count": 0,
                    "assigned_regions": set(),
                    "regulations": set()
                }
            
            client_distribution[country]["count"] += 1
            client_distribution[country]["assigned_regions"].add(location.assigned_region.value)
            client_distribution[country]["regulations"].update([r.value for r in location.applicable_regulations])
        
        # Convert sets to lists for JSON serialization
        for country_data in client_distribution.values():
            country_data["assigned_regions"] = list(country_data["assigned_regions"])
            country_data["regulations"] = list(country_data["regulations"])
        
        return {
            "regulations_by_region": regulations_by_region,
            "client_distribution": client_distribution,
            "total_jurisdictions": len(client_distribution),
            "compliance_zones": len(set(
                config.residency_zone.value 
                for config in self.region_configs.values()
                if config.region in self.active_regions
            ))
        }


def create_multi_region_manager(config: FederatedConfig) -> RegionManager:
    """Create multi-region deployment manager."""
    return RegionManager(config)


async def deploy_global_federation(
    config: FederatedConfig,
    target_regions: List[DeploymentRegion]
) -> RegionManager:
    """Deploy federated learning system across multiple regions."""
    manager = create_multi_region_manager(config)
    
    # Activate target regions
    activation_tasks = []
    for region in target_regions:
        activated = manager.activate_region(region)
        if activated:
            logger.info(f"Successfully activated region: {region.value}")
        else:
            logger.error(f"Failed to activate region: {region.value}")
    
    logger.info(f"Global federation deployed across {len(manager.active_regions)} regions")
    
    return manager
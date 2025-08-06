terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Multi-region deployment configuration
locals {
  regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
  
  common_tags = {
    Project     = "dp-federated-lora"
    Environment = var.environment
    Owner       = "terragon-labs"
    Compliance  = "gdpr-ccpa-ready"
  }
}

# AWS Provider configurations for multi-region
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = local.common_tags
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = local.common_tags
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  
  default_tags {
    tags = local.common_tags
  }
}

# Global Infrastructure
module "global_infrastructure" {
  source = "./modules/global"
  
  project_name = "dp-federated-lora"
  environment  = var.environment
  
  # Global DNS and CDN
  domain_name = var.domain_name
  
  # Global security and compliance
  enable_waf           = true
  enable_shield        = true
  enable_cloudtrail    = true
  enable_config        = true
  
  # Privacy and compliance
  gdpr_compliance     = true
  ccpa_compliance     = true
  data_residency_regions = local.regions
  
  tags = local.common_tags
}

# Regional EKS clusters
module "eks_us_east_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.us_east_1
  }
  
  cluster_name    = "dp-federated-lora-us-east-1"
  cluster_version = "1.28"
  region          = "us-east-1"
  environment     = var.environment
  
  # Node groups
  node_groups = {
    federated_servers = {
      instance_types = ["m5.xlarge", "m5.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      
      labels = {
        role = "federated-server"
      }
      
      taints = []
    }
    
    privacy_workers = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      min_size       = 1
      max_size       = 5
      desired_size   = 2
      
      labels = {
        role = "privacy-computation"
      }
      
      taints = [
        {
          key    = "privacy-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Network configuration
  vpc_cidr = "10.1.0.0/16"
  
  # Security and compliance
  enable_encryption         = true
  enable_logging           = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  enable_private_endpoint  = true
  
  tags = local.common_tags
}

module "eks_eu_west_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  cluster_name    = "dp-federated-lora-eu-west-1"
  cluster_version = "1.28"
  region          = "eu-west-1"
  environment     = var.environment
  
  node_groups = {
    federated_servers = {
      instance_types = ["m5.xlarge", "m5.2xlarge"]
      min_size       = 2
      max_size       = 8
      desired_size   = 3
    }
  }
  
  vpc_cidr = "10.2.0.0/16"
  
  # GDPR-specific configuration
  enable_encryption        = true
  enable_logging          = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  data_residency_eu       = true
  
  tags = merge(local.common_tags, {
    DataResidency = "EU"
    GDPRCompliant = "true"
  })
}

module "eks_ap_southeast_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.ap_southeast_1
  }
  
  cluster_name    = "dp-federated-lora-ap-southeast-1"
  cluster_version = "1.28"
  region          = "ap-southeast-1"
  environment     = var.environment
  
  node_groups = {
    federated_servers = {
      instance_types = ["m5.large", "m5.xlarge"]
      min_size       = 1
      max_size       = 6
      desired_size   = 2
    }
  }
  
  vpc_cidr = "10.3.0.0/16"
  
  enable_encryption = true
  enable_logging    = ["api", "audit"]
  
  tags = merge(local.common_tags, {
    DataResidency = "APAC"
  })
}

# Global database with read replicas
module "global_database" {
  source = "./modules/database"
  
  providers = {
    aws.primary = aws.us_east_1
    aws.replica_eu = aws.eu_west_1
    aws.replica_ap = aws.ap_southeast_1
  }
  
  db_name     = "federated-lora-db"
  environment = var.environment
  
  # Primary instance configuration
  primary_region        = "us-east-1"
  instance_class        = "db.r6g.xlarge"
  allocated_storage     = 100
  max_allocated_storage = 1000
  
  # Read replicas
  read_replicas = {
    eu_west_1 = {
      instance_class = "db.r6g.large"
      region        = "eu-west-1"
    }
    ap_southeast_1 = {
      instance_class = "db.r6g.large" 
      region        = "ap-southeast-1"
    }
  }
  
  # Backup and security
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enable_encryption     = true
  enable_performance_insights = true
  
  # Privacy compliance
  enable_deletion_protection = true
  enable_point_in_time_recovery = true
  
  tags = local.common_tags
}

# Global Redis cluster
module "global_cache" {
  source = "./modules/cache"
  
  cache_name = "federated-lora-cache"
  environment = var.environment
  
  # Multi-region configuration
  regions = local.regions
  
  # Cache configuration
  node_type               = "cache.r6g.large"
  num_cache_clusters      = 3
  parameter_group_name    = "default.redis7"
  port                    = 6379
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  tags = local.common_tags
}

# Monitoring and observability
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name = "dp-federated-lora"
  environment  = var.environment
  regions      = local.regions
  
  # EKS clusters to monitor
  eks_clusters = [
    module.eks_us_east_1.cluster_name,
    module.eks_eu_west_1.cluster_name,
    module.eks_ap_southeast_1.cluster_name,
  ]
  
  # Metrics and alerting
  enable_container_insights = true
  enable_service_map       = true
  
  # Privacy-specific monitoring
  privacy_metrics_enabled = true
  epsilon_budget_alerts   = true
  
  # Compliance monitoring
  compliance_monitoring = {
    gdpr_audit_logs    = true
    ccpa_data_tracking = true
    retention_policies = true
  }
  
  tags = local.common_tags
}

# Security and compliance
module "security" {
  source = "./modules/security"
  
  project_name = "dp-federated-lora"
  environment  = var.environment
  
  # WAF and DDoS protection
  enable_waf    = true
  enable_shield = true
  
  # Certificate management
  domain_name = var.domain_name
  
  # Secrets management
  enable_secrets_manager = true
  
  # Network security
  enable_vpc_flow_logs = true
  enable_guardduty     = true
  
  # Compliance
  enable_config       = true
  enable_cloudtrail   = true
  enable_security_hub = true
  
  # Privacy-specific security
  data_encryption_at_rest    = true
  data_encryption_in_transit = true
  privacy_budget_protection  = true
  
  tags = local.common_tags
}

# Outputs
output "eks_clusters" {
  description = "EKS cluster endpoints"
  value = {
    us_east_1      = module.eks_us_east_1.cluster_endpoint
    eu_west_1      = module.eks_eu_west_1.cluster_endpoint
    ap_southeast_1 = module.eks_ap_southeast_1.cluster_endpoint
  }
}

output "database_endpoints" {
  description = "Database connection endpoints"
  value = {
    primary_endpoint = module.global_database.primary_endpoint
    read_endpoints   = module.global_database.read_replica_endpoints
  }
  sensitive = true
}

output "load_balancer_dns" {
  description = "Global load balancer DNS names"
  value = module.global_infrastructure.load_balancer_dns
}

output "monitoring_dashboard_url" {
  description = "Monitoring dashboard URL"
  value = module.monitoring.dashboard_url
}
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "domain_name" {
  description = "Domain name for the federated learning service"
  type        = string
  default     = "federated.terragonlabs.com"
}

variable "aws_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS clusters"
  type        = string
  default     = "1.28"
}

variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = true
}

variable "enable_gdpr_compliance" {
  description = "Enable GDPR compliance features"
  type        = bool
  default     = true
}

variable "enable_ccpa_compliance" {
  description = "Enable CCPA compliance features"
  type        = bool
  default     = true
}

variable "privacy_budget_limit" {
  description = "Global privacy budget limit (epsilon)"
  type        = number
  default     = 10.0
  
  validation {
    condition     = var.privacy_budget_limit > 0 && var.privacy_budget_limit <= 50
    error_message = "Privacy budget limit must be between 0 and 50."
  }
}

variable "max_clients_per_region" {
  description = "Maximum number of federated learning clients per region"
  type        = number
  default     = 1000
}

variable "database_backup_retention_days" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.database_backup_retention_days >= 7 && var.database_backup_retention_days <= 35
    error_message = "Database backup retention must be between 7 and 35 days."
  }
}

variable "monitoring_retention_days" {
  description = "Monitoring data retention period in days"
  type        = number
  default     = 90
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for federated learning servers"
  type        = bool
  default     = true
}

variable "min_replicas" {
  description = "Minimum number of federated server replicas per region"
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "Maximum number of federated server replicas per region"
  type        = number
  default     = 20
}

variable "enable_disaster_recovery" {
  description = "Enable disaster recovery across regions"
  type        = bool
  default     = true
}

variable "enable_data_residency" {
  description = "Enable data residency controls"
  type        = bool
  default     = true
}

variable "compliance_tags" {
  description = "Compliance and regulatory tags"
  type = object({
    data_classification = string
    retention_policy    = string
    privacy_level       = string
    regulatory_scope    = list(string)
  })
  
  default = {
    data_classification = "confidential"
    retention_policy    = "7-years"
    privacy_level       = "high"
    regulatory_scope    = ["GDPR", "CCPA", "PDPA"]
  }
}

variable "network_security" {
  description = "Network security configuration"
  type = object({
    enable_vpc_flow_logs        = bool
    enable_network_segmentation = bool
    enable_private_endpoints    = bool
    allowed_cidr_blocks        = list(string)
  })
  
  default = {
    enable_vpc_flow_logs        = true
    enable_network_segmentation = true
    enable_private_endpoints    = true
    allowed_cidr_blocks        = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
  }
}

variable "encryption_config" {
  description = "Encryption configuration"
  type = object({
    kms_key_rotation_enabled = bool
    encryption_at_rest       = bool
    encryption_in_transit    = bool
    use_customer_managed_keys = bool
  })
  
  default = {
    kms_key_rotation_enabled = true
    encryption_at_rest       = true
    encryption_in_transit    = true
    use_customer_managed_keys = true
  }
}

variable "privacy_settings" {
  description = "Differential privacy settings"
  type = object({
    default_epsilon              = number
    default_delta               = number
    max_queries_per_client      = number
    privacy_budget_refresh_hours = number
    enable_privacy_amplification = bool
  })
  
  default = {
    default_epsilon              = 1.0
    default_delta               = 1e-5
    max_queries_per_client      = 100
    privacy_budget_refresh_hours = 24
    enable_privacy_amplification = true
  }
}

variable "resource_limits" {
  description = "Resource limits for different components"
  type = object({
    federated_server = object({
      cpu_limit    = string
      memory_limit = string
      cpu_request  = string
      memory_request = string
    })
    privacy_engine = object({
      cpu_limit    = string
      memory_limit = string
      cpu_request  = string
      memory_request = string
    })
    monitoring = object({
      cpu_limit    = string
      memory_limit = string
      cpu_request  = string
      memory_request = string
    })
  })
  
  default = {
    federated_server = {
      cpu_limit      = "2000m"
      memory_limit   = "4Gi"
      cpu_request    = "500m"
      memory_request = "1Gi"
    }
    privacy_engine = {
      cpu_limit      = "4000m"
      memory_limit   = "8Gi"
      cpu_request    = "1000m"
      memory_request = "2Gi"
    }
    monitoring = {
      cpu_limit      = "500m"
      memory_limit   = "1Gi"
      cpu_request    = "100m"
      memory_request = "256Mi"
    }
  }
}

variable "alerts_config" {
  description = "Alerting configuration"
  type = object({
    slack_webhook_url     = string
    email_notifications   = list(string)
    pagerduty_service_key = string
    enable_sms_alerts     = bool
  })
  
  default = {
    slack_webhook_url     = ""
    email_notifications   = ["admin@terragonlabs.com"]
    pagerduty_service_key = ""
    enable_sms_alerts     = false
  }
}

variable "maintenance_window" {
  description = "Maintenance window configuration"
  type = object({
    day_of_week   = string
    start_time    = string
    duration_hours = number
  })
  
  default = {
    day_of_week   = "sunday"
    start_time    = "03:00"
    duration_hours = 4
  }
}

variable "cost_optimization" {
  description = "Cost optimization settings"
  type = object({
    enable_spot_instances    = bool
    spot_instance_percentage = number
    enable_scheduled_scaling = bool
    off_hours_min_replicas  = number
  })
  
  default = {
    enable_spot_instances    = true
    spot_instance_percentage = 30
    enable_scheduled_scaling = true
    off_hours_min_replicas  = 1
  }
}

# Localization and internationalization
variable "i18n_config" {
  description = "Internationalization configuration"
  type = object({
    supported_languages = list(string)
    default_language    = string
    enable_rtl_support  = bool
  })
  
  default = {
    supported_languages = ["en", "es", "fr", "de", "ja", "zh"]
    default_language    = "en"
    enable_rtl_support  = false
  }
}
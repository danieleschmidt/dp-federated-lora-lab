
# Terraform configuration for production environment
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "dp-federated-lora-terraform-state"
    key    = "production/eu-central-1/terraform.tfstate"
    region = "eu-central-1"
  }
}

# Provider configurations
provider "aws" {
  region = "eu-central-1"
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "dp-federated-lora"
      ManagedBy   = "terraform"
    }
  }
}

# VPC and networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "dp-federated-lora-production"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "dp-federated-lora-production-igw"
  }
}

# Subnets
resource "aws_subnet" "public" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "dp-federated-lora-production-public-${count.index + 1}"
    Type = "public"
  }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "dp-federated-lora-production-private-${count.index + 1}"
    Type = "private"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "dp-federated-lora-production"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = false
    
    public_access_cidrs = ["0.0.0.0/0"]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
    aws_cloudwatch_log_group.eks,
  ]
  
  tags = {
    Environment = "production"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "dp-federated-lora-production-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["c5.xlarge"]
  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"
  
  scaling_config {
    desired_size = 5
    max_size     = 15
    min_size     = 2
  }
  
  update_config {
    max_unavailable_percentage = 25
  }
  
  remote_access {
    ec2_ssh_key = aws_key_pair.eks_nodes.key_name
    source_security_group_ids = [aws_security_group.eks_remote_access.id]
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
  
  tags = {
    Environment = "production"
  }
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name        = "dp-federated-lora-production-cluster-sg"
  description = "Security group for EKS cluster"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "dp-federated-lora-production-cluster-sg"
  }
}

# RDS for persistent storage
resource "aws_db_instance" "main" {
  identifier = "dp-federated-lora-production"
  
  engine         = "postgresql"
  engine_version = "14.9"
  instance_class = "db.r5.large"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn
  
  db_name  = "dpfederatedlora"
  username = "federated_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
  
  tags = {
    Environment = "production"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
}

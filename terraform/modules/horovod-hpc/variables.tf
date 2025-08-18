# ============================================================================
# AWS Horovod HPC Cluster Variables
# ============================================================================

# ============================================================================
# General Configuration
# ============================================================================

variable "cluster_name" {
  description = "Name of the HPC cluster"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9-]*$", var.cluster_name))
    error_message = "Cluster name must start with a letter and contain only letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "arn_prefix" {
  description = "AWS ARN prefix (use 'arn:aws-us-gov' for GovCloud regions)"
  type        = string
  default     = "arn:aws"
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# ============================================================================
# Compute Configuration
# ============================================================================

variable "instance_type" {
  description = "EC2 instance type for HPC nodes"
  type        = string
  default     = "p4d.24xlarge"

  validation {
    condition = contains([
      "p3.2xlarge", "p3.8xlarge", "p3.16xlarge", "p3dn.24xlarge",
      "p4d.24xlarge", "p4de.24xlarge",
      "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge", "g4dn.8xlarge", "g4dn.12xlarge", "g4dn.16xlarge"
    ], var.instance_type)
    error_message = "Instance type must be a GPU-enabled instance type."
  }
}

variable "instance_count" {
  description = "Number of EC2 instances in the cluster"
  type        = number
  default     = 2

  validation {
    condition     = var.instance_count >= 1 && var.instance_count <= 100
    error_message = "Instance count must be between 1 and 100."
  }
}

variable "ami_id" {
  description = "AMI ID to use for the instances"
  type        = string
  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
  default     = "ami-0f20cc6143e3cdb84"
}

# ============================================================================
# Networking Configuration
# ============================================================================

variable "create_vpc" {
  description = "Whether to create a new VPC"
  type        = bool
  default     = true
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "private_subnet_cidr" {
  description = "CIDR block for private subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "existing_vpc_id" {
  description = "ID of existing VPC (used when create_vpc is false)"
  type        = string
  default     = ""
}

variable "existing_public_subnet_id" {
  description = "ID of existing public subnet (used when create_vpc is false)"
  type        = string
  default     = ""
}

variable "existing_private_subnet_id" {
  description = "ID of existing private subnet (used when create_vpc is false)"
  type        = string
  default     = ""
}

variable "use_public_subnet" {
  description = "Whether to deploy instances in public subnet (true) or private subnet (false)"
  type        = bool
  default     = true
}

variable "create_nat_gateway" {
  description = "Whether to create NAT gateway for private subnet internet access"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ============================================================================
# High-Performance Networking
# ============================================================================

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter (EFA) for high-performance networking"
  type        = bool
  default     = true
}

variable "placement_group_strategy" {
  description = "Placement group strategy (cluster, partition, spread)"
  type        = string
  default     = "cluster"

  validation {
    condition     = contains(["cluster", "partition", "spread"], var.placement_group_strategy)
    error_message = "Placement group strategy must be cluster, partition, or spread."
  }
}

variable "nccl_port_range" {
  description = "Port ranges for NCCL communication"
  type = list(object({
    from = number
    to   = number
  }))
  default = [
    {
      from = 61000
      to   = 61999
    }
  ]
}

# ============================================================================
# SSH Configuration
# ============================================================================

variable "create_ssh_key" {
  description = "Whether to create a new SSH key pair"
  type        = bool
  default     = true
}

variable "existing_key_pair_name" {
  description = "Name of existing EC2 key pair (used when create_ssh_key is false)"
  type        = string
  default     = ""
}

variable "existing_private_key_path" {
  description = "Path to existing private key file (used when create_ssh_key is false)"
  type        = string
  default     = ""
}

variable "setup_passwordless_ssh" {
  description = "Setup passwordless SSH between cluster nodes"
  type        = bool
  default     = true
}

variable "save_private_key_locally" {
  description = "Whether to save the generated private key to a local file (in addition to Secrets Manager)"
  type        = bool
  default     = false
}

# ============================================================================
# Storage Configuration
# ============================================================================

variable "root_volume_type" {
  description = "Root volume type"
  type        = string
  default     = "gp3"

  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.root_volume_type)
    error_message = "Root volume type must be gp2, gp3, io1, or io2."
  }
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 200
}

variable "root_volume_iops" {
  description = "Root volume IOPS (for gp3, io1, io2)"
  type        = number
  default     = 3000
}

variable "root_volume_throughput" {
  description = "Root volume throughput in MB/s (for gp3)"
  type        = number
  default     = 125
}

variable "enable_ebs_encryption" {
  description = "Enable EBS volume encryption"
  type        = bool
  default     = true
}

variable "additional_ebs_volumes" {
  description = "Additional EBS volumes to attach"
  type = list(object({
    device_name = string
    volume_type = string
    volume_size = number
    iops        = optional(number)
    throughput  = optional(number)
  }))
  default = [
    {
      device_name = "/dev/sdf"
      volume_type = "gp3"
      volume_size = 500
      iops        = 3000
      throughput  = 125
    }
  ]
}

variable "shared_storage_mount_point" {
  description = "Mount point for shared storage"
  type        = string
  default     = "/shared"
}

# ============================================================================
# Software Configuration
# ============================================================================

variable "horovod_version" {
  description = "Horovod version to install"
  type        = string
  default     = "0.28.1"
}

variable "nccl_version" {
  description = "NCCL version to install"
  type        = string
  default     = "2.15.5"
}

variable "cuda_version" {
  description = "CUDA version"
  type        = string
  default     = "11.8"
}

# ============================================================================
# Monitoring and Observability
# ============================================================================

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring for EC2 instances"
  type        = bool
  default     = false
}

# ============================================================================
# IAM and Security
# ============================================================================

variable "s3_bucket_arn" {
  description = "ARN of S3 bucket for storing training data/checkpoints"
  type        = string
  default     = ""
}

variable "secret_recovery_window_days" {
  description = "Number of days to retain secret in Secrets Manager after deletion (0 to force immediate deletion)"
  type        = number
  default     = 7

  validation {
    condition     = var.secret_recovery_window_days >= 0 && var.secret_recovery_window_days <= 30
    error_message = "Secret recovery window must be between 0 and 30 days."
  }
}

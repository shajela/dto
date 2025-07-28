# ============================================================================
# Local Variables
# ============================================================================

locals {
  cluster_name = var.cluster_name

  # Common tags
  common_tags = merge(var.tags, {
    environment = var.environment
    project     = "horovod-hpc"
    managed_by  = "terraform"
    cluster_name = local.cluster_name
  })

  # Instance configuration
  gpu_count_map = {
    "p3.2xlarge"    = 1
    "p3.8xlarge"    = 4
    "p3.16xlarge"   = 8
    "p3dn.24xlarge" = 8
    "p4d.24xlarge"  = 8
    "p4de.24xlarge" = 8
    "g4dn.xlarge"   = 1
    "g4dn.2xlarge"  = 1
    "g4dn.4xlarge"  = 1
    "g4dn.8xlarge"  = 1
    "g4dn.12xlarge" = 4
    "g4dn.16xlarge" = 1
  }

  gpus_per_instance = lookup(local.gpu_count_map, var.instance_type, 1)
  total_gpus        = var.instance_count * local.gpus_per_instance

  # User data for instance initialization
  user_data = base64encode(templatefile("${path.module}/scripts/setup.sh", {
    cluster_name         = local.cluster_name
    region               = var.region
    enable_efa           = var.enable_efa
    horovod_version      = var.horovod_version
    nccl_version         = var.nccl_version
    cuda_version         = var.cuda_version
    instance_type        = var.instance_type
    enable_monitoring    = var.enable_monitoring
    shared_storage_mount = var.shared_storage_mount_point
    setup_ssh_keys       = var.setup_passwordless_ssh
  }))
}

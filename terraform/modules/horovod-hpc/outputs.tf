# ============================================================================
# AWS Horovod HPC Cluster Outputs
# ============================================================================

# ============================================================================
# Cluster Information
# ============================================================================

output "cluster_name" {
  description = "Name of the HPC cluster"
  value       = local.cluster_name
}

output "cluster_region" {
  description = "AWS region where cluster is deployed"
  value       = var.region
}

output "total_gpus" {
  description = "Total number of GPUs in the cluster"
  value       = local.total_gpus
}

output "gpus_per_instance" {
  description = "Number of GPUs per instance"
  value       = local.gpus_per_instance
}

# ============================================================================
# Instance Information
# ============================================================================

output "instance_ids" {
  description = "List of EC2 instance IDs"
  value       = aws_instance.hpc_nodes[*].id
}

output "instance_details" {
  description = "Detailed information about each instance"
  value = [
    for i, instance in aws_instance.hpc_nodes : {
      id                = instance.id
      name              = instance.tags.Name
      instance_type     = instance.instance_type
      private_ip        = instance.private_ip
      public_ip         = instance.public_ip
      role              = instance.tags.Role
      node_index        = instance.tags.NodeIndex
      availability_zone = instance.availability_zone
    }
  ]
}

output "master_node" {
  description = "Information about the master node"
  value = {
    id         = aws_instance.hpc_nodes[0].id
    private_ip = aws_instance.hpc_nodes[0].private_ip
    public_ip  = aws_instance.hpc_nodes[0].public_ip
  }
}

output "worker_nodes" {
  description = "Information about worker nodes"
  value = [
    for i, instance in slice(aws_instance.hpc_nodes, 1, length(aws_instance.hpc_nodes)) : {
      id         = instance.id
      private_ip = instance.private_ip
      public_ip  = instance.public_ip
      node_index = i + 1
    }
  ]
}

# ============================================================================
# Networking Information
# ============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = var.create_vpc ? aws_vpc.cluster_vpc[0].id : var.existing_vpc_id
}

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = var.create_vpc ? aws_subnet.public_subnet[0].id : var.existing_public_subnet_id
}

output "private_subnet_id" {
  description = "ID of the private subnet"
  value       = var.create_vpc ? aws_subnet.private_subnet[0].id : var.existing_private_subnet_id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.hpc_cluster_sg.id
}

output "placement_group_name" {
  description = "Name of the placement group"
  value       = aws_placement_group.hpc_cluster_pg.name
}

# ============================================================================
# SSH and Access Information
# ============================================================================

output "ssh_key_name" {
  description = "Name of the SSH key pair"
  value       = var.create_ssh_key ? aws_key_pair.cluster_key[0].key_name : var.existing_key_pair_name
}

output "use_secrets_manager" {
  description = "Whether SSH key is stored in AWS Secrets Manager"
  value       = var.create_ssh_key
}

output "secret_name" {
  description = "Name of the secret in AWS Secrets Manager"
  value       = var.create_ssh_key ? aws_secretsmanager_secret.ssh_private_key[0].name : null
}

# ============================================================================
# Monitoring and Management
# ============================================================================

output "cloudwatch_log_group" {
  description = "CloudWatch log group for cluster logs"
  value       = var.enable_monitoring ? "/aws/ec2/${local.cluster_name}" : null
}

output "iam_instance_profile_arn" {
  description = "ARN of the IAM instance profile"
  value       = aws_iam_instance_profile.hpc_instance_profile.arn
}

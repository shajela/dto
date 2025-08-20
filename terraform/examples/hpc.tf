module "hpc" {
  source = "../modules/horovod-hpc"

  cluster_name = "example-hpc-cluster"
  environment  = "dev"

  # ~$0.526/hour (1x T4 GPU, 4 vCPUs, 16 GB RAM)
  # Defaults to 2 instances
  instance_type = "g4dn.xlarge"

  # g4dn.xlarge does not support EFA
  # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types
  enable_efa = false

  create_vpc        = true

  # For dev purposes deploy in a public subnet
  # If using a private subnet, a client vpn
  # endpoint is required but not provided
  use_public_subnet = true
}

output "cluster_name" {
  value = module.hpc.cluster_name
}

output "master_node" {
  value = module.hpc.master_node
}

output "worker_nodes" {
  value = module.hpc.worker_nodes
}

output "total_gpus" {
  value = module.hpc.total_gpus
}

output "ssh_key_name" {
  value = module.hpc.ssh_key_name
}

output "secret_name" {
  value = module.hpc.secret_name
}

output "cluster_region" {
  value = module.hpc.cluster_region
}

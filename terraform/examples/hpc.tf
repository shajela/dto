module "hpc" {
  source = "../modules/horovod-hpc"

  cluster_name = "example-hpc-cluster"
  environment  = "dev"

  # ~$0.526/hour (1x T4 GPU, 4 vCPUs, 16 GB RAM)
  # Defaults to 2 instances
  instance_type = "g4dn.xlarge"

  create_vpc        = true
  use_public_subnet = false
}

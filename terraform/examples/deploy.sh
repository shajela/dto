#!/bin/bash
set -euo pipefail

# Infra deployment
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Extract values from Terraform outputs
CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "hpc-cluster")
MASTER_IP=$(terraform output -json master_node 2>/dev/null | jq -r '.private_ip' || echo "")
WORKER_NODES_JSON=$(terraform output -json worker_nodes 2>/dev/null || echo "[]")
WORKER_IPS=$(echo "$WORKER_NODES_JSON" | jq -r '.[].private_ip' | tr '\n' ' ' | sed 's/ $//')
TOTAL_GPUS=$(terraform output -raw total_gpus 2>/dev/null || echo "")
SSH_KEY_NAME=$(terraform output -raw ssh_key_name 2>/dev/null || echo "")
SECRET_NAME=$(terraform output -raw secret_name 2>/dev/null || echo "")
REGION=$(terraform output -raw cluster_region 2>/dev/null || echo "")

../../terraform/modules/horovod-hpc/scripts/cluster-setup.sh --cluster-name "$CLUSTER_NAME" \
    --master-ip "$MASTER_IP" \
    --worker-ips "$WORKER_IPS" \
    --total-gpus "$TOTAL_GPUS" \
    --secret-name "$SECRET_NAME" \
    --region "$REGION"

#!/bin/bash
# ============================================================================
# HPC Cluster Post-Deployment Setup Script
# ============================================================================

set -euo pipefail

# Function to show usage
show_usage() {
    echo "Usage: $0 --cluster-name NAME --master-ip IP --worker-ips \"IP1 IP2 ...\" --total-gpus N [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --cluster-name NAME          Name of the HPC cluster"
    echo "  --master-ip IP               IP address of the master node"
    echo "  --worker-ips \"IP1 IP2 ...\"   Space-separated list of worker node IPs"
    echo "  --total-gpus N               Total number of GPUs in the cluster"
    echo ""
    echo "Optional arguments:"
    echo "  --private-key PATH           Path to SSH private key (default: ~/.ssh/id_rsa)"
    echo "  --use-secrets-manager        Use AWS Secrets Manager for SSH key"
    echo "  --secret-name NAME           Name of secret in AWS Secrets Manager"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --cluster-name my-cluster --master-ip 10.0.1.10 --worker-ips \"10.0.1.11 10.0.1.12\" --total-gpus 4"
    echo "  $0 --cluster-name my-cluster --master-ip 10.0.1.10 --worker-ips \"10.0.1.11\" --total-gpus 2 --private-key /path/to/key.pem"
}

# Default values
PRIVATE_KEY_PATH="$HOME/.ssh/id_rsa"
USE_SECRETS_MANAGER="false"
SECRET_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --master-ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --worker-ips)
            read -ra WORKER_IPS <<< "$2"
            shift 2
            ;;
        --total-gpus)
            TOTAL_GPUS="$2"
            shift 2
            ;;
        --private-key)
            PRIVATE_KEY_PATH="$2"
            shift 2
            ;;
        --use-secrets-manager)
            USE_SECRETS_MANAGER="true"
            shift
            ;;
        --secret-name)
            SECRET_NAME="$2"
            USE_SECRETS_MANAGER="true"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "${CLUSTER_NAME:-}" ]]; then
    echo "Error: --cluster-name is required"
    show_usage
    exit 1
fi

if [[ -z "${MASTER_IP:-}" ]]; then
    echo "Error: --master-ip is required"
    show_usage
    exit 1
fi

if [[ ${#WORKER_IPS[@]} -eq 0 ]]; then
    echo "Error: --worker-ips is required"
    show_usage
    exit 1
fi

if [[ -z "${TOTAL_GPUS:-}" ]]; then
    echo "Error: --total-gpus is required"
    show_usage
    exit 1
fi

if [[ "$USE_SECRETS_MANAGER" == "true" && -z "$SECRET_NAME" ]]; then
    echo "Error: --secret-name is required when using secrets manager"
    show_usage
    exit 1
fi

echo "============================================================================"
echo "Setting up HPC Cluster: $CLUSTER_NAME"
echo "Master: $MASTER_IP"
echo "Workers: ${WORKER_IPS[@]}"
echo "Total GPUs: $TOTAL_GPUS"
echo "============================================================================"

# Set up SSH key
if [[ "$USE_SECRETS_MANAGER" == "true" ]]; then
    echo "Retrieving SSH private key from AWS Secrets Manager..."
    TEMP_KEY_FILE="/tmp/cluster-key-$$.pem"
    aws secretsmanager get-secret-value --secret-id "$SECRET_NAME" --query SecretString --output text | jq -r .private_key > "$TEMP_KEY_FILE"
    chmod 600 "$TEMP_KEY_FILE"
    PRIVATE_KEY_PATH="$TEMP_KEY_FILE"
    
    # Cleanup function to remove temporary key file
    cleanup() {
        if [[ -f "$TEMP_KEY_FILE" ]]; then
            rm -f "$TEMP_KEY_FILE"
        fi
    }
    trap cleanup EXIT
fi

# Function to execute commands on remote hosts
ssh_exec() {
    local host=$1
    shift
    ssh -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$host "$@"
}

# Function to copy files to remote hosts
ssh_copy() {
    local host=$1
    local src=$2
    local dst=$3
    scp -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no "$src" ubuntu@$host:"$dst"
}

# Wait for all nodes to be ready
echo "Waiting for all nodes to complete setup..."
all_ready=false
while [[ "$all_ready" == "false" ]]; do
    all_ready=true
    
    # Check master
    if ! ssh_exec $MASTER_IP "test -f /var/log/hpc-setup-complete" 2>/dev/null; then
        echo "Master $MASTER_IP not ready yet..."
        all_ready=false
    fi
    
    # Check workers
    for worker_ip in "${WORKER_IPS[@]}"; do
        if ! ssh_exec $worker_ip "test -f /var/log/hpc-setup-complete" 2>/dev/null; then
            echo "Worker $worker_ip not ready yet..."
            all_ready=false
        fi
    done
    
    if [[ "$all_ready" == "false" ]]; then
        sleep 30
    fi
done

echo "All nodes are ready!"

# Setup passwordless SSH between nodes
echo "Setting up passwordless SSH between nodes..."

# Get master's public key
MASTER_PUBKEY=$(ssh_exec $MASTER_IP "cat /home/ubuntu/.ssh/id_rsa.pub")

# Distribute master's public key to all nodes (including itself)
ssh_exec $MASTER_IP "echo '$MASTER_PUBKEY' >> /home/ubuntu/.ssh/authorized_keys"

for worker_ip in "${WORKER_IPS[@]}"; do
    ssh_exec $worker_ip "echo '$MASTER_PUBKEY' >> /home/ubuntu/.ssh/authorized_keys"
done

# Detect GPUs per node dynamically for SLURM configuration
GPUS_PER_NODE=$(ssh_exec $MASTER_IP "nvidia-smi -L | wc -l" 2>/dev/null)

# Install and configure SLURM
echo "Installing and configuring SLURM..."

# Install SLURM on master node
ssh_exec $MASTER_IP "sudo apt-get update && sudo apt-get install -y slurm-wlm slurm-wlm-doc"

# Install SLURM on worker nodes
for worker_ip in "${WORKER_IPS[@]}"; do
    ssh_exec $worker_ip "sudo apt-get update && sudo apt-get install -y slurmd slurm-client"
done

# Create SLURM configuration
ssh_exec $MASTER_IP "cat > /tmp/slurm.conf" << EOF
# SLURM Configuration for HPC Cluster
ClusterName=$CLUSTER_NAME
ControlMachine=\$(hostname)
ControlAddr=$MASTER_IP
SlurmUser=slurm
SlurmdUser=root
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/none
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
SwitchType=switch/none
MpiDefault=pmix
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup
ReturnToService=2
MaxJobCount=10000
MaxArraySize=1001
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
DefMemPerCPU=1000
MaxMemPerCPU=0
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/none
JobAcctGatherFrequency=30
JobAcctGatherType=jobacct_gather/linux
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm/slurmd.log

# Node definitions
PartitionName=gpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP DefaultTime=60
EOF

# Add node definitions to SLURM config
ssh_exec $MASTER_IP "echo 'NodeName=\$(hostname) CPUs=\$(nproc) Sockets=\$(lscpu | grep 'Socket(s):' | awk '{print \$2}') CoresPerSocket=\$((\$(nproc)/\$(lscpu | grep 'Socket(s):' | awk '{print \$2}'))) ThreadsPerCore=1 RealMemory=\$((\$(free -m | grep '^Mem:' | awk '{print \$2}') - 1000)) Gres=gpu:$GPUS_PER_NODE State=UNKNOWN' >> /tmp/slurm.conf"

for worker_ip in "${WORKER_IPS[@]}"; do
    ssh_exec $worker_ip "echo 'NodeName=\$(hostname) CPUs=\$(nproc) Sockets=\$(lscpu | grep 'Socket(s):' | awk '{print \$2}') CoresPerSocket=\$((\$(nproc)/\$(lscpu | grep 'Socket(s):' | awk '{print \$2}'))) ThreadsPerCore=1 RealMemory=\$((\$(free -m | grep '^Mem:' | awk '{print \$2}') - 1000)) Gres=gpu:$GPUS_PER_NODE State=UNKNOWN' | ssh_exec $MASTER_IP 'cat >> /tmp/slurm.conf'"
done

# Copy SLURM config to all nodes
ssh_exec $MASTER_IP "sudo cp /tmp/slurm.conf /etc/slurm/slurm.conf"
for worker_ip in "${WORKER_IPS[@]}"; do
    # Copy and then move on local due to permissions
    ssh_copy $MASTER_IP "/tmp/slurm.conf" "$worker_ip:/tmp/slurm.conf"
    ssh_exec $worker_ip "sudo cp /tmp/slurm.conf /etc/slurm/slurm.conf"
done

# Configure GPU resources
ssh_exec $MASTER_IP "sudo mkdir -p /etc/slurm && echo 'Name=gpu File=/dev/nvidia[0-$((GPUS_PER_NODE-1))]' | sudo tee /etc/slurm/gres.conf"
for worker_ip in "${WORKER_IPS[@]}"; do
    ssh_exec $worker_ip "sudo mkdir -p /etc/slurm && echo 'Name=gpu File=/dev/nvidia[0-$((GPUS_PER_NODE-1))]' | sudo tee /etc/slurm/gres.conf"
done

# Create SLURM directories and set permissions
for node_ip in $MASTER_IP "${WORKER_IPS[@]}"; do
    ssh_exec $node_ip "sudo mkdir -p /var/spool/slurm/{ctld,d} /var/log/slurm && sudo chown slurm:slurm /var/spool/slurm/{ctld,d} /var/log/slurm"
done

# Setup DTO framework directory on all nodes for Python imports
echo "Setting up DTO framework directory structure..."
for node_ip in $MASTER_IP "${WORKER_IPS[@]}"; do
    ssh_exec $node_ip "mkdir -p /home/ubuntu/dto"
done

# Deploy DTO framework
echo "Deploying DTO framework files to all nodes..."
# Copy framework files to master node
ssh_copy distributed_trainer.py $MASTER_IP "/home/ubuntu/dto/distributed_trainer.py"

# Copy framework files to all worker nodes
for worker_ip in "${WORKER_IPS[@]}"; do
    echo "Copying framework files to worker: $worker_ip"
    ssh_copy distributed_trainer.py $worker_ip "/home/ubuntu/dto/distributed_trainer.py"
done

echo "Framework deployed to all nodes in /home/ubuntu/dto/"
echo "Training scripts can now import: from distributed_trainer import DistributedTrainer"

# Start SLURM services
ssh_exec $MASTER_IP "sudo systemctl enable slurmctld && sudo systemctl start slurmctld"
for worker_ip in "${WORKER_IPS[@]}"; do
    ssh_exec $worker_ip "sudo systemctl enable slurmd && sudo systemctl start slurmd"
done

# Create submission script
ssh_exec $MASTER_IP "cat > /home/ubuntu/submit_job.sh" << 'EOF'
#!/bin/bash
# SLURM job submission script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [args...]"
    echo "Example: $0 train.py"
    exit 1
fi

SCRIPT=$1
shift
ARGS="$@"

# Extract script name without path and extension for job naming
SCRIPT_NAME=$(basename "$SCRIPT" .py)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="${SCRIPT_NAME}_${TIMESTAMP}"

# Create SLURM batch script
cat > /tmp/dto_job.sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --nodes=\$(sinfo -N -h | wc -l)
#SBATCH --ntasks=\$((\$(sinfo -N -h | wc -l) * $GPUS_PER_NODE))
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --time=24:00:00
#SBATCH --output=/home/ubuntu/slurm-%j.out
#SBATCH --error=/home/ubuntu/slurm-%j.err

# Load environment
source /etc/environment

# Environment setup
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_PYTORCH=1

# Add DTO framework to Python path for imports
export PYTHONPATH="/home/ubuntu/dto:\$PYTHONPATH"

# Run training
echo "Starting training job \$SLURM_JOB_ID"
echo "Script: $SCRIPT"
echo "Nodes: \$SLURM_JOB_NODELIST"
echo "Total GPUs: \$SLURM_NTASKS"

srun --mpi=pmix python $SCRIPT $ARGS
SBATCH_EOF

# Submit the job
echo "Submitting training job..."
JOB_ID=\$(sbatch /tmp/dto_job.sbatch | awk '{print \$4}')
echo "Job submitted with ID: \$JOB_ID"
echo "Monitor with: squeue -j \$JOB_ID"
echo "View output: tail -f /home/ubuntu/slurm-\$JOB_ID.out"
EOF

ssh_exec $MASTER_IP "cat > /home/ubuntu/slurm_status.sh" << 'EOF'
#!/bin/bash
# SLURM cluster and job status script

echo "============================================================================"
echo "SLURM Cluster Status"
echo "============================================================================"
echo "Cluster: $CLUSTER_NAME"
echo "Date: $(date)"
echo ""

echo "=== Node Information ==="
sinfo -N -l

echo ""
echo "=== Partition Information ==="
sinfo -s

echo ""
echo "=== Job Queue ==="
squeue -l

echo ""
echo "=== Recent Jobs ==="
sacct -S today --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Start,End,Elapsed

echo ""
echo "=== GPU Usage ==="
sinfo -o "%.20N %.10c %.10m %.25f %.10G %.5a"

echo ""
echo "=== Usage Examples ==="
echo "Submit training job:"
echo "  ./submit_job.sh train.py"
echo "  ./submit_job.sh train.py"
echo ""
echo "Monitor jobs:"
echo "  squeue -u ubuntu"
echo "  watch squeue"
echo ""
echo "Cancel job:"
echo "  scancel <job_id>"
echo ""
echo "Job details:"
echo "  scontrol show job <job_id>"
echo ""
EOF

# Make scripts executable
ssh_exec $MASTER_IP "chmod +x /home/ubuntu/submit_job.sh /home/ubuntu/slurm_status.sh"

# Run initial cluster test
echo "Running initial cluster test..."

ssh_exec $MASTER_IP "cat > /home/ubuntu/cluster_test.py" << 'EOF'
#!/usr/bin/env python3
import torch
import horovod.torch as hvd
import socket
import os

# Initialize Horovod
hvd.init()

hostname = socket.gethostname()
rank = hvd.rank()
size = hvd.size()
local_rank = hvd.local_rank()

print(f"Rank {rank}/{size} on {hostname} (local rank: {local_rank})")

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')
    
    # Test basic tensor operations
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x.t())
    
    # Test Horovod allreduce
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    summed = hvd.allreduce(tensor, average=False)
    
    print(f"Rank {rank}: GPU {torch.cuda.current_device()}, "
          f"Device: {torch.cuda.get_device_name()}, "
          f"Allreduce sum: {summed.item()}")
else:
    print(f"Rank {rank}: No CUDA available")

# Synchronize all processes
hvd.allreduce(torch.tensor(0.0))

if rank == 0:
    print(f"Cluster test completed successfully with {size} processes!")
EOF

echo "Running SLURM cluster test..."
if ssh_exec $MASTER_IP "cd /home/ubuntu && ./submit_job.sh cluster_test.py" 2>/dev/null; then
    echo "SLURM cluster test job submitted successfully!"
    echo "Monitor with: ssh $MASTER_IP 'squeue -u ubuntu'"
else
    echo "SLURM cluster test job submission failed!"
fi

# Create cluster information summary
echo "Creating cluster summary..."

ssh_exec $MASTER_IP "cat > /home/ubuntu/cluster_summary.txt" << EOF
HPC Cluster Summary
==================
Cluster Name: $CLUSTER_NAME
Total Nodes: $((1 + ${#WORKER_IPS[@]}))
Total GPUs: $TOTAL_GPUS
Setup Date: $(date)

Master Node: $MASTER_IP
Worker Nodes: ${WORKER_IPS[@]}

SLURM Configuration:
- Cluster managed by SLURM workload manager
- GPU partition with all nodes
- MPI support via PMIx
- Job scheduling and resource allocation
- All training execution enforced through SLURM

Files Created:
- submit_job.sh: Job submission script
- slurm_status.sh: Check SLURM cluster and job status
- cluster_test.py: Test cluster connectivity (for setup validation only)

SLURM Job Submission:
1. Copy your DTO training script to the master node
2. Submit job: ./submit_job.sh train.py
3. Monitor: ./slurm_status.sh or squeue

Example SLURM Commands:
- Submit job: ./submit_job.sh train.py
- Check queue: squeue -u ubuntu
- Monitor jobs: watch squeue
- Cancel job: scancel <job_id>
- Job details: scontrol show job <job_id>
- Cluster status: ./slurm_status.sh

Access:
- View SLURM logs: tail -f /var/log/slurm/slurmctld.log
- View job output: tail -f /home/ubuntu/slurm-<job_id>.out

Note: Direct execution of training scripts (bypassing SLURM) is not supported.
All training workloads must be submitted through SLURM job scheduling.
EOF

echo "============================================================================"
echo "HPC Cluster Setup Complete!"
echo "============================================================================"
echo "Master Node: $MASTER_IP"
echo "Total GPUs: $TOTAL_GPUS"
echo ""
echo "SLURM Job Submission:
echo "1. SSH to master: ssh -i $PRIVATE_KEY_PATH ubuntu@$MASTER_IP"
echo "2. Submit job: ./submit_job.sh train.py" 
echo "3. Monitor jobs: ./slurm_status.sh or squeue -u ubuntu"
echo "4. View output: tail -f /home/ubuntu/slurm-<job_id>.out"
echo ""
echo "Useful files on master node:"
echo "- /home/ubuntu/cluster_summary.txt: Cluster information"
echo "- /home/ubuntu/slurm_status.sh: SLURM status and job queue"
echo "- /home/ubuntu/submit_job.sh: Job submission script"
echo ""
echo "Note: All training execution is enforced through SLURM job scheduling."
echo "Only framework training scripts are supported."

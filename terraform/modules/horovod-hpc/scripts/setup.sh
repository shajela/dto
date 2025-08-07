#!/bin/bash
# ============================================================================
# EC2 Instance Setup Script for HPC Cluster
# ============================================================================
# This script runs on each EC2 instance during initial boot via user data.
# It installs and configures the necessary software for distributed training.

set -euo pipefail

# Template variables (replaced by Terraform)
CLUSTER_NAME="${cluster_name}"
REGION="${region}"
ENABLE_EFA="${enable_efa}"
HOROVOD_VERSION="${horovod_version}"
NCCL_VERSION="${nccl_version}"
CUDA_VERSION="${cuda_version}"
INSTANCE_TYPE="${instance_type}"
ENABLE_MONITORING="${enable_monitoring}"
SHARED_STORAGE_MOUNT="${shared_storage_mount}"
SETUP_SSH_KEYS="${setup_ssh_keys}"

# Logging function
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a /var/log/hpc-setup.log
}

log "Starting HPC cluster setup for instance in cluster: $CLUSTER_NAME"
log "Instance type: $INSTANCE_TYPE"
log "Region: $REGION"

# Update system packages
log "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# Install essential packages
log "Installing essential packages..."
apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    unzip \
    jq \
    build-essential \
    cmake \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-pip \
    python3-dev \
    python3-venv \
    openssh-server \
    nfs-common \
    awscli

# Configure AWS CLI with region
log "Configuring AWS CLI..."
mkdir -p /home/ubuntu/.aws
cat > /home/ubuntu/.aws/config << EOF
[default]
region = $REGION
output = json
EOF
chown -R ubuntu:ubuntu /home/ubuntu/.aws

# Install NVIDIA drivers and CUDA (if GPU instance)
if [[ "$INSTANCE_TYPE" == g* ]] || [[ "$INSTANCE_TYPE" == p* ]]; then
    log "Installing NVIDIA drivers and CUDA..."
    
    # Install NVIDIA drivers
    apt-get install -y nvidia-driver-470
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update -y
    apt-get install -y cuda-toolkit-$${CUDA_VERSION//./-}
    
    # Add CUDA to PATH
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /etc/environment
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment

    # Install NVIDIA Container Toolkit for Docker
    source /etc/os-release
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$ID$VERSION_ID/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update -y
    apt-get install -y nvidia-docker2
    systemctl restart docker
fi

# Install Python packages and create virtual environment
log "Setting up Python environment..."
pip3 install --upgrade pip setuptools wheel

# Create a virtual environment for the project
python3 -m venv /opt/dto-env
source /opt/dto-env/bin/activate

# Install Python dependencies for distributed training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib seaborn pandas jupyter tensorboard psutil nvidia-ml-py3 boto3

# Install Horovod with NCCL support
log "Installing Horovod version $HOROVOD_VERSION..."
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install "horovod[pytorch]==$HOROVOD_VERSION"

# Make virtual environment accessible to all users
chmod -R 755 /opt/dto-env
echo "source /opt/dto-env/bin/activate" >> /home/ubuntu/.bashrc
echo "source /opt/dto-env/bin/activate" >> /etc/skel/.bashrc

# Install and configure EFA (if enabled)
if [[ "$ENABLE_EFA" == "true" ]]
then
    log "Installing EFA drivers..."
    cd /tmp
    curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-latest.tar.gz
    tar -xf aws-efa-installer-latest.tar.gz
    cd aws-efa-installer
    ./efa_installer.sh -y
    
    # Configure EFA
    echo "export FI_PROVIDER=efa" >> /etc/environment
    echo "export FI_EFA_USE_DEVICE_RDMA=1" >> /etc/environment
fi

# Setup SSH key generation for passwordless SSH
if [[ "$SETUP_SSH_KEYS" == "true" ]]
then
    log "Setting up SSH keys for passwordless access..."
    
    # Generate SSH key for ubuntu user if it doesn't exist
    if [[ ! -f /home/ubuntu/.ssh/id_rsa ]]
    then
        sudo -u ubuntu ssh-keygen -t rsa -b 4096 -f /home/ubuntu/.ssh/id_rsa -N ""
        chmod 600 /home/ubuntu/.ssh/id_rsa
        chmod 644 /home/ubuntu/.ssh/id_rsa.pub
        chown ubuntu:ubuntu /home/ubuntu/.ssh/id_rsa*
    fi
    
    # Configure SSH client settings
    cat >> /home/ubuntu/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF
    chown ubuntu:ubuntu /home/ubuntu/.ssh/config
    chmod 600 /home/ubuntu/.ssh/config
fi

# Setup shared storage mount point
if [[ -n "$SHARED_STORAGE_MOUNT" ]]
then
    log "Creating shared storage mount point: $SHARED_STORAGE_MOUNT"
    mkdir -p "$SHARED_STORAGE_MOUNT"
    chown ubuntu:ubuntu "$SHARED_STORAGE_MOUNT"
    chmod 755 "$SHARED_STORAGE_MOUNT"
fi

# Install and configure monitoring (if enabled)
if [[ "$ENABLE_MONITORING" == "true" ]]
then
    log "Setting up CloudWatch monitoring..."
    
    # Install CloudWatch agent
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i -E amazon-cloudwatch-agent.deb
    
    # Configure CloudWatch agent
    cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "ubuntu"
    },
    "metrics": {
        "namespace": "HPC/Cluster",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "net": {
                "measurement": [
                    "bytes_sent",
                    "bytes_recv",
                    "packets_sent",
                    "packets_recv"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/hpc-setup.log",
                        "log_group_name": "/aws/ec2/hpc-cluster",
                        "log_stream_name": "{instance_id}/setup.log"
                    }
                ]
            }
        }
    }
}
EOF
    
    # Start CloudWatch agent
    systemctl enable amazon-cloudwatch-agent
    systemctl start amazon-cloudwatch-agent
fi

# Configure system settings for HPC workloads
log "Configuring system settings for HPC..."

# Increase file descriptor limits
cat >> /etc/security/limits.conf << EOF
ubuntu soft nofile 65536
ubuntu hard nofile 65536
ubuntu soft nproc 65536
ubuntu hard nproc 65536
EOF

# Configure sysctl for network performance
cat >> /etc/sysctl.conf << EOF
# Network performance tuning for HPC
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
EOF
sysctl -p

# Create DTO framework directory structure
log "Creating DTO framework directory structure..."
mkdir -p /home/ubuntu/dto/{src,examples,checkpoints,logs}
chown -R ubuntu:ubuntu /home/ubuntu/dto

# Create a simple training environment test script
cat > /home/ubuntu/test_environment.py << EOF
#!/usr/bin/env python3
""" Test script to verify the HPC environment setup. """
import sys
import torch
import subprocess

def test_cuda():
    """Test CUDA availability."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def test_horovod():
    """Test Horovod installation."""
    try:
        import horovod.torch as hvd
        print(f"Horovod imported successfully")
        print(f"Horovod version: {hvd.__version__}")
        return True
    except ImportError as e:
        print(f"Horovod import failed: {e}")
        return False

def test_system():
    """Test system configuration."""
    try:
        # Test nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi working correctly")
            print(result.stdout.split("\n")[2])  # GPU info line
        else:
            print("nvidia-smi failed or not available")
    except FileNotFoundError:
        print("nvidia-smi not found")

if __name__ == "__main__":
    print("=" * 50)
    print("HPC Environment Test")
    print("=" * 50)
    
    test_cuda()
    print("-" * 30)
    test_horovod()
    print("-" * 30)
    test_system()
    print("=" * 50)
EOF

chmod +x /home/ubuntu/test_environment.py
chown ubuntu:ubuntu /home/ubuntu/test_environment.py

# Create startup script that will be run on every boot
cat > /etc/systemd/system/hpc-startup.service << EOF
[Unit]
Description=HPC Cluster Startup Service
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c "source /opt/dto-env/bin/activate && echo \"HPC environment ready\" > /tmp/hpc-ready"
User=ubuntu
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable hpc-startup.service

# Set environment variables system-wide
cat >> /etc/environment << EOF
HOROVOD_GPU_OPERATIONS=NCCL
HOROVOD_WITH_PYTORCH=1
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=eth0
OMP_NUM_THREADS=1
EOF

# Configure hostname resolution for cluster communication
echo "127.0.0.1 $(hostname)" >> /etc/hosts

# Final setup steps
log "Finalizing setup..."

# Ensure ubuntu user owns their home directory
chown -R ubuntu:ubuntu /home/ubuntu

# Create completion marker
touch /var/log/hpc-setup-complete
echo "$(date): HPC setup completed successfully" >> /var/log/hpc-setup-complete

# Reboot to ensure all drivers and configurations are loaded
log "Setup completed successfully. System will reboot in 30 seconds..."
log "After reboot, test the environment with: python3 /home/ubuntu/test_environment.py"

# Schedule a reboot
shutdown -r +1 "HPC setup completed, rebooting to finalize configuration"

log "HPC cluster setup script finished"

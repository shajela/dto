# ============================================================================
# Data Sources
# ============================================================================

# Get availability zones
data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# Get Deep Learning AMI
data "aws_ami" "deep_learning_ami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.ami_name_pattern]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Get current caller identity
data "aws_caller_identity" "current" {}

# ============================================================================
# Random Resources
# ============================================================================

resource "random_id" "cluster_suffix" {
  byte_length = 4
}

# ============================================================================
# SSH Key Pair
# ============================================================================

resource "tls_private_key" "cluster_key" {
  count     = var.create_ssh_key ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "cluster_key" {
  count      = var.create_ssh_key ? 1 : 0
  key_name   = "${local.cluster_name}-key-${random_id.cluster_suffix.hex}"
  public_key = tls_private_key.cluster_key[0].public_key_openssh

  tags = local.common_tags
}

# ============================================================================
# Secrets Manager for SSH Key
# ============================================================================

resource "aws_secretsmanager_secret" "ssh_private_key" {
  count                   = var.create_ssh_key ? 1 : 0
  name                    = "${local.cluster_name}-ssh-private-key-${random_id.cluster_suffix.hex}"
  description             = "SSH private key for ${local.cluster_name} HPC cluster"
  recovery_window_in_days = var.secret_recovery_window_days

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-ssh-key-secret"
    Type = "ssh-key"
  })
}

resource "aws_secretsmanager_secret_version" "ssh_private_key" {
  count         = var.create_ssh_key ? 1 : 0
  secret_id     = aws_secretsmanager_secret.ssh_private_key[0].id
  secret_string = jsonencode({
    private_key = tls_private_key.cluster_key[0].private_key_pem
    public_key  = tls_private_key.cluster_key[0].public_key_openssh
    key_name    = aws_key_pair.cluster_key[0].key_name
  })
}

# ============================================================================
# Networking
# ============================================================================

# VPC
resource "aws_vpc" "cluster_vpc" {
  count = var.create_vpc ? 1 : 0

  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "cluster_igw" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.cluster_vpc[0].id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-igw"
  })
}

# Public Subnet
resource "aws_subnet" "public_subnet" {
  count = var.create_vpc ? 1 : 0

  vpc_id                  = aws_vpc.cluster_vpc[0].id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-subnet"
    Type = "public"
  })
}

# Private Subnet for high-performance computing
resource "aws_subnet" "private_subnet" {
  count = var.create_vpc ? 1 : 0

  vpc_id            = aws_vpc.cluster_vpc[0].id
  cidr_block        = var.private_subnet_cidr
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-subnet"
    Type = "private"
  })
}

# NAT Gateway for private subnet internet access
resource "aws_eip" "nat_eip" {
  count  = var.create_vpc && var.create_nat_gateway ? 1 : 0
  domain = "vpc"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-eip"
  })

  depends_on = [aws_internet_gateway.cluster_igw]
}

resource "aws_nat_gateway" "cluster_nat" {
  count = var.create_vpc && var.create_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat_eip[0].id
  subnet_id     = aws_subnet.public_subnet[0].id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-gateway"
  })

  depends_on = [aws_internet_gateway.cluster_igw]
}

# Route Tables
resource "aws_route_table" "public_rt" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.cluster_vpc[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.cluster_igw[0].id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-rt"
  })
}

resource "aws_route_table" "private_rt" {
  count  = var.create_vpc && var.create_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.cluster_vpc[0].id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.cluster_nat[0].id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-rt"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public_rta" {
  count          = var.create_vpc ? 1 : 0
  subnet_id      = aws_subnet.public_subnet[0].id
  route_table_id = aws_route_table.public_rt[0].id
}

resource "aws_route_table_association" "private_rta" {
  count          = var.create_vpc && var.create_nat_gateway ? 1 : 0
  subnet_id      = aws_subnet.private_subnet[0].id
  route_table_id = aws_route_table.private_rt[0].id
}

# ============================================================================
# Security Groups
# ============================================================================

# Security group for HPC cluster
resource "aws_security_group" "hpc_cluster_sg" {
  name_prefix = "${local.cluster_name}-hpc-"
  description = "Security group for HPC cluster nodes"
  vpc_id      = var.create_vpc ? aws_vpc.cluster_vpc[0].id : var.existing_vpc_id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Internal cluster communication
  ingress {
    description = "Internal cluster communication"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Internal cluster communication UDP"
    from_port   = 0
    to_port     = 65535
    protocol    = "udp"
    self        = true
  }

  # MPI communication
  ingress {
    description = "MPI communication"
    from_port   = 1024
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  # NCCL/Horovod communication
  dynamic "ingress" {
    for_each = var.nccl_port_range
    content {
      description = "NCCL communication"
      from_port   = ingress.value.from
      to_port     = ingress.value.to
      protocol    = "tcp"
      self        = true
    }
  }

  # EFA communication
  dynamic "ingress" {
    for_each = var.enable_efa ? [1] : []
    content {
      description = "EFA communication"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      self        = true
    }
  }

  # All outbound traffic
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-hpc-sg"
  })
}

# ============================================================================
# Placement Group
# ============================================================================

resource "aws_placement_group" "hpc_cluster_pg" {
  name         = "${local.cluster_name}-pg"
  strategy     = var.placement_group_strategy
  spread_level = var.placement_group_strategy == "spread" ? "rack" : null

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-placement-group"
  })
}

# ============================================================================
# IAM Role and Instance Profile
# ============================================================================

# IAM role for EC2 instances
resource "aws_iam_role" "hpc_instance_role" {
  name_prefix = "${local.cluster_name}-instance-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM policy for instance permissions
resource "aws_iam_role_policy" "hpc_instance_policy" {
  name_prefix = "${local.cluster_name}-policy-"
  role        = aws_iam_role.hpc_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribePlacementGroups",
          "ec2:CreateTags",
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "${var.arn_prefix}:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/${local.cluster_name}/*"
      }
    ],
    var.s3_bucket_arn != "" ? [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        var.s3_bucket_arn,
        "${var.s3_bucket_arn}/*"
      ]
    }] : [],
    var.create_ssh_key ? [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ]
      Resource = "${var.arn_prefix}:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:${local.cluster_name}-ssh-private-key-??????"
    }] : [])
  })
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "ssm_managed_instance_core" {
  role       = aws_iam_role.hpc_instance_role.name
  policy_arn = "${var.arn_prefix}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cloudwatch_agent_server_policy" {
  count      = var.enable_monitoring ? 1 : 0
  role       = aws_iam_role.hpc_instance_role.name
  policy_arn = "${var.arn_prefix}:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# Instance profile
resource "aws_iam_instance_profile" "hpc_instance_profile" {
  name_prefix = "${local.cluster_name}-profile-"
  role        = aws_iam_role.hpc_instance_role.name

  tags = local.common_tags
}

# ============================================================================
# Launch Template
# ============================================================================

resource "aws_launch_template" "hpc_launch_template" {
  name_prefix   = "${local.cluster_name}-lt-"
  image_id      = data.aws_ami.deep_learning_ami.id
  instance_type = var.instance_type
  key_name      = var.create_ssh_key ? aws_key_pair.cluster_key[0].key_name : var.existing_key_pair_name

  iam_instance_profile {
    name = aws_iam_instance_profile.hpc_instance_profile.name
  }

  user_data = local.user_data

  # Block device mappings
  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_type = var.root_volume_type
      volume_size = var.root_volume_size
      encrypted   = var.enable_ebs_encryption
      iops        = var.root_volume_type == "gp3" ? var.root_volume_iops : null
      throughput  = var.root_volume_type == "gp3" ? var.root_volume_throughput : null
    }
  }

  # Additional EBS volumes for shared storage
  dynamic "block_device_mappings" {
    for_each = var.additional_ebs_volumes
    content {
      device_name = block_device_mappings.value.device_name
      ebs {
        volume_type = block_device_mappings.value.volume_type
        volume_size = block_device_mappings.value.volume_size
        encrypted   = var.enable_ebs_encryption
        iops        = block_device_mappings.value.volume_type == "gp3" ? block_device_mappings.value.iops : null
        throughput  = block_device_mappings.value.volume_type == "gp3" ? block_device_mappings.value.throughput : null
      }
    }
  }

  # Instance metadata options
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  # Monitoring
  monitoring {
    enabled = var.enable_detailed_monitoring
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${local.cluster_name}-node"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(local.common_tags, {
      Name = "${local.cluster_name}-volume"
    })
  }

  tags = local.common_tags
}

# ============================================================================
# EC2 Instances
# ============================================================================

resource "aws_instance" "hpc_nodes" {
  count = var.instance_count

  launch_template {
    id      = aws_launch_template.hpc_launch_template.id
    version = "$Latest"
  }

  subnet_id                   = var.use_public_subnet ? (var.create_vpc ? aws_subnet.public_subnet[0].id : var.existing_public_subnet_id) : (var.create_vpc ? aws_subnet.private_subnet[0].id : var.existing_private_subnet_id)
  vpc_security_group_ids      = [aws_security_group.hpc_cluster_sg.id]
  placement_group             = aws_placement_group.hpc_cluster_pg.id
  availability_zone           = data.aws_availability_zones.available.names[0]
  associate_public_ip_address = var.use_public_subnet

  tags = merge(local.common_tags, {
    Name      = "${local.cluster_name}-node-${count.index + 1}"
    Role      = count.index == 0 ? "master" : "worker"
    NodeIndex = count.index
  })
}

# EFA Network Interfaces (attached after instance creation when EFA is enabled)
resource "aws_network_interface" "efa_interface" {
  count           = var.enable_efa ? var.instance_count : 0
  subnet_id       = var.use_public_subnet ? (var.create_vpc ? aws_subnet.public_subnet[0].id : var.existing_public_subnet_id) : (var.create_vpc ? aws_subnet.private_subnet[0].id : var.existing_private_subnet_id)
  security_groups = [aws_security_group.hpc_cluster_sg.id]
  interface_type  = "efa"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-efa-interface-${count.index + 1}"
  })
}

# Attach EFA interfaces to instances
resource "aws_network_interface_attachment" "efa_attachment" {
  count                = var.enable_efa ? var.instance_count : 0
  instance_id          = aws_instance.hpc_nodes[count.index].id
  network_interface_id = aws_network_interface.efa_interface[count.index].id
  device_index         = 1  # Use device index 1 since 0 is the primary interface
}

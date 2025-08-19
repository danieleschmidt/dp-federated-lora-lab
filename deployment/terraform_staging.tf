# DP-Federated LoRA Lab - staging Environment
# Generated on 2025-08-19T22:32:33.313073

terraform {
  required_version = ">= 1.0"
}

resource "aws_instance" "dp_fed_lora" {
  count         = 2
  instance_type = "t3.xlarge"
  
  tags = {
    Name        = "dp-fed-lora-staging-${count.index}"
    Environment = "staging"
  }
}

output "instance_ips" {
  value = aws_instance.dp_fed_lora[*].public_ip
}

# DP-Federated LoRA Lab - production Environment
# Generated on 2025-08-19T22:32:40.318792

terraform {
  required_version = ">= 1.0"
}

resource "aws_instance" "dp_fed_lora" {
  count         = 5
  instance_type = "t3.xlarge"
  
  tags = {
    Name        = "dp-fed-lora-production-${count.index}"
    Environment = "production"
  }
}

output "instance_ips" {
  value = aws_instance.dp_fed_lora[*].public_ip
}

# DP-Federated LoRA Lab - development Environment
# Generated on 2025-08-19T22:32:27.307888

terraform {
  required_version = ">= 1.0"
}

resource "aws_instance" "dp_fed_lora" {
  count         = 1
  instance_type = "t3.large"
  
  tags = {
    Name        = "dp-fed-lora-development-${count.index}"
    Environment = "development"
  }
}

output "instance_ips" {
  value = aws_instance.dp_fed_lora[*].public_ip
}

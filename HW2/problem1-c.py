import torch
import torchvision.models as models

# Load the ResNet-18 model
model = models.resnet18(weights=None)

# Iterate over all layers and count parameters
print(f"{'Layer Name':<50} {'# Parameters'}")
print("=" * 70)
total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"{name:<50} {num_params}")

print("=" * 70)
print(f"Total parameters in ResNet-18: {total_params:,}")

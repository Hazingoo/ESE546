import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from allcnn import allcnn_t

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 normalization constants
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])

# Clamping bounds in normalized space
MIN_NORM = (0.0 - MEAN) / STD
MAX_NORM = (1.0 - MEAN) / STD
MIN_NORM_T = torch.tensor(MIN_NORM, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
MAX_NORM_T = torch.tensor(MAX_NORM, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def one_step_attack(model, x, y, criterion, epsilon=8):
    """Perform 1-step signed gradient attack"""
    # Convert epsilon from pixel space to normalized space
    eps_pixel = epsilon / 255.0
    eps_normalized = eps_pixel / STD
    eps_tensor = torch.tensor(eps_normalized, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    
    x.requires_grad_(True)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    
    dx = x.grad.data.clone()
    
    with torch.no_grad():
        x_perturbed = x + eps_tensor * dx.sign()
        x_perturbed = torch.max(torch.min(x_perturbed, MAX_NORM_T), MIN_NORM_T)
    
    return x_perturbed

def compute_accuracy(model, dataloader, criterion, device):
    """Compute accuracy on clean images"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100.0 * correct / total

def compute_adversarial_accuracy(model, dataloader, criterion, device, epsilon=8):
    """Compute accuracy on 1-step adversarially perturbed images"""
    model.eval()
    correct = 0
    total = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        # Perform 1-step attack on each image
        perturbed_data = []
        for i in range(data.size(0)):
            x_single = data[i:i+1]
            y_single = target[i:i+1]
            x_perturbed = one_step_attack(model, x_single, y_single, criterion, epsilon)
            perturbed_data.append(x_perturbed)
        
        # Concatenate all perturbed images
        perturbed_batch = torch.cat(perturbed_data, dim=0)
        
        # Get predictions on perturbed images
        with torch.no_grad():
            outputs = model(perturbed_batch)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100.0 * correct / total

if __name__ == '__main__':
    # Load validation data (small subset for demo)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # Use only first 1000 images for faster testing
    valset_subset = torch.utils.data.Subset(valset, range(1000))
    valloader = torch.utils.data.DataLoader(valset_subset, batch_size=100, shuffle=False, num_workers=0)
    
    # Load model (untrained for demo)
    model = allcnn_t().to(device)
    model.eval()
    print("Using untrained model for demonstration")
    
    criterion = nn.CrossEntropyLoss()
    
    print("Computing accuracies...")
    
    # Compute clean accuracy
    clean_accuracy = compute_accuracy(model, valloader, criterion, device)
    print(f"Clean validation accuracy: {clean_accuracy:.2f}%")
    
    # Compute adversarial accuracy
    adversarial_accuracy = compute_adversarial_accuracy(model, valloader, criterion, device, epsilon=8)
    print(f"1-step adversarial accuracy (ε=8): {adversarial_accuracy:.2f}%")
    
    # Compare accuracies
    accuracy_drop = clean_accuracy - adversarial_accuracy
    print(f"\nAccuracy drop: {accuracy_drop:.2f} percentage points")
    print(f"Relative drop: {(accuracy_drop/clean_accuracy)*100:.1f}%")
    
    if adversarial_accuracy < clean_accuracy:
        print("The model is vulnerable to 1-step adversarial attacks.")
    else:
        print("The model shows some robustness to 1-step attacks.")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

def compute_gradient_wrt_input(model, x, y, criterion):
    model.zero_grad()
    x = x.clone().detach().requires_grad_(True)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    dx = x.grad.data.clone()
    return dx, loss.item()

def visualize_gradients(model, testloader, criterion):
    model.eval()
    
    correct_gradients = []
    incorrect_gradients = []
    correct_images = []
    incorrect_images = []
    
    for data, target in testloader:
        if len(correct_gradients) >= 3 and len(incorrect_gradients) >= 3:
            break
            
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
        
        for i in range(data.size(0)):
            if len(correct_gradients) >= 3 and len(incorrect_gradients) >= 3:
                break
                
            x_single = data[i:i+1]
            y_single = target[i:i+1]
            dx, _ = compute_gradient_wrt_input(model, x_single, y_single, criterion)
            
            # Denormalize for display
            img = x_single.cpu().numpy()[0].transpose(1, 2, 0)
            img = img * STD + MEAN
            img = np.clip(img, 0, 1)
            
            grad_mag = np.linalg.norm(dx.cpu().numpy()[0].transpose(1, 2, 0), axis=2)
            
            if predicted[i] == target[i] and len(correct_gradients) < 3:
                correct_images.append(img)
                correct_gradients.append(grad_mag)
            elif predicted[i] != target[i] and len(incorrect_gradients) < 3:
                incorrect_images.append(img)
                incorrect_gradients.append(grad_mag)
    
    # Simplified plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i in range(3):
        axes[0, i].imshow(correct_images[i])
        axes[0, i].set_title(f'Correct {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(incorrect_images[i])
        axes[1, i].set_title(f'Incorrect {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradient_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Correct avg grad: {np.mean([np.mean(g) for g in correct_gradients])}")
    print(f"Incorrect avg grad: {np.mean([np.mean(g) for g in incorrect_gradients])}")

def signed_gradient_attack(model, x, y, criterion, epsilon=8, steps=5):
    # Convert epsilon from pixel space to normalized space
    eps_pixel = epsilon / 255.0
    eps_normalized = eps_pixel / STD
    eps_tensor = torch.tensor(eps_normalized, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    
    x_perturbed = x.clone().detach().requires_grad_(True)
    losses = []
    
    for step in range(steps):
        model.zero_grad()
        if x_perturbed.grad is not None:
            x_perturbed.grad.zero_()
            
        y_pred = model(x_perturbed)
        loss = criterion(y_pred, y)
        loss.backward()
        
        dx = x_perturbed.grad.data.clone()
        
        with torch.no_grad():
            x_perturbed = x_perturbed + eps_tensor * dx.sign()
            # Clamp to normalized space bounds 
            x_perturbed = torch.max(torch.min(x_perturbed, MAX_NORM_T), MIN_NORM_T)
            
            y_pred_perturbed = model(x_perturbed)
            loss_perturbed = criterion(y_pred_perturbed, y)
            losses.append(loss_perturbed.item())
        
        x_perturbed = x_perturbed.detach().requires_grad_(True)
    
    return x_perturbed, losses

def adversarial_attack_analysis(model, testloader, criterion):
    model.eval()
    
    data, target = next(iter(testloader))
    data, target = data.to(device), target.to(device)
    
    all_losses = []
    for i in range(data.size(0)):
        x_single = data[i:i+1]
        y_single = target[i:i+1]
        _, losses = signed_gradient_attack(model, x_single, y_single, criterion)
        all_losses.append(losses)
    
    avg_losses = np.mean(all_losses, axis=0)
    
    # Simplified plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), avg_losses, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Attack Step')
    plt.ylabel('Average Loss')
    plt.title('Loss vs Attack Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('adversarial_attack_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Initial loss: {avg_losses[0]}")
    print(f"Final loss: {avg_losses[-1]}")
    print(f"Loss increase: {avg_losses[-1] - avg_losses[0]}")
    
    return avg_losses

if __name__ == '__main__':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    try:
        model = allcnn_t().to(device)
        model.load_state_dict(torch.load('best_allcnn_model.pth', map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model not found. Please run training first.")
        exit()
    
    criterion = nn.CrossEntropyLoss()
    
    visualize_gradients(model, testloader, criterion)
    adversarial_attack_analysis(model, testloader, criterion)
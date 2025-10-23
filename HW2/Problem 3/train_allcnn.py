import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from allcnn import allcnn_t

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

# Initialize model
model = allcnn_t().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

# Training history
train_losses = []
train_errors = []
val_losses = []
val_errors = []

def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_error = 100.0 * (1 - correct / total)
    
    return epoch_loss, epoch_error

def validate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(testloader)
    epoch_error = 100.0 * (1 - correct / total)
    
    return epoch_loss, epoch_error

if __name__ == '__main__':
    # Training loop
    num_epochs = 100
    best_val_error = 100.0

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_loss, train_error = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_error = validate(model, testloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        train_errors.append(train_error)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        
        print(f'Train Loss: {train_loss:.4f}, Train Error: {train_error:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Error: {val_error:.2f}%')
        
        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), 'best_allcnn_model.pth')
            print(f'New best validation error: {val_error:.2f}% - Model saved!')

    print(f'\nTraining completed! Best validation error: {best_val_error:.2f}%')

    # Save final model
    torch.save(model.state_dict(), 'final_allcnn_model.pth')

    # Plot training progress
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_errors, label='Training Error')
    plt.plot(val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Training and Validation Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training progress plots saved as 'training_progress.png'")

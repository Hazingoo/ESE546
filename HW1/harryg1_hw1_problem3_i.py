import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
ds = fetch_openml('mnist_784', as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

train_data = torch.tensor(x.astype(np.float32) / 255.0)
train_labels = torch.tensor(y.astype(int))
test_data = torch.tensor(x_test.astype(np.float32) / 255.0)
test_labels = torch.tensor(y_test.astype(int))

# Reshape data for Conv2d layer)
train_data = train_data.view(-1, 1, 28, 28)
test_data = test_data.view(-1, 1, 28, 28)

# Split training data 50/50 for train/validation
n_train = len(train_data) // 2
x_train = train_data[:n_train]
y_train = train_labels[:n_train]
x_val = train_data[n_train:]
y_val = train_labels[n_train:]

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Extracts 4x4 patches
        self.embedding = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, bias=True)
        self.linear = nn.Linear(392, 10)
        self.relu = nn.ReLU()
        
        # Initialize weights 
        nn.init.normal_(self.embedding.weight, mean=0, std=1)
        nn.init.normal_(self.embedding.bias, mean=0, std=1)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.normal_(self.linear.bias, mean=0, std=1)
        
        # Normalize weights
        with torch.no_grad():
            self.embedding.weight.data = self.embedding.weight.data / torch.norm(self.embedding.weight.data)
            self.embedding.bias.data = self.embedding.bias.data / torch.norm(self.embedding.bias.data)
            self.linear.weight.data = self.linear.weight.data / torch.norm(self.linear.weight.data)
            self.linear.bias.data = self.linear.bias.data / torch.norm(self.linear.bias.data)
    
    def forward(self, x):
        x = self.embedding(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        
        return x

# Initialize network
net = NeuralNetwork().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Training parameters
batch_size = 32
n_iterations = 10000  

# Storage for logging
train_losses = []
train_errors = []
val_losses = []
val_errors = []
iterations = []


def validate(model, x_val, y_val, criterion):
    model.eval()
    total_loss = 0
    total_error = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            x_batch = x_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Calculate error
            _, predicted = torch.max(outputs.data, 1)
            error = (predicted != y_batch).float().mean()
            
            total_loss += loss.item()
            total_error += error.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_error = total_error / n_batches
    return avg_loss, avg_error

# Training loop
net.train()
for t in range(n_iterations):
    batch_indices = torch.randperm(len(x_train))[:batch_size]
    x_batch = x_train[batch_indices]
    y_batch = y_train[batch_indices]
    
    
    # Forward pass
    optimizer.zero_grad()
    outputs = net(x_batch)
    loss = criterion(outputs, y_batch)
    
    # Calculate training error
    _, predicted = torch.max(outputs.data, 1)
    train_error = (predicted != y_batch).float().mean()
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Validation every 1000 weight updates
    if t % 1000 == 0:
        val_loss, val_error = validate(net, x_val, y_val, criterion)
        
        print(f"Iteration {t}: train_loss={loss.item()}, train_error={train_error.item()}")
        print(f"           val_loss={val_loss:.4f}, val_error={val_error:.4f}")
        
        # Store for plotting
        train_losses.append(loss.item())
        train_errors.append(train_error.item())
        val_losses.append(val_loss)
        val_errors.append(val_error)
        iterations.append(t)

print("\nTraining complete!")

# Final validation
final_val_loss, final_val_error = validate(net, x_val, y_val, criterion)
print(f"Final validation error: {final_val_error} ({final_val_error*100}%)")

# Plot training
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.plot(iterations, val_losses, 'r-', linewidth=2, label='Validation Loss')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Weight Updates')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(iterations, train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(iterations, val_errors, 'r-', linewidth=2, label='Validation Error')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Error')
plt.title('Training and Validation Error vs. Weight Updates')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

print(f"Final validation error: {final_val_error} ({final_val_error*100}%)")



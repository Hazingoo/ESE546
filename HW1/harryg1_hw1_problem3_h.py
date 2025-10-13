import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

from harryg1_hw1_problem3_b import embedding_t
from harryg1_hw1_problem3_c import linear_t
from harryg1_hw1_problem3_d import relu_t
from harryg1_hw1_problem3_e import softmax_cross_entropy_t

# Load dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
ds = fetch_openml('mnist_784', as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

train_data = x.astype(np.float32) / 255.0
train_labels = y.astype(int)
test_data = x_test.astype(np.float32) / 255.0
test_labels = y_test.astype(int)

train_data = train_data.reshape(-1, 28, 28)
test_data = test_data.reshape(-1, 28, 28)

# Split training data 50/50 for train/validation
n_train = len(train_data) // 2
x_train = train_data[:n_train]
y_train = train_labels[:n_train]
x_val = train_data[n_train:]
y_val = train_labels[n_train:]


class val:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

val = val(x_val, y_val)

# Initialize all the layers
l1 = embedding_t()  
l2 = linear_t()     
l3 = relu_t()       
l4 = softmax_cross_entropy_t() 

net = [l1, l2, l3, l4]

def validate(l1, l2, l3, l4):
    loss, tot_error = 0, 0
    batch_size = 32
    
    for i in range(0, 5000, batch_size): 
        x, y = val.data[i:i+32], val.targets[i:i+32]
        
        h1 = l1.forward(x)
        h2 = l2.forward(h1)
        h3 = l3.forward(h2)
        ell, error = l4.forward(h3, y)
        
        loss += ell
        tot_error += error
    
    n_batches = (5000 + batch_size - 1) // batch_size  
    avg_loss = loss / n_batches
    avg_error = tot_error / n_batches
    
    return avg_loss, avg_error

# Training parameters
lr = 0.1  
batch_size = 32
n_iterations = 10000

# Storage for logging
train_losses = []
train_errors = []
val_losses = []
val_errors = []
iterations = []

for t in range(n_iterations):
    batch_indices = np.random.choice(len(x_train), batch_size, replace=False)
    x_batch = x_train[batch_indices]
    y_batch = y_train[batch_indices]
    
    # Forward pass
    h1 = l1.forward(x_batch)
    h2 = l2.forward(h1)
    h3 = l3.forward(h2)
    ell, error = l4.forward(h3, y_batch)
    
    # Backward pass
    dh3 = l4.backward()
    dh2 = l3.backward(dh3)
    dh1 = l2.backward(dh2)
    dx = l1.backward(dh1)
    
    # Gather backprop gradients
    dw1, db1 = l1.dw, l1.db
    dw2, db2 = l2.dw, l2.db
    
    # One step of SGD
    l1.w = l1.w - lr * dw1
    l1.b = l1.b - lr * db1
    l2.w = l2.w - lr * dw2
    l2.b = l2.b - lr * db2
    
    # Logging every 1000 iterations
    if t % 1000 == 0:
        val_loss, val_error = validate(l1, l2, l3, l4)
        
        print(f"Iteration {t}: train_loss={ell}, train_error={error}")
        print(f"           val_loss={val_loss}, val_error={val_error}")
        
        # Store for plotting
        train_losses.append(ell)
        train_errors.append(error)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        iterations.append(t)

print("\nTraining complete!")

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.plot(iterations, val_losses, 'r-', linewidth=2, label='Validation Loss')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Weight Updates')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(iterations, train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(iterations, val_errors, 'r-', linewidth=2, label='Validation Error')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Error')
plt.title('Training and Validation Error vs. Weight Updates')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('validation_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final validation error: {val_errors[-1]} ({val_errors[-1]*100}%)")

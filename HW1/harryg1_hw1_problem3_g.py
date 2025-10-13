import numpy as np
import matplotlib.pyplot as plt

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

x_train = train_data
y_train = train_labels

# Initialize all the layers
l1 = embedding_t()  
l2 = linear_t()     
l3 = relu_t()      
l4 = softmax_cross_entropy_t()  

net = [l1, l2, l3, l4]
# Training parameters
lr = 0.1
batch_size = 32
n_iterations = 10000  

# Storage for logging
train_losses = []
train_errors = []
iterations = []

print(f"Starting training for {n_iterations} iterations...")
print(f"Batch size: {batch_size}, Learning rate: {lr}")

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
        print(f"Iteration {t}: train_loss={ell}, train_error={error:}")
        
        # Store for plotting
        train_losses.append(ell)
        train_errors.append(error)
        iterations.append(t)

print("\nTraining complete!")

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Loss')
plt.title('Training Loss vs. Weight Updates')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(iterations, train_errors, 'b-', linewidth=2, label='Training Error')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Error')
plt.title('Training Error vs. Weight Updates')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final training error: {train_errors[-1]} ({train_errors[-1]*100}%)")

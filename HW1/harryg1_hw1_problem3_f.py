import numpy as np
import sys

from harryg1_hw1_problem3_b import embedding_t
from harryg1_hw1_problem3_c import linear_t
from harryg1_hw1_problem3_d import relu_t
from harryg1_hw1_problem3_e import softmax_cross_entropy_t

def gradient_check_linear():
    print("\nGradient checking: Linear Layer")
    linear_layer = linear_t()
    h_l = np.random.normal(0, 0.1, (1, 392))
    
    epsilon = 1e-5
    
    print("Checking weights:")
    for test in range(10):
        i = np.random.randint(0, 10)
        j = np.random.randint(0, 392)
        k = np.random.randint(0, 10)
        
        dh_l_plus_1 = np.zeros((1, 10))
        dh_l_plus_1[0, k] = 1
        
        linear_layer.forward(h_l)
        linear_layer.backward(dh_l_plus_1)
        analytical = linear_layer.dw[i, j]
        
        # Numerical gradient
        linear_layer.w[i, j] += epsilon
        f_pos = linear_layer.forward(h_l)[0, k]
        
        linear_layer.w[i, j] -= 2 * epsilon
        f_neg = linear_layer.forward(h_l)[0, k]
        
        linear_layer.w[i, j] += epsilon  # restore
        
        numerical = (f_pos - f_neg) / (2 * epsilon)
        diff = abs(analytical - numerical)
        
        print(f"W[{i},{j}]: analytical={analytical}, numerical={numerical}, diff={diff}")
    
    print("Checking bias:")
    for test in range(5):
        i = np.random.randint(0, 10)
        k = np.random.randint(0, 10)
        
        dh_l_plus_1 = np.zeros((1, 10))
        dh_l_plus_1[0, k] = 1
        
        linear_layer.forward(h_l)
        linear_layer.backward(dh_l_plus_1)
        analytical = linear_layer.db[i]
        
        linear_layer.b[i] += epsilon
        f_pos = linear_layer.forward(h_l)[0, k]

        linear_layer.b[i] -= 2 * epsilon
        f_neg = linear_layer.forward(h_l)[0, k]
        linear_layer.b[i] += epsilon  
        
        numerical = (f_pos - f_neg) / (2 * epsilon)
        diff = abs(analytical - numerical)
        
        print(f"b[{i}]: analytical={analytical}, numerical={numerical}, diff={diff}")

def gradient_check_embedding():
    print("\nGradient checking: Embedding Layer")
    
    embedding_layer = embedding_t()
    h_l = np.random.normal(0, 0.1, (1, 28, 28))
    
    epsilon = 1e-5
    
    print("Checking weights:")
    for test in range(10):
        i = np.random.randint(0, 4)
        j = np.random.randint(0, 4) 
        k = np.random.randint(0, 8)
        output_idx = np.random.randint(0, 392)
        
        dh_l_plus_1 = np.zeros((1, 392))
        dh_l_plus_1[0, output_idx] = 1
        
        embedding_layer.forward(h_l)
        embedding_layer.backward(dh_l_plus_1)
        analytical = embedding_layer.dw[i, j, k]
        
        embedding_layer.w[i, j, k] += epsilon
        f_pos = embedding_layer.forward(h_l)[0, output_idx]
        
        embedding_layer.w[i, j, k] -= 2 * epsilon
        f_neg = embedding_layer.forward(h_l)[0, output_idx]
        
        embedding_layer.w[i, j, k] += epsilon 
        
        numerical = (f_pos - f_neg) / (2 * epsilon)
        diff = abs(analytical - numerical)
        
        print(f"W[{i},{j},{k}]: analytical={analytical}, numerical={numerical}, diff={diff}")
    
    print("Checking bias:")
    for test in range(5):
        i = np.random.randint(0, 8)
        output_idx = np.random.randint(0, 392)
        
        dh_l_plus_1 = np.zeros((1, 392))
        dh_l_plus_1[0, output_idx] = 1
        
        embedding_layer.forward(h_l)
        embedding_layer.backward(dh_l_plus_1)
        analytical = embedding_layer.db[i]
        
        embedding_layer.b[i] += epsilon
        f_pos = embedding_layer.forward(h_l)[0, output_idx]
        
        embedding_layer.b[i] -= 2 * epsilon
        f_neg = embedding_layer.forward(h_l)[0, output_idx]
        
        embedding_layer.b[i] += epsilon  
        
        numerical = (f_pos - f_neg) / (2 * epsilon)
        diff = abs(analytical - numerical)
        
        print(f"b[{i}]: analytical={analytical}, numerical={numerical}, diff={diff}")

def gradient_check_softmax():
    print("\nGradient checking: Softmax Cross-Entropy Layer")
    
    softmax_layer = softmax_cross_entropy_t()
    h_l = np.random.normal(0, 1, (1, 4))
    y = np.array([2])
    
    epsilon = 1e-5
    
    print("Checking gradients:")
    for i in range(4):
        ell, _ = softmax_layer.forward(h_l, y)
        dh_l_analytical = softmax_layer.backward()
        analytical = dh_l_analytical[0, i]
        
        h_l_perturbed = h_l.copy()
        h_l_perturbed[0, i] += epsilon
        ell_pos, _ = softmax_layer.forward(h_l_perturbed, y)
        
        h_l_perturbed[0, i] -= 2 * epsilon
        ell_neg, _ = softmax_layer.forward(h_l_perturbed, y)
        
        numerical = (ell_pos - ell_neg) / (2 * epsilon)
        diff = abs(analytical - numerical)
        
        input_val = h_l[0, i]
        print(f"logit {i} analytical={analytical}, numerical={numerical}, diff={diff}")

if __name__ == "__main__":
    
    gradient_check_linear()
    gradient_check_embedding()
    gradient_check_softmax()
    
    print("\nComplete")

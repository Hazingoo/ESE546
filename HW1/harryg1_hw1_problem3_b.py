import numpy as np

class embedding_t:
    def __init__(self):
        # Initialize to appropriate sizes, fill with random entries
        self.w = np.random.random((4, 4, 8))  
        self.b = np.random.random(8)         
        
        w_norm = np.linalg.norm(self.w)
        b_norm = np.linalg.norm(self.b)
        self.w = self.w / w_norm
        self.b = self.b / b_norm

    def forward(self, h_l):
        
        batch_size = h_l.shape[0]
        h_l_plus_1 = np.zeros((batch_size, 7, 7, 8))
        
        for b in range(batch_size):
            for i in range(7):  
                for j in range(7):  
                    # Extract 4×4 patch starting at position (4*i, 4*j)
                    patch = h_l[b, 4*i:4*i+4, 4*j:4*j+4]
                    for k in range(8):  # 8 feature channels
                        # sum(W_{i',j',k} * x_{i',j'}) + b_k
                        h_l_plus_1[b, i, j, k] = np.sum(self.w[:, :, k] * patch) + self.b[k]
        
        # Cache h^l for backward pass
        self.h_l = h_l
        
        # Flatten to 392 dimensions
        return h_l_plus_1.reshape(batch_size, -1)
    
    def backward(self, dh_l_plus_1_flat):
        
        batch_size = dh_l_plus_1_flat.shape[0]

        dh_l_plus_1 = dh_l_plus_1_flat.reshape(batch_size, 7, 7, 8)
        
        # Initialize gradients
        dw = 0 * self.w  
        db = 0 * self.b  
        dh_l = np.zeros((batch_size, 28, 28))
        
        # Compute gradients using chain rule
        for b in range(batch_size):
            for i in range(7):
                for j in range(7):
                    # Get the original patch in forward pass
                    patch = self.h_l[b, 4*i:4*i+4, 4*j:4*j+4]
                    for k in range(8):
                        # dL/dW = dL/dh^(l+1) × patch
                        dw[:, :, k] += dh_l_plus_1[b, i, j, k] * patch
                        # dL/db = dL/dh^(l+1)
                        db[k] += dh_l_plus_1[b, i, j, k]
                        # dL/dh^l = dL/dh^(l+1) × W
                        dh_l[b, 4*i:4*i+4, 4*j:4*j+4] += dh_l_plus_1[b, i, j, k] * self.w[:, :, k]
        
        # Store gradients for SGD update
        self.dw = dw
        self.db = db
        return dh_l
    
    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
import numpy as np


class linear_t:
    def __init__(self):
        self.w = np.random.random((10, 392))  
        self.b = np.random.random(10)  
        
        w_norm = np.linalg.norm(self.w)
        b_norm = np.linalg.norm(self.b)
        self.w = self.w / w_norm
        self.b = self.b / b_norm

    def forward(self, h_l):
        # computes the linear transformation
        h_l_plus_1 = h_l @ self.w.T + self.b
        
        # Cache h^l in forward pass
        self.h_l = h_l
        
        return h_l_plus_1

    def backward(self, dh_l_plus_1):        
        # dL/dh^l = dL/dh^(l+1) * dh^(l+1)/dh^l 
        dh_l = dh_l_plus_1 @ self.w
        
        # dL/dW = dL/dh^(l+1) * dh^(l+1)/dW 
        dw = dh_l_plus_1.T @ self.h_l
        
        # dL/db = dL/dh^(l+1) * dh^(l+1)/db = sum over batch dimension
        db = np.sum(dh_l_plus_1, axis=0)
        
        # Store gradients for SGD update
        self.dw = dw
        self.db = db
        
        return dh_l

    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
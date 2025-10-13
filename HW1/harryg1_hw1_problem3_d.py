import numpy as np

class relu_t:
    def __init__(self):
        pass

    def forward(self, h_l):
        h_l_plus_1 = np.maximum(0, h_l)
        
        self.h_l = h_l
        
        return h_l_plus_1

    def backward(self, dh_l_plus_1):
        # ReLU derivative is 1 if input > 0, else 0
        dh_l = dh_l_plus_1 * (self.h_l > 0)
        
        return dh_l
    def zero_grad(self):
        # ReLU has no parameters, so nothing to zero
        pass

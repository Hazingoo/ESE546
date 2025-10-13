import numpy as np

class softmax_cross_entropy_t:
    def __init__(self):
        # No parameters, nothing to initialize
        pass

    def forward(self, h_l, y):
        batch_size = h_l.shape[0]
        
        # Compute softmax probabilities 
        exp_logits = np.exp(h_l)
        h_l_plus_1 = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cache for backward pass
        self.h_l_plus_1 = h_l_plus_1
        self.y = y
        
        # Compute cross-entropy loss for each sample
        batch_indices = np.arange(batch_size)
        individual_losses = -np.log(h_l_plus_1[batch_indices, y])
        
        # Compute average loss over mini-batch
        ell = np.mean(individual_losses)
        
        # Compute classification error error
        predictions = np.argmax(h_l_plus_1, axis=1)
        error = np.mean(y != predictions)
        
        return ell, error

    def backward(self):
        batch_size = self.h_l_plus_1.shape[0]
        
        # Initialize gradient
        dh_l = np.copy(self.h_l_plus_1)
        
        # Subtract 1 from the true class probabilities
        batch_indices = np.arange(batch_size)
        dh_l[batch_indices, self.y] -= 1
        
        # Average over mini-batch
        dh_l = dh_l / batch_size
        
        return dh_l
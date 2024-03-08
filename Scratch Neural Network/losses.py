import numpy as np

from nn import Module


class L2Loss(Module):
    def __init__(self):
        pass

    def initialize(self, rng):
        pass

    def forward(self, input, target) -> np.float32:
        self.input = input
        self.target = target
        diff = input.reshape(input.shape[0], -1) - target.reshape(target.shape[0], -1)
        output = np.sum(diff ** 2, axis=1)
        self.n = output.shape[0]
        output = np.sum(output) / self.n
        return output

    def backward(self, delta):
        return 2 * (self.input - self.target) * delta / self.n


class SoftmaxWithLogitsLoss(Module):
    # the Softmax function is a loss function, not like the linear in layers.py, there is not any learnable parameters like weight and bias
    def __init__(self):
        pass
    
    def initialize(self, rng):
        pass
    
    def forward(self, input, target) -> np.float32:
        """
        Forward pass of the softmax cross-entropy loss.
        Hint: store the input and target or other necessary intermediate values for the backward pass.
        Args:
            input: n x n_class matrix with a d-dimensional feature for each of the n images
            target: n x n_class vector for each of the images
        Returns:
            loss: scalar, the average negative log likelihood loss over the n images
        """
        #store the input and target or other necessary intermediate values for the backward pass.
        self.input = input
        self.target = target
        samples = input.shape[0]
        # find the max value of each sample
        #the use of reshape can help avoid issues related to axis bounds
        max_value = np.max(input,axis = 1).reshape(-1,1)
        # I need to substract the max value from each logit before exp
        exp_input = np.exp(input - max_value)
        # sum the exp value for each sample
        sum = np.sum(exp_input, axis = 1).reshape(-1,1)
        # get the log of sum and add back the max value
        log_sum = np.log(sum) + max_value
        # get the log-softmax values
        log_softmax = input - log_sum
        # consider all negative log likelihood
        loss = -np.sum(log_softmax * target) / samples
        
        return loss
    
    # in this way, avoid taking the log of 0
        
        
     

        
        
        
        
        

    def backward(self, delta):
        """
        Backward pass of the softmax cross-entropy loss.
        Hint: use the stored input and target.
        Args:
            delta: scalar, the upstream gradient.
        Returns:
            gradient: n x n_class, gradient with respect to the input.
        """
        
        #first we still need get the softmax values that we stored in forward
        # the use of reshape can help avoid issues related to axis bounds
        # compute the exp of the logits
        exp_input = np.exp(self.input - np.max(self.input, axis=1).reshape(-1, 1))
        # get the softmax values
        output = exp_input / np.sum(exp_input, axis=1).reshape(-1, 1)
        # also we need to find the average gradient
        gradient = (output - self.target) / self.input.shape[0]
        
        return gradient * delta
        

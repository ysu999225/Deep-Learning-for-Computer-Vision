import numpy as np

from nn import Module


class Linear(Module):
    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.params = {'weight': None, 'bias': None}
        self.grads = {'weight': None, 'bias': None}

    def initialize(self, rng):
        gain = np.sqrt(2)
        fan_in = self.input_channel
        fan_out = self.output_channel
        bound = gain * np.sqrt(3 / fan_in)
        self.params['weight'] = rng.uniform(-bound, bound,
                                            (self.output_channel, self.input_channel))
        bound = 1 / np.sqrt(fan_in)
        self.params['bias'] = rng.uniform(-bound, bound, (self.output_channel,))

    def forward(self, input):
        """
        The forward pass of a linear layer.
        Store anything you need for the backward pass in self.
        Args:
            input: N x input_channel
        Returns:
            output: N x output_channel
        """
        assert (input.ndim == 2)
        assert (input.shape[1] == self.input_channel)

        self.input = input
        self.output = np.dot(input, self.params['weight'].T) + self.params['bias']
        return self.output

    def backward(self, delta):
        """
        Backward pass of a linear layer.
        Use the values stored from the forward pass to compute gradients.
        Store the gradients in `self.grads` dict.
        :param delta: Upstream gradient, N x output_channel.
        :return: downstream gradient, N x input_channel.
        """
        assert (delta.ndim == 2)
        assert (delta.shape[1] == self.output_channel)

        self.grads['weight'] = np.dot(delta.T, self.input)
        self.grads['bias'] = np.sum(delta, axis=0)
        return np.dot(delta, self.params['weight'])


class Flatten(Module):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Args:
            input: (N, any shape)
        Returns:
            output: (N, product of input shape)
        """
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, delta):
        """
        Args:
            delta: (N, product of input shape)
        Returns:
            output: (N, any shape)
        """
        return delta.reshape(self.input_shape)


class ReLU(Module):
    def __init__(self):
        self.params = {}
        ###########
        self.grads = {}

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Args:
            input: any shape
        Returns:
            output: same shape as the input.
        """
        #ReLU(x) = max (0,x)
        self.input = input
        return np.maximum(0,input)

    def backward(self, delta):
        """
        Args:
            delta: upstream gradient, any shape
        Returns:
            gradient: same shape as the input.
        """
        # We need to take the derivative of the two outputs with respect to x
        # The gradient of ReLU is 1 for x >0 and 0 for x<0
        # np.where function help me check the condition > 0 or not, > 0 replace 1, <= 0 replace 0
        gradient = np.where(self.input > 0, 1, 0)
        return delta * gradient
        
        
        pass


class Sequential(Module):
    def __init__(self, *layers):
        self.params = {}
        self.layers = layers
        self.rng = np.random.RandomState(1234)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def initialize(self, rng):
        for layer in self.layers:
            layer.initialize(self.rng)

    def forward(self, x):
        """
        Args:
            x: input to the network
        Returns:
            output: output of the network (after the last layer)
        """
        #as did above, use forward function to forward pass
        for layer in self.layers:
            x = layer.forward(x)
        return x
        

    def backward(self, delta):
        """
        Args:
            delta: gradient from the loss
        Returns:
            delta: gradient to be passed to the previous layer
        """
        # as did above, use backward function to backward pass
        # should in reverse order 
        #operands could not be broadcast together with shapes (4,2) (4,3) 
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
            


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        self.params = {}
        self.grads = {}
        self.kernel_size = kernel_size

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Pool the input by taking the max value over non-overlapping kernel_size x kernel_size blocks.
        Hint: Use a double for-loop to do this.
        Args:
            input: images of (N, H, W, C)
        Returns:
            output: pooled images of (N, H // k_h, W // k_w, C).
        """
        assert (input.ndim == 4)
        assert (input.shape[1] % self.kernel_size[0] == 0)
        assert (input.shape[2] % self.kernel_size[1] == 0)
        #The input size is (N, H, W, C), it is a 4D 
        N,H,W,C = input.shape 
        #The kernel size is (k_h, k_w)
        k_h, k_w = self.kernel_size
        #The out size should be (N, H // k_h, W // k_w, C). To simplify matters, you can assume that the height and the width are divisible by the kernel size.
        out_height = H // k_h
        out_width = W // k_w
        #You don't need to vectorize the implementation. You may use a double for-loop to iterate over the input.
        # use np.zeros to initialize it filled with zeros
        output = np.zeros((N,out_height,out_width,C))
        #You may use a double for-loop to iterate over the input.
        # consider the speed, so only loop literate over height and width
    #for n in range(N):
        for h in range(0, H, k_h):
            for w in range (0,W, k_w):
                    #for c in range (C):
                # slice the region
                region = input[ :, h:h+k_h, w:w+k_w, :]
                 # combine the height dimension1 and width dimension2 in one dimension
                 # find the max_value
                region_max = np.max(region, axis=(1, 2))
                output[:,h // k_h,w//k_w,:] = region_max
        #You may want to store the input for the backward pass.
        # store the entire input
        self.input = input
        return output
        
        
        

    def backward(self, delta):
        """
        Args:
            delta: upstream gradient, same shape as the output
        Returns:
            gradient: same shape as the input.
        """
        #The input size is (N, H, W, C)
        # use the store input above
        N,H,W,C = self.input.shape
        #The kernel size is (k_h, k_w)
        k_h, k_w = self.kernel_size
        #The out size should be (N, H // k_h, W // k_w, C). To simplify matters, you can assume that the height and the width are divisible by the kernel size.
        out_height = H // k_h
        out_width = W // k_w
        #You don't need to vectorize the implementation. You may use a double for-loop to iterate over the input.
        # gradient same data shape as the self.input
        gradient = np.zeros(self.input.shape, dtype=self.input.dtype)
        # almost the same above write double loop
        #You may use a double for-loop to iterate over the input.
        # consider the speed, so only loop literate over height and width
        #for n in range(N):
        for h in range(0, H, k_h):
            for w in range (0,W, k_w):
                    #for c in range (C):
                # slice the region by consider the self.input
                region = self.input[ :, h:h+k_h, w:w+k_w, :]
                 # combine the height dimension1 and width dimension2 in one dimension
                 # find the max_value in this region
                region_max = np.max(region, axis=(1, 2))
                # consider the reshape functionm and reshape it to (n,1,1,c)
                #because Only the maximum value in each window is kept. Therefore, only these pixels have non-zero gradients.
                # 1 means true here
                region_max = region_max.reshape(region_max.shape[0], 1, 1, region_max.shape[1])
                #create a mask equals to maximum value means True, and other mean False
                mask = (region == region_max)
                #reshape to match the k_h,k_w of mask
                gradient[ :, h:h+k_h, w:w+k_w, :] = mask * delta[ :,h//k_h,w//k_w, :].reshape(N,1,1,C)
        return gradient



class Conv2d(Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.params = {
            'weight': np.zeros((self.output_channel,
                                self.input_channel,
                                self.kernel_size[0],
                                self.kernel_size[1])),
            'bias': np.zeros((self.output_channel,)),
        }
        self.grads = {
            'weight': np.zeros((self.output_channel,
                                self.input_channel,
                                self.kernel_size[0],
                                self.kernel_size[1])),
            'bias': np.zeros((self.output_channel,)),
        }

    def initialize(self, rng):
        gain = np.sqrt(2)
        fan_in = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.output_channel
        bound = gain * np.sqrt(3 / fan_in)
        self.params['weight'] = rng.uniform(-bound, bound, self.params['weight'].shape)
        bound = 1 / np.sqrt(fan_in)
        self.params['bias'] = rng.uniform(-bound, bound, self.params['bias'].shape)

    def forward(self, input):
        """
        Convolve the input with the kernel and return the result.
        Hint:
            1. Use a double for-loop to do this.
            2. Recall the size of the kernel weight is (C_out, C_in, k_h, k_w)
                and the size of the kernel bias is (C_out,).
        Args:
            input: images of (N, H, W, C_in)
        Returns:
            output: images of (N, H', W', C_out) where H' = H - k_h + 1, W' = W - k_w + 1
        """
        #The input size is (N, H, W, C_in) with C_in denoting the number of input channels.
        N, H, W, C_in = input.shape
        #Recall the size of the kernel weight is (C_out, C_in, k_h, k_w) and the size of the kernel bias is (C_out,), where C_out denotes the number of output channels.
        #k_h,k_w = self.kernel_size
        #C_in = self.input_channel
        #C_out = self.output_channel
        C_out,C_in, k_h, k_w = self.params['weight'].shape
        #The out size should be (N, H - k_h + 1, W - k_w + 1, C_out). This means we don't pad the input and use a stride of 1
        out_height = H - k_h + 1
        out_width = W - k_w + 1
        #still initialize the output
        output = np.zeros((N,out_height,out_width,C_out))
        #You don't need to vectorize the implementation.
        #still consider the loop
        #we must add this loop, if I did not add this, will show axes do not match array
        for n in range(N):
            
            for h in range(out_height):
                for w in range (out_width):
                    for c_out in range(C_out):
                        weights = self.params['weight'][c_out]
                        biases = self.params['bias'][c_out]
                                # slice the region
                        region = input[n, h:h+k_h, w:w+k_w, :]
                                # rearranges the dimensions of an array according to a given sequence.
                                # transfer the shape the shape (k_h, k_w, C_in) to (C_in, k_h, k_w)
                        region = np.transpose(region, (2, 0, 1))
                                # do the matrix multiplication
                        output[n, h, w, c_out] = np.sum(region * weights) + biases
            #You may want to store the input for the backward pass.
        # store the entire input
        self.input = input
        return output
    
        
     
        
    



    def backward(self, delta):
        """
        Gradient with respect to the weights should be calculated and stored here.
        Args:
            delta: upstream gradient, same shape as the output
        Returns:
            gradient: same shape as the input.
            
        """

     
        # use the store input above
        N, H, W, C_in = self.input.shape
        #C_in = self.input_channel
        #C_out = self.output_channel
        #k_h,k_w = self.kernel_size
        C_out,C_in, k_h, k_w = self.grads['weight'].shape
        #The out size should be (N, H - k_h + 1, W - k_w + 1, C_out). This means we don't pad the input and use a stride of 1
        out_height = H - k_h + 1
        out_width = W - k_w + 1
        
        N,out_height,out_width,C_out = delta.shape
        
        
        
        #still initialize the output
        #gradient = np.zeros(self.input.shape)
        
        
        
        #initialize the gradients and gradient_weights and graident_bias
        # Initializing the gradients that has the same shape as self.input. 
        # Every element of this new array is set to zero.
        gradient = np.zeros_like(self.input)
        #initializes an array called gweights that has the same shape as self.params['weight']
        gweights = np.zeros_like(self.params['weight'])
        #calculates the gradient of the loss with respect to the biases
        #reshape
        reshaped_delta = delta.reshape(-1, delta.shape[-1])
        gbias = reshaped_delta.sum(axis=0)
     
    
        # initialize the weights and bias
        # write three loops for each
        # the first is for the input
        # Gradient with respect to Weights and Bias
        # Iterate over the spatial dimensions and channels
        #weights
        for n in range(delta.shape[0]):
            for h in range(delta.shape[1]):
                for w in range(delta.shape[2]):
                    for c_out in range(delta.shape[3]):
                        #Extract the new value
                        new_delta  = delta [n,h,w,c_out]
                        #slice the region
                        region = self.input[n, h:h+k_h, w:w+k_w, :]
                        # like I did above,
                        ## rearranges the dimensions of an array according to a given sequence.
                        # transfer the shape the shape (k_h, k_w, C_in) to (C_in, k_h, k_w)
                        region = np.transpose(region, (2, 0, 1))
                        #update gradient weights
                        gweights[c_out] += region * delta[n, h, w, c_out]
                        # update the gradient
                        gradient[n, h:h+k_h, w:w+k_w, :] += new_delta * np.transpose(self.params['weight'][c_out],(1, 2, 0))
            
        #gradient 
        # this loop will take lots of time   
        #for n in range(delta.shape[0]):
            #for h in range(delta.shape[1]):
                #for w in range(delta.shape[2]):
                    #for c_out in range(delta.shape[3]):
                        #for i in range(k_h):
                            #for j in range(k_w):
                                #for c_in in range(C_in):
                                    #gradient[n, h+i, w+j, c_in] += delta[n, h, w, c_out] * self.params['weight'][c_out, c_in, i, j]
                       
        #this is important
        # should be self.grads not the self.params  
        #This dictionary holds the gradients of the loss with respect to the parameters.             
        self.grads['weight'] = gweights
        self.grads['bias'] = gbias
                         
        return gradient
            
  
       

        
        
        

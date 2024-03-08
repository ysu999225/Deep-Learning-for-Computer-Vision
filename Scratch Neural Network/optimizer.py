import numpy as np


class SGD:
    def __init__(self, net):
        self.net = net
        self.steps = 0
        self.grads = {}

    def step(self, lr):
        for i, layer in enumerate(self.net.layers):
            for param in layer.params:
                grad = layer.grads[param]
                self.grads[f'{i}_{param}'] = grad
                layer.params[param] -= lr * self.grads[f'{i}_{param}']
        self.steps += 1


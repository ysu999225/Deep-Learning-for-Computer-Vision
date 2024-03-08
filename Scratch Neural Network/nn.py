from abc import ABC, abstractmethod
import numpy as np


class Module(ABC):
    @abstractmethod
    def initialize(self, rng):
        pass

    @abstractmethod
    def forward(self, input):
        pass
    
    @abstractmethod
    def backward(self, delta):
        pass

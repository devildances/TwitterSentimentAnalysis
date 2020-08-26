import numpy
from trax import fastmath

class Layer(object):
    def __init__(self):
        self.weights = None
    def forward(self, x):
        raise NotImplementedError
    def init_weights_and_state(self, input_signature, random_key):
        pass
    def init(self, input_signature, random_key):
        self.init_weights_and_state(input_signature, random_key)
        return self.weights
    def __call__(self, x):
        return self.forward(x)

class ReLU(Layer):
    def forward(self, x):
        activation = numpy.maximum(x, 0)
        return activation

class Dense(Layer):
    def __init__(self, n_units, init_std=0.1):
        self.n_units = n_units
        self.init_std = init_std
    def forward(self, x):
        dense = fastmath.numpy.dot(x, self.weights)
        return dense
    def init_weights_and_state(self, input_signature, random_key):
        input_shape = input_signature.shape
        w = self.init_std * fastmath.random.normal(key=random_key, shape=(input_shape[-1], self.n_units))
        self.weights = w
        return self.weights
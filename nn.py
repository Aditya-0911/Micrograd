import numpy as np
from engine import Value, build_topo

class Neuron:

    def __init__(self, input_size):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(input_size)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        z = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act = z.tanh()
        return act
    
    def parameters(self):
        return self.w + [self.b]
   
class Layer:

    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
 
class MLP:

    def __init__(self, input_size, hidden_sizes, output_size): # hidden_sizes is a list of sizes of hidden layers
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
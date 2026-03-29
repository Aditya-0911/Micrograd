import numpy as np 
import pandas as pd
import math

# topological sort to make sure we call the backward functions in the right order.
def build_topo(v,topo,visited):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child,topo,visited)
        topo.append(v)

class Value:

    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self._prev = set(_children) # _ means dont call it outside the class
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self): # for printing purposes
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = backward
        
        return out
    
    def __rmul__(self, other):
        """ This helps when we are doing something like 2*a where 2 is not a Value class object
        so this reverse the multiplication order and calls __mul__ function"""
        return self * other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")

        def backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        
        out._backward = backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        
        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = backward
    
        return out
    
    def exp(self):
        x = self.data 
        out = Value(math.exp(x),(self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()
        build_topo(self, topo, visited)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
![header](header.jpg)

# Micrograd

A scalar-valued autograd engine built from scratch in Python, following 
Karpathy's micrograd. Implements backpropagation over a dynamical computational 
graph with a small neural network library on top.

## Structure
- `engine.py` — Value class with full operator overloading and backward pass
- `nn.py` — Neuron, Layer, and MLP built on top of the Value class
- `test.py` — verifies micrograd gradients match PyTorch exactly

## What it implements
- Forward and backward pass for all basic operations (+, *, **, tanh, exp)
- Topological sort for correct gradient accumulation order
- Automatic handling of variables used multiple times in the graph
- A working MLP trained with gradient descent

## Computational Graph
Visualization of a sample forward pass showing how the Value class 
builds the computation graph automatically:

![Computational Graph](output.svg)

The full MLP graph is significantly larger — each arrow represents 
a Value node that tracks its own gradient during backpropagation.

## Verified against PyTorch
Running `python test.py` confirms all gradients match PyTorch to 1e-6 precision.

## Reference
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [Video walkthrough](https://www.youtube.com/watch?v=VMj-3S1tku0)

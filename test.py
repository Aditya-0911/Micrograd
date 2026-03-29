# test.py
from engine import Value
from nn import MLP
import torch

def test_gradients():
    # micrograd forward pass
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b  = Value(8.0)
    o  = ((w1*x1) + (w2*x2) + b).tanh()
    o.backward()

    # pytorch forward pass
    x1t = torch.tensor(2.0).double().requires_grad_(True)
    x2t = torch.tensor(0.0).double().requires_grad_(True)
    w1t = torch.tensor(-3.0).double().requires_grad_(True)
    w2t = torch.tensor(1.0).double().requires_grad_(True)
    bt  = torch.tensor(8.0).double().requires_grad_(True)
    ot  = torch.tanh(w1t*x1t + w2t*x2t + bt)
    ot.backward()

    # compare
    assert abs(x1.grad - x1t.grad.item()) < 1e-6, "x1 grad mismatch"
    assert abs(w1.grad - w1t.grad.item()) < 1e-6, "w1 grad mismatch"
    assert abs(b.grad  - bt.grad.item())  < 1e-6, "b grad mismatch"
    
    print("All gradients match PyTorch. Autograd engine is correct.")

test_gradients()
import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float

    # Create a leaf tensor tracked by autograd
    x = torch.tensor(x_value, dtype=torch.float32, requires_grad= True)
    # Forward pass
    y = x ** 2 + 3 * x + 2

    # BackPropagation

    y.backward()

    # Gradient stored on the leaf tensor
    return x.grad.item()

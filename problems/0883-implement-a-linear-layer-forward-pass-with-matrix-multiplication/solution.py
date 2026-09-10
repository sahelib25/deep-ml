import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: implement y = x W^T + b using PyTorch ops
    y = x @ W.transpose(-1, -2) + b
    return y

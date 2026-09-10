import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    torch.flatten(x)
    return torch.reshape(x, new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    # TODO: swap the last two dimensions of x
    return torch.transpose(x, -1, -2)

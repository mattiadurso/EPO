import torch
import torch.nn as nn


class Depth(nn.Module):  # Changed: inherit from nn.Module
    def __init__(
        self,
        height: int,
        width: int,
        depth: torch.Tensor,
        grad=True,
        device="cuda",
    ):
        """Depth module to hold depth parameters"""
        super().__init__()
        self.device = device
        self.grad = grad

        self.depth = depth.to(device)

        self.params = nn.Parameter(
            self.depth.clone().detach().to(device), requires_grad=self.grad
        )
        self.hw = (height, width)

    def forward(self):
        """Return depth parameters - ensures gradient flow"""
        return self.params

    def parameters(self, recurse: bool = True):
        return iter([self.params])

    def __repr__(self):
        out = (
            f"Depth(height={self.hw[0]}, width={self.hw[1]}, "
            + f"parameters={len(self.params.data.detach().tolist()):,})"
        )
        return out

    def __str__(self):
        return self.__repr__()

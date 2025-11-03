import torch
import torch.nn as nn


class Camera:
    def __init__(
        self,
        cam_id: int,
        model: str,
        parameters: torch.Tensor,
        grad=True,
        device="cuda",
    ):
        """
        Camera class representing camera intrinsics.
        """
        self.device = device
        self.grad = grad
        self.id = cam_id
        self.model = model

        # Create parameter as leaf tensor - this is what gets optimized
        self.params = nn.Parameter(
            parameters.clone().detach().to(device), requires_grad=self.grad
        )

    def intrinsic_matrix(self, inverse: bool = False) -> torch.Tensor:
        """Construct K matrix from parameters on-the-fly"""
        if self.model == "PINHOLE":
            assert (
                self.params.shape[0] == 4
            ), "Pinhole model requires 4 parameters: fx, fy, cx, cy"
            fx = self.params[0]
            fy = self.params[1]
            cx = self.params[2]
            cy = self.params[3]

            # Build K matrix - this will maintain gradient connection to self.params
            K = torch.zeros(3, 3, dtype=self.params.dtype, device=self.device)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1.0

        elif self.model == "SIMPLE_PINHOLE":
            assert (
                self.params.shape[0] == 3
            ), "Simple pinhole model requires 3 parameters: f, cx, cy"
            f = self.params[0]
            cx = self.params[1]
            cy = self.params[2]

            K = torch.zeros(3, 3, dtype=self.params.dtype, device=self.device)
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1.0
        else:
            raise ValueError(f"Unsupported camera model: {self.model}")

        if inverse:
            K = torch.inverse(K)

        return K

    def parameters(self):
        """Return list of trainable parameters - only self.params is a leaf tensor"""
        return [self.params]

    def __repr__(self):
        return f"Camera(id={self.id}, model={self.model}, parameters={self.params.data.detach().tolist()})"

    def __str__(self):
        return self.__repr__()

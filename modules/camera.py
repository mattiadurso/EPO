import torch
import torch.nn as nn


# camera class as nn.Module
class Camera:
    def __init__(
        self,
        cam_id: int,
        model: str,
        parameters: torch.Tensor,
        grad=True,
    ):
        """
        Camera class representing camera intrinsics.
            Args:
            cam_id (int): Camera ID.
            model (str): Camera model type.
            parameters (torch.Tensor): Camera intrinsic parameters.
            grad (bool): Whether to track gradients.
        """
        self.grad = grad

        self.id = cam_id
        self.model = model
        self.parameters = nn.Parameter(parameters, requires_grad=self.grad).float()

        self.matrix = self.intrinsic_matrix()

    def intrinsic_matrix(self, inverse: bool = False) -> torch.Tensor:
        if self.model == "PINHOLE":
            assert (
                self.parameters.shape[0] == 4
            ), "Pinhole model requires 4 parameters: fx, fy, cx, cy"
            self.fx = self.parameters[0]
            self.fy = self.parameters[1]
            self.cx = self.parameters[2]
            self.cy = self.parameters[3]

            K = torch.stack(
                [
                    torch.stack([self.fx, torch.zeros_like(self.fx), self.cx]),
                    torch.stack([torch.zeros_like(self.fy), self.fy, self.cy]),
                    torch.stack(
                        [
                            torch.zeros_like(self.fx),
                            torch.zeros_like(self.fx),
                            torch.ones_like(self.fx),
                        ]
                    ),
                ]
            )
        elif self.model == "SIMPLE_PINHOLE":
            assert (
                self.parameters.shape[0] == 3
            ), "Simple pinhole model requires 3 parameters: f, cx, cy"
            self.f = self.parameters[0]
            self.cx = self.parameters[1]
            self.cy = self.parameters[2]
            K = torch.stack(
                [
                    torch.stack([self.f, torch.zeros_like(self.f), self.cx]),
                    torch.stack([torch.zeros_like(self.f), self.f, self.cy]),
                    torch.stack(
                        [
                            torch.zeros_like(self.f),
                            torch.zeros_like(self.f),
                            torch.ones_like(self.f),
                        ]
                    ),
                ]
            )

        if inverse:
            K = torch.inverse(K)

        return K

    def __repr__(self):
        return f"Camera(id={self.id}, model={self.model}, parameters={self.parameters.data.detach().tolist()})"

    def __str__(self):
        return self.__repr__()

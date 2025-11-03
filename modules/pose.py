import torch
import torch.nn as nn
import kornia.geometry.conversions as kgc


class Pose:
    def __init__(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        grad_q: bool = False,
        grad_t: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        # Convert rotation to quaternion
        q = self.rotation_matrix_to_quaternion(R)
        t_vec = t.squeeze().reshape(3)

        # Create leaf parameters - critical for optimization
        self.q = nn.Parameter(
            q.clone().detach().to(device),
            requires_grad=False,
        )
        self.t = nn.Parameter(
            t_vec.clone().detach().to(device),
            requires_grad=grad_t,
        )

    def normalize_quat(self, quaternion):
        """Normalize a quaternion."""
        return kgc.normalize_quaternion(quaternion)

    def quaternion_to_rotation_matrix(self, quaternion):
        """Convert a quaternion to a rotation matrix."""
        quaternion = self.normalize_quat(quaternion)
        return kgc.quaternion_to_rotation_matrix(quaternion)

    def rotation_matrix_to_quaternion(self, rotmat):
        """Convert a rotation matrix to a quaternion."""
        quaternion = kgc.rotation_matrix_to_quaternion(rotmat)
        return self.normalize_quat(quaternion)

    def rotation_matrix(self) -> torch.Tensor:
        """Return rotation matrix from quaternion"""
        return self.quaternion_to_rotation_matrix(self.q)

    def projection_matrix(self, inverse: bool = False) -> torch.Tensor:
        """Return 4x4 projection matrix (P)"""
        R = self.rotation_matrix()
        t = self.t.unsqueeze(1)  # (3, 1)

        # Build P matrix
        P = torch.zeros(4, 4, dtype=R.dtype, device=R.device)
        P[:3, :3] = R
        P[:3, 3] = t.squeeze()
        P[3, 3] = 1.0

        if inverse:
            P = torch.inverse(P)

        return P

    def parameters(self):
        """Return list of trainable parameters - only leaf tensors"""
        return [self.q, self.t]

    def __repr__(self):
        return f"q: {self.q.cpu().detach()} \nt: {self.t.cpu().detach()}"

    def __str__(self):
        return self.__repr__()

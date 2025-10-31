import torch
import torch.nn as nn
import kornia.geometry.conversions as kgc


class Pose:
    def __init__(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        grad_q: bool = True,
        grad_t: bool = True,
    ):
        super().__init__()

        # store rotation as quaternion and translation
        self.q = self.rotation_matrix_to_quaternion(R)
        self.t = t.squeeze().reshape(3)

        # make parameters trainable
        self.q = nn.Parameter(self.q, requires_grad=False)  # keep rotation fixed
        self.t = nn.Parameter(self.t, requires_grad=True)

    def normalize_quat(self, quaternion):
        """
        Normalize a quaternion.
        """
        return kgc.normalize_quaternion(quaternion)

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert a quaternion to a rotation matrix.
        """
        # Normalize quaternion
        quaternion = self.normalize_quat(quaternion)
        return kgc.quaternion_to_rotation_matrix(quaternion)

    def rotation_matrix_to_quaternion(self, rotmat):
        """
        Convert a rotation matrix to a quaternion.
        """
        quaternion = kgc.rotation_matrix_to_quaternion(rotmat)

        return self.normalize_quat(quaternion)

    def rotation_matrix(self) -> torch.Tensor:
        # return rotation matrix from quaternion
        return self.quaternion_to_rotation_matrix(self.q)

    def projection_matrix(self, inverse: bool = False) -> torch.Tensor:
        # return 4x4 projection matrix (P)
        R = self.rotation_matrix()
        t = self.t.unsqueeze(1)  # make it (3, 1)
        Rt = torch.cat([R, t], dim=1)  # (3, 4)

        # Create bottom row - explicitly detached to ensure it's not trainable
        bottom_row = torch.zeros((1, 4), dtype=Rt.dtype, device=Rt.device)
        bottom_row[0, 3] = 1.0
        bottom_row = bottom_row.detach()  # Explicit: ensure no gradients

        P = torch.cat([Rt, bottom_row], dim=0)  # (4, 4)

        if inverse:
            P = torch.inverse(P)

        return P

    def __repr__(self):
        return f"Pose(R={self.rotation_matrix().cpu().detach()}, t={self.t.cpu().detach()})"

    def __str__(self):
        return self.__repr__()

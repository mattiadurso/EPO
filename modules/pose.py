import torch
import torch.nn as nn
import pypose as pp
import kornia.geometry.conversions as kgc


class Pose(nn.Module):  # Changed: inherit from nn.Module
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
        q = self.rotation_matrix_to_quaternion(R).detach().clone().float().to(device)
        q = torch.roll(q, shifts=-1)  # convert wxyz -> to xyzw
        self.q_param = nn.Parameter(q, requires_grad=grad_q)
        self.q = pp.SO3(self.q_param)

        # Translation vector
        t_vec = t.squeeze().reshape(3).detach().clone().float().to(device)
        self.t = nn.Parameter(t_vec, requires_grad=grad_t)

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
        # if self.q is a LieTensor, use its matrix method
        if isinstance(self.q, pp.LieTensor):
            return self.q.matrix()
        else:
            return self.quaternion_to_rotation_matrix(self.q_param)

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

    def parameters(self, t=True, q=True):
        """Return iterator of trainable parameters - only leaf tensors"""
        params = []
        if q:
            params.append(self.q_param)
        if t:
            params.append(self.t)
        return iter(params)  # Changed: return iterator instead of list

    def __repr__(self):
        return f"q: {self.q_param.cpu().detach()} \nt: {self.t.cpu().detach()}"

    def __str__(self):
        return self.__repr__()

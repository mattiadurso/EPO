"""Small MLP that refines a 3x4 pose matrix with a residual + orthonormalization.

Architecture follows ACE Zero (supplementary material): six fully-connected
layers, a residual connection between layers 1 and 3, and a final
Gram-Schmidt step that re-orthonormalizes the rotation block of the output
3x4 pose matrix.

The Gram-Schmidt primitive is exposed at module level as
:func:`gram_schmidt_rotation` so it can be reused by other modules (e.g. the
:class:`PoseModule`, which stores its rotation as a learnable 3x3 matrix and
needs to re-orthonormalize on every fetch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_schmidt_rotation(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Re-orthonormalize a batch of (approximately) rotation matrices.

    Uses the standard 6D representation: keep rows 1 and 2, normalize r1,
    project r2 onto the plane orthogonal to r1, normalize r2, then set
    r3 = r1 x r2 to guarantee a right-handed orthonormal frame.

    Args:
        R: ``(..., 3, 3)`` tensor whose row-vectors are treated as the rotation
           basis (world-to-camera convention is unaffected — we only
           orthonormalize).
        eps: Numerical floor passed to :func:`F.normalize`.

    Returns:
        ``(..., 3, 3)`` orthonormal rotation matrix with ``det(R) = +1``.
    """
    r1 = R[..., 0, :]
    r2 = R[..., 1, :]

    r1 = F.normalize(r1, p=2, dim=-1, eps=eps)
    # Project r2 onto r1 and subtract; (...,1) dot product
    dot = torch.sum(r1 * r2, dim=-1, keepdim=True)
    r2 = F.normalize(r2 - dot * r1, p=2, dim=-1, eps=eps)
    r3 = torch.cross(r1, r2, dim=-1)

    return torch.stack([r1, r2, r3], dim=-2)


class PoseRefinementMLP(nn.Module):
    """Pose Refinement MLP as described in ACE Zero (Supplementary Material).

    Architecture:
    - Input: 3x4 pose matrix
    - Layers: 6 Linear Layers
    - Residual Connection: Between Layer 1 and Layer 3
    - Output: 3x4 pose matrix with orthonormalized rotation via Gram-Schmidt.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """Args:
        input_dim: Flattened input size (12 for a 3x4 pose matrix).
        output_dim: Flattened output size (12 for a 3x4 pose matrix).
        hidden_dim: Hidden width of the fully-connected stack.
        """
        super().__init__()
        # The 6 layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        self.skip = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Standard initialization.
        Note: For refinement networks, it is often beneficial to initialize the
        final layer with very small weights so the initial prediction is close
        to identity (zero offset), preventing large jumps at the start of training.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize the last layer to near-zero to start with small updates
        # at step 0: mlp(x) = x
        nn.init.normal_(self.fc6.weight, mean=0, std=1e-8)
        nn.init.constant_(self.fc6.bias, 0)

    def gram_schmidt(self, poses):
        """Orthonormalize the rotation block of (B, 3, 4) pose matrices.

        Thin wrapper around :func:`gram_schmidt_rotation` that keeps the
        translation column untouched.

        Args:
            poses: Tensor of shape (B, 3, 4)

        Returns:
            poses: Tensor of shape (B, 3, 4) with orthonormalized rotations.
        """
        R_ortho = gram_schmidt_rotation(poses[:, :3, :3])
        return torch.cat([R_ortho, poses[:, :3, 3:]], dim=2)

    def forward(self, x):
        """Args:
            initial_pose_flat: (Batch_Size, 3, 4) tensor containing 3x4 matrices pose.

        Returns:
            refined_pose: (Batch_Size, 3, 4) tensor, orthonormalized.
        """
        # flatten x
        B, H, W = x.shape
        x0 = x.view(B, -1)

        x1 = self.act(self.fc1(x0))
        x2 = self.act(self.fc2(x1))
        x3 = self.act(self.fc3(x2) + self.skip(x1))  # Residual connection

        x4 = self.act(self.fc4(x3))
        x5 = self.act(self.fc5(x4))

        x6 = self.fc6(x5) + x0  # P = MLP(P) + P

        # Orthonormalisation in FP32. The caller may have us inside a BF16
        # autocast scope (see PoseModule.apply_mlp with use_amp=True); the
        # cross products and per-vector normalisations inside Gram-Schmidt are
        # precision-sensitive and lose ~2 digits in BF16, which distorts R
        # enough to slow / destabilise the rotation refinement. We unconditionally
        # disable autocast here and upcast the input — a no-op when the outer
        # scope is already FP32.
        with torch.autocast(device_type="cuda", enabled=False):
            return self.gram_schmidt(x6.view(B, H, W).float())


if __name__ == "__main__":
    try:
        from torchinfo import summary

        torchinfo_available = True
    except:
        torchinfo_available = False

    # Simple test
    mlp = PoseRefinementMLP()
    if torchinfo_available:
        summary(mlp, input_size=(16, 3, 4))
    else:
        # print num of parameters
        num_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print(f"PoseRefinementMLP has {num_params} trainable parameters.")

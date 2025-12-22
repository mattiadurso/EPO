import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRefinementMLP(nn.Module):
    """
    Pose Refinement MLP as described in ACE Zero (Supplementary Material).

    Architecture:
    - Input: 3x4 pose matrix
    - Layers: 6 Linear Layers
    - Residual Connection: Between Layer 1 and Layer 3
    - Output: 3x4 pose matrix with orthonormalized rotation via Gram-Schmidt.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        # The 6 layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Standard initialization.
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
        """
        Applies Gram-Schmidt orthonormalization to the 3x3 rotation parts
        of the 3x4 pose matrices.

        Args:
            poses: Tensor of shape (B, 3, 4)

        Returns:
            poses: Tensor of shape (B, 3, 4) with orthonormalized rotations.
        """
        # Split into Rotation (3x3) and Translation (3x1)
        R = poses[:, :3, :3]
        t = poses[:, :3, 3:]

        # Extract row vectors (assuming standard World-to-Cam convention)
        r1 = R[:, 0, :]
        r2 = R[:, 1, :]
        # r3 is re-computed via cross product

        # Normalize first vector
        r1 = F.normalize(r1, p=2, dim=1, eps=1e-8)

        # Project r2 onto r1 and subtract to make orthogonal
        # dot product shape: (B, 1)
        dot = torch.sum(r1 * r2, dim=1, keepdim=True)
        r2 = r2 - dot * r1

        # Normalize second vector
        r2 = F.normalize(r2, p=2, dim=1, eps=1e-8)

        # Compute third vector via cross product (ensures right-handed system)
        r3 = torch.cross(r1, r2, dim=1)

        # Stack back into rotation matrix
        # Stack dim=1 results in (B, 3, 3)
        R_ortho = torch.stack([r1, r2, r3], dim=1)

        # Recombine with translation
        poses_ortho = torch.cat([R_ortho, t], dim=2)

        return poses_ortho

    def forward(self, x):
        """
        Args:
            initial_pose_flat: (Batch_Size, 3, 4) tensor containing 3x4 matrices pose.

        Returns:
            refined_pose: (Batch_Size, 3, 4) tensor, orthonormalized.
        """
        # flatten x
        B = x.shape[0]
        x = x.view(B, -1)

        # Forward pass with residual connections
        x0 = self.act(self.fc1(x))
        x1 = self.act(self.fc2(x0))
        x2 = self.act(self.fc3(x1) + x0)  # Residual connection

        x3 = self.act(self.fc4(x2))
        x4 = self.act(self.fc5(x3))
        x5 = self.fc6(x4) + x  # Residual connection

        return self.gram_schmidt(x5.view(B, 3, 4))

    def predict_residuals(self, x):
        """
        Predicts only the residuals to be added to the input poses.

        Args:
            x: (Batch_Size, 3, 4) tensor containing 3x4 matrices pose.

        Returns:
            residuals: (Batch_Size, 3, 4) tensor of predicted residuals.
        """
        # flatten x
        B = x.shape[0]
        x = x.view(B, -1)

        # Forward pass with residual connections
        x0 = self.act(self.fc1(x))
        x1 = self.act(self.fc2(x0))
        x2 = self.act(self.fc3(x1) + x0)  # Residual connection

        x3 = self.act(self.fc4(x2))
        x4 = self.act(self.fc5(x3))
        residuals = self.fc6(x4)  # No addition of input here

        return residuals.view(B, 3, 4)


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

import torch
import torch.nn as nn
import pypose as pp

# import kornia.geometry.conversions as kgc  # maybe I can remove kornia
from modules.mlp import PoseRefinementMLP
from modules.base_module import BaseModule


class PoseModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict[str, int],
        hw: tuple[int, int],
        R: torch.Tensor,
        t: torch.Tensor,
        t_lr: float = 1e-3,
        q_lr: float = 1e-4,
        grad_q: bool = False,
        grad_t: bool = False,
        grad_t_offset: bool = False,
        use_mlp: bool = True,
        mlp_lr: float = 3e-3,
        device: str = "cuda",
        max_num_iterations: int = 2048,
        warmup_steps: int = 25,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Class storing extrinsics (Pose) for multiple cameras.
        Assumes World-to-Camera convention (T_cw).

        Args:
            image_id_map: Mapping from image IDs to tensor indices
            hw: Height and width of the input images
            R: (N, 3, 3) tensor of rotation matrices
            t: (N, 3) or (N, 3, 1) tensor of translation vectors
            t_lr: Learning rate for translation optimizer
            q_lr: Learning rate for rotation optimizer
            grad_q: Whether to compute gradients for rotation
            grad_t: Whether to compute gradients for translation
            grad_t_offset: Whether to compute gradients for translation offset
            use_mlp: Whether to use MLP for pose refinement
            mlp_lr: Learning rate for MLP optimizer
            device: Device to run the module on
            dtype: Data type for the tensors
        Note:
            To make this module more readable, some variables and methods share between
            pose, camera and depth modules are in base_module.
        """
        super().__init__(
            image_id_map=image_id_map,
            device=device,
            dtype=dtype,
        )
        self.hw = hw  # (H, W)
        self.max_num_iterations = max_num_iterations

        if use_mlp:
            print(f"When using MLP pose refinement, q and t gradients are disabled.")

        grad_q = False if use_mlp else grad_q
        grad_t = False if use_mlp else grad_t
        self.grad_t_offset = grad_t_offset
        self.t_lr = t_lr
        self.q_lr = q_lr
        self.mlp_lr = mlp_lr
        num_cams = len(image_id_map)

        # --- Rotation Prep (Matrix -> Quat) ---
        # # Kornia returns (w, x, y, z)
        # q_init = kgc.rotation_matrix_to_quaternion(R)

        # # PyPose expects (x, y, z, w) (scalar last). We roll -1 to move w from index 0 to index 3
        # q_init = torch.roll(q_init, shifts=-1, dims=1)

        q_init = pp.mat2SO3(R).tensor()  # Returns (x,y,z,w) directly

        # Store as Raw Parameter (requires normalization on usage)
        self.q_param = nn.Parameter(
            q_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=grad_q,
        )

        # --- Translation Prep ---
        t_init = t.reshape(num_cams, 3).to(self.device, dtype=self.dtype)
        self.t_mean = t_init.mean(dim=0, keepdim=True)
        dist = torch.norm(t_init - self.t_mean, dim=1)
        self.t_scale = torch.clamp(torch.mean(dist), min=1e-10)
        t_init = (t_init - self.t_mean) / self.t_scale

        # Store as Raw Parameter
        self.t_param = nn.Parameter(
            t_init.clone().detach(),
            requires_grad=grad_t,
        )

        self.t_offset = nn.Parameter(
            torch.zeros_like(self.t_param).to(self.device, dtype=self.dtype),
            requires_grad=self.grad_t_offset,
        )

        if use_mlp:
            self.use_mlp = True
            self.mlp = PoseRefinementMLP(
                input_dim=12,  # 3x4 pose matrix flattened
                output_dim=12,  # 3x4 pose matrix flattened
                hidden_dim=256,  # or 256 for complex scenarios
            ).to(self.device, dtype=self.dtype)
            # Initialize optimizer with MLP parameters
            self.init_optimizer(mlp_lr=mlp_lr)
        else:
            self.use_mlp = False
            # Initialize optimizer with q and t parameters
            self.init_optimizer(t_lr=self.t_lr, q_lr=self.q_lr)

        # Initialize learning rate scheduler
        self.init_scheduler(warmup_steps, max_num_iterations)

        # Precompute all extrinsic matrices
        self.update_all_matrices()

    def init_optimizer(
        self, t_lr: float = 1e-3, q_lr: float = 1e-4, mlp_lr: float = 3e-3
    ):
        """Re-initialize optimizer with new learning rate."""
        if self.use_mlp:
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.mlp.parameters(),
                        "lr": mlp_lr,
                        "name": "mlp",
                        "weight_decay": 0.01,  # default
                        "eps": 1e-10,
                    },
                    {
                        "params": [self.t_offset],
                        "lr": mlp_lr,
                        "name": "t_offset",
                        "weight_decay": 0.1,  # (more aggressive) L2 penalty
                        "eps": 1e-10,
                    },
                ]
            )

        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [self.t_param], "lr": t_lr, "name": "t", "eps": 1e-10},
                    {"params": [self.q_param], "lr": q_lr, "name": "q", "eps": 1e-10},
                    {
                        "params": [self.t_offset],
                        "lr": t_lr,
                        "name": "t_offset",
                        "weight_decay": 0.1,
                        "eps": 1e-10,
                    },
                ],
            )

    def get_rotation_matrix(self, indices) -> torch.Tensor:
        """
        Returns (B, 3, 3) Rotation Matrix for the requested images.
        """
        # 2. Get quaternions for batch
        q_batch = self.q_param[indices]

        # 1. Normalize all quaternions (important for stability)
        q_norm = q_batch / (torch.norm(q_batch, dim=1, keepdim=True) + 1e-10)

        # 3. Convert to Matrix via PyPose
        return pp.SO3(q_norm).matrix()

    def get_translation(self, indices) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        # Apply scaling to the learnable parameter
        return self.t_param[indices]

    def get_translation_offset(self, indices) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        # Apply scaling to the learnable parameter
        return self.t_offset[indices]

    def get_projection_matrix(self, indices) -> torch.Tensor:
        """
        Constructs 4x4 SE3 Matrix [R|t].
        Standard convention: World-to-Camera.
        """
        batch_size = len(indices)
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        # Retrieve components
        R = self.get_rotation_matrix(indices)  # (B, 3, 3)
        t = self.get_translation(indices)  # (B, 3)
        R, t = self.apply_mlp(R, t, indices)
        t = t * self.t_scale + self.t_mean  # Rescale to physical units

        # Build 4x4 Matrix
        P = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t

        return P

    def apply_mlp(self, R, t, indices) -> torch.Tensor:
        """Apply MLP refinement to given poses."""
        if not self.use_mlp:
            return R, t

        # Build 3x4 pose matrices and apply MLP
        P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)

        # adding mlp residual and orthonormalization
        P_3x4_refined = self.mlp(P_3x4)

        R = P_3x4_refined[:, :3, :3]
        t = P_3x4_refined[:, :3, 3] + self.get_translation_offset(indices)

        return R, t

    def __repr__(self):
        limit = 3
        s = f"PoseModel ({len(self.image_to_tensor_idx)} poses):\n"
        for i in range(min(limit, len(self.t_param))):
            name = self.tensor_idx_to_image[i]
            q_val = self.q_param[i].detach().cpu().numpy()

            # Fetch physical translation (scaled) for representation
            t_val = self.get_translation([i]).detach().cpu().numpy().flatten()

            # s += f"  Image '{name}': R={q_val:.3e}, t={t_val:.3e}\n"
            # print arrays with 3 decimal places signed, align them considering the sign too
            s += f"  Image '{name}': q=[{q_val[0]:+.3e}, {q_val[1]:+.3e}, {q_val[2]:+.3e}, {q_val[3]:+.3e}], t=[{t_val[0]:+.3e}, {t_val[1]:+.3e}, {t_val[2]:+.3e}]\n"

        if len(self.t_param) > limit:
            s += f"  ... {len(self.t_param) - limit} more."
        return s

    def parameters(self, t=False, q=False, mlp=False, recurse: bool = True):
        """Return list of trainable parameters - only leaf tensors"""

        params = []

        if mlp and self.use_mlp:  # no q and t when mlp is used
            params.extend([p for p in self.mlp.parameters() if p.requires_grad])
        else:
            if q and self.q_param.requires_grad:
                params.append(self.q_param)
            if t and self.t_param.requires_grad:
                params.append(self.t_param)
        return params

    def get_image_qt(self, indices):
        """Get quaternion and translation for given image names."""
        indices = (
            self.map_names_to_indices(indices)
            if isinstance(indices[0], str)
            else indices
        )
        R = self.get_rotation_matrix(indices)
        t = self.get_translation(indices)

        # Ensure MLP is applied before returning values
        if self.use_mlp:
            R, t = self.apply_mlp(R, t, indices)

        # Convert R to quaternion
        q = kgc.rotation_matrix_to_quaternion(R)
        q = torch.roll(q, shifts=-1, dims=1)  # (x, y, z, w) to (w, x, y, z)
        t = self.t_scale * t + self.t_mean  # Rescale to physical units

        return q.squeeze(), t.squeeze()

    def update_all_matrices(self):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        all_names = list(self.image_to_tensor_idx.keys())
        self.poses = self.get_projection_matrix(all_names)

    def get_all_matrices(self):
        """Get all extrinsic matrices for all images."""
        all_names = list(self.image_to_tensor_idx.keys())
        return self.get_projection_matrix(all_names)

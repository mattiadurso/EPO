import torch
import torch.nn as nn
import pypose as pp
import kornia.geometry.conversions as kgc  # maybe I can remove kornia
from modules.mlp import PoseRefinementMLP
from modules.base_module import BaseModule


class PoseModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict[str, int],
        R: torch.Tensor,
        t: torch.Tensor,
        t_lr: float = 1e-3,
        q_lr: float = 1e-4,
        grad_q: bool = False,
        grad_t: bool = False,
        use_mlp: bool = True,
        mlp_lr: float = 1e-3,
        device: str = "cuda",
        total_steps: int = 1000,
        warmup_steps: int = 25,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Class storing extrinsics (Pose) for multiple cameras.
        Assumes World-to-Camera convention (T_cw).

        Args:
            image_id_map: dict mapping image filenames (str) to tensor indices (int)
            R: (N, 3, 3) tensor of rotation matrices
            t: (N, 3) or (N, 3, 1) tensor of translation vectors
            grad_q: Optimize rotation?
            grad_t: Optimize translation?
            device: torch device
        """
        super().__init__(
            image_id_map=image_id_map,
            device=device,
            dtype=dtype,
        )
        if use_mlp:
            print(f"When using MLP pose refinement, q and t gradients are disabled.")
        grad_q = False if use_mlp else grad_q
        grad_t = False if use_mlp else grad_t
        self.mlp_lr = mlp_lr

        # --- ID Mappings ---
        # string name -> tensor index (0..N)
        self.image_to_tensor_idx = image_id_map
        # tensor index -> string name (inverse)
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        num_cams = len(image_id_map)
        assert (
            R.shape[0] == t.shape[0] == num_cams
        ), f"R shape {R.shape} mismatch with num images {num_cams}"

        # --- Rotation Prep (Matrix -> Quat) ---
        # Kornia returns (w, x, y, z)
        q_init = kgc.rotation_matrix_to_quaternion(R)

        # PyPose expects (x, y, z, w) (scalar last)
        # We roll -1 to move w from index 0 to index 3
        q_init = torch.roll(q_init, shifts=-1, dims=1)

        # Store as Raw Parameter (requires normalization on usage)
        self.q_param = nn.Parameter(
            q_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=grad_q,
        )

        # --- Translation Prep ---
        t_init = t.reshape(num_cams, 3).to(self.device, dtype=self.dtype)

        # I might/should find an effective way to scale translations
        # self.t_scale = ...

        # Store as Raw Parameter
        self.t_param = nn.Parameter(
            t_init.clone().detach(),
            requires_grad=grad_t,
        )

        if use_mlp:
            self.use_mlp = True
            self.mlp = PoseRefinementMLP().to(self.device, dtype=self.dtype)
            self.init_optimizer(mlp_lr=mlp_lr)

            # --- Configuration ---
            # 1. The Warmup Phase
            # Starts at lr * start_factor and linearly increases to lr over 'total_iters'
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,  # Start at 1% of your defined LR
                total_iters=warmup_steps,
            )

            # 2. The Decay Phase
            # Smoothly decreases from lr to min_lr
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,  # Remaining steps
                eta_min=5e-5,
            )

            # 3. Combine them
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
            )
        else:
            self.use_mlp = False
            # init optimizer lr
            self.t_lr = float(t_lr)
            self.min_t_lr = self.t_lr / 20
            self.q_lr = float(q_lr)
            self.min_q_lr = self.q_lr / 20
            self.init_optimizer(t_lr=self.t_lr, q_lr=self.q_lr)
            self.init_scheduler(
                lr_reduce_factor=0.75,
                patience=3,
                min_q_lr=self.min_q_lr,
                min_t_lr=self.min_t_lr,
            )  # this params seem good

        self.update_all_matrices()  # Precompute all extrinsic matrices

    def init_optimizer(
        self, t_lr: float = 1e-3, q_lr: float = 1e-4, mlp_lr: float = 1e-3
    ):
        """Re-initialize optimizer with new learning rate."""
        if self.use_mlp:
            self.optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=mlp_lr)
        else:
            # Single optimizer with separate parameter groups for different LRs
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [self.t_param], "lr": t_lr, "name": "t"},
                    {"params": [self.q_param], "lr": q_lr, "name": "q"},
                ],
                weight_decay=0,
            )

    def init_scheduler(
        self, lr_reduce_factor: float, patience: int, min_q_lr: float, min_t_lr: float
    ):
        """Initialize LR scheduler for the optimizer."""

        if not hasattr(self, "optimizer") and not self.use_mlp:
            raise ValueError("Optimizer must be initialized before scheduler.")

        # Single scheduler - will reduce all param group LRs by the same factor
        # Note: ReduceLROnPlateau applies same factor to all groups
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=lr_reduce_factor,
            patience=patience,
            min_lr=min(min_t_lr, min_q_lr),  # Use the smaller min_lr
        )

    def get_rotation_matrix(self, indices) -> torch.Tensor:
        """
        Returns (B, 3, 3) Rotation Matrix for the requested images.
        """
        # 2. Get quaternions for batch
        q_batch = self.q_param[indices]

        # 1. Normalize all quaternions (important for stability)
        q_norm = q_batch / torch.norm(q_batch, dim=1, keepdim=True)

        # 3. Convert to Matrix via PyPose
        return pp.SO3(q_norm).matrix()

    def get_translation(self, indices) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        # Apply scaling to the learnable parameter
        return self.t_param[indices]

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
        R, t = self.apply_mlp(R, t)

        # Build 4x4 Matrix
        P = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t

        return P

    def apply_mlp(self, R, t) -> torch.Tensor:
        """Apply MLP refinement to given poses."""
        if not self.use_mlp:
            return R, t

        # Build 3x4 pose matrices and apply MLP
        P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)

        # Use mixed precision for MLP inference
        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            P_3x4_refined = self.mlp(
                P_3x4
            )  # adding mlp residual and orthonormalization

        R = P_3x4_refined[:, :3, :3]
        t = P_3x4_refined[:, :3, 3]

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
            R, t = self.apply_mlp(R, t)

        # Convert R to quaternion
        q = kgc.rotation_matrix_to_quaternion(R)
        q = torch.roll(q, shifts=-1, dims=1)  # (x, y, z, w) to (w, x, y, z)

        return q.squeeze(), t.squeeze()

    def update_all_matrices(self):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        all_names = list(self.image_to_tensor_idx.keys())
        self.poses = self.get_projection_matrix(all_names)

    def get_all_matrices(self):
        """Get all extrinsic matrices for all images."""
        all_names = list(self.image_to_tensor_idx.keys())
        return self.get_projection_matrix(all_names)

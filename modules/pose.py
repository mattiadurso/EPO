import torch
import torch.nn as nn
import pypose as pp
import kornia.geometry.conversions as kgc  # maybe I can remove kornia


class PoseModule(nn.Module):
    def __init__(
        self,
        image_id_map: dict[str, int],
        R: torch.Tensor,
        t: torch.Tensor,
        t_lr: float = 1e-3,
        q_lr: float = 1e-4,
        device: str = "cuda",
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
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # --- ID Mappings ---
        # string name -> tensor index (0..N)
        self.image_to_tensor_idx = image_id_map
        # tensor index -> string name (inverse)
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        num_cams = len(image_id_map)
        assert (
            R.shape[0] == num_cams
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
            requires_grad=True,
        )

        # --- Translation Prep ---
        t_init = t.reshape(num_cams, 3)
        self.t_param = nn.Parameter(
            t_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=True,
        )

        # init optimizer lr
        self.t_lr = float(t_lr)
        self.q_lr = float(q_lr)
        self.init_optimizer(t_lr=self.t_lr, q_lr=self.q_lr)
        self.init_scheduler(
            lr_reduce_factor=0.75,
            patience=3,
            min_q_lr=self.q_lr / 20,
            min_t_lr=self.t_lr / 20,
        )  # this params seem good

        self.update_all_matrices()  # Precompute all extrinsic matrices

    def init_optimizer(self, t_lr: float, q_lr: float):
        """Re-initialize optimizer with new learning rate."""
        self.optimizer_t = torch.optim.AdamW(
            [
                {"params": self.t_param, "lr": t_lr},
            ]
        )
        self.optimizer_q = torch.optim.AdamW(
            [
                {"params": self.q_param, "lr": q_lr},
            ]
        )

    def init_scheduler(
        self, lr_reduce_factor: float, patience: int, min_q_lr: float, min_t_lr: float
    ):
        """Initialize LR scheduler for the optimizers."""

        if not hasattr(self, "optimizer_t") or not hasattr(self, "optimizer_q"):
            raise ValueError("Optimizers must be initialized before scheduler.")

        self.scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_t,
            factor=lr_reduce_factor,
            patience=patience,
            min_lr=min_t_lr,
        )
        self.scheduler_q = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_q,
            factor=lr_reduce_factor,
            patience=patience,
            min_lr=min_q_lr,
        )

    def map_names_to_indices(self, indices) -> torch.LongTensor:
        """Robustly maps string names to tensor indices."""
        # Handle single string input
        if isinstance(indices, str):
            indices = [indices]

        elif isinstance(indices, torch.Tensor):
            return torch.tensor(indices, dtype=torch.long, device=self.device)

        try:
            indices = [self.image_to_tensor_idx[name] for name in indices]
        except KeyError as e:
            raise ValueError(
                f"Image name {e} not found in PoseModel initialization dict."
            )
        return torch.tensor(indices, dtype=torch.long, device=self.device)

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
        t = self.t_param[indices]  # (B, 3)

        # # mlp Refinement
        # if self.use_mlp:
        #     R, t = self.apply_mlp(R, t)

        # Build 4x4 Matrix
        P = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t

        return P

    def forward(self, image_names):
        """Convenience method to get RT matrices"""
        return self.get_projection_matrix(image_names)

    # def apply_and_init_mlp(self):
    #     """Apply MLP refinement to all poses, then initialize a new MLP."""
    #     # Get all current poses
    #     all_poses_ids = list(range(len(self.t_param)))
    #     R = self.get_rotation_matrix(all_poses_ids)
    #     t = self.get_translation(all_poses_ids)

    #     # Apply MLP
    #     R_refined, t_refined = self.apply_mlp(R, t)

    #     # Update q_param and t_param with refined poses
    #     # Convert R_refined to quaternions
    #     q_refined = kgc.rotation_matrix_to_quaternion(R_refined)
    #     q_refined = torch.roll(q_refined, shifts=-1, dims=1)

    #     self.q_param[all_poses_ids] = q_refined.detach()
    #     self.t_param[all_poses_ids] = t_refined.detach()

    #     # Re-initialize MLP
    #     self.mlp = PoseRefinementMLP().to(self.device, dtype=self.dtype)

    # def apply_mlp(self, R, t) -> torch.Tensor:
    #     """Apply MLP refinement to given poses."""
    #     if not self.use_mlp:
    #         return R, t

    #     P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)
    #     P_3x4_refined = self.mlp(P_3x4)  # adding mlp residual and orthonormalization
    #     R = P_3x4_refined[:, :3, :3]
    #     t = P_3x4_refined[:, :3, 3]

    #     return R, t

    def __repr__(self):
        limit = 3
        s = f"PoseModel ({len(self.image_to_tensor_idx)} poses):\n"
        for i in range(min(limit, len(self.t_param))):
            name = self.tensor_idx_to_image[i]
            q_val = self.q_param[i].detach().cpu().numpy()
            t_val = self.t_param[i].detach().cpu().numpy()
            # s += f"  Image '{name}': R={q_val:.3e}, t={t_val:.3e}\n"
            # print arrays with 3 decimal places signed, align them considering the sign too
            s += f"  Image '{name}': q=[{q_val[0]:+.3e}, {q_val[1]:+.3e}, {q_val[2]:+.3e}, {q_val[3]:+.3e}], t=[{t_val[0]:+.3e}, {t_val[1]:+.3e}, {t_val[2]:+.3e}]\n"
        if len(self.t_param) > limit:
            s += f"  ... {len(self.t_param) - limit} more."
        return s

    def parameters(self, t=False, q=False, recurse: bool = True):
        """Return list of trainable parameters - only leaf tensors"""
        params = []
        if q and self.q_param.requires_grad:
            params.append(self.q_param)
        if t and self.t_param.requires_grad:
            params.append(self.t_param)
        # if mlp and self.use_mlp:
        #     params.extend([p for p in self.mlp.parameters() if p.requires_grad])
        return params

    def get_image_qt(self, indices):
        """Get quaternion and translation for given image names."""
        indices = (
            self.map_names_to_indices(indices)
            if isinstance(indices[0], str)
            else indices
        )

        # # Ensure MLP is applied before returning values
        # if self.use_mlp:
        #     self.apply_and_init_mlp()

        q = self.q_param[indices].squeeze()
        t = self.t_param[indices].squeeze()

        return q, t

    def update_all_matrices(self):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        all_names = list(self.image_to_tensor_idx.keys())
        self.poses = self.get_projection_matrix(all_names)

"""Learnable camera-pose module (world-to-camera, ``T_cw``).

Stores the per-image pose as a 3x3 rotation matrix + translation and exposes
either direct optimization of those parameters or, when ``use_mlp=True``, a
fixed rotation + translation refined by a small :class:`PoseRefinementMLP`
residual plus a learnable per-image translation offset.

The raw 3x3 rotation parameter is unconstrained during optimization; we
re-orthonormalize on every fetch via :func:`gram_schmidt_rotation`, which is
the same primitive the MLP uses internally to project its output back onto
SO(3).
"""

import torch
import torch.nn as nn

from modules.mlp import PoseRefinementMLP, gram_schmidt_rotation
from modules.base_module import BaseModule


class PoseModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict[str, int],
        hw: tuple[int, int],
        R: torch.Tensor,
        t: torch.Tensor,
        t_lr: float = 1e-3,
        R_lr: float = 1e-4,
        grad_R: bool = False,
        grad_t: bool = False,
        grad_t_offset: bool = False,
        use_mlp: bool = True,
        mlp_lr: float = 3e-3,
        use_amp: bool = False,
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
            R_lr: Learning rate for rotation optimizer
            grad_R: Whether to compute gradients for rotation
            grad_t: Whether to compute gradients for translation
            grad_t_offset: Whether to compute gradients for translation offset
            use_mlp: Whether to use MLP for pose refinement
            mlp_lr: Learning rate for MLP optimizer
            use_amp: If True, run the pose-refinement MLP's linear layers in
                BF16 via ``torch.autocast``. The Gram-Schmidt orthonormalisation
                stays in FP32 (precision-sensitive). No ``GradScaler`` is
                needed for BF16. Default False (FP32 throughout).
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

        # When the MLP refines poses, the raw R/t parameters are frozen
        # (the MLP residual + t_offset carry the optimization).
        if use_mlp:
            grad_R = False
            grad_t = False
        self.grad_t_offset = grad_t_offset
        self.t_lr = t_lr
        self.R_lr = R_lr
        # Per-call autocast switch — only relevant when use_mlp=True.
        self.use_amp = use_amp
        self.mlp_lr = mlp_lr
        num_cams = len(image_id_map)

        # --- Rotation Prep ---
        # Stored as an unconstrained 3x3; Gram-Schmidt is applied on every
        # fetch (see `get_rotation_matrix`) to project back onto SO(3).
        R_init = R.reshape(num_cams, 3, 3).to(self.device, dtype=self.dtype)
        self.R_param = nn.Parameter(
            R_init.clone().detach(),
            requires_grad=grad_R,
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
                hidden_dim=128,  # or 256 for complex scenarios
            ).to(self.device, dtype=self.dtype)
            # Initialize optimizer with MLP parameters
            self.init_optimizer(mlp_lr=mlp_lr)
        else:
            self.use_mlp = False
            # Initialize optimizer with R and t parameters
            self.init_optimizer(t_lr=self.t_lr, R_lr=self.R_lr)

        # Initialize learning rate scheduler
        self.init_scheduler(warmup_steps, max_num_iterations)

        # Precompute all extrinsic matrices
        self.update_all_matrices()

    def init_optimizer(
        self, t_lr: float = 1e-3, R_lr: float = 1e-4, mlp_lr: float = 3e-3
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
                        "weight_decay": 0.1,  # more aggressive decay
                        "eps": 1e-10,
                    },
                ]
            )

        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [self.t_param], "lr": t_lr, "name": "t", "eps": 1e-10},
                    {"params": [self.R_param], "lr": R_lr, "name": "R", "eps": 1e-10},
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
        Returns (B, 3, 3) rotation matrices for the requested images,
        re-orthonormalized via Gram-Schmidt so the result is always on SO(3).
        """
        return gram_schmidt_rotation(self.R_param[indices])

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

        if self.poses is not None:
            return self.poses[indices]

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

        P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4) — FP32

        # BF16 autocast only for the linear stack, and only when use_amp=True.
        # ``MLP.forward`` opts the Gram-Schmidt orthonormalisation back out to
        # FP32 internally (cross products / normalisations there lose ~2 digits
        # in BF16 and would distort the rotation noticeably).
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp,
        ):
            P_3x4_refined = self.mlp(P_3x4)

        # Autocast only forces matmul/conv inputs into BF16; the output dtype is
        # whatever the last op produced. Cast back explicitly so the Triton kernel
        # downstream (which expects FP32) is never handed a BF16 tensor.
        P_3x4_refined = P_3x4_refined.float()

        R = P_3x4_refined[:, :3, :3]
        t = P_3x4_refined[:, :3, 3] + self.get_translation_offset(indices)
        return R, t

    def __repr__(self) -> str:
        """Return a short summary of the first few stored poses."""
        limit = 3
        s = f"PoseModel ({len(self.image_to_tensor_idx)} poses):\n"
        for i in range(min(limit, len(self.t_param))):
            name = self.tensor_idx_to_image[i]
            # Print the (orthonormalized) first row of R as a compact summary.
            R_row0 = self.get_rotation_matrix([i])[0, 0].detach().cpu().numpy()

            # Fetch physical translation (scaled) for representation
            t_val = self.get_translation([i]).detach().cpu().numpy().flatten()

            s += (
                f"  Image '{name}': "
                f"R[0]=[{R_row0[0]:+.3e}, {R_row0[1]:+.3e}, {R_row0[2]:+.3e}], "
                f"t=[{t_val[0]:+.3e}, {t_val[1]:+.3e}, {t_val[2]:+.3e}]\n"
            )

        if len(self.t_param) > limit:
            s += f"  ... {len(self.t_param) - limit} more."
        return s

    def parameters(self, t=False, R=False, mlp=False, recurse: bool = True):
        """Return list of trainable parameters - only leaf tensors"""

        params = []

        if mlp and self.use_mlp:  # no R and t when mlp is used
            params.extend([p for p in self.mlp.parameters() if p.requires_grad])
        else:
            if R and self.R_param.requires_grad:
                params.append(self.R_param)
            if t and self.t_param.requires_grad:
                params.append(self.t_param)
        return params

    def get_image_Rt(self, indices):
        """Get rotation matrix and physical translation for given image names.

        Returns:
            R: ``(B, 3, 3)`` orthonormal rotation matrices (or ``(3, 3)`` if a
               single image was requested).
            t: ``(B, 3)`` translations rescaled to the original physical units
               (or ``(3,)`` for a single image).
        """
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

        t = self.t_scale * t + self.t_mean  # Rescale to physical units

        return R.squeeze(), t.squeeze()

    def update_all_matrices(self):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        all_names = list(self.image_to_tensor_idx.keys())
        self.poses = None  # Invalidate cache before recomputing
        self.poses = self.get_projection_matrix(all_names)

    def get_all_matrices(self):
        """Get all extrinsic matrices for all images."""
        if self.poses is None:
            self.update_all_matrices()

        return self.poses

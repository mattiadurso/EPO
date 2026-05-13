"""Learnable camera-intrinsics module.

Stores intrinsics as a simple-pinhole tuple ``(f, cx, cy)`` per camera and
optimizes a single scalar focal-length scale per camera (``f_eff = f *
(1 + alpha)``). Principal points are kept fixed; ``PINHOLE`` cameras are
collapsed to ``SIMPLE_PINHOLE`` by averaging ``fx`` and ``fy``.
"""

import torch
import torch.nn as nn

from helpers.reprojection import invert_K
from modules.base_module import BaseModule


class CameraModule(BaseModule):
    """Per-camera learnable focal-length scale (simple-pinhole)."""

    def __init__(
        self,
        image_id_map: dict,
        k_models: list[str],
        k_params: torch.Tensor,
        lr: float = 1e-3,
        grad: bool = True,
        warmup_steps: int = 25,
        max_num_iterations: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Class storing camera intrinsics as simple pinhole (f, cx, cy) for all cameras.

        Args:
            image_id_map: dict mapping camera ids to tensor indices (0..N)
            k_models: list of camera model names ("PINHOLE" or "SIMPLE_PINHOLE")
            k_params: camera parameters per camera; PINHOLE expects (fx, fy, cx, cy),
                      SIMPLE_PINHOLE expects (f, cx, cy). PINHOLE inputs are averaged: f=(fx+fy)/2.
        """
        super().__init__(image_id_map, device=device, dtype=dtype)
        self.max_num_iterations = max_num_iterations
        self.lr = float(lr)
        self.keys = list(self.image_to_tensor_idx.keys())

        # Normalize all cameras to simple pinhole format: [f, cx, cy]
        k_params = k_params.clone().detach().to(self.device, dtype=self.dtype)
        normalized = []
        for model, p in zip(k_models, k_params):
            if model == "PINHOLE":
                f = (p[0] + p[1]) / 2.0
                normalized.append(torch.stack([f, p[2], p[3]]))
            else:  # SIMPLE_PINHOLE
                normalized.append(p[:3])
        self.k_params = torch.stack(normalized)  # (N, 3): [f, cx, cy]

        # Single learnable scale per camera: f_eff = f * (1 + alpha)
        self.params = nn.Parameter(
            torch.zeros(
                (self.k_params.shape[0], 1), device=self.device, dtype=self.dtype
            ),
            requires_grad=grad,
        )

        if grad:
            self.init_optimizer(lr=self.lr)
            self.init_scheduler(warmup_steps, max_num_iterations)

        self.update_all_matrices()  # Pre-compute all intrinsic matrices

    def get_all_intrinsic_matrix(self) -> tuple[list[str], torch.Tensor]:
        """Return (camera names, stacked 3x3 intrinsics) for all cameras."""
        return self.keys, self.get_intrinsic_matrix(self.keys)

    def get_intrinsic_matrix(self, indices) -> torch.Tensor:
        """Return ``(B, 3, 3)`` intrinsic matrices for the given cameras."""
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        if self.cameras is not None:
            return self.cameras[indices]

        return self._build_K(indices)

    def get_inverse_intrinsic_matrix(self, indices) -> torch.Tensor:
        """Return ``(B, 3, 3)`` inverse intrinsic matrices for the given cameras."""
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        if self.cameras_inv is not None:
            return self.cameras_inv[indices]

        return invert_K(self.get_intrinsic_matrix(indices))

    def _build_K(self, tensor_indices: torch.Tensor) -> torch.Tensor:
        """Compose ``(B, 3, 3)`` intrinsic matrices from base params + ``alpha``."""
        params = self.k_params[tensor_indices]  # (B, 3): [f, cx, cy]
        alpha = self.params[tensor_indices]  # (B, 1)

        f = params[:, 0] * (1 + alpha[:, 0])
        cx = params[:, 1]
        cy = params[:, 2]

        B = tensor_indices.shape[0]
        K = torch.zeros((B, 3, 3), dtype=params.dtype, device=self.device)
        K[:, 0, 0] = f
        K[:, 1, 1] = f
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def get_camera_parameters(self, indices) -> tuple[str, torch.Tensor]:
        """Return ``("SIMPLE_PINHOLE", params)`` for the requested cameras.

        ``params`` is a ``(B, 3)`` tensor of effective ``[f, cx, cy]`` values
        (i.e. with the learnable focal-length scale already applied).
        """
        if isinstance(indices, str):
            indices = [indices]
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        params = self.k_params[indices].clone()  # (B, 3): [f, cx, cy]
        alpha = self.params[indices]  # (B, 1)
        params[:, 0] = params[:, 0] * (1 + alpha[:, 0])

        return "SIMPLE_PINHOLE", params.squeeze()

    def update_all_matrices(self) -> None:
        """Refresh the cached intrinsic matrices after parameters change."""
        self.cameras = None
        self.cameras = self.get_intrinsic_matrix(self.keys)
        self.cameras_inv = None

    def __repr__(self) -> str:
        s = "CameraModel:\n"
        limit = 5
        for i, params in enumerate(self.k_params[:limit]):
            s += f"  Camera {self.tensor_idx_to_image[i]}: Model=SIMPLE_PINHOLE, Params={params.detach().cpu().tolist()}\n"
        if len(self.k_params) > limit:
            s += f"  ... and {len(self.k_params) - limit} more.\n"
        return s

"""Learnable per-image depth correction.

Stores a parametric depth correction ``z' = clamp(z * (1 + a) + b,
min=min_depth)`` per image, with either one ``(a, b)`` pair shared across
the depth map (``per_pixel=False``) or one pair per pixel
(``per_pixel=True``). The original depth is kept frozen; only ``(a, b)``
are learnable.

If ``direct_backprop=True``, the depth ``z`` itself becomes the learnable
parameter and ``(a, b)`` are kept as a non-learnable zero buffer (so
``(1 + a) = 1`` and ``b = 0``) for API compatibility.
"""

import torch
import torch.nn as nn

from modules.base_module import BaseModule


class DepthModule(BaseModule):
    """Per-image (optionally per-pixel) parametric depth correction."""

    def __init__(
        self,
        image_id_map: dict,
        depth: torch.Tensor,
        lr: float = 5e-3,
        grad: bool = True,
        warmup_steps: int = 25,
        max_num_iterations: int = 1000,
        per_pixel: bool = True,
        min_depth: float = 1e-3,
        direct_backprop: bool = False,
        device="cuda",
        dtype=torch.float32,
    ):
        """Args:
            image_id_map: Mapping from image IDs to tensor indices
            depth: Depth tensor (N, 1) or (N, 2) for parametric representation
            lr: Learning rate for optimizer
            grad: Whether to compute gradients
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_num_iterations: Maximum number of optimization iterations
            per_pixel: If True, optimize a (scale, shift) pair per pixel of each
                depth map. If False, optimize a single (scale, shift) pair per
                depth map (2 params per image). Ignored when
                ``direct_backprop=True``.
            min_depth: Minimum allowed depth value. The output depth is clamped
                to be at least this value to keep points in front of the camera
                and avoid negative/zero depths.
            direct_backprop: If True, optimize the depth ``z`` directly instead
                of the ``(a, b)`` correction. ``(a, b)`` are kept as a
                non-learnable zero buffer for compatibility.
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
        self.max_num_iterations = max_num_iterations
        self.lr = float(lr)
        self.depth = depth
        self.per_pixel = per_pixel
        self.min_depth = float(min_depth)
        self.direct_backprop = direct_backprop

        if direct_backprop:
            # z itself is learnable; (a, b) kept as identity buffer.
            self.params = nn.Parameter(
                self.depth.clone().detach().to(device=self.device, dtype=self.dtype),
                requires_grad=grad,
            )
            ab_shape = (
                (self.depth.shape[0], self.depth.shape[1], 2)
                if per_pixel
                else (self.depth.shape[0], 2)
            )
            self.register_buffer(
                "ab",
                torch.zeros(*ab_shape, device=self.device, dtype=self.dtype),
            )
        else:
            # Depth params: (N, P, 2) per-pixel or (N, 2) per-image
            if per_pixel:
                params_shape = (self.depth.shape[0], self.depth.shape[1], 2)
            else:
                params_shape = (self.depth.shape[0], 2)

            self.params = nn.Parameter(
                torch.zeros(*params_shape)
                .clone()
                .detach()
                .to(device=self.device, dtype=self.dtype),
                requires_grad=grad,
            )

        if grad:
            self.init_optimizer(lr=self.lr)
            self.init_scheduler(warmup_steps, max_num_iterations)

    def get_parameters(self, ids):
        """Return depth parameters - ensures gradient flow"""
        indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
        if self.direct_backprop:
            # z is learnable; (a, b) are frozen at identity, so no-op.
            return torch.clamp(self.params[indices], min=self.min_depth)
        # Need to return depth, not inverse depth
        z = self.depth[indices]
        if self.per_pixel:
            a = self.params[indices][:, :, 0]
            b = self.params[indices][:, :, 1]
        else:
            # broadcast a single (scale, shift) pair across all pixels
            a = self.params[indices][:, 0:1]
            b = self.params[indices][:, 1:2]
        # clamp to keep depths in front of the camera (z > 0)
        return torch.clamp(z * (1 + a) + b, min=self.min_depth)

    def get_all_parameters(self) -> torch.Tensor:
        """Return corrected depth maps for every image, in storage order."""
        return self.get_parameters(list(self.image_to_tensor_idx.keys()))

    def __repr__(self) -> str:
        """Return a one-line summary with the total learnable parameter count."""
        return f"DepthModule(parameters={self.params.numel():,})"

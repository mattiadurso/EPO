"""Shared base class for the per-image learnable modules (pose, camera, depth).

Centralises the bookkeeping that all three submodules need: an
image-name → tensor-row index map, optimizer/scheduler construction with
warmup + cosine decay, and a uniform :py:meth:`get_parameters` API that
accepts either string image names or integer indices.
"""

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    """Base for modules whose parameters are indexed by image name."""

    def __init__(
        self,
        image_id_map: dict[str, int],
        parameters: torch.Tensor | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            image_id_map: Mapping from image name (str) to row index in the
                parameter tensor.
            parameters: Optional pre-allocated parameter tensor. Subclasses
                normally allocate their own ``nn.Parameter`` instead.
            device: Torch device for the parameters.
            dtype: Floating-point dtype for the parameters.
        """
        super(BaseModule, self).__init__()

        self.device = torch.device(device)
        self.dtype = dtype

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # Optional pre-supplied parameter tensor; subclasses usually allocate
        # their own learnable nn.Parameter and overwrite this attribute.
        self.params = parameters.to(self.device) if parameters is not None else None

    def forward(self, x):
        """Subclasses must override; the base class does not define a forward pass."""
        raise NotImplementedError("Forward method not implemented yet.")

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

    def get_parameters(self, ids) -> torch.Tensor:
        """
        Returns parameters for the requested IDs.

        Args:
            ids: Single ID (str/int), list, tuple, or tensor of IDs
        """
        indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
        return self.params[indices]

    def __repr__(self) -> str:
        """Return a short, truncated summary of the module contents."""
        num_items = len(self.tensor_idx_to_image)
        s = f"{self.__class__.__name__} ({num_items} items):\n"
        limit = 3
        for i in range(min(limit, num_items)):
            id_val = self.tensor_idx_to_image[i]
            s += f"  [{i}] ID: {id_val}\n"
        if num_items > limit:
            s += f"  ... {num_items - limit} more."
        return s

    def parameters(self, recurse=True):  # Changed: match nn.Module signature
        """Return list of trainable parameters - only self.params is a leaf tensor"""
        return [self.params] if self.params.requires_grad else []

    def init_optimizer(self, lr: float, w_decay: float = 1e-2, eps: float = 1e-10):
        """Initialize optimizer."""
        args = {"lr": lr, "weight_decay": w_decay, "eps": eps, "fused": True}
        self.optimizer = torch.optim.AdamW(
            [self.params],
            **args,
        )

    def init_scheduler(self, warmup_steps: int, max_num_iterations: int):
        """Initialize learning rate scheduler."""

        # Linearly increase learning rate
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1 / 100,  # Start at 1% of your defined LR
            total_iters=warmup_steps,
        )

        # Smoothly decreases from lr to min_lr
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_num_iterations - warmup_steps,  # Remaining steps
            eta_min=1e-6,  # this was 1e-6
        )

        # 3. Combine them
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

    def optimizer_and_scheduler_step(self):
        """Perform optimizer step and update scheduler based on loss."""
        self.optimizer.step()
        self.scheduler.step()

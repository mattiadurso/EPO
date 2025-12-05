import torch
import torch.nn as nn
from modules.base_module import BaseModule


class DepthModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict,
        parameters: torch.Tensor,
        lr: float = 5e-3,
        grad: bool = True,
        device="cuda",
        dtype=torch.float32,
    ):
        """Depth module to hold depth parameters"""
        super().__init__(
            image_id_map=image_id_map,
            device=device,
            dtype=dtype,
        )

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # storing as inverse depth for better numerical stability
        depth = parameters.pow(-1)

        self.params = nn.Parameter(
            depth.clone().detach().to(device=self.device, dtype=self.dtype),
            requires_grad=grad,
        )

        self.lr = float(lr)
        self.lr_min = self.lr / 20
        if grad:
            self.init_optimizer(lr=self.lr)
            self.init_scheduler(lr_reduce_factor=0.75, patience=3, min_lr=self.lr_min)

    def init_scheduler(self, lr_reduce_factor: float, patience: int, min_lr: float):
        """Initialize LR scheduler for the optimizer."""
        if not hasattr(self, "optimizer"):
            raise ValueError("Optimizer must be initialized before scheduler.")

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=lr_reduce_factor,
            patience=patience,
            min_lr=min_lr,
        )

    def get_parameters(self, ids):
        """Return depth parameters - ensures gradient flow"""
        indices = self.map_ids_to_indices(ids) if isinstance(ids[0], str) else ids
        # Need to return depth, not inverse depth
        return self.params[indices].pow(-1)

    def __repr__(self):
        out = f"Depth" + f"parameters={len(self.params.data.detach().tolist()):,})"
        return out

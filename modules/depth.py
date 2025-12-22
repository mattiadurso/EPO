import torch
import torch.nn as nn
from modules.base_module import BaseModule

# optimizing depth as
# - free variable is okish
# - z = z_param * a + b  (a,b per depth) better
# - z = z_param * a + b  (a,b per image) not as good as previous

# next
# z = z+mlp(f(x,y), z) with cooordinates-based MLP and f encoding for coordinates

# at certain point I should optimizid K as well


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

        # self.params = nn.Parameter(
        #     depth.clone().detach().to(device=self.device, dtype=self.dtype),
        #     requires_grad=grad,
        # )
        self.depth = depth

        alphas = torch.ones_like(depth)
        betas = torch.zeros_like(depth)
        params = torch.stack([alphas, betas], dim=-1)
        self.params = nn.Parameter(
            params.clone().detach().to(device=self.device, dtype=self.dtype),
            requires_grad=grad,
        )

        self.lr = float(lr)
        # self.lr_min = self.lr / 20

        # if grad:
        #     self.init_optimizer(lr=self.lr)
        #     self.init_scheduler(lr_reduce_factor=0.75, patience=3, min_lr=self.lr_min)

        self.optimizer = torch.optim.AdamW(
            [self.params],
            lr=self.lr,
            weight_decay=1e-2,
        )

        # --- Configuration ---
        warmup_steps = 25
        total_steps = 1000
        # 1. The Warmup Phase
        # Starts at lr * start_factor and linearly increases to lr over 'total_iters'
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,  # Start at 1% of your defined LR
            total_iters=25,
        )

        # 2. The Decay Phase
        # Smoothly decreases from lr to min_lr
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,  # Remaining steps
            eta_min=1e-6,
        )

        # 3. Combine them
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

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
        indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
        # Need to return depth, not inverse depth
        z = self.depth[indices].pow(-1)
        a = self.params[indices][:, :, 0]
        b = self.params[indices][:, :, 1]
        return z * a + b

    def get_all_parameters(self):
        return self.get_parameters(list(self.image_to_tensor_idx.keys()))

    def __repr__(self):
        out = f"Depth" + f"parameters={len(self.params.data.detach().tolist()):,})"
        return out

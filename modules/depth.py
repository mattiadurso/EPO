import torch
import torch.nn as nn


class DepthModule(nn.Module):
    def __init__(
        self,
        image_id_map: dict,
        depth: torch.Tensor,
        lr: float = 5e-3,
        device="cuda",
        grad: bool = True,
    ):
        """Depth module to hold depth parameters"""
        super().__init__()
        self.device = device

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # storing as inverse depth for better numerical stability
        depth = depth.pow(-1)

        self.params = nn.Parameter(
            depth.clone().detach().to(device), requires_grad=grad
        )

        self.lr = float(lr)
        self.lr_min = self.lr / 20
        if grad:
            self.init_optimizer(z_lr=self.lr)
            self.init_scheduler(
                lr_reduce_factor=0.75, patience=3, min_lr=self.lr_min
            )  # this params seem good

    def init_optimizer(self, z_lr: float):
        """Re-initialize optimizer with new learning rate."""
        self.optimizer = torch.optim.AdamW([self.params], lr=z_lr)

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

    def optimizer_and_scheduler_step(self, loss):
        """Perform optimizer step and update scheduler based on loss."""
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step(loss.detach())

    def forward(self, ids):
        """Return depth parameters - ensures gradient flow"""
        return self.get_parameters(ids)

    def map_ids_to_indices(self, ids) -> torch.LongTensor:
        """
        Robustly maps identifiers to internal tensor indices.

        Args:
            ids: Single ID (str/int), list, tuple, or tensor of IDs

        Returns:
            torch.LongTensor of indices on the correct device
        """
        # Ensure input is iterable on CPU for dictionary lookup
        if isinstance(ids, torch.Tensor):
            ids_cpu = ids.detach().cpu().tolist()  # why? cant this stay on gpu?
        elif isinstance(ids, (list, tuple)):
            ids_cpu = ids
        else:
            ids_cpu = [ids]  # Handle single ID

        # Perform lookup
        try:
            indices = [self.image_to_tensor_idx[id_val] for id_val in ids_cpu]
        except KeyError as e:
            raise ValueError(
                f"ID {e} not found. "
                f"Available IDs: {list(self.image_to_tensor_idx.keys())}"
            )

        # Return as LongTensor on the correct device for indexing
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def get_parameters(self, ids):
        """Return depth parameters - ensures gradient flow"""
        indices = self.map_ids_to_indices(ids) if isinstance(ids[0], str) else ids
        # Return depth in linear space by exponentiating the stored log-depths
        return self.params[indices].pow(-1)

    def parameters(self, recurse: bool = True):
        return [self.params] if self.params.requires_grad else []

    def __repr__(self):
        out = (
            f"Depth"
            # + f"(height={self.hw[0]}, width={self.hw[1]}, "
            + f"parameters={len(self.params.data.detach().tolist()):,})"
        )
        return out

    def __str__(self):
        return self.__repr__()

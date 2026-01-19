import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(
        self,
        image_id_map: dict[str, int],
        parameters: [torch.Tensor | None] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super(BaseModule, self).__init__()

        self.image_id_map = image_id_map
        self.device = torch.device(device)
        self.dtype = dtype

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # Initialize layers or parameters here
        self.params = parameters.to(self.device) if parameters is not None else None

    def forward(self, x):
        # Define the forward pass
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
        s = f"{self.__class__.__name__} ({self.num_items} items):\n"
        limit = 3
        for i in range(min(limit, self.num_items)):
            id_val = self.tensor_idx_to_id[i]
            s += f"  [{i}] ID: {id_val}\n"
        if self.num_items > limit:
            s += f"  ... {self.num_items - limit} more."
        return s

    def parameters(self, recurse=True):  # Changed: match nn.Module signature
        """Return list of trainable parameters - only self.params is a leaf tensor"""
        return [self.params] if self.params.requires_grad else []

    def init_optimizer(self, lr: float, w_decay: float = 0, eps: float = 1e-10):
        """Initialize optimizer."""
        args = {"lr": lr, "weight_decay": w_decay, "eps": eps}
        self.optimizer = torch.optim.AdamW(
            [self.params],
            **args,
        )

    def optimizer_and_scheduler_step(self, loss):
        """Perform optimizer step and update scheduler based on loss."""
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.detach())
            else:  # other schedulers do not need loss input
                self.scheduler.step()

    def get_all_parameters(self) -> torch.Tensor:
        """Return all parameters as a tensor."""
        return self.params.detach().clone()

import torch
import torch.nn as nn


class DepthMap(nn.Module):  # Changed: inherit from nn.Module
    def __init__(
        self,
        height: int,
        width: int,
        depth: torch.Tensor,
        grad=True,
        device="cuda",
    ):
        """Depth module to hold depth parameters"""
        super().__init__()
        self.device = device
        self.grad = grad

        self.params = nn.Parameter(
            depth.clone().detach().to(device), requires_grad=self.grad
        )
        self.hw = (height, width)

    def forward(self):
        """Return depth parameters - ensures gradient flow"""
        return self.params

    def get_depth(self):
        """Return depth parameters - ensures gradient flow"""
        return self.params

    def parameters(self, recurse: bool = True):
        return [self.params]

    def __repr__(self):
        out = (
            f"Depth(height={self.hw[0]}, width={self.hw[1]}, "
            + f"parameters={len(self.params.data.detach().tolist()):,})"
        )
        return out

    def __str__(self):
        return self.__repr__()


class DepthModule(nn.Module):
    def __init__(
        self,
        image_id_map: dict,
        depth: torch.Tensor,
        device="cuda",
        grad=True,
    ):
        """Depth module to hold depth parameters"""
        super().__init__()
        self.device = device
        self.grad = grad

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # store loss in log space to avoid negative depths
        depth = torch.log(depth)

        self.params = nn.Parameter(
            depth.clone().detach().to(device), requires_grad=self.grad
        )

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
        indices = self.map_ids_to_indices(ids)
        # Return depth in linear space by exponentiating the stored log-depths
        return torch.exp(self.params[indices])

    def parameters(self, recurse: bool = True):
        return [self.params]

    def __repr__(self):
        out = (
            f"Depth"
            # + f"(height={self.hw[0]}, width={self.hw[1]}, "
            + f"parameters={len(self.params.data.detach().tolist()):,})"
        )
        return out

    def __str__(self):
        return self.__repr__()

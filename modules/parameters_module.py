import torch
import torch.nn as nn


class ParameterModule(nn.Module):
    """
    Generic module to manage parameters without grad with custom identifiers.
    """

    def __init__(
        self,
        image_id_map: dict,
        parameters: torch.Tensor,
        device: str = "cuda",
    ):
        """
        Args:
            image_id_map: dict mapping identifiers (str or int) to tensor indices (0..N)
            parameters: torch.Tensor containing the parameters
            device: torch device
        """
        super().__init__()
        self.device = torch.device(device)

        # ID Mappings
        self.image_to_tensor_idx = image_id_map
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        # store parameters
        self.params = parameters.to(self.device)

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

    def get_parameters(self, ids) -> torch.Tensor:
        """
        Returns parameters for the requested IDs.

        Args:
            ids: Single ID (str/int), list, tuple, or tensor of IDs
        """
        indices = self.map_ids_to_indices(ids)
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

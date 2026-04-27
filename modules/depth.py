import torch
import torch.nn as nn
from modules.base_module import BaseModule


# depth as parametric z=z*a+b
class DepthModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict,
        depth: torch.Tensor,
        lr: float = 5e-3,
        grad: bool = True,
        warmup_steps: int = 25, #25
        max_num_iterations: int = 1000,
        device="cuda",
        dtype=torch.float32,
    ):
        """
        Args:
            image_id_map: Mapping from image IDs to tensor indices
            depth: Depth tensor (N, 1) or (N, 2) for parametric representation
            lr: Learning rate for optimizer
            grad: Whether to compute gradients
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_num_iterations: Maximum number of optimization iterations
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

        # Depth params
        self.params = nn.Parameter(
            torch.zeros(self.depth.shape[0], self.depth.shape[1], 2)
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
        # Need to return depth, not inverse depth
        z = self.depth[indices]
        a = self.params[indices][:, :, 0]
        b = self.params[indices][:, :, 1]
        return z * (1 + a) + b

    def get_all_parameters(self):
        return self.get_parameters(list(self.image_to_tensor_idx.keys()))

    def __repr__(self):
        out = f"Depth" + f"parameters={len(self.params.data.detach().tolist()):,})"
        return out
    
    #####################################################################################################################

    def get_parameters_idx(self, indices, recurse=True):
        return self.params[indices]


    def add_element(self, 
                    new_image_name, 
                    new_id, 
                    new_sampled_depth,
                    lr: float = 5e-4
                    ):


        super().add_element(new_image_name, new_id)
        
        # Depth
        # 1- concat new depth to old tensor
        new_z = new_sampled_depth.unsqueeze(0).to(self.params.device)
        self.depth = torch.cat([self.depth, new_z], dim=0) 

        # Depth params
        # 2. Create new parameters (a=0, b=0) for the new element
        new_params_val = torch.zeros(
            (1, self.depth.shape[1], 2), 
            device=self.device, 
            dtype=self.dtype
        )

        # 3. Concatenate and re-wrap as nn.Parameter
        # We use .data or .detach() to avoid tracking the cat operation in the gradient history
        updated_params = torch.cat([self.params.data, new_params_val], dim=0)
        self.params = nn.Parameter(updated_params, requires_grad=self.params.requires_grad)


        # Update params

        self.lr = float(lr)

        self.init_optimizer(lr=self.lr)
        self.init_scheduler(warmup_steps=0, max_num_iterations = self.max_num_iterations)

        return 
    


# # depth class to use for test with Z optimized as free variable
# class DepthModule(BaseModule):
#     def __init__(
#         self,
#         image_id_map: dict,
#         depth: torch.Tensor,
#         lr: float = 5e-3,
#         grad: bool = True,
#         warmup_steps: int = 25,
#         max_num_iterations: int = 1000,
#         device="cuda",
#         dtype=torch.float32,
#     ):
#         """Depth module to hold depth parameters"""
#         super().__init__(
#             image_id_map=image_id_map,
#             device=device,
#             dtype=dtype,
#         )
#         self.lr = float(lr)

#         # ID Mappings
#         self.image_to_tensor_idx = image_id_map
#         self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

#         # storing as inverse depth for better numerical stability
#         depth = depth.pow(-1)

#         self.params = nn.Parameter(
#             depth.clone().detach().to(device=self.device, dtype=self.dtype),
#             requires_grad=grad,
#         )

#         if grad:
#             self.init_optimizer(lr=self.lr)
#             self.init_scheduler(warmup_steps, max_num_iterations)

#     def get_parameters(self, ids):
#         """Return depth parameters - ensures gradient flow"""
#         indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
#         # Need to return depth, not inverse depth
#         z = self.params[indices].pow(-1)

#         return z

#     def get_all_parameters(self):
#         return self.get_parameters(list(self.image_to_tensor_idx.keys()))

#     def __repr__(self):
#         out = f"Depth" + f"parameters={len(self.params.data.detach().tolist()):,})"
#         return out

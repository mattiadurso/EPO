import torch
import torch.nn as nn
from modules.base_module import BaseModule
from helpers.reprojection_compiled import invert_K


# paramtric f
class CameraModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict,
        k_models: list[str],
        k_params: torch.Tensor,
        lr: float = 1e-3,
        grad: bool = True,
        warmup_steps: int = 25, #25
        max_num_iterations: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Class storing camera intrinsics for multiple cameras and poses.
        Args:
            cam_id: dict mapping reconstruction camera ids (real world) to tensor indices (0..N)
            k_models: list of strings, camera model names per camera
            k_params: Nx4 tensor of camera parameters, fx, fy, cx, cy
            device: torch device
        Note:
            To make this module more readable, some variables and methods share between
            pose, camera and depth modules are in base_module.
        """
        super().__init__(image_id_map, device=device, dtype=dtype)
        self.max_num_iterations = max_num_iterations
        self.lr = float(lr)
        self.keys = list(self.image_to_tensor_idx.keys())

        # Define model ID mapping
        self.camera_model_name_to_id = {
            "PINHOLE": 0,
            "SIMPLE_PINHOLE": 1,
        }

        # --- Model Types ---
        self.k_models = k_models
        # Store model types as a Tensor for fast masking during forward pass
        model_ids_list = [self.camera_model_name_to_id[m] for m in k_models]
        self.register_buffer(
            "k_models_ids", torch.tensor(model_ids_list, device=self.device)
        )

        # --- Parameters ---
        self.k_params = k_params.clone().detach().to(self.device, dtype=self.dtype)
        # alphas for f = f * (1 + alpha)
        self.params = nn.Parameter(
            torch.zeros(
                (self.k_params.shape[0], 2), device=self.device, dtype=self.dtype
            ),
            requires_grad=grad,
        )

        if grad:
            self.init_optimizer(lr=self.lr)
            self.init_scheduler(warmup_steps, max_num_iterations)

        self.update_all_matrices()  # Pre-compute all intrinsic matrices

    def get_all_intrinsic_matrix(self):
        """Get all intrinsic matrices."""
        return self.keys, self.get_intrinsic_matrix(self.keys)

    def get_intrinsic_matrix(self, indices) -> torch.Tensor:
        """Construct K matrix from parameters on-the-fly. Handles mixed models."""

        # 1. Get the internal row indices for the requested cameras
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        if self.cameras is not None:
            return self.cameras[indices]

        batch_size = indices.shape[0]

        # 2. Retrieve the model type for these specific cameras
        # Use standard indexing, k_models_ids is a buffer
        models_in_batch = self.k_models_ids[indices]

        # 3. Check if we can take a fast path (all cameras are the same type)
        if (models_in_batch == 0).all():
            return self._build_pinhole(indices, batch_size)

        elif (models_in_batch == 1).all():
            return self._build_simple_pinhole(indices, batch_size)

        # 4. Mixed Case: Vectorized Masking
        else:
            K = torch.zeros(
                (batch_size, 3, 3), dtype=self.k_params.dtype, device=self.device
            )

            # Create masks
            mask_pinhole = models_in_batch == 0
            mask_simple = models_in_batch == 1

            # Process PINHOLE cameras
            if mask_pinhole.any():
                # Filter indices relevant to pinhole
                idx_pinhole = indices[mask_pinhole]
                # Calculate and assign to the masked slice of K
                K[mask_pinhole] = self._build_pinhole(idx_pinhole, idx_pinhole.shape[0])

            # Process SIMPLE_PINHOLE cameras
            if mask_simple.any():
                idx_simple = indices[mask_simple]
                K[mask_simple] = self._build_simple_pinhole(
                    idx_simple, idx_simple.shape[0]
                )

            return K

    def get_inverse_intrinsic_matrix(self, indices) -> torch.Tensor:
        """Get inverse of intrinsic matrix for given camera IDs."""

        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        if self.cameras_inv is not None:
            return self.cameras_inv[indices]

        K = self.get_intrinsic_matrix(indices)
        K_inv = invert_K(K)
        return K_inv

    def _build_pinhole(
        self, tensor_indices: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Internal helper: Pinhole (fx, fy, cx, cy)"""
        params = self.k_params[tensor_indices]  # Shape (B, 4)

        fx, fy = params[:, 0], params[:, 1]
        cx, cy = params[:, 2], params[:, 3]
        alpha = self.params[tensor_indices]

        K = torch.zeros((batch_size, 3, 3), dtype=params.dtype, device=self.device)
        K[:, 0, 0] = fx * (1 + alpha[:, 0])
        K[:, 1, 1] = fy * (1 + alpha[:, 1])
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def _build_simple_pinhole(
        self, tensor_indices: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Internal helper: Simple Pinhole (f, cx, cy)"""
        params = self.k_params[
            tensor_indices
        ]  # Shape (B, 4) - we assume unused cols are ignored
        alpha = self.params[tensor_indices]

        f = params[:, 0]
        cx = params[:, 1]
        cy = params[:, 2]

        K = torch.zeros((batch_size, 3, 3), dtype=params.dtype, device=self.device)
        K[:, 0, 0] = f * (1 + alpha[:, 0])
        K[:, 1, 1] = f * (1 + alpha[:, 1])
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def get_camera_parameters(self, indices) -> torch.Tensor:
        """Get camera parameters for given camera IDs."""
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        k_model = self.k_models[indices]
        k_params = self.k_params[indices]
        params = self.params[indices]

        if k_params.shape[1] == 3 and k_model[0] == "SIMPLE_PINHOLE":
            k_params[:, 0] = k_params[:, 0] * (1 + params[:, 0])  # fx

        elif k_params.shape[1] == 4 and k_model[0] == "PINHOLE":
            k_params[:, 0] = k_params[:, 0] * (1 + params[:, 0])  # fx
            k_params[:, 1] = k_params[:, 1] * (1 + params[:, 1])  # fy

        return k_model, k_params.squeeze()

    def update_all_matrices(self, indices = None):
        """Init/Update all intrinsic matrices for all cameras and store them internally."""
      
      
        all_ids = self.keys
        self.cameras = None
        self.cameras = self.get_intrinsic_matrix(all_ids)

        # don't using to have simpler camera management
        # self.cameras_inv = None
        # self.cameras_inv = self.get_inverse_intrinsic_matrix(all_ids)

    def __repr__(self):
        s = "CameraModel:\n"
        # Only print first 5 to avoid clutter if there are thousands of cams
        limit = 5
        count = 0
        for i, (model, params) in enumerate(zip(self.k_models, self.k_params)):
            if count >= limit:
                s += f"  ... and {len(self.k_models) - limit} more.\n"
                break
            s += f"  Camera {self.tensor_idx_to_image[i]}: Model={model}, Params={params.detach().cpu().tolist()}\n"
            count += 1
        return s

####################################################################3

    def add_element(self, new_image_name, new_id, new_params = None):
        self.init_scheduler(warmup_steps=0, max_num_iterations = self.max_num_iterations)

        super().add_element(new_image_name, new_id, new_params)

        return 

    def get_parameters_idx(self, indices, recurse=True):
        return self.params[indices]

import torch
import torch.nn as nn


class CameraModule(nn.Module):
    def __init__(
        self,
        cam_id: dict,
        k_models: list[str],
        k_params: torch.Tensor,
        lr: float = 1e-3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Class storing camera intrinsics for multiple cameras and poses.
        Args:
            cam_id: dict mapping reconstruction camera ids (real world) to tensor indices (0..N)
            k_models: list of strings, camera model names per camera
            k_params: Nx4 tensor of camera parameters, fx, fy, cx, cy
            device: torch device
        """
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Define model ID mapping
        self.camera_model_name_to_id = {
            "PINHOLE": 0,
            "SIMPLE_PINHOLE": 1,
        }

        # --- ID Mappings ---
        self.recon_to_tensor_cam_id = cam_id
        self.tensor_to_recon_cam_id = {v: k for k, v in cam_id.items()}
        self.keys = list(self.recon_to_tensor_cam_id.keys())

        # --- Model Types ---
        self.k_models = k_models
        # Store model types as a Tensor for fast masking during forward pass
        model_ids_list = [self.camera_model_name_to_id[m] for m in k_models]
        self.register_buffer(
            "k_models_ids", torch.tensor(model_ids_list, device=self.device)
        )

        # --- Parameters ---
        # Use nn.Parameter so nn.Module tracks it automatically
        # self.k_params = nn.Parameter(
        #     k_params.clone().detach().to(self.device, dtype=self.dtype),
        #     requires_grad=True,
        # )
        self.k_params = k_params.clone().detach().to(self.device, dtype=self.dtype)
        self.alphas = nn.Parameter(
            torch.zeros((self.k_params.shape[0], 2), device=self.device),
            requires_grad=True,
        )

        self.lr = float(lr)
        self.init_optimizer(k_lr=self.lr)
        self.init_scheduler(
            lr_reduce_factor=0.75, patience=3, min_lr=self.lr / 20
        )  # this params seem good

        self.update_all_matrices()  # Precompute all intrinsic matrices

    def init_optimizer(self, k_lr: float):
        """Re-initialize optimizer with new learning rate."""
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.alphas, "lr": k_lr},
            ]
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

    def map_camera_ids_to_indices(self, camera_ids) -> torch.LongTensor:
        """
        Robustly maps reconstruction IDs (arbitrary ints) to internal tensor indices (0..N).
        Handles inputs as list, numpy, or tensor.
        """
        # Ensure input is iterable on CPU for dictionary lookup
        if isinstance(camera_ids, torch.Tensor):
            ids_cpu = camera_ids.detach().cpu().tolist()
        elif isinstance(camera_ids, (list, tuple)):
            ids_cpu = camera_ids
        else:
            ids_cpu = [camera_ids]  # Handle single int

        # Perform lookup
        try:
            indices = [self.recon_to_tensor_cam_id[cid] for cid in ids_cpu]
        except KeyError as e:
            raise ValueError(
                f"Camera ID {e} found in batch but not in initialization dictionary."
            )

        # Return as LongTensor on the correct device for indexing
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def get_all_intrinsic_matrix(self):
        return self.keys, self.get_intrinsic_matrix(self.keys)

    def get_intrinsic_matrix(self, indices) -> torch.Tensor:
        """Construct K matrix from parameters on-the-fly. Handles mixed models."""

        # 1. Get the internal row indices for the requested cameras
        if isinstance(indices[0], str):
            indices = self.map_camera_ids_to_indices(indices)

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

    def _build_pinhole(
        self, tensor_indices: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Internal helper: Pinhole (fx, fy, cx, cy)"""
        params = self.k_params[tensor_indices]  # Shape (B, 4)

        fx, fy = params[:, 0], params[:, 1]
        cx, cy = params[:, 2], params[:, 3]
        alpha = self.alphas[tensor_indices]

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
        alpha = self.alphas[tensor_indices]

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

    def __repr__(self):
        s = "CameraModel:\n"
        # Only print first 5 to avoid clutter if there are thousands of cams
        limit = 5
        count = 0
        for i, (model, params) in enumerate(zip(self.k_models, self.k_params)):
            if count >= limit:
                s += f"  ... and {len(self.k_models) - limit} more.\n"
                break
            s += f"  Camera {self.tensor_to_recon_cam_id[i]}: Model={model}, Params={params.detach().cpu().tolist()}\n"
            count += 1
        return s

    def parameters(self, recurse=True):  # Changed: match nn.Module signature
        """Return list of trainable parameters - only self.params is a leaf tensor"""
        return [self.alphas]  # Changed: return iterator like nn.Module

    def get_camera_parameters(self, indices) -> torch.Tensor:
        """Get camera parameters for given camera IDs."""
        if isinstance(indices[0], str):
            indices = self.map_camera_ids_to_indices(indices)

        k_model = self.k_models[indices]
        k_params = self.k_params[indices]
        alphas = self.alphas[indices]

        if k_params.shape[1] == 3 and k_model[0] == "SIMPLE_PINHOLE":
            k_params[:, 0] = k_params[:, 0] * (1 + alphas[:, 0])  # fx

        elif k_params.shape[1] == 4 and k_model[0] == "PINHOLE":
            k_params[:, 0] = k_params[:, 0] * (1 + alphas[:, 0])  # fx
            k_params[:, 1] = k_params[:, 1] * (1 + alphas[:, 1])  # fy

        return k_model, k_params.squeeze()

    def update_all_matrices(self):
        """Init/Update all intrinsic matrices for all cameras and store them internally."""
        all_ids = self.keys
        self.cameras = self.get_intrinsic_matrix(all_ids)
        self.cameras_inv = torch.linalg.inv(self.cameras)

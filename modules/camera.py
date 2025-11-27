import torch
import torch.nn as nn


class CameraModule(nn.Module):
    def __init__(
        self,
        cam_id: dict,
        k_models: list[str],
        k_params: torch.Tensor,
        k_grad: bool = True,
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
        self.k_grad = k_grad
        # Use nn.Parameter so nn.Module tracks it automatically
        self.k_params = nn.Parameter(
            k_params.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=self.k_grad,
        )

        self.update_all_matrices()  # Precompute all intrinsic matrices

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

    def get_intrinsic_matrix(self, camera_ids) -> torch.Tensor:
        """Construct K matrix from parameters on-the-fly. Handles mixed models."""

        # 1. Get the internal row indices for the requested cameras
        tensor_indices = self.map_camera_ids_to_indices(camera_ids)
        batch_size = tensor_indices.shape[0]

        # 2. Retrieve the model type for these specific cameras
        # Use standard indexing, k_models_ids is a buffer
        models_in_batch = self.k_models_ids[tensor_indices]

        # 3. Check if we can take a fast path (all cameras are the same type)
        if (models_in_batch == 0).all():
            return self._build_pinhole(tensor_indices, batch_size)

        elif (models_in_batch == 1).all():
            return self._build_simple_pinhole(tensor_indices, batch_size)

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
                idx_pinhole = tensor_indices[mask_pinhole]
                # Calculate and assign to the masked slice of K
                K[mask_pinhole] = self._build_pinhole(idx_pinhole, idx_pinhole.shape[0])

            # Process SIMPLE_PINHOLE cameras
            if mask_simple.any():
                idx_simple = tensor_indices[mask_simple]
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

        K = torch.zeros((batch_size, 3, 3), dtype=params.dtype, device=self.device)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
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

        f = params[:, 0]
        cx = params[:, 1]
        cy = params[:, 2]

        K = torch.zeros((batch_size, 3, 3), dtype=params.dtype, device=self.device)
        K[:, 0, 0] = f
        K[:, 1, 1] = f
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    # --- Wrappers for backward compatibility or specific calls ---
    def intrinsic_matrix_pinhole(self, camera_ids) -> torch.Tensor:
        """Assumes all cameras are PINHOLE
        Args:
            camera_ids: list or tensor of camera IDs in the reconstruction
        Returns:
            K: Bx3x3 intrinsic matrices
        """
        indices = self.map_camera_ids_to_indices(camera_ids)
        return self._build_pinhole(indices, indices.shape[0])

    def intrinsic_matrix_simple_pinhole(self, camera_ids) -> torch.Tensor:
        """Assumes all cameras are SIMPLE_PINHOLE
        Args:
            camera_ids: list or tensor of camera IDs in the reconstruction
        Returns:
            K: Bx3x3 intrinsic matrices
        """
        indices = self.map_camera_ids_to_indices(camera_ids)
        return self._build_simple_pinhole(indices, indices.shape[0])

    def get_intrinsic_matrix_inverse(self, camera_ids) -> torch.Tensor:
        """Get inverse intrinsic matrices for given camera IDs. Uses precomputed inverses."""
        indices = self.map_camera_ids_to_indices(camera_ids)
        return self.cameras_inv[indices]

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
        return [self.k_params]  # Changed: return iterator like nn.Module

    def get_camera_parameters(self, camera_ids) -> torch.Tensor:
        """Get camera parameters for given camera IDs."""
        indices = self.map_camera_ids_to_indices(camera_ids)
        return self.k_models[indices], self.k_params[indices].squeeze()

    def update_all_matrices(self):
        """Init/Update all intrinsic matrices for all cameras and store them internally."""
        all_ids = self.keys
        self.cameras = self.get_intrinsic_matrix(all_ids)
        self.cameras_inv = torch.linalg.inv(self.cameras)

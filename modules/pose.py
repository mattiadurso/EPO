import torch
import torch.nn as nn
import pypose as pp

# import kornia.geometry.conversions as kgc  # maybe I can remove kornia
from modules.mlp import PoseRefinementMLP
from modules.base_module import BaseModule


class PoseModule(BaseModule):
    def __init__(
        self,
        image_id_map: dict[str, int],
        hw: tuple[int, int],
        R: torch.Tensor,
        t: torch.Tensor,
        t_lr: float = 1e-3,
        q_lr: float = 1e-4,
        grad_q: bool = False,
        grad_t: bool = False,
        grad_t_offset: bool = False,
        use_mlp: bool = True,
        mlp_lr: float = 3e-3,
        device: str = "cuda",
        max_num_iterations: int = 2048,
        warmup_steps: int = 25, #25
        dtype: torch.dtype = torch.float32,
    ):
        """
        Class storing extrinsics (Pose) for multiple cameras.
        Assumes World-to-Camera convention (T_cw).

        Args:
            image_id_map: Mapping from image IDs to tensor indices
            hw: Height and width of the input images
            R: (N, 3, 3) tensor of rotation matrices
            t: (N, 3) or (N, 3, 1) tensor of translation vectors
            t_lr: Learning rate for translation optimizer
            q_lr: Learning rate for rotation optimizer
            grad_q: Whether to compute gradients for rotation
            grad_t: Whether to compute gradients for translation
            grad_t_offset: Whether to compute gradients for translation offset
            use_mlp: Whether to use MLP for pose refinement
            mlp_lr: Learning rate for MLP optimizer
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
        self.hw = hw  # (H, W)
        self.max_num_iterations = max_num_iterations

        if use_mlp:
            print(f"When using MLP pose refinement, q and t gradients are disabled.")

        grad_q = False if use_mlp else grad_q
        grad_t = False if use_mlp else grad_t
        self.grad_t_offset = grad_t_offset
        self.t_lr = t_lr
        self.q_lr = q_lr
        self.mlp_lr = mlp_lr
        num_cams = len(image_id_map)

        # --- Rotation Prep (Matrix -> Quat) ---
        # # Kornia returns (w, x, y, z)
        # q_init = kgc.rotation_matrix_to_quaternion(R)

        # # PyPose expects (x, y, z, w) (scalar last). We roll -1 to move w from index 0 to index 3
        # q_init = torch.roll(q_init, shifts=-1, dims=1)

        q_init = pp.mat2SO3(R).tensor()  # Returns (x,y,z,w) directly

        # Store as Raw Parameter (requires normalization on usage)
        self.q_param = nn.Parameter(
            q_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=grad_q,
        )

        # --- Translation Prep ---
        t_init = t.reshape(num_cams, 3).to(self.device, dtype=self.dtype)
        self.t_mean = t_init.mean(dim=0, keepdim=True)
        dist = torch.norm(t_init - self.t_mean, dim=1)
        self.t_scale = torch.clamp(torch.mean(dist), min=1e-10)
        t_init = (t_init - self.t_mean) / self.t_scale

        # Store as Raw Parameter
        self.t_param = nn.Parameter(
            t_init.clone().detach(),
            requires_grad=grad_t,
        )

        self.t_offset = nn.Parameter(
            torch.zeros_like(self.t_param).to(self.device, dtype=self.dtype),
            requires_grad=self.grad_t_offset,
        )

        if use_mlp:
            self.use_mlp = True
            self.mlp = PoseRefinementMLP(
                input_dim=12,  # 3x4 pose matrix flattened
                output_dim=12,  # 3x4 pose matrix flattened
                hidden_dim=256,  # or 256 for complex scenarios
            ).to(self.device, dtype=self.dtype)
            # Initialize optimizer with MLP parameters
            self.init_optimizer(mlp_lr=mlp_lr)
        else:
            self.use_mlp = False
            # Initialize optimizer with q and t parameters
            self.init_optimizer(t_lr=self.t_lr, q_lr=self.q_lr)

        # Initialize learning rate scheduler
        self.init_scheduler(warmup_steps, max_num_iterations)

        # Precompute all extrinsic matrices
        self.update_all_matrices()

    def init_optimizer(
        self, t_lr: float = 1e-3, q_lr: float = 1e-4, mlp_lr: float = 3e-3
    ):
        """Re-initialize optimizer with new learning rate."""
        if self.use_mlp:
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.mlp.parameters(),
                        "lr": mlp_lr,
                        "name": "mlp",
                        "weight_decay": 0.01,  # default
                        "eps": 1e-10,
                    },
                    # {
                    #     "params": [self.t_offset],
                    #     "lr": mlp_lr,
                    #     "name": "t_offset",
                    #     "weight_decay": 0.1,  # more aggressive decay
                    #     "eps": 1e-10,
                    # },
                ]
            )

        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [self.t_param], "lr": t_lr, "name": "t", "eps": 1e-10},
                    {"params": [self.q_param], "lr": q_lr, "name": "q", "eps": 1e-10},
                    # {
                    #     "params": [self.t_offset],
                    #     "lr": t_lr,
                    #     "name": "t_offset",
                    #     "weight_decay": 0.1,
                    #     "eps": 1e-10,
                    # },
                ],
            )

    def get_rotation_matrix(self, indices) -> torch.Tensor:
        """
        Returns (B, 3, 3) Rotation Matrix for the requested images.
        """
        # 2. Get quaternions for batch
        q_batch = self.q_param[indices]

        # 1. Normalize all quaternions (important for stability)
        q_norm = q_batch / (torch.norm(q_batch, dim=1, keepdim=True) + 1e-10)

        # 3. Convert to Matrix via PyPose
        return pp.SO3(q_norm).matrix()

    def get_translation(self, indices) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        # Apply scaling to the learnable parameter
        return self.t_param[indices]

    def get_translation_offset(self, indices) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        # Apply scaling to the learnable parameter
        return self.t_offset[indices]

    def get_projection_matrix(self, indices) -> torch.Tensor:
        """
        Constructs 4x4 SE3 Matrix [R|t].
        Standard convention: World-to-Camera.
        Handles both main poses and the locally-stored new pose.
        """
        batch_size = len(indices)
        if isinstance(indices[0], str):
            indices = self.map_names_to_indices(indices)

        if self.poses is not None:
            return self.poses[indices]

        # Retrieve components
        R = self.get_rotation_matrix(indices)  # (B, 3, 3)
        t = self.get_translation(indices)  # (B, 3)
        R, t = self.apply_mlp(R, t, indices)
        t = t * self.t_scale + self.t_mean  # Rescale to physical units

        # Build 4x4 Matrix
        P = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t

        return P
    
    def get_rotation_matrix(self, indices) -> torch.Tensor:
        """
        Returns (B, 3, 3) Rotation Matrix for the requested images.
        Handles both main poses and the locally-stored new pose.
        """
        # Check if this is the new local pose
        if (hasattr(self, 'new_image_idx') and 
            len(indices) == 1 and indices[0] == self.new_image_idx):
            q_batch = self.q_new_local
        else:
            q_batch = self.q_param[indices]

        # 1. Normalize all quaternions (important for stability)
        q_norm = q_batch / (torch.norm(q_batch, dim=1, keepdim=True) + 1e-10)

        # 3. Convert to Matrix via PyPose
        return pp.SO3(q_norm).matrix()

    def get_translation(self, indices) -> torch.Tensor:
        """
        Returns (B, 3) Translation vectors.
        Handles both main poses and the locally-stored new pose.
        """
        # Check if this is the new local pose
        if (hasattr(self, 'new_image_idx') and 
            len(indices) == 1 and indices[0] == self.new_image_idx):
            return self.t_new_local
        else:
            return self.t_param[indices]

    def get_translation_offset(self, indices) -> torch.Tensor:
        """
        Returns (B, 3) Translation offsets.
        Handles both main poses and the locally-stored new pose.
        """
        # Check if this is the new local pose
        if (hasattr(self, 'new_image_idx') and 
            len(indices) == 1 and indices[0] == self.new_image_idx):
            return self.t_offset_new_local
        else:
            return self.t_offset[indices]

    def __repr__(self):
        limit = 3
        s = f"PoseModel ({len(self.image_to_tensor_idx)} poses):\n"
        for i in range(min(limit, len(self.t_param))):
            name = self.tensor_idx_to_image[i]
            q_val = self.q_param[i].detach().cpu().numpy()

            # Fetch physical translation (scaled) for representation
            t_val = self.get_translation([i]).detach().cpu().numpy().flatten()

            # s += f"  Image '{name}': R={q_val:.3e}, t={t_val:.3e}\n"
            # print arrays with 3 decimal places signed, align them considering the sign too
            s += f"  Image '{name}': q=[{q_val[0]:+.3e}, {q_val[1]:+.3e}, {q_val[2]:+.3e}, {q_val[3]:+.3e}], t=[{t_val[0]:+.3e}, {t_val[1]:+.3e}, {t_val[2]:+.3e}]\n"

        if len(self.t_param) > limit:
            s += f"  ... {len(self.t_param) - limit} more."
        return s

    def parameters(self, t=False, q=False, mlp=False, recurse: bool = True):
        """Return list of trainable parameters - only leaf tensors"""

        params = []

        if mlp and self.use_mlp:  # no q and t when mlp is used
            params.extend([p for p in self.mlp.parameters() if p.requires_grad])
        else:
            if q and self.q_param.requires_grad:
                params.append(self.q_param)
            if t and self.t_param.requires_grad:
                params.append(self.t_param)
        return params

    def get_image_qt(self, indices):
        """Get quaternion and translation for given image names."""
        indices = (
            self.map_names_to_indices(indices)
            if isinstance(indices[0], str)
            else indices
        )
        R = self.get_rotation_matrix(indices)
        t = self.get_translation(indices)

        # Ensure MLP is applied before returning values
        if self.use_mlp:
            R, t = self.apply_mlp(R, t, indices)

        # Convert R to quaternion
        q = pp.mat2SO3(R).tensor()  # Returns (x,y,z,w) directly
        t = self.t_scale * t + self.t_mean  # Rescale to physical units

        return q.squeeze(), t.squeeze()

    def update_all_matrices(self, indices = None):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        if indices is not None:
            # 1. Get the current poses but "cut" the link to previous iterations
            # This prevents the "backward through the graph a second time" error
            updated_poses = self.poses.detach().clone()
            
            all_names = list(self.image_to_tensor_idx.keys())
            self.poses = None  # Invalidate cache before recomputing
            self.poses = self.get_projection_matrix(all_names)

            # 2. Compute the update (this will have a fresh graph for THIS iteration)
            #new_matrix = self.get_projection_matrix(indices)
            
            # 3. Use clone to make a unique memory copy (safest practice)
            updated_poses[indices] = self.get_projection_matrix(indices)

            # 4. Assign back
            self.poses = updated_poses
        else:
            all_names = list(self.image_to_tensor_idx.keys())
            self.poses = None  # Invalidate cache before recomputing
            self.poses = self.get_projection_matrix(all_names)

    

    def get_all_matrices(self):
        """Get all extrinsic matrices for all images."""
        if self.poses is None:
            self.update_all_matrices()

        return self.poses
    
    def apply_mlp(self, R, t, indices) -> torch.Tensor:
        """Apply MLP refinement to given poses."""
        if not self.use_mlp:
            return R, t

        # Build 3x4 pose matrices and apply MLP
        P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)

        # adding mlp residual and orthonormalization
        P_3x4_refined = self.mlp(P_3x4)


        R = P_3x4_refined[:, :3, :3]
        t = P_3x4_refined[:, :3, 3] #+ self.get_translation_offset(indices)

        return R, t
    
    def mlp_offset(self, indices) -> torch.Tensor:
        """Apply MLP refinement to given poses."""
        indices = (
            self.map_names_to_indices(indices)
            if isinstance(indices[0], str)
            else indices
        )
        R = self.get_rotation_matrix(indices)
        t = self.get_translation(indices)

        # Build 3x4 pose matrices and apply MLP
        P_3x4 = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)

        # adding mlp residual and orthonormalization
        return self.mlp(P_3x4, sum=True)

    
    def get_new_pose_params(self):
        """
        Get parameters for the newly added pose (before merging into main tensors).
        Use this to create an optimizer for local pose refinement.
        
        Returns:
            list: [q_new, t_new, t_offset_new] parameters
        """
        if not hasattr(self, 'q_new_local'):
            raise RuntimeError("No new pose added yet. Call add_element() first.")
        
        return [self.q_new_local, self.t_new_local, self.t_offset_new_local]
    
    def merge_new_pose_to_main(self):
        """
        Merge the locally-optimized new pose into the main parameter tensors.
        Call this after optimization of the new pose completes.
        """
        if not hasattr(self, 'q_new_local'):
            raise RuntimeError("No new pose to merge.")
        
        # Concatenate into main parameter tensors
        self.q_param = nn.Parameter(
            torch.cat([self.q_param.data, self.q_new_local.data], dim=0),
            requires_grad=self.q_param.requires_grad
        )
        self.t_param = nn.Parameter(
            torch.cat([self.t_param.data, self.t_new_local.data], dim=0),
            requires_grad=self.t_param.requires_grad
        )
        self.t_offset = nn.Parameter(
            torch.cat([self.t_offset.data, self.t_offset_new_local.data], dim=0),
            requires_grad=self.t_offset.requires_grad
        )
        
        # Reinitialize optimizer to include the merged parameters
        if self.use_mlp:
            self.init_optimizer(mlp_lr=self.mlp_lr)
        else:
            self.init_optimizer(t_lr=self.t_lr, q_lr=self.q_lr)
        
        # Reinitialize scheduler
        self.init_scheduler(warmup_steps=0, max_num_iterations=self.max_num_iterations)
        
        # Clean up temporary local parameters
        del self.q_new_local, self.t_new_local, self.t_offset_new_local
        del self.new_image_name, self.new_image_idx
        
        # Update cached matrices
        self.update_all_matrices()


    #####################################################################################################################

    def add_element(self, 
                    image_name: str, 
                    R_new: torch.Tensor, 
                    t_new: torch.Tensor, 
                    t_offset_new : torch.Tensor = None,
                    t_lr: float = 1e-4, 
                    q_lr: float = 1e-5, 
                    mlp_lr: float = 3e-4
                    ):
        """
        Args:
            image_id: Unique string ID for the new camera
            R_new: (3, 3) rotation matrix
            t_new: (3,) or (3, 1) translation vector
        """
        # if image_name in self.image_to_tensor_idx:
        #     print(f"Pose for {image_name} already exists.")
        #     return

        # 1. Update Mapping
        new_idx = len(self.image_to_tensor_idx)
        self.image_to_tensor_idx[image_name] = new_idx
        self.tensor_idx_to_image[new_idx] = image_name  # Keep reverse mapping in sync

        # 2. Process New Rotation (Matrix -> SO3 Quat x,y,z,w)
        q_new = pp.mat2SO3(R_new.view(1, 3, 3), check=False).tensor().to(self.device, dtype=self.dtype)
        
        # 3. Process New Translation (Normalize using existing scale/mean)
        t_new = t_new.view(1, 3).to(self.device, dtype=self.dtype)

        # 4. Concatenate and Re-assign Parameters
        self.q_param = nn.Parameter(
            torch.cat([self.q_param.data, q_new], dim=0),
            requires_grad=self.q_param.requires_grad
        )
        self.t_param = nn.Parameter(
            torch.cat([self.t_param.data, t_new], dim=0),
            requires_grad=self.t_param.requires_grad
        )

        t_offset = t_offset_new.view(1, 3) if t_offset_new is not None else torch.zeros_like(t_new)
        
        self.t_offset = nn.Parameter(
            torch.cat([self.t_offset.data, t_offset], dim=0),
            requires_grad=self.t_offset.requires_grad
        )

        self.mlp_lr = mlp_lr
        self.q_lr = q_lr
        self.t_lr = t_lr

        self.refresh_after_addition()
        self.update_matrix_direct(new_idx) # ! Important to ensure the detachment from the MLP

     
        
    def get_parameters_idx(self, indices, t=False, q=False, mlp=False):
        """Return list of trainable parameters - only leaf tensors"""

        params = []

        if mlp and self.use_mlp:  # no q and t when mlp is used
            params.extend([p for p in self.mlp.parameters() if p.requires_grad])
        else:
            if q and self.q_param.requires_grad:
                params.append(self.q_param[indices])
            if t and self.t_param.requires_grad:
                params.append(self.t_param[indices])
        return params
    
    
    def refresh_after_addition(self):
        # Re-init optimizer
        if self.use_mlp:
            self.init_optimizer(mlp_lr = self.mlp_lr)
        else:
            self.init_optimizer(t_lr = self.t_lr, q_lr=self.q_lr)
        
        # Re-init scheduler to match the new optimizer
        self.init_scheduler(warmup_steps=0, max_num_iterations=self.max_num_iterations)
        
        #self.update_all_matrices()
    
  
    def get_projection_matrix_direct(self, indices, q_override=None, t_override=None):
        """
        Builds projection matrix bypassing MLP.
        q_override/t_override are standalone leaf tensors from forward_frame
        so gradients flow directly to them.
        """
        if q_override is not None:
            q_norm = q_override / (q_override.norm(dim=-1, keepdim=True) + 1e-10)
            R = pp.SO3(q_norm).matrix()
        else:
            R = self.get_rotation_matrix(indices)

        if t_override is not None:
            t = t_override
        else:
            t = self.get_translation(indices)

        t = t * self.t_scale + self.t_mean

        batch_size = len(indices)
        P = torch.eye(4, device=self.device, dtype=self.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t
        return P  

    def update_matrix_direct(self, idx, q_override=None, t_override=None):
        """
        Rebuilds self.poses with a fresh differentiable matrix at idx.
        Uses torch.cat so autograd can track the gradient at idx.
        """
        P_new = self.get_projection_matrix_direct([idx], q_override, t_override)  # (1, 4, 4)
        P_new = self.mlp.gram_schmidt(P_new)  # (1, 3, 4)
        bottom = torch.tensor([[[0., 0., 0., 1.]]], device=P_new.device, dtype=P_new.dtype).expand(P_new.shape[0], -1, -1)
        P_new = torch.cat([P_new, bottom], dim=1)  # (N, 4, 4)

        if idx == 0:
            self.poses = torch.cat([P_new, self.poses[1:].detach()], dim=0)
        elif idx == len(self.poses) - 1:
            self.poses = torch.cat([self.poses[:-1].detach(), P_new], dim=0)
        else:
            self.poses = torch.cat([
                self.poses[:idx].detach(),
                P_new,
                self.poses[idx+1:].detach()
            ], dim=0)
import torch
import torch.nn as nn
import pypose as pp
import kornia.geometry.conversions as kgc


class PoseModule(nn.Module):
    def __init__(
        self,
        image_id_map: dict[str, int],
        R: torch.Tensor,
        t: torch.Tensor,
        grad_q: bool = True,
        grad_t: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Class storing extrinsics (Pose) for multiple cameras.
        Assumes World-to-Camera convention (T_cw).

        Args:
            image_id_map: dict mapping image filenames (str) to tensor indices (int)
            R: (N, 3, 3) tensor of rotation matrices
            t: (N, 3) or (N, 3, 1) tensor of translation vectors
            grad_q: Optimize rotation?
            grad_t: Optimize translation?
            device: torch device
        """
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # --- ID Mappings ---
        # string name -> tensor index (0..N)
        self.image_to_tensor_idx = image_id_map
        # tensor index -> string name (inverse)
        self.tensor_idx_to_image = {v: k for k, v in image_id_map.items()}

        num_cams = len(image_id_map)
        assert (
            R.shape[0] == num_cams
        ), f"R shape {R.shape} mismatch with num images {num_cams}"

        # --- Rotation Prep (Matrix -> Quat) ---
        # Kornia returns (w, x, y, z)
        q_init = kgc.rotation_matrix_to_quaternion(R)

        # PyPose expects (x, y, z, w) (scalar last)
        # We roll -1 to move w from index 0 to index 3
        q_init = torch.roll(q_init, shifts=-1, dims=1)

        # Store as Raw Parameter (requires normalization on usage)
        self.q_param = nn.Parameter(
            q_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=grad_q,
        )

        # --- Translation Prep ---
        t_init = t.reshape(num_cams, 3)
        self.t_param = nn.Parameter(
            t_init.clone().detach().to(self.device, dtype=self.dtype),
            requires_grad=grad_t,
        )

        self.update_all_matrices()  # Precompute all extrinsic matrices

    def map_names_to_indices(self, image_names) -> torch.LongTensor:
        """Robustly maps string names to tensor indices."""
        # Handle single string input
        if isinstance(image_names, str):
            image_names = [image_names]

        try:
            indices = [self.image_to_tensor_idx[name] for name in image_names]
        except KeyError as e:
            raise ValueError(
                f"Image name {e} not found in PoseModel initialization dict."
            )

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def get_rotation_matrix(self, image_names) -> torch.Tensor:
        """
        Returns (B, 3, 3) Rotation Matrix for the requested images.
        """
        # First normalize all quaternions (important for stability)
        self.q_batch = self.q_param / torch.norm(self.q_param, dim=1, keepdim=True)

        # Map names to indices
        indices = self.map_names_to_indices(image_names)

        # 1. Get raw quaternions for batch
        q_batch = self.q_param[indices]

        # 3. Convert to Matrix via PyPose
        # PyPose SO3 wraps (x,y,z,w)
        return pp.SO3(q_batch).matrix()

    def get_translation(self, image_names) -> torch.Tensor:
        """Returns (B, 3) Translation vectors"""
        indices = self.map_names_to_indices(image_names)
        return self.t_param[indices]

    def get_projection_matrix(self, image_names) -> torch.Tensor:
        """
        Constructs 4x4 SE3 Matrix [R|t].
        Standard convention: World-to-Camera.
        """
        indices = self.map_names_to_indices(image_names)
        batch_size = len(indices)

        # Retrieve components
        R = self.get_rotation_matrix(image_names)  # (B, 3, 3)
        t = self.t_param[indices]  # (B, 3)

        # Build 4x4 Matrix
        P = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P[:, :3, :3] = R
        P[:, :3, 3] = t

        return P

    def get_projection_matrix_inverse(self, image_names) -> torch.Tensor:
        """
        Constructs 4x4 SE3 Inverse Matrix [R^T | -R^T * t].
        Numerically stable inversion (Camera-to-World).
        """
        indices = self.map_names_to_indices(image_names)
        batch_size = len(indices)

        # Get components
        R = self.get_rotation_matrix(image_names)  # (B, 3, 3)
        t = self.t_param[indices]  # (B, 3)

        # Stable Inversion Logic:
        # P_inv = | R^T   -R^T * t |
        #         | 0        1     |

        R_T = R.permute(0, 2, 1)  # Transpose R
        # -R^T * t (Need t as Bx3x1 for matmul)
        t_inv = -torch.bmm(R_T, t.unsqueeze(2))

        # Build Matrix
        P_inv = torch.eye(4, device=self.device, dtype=R.dtype).repeat(batch_size, 1, 1)
        P_inv[:, :3, :3] = R_T
        P_inv[:, :3, 3:4] = t_inv  # Assign Bx3x1 directly to column slice

        return P_inv

    def forward(self, image_names):
        """Convenience method to get RT matrices"""
        return self.get_projection_matrix(image_names)

    def __repr__(self):
        limit = 3
        s = f"PoseModel ({len(self.image_to_tensor_idx)} poses):\n"
        for i in range(min(limit, len(self.t_param))):
            name = self.tensor_idx_to_image[i]
            q_val = self.q_param[i].detach().cpu().numpy()
            t_val = self.t_param[i].detach().cpu().numpy()
            # s += f"  Image '{name}': R={q_val:.3e}, t={t_val:.3e}\n"
            # print arrays with 3 decimal places signed, align them considering the sign too
            s += f"  Image '{name}': q=[{q_val[0]:+.3e}, {q_val[1]:+.3e}, {q_val[2]:+.3e}, {q_val[3]:+.3e}], t=[{t_val[0]:+.3e}, {t_val[1]:+.3e}, {t_val[2]:+.3e}]\n"
        if len(self.t_param) > limit:
            s += f"  ... {len(self.t_param) - limit} more."
        return s

    def parameters(self, t=True, q=True, recurse: bool = True):
        """Return list of trainable parameters - only leaf tensors"""
        params = []
        if q and self.q_param.requires_grad:
            params.append(self.q_param)
        if t and self.t_param.requires_grad:
            params.append(self.t_param)
        return params  # Changed: return list instead of iterator

    def get_image_qt(self, image_names):
        """Get quaternion and translation for given image names."""
        indices = self.map_names_to_indices(image_names)
        return self.q_param[indices].squeeze(), self.t_param[indices].squeeze()

    def update_all_matrices(self):
        """Init/Update all extrinsic matrices for all images and store them internally."""
        all_names = list(self.image_to_tensor_idx.keys())
        self.poses = self.get_projection_matrix(all_names)

import torch
from torch import Tensor
from losses.dt_loss import sample_distance_field


### From 2D to 3D world coordinates


def to_homogeneous(xy: Tensor) -> Tensor:
    """Converts 2D points to homogeneous coordinates."""
    batch_shape = xy.shape[:-1]
    ones = torch.ones(*batch_shape, 1, dtype=xy.dtype, device=xy.device)
    return torch.cat((xy, ones), dim=-1)


def unproject_to_virtual_plane(
    xy: Tensor,
    K_inv: Tensor,
) -> Tensor:
    """unproject points to the camera virtual plane at depth 1
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
    Returns:
        xyz: 3D points laying on the virtual plane
            B,n,3
    """
    xy_hom = to_homogeneous(xy)  # B,n,3
    xyz = K_inv @ (xy_hom.permute(0, 2, 1))
    return xyz.permute(0, 2, 1)


def unproject_to_3D(xy: Tensor, K_inv: Tensor, depths: Tensor) -> Tensor:
    """unproject points to 3D in the camera ref system
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        depths: the points depth
            B,n
    Returns:
        xyz: unprojected 3D points in the camera reference system
            B,n,3
    """
    # commented to make torch.compile happy
    # assert xy.shape[0] == K.shape[0] and xy.shape[0] == depths.shape[0]
    # assert (
    #     xy.shape[1] == depths.shape[1]
    # ), f"Expected xy and depths to have the same number of points, got {xy.shape[1]} and {depths.shape[1]}"
    # assert xy.shape[2] == 2

    xyz = unproject_to_virtual_plane(xy, K_inv)  # B,n,3
    depths = depths.unsqueeze(-1)  # B,n,1
    xyz_scaled = xyz * depths  # B,n,3

    return xyz_scaled


def invert_P(P: Tensor) -> Tensor:
    """invert the extrinsics P matrix in a more stable way
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    """
    B = P.shape[0]
    R = P[:, 0:3, 0:3]
    t = P[:, 0:3, 3:4]
    # 3x4
    P_inv = torch.cat((R.permute(0, 2, 1), -R.permute(0, 2, 1) @ t), dim=2)

    # Ensure device matches P
    template = torch.tensor([[[0, 0, 0, 1]]], device=P.device, dtype=P.dtype)
    # Expand
    bottom_row = template.expand(B, 1, 4)

    # 4x4
    P_inv = torch.cat((P_inv, bottom_row), dim=1)
    return P_inv


def unproject_2D_to_world(
    xy0: Tensor, K0: Tensor, depth0: Tensor, P0: Tensor
) -> Tensor:
    """unproject points to world coordinates
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        depths: the points depth
            B,n
        P: camera extrinsics matrix
            B,4,4
    Returns:
        xyz_world: unprojected 3D points in the world reference system
            B,n,3
    """
    # invert K and P
    K0_inv = torch.linalg.inv(K0)
    P0_inv = invert_P(P0)

    # 2D -> 3D camera
    xyz_camera = unproject_to_3D(xy0, K0_inv, depth0)  # B,n,3

    # 3D camera -> world
    R_inv, t_inv = P0_inv[:, :3, :3], P0_inv[:, :3, 3:]  # B,3,3 , B,3,1
    xyz_world = (R_inv @ xyz_camera.permute(0, 2, 1) + t_inv).permute(0, 2, 1)  # B,n,3

    return xyz_world


#### From homogeneous coordinates to 2D


def from_homogeneous(points: Tensor) -> Tensor:
    """Converts homogeneous coordinates to 2D points."""
    eps = 1e-8
    z_vec = points[..., -1:]
    # set the results of division by zero/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))
    output = scale * points[..., :-1]
    return output


def filter_outside_safe(
    xy: Tensor, shape: Tensor | tuple[int, int], border: int = 0
) -> tuple[Tensor, Tensor]:
    """
    Identifies points outside the image.
    Returns:
        xy_safe: Points with outside values replaced by 0.0 (safe for graph execution)
        outside_mask: Boolean mask where True indicates the point was outside.
    """
    # Handle both single shape [H, W] and batched shapes [B, 2]
    if isinstance(shape, tuple) or (shape).ndim == 1:
        H, W = shape[0], shape[1]
    else:
        # shape is (B, 2). We need (B, 1) for broadcasting against (B, n)
        H = shape[..., 0:1]
        W = shape[..., 1:2]

    outside_mask = (
        (xy[..., 0] < border)
        | (xy[..., 0] >= W - border)
        | (xy[..., 1] < border)
        | (xy[..., 1] >= H - border)
    )

    # Replace outside points with 0.0 (valid coordinate, but masked later)
    # This prevents NaNs from propagating through the graph
    xy_safe = torch.where(
        outside_mask[..., None], torch.tensor(0.0, device=xy.device, dtype=xy.dtype), xy
    )

    return xy_safe, outside_mask


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    img_shape: Tensor,
    border: int = 0,
) -> tuple[Tensor, Tensor]:
    """project 3D points to 2D using the provided intrinsics matrix K."""
    original_dtype = xyz.dtype
    # B,3,3 * B,3,n =  B,3,n  -> B,n,3 after permutation
    xy_proj_hom = (K @ xyz.permute(0, 2, 1)).permute(0, 2, 1)
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    # Use safe filtering: sets outside points to 0.0, returns mask
    xy_proj, outside_mask = filter_outside_safe(xy_proj, img_shape, border)

    return xy_proj, outside_mask


def project_world_to_2D(
    xyz_world: Tensor,
    P1: Tensor,
    K1: Tensor,
    img1_shape: Tensor,
    border: int = 0,
) -> tuple[Tensor, Tensor]:

    # Extract R, t
    R1 = P1[:, :3, :3]
    t1 = P1[:, :3, 3:].transpose(-2, -1)  # B, 1, 3

    # World -> Camera
    # Formula: Camera = R * World + t
    # Shape optimized: (World @ R^T) + t
    xyz_camera1 = xyz_world @ R1.transpose(-2, -1) + t1

    xy_proj, outside_mask = project_to_2D(xyz_camera1, K1, img1_shape, border)

    return xy_proj, outside_mask


### Forward step


# @torch.compile(mode="reduce-overhead")
def project_and_sample_logic(
    xyz_world: torch.Tensor,
    K1: torch.Tensor,
    P1: torch.Tensor,
    img1_shape: torch.Tensor,
    dt_fields: torch.Tensor,
    border: int = 0,
):
    """
    Fused operation: Projection -> 2D -> Sampling.
    Guarantees no NaNs are produced. Invalid points are zeroed out and tracked via outside_mask.
    """
    # 1. Project World Points to 2D
    # uv_proj contains safe values (0.0) where points are outside
    uv_proj, outside_mask = project_world_to_2D(xyz_world, P1, K1, img1_shape, border)

    # 2. Prepare Distance Fields
    # Ensure 4D shape (B, C, H, W) for grid_sample
    if dt_fields.dim() == 3:
        dt_fields = dt_fields.unsqueeze(1)

    # 3. Sample Distance Field
    # uv_proj is guaranteed safe, so we don't need NaN checks inside
    sampled_vals = sample_distance_field(dt_fields, uv_proj)

    return sampled_vals, ~outside_mask

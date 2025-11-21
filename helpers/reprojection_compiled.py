import torch
from torch import Tensor

BOTTOM_ROW_TEMPLATE = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
### From 2D to 3D world coordinates


def to_homogeneous(xy: Tensor) -> Tensor:
    """Converts 2D points to homogeneous coordinates."""
    batch_shape = xy.shape[:-1]
    ones = torch.ones(*batch_shape, 1, dtype=xy.dtype, device=xy.device)
    return torch.cat((xy, ones), dim=-1)


def unproject_to_virtual_plane(
    xy: Tensor,
    K: Tensor,
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
    xyz = (torch.linalg.inv(K) @ (xy_hom.permute(0, 2, 1))).permute(0, 2, 1)
    return xyz


def unproject_to_3D(xy: Tensor, K: Tensor, depths: Tensor) -> Tensor:
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

    xyz = unproject_to_virtual_plane(xy, K)  # B,n,3
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
    template = BOTTOM_ROW_TEMPLATE.to(P.device).type(P.dtype)
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
    # 2D -> 3D camera
    xyz_camera = unproject_to_3D(xy0, K0, depth0)  # B,n,3

    # 3D camera -> world
    P_inv = invert_P(P0)  # B,4,4
    R_inv, t_inv = P_inv[:, :3, :3], P_inv[:, :3, 3:]  # B,3,3 , B,3,1
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


def filter_outside(xy: Tensor, shape: Tensor, border: int = 0) -> Tensor:
    """set as nan all the points that are not inside rectangle
    Args:
        xy: Points to filter (B, n, 2)
        shape: 1D Tensor [H, W] of the image shape
        border: Border margin
    """
    # MODIFICATION: Use | (logical or) instead of + for boolean tensors.
    # shape[0] is H, shape[1] is W
    outside_mask = (
        (xy[..., 0] < border)
        | (xy[..., 0] >= shape[1] - border)  # W
        | (xy[..., 1] < border)
        | (xy[..., 1] >= shape[0] - border)  # H
    )
    # Use torch.where instead of indexing assignment
    xy_filtered = torch.where(
        outside_mask[..., None], torch.full_like(xy, float("nan")), xy
    )
    return xy_filtered, outside_mask # Return the outside mask as well


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    # MODIFICATION: img_shape must be a Tensor or None.
    # A Python tuple will cause recompilations.
    img_shape: Tensor,
    border: int = 0,
) -> tuple[Tensor, Tensor]:  # MODIFICATION: Always return a tuple
    """project 3D points to 2D using the provided intrinsics matrix K.
    Args:
        xyz: the 3D points
            B,n,3
        K: the camera intrinsics matrix
            B,3,3
        img_shape: if provided, set to nan the points that map out of the image and additionally return mask_outside
        border: if img_shape is provided, set to nan the points that map out of the image border
    Returns
        xy_proj: the 2D projection of the 3D points
            B,n,2
        mask_outside: optional (if img_shape is provided). True where the point map outside img_shape
            B,n bool
    """
    original_dtype = xyz.dtype
    # B,3,3 * B,3,n =  B,3,n  -> B,n,3 after permutation
    xy_proj_hom = (K @ xyz.permute(0, 2, 1)).permute(0, 2, 1)
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    # filter points outside img_shape
    xy_proj, outside_mask = filter_outside(xy_proj, img_shape, border)
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

    return xy_proj

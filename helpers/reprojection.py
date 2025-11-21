# visual test: can I reproject the points correctly?
import torch
import h5py
from tqdm.auto import tqdm


def create_grid(image, permute=False, sampling_factor=10, border=30):
    """
    Function to create a grid of the same size as the image.
    Args:
        image: image of shape BxCxHxW or CxHxW
        permute: if True, the grid is permuted
    Returns:
        grid: grid of the same size as the image HxWx2
    """
    image = image[None] if image.dim() == 3 else image
    H, W = image.shape[-2:]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(border, H - border, sampling_factor),
        torch.arange(border, W - border, sampling_factor),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).float()

    grid = grid[torch.randperm(grid.shape[0])] if permute else grid

    return grid


def dist(p0, p1):
    """
    Euclidean distance between two points
    Args:
        p0: point 0 (N,2)
        p1: point 1 (N,2)
    Returns:
        dist: distance between the points (N,)
    """
    return torch.sqrt(((p0 - p1) ** 2).sum(dim=-1))


def compute_121_reprojection(
    data,
    img0,
    img1,
    verbose=True,
    reprojection_error=3.0,
    border=30,
    sampling_factor=10,
    device="cuda",
):
    # create a grid of points in img 0
    kpts0 = create_grid(img0, sampling_factor=sampling_factor, border=border)[None].to(
        device
    )
    # starting from depth valid locations, in nan is invalid in any case
    # kpts0 = torch.nonzero(~torch.isnan(data['depth0'][0]))[None].float() # why not working?
    tot = kpts0.numel()

    # project the points to img1
    kpts1 = reproject_2D_2D(
        xy0=kpts0,
        depthmap0=data["depth0"],
        P0=data["P0"],
        P1=data["P1"],
        K0=data["K0"],
        K1=data["K1"],
        img1_shape=(img1.shape[-2], img1.shape[-1]),
    )

    # back project the points to img0
    kpts0_back = reproject_2D_2D(
        xy0=kpts1,
        depthmap0=data["depth1"],
        P0=data["P1"],
        P1=data["P0"],
        K0=data["K1"],
        K1=data["K0"],
        img1_shape=(img0.shape[-2], img0.shape[-1]),
    )

    if verbose:
        print(kpts0.shape, kpts1.shape, kpts0_back.shape, "projected")

    # detect nans and remove if any, no need for kpts0
    nan_mask = torch.logical_and(
        torch.isnan(kpts1).any(dim=-1), torch.isnan(kpts0_back)[0].any(dim=-1)
    )
    kpts0 = kpts0[~nan_mask]
    kpts1 = kpts1[~nan_mask]
    kpts0_back = kpts0_back[~nan_mask]

    if verbose:
        print(kpts0.shape, kpts1.shape, "removed nan")

    # check if back projections is close enough to the original points
    mask = dist(kpts0, kpts0_back) < reprojection_error
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]

    # check projection to be within border margin for kpt1
    mask_x = torch.logical_and(
        kpts1[:, 0] > border, kpts1[:, 0] < img1.shape[-1] - border
    )
    mask_y = torch.logical_and(
        kpts1[:, 1] > border, kpts1[:, 1] < img1.shape[-2] - border
    )
    mask = torch.logical_and(mask_x, mask_y)
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]

    return kpts0, kpts1, tot


@torch.no_grad()
def filter_viewgraph_by_reprojection(
    self,
    viewgraph,
    images,
    th=0.025,
    min_points=100,  # do I want this in my scheme?
    border=0,
    sampling_factor=10,
    reprojection_error=5.0,
    device="cuda",
    use_amp=False,
):
    """Filters viewgraph with batched reprojection."""

    grid_cache = {}

    def get_or_create_grid(h, w, sampling_factor, border, device):
        """Cache grids by (h, w, sampling_factor, border) to avoid recomputation"""
        key = (h, w, sampling_factor, border)
        if key not in grid_cache:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(border, h - border, sampling_factor, device=device),
                torch.arange(border, w - border, sampling_factor, device=device),
                indexing="ij",
            )
            grid_cache[key] = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).float()
        return grid_cache[key]

    filtered_viewgraph = []

    for i, j in tqdm(viewgraph, desc="Computing viewgraph"):
        ix1, iy1, ix2, iy2, ih, iw = [int(x) for x in images[i]["coords"]]
        jx1, jy1, jx2, jy2, jh, jw = [int(x) for x in images[j]["coords"]]

        # Convert depth to float32
        Z1 = images[i]["depth"][iy1:iy2, ix1:ix2][None].float()
        Z2 = images[j]["depth"][jy1:jy2, jx1:jx2][None].float()

        # Get cached grid instead of creating new one
        grid = get_or_create_grid(ih, iw, sampling_factor, border, device)

        # Use cached matrices
        data = {
            "P0": self.poses.get_projection_matrix(i),
            "P1": self.poses.get_projection_matrix(j),
            "K0": self.intrinsics.get_intrinsic_matrix(images[i]["cam_id"]),
            "K1": self.intrinsics.get_intrinsic_matrix(images[j]["cam_id"]),
            "depth0": Z1,
            "depth1": Z2,
        }

        with torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=use_amp
        ):
            # project the points to img1
            kpts1 = reproject_2D_2D(
                xy0=grid[None],
                depthmap0=data["depth0"],
                P0=data["P0"],
                P1=data["P1"],
                K0=data["K0"],
                K1=data["K1"],
                img1_shape=(jh, jw),
            )

            # back project the points to img0
            kpts0_back = reproject_2D_2D(
                xy0=kpts1,
                depthmap0=data["depth1"],
                P0=data["P1"],
                P1=data["P0"],
                K0=data["K1"],
                K1=data["K0"],
                img1_shape=(ih, iw),
            )

        # ================================================================
        # AOT it's not clear to me why this works...
        # kpt1 shouldn't be checked, only the reprojected points k0 should count
        #
        nan_mask = torch.logical_or(
            torch.isnan(kpts1).any(dim=-1), torch.isnan(kpts0_back).any(dim=-1)
        )
        kpts0_valid = grid[~nan_mask.squeeze()]
        kpts1_valid = kpts1[~nan_mask]
        kpts0_back_valid = kpts0_back[~nan_mask]

        # Check reprojection consistency
        reprojection_dist = torch.sqrt(
            ((kpts0_valid - kpts0_back_valid) ** 2).sum(dim=-1)
        )
        consistent_mask = reprojection_dist < reprojection_error

        kpts0_consistent = kpts0_valid[consistent_mask]

        # Check border constraints
        if kpts0_consistent.numel() > 0:
            mask_x = torch.logical_and(
                kpts1_valid[:, 0] > border, kpts1_valid[:, 0] < jw - border
            )
            mask_y = torch.logical_and(
                kpts1_valid[:, 1] > border, kpts1_valid[:, 1] < jh - border
            )
            mask = torch.logical_and(mask_x, mask_y)
            kpt0 = kpts0_valid[mask]
        else:
            kpt0 = kpts0_consistent

        tot_kpts = grid.shape[0]
        if tot_kpts > 0:
            perc = len(kpt0) / tot_kpts
            if perc >= th or len(kpt0) >= min_points:
                filtered_viewgraph.append((i, j))

    print(f"Filtered viewgraph: {len(filtered_viewgraph):,} pairs retained")
    return filtered_viewgraph


#####
import torch as th
import numpy as np
import torch.nn.functional as F
from torch import Tensor


def grid_sample_nan(xy: Tensor, img: Tensor, mode="nearest") -> tuple[Tensor, Tensor]:
    """pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
    the output will be nan). Works both with input with shape B,n,2 and B,n0,n1,2
    xy point that fall outside the image are treated as nan (those which are really close are interpolated using
    border padding mode)
    Args:
        xy: input coordinates (with the convention top-left pixel center at (0.5, 0.5))
            B,n,2 or B,n0,n1,2
        img: the image where the sampling is done
            BxCxHxW or BxHxW
        mode: the interpolation mode
    Returns:
        sampled: the sampled values
            BxCxN or BxCxN0xN1 (if no C dimension in input BxN or BxN0xN1)
        mask_img_nan: mask of the points that had a nan in the img. The points xy that were nan appear as false in the
            mask in the same way as point that had a valid img value. This is done to discriminate between invalid
            sampling position and valid sampling position with a nan value in the image
            BxN or BxN0xN1
    """
    assert img.dim() in {3, 4}
    if img.dim() == 3:
        # ? remove the channel dimension from the result at the end of the function
        squeeze_result = True
        img = img.unsqueeze(1)
    else:
        squeeze_result = False

    assert xy.shape[-1] == 2
    assert xy.dim() == 3 or xy.dim() == 4
    B, C, H, W = img.shape

    xy_norm = normalize_pixel_coordinates(xy, img.shape[-2:])  # BxNx2 or BxN0xN1x2
    # ? set to nan the point that fall out of the second image
    xy_norm[(xy_norm < -1) + (xy_norm > 1)] = float("nan")
    if xy.ndim == 3:
        sampled = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=False,
            mode=mode,
            padding_mode="border",
        ).view(
            B, C, xy.shape[1]
        )  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=False, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # ? points xy that are not nan and have nan img. The sum is just to squash the channel dimension
    mask_img_nan = th.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # ? set to nan the sampled values for points xy that were nan (grid_sample consider those as (-1, -1))
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    if xy.ndim == 3:
        sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    else:
        sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img = img.squeeze(1)
        sampled = sampled.squeeze(1)

    return sampled, mask_img_nan


def normalize_pixel_coordinates(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray
) -> Tensor:
    """normalize pixel coordinates from -1 to +1. Being (-1,-1) the exact top left corner of the image
    the coordinates must be given in a way that the center of pixel is at .float() coordinates (0.5,0.5)
    xy ordered as (x, y) and shape ordered as (H, W)
    Args:
        xy: input coordinates in order (x,y) with the convention top-left pixel center is at coordinates (0.5, 0.5)
            ...x2
        shape: shape of the image in the order (H, W)
    Returns:
        xy_norm: normalized coordinates between [-1, 1]
    """
    xy_norm_x = 2 * xy[..., 0] / shape[1] - 1
    xy_norm_y = 2 * xy[..., 1] / shape[0] - 1
    xy_norm = th.stack([xy_norm_x, xy_norm_y], dim=-1)
    return xy_norm


def to_homogeneous(xy: Tensor) -> Tensor:
    return th.cat((xy, th.ones_like(xy[..., 0:1])), dim=-1)


def from_homogeneous(points: Tensor) -> Tensor:
    eps = 1e-8
    z_vec = points[..., -1:]
    # set the results of division by zero/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = th.abs(z_vec) > eps
    scale = th.where(mask, 1.0 / (z_vec + eps), th.ones_like(z_vec))
    output = scale * points[..., :-1]
    return output


def unproject_to_virtual_plane(
    xy: Tensor,
    K: Tensor,  # cast_to_double: bool = True
) -> Tensor:
    """unproject points to the camera virtual plane at depth 1
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns:
        xyz: 3D points laying on the virtual plane
            B,n,3
    """
    xy_hom = to_homogeneous(xy)  # B,n,3
    # if cast_to_double:
    original_type = xy.dtype
    # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
    xyz = (
        (th.linalg.inv(K.to(th.double)) @ (xy_hom.permute(0, 2, 1).to(th.double)))
        .permute(0, 2, 1)
        .to(original_type)
    )
    # else:
    # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
    # xyz = (th.inverse(K) @ (xy_hom.permute(0, 2, 1))).permute(0, 2, 1)

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
    assert xy.shape[0] == K.shape[0] and xy.shape[0] == depths.shape[0]
    assert (
        xy.shape[1] == depths.shape[1]
    ), f"Expected xy and depths to have the same number of points, got {xy.shape[1]} and {depths.shape[1]}"
    assert xy.shape[2] == 2

    xyz = unproject_to_virtual_plane(xy, K)  # B,n,3
    depths = depths.unsqueeze(-1)  # B,n,1
    xyz_scaled = xyz * depths  # B,n,3

    return xyz_scaled


def invert_P(P: Tensor) -> Tensor:
    """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    Raises:
        None
    """
    B = P.shape[0]
    R = P[:, 0:3, 0:3]
    t = P[:, 0:3, 3:4]
    P_inv = th.cat((R.permute(0, 2, 1), -R.permute(0, 2, 1) @ t), dim=2)
    P_inv = th.cat(
        (P_inv, P.new_tensor([[0.0, 0.0, 0.0, 1.0]])[None, ...].repeat(B, 1, 1)), dim=1
    )
    return P_inv


def change_reference_3D_points(
    xyz0: Tensor,
    P0: Tensor,
    P1: Tensor,  # cast_to_double: bool = True
) -> Tensor:
    """move 3D points from P0 to P1 reference systems
    Args:
        xyz0: the 3D points in the P0 coordinate system
            B,n,3
        P0: the source coordinate system
            B,4,4
        P1: the destination coordinate system
            B,4,4
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns
        xyz1: the 3D points in the P1 coordinate system
            B,n,3
    """
    assert (
        xyz0.shape[0] == P0.shape[0] and xyz0.shape[0] == P1.shape[0]
    ), f"Expected xyz0 and P0 to have the same batch size, got {xyz0.shape[0]} and {P0.shape[0]}"
    assert xyz0.shape[2] == 3, f"Expected xyz0 to have 3 channels, got {xyz0.shape[2]}"
    assert (
        P0.shape[1] == 4 and P0.shape[2] == 4
    ), f"Expected P0 to have shape Bx4x4, got {P0.shape}"
    assert (
        P1.shape[1] == 4 and P1.shape[2] == 4
    ), f"Expected P1 to have shape Bx4x4, got {P1.shape}"

    xyz0_hom = to_homogeneous(xyz0)  # B,n,4
    # if cast_to_double:
    original_dtype = xyz0.dtype
    P0_inv = invert_P(P0.to(th.double))
    xyz1_hom = (
        P1.to(th.double) @ P0_inv @ xyz0_hom.permute(0, 2, 1).to(th.double)
    )  # B,4,n
    xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1)).to(original_dtype)  # B,n,3
    # else:
    #     P0_inv = invert_P(P0)
    #     xyz1_hom = P1 @ P0_inv @ xyz0_hom.permute(0, 2, 1)  # B,4,n
    #     xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1))  # B,n,3

    return xyz1


def filter_outside(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray, border: int = 0
) -> Tensor:
    """set as nan all the points that are not inside rectangle defined with shape HxW
    Args:
        xy: keypoints with coordinate (x, y)
            (B)xnx2
        shape: shape where the keypoints should be contained (H, W)
            2
        border: the minimum border to apply when masking
    Returns:
        Tensor: input keypoints with 'nan' where one of the two coordinates was not contained inside shape
        xy_filtered     (B)xnx2
    """
    # assert xy.shape[-1] == 2, f"xy must have last dimension of size 2, got {xy.shape}"
    # assert len(shape) == 2, f"shape must be a tuple of 2 elements, got {shape}"
    # assert border < max(
    #     shape
    # ), f"border must be smaller than the smallest shape dimension, got {border} and {shape}"

    xy = xy.clone()
    outside_mask = (
        (xy[..., 0] < border)
        + (xy[..., 0] >= shape[1] - border)
        + (xy[..., 1] < border)
        + (xy[..., 1] >= shape[0] - border)
    )
    # Use torch.where instead of indexing assignment
    xy_filtered = th.where(outside_mask[..., None], th.full_like(xy, float("nan")), xy)
    return xy_filtered


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    img_shape: tuple[int, int],
    border: int = 0,
) -> Tensor | tuple[Tensor, Tensor]:
    """project 3D points to 2D using the provided intrinsics matrix K. If img_shape is provided, set to nan the points
    that project out of the img and additionally return mask_outside boolean tensor
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
    xy_proj_hom = (K.to(th.double) @ xyz.permute(0, 2, 1).to(th.double)).permute(
        0, 2, 1
    )
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    # if img_shape is not None:
    # ? filter points that fall outside the second image but have depth valid
    # ? as the comparison of a 'nan' values with something else is always false, only the points that had valid
    # ? depth will appear in mask_outside
    mask_outside = (
        (xy_proj[..., 0] < border)
        + (xy_proj[..., 0] >= img_shape[1] - border)
        + (xy_proj[..., 1] < border)
        + (xy_proj[..., 1] >= img_shape[0] - border)
    )
    xy_proj = filter_outside(xy_proj, img_shape, border)
    return xy_proj, mask_outside  #  Always return both


def reproject_2D_2D(
    xy0: Tensor,
    depthmap0: Tensor,
    P0: Tensor,
    P1: Tensor,
    K0: Tensor,
    K1: Tensor,
    img1_shape: tuple[int, int] | None = None,
    border: int = 0,
    mode: str = "nearest",
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """projects xy0 points from img0 to img1 using depth0. Points that have an invalid depth='nan' are
        set to 'nan' (if bilinear sampling is used, all the 4 closest depth values must be valid to get a valid projection).
        If img1_shape is provided, also the points that project out of the second image are set to Nan
    Args:
        xy0: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        depthmap0: depthmap of img0
            B,H,W or B,n
        P0: camera0 extrinsics matrix
            B,4,4
        P1: camera1 extrinsics matrix
            B,4,4
        K0: camera0 intrinsics matrix
            B,3,3
        K1: camera1 intrinsics matrix
            B,3,3
        img1_shape: shape of img1 (H, W)
        border: if > 0, the points that project closer to the image borders are set to nan
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
    Returns:
        xy0_proj: the projected keypoints in img1
            B,n,2
        mask_invalid_depth: mask of points that had invalid depth
            B,n  bool
        mask_outside: optional (if img1_shape is provided) mask of points that had valid depth but project out of the
            second image
            B,n  bool
    """
    # ? interpolate depths
    if depthmap0.dim() == 3:
        selected_depths0, mask_invalid_depth0 = grid_sample_nan(
            xy0, depthmap0, mode=mode
        )  # Bxn, Bxn
    else:
        # pre-sampled depths
        assert (
            depthmap0.shape == xy0.shape[:2]
        ), f"If depthmap0 is not BxHxW, it must be Bxn, got {depthmap0.shape} and {xy0.shape}"
        selected_depths0 = depthmap0

    # ? use the depth to define the 3D coordinates of points in the ref system of camera0
    xyz0 = unproject_to_3D(xy0, K0, selected_depths0)  # B,n,3

    # ? change the ref system of the 3d point to camera1
    xyz0_proj = change_reference_3D_points(xyz0, P0, P1)  # B,n,3

    # ? project the point in the destination image
    if img1_shape is not None:
        xy0_proj, mask_outside0 = project_to_2D(
            xyz0_proj, K1, img1_shape, border
        )  # B,n,2, B,n,2
        return xy0_proj
    else:
        assert border == 0, "border must be 0 if img1_shape is not provided"
        xy0_proj = project_to_2D(xyz0_proj, K1)  # B,n,2, B,n,2
        return xy0_proj

"""Reprojection utilities and compile-friendly geometric primitives.

This module is the single home for reprojection code: NaN-safe sampling,
viewgraph filtering, the full 1-to-1 reprojection used to seed the viewgraph,
and the tensor-only primitives used in the optimized EPO forward pass.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.dt_loss import sample_distance_field


def normalize_pixel_coordinates(xy: Tensor, shape: tuple[int, int] | Tensor) -> Tensor:
    """Normalize integer pixel indices to ``[-1, +1]`` with ``align_corners=True``
    semantics: pixel 0 maps to -1 and pixel ``size-1`` maps to +1, so the pixel
    *center* sits at integer coordinates (the same convention used by the
    Triton kernels in ``helpers/triton_ops.py`` and by ``sample_distance_field``
    in ``losses/dt_loss.py``). Edges produced by ``edges_map.nonzero()`` are
    integer indices, so callers should pass them without a ``+0.5`` offset.

    xy ordered as (x, y) and shape ordered as (H, W).

    Args:
        xy: input coordinates in order (x, y), integer pixel indices in
            ``[0, W-1] x [0, H-1]``.
            ...x2
        shape: shape of the image in the order (H, W).

    Returns:
        xy_norm: normalized coordinates between [-1, 1].
    """
    xy_norm_x = 2 * xy[..., 0] / (shape[1] - 1) - 1
    xy_norm_y = 2 * xy[..., 1] / (shape[0] - 1) - 1
    xy_norm = torch.stack([xy_norm_x, xy_norm_y], dim=-1)
    return xy_norm


def grid_sample_nan(xy: Tensor, img: Tensor, mode="nearest") -> tuple[Tensor, Tensor]:
    """Pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
    the output will be nan). Works both with input with shape B,n,2 and B,n0,n1,2
    xy point that fall outside the image are treated as nan (those which are really close are interpolated using
    border padding mode)

    Args:
        xy: input coordinates with integer-pixel-center convention (pixel 0
            center at (0, 0), pixel (W-1) center at (W-1, 0)). Matches the
            ``align_corners=True`` semantics used by ``normalize_pixel_coordinates``
            and the Triton kernels.
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
        sampled_raw = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        sampled = sampled_raw.view(B, C, xy.shape[1])  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=True, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # ? points xy that are not nan and have nan img. The sum is just to squash the channel dimension
    mask_img_nan = torch.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # ? set to nan the sampled values for points xy that were nan (grid_sample consider those as (-1, -1))
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    # if xy.ndim == 3:
    sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    # else:
    #     sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img = img.squeeze(1)
        sampled = sampled.squeeze(1)

    return sampled, mask_img_nan


def create_grid(image, permute=False, sampling_factor=10, border=0):  # noqa: D417
    """Function to create a grid of the same size as the image.

    Args:
        image: image of shape BxCxHxW or CxHxW
        permute: if True, the grid is permuted
    Returns:
        grid: grid of the same size as the image HxWx2
    """
    dtype = image.dtype
    image = image[None] if image.dim() == 3 else image
    H, W = image.shape[-2:]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(border, H - border, sampling_factor),
        torch.arange(border, W - border, sampling_factor),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).to(dtype=dtype)

    grid = grid[torch.randperm(grid.shape[0])] if permute else grid

    return grid


def compute_121_reprojection(  # noqa: D417
    data,
    img0,
    img1,
    verbose=True,
    reprojection_error=3.0,
    border=1,
    sampling_factor=10,
    device="cuda",
):
    """Reproject a regular grid of points from ``img0`` to ``img1`` and back.

    Used as a coarse cycle-consistency check when seeding the viewgraph: a
    point is kept if its round-trip (img0 → img1 → img0) error is below
    ``reprojection_error`` pixels.

    Args:
        data: Dict with the per-image extrinsics, intrinsics and depths
            indexed by image name.
        img0, img1: Image names (keys in ``data``).
        verbose: Currently unused. Kept for API compatibility.
        reprojection_error: Round-trip pixel error threshold.
        border: Pixels excluded around the image boundary.
        sampling_factor: Stride of the regular grid in img0.
        device: Torch device for the computation.

    Returns:
        Dict with the kept points and per-stage masks. See call sites in
        :mod:`epo` for the exact fields consumed.
    """
    border = max(border, 1)  # to cut out pixels at 0,0

    # create a grid of points in img 0
    kpts0 = create_grid(img0, sampling_factor=sampling_factor, border=border)[None].to(
        device, dtype=img0.dtype
    )

    tot = kpts0.shape[1]  # B, num_points, z

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

    # if verbose:
    #     print(kpts0.shape, kpts1.shape, kpts0_back.shape, "projected")

    # detect nans and remove if any, no need for kpts0
    # it should be a problem, invalid points are set to (0,0)
    nan_mask = torch.logical_and(
        torch.isnan(kpts1).any(dim=-1), torch.isnan(kpts0_back)[0].any(dim=-1)
    )
    kpts0 = kpts0[~nan_mask]
    kpts1 = kpts1[~nan_mask]
    kpts0_back = kpts0_back[~nan_mask]

    # if verbose:
    #     print(kpts0.shape, kpts1.shape, "removed nan")

    # check if back projections is close enough to the original points
    mask = torch.sqrt(((kpts0 - kpts0_back) ** 2).sum(dim=-1)) < reprojection_error
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


def change_reference_3D_points(
    xyz0: Tensor,
    P0: Tensor,
    P1: Tensor,  # cast_to_double: bool = True
) -> Tensor:
    """Move 3D points from P0 to P1 reference systems
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
    if not (xyz0.shape[0] == P0.shape[0] and xyz0.shape[0] == P1.shape[0]):
        raise AssertionError(
            f"Expected xyz0/P0 same batch, got {xyz0.shape[0]} and {P0.shape[0]}"
        )
    if xyz0.shape[2] != 3:
        raise AssertionError(f"Expected xyz0 to have 3 channels, got {xyz0.shape[2]}")
    if not (P0.shape[1] == 4 and P0.shape[2] == 4):
        raise AssertionError(f"Expected P0 shape Bx4x4, got {P0.shape}")
    if not (P1.shape[1] == 4 and P1.shape[2] == 4):
        raise AssertionError(f"Expected P1 shape Bx4x4, got {P1.shape}")

    xyz0_hom = to_homogeneous(xyz0)  # B,n,4
    # if cast_to_double:
    original_dtype = xyz0.dtype
    P0_inv = invert_P(P0.to(torch.double))
    xyz1_hom = (
        P1.to(torch.double) @ P0_inv @ xyz0_hom.permute(0, 2, 1).to(torch.double)
    )  # B,4,n
    xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1)).to(original_dtype)  # B,n,3
    # else:
    #     P0_inv = invert_P(P0)
    #     xyz1_hom = P1 @ P0_inv @ xyz0_hom.permute(0, 2, 1)  # B,4,n
    #     xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1))  # B,n,3

    return xyz1


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
    backend: str = "torch",
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Projects xy0 points from img0 to img1 using depth0. Points that have an invalid depth='nan' are
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
    if backend == "triton":
        if depthmap0.dim() != 3:
            raise ValueError("Triton reproject_2D_2D requires depthmap0=(B,H,W)")
        if mode != "nearest":
            raise ValueError("Triton reproject_2D_2D supports mode='nearest' only")
        from helpers.triton_ops import reproject_2D_2D_triton

        xy_proj = reproject_2D_2D_triton(xy0, depthmap0, P0, P1, K0, K1)
        return xy_proj

    if backend != "torch":
        raise ValueError(f"Unknown backend {backend!r}; expected 'torch' or 'triton'")

    if depthmap0.dim() == 3:
        selected_depths0, mask_invalid_depth0 = grid_sample_nan(
            xy0, depthmap0, mode=mode
        )  # Bxn, Bxn
    else:
        # pre-sampled depths
        if depthmap0.shape != xy0.shape[:2]:
            raise AssertionError(
                f"If depthmap0 is not BxHxW, it must be Bxn, got {depthmap0.shape} and {xy0.shape}"
            )
        selected_depths0 = depthmap0

    # ? use the depth to define the 3D coordinates of points in the ref system of camera0
    K0_dtype = K0.dtype
    K0_inv = invert_K(K0.float()).to(K0_dtype)  # does not support bfloat16
    xyz0 = unproject_to_3D(xy0, K0_inv, selected_depths0)  # B,n,3

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


@torch.no_grad()
def filter_viewgraph_by_reprojection_batched(
    viewgraph: list[tuple[str, str]],
    images: dict,
    intrinsics,  # CameraModule
    poses,  # PoseModule
    min_points: int = 100,
    border: int = 10,
    sampling_factor: int = 5,
    reprojection_error: float = 5.0,
    device: str = "cuda",
    batch_size: int = 512,
    verbose: bool = False,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    """Batched version of filter_viewgraph_by_reprojection.
    Processes multiple pairs in parallel for better GPU utilization.

    Note: This assumes all images have the same resolution for batching.
    """
    image_names = sorted(list(images.keys()))
    name_to_idx = {name: idx for idx, name in enumerate(image_names)}

    all_Ps = poses.get_projection_matrix(image_names)  # (N, 4, 4)
    cam_ids = [images[name]["cam_id"] for name in image_names]
    all_Ks = intrinsics.get_intrinsic_matrix(cam_ids)  # (N, 3, 3)

    # Stack ALL depths into one (N_img, H, W) tensor once, instead of
    # per-batch torch.stack on Python-list comprehensions.
    all_depths_stacked = torch.stack(
        [images[name]["depth"].to(device) for name in image_names], dim=0
    )
    # Likewise stack image shapes once.
    all_shapes = torch.tensor(
        [images[name]["image"].shape[-2:] for name in image_names],
        device=device,
        dtype=torch.long,
    )
    # Sort pairs by (idx_i, idx_j) so consecutive pairs share their source
    # image. The Triton kernel grid is (n_tiles, B=pair_idx), with pid_b on
    # the slow-varying axis — consecutive program blocks now reuse the same
    # depthmap0/K0/P0 in L2 (same cycle-8/9 locality argument that gave
    # +5-7% on the EPO inner-loop kernels). For exhaustive matchers
    # combinations() already returns this order, so the sort is a no-op
    # there; for sequential / pre-loaded viewgraphs it's a real reorder.
    viewgraph = sorted(viewgraph, key=lambda p: (name_to_idx[p[0]], name_to_idx[p[1]]))

    # Pair-index tensors for the entire viewgraph (no Python comprehension
    # in the per-batch loop).
    idx_i_all = torch.tensor(
        [name_to_idx[p[0]] for p in viewgraph], device=device, dtype=torch.long
    )
    idx_j_all = torch.tensor(
        [name_to_idx[p[1]] for p in viewgraph], device=device, dtype=torch.long
    )

    first_img_name = image_names[0]
    grid = create_grid(
        images[first_img_name]["image"], sampling_factor=sampling_factor, border=border
    ).to(device)[None]

    num_pairs = len(viewgraph)
    # On-device num_valid accumulator → single bulk .cpu() at end instead of
    # one .item() (CUDA sync) per pair.
    num_valid_all = torch.empty(num_pairs, device=device, dtype=torch.long)

    for batch_start in range(0, num_pairs, batch_size):
        batch_end = min(batch_start + batch_size, num_pairs)
        current_batch_size = batch_end - batch_start

        idx_i = idx_i_all[batch_start:batch_end]
        idx_j = idx_j_all[batch_start:batch_end]

        P0_batch = all_Ps[idx_i]
        P1_batch = all_Ps[idx_j]
        K0_batch = all_Ks[idx_i]
        K1_batch = all_Ks[idx_j]
        depths_i = all_depths_stacked[idx_i]
        depths_j = all_depths_stacked[idx_j]
        shapes_i = all_shapes[idx_i]
        shapes_j = all_shapes[idx_j]

        kpts0 = grid.expand(current_batch_size, -1, -1)  # (B, N, 2)

        kpts1 = reproject_2D_2D(
            xy0=kpts0,
            depthmap0=depths_i,
            P0=P0_batch,
            P1=P1_batch,
            K0=K0_batch,
            K1=K1_batch,
            img1_shape=shapes_j,
            border=border,
            backend="triton",
        )

        kpts0_back = reproject_2D_2D(
            xy0=kpts1,
            depthmap0=depths_j,
            P0=P1_batch,
            P1=P0_batch,
            K0=K1_batch,
            K1=K0_batch,
            img1_shape=shapes_i,
            border=border,
            backend="triton",
        )

        nan_mask = torch.isnan(kpts1).any(dim=-1) | torch.isnan(kpts0_back).any(dim=-1)

        reproj_dist = torch.sqrt(((kpts0 - kpts0_back) ** 2).sum(dim=-1))
        reproj_mask = reproj_dist < reprojection_error

        H_j = shapes_j[:, 0:1]
        W_j = shapes_j[:, 1:2]

        border_mask_x = (kpts1[..., 0] > border) & (kpts1[..., 0] < W_j - border)
        border_mask_y = (kpts1[..., 1] > border) & (kpts1[..., 1] < H_j - border)
        border_mask = border_mask_x & border_mask_y

        valid_mask = (~nan_mask) & reproj_mask & border_mask
        num_valid_all[batch_start:batch_end] = valid_mask.sum(dim=1)

    num_valid_cpu = num_valid_all.cpu().tolist()
    filtered_viewgraph = [
        viewgraph[k] for k in range(num_pairs) if num_valid_cpu[k] >= min_points
    ]
    valid_points_per_pair = {viewgraph[k]: num_valid_cpu[k] for k in range(num_pairs)}
    if verbose:
        print(
            f"Filtered viewgraph: {len(filtered_viewgraph):,}/{len(viewgraph):,} pairs retained"
        )
    return filtered_viewgraph, valid_points_per_pair


# Compile-friendly geometric primitives used in the EPO inner loop.
# Keep argument and return shapes uniform so torch.compile can fuse the hot path.


def to_homogeneous(xy: Tensor) -> Tensor:
    """Converts 2D points to homogeneous coordinates."""
    batch_shape = xy.shape[:-1]
    ones = torch.ones(*batch_shape, 1, dtype=xy.dtype, device=xy.device)
    return torch.cat((xy, ones), dim=-1)


def unproject_to_virtual_plane(
    xy: Tensor,
    K_inv: Tensor,
) -> Tensor:
    """Unproject points to the camera virtual plane at depth 1
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
    """Unproject points to 3D in the camera ref system
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


def invert_K(K: Tensor) -> Tensor:
    """Closed-form inversion of batched 3x3 intrinsic matrices.
    K is expected to have shape (B, 3, 3).
    """
    K_inv = torch.zeros_like(K)

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    K_inv[:, 0, 0] = 1.0 / fx
    K_inv[:, 1, 1] = 1.0 / fy
    K_inv[:, 0, 2] = -cx / fx
    K_inv[:, 1, 2] = -cy / fy
    K_inv[:, 2, 2] = 1.0

    return K_inv


def invert_P(P: Tensor, return_Rt=False) -> Tensor:
    """Invert the extrinsics P matrix in a more stable way
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    """
    if return_Rt:
        # Extract R and t directly from the 3x4 or 4x4 P0 matrix
        R = P[:, :3, :3]
        t = P[:, :3, 3:]

        # Mathematical inverse of SE(3) is just R^T and -R^T @ t
        R_inv = R.transpose(-2, -1)
        t_inv = -R_inv @ t
        return R_inv, t_inv

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
    xy0: Tensor,
    K0: Tensor,
    depth0: Tensor,
    P0: Tensor,
    backend: str = "torch",
) -> Tensor:
    """Unproject points to world coordinates
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        depths: the points depth
            B,n
        P: camera extrinsics matrix
            B,4,4
        backend: ``"torch"`` (default) uses the closed-form PyTorch chain
            ``invert_K → unproject_to_3D → R_inv @ xyz_cam + t_inv``.
            ``"triton"`` swaps in a fused CUDA kernel with an analytical
            backward (numerically equivalent up to fp32 noise; requires
            CUDA tensors).

    Returns:
        xyz_world: unprojected 3D points in the world reference system
            B,n,3
    """
    if backend == "triton":
        # Local import so CPU-only setups don't need triton installed.
        from helpers.triton_ops import unproject_2D_to_world_triton

        return unproject_2D_to_world_triton(xy0, K0, depth0, P0)

    if backend != "torch":
        raise ValueError(f"Unknown backend {backend!r}; expected 'torch' or 'triton'")

    # invert K and P
    K0_inv = invert_K(K0)
    R_inv, t_inv = invert_P(P0, return_Rt=True)

    # 2D -> 3D camera
    xyz_camera = unproject_to_3D(xy0, K0_inv, depth0)  # B,n,3

    # 3D camera -> world
    xyz_world = (R_inv @ xyz_camera.permute(0, 2, 1) + t_inv).permute(0, 2, 1)  # B,n,3

    return xyz_world


#### From homogeneous coordinates to 2D


def from_homogeneous(points: Tensor) -> Tensor:
    """Converts homogeneous coordinates to 2D points."""
    eps = 1e-10
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
    """Identifies points outside the image.

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
    img_shape: Tensor | tuple[int, int] | None = None,
    border: int = 0,
) -> Tensor | tuple[Tensor, Tensor]:
    """Project 3D points to 2D using the provided intrinsics matrix K."""
    original_dtype = xyz.dtype
    # B,3,3 * B,3,n =  B,3,n  -> B,n,3 after permutation
    xy_proj_hom = (K @ xyz.permute(0, 2, 1)).permute(0, 2, 1)
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    if img_shape is None:
        assert border == 0, "border must be 0 if img_shape is not provided"
        return xy_proj

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
    """Project world-frame 3D points into pixel coordinates of the target view.

    Args:
        xyz_world: ``(B, N, 3)`` 3D points in world coordinates.
        P1: ``(B, 4, 4)`` world-to-camera extrinsics of the target view.
        K1: ``(B, 3, 3)`` intrinsics of the target view.
        img1_shape: ``(2,)`` ``(H, W)`` of the target image.
        border: Pixels excluded around the image boundary.

    Returns:
        ``(uv, outside_mask)`` where ``uv`` is ``(B, N, 2)`` (with values
        zeroed for points falling outside) and ``outside_mask`` is the
        boolean mask of those invalid points.
    """
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


def project_and_sample_logic(  # noqa: D417
    xyz_world: torch.Tensor,
    K1: torch.Tensor,
    P1: torch.Tensor,
    img1_shape: torch.Tensor,
    dt_fields: torch.Tensor,
    dt_indices: torch.Tensor | None = None,
    border: int = 0,
    backend: str = "torch",
):
    """Fused operation: Projection -> 2D -> Sampling.
    Guarantees no NaNs are produced. Invalid points are zeroed out and tracked via outside_mask.

    Args:
        dt_fields: Either the *source* ``(N_img, 1, H, W)`` distance-field
            tensor (when ``dt_indices`` is provided), or a pre-gathered
            per-batch ``(B, 1, H, W)`` tensor (when ``dt_indices`` is ``None``,
            legacy path).
        dt_indices: Optional ``(B,)`` int64 — for each batch row, the image
            index in ``dt_fields``. When provided, the Triton backend reads
            ``dt_fields`` lazily and **never materialises a per-batch copy**;
            the torch backend gathers internally to feed ``F.grid_sample``.
        backend: ``"torch"`` (default) uses the PyTorch chain
            ``project_world_to_2D → sample_distance_field``. ``"triton"``
            calls a fused Triton kernel with an analytical PyTorch backward;
            it requires ``border == 0``, CUDA tensors, and assumes
            ``dt_fields`` does not require gradients (true for EPO).
    """
    if backend == "triton":
        assert border == 0, "Triton backend currently supports border=0 only"
        assert dt_indices is not None, (
            "Triton backend requires dt_indices (the (B,) image-index "
            "lookup into the source dt_fields tensor)."
        )
        # Local import keeps Triton off the import path for CPU-only setups.
        from helpers.triton_ops import project_and_sample_triton

        return project_and_sample_triton(
            xyz_world, K1, P1, dt_fields, dt_indices, img1_shape
        )

    if backend != "torch":
        raise ValueError(f"Unknown backend {backend!r}; expected 'torch' or 'triton'")

    # Torch backend: F.grid_sample needs (B, C, H, W). If we were given the
    # source + indices, materialise the per-batch view here. (The Triton
    # backend avoids this copy entirely.)
    if dt_indices is not None:
        dt_fields = dt_fields[dt_indices]

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

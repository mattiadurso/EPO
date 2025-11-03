# visual test: can I reproject the points correctly?
import torch
import h5py
from tqdm.auto import tqdm

from mylib.from_eslibutils import reproject_2D_2D
from mylib.conversions import to_torch


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


def filter_viewgraph_by_reprojection_old(
    viewgraph,
    images,
    intrinsics,
    th=0.025,
    min_points=100,
    border=0,
    sampling_factor=10,
    reprojection_error=3.0,
):
    """
    Filters a viewgraph by filtering the percentage of points survived to a round-trip of reprojection.
    Args:
        viewgraph: list of images names.
        images: dict[images_name] = {'image', 'coords', 'edges', 'P', 'cam_id', 'depth'}
        intrinsics: dict of camera intrinsics classes
        th: threshold percentage of points survived to reprojection (survived/total points)
        min_points: minimum number of points to consider a pair
        border: border margin to avoid points too close to the border
        sampling_factor: factor to downsample the grid of points to reproject
        reprojection_error: maximum reprojection error to consider a point as survived
    Returns:
    """
    # print(
    #     f"Filtering with threshold: {th}, reprojection error: {reprojection_error}, border: {border}, sampling factor: {sampling_factor}"
    # )

    filtered_viewgraph = []
    for i, j in tqdm(viewgraph, desc="Filtering viewgraph by reprojection"):
        ix1, iy1, ix2, iy2, ih, iw = [int(x) for x in images[i]["coords"]]
        jx1, jy1, jx2, jy2, jh, jw = [int(x) for x in images[j]["coords"]]

        Z1 = images[i]["depth"][iy1:iy2, ix1:ix2][None]
        Z2 = images[j]["depth"][jy1:jy2, jx1:jx2][None]

        data = {
            "P0": images[i]["P"].projection_matrix()[None],
            "P1": images[j]["P"].projection_matrix()[None],
            "K0": intrinsics[images[i]["cam_id"]].intrinsic_matrix()[None],
            "K1": intrinsics[images[j]["cam_id"]].intrinsic_matrix()[None],
            "depth0": Z1,
            "depth1": Z2,
        }

        kpt0, kpt1, tot_kpts = compute_121_reprojection(
            data,
            images[i]["image"],
            images[j]["image"],
            reprojection_error=reprojection_error,
            border=border,
            sampling_factor=sampling_factor,
            verbose=False,
        )

        perc = len(kpt0) / tot_kpts

        if perc >= th or len(kpt0) >= min_points:
            filtered_viewgraph.append((i, j))

    return filtered_viewgraph


@torch.no_grad()
def filter_viewgraph_by_reprojection(
    viewgraph,
    images,
    intrinsics,
    th=0.025,
    min_points=100,
    border=0,
    sampling_factor=10,
    reprojection_error=3.0,
    device="cuda",
):
    """
    Filters a viewgraph by filtering the percentage of points survived to a round-trip of reprojection.
    """

    # Pre-cache all projection matrices and intrinsics in float16 (MAJOR speedup)
    cached_P = {
        name: img_data["P"].projection_matrix().half()
        for name, img_data in images.items()
    }
    cached_K = {
        cam_id: cam.intrinsic_matrix().half() for cam_id, cam in intrinsics.items()
    }

    filtered_viewgraph = []

    for i, j in tqdm(viewgraph, desc="Filtering viewgraph"):
        ix1, iy1, ix2, iy2, ih, iw = [int(x) for x in images[i]["coords"]]
        jx1, jy1, jx2, jy2, jh, jw = [int(x) for x in images[j]["coords"]]

        # Convert depth to float16
        Z1 = images[i]["depth"][iy1:iy2, ix1:ix2][None].half()
        Z2 = images[j]["depth"][jy1:jy2, jx1:jx2][None].half()

        # Use cached matrices (already in float16)
        data = {
            "P0": cached_P[i][None],
            "P1": cached_P[j][None],
            "K0": cached_K[images[i]["cam_id"]][None],
            "K1": cached_K[images[j]["cam_id"]][None],
            "depth0": Z1,
            "depth1": Z2,
        }

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            kpt0, _, tot_kpts = compute_121_reprojection(
                data,
                images[i]["image"],
                images[j]["image"],
                reprojection_error=reprojection_error,
                border=border,
                sampling_factor=sampling_factor,
                verbose=False,
                device=device,
            )

        if tot_kpts > 0:  # Avoid division by zero
            perc = len(kpt0) / tot_kpts
            if perc >= th or len(kpt0) >= min_points:
                filtered_viewgraph.append((i, j))

    return filtered_viewgraph

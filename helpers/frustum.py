import torch

from pathlib import Path
from tqdm.auto import tqdm

from .load import load_reconstruction


def compute_frustum_corners(K, width, height, z_near, z_far, R, t, device):
    """Compute 8 frustum corners in world coordinates."""
    dtype = K.dtype
    corners_px = torch.tensor(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=dtype,
        device=device,
    )

    invK = torch.linalg.inv(K.float()).to(dtype=dtype)
    near_pts = (invK @ corners_px.T).T * z_near
    far_pts = (invK @ corners_px.T).T * z_far
    pts_cam = torch.cat([near_pts, far_pts], dim=0)
    # Xw = R^T (Xc - t)
    Xw = (R.T @ (pts_cam.T - t.reshape(3, 1))).T
    return Xw


def aabb_from_points(points):
    """Compute axis-aligned bounding box."""
    return points.min(dim=0).values, points.max(dim=0).values


def aabb_overlap(a_min, a_max, b_min, b_max):
    """Check if AABBs intersect."""
    return torch.all(a_min <= b_max) and torch.all(b_min <= a_max)


def build_view_graph_from_frustums(
    recon,  # or recon_path
    z_near_default=0.1,
    z_far_default=5.0,
    max_view_angle_deg=30.0,
    distance_factor=2,
    verbose=True,
    images_with_depth=None,
    dtype=torch.float32,
):
    """
    Compute view-graph image pairs by frustum intersection,
    with tighter geometric filtering. On average finds 90% (median is 95%) of the original COLMAP pairs,
    with at least 30 geometric inliers. Measured on mydataset (90 scenes).

    Args:
        recon_path: path to COLMAP reconstruction folder
        device: torch device
        z_near_default: default near plane distance
        z_far_default: default far plane distance
        max_view_angle_deg: maximum allowed view-direction angle difference between cameras. (To reduce pairs: lower the value (e.g., 20°)
        distance_factor: maximum allowed distance between camera centers as a factor of scene size (To reduce pairs: set to 1.0-1.5, to increase pairs: set to 3.0-4.0.)
    """

    recon, cams, imgs, id_to_name, recon_path = load_reconstruction(recon)
    device = "cpu"

    frustums = {}
    aabbs = {}
    centers = {}
    directions = {}

    if verbose is True:
        print("Building camera frustums and computing metadata...")
        bar = tqdm(imgs.values())
    else:
        bar = imgs.values()

    v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    for img in bar:
        cam = cams[img.camera_id]
        K = torch.tensor(cam.calibration_matrix(), device=device, dtype=dtype)
        R = torch.tensor(
            img.cam_from_world.rotation.matrix(), device=device, dtype=dtype
        )
        t = torch.tensor(img.cam_from_world.translation, device=device, dtype=dtype)

        if images_with_depth is not None and img.name in images_with_depth:
            depth = images_with_depth[img.name]["depth"]
            # Filter out NaN and invalid depth values
            valid_depth = depth[torch.isfinite(depth) & (depth > 0)]
            if valid_depth.numel() > 0:
                z_near = valid_depth.min().detach().cpu().item()
                z_far = valid_depth.max().detach().cpu().item()
            else:
                z_near, z_far = z_near_default, z_far_default
        else:
            z_near, z_far = z_near_default, z_far_default
        # shrink far plane slightly (avoid wide skinny cones)
        z_far *= 0.9

        corners = compute_frustum_corners(
            K, cam.width, cam.height, z_near, z_far, R, t, device
        )
        aabbs[img.image_id] = aabb_from_points(corners)
        frustums[img.image_id] = corners

        # camera center in world = -R^T t
        c_world = -(R.T @ t)
        centers[img.image_id] = c_world
        # camera forward vector in world
        d_world = R.T @ v
        directions[img.image_id] = d_world / torch.norm(d_world)

    ids = list(imgs.keys())
    pairs = []

    cos_angle_thresh = torch.cos(
        torch.deg2rad(torch.tensor(max_view_angle_deg, device=device))
    )

    if verbose:
        print(f"\nChecking {len(ids)} cameras for tight frustum overlaps...")

    for i_idx, i in enumerate(ids):
        a_min, a_max = aabbs[i]
        ci, di = centers[i], directions[i]
        for j in ids[i_idx + 1 :]:
            b_min, b_max = aabbs[j]
            cj, dj = centers[j], directions[j]

            # Step 1: AABB intersection (coarse)
            if not aabb_overlap(a_min, a_max, b_min, b_max):
                continue

            # Step 2: view direction consistency
            cos_angle = torch.dot(di, dj)
            if cos_angle < cos_angle_thresh:
                continue  # too divergent (e.g., opposite sides)

            # Step 3: distance filter
            dist = torch.norm(ci - cj)
            scene_scale = torch.norm(a_max - a_min)
            if dist > distance_factor * scene_scale:
                continue

            pairs.append([i, j])

    if verbose:
        print(f"\nFound {len(pairs):,} tight overlapping pairs.")

    # Link image names
    out_pairs = []
    for i, j in pairs:
        sorted_ij = sorted([id_to_name[i], id_to_name[j]])
        out_pairs.append([sorted_ij[0], sorted_ij[1]])

    # sort pairs by first image name and then second image name (colmap convention)
    out_pairs = sorted(out_pairs, key=lambda x: (x[0], x[1]))

    if recon_path is not None:
        with open(Path(recon_path) / "pairs.txt", "w") as f:
            for i, j in out_pairs:
                f.write(f"{i} {j}\n")

    return out_pairs

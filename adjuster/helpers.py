# Helpers
import os
import glob
from typing import List
import torch
import pycolmap
import numpy as np
from tqdm import tqdm
from pathlib import Path

from PIL import Image
from torchvision import transforms as TF
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import torch.nn.functional as F


### Loading and preprocessing images and depths
def find_images(images_path: str) -> List[str]:
    """
    Find all images in the given path, including subdirectories.

    Args:
        images_path: Path to directory containing images.

    Returns:
        List of image file paths.
    """
    valid_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    image_paths = []

    for ext in valid_extensions:
        # Search in root and one level deep
        image_paths.extend(glob.glob(os.path.join(images_path, f"*.{ext}")))
        image_paths.extend(glob.glob(os.path.join(images_path, "*", f"*.{ext}")))

    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        raise ValueError(
            f"No images found in {images_path}. Path {images_path} is invalid or empty."
        )

    print(f"Found {len(image_paths)} images in {images_path}")
    return image_paths


def _process_single_image(image_path, images_path, target_size, to_tensor):
    """Helper function to process a single image."""
    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    width, height = img.size

    # Make the image square by padding the shorter dimension
    max_dim = max(width, height)

    # Calculate padding
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2

    # Calculate scale factor for resizing
    scale = target_size / max_dim

    # Calculate final coordinates of original image in target space
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale

    # Store original image coordinates and scale
    coords = np.array([x1, y1, x2, y2, width, height])

    # Create a new black square image and paste original
    square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square_img.paste(img, (left, top))

    # Resize to target size
    square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

    # Convert to tensor
    img_tensor = to_tensor(square_img)

    # Get image relative path wrt images_path
    image_name = Path(image_path).relative_to(images_path).as_posix()

    return image_name, img_tensor, coords


def load_and_preprocess_images_square(
    image_path_list, images_path, target_size=1024, max_workers=20
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Returns a dictionary mapping image names to tensors.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        max_workers (int, optional): Maximum number of threads for parallel processing.
                                    Defaults to None (uses default ThreadPoolExecutor behavior).

    Returns:
        dict: Dictionary mapping image names (str) to image tensors (torch.Tensor)
              with shape (3, target_size, target_size)

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    to_tensor = TF.ToTensor()
    images_dict = {}

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                _process_single_image, img_path, images_path, target_size, to_tensor
            )
            for img_path in image_path_list
        ]

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading images"
        ):
            image_name, img_tensor, coords = future.result()
            images_dict[image_name] = {"image": img_tensor, "coords": coords}

    return images_dict


def _process_single_depth(depth_path, image_name, image_info, target_size):
    """Helper function to process a single depth map."""
    # Load depth from h5 file
    depth_file = Path(depth_path) / (image_name.split(".")[0] + ".h5")

    if not depth_file.exists():
        return image_name, None

    depth = h5py.File(depth_file, "r")["depth"][()]

    # Convert to tensor and add batch+channel dimensions
    depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Get original dimensions
    h, w = depth_tensor.shape[-2:]

    # Pad to square (same logic as image loading)
    max_dim = max(h, w)

    pad_h = max_dim - h
    pad_w = max_dim - w

    # Symmetric pad: (left, right, top, bottom)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    # Use NaN for invalid depth padding
    depth_tensor = F.pad(depth_tensor, pad, mode="constant", value=float("nan"))

    # Resize to target size
    depth_tensor = F.interpolate(
        depth_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    # Remove batch and channel dimensions
    depth_tensor = depth_tensor.squeeze()

    return image_name, depth_tensor


def load_and_preprocess_depths_square(
    depth_path,
    images_dict,
    target_size=518,
    max_workers=20,
):
    """
    Load and preprocess depth maps by center padding to square and resizing to target size.
    Updates images_dict with depth information.

    Args:
        depth_path (str): Path to directory containing depth maps (.h5 files)
        images_dict (dict): Dictionary mapping image names to image data
        target_size (int, optional): Target size for both width and height. Defaults to 518.
        max_workers (int, optional): Maximum number of threads for parallel processing.

    Returns:
        dict: Updated images_dict with depth maps added
    """
    depth_path = Path(depth_path)

    if not depth_path.exists():
        print(
            f"Warning: Depth path {depth_path} does not exist. Skipping depth loading."
        )
        return images_dict

    # Process depths in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                _process_single_depth,
                depth_path,
                image_name,
                images_dict[image_name],
                target_size,
            )
            for image_name in images_dict.keys()
        ]

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading depth maps"
        ):
            image_name, depth_tensor = future.result()

            if depth_tensor is not None:
                images_dict[image_name].update({"depth": depth_tensor})
            else:
                print(f"Warning: Depth map not found for {image_name}")

    return images_dict


def load_reconstruction(recon_path):
    """Load COLMAP reconstruction."""
    if isinstance(recon_path, str):
        recon = pycolmap.Reconstruction(recon_path)
    else:
        recon = recon_path  # already loaded

    cams = recon.cameras
    imgs = recon.images
    id_to_name = {img.image_id: img.name for img in imgs.values()}
    return recon, cams, imgs, id_to_name, recon_path


### View graph from frustum overlaps
def compute_frustum_corners(K, width, height, z_near, z_far, R, t, device):
    """Compute 8 frustum corners in world coordinates."""
    corners_px = torch.tensor(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=torch.float32,
        device=device,
    )

    invK = torch.inverse(K)
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
    max_view_angle_deg=75.0,
    distance_factor=2,
    verbose=True,
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

    for img in bar:
        cam = cams[img.camera_id]
        K = torch.tensor(cam.calibration_matrix(), dtype=torch.float32, device=device)
        R = torch.tensor(
            img.cam_from_world.rotation.matrix(), dtype=torch.float32, device=device
        )
        t = torch.tensor(
            img.cam_from_world.translation, dtype=torch.float32, device=device
        )

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
        d_world = R.T @ torch.tensor([0.0, 0.0, 1.0], device=device)
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

    return recon, out_pairs

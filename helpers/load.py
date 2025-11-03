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


def _process_single_image(
    image_path, images_path, target_size, to_tensor, load_with_pad
):
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

    if load_with_pad:

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
        square_img = square_img.resize(
            (target_size, target_size), Image.Resampling.BICUBIC
        )

    else:
        # resize such that longest edge is target_size
        if width >= height:
            scale = target_size / width
        else:
            scale = target_size / height

        square_img = img.resize(
            (int(width * scale), int(height * scale)), Image.Resampling.BICUBIC
        )
        coords = np.array([0, 0, width * scale, height * scale, width, height])

    # Convert to tensor
    img_tensor = to_tensor(square_img)

    # Get image relative path wrt images_path
    image_name = Path(image_path).relative_to(images_path).as_posix()

    return image_name, img_tensor, coords, scale


def load_and_preprocess_images(
    image_path_list,
    images_path,
    target_size=1024,
    max_workers=20,
    load_with_pad=True,
    device="cuda",
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Returns a dictionary mapping image names to tensors.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        max_workers (int, optional): Maximum number of threads for parallel processing.
                                    Defaults to None (uses default ThreadPoolExecutor behavior).
        load_with_pad (bool, optional): If True, images are resized to square with padding. Depth maps are resized accordingly.
                                        Use this if images might have different aspect ratios.

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
                _process_single_image,
                img_path,
                images_path,
                target_size,
                to_tensor,
                load_with_pad,
            )
            for img_path in image_path_list
        ]

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading images"
        ):
            image_name, img_tensor, coords, scale = future.result()
            images_dict[image_name] = {
                "image": img_tensor.to(device),
                "coords": torch.from_numpy(coords).to(device),
                "scale": scale,
            }

    return images_dict


def _process_single_depth(
    depth_path, image_name, image_info, target_size, load_with_pad
):
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

    if load_with_pad:
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

    else:
        # Resize such that longest edge is target_size. It might be square if from vggt.
        # cut
        x1, y1, x2, y2, _, _ = [int(_) for _ in image_info["coords"]]

        # estimate pad
        pad_x = (depth_tensor.shape[-1] - (x2 - x1)) // 2  # cutting
        pad_y = (depth_tensor.shape[-2] - (y2 - y1)) // 2

        # cutting
        depth_tensor = depth_tensor[
            :, :, y1 + pad_y : y2 + pad_y, x1 + pad_x : x2 + pad_x
        ]

        h, w = depth_tensor.shape[-2:]
        if w >= h:
            scale = target_size / w
        else:
            scale = target_size / h

        # Resize to target size
        depth_tensor = F.interpolate(
            depth_tensor,
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
            align_corners=False,
        )

    # Remove batch and channel dimensions
    depth_tensor = depth_tensor.squeeze()

    return image_name, depth_tensor


def load_and_preprocess_depths(
    depth_path,
    images_dict,
    target_size=518,
    max_workers=20,
    load_with_pad=False,
    device="cuda",
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
                load_with_pad,
            )
            for image_name in images_dict.keys()
        ]

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading depth maps"
        ):
            image_name, depth_tensor = future.result()

            if depth_tensor is not None:
                images_dict[image_name].update({"depth": depth_tensor.to(device)})
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

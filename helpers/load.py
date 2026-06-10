# Helpers
"""I/O helpers: image / depth loading and COLMAP intrinsics & pose unpacking.

All loaders return tensors already on the requested device and dtype, with
shapes matching the optional padding + resize convention used by EPO.
"""

import glob
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image  # fallback only — kept for unusual formats / RGBA blends
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import resize as tv_resize

logger = logging.getLogger(__name__)

# torchvision.io uses libjpeg-turbo + libpng under the hood — same decoders
# as cv2 — and returns CHW uint8 tensors directly (no ndarray → permute hop).
# PIL stays as the fallback for anything torchvision refuses (16-bit PNGs,
# exotic formats, RGBA we want to explicitly alpha-blend).


### Loading and preprocessing images and depths
def find_images(images_path: str, verbose: bool = False) -> list[str]:
    """Find all images in the given path, including subdirectories.

    Args:
        images_path: Path to directory containing images.
        verbose: If True, log the number of images found.

    Returns:
        Sorted list of image file paths.
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

    if verbose:
        logger.info(f"Found {len(image_paths)} images in {images_path}")
    return image_paths


def _decode_via_pil(image_path: str) -> torch.Tensor:
    """PIL fallback path: decode → RGB (alpha-composited onto white) → CHW uint8."""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    arr = np.asarray(img)  # HWC uint8 RGB
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _decode_image_rgb_chw_uint8(image_path: str) -> torch.Tensor:
    """Decode an image into a CHW RGB uint8 tensor.

    Primary path uses ``torchvision.io.read_image`` (libjpeg-turbo + libpng,
    GIL-released during decode). PIL is the fallback for anything torchvision
    refuses (16-bit PNGs, exotic formats) or for channel counts we want to
    handle explicitly (gray+alpha).
    """
    try:
        img = read_image(image_path, mode=ImageReadMode.UNCHANGED)
    except RuntimeError:
        return _decode_via_pil(image_path)

    c = img.shape[0]
    if c == 1:
        # Grayscale → RGB by repeating the channel.
        return img.expand(3, -1, -1).contiguous()
    if c == 3:
        # torchvision returns RGB (not BGR), so no channel swap needed.
        return img
    if c == 4:
        # RGBA → blend onto white, then drop alpha (matches historic PIL path).
        rgb = img[:3].float()
        a = img[3:4].float() / 255.0
        blended = rgb * a + 255.0 * (1.0 - a)
        return blended.clamp_(0, 255).to(torch.uint8)
    # 2-channel (gray+alpha) or anything else: defer to PIL.
    return _decode_via_pil(image_path)


def _process_single_image(image_path, images_path, target_size, load_with_pad):
    """Helper function to process a single image."""
    rgb = _decode_image_rgb_chw_uint8(image_path)  # CHW uint8
    height, width = rgb.shape[-2:]

    # Bicubic + antialias matches PIL's BICUBIC, which is what we ultimately
    # want parity with. Works for both up- and downsample (antialias kicks in
    # only when downsampling).
    interp = InterpolationMode.BICUBIC

    if load_with_pad:
        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale
        coords = np.array([x1, y1, x2, y2, width, height])

        # Black-padded square, then antialiased resize to target.
        square = torch.zeros((3, max_dim, max_dim), dtype=torch.uint8)
        square[:, top : top + height, left : left + width] = rgb
        square = tv_resize(
            square,
            [target_size, target_size],
            interpolation=interp,
            antialias=True,
        )
    else:
        # Resize such that longest edge is target_size, preserve aspect ratio
        if width >= height:
            scale = target_size / width
        else:
            scale = target_size / height

        new_w = int(width * scale)
        new_h = int(height * scale)
        square = tv_resize(
            rgb,
            [new_h, new_w],
            interpolation=interp,
            antialias=True,
        )
        coords = np.array([0, 0, width * scale, height * scale, width, height])

    # CHW uint8 → CHW float32 ∈ [0, 1] (same semantics as torchvision ToTensor)
    img_tensor = square.contiguous().float().div_(255.0)

    # Get image relative path wrt images_path
    image_name = Path(image_path).relative_to(images_path).as_posix()

    return image_name, img_tensor, coords, scale


def load_and_preprocess_images(
    image_path_list,
    images_path,
    target_size=1024,
    max_workers=20,
    load_with_pad=True,
    dtype=torch.float32,
    device="cuda",
):
    """Load and preprocess images by center padding to square and resizing to target size.
    Returns a dictionary mapping image names to tensors.

    Args:
        image_path_list (list): List of paths to image files
        images_path (str | Path): Root directory used to derive relative image names.
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        max_workers (int, optional): Maximum number of threads for parallel processing.
                                    Defaults to None (uses default ThreadPoolExecutor behavior).
        load_with_pad (bool, optional): If True, images are resized to square with padding. Depth maps are resized accordingly.
                                        Use this if images might have different aspect ratios.
        dtype (torch.dtype, optional): Dtype of the returned image tensors. Defaults to ``torch.float32``.
        device (str, optional): Device on which the image tensors are placed. Defaults to ``"cuda"``.

    Returns:
        dict: Dictionary mapping image names (str) to image tensors (torch.Tensor)
              with shape (3, target_size, target_size)

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

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
                load_with_pad,
            )
            for img_path in image_path_list
        ]

        for future in as_completed(futures):
            image_name, img_tensor, coords, scale = future.result()
            images_dict[image_name] = {
                "image": img_tensor.to(device, dtype=dtype),
                "coords": torch.from_numpy(coords).to(device),
                "scale": scale,
                "hw": (img_tensor.shape[-2], img_tensor.shape[-1]),
            }

    # ``as_completed`` returns futures in completion order — non-deterministic
    # across runs (depends on OS thread scheduling). Several downstream sites
    # iterate ``self.images.keys()`` and consume from a shared RNG (e.g.
    # ``_pad_edges`` does ``torch.randperm(...)`` per image, advancing the
    # generator). If dict order varies run-to-run, each image gets a different
    # permutation and the initial loss + convergence step count drift by ~1e-4
    # / ±10s of steps. Sort here so all downstream iteration is deterministic.
    return {k: images_dict[k] for k in sorted(images_dict.keys())}


def _process_single_depth(
    depth_data, image_name, image_info, target_size, load_with_pad
):
    """Helper function to process a single depth map.

    Looks the entry up in ``depth_data`` (a dict keyed by image stem), crops to
    match the original image dimensions (center crop if depth is larger), then
    resizes it to match the preprocessed image exactly.
    """
    entry = depth_data.get(image_name.split(".")[0])
    if entry is None:
        return image_name, None, None

    depth_tensor = entry["depth"]
    if not torch.is_tensor(depth_tensor):
        depth_tensor = torch.as_tensor(depth_tensor)
    # Ensure (1, 1, H, W) for F.pad / F.interpolate.
    while depth_tensor.ndim < 4:
        depth_tensor = depth_tensor.unsqueeze(0)

    confidence = entry.get("confidence")
    if confidence is not None:
        confidence_tensor = (
            confidence if torch.is_tensor(confidence) else torch.as_tensor(confidence)
        )
        while confidence_tensor.ndim < 4:
            confidence_tensor = confidence_tensor.unsqueeze(0)
    else:
        confidence_tensor = None

    # Get original image dimensions from image_info
    # coords stores [x1, y1, x2, y2, orig_width, orig_height]
    orig_w = int(image_info["coords"][4].item())
    orig_h = int(image_info["coords"][5].item())
    # scale according to target size
    scale = target_size / max(orig_w, orig_h)
    orig_w, orig_h = int(orig_w * scale), int(orig_h * scale)

    # Get depth map dimensions
    depth_h, depth_w = depth_tensor.shape[-2:]

    # If depth map is larger than the original image, center crop to original image size
    if depth_h > orig_h or depth_w > orig_w:
        crop_top = max((depth_h - orig_h) // 2, 0)
        crop_left = max((depth_w - orig_w) // 2, 0)
        crop_h = min(orig_h, depth_h)
        crop_w = min(orig_w, depth_w)

        depth_tensor = depth_tensor[
            :,
            :,
            crop_top : crop_top + crop_h,
            crop_left : crop_left + crop_w,
        ]
        if confidence_tensor is not None:
            confidence_tensor = confidence_tensor[
                :,
                :,
                crop_top : crop_top + crop_h,
                crop_left : crop_left + crop_w,
            ]
    else:
        # Depth map is smaller than (or equal to) the original image: leave it
        # as-is here; the resize step below brings it to the target size.
        pass

    # Now depth_tensor should correspond to the original image content.
    # Resize it exactly the same way the image was resized.
    depth_h, depth_w = depth_tensor.shape[-2:]

    if load_with_pad:
        # Pad to square (same logic as image loading)
        max_dim = max(depth_h, depth_w)

        pad_h = max_dim - depth_h
        pad_w = max_dim - depth_w

        # Symmetric pad: (left, right, top, bottom)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

        # Use NaN for invalid depth padding
        depth_tensor = F.pad(depth_tensor, pad, mode="constant", value=float("nan"))
        if confidence_tensor is not None:
            confidence_tensor = F.pad(
                confidence_tensor, pad, mode="constant", value=0.0
            )

        # Resize to target size (square)
        depth_tensor = F.interpolate(
            depth_tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        if confidence_tensor is not None:
            confidence_tensor = F.interpolate(
                confidence_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
    else:
        # Resize preserving aspect ratio such that longest edge is target_size
        # This matches the logic in _process_single_image
        if depth_w >= depth_h:
            scale = target_size / depth_w
        else:
            scale = target_size / depth_h

        new_h = int(depth_h * scale)
        new_w = int(depth_w * scale)

        depth_tensor = F.interpolate(
            depth_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        if confidence_tensor is not None:
            confidence_tensor = F.interpolate(
                confidence_tensor,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )

    # Remove batch and channel dimensions
    depth_tensor = depth_tensor.squeeze()
    if confidence_tensor is not None:
        confidence_tensor = confidence_tensor.squeeze()

    return image_name, depth_tensor, confidence_tensor


def load_and_preprocess_depths(
    depth_path,
    images_dict,
    target_size=518,
    max_workers=20,
    load_with_pad=False,
    dtype=torch.float32,
    device="cuda",
):
    """Load and preprocess depth maps by center padding to square and resizing to target size.
    Updates images_dict with depth information.

    Args:
        depth_path (str): Path to a ``.pth`` file (or directory containing
            ``depths.pth``) mapping image stem → ``{"depth": tensor,
            optional "confidence": tensor}``.
        images_dict (dict): Dictionary mapping image names to image data
        target_size (int, optional): Target size for both width and height. Defaults to 518.
        max_workers (int, optional): Maximum number of threads for parallel processing.
        load_with_pad (bool, optional): If True, depth maps are padded to square before
                                        being resized so they align with padded images.
        dtype (torch.dtype, optional): Dtype of the returned depth tensors. Defaults to ``torch.float32``.
        device (str, optional): Device on which the depth tensors are placed. Defaults to ``"cuda"``.

    Returns:
        dict: Updated images_dict with depth maps added
    """
    depth_path = Path(depth_path)
    if depth_path.is_dir():
        depth_file = depth_path / "depths.pth"
    else:
        depth_file = depth_path

    if not depth_file.exists():
        warnings.warn(
            f"Depth file {depth_file} does not exist. Skipping depth loading.",
            stacklevel=2,
        )
        return images_dict

    # Single read up front; downstream workers only touch the in-memory dict,
    # so no file-handle / threading concerns. Crop + resize still release the
    # GIL via PyTorch ops, so threading remains worthwhile.
    depth_data = torch.load(depth_file, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map future → image name so failures can be attributed (result()
        # raises, so the name can't be recovered from the return value).
        future_names = {
            executor.submit(
                _process_single_depth,
                depth_data,
                image_name,
                images_dict[image_name],
                target_size,
                load_with_pad,
            ): image_name
            for image_name in images_dict.keys()
        }

        for future in as_completed(future_names):
            image_name = future_names[future]
            try:
                _, depth_tensor, confidence_tensor = future.result()
            except Exception as e:
                warnings.warn(
                    f"Error processing depth for {image_name}: {e}", stacklevel=2
                )
                continue

            if depth_tensor is not None:
                images_dict[image_name].update(
                    {"depth": depth_tensor.to(device, dtype=dtype)}
                )
            else:
                warnings.warn(f"Depth map not found for {image_name}", stacklevel=2)

            if confidence_tensor is not None:
                images_dict[image_name].update(
                    {"confidence": confidence_tensor.to(device, dtype=dtype)}
                )
    return images_dict


def load_reconstruction(recon_path):
    """Load COLMAP reconstruction."""
    if isinstance(recon_path, str):
        recon = pycolmap.Reconstruction(recon_path)
        path = recon_path
    else:
        recon = recon_path  # already loaded
        path = None  # No file path available

    cams = recon.cameras
    imgs = recon.images
    id_to_name = {img.image_id: img.name for img in imgs.values()}
    return recon, cams, imgs, id_to_name, path


def process_camera(camera, load_with_pad: bool = False, images_size: int = 518):
    """Convert a ``pycolmap.Camera`` into the layout expected by EPO.

    Returns ``(cam_id, model, params)`` where ``params`` is a 1D tensor with
    intrinsics rescaled to match the resized (and optionally padded-to-square)
    image. Supports ``SIMPLE_PINHOLE`` and ``PINHOLE``.
    """
    cam_id = camera.camera_id
    model = camera.model.name
    params = camera.params
    width = camera.width
    height = camera.height

    if model == "SIMPLE_PINHOLE":  # or model == "SIMPLE_RADIAL":
        f = torch.tensor(params[0])
        cx, cy = params[1], params[2]

    elif model == "PINHOLE":  # or model == "RADIAL":
        f = torch.tensor([params[0], params[1]])
        cx, cy = params[2], params[3]

    else:
        raise NotImplementedError(f"Camera model {model} not supported.")

    # Account for padding when making square
    max_dim = max(width, height)
    pad_x = (max_dim - width) // 2 if load_with_pad else 0
    pad_y = (max_dim - height) // 2 if load_with_pad else 0

    # Scale factor after resize
    scale = images_size / max_dim

    # Apply padding shift + scale
    f = f * scale
    cx = (cx + pad_x) * scale
    cy = (cy + pad_y) * scale

    params = torch.cat([f.flatten(), torch.tensor([cx, cy])], dim=0)
    return cam_id, model, params


def process_pose(image):
    """Convert a ``pycolmap.Image`` into ``(R, t, cam_id)`` torch tensors.

    ``R`` and ``t`` are world-to-camera (COLMAP's native convention); ``t`` is
    returned with shape ``(3, 1)`` to match downstream stacking.
    """
    P = torch.tensor(image.cam_from_world().matrix())
    R = P[:3, :3]
    t = P[:3, 3].unsqueeze(1)

    return R, t, image.camera_id

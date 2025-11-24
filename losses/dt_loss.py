import torch
import torch.nn.functional as F

import torch
import cv2
import numpy as np


import torch
import cv2
import numpy as np


@torch.no_grad()
def compute_distance_field_cv2(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    """
    Compute the Euclidean distance field from edges coordinates using OpenCV.

    Args:
        edges_map: Tensor of shape (H, W). Values > 0 are treated as edges.
        device: Device to place the result on.
        dtype: Data type for the result (e.g., torch.float16).
    Returns:
        field: Distance field of shape (H, W).
    """
    # 1. Move to CPU and convert to Numpy
    # We detach just in case, though @no_grad handles most of it.
    edges_np = edges_map.detach().cpu().float().numpy()

    # 2. Prepare Binary Mask for OpenCV
    # Your logic: 1 (or >0) is an edge.
    # OpenCV logic: 0 is the target (distance 0), non-zero is the background.
    # We invert the logic: Set edges to 0, background to 1.
    mask = np.where(edges_np > 0, 0, 1).astype(np.uint8)

    # 3. Run Distance Transform
    # cv2.DIST_L2: Euclidean distance.
    # cv2.DIST_MASK_PRECISE: Calculates exact geometric distance (matches torch.cdist).
    # If you used '5' or '3', it would be an approximation and fail torch.allclose().
    field_np = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 4. Convert back to Torch
    field = torch.from_numpy(field_np)

    # 5. Match requested device and dtype
    return field.to(device=device, dtype=dtype)


# this has 1:1 correspondence with cv2 version, but is slower
@torch.no_grad()
def compute_distance_field_torch(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    # TODO: run all this in fp16. carefully normalize EVERITHING in 0-1 range
    """
    Compute the Euclidean distance field from edges coordinates.
    Args:
        edges_map: Tensor of shape (H, W) with binary edge maps.
        device: Device to use for computation.
    Returns:
        field: Distance field of shape (H, W) showing distance from each pixel to nearest edge.
    """
    # dtype = torch.float16 if normalize else dtype
    h, w = edges_map.shape[-2:]

    edges_dtype = edges_map.dtype
    edges_map = edges_map.to(device).to(dtype)
    edges = edges_map.nonzero().flip(dims=(0, 1)).to(dtype)

    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    pixel_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).to(
        dtype
    )

    # Compute distances from all pixels to target points
    pixel_dists = torch.cdist(
        pixel_coords.unsqueeze(0), edges.unsqueeze(0), p=2
    )  # (1, h*w, M)
    min_pixel_dists, _ = torch.min(pixel_dists[0], dim=1)  # (h*w,)

    # Reshape to image
    full_field = min_pixel_dists.view(h, w).to(edges_dtype)

    return full_field


def sample_distance_field(
    dt_field: torch.Tensor,
    edge_coords: torch.Tensor,
    device="cuda",
    sampling_mode="bilinear",
):
    """
    Sample the distance field at given edge coordinates.
    Preserves gradient flow through grid_sample.

    Args:
        dt_field: Tensor of shape (B, H, W) or (H, W) with the distance field.
        edge_coords: Tensor of shape (B, N, 2) or (N, 2) with edge coordinates (x, y) to sample.
        device: Device to use for computation.
        sampling_mode: Sampling mode for grid_sample ('bilinear' or 'nearest').
    Returns:
        sampled_dists: Tensor of shape (B, N) or (N,) with sampled distances at edge coordinates.
                       NaN values are preserved but masked for loss computation.
    """
    dt_field = dt_field.to(device)
    edge_coords = edge_coords.to(device)

    # Handle both batched and unbatched inputs
    if dt_field.dim() == 2:
        dt_field = dt_field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        edge_coords = edge_coords.unsqueeze(0)  # (1, N, 2)
        unbatched = True
    else:
        dt_field = dt_field.unsqueeze(1)  # (B, 1, H, W)
        unbatched = False

    b, _, h, w = dt_field.shape

    # **CRITICAL: Detect NaN coordinates BEFORE normalization**
    nan_mask_coords = torch.isnan(edge_coords).any(dim=-1)  # (B, N)

    # Replace NaN coordinates with valid placeholder (e.g., center of image)
    # This prevents NaN from entering grid_sample
    edge_coords_safe = edge_coords.clone()
    edge_coords_safe[nan_mask_coords] = torch.tensor(
        [w / 2.0, h / 2.0], device=device, dtype=edge_coords.dtype
    )

    # Normalize coordinates to [-1, 1] for grid_sample
    norm_x = (edge_coords_safe[..., 0] / (w - 1)) * 2 - 1
    norm_y = (edge_coords_safe[..., 1] / (h - 1)) * 2 - 1

    # Clamp to valid range to prevent out-of-bounds issues
    norm_x = torch.clamp(norm_x, -1.0, 1.0)
    norm_y = torch.clamp(norm_y, -1.0, 1.0)

    # Stack as [x, y] for grid_sample: (B, 1, N, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)

    # Sample without NaNs in the computation graph
    sampled_dists = torch.nn.functional.grid_sample(
        dt_field,
        grid,
        mode=sampling_mode,
        align_corners=True,
    )

    sampled_dists = sampled_dists.squeeze(1)  # (B, N)

    # **CRITICAL: Restore NaN values AFTER sampling (outside computation graph)**
    # Expand nan_mask_coords to match sampled_dists shape if needed
    if sampled_dists.dim() > nan_mask_coords.dim():
        # This shouldn't happen with squeeze(1), but being safe
        nan_mask_coords = nan_mask_coords.unsqueeze(1).expand_as(sampled_dists)

    sampled_dists = sampled_dists.clone()
    sampled_dists[nan_mask_coords] = float("nan")

    if unbatched:
        sampled_dists = sampled_dists.squeeze(0)  # (N,)

    return sampled_dists


# # Not tused for now
# # This seem to have same results, but masking with zeros instaed of nans which is better for gradients
# # @torch.compile(mode="reduce-overhead")
# def sample_distance_field(
#     dt_field: torch.Tensor,
#     edge_coords: torch.Tensor,
#     device="cuda",
#     sampling_mode="bilinear",
# ) -> tuple[torch.Tensor, torch.Tensor]:  # <--- RETURN MASK TOO
#     """
#     Sample the distance field at given edge coordinates.
#     Preserves gradient flow through grid_sample.

#     Args:
#         dt_field: Tensor of shape (B, H, W) or (H, W) with the distance field.
#         edge_coords: Tensor of shape (B, N, 2) or (N, 2) with edge coordinates (x, y) to sample.
#         device: Device to use for computation.
#         sampling_mode: Sampling mode for grid_sample ('bilinear' or 'nearest').
#     Returns:
#         sampled_dists: Tensor of shape (B, N) or (N,) with sampled distances at edge coordinates.
#         valid_mask: Tensor of shape (B, N) or (N,) boolean mask. True = valid point, False = invalid (NaN/out-of-bounds).
#     """
#     dt_field = dt_field.to(device).float()
#     edge_coords = edge_coords.to(device).float()

#     # Handle both batched and unbatched inputs
#     if dt_field.dim() == 2:
#         dt_field = dt_field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#         edge_coords = edge_coords.unsqueeze(0)  # (1, N, 2)
#         unbatched = True
#     else:
#         dt_field = dt_field.unsqueeze(1)  # (B, 1, H, W)
#         unbatched = False

#     b, _, h, w = dt_field.shape

#     # **CRITICAL: Detect NaN coordinates BEFORE normalization**
#     nan_mask_coords = torch.isnan(edge_coords).any(dim=-1)  # (B, N)
#     valid_mask = ~nan_mask_coords  # <--- Store validity ONCE

#     # Replace NaN coordinates with valid placeholder (e.g., center of image)
#     edge_coords_safe = edge_coords.clone()
#     edge_coords_safe[nan_mask_coords] = torch.tensor(
#         [w / 2.0, h / 2.0], device=device, dtype=edge_coords.dtype
#     )

#     # Normalize coordinates to [-1, 1] for grid_sample
#     norm_x = (edge_coords_safe[..., 0] / (w - 1)) * 2 - 1
#     norm_y = (edge_coords_safe[..., 1] / (h - 1)) * 2 - 1

#     # Clamp to valid range
#     norm_x = torch.clamp(norm_x, -1.0, 1.0)
#     norm_y = torch.clamp(norm_y, -1.0, 1.0)

#     grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)

#     # Sample
#     sampled_dists = torch.nn.functional.grid_sample(
#         dt_field,
#         grid,
#         mode=sampling_mode,
#         align_corners=True,
#     )

#     sampled_dists = sampled_dists.squeeze()  # (B, N)

#     if unbatched:
#         sampled_dists = sampled_dists.squeeze(0)  # (N,)
#         valid_mask = valid_mask.squeeze(0)  # (N,)

#     return sampled_dists, valid_mask  # <--- RETURN BOTH

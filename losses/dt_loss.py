import torch
import cv2
import numpy as np
import torch.nn.functional as F


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
    field = torch.from_numpy(field_np).to(device=device, dtype=dtype)

    return field


# this has 1:1 correspondence with cv2 version, but is slower since it is quadratic
@torch.no_grad()
def compute_distance_field_torch(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    """
    Compute the Euclidean distance field from edges coordinates.
    Args:
        edges_map: Tensor of shape (H, W) with binary edge maps.
        device: Device to use for computation.
    Returns:
        field: Distance field of shape (H, W) showing distance from each pixel to nearest edge.
    """
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

    pixel_dists = torch.cdist(pixel_coords.unsqueeze(0), edges.unsqueeze(0), p=2)
    min_pixel_dists, _ = torch.min(pixel_dists[0], dim=1)

    # Reshape to image
    full_field = min_pixel_dists.view(h, w).to(edges_dtype)

    return full_field


def sample_distance_field(
    dt_field: torch.Tensor,
    edge_coords: torch.Tensor,
):
    """
    Simplified version of sample_distance_field.
    Assumes input edge_coords are free of NaNs (pre-filtered).

    Args:
        dt_field: (B, C, H, W) or (B, H, W) distance fields
        edge_coords: (B, N, 2) projected coordinates, assumed safe.
    """
    B, H, W = dt_field.shape[-3], dt_field.shape[-2], dt_field.shape[-1]
    device = dt_field.device
    dtype = dt_field.dtype

    # Normalize coordinates to [-1, 1]
    x = edge_coords[..., 0]
    y = edge_coords[..., 1]

    norm_x = (x / (W - 1)) * 2 - 1
    norm_y = (y / (H - 1)) * 2 - 1

    # Stack: (B, N, 1, 2) required for grid_sample 4D input
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)

    # Grid Sample
    # Input dt_field must be (B, C, H, W)
    if dt_field.dim() == 3:
        dt_field = dt_field.unsqueeze(1)

    sampled = F.grid_sample(
        dt_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # Output Shaping
    sampled_dists = sampled.squeeze(-1).squeeze(1)  # (B, N)

    return sampled_dists.to(device=device, dtype=dtype)


# @torch.compile(mode="reduce-overhead")
def compute_chunk_loss_logic(residuals_chunk: torch.Tensor, mask_chunk: torch.Tensor):
    """
    Pure tensor logic for the loss reduction of a single chunk.
    Assumes mask_chunk correctly identifies valid entries.
    """
    # Masked Summation
    zero = torch.tensor(0.0, device=residuals_chunk.device, dtype=residuals_chunk.dtype)

    # We trust the mask_chunk. Any residual where mask is False is ignored.
    valid_values = torch.where(mask_chunk, residuals_chunk, zero)

    sum_val = valid_values.sum(dim=1)
    count_val = mask_chunk.sum(dim=1).to(torch.long)

    return sum_val, count_val

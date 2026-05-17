"""Edge-distance-field loss building blocks.

Provides the exact Euclidean distance-field implementation (Felzenszwalb-
Huttenlocher via Triton, with a lazy OpenCV fallback for non-CUDA paths or
kernel failures), a quadratic torch reference, a bilinear sampler that reads
the field at floating-point edge coordinates, and the per-chunk residual
reduction used by the EPO forward pass (per-edge clamp → Huber → per-direction
mean).
"""

import numpy as np
import torch
import torch.nn.functional as F

# Try to load the Triton fast path at import time. If Triton (or CUDA) isn't
# available, ``_distance_transform_l2_triton`` stays ``None`` and the cv2
# fallback is used unconditionally.
try:
    from helpers.triton_ops import (
        distance_transform_l2_triton as _distance_transform_l2_triton,
    )
except Exception:
    _distance_transform_l2_triton = None


@torch.no_grad()
def compute_distance_field(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    """Exact Euclidean distance field via Felzenszwalb-Huttenlocher.

    Primary path is a Triton kernel; on non-CUDA tensors, missing Triton
    install, or kernel failure it falls back to
    ``cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)``
    (cv2 is imported lazily so it's only required on the fallback path).

    Args:
        edges_map: Tensor of shape (H, W). Values > 0 are treated as edges.
        device: Device to place the result on.
        dtype: Data type for the result (e.g., torch.float16).

    Returns:
        field: Distance field of shape (H, W).
    """
    if _distance_transform_l2_triton is not None and edges_map.is_cuda:
        try:
            field = _distance_transform_l2_triton(edges_map)
            return field.to(device=device, dtype=dtype)
        except Exception:
            pass  # fall through to cv2 path

    return _compute_distance_field_cv2_fallback(edges_map, device=device, dtype=dtype)


def _compute_distance_field_cv2_fallback(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    """Lazy cv2 fallback. Only imports cv2 when actually called."""
    import cv2  # lazy import — keeps cv2 off the import graph for the fast path

    edges_np = edges_map.detach().cpu().float().numpy()
    # cv2 convention: 0 is the target (distance 0), non-zero is background.
    mask = np.where(edges_np > 0, 0, 1).astype(np.uint8)
    field_np = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return torch.from_numpy(field_np).to(device=device, dtype=dtype)


# this has 1:1 correspondence with cv2 version, but is slower since it is quadratic
@torch.no_grad()
def compute_distance_field_torch(
    edges_map: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
):
    """Compute the Euclidean distance field from edges coordinates.

    Args:
        edges_map: Tensor of shape (H, W) with binary edge maps.
        device: Device to use for computation.
        dtype: Data type for the output distance field (e.g., torch.float32).

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
) -> torch.Tensor:
    """Bilinearly sample a distance field at floating-point edge coordinates.

    Args:
        dt_field: ``(B, 1, H, W)`` distance field per image.
        edge_coords: ``(B, N, 2)`` ``(x, y)`` pixel coordinates to sample.

    Returns:
        ``(B, N)`` sampled distance values.
    """
    # Safely extract H and W from the last two dimensions of the (B, 1, H, W) tensor
    H, W = dt_field.shape[-2:]
    device = dt_field.device
    dtype = dt_field.dtype

    # Normalize coordinates to [-1, 1]
    x = edge_coords[..., 0]
    y = edge_coords[..., 1]

    norm_x = (x / (W - 1)) * 2 - 1
    norm_y = (y / (H - 1)) * 2 - 1

    # Stack: (B, N, 1, 2) required for grid_sample
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)

    # dt_field is already (B, 1, H, W), so we feed it straight in
    sampled = F.grid_sample(
        dt_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # Output is (B, 1, N, 1). Squeeze the dummy dimensions to get (B, N)
    sampled_dists = sampled.squeeze(-1).squeeze(1)

    return sampled_dists.to(device=device, dtype=dtype)


# @torch.compile(mode="reduce-overhead")
def compute_chunk_loss_logic(
    residuals_chunk: torch.Tensor,
    mask_chunk: torch.Tensor,
    clamp_max: float = 10.0,
    huber_delta: float = 1.0,
):
    """Pure tensor logic for the loss reduction of a single chunk.

    The clamp + Huber are applied per-edge sample (matching Eq. (7)-(8) of the
    paper) so that individual outlier edges cannot dominate the per-pair mean.
    The returned ``sum_val`` is therefore the sum of robustified residuals;
    dividing by ``count_val`` downstream yields the masked per-pair mean Huber.

    Args:
        residuals_chunk: (B, n) non-negative DTF distances per sampled edge.
        mask_chunk: (B, n) boolean validity mask.
        clamp_max: per-sample upper bound on the raw distance (pixels).
        huber_delta: Huber transition between quadratic and linear regimes.
    """
    zero = torch.tensor(0.0, device=residuals_chunk.device, dtype=residuals_chunk.dtype)

    # Per-sample clamp (residuals are >= 0 by construction; only the upper tail matters)
    r_clamped = residuals_chunk.clamp(min=0.0, max=clamp_max)

    # Per-sample Huber: rho(r) = 0.5*r^2 if r<=delta else delta*(r - 0.5*delta)
    rho = F.huber_loss(
        r_clamped,
        torch.zeros_like(r_clamped),
        reduction="none",
        delta=huber_delta,
    )

    # Mask out invalid samples *after* robustification so they contribute nothing.
    valid_rho = torch.where(mask_chunk, rho, zero)

    sum_val = valid_rho.sum(dim=1)
    count_val = mask_chunk.sum(dim=1).to(torch.long)

    return sum_val, count_val

import torch


@torch.no_grad()
def compute_distance_field(
    edges_map: torch.Tensor,
    device="cuda",
):
    """
    Compute the Euclidean distance field from edges coordinates.
    Args:
        edges_map: Tensor of shape (H, W) with binary edge maps.
        device: Device to use for computation.
    Returns:
        field: Distance field of shape (H, W) showing distance from each pixel to nearest edge.
    """
    edges_map = edges_map.to(device).float()
    edges = edges_map.nonzero().flip(dims=(0, 1)).float()

    h, w = edges_map.shape

    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    pixel_coords = torch.stack(
        [x_coords.flatten(), y_coords.flatten()], dim=1
    ).float()  # (h*w, 2)

    # Compute distances from all pixels to target points
    pixel_dists = torch.cdist(
        pixel_coords.unsqueeze(0), edges.unsqueeze(0), p=2
    )  # (1, h*w, M)
    min_pixel_dists, _ = torch.min(pixel_dists[0], dim=1)  # (h*w,)

    # Reshape to image
    full_field = min_pixel_dists.view(h, w)

    return full_field


def sample_distance_field(
    dt_field: torch.Tensor,
    edge_coords: torch.Tensor,
    device="cuda",
    sampling_mode="bilinear",
):
    """
    Sample the distance field at given edge coordinates.
    Args:
        dt_field: Tensor of shape (B, H, W) or (H, W) with the distance field.
        edge_coords: Tensor of shape (B, N, 2) or (N, 2) with edge coordinates (x, y) to sample.
        device: Device to use for computation.
        sampling_mode: Sampling mode for grid_sample ('bilinear' or 'nearest').
    Returns:
        sampled_dists: Tensor of shape (B, N) or (N,) with sampled distances at edge coordinates.
    """
    dt_field = dt_field.to(device).float()
    edge_coords = edge_coords.to(device).float()

    # Handle both batched and unbatched inputs
    if dt_field.dim() == 2:
        dt_field = dt_field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        edge_coords = edge_coords.unsqueeze(0)  # (1, N, 2)
        unbatched = True
    else:
        dt_field = dt_field.unsqueeze(1)  # (B, 1, H, W)
        unbatched = False

    b, _, h, w = dt_field.shape

    # Normalize coordinates to [-1, 1] for grid_sample
    norm_x = (edge_coords[..., 0] / (w - 1)) * 2 - 1
    norm_y = (edge_coords[..., 1] / (h - 1)) * 2 - 1

    # Stack as [x, y] for grid_sample: (B, N, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)  # (B, 1, N, 2)

    sampled_dists = torch.nn.functional.grid_sample(
        dt_field,
        grid,
        mode=sampling_mode,
        align_corners=True,
    )

    sampled_dists = sampled_dists.squeeze(1)  # (B, N)

    if unbatched:
        sampled_dists = sampled_dists.squeeze(0)  # (N,)

    return sampled_dists

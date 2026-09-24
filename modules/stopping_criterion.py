"""Convergence helpers for the optimization loop.

Provides per-image rotation/translation error metrics and the
:func:`evaluate_pose_changes` utility the EPO loop uses to decide when
to stop.
"""

import torch


### Pose changes ###
def evaluate_t_err(t_past, t_present, deg=True):
    """Per-sample translation-direction error via normalised inner product.

    Args:
        t_past: ``(..., 3)`` or ``(..., 3, 1)`` previous translations.
        t_present: ``(..., 3)`` or ``(..., 3, 1)`` current translations.
        deg: If True, return degrees; otherwise radians.

    Returns:
        ``(...,)`` tensor with the angle between the (normalised) translation
        directions. Translation magnitude is intentionally ignored.
    """
    eps = 1e-15

    # Handle shapes (B, 3, 1) -> (B, 3)
    if t_past.dim() > 1 and t_past.shape[-1] == 1:
        t_past = t_past.squeeze(-1)
    if t_present.dim() > 1 and t_present.shape[-1] == 1:
        t_present = t_present.squeeze(-1)

    t_past = t_past / (torch.norm(t_past, dim=-1, keepdim=True) + eps)
    t_present = t_present / (torch.norm(t_present, dim=-1, keepdim=True) + eps)

    inner = torch.sum(t_past * t_present, dim=-1)
    loss_t = torch.clamp(1.0 - inner**2, min=eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    if deg:
        err_t = torch.rad2deg(err_t)

    return err_t


def evaluate_R_err_fast(R_past, R_present, deg=True):
    """Computes rotation error directly from Rotation Matrices using the trace.
    Formula: theta = arccos( (tr(R_diff) - 1) / 2 )
    """
    # R_diff = R_past^T @ R_present
    # We want the trace of R_diff.
    # Efficiently: sum(elementwise_product(R_past, R_present))

    # This calculates trace(R_past^T @ R_present) without full matmul
    # equivalent to: torch.diagonal(torch.matmul(R_past.transpose(-1,-2), R_present), dim1=-2, dim2=-1).sum(-1)
    # simpler: sum(R_past * R_present)
    trace = torch.sum(R_past * R_present, dim=(-2, -1))

    # Numerical stability clamp (trace should be in [-1, 3] for 3x3 matrices)
    trace = torch.clamp(trace, -1.0, 3.0)

    # theta = arccos((trace - 1) / 2)
    err_rad = torch.acos((trace - 1.0) / 2.0)

    if deg:
        return torch.rad2deg(err_rad)
    return err_rad


def evaluate_center_err(R_past, R_present, t_past, t_present, deg=True):
    """Per-image camera-centre displacement, as an angle at the scene scale.

    ``atan(|c' - c| / s)`` with ``c = -R^T t`` the camera centre and ``s`` the
    median distance of the present centres from their median. Unlike the
    direction of ``t_cw`` (:func:`evaluate_t_err`) it does not depend on
    where the world origin is: a camera next to the origin (the 3DFM's
    reference view) turns its ``t_cw`` by degrees per step while barely
    moving, and on a scene of ~20 images the 0.95 quantile is that camera.

    Args:
        R_past: ``(N, 3, 3)`` previous rotations (world-to-camera).
        R_present: ``(N, 3, 3)`` current rotations.
        t_past: ``(N, 3)`` previous translations.
        t_present: ``(N, 3)`` current translations.
        deg: If True, return degrees; otherwise radians.

    Returns:
        ``(N,)`` tensor of per-image angles.
    """
    c_past = -(R_past.transpose(-1, -2) @ t_past[..., None])[..., 0]
    c_present = -(R_present.transpose(-1, -2) @ t_present[..., None])[..., 0]
    spread = c_present - c_present.median(dim=0).values
    scale = spread.norm(dim=-1).median()
    err = torch.atan((c_present - c_past).norm(dim=-1) / scale)
    return torch.rad2deg(err) if deg else err


def evaluate_pose_changes(P_past, P_present, quantile=0.95, deg=True):
    """Evaluate the rotation and translation errors between two poses.

    Args:
        P_past: Past relative pose matrix.
        P_present: Present relative pose matrix.
        quantile: Quantile of the per-image errors used as the summary value.
        deg: If True, report errors in degrees; otherwise in radians.

    Returns:
        ``(3,)`` tensor with the quantiles of the rotation, translation-
        direction and camera-centre changes (:func:`evaluate_center_err`), in
        degrees (or radians). Kept on-device — no GPU→CPU sync here; the
        caller batches the transfer of all per-step scalars into one read.
    """
    # R and t from past iteration
    R_past = P_past[:, :3, :3]
    t_past = P_past[:, :3, 3]

    # R and t from present iteration
    R_present = P_present[:, :3, :3]
    t_present = P_present[:, :3, 3]

    err_q = evaluate_R_err_fast(R_past, R_present, deg=deg)  # (N,)
    err_t = evaluate_t_err(t_past, t_present, deg=deg)  # (N,)
    err_c = evaluate_center_err(R_past, R_present, t_past, t_present, deg=deg)

    return torch.quantile(torch.stack([err_q, err_t, err_c]), quantile, dim=1)

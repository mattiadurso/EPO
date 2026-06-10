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


def evaluate_pose_changes(P_past, P_present, quantile=0.95, deg=True):
    """Evaluate the rotation and translation errors between two poses.

    Args:
        P_past: Past relative pose matrix.
        P_present: Present relative pose matrix.
        quantile: Quantile of the per-image errors used as the summary value.
        deg: If True, report errors in degrees; otherwise in radians.

    Returns:
        ``(2,)`` tensor with the rotation and translation error quantiles in
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

    return torch.quantile(torch.stack([err_q, err_t]), quantile, dim=1)

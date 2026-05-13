"""Convergence helpers for the optimization loop.

Provides per-image rotation/translation error metrics and the
:func:`evaluate_pose_changes` / :func:`evaluate_depth_changes` utilities
the EPO loop uses to decide when to stop. Quaternion / rotation-matrix
conversions follow COLMAP's convention.
"""

import torch


### Pose changes ###
def qvec2rotmat(qvec):
    """From COLMAP implementation.

    Args:
        qvec: (..., 4) tensor
    Returns:
        R: (..., 3, 3) tensor
    """
    q0 = qvec[..., 0]
    q1 = qvec[..., 1]
    q2 = qvec[..., 2]
    q3 = qvec[..., 3]

    row0 = torch.stack(
        [
            1 - 2 * q2**2 - 2 * q3**2,
            2 * q1 * q2 - 2 * q0 * q3,
            2 * q3 * q1 + 2 * q0 * q2,
        ],
        dim=-1,
    )

    row1 = torch.stack(
        [
            2 * q1 * q2 + 2 * q0 * q3,
            1 - 2 * q1**2 - 2 * q3**2,
            2 * q2 * q3 - 2 * q0 * q1,
        ],
        dim=-1,
    )

    row2 = torch.stack(
        [
            2 * q3 * q1 - 2 * q0 * q2,
            2 * q2 * q3 + 2 * q0 * q1,
            1 - 2 * q1**2 - 2 * q2**2,
        ],
        dim=-1,
    )

    return torch.stack([row0, row1, row2], dim=-2)


def rotmat2qvec(R):
    """From COLMAP implementation.

    Args:
        R: (..., 3, 3) tensor
    Returns:
        qvec: (..., 4) tensor
    """
    Rxx = R[..., 0, 0]
    Ryx = R[..., 1, 0]
    Rzx = R[..., 2, 0]
    Rxy = R[..., 0, 1]
    Ryy = R[..., 1, 1]
    Rzy = R[..., 2, 1]
    Rxz = R[..., 0, 2]
    Ryz = R[..., 1, 2]
    Rzz = R[..., 2, 2]

    zeros = torch.zeros_like(Rxx)

    # Row 0
    k00 = Rxx - Ryy - Rzz
    k01 = zeros
    k02 = zeros
    k03 = zeros

    # Row 1
    k10 = Ryx + Rxy
    k11 = Ryy - Rxx - Rzz
    k12 = zeros
    k13 = zeros

    # Row 2
    k20 = Rzx + Rxz
    k21 = Rzy + Ryz
    k22 = Rzz - Rxx - Ryy
    k23 = zeros

    # Row 3
    k30 = Ryz - Rzy
    k31 = Rzx - Rxz
    k32 = Rxy - Ryx
    k33 = Rxx + Ryy + Rzz

    row0 = torch.stack([k00, k01, k02, k03], dim=-1)
    row1 = torch.stack([k10, k11, k12, k13], dim=-1)
    row2 = torch.stack([k20, k21, k22, k23], dim=-1)
    row3 = torch.stack([k30, k31, k32, k33], dim=-1)

    K = torch.stack([row0, row1, row2, row3], dim=-2) / 3.0

    # torch.linalg.eigh returns eigenvalues in ascending order
    eigvals, eigvecs = torch.linalg.eigh(K)

    # Select eigenvector corresponding to max eigenvalue (last one)
    qvec = eigvecs[..., :, -1]

    # Reorder to match COLMAP implementation: [3, 0, 1, 2]
    qvec = qvec[..., [3, 0, 1, 2]]

    # Handle sign ambiguity
    mask = qvec[..., 0] < 0
    qvec[mask] *= -1

    return qvec


def evaluate_R_err(R_past, R_present, deg=True):
    """Per-sample relative rotation error via quaternion inner product.

    Args:
        R_past: ``(..., 3, 3)`` previous rotation matrices.
        R_present: ``(..., 3, 3)`` current rotation matrices.
        deg: If True, return degrees; otherwise radians.

    Returns:
        ``(...,)`` tensor of rotation errors (always non-negative).
    """
    eps = 1e-15

    # Make and normalize the quaternions.
    q_past = rotmat2qvec(R_past)
    q_present = rotmat2qvec(R_present)

    q_past = q_past / (torch.norm(q_past, dim=-1, keepdim=True) + eps)
    q_present = q_present / (torch.norm(q_present, dim=-1, keepdim=True) + eps)

    # Relative Rotation Angle in radians.
    inner = torch.sum(q_past * q_present, dim=-1)
    loss_q = torch.clamp(1.0 - inner**2, min=eps)
    err_q = torch.acos(1 - 2 * loss_q)

    if deg:
        err_q = torch.rad2deg(err_q)

    return err_q


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

    Returns:
        err_q: Rotation error in degrees (or radians).
        err_t: Translation error in degrees (or radians).
    """
    # R and t from past iteration
    R_past = P_past[:, :3, :3]
    t_past = P_past[:, :3, 3]

    # R and t from present iteration
    R_present = P_present[:, :3, :3]
    t_present = P_present[:, :3, 3]

    err_q = evaluate_R_err_fast(R_past, R_present, deg=deg)  # N,1
    err_t = evaluate_t_err(t_past, t_present, deg=deg)  # N,1

    qq = torch.quantile(err_q, quantile).item()
    qt = torch.quantile(err_t, quantile).item()

    return qq, qt, max(qq, qt)


### depth changes ###
def evaluate_depth_changes(depth_past, depth_present, pad_masks, quantile=0.95):
    """Evaluate the relative depth changes between two depth maps.

    Args:
        depth_past: Past depth map. (N,)
        depth_present: Present depth map. (N,)

    Returns:
        qd: Quantile of relative depth changes.
    """
    eps = 1e-15

    # I need to take care of padded parts here

    # Avoid division by zero
    depth_past = depth_past.clone()
    depth_present = depth_present.clone()
    depth_past[depth_past < eps] = eps
    depth_present[depth_present < eps] = eps

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    rel_change = torch.abs(depth_present - depth_past) / (
        (depth_present + depth_past) / 2.0 + eps
    )  # (N, edges)

    # now I need to reduce rel_change to 1 per image (N,) by keeping the quantile of valid points, valid points might differ from image to image
    qz_list = []
    for i in range(rel_change.shape[0]):
        valid_rel_change = rel_change[i][pad_masks[i]]
        if valid_rel_change.numel() == 0:
            qz_list.append(torch.tensor(0.0, device=rel_change.device))
        else:
            qz = torch.quantile(valid_rel_change, quantile)
            qz_list.append(qz)

    qz_list = torch.stack(qz_list, dim=0)  # (N,)
    return torch.quantile(qz_list, quantile).item()

"""Quaternion / AUC utilities for pose benchmarking.

Vendored from the sibling ``posebench`` repo so this codebase no longer
depends on a ``sys.path`` hack. Kept byte-for-byte equivalent to the
reference implementation — see ``benchmark_pose_fast`` for a vectorised
drop-in replacement used by ``eval_colmap_model``.
"""

import numpy as np
import torch


def is_torch(vector):
    """Return ``True`` iff ``vector`` is a ``torch.Tensor``."""
    if isinstance(vector, torch.Tensor):
        return True
    else:
        return False


def to_numpy(vector):
    """Convert a torch tensor to a numpy array (pass-through for arrays)."""
    if is_torch(vector):
        return vector.detach().cpu().numpy()
    return vector


def evaluate_R_err(R_gt, R, deg=True):
    """Relative rotation error via quaternion inner product."""
    eps = 1e-15

    # Make and normalize the quaternions.
    q = rotmat2qvec(R)
    q_gt = rotmat2qvec(R_gt)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    # Relative Rotation Angle in radians. Equivalent to acos(trace(R)*.5) with R = R_gt*R^T but more stable.
    loss_q = np.maximum(
        eps, (1.0 - np.inner(q, q_gt) ** 2)
    )  # Max to void NaNs, always > 0 due to **2.
    err_q = np.arccos(1 - 2 * loss_q)

    if deg:
        err_q = np.rad2deg(err_q)  # rad*180/np.pi

    if np.sum(np.isnan(err_q)):
        # This should never happen! Debug here
        import IPython

        IPython.embed()

    return err_q.item()


def evaluate_t_err(t_gt, t, deg=True):
    """Relative translation-direction error via normalised inner product."""
    t_gt = to_numpy(t_gt)
    t = to_numpy(t)
    # Flatten
    t = t.flatten()
    t_gt = t_gt.flatten()
    eps = 1e-15

    # Equivalent to arccos(cosine_sim(t,t_gt))
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.inner(t, t_gt) ** 2))  # Max to void NaNs
    err_t = np.arccos(np.sqrt(1 - loss_t))
    # err_t = np.arccos(np.clip(np.inner(t,t_gt), -1.0, 1.0)) # Equivalent to above

    if np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython

        IPython.embed()

    if deg:
        err_t = np.rad2deg(err_t)  # rad*180/np.pi

    return err_t.item()


def evaluate_R_t(R_gt, t_gt, R, t, deg=True):
    """Evaluate the rotation and translation errors between two poses. From IMC2020.

    Args:
        R_gt: Ground truth relative rotation matrix.
        t_gt: Ground truth relative translation vector.
        R:    Predicted relative rotation matrix.
        t:    Predicted relative translation vector.
        deg:  If True, return errors in degrees; otherwise in radians.

    Returns:
        err_q: Rotation error in radians.
        err_t: Translation error in radians.
    """
    err_q = evaluate_R_err(R_gt, R, deg=deg)
    err_t = evaluate_t_err(t_gt, t, deg=deg)

    return np.stack([err_q, err_t])


def qvec2rotmat(qvec):
    """From COLMAP implementation."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    """From COLMAP implementation."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def compute_recall(errors):
    """Compute the recall for the errors. From Pixel-Perfect SfM.

    Args:
        errors: numpy array or errors.

    Returns:
        errors: sorted errors.
        recall: recall for each.
    """
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements  # cumsum accuracy?
    return errors, recall


def compute_AUC(errors, thresholds, min_error=None):
    """Compute the AUC for one array of errors. From Pixel-Perfect SfM.

    Args:
        errors: numpy array or errors.
        thresholds: list of thresholds for the AUC computation.
        min_error: minimum error to consider.

    Returns:
        aucs: list with the AUC values for each threshold.

    Note:
        - It is computed as the defined integral of the recall over the error.
    """
    n_errors = len(errors)

    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / n_errors
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:  # [1,3,5]
        last_index = np.searchsorted(
            errors, t, side="right"
        )  # index of the first element >= t
        r = np.r_[recall[:last_index], recall[last_index - 1]]  # error < t
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t  # ?
        aucs.append(auc * 100)
    return aucs

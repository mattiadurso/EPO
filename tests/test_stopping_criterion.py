"""The phase-2 stop metric does not depend on where the world origin is (CPU).

``evaluate_center_err`` measures how far each camera centre moved, at the
scene's scale; the direction of ``t_cw`` (``evaluate_t_err``) instead turns
by degrees for a camera next to the origin that barely moves.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.stopping_criterion import (  # noqa: E402
    evaluate_center_err,
    evaluate_pose_changes,
    evaluate_t_err,
)


def _poses(n=20, seed=0):
    """Random world-to-camera poses on a ring, camera 0 at the origin."""
    g = torch.Generator().manual_seed(seed)
    angles = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
    centres = torch.stack([angles.cos(), angles.sin(), 0.1 * angles], dim=-1)
    centres = centres - centres[0]
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    w, x, y, z = q.double().unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(n, 3, 3)  # fmt: skip
    t = -(R @ centres[..., None])[..., 0]
    return R, t, centres


def _move(R, centres, step):
    """Poses after moving every centre by ``step`` (world units)."""
    moved = centres + step
    return R, -(R @ moved[..., None])[..., 0]


def test_center_err_ignores_the_world_origin():
    """Shifting the world frame leaves the centre metric unchanged."""
    R, t, c = _poses()
    step = 1e-3 * torch.randn(c.shape, generator=torch.Generator().manual_seed(1))
    _, t1 = _move(R, c, step)
    err = evaluate_center_err(R, R, t, t1)
    shift = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    Rs, t_s = _move(R, c, shift)
    _, t1_s = _move(R, c, shift + step)
    assert torch.allclose(evaluate_center_err(Rs, Rs, t_s, t1_s), err, atol=1e-10)
    # The t-direction term is not: it depends on the camera's distance to
    # the origin, and camera 0 (at the origin) swings by degrees.
    t_err = evaluate_t_err(t, t1)
    assert t_err[0] > 1.0
    assert not torch.allclose(evaluate_t_err(t_s, t1_s), t_err, atol=1e-3)


def test_center_err_scale():
    """A move of ``tan(1 deg)`` scene scales reads as 1 degree."""
    R, t, c = _poses()
    scale = (c - c.median(dim=0).values).norm(dim=-1).median()
    step = torch.zeros_like(c)
    step[:, 0] = scale * math.tan(math.radians(1.0))
    _, t1 = _move(R, c, step)
    err = evaluate_center_err(R, R, t, t1)
    assert torch.allclose(err, torch.ones_like(err), atol=1e-9)


def test_pose_changes_quantiles():
    """``evaluate_pose_changes`` returns the three quantiles, the first two
    unchanged by the added centre term."""
    R, t, c = _poses()
    _, t1 = _move(R, c, 1e-3)
    P0 = torch.eye(4, dtype=torch.float64).repeat(len(R), 1, 1)
    P1 = P0.clone()
    P0[:, :3, :3], P0[:, :3, 3] = R, t
    P1[:, :3, :3], P1[:, :3, 3] = R, t1
    out = evaluate_pose_changes(P0, P1)
    assert out.shape == (3,)
    assert torch.allclose(out[1], torch.quantile(evaluate_t_err(t, t1), 0.95))
    assert torch.allclose(
        out[2], torch.quantile(evaluate_center_err(R, R, t, t1), 0.95)
    )

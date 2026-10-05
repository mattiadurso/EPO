"""SIMPLE_RADIAL cameras on EPO's GD path (``camera_model="SIMPLE_RADIAL"``).

Checks:
  1. ``undistort_pixels`` inverts the distortion the projection applies.
  2. The projection with ``k1`` passes ``gradcheck`` (torch path, float64).
  3. Triton project+sample with ``k1`` matches the torch reference: forward,
     and the gradients w.r.t. points, focal / principal point, pose and k1.
  4. The fused loss variants with ``k1`` match the unfused chain.
  5. Unprojection with ``k1``: Triton matches torch, gradients included
     (the k1 / focal gradient flows through the undistorted pixels).
  6. ``k1 = 0`` reproduces the pinhole kernels.
  7. ``CameraModule(radial=True)`` exposes k1 and a SIMPLE_RADIAL export.
  8. ``EPO(camera_model=...)`` rejects models it cannot refine.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from helpers.reprojection import (
    project_and_sample_logic,
    project_world_to_2D,
    undistort_pixels,
    unproject_2D_to_world,
)
from losses.dt_loss import compute_chunk_loss_logic

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
K1 = (0.15, -0.1, 0.05, 0.0)  # one per batch row: barrel, pincushion, mild, none
CAM = (0, 0), (1, 1), (0, 2), (1, 2)  # the K entries a SIMPLE_RADIAL camera uses


def _scene(B=4, N=256, H=64, W=80, dtype=torch.float32, device="cuda", seed=0):
    """Random points in front of near-identity cameras, a DT field per row."""
    g = torch.Generator(device=device).manual_seed(seed)
    xyz = torch.randn(B, N, 3, device=device, dtype=dtype, generator=g)
    xyz[..., 2] = xyz[..., 2].abs() + 1.5
    xyz[..., :2] *= 0.4  # most points land inside the 80 x 64 image
    K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    K[:, 0, 0], K[:, 1, 1] = 100.0, 102.0
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2.0, H / 2.0, 1.0
    P = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)
    P[:, :3, 3] = 0.05 * torch.randn(B, 3, device=device, dtype=dtype, generator=g)
    dt = 5.0 * torch.rand(B, 1, H, W, device=device, dtype=dtype, generator=g)
    idx = torch.arange(B, device=device)
    hw = torch.tensor([H, W], device=device, dtype=torch.int32).repeat(B, 1)
    k1 = torch.tensor(K1[:B], device=device, dtype=dtype)
    return xyz, K, P, dt, idx, hw, k1


def _leaves(*ts):
    return [t.detach().clone().requires_grad_(True) for t in ts]


def _close(name, a, b, rtol=2e-3, atol=2e-3):
    scale = b.abs().max().clamp(min=1.0)
    err = (a - b).abs().max() / scale
    assert err < max(rtol, atol), f"{name}: rel err {err:.2e}"


def test_undistort_inverts_distortion():
    """Pixels -> undistorted pixels -> re-distorted pixels is the identity."""
    g = torch.Generator().manual_seed(0)
    B, N = 4, 500
    K = torch.zeros(B, 3, 3, dtype=torch.float64)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = 300, 310, 260, 190, 1
    xy = torch.rand(B, N, 2, generator=g, dtype=torch.float64) * torch.tensor(
        [518.0, 388.0], dtype=torch.float64
    )
    k1 = torch.tensor(K1, dtype=torch.float64)
    xu = undistort_pixels(xy, K, k1)
    xn = (xu - K[:, None, :2, 2]) / torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None]
    s = 1 + k1[:, None, None] * (xn * xn).sum(-1, keepdim=True)
    xd = K[:, None, :2, 2] + torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None] * xn * s
    assert (xd - xy).abs().max() < 1e-6


def test_projection_gradcheck_k1():
    """Finite differences agree with autograd through the distortion."""
    xyz, K, P, _, _, hw, k1 = _scene(B=2, N=20, dtype=torch.float64, device="cpu")
    hw = hw * 100  # keep every point inside: the test is about the smooth part
    xyz, k1 = _leaves(xyz, k1)

    def uv(xyz, k1):
        return project_world_to_2D(xyz, P, K, hw, k1=k1)[0]

    assert torch.autograd.gradcheck(uv, (xyz, k1), eps=1e-6, atol=1e-6)


@cuda
def test_project_sample_k1_matches_torch():
    """Triton forward and every gradient match the torch chain with k1."""
    from helpers.triton_ops import project_and_sample_triton

    xyz, K, P, dt, idx, hw, k1 = _scene()
    ref_in = _leaves(xyz, K, P, k1)
    tri_in = _leaves(xyz, K, P, k1)
    r_ref, m_ref = project_and_sample_logic(
        ref_in[0], ref_in[1], ref_in[2], hw, dt, dt_indices=idx, k1=ref_in[3]
    )
    r_tri, m_tri = project_and_sample_triton(
        tri_in[0], tri_in[1], tri_in[2], dt, idx, hw, k1=tri_in[3]
    )
    assert torch.equal(m_ref, m_tri), "inside masks differ"
    assert m_ref.float().mean() > 0.5, "degenerate: most points outside"
    _close("residuals", r_tri[m_tri], r_ref[m_ref], rtol=1e-4, atol=1e-4)

    up = torch.randn_like(r_ref)
    (r_ref * up).sum().backward()
    (r_tri * up).sum().backward()
    _close("grad xyz", tri_in[0].grad, ref_in[0].grad)
    for i, j in CAM:
        _close(f"grad K[{i},{j}]", tri_in[1].grad[:, i, j], ref_in[1].grad[:, i, j])
    _close("grad R|t", tri_in[2].grad[:, :3], ref_in[2].grad[:, :3])
    _close("grad k1", tri_in[3].grad, ref_in[3].grad)
    assert ref_in[3].grad[:3].abs().min() > 0, "degenerate: k1 has no gradient"


@cuda
@pytest.mark.parametrize("reduce", [False, True])
def test_fused_loss_k1_matches_unfused(reduce):
    """Fused clamp + Huber (+ row sum) with k1 equals the unfused chain."""
    from helpers.triton_ops import (
        project_and_sample_triton,
        project_sample_huber_sum_triton,
        project_sample_huber_triton,
    )

    xyz, K, P, dt, idx, hw, k1 = _scene()
    pad = torch.rand(xyz.shape[:2], device="cuda") > 0.1
    a = _leaves(xyz, K, P, k1)
    b = _leaves(xyz, K, P, k1)
    res, inside = project_and_sample_triton(a[0], a[1], a[2], dt, idx, hw, k1=a[3])
    s_ref, n_ref = compute_chunk_loss_logic(
        res, pad & inside, clamp_max=3.0, huber_delta=1.0
    )
    if reduce:
        s_tri, n_tri = project_sample_huber_sum_triton(
            b[0], b[1], b[2], dt, idx, hw, pad, clamp_max=3.0, k1=b[3]
        )
    else:
        rho, valid = project_sample_huber_triton(
            b[0], b[1], b[2], dt, idx, hw, pad, clamp_max=3.0, k1=b[3]
        )
        s_tri, n_tri = rho.sum(1), valid.sum(1)
    _close("row sums", s_tri, s_ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(n_tri.float(), n_ref.float())
    s_ref.sum().backward()
    s_tri.sum().backward()
    _close("grad xyz", b[0].grad, a[0].grad)
    _close("grad k1", b[3].grad, a[3].grad)
    for i, j in CAM:
        _close(f"grad K[{i},{j}]", b[1].grad[:, i, j], a[1].grad[:, i, j])


@cuda
def test_unproject_k1_triton_matches_torch():
    """Unprojection through undistorted pixels: values and gradients agree."""
    g = torch.Generator(device="cuda").manual_seed(1)
    B, N, H, W = 4, 300, 64, 80
    xy = torch.rand(B, N, 2, device="cuda", generator=g) * torch.tensor(
        [W - 1.0, H - 1.0], device="cuda"
    )
    depth = 1.0 + 2.0 * torch.rand(B, N, device="cuda", generator=g)
    _, K, P, _, _, _, k1 = _scene(B=B)
    ref = _leaves(depth, K, P, k1)
    tri = _leaves(depth, K, P, k1)
    x_ref = unproject_2D_to_world(xy, ref[1], ref[0], ref[2], "torch", k1=ref[3])
    x_tri = unproject_2D_to_world(xy, tri[1], tri[0], tri[2], "triton", k1=tri[3])
    _close("xyz", x_tri, x_ref, rtol=1e-5, atol=1e-5)
    up = torch.randn_like(x_ref)
    (x_ref * up).sum().backward()
    (x_tri * up).sum().backward()
    _close("grad depth", tri[0].grad, ref[0].grad)
    for i, j in CAM:
        _close(f"grad K[{i},{j}]", tri[1].grad[:, i, j], ref[1].grad[:, i, j])
    _close("grad R|t", tri[2].grad[:, :3], ref[2].grad[:, :3])
    _close("grad k1", tri[3].grad, ref[3].grad)
    assert ref[3].grad[:3].abs().min() > 0, "degenerate: k1 has no gradient"


@cuda
def test_k1_zero_matches_pinhole():
    """k1 = 0 reproduces the pinhole kernels (values, masks, gradients)."""
    from helpers.triton_ops import project_sample_huber_triton

    xyz, K, P, dt, idx, hw, _ = _scene()
    pad = torch.ones(xyz.shape[:2], device="cuda", dtype=torch.bool)
    a = _leaves(xyz)
    b = _leaves(xyz)
    rho0, v0 = project_sample_huber_triton(a[0], K, P, dt, idx, hw, pad, clamp_max=3.0)
    rho1, v1 = project_sample_huber_triton(
        b[0], K, P, dt, idx, hw, pad, clamp_max=3.0, k1=torch.zeros(4, device="cuda")
    )
    assert torch.equal(v0, v1)
    _close("rho", rho1, rho0, rtol=1e-5, atol=1e-5)
    rho0.sum().backward()
    rho1.sum().backward()
    _close("grad xyz", b[0].grad, a[0].grad, rtol=1e-4, atol=1e-4)


def test_camera_module_radial():
    """k1 is the second learnable column and exports as SIMPLE_RADIAL."""
    from modules.camera import CameraModule

    params = torch.tensor([[300.0, 259.0, 194.0], [500.0, 300.0, 200.0]])
    cam = CameraModule({"a": 0, "b": 1}, ["SIMPLE_PINHOLE"] * 2, params,
                       grad=True, device="cpu", radial=True)  # fmt: skip
    assert cam.params.shape == (2, 2)
    with torch.no_grad():
        cam.params[:, 0] = torch.tensor([0.1, 0.0])
        cam.params[:, 1] = torch.tensor([0.05, -0.02])
    model, p = cam.get_camera_parameters("a")
    assert model == "SIMPLE_RADIAL"
    assert torch.allclose(p, torch.tensor([330.0, 259.0, 194.0, 0.05]))
    assert torch.allclose(cam.get_k1(torch.tensor([1, 0])), torch.tensor([-0.02, 0.05]))
    pin = CameraModule({"a": 0, "b": 1}, ["SIMPLE_PINHOLE"] * 2, params,
                       grad=False, device="cpu")  # fmt: skip
    assert pin.get_k1(None) is None
    assert pin.get_camera_parameters("a")[0] == "SIMPLE_PINHOLE"


@cuda
def test_epo_rejects_unknown_camera_model():
    """Only SIMPLE_PINHOLE / SIMPLE_RADIAL; checked before anything is loaded."""
    from epo import EPO

    with pytest.raises(ValueError, match="camera_model"):
        EPO(reconstruction_path="/nonexistent", camera_model="OPENCV")

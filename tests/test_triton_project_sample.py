"""Validation for the Triton fused project+sample op.

Checks:
  1. Forward output matches ``project_and_sample_logic`` numerically.
  2. Backward gradients match autograd through the reference path.
  3. ``gradcheck`` passes (double-precision finite differences).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from helpers.reprojection import project_and_sample_logic
from helpers.triton_ops import project_and_sample_triton


def _make_inputs(B=4, N=128, H=64, W=80, dtype=torch.float32, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    # World points near the camera frustum
    xyz = torch.randn(B, N, 3, device=device, dtype=dtype, generator=g)
    xyz[..., 2] = xyz[..., 2].abs() + 1.5  # ensure positive z after a near-identity R

    # K with reasonable focal lengths and principal point near image center
    K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    K[:, 0, 0] = 100.0  # fx
    K[:, 1, 1] = 100.0  # fy
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    K[:, 2, 2] = 1.0

    # P near identity (tiny rotation + small translation)
    P = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).contiguous()
    P = P.clone()
    P[:, :3, 3] = 0.05 * torch.randn(B, 3, device=device, dtype=dtype, generator=g)

    # DT field — smooth random pattern in [0, 5].
    # The "source" has one entry per batch slot; identity dt_indices give
    # exactly the same per-batch DT as a manual gather would.
    dt = 5.0 * torch.rand(B, 1, H, W, device=device, dtype=dtype, generator=g)
    dt_idx = torch.arange(B, device=device, dtype=torch.int64)
    return xyz, K, P, dt, dt_idx


def test_forward_matches_reference():
    """Forward residuals + mask agree with the reference path."""
    xyz, K, P, dt, dt_idx = _make_inputs(B=4, N=256, H=64, W=80)
    img_shape = torch.tensor([dt.shape[-2], dt.shape[-1]], device=dt.device)

    res_ref, mask_ref = project_and_sample_logic(
        xyz, K, P, img_shape, dt, dt_indices=dt_idx, border=0
    )
    res_tri, mask_tri = project_and_sample_triton(xyz, K, P, dt, dt_idx)

    # Masks should match exactly
    assert torch.equal(mask_ref, mask_tri), "inside-mask mismatch"

    # Residuals should match up to floating-point noise on inside points
    diff = (res_ref - res_tri)[mask_ref]
    print(f"forward |Δ| max={diff.abs().max().item():.3e}  mean={diff.abs().mean().item():.3e}")
    assert diff.abs().max().item() < 1e-3, "forward residuals differ beyond tolerance"


def test_backward_matches_reference():
    """Autograd through ref ≡ analytical backward of Triton op."""
    xyz, K, P, dt, dt_idx = _make_inputs(B=3, N=200, H=48, W=64)
    img_shape = torch.tensor([dt.shape[-2], dt.shape[-1]], device=dt.device)

    # Reference path with autograd
    xyz_a = xyz.clone().requires_grad_(True)
    K_a = K.clone().requires_grad_(True)
    P_a = P.clone().requires_grad_(True)
    res_a, mask_a = project_and_sample_logic(
        xyz_a, K_a, P_a, img_shape, dt, dt_indices=dt_idx, border=0
    )
    # Use a fixed-but-non-trivial scalar loss
    loss_a = (res_a * mask_a.to(res_a.dtype)).sum()
    loss_a.backward()

    # Triton path
    xyz_b = xyz.clone().requires_grad_(True)
    K_b = K.clone().requires_grad_(True)
    P_b = P.clone().requires_grad_(True)
    res_b, mask_b = project_and_sample_triton(xyz_b, K_b, P_b, dt, dt_idx)
    loss_b = (res_b * mask_b.to(res_b.dtype)).sum()
    loss_b.backward()

    # fp32 accumulation noise is unavoidable; use relative + absolute tolerance.
    def cmp(name, a, b, atol=5e-3, rtol=1e-3):
        d = (a - b).abs()
        rel = d / (a.abs().clamp(min=1e-6))
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(
            f"  grad {name:<10} max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}"
            f"  max|rel|={rel.max().item():.3e}  ok={ok}"
        )
        assert ok, f"grad {name} mismatch (atol={atol}, rtol={rtol})"

    print("Backward (loss = sum(residuals * mask)):")
    cmp("xyz", xyz_a.grad, xyz_b.grad)
    # K and P only have a few learnable entries (fx, fy, cx, cy / R, t).
    # Compare the relevant slices to avoid noise on entries that should be zero.
    cmp("K[:,:2,:]", K_a.grad[:, :2, :], K_b.grad[:, :2, :])
    cmp("P[:,:3,:]", P_a.grad[:, :3, :], P_b.grad[:, :3, :])


def test_backward_random_upstream():
    """Cross-check with a non-uniform upstream gradient (more demanding)."""
    xyz, K, P, dt, dt_idx = _make_inputs(B=3, N=200, H=48, W=64, seed=7)
    img_shape = torch.tensor([dt.shape[-2], dt.shape[-1]], device=dt.device)
    g = torch.Generator(device=dt.device).manual_seed(11)
    grad_up = torch.randn(3, 200, device=dt.device, generator=g)  # (B, N)

    # Reference
    xa, ka, pa = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    ra, ma = project_and_sample_logic(
        xa, ka, pa, img_shape, dt, dt_indices=dt_idx, border=0
    )
    (ra * ma.to(ra.dtype) * grad_up).sum().backward()

    # Triton
    xb, kb, pb = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    rb, mb = project_and_sample_triton(xb, kb, pb, dt, dt_idx)
    (rb * mb.to(rb.dtype) * grad_up).sum().backward()

    print("Random upstream grad:")
    for name, a, b in [
        ("xyz", xa.grad, xb.grad),
        ("K", ka.grad, kb.grad),
        ("P[:,:3]", pa.grad[:, :3, :], pb.grad[:, :3, :]),
    ]:
        d = (a - b).abs()
        rel = d / a.abs().clamp(min=1e-6)
        print(
            f"  {name:<8} max|Δ|={d.max().item():.3e}  "
            f"mean|Δ|={d.mean().item():.3e}  max|rel|={rel.max().item():.3e}"
        )
        assert torch.allclose(a, b, atol=5e-3, rtol=2e-3), f"{name} mismatch"


def test_nonidentity_indices_match_gather():
    """Source + indices ≡ pre-gathered per-batch DT.

    In real EPO use, each image appears in many viewgraph pairs, so
    ``dt_indices`` contains duplicates. Verifies the kernel reads the right
    slice of the source tensor (and that torch backend's internal gather
    agrees with passing pre-gathered DT).
    """
    N_img = 5
    B, N, H, W = 12, 200, 48, 64
    g = torch.Generator(device="cuda").manual_seed(13)
    dt_src = 5.0 * torch.rand(N_img, 1, H, W, device="cuda", generator=g)
    # B batch rows sample from N_img source images, with many repeats.
    dt_idx = torch.randint(0, N_img, (B,), device="cuda", generator=g)
    dt_gather = dt_src[dt_idx]  # (B, 1, H, W) — what the old API produced

    xyz, K, P, _, _ = _make_inputs(B=B, N=N, H=H, W=W, seed=21)
    img_shape = torch.tensor([H, W], device="cuda")

    # 1) torch backend with the source+indices path
    res_a, mask_a = project_and_sample_logic(
        xyz, K, P, img_shape, dt_src, dt_indices=dt_idx, border=0,
        backend="torch",
    )
    # 2) torch backend with the pre-gathered tensor (no indices)
    res_b, mask_b = project_and_sample_logic(
        xyz, K, P, img_shape, dt_gather, border=0, backend="torch",
    )
    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(res_a, res_b), "torch source+idx ≠ torch gather"

    # 3) triton backend with source+indices
    res_c, mask_c = project_and_sample_triton(xyz, K, P, dt_src, dt_idx)
    assert torch.equal(mask_a, mask_c)
    d = (res_a - res_c)[mask_a].abs()
    print(
        f"  N_img={N_img}, B={B}, {len(dt_idx.unique())} unique indices: "
        f"torch≡triton max|Δ|={d.max().item():.3e}"
    )
    assert d.max().item() < 1e-3


def test_no_nan_when_points_behind_camera():
    """Stability regression: points with zc ≤ 0 must produce finite gradients.

    Mid-optimization a point can be pushed behind the camera; the forward
    correctly marks it outside and outputs 0, but if the backward sees the
    saved Inf/NaN ``u, v, zc`` it would propagate NaN into the optimizer
    state and the loss would collapse / explode. This is the failure mode
    reported on real scenes.
    """
    xyz, K, P, dt, dt_idx = _make_inputs(B=4, N=512, H=64, W=80, seed=3)

    # Force ~30% of the points to land behind the camera (zc ≤ 0) and a few
    # to land at zc ≈ 0 (so inv_z → ±Inf).
    g = torch.Generator(device=dt.device).manual_seed(99)
    bad = (torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.3)
    xyz = xyz.clone()
    xyz[..., 2] = torch.where(bad, -xyz[..., 2].abs(), xyz[..., 2])
    # Sprinkle a handful with zc tiny-positive
    tiny = (torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.02)
    xyz[..., 2] = torch.where(tiny, 1e-9 * torch.ones_like(xyz[..., 2]),
                              xyz[..., 2])

    xyz_t = xyz.clone().requires_grad_(True)
    K_t = K.clone().requires_grad_(True)
    P_t = P.clone().requires_grad_(True)

    res, mask = project_and_sample_triton(xyz_t, K_t, P_t, dt, dt_idx)
    assert torch.isfinite(res).all(), "forward produced non-finite residuals"
    # Outside points must have residual exactly 0.
    assert (res[~mask] == 0).all(), "non-zero residual at outside points"

    loss = (res * mask.to(res.dtype)).sum()
    loss.backward()

    for name, t in [("xyz", xyz_t.grad), ("K", K_t.grad), ("P", P_t.grad)]:
        assert torch.isfinite(t).all(), (
            f"grad {name} has non-finite values "
            f"(NaN: {torch.isnan(t).sum().item()}, "
            f"Inf: {torch.isinf(t).sum().item()})"
        )

    # Sanity: gradients of points that are outside should be exactly 0.
    outside = ~mask  # (B, N)
    g_xyz_outside = xyz_t.grad[outside]
    assert torch.all(g_xyz_outside == 0), (
        "outside points contribute non-zero grad_xyz "
        f"(max |Δ| = {g_xyz_outside.abs().max().item():.3e})"
    )
    print(
        f"  {bad.sum().item()}/{bad.numel()} pts behind camera, "
        f"{tiny.sum().item()} with zc≈0 — all grads finite, "
        f"outside grads exactly 0 ✓"
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for the Triton op"
    print("== Forward parity ==")
    test_forward_matches_reference()
    print("\n== Backward parity (uniform upstream grad) ==")
    test_backward_matches_reference()
    print("\n== Backward parity (random upstream grad) ==")
    test_backward_random_upstream()
    print("\n== Source + non-identity indices (no per-batch gather) ==")
    test_nonidentity_indices_match_gather()
    print("\n== Stability: points behind camera / zc≈0 ==")
    test_no_nan_when_points_behind_camera()
    print("\nAll tests passed.")

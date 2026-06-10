"""Validation for the Triton fused project+sample op.

Checks:
  1. Forward output matches ``project_and_sample_logic`` numerically.
  2. Backward gradients match autograd through the reference path.
  3. ``gradcheck`` passes (double-precision finite differences).
"""

from __future__ import annotations

import os
import sys

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
    # Uniform per-row real H/W (every row uses the full DT canvas — the
    # degenerate case where padded == real and the kernel matches the
    # pre-fix behavior).
    img_hw = (
        torch.tensor([H, W], device=device, dtype=torch.int32).expand(B, 2).contiguous()
    )
    return xyz, K, P, dt, dt_idx, img_hw


def test_forward_matches_reference():
    """Forward residuals + mask agree with the reference path."""
    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=4, N=256, H=64, W=80)

    res_ref, mask_ref = project_and_sample_logic(
        xyz, K, P, img_hw, dt, dt_indices=dt_idx, border=0
    )
    res_tri, mask_tri = project_and_sample_triton(xyz, K, P, dt, dt_idx, img_hw)

    # Masks should match exactly
    assert torch.equal(mask_ref, mask_tri), "inside-mask mismatch"

    # Residuals should match up to floating-point noise on inside points
    diff = (res_ref - res_tri)[mask_ref]
    print(
        f"forward |Δ| max={diff.abs().max().item():.3e}  mean={diff.abs().mean().item():.3e}"
    )
    assert diff.abs().max().item() < 1e-3, "forward residuals differ beyond tolerance"


def test_backward_matches_reference():
    """Autograd through ref ≡ analytical backward of Triton op."""
    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=3, N=200, H=48, W=64)

    # Reference path with autograd
    xyz_a = xyz.clone().requires_grad_(True)
    K_a = K.clone().requires_grad_(True)
    P_a = P.clone().requires_grad_(True)
    res_a, mask_a = project_and_sample_logic(
        xyz_a, K_a, P_a, img_hw, dt, dt_indices=dt_idx, border=0
    )
    # Use a fixed-but-non-trivial scalar loss
    loss_a = (res_a * mask_a.to(res_a.dtype)).sum()
    loss_a.backward()

    # Triton path
    xyz_b = xyz.clone().requires_grad_(True)
    K_b = K.clone().requires_grad_(True)
    P_b = P.clone().requires_grad_(True)
    res_b, mask_b = project_and_sample_triton(xyz_b, K_b, P_b, dt, dt_idx, img_hw)
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
    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=3, N=200, H=48, W=64, seed=7)
    g = torch.Generator(device=dt.device).manual_seed(11)
    grad_up = torch.randn(3, 200, device=dt.device, generator=g)  # (B, N)

    # Reference
    xa, ka, pa = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    ra, ma = project_and_sample_logic(
        xa, ka, pa, img_hw, dt, dt_indices=dt_idx, border=0
    )
    (ra * ma.to(ra.dtype) * grad_up).sum().backward()

    # Triton
    xb, kb, pb = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    rb, mb = project_and_sample_triton(xb, kb, pb, dt, dt_idx, img_hw)
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

    xyz, K, P, _, _, _ = _make_inputs(B=B, N=N, H=H, W=W, seed=21)
    img_hw = (
        torch.tensor([H, W], device="cuda", dtype=torch.int32).expand(B, 2).contiguous()
    )

    # 1) torch backend with the source+indices path
    res_a, mask_a = project_and_sample_logic(
        xyz,
        K,
        P,
        img_hw,
        dt_src,
        dt_indices=dt_idx,
        border=0,
        backend="torch",
    )
    # 2) torch backend with the pre-gathered tensor (no indices)
    res_b, mask_b = project_and_sample_logic(
        xyz,
        K,
        P,
        img_hw,
        dt_gather,
        border=0,
        backend="torch",
    )
    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(res_a, res_b), "torch source+idx ≠ torch gather"

    # 3) triton backend with source+indices
    res_c, mask_c = project_and_sample_triton(xyz, K, P, dt_src, dt_idx, img_hw)
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
    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=4, N=512, H=64, W=80, seed=3)

    # Force ~30% of the points to land behind the camera (zc ≤ 0) and a few
    # to land at zc ≈ 0 (so inv_z → ±Inf).
    g = torch.Generator(device=dt.device).manual_seed(99)
    bad = torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.3
    xyz = xyz.clone()
    xyz[..., 2] = torch.where(bad, -xyz[..., 2].abs(), xyz[..., 2])
    # Sprinkle a handful with zc tiny-positive
    tiny = torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.02
    xyz[..., 2] = torch.where(tiny, 1e-9 * torch.ones_like(xyz[..., 2]), xyz[..., 2])

    xyz_t = xyz.clone().requires_grad_(True)
    K_t = K.clone().requires_grad_(True)
    P_t = P.clone().requires_grad_(True)

    res, mask = project_and_sample_triton(xyz_t, K_t, P_t, dt, dt_idx, img_hw)
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


def test_torch_matches_triton_behind_camera_and_nan():
    """Adversarial parity: behind-camera and NaN world points.

    The torch reference path historically lacked the ``zc > 0`` and
    finiteness gating of the Triton kernel, so behind-camera points that the
    perspective divide mirrors into the frame counted as inside (with live
    gradients) and NaN points passed as "inside". Locks in the parity fix in
    ``project_world_to_2D``.

    Backward is checked on the no-NaN variant only: with NaN *inputs* the
    torch autograd chain unavoidably produces NaN grads through 0·NaN
    products, while the Triton backward selects exact zeros for masked
    points (covered by ``test_no_nan_when_points_behind_camera``).
    """
    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=4, N=512, H=64, W=80, seed=17)
    g = torch.Generator(device=dt.device).manual_seed(23)
    xyz = xyz.clone()
    # ~30% behind the camera, a few at zc ≈ 0.
    bad = torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.3
    xyz[..., 2] = torch.where(bad, -xyz[..., 2].abs(), xyz[..., 2])
    tiny = torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.02
    xyz[..., 2] = torch.where(tiny, torch.full_like(xyz[..., 2], 1e-9), xyz[..., 2])

    # --- Backward parity on the no-NaN adversarial input -------------------
    xa, ka, pa = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    res_ref, mask_ref = project_and_sample_logic(
        xa, ka, pa, img_hw, dt, dt_indices=dt_idx, border=0
    )
    xb, kb, pb = (
        xyz.clone().requires_grad_(True),
        K.clone().requires_grad_(True),
        P.clone().requires_grad_(True),
    )
    res_tri, mask_tri = project_and_sample_triton(xb, kb, pb, dt, dt_idx, img_hw)

    assert torch.equal(mask_ref, mask_tri), "inside-mask mismatch (behind-camera)"
    behind = bad & ~tiny
    assert not mask_ref[behind].any(), "behind-camera point marked inside"
    assert torch.isfinite(res_ref).all(), "torch forward produced non-finite residuals"
    d_fwd = (res_ref - res_tri).abs()
    msg = "residuals differ (invalid points must be 0 in both)"
    assert d_fwd.max().item() < 1e-3, msg

    (res_ref * mask_ref.to(res_ref.dtype)).sum().backward()
    (res_tri * mask_tri.to(res_tri.dtype)).sum().backward()
    for name, a, b in [
        ("xyz", xa.grad, xb.grad),
        ("K[:,:2,:]", ka.grad[:, :2, :], kb.grad[:, :2, :]),
        ("P[:,:3,:]", pa.grad[:, :3, :], pb.grad[:, :3, :]),
    ]:
        assert torch.isfinite(a).all(), f"torch grad {name} has non-finite values"
        assert torch.allclose(a, b, atol=5e-3, rtol=2e-3), f"grad {name} mismatch"

    # --- Forward parity with NaN world points sprinkled in -----------------
    nan_pts = torch.rand(xyz.shape[:2], device=dt.device, generator=g) < 0.05
    xyz_nan = torch.where(nan_pts[..., None], torch.full_like(xyz, float("nan")), xyz)
    res_ref_n, mask_ref_n = project_and_sample_logic(
        xyz_nan, K, P, img_hw, dt, dt_indices=dt_idx, border=0
    )
    res_tri_n, mask_tri_n = project_and_sample_triton(xyz_nan, K, P, dt, dt_idx, img_hw)
    assert torch.equal(mask_ref_n, mask_tri_n), "inside-mask mismatch (NaN points)"
    assert not mask_ref_n[nan_pts].any(), "NaN point marked inside"
    assert torch.isfinite(res_ref_n).all(), "NaN leaked into torch residuals"
    d_nan = (res_ref_n - res_tri_n).abs()
    assert d_nan.max().item() < 1e-3
    print(
        f"  {behind.sum().item()} behind-camera, {tiny.sum().item()} zc≈0, "
        f"{nan_pts.sum().item()} NaN pts — masks equal, residuals/grads match ✓"
    )


def test_per_row_img_hw_gates_padded_region():
    """Mixed per-row real H/W: projections into the padded zone must be rejected.

    Regression test for the bug where the Triton inside-check used the padded
    DT canvas H, W instead of each row's real image extent. On mixed-resolution
    datasets (e.g. mipnerf360) this let projections land in the padded zone
    and contribute spurious DT values to the loss.

    Setup: padded canvas is 80×80; rows alternate between real shape (80, 80)
    (uses full canvas) and (40, 40) (only the top-left quadrant is "real").
    Construct points that project into the (40, 80) × (40, 80) padded zone of
    the small rows — those must be outside under the fix and inside under
    the old behavior, so torch and triton must agree (both correct) and
    differ from a buggy "use padded H, W" reference.
    """
    device = "cuda"
    B, N, H_pad, W_pad = 4, 256, 80, 80
    H_small, W_small = 40, 40

    g = torch.Generator(device=device).manual_seed(31)
    xyz = torch.randn(B, N, 3, device=device, generator=g)
    xyz[..., 2] = xyz[..., 2].abs() + 1.5
    # Force the projected u, v to span the entire padded canvas (so plenty
    # of points fall into the padded zone of the small rows).
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 0] = 30.0
    K[:, 1, 1] = 30.0
    K[:, 0, 2] = W_pad / 2.0
    K[:, 1, 2] = H_pad / 2.0
    K[:, 2, 2] = 1.0
    P = torch.eye(4, device=device).expand(B, 4, 4).contiguous().clone()

    dt = 5.0 * torch.rand(B, 1, H_pad, W_pad, device=device, generator=g)
    dt_idx = torch.arange(B, device=device, dtype=torch.int64)

    # Per-row real H/W: rows 0, 2 use the full canvas; rows 1, 3 are small.
    img_hw = torch.tensor(
        [[H_pad, W_pad], [H_small, W_small], [H_pad, W_pad], [H_small, W_small]],
        device=device,
        dtype=torch.int32,
    )

    res_torch, mask_torch = project_and_sample_logic(
        xyz,
        K,
        P,
        img_hw,
        dt,
        dt_indices=dt_idx,
        border=0,
        backend="torch",
    )
    res_tri, mask_tri = project_and_sample_triton(xyz, K, P, dt, dt_idx, img_hw)

    msg = "per-row img_hw: torch and triton disagree on inside-mask"
    assert torch.equal(mask_torch, mask_tri), msg

    diff = (res_torch - res_tri)[mask_torch].abs()
    print(
        f"  per-row H/W: mask matches, "
        f"max|Δres|={diff.max().item():.3e} on {mask_torch.sum().item()} inside points"
    )
    assert diff.max().item() < 1e-3

    # Sanity: the small-canvas rows must have STRICTLY fewer inside points
    # than a buggy "use padded H, W" check would yield. Equivalent to: at
    # least one projection lands in the padded zone of a small row.
    img_hw_buggy = torch.tensor(
        [[H_pad, W_pad]] * B,
        device=device,
        dtype=torch.int32,
    )
    _, mask_buggy = project_and_sample_triton(xyz, K, P, dt, dt_idx, img_hw_buggy)
    extra_inside = (mask_buggy & ~mask_tri).sum().item()
    print(
        f"  buggy 'use padded H, W' would mark {extra_inside} extra points as inside "
        f"(must be > 0 for this test to be meaningful)"
    )
    assert extra_inside > 0, (
        "test is vacuous: no projection landed in the padded zone — "
        "adjust K / xyz so the bug case actually triggers"
    )


def test_fused_loss_epilogue_bit_exact():
    """Fused loss epilogue ≡ unfused kernel + torch chain, bit-for-bit.

    Replicates the exact downstream of ``EPO.compute_forward_step``
    (clamp → Huber → mask → per-direction mean → per-pair sum → mean) on
    both paths and requires ``torch.equal`` on the loss AND on every input
    gradient. Inputs include behind-camera points (inside=False), padded
    points (pad_mask=False), residuals above ``clamp_max`` (clamped branch)
    and on both sides of ``huber_delta`` (both Huber branches).
    """
    from helpers.triton_ops import project_sample_huber_triton
    from losses.dt_loss import compute_chunk_loss_logic

    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=4, N=512, H=64, W=80, seed=11)
    g = torch.Generator(device=xyz.device).manual_seed(99)
    # ~20% of points behind the camera, ~15% pad-masked.
    behind = torch.rand(xyz.shape[:2], device=xyz.device, generator=g) < 0.2
    xyz[..., 2] = torch.where(behind, -xyz[..., 2].abs() - 0.5, xyz[..., 2])
    pad_mask = torch.rand(xyz.shape[:2], device=xyz.device, generator=g) >= 0.15
    # DT in [0, 5] with clamp_max=3.0 exercises the clamped branch;
    # huber_delta=1.0 exercises both Huber branches.
    clamp_max, huber_delta = 3.0, 1.0

    def downstream(rho_sum, count, dtype):
        mean_losses = torch.where(
            count > 0,
            rho_sum / count.to(dtype).clamp(min=1.0),
            rho_sum.new_zeros(()),
        )
        return mean_losses.view(-1, 2).sum(dim=1).mean()

    # Reference: unfused Triton op + torch loss chain
    xyz_a = xyz.clone().requires_grad_(True)
    K_a = K.clone().requires_grad_(True)
    P_a = P.clone().requires_grad_(True)
    res_a, inside_a = project_and_sample_triton(xyz_a, K_a, P_a, dt, dt_idx, img_hw)
    valid_a = pad_mask & inside_a
    sum_a, count_a = compute_chunk_loss_logic(
        res_a, valid_a, clamp_max=clamp_max, huber_delta=huber_delta
    )
    loss_a = downstream(sum_a, count_a, res_a.dtype)
    loss_a.backward()

    # Fused path
    xyz_b = xyz.clone().requires_grad_(True)
    K_b = K.clone().requires_grad_(True)
    P_b = P.clone().requires_grad_(True)
    rho_b, valid_b = project_sample_huber_triton(
        xyz_b, K_b, P_b, dt, dt_idx, img_hw, pad_mask, clamp_max, huber_delta
    )
    loss_b = downstream(rho_b.sum(dim=1), valid_b.sum(dim=1), rho_b.dtype)
    loss_b.backward()

    n_clamped = (res_a[valid_a] > clamp_max).sum().item()
    n_linear = (res_a[valid_a].clamp(max=clamp_max) > huber_delta).sum().item()
    print(
        f"  valid={valid_a.sum().item()}/{valid_a.numel()}  clamped={n_clamped}  "
        f"linear-branch={n_linear}  loss={loss_a.item():.6f}"
    )
    assert n_clamped > 0 and n_linear > 0, "test is vacuous: branches not exercised"

    assert torch.equal(valid_a, valid_b), "valid mask mismatch"
    rho_a = torch.where(
        valid_a,
        torch.where(
            res_a.clamp(max=clamp_max) <= huber_delta,
            0.5 * res_a.clamp(max=clamp_max) * res_a.clamp(max=clamp_max),
            huber_delta * (res_a.clamp(max=clamp_max) - 0.5 * huber_delta),
        ),
        torch.zeros_like(res_a),
    )
    assert torch.equal(rho_a, rho_b), "per-point rho not bit-identical"
    assert torch.equal(loss_a, loss_b), "loss not bit-identical"
    assert torch.equal(xyz_a.grad, xyz_b.grad), "grad_xyz not bit-identical"
    assert torch.equal(K_a.grad, K_b.grad), "grad_K not bit-identical"
    assert torch.equal(P_a.grad, P_b.grad), "grad_P not bit-identical"
    print("  loss + all grads bit-identical (torch.equal)")


def test_fused_reduction_matches_within_tolerance():
    """Fully-fused reduction ≡ torch row-sum up to fp reordering noise.

    The fused row reduction accumulates in a different order than torch's
    ``sum(dim=1)``, so exact equality is impossible by design — but counts
    are integer-valued (exact in fp32) and sums/grads must agree to fp32
    reduction tolerance.
    """
    from helpers.triton_ops import (
        project_sample_huber_sum_triton,
        project_sample_huber_triton,
    )

    xyz, K, P, dt, dt_idx, img_hw = _make_inputs(B=4, N=512, H=64, W=80, seed=23)
    g = torch.Generator(device=xyz.device).manual_seed(5)
    pad_mask = torch.rand(xyz.shape[:2], device=xyz.device, generator=g) >= 0.15
    clamp_max, huber_delta = 3.0, 1.0

    def downstream(rho_sum, count, dtype):
        mean_losses = torch.where(
            count > 0,
            rho_sum / count.to(dtype).clamp(min=1.0),
            rho_sum.new_zeros(()),
        )
        return mean_losses.view(-1, 2).sum(dim=1).mean()

    xyz_a = xyz.clone().requires_grad_(True)
    rho_a, valid_a = project_sample_huber_triton(
        xyz_a, K, P, dt, dt_idx, img_hw, pad_mask, clamp_max, huber_delta
    )
    sum_a, count_a = rho_a.sum(dim=1), valid_a.sum(dim=1)
    downstream(sum_a, count_a, rho_a.dtype).backward()

    xyz_b = xyz.clone().requires_grad_(True)
    sum_b, count_b = project_sample_huber_sum_triton(
        xyz_b, K, P, dt, dt_idx, img_hw, pad_mask, clamp_max, huber_delta
    )
    downstream(sum_b, count_b, sum_b.dtype).backward()

    assert torch.equal(count_a.to(count_b.dtype), count_b), "counts must be exact"
    assert torch.allclose(sum_a, sum_b, rtol=1e-5, atol=1e-4), "row sums diverged"
    grad_msg = "grad_xyz diverged"
    assert torch.allclose(xyz_a.grad, xyz_b.grad, rtol=1e-4, atol=1e-6), grad_msg
    d = (sum_a - sum_b).abs().max().item()
    print(f"  counts exact; max|Δsum|={d:.3e} (fp32 reordering envelope)")


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
    print("\n== Torch ≡ Triton on behind-camera / NaN points ==")
    test_torch_matches_triton_behind_camera_and_nan()
    print("\n== Per-row img_hw gates padded region ==")
    test_per_row_img_hw_gates_padded_region()
    print("\n== Fused loss epilogue bit-exactness ==")
    test_fused_loss_epilogue_bit_exact()
    print("\n== Fused reduction tolerance ==")
    test_fused_reduction_matches_within_tolerance()
    print("\nAll tests passed.")

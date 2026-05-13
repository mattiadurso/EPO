"""Validation for the fused unproject Triton op.

Checks:
  1. Forward output matches ``unproject_2D_to_world`` numerically.
  2. Backward gradients match autograd through the reference chain.
  3. Stability under degenerate depths (very small / large).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from helpers.reprojection import unproject_2D_to_world


def _make_inputs(B=4, N=512, dtype=torch.float32, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    # Edge-pixel coords in a plausible image plane (518×518)
    H, W = 518, 518
    u = torch.randint(0, W, (B, N), device=device, generator=g).to(dtype)
    v = torch.randint(0, H, (B, N), device=device, generator=g).to(dtype)
    xy0 = torch.stack([u, v], dim=-1)

    # Reasonable depth in [0.5, 5.0]
    depth0 = 0.5 + 4.5 * torch.rand(B, N, device=device, dtype=dtype, generator=g)

    # Intrinsics: focal length ~300 px, principal point at image center.
    K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    K[:, 0, 0] = 300.0
    K[:, 1, 1] = 300.0
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    K[:, 2, 2] = 1.0

    # Extrinsics: small random rotation + translation around identity
    P = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).contiguous().clone()
    P[:, :3, 3] = 0.1 * torch.randn(B, 3, device=device, dtype=dtype, generator=g)
    # tiny rotation: skew + add to identity, then re-orthonormalize via QR
    skew = 0.05 * torch.randn(B, 3, 3, device=device, dtype=dtype, generator=g)
    R = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3).contiguous() + skew
    R, _ = torch.linalg.qr(R)
    P[:, :3, :3] = R

    return xy0, K, depth0, P


def test_forward_matches_reference():
    """Triton forward must match the reference unproject within fp32 noise."""
    xy0, K, depth0, P = _make_inputs(B=4, N=512)

    ref = unproject_2D_to_world(xy0, K, depth0, P, backend="torch")
    tri = unproject_2D_to_world(xy0, K, depth0, P, backend="triton")

    diff = (ref - tri).abs()
    print(
        f"forward |Δ| max={diff.max().item():.3e}  "
        f"mean={diff.mean().item():.3e}"
    )
    assert diff.max().item() < 5e-3, "forward output mismatch"


def test_backward_matches_reference():
    """Triton analytical grads must match autograd through the reference path."""
    xy0, K, depth0, P = _make_inputs(B=3, N=400, seed=1)

    # Reference
    Ka = K.clone().requires_grad_(True)
    da = depth0.clone().requires_grad_(True)
    Pa = P.clone().requires_grad_(True)
    ra = unproject_2D_to_world(xy0, Ka, da, Pa, backend="torch")
    ra.sum().backward()

    # Triton
    Kb = K.clone().requires_grad_(True)
    db = depth0.clone().requires_grad_(True)
    Pb = P.clone().requires_grad_(True)
    rb = unproject_2D_to_world(xy0, Kb, db, Pb, backend="triton")
    rb.sum().backward()

    def cmp(name, a, b, atol=5e-3, rtol=2e-3):
        d = (a - b).abs()
        rel = d / a.abs().clamp(min=1e-6)
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(
            f"  grad {name:<10} max|Δ|={d.max().item():.3e}  "
            f"mean|Δ|={d.mean().item():.3e}  "
            f"max|rel|={rel.max().item():.3e}  ok={ok}"
        )
        assert ok, f"grad {name} mismatch"

    print("Backward (loss = sum(xyz_world)):")
    cmp("depth", da.grad, db.grad)
    # Only K[0,0], K[1,1], K[0,2], K[1,2] are touched.
    cmp("K_pinhole", Ka.grad[:, :2, :3], Kb.grad[:, :2, :3])
    # Only P[:3, :] is touched (R + t).
    cmp("P_R_t", Pa.grad[:, :3, :], Pb.grad[:, :3, :])


def test_backward_random_upstream():
    """Cross-check with a non-uniform upstream gradient."""
    xy0, K, depth0, P = _make_inputs(B=3, N=400, seed=7)
    g = torch.Generator(device=K.device).manual_seed(11)
    grad_up = torch.randn(3, 400, 3, device=K.device, generator=g)

    Ka = K.clone().requires_grad_(True); da = depth0.clone().requires_grad_(True); Pa = P.clone().requires_grad_(True)
    ra = unproject_2D_to_world(xy0, Ka, da, Pa, backend="torch")
    (ra * grad_up).sum().backward()

    Kb = K.clone().requires_grad_(True); db = depth0.clone().requires_grad_(True); Pb = P.clone().requires_grad_(True)
    rb = unproject_2D_to_world(xy0, Kb, db, Pb, backend="triton")
    (rb * grad_up).sum().backward()

    print("Random upstream grad:")
    for name, a, b in [
        ("depth", da.grad, db.grad),
        ("K_pinhole", Ka.grad[:, :2, :3], Kb.grad[:, :2, :3]),
        ("P_R_t", Pa.grad[:, :3, :], Pb.grad[:, :3, :]),
    ]:
        d = (a - b).abs()
        rel = d / a.abs().clamp(min=1e-6)
        print(
            f"  {name:<10} max|Δ|={d.max().item():.3e}  "
            f"mean|Δ|={d.mean().item():.3e}  "
            f"max|rel|={rel.max().item():.3e}"
        )
        assert torch.allclose(a, b, atol=5e-3, rtol=5e-3), f"{name} mismatch"


def test_stability_small_depth():
    """Very small depths should produce finite gradients (no Inf/NaN)."""
    xy0, K, depth0, P = _make_inputs(B=2, N=128, seed=3)
    # Mix in some near-min depths (matches DepthModule's min_depth=1e-3).
    depth0 = depth0.clone()
    depth0[:, :30] = 1e-3
    K = K.clone().requires_grad_(True)
    depth0 = depth0.requires_grad_(True)
    P = P.clone().requires_grad_(True)

    out = unproject_2D_to_world(xy0, K, depth0, P, backend="triton")
    assert torch.isfinite(out).all(), "forward produced non-finite values"
    out.sum().backward()
    assert torch.isfinite(K.grad).all(), "grad_K non-finite"
    assert torch.isfinite(depth0.grad).all(), "grad_depth non-finite"
    assert torch.isfinite(P.grad).all(), "grad_P non-finite"
    print("  small-depth gradients all finite ✓")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for the Triton op"
    print("== Forward parity ==")
    test_forward_matches_reference()
    print("\n== Backward parity (uniform upstream grad) ==")
    test_backward_matches_reference()
    print("\n== Backward parity (random upstream grad) ==")
    test_backward_random_upstream()
    print("\n== Stability under small depths ==")
    test_stability_small_depth()
    print("\nAll tests passed.")

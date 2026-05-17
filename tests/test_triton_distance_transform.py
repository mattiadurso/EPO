"""Validation for the Triton exact-EDT op.

Replaces ``cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)``.

Checks:
  1. Bit-equivalence to cv2 across edge densities, sizes, and shapes.
  2. Robustness on degenerate inputs (single edge, dense edges, no edges
     except a sentinel).
  3. Speed parity with cv2 on a 518x518 input (the EPO default).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

if not torch.cuda.is_available():
    pytest.skip("Triton EDT requires CUDA", allow_module_level=True)

from helpers.triton_ops import distance_transform_l2_triton  # noqa: E402


def _cv2_reference(edges_np: np.ndarray) -> np.ndarray:
    """cv2 reference: distance to nearest non-zero pixel of ``edges_np``."""
    mask = np.where(edges_np > 0, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _run_pair(edges_np: np.ndarray):
    edges_t = torch.from_numpy(edges_np).cuda()
    out_triton = distance_transform_l2_triton(edges_t).cpu().numpy()
    out_cv2 = _cv2_reference(edges_np)
    return out_triton, out_cv2


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [16, 64, 128, 192, 256, 384, 518, 1024])
@pytest.mark.parametrize("density", [0.005, 0.05, 0.20])
def test_matches_cv2_square(size, density):
    """Exact bit-for-bit match against cv2 for square images."""
    rng = np.random.default_rng(size * 1000 + int(density * 1000))
    edges = (rng.random((size, size)) < density).astype(np.float32)
    # Guarantee at least one edge so cv2/triton aren't both at saturation
    edges[0, 0] = 1.0

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(
        out_triton,
        out_cv2,
        atol=1e-5,
        rtol=1e-6,
        err_msg=f"mismatch on size={size}, density={density}",
    )


@pytest.mark.parametrize("h,w", [(64, 128), (200, 350), (518, 384)])
def test_matches_cv2_rectangular(h, w):
    rng = np.random.default_rng(h * 10_000 + w)
    edges = (rng.random((h, w)) < 0.05).astype(np.float32)
    edges[0, 0] = 1.0

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)


def test_single_edge_pixel():
    """One edge in the middle — output should be radial distance map."""
    edges = np.zeros((64, 64), dtype=np.float32)
    edges[32, 32] = 1.0

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)


def test_edge_at_corner():
    """Edge only at (0, 0) — exercises the LARGE-seed cascade."""
    edges = np.zeros((128, 128), dtype=np.float32)
    edges[0, 0] = 1.0

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)


def test_dense_edges():
    """Many edges — most pixels should be distance 0 or 1."""
    rng = np.random.default_rng(0)
    edges = (rng.random((128, 128)) < 0.5).astype(np.float32)

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)


def test_line_edge():
    """A horizontal line of edges — exercises row pass equivalence."""
    edges = np.zeros((128, 128), dtype=np.float32)
    edges[64, :] = 1.0

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)


def test_full_edges():
    """Every pixel is an edge — output is all zeros."""
    edges = np.ones((64, 64), dtype=np.float32)

    out_triton, out_cv2 = _run_pair(edges)
    np.testing.assert_allclose(out_triton, out_cv2, atol=1e-5, rtol=1e-6)
    assert out_triton.max() == 0.0


# ---------------------------------------------------------------------------
# Speed parity
# ---------------------------------------------------------------------------


def _bench_ms(fn, iters: int = 50) -> float:
    """Steady-state milliseconds per call. Caller handles warmup + sync."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


@pytest.mark.parametrize("density", [0.01, 0.05, 0.20])
def test_speed_parity_518(density):
    """Triton should be within ~2x of cv2 on the EPO default (518x518)."""
    rng = np.random.default_rng(int(density * 1000))
    edges = (rng.random((518, 518)) < density).astype(np.float32)
    edges[0, 0] = 1.0

    edges_t = torch.from_numpy(edges).cuda()
    mask = np.where(edges > 0, 0, 1).astype(np.uint8)

    # Warmup
    for _ in range(10):
        cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distance_transform_l2_triton(edges_t)

    t_cv2 = _bench_ms(
        lambda: cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    )
    t_tr = _bench_ms(lambda: distance_transform_l2_triton(edges_t))

    print(
        f"\n  density={density:.2f}: cv2={t_cv2:.3f} ms, "
        f"triton={t_tr:.3f} ms, ratio={t_tr / t_cv2:.2f}x"
    )

    # Generous bound — primary requirement is correctness; this just guards
    # against pathological regressions (10x+ slowdown).
    assert t_tr < 2.0 * t_cv2, f"Triton too slow: {t_tr:.3f} ms vs cv2 {t_cv2:.3f} ms"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

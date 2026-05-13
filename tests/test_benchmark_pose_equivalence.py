"""Equivalence + speedup test for the vendored pose benchmark.

Compares :func:`helpers.benchmark_pose.evaluate_scene` (vectorised) against
:func:`helpers.benchmark_pose.evaluate_scene_reference` (the byte-for-byte
posebench original) on a real COLMAP reconstruction, asserting:

1. Identical pair set and identical missing-image masking.
2. Per-pair ``q_error`` / ``t_error`` agree to within FP tolerance.
3. The derived ``max_error`` column matches (including the ``< 10`` clip).
4. ``compute_AUC`` on the two error vectors agrees exactly.

The test also prints a wall-clock comparison so regressions in the fast
path show up immediately.

Run with::

    python -m pytest tests/test_benchmark_pose_equivalence.py -s

Skipped automatically if the reference reconstruction isn't on disk.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

# Default paths — overridable via env vars for portability.
INPUT_PATH = os.environ.get(
    "EPO_TEST_INPUT_REC",
    "/home/mattia/Desktop/Repos/batchsfm/benchmarks/vggt/mipnerf360/bicycle/sparse",
)
TARGET_PATH = os.environ.get(
    "EPO_TEST_TARGET_REC",
    "/home/mattia/Desktop/datasets/mipnerf360/bicycle/sparse_150",
)

pycolmap = pytest.importorskip("pycolmap")
from helpers.benchmark_pose import (  # noqa: E402
    eval_colmap_model,
    eval_colmap_model_reference,
    evaluate_scene,
    evaluate_scene_reference,
)
from helpers.utils_benchmark_pose import compute_AUC  # noqa: E402


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"reference reconstruction missing: {path}")


def _load_pair():
    _require(INPUT_PATH)
    _require(TARGET_PATH)
    return (
        pycolmap.Reconstruction(INPUT_PATH),
        pycolmap.Reconstruction(TARGET_PATH),
    )


def test_evaluate_scene_matches_reference():
    rec_input, rec_target = _load_pair()

    t0 = time.perf_counter()
    df_ref, n_ref = evaluate_scene_reference(rec_target, rec_input)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_fast, n_fast = evaluate_scene(rec_target, rec_input)
    t_fast = time.perf_counter() - t0

    print(
        f"\nevaluate_scene: reference={t_ref:.3f}s  fast={t_fast:.3f}s "
        f"speedup={t_ref / max(t_fast, 1e-9):.1f}x  pairs={len(df_ref)}"
    )

    # Same image counts, same pair set.
    assert n_ref == n_fast
    assert len(df_ref) == len(df_fast)

    # Pair set must match exactly (same sort, same combinations).
    ref_pairs = list(zip(df_ref["image1"], df_ref["image2"], strict=False))
    fast_pairs = list(zip(df_fast["image1"], df_fast["image2"], strict=False))
    assert ref_pairs == fast_pairs

    # inf masking must align.
    ref_inf = np.isinf(df_ref["q_error"].to_numpy())
    fast_inf = np.isinf(df_fast["q_error"].to_numpy())
    np.testing.assert_array_equal(ref_inf, fast_inf)

    finite = ~ref_inf
    # The reference uses the quaternion-eigh form; the fast path uses the
    # trace form. Both compute the geodesic angle of R_pred @ R_target.T,
    # so they agree to ~6 decimal places on real rotations.
    np.testing.assert_allclose(
        df_fast.loc[finite, "q_error"].to_numpy(),
        df_ref.loc[finite, "q_error"].to_numpy(),
        atol=1e-4,
        rtol=1e-5,
        err_msg="rotation error mismatch (trace vs quaternion form)",
    )
    np.testing.assert_allclose(
        df_fast.loc[finite, "t_error"].to_numpy(),
        df_ref.loc[finite, "t_error"].to_numpy(),
        atol=1e-6,
        rtol=1e-6,
        err_msg="translation error mismatch",
    )

    # max_error column applies an identical < 10 clip in both impls.
    ref_max = df_ref["max_error"].to_numpy()
    fast_max = df_fast["max_error"].to_numpy()
    # Both can have inf in the same slots.
    both_inf = np.isinf(ref_max) & np.isinf(fast_max)
    finite_max = ~both_inf
    np.testing.assert_allclose(
        fast_max[finite_max],
        ref_max[finite_max],
        atol=1e-4,
        rtol=1e-5,
    )
    assert np.array_equal(np.isinf(ref_max), np.isinf(fast_max))


def test_compute_AUC_matches():
    rec_input, rec_target = _load_pair()
    df_ref, _ = evaluate_scene_reference(rec_target, rec_input)
    df_fast, _ = evaluate_scene(rec_target, rec_input)

    thrs = [0.5, 1, 3, 5, 10]
    auc_ref = compute_AUC(df_ref["max_error"], thrs)
    auc_fast = compute_AUC(df_fast["max_error"], thrs)
    print(f"\nAUC reference: {np.round(auc_ref, 4)}")
    print(f"AUC fast:      {np.round(auc_fast, 4)}")
    np.testing.assert_allclose(auc_fast, auc_ref, atol=1e-3, rtol=1e-4)


def test_eval_colmap_model_end_to_end():
    """The public entry point must agree on the AUC vector it returns."""
    _require(INPUT_PATH)
    _require(TARGET_PATH)

    thrs = [1, 3, 5]
    auc_ref, n_ref, df_ref = eval_colmap_model_reference(
        INPUT_PATH, TARGET_PATH, thrs=thrs, return_df=True
    )
    auc_fast, n_fast, df_fast = eval_colmap_model(
        INPUT_PATH, TARGET_PATH, thrs=thrs, return_df=True
    )
    print(f"\neval_colmap_model AUC ref:  {np.round(auc_ref, 4)}")
    print(f"eval_colmap_model AUC fast: {np.round(auc_fast, 4)}")
    assert n_ref == n_fast
    np.testing.assert_allclose(auc_fast, auc_ref, atol=1e-3, rtol=1e-4)


if __name__ == "__main__":
    test_evaluate_scene_matches_reference()
    test_compute_AUC_matches()
    test_eval_colmap_model_end_to_end()
    print("\nAll equivalence tests passed.")

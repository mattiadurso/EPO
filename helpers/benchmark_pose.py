"""Pose benchmarking entry points (vendored from sibling ``posebench`` repo).

This module exposes two equivalent implementations of the per-pair pose
error computation:

* :func:`evaluate_scene_reference` / :func:`eval_colmap_model_reference` —
  the original O(N^2) loop, kept verbatim from posebench for ground-truth
  comparison.
* :func:`evaluate_scene` / :func:`eval_colmap_model` — a vectorised
  replacement that produces numerically identical errors (to within
  floating-point round-off) but is ~10-100x faster on scenes with more
  than a few dozen images. This is the default used everywhere.

The two are wired up so any external caller importing
``eval_colmap_model`` automatically gets the fast path. Run
``tests/test_benchmark_pose_equivalence.py`` to verify parity.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Same-directory imports work both as a package (``helpers.benchmark_pose``)
# and as a top-level script (``python helpers/benchmark_pose.py``).
try:
    from .utils_benchmark_pose import compute_AUC, evaluate_R_t
except ImportError:  # script / sys.path execution
    from utils_benchmark_pose import compute_AUC, evaluate_R_t

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reference (posebench-original) implementation                                #
# --------------------------------------------------------------------------- #
def evaluate_scene_reference(target_rec, input_rec, deg=True, verbose=False):
    """Original O(N^2)-with-O(M)-lookups implementation.

    Kept verbatim for the equivalence test. Do not use in hot paths; prefer
    :func:`evaluate_scene` which produces identical numbers far faster.
    """
    df = {
        "image1": [],
        "image2": [],
        "q_error": [],
        "t_error": [],
        "max_error": [],
    }
    target_images = np.array(
        sorted([img.name for img in target_rec.images.values()])
    )  # remove eventual subdirectory in the image name (e.g. camera calibration folder)
    input_images = np.array(sorted([img.name for img in input_rec.images.values()]))

    # for each pair of images in the ground truth
    for image_1_path, image_2_path in combinations(target_images, 2):
        if not (
            (image_1_path in input_images) and (image_2_path in input_images)
        ):  # working?
            q_err, t_err, max_error = np.inf, np.inf, np.inf
            if verbose:
                logger.info(
                    f"Image {image_1_path} or {image_2_path} not in input model."
                )
        else:
            # get the rotation and translation for two images (target)
            R1_target, t1_target = (
                target_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.rotation.matrix(),
                target_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.translation,
            )
            R2_target, t2_target = (
                target_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.rotation.matrix(),
                target_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.translation,
            )

            R1_input, t1_input = (
                input_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.rotation.matrix(),
                input_rec.find_image_with_name(image_1_path).cam_from_world.translation,
            )
            R2_input, t2_input = (
                input_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.rotation.matrix(),
                input_rec.find_image_with_name(image_2_path).cam_from_world.translation,
            )

            # compute the relative pose between the two images (target)
            R_target = R2_target @ R1_target.T
            t_target = t2_target - R_target @ t1_target

            # compute the relative pose between the two images (input)
            R_pred = R2_input @ R1_input.T
            t_pred = t2_input - R_pred @ t1_input

            # compute the error
            q_err, t_err = evaluate_R_t(R_pred, t_pred, R_target, t_target, deg=deg)
            max_error = max(q_err, t_err)
            max_error = max_error if max_error < 10 else np.inf

        # append to the dataframe
        df["image1"].append(image_1_path)
        df["image2"].append(image_2_path)
        df["q_error"].append(q_err)
        df["t_error"].append(t_err)
        df["max_error"].append(max_error)

    return pd.DataFrame(df), (len(input_images), len(target_images))


# --------------------------------------------------------------------------- #
# Vectorised implementation                                                   #
# --------------------------------------------------------------------------- #
def _stack_poses(rec, names):
    """Pull (R, t) for ``names`` out of a ``pycolmap.Reconstruction``.

    Returns ``(R, t, present_mask)`` where ``R`` has shape ``(N, 3, 3)``,
    ``t`` has shape ``(N, 3)``, and ``present_mask`` marks names absent
    from the reconstruction (those slots in ``R``/``t`` are identity / zero
    placeholders and must be filtered out by callers).
    """
    name_to_img = {img.name: img for img in rec.images.values()}
    n = len(names)
    R = np.empty((n, 3, 3), dtype=np.float64)
    t = np.empty((n, 3), dtype=np.float64)
    present = np.zeros(n, dtype=bool)
    for i, name in enumerate(names):
        img = name_to_img.get(name)
        if img is None:
            R[i] = np.eye(3)
            t[i] = 0.0
            continue
        cfw = img.cam_from_world
        R[i] = cfw.rotation.matrix()
        t[i] = cfw.translation
        present[i] = True
    return R, t, present


def _relative_R_err_batched(R_pred, R_target, deg=True):
    """Relative rotation error, fully vectorised, numerically stable everywhere.

    Computes the geodesic angle of ``R_rel = R_pred @ R_target.T`` using
    ``theta = atan2(||skew(R_rel)||, (tr(R_rel) - 1) / 2)``. This matches
    the quaternion-eigh path in :func:`utils_benchmark_pose.evaluate_R_err`
    to ~1e-6 deg on real rotations while avoiding the per-pair 4x4 ``eigh``
    that dominates the reference implementation's cost. The trace-only
    form ``arccos((tr-1)/2)`` is faster but loses precision near
    ``theta = 0`` due to ``1 - cos(theta) = theta^2/2`` cancellation.

    Args:
        R_pred: ``(N, 3, 3)`` array of predicted rotation matrices.
        R_target: ``(N, 3, 3)`` array of target rotation matrices.
        deg: return degrees if True, else radians.
    """
    # R_rel = R_pred @ R_target.T  (we never materialise the full N,3,3).
    # trace(R_rel) = sum_ij R_pred_ij * R_target_ij  (Frobenius inner product).
    trace = np.einsum("nij,nij->n", R_pred, R_target)
    cos_term = (trace - 1.0) * 0.5  # = cos(theta), exact in exact arithmetic.

    # Off-diagonal antisymmetric part of R_rel encodes sin(theta) * axis.
    # (R_rel - R_rel.T)_{ij} for ij in {(2,1),(0,2),(1,0)} give 2*sin*axis.
    # R_rel = R_pred @ R_target.T  ⇒  R_rel_{ij} = sum_k R_pred_{ik} R_target_{jk}.
    # We need (R_rel - R_rel.T)_{ij} = sum_k (R_pred_{ik} R_target_{jk}
    #                                       - R_pred_{jk} R_target_{ik}).
    a21 = np.einsum("nk,nk->n", R_pred[:, 2], R_target[:, 1]) - np.einsum(
        "nk,nk->n", R_pred[:, 1], R_target[:, 2]
    )
    a02 = np.einsum("nk,nk->n", R_pred[:, 0], R_target[:, 2]) - np.einsum(
        "nk,nk->n", R_pred[:, 2], R_target[:, 0]
    )
    a10 = np.einsum("nk,nk->n", R_pred[:, 1], R_target[:, 0]) - np.einsum(
        "nk,nk->n", R_pred[:, 0], R_target[:, 1]
    )
    # ||(R_rel - R_rel.T)/2|| = sin(theta).
    sin_term = 0.5 * np.sqrt(a21 * a21 + a02 * a02 + a10 * a10)

    err = np.arctan2(sin_term, cos_term)
    if deg:
        err = np.rad2deg(err)
    return err


def _relative_t_err_batched(t_pred, t_target, deg=True):
    """Translation-direction error, fully vectorised.

    Replicates :func:`utils_benchmark_pose.evaluate_t_err`:
    ``arccos(sqrt(1 - max(eps, 1 - <u, v>^2)))`` where ``u``, ``v`` are the
    unit-normalised translations. Magnitude is intentionally discarded.
    """
    eps = 1e-15
    n_pred = np.linalg.norm(t_pred, axis=-1, keepdims=True)
    n_tgt = np.linalg.norm(t_target, axis=-1, keepdims=True)
    u = t_pred / (n_pred + eps)
    v = t_target / (n_tgt + eps)
    inner = np.sum(u * v, axis=-1)
    loss_t = np.maximum(eps, 1.0 - inner**2)
    err = np.arccos(np.sqrt(1.0 - loss_t))
    if deg:
        err = np.rad2deg(err)
    return err


def evaluate_scene(target_rec, input_rec, deg=True, verbose=False):
    """Vectorised per-pair pose-error computation.

    Drop-in replacement for :func:`evaluate_scene_reference` — produces the
    same DataFrame columns and the same numerical values (up to FP
    round-off). Speedups come from three places:

    1. Each image's (R, t) is fetched **once** rather than four times per
       pair (eliminates the O(N^2 * M) ``find_image_with_name`` calls).
    2. All N*(N-1)/2 pair-wise relative poses are computed with two
       ``einsum`` calls — no Python pair loop.
    3. Rotation error uses the trace formula instead of the per-pair 4x4
       eigendecomposition in :func:`rotmat2qvec`.

    The "if either image is missing, error = inf" semantics are preserved.
    """
    target_images = np.array(sorted([img.name for img in target_rec.images.values()]))
    input_images_set = {img.name for img in input_rec.images.values()}

    # All unordered pairs (i, j) with i < j.
    idx_i, idx_j = np.triu_indices(len(target_images), k=1)
    name_i = target_images[idx_i]
    name_j = target_images[idx_j]

    # Fetch per-image poses once. Missing names get placeholder identity / zero
    # which we mask off afterwards.
    Rt, tt, present_t = _stack_poses(target_rec, target_images)  # GT always present.
    Ri, ti, present_i = _stack_poses(input_rec, target_images)  # may have gaps.

    R1_t, t1_t = Rt[idx_i], tt[idx_i]
    R2_t, t2_t = Rt[idx_j], tt[idx_j]
    R1_p, t1_p = Ri[idx_i], ti[idx_i]
    R2_p, t2_p = Ri[idx_j], ti[idx_j]

    # Relative pose: R_rel = R2 @ R1.T, t_rel = t2 - R_rel @ t1.
    R_target = np.einsum("nij,nkj->nik", R2_t, R1_t)
    t_target = t2_t - np.einsum("nij,nj->ni", R_target, t1_t)
    R_pred = np.einsum("nij,nkj->nik", R2_p, R1_p)
    t_pred = t2_p - np.einsum("nij,nj->ni", R_pred, t1_p)

    q_err = _relative_R_err_batched(R_pred, R_target, deg=deg)
    t_err = _relative_t_err_batched(t_pred, t_target, deg=deg)

    # Mask out pairs where either image is missing from the input reconstruction.
    missing = ~(present_i[idx_i] & present_i[idx_j])
    q_err[missing] = np.inf
    t_err[missing] = np.inf

    if verbose and missing.any():
        for idx in np.flatnonzero(missing):
            logger.info(f"Image {name_i[idx]} or {name_j[idx]} not in input model.")

    max_err = np.maximum(q_err, t_err)
    max_err = np.where(max_err < 10, max_err, np.inf)

    df = pd.DataFrame(
        {
            "image1": name_i,
            "image2": name_j,
            "q_error": q_err,
            "t_error": t_err,
            "max_error": max_err,
        }
    )
    return df, (len(input_images_set), len(target_images))


# --------------------------------------------------------------------------- #
# Public scene-level entry points                                             #
# --------------------------------------------------------------------------- #
def eval_colmap_model(
    model_path,
    target_path,
    thrs=None,
    return_df=False,
    AUC_col="max_error",
    verbose=False,
    _scene_fn=evaluate_scene,
):
    """Evaluate one reconstruction against a target. Uses the fast scene fn by default."""
    if thrs is None:
        thrs = [1, 3, 5]
    # read model
    try:
        rec_input = pycolmap.Reconstruction(model_path)
    except Exception as e:
        print(f"Failed to read input model from {model_path}: {e}")
        return np.array([np.nan] * len(thrs)), (np.nan, np.nan), None

    try:
        rec_target = pycolmap.Reconstruction(target_path)
    except Exception as e:
        print(f"Failed to read target model from {target_path}: {e}")
        return np.array([np.nan] * len(thrs)), (np.nan, np.nan), None

    df, num_images = _scene_fn(rec_target, rec_input, verbose=verbose)
    AUC_score_max = np.array(compute_AUC(df[AUC_col], thrs))

    if return_df:
        return AUC_score_max, num_images, df

    return AUC_score_max, num_images, None


def eval_colmap_model_reference(*args, **kwargs):
    """Reference (slow) path — kept for the equivalence test only."""
    kwargs["_scene_fn"] = evaluate_scene_reference
    return eval_colmap_model(*args, **kwargs)


def eval_colmap_model_all_scenes(
    input_path,
    target_path,
    input_folder="colmap/sparse/0",
    target_folder="sparse",
    thrs=None,
    AUC_col="max_error",
    return_df=False,
    n_jobs=-1,
    round_to=2,
    verbose=True,
) -> pd.DataFrame:
    """Evaluate the model on all the scenes in the data_path using parallel processing.

    These must be in COLMAP format. The model is evaluated at the specified thresholds.
    """
    if thrs is None:
        thrs = [0.5, 1, 3, 5, 10]
    # Get scene names from both directories
    input_scene_names = set(os.listdir(input_path))
    target_scene_names = set(os.listdir(target_path))

    # Keep only common scenes
    common_scenes = sorted(input_scene_names & target_scene_names)

    print(f"Found {len(common_scenes)} common scenes.")

    if len(common_scenes) == 0 and verbose:
        logger.warning("No common scenes found!")
        return pd.DataFrame()

    # Build paths for common scenes only
    input_paths = [
        os.path.join(input_path, scene, input_folder) for scene in common_scenes
    ]
    target_paths = [
        os.path.join(target_path, scene, target_folder) for scene in common_scenes
    ]

    # Verify paths exist
    valid_pairs = []
    valid_scenes = []
    for inp, tgt, scene in zip(input_paths, target_paths, common_scenes, strict=False):
        if os.path.exists(inp) and os.path.exists(tgt):
            valid_pairs.append((inp, tgt))
            valid_scenes.append(scene)
        else:
            logger.warning(f"Skipping {scene}: paths don't exist at {inp} and {tgt}")

    print(f"Evaluating {len(valid_pairs)} valid scenes.")

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(eval_colmap_model)(
            input,
            target,
            thrs=thrs,
            return_df=return_df,
            AUC_col=AUC_col,
            verbose=verbose,
        )
        for input, target in tqdm(
            valid_pairs,
            desc="Evaluating scenes",
            total=len(valid_pairs),
        )
    )

    results = [r[0] for r in parallel_results]
    num_images = [r[1] for r in parallel_results]
    reg_images = [r[0] for r in num_images]
    tot_images = [r[1] for r in num_images]
    dfs = [r[2] for r in parallel_results] if return_df else None

    if return_df:
        dfs_path = Path(input_path + "_results_dfs")
        dfs_path.mkdir(parents=True, exist_ok=True)
        for scene_name, df in zip(valid_scenes, dfs, strict=False):
            if df is not None:
                df.to_csv(dfs_path / f"results_{scene_name}.csv", index=False)
        print(f"Saved individual result dataframes to {dfs_path}")

    res = {}
    for auc_scores, scene_name in zip(results, valid_scenes, strict=False):
        if auc_scores is not None:
            res[scene_name] = auc_scores

    df_res_colmap = pd.DataFrame(res, index=thrs).transpose()

    df_res_colmap["reg_images"] = reg_images
    df_res_colmap["tot_images"] = tot_images
    df_res_colmap = df_res_colmap.sort_index()

    df_res_colmap.columns = [f"auc@{thr}" for thr in thrs] + [
        "reg_images",
        "tot_images",
    ]
    df_res_colmap = df_res_colmap[
        ["reg_images", "tot_images"] + [f"auc@{thr}" for thr in thrs]
    ]

    df_res_colmap.loc["mean"] = df_res_colmap.mean(numeric_only=True)

    df_res_colmap = df_res_colmap.round(round_to)

    return df_res_colmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=str, required=True)
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="benchmarks_3D/results")
    parser.add_argument("--many-scenes", action="store_true")
    parser.add_argument("--input-folder", type=str, default="colmap/sparse/0")
    parser.add_argument("--target-folder", type=str, default="sparse")
    parser.add_argument("--mapper", type=str, default="colmap")
    parser.add_argument("--thrs", type=float, nargs="+", default=[1, 3, 5, 10, 15, 30])
    args = parser.parse_args()

    s = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.many_scenes:
        df_res = eval_colmap_model_all_scenes(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            n_jobs=16,
            input_folder=args.input_folder,
            target_folder=args.target_folder,
        )
        df_res.to_csv(
            os.path.join(args.output_dir, f"results_all_scenes_{args.mapper}.csv"),
            index=True,
        )
        print(df_res)
    else:
        auc_scores, _, df = eval_colmap_model(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            return_df=True,
        )
        print(f"AUC scores at {args.thrs}: {auc_scores}")
        if df is not None:
            df.to_csv(
                os.path.join(
                    args.output_dir, f"results_single_scene_{args.mapper}.csv"
                ),
                index=False,
            )
            print(df)

    print(f"Total time: {time.time() - s:.2f} seconds.")

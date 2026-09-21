"""VGGSfM-tracker bundle adjustment on top of an exported 3DFM COLMAP model.

Generalises the BA path of :mod:`wrapper.vggt_wrapper` (``use_ba=True``) to
any model already on disk (``vggt``, ``vggt_omega``): load the model, its
``depths.pth`` and the images, run the VGGSfM tracker with ALIKED + SuperPoint
query points on DINO-ranked query frames, initialise every track's 3D point
from the model's own depth (no triangulation against the poses), bundle-adjust
with pycolmap and export at the original image resolution to
``benchmarks/<model>_vggsfm_ba/<dataset>/<scene>/sparse``.

The tracker is called through ``vggt.dependency.track_predict``: that module
is VGGSfM's tracker (same ``vggsfm_v2_tracker.pt`` weights, query-frame
ranking and keypoint extractors as ``third_party/vggsfm``) without the hydra
runner the upstream repo wraps it in. Like ``any2full_wrapper``, this is a
post-processor, not a 3DFM, so it is not in the ``WRAPPERS`` registry.

Example:
    python wrapper/vggsfm_ba_wrapper.py --model vggt --dataset mipnerf360 \
        --scenes bicycle
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pycolmap
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for path in (_ROOT, os.path.join(_ROOT, "third_party", "vggt")):
    if path not in sys.path:
        sys.path.insert(0, path)

from helpers.load import (  # noqa: E402
    find_images,
    load_and_preprocess_depths,
    load_and_preprocess_images,
)
from wrapper.np_to_colmap import batch_np_matrix_to_pycolmap  # noqa: E402

TRACK_RESOLUTION = (
    768  # images fed to the tracker (img_load_resolution of the vggt BA path)
)
DEPTH_RESOLUTION = 518  # padded-square grid the depth / intrinsics live on


def load_scene(model_sparse: str, images_dir: str, device: str) -> tuple:
    """Load the model's images (padded square, 1024) and depths (padded, 518).

    Returns:
        image names in model order, the images dict from ``helpers.load``,
        and the ``pycolmap.Reconstruction``.
    """
    recon = pycolmap.Reconstruction(model_sparse)
    names = sorted(image.name for image in recon.images.values())
    on_disk = {os.path.relpath(p, images_dir): p for p in find_images(images_dir)}
    paths = [on_disk[name] for name in names]
    images = load_and_preprocess_images(
        paths,
        images_dir,
        target_size=TRACK_RESOLUTION,
        load_with_pad=True,
        device=device,
    )
    images = load_and_preprocess_depths(
        model_sparse,
        images,
        target_size=DEPTH_RESOLUTION,
        load_with_pad=True,
        device=device,
    )
    return names, images, recon


def padded_intrinsics(camera: pycolmap.Camera, orig_wh: tuple[int, int]) -> np.ndarray:
    """PINHOLE K on the padded-square ``DEPTH_RESOLUTION`` grid (COLMAP convention).

    A model exported at network resolution (VGGT-Omega) is first rescaled to
    the on-disk image size, then padded to square and scaled like the loaders.
    """
    if (camera.width, camera.height) != orig_wh:
        camera = pycolmap.Camera(
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params,
            camera_id=camera.camera_id,
        )
        camera.rescale(*orig_wh)
    fx, fy, cx, cy = camera.params
    width, height = orig_wh
    max_dim = max(width, height)
    pad_x, pad_y = (max_dim - width) // 2, (max_dim - height) // 2
    scale = DEPTH_RESOLUTION / max_dim
    return np.array(
        [
            [fx * scale, 0, (cx + pad_x) * scale],
            [0, fy * scale, (cy + pad_y) * scale],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def model_arrays(names: list[str], images: dict, recon: pycolmap.Reconstruction):
    """Stack extrinsics, intrinsics, depth and confidence in model order."""
    by_name = {image.name: image for image in recon.images.values()}
    extrinsics, intrinsics, depths, confs = [], [], [], []
    for name in names:
        image = by_name[name]
        orig_wh = tuple(int(v) for v in images[name]["coords"][-2:].tolist())
        extrinsics.append(image.cam_from_world().matrix()[:3])
        intrinsics.append(padded_intrinsics(recon.camera(image.camera_id), orig_wh))
        depths.append(images[name]["depth"])
        confs.append(
            images[name].get("confidence", torch.ones_like(images[name]["depth"]))
        )
    return (
        np.stack(extrinsics),
        np.stack(intrinsics),
        torch.stack(depths).float().cpu().numpy(),
        torch.stack(confs).float().cpu().numpy(),
    )


@torch.no_grad()
def track_and_adjust(names, images, extrinsics, intrinsics, depth, conf, args):
    """VGGSfM tracks -> depth-initialised points -> pycolmap BA."""
    from vggt.dependency.track_predict import predict_tracks
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    points_3d = unproject_depth_map_to_point_map(
        depth[..., None], extrinsics, intrinsics
    )
    image_stack = torch.stack([images[name]["image"] for name in names])

    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        tracks, vis, _, track_points, colors = predict_tracks(
            image_stack,
            conf=conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            max_points_num=args.max_points_num,
            fine_tracking=False,
        )
    torch.cuda.empty_cache()
    track_time = time.time() - t0

    t0 = time.time()
    intrinsics = intrinsics.copy()
    intrinsics[:, :2, :] *= TRACK_RESOLUTION / DEPTH_RESOLUTION
    recon, _ = batch_np_matrix_to_pycolmap(
        track_points,
        extrinsics,
        intrinsics,
        tracks,
        np.array([TRACK_RESOLUTION, TRACK_RESOLUTION]),
        masks=vis > args.vis_thresh,
        max_reproj_error=args.max_reproj_error,
        shared_camera=True,
        camera_type="PINHOLE",
        points_rgb=colors,
        image_names=names,
    )
    if recon is None:
        raise SystemExit("no valid tracks after the reprojection filter")
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_principal_point = True  # as the vggt BA path
    pycolmap.bundle_adjustment(recon, options)
    # BA's robust loss tolerates degenerate points rather than removing them;
    # apply the mapper's own post-BA point filter so the export is sane.
    pycolmap.ObservationManager(recon).filter_all_points3D(4.0, 1.5)
    ba_time = time.time() - t0
    stats = {
        "num_tracks": int(tracks.shape[1]),
        "num_points3D": recon.num_points3D(),
        "mean_track_length": round(recon.compute_mean_track_length(), 2),
        "mean_reproj_error_px": round(recon.compute_mean_reprojection_error(), 3),
    }
    return recon, track_time, ba_time, stats


def export_original_resolution(recon, names, images, recon_in):
    """Rename images and bring cameras + 2D observations back to the original pixels.

    Mirrors ``VGGTWrapper._rescale_reconstruction`` (square, letterboxed frame:
    one max-side ratio, principal point back at the image centre).
    """
    done = set()
    for image_id, image in recon.images.items():
        name = names[image_id - 1]
        image.name = name
        coords = images[name]["coords"].cpu().numpy()
        orig_wh = coords[-2:]
        ratio = float(max(orig_wh)) / TRACK_RESOLUTION
        camera = recon.cameras[image.camera_id]
        if image.camera_id not in done:
            params = np.asarray(camera.params, dtype=np.float64) * ratio
            params[-2:] = orig_wh / 2
            camera.params = params
            camera.width, camera.height = int(orig_wh[0]), int(orig_wh[1])
            done.add(image.camera_id)
        for point2D in image.points2D:
            point2D.xy = (point2D.xy - coords[:2]) * ratio
    return recon


def run_scene(model_sparse: str, images_dir: str, out_sparse: str, args) -> dict:
    """Full pipeline for one scene; writes the model and ``timings.txt``."""
    t0 = time.time()
    names, images, recon_in = load_scene(model_sparse, images_dir, "cuda")
    extrinsics, intrinsics, depth, conf = model_arrays(names, images, recon_in)
    load_time = time.time() - t0

    recon, track_time, ba_time, stats = track_and_adjust(
        names, images, extrinsics, intrinsics, depth, conf, args
    )
    recon = export_original_resolution(recon, names, images, recon_in)

    os.makedirs(out_sparse, exist_ok=True)
    recon.write_text(out_sparse)
    timings = {
        "load": load_time,
        "track_establishment": track_time,
        "bundle_adjustment": ba_time,
    }
    with open(os.path.join(out_sparse, "timings.txt"), "w") as f:
        for key, value in timings.items():
            f.write(f"{key}: {value:.4f} s\n")
        f.write(f"total: {sum(timings.values()):.4f} s\n")
    stats.update({k: round(v, 2) for k, v in timings.items()})
    return stats


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vggt", help="3DFM run under benchmarks/.")
    parser.add_argument("--dataset", required=True, help="Dataset key in paths.json.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Subset of scenes.")
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--query_frame_num", type=int, default=30)
    parser.add_argument("--vis_thresh", type=float, default=0.3)
    parser.add_argument("--max_reproj_error", type=float, default=10.0)
    parser.add_argument(
        "--max_points_num",
        type=int,
        default=65536,
        help="Tracker chunk size (frames x points). The upstream default of "
        "163840 needs more than 24 GB at 150 frames / 768 px.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(_ROOT, "benchmarks", "paths.json")) as f:
        cfg = json.load(f)[args.dataset]
    base = os.path.join(_ROOT, "benchmarks", args.model, args.dataset)
    for scene in args.scenes or sorted(os.listdir(base)):
        model_sparse = os.path.join(base, scene, cfg["reconstruction_folder"])
        images_dir = os.path.join(cfg["images_path"], scene, cfg["images_folder"])
        out_sparse = os.path.join(
            _ROOT,
            "benchmarks",
            f"{args.model}_vggsfm_ba",
            args.dataset,
            scene,
            "sparse",
        )
        if not os.path.isdir(model_sparse):
            continue
        if (
            os.path.isfile(os.path.join(out_sparse, "timings.txt"))
            and not args.overwrite
        ):
            print(f"Skipping {scene}: already done", flush=True)
            continue
        print(f"=== {args.model} / {args.dataset} / {scene}", flush=True)
        print(
            f"    {run_scene(model_sparse, images_dir, out_sparse, args)}", flush=True
        )


if __name__ == "__main__":
    main()

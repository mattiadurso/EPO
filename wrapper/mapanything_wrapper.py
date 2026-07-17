"""Standalone MapAnything wrapper for EPO.

Drop-in replacement for ``VGGTWrapper`` backed by the pristine
facebookresearch/map-anything clone at ``third_party/mapanything``
(pip-installed editable, no-deps). Same ``forward()`` signature and
``(ff_data, reconstruction, depths)`` return, so the two wrappers swap 1:1:

    from wrapper.mapanything_wrapper import MapAnythingWrapper

    model = MapAnythingWrapper("facebook/map-anything")
    ff_data, reconstruction, depths = model.forward(images_path, output_path)

Differences from VGGT worth knowing:

- MapAnything preprocesses with an isotropic "cover" resize to a fixed
  aspect-ratio bucket (long side 518) followed by a small center crop, so
  the model's frame is a cropped sub-window of the original image. To keep
  EPO's image/depth pixel alignment, the wrapper pads depth (NaN) and
  confidence (0) back to the *uncropped* resized frame and shifts the
  principal point accordingly; the saved reconstruction is expressed in
  original-image pixel space like every other wrapper.
- Poses come out camera-to-world (OpenCV); the wrapper inverts to
  world-to-camera for EPO/COLMAP. Outputs are metric.
- Dense depth is saved raw (``apply_mask=False``); the sparse cloud is
  filtered by the model's non-ambiguous mask + a confidence percentile.
- Only the feed-forward path exists: ``use_ba=True`` raises
  ``NotImplementedError``.

Weights resolve via ``MapAnything.from_pretrained``: pass a Hugging Face
repo id (default ``facebook/map-anything``; the Apache-licensed variant is
``facebook/map-anything-apache``).
"""

import os
import sys

# Make this folder importable for the local ``np_to_colmap`` companion,
# regardless of caller CWD (``mapanything`` itself is pip-installed).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from base_wrapper import BaseWrapper  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402
from PIL import Image  # noqa: E402
from PIL.ImageOps import exif_transpose  # noqa: E402


class MapAnythingWrapper(BaseWrapper):
    """Wrapper class for MapAnything to perform 3D reconstruction."""

    def __init__(
        self,
        model_path: str = "facebook/map-anything",
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the MapAnything wrapper.

        Args:
            model_path: Hugging Face repo id (e.g. ``facebook/map-anything``)
                resolved via ``MapAnything.from_pretrained``.
            cuda_id: CUDA device index (CPU fallback if unavailable).
            seed: Random seed for reproducibility.
            oom_safe: Unused (no BA path); kept for interface parity.
        """
        self.model_path = model_path
        self.seed = seed
        self._set_seed(seed)
        self.oom_safe = oom_safe

        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        )

        self.model = self._load_model(self.model_path)

        print(f"MapAnythingWrapper initialized on {self.device}")

    def _load_model(self, model_path: str) -> MapAnything:
        """Load MapAnything from a Hugging Face repo id."""
        model = MapAnything.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        print(f"MapAnything model loaded from {model_path}")
        return model

    @staticmethod
    def _crop_geometry(
        orig_w: int, orig_h: int, target_w: int, target_h: int
    ) -> tuple[int, int, int, int]:
        """Replicate MapAnything's cover-resize + centered-crop geometry.

        Mirrors ``crop_resize_if_necessary`` (no-intrinsics path): isotropic
        scale so the resized image covers the target, floor-rounded dims,
        centered crop with floor-division offsets.

        Returns:
            ``(resized_w, resized_h, crop_left, crop_top)``.
        """
        scale = max(target_w / orig_w, target_h / orig_h) + 1e-8
        resized_w = int(np.floor(orig_w * scale))
        resized_h = int(np.floor(orig_h * scale))
        crop_left = (resized_w - target_w) // 2
        crop_top = (resized_h - target_h) // 2
        return resized_w, resized_h, crop_left, crop_top

    def _run_mapanything(self, image_paths: list[str]) -> dict:
        """Run MapAnything inference (preprocessing included) on image paths.

        Returns a dict of numpy arrays in the model's *cropped* processed
        frame: ``extrinsic`` (N, 3, 4) world-to-camera, ``intrinsic``
        (N, 3, 3), ``depth_map`` / ``depth_conf`` (N, H, W), ``valid_mask``
        (N, H, W), ``points`` (N, H, W, 3) world coordinates,
        ``processed_images`` (N, H, W, 3) uint8.
        """
        views = load_images(image_paths)
        predictions = self.model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=False,  # keep dense depth raw; mask only the cloud
            apply_confidence_mask=False,
        )

        extrinsic, intrinsic, depth, conf, valid, points, images = (
            [] for _ in range(7)
        )
        for pred in predictions:
            c2w = pred["camera_poses"][0].float().cpu().numpy()
            r_wc = c2w[:3, :3].T
            t_wc = -r_wc @ c2w[:3, 3]
            extrinsic.append(np.concatenate([r_wc, t_wc[:, None]], axis=1))
            intrinsic.append(pred["intrinsics"][0].float().cpu().numpy())
            depth.append(pred["depth_z"][0, ..., 0].float().cpu().numpy())
            conf.append(pred["conf"][0].float().cpu().numpy())
            valid.append(pred["non_ambiguous_mask"][0].cpu().numpy() > 0)
            points.append(pred["pts3d"][0].float().cpu().numpy())
            images.append(
                (pred["img_no_norm"][0].float().cpu().numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )

        torch.cuda.empty_cache()

        return {
            "extrinsic": np.stack(extrinsic).astype(np.float32),
            "intrinsic": np.stack(intrinsic).astype(np.float32),
            "depth_map": np.stack(depth).astype(np.float32),
            "depth_conf": np.stack(conf).astype(np.float32),
            "valid_mask": np.stack(valid),
            "points": np.stack(points).astype(np.float32),
            "processed_images": np.stack(images),
        }

    def _rescale_camera_params(
        self,
        params: np.ndarray,
        orig_wh: tuple[int, int],
        proc_hw: tuple[int, int],
    ) -> np.ndarray:
        """Per-axis rescale that also undoes MapAnything's centered crop.

        The principal point is first shifted into the uncropped resized frame,
        then everything is scaled by the per-axis resized→original ratios.
        """
        orig_w, orig_h = orig_wh
        proc_h, proc_w = proc_hw
        resized_w, resized_h, left, top = self._crop_geometry(
            orig_w, orig_h, proc_w, proc_h
        )
        return self._rescale_camera_params_per_axis(
            params, orig_wh, (resized_h, resized_w), crop_offset=(left, top)
        )

    def _pad_to_uncropped(
        self,
        preds: dict,
        original_sizes: list[tuple[int, int]],
    ) -> tuple[dict, list[np.ndarray]]:
        """Pad depth/confidence back to the uncropped resized frame.

        MapAnything center-crops a few rows/columns after its isotropic
        resize; EPO pairs depth maps with *full* images, so the crop would
        shift everything by a few pixels. Padding with NaN depth (0
        confidence) restores alignment — EPO's NaN-safe sampling ignores
        the unknown border. Also returns per-image intrinsics with the
        principal point shifted into the uncropped frame.

        Returns:
            ``(padded, intrinsics_uncropped)`` where ``padded`` maps
            ``"depth"``/``"confidence"`` to per-image arrays.
        """
        n, target_h, target_w = preds["depth_map"].shape
        depths_padded, confs_padded, intrinsics_unc = [], [], []
        for i in range(n):
            orig_w, orig_h = original_sizes[i]
            resized_w, resized_h, left, top = self._crop_geometry(
                orig_w, orig_h, target_w, target_h
            )
            d = np.full((resized_h, resized_w), np.nan, dtype=np.float32)
            c = np.zeros((resized_h, resized_w), dtype=np.float32)
            d[top : top + target_h, left : left + target_w] = preds["depth_map"][i]
            c[top : top + target_h, left : left + target_w] = preds["depth_conf"][i]
            depths_padded.append(d)
            confs_padded.append(c)

            k = preds["intrinsic"][i].copy()
            k[0, 2] += left
            k[1, 2] += top
            intrinsics_unc.append(k)

        return (
            {"depth": depths_padded, "confidence": confs_padded},
            intrinsics_unc,
        )

    def _conf_mask(self, preds: dict, conf_thres: float) -> np.ndarray:
        """Depth validity + the model's non-ambiguous mask + a confidence percentile."""
        depth_map = preds["depth_map"]
        conf_thresh = np.percentile(preds["depth_conf"], conf_thres)
        mask = np.isfinite(depth_map) & (depth_map > 0)
        mask &= preds["valid_mask"]
        return mask & (preds["depth_conf"] >= conf_thresh)

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Assemble EPO's feed-forward dict from raw MapAnything outputs.

        Normalizes MapAnything's cover-resize + centered-crop frame into the
        per-image entries ``BaseWrapper._build_ff_data`` consumes.
        Everything lives in the *uncropped* resized frame: the image is resized
        (never cropped) and the depth/confidence are the NaN-padded maps
        (``preds["padded"]``), so image and depth stay pixel-aligned. Keyed by
        the relative image path (``"cam_id/image_name"``).
        """
        target_h, target_w = preds["depth_map"].shape[-2:]
        entries = []

        for i, key in enumerate(base_image_paths):
            orig_w, orig_h = preds["original_sizes"][i]
            resized_w, resized_h, _, _ = self._crop_geometry(
                orig_w, orig_h, target_w, target_h
            )

            entries.append(
                {
                    "key": key,
                    "image_path": image_paths[i],
                    "resize_hw": (resized_h, resized_w),
                    "crop_box": None,
                    "depth": preds["padded"]["depth"][i],
                    "confidence": preds["padded"]["confidence"][i],
                    "pose": preds["extrinsic"][i].copy(),
                    "intrinsic": preds["intrinsics_unc"][i].copy(),
                }
            )

        return entries

    @torch.no_grad()
    def forward(
        self,
        images_path: str,
        output_path: str,
        max_images: int = 150,
        use_ba: bool = False,
        save_depth: bool = True,
        save: bool = True,
        # BA-specific parameters (unsupported; kept for signature parity)
        max_reproj_error: float = 10.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.3,
        query_frame_num: int = 30,
        max_query_pts: int = 4096,
        fine_tracking: bool = False,
        # Non-BA parameters
        conf_thres_percentile: float = 10.0,
        max_points_for_colmap: int = 100_000,
    ):
        """Run MapAnything reconstruction on images and save results.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images (randomly sampled if exceeded).
            use_ba: Unsupported for MapAnything; must stay False.
            save_depth: Whether to write ``depths.pth`` (only when ``save``).
            save: Whether to write the reconstruction / depths / timings to
                disk. Set ``False`` to skip all disk I/O; the returned
                ``(ff_data, reconstruction, depths)`` can be written later.
            max_reproj_error: Ignored (BA only).
            shared_camera: Ignored (cameras are per-image, like VGGT non-BA).
            camera_type: Ignored (PINHOLE is always used).
            vis_thresh: Ignored (BA only).
            query_frame_num: Ignored (BA only).
            max_query_pts: Ignored (BA only).
            fine_tracking: Ignored (BA only).
            conf_thres_percentile: Confidence percentile below which pixels
                are dropped from the sparse point cloud (MapAnything's README
                filtering default is the 10th percentile).
            max_points_for_colmap: Maximum 3D points for COLMAP.

        Returns:
            ``(ff_data, reconstruction, depths)`` — same contract as
            ``VGGTWrapper.forward``.
        """
        if use_ba:
            raise NotImplementedError(
                "MapAnythingWrapper has no bundle-adjustment path; use "
                "use_ba=False (or the VGGT wrappers for BA)."
            )

        os.makedirs(output_path, exist_ok=True)

        timings = {}
        t_total_start = time.time()

        # Find and sample images
        t_start = time.time()
        image_paths = self._find_images(images_path)
        image_paths = self._sample_images(image_paths, max_images)
        timings["find_and_sample_images"] = time.time() - t_start

        # Get base paths and original sizes (EXIF-corrected, matching
        # MapAnything's exif_transpose preprocessing).
        base_image_paths = [os.path.relpath(path, images_path) for path in image_paths]
        original_sizes = [exif_transpose(Image.open(p)).size for p in image_paths]

        # Run MapAnything (its load_images handles preprocessing).
        print("Running MapAnything model...")
        t_start = time.time()
        preds = self._run_mapanything(image_paths)
        timings["run_mapanything"] = time.time() - t_start

        print("Running reconstruction without Bundle Adjustment...")
        t_start = time.time()
        proc_hw = preds["depth_map"].shape[-2:]
        preds["points_rgb"] = preds["processed_images"]
        reconstruction = self._reconstruct(
            preds, conf_thres_percentile, max_points_for_colmap
        )
        timings["reconstruction_without_ba"] = time.time() - t_start

        # Pad depth/confidence back to the uncropped resized frame so image
        # and depth stay pixel-aligned for EPO.
        padded, intrinsics_unc = self._pad_to_uncropped(preds, original_sizes)

        # Always build the depths dict for downstream tasks (disk save).
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            stem = Path(img_path).with_suffix("").as_posix()
            depths[stem] = {
                "depth": torch.from_numpy(padded["depth"][i]).float(),
                "confidence": torch.from_numpy(padded["confidence"][i]).float(),
            }

        # Build EPO's feed-forward dict: MapAnything depth/pose/intrinsic
        # plus the original sharp image for the edge detector.
        t_start = time.time()
        preds["padded"] = padded
        preds["intrinsics_unc"] = intrinsics_unc
        preds["original_sizes"] = original_sizes
        ff_data = self._build_ff_data(preds, base_image_paths, image_paths)
        timings["build_ff_data"] = time.time() - t_start
        gc.collect()
        torch.cuda.empty_cache()

        # Rescale reconstruction to original resolution
        if reconstruction is not None:
            t_start = time.time()
            reconstruction = self._rescale_reconstruction(
                reconstruction, base_image_paths, original_sizes, proc_hw
            )
            timings["rescale_reconstruction"] = time.time() - t_start

            t_start = time.time()
            if save:
                os.makedirs(output_path, exist_ok=True)
                reconstruction.write_text(output_path)

                if save_depth:
                    print("Saving depth maps...")
                    torch.save(depths, os.path.join(output_path, "depths.pth"))

            timings["save_reconstruction"] = time.time() - t_start
            timings["total"] = time.time() - t_total_start
            if save:
                print(f"Reconstruction saved to {output_path}")
        else:
            timings["rescale_reconstruction"] = 0.0
            timings["save_reconstruction"] = 0.0
            timings["total"] = time.time() - t_total_start
            print("No reconstruction could be built.")

        # Print timing summary
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        print(
            f"Find and sample images:      {timings['find_and_sample_images']:>8.2f}s"
        )
        print(f"Run MapAnything model:       {timings['run_mapanything']:>8.2f}s")
        print(
            f"Reconstruction (without BA): {timings['reconstruction_without_ba']:>8.2f}s"
        )
        print(f"Build ff_data (EPO input):   {timings['build_ff_data']:>8.2f}s")
        print(
            f"Rescale reconstruction:      {timings['rescale_reconstruction']:>8.2f}s"
        )
        print(f"Save reconstruction:         {timings['save_reconstruction']:>8.2f}s")
        print("-" * 60)
        print(f"TOTAL TIME:                  {timings['total']:>8.2f}s")
        print("=" * 60 + "\n")

        # save timings to a text file
        if save:
            with open(os.path.join(output_path, "timings.txt"), "w") as f:
                for key, value in timings.items():
                    f.write(f"{key}: {value:.4f} s\n")

        del preds, padded
        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown (e.g. `run_mapanything`) for callers
        # that want to report inference time separately from I/O.
        self.last_timings = timings

        return ff_data, reconstruction, depths


if __name__ == "__main__":
    MapAnythingWrapper._cli_main(default_model_path="facebook/map-anything")

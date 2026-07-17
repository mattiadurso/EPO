"""Standalone Depth Anything 3 wrapper for EPO.

Drop-in replacement for ``VGGTWrapper`` backed by the pristine
ByteDance-Seed/Depth-Anything-3 clone at ``third_party/depth_anything_3``.
Same ``forward()`` signature and ``(ff_data, reconstruction, depths)``
return, so the two wrappers swap 1:1:

    from wrapper.da3_wrapper import DA3Wrapper

    model = DA3Wrapper("depth-anything/DA3-LARGE")
    ff_data, reconstruction, depths = model.forward(images_path, output_path)

Differences from VGGT worth knowing:

- DA3 runs on aspect-preserving (non-square) inputs: the long side is
  resized to ``process_res`` (default 504) and each dimension rounded to
  the nearest multiple of 14, so depth maps / ``ff_data`` keep the image
  aspect ratio instead of VGGT's 518x518 square.
- Extrinsics are OpenCV/COLMAP world-to-camera (same convention as VGGT);
  intrinsics are PINHOLE in processed-pixel space with a *predicted*
  principal point (not forced to the image centre).
- Confidence is thresholded by percentile (DA3's own convention) rather
  than by absolute value.
- Only the feed-forward path exists: ``use_ba=True`` raises
  ``NotImplementedError`` (DA3 ships no track-based BA pipeline).

Weights resolve via ``DepthAnything3.from_pretrained``: pass a Hugging Face
repo id (e.g. ``depth-anything/DA3-LARGE``, cached under
``~/.cache/huggingface``) or a local directory containing
``model.safetensors`` + ``config.json``.
"""

import os
import sys

# Make the pristine Depth-Anything-3 clone importable as
# ``depth_anything_3`` and this folder importable for the local
# ``np_to_colmap`` companion, regardless of caller CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "third_party", "depth_anything_3", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from base_wrapper import BaseWrapper  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from PIL import Image  # noqa: E402


class DA3Wrapper(BaseWrapper):
    """Wrapper class for Depth Anything 3 to perform 3D reconstruction."""

    def __init__(
        self,
        model_path: str = "depth-anything/DA3-LARGE",
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the DA3 wrapper.

        Args:
            model_path: Hugging Face repo id (e.g. ``depth-anything/DA3-LARGE``)
                or local directory with ``model.safetensors`` + ``config.json``,
                resolved via ``DepthAnything3.from_pretrained``.
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

        # DA3's native processing resolution (long side, multiple of 14).
        self.process_res = 504
        self.patch_size = 14

        print(f"DA3Wrapper initialized on {self.device}")

    def _load_model(self, model_path: str) -> DepthAnything3:
        """Load DA3 from a Hugging Face repo id or a local weights directory."""
        model = DepthAnything3.from_pretrained(model_path)
        model.eval()
        model = model.to(self.device)
        print(f"DA3 model loaded from {model_path}")
        return model

    def _processed_size(self, width: int, height: int) -> tuple[int, int]:
        """Replicate DA3's ``upper_bound_resize`` sizing for one image.

        Long side scaled to ``process_res``, then each dimension rounded to
        the nearest multiple of ``patch_size`` (ties round up), mirroring
        ``InputProcessor._resize_longest_side`` + ``_make_divisible_by_resize``.

        Returns:
            ``(height, width)`` of the processed image.
        """
        scale = self.process_res / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        return (
            max(self.patch_size, nearest_multiple(new_h, self.patch_size)),
            max(self.patch_size, nearest_multiple(new_w, self.patch_size)),
        )

    def _run_da3(
        self, image_paths: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run DA3 inference (preprocessing included) on image paths.

        Returns:
            extrinsic (N, 3, 4) world-to-camera, intrinsic (N, 3, 3) in
            processed-pixel space, depth_map (N, H, W), depth_conf (N, H, W),
            processed_images (N, H, W, 3) uint8.
        """
        prediction = self.model.inference(
            image_paths,
            process_res=self.process_res,
        )

        extrinsic = np.asarray(prediction.extrinsics, dtype=np.float32)[:, :3, :4]
        intrinsic = np.asarray(prediction.intrinsics, dtype=np.float32)
        depth_map = np.asarray(prediction.depth, dtype=np.float32)
        depth_conf = np.asarray(prediction.conf, dtype=np.float32)
        processed_images = prediction.processed_images

        torch.cuda.empty_cache()

        return extrinsic, intrinsic, depth_map, depth_conf, processed_images

    def _rescale_camera_params(
        self,
        params: np.ndarray,
        orig_wh: tuple[int, int],
        proc_hw: tuple[int, int],
    ) -> np.ndarray:
        """Per-axis rescale: DA3's frame keeps the aspect ratio and is uncropped.

        fx/cx and fy/cy scale by their own ratios (matching DA3's own COLMAP
        export) and DA3's *predicted* principal point is kept, rather than the
        VGGT family's single max-side ratio + re-centred principal point.
        """
        return self._rescale_camera_params_per_axis(params, orig_wh, proc_hw)

    def _conf_mask(self, preds: dict, conf_thres: float) -> np.ndarray:
        """Depth validity + a confidence *percentile* (DA3's own convention)."""
        depth_map = preds["depth_map"]
        conf_thresh = np.percentile(preds["depth_conf"], conf_thres)
        mask = np.isfinite(depth_map) & (depth_map > 0)
        return mask & (preds["depth_conf"] >= conf_thresh)

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Assemble EPO's feed-forward dict from raw DA3 outputs.

        Normalizes DA3's aspect-preserving frame into the per-image entries
        ``BaseWrapper._build_ff_data`` consumes: each image is resized to
        its DA3 processed size and, when the batch mixed sizes (DA3
        center-crops to the smallest), center-cropped the same way, so image
        and depth stay pixel-aligned. Depth/confidence/pose/intrinsic come
        straight from DA3 in its processed-pixel space. Keyed by the relative
        image path (``"cam_id/image_name"``).
        """
        out_h, out_w = preds["depth_map"].shape[-2:]
        entries = []

        for i, key in enumerate(base_image_paths):
            width, height = preds["original_sizes"][i]
            new_h, new_w = self._processed_size(width, height)

            # Center-crop to the batch-unified size (identity when equal).
            top = max((new_h - out_h) // 2, 0)
            left = max((new_w - out_w) // 2, 0)

            entries.append(
                {
                    "key": key,
                    "image_path": image_paths[i],
                    "resize_hw": (new_h, new_w),
                    "crop_box": (top, left, out_h, out_w),
                    "depth": preds["depth_map"][i],
                    "confidence": preds["depth_conf"][i],
                    "pose": preds["extrinsic"][i].copy(),
                    "intrinsic": preds["intrinsic"][i].copy(),
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
        conf_thres_percentile: float = 40.0,
        max_points_for_colmap: int = 100_000,
    ):
        """Run DA3 reconstruction on images and save results.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images (randomly sampled if exceeded).
            use_ba: Unsupported for DA3; must stay False.
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
            conf_thres_percentile: Confidence percentile below which depth
                pixels are dropped from the sparse point cloud (DA3 uses a
                percentile, not VGGT's absolute threshold).
            max_points_for_colmap: Maximum 3D points for COLMAP.

        Returns:
            ``(ff_data, reconstruction, depths)`` — same contract as
            ``VGGTWrapper.forward``.
        """
        if use_ba:
            raise NotImplementedError(
                "DA3Wrapper has no bundle-adjustment path; use use_ba=False "
                "(or the VGGT wrappers for BA)."
            )

        os.makedirs(output_path, exist_ok=True)

        timings = {}
        t_total_start = time.time()

        # Find and sample images
        t_start = time.time()
        image_paths = self._find_images(images_path)
        image_paths = self._sample_images(image_paths, max_images)
        timings["find_and_sample_images"] = time.time() - t_start

        # Get base paths and original sizes (header read only)
        base_image_paths = [os.path.relpath(path, images_path) for path in image_paths]
        original_sizes = [Image.open(p).size for p in image_paths]

        # Run DA3 (its InputProcessor handles loading + preprocessing).
        print("Running DA3 model...")
        t_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, processed_images = self._run_da3(
            image_paths
        )
        timings["run_da3"] = time.time() - t_start

        print("Running reconstruction without Bundle Adjustment...")
        t_start = time.time()
        proc_hw = depth_map.shape[-2:]
        # No "points": DA3 predicts depth, so only the selected pixels are
        # unprojected (BaseWrapper._reconstruct falls back to _unproject_masked).
        recon_preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "points_rgb": processed_images,
        }
        reconstruction = self._reconstruct(
            recon_preds, conf_thres_percentile, max_points_for_colmap
        )
        timings["reconstruction_without_ba"] = time.time() - t_start

        # Always build the depths dict for downstream tasks (disk save).
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            stem = Path(img_path).with_suffix("").as_posix()
            depths[stem] = {
                "depth": torch.from_numpy(depth_map[i]).float(),
                "confidence": torch.from_numpy(depth_conf[i]).float(),
            }

        # Build EPO's feed-forward dict: DA3 depth/pose/intrinsic plus the
        # original sharp image for the edge detector.
        t_start = time.time()
        ff_preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "original_sizes": original_sizes,
        }
        ff_data = self._build_ff_data(ff_preds, base_image_paths, image_paths)
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
        print(f"Run DA3 model:               {timings['run_da3']:>8.2f}s")
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

        del extrinsic, intrinsic, depth_map, depth_conf, processed_images
        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown (e.g. `run_da3`) for callers that want
        # to report inference time separately from I/O.
        self.last_timings = timings

        return ff_data, reconstruction, depths


if __name__ == "__main__":
    DA3Wrapper._cli_main(default_model_path="depth-anything/DA3-LARGE")

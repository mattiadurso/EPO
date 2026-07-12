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
import glob  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from np_to_colmap import batch_np_matrix_to_pycolmap_wo_track  # noqa: E402
from PIL import Image  # noqa: E402


def _randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """Keep at most ``max_trues`` True entries of ``mask``, chosen uniformly."""
    true_idx = np.flatnonzero(mask)
    if true_idx.size <= max_trues:
        return mask
    keep = np.random.choice(true_idx, size=max_trues, replace=False)
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep] = True
    return limited.reshape(mask.shape)


class DA3Wrapper:
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

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: str) -> DepthAnything3:
        """Load DA3 from a Hugging Face repo id or a local weights directory."""
        model = DepthAnything3.from_pretrained(model_path)
        model.eval()
        model = model.to(self.device)
        print(f"DA3 model loaded from {model_path}")
        return model

    def _find_images(self, images_path: str) -> list[str]:
        """Find all images in the given path, including subdirectories.

        Args:
            images_path: Path to directory containing images.

        Returns:
            List of image file paths.
        """
        valid_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        image_paths = []

        for ext in valid_extensions:
            # Search in root and one level deep
            image_paths.extend(glob.glob(os.path.join(images_path, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(images_path, "*", f"*.{ext}")))

        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))

        if len(image_paths) == 0:
            raise ValueError(
                f"No images found in {images_path}. Path {images_path} is invalid or empty."
            )

        print(f"Found {len(image_paths)} images in {images_path}")
        return image_paths

    def _sample_images(
        self, image_paths: list[str], max_images: int | None = None
    ) -> list[str]:
        """Randomly sample images if needed.

        Args:
            image_paths: List of all image paths.
            max_images: Maximum number of images to use. None means use all.

        Returns:
            Sampled list of image paths.
        """
        max_images = max_images if max_images > 0 else 100_000
        if max_images is not None and len(image_paths) > max_images:
            sampled_paths = random.sample(image_paths, max_images)
            sampled_paths = sorted(sampled_paths)  # Keep sorted order
            print(f"Randomly sampled {max_images} images from {len(image_paths)}")
            return sampled_paths
        return image_paths

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

    def _unproject_masked(
        self,
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unproject masked depth pixels to world points.

        Args:
            depth_map: (N, H, W) depth.
            intrinsic: (N, 3, 3) pinhole intrinsics (processed-pixel space).
            extrinsic: (N, 3, 4) world-to-camera.
            mask: (N, H, W) boolean selection.

        Returns:
            points_3d (P, 3) world points and points_xyf (P, 3) with
            per-point ``(x, y, frame_idx)`` in processed-pixel coordinates.
        """
        points, xyf = [], []
        for i in range(len(depth_map)):
            v, u = np.nonzero(mask[i])
            if v.size == 0:
                continue
            z = depth_map[i, v, u]
            fx, fy = intrinsic[i, 0, 0], intrinsic[i, 1, 1]
            cx, cy = intrinsic[i, 0, 2], intrinsic[i, 1, 2]
            x_cam = (u - cx) / fx * z
            y_cam = (v - cy) / fy * z
            pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
            r_cw = extrinsic[i, :3, :3]
            t_cw = extrinsic[i, :3, 3]
            points.append((pts_cam - t_cw) @ r_cw)  # R^T (Xc - t)
            xyf.append(np.stack([u, v, np.full_like(u, i)], axis=-1).astype(np.float64))
        return (
            np.concatenate(points, axis=0),
            np.concatenate(xyf, axis=0),
        )

    def _reconstruct_without_ba(
        self,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        processed_images: np.ndarray,
        conf_thres_percentile: float,
        max_points_for_colmap: int,
    ):
        """Build a track-less pycolmap reconstruction from DA3 outputs."""
        height, width = depth_map.shape[-2:]
        image_size = np.array([width, height])

        # Filter by depth validity + confidence percentile (DA3 convention).
        conf_thresh = np.percentile(depth_conf, conf_thres_percentile)
        conf_mask = np.isfinite(depth_map) & (depth_map > 0)
        conf_mask &= depth_conf >= conf_thresh
        conf_mask = _randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d, points_xyf = self._unproject_masked(
            depth_map, intrinsic, extrinsic, conf_mask
        )
        points_rgb = processed_images[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )

        return reconstruction, (height, width)

    def _rescale_reconstruction(
        self,
        reconstruction,
        base_image_paths: list[str],
        original_sizes: list[tuple[int, int]],
        proc_hw: tuple[int, int],
    ):
        """Rescale cameras to original resolutions and rename images.

        DA3's processed frame is non-square, so fx/cx and fy/cy are scaled
        by the per-axis ratios (matching DA3's own COLMAP export) instead of
        VGGT's single max-side ratio. Points2D stay in processed-pixel
        coordinates, mirroring ``VGGTWrapper``'s non-BA behavior.
        """
        proc_h, proc_w = proc_hw
        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = base_image_paths[pyimageid - 1]

            orig_w, orig_h = original_sizes[pyimageid - 1]
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h

            fx, fy, cx, cy = pycamera.params  # PINHOLE, per-image camera
            pycamera.params = np.array(
                [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y]
            )
            pycamera.width = orig_w
            pycamera.height = orig_h

        return reconstruction

    def _build_ff_data(
        self,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        base_image_paths: list[str],
        image_paths: list[str],
        original_sizes: list[tuple[int, int]],
    ):
        """Assemble EPO's feed-forward dict from raw DA3 outputs.

        Follows the ``VGGTWrapper._build_ff_data`` recipe: the *original*
        sharp pixels (torchvision decode, PIL fallback, antialiased BICUBIC
        resize) feed the edge detector, while depth/confidence/pose/intrinsic
        come straight from DA3 in its processed-pixel space. Each image is
        resized to its DA3 processed size and, when the batch mixed sizes
        (DA3 center-crops to the smallest), center-cropped the same way, so
        image and depth stay pixel-aligned. Keyed by the relative image path
        (``"cam_id/image_name"``).
        """
        from concurrent.futures import ThreadPoolExecutor

        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize as tv_resize

        out_h, out_w = depth_map.shape[-2:]

        def _process_one(i):
            """Build one image's ff_data entry (independent per image)."""
            base = base_image_paths[i]
            width, height = original_sizes[i]
            new_h, new_w = self._processed_size(width, height)

            # Decode -> CHW uint8 RGB, same decoder + fallback as VGGTWrapper.
            try:
                rgb = read_image(image_paths[i], mode=ImageReadMode.UNCHANGED)
                if rgb.shape[0] == 1:
                    rgb = rgb.expand(3, -1, -1).contiguous()
                elif rgb.shape[0] == 4:
                    # RGBA -> blend onto white, drop alpha.
                    a = rgb[3:4].float() / 255.0
                    rgb = (
                        (rgb[:3].float() * a + 255.0 * (1.0 - a))
                        .clamp_(0, 255)
                        .to(torch.uint8)
                    )
                elif rgb.shape[0] != 3:
                    raise RuntimeError("defer to PIL")
            except RuntimeError:
                img = Image.open(image_paths[i])
                if img.mode == "RGBA":
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(bg, img)
                arr = np.asarray(img.convert("RGB"))
                rgb = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            img_t = (
                tv_resize(
                    rgb,
                    [new_h, new_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                .float()
                .div_(255.0)
            )

            # Center-crop to the batch-unified size (identity when equal).
            crop_top = max((new_h - out_h) // 2, 0)
            crop_left = max((new_w - out_w) // 2, 0)
            img_t = img_t[:, crop_top : crop_top + out_h, crop_left : crop_left + out_w]

            return base, {
                "image": img_t,
                "depth": torch.from_numpy(depth_map[i]).float(),
                "confidence": torch.from_numpy(depth_conf[i]).float(),
                "pose": torch.from_numpy(extrinsic[i].copy()).float(),
                "intrinsic": torch.from_numpy(intrinsic[i].copy()).float(),
            }

        # Decode + resize is the bottleneck and releases the GIL, so thread it.
        n = len(base_image_paths)
        max_workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_process_one, range(n)))

        return {base: entry for base, entry in results}

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
        reconstruction, proc_hw = self._reconstruct_without_ba(
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            processed_images,
            conf_thres_percentile,
            max_points_for_colmap,
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
        ff_data = self._build_ff_data(
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            base_image_paths,
            image_paths,
            original_sizes,
        )
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

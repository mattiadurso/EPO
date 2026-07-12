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
import glob  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402
from np_to_colmap import batch_np_matrix_to_pycolmap_wo_track  # noqa: E402
from PIL import Image  # noqa: E402
from PIL.ImageOps import exif_transpose  # noqa: E402


def _randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """Keep at most ``max_trues`` True entries of ``mask``, chosen uniformly."""
    true_idx = np.flatnonzero(mask)
    if true_idx.size <= max_trues:
        return mask
    keep = np.random.choice(true_idx, size=max_trues, replace=False)
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep] = True
    return limited.reshape(mask.shape)


class MapAnythingWrapper:
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

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: str) -> MapAnything:
        """Load MapAnything from a Hugging Face repo id."""
        model = MapAnything.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        print(f"MapAnything model loaded from {model_path}")
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

    def _reconstruct_without_ba(
        self,
        preds: dict,
        conf_thres_percentile: float,
        max_points_for_colmap: int,
    ):
        """Build a track-less pycolmap reconstruction from MapAnything outputs."""
        depth_map = preds["depth_map"]
        height, width = depth_map.shape[-2:]
        image_size = np.array([width, height])

        conf_thresh = np.percentile(preds["depth_conf"], conf_thres_percentile)
        conf_mask = np.isfinite(depth_map) & (depth_map > 0)
        conf_mask &= preds["valid_mask"]
        conf_mask &= preds["depth_conf"] >= conf_thresh
        conf_mask = _randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d, points_xyf = [], []
        for i in range(len(depth_map)):
            v, u = np.nonzero(conf_mask[i])
            if v.size == 0:
                continue
            points_3d.append(preds["points"][i, v, u])
            points_xyf.append(
                np.stack([u, v, np.full_like(u, i)], axis=-1).astype(np.float64)
            )
        points_3d = np.concatenate(points_3d, axis=0)
        points_xyf = np.concatenate(points_xyf, axis=0)
        points_rgb = preds["processed_images"][conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            preds["extrinsic"],
            preds["intrinsic"],
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

        Undoes the cover-resize + center crop: the principal point is first
        shifted into the uncropped resized frame, then everything is scaled
        by the per-axis resized→original ratios. Points2D stay in
        processed-pixel coordinates, mirroring ``VGGTWrapper``'s non-BA
        behavior.
        """
        proc_h, proc_w = proc_hw
        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = base_image_paths[pyimageid - 1]

            orig_w, orig_h = original_sizes[pyimageid - 1]
            resized_w, resized_h, left, top = self._crop_geometry(
                orig_w, orig_h, proc_w, proc_h
            )
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h

            fx, fy, cx, cy = pycamera.params  # PINHOLE, per-image camera
            pycamera.params = np.array(
                [
                    fx * scale_x,
                    fy * scale_y,
                    (cx + left) * scale_x,
                    (cy + top) * scale_y,
                ]
            )
            pycamera.width = orig_w
            pycamera.height = orig_h

        return reconstruction

    def _build_ff_data(
        self,
        preds: dict,
        padded: dict,
        intrinsics_unc: list[np.ndarray],
        base_image_paths: list[str],
        image_paths: list[str],
        original_sizes: list[tuple[int, int]],
    ):
        """Assemble EPO's feed-forward dict from raw MapAnything outputs.

        Follows the ``VGGTWrapper._build_ff_data`` recipe: the *original*
        sharp pixels (torchvision decode, PIL fallback, antialiased BICUBIC
        resize) feed the edge detector, while depth/confidence/pose/intrinsic
        come from MapAnything. Everything lives in the *uncropped* resized
        frame (NaN-padded depth), so image and depth stay pixel-aligned.
        Keyed by the relative image path (``"cam_id/image_name"``).
        """
        from concurrent.futures import ThreadPoolExecutor

        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize as tv_resize

        target_h, target_w = preds["depth_map"].shape[-2:]

        def _process_one(i):
            """Build one image's ff_data entry (independent per image)."""
            base = base_image_paths[i]
            orig_w, orig_h = original_sizes[i]
            resized_w, resized_h, _, _ = self._crop_geometry(
                orig_w, orig_h, target_w, target_h
            )

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
                    [resized_h, resized_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                .float()
                .div_(255.0)
            )

            return base, {
                "image": img_t,
                "depth": torch.from_numpy(padded["depth"][i]).float(),
                "confidence": torch.from_numpy(padded["confidence"][i]).float(),
                "pose": torch.from_numpy(preds["extrinsic"][i].copy()).float(),
                "intrinsic": torch.from_numpy(intrinsics_unc[i].copy()).float(),
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
        reconstruction, proc_hw = self._reconstruct_without_ba(
            preds,
            conf_thres_percentile,
            max_points_for_colmap,
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
        ff_data = self._build_ff_data(
            preds,
            padded,
            intrinsics_unc,
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

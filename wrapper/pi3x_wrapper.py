"""Standalone Pi3X (π³-X) wrapper for EPO.

Drop-in replacement for ``VGGTWrapper`` backed by the pristine yyfz/Pi3
clone at ``third_party/pi3``. Same ``forward()`` signature and
``(ff_data, reconstruction, depths)`` return, so the two wrappers swap 1:1:

    from wrapper.pi3x_wrapper import Pi3XWrapper

    model = Pi3XWrapper("yyfz233/Pi3X")
    ff_data, reconstruction, depths = model.forward(images_path, output_path)

Differences from VGGT worth knowing:

- Pi3 resizes every image to one uniform size derived from the *first*
  image: the area is scaled to ~255k pixels and each dimension rounded to
  a multiple of 14 (no cropping, so mixed-aspect batches get slightly
  distorted — Pi3's own convention).
- The model predicts metric-scaled camera-to-world poses (OpenCV); the
  wrapper inverts them to world-to-camera for EPO/COLMAP.
- There is no depth head: z-depth comes from the metric-scaled per-pixel
  camera-frame points (``local_points``). Intrinsics are recovered from
  the point-ray field with a centred principal point.
- Confidence is a logit map; the wrapper stores ``sigmoid(conf)`` and the
  sparse cloud keeps pixels with probability > ``conf_thres_prob`` (0.1,
  the repo demo's threshold) that also survive the repo's depth/normal
  edge filter.
- Only the feed-forward path exists: ``use_ba=True`` raises
  ``NotImplementedError``.

Weights resolve via ``Pi3X.from_pretrained``: pass a Hugging Face repo id
(default ``yyfz233/Pi3X``, cached under ``~/.cache/huggingface``).
"""

import os
import sys

# Make the pristine Pi3 clone importable as ``pi3`` and this folder
# importable for the local ``np_to_colmap`` companion, regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "third_party", "pi3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc  # noqa: E402
import glob  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from np_to_colmap import batch_np_matrix_to_pycolmap_wo_track  # noqa: E402
from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.geometry import (  # noqa: E402
    depth_normal_edge,
    recover_intrinsic_from_rays_d,
    se3_inverse,
)
from PIL import Image  # noqa: E402

# Pi3's target pixel budget (``load_images_as_tensor`` PIXEL_LIMIT).
_PIXEL_LIMIT = 255_000
_PATCH = 14


def _randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """Keep at most ``max_trues`` True entries of ``mask``, chosen uniformly."""
    true_idx = np.flatnonzero(mask)
    if true_idx.size <= max_trues:
        return mask
    keep = np.random.choice(true_idx, size=max_trues, replace=False)
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep] = True
    return limited.reshape(mask.shape)


class Pi3XWrapper:
    """Wrapper class for Pi3X to perform 3D reconstruction."""

    def __init__(
        self,
        model_path: str = "yyfz233/Pi3X",
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the Pi3X wrapper.

        Args:
            model_path: Hugging Face repo id (e.g. ``yyfz233/Pi3X``) resolved
                via ``Pi3X.from_pretrained``.
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

        print(f"Pi3XWrapper initialized on {self.device}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: str) -> Pi3X:
        """Load Pi3X from a Hugging Face repo id (image-only branch)."""
        model = Pi3X.from_pretrained(model_path).eval()
        # No pose/depth/intrinsic conditions are ever fed, so drop the
        # multimodal branch to save memory (repo demo does the same).
        model.disable_multimodal()
        model = model.to(self.device)
        print(f"Pi3X model loaded from {model_path}")
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

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        """Replicate ``pi3.utils.basic.load_images_as_tensor`` sizing.

        The whole batch is resized to one size derived from the given
        (first) image: scale the area to ``_PIXEL_LIMIT``, round each
        dimension to a multiple of 14, then shrink the larger-ratio side
        until the area fits the budget again.

        Returns:
            ``(width, height)`` of the processed images.
        """
        scale = (_PIXEL_LIMIT / (width * height)) ** 0.5 if width * height > 0 else 1.0
        w_target, h_target = width * scale, height * scale
        k, m = round(w_target / _PATCH), round(h_target / _PATCH)
        while (k * _PATCH) * (m * _PATCH) > _PIXEL_LIMIT:
            if k / m > w_target / h_target:
                k -= 1
            else:
                m -= 1
        return max(1, k) * _PATCH, max(1, m) * _PATCH

    def _load_images_tensor(
        self, image_paths: list[str]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load and resize images exactly like Pi3's own loader.

        PIL decode → RGB → LANCZOS resize to the uniform target size
        (derived from the first image) → float tensor in [0, 1].

        Returns:
            ``(N, 3, H, W)`` tensor and the ``(width, height)`` target.
        """
        first = Image.open(image_paths[0])
        target_w, target_h = self._target_size(*first.size)

        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(tensors, dim=0), (target_w, target_h)

    def _run_pi3x(self, imgs: torch.Tensor) -> dict:
        """Run Pi3X inference on a ``(N, 3, H, W)`` image tensor.

        Returns a dict of numpy arrays: ``extrinsic`` (N, 3, 4)
        world-to-camera, ``intrinsic`` (N, 3, 3) in processed-pixel space,
        ``depth_map`` / ``depth_conf`` (N, H, W), ``points`` (N, H, W, 3)
        world coordinates, ``edge_mask`` (N, H, W) True at depth/normal
        discontinuities.
        """
        imgs = imgs.to(self.device)
        if self.device.type == "cuda":
            major = torch.cuda.get_device_capability(self.device)[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                res = self.model(imgs=imgs[None])
        else:
            res = self.model(imgs=imgs[None])

        local_points = res["local_points"][0].float()  # (N, H, W, 3), metric
        conf_prob = torch.sigmoid(res["conf"][0, ..., 0].float())  # (N, H, W)

        # Repo-demo mask: drop depth/normal discontinuities (used only for
        # the sparse cloud; the dense depth map stays untouched).
        prob_mask = conf_prob > 0.1
        edge_mask = depth_normal_edge(
            res["local_points"], rtol=0.03, mask=prob_mask[None]
        )[0]

        # Intrinsics from the point-ray field, centred principal point
        # (mirrors example_mm.py).
        rays_d = F.normalize(res["local_points"].float(), dim=-1)
        intrinsic = recover_intrinsic_from_rays_d(
            rays_d, force_center_principal_point=True
        )[0]

        extrinsic = se3_inverse(res["camera_poses"][0].float())[:, :3, :4]

        out = {
            "extrinsic": extrinsic.cpu().numpy().astype(np.float32),
            "intrinsic": intrinsic.cpu().numpy().astype(np.float32),
            "depth_map": local_points[..., 2].cpu().numpy().astype(np.float32),
            "depth_conf": conf_prob.cpu().numpy().astype(np.float32),
            "points": res["points"][0].float().cpu().numpy().astype(np.float32),
            "edge_mask": edge_mask.cpu().numpy(),
        }

        del res, local_points, conf_prob, prob_mask, edge_mask, rays_d
        torch.cuda.empty_cache()
        return out

    def _reconstruct_without_ba(
        self,
        preds: dict,
        processed_images: np.ndarray,
        conf_thres_prob: float,
        max_points_for_colmap: int,
    ):
        """Build a track-less pycolmap reconstruction from Pi3X outputs."""
        depth_map = preds["depth_map"]
        height, width = depth_map.shape[-2:]
        image_size = np.array([width, height])

        conf_mask = np.isfinite(depth_map) & (depth_map > 0)
        conf_mask &= preds["depth_conf"] > conf_thres_prob
        conf_mask &= ~preds["edge_mask"]
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
        points_rgb = processed_images[conf_mask]

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

        Pi3's processed frame is non-square (and possibly anisotropically
        resized), so fx/cx and fy/cy are scaled by the per-axis ratios.
        Points2D stay in processed-pixel coordinates, mirroring
        ``VGGTWrapper``'s non-BA behavior.
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
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
        proc_wh: tuple[int, int],
    ):
        """Assemble EPO's feed-forward dict from raw Pi3X outputs.

        Follows the ``VGGTWrapper._build_ff_data`` recipe: the *original*
        sharp pixels (torchvision decode, PIL fallback, antialiased BICUBIC
        resize) feed the edge detector, while depth/confidence/pose/intrinsic
        come straight from Pi3X in its processed-pixel space. Every image is
        resized to the batch-uniform target size (Pi3 never crops). Keyed by
        the relative image path (``"cam_id/image_name"``).
        """
        from concurrent.futures import ThreadPoolExecutor

        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize as tv_resize

        target_w, target_h = proc_wh

        def _process_one(i):
            """Build one image's ff_data entry (independent per image)."""
            base = base_image_paths[i]

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
                    [target_h, target_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                .float()
                .div_(255.0)
            )

            return base, {
                "image": img_t,
                "depth": torch.from_numpy(preds["depth_map"][i]).float(),
                "confidence": torch.from_numpy(preds["depth_conf"][i]).float(),
                "pose": torch.from_numpy(preds["extrinsic"][i].copy()).float(),
                "intrinsic": torch.from_numpy(preds["intrinsic"][i].copy()).float(),
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
        conf_thres_prob: float = 0.1,
        max_points_for_colmap: int = 100_000,
    ):
        """Run Pi3X reconstruction on images and save results.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images (randomly sampled if exceeded).
            use_ba: Unsupported for Pi3X; must stay False.
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
            conf_thres_prob: Sigmoid-confidence threshold below which pixels
                are dropped from the sparse point cloud (Pi3 demo uses 0.1).
            max_points_for_colmap: Maximum 3D points for COLMAP.

        Returns:
            ``(ff_data, reconstruction, depths)`` — same contract as
            ``VGGTWrapper.forward``.
        """
        if use_ba:
            raise NotImplementedError(
                "Pi3XWrapper has no bundle-adjustment path; use use_ba=False "
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

        # Load + resize images the way Pi3's own loader does.
        t_start = time.time()
        imgs, proc_wh = self._load_images_tensor(image_paths)
        processed_images = (
            (imgs.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        )
        timings["load_images"] = time.time() - t_start

        # Run Pi3X
        print("Running Pi3X model...")
        t_start = time.time()
        preds = self._run_pi3x(imgs)
        timings["run_pi3x"] = time.time() - t_start

        print("Running reconstruction without Bundle Adjustment...")
        t_start = time.time()
        reconstruction, proc_hw = self._reconstruct_without_ba(
            preds,
            processed_images,
            conf_thres_prob,
            max_points_for_colmap,
        )
        timings["reconstruction_without_ba"] = time.time() - t_start

        # Always build the depths dict for downstream tasks (disk save).
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            stem = Path(img_path).with_suffix("").as_posix()
            depths[stem] = {
                "depth": torch.from_numpy(preds["depth_map"][i]).float(),
                "confidence": torch.from_numpy(preds["depth_conf"][i]).float(),
            }

        # Build EPO's feed-forward dict: Pi3X depth/pose/intrinsic plus the
        # original sharp image for the edge detector.
        t_start = time.time()
        ff_data = self._build_ff_data(preds, base_image_paths, image_paths, proc_wh)
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
        print(f"Load images:                 {timings['load_images']:>8.2f}s")
        print(f"Run Pi3X model:              {timings['run_pi3x']:>8.2f}s")
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

        del preds, imgs, processed_images
        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown (e.g. `run_pi3x`) for callers that want
        # to report inference time separately from I/O.
        self.last_timings = timings

        return ff_data, reconstruction, depths

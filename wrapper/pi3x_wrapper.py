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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from base_wrapper import BaseWrapper  # noqa: E402
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


class Pi3XWrapper(BaseWrapper):
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

    def _load_model(self, model_path: str) -> Pi3X:
        """Load Pi3X from a Hugging Face repo id (image-only branch)."""
        model = Pi3X.from_pretrained(model_path).eval()
        # No pose/depth/intrinsic conditions are ever fed, so drop the
        # multimodal branch to save memory (repo demo does the same).
        model.disable_multimodal()
        model = model.to(self.device)
        print(f"Pi3X model loaded from {model_path}")
        return model

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

    def _rescale_camera_params(
        self,
        params: np.ndarray,
        orig_wh: tuple[int, int],
        proc_hw: tuple[int, int],
    ) -> np.ndarray:
        """Per-axis rescale: Pi3X's frame keeps the aspect ratio and is uncropped."""
        return self._rescale_camera_params_per_axis(params, orig_wh, proc_hw)

    def _conf_mask(self, preds: dict, conf_thres: float) -> np.ndarray:
        """Depth validity + sigmoid probability + Pi3X's depth/normal edge filter."""
        depth_map = preds["depth_map"]
        mask = np.isfinite(depth_map) & (depth_map > 0)
        mask &= preds["depth_conf"] > conf_thres
        return mask & ~preds["edge_mask"]

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Assemble EPO's feed-forward dict from raw Pi3X outputs.

        Normalizes Pi3X's batch-uniform frame into the per-image entries
        ``BaseWrapper._build_ff_data`` consumes: every image is resized to
        ``preds["proc_wh"]`` (Pi3 never crops), while depth / confidence /
        pose / intrinsic come straight from Pi3X in its processed-pixel
        space. Keyed by the relative image path (``"cam_id/image_name"``).
        """
        target_w, target_h = preds["proc_wh"]

        entries = [
            {
                "key": key,
                "image_path": image_paths[i],
                "resize_hw": (target_h, target_w),
                "crop_box": None,
                "depth": preds["depth_map"][i],
                "confidence": preds["depth_conf"][i],
                "pose": preds["extrinsic"][i].copy(),
                "intrinsic": preds["intrinsic"][i].copy(),
            }
            for i, key in enumerate(base_image_paths)
        ]

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
        proc_hw = preds["depth_map"].shape[-2:]
        preds["points_rgb"] = processed_images
        reconstruction = self._reconstruct(
            preds, conf_thres_prob, max_points_for_colmap
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
        preds["proc_wh"] = proc_wh
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


if __name__ == "__main__":
    Pi3XWrapper._cli_main(default_model_path="yyfz233/Pi3X")

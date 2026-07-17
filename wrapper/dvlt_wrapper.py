"""Standalone DVLT (Déjà View) wrapper for EPO.

Drop-in replacement for ``wrapper.vggt_wrapper.VGGTWrapper``: same constructor
arguments, same ``forward`` signature, and the same
``(ff_data, reconstruction, depths)`` return contract, so the two models can
be swapped by changing only the import. Runs against the unmodified
``third_party/dvlt`` checkout (nv-tlabs/dvlt); only the NumPy→pycolmap
conversion comes from the local ``np_to_colmap`` companion module.

Differences inherent to the model:
- DVLT works at 504 px (patch 14) and keeps the image aspect ratio (center
  crop to a multiple of 14) instead of VGGT's 518 px square padding, so
  ``ff_data``/``depths`` maps are at 504-scale dims.
- Pose comes out camera-to-world and is inverted here to COLMAP's
  world-to-camera (T_cw), matching what EPO's ``from_ff`` expects.
- DVLT has a strong landscape prior: portrait input breaks its intrinsics/
  pose fit (AUC@5 1.3 vs 68.3 on terrasky3D's munich_frauenkirche). Portrait
  frames (H > W) are therefore rotated 90° CCW for inference and every
  prediction is rotated back afterwards (poses via the in-plane camera
  rotation, depth/confidence maps via ``rot90``, focals swapped).
- Bundle adjustment is not supported (``use_ba=True`` raises); the BA-specific
  keyword arguments exist only for signature parity.

Used via ``from wrapper.dvlt_wrapper import DVLTWrapper``.
"""

import os
import sys

# Make the pristine dvlt checkout importable as ``dvlt`` and this folder
# importable for the local ``np_to_colmap`` companion, regardless of caller CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "third_party", "dvlt", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import PartialState  # noqa: E402
from base_wrapper import BaseWrapper  # noqa: E402
from dvlt.common.constants import DataField, PredictionField  # noqa: E402
from dvlt.model.dvlt.model import DVLT  # noqa: E402
from dvlt.util.preprocess import preprocess_images  # noqa: E402
from PIL import Image  # noqa: E402


class DVLTWrapper(BaseWrapper):
    """Wrapper class for DVLT model to perform 3D reconstruction from images."""

    def __init__(
        self,
        model_path: str,
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the DVLT wrapper.

        Args:
            model_path: DVLT weights: a local file/directory, a URL, or a
                Hugging Face Hub repo id (the release checkpoint is
                ``"nvidia/dvlt"``).
            cuda_id: CUDA device index (CPU fallback if unavailable).
            seed: Random seed for reproducibility.
            oom_safe: Unused (BA-path knob in VGGTWrapper); kept for parity.
        """
        self.model_path = model_path
        self.seed = seed
        self._set_seed(seed)
        self.oom_safe = oom_safe

        # Setup device and dtype
        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        )

        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Configure CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Fixed resolutions (release checkpoint: img_size=504, patch 14).
        self.dvlt_fixed_resolution = 504
        self.patch_size = 14

        # dvlt logs through accelerate's logger, which requires the accelerate
        # state to exist; a bare PartialState() is enough (no Accelerator).
        PartialState()

        # Load model
        self.model = self._load_model(self.model_path)

        print(f"DVLTWrapper initialized on {self.device} with dtype {self.dtype}")

    def _load_model(self, model_path: str) -> DVLT:
        """Load the DVLT model from a local path, URL, or HF Hub repo id."""
        # The release checkpoint carries the DINOv2 patch-embed weights
        # (load_pretrained is strict), so skip the separate hub download.
        # decode_chunk_size: frames decoded per chunk in the fp32 ray/depth
        # heads (results are identical); the default 128 OOMs a 24 GB GPU on
        # 150-frame 4:3 scenes.
        model = DVLT(
            img_size=self.dvlt_fixed_resolution,
            patch_size=self.patch_size,
            load_patch_embed_weights=False,
            decode_chunk_size=32,
        )
        model.load_pretrained(model_path, strict=True)
        model.model.eval()
        model.model.to(self.device)
        print(f"DVLT model loaded from {model_path}")
        return model

    def _crop_geometry(
        self, original_sizes: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Replicate ``dvlt.util.preprocess.preprocess_images`` geometry.

        For each image with original size (w, h) the preprocessor resizes the
        longest side to ``dvlt_fixed_resolution``, center-crops both dims to a
        multiple of ``patch_size``, then center-pads all frames to the batch
        max. Returns per-frame ``(crop_h, crop_w, pad_top, pad_left)`` that
        locate each frame's real content inside the padded model output.
        """
        res, patch = self.dvlt_fixed_resolution, self.patch_size
        crops = []
        for width, height in original_sizes:
            scale = res / max(width, height)
            new_h, new_w = int(round(height * scale)), int(round(width * scale))
            crop_h = max(patch, (new_h // patch) * patch)
            crop_w = max(patch, (new_w // patch) * patch)
            crops.append((crop_h, crop_w))
        max_h = max(c[0] for c in crops)
        max_w = max(c[1] for c in crops)
        return [
            (crop_h, crop_w, (max_h - crop_h) // 2, (max_w - crop_w) // 2)
            for crop_h, crop_w in crops
        ]

    # In-plane camera rotation for a 90° CCW image rotation: PIL ROTATE_90
    # maps pixel (x, y) -> (y, W-1-x), so the virtual camera axes are
    # X' = Y, Y' = -X, Z' = Z, i.e. x'_c = A @ x_c.
    _ROT90_CCW = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def _unrotate_predictions(
        self, extrinsic: np.ndarray, intrinsic: np.ndarray, rotated: list[bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map predictions for rotated (portrait) frames back to portrait.

        The model saw those frames rotated 90° CCW; its w2c pose maps world
        into the *virtual* landscape camera. The true portrait pose is
        R = A^T R', t = A^T t' — the world frame/gauge is untouched, so the
        model's ``world_points`` stay consistent with the corrected poses.
        Focals swap (fx = fy', fy = fx'); the principal point is re-centred
        downstream (recon rescale + ff_data), so a plain cx/cy swap suffices.
        """
        extrinsic = extrinsic.copy()
        intrinsic = intrinsic.copy()
        A_T = self._ROT90_CCW.T
        for i, rot in enumerate(rotated):
            if not rot:
                continue
            extrinsic[i] = A_T @ extrinsic[i]
            fx, fy = intrinsic[i, 0, 0], intrinsic[i, 1, 1]
            intrinsic[i, 0, 0], intrinsic[i, 1, 1] = fy, fx
            cx, cy = intrinsic[i, 0, 2], intrinsic[i, 1, 2]
            intrinsic[i, 0, 2], intrinsic[i, 1, 2] = cy, cx
        return extrinsic, intrinsic

    def _run_dvlt(
        self, batch: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run DVLT and return VGGT-convention numpy outputs.

        Returns ``(extrinsic, intrinsic, depth_map, depth_conf, world_points)``
        with ``extrinsic`` already inverted to world-to-camera (S, 3, 4) and
        all maps at the padded model resolution (S, H, W[, 3]).
        """
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                predictions = self.model.predict(batch, accelerator=None)

        cameras = predictions[PredictionField.CAMERAS][0]
        c2w = cameras.camera_to_worlds.float().cpu().numpy()  # (S, 3, 4)
        intrinsic = cameras.get_intrinsics_matrices().float().cpu().numpy()
        depth_map = predictions[PredictionField.DEPTHS][0].float().cpu().numpy()
        depth_conf = predictions[PredictionField.DEPTHS_CONF][0].float().cpu().numpy()
        world_points = (
            predictions[PredictionField.WORLD_POINTS][0].float().cpu().numpy()
        )

        # c2w -> w2c (T_cw): R' = R^T, t' = -R^T t.
        R_T = c2w[:, :3, :3].transpose(0, 2, 1)
        extrinsic = np.concatenate(
            [R_T, -np.einsum("sij,sj->si", R_T, c2w[:, :3, 3])[..., None]], axis=-1
        )

        del predictions
        torch.cuda.empty_cache()

        return extrinsic, intrinsic, depth_map, depth_conf, world_points

    def _conf_mask(self, preds: dict, conf_thres: float) -> np.ndarray:
        """Absolute confidence threshold, minus the synthetic batch-pad pixels."""
        return (preds["depth_conf"] >= conf_thres) & preds["valid_pixels"]

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Assemble EPO's feed-forward dict from raw DVLT outputs.

        Normalizes DVLT's padded (and, for portrait frames, rotated) model
        space into the per-image entries ``BaseWrapper._build_ff_data``
        consumes:

        - image: antialiased BICUBIC resize to DVLT's model dims, then the
          same center crop to a multiple of ``patch_size`` the preprocessor
          applies — so it matches the depth maps pixel for pixel;
        - depth/confidence: the frame's real content cropped out of the padded
          model output, rotated back to portrait where the frame went in
          rotated;
        - intrinsic: DVLT's predicted focals with the principal point at the
          float image centre (the disk-loader convention);
        - pose: world-to-camera (3, 4), as EPO expects.

        Keyed by the relative image path (``"cam_id/image_name"``).
        """
        res = self.dvlt_fixed_resolution
        entries = []

        for i, key in enumerate(base_image_paths):
            # Model-space crop (landscape for rotated frames) locates the
            # frame in the padded output; image-space dims follow the
            # original orientation (transposed when the frame was rotated).
            crop_h, crop_w, pad_top, pad_left = preds["crop_geometry"][i]
            rotated = preds["rotated"][i]
            img_h, img_w = (crop_w, crop_h) if rotated else (crop_h, crop_w)

            # Same resize + divisible center crop as `preprocess_images` (the
            # crop math commutes with rotation), so the image lands on the
            # rotated-back depth map's exact dims.
            width, height = preds["original_sizes"][i]
            scale = res / max(width, height)
            new_h, new_w = int(round(height * scale)), int(round(width * scale))

            # Crop the frame's real content out of the padded model output;
            # rotate back to portrait where the frame went in rotated.
            d = preds["depth_map"][
                i, pad_top : pad_top + crop_h, pad_left : pad_left + crop_w
            ]
            c = preds["depth_conf"][
                i, pad_top : pad_top + crop_h, pad_left : pad_left + crop_w
            ]
            if rotated:
                d = np.rot90(d, k=-1).copy()
                c = np.rot90(c, k=-1).copy()

            intr = preds["intrinsic"][i].copy()
            intr[0, 2] = img_w / 2.0
            intr[1, 2] = img_h / 2.0

            entries.append(
                {
                    "key": key,
                    "image_path": image_paths[i],
                    "resize_hw": (new_h, new_w),
                    "crop_box": (
                        (new_h - img_h) // 2,
                        (new_w - img_w) // 2,
                        img_h,
                        img_w,
                    ),
                    "depth": d,
                    "confidence": c,
                    "pose": preds["extrinsic"][i][:3, :4],
                    "intrinsic": intr,
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
        # BA-specific parameters (signature parity with VGGTWrapper; unused)
        max_reproj_error: float = 10.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.3,
        query_frame_num: int = 30,
        max_query_pts: int = 4096,
        fine_tracking: bool = False,
        # Non-BA parameters
        conf_thres_value: float = 2.5,
        max_points_for_colmap: int = 100_000,
    ):
        """Run DVLT reconstruction on images and save results.

        Same signature and return contract as ``VGGTWrapper.forward``:
        returns ``(ff_data, reconstruction, depths)``.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images to process (randomly sampled if exceeded).
            use_ba: Not supported for DVLT; ``True`` raises NotImplementedError.
            save_depth: Whether to write ``depths.pth`` (only when ``save``).
            save: Whether to write the reconstruction / depths / timings to disk.
                Set ``False`` to skip all disk I/O for clean benchmarking; the
                returned ``(ff_data, reconstruction, depths)`` can be written by
                the caller afterwards.
            max_reproj_error: Unused (BA-only); kept for signature parity.
            shared_camera: Rescale each shared camera only once on export.
            camera_type: Unused (BA-only); kept for signature parity.
            vis_thresh: Unused (BA-only); kept for signature parity.
            query_frame_num: Unused (BA-only); kept for signature parity.
            max_query_pts: Unused (BA-only); kept for signature parity.
            fine_tracking: Unused (BA-only); kept for signature parity.
            conf_thres_value: Confidence threshold for depth filtering (DVLT's
                confidence is ``exp(x) + 1``, same >= 1 family as VGGT's).
            max_points_for_colmap: Maximum 3D points for COLMAP.
        """
        if use_ba:
            raise NotImplementedError(
                "DVLTWrapper does not support bundle adjustment (no track "
                "prediction). Use VGGTWrapper for the BA path."
            )

        os.makedirs(output_path, exist_ok=True)

        timings = {}
        t_total_start = time.time()

        # Find and sample images
        t_start = time.time()
        image_paths = self._find_images(images_path)
        image_paths = self._sample_images(image_paths, max_images)
        timings["find_and_sample_images"] = time.time() - t_start

        # Get base paths
        base_image_paths = [os.path.relpath(path, images_path) for path in image_paths]

        # Load and preprocess images: longest side to 504, center crop to a
        # multiple of 14, center pad across the batch (aspect kept).
        print(f"Loading {len(image_paths)} images...")
        t_start = time.time()
        pil_images = [Image.open(p) for p in image_paths]
        original_sizes = np.array([img.size for img in pil_images])  # (S, 2) = (w, h)
        # Rotate portrait frames to landscape for inference (see module
        # docstring); predictions are rotated back after the forward pass.
        rotated = [img.height > img.width for img in pil_images]
        if any(rotated):
            print(f"Rotating {sum(rotated)} portrait frame(s) to landscape...")
            pil_images = [
                img.transpose(Image.ROTATE_90) if rot else img
                for img, rot in zip(pil_images, rotated, strict=True)
            ]
        # Model-space geometry follows the (possibly rotated) frames.
        effective_sizes = np.array(
            [
                (h, w) if rot else (w, h)
                for (w, h), rot in zip(original_sizes, rotated, strict=True)
            ]
        )
        batch = preprocess_images(
            pil_images,
            img_size=self.dvlt_fixed_resolution,
            patch_size=self.patch_size,
            device=self.device,
        )
        crop_geometry = self._crop_geometry(effective_sizes)
        timings["load_and_preprocess"] = time.time() - t_start

        # Run DVLT.
        print("Running DVLT model...")
        t_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, world_points = self._run_dvlt(
            batch
        )
        extrinsic, intrinsic = self._unrotate_predictions(extrinsic, intrinsic, rotated)
        timings["run_dvlt"] = time.time() - t_start

        # Reconstruct (no BA path for DVLT)
        print("Running reconstruction without Bundle Adjustment...")
        t_start = time.time()
        recon_resolution = self.dvlt_fixed_resolution
        recon_preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_conf": depth_conf,
            "points": world_points,
            # DVLT's batch is already at the padded model size, so no resize.
            "points_rgb": self._points_rgb(batch[DataField.IMAGES][0]),
            "valid_pixels": batch["gradio_valid_pixels"][0].cpu().numpy(),
        }
        reconstruction = self._reconstruct(
            recon_preds, conf_thres_value, max_points_for_colmap
        )
        timings["reconstruction_without_ba"] = time.time() - t_start

        # Build the depths dict for downstream tasks (disk save), maps cropped
        # to each frame's real content.
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            crop_h, crop_w, pad_top, pad_left = crop_geometry[i]
            stem = Path(img_path).with_suffix("").as_posix()
            d = depth_map[i, pad_top : pad_top + crop_h, pad_left : pad_left + crop_w]
            c = depth_conf[i, pad_top : pad_top + crop_h, pad_left : pad_left + crop_w]
            if rotated[i]:
                # Back to portrait: inverse of the CCW image rotation.
                d = np.rot90(d, k=-1).copy()
                c = np.rot90(c, k=-1).copy()
            depths[stem] = {
                "depth": torch.from_numpy(d).float(),
                "confidence": torch.from_numpy(c).float(),
            }

        # Build EPO's feed-forward dict: DVLT depth/pose/intrinsic with the
        # batch padding cropped out, and the original sharp image for the edge
        # detector. Free the GPU batch afterwards.
        t_start = time.time()
        ff_preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "crop_geometry": crop_geometry,
            "rotated": rotated,
            "original_sizes": original_sizes,
        }
        ff_data = self._build_ff_data(ff_preds, base_image_paths, image_paths)
        timings["build_ff_data"] = time.time() - t_start
        del batch, pil_images
        gc.collect()
        torch.cuda.empty_cache()

        # Rescale reconstruction to original resolution
        if reconstruction is not None:
            t_start = time.time()
            reconstruction = self._rescale_reconstruction(
                reconstruction,
                base_image_paths,
                original_sizes,  # per-image (width, height)
                (recon_resolution, recon_resolution),  # square model frame
            )
            timings["rescale_reconstruction"] = time.time() - t_start

            # Save reconstruction (skipped when save=False; the caller can
            # write the returned objects after timing the inference).
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
        print(f"Load and preprocess images:  {timings['load_and_preprocess']:>8.2f}s")
        print(f"Run DVLT model:              {timings['run_dvlt']:>8.2f}s")
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
            t_path = "timings.txt"
            with open(os.path.join(output_path, t_path), "w") as f:
                for key, value in timings.items():
                    f.write(f"{key}: {value:.4f} s\n")

        # Clean up all memory before returning
        del (
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            world_points,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown for callers that want to report
        # inference time separately from I/O. The model-forward key is
        # "run_dvlt" (VGGTWrapper's analog is "run_vggt").
        self.last_timings = timings

        # ff_data: EPO-ready feed-forward dict; reconstruction: original-res
        # pycolmap model; depths: per-image {"depth", "confidence"} dict (the
        # depths.pth contents) — returned so a caller can write it after
        # timing the inference.
        return ff_data, reconstruction, depths


if __name__ == "__main__":
    DVLTWrapper._cli_main(default_model_path="nvidia/dvlt")

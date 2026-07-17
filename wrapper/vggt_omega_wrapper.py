"""Standalone VGGT-Omega wrapper for EPO.

Drop-in replacement for ``wrapper.vggt_wrapper.VGGTWrapper``: same constructor
arguments, same ``forward`` signature, and the same
``(ff_data, reconstruction, depths)`` return contract, so the two models can
be swapped by changing only the import. Runs against the pristine
``third_party/vggt-omega`` checkout (facebookresearch/vggt-omega); only the
NumPy→pycolmap conversion comes from the local ``np_to_colmap`` companion
module.

Differences from ``VGGTWrapper``:
- Model class is ``VGGTOmega`` instead of ``VGGT``.
- Native square resolution is 512 (patch size 16) instead of VGGT's 518.
- Inference is a single ``model(images)`` dict call; the 9D pose encoding
  (translation, quaternion, FoV) is decoded via ``encoding_to_camera``
  instead of ``pose_encoding_to_extri_intri``.
- VGGT-Omega has no bundle-adjustment path, so ``use_ba=True`` raises
  ``NotImplementedError`` (as in ``DA3Wrapper`` / ``DVLTWrapper``) and this
  module never needs lightglue.

The checkpoint (``vggt_omega_1b_512.pt``) is gated on Hugging Face
(https://huggingface.co/facebook/VGGT-Omega), so pass a local path.

Used via ``from wrapper.vggt_omega_wrapper import VGGTOmegaWrapper``.
"""

import os
import sys

# Make the pristine vggt-omega submodule importable as ``vggt_omega``, the
# pristine vggt submodule importable as ``vggt`` (shared vggsfm-derived
# loading/geometry helpers), and this folder importable for the local
# ``np_to_colmap`` companion — regardless of caller CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (
    _HERE,
    os.path.join(_ROOT, "third_party", "vggt-omega"),
    os.path.join(_ROOT, "third_party", "vggt"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from base_wrapper import BaseWrapper  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


class VGGTOmegaWrapper(BaseWrapper):
    """Wrapper class for VGGT-Omega model to perform 3D reconstruction from images."""

    def __init__(
        self,
        model_path: str,
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the VGGT-Omega wrapper.

        Args:
            model_path: Path to the VGGT-Omega weights (a local ``.pt``
                checkpoint, e.g. ``vggt_omega_1b_512.pt``). A URL is also
                accepted and downloaded/cached via ``torch.hub``.
            cuda_id: CUDA device index (CPU fallback if unavailable).
            seed: Random seed for reproducibility.
            oom_safe: Part of the shared wrapper interface (``run_for_dataset.py``
                passes it); inert here — the model is held for the whole run.
        """
        self._set_seed(seed)

        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        )

        # Configure CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load model
        self.model = self._load_model(model_path)

        # Fixed resolutions (VGGT-Omega-1B-512 native resolution, patch 16).
        self.vggt_fixed_resolution = 512
        self.img_load_resolution = 768

        print(f"VGGTOmegaWrapper initialized on {self.device}")

    def _load_model(self, model_path: str) -> VGGTOmega:
        """Load the VGGT-Omega model from a local checkpoint file or a URL."""
        model = VGGTOmega()
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        print(f"VGGT-Omega model loaded from {model_path}")
        return model

    def _run_vggt(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run VGGT-Omega to estimate cameras and depth.

        Same contract as ``VGGTWrapper._run_vggt``: square input resized to
        ``vggt_fixed_resolution``, numpy outputs of identical shapes —
        extrinsic (S, 3, 4), intrinsic (S, 3, 3), depth_map (S, H, W, 1),
        depth_conf (S, H, W). VGGT-Omega handles mixed precision internally
        and returns a 9D pose encoding (translation, quaternion, FoV) that
        ``encoding_to_camera`` decodes against the square frame.
        """
        assert len(images.shape) == 4 and images.shape[1] == 3

        # Resize to VGGT-Omega resolution (square).
        images_resized = F.interpolate(
            images,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            predictions = self.model(images_resized)

        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"], images_resized.shape[-2:]
        )

        # Convert to numpy (drop the batch dim VGGT-Omega adds internally).
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = predictions["depth"].squeeze(0).cpu().numpy()
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

        # Clean up intermediate tensors
        del images_resized, predictions
        torch.cuda.empty_cache()

        return extrinsic, intrinsic, depth_map, depth_conf

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Place VGGT-Omega's outputs relative to the original images.

        VGGT-Omega infers on a square, letterboxed 512 px frame, so this is the
        shared letterbox recipe (``BaseWrapper._ff_entries_letterbox``),
        identical to VGGT's apart from the resolution.
        """
        return self._ff_entries_letterbox(
            preds, base_image_paths, image_paths, self.vggt_fixed_resolution
        )

    @torch.no_grad()
    def forward(
        self,
        images_path: str,
        output_path: str,
        max_images: int = 150,
        use_ba: bool = False,
        save_depth: bool = True,
        save: bool = True,
        # BA-specific parameters (signature parity with VGGTWrapper; unused).
        # run_for_dataset.py passes these to whichever wrapper it selects.
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
        """Run VGGT-Omega reconstruction on images and save results.

        Same return contract as ``VGGTWrapper.forward``: returns
        ``(ff_data, reconstruction, depths)``.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images to process (randomly sampled if exceeded).
            use_ba: Unsupported for VGGT-Omega; must stay False.
            save_depth: Whether to write ``depths.pth`` (only when ``save``).
            save: Whether to write the reconstruction / depths / timings to disk.
                Set ``False`` to skip all disk I/O for clean benchmarking; the
                returned ``(ff_data, reconstruction, depths)`` can be written by
                the caller afterwards.
            max_reproj_error: Unused (BA-only).
            shared_camera: Unused (BA-only); the feed-forward path always builds
                one camera per image.
            camera_type: Unused (BA-only); the feed-forward path uses PINHOLE.
            vis_thresh: Unused (BA-only).
            query_frame_num: Unused (BA-only).
            max_query_pts: Unused (BA-only).
            fine_tracking: Unused (BA-only).
            conf_thres_value: Confidence threshold for depth filtering.
            max_points_for_colmap: Maximum 3D points for COLMAP.

        Raises:
            NotImplementedError: If ``use_ba`` is True.
        """
        if use_ba:
            raise NotImplementedError(
                "VGGTOmegaWrapper has no bundle-adjustment path; use use_ba=False "
                "(the feed-forward output is what EPO refines)."
            )

        os.makedirs(output_path, exist_ok=True)

        timings = {}
        t_total_start = time.time()

        # Find and sample images
        t_start = time.time()
        image_paths = self._find_images(images_path)
        image_paths = self._sample_images(image_paths, max_images)
        timings["find_and_sample_images"] = time.time() - t_start

        # Load and preprocess images. VGGT-Omega's camera head is tuned for a
        # square frame, so the square-padded loader is used and the square
        # outputs are handed to EPO via ff_data.
        print(f"Loading {len(image_paths)} images...")
        t_start = time.time()
        images, original_coords = self._load_and_preprocess_images_square(
            image_paths, self.img_load_resolution
        )
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)
        timings["load_and_preprocess"] = time.time() - t_start

        base_image_paths = [os.path.relpath(path, images_path) for path in image_paths]

        # Run VGGT-Omega.
        print("Running VGGT-Omega model...")
        t_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images)
        points_3d = self._unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsic
        )
        timings["run_vggt"] = time.time() - t_start

        print("Running reconstruction...")
        t_start = time.time()
        recon_resolution = self.vggt_fixed_resolution
        recon_preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_conf": depth_conf,
            "points": points_3d,
            "points_rgb": self._points_rgb(
                images, (recon_resolution, recon_resolution)
            ),
        }
        reconstruction = self._reconstruct(
            recon_preds, conf_thres_value, max_points_for_colmap
        )
        timings["reconstruction"] = time.time() - t_start

        # Always build the depths dict for downstream tasks (disk save).
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            stem = Path(img_path).with_suffix("").as_posix()
            depth_hw = torch.from_numpy(np.asarray(depth_map[i]).squeeze()).float()
            conf_hw = torch.from_numpy(np.asarray(depth_conf[i]).squeeze()).float()
            depths[stem] = {"depth": depth_hw, "confidence": conf_hw}

        # Build EPO's feed-forward dict: VGGT-Omega depth/pose/intrinsic with
        # the letterbox cropped out, and the original sharp image for the edge
        # detector. Free the GPU image tensor afterwards.
        t_start = time.time()
        preds = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "original_coords": original_coords.cpu().numpy(),
        }
        ff_data = self._build_ff_data(preds, base_image_paths, image_paths)
        timings["build_ff_data"] = time.time() - t_start
        del images
        gc.collect()
        torch.cuda.empty_cache()

        # Rescale reconstruction to original resolution
        if reconstruction is not None:
            t_start = time.time()
            reconstruction = self._rescale_reconstruction(
                reconstruction,
                base_image_paths,
                original_coords.cpu().numpy()[:, -2:],  # per-image (width, height)
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
        print(f"Run VGGT-Omega model:        {timings['run_vggt']:>8.2f}s")
        print(f"Reconstruction:              {timings['reconstruction']:>8.2f}s")
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

        # Clean up all memory before returning (these arrays can be large)
        del (
            original_coords,
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            points_3d,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown (e.g. `run_vggt`) for callers that want
        # to report inference time separately from I/O.
        self.last_timings = timings

        # ff_data: EPO-ready feed-forward dict at native resolution.
        # reconstruction: original-res pycolmap model.
        # depths: per-image {"depth", "confidence"} dict (the depths.pth
        # contents) — returned so a caller can write it after timing inference.
        return ff_data, reconstruction, depths

    @staticmethod
    def _load_and_preprocess_images_square(image_paths: list[str], target_size: int):
        """Load images and preprocess to square, padded shape (from vggt.utils.load_fn)."""
        from vggt.utils.load_fn import load_and_preprocess_images_square

        return load_and_preprocess_images_square(image_paths, target_size)

    @staticmethod
    def _unproject_depth_map_to_point_map(
        depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
    ) -> np.ndarray:
        """Unproject depth maps to world-space point maps (from vggt.utils.geometry)."""
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        return unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)


if __name__ == "__main__":
    VGGTOmegaWrapper._cli_main(
        default_model_path="https://huggingface.co/facebook/VGGT-Omega/resolve/main/vggt_omega_1b_512.pt"
    )

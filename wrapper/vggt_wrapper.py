"""Standalone VGGT wrapper for EPO.

1:1 port of the VGGT fork's ``vggt/wrapper.py`` so it runs against the
*unmodified* facebookresearch/vggt submodule. Everything the wrapper needs
(model, aggregator, heads, geometry, tracking) is imported from the pristine
``vggt`` package; only the NumPy→pycolmap conversion is taken from the local
``np_to_colmap`` companion module (pycolmap 4.x compatible) so vggt stays
unmodified. Outputs are identical to the fork.

Used by ``demo_epo.py`` via ``from wrapper.vggt_wrapper import VGGTWrapper``.
"""

import os
import sys

# Make the pristine vggt submodule importable as ``vggt`` and this folder
# importable for the local ``np_to_colmap`` companion, regardless of caller CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "third_party", "vggt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import copy  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from np_to_colmap import (  # noqa: E402
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.helper import (  # noqa: E402
    create_pixel_coordinate_grid,
    randomly_limit_trues,
)
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

# NOTE: `predict_tracks` (the only user of lightglue, via vggt.dependency.
# vggsfm_utils) is imported lazily inside `_reconstruct_with_ba` so the default
# non-BA / EPO path — and importing this module — never requires lightglue.


class VGGTWrapper:
    """Wrapper class for VGGT model to perform 3D reconstruction from images."""

    def __init__(
        self,
        model_path: str,
        cuda_id: int = 0,
        seed: int = 42,
        oom_safe: bool = False,
    ):
        """Initialize the VGGT wrapper.

        Args:
            model_path: Path to the VGGT weights (a local ``.pt`` checkpoint).
                A URL is also accepted and downloaded/cached via ``torch.hub``.
            cuda_id: CUDA device index (CPU fallback if unavailable).
            seed: Random seed for reproducibility.
            oom_safe: Free the model after track prediction (BA path) to save VRAM.
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

        # Load model
        self.model = self._load_model(self.model_path)

        # Fixed resolutions
        self.vggt_fixed_resolution = 518
        self.img_load_resolution = 768

        print(f"VGGTWrapper initialized on {self.device} with dtype {self.dtype}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: str) -> VGGT:
        """Load the VGGT model from a local checkpoint file or a URL."""
        model = VGGT()
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        print(f"VGGT model loaded from {model_path}")
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

    def _run_vggt(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run VGGT model to estimate cameras and depth.

        Images are resized to a ``vggt_fixed_resolution`` square — VGGT's
        camera head is tuned for this framing; non-square input degrades the
        estimated poses badly.
        """
        assert len(images.shape) == 4 and images.shape[1] == 3

        # Resize to VGGT resolution (square).
        images_resized = F.interpolate(
            images,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)

            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_batch.shape[-2:]
            )

            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images_batch, ps_idx
            )

        # Convert to numpy
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()

        # Clean up intermediate tensors
        del images_resized, images_batch, pose_enc, aggregated_tokens_list, ps_idx
        torch.cuda.empty_cache()

        return extrinsic, intrinsic, depth_map, depth_conf

    @torch.no_grad()
    def _reconstruct_with_ba(
        self,
        images: torch.Tensor,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        points_3d: np.ndarray,
        max_query_pts: int,
        query_frame_num: int,
        vis_thresh: float,
        max_reproj_error: float,
        shared_camera: bool,
        camera_type: str,
        fine_tracking: bool,
        image_names: list[str] | None = None,
    ):
        """Reconstruct with bundle adjustment."""
        # Lazy import: pulls in lightglue (via vggt.dependency.vggsfm_utils),
        # which is only needed for this BA path.
        from vggt.dependency.track_predict import predict_tracks

        image_size = np.array(images.shape[-2:])
        scale = self.img_load_resolution / self.vggt_fixed_resolution

        # Track establishment timing
        t_track_start = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = (
                predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=max_query_pts,
                    query_frame_num=query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=fine_tracking,
                )
            )
            torch.cuda.empty_cache()
        track_time = time.time() - t_track_start

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh

        # Reconstruction timing (BA only)
        t_ba_start = time.time()
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=points_rgb,
            image_names=image_names,
        )

        if reconstruction is None:
            return None, None, track_time, 0.0

        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_principal_point = True
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        print("Bundle adjustment skipped (run it later from CLI.)")

        ba_time = time.time() - t_ba_start

        reconstruction_resolution = self.img_load_resolution

        return reconstruction, reconstruction_resolution, track_time, ba_time

    def _reconstruct_without_ba(
        self,
        images: torch.Tensor,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_conf: np.ndarray,
        points_3d: np.ndarray,
        conf_thres_value: float,
        max_points_for_colmap: int,
    ):
        """Reconstruct without bundle adjustment."""
        image_size = np.array([self.vggt_fixed_resolution, self.vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        # Get RGB values
        points_rgb = F.interpolate(
            images,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # Create coordinate grid
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        # Filter by confidence
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

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

        return reconstruction, self.vggt_fixed_resolution

    def _rescale_reconstruction(
        self,
        reconstruction,
        base_image_paths: list[str],
        original_coords: np.ndarray,
        img_size: int,
        shift_point2d: bool,
        shared_camera: bool,
    ):
        """Rescale and rename reconstruction to match original images."""
        rescale_camera = {
            camera_id: True for camera_id in reconstruction.cameras.keys()
        }  # rescale all cameras but only once each

        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = base_image_paths[pyimageid - 1]

            if rescale_camera[pyimage.camera_id]:
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = original_coords[pyimageid - 1, -2:]
                resize_ratio = max(real_image_size) / img_size
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp

                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

            if shift_point2d:
                top_left = original_coords[pyimageid - 1, :2]
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                rescale_camera[pyimage.camera_id] = False

        return reconstruction

    def _build_ff_data(
        self,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        base_image_paths: list[str],
        image_paths: list[str],
        original_coords: np.ndarray,
    ):
        """Assemble EPO's feed-forward dict from raw VGGT outputs.

        Mirrors EPO's disk loaders step for step (``helpers/load.py``:
        ``_process_single_image``, ``_process_single_depth``,
        ``process_camera`` with ``load_with_pad=False``), so ``EPO.from_ff``
        sees the same inputs as a save-to-disk round-trip and lands in the
        same optimum:

        - ``"image"``: original pixels decoded to CHW uint8 (torchvision,
          PIL fallback), antialiased BICUBIC resize to
          ``(int(h*s), int(w*s))`` with ``s = res / max(w, h)``;
        - ``"depth"``/``"confidence"``: centered ``//2`` crop of VGGT's
          square map to those same dims (the disk path's follow-up resize
          is an exact identity at these sizes, so it is skipped);
        - ``"intrinsic"``: VGGT's square-space focals with the principal
          point at the float image centre ``(w*s/2, h*s/2)``.

        Extrinsics are untouched. Keyed by the relative image path
        (``"cam_id/image_name"``).
        """
        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image
        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize as tv_resize

        res = self.vggt_fixed_resolution

        def _process_one(i):
            """Build one image's ff_data entry (independent per image)."""
            base = base_image_paths[i]
            # original_coords[:, -2:] holds the original (width, height).
            width, height = original_coords[i, -2:]
            scale = res / max(width, height)
            new_w, new_h = int(width * scale), int(height * scale)

            # Decode → CHW uint8 RGB, same decoder + fallback as the disk path.
            try:
                rgb = read_image(image_paths[i], mode=ImageReadMode.UNCHANGED)
                if rgb.shape[0] == 1:
                    rgb = rgb.expand(3, -1, -1).contiguous()
                elif rgb.shape[0] == 4:
                    # RGBA → blend onto white, drop alpha (same float math +
                    # truncating uint8 cast as the disk decoder).
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

            # Centered crop of the square depth/confidence maps.
            crop_top = max((res - new_h) // 2, 0)
            crop_left = max((res - new_w) // 2, 0)
            depth_hw = torch.from_numpy(np.asarray(depth_map[i]).squeeze()).float()
            conf_hw = torch.from_numpy(np.asarray(depth_conf[i]).squeeze()).float()
            depth_hw = depth_hw[
                crop_top : crop_top + new_h, crop_left : crop_left + new_w
            ]
            conf_hw = conf_hw[
                crop_top : crop_top + new_h, crop_left : crop_left + new_w
            ]

            intr = torch.from_numpy(intrinsic[i].copy()).float()
            intr[0, 2] = width * scale / 2.0
            intr[1, 2] = height * scale / 2.0

            return base, {
                "image": img_t,
                "depth": depth_hw,
                "confidence": conf_hw,
                "pose": torch.from_numpy(extrinsic[i][:3, :4]).float(),
                "intrinsic": intr,
            }

        # Decode + resize is the bottleneck and releases the GIL, so thread it.
        # `map` preserves input order, so the result is identical to the
        # sequential loop (bit-exact ff_data parity is required).
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
        # BA-specific parameters
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
        """Run VGGT reconstruction on images and save results.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images to process (randomly sampled if exceeded).
            use_ba: Whether to use bundle adjustment.
            save_depth: Whether to write ``depths.pth`` (only when ``save``).
            save: Whether to write the reconstruction / depths / timings to disk.
                Set ``False`` to skip all disk I/O for clean benchmarking; the
                returned ``(ff_data, reconstruction, depths)`` can be written by
                the caller afterwards.
            one_camera_per_folder: Whether to assume one camera per folder.
            max_reproj_error: Maximum reprojection error for BA.
            shared_camera: Whether to use shared camera for all images.
            camera_type: Camera type for reconstruction.
            vis_thresh: Visibility threshold for tracks.
            query_frame_num: Number of frames to query for tracking.
            max_query_pts: Maximum number of query points.
            fine_tracking: Use fine tracking (slower but more accurate).
            conf_thres_value: Confidence threshold for depth filtering (without BA).
            max_points_for_colmap: Maximum 3D points for COLMAP (without BA).
        """
        if self.model is None:
            self.__init__(self.model_path)

        # if not output_path.endswith("sparse"):
        #     output_path = os.path.join(output_path, "sparse")
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

        # Load and preprocess images. VGGT's camera head is tuned for a
        # square frame, so both paths use the square-padded loader; the
        # non-BA path then hands the square outputs to EPO via ff_data.
        print(f"Loading {len(image_paths)} images...")
        t_start = time.time()
        images, original_coords = load_and_preprocess_images_square(
            image_paths, self.img_load_resolution
        )
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)
        timings["load_and_preprocess"] = time.time() - t_start

        # Run VGGT.
        print("Running VGGT model...")
        t_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        timings["run_vggt"] = time.time() - t_start

        # Reconstruct with or without BA
        if use_ba:
            if self.oom_safe:
                print("OOM-safe mode enabled: freeing VGGT model from memory...")
                del self.model  # free memory
                gc.collect()
                torch.cuda.empty_cache()
                self.model = None

            print("Running reconstruction with Bundle Adjustment...")
            t_start = time.time()
            reconstruction, recon_resolution, track_time, ba_time = (
                self._reconstruct_with_ba(
                    images,
                    extrinsic,
                    intrinsic,
                    depth_map,
                    depth_conf,
                    points_3d,
                    max_query_pts,
                    query_frame_num,
                    vis_thresh,
                    max_reproj_error,
                    shared_camera,
                    camera_type,
                    fine_tracking,
                    image_names=base_image_paths,
                )
            )
            timings["track_establishment"] = track_time
            timings["bundle_adjustment"] = ba_time
            timings["reconstruction_with_ba"] = time.time() - t_start
        else:
            print("Running reconstruction without Bundle Adjustment...")
            t_start = time.time()
            reconstruction, recon_resolution = self._reconstruct_without_ba(
                images,
                extrinsic,
                intrinsic,
                depth_conf,
                points_3d,
                conf_thres_value,
                max_points_for_colmap,
            )
            timings["reconstruction_without_ba"] = time.time() - t_start

        # Always build the depths dict for downstream tasks (disk save).
        depths = {}
        for i, img_path in enumerate(base_image_paths):
            stem = Path(img_path).with_suffix("").as_posix()
            depth_hw = torch.from_numpy(np.asarray(depth_map[i]).squeeze()).float()
            conf_hw = torch.from_numpy(np.asarray(depth_conf[i]).squeeze()).float()
            depths[stem] = {"depth": depth_hw, "confidence": conf_hw}

        # Build EPO's feed-forward dict (non-BA path only): VGGT depth/pose/
        # intrinsic with the letterbox cropped out, and the original sharp
        # image for the edge detector. Free the GPU image tensor afterwards.
        ff_data = None
        if not use_ba:
            t_start = time.time()
            ff_data = self._build_ff_data(
                extrinsic,
                intrinsic,
                depth_map,
                depth_conf,
                base_image_paths,
                image_paths,
                original_coords.cpu().numpy(),
            )
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
                original_coords.cpu().numpy(),
                recon_resolution,
                shift_point2d=use_ba,
                shared_camera=shared_camera,
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
        print(f"Run VGGT model:              {timings['run_vggt']:>8.2f}s")
        if use_ba:
            print(
                f"Track establishment:         {timings['track_establishment']:>8.2f}s"
            )
            print(f"Bundle adjustment:           {timings['bundle_adjustment']:>8.2f}s")
            print(
                f"Reconstruction (with BA):    {timings['reconstruction_with_ba']:>8.2f}s"
            )
        else:
            print(
                f"Reconstruction (without BA): {timings['reconstruction_without_ba']:>8.2f}s"
            )
            if "build_ff_data" in timings:
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
            original_coords,
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            points_3d,
        )

        # Delete reconstruction-related objects (they can be large)
        if "pred_tracks" in locals():
            del pred_tracks  # noqa: F821
        if "pred_vis_scores" in locals():
            del pred_vis_scores  # noqa: F821
        if "pred_confs" in locals():
            del pred_confs  # noqa: F821
        if "points_rgb" in locals():
            del points_rgb  # noqa: F821
        if "track_mask" in locals():
            del track_mask  # noqa: F821
        if "valid_track_mask" in locals():
            del valid_track_mask  # noqa: F821
        if "images" in locals():  # BA path still holds the GPU tensor
            del images

        gc.collect()
        torch.cuda.empty_cache()

        # Expose the timing breakdown (e.g. `run_vggt`) for callers that want
        # to report inference time separately from I/O.
        self.last_timings = timings

        # ff_data: EPO-ready feed-forward dict (native res, non-BA path);
        # None for the BA path. reconstruction: original-res pycolmap model.
        # depths: per-image {"depth", "confidence"} dict (the depths.pth
        # contents) — returned so a caller can write it after timing inference.
        return ff_data, reconstruction, depths


if __name__ == "__main__":
    import argparse

    # add dataset scene, and use-ba as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eth3d")
    parser.add_argument("--scene", type=str, default="door")

    parser.add_argument("--max-images", type=int, default=170)
    parser.add_argument("--use-ba", action="store_true")
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument(
        "--weights",
        type=str,
        default="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        help="Path to VGGT weights (.pt) or a URL.",
    )
    args = parser.parse_args()

    # print("Hardcoded BA")
    # args.use_ba = True

    vggt = VGGTWrapper(args.weights, cuda_id=args.cuda_id, oom_safe=args.use_ba)

    # setting paths
    base_path = "/data/mdurso"
    if os.path.exists(base_path):
        if args.dataset == "eth3d":
            input = f"{base_path}/eth3d/{args.scene}/images_by_k"
            output = f"{base_path}/results/vggt/eth3d/{args.scene}/sparse"

        elif args.dataset == "imc":
            input = f"{base_path}/imc/phototourism/{args.scene}/set_100/images"
            output = f"{base_path}/results/vggt/imc/{args.scene}/sparse"

        elif args.dataset == "mydataset":
            input = f"{base_path}/mydataset/{args.scene}/frames"
            output = f"{base_path}/results/vggt/mydataset/{args.scene}/sparse"

    else:
        base_path = "/home/mattia/Desktop/Repos/wrapper_factory"
        if args.dataset == "eth3d":
            input = f"{base_path}/benchmarks_3D/eth3d/{args.scene}/images_by_k"
            output = f"{base_path}/benchmarks_3D/results/vggt/eth3d/{args.scene}/sparse"

        elif args.dataset == "imc":
            input = f"{base_path}/benchmarks_2D/imc/data/phototourism/{args.scene}/set_100/images"
            output = f"{base_path}/benchmarks_3D/results/vggt/imc/{args.scene}/sparse"

        elif args.dataset == "mydataset":
            input = f"/home/mattia/Desktop/datasets/mydataset/data/{args.scene}/frames"
            output = (
                f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{args.scene}/sparse"
            )

    # reconstruction
    # scene = args.scene
    scene = "vienna_state_opera"
    input = f"/home/mattia/Desktop/datasets/mydataset/data_test/{scene}/frames"
    output = f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{scene}"

    ff_data, rec, depths = vggt.forward(
        input,
        output,
        max_images=args.max_images,
        use_ba=args.use_ba,
        query_frame_num=10,
        max_query_pts=2048,
        fine_tracking=False,
        shared_camera=True,
    )

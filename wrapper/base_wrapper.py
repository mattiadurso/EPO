"""Shared base class for the wrapper/ 3D foundation model wrappers.

Holds everything the 3DFM wrappers do identically:

- seeding, image discovery/sampling, boolean-mask thinning;
- ``_build_ff_data``: THE EPO feed-forward dict builder (image decode + resize
  + crop + tensor packing, threaded). There is exactly one implementation; a
  wrapper never overrides it. What differs per model is only *where its outputs
  sit relative to the original image*, and that is expressed by ``_ff_entries``
  (see below);
- ``_reconstruct``: THE track-less COLMAP builder (select pixels → thin →
  unproject → convert). Also a single implementation; what differs per model is
  only *which pixels are trustworthy*, expressed by ``_conf_mask``;
- ``_rescale_reconstruction``: cameras back to original resolution, with the
  per-model camera convention behind the ``_rescale_camera_params`` hook.

So each wrapper's ``forward()`` runs its model, packs the raw outputs into a
plain ``preds`` dict, and calls the shared builders. Only three small hooks are
model-specific:

- ``_ff_entries``: where the model's outputs sit relative to the original image
  — letterbox (VGGT family), aspect-preserving resize (DA3/Pi3X), cover-resize
  + centered crop (MapAnything), padded/rotated model space (DVLT);
- ``_conf_mask``: which pixels become 3D points — absolute threshold (default),
  percentile (DA3, MapAnything), sigmoid probability (Pi3X), plus per-model
  validity masks;
- ``_rescale_camera_params``: how the camera maps back to the original image.

What stays per-model: ``__init__``, ``_load_model``, the inference call, those
three hooks, and ``forward()``. See ``wrapper/__init__.py``'s ``WRAPPERS``
registry for the interface every wrapper must satisfy.
"""

import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


class BaseWrapper:
    """Helpers shared by every 3DFM wrapper."""

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _find_images(self, images_path: str) -> list[str]:
        """Find all images in the given path, including subdirectories.

        Args:
            images_path: Path to directory containing images.

        Returns:
            List of image file paths.

        Raises:
            ValueError: If no images are found.
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
    def _randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
        """Randomly keep at most ``max_trues`` True entries of a boolean mask.
        Used to limit the number of points for COLMAP.
        """
        true_indices = np.flatnonzero(mask)
        if true_indices.size <= max_trues:
            return mask
        sampled = np.random.choice(true_indices, size=max_trues, replace=False)
        limited = np.zeros(mask.size, dtype=bool)
        limited[sampled] = True
        return limited.reshape(mask.shape)

    def _ff_entries(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> list[dict]:
        """Normalize this model's outputs into per-image ff_data entries.

        Implemented by every wrapper; this is the *only* model-specific part of
        the feed-forward path. It says where the model's depth/pose/intrinsic
        sit relative to the original image, as plain dicts with keys:

            ``key``: ff_data key (the relative image path);
            ``image_path``: absolute path of the original image;
            ``resize_hw``: ``(h, w)`` the decoded image is resized to;
            ``crop_box``: ``(top, left, h, w)`` crop applied to the resized
                image afterwards, or ``None``;
            ``depth`` / ``confidence``: final ``(H, W)`` numpy maps,
                pixel-aligned with the (cropped) resized image;
            ``pose``: ``(3, 4)`` world-to-camera numpy matrix;
            ``intrinsic``: ``(3, 3)`` numpy matrix in the same frame.

        Raises:
            NotImplementedError: If the wrapper does not implement it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _ff_entries()")

    def _build_ff_data(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
    ) -> dict:
        """Build EPO's feed-forward dict. Shared by every wrapper; never overridden.

        Mirrors EPO's disk loaders step for step (``helpers/load.py``:
        ``_process_single_image`` with ``load_with_pad=False``), so
        ``EPO.from_ff`` sees the same inputs as a save-to-disk round-trip:
        the *original* sharp pixels (torchvision decode, PIL fallback,
        antialiased BICUBIC resize) feed the edge detector, while depth /
        confidence / pose / intrinsic come from the model, placed by the
        wrapper's :meth:`_ff_entries`.

        Args:
            preds: The model's raw outputs, as assembled by ``forward()``.
            base_image_paths: Relative image paths (the ff_data keys).
            image_paths: Absolute paths of the original images.

        Returns:
            Dict mapping each relative image path to EPO's per-image tensors.
        """
        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image
        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize as tv_resize

        entries = self._ff_entries(preds, base_image_paths, image_paths)

        def _process_one(entry):
            """Build one image's ff_data entry (independent per image)."""
            new_h, new_w = entry["resize_hw"]

            # Decode → CHW uint8 RGB, same decoder + fallback as the disk path.
            try:
                rgb = read_image(entry["image_path"], mode=ImageReadMode.UNCHANGED)
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
                img = Image.open(entry["image_path"])
                if img.mode == "RGBA":
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(bg, img)
                arr = np.asarray(img.convert("RGB"))
                rgb = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

            img_t = tv_resize(
                rgb,
                [new_h, new_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            if entry["crop_box"] is not None:
                top, left, crop_h, crop_w = entry["crop_box"]
                img_t = img_t[:, top : top + crop_h, left : left + crop_w]
            img_t = img_t.float().div_(255.0)

            return entry["key"], {
                "image": img_t,
                "depth": torch.from_numpy(entry["depth"]).float(),
                "confidence": torch.from_numpy(entry["confidence"]).float(),
                "pose": torch.from_numpy(entry["pose"]).float(),
                "intrinsic": torch.from_numpy(entry["intrinsic"]).float(),
            }

        # Decode + resize is the bottleneck and releases the GIL, so thread it.
        # `map` preserves input order, so the result is identical to the
        # sequential loop (bit-exact ff_data parity is required).
        max_workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_process_one, entries))

        result = {key: entry for key, entry in results}
        # sort the dict by key to make the ff_data deterministic (the thread pool
        # returns results in arbitrary order).
        return dict(sorted(result.items(), key=lambda item: item[0]))

    def _rescale_camera_params(
        self,
        params: np.ndarray,
        orig_wh: tuple[int, int],
        proc_hw: tuple[int, int],
    ) -> np.ndarray:
        """Map PINHOLE ``params`` from the processed frame to the original image.

        Default convention (VGGT-family): the model saw a *square*, letterboxed
        frame, so a single max-side ratio rescales the focals and the principal
        point is forced back to the image centre. Wrappers whose processed frame
        keeps the image's aspect ratio override this with
        :meth:`_rescale_camera_params_per_axis`.

        Args:
            params: PINHOLE ``[fx, fy, cx, cy]`` in processed-pixel space.
            orig_wh: Original image ``(width, height)``.
            proc_hw: Processed frame ``(height, width)``.

        Returns:
            Rescaled ``[fx, fy, cx, cy]``.
        """
        orig = np.asarray(orig_wh, dtype=np.float64)
        ratio = max(orig) / max(proc_hw)
        rescaled = np.asarray(params, dtype=np.float64) * ratio
        rescaled[-2:] = orig / 2  # principal point at the image centre
        return rescaled

    def _rescale_camera_params_per_axis(
        self,
        params: np.ndarray,
        orig_wh: tuple[int, int],
        frame_hw: tuple[int, int],
        crop_offset: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Rescale PINHOLE ``params`` by independent per-axis ratios.

        For models whose processed frame preserves the image aspect ratio: fx/cx
        and fy/cy scale by their own ratios and the *predicted* principal point
        is kept (not re-centred).

        Args:
            params: PINHOLE ``[fx, fy, cx, cy]`` in processed-pixel space.
            orig_wh: Original image ``(width, height)``.
            frame_hw: The ``(height, width)`` the params live in *before* any
                centered crop the model applied — i.e. the uncropped resized
                frame. Equal to the processed size when the model never crops.
            crop_offset: ``(left, top)`` of the model's centered crop, added to
                the principal point to lift it back into ``frame_hw``.
        """
        orig_w, orig_h = orig_wh
        frame_h, frame_w = frame_hw
        left, top = crop_offset
        fx, fy, cx, cy = np.asarray(params, dtype=np.float64)
        scale_x = orig_w / frame_w
        scale_y = orig_h / frame_h
        return np.array(
            [
                fx * scale_x,
                fy * scale_y,
                (cx + left) * scale_x,
                (cy + top) * scale_y,
            ]
        )

    def _rescale_reconstruction(
        self,
        reconstruction,
        base_image_paths: list[str],
        original_sizes,
        proc_hw: tuple[int, int],
    ):
        """Rescale cameras to original resolutions and rename images.

        The feed-forward path builds one camera per image and carries no 2D
        observations, so each camera is rescaled exactly once and nothing needs
        shifting. The per-model camera convention lives in
        :meth:`_rescale_camera_params`.

        Args:
            reconstruction: pycolmap model in processed-pixel space.
            base_image_paths: Relative image paths, indexed by ``image_id - 1``.
            original_sizes: Per-image original ``(width, height)``.
            proc_hw: Processed frame ``(height, width)``.
        """
        for image_id in reconstruction.images:
            pyimage = reconstruction.images[image_id]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = base_image_paths[image_id - 1]

            # int(): callers pass (w, h) as python ints, np.float32 (VGGT's
            # original_coords) or np.int64 — pycolmap's width/height setters
            # only accept an integral type.
            orig_w = int(original_sizes[image_id - 1][0])
            orig_h = int(original_sizes[image_id - 1][1])
            pycamera.params = self._rescale_camera_params(
                pycamera.params, (orig_w, orig_h), proc_hw
            )
            pycamera.width = orig_w
            pycamera.height = orig_h

        return reconstruction

    def _ff_entries_letterbox(
        self,
        preds: dict,
        base_image_paths: list[str],
        image_paths: list[str],
        res: int,
    ) -> list[dict]:
        """``_ff_entries`` for models that infer on a square, letterboxed frame.

        Shared by the VGGT family (VGGT, VGGT-Omega), which pad the image to a
        square of side ``res`` before inference. Mirrors EPO's disk loaders
        (``helpers/load.py`` with ``load_with_pad=False``):

        - image: antialiased BICUBIC resize to ``(int(h*s), int(w*s))`` with
          ``s = res / max(w, h)`` — no crop, the *model* padded, not the image;
        - depth/confidence: centered ``//2`` crop of the square map to those
          same dims (the disk path's follow-up resize is an exact identity at
          these sizes, so it is skipped);
        - intrinsic: square-space focals with the principal point moved to the
          float image centre ``(w*s/2, h*s/2)``.

        Expects ``preds`` with ``extrinsic``, ``intrinsic``, ``depth_map``,
        ``depth_conf`` and ``original_coords`` (whose last two columns are the
        original ``(width, height)``).
        """
        entries = []

        for i, key in enumerate(base_image_paths):
            width, height = preds["original_coords"][i, -2:]
            scale = res / max(width, height)
            new_w, new_h = int(width * scale), int(height * scale)

            top = max((res - new_h) // 2, 0)
            left = max((res - new_w) // 2, 0)
            depth = np.asarray(preds["depth_map"][i]).squeeze()
            conf = np.asarray(preds["depth_conf"][i]).squeeze()

            intr = preds["intrinsic"][i].copy()
            intr[0, 2] = width * scale / 2.0
            intr[1, 2] = height * scale / 2.0

            entries.append(
                {
                    "key": key,
                    "image_path": image_paths[i],
                    "resize_hw": (new_h, new_w),
                    "crop_box": None,
                    "depth": depth[top : top + new_h, left : left + new_w],
                    "confidence": conf[top : top + new_h, left : left + new_w],
                    "pose": preds["extrinsic"][i][:3, :4],
                    "intrinsic": intr,
                }
            )

        return entries

    @staticmethod
    def _gather_masked_points(
        conf_mask: np.ndarray, points_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather world points and pixel coordinates at masked locations.

        Args:
            conf_mask: (S, H, W) boolean selection.
            points_map: (S, H, W, 3) per-pixel world points.

        Returns:
            points_3d (P, 3) and points_xyf (P, 3) with per-point
            ``(x, y, frame_idx)``, in frame-major row-major order (the same
            order boolean indexing ``points_map[conf_mask]`` produces).
        """
        points_3d, points_xyf = [], []
        for i in range(conf_mask.shape[0]):
            v, u = np.nonzero(conf_mask[i])
            if v.size == 0:
                continue
            points_3d.append(points_map[i, v, u])
            points_xyf.append(
                np.stack([u, v, np.full_like(u, i)], axis=-1).astype(np.float64)
            )
        return np.concatenate(points_3d, axis=0), np.concatenate(points_xyf, axis=0)

    def _masked_points_to_colmap(
        self,
        points_3d: np.ndarray,
        points_xyf: np.ndarray,
        points_rgb: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        image_size: np.ndarray,
    ):
        """Convert gathered points + cameras to a track-less reconstruction."""
        from np_to_colmap import batch_np_matrix_to_pycolmap_wo_track

        print("Converting to COLMAP format")
        return batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )

    @staticmethod
    def _points_rgb(
        images: torch.Tensor, size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Model-input batch → per-pixel colors for the sparse cloud.

        Args:
            images: (S, 3, H, W) float tensor in [0, 1] (the model's input).
            size: Optional ``(h, w)`` to resize to first — needed when the model
                infers at a different size than the batch was loaded at.

        Returns:
            (S, H, W, 3) uint8 colors.
        """
        if size is not None:
            images = F.interpolate(
                images, size=size, mode="bilinear", align_corners=False
            )
        rgb = (images.cpu().numpy() * 255).astype(np.uint8)
        return rgb.transpose(0, 2, 3, 1)

    @staticmethod
    def _unproject_masked(
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unproject masked depth pixels to world points.

        Used for models that predict depth but no pointmap, so only the selected
        pixels are ever unprojected.

        Args:
            depth_map: (S, H, W) depth.
            intrinsic: (S, 3, 3) pinhole intrinsics (processed-pixel space).
            extrinsic: (S, 3, 4) world-to-camera.
            mask: (S, H, W) boolean selection.

        Returns:
            points_3d (P, 3) world points and points_xyf (P, 3) with per-point
            ``(x, y, frame_idx)`` in processed-pixel coordinates.
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
        return np.concatenate(points, axis=0), np.concatenate(xyf, axis=0)

    def _conf_mask(self, preds: dict, conf_thres: float) -> np.ndarray:
        """Select the pixels that become 3D points. The per-model policy.

        Default (VGGT family): an absolute threshold on the model's confidence.
        Wrappers whose confidence means something else — a percentile (DA3,
        MapAnything), a sigmoid probability (Pi3X) — or that carry extra
        validity masks (DVLT's pad pixels, MapAnything's ambiguity mask, Pi3X's
        depth-discontinuity mask) override this.

        Args:
            preds: The model's raw outputs, as assembled by ``forward()``.
            conf_thres: The wrapper's confidence threshold, whatever it means.

        Returns:
            (S, H, W) boolean mask.
        """
        return preds["depth_conf"] >= conf_thres

    def _reconstruct(
        self,
        preds: dict,
        conf_thres: float,
        max_points_for_colmap: int,
    ):
        """Build the track-less COLMAP model. Shared by every wrapper.

        Selects pixels with :meth:`_conf_mask` (the only model-specific step),
        thins them to ``max_points_for_colmap``, then unprojects and converts.
        World points come from ``preds["points"]`` when the model predicts a
        pointmap, otherwise from unprojecting ``preds["depth_map"]``.

        Args:
            preds: Model outputs; needs ``extrinsic``, ``intrinsic``,
                ``depth_conf``, ``points_rgb`` (S, H, W, 3) uint8, and either
                ``points`` (S, H, W, 3) or ``depth_map`` (S, H, W).
            conf_thres: Confidence threshold, passed to :meth:`_conf_mask`.
            max_points_for_colmap: Cap on the points kept across all images.

        Returns:
            The ``pycolmap.Reconstruction``, in processed-pixel space.
        """
        conf_mask = self._conf_mask(preds, conf_thres)
        conf_mask = self._randomly_limit_trues(conf_mask, max_points_for_colmap)
        height, width = conf_mask.shape[-2:]

        if "points" in preds:
            points_3d, points_xyf = self._gather_masked_points(
                conf_mask, preds["points"]
            )
        else:
            points_3d, points_xyf = self._unproject_masked(
                preds["depth_map"], preds["intrinsic"], preds["extrinsic"], conf_mask
            )

        return self._masked_points_to_colmap(
            points_3d,
            points_xyf,
            preds["points_rgb"][conf_mask],
            preds["extrinsic"],
            preds["intrinsic"],
            image_size=np.array([width, height]),
        )

    @classmethod
    def _cli_main(cls, default_model_path: str | None = None) -> None:
        """Minimal standalone CLI so a single wrapper can be smoke-tested.

        Runs the wrapper on one folder of images and writes a COLMAP model +
        ``depths.pth``, e.g. ``python wrapper/vggt_wrapper.py --images_path
        scene/images/1 --output_path /tmp/out/sparse``. For batch runs across
        benchmark datasets use ``wrapper/run_for_dataset.py``.

        Args:
            default_model_path: Weights path/URL/HF repo id used when
                ``--model_path`` is omitted; ``None`` falls back to the
                wrapper's ``__init__`` default.
        """
        import argparse

        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument(
            "--images_path", required=True, help="Folder of input images."
        )
        parser.add_argument(
            "--output_path", required=True, help="Where to write the COLMAP model."
        )
        parser.add_argument(
            "--model_path",
            default=default_model_path,
            help="Weights path/URL/HF repo id; defaults to the wrapper's checkpoint.",
        )
        parser.add_argument("--cuda_id", type=int, default=0)
        parser.add_argument(
            "--max_images", type=int, default=150, help="Cap on images (-1 for all)."
        )
        parser.add_argument(
            "--ba", action="store_true", help="Bundle adjustment (VGGT only)."
        )
        args = parser.parse_args()

        kwargs = {"cuda_id": args.cuda_id}
        if args.model_path is not None:
            kwargs["model_path"] = args.model_path
        wrapper = cls(**kwargs)
        wrapper.forward(
            args.images_path,
            args.output_path,
            max_images=args.max_images,
            use_ba=args.ba,
        )

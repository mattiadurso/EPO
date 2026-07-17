"""Any2Full depth-densification wrapper for EPO.

Unlike the other ``wrapper/`` entries, this is **not** a 3D foundation model:
it takes an *existing* COLMAP reconstruction whose depth maps are sparse and
completes them into dense ones. The intended input is EPO's own export
(``to_colmap(..., save_depth=True)``), whose ``depths.pth`` only carries depth
at the sampled edge pixels — exactly the sparse-prompt regime Any2Full
(https://github.com/zhiyuandaily/Any2Full) is trained for.

Poses and intrinsics are passed through untouched; only the depth (and the
point cloud unprojected from it) changes::

    from wrapper.any2full_wrapper import Any2FullWrapper

    model = Any2FullWrapper()
    recon, dense = model.forward("out/sparse_vggt_epo", "scene/images")
    # → writes out/dense_vggt_epo/{cameras,images,points3D}.bin + depths.pth

The output folder defaults to a sibling of the input with ``sparse`` swapped
for ``dense`` (``sparse_vggt_epo`` → ``dense_vggt_epo``).

Weights: ``third_party/Any2Full/checkpoints/Any2Full_vitl.pth.tar``
(https://huggingface.co/zhiyuandaily/Any2Full/tree/main/checkpoints).
"""

import os
import sys

# The Any2Full repo uses top-level ``model`` / ``utils`` packages, so its root
# only goes on sys.path when this wrapper is imported (it is imported lazily by
# demo_epo.py) to keep those generic names out of the way of normal runs.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ANY2FULL_ROOT = os.path.join(_ROOT, "third_party", "Any2Full")
if _ANY2FULL_ROOT not in sys.path:
    sys.path.insert(0, _ANY2FULL_ROOT)

import logging  # noqa: E402
import time  # noqa: E402
from argparse import Namespace  # noqa: E402
from collections import OrderedDict, defaultdict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
import torch  # noqa: E402
from model.ours.any2full import Any2Full  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = os.path.join(
    _ANY2FULL_ROOT, "checkpoints", "Any2Full_vitl.pth.tar"
)

# ImageNet statistics — Any2Full's DINOv2 encoder expects them (run_any2full.py).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def dense_output_path(reconstruction_path: str) -> str:
    """Sibling folder of ``reconstruction_path`` with ``sparse`` → ``dense``.

    ``out/sparse_vggt_epo`` → ``out/dense_vggt_epo``. Folder names that do not
    start with ``sparse`` are simply prefixed with ``dense_``.
    """
    path = Path(reconstruction_path)
    name = path.name
    if name.startswith("sparse"):
        dense_name = "dense" + name[len("sparse") :]
    else:
        dense_name = f"dense_{name}"
    return str(path.parent / dense_name)


class Any2FullWrapper:
    """Densify the depth maps of a COLMAP reconstruction with Any2Full."""

    def __init__(
        self,
        model_path: str = DEFAULT_CHECKPOINT,
        cuda_id: int = 0,
        encoder: str = "vitl",
        max_depth: float = 1e3,
        min_depth: float = 1e-6,
    ):
        """Initialize the Any2Full wrapper.

        Args:
            model_path: Path to the Any2Full checkpoint (``.pth.tar``).
            cuda_id: CUDA device index.
            encoder: DINOv2 backbone size (``vits`` / ``vitb`` / ``vitl``);
                must match the checkpoint.
            max_depth: Upper clamp on the predicted depth.
            min_depth: Lower clamp on the predicted depth.
        """
        self.device = (
            torch.device(f"cuda:{cuda_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.last_timings = {}
        # Any2Full reads its hyper-parameters off an argparse Namespace.
        # ``stage`` only matters when a DepthAnything checkpoint is loaded on
        # top (``da_ckpt_path``), which we never do: the released checkpoint is
        # complete and loads strict.
        self.args = Namespace(
            init_scailing=True,
            stage=1,
            max_depth=max_depth,
            min_depth=min_depth,
        )
        self.model = self._load_model(model_path, encoder)

    def _load_model(self, model_path: str, encoder: str) -> Any2Full:
        """Build Any2Full and load ``model_path`` into it."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Any2Full checkpoint not found at {model_path}. Download it from "
                "https://huggingface.co/zhiyuandaily/Any2Full/tree/main/checkpoints"
            )
        model = Any2Full(encoder=encoder, da_ckpt_path=None, args=self.args)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        model.load_state_dict(cleaned, strict=True)
        del checkpoint, state, cleaned
        return model.to(self.device).eval()

    def _load_rgb(
        self, image_path: str, hw: tuple[int, int]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Load an image and resize it to ``hw`` (the sparse depth's size).

        Returns:
            Tuple of the ImageNet-normalized (1, 3, H, W) tensor fed to the
            model and the matching (H, W, 3) uint8 array used for point colors.
        """
        rgb = Image.open(image_path).convert("RGB")
        rgb = rgb.resize((hw[1], hw[0]), Image.BICUBIC)
        rgb_uint8 = np.array(rgb)  # (H, W, 3)

        tensor = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device), rgb_uint8

    @torch.no_grad()
    def _densify(self, rgb: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """Run Any2Full on a batch of (rgb, sparse depth) pairs.

        Args:
            rgb: (B, 3, H, W) ImageNet-normalized images.
            sparse: (B, 1, H, W) sparse depth prompts, 0 where unknown.

        Returns:
            (B, H, W) dense depth. All items must share H, W — Any2Full resizes
            the whole batch tensor in one go.
        """
        pred = self.model({"rgb": rgb, "dep": sparse.to(self.device).float()})["pred"]
        return pred.squeeze(1)

    def _unproject(
        self,
        depth: torch.Tensor,
        rgb_uint8: np.ndarray,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unproject a depth map into world points + colors.

        Each image contributes at most ``max_points``, drawn uniformly at random
        from its valid pixels. Sampling per depth map (rather than thinning the
        fused cloud afterwards) bounds the merged cloud at the source; drawing
        at random rather than on a stride grid avoids aliasing the sample
        pattern against image structure.

        The depth map lives at the resolution EPO ran at, while the COLMAP
        camera is at the original image resolution, so both the intrinsics and
        the depth are converted by the ratio between the two (this mirrors
        ``helpers.reconstruction.build_reconstruction``, which unprojects at
        EPO's resolution and then divides the points by that same scale).
        """
        h, w = depth.shape[-2:]
        scale = 0.5 * (h / camera.height + w / camera.width)

        # Rays are scale-invariant, so use the depth-map pixel grid together
        # with the intrinsics scaled into that same space.
        fx = camera.focal_length_x * scale
        fy = camera.focal_length_y * scale
        cx = camera.principal_point_x * scale
        cy = camera.principal_point_y * scale

        vs, us = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        # Depth is in EPO's scaled space; the reconstruction's translations are
        # not, so bring z back to the reconstruction's world scale.
        z = depth / scale
        points_cam = torch.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], dim=-1)

        valid = torch.isfinite(z) & (z > 0)
        points_cam = points_cam[valid]
        colors = torch.from_numpy(rgb_uint8).to(depth.device)[valid]

        if points_cam.shape[0] > max_points:
            keep = torch.randperm(points_cam.shape[0], device=depth.device)[:max_points]
            points_cam, colors = points_cam[keep], colors[keep]

        cam_from_world = image.cam_from_world()
        rotation = torch.from_numpy(cam_from_world.rotation.matrix()).to(points_cam)
        translation = torch.from_numpy(cam_from_world.translation).to(points_cam)
        points_world = (points_cam - translation) @ rotation  # R^T @ (X_c - t)

        return (
            points_world.double().cpu().numpy(),
            colors.to(torch.uint8).cpu().numpy(),
        )

    def forward(
        self,
        reconstruction_path: str,
        images_path: str,
        output_path: str | None = None,
        depths: dict | None = None,
        max_points_per_image: int = 10_000,
        batch_size: int = 4,
        save: bool = True,
    ) -> tuple[pycolmap.Reconstruction, dict]:
        """Densify the depth maps of a COLMAP reconstruction.

        Args:
            reconstruction_path: COLMAP folder with sparse per-image depth in
                ``depths.pth`` (e.g. EPO's ``sparse_<model>_epo`` export).
            images_path: Root folder of the RGB images; the reconstruction's
                image names are resolved relative to it.
            output_path: Where to write the densified model. Defaults to
                :func:`dense_output_path` of ``reconstruction_path``.
            depths: Sparse depths dict (``{image_stem: {"depth": (H, W)}}``) to
                use instead of reading ``reconstruction_path/depths.pth``.
            max_points_per_image: Cap on the points each image contributes,
                drawn uniformly at random from its depth map before fusion.
                This avoids OOM.
            batch_size: Images per Any2Full forward pass. Only images with the
                same depth-map size can share a batch (Any2Full resizes the
                whole tensor at once), so mixed-resolution scenes batch within
                each size group.
            save: If False, skip the disk writes and only return the results.

        Returns:
            Tuple of the densified ``pycolmap.Reconstruction`` and the dense
            depths dict (same layout as the input ``depths.pth``).

        Raises:
            FileNotFoundError: If the sparse ``depths.pth`` is missing.
        """
        if depths is None:
            depths_file = Path(reconstruction_path) / "depths.pth"
            if not depths_file.exists():
                raise FileNotFoundError(
                    f"No depths.pth in {reconstruction_path}. EPO writes it with "
                    "`to_colmap(..., save_depth=True)`."
                )
            depths = torch.load(depths_file, map_location="cpu", weights_only=False)

        if output_path is None:
            output_path = dense_output_path(reconstruction_path)

        recon = pycolmap.Reconstruction(reconstruction_path)
        # Poses, cameras and images are passed through; only the point cloud
        # (unprojected from the densified depth) is rebuilt.
        for point3D_id in list(recon.points3D.keys()):
            recon.delete_point3D(point3D_id)

        # Group by depth-map size: Any2Full resizes the whole batch tensor in
        # one call, so only same-sized images can share a forward pass.
        by_size = defaultdict(list)
        for image in sorted(recon.images.values(), key=lambda im: im.name):
            entry = depths.get(image.name.split(".")[0])
            if entry is None:
                continue
            sparse = torch.as_tensor(entry["depth"]).squeeze()
            by_size[tuple(sparse.shape[-2:])].append((image, sparse))
        num_images = sum(len(v) for v in by_size.values())

        dense_depths = {}
        total_points = 0
        start = time.time()
        with tqdm(total=num_images, desc="Densifying depths") as pbar:
            for hw, items in by_size.items():
                for i in range(0, len(items), batch_size):
                    chunk = items[i : i + batch_size]

                    # Decode + resize releases the GIL, so loading the batch's
                    # images in parallel overlaps with nothing else running here.
                    with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                        loaded = list(
                            pool.map(
                                lambda it, hw=hw: self._load_rgb(
                                    os.path.join(images_path, it[0].name), hw
                                ),
                                chunk,
                            )
                        )
                    rgb = torch.cat([t for t, _ in loaded])  # (B, 3, H, W)
                    sparse = torch.stack([s for _, s in chunk]).unsqueeze(1)

                    dense = self._densify(rgb, sparse)  # (B, H, W)

                    for (image, _), depth, (_, rgb_uint8) in zip(
                        chunk, dense, loaded, strict=True
                    ):
                        dense_depths[image.name.split(".")[0]] = {"depth": depth.cpu()}
                        points, colors = self._unproject(
                            depth,
                            rgb_uint8,
                            image,
                            recon.cameras[image.camera_id],
                            max_points_per_image,
                        )
                        # Empty track: these points have no 2D observations (see
                        # the same note in helpers/reconstruction.py — a dangling
                        # track element produces a model COLMAP can serialize but
                        # not read back).
                        for point, color in zip(points, colors, strict=True):
                            recon.add_point3D(point, pycolmap.Track(), color)
                        total_points += len(points)
                    pbar.update(len(chunk))

        self.last_timings = {"run_any2full": time.time() - start}
        logger.info(
            f"Fused {total_points:,} points from {len(dense_depths)} depth maps "
            f"(randomly subsampled to <= {max_points_per_image:,} per image)"
        )

        if save:
            os.makedirs(output_path, exist_ok=True)
            recon.write_binary(output_path)
            torch.save(dense_depths, os.path.join(output_path, "depths.pth"))

        return recon, dense_depths


if __name__ == "__main__":
    # Standalone smoke test: densify an existing COLMAP model's depths.pth.
    # Any2Full is a depth-completion step, not a 3DFM, so its CLI takes an
    # existing reconstruction rather than raw images.
    import argparse

    _parser = argparse.ArgumentParser(description=Any2FullWrapper.__doc__)
    _parser.add_argument(
        "--reconstruction_path",
        required=True,
        help="COLMAP folder with sparse per-image depths.pth (e.g. an EPO export).",
    )
    _parser.add_argument(
        "--images_path", required=True, help="Root folder of the RGB images."
    )
    _parser.add_argument(
        "--output_path",
        default=None,
        help="Where to write the densified model; defaults to a dense_* sibling.",
    )
    _parser.add_argument("--model_path", default=DEFAULT_CHECKPOINT)
    _parser.add_argument("--cuda_id", type=int, default=0)
    _args = _parser.parse_args()

    Any2FullWrapper(_args.model_path, cuda_id=_args.cuda_id).forward(
        _args.reconstruction_path,
        _args.images_path,
        output_path=_args.output_path,
    )

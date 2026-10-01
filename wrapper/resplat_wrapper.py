"""ReSplat feed-forward 3D Gaussian Splatting wrapper for EPO.

Like ``any2full_wrapper``, this is **not** a 3D foundation model: it takes an
*existing* posed COLMAP reconstruction (EPO's ``sparse_<model>_epo`` export)
plus its images, and predicts 3D Gaussians in one forward pass with ReSplat
(Xu et al., https://github.com/cvg/resplat). Poses and intrinsics are only
read; nothing is optimised per scene::

    from wrapper.resplat_wrapper import ReSplatWrapper

    model = ReSplatWrapper()
    ply = model.forward("out/sparse_vggt_omega_epo", "scene/images")
    # -> out/splat_vggt_omega_epo/gaussians.ply

The PLY uses the standard 3DGS layout (positions, SH degree 3, logit
opacity, log scale, wxyz quaternion), so any 3DGS viewer opens it, and it is
in the reconstruction's world frame, so it overlays the COLMAP model.

All registered images are context views at 256x384 (portrait views are
turned 90 deg to landscape: ReSplat's encoder needs W >= H):
peak ~19 GiB at 131 views, ~21.5 GiB at 150, on a 24 GB GPU. Setup (CUDA
extensions, patch for many views): ``bash scripts/install_resplat.sh``. Weights
(``resplat-base-dl3dv-256x448-view32``) download from the Hugging Face Hub on
first use unless they are in ``third_party/resplat/pretrained/``.
"""

import os
import sys

# ReSplat's code lives in a top-level ``src`` package, so its root only goes
# on sys.path when this wrapper is imported (lazily, by demo_epo.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESPLAT_ROOT = os.path.join(os.path.dirname(_HERE), "third_party", "resplat")
if _RESPLAT_ROOT not in sys.path:
    sys.path.insert(0, _RESPLAT_ROOT)

import logging  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
import torch  # noqa: E402
from gsplat import spherical_harmonics  # noqa: E402
from PIL import Image  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

logger = logging.getLogger(__name__)

CHECKPOINT = "resplat-base-dl3dv-256x448-view32-439b63a6.pth"
HF_REPO = "haofeixu/resplat"
NUM_REFINE = 4  # recurrent refinement steps the checkpoint was trained with
NEAR, FAR = 0.01, 200.0  # ReSplat's DL3DV depth range


def splat_output_path(reconstruction_path: str) -> str:
    """Sibling folder of ``reconstruction_path`` with ``sparse`` -> ``splat``.

    ``out/sparse_vggt_epo`` -> ``out/splat_vggt_epo``. Folder names that do
    not start with ``sparse`` are simply prefixed with ``splat_``.
    """
    path = Path(reconstruction_path)
    name = path.name
    if name.startswith("sparse"):
        return str(path.parent / ("splat" + name[len("sparse") :]))
    return str(path.parent / f"splat_{name}")


def _sh_basis(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """[M, K] values of gsplat's real SH basis functions at unit ``dirs``."""
    k = (degree + 1) ** 2
    one_hot = torch.eye(k, device=dirs.device)[None, :, :, None]
    coeffs = one_hot.expand(len(dirs), k, k, 3).reshape(-1, k, 3)
    flat_dirs = dirs[:, None].expand(-1, k, 3).reshape(-1, 3)
    return spherical_harmonics(degree, flat_dirs, coeffs)[:, 0].view(len(dirs), k)


def sh_rotation(rotation: torch.Tensor, degree: int) -> torch.Tensor:
    """[K, K] matrix ``A`` taking SH coefficients ``c`` to ``A @ c`` under ``rotation``.

    Gaussians moved by ``rotation`` (3x3) look the same along ``rotation @ d``
    as before along ``d`` once their coefficients are mapped by ``A``. Fitted
    by least squares in gsplat's own basis: e3nn's Wigner-D (ReSplat's
    ``rotate_sh``) uses a different basis and is off by O(1) for degree >= 1.
    """
    gen = torch.Generator().manual_seed(0)
    dirs = torch.randn(512, 3, generator=gen).to(rotation.device)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    y_new = _sh_basis(dirs, degree)  # Y(d), d in the new frame
    y_old = _sh_basis(dirs @ rotation.float(), degree)  # Y(R^T d)
    return torch.linalg.lstsq(y_new.double().cpu(), y_old.double().cpu()).solution


class ReSplatWrapper:
    """Predict 3D Gaussians for a posed COLMAP reconstruction with ReSplat."""

    def __init__(
        self,
        model_path: str | None = None,
        cuda_id: int = 0,
        image_shape: tuple[int, int] = (256, 384),
    ):
        """Initialize the ReSplat wrapper.

        Args:
            model_path: ReSplat checkpoint. Defaults to
                ``third_party/resplat/pretrained/<CHECKPOINT>`` if present,
                else the Hugging Face Hub copy.
            cuda_id: CUDA device index.
            image_shape: Landscape (H, W) of the context views, multiples
                of 64. 256x384 is the largest that fits ~150 views in 24 GB.
        """
        self.device = torch.device(f"cuda:{cuda_id}")
        # Own the cuDNN state (see Any2FullWrapper): EPO's fix_seed leaves
        # benchmark=True process-wide, and autotune only adds start-up time.
        torch.backends.cudnn.benchmark = False
        # 150 views peak at ~21.5 GiB; without expandable segments ~2.8 GiB
        # of fragmentation tips that over a 24 GB card. Applies to segments
        # allocated from here on, so release the earlier stages' cache first.
        torch.cuda.empty_cache()
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        self.image_shape = tuple(image_shape)
        self.last_timings = {}
        self.encoder, self.decoder = self._load_model(model_path)

    def _load_model(self, model_path: str | None):
        """Build ReSplat's encoder + decoder from its Hydra config and load weights."""
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from src.config import load_typed_root_config
        from src.global_cfg import set_cfg
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder

        h, w = self.image_shape
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=os.path.join(_RESPLAT_ROOT, "config"), version_base=None
        ):
            cfg_dict = compose(
                config_name="main",
                overrides=[
                    "+experiment=dl3dv",
                    "mode=test",
                    f"model.encoder.num_refine={NUM_REFINE}",
                    f"dataset.image_shape=[{h},{w}]",
                    f"dataset.ori_image_shape=[{h},{w}]",
                ],
            )
        set_cfg(cfg_dict)
        cfg = load_typed_root_config(cfg_dict)
        encoder, _ = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder, cfg.dataset)

        state = torch.load(self._checkpoint(model_path), map_location="cpu")
        state = state.get("state_dict", state)
        encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in state.items()}, strict=True
        )
        return encoder.to(self.device).eval(), decoder.to(self.device).eval()

    @staticmethod
    def _checkpoint(model_path: str | None) -> str:
        """Local checkpoint path, downloading it from the Hub if needed."""
        if model_path is not None:
            return model_path
        local = os.path.join(_RESPLAT_ROOT, "pretrained", CHECKPOINT)
        if os.path.exists(local):
            return local
        from huggingface_hub import hf_hub_download

        return hf_hub_download(HF_REPO, CHECKPOINT)

    @staticmethod
    def _load_images(paths: list[str], h: int, w: int) -> torch.Tensor:
        """[V, 3, H, W] in [0, 1]; JPEG draft decoding keeps 16 MP inputs fast."""

        def load(path):
            img = Image.open(path)
            img.draft("RGB", (2 * w, 2 * h))
            img = img.convert("RGB").resize((w, h), Image.LANCZOS)
            return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)

        # Decode + resize release the GIL, so threads overlap them.
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
            return torch.stack(list(pool.map(load, paths))).float() / 255

    @staticmethod
    def _cameras(recon: pycolmap.Reconstruction):
        """Sorted images, their c2w [N, 4, 4] and image-normalised K [N, 3, 3]."""
        images = sorted(recon.images.values(), key=lambda im: im.name)
        c2w, intrinsics = [], []
        for image in images:
            camera = recon.cameras[image.camera_id]
            if camera.model.name not in ("SIMPLE_PINHOLE", "PINHOLE"):
                raise ValueError(f"ReSplat needs pinhole cameras, got {camera.model}")
            norm = np.diag([1 / camera.width, 1 / camera.height, 1.0])
            intrinsics.append(norm @ camera.calibration_matrix())
            w2c = np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]])
            c2w.append(np.linalg.inv(w2c))
        c2w = torch.from_numpy(np.stack(c2w).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
        return images, c2w, intrinsics

    @staticmethod
    def _to_landscape(rgb, c2w, intrinsics):
        """Views turned 90 deg clockwise, as ReSplat's encoder needs W >= H.

        Normalised pixels map (u, v) -> (1 - v, u) and camera axes to
        x' = -y, y' = x, so the world-frame Gaussians are unchanged.
        """
        rot = torch.eye(4)
        rot[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        pix = torch.tensor([[0.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rgb = torch.rot90(rgb, k=-1, dims=(-2, -1))
        return rgb, c2w @ rot.T, pix @ intrinsics @ rot[:3, :3].T

    @torch.no_grad()
    def _predict(
        self, images: torch.Tensor, c2w: torch.Tensor, intrinsics: torch.Tensor
    ):
        """ReSplat's Gaussians for all views, in the middle view's camera frame."""
        num_views = len(c2w)
        pivot = c2w[num_views // 2]  # ReSplat aligns poses to the middle view
        context = {
            "image": images[None],
            "extrinsics": (torch.linalg.inv(pivot) @ c2w)[None],
            "intrinsics": intrinsics[None],
            "near": torch.full((1, num_views), NEAR),
            "far": torch.full((1, num_views), FAR),
            "index": torch.arange(num_views)[None],
        }
        context = {k: v.to(self.device) for k, v in context.items()}
        shim = self.encoder.get_data_shim()  # crops to the patch size; needs a target
        context = shim({"context": context, "target": context})["context"]
        # TF32 matmuls, as ReSplat's own inference script sets them.
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        try:
            out = self.encoder(context, global_step=0, deterministic=False)
            out = self.encoder.forward_update(
                context,
                None,  # no target views: only the context renders are needed
                out["condition_features"],
                out["gaussians"],
                self.decoder,
                None,  # context_remain
            )
        finally:
            torch.set_float32_matmul_precision(precision)
        return out["gaussian"][-1], pivot.to(self.device)

    @staticmethod
    def _to_world(gaussians, pivot: torch.Tensor) -> dict:
        """Gaussians moved from the pivot camera frame to world, as PLY fields."""
        rot, trans = pivot[:3, :3], pivot[:3, 3]
        means = gaussians.means[0] @ rot.T + trans
        # Scales/rotations come from the world covariance: ReSplat's own
        # ``rotations`` field stays in each source camera's frame.
        rot64 = rot.double().cpu()  # batched CUDA eigh asks for ~400 GiB at 800k
        cov = rot64 @ gaussians.covariances[0].double().cpu() @ rot64.T
        var, axes = torch.linalg.eigh(cov)
        axes[..., 2] *= torch.linalg.det(axes)[..., None]  # proper rotations
        quat = Rotation.from_matrix(axes.cpu().numpy()).as_quat()  # xyzw
        sh = gaussians.harmonics[0].transpose(1, 2)  # [N, K, 3]
        degree = int(sh.shape[1] ** 0.5) - 1
        sh = torch.einsum("jk,nkc->njc", sh_rotation(rot, degree).to(sh), sh)
        opacity = gaussians.opacities[0].clamp(1e-6, 1 - 1e-6)
        return {
            "xyz": means.cpu().numpy(),
            "f_dc": sh[:, 0].cpu().numpy(),
            "f_rest": sh[:, 1:].transpose(1, 2).flatten(1).cpu().numpy(),
            "opacity": torch.logit(opacity)[:, None].cpu().numpy(),
            "scale": 0.5 * var.clamp_min(1e-20).log().float().cpu().numpy(),
            "rot": np.roll(quat, 1, axis=1),  # wxyz
        }

    @staticmethod
    def _write_ply(fields: dict, path: str) -> None:
        """Standard 3DGS PLY (the layout of the original 3DGS ``save_ply``)."""
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{i}" for i in range(3)]
        names += [f"f_rest_{i}" for i in range(fields["f_rest"].shape[1])]
        names += ["opacity", "scale_0", "scale_1", "scale_2"]
        names += [f"rot_{i}" for i in range(4)]
        normals = np.zeros_like(fields["xyz"])
        keys = ["f_dc", "f_rest", "opacity", "scale", "rot"]
        columns = np.concatenate(
            [fields["xyz"], normals] + [fields[k] for k in keys], axis=1
        )
        vertices = np.empty(len(columns), dtype=[(n, "f4") for n in names])
        for i, name in enumerate(names):
            vertices[name] = columns[:, i]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PlyData([PlyElement.describe(vertices, "vertex")]).write(path)

    def forward(
        self,
        reconstruction_path: str,
        images_path: str,
        output_path: str | None = None,
    ) -> str:
        """Predict Gaussians for every registered image and write them as a PLY.

        Args:
            reconstruction_path: Posed COLMAP model (e.g. EPO's
                ``sparse_<model>_epo``); pinhole cameras only.
            images_path: Root folder of the RGB images; the reconstruction's
                image names are resolved relative to it.
            output_path: Output folder. Defaults to
                :func:`splat_output_path` of ``reconstruction_path``.

        Returns:
            Path of the written ``gaussians.ply``.
        """
        recon = pycolmap.Reconstruction(reconstruction_path)
        images, c2w, intrinsics = self._cameras(recon)
        cam = recon.cameras[images[0].camera_id]
        portrait = cam.height > cam.width
        h, w = self.image_shape[::-1] if portrait else self.image_shape
        start = time.time()
        rgb = self._load_images(
            [os.path.join(images_path, im.name) for im in images], h, w
        )
        if portrait:
            rgb, c2w, intrinsics = self._to_landscape(rgb, c2w, intrinsics)
        gaussians, pivot = self._predict(rgb, c2w, intrinsics)
        fields = self._to_world(gaussians, pivot)
        self.last_timings = {"run_resplat": time.time() - start}
        out_dir = output_path or splat_output_path(reconstruction_path)
        path = os.path.join(out_dir, "gaussians.ply")
        self._write_ply(fields, path)
        logger.info(f"{len(fields['xyz']):,} Gaussians from {len(images)} views")
        return path

"""Standalone VGGT-Omega wrapper for EPO.

Drop-in replacement for ``VGGTWrapper`` backed by the pristine
facebookresearch/vggt-omega submodule. Subclasses ``VGGTWrapper`` and
overrides only what differs:

- the model class (``VGGTOmega`` instead of ``VGGT``);
- the native square resolution (512 instead of 518);
- the inference call (single ``model(images)`` dict, 9D pose encoding
  decoded via ``encoding_to_camera`` instead of
  ``pose_encoding_to_extri_intri``).

Everything else — image loading, BA / non-BA reconstruction, rescaling,
``ff_data`` assembly, ``forward``'s signature and its
``(ff_data, reconstruction, depths)`` return — is inherited unchanged, so
the two wrappers can be swapped 1:1:

    from wrapper.vggt_omega_wrapper import VGGTOmegaWrapper

    model = VGGTOmegaWrapper("path/to/vggt_omega_1b_512.pt")
    ff_data, reconstruction, depths = model.forward(images_path, output_path)

The checkpoint (``vggt_omega_1b_512.pt``) is gated on Hugging Face
(https://huggingface.co/facebook/VGGT-Omega), so pass a local path.
"""

import os
import sys

# Make the pristine vggt-omega submodule importable as ``vggt_omega`` and
# this folder importable for ``vggt_wrapper``, regardless of caller CWD.
# (``vggt_wrapper`` itself adds third_party/vggt for the shared helpers.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "third_party", "vggt-omega")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_wrapper import VGGTWrapper  # noqa: E402


class VGGTOmegaWrapper(VGGTWrapper):
    """VGGTWrapper variant running VGGT-Omega for 3D reconstruction."""

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
            oom_safe: Free the model after track prediction (BA path) to save VRAM.
        """
        super().__init__(model_path, cuda_id=cuda_id, seed=seed, oom_safe=oom_safe)
        # VGGT-Omega-1B-512 native resolution (patch size 16), vs VGGT's 518.
        self.vggt_fixed_resolution = 512

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

    def _run_vggt(self, images: torch.Tensor):
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

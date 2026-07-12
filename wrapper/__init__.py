"""Shared registry for the wrapper/ 3D foundation model wrappers.

Selecting a model (``vggt``, ``vggt_omega``, ...) resolves to a (module,
class, default weights path or URL — downloaded/cached via torch.hub when a
URL) triple; the module is imported lazily via ``load_wrapper_class`` so
only the selected backend's dependencies are needed. Every wrapper must
expose the ``VGGTWrapper`` interface: ``ctor(model_path, cuda_id=...,
oom_safe=...)`` and ``forward(images_path, output_path, ...)``.
"""

import importlib

WRAPPERS = {
    "vggt": (
        "vggt_wrapper",
        "VGGTWrapper",
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    ),
    "vggt_omega": (
        "vggt_omega_wrapper",
        "VGGTOmegaWrapper",
        # Gated checkpoint: request access on HF or pass a local --model_path.
        "https://huggingface.co/facebook/VGGT-Omega/resolve/main/vggt_omega_1b_512.pt",
    ),
    "dvlt": (
        "dvlt_wrapper",
        "DVLTWrapper",
        # HF Hub repo id; resolved via huggingface_hub inside the wrapper.
        "nvidia/dvlt",
    ),
    "da3": (
        "da3_wrapper",
        "DA3Wrapper",
        # HF repo id resolved via DepthAnything3.from_pretrained (a local
        # directory with model.safetensors + config.json also works).
        "depth-anything/DA3-LARGE",
    ),
    "mapanything": (
        "mapanything_wrapper",
        "MapAnythingWrapper",
        # HF repo id resolved via MapAnything.from_pretrained.
        "facebook/map-anything",
    ),
    "pi3x": (
        "pi3x_wrapper",
        "Pi3XWrapper",
        # HF repo id resolved via Pi3X.from_pretrained.
        "yyfz233/Pi3X",
    ),
}


def load_wrapper_class(name: str):
    """Import and return the wrapper class registered under ``name``."""
    module_name, class_name, _ = WRAPPERS[name]
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, class_name)

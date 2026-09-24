"""Per-axis mapping of input intrinsics onto EPO's resized image (CPU only).

``process_camera`` maps a COLMAP camera from its own frame to the image file
and from the file to the resized image axis by axis: the resize rounds the
short side down, and a 3DFM may export its network frame (VGGT-Omega:
512 x 336 for a 6214 x 4138 file, an anisotropic resize). The export in
``build_reconstruction`` inverts the mapping into the file frame.
"""

import os
import sys

import pycolmap
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.load import process_camera  # noqa: E402
from modules.camera import CameraModule  # noqa: E402


def _camera(width, height, params, model="PINHOLE"):
    return pycolmap.Camera(model=model, width=width, height=height, params=params)


def _export(params, aspect, sx, sy):
    """``build_reconstruction``'s inverse mapping, into the file frame."""
    f, cx, cy = params.tolist()
    return f / sx, f * aspect / sy, (cx + 0.5) / sx, (cy + 0.5) / sy


def test_file_resolution_camera_short_axis():
    """A camera at the file's size: the long axis is unchanged, the short one
    follows the rounded resize (1440 * 518 / 1920 = 388.5 -> 388)."""
    fx, fy, cx, cy = 1466.6, 1470.9, 960.0, 720.0
    _, p, aspect = process_camera(
        _camera(1920, 1440, [fx, fy, cx, cy]),
        image_wh=(1920, 1440),
        resized_wh=(518, 388),
    )
    sx, sy = 518 / 1920, 388 / 1440
    f = (fx + fy) / 2
    assert p.tolist() == pytest.approx([f * sx, cx * sx - 0.5, cy * sy - 0.5])
    assert aspect == pytest.approx(sy / sx)
    # The short axis differs from the old isotropic 518 / 1920 by 0.13 %.
    assert abs(aspect - 1) > 1e-3
    f_x, f_y, cx_o, cy_o = _export(p, aspect, sx, sy)
    assert (f_x, f_y, cx_o, cy_o) == pytest.approx((f, f, cx, cy))


def test_network_frame_camera_is_mapped_per_axis():
    """A 512 x 336 network-frame camera of a 6214 x 4138 file."""
    fx, fy, cx, cy = 287.7, 293.0, 256.0, 168.0
    W, H, w, h = 6214, 4138, 518, 344
    _, p, aspect = process_camera(
        _camera(512, 336, [fx, fy, cx, cy]),
        image_wh=(W, H),
        resized_wh=(w, h),
    )
    ax, ay = W / 512, H / 336
    f_file = (fx * ax + fy * ay) / 2
    assert p.tolist() == pytest.approx(
        [f_file * w / W, cx * w / 512 - 0.5, cy * h / 336 - 0.5]
    )
    # The image centre stays the image centre (the old isotropic mapping put
    # cy at 168 * 518 / 512 - 0.5 = 169.5, 2 px above it).
    assert p[2].item() == pytest.approx(h / 2 - 0.5)
    f_x, f_y, cx_o, cy_o = _export(p, aspect, w / W, h / H)
    assert (f_x, f_y, cx_o, cy_o) == pytest.approx((f_file, f_file, cx * ax, cy * ay))


def test_simple_pinhole_without_sizes_keeps_old_mapping():
    """No sizes given: the isotropic ``images_size / max_dim`` mapping."""
    _, p, aspect = process_camera(
        _camera(1000, 500, [800.0, 500.0, 250.0], model="SIMPLE_PINHOLE"),
        images_size=518,
    )
    s = 518 / 1000
    assert p.tolist() == pytest.approx([800 * s, 500 * s - 0.5, 250 * s - 0.5])
    assert aspect == 1.0


def test_camera_module_applies_aspect_to_fy_only():
    """``K[1, 1] = f * aspect``; without an aspect K stays square-pixel."""
    k = torch.tensor([[300.0, 258.5, 171.5], [200.0, 100.0, 80.0]])
    ids = {"a": 0, "b": 1}
    kw = {"k_models": ["SIMPLE_PINHOLE"] * 2, "k_params": k, "grad": False}
    square = CameraModule(ids, device="cpu", **kw).get_intrinsic_matrix(None)
    assert torch.equal(square[:, 0, 0], square[:, 1, 1])
    aspect = torch.tensor([0.9973, 1.0013])
    K = CameraModule(ids, device="cpu", aspect=aspect, **kw).get_intrinsic_matrix(None)
    assert torch.equal(K[:, 0, 0], square[:, 0, 0])
    assert torch.allclose(K[:, 1, 1], k[:, 0] * aspect)
    assert torch.equal(K[:, :2, 2], square[:, :2, 2])

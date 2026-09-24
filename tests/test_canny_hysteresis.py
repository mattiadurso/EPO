"""``CannyEdgeDetector``'s hysteresis against kornia's own (GPU).

The wrapper runs kornia's Canny without hysteresis and does the weak-edge
promotion itself (``CannyEdgeDetector._hysteresis``); the edge maps must be
identical to ``kornia.filters.canny(..., hysteresis=True)``.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not torch.cuda.is_available():
    pytest.skip("the Canny wrapper runs under CUDA autocast", allow_module_level=True)

import kornia  # noqa: E402

from extractors.canny import CannyEdgeDetector  # noqa: E402


def _images(seed, b=4, h=388, w=518):
    """Smooth random images: long edges, so hysteresis needs many steps."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.rand((b, 3, h // 8, w // 8), device="cuda", generator=g)
    return torch.nn.functional.interpolate(x, (h, w), mode="bicubic").clamp(0, 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("thresholds", [(0.15, 0.25), (0.05, 0.3)])
def test_matches_kornia_hysteresis(seed, thresholds):
    """Same edge map as kornia's hysteresis loop, under the same autocast."""
    low, high = thresholds
    images = _images(seed)
    detector = CannyEdgeDetector(low_threshold=low, high_threshold=high)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, ref = kornia.filters.canny(
            images, low, high, (7, 7), (2.0, 2.0), hysteresis=True
        )
    out = detector(images)
    assert out.dtype == ref.dtype
    assert torch.equal(out, ref)
    assert 0 < out.sum() < out.numel()

"""Canny edge-detector wrapper used as the default detector for EPO."""

import logging

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CannyEdgeDetector(nn.Module):
    """A simple canny edge detector using kornia implementation."""

    def __init__(
        self,
        low_threshold: float = 0.2,
        high_threshold: float = 0.25,
        hysteresis: bool = True,
        kernel_size: int = 7,
        sigma: float = 2.0,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """Args:
            low_threshold (float): Low threshold for hysteresis.
            high_threshold (float): High threshold for hysteresis.
            hysteresis (bool): Whether to use hysteresis.
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian kernel.
            device (str): Device to run the detector on.

        Outputs:
            edges_binary (torch.Tensor): Binary edge map of shape (B, 1, H, W).

        Notes:
        Hysteresis edges refer to the process of using two thresholds—a low and
        a high—to determine which edge pixels are part of a final edge map. Pixels
        above the high threshold are automatically considered strong edges, while pixels
        below the low threshold are discarded. Pixels between the two thresholds are
        included only if they are "connected" (e.g., 8-connected) to a strong edge pixel.
        This technique, famously used in the Canny edge detector, helps to preserve weak
        but connected edge segments while suppressing noise. (Source: Gemini/Google)

        Increase kernel_size and sigma to reduce granularity of edges. Tune according to
        image resolution.
        """
        super().__init__()

        if verbose:
            logger.info(
                f"CannyEdgeDetector initialized with low_threshold={low_threshold}, "
                f"high_threshold={high_threshold}, hysteresis={hysteresis}, "
                f"kernel_size={kernel_size}, sigma={sigma}, device={device}"
            )

        self.device = torch.device(device)

        # Hysteresis runs in `_hysteresis` instead of inside kornia (same output).
        self.hysteresis = hysteresis
        self.canny = kornia.filters.Canny(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            hysteresis=False,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run Canny on a batch of images.

        Args:
            images: ``(B, C, H, W)`` or ``(C, H, W)`` image(s) in ``[0, 1]``.

        Returns:
            ``(B, 1, H, W)`` binary edge map in ``{0, 1}``.
        """
        assert images.dim() in [
            4,
            3,
        ], (
            "Input images should be a batch of images with shape (B, C, H, W) "
            + "or (C, H, W)"
        )
        if images.dim() == 3:
            # kornia's Canny enforces (B, C, H, W)
            images = images.unsqueeze(0)

        images = images.to(self.device)
        images = images if images.is_floating_point() else images.float()
        # No cuDNN autotuning for these filters: they run once per image
        # shape, and benchmarking each new shape cost ~0.3 s of a scene's
        # setup for identical edge maps.
        cudnn = torch.backends.cudnn
        benchmark, cudnn.benchmark = cudnn.benchmark, False
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, edges = self.canny(images)  # 1 strong, 0.5 weak, 0 none
        finally:
            cudnn.benchmark = benchmark
        if self.hysteresis:
            edges = self._hysteresis(edges)
        return edges

    @staticmethod
    def _hysteresis(edges: torch.Tensor, check_every: int = 8) -> torch.Tensor:
        """Keep the weak edges 8-connected to a strong one; kornia's output.

        Kornia promotes the weak pixels next to a strong one, one pixel ring
        per iteration, with an 8-tap conv and a blocking ``.any()`` test per
        iteration, and the longest weak chain takes 40-150 iterations here. A
        3x3 max-pool of the strong mask does the same promotion, and testing
        for the fixed point every ``check_every`` iterations cannot change it
        (iterations past it are no-ops): about 5x faster, bit-identical.

        Args:
            edges: ``(B, 1, H, W)`` map with 1 (strong), 0.5 (weak) and 0.
            check_every: Iterations between two fixed-point tests.

        Returns:
            ``(B, 1, H, W)`` binary edge map in ``edges``' dtype.
        """
        weak = edges == 0.5
        strong = edges == 1
        while True:
            prev = strong
            for _ in range(check_every):
                grown = F.max_pool2d(strong.to(edges.dtype), 3, stride=1, padding=1)
                strong = strong | (weak & (grown > 0))
            if torch.equal(prev, strong):
                return strong.to(edges.dtype)

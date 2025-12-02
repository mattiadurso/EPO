# This script should contain the pose, intrinsics and depth optimizer.
# It should work with PyTorch and be compatible with CUDA if available.
# It should works usign edges and photometric losses.

# I might want to use pypose, kornia and/or pytorch3d for this.
import os
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import pycolmap
import warnings
import random
import torch.nn.functional as F
from tqdm import tqdm

from helpers.load import (
    find_images,
    load_and_preprocess_images,
    load_and_preprocess_depths,
    process_pose,
    process_camera,
    load_reconstruction,
)
from losses.dt_loss import (
    compute_distance_field,
    sample_distance_field,
)

from helpers.reprojection import (
    filter_viewgraph_by_reprojection,
    reproject_2D_2D,
    grid_sample_nan,
)
from helpers.frustum import build_view_graph_from_frustums
from extractors.canny import CannyEdgeDetector
from modules.camera import Camera
from modules.pose import Pose
from modules.scene_model import SceneModel
from modules.depth import DepthMap

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ignore warnings
warnings.filterwarnings(
    "ignore",
    message=".*cudnnException.*CUDNN_STATUS_NOT_SUPPORTED.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*IProgress not found.*",
)


class AdjusterLM(nn.Module):
    """
    AdjusterLM class for optimizing poses, intrinsics, and depths using edge alignment losses.
    """

    def __init__(
        self,
        reconstruction_path,
        images_path,
        depths_path,
        viewgraph_path=None,
        distance_field_path=None,
        single_camera_per_folder=True,
        load_with_pad=True,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={},
        seed=0,
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=True,
        gt_path=None,
    ):
        super().__init__()

        assert detector in ["canny"], f"Detector {detector} not supported."

        self.device = device
        self.images_size = 518  # square size to which images are resized
        self.load_with_pad = load_with_pad
        self.images_path = images_path
        self.depths_path = depths_path
        self.reconstruction_path = reconstruction_path
        self.single_camera_per_folder = single_camera_per_folder
        self.viewgraph_path = viewgraph_path
        self.gt_path = gt_path
        self.grad_q = grad_q
        self.grad_t = grad_t
        self.grad_k = grad_k
        self.grad_z = grad_z
        self.seed = seed
        self.convergence = False
        self.auc_list = []

        # Set max workers for parallel processing
        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers

        # Initialize edge extractor
        if detector == "canny":
            self.edge_extractor = CannyEdgeDetector(
                low_threshold=detector_params.get("low_threshold", 0.15),
                high_threshold=detector_params.get("high_threshold", 0.25),
                hysteresis=detector_params.get("hysteresis", True),
                kernel_size=detector_params.get("kernel_size", 5),
                sigma=detector_params.get("sigma", 3.0),
                device=device,
            )

        # Timings for profiling
        self.timings = {}
        time_start = time.time()

        # ===================================================
        # Load reconstruction, images, depths, intrinsics, and poses
        # ===================================================
        self.recon = pycolmap.Reconstruction(self.reconstruction_path)

        # Load images
        s_time = time.time()
        self.image_path_list = find_images(self.images_path)
        self._load_and_preprocess_images()
        self.timings["load_images"] = time.time() - s_time

        # Load poses and intrinsics
        s_time = time.time()
        self._read_cameras_from_reconstruction()
        self.timings["load_poses_and_intrinsics"] = time.time() - s_time

        # Extract edges
        s_time = time.time()
        self._extract_edges()
        self.timings["extract_edges"] = time.time() - s_time

        # Load and preprocess depth maps
        s_time = time.time()
        self._load_and_preprocess_depths()
        self.timings["load_depth_maps"] = time.time() - s_time

        # Compute distance fields
        s_time = time.time()
        if distance_field_path is not None:
            distance_fields = torch.load(distance_field_path)
            for image_name in self.images.keys():
                self.images[image_name]["dt_field"] = distance_fields[image_name]
        self._compute_distance_fields()
        self.timings["compute_distance_fields"] = time.time() - s_time

        # Compute or load viewgraph
        s_time = time.time()
        if self.viewgraph_path is not None:
            self._load_viewgraph()
        else:
            self._compute_viewgraph()
        self.viewgraph.sort(key=lambda x: (x[0], x[1]))
        self.timings["compute_viewgraph"] = time.time() - s_time

        # Create scene model
        self.model = SceneModel(
            self.images, self.intrinsics, self.viewgraph, device=self.device
        )

        # ===================================================
        # Finalize initialization
        # ===================================================
        self.timings["total_loading"] = time.time() - time_start
        self.timings["total_optimization"] = 0
        self.loss_list = []

        # Set random seed for reproducibility
        self.fix_seed()

        # Free unused memory
        gc.collect()
        torch.cuda.empty_cache()

    def fix_seed(self):
        """Fix random seed for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def print_summary(self, w=None):
        # Column widths
        key_width = 30
        val_width = 10
        perc_width = 8
        avg_width = 12

        # Total line width
        if w is None:
            w = key_width + val_width + perc_width + avg_width + 6

        print("\n" + "=" * w)
        print(f"{'Summary':^{w}}")
        print("-" * w)

        # Compute total
        self.timings["total"] = self.timings.get("total_loading", 0) + self.timings.get(
            "total_optimization", 0
        )

        # Header row
        print(
            f"{'Stage':<{key_width}}"
            f"{'Time (s)':>{val_width}}"
            f"{'%':>{perc_width+2}}"
            f"{'Per Iter':>{avg_width}}"
        )
        print("-" * w)

        num_iters = len(getattr(self, "loss_list", []))
        per_iter_keys = {
            "prepare_batched_inputs",
            "batched_reprojection_and_loss",
            "gradient_computation",
            "parameter_update",
            "batched_loss_computation",
            "batched_reprojection",
        }

        for key, value in self.timings.items():
            if (key in ["total", "total_loading"]) or (
                key == "total_optimization" and value == 0
            ):
                continue

            if key in per_iter_keys and num_iters > 0:
                # Show only per-iteration column
                value_avg = value / num_iters
                row_str = (
                    f"{key:<{key_width}}"
                    f"{'':>{val_width}}"  # Time blank
                    f"{'':>{perc_width+3}}"  # % blank
                    f"{value_avg:>{avg_width}.4f}"
                )
            else:
                # Show total time and percentage
                perc = (
                    (value / self.timings["total"]) * 100
                    if self.timings["total"] > 0
                    else 0
                )
                row_str = (
                    f"{key:<{key_width}}"
                    f"{value:>{val_width}.2f}"
                    f"{perc:>{perc_width}.1f}%"
                    f"{'':>{avg_width}}"
                )

            print(row_str)

        print("-" * w)
        print(
            f"{'Total':<{key_width}}"
            f"{self.timings['total']:>{val_width}.2f}"
            f"{'':>{perc_width + avg_width + 3}}"
        )

        # Loss summary
        if len(self.loss_list) > 0:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss

            print("-" * w)
            print(f"{'Initial loss:':<{key_width}}{initial_loss:>{val_width}.6f}")
            print(f"{'Final loss:':<{key_width}}{final_loss:>{val_width}.6f}")
            print(f"{'Loss reduction:':<{key_width}}{delta:>{val_width}.6f}")
            steps = len(self.loss_list)
            conv = " (converged)" if getattr(self, "convergence", False) else ""
            print(f"{f'Total steps{conv}:':<{key_width}}{steps:>{val_width}d}")

        print("=" * w)

    def _load_and_preprocess_images(self):
        """Load and preprocess images."""
        self.images = load_and_preprocess_images(
            self.image_path_list,
            self.images_path,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            device=self.device,
        )

    def read_cameras_from_reconstruction(
        self,
        reconstruction,
        images,
        single_camera_per_folder=False,
        load_with_pad=False,
    ):

        intrinsics = {}

        # Read cameras intrinsics
        if single_camera_per_folder:
            # Reading cameras from images (to handle multiple images with same camera)
            for image in reconstruction.images.values():
                _, model, new_params = process_camera(
                    image.camera, load_with_pad, images_size=self.images_size
                )
                cam_id = image.name.split("/")[
                    0
                ]  # assuming image names are like "cam_id/image_name"

                # I want to stack params of same cam_id to then averaged them
                if cam_id not in intrinsics:
                    intrinsics[cam_id] = {
                        "cam_id": cam_id,
                        "model": model,
                        "parameters": [new_params],  # Changed: store as list initially
                    }
                else:
                    # Append new params to the list
                    intrinsics[cam_id]["parameters"].append(new_params)

            # Average params for each cam_id
            for cam_id in intrinsics.keys():
                params = intrinsics[cam_id]["parameters"]
                if len(params) == 1:
                    # only one image with this cam_id
                    intrinsics[cam_id]["parameters"] = params[0]
                else:
                    # multiple images with this cam_id - stack and average
                    intrinsics[cam_id]["parameters"] = torch.stack(params, dim=0).mean(
                        dim=0
                    )
        else:  # one camera per image
            # Reading cameras from images
            for cam in reconstruction.cameras.values():
                cam_id, model, new_params = self.process_camera(cam)
                intrinsics[cam.camera_id] = {
                    "cam_id": cam_id,
                    "model": model,
                    "parameters": new_params.to(self.device),
                }
        # Sort dict by keys
        intrinsics = dict(sorted(intrinsics.items()))

        # Convert to Camera objects
        for cam_id in intrinsics.keys():
            intrinsics[cam_id] = Camera(
                **intrinsics[cam_id], grad=self.grad_k, device=self.device
            )

        # Read poses from images
        for image in reconstruction.images.values():
            R, t, cam_id = process_pose(image)
            pose = Pose(R=R, t=t, grad_q=self.grad_q, device=self.device)

            if single_camera_per_folder:
                cam_id = image.name.split("/")[0]
            else:
                cam_id = image.camera_id

            images[image.name].update({"P": pose, "cam_id": cam_id})

        return images, intrinsics

    def _read_cameras_from_reconstruction(self):
        """Read cameras and poses from the reconstruction."""
        self.images, self.intrinsics = self.read_cameras_from_reconstruction(
            self.recon,
            self.images,
            single_camera_per_folder=self.single_camera_per_folder,
            load_with_pad=self.load_with_pad,
        )

    def _load_and_preprocess_depths(self):
        """Load and preprocess depth maps."""
        self.images = load_and_preprocess_depths(
            self.depths_path,
            self.images,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            device=self.device,
        )

        # Check all depths have the same size
        depth_shapes = set()
        for image_name in self.images.keys():
            depth_shapes.add(self.images[image_name]["depth"].shape[-2:])
        if len(depth_shapes) > 1:
            # Pad bottom-right to make them equal
            max_h = max([shape[0] for shape in depth_shapes])
            max_w = max([shape[1] for shape in depth_shapes])
            for image_name in self.images.keys():
                depth = self.images[image_name]["depth"]
                h, w = depth.shape[-2:]
                pad_h = max_h - h
                pad_w = max_w - w
                depth = F.pad(
                    depth,
                    (0, pad_w, 0, pad_h),
                    mode="constant",
                    value=float("nan"),
                )
                self.images[image_name]["depth"] = depth

        # Add sampled depth at `edges_padded` locations
        for image_name in self.images.keys():
            edges_padded = self.images[image_name]["edges_padded"]  # (N, 2)
            depth = self.images[image_name]["depth"]  # (H, W)
            sampled_depth, _ = grid_sample_nan(edges_padded[None], depth[None])
            self.images[image_name]["sampled_depth"] = DepthMap(
                height=self.images[image_name]["hw"][0],
                width=self.images[image_name]["hw"][1],
                depth=sampled_depth.squeeze(),
                grad=self.grad_z,
            )

    def _extract_edges(self):
        """Extract edges from images."""
        for image_name in tqdm(self.images.keys(), desc="Extracting edges"):
            img_tensor = self.images[image_name]["image"].unsqueeze(0).to(self.device)
            edges_map = self.edge_extractor(img_tensor)
            edges = edges_map.squeeze().nonzero().flip(dims=(1, 0)).float()  # (N, 2)
            self.images[image_name].update(
                {"edges_map": edges_map.squeeze(), "edges": edges}
            )

        # Pad edges to have the same number per image
        self._pad_edges()

    def _pad_edges(self):
        """Pad all edges to have same number (max_edges) of edges per image."""
        max_edges = max(
            [self.images[img]["edges"].shape[0] for img in self.images.keys()]
        )

        for image_name in self.images.keys():
            edges = self.images[image_name]["edges"]
            n_edges = edges.shape[0]
            if n_edges < max_edges:
                pad_size = max_edges - n_edges
                pad = torch.zeros((pad_size, 2), device=edges.device)
                edges = torch.cat([edges, pad], dim=0)

                pad_mask = torch.zeros(
                    (max_edges,), device=edges.device, dtype=torch.float32
                )
                pad_mask[:n_edges] = 1.0
            else:
                # n_edges == max_edges: all edges are valid
                pad_mask = torch.ones(
                    (max_edges,), device=edges.device, dtype=torch.float32
                )

            self.images[image_name].update(
                {"edges_padded": edges, "pad_mask": pad_mask}
            )

        self.max_edges = max_edges
        print(f"Max edges per image: {self.max_edges:,}")

    @torch.no_grad()
    def _compute_distance_fields(self):
        """Compute distance fields for all images."""
        for image_name in tqdm(self.images.keys(), desc="Computing distance fields"):
            edges_map = self.images[image_name]["edges_map"]
            dt_field = compute_distance_field(edges_map, device=self.device)
            self.images[image_name].update({"dt_field": dt_field})

    @torch.no_grad()
    def _compute_viewgraph(self):
        """Compute the viewgraph from frustums."""
        viewgraph = build_view_graph_from_frustums(
            self.recon,
            z_near_default=0.1,
            z_far_default=5.0,
            max_view_angle_deg=30.0,
            distance_factor=2,
            verbose=False,
        )
        viewgraph = filter_viewgraph_by_reprojection(
            viewgraph,
            self.images,
            self.intrinsics,
            th=0.025,
            sampling_factor=10,
            reprojection_error=3.0,
        )
        self.viewgraph = viewgraph

    def _load_viewgraph(self):
        """Load the viewgraph from a file."""
        self.viewgraph = []
        with open(self.viewgraph_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                i, j = line.strip().split()
                self.viewgraph.append((i, j))

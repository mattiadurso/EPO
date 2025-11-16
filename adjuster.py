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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Ignore the cuDNN warning
warnings.filterwarnings(
    "ignore",
    message=".*cudnnException.*CUDNN_STATUS_NOT_SUPPORTED.*",
)

# Ignore the tqdm / IProgress warning
warnings.filterwarnings(
    "ignore",
    message=".*IProgress not found.*",
)

import glob
from PIL import Image
from tqdm import tqdm
from helpers.load import (
    find_images,
    load_and_preprocess_images,
    load_and_preprocess_depths,
)
from losses.loss import (
    compute_distance_field,
    sample_distance_field,
)
from helpers.reprojection import (
    filter_viewgraph_by_reprojection,
    reproject_2D_2D,
    grid_sample_nan,
)
from helpers.reprojection_compiled import (
    unproject_2D_to_world,
    project_world_to_2D,
)
from helpers.frustum import build_view_graph_from_frustums
from extractors.canny import CannyEdgeDetector
from modules.camera import Camera
from modules.pose import Pose
from modules.depth import DepthMap


import sys

sys.path.append("/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_3D")
# from utils_benchmark_pose import eval_colmap_model
from benchmark_pose import eval_colmap_model


class Adjuster(nn.Module):
    """
    Module to adjust poses and intrinsics of a given reconstruction using edge alignment losses.
    Args:
        reconstruction_path (str): Path to the COLMAP reconstruction folder.
        images_path (str): Path to the folder containing input images.
        depths_path (str): Path to the folder containing depth maps.
        viewgraph_path (str, optional): Path to a precomputed viewgraph file. If None, it will be computed from frustums.
        sky_mask_path (str, optional): Path to the folder containing sky masks. Default is None.
            This excludes edges from sky regions. Might lead to less constraints in optimization and slightly worse results.
            Use when sky changes a lot between images. PRovide massk as png images with 1 for sky and 0 for non-sky.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
        single_camera_per_folder (bool, optional): Whether to assume a single camera per folder. Default is True.
        load_with_pad (bool, optional): Whether to load images with padding to make them square. Default is True.
        detector (str, optional): Edge detector to use. Default is "canny".
        device (str, optional): Device to use for computation. Default is "cuda".
        max_workers (int, optional): Maximum number of workers for parallel loading. Default is -1 (all available).
        detector_params (dict, optional): Parameters for the edge detector.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        optim (str, optional): Optimizer to use ("adamw" or "lm"). Default is "adamw".
        scheduler_name (str, optional): Learning rate scheduler name. Default is None.
        scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        grad_q (bool, optional): Whether to optimize rotation quaternions. Default is True.
        grad_t (bool, optional): Whether to optimize translation vectors. Default is True.
        grad_k (bool, optional): Whether to optimize camera intrinsics. Default is True.
        grad_z (bool, optional): Whether to optimize depth scale. Default is False.
        viz (bool, optional): Whether to visualize intermediate results. Default is False.
        gt_path (str, optional): Path to ground truth data for evaluation. Default is None.
    """

    def __init__(
        self,
        reconstruction_path,
        images_path,
        depths_path,
        viewgraph_path=None,  # for testing with GT viewgraph
        sky_mask_path=None,
        lr=1e-3,
        single_camera_per_folder=True,
        load_with_pad=True,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={},
        seed=0,
        scheduler_name=None,
        scheduler_params={"factor": 0.75, "patience": 2, "min_lr": 1e-6},
        matcher_type="exhaustive",  # or "sequential"
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=False,
        q_lr_scale=0.1,
        t_lr_scale=1.0,
        k_lr_scale=0.5,
        z_lr_scale=0.1,
        viz=False,  # it true del non used stuff during computation
        gt_path=None,
    ):
        super().__init__()

        assert detector in ["canny"], f"Detector {detector} not supported."

        self.max_workers = os.cpu_count() if max_workers < 0 else max_workers
        self.images_size = 518  # square size to which images are resized
        self.device = device
        self.lr = lr
        self.load_with_pad = load_with_pad
        self.images_path = images_path
        self.depths_path = depths_path
        self.reconstruction_path = reconstruction_path
        self.single_camera_per_folder = single_camera_per_folder
        self.convergence = False
        self.gt_path = gt_path
        self.viewgraph_path = viewgraph_path
        self.matcher_type = matcher_type
        self.scheduler_params = scheduler_params
        self.auc_list = []

        # Edge extractor
        if detector == "canny":
            self.edge_extractor = CannyEdgeDetector(
                low_threshold=detector_params.get("low_threshold", 0.15),
                high_threshold=detector_params.get("high_threshold", 0.25),
                hysteresis=detector_params.get("hysteresis", True),
                kernel_size=detector_params.get("kernel_size", 5),
                sigma=detector_params.get("sigma", 3.0),
                device=device,
            )

        # what to train
        self.grad_q = grad_q
        self.grad_t = grad_t
        self.grad_k = grad_k
        self.grad_z = grad_z
        self.q_lr_scale = q_lr_scale
        self.t_lr_scale = t_lr_scale
        self.k_lr_scale = k_lr_scale
        self.z_lr_scale = z_lr_scale

        # Loading
        self.timings = {}
        time_start = time.time()

        ## Load Reconstruction
        self.recon = pycolmap.Reconstruction(self.reconstruction_path)

        ## Load Images as dict {image_name: image_tensor}
        s_time = time.time()
        self.image_path_list = find_images(self.images_path)
        self._load_and_preprocess_images()  # into self.images
        self.timings["load_images"] = time.time() - s_time

        ## Load poses and intrinsics
        s_time = time.time()
        self._read_cameras_from_reconstruction()  # into self.images and self.intrinsics
        self.timings["load_poses_and_intrinsics"] = time.time() - s_time

        ## Extract edges
        s_time = time.time()
        self._extract_edges()  # into self.images
        self.timings["extract_edges"] = time.time() - s_time

        ## Load depth maps
        s_time = time.time()
        self._load_and_preprocess_depths()
        self.timings["load_depth_maps"] = time.time() - s_time

        ## Viewgraph from frustums
        s_time = time.time()
        # compute and cache P and K matrices once
        self.P_cache, self.K_cache = None, None
        self._cache_projection_matrices()

        self._compute_viewgraph(type=self.matcher_type)
        # vg is a lsit of names of image pairs, sort by the first name
        self.viewgraph.sort(key=lambda x: (x[0], x[1]))
        self.timings["compute_viewgraph"] = time.time() - s_time

        ## Load sky mask if any
        s_time = time.time()
        self._load_and_preprocess_sky_masks(sky_mask_path)
        self.timings["load_sky_masks"] = time.time() - s_time

        ## Compute Distance Fields
        s_time = time.time()
        self._compute_distance_fields()  # into self.images
        self._cache_pad_and_dt_fields()
        self.timings["compute_distance_fields"] = time.time() - s_time

        # Create optimizer
        params_to_optimize = self._collect_parameters_to_optimize()
        self._print_params_summary(params_to_optimize)

        # Create optimizer with collected parameters
        self._load_optimizer(params_to_optimize)
        self._load_scheduler(scheduler_name, self.optimizer, scheduler_params)
        self.scaler = torch.cuda.amp.GradScaler()

        self.timings["total_loading"] = time.time() - time_start
        self.timings["total_optimization"] = 0
        self.timings["prepare_batched_inputs"] = 0
        self.timings["batched_reprojection"] = 0
        self.timings["batched_loss_computation"] = 0
        self.timings["gradient_computation"] = 0
        self.timings["parameter_update"] = 0

        # # At this point I might get rid of rgb images to save memory as not needed for edge loss
        if not viz:
            for image_name in self.images.keys():
                self.images[image_name].pop("image")
                self.images[image_name].pop("depth")
                self.images[image_name].pop("edges_map")
                if "sky_mask" in self.images[image_name]:
                    self.images[image_name].pop("sky_mask")

        self.loss_list = []
        self.lr_list = []
        self.seed = seed
        self.fix_seed()

        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        max_steps=1_000,
        type="batched",
        gradient_tolerance=1e-5,
        quick_mode=False,  # "quick" (sample & backprop) or "full" (accumulate over all)
        batch_size=128,  # faster than 64 and 256
        loss_robustifier=True,
    ):
        """
        Main optimization loop.
        Args:
            max_steps (int): Maximum number of optimization steps.
            batch_size (int): Maximum number of image pairs per batch.
                            In "quick" mode: samples this many pairs and backprops immediately.
                            In "full" mode: batch size for accumulation over full viewgraph.
            type (str): Type of optimization to perform. Options are "sequential" or "batched".
            gradient_tolerance (float): Tolerance for gradient-based convergence.
            quick_mode (bool): Whether to use quick mode (sample & backprop) or full accumulation.
        """
        assert type in ["sequential", "batched"], f"Type {type} not supported."
        time_start = time.time()

        if quick_mode:
            print(
                f"Quick mode ON. Randomly sampling viewgraph from {len(self.viewgraph):,} to {batch_size:,} pairs before every iteration"
            )
        else:
            num_batches = math.ceil(len(self.viewgraph) / batch_size)
            print(
                f"Full accumulation mode ON. Processing {len(self.viewgraph):,}",
                (
                    f"pairs with batch size {batch_size:,} ({num_batches} batches per iteration)"
                    if type == "batched"
                    else ""
                ),
                # edges per image
                f"Using {self.images[list(self.images.keys())[0]]['edges_padded'].numel()//2:,} edges per image",  # // due to x and y
            )

        max_steps = max_steps if max_steps > 0 else 1_000
        total_points = self.max_edges * (
            batch_size if quick_mode else len(self.viewgraph)
        )
        print(f"Total points to process per iteration: {total_points:,}")
        bar = tqdm(range(max_steps), desc="Adjusting poses and intrinsics")

        for step in bar:
            # Initialize optimizer gradients
            self.optimizer.zero_grad()

            # Update cache at START of each step (after parameters changed)
            self._cache_projection_matrices()

            # Unproject point to world coordinates
            self._unproject_edges_to_3D()

            if quick_mode:  # reduce pairs randomly each step
                if len(self.viewgraph) > batch_size:
                    sampled_indices = torch.randperm(len(self.viewgraph))[:batch_size]
                    sampled_viewgraph = [self.viewgraph[i] for i in sampled_indices]
            else:
                sampled_viewgraph = self.viewgraph

            if type == "sequential":
                residuals = self.compute_sequential_step(sampled_viewgraph)

            elif type == "batched":
                residuals = self.compute_batched_step(
                    sampled_viewgraph, batch_size=batch_size
                )  # (num_pairs,)

            loss = self._compute_batched_loss(residuals, robustifier=loss_robustifier)

            # Backpropagate and update using GradScaler if available
            self._scaler_and_scheduler_steps(loss)

            # Logging
            self.loss_list.append(loss.detach().item())
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.lr
            )
            self.lr_list.append(current_lr)

            # DEBUG: Evaluate AUC if GT available
            if self.gt_path is not None and step % 5 == 0:
                opt = "/home/mattia/Desktop/Repos/batchsfm/optimized_reconstruction"
                self.build_reconstruction(opt, save_points=False, verbose=False)
                AUC_score_max, num_images, df_optim = eval_colmap_model(
                    opt, self.gt_path, return_df=False, thrs=[1, 3, 5]
                )
                self.auc_list.append(AUC_score_max)

            # stopping criteria: stop if relative change is below tolerance
            if step > 0:
                rel_change = abs(self.loss_list[-2] - self.loss_list[-1]) / abs(
                    self.loss_list[-2]
                )
                if rel_change < gradient_tolerance:
                    print(
                        f"Converged at step {step} with relative loss ",
                        f"change {rel_change:.3e} < {gradient_tolerance}",
                    )
                    if step > self.scheduler_params.get("patience", 2):
                        self.convergence = True
                        break

            bar.set_postfix(
                loss=f"{self.loss_list[-1]:.4f}",
                auc5=(
                    f"{self.auc_list[-1][-1]:.4f}" if self.gt_path is not None else -1
                ),
                rel_change=f"{rel_change:.3e}" if step > 0 else -1,
            )

        self.timings["total_optimization"] += time.time() - time_start
        self.print_summary()

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

    def fix_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    def process_camera(self, camera, load_with_pad=False):
        # Convert a single pycolmap.Camera to torch tensor
        cam_id = camera.camera_id
        model = camera.model.name
        params = camera.params
        width = camera.width
        height = camera.height

        if model == "SIMPLE_PINHOLE":  # or model == "SIMPLE_RADIAL":
            f = params[0]
            cx, cy = params[1], params[2]

        elif model == "PINHOLE":  # or model == "RADIAL":
            f = torch.tensor([params[0], params[1]], dtype=torch.float32)
            cx, cy = params[2], params[3]

        else:
            raise NotImplementedError(f"Camera model {model} not supported.")

        # Account for padding when making square
        max_dim = max(width, height)
        pad_x = (max_dim - width) // 2 if load_with_pad else 0
        pad_y = (max_dim - height) // 2 if load_with_pad else 0

        # Scale factor after resize
        scale = self.images_size / max_dim

        # Apply padding shift + scale
        f = f * scale
        cx = (cx + pad_x) * scale
        cy = (cy + pad_y) * scale

        params = torch.cat([f, torch.tensor([cx, cy], dtype=torch.float32)], dim=0)
        return cam_id, model, params

    def process_pose(self, image):
        # Convert a single pycolmap.Image to torch tensor
        # COLMAP's cam_from_world is already R_cw (world-to-camera rotation)
        R = torch.tensor(image.cam_from_world.rotation.matrix(), dtype=torch.float32)
        t = torch.tensor(
            image.cam_from_world.translation, dtype=torch.float32
        ).unsqueeze(1)

        return R, t, image.camera_id

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
                _, model, new_params = self.process_camera(image.camera, load_with_pad)
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
            R, t, cam_id = self.process_pose(image)
            pose = Pose(R=R, t=t, grad_q=self.grad_q, device=self.device)

            if single_camera_per_folder:
                cam_id = image.name.split("/")[0]
            else:
                cam_id = image.camera_id

            images[image.name].update({"P": pose, "cam_id": cam_id})

        return images, intrinsics

    def compute_sequential_step(self, sampled_viewgraph):
        """Compute one optimization step over the sampled viewgraph sequentially and return the loss."""

        residuals_list = []
        # Use autocast (not really helping)
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            for pair in sampled_viewgraph:
                i, j = pair
                image_i = self.images[i]
                image_j = self.images[j]

                # project edges 1->2
                edges_12 = reproject_2D_2D(
                    xy0=image_i["edges"][None],
                    depthmap0=image_i["depth"][None],
                    P0=self.P_cache[i][None],
                    P1=self.P_cache[j][None],
                    K0=self.K_cache[image_i["cam_id"]][None],
                    K1=self.K_cache[image_j["cam_id"]][None],
                    img1_shape=image_j["hw"],
                )

                # project edges 2->1
                edges_21 = reproject_2D_2D(
                    xy0=image_j["edges"][None],
                    depthmap0=image_j["depth"][None],
                    P0=self.P_cache[j][None],
                    P1=self.P_cache[i][None],
                    K0=self.K_cache[image_j["cam_id"]][None],
                    K1=self.K_cache[image_i["cam_id"]][None],
                    img1_shape=image_i["hw"],
                )

                # Remove batch dimension before sampling
                edges_12 = edges_12.squeeze(0)  # (N, 2)
                edges_21 = edges_21.squeeze(0)  # (N, 2)

                # Filter out NaN values
                if edges_12.numel() > 0:
                    valid_12 = ~torch.isnan(edges_12).any(dim=1)
                    edges_12 = edges_12[valid_12]

                if edges_21.numel() > 0:
                    valid_21 = ~torch.isnan(edges_21).any(dim=1)
                    edges_21 = edges_21[valid_21]

                # compute loss, clone and detach since they are created without grads
                dt_field_1 = image_i["dt_field"]
                dt_field_2 = image_j["dt_field"]
                edge_loss_12 = sample_distance_field(
                    dt_field_2, edges_12, device=self.device
                )

                edge_loss_21 = sample_distance_field(
                    dt_field_1, edges_21, device=self.device
                )

                # average over points per image
                residuals_list.append(edge_loss_12.mean())
                residuals_list.append(edge_loss_21.mean())

        residuals = torch.stack(residuals_list)  # (num_pairs,)
        return residuals

    @torch.no_grad()
    def _dbscan_filter(self, reconstruction, eps=0.5, min_samples=20):
        """
        Filter 3D points in reconstruction using DBSCAN clustering.
        Keeps only the largest cluster to remove outliers.

        Args:
            reconstruction: pycolmap.Reconstruction object
            eps: Maximum distance between two samples for DBSCAN
            min_samples: Minimum number of samples in a neighborhood for DBSCAN

        Returns:
            pycolmap.Reconstruction: Filtered reconstruction
        """
        import numpy as np
        from sklearn.cluster import DBSCAN

        if len(reconstruction.points3D) == 0:
            return reconstruction

        # Extract 3D point coordinates
        point_ids = list(reconstruction.point3D_ids())
        xyz = np.array([reconstruction.point3D(p_id).xyz for p_id in point_ids])

        # Run DBSCAN with fallback for memory errors
        labels = None
        n_jobs = 4
        while n_jobs >= 1 and labels is None:
            try:
                clustering = DBSCAN(
                    eps=eps, min_samples=min_samples, n_jobs=n_jobs
                ).fit(xyz)
                labels = clustering.labels_
            except MemoryError:
                n_jobs //= 2
            except Exception as e:
                print(f"DBSCAN failed: {e}")
                return reconstruction

        # Find largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        non_noise_indices = np.where(unique_labels != -1)

        if len(counts[non_noise_indices]) == 0:
            print("Warning: DBSCAN found no clusters. Skipping filtering.")
            return reconstruction

        main_cluster_label = unique_labels[non_noise_indices][
            np.argmax(counts[non_noise_indices])
        ]

        # Get indices of points in main cluster
        cluster_indices = np.where(labels == main_cluster_label)[0]
        ids_to_keep = set([point_ids[i] for i in cluster_indices])

        print(
            f"DBSCAN: Keeping largest cluster with {len(ids_to_keep):,} / {len(point_ids):,} points"
        )

        # Create new reconstruction with only kept points
        filtered_reconstruction = pycolmap.Reconstruction()

        # Copy cameras
        for camera_id in reconstruction.cameras.keys():
            filtered_reconstruction.add_camera(reconstruction.cameras[camera_id])

        # Recreate images with new camera references
        for image_id in reconstruction.images.keys():
            old_image = reconstruction.images[image_id]
            new_image = pycolmap.Image(
                id=old_image.image_id,
                name=old_image.name,
                camera_id=old_image.camera_id,
                cam_from_world=old_image.cam_from_world,
            )
            filtered_reconstruction.add_image(new_image)

        # Copy only kept points with fresh empty tracks
        for p_id in ids_to_keep:
            point3d = reconstruction.point3D(p_id)
            # Create empty track instead of copying old one
            empty_track = pycolmap.Track()
            filtered_reconstruction.add_point3D(point3d.xyz, empty_track, point3d.color)

        return filtered_reconstruction

    @torch.no_grad()
    def build_reconstruction(
        self,
        output_path="optimized_reconstruction",
        save_points=True,
        verbose=False,
        max_points_per_image=100_000,
        final_dbscan_filtering=False,
        dbscan_eps=0.05,
        dbscan_min_samples=5,
    ):
        """
        Create a pycolmap.Reconstruction from images and intrinsics dictionaries.

        Args:
            output_path: path to save the reconstruction
            save_points: whether to save 3D points from depth unprojection
            max_points_per_image: maximum number of 3D points per image (default: 100_000)
            dbscan_eps: epsilon for DBSCAN clustering
            dbscan_min_samples: min samples for DBSCAN clustering
        """
        # Create empty reconstruction
        reconstruction = pycolmap.Reconstruction()

        # 1. Add cameras - we need to handle different scales per image
        # Group images by camera to find the appropriate scale
        camera_scales = {}
        for image_name, image_data in self.images.items():
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            if cam_id not in camera_scales:
                camera_scales[cam_id] = []
            camera_scales[cam_id].append(scale)

        # Use median scale for each camera
        for cam_id in camera_scales:
            camera_scales[cam_id] = np.median(camera_scales[cam_id])

        for cam_id, camera in self.intrinsics.items():
            # Get camera parameters as numpy array
            params = camera.params.detach().cpu().numpy()

            # Get scale for this camera
            scale = camera_scales.get(cam_id, 1.0)

            # Apply inverse scaling to focal lengths (scale back to original)
            if camera.model == "PINHOLE":
                params = params.copy()
                params[0] /= scale  # fx
                params[1] /= scale  # fy
                params[2] /= scale  # cx
                params[3] /= scale  # cy
                model = pycolmap.CameraModelId.PINHOLE
            elif camera.model == "SIMPLE_PINHOLE":
                params = params.copy()
                params[0] /= scale  # f
                params[1] /= scale  # cx
                params[2] /= scale  # cy
                model = pycolmap.CameraModelId.SIMPLE_PINHOLE
            else:
                raise ValueError(f"Unsupported camera model: {camera.model}")

            # Get image dimensions from first image with this cam_id
            sample_image = next(
                (img for img in self.images.values() if img["cam_id"] == cam_id), None
            )

            if sample_image is None:
                print(f"Warning: No images found for camera {cam_id}, skipping...")
                continue

            height, width = sample_image["hw"]

            # Scale image dimensions back to original
            width = int(width / scale)
            height = int(height / scale)

            # Convert cam_id to int for COLMAP
            cam_id_int = int(cam_id) if isinstance(cam_id, str) else cam_id

            # Create and register camera
            cam = pycolmap.Camera(
                model=model,
                width=width,
                height=height,
                params=params,
                camera_id=cam_id_int,
            )
            reconstruction.add_camera(cam)

        # 2. Add images (poses)
        for image_id, (image_name, image_data) in enumerate(
            self.images.items(), start=1
        ):
            pose = image_data["P"]
            cam_id = image_data["cam_id"]
            scale = image_data.get("scale", 1.0)

            # Convert cam_id to int for COLMAP
            cam_id_int = int(cam_id) if isinstance(cam_id, str) else cam_id

            # Get rotation matrix and translation
            q = pose.q.detach().cpu().numpy()
            t = pose.t.detach().cpu().numpy().squeeze()

            # Apply inverse scaling to translation (scale back to original)
            t = t / scale

            # Create image
            img = pycolmap.Image(
                id=image_id,
                name=image_name,
                camera_id=cam_id_int,
                cam_from_world=pycolmap.Rigid3d(
                    rotation=pycolmap.Rotation3d(q), translation=t
                ),
            )
            reconstruction.add_image(img)

        # 3. Add Points3D from depth unprojection using fresh computation
        if save_points:
            if verbose:
                print("Unprojecting depth maps to 3D points...")

            # Compute fresh 3D world coordinates
            self._unproject_edges_to_3D()

            total_points = 0

            for image_id, (image_name, image_data) in enumerate(
                self.images.items(), start=1
            ):
                cam_id = image_data["cam_id"]
                scale = image_data.get("scale", 1.0)

                # Get unprojected 3D points in world coordinates (at scaled resolution)
                edges_3D = image_data.get("edges_3D", None)  # (N, 3)
                pad_mask = image_data.get("pad_mask", None)  # (N,)

                if edges_3D is None or pad_mask is None:
                    if verbose:
                        print(f"No edges_3D or pad_mask for {image_name}, skipping...")
                    continue

                # Convert to numpy
                edges_3D_np = edges_3D.detach().cpu().numpy()  # (N, 3)
                pad_mask_np = pad_mask.detach().cpu().numpy()  # (N,)

                # Filter by pad mask (only valid edges, ignore padded entries)
                valid_mask = pad_mask_np > 0
                valid_3D = edges_3D_np[valid_mask]  # (M, 3)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_3D) == 0:
                    if verbose:
                        print(f"No valid edges for {image_name}")
                    continue

                # Scale points back to original resolution
                # points_3D is in downsampled world space, multiply by scale to expand to original
                valid_3D = valid_3D / scale

                # Sample uniformly up to max_points_per_image
                num_valid = len(valid_3D)
                if num_valid > max_points_per_image:
                    sample_idx = np.random.choice(
                        num_valid, size=max_points_per_image, replace=False
                    )
                    valid_3D = valid_3D[sample_idx]
                    valid_indices = valid_indices[sample_idx]

                # Get RGB values from original image
                if "image" in image_data:
                    image = image_data["image"].detach().cpu().numpy()  # (3, H, W)
                    edges_padded = (
                        image_data["edges_padded"].detach().cpu().numpy()
                    )  # (N, 2)

                    # Get coordinates of valid edges
                    valid_edges = edges_padded[valid_indices]
                    y_coords_int = valid_edges[:, 1].astype(np.int32)
                    x_coords_int = valid_edges[:, 0].astype(np.int32)

                    # Clamp to valid range (at scaled resolution)
                    y_coords_int = np.clip(y_coords_int, 0, image.shape[1] - 1)
                    x_coords_int = np.clip(x_coords_int, 0, image.shape[2] - 1)

                    rgb = image[:, y_coords_int, x_coords_int]  # (3, M)
                    rgb = (rgb * 255).astype(np.uint8).T  # (M, 3)
                else:
                    # Default to black if no image available
                    rgb = np.full((len(valid_3D), 3), 0, dtype=np.uint8)

                # Add points to reconstruction
                for pt_world, rgb_val in zip(valid_3D, rgb):
                    point3D_id = reconstruction.add_point3D(
                        pt_world, pycolmap.Track(), rgb_val
                    )
                    track = reconstruction.point3D(point3D_id).track
                    track.add_element(image_id, int(0))  # dummy keypoint index

                total_points += len(valid_3D)
                if verbose:
                    print(f"Added {len(valid_3D)} points from {image_name}")

            if verbose:
                print(f"Total points added: {total_points:,}")

        # 4. DBSCAN filtering
        if save_points and final_dbscan_filtering:
            if verbose:
                print("Running DBSCAN filtering...")
            reconstruction = self._dbscan_filter(
                reconstruction, eps=dbscan_eps, min_samples=dbscan_min_samples
            )

        # 5. Save reconstruction
        if verbose:
            print(f"Cameras: {len(reconstruction.cameras)}")
            print(f"Images: {len(reconstruction.images)}")
            print(f"Points3D: {len(reconstruction.points3D):,}")

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            reconstruction.write_text(output_path)
            if verbose:
                print(f"Reconstruction saved to: {output_path}")

        return reconstruction

    def _create_batched_inputs(self, sampled_viewgraph):
        """Prepare batched inputs for the batched optimization step given a list of pairs from the viewgraph."""
        batch = {
            "xyz_world": [],
            "P1": [],
            "K1": [],
            "img1_shape": [],
        }
        pad_masks, dt_fields = [], []

        for i, j in sampled_viewgraph:
            batch["xyz_world"].append(self.images[i]["edges_3D"])
            batch["P1"].append(self.P_cache[j])
            batch["K1"].append(self.K_cache[self.images[j]["cam_id"]])
            batch["img1_shape"].append(self.images[j]["hw"])

            batch["xyz_world"].append(self.images[j]["edges_3D"])
            batch["P1"].append(self.P_cache[i])
            batch["K1"].append(self.K_cache[self.images[i]["cam_id"]])
            batch["img1_shape"].append(self.images[i]["hw"])

            # this can be cached once and indexed
            pad_masks.append(self.pad_masks_cache[i])
            pad_masks.append(self.pad_masks_cache[j])

            dt_fields.append(self.dt_fields_cache[j])
            dt_fields.append(self.dt_fields_cache[i])

        for key in batch:
            if key in ["img1_shape"]:
                continue
            batch[key] = torch.stack(batch[key], dim=0)

        # if img_shape us a list of tuple all equal, use a tuple to run faster
        batch["img1_shape"] = (
            batch["img1_shape"][0]
            if all(
                batch["img1_shape"][k] == batch["img1_shape"][0]
                for k in range(len(batch["img1_shape"]))
            )
            else batch["img1_shape"]
        )

        pad_masks = torch.stack(pad_masks, dim=0)  # (B,)
        dt_fields = torch.stack(dt_fields, dim=0)

        return batch, pad_masks, dt_fields

    def compute_batched_step(self, sampled_viewgraph, batch_size=1024, chunk_size=4096):
        """Compute one optimization step over the sampled_viewgraph in a batched manner and return the loss."""

        # divide self.viewgraph in batches if len(self.viewgraph) > batch size
        sampled_viewgraphs = []
        if len(sampled_viewgraph) > batch_size:
            for i in range(0, len(sampled_viewgraph), batch_size):
                end = min(i + batch_size, len(sampled_viewgraph))
                sampled_viewgraphs.append(sampled_viewgraph[i:end])
        else:
            sampled_viewgraphs.append(sampled_viewgraph)

        # collect per-batch results in a python list (tensors)
        residuals_list = []

        for sampled_viewgraph in sampled_viewgraphs:
            s_time = time.time()
            # prepare batched inputs
            batch, pad_masks, dt_fields = self._create_batched_inputs(sampled_viewgraph)
            self.timings["prepare_batched_inputs"] += time.time() - s_time

            # actual inference - keep NaN values
            s_time = time.time()
            # amp not really helping
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                s_time = time.time()
                edges_reprojected = project_world_to_2D(**batch)  # (B, N, 2)
                self.timings["batched_reprojection"] += time.time() - s_time

                # residuals sampling
                residuals = sample_distance_field(
                    dt_fields, edges_reprojected, device=self.device
                ).squeeze(
                    1
                )  # (B, 1, N) -> (B, N)

                # chunked reduction to avoid OOM
                B, N = residuals.shape
                dtype = residuals.dtype
                zero = torch.tensor(0.0, device=self.device, dtype=dtype)
                valid_sums = torch.zeros(B, device=self.device, dtype=dtype)
                valid_counts = torch.zeros(B, device=self.device, dtype=torch.long)

                for i in range(0, N, chunk_size):
                    r_chunk = residuals[:, i : i + chunk_size]  # (B, chunk)
                    m_chunk = pad_masks[:, i : i + chunk_size]  # (B, chunk)
                    non_nan = ~torch.isnan(r_chunk)
                    mask = non_nan & (m_chunk > 0)
                    valid_sums += torch.where(mask, r_chunk, zero).sum(dim=1)
                    valid_counts += mask.sum(dim=1).to(torch.long)

                valid_counts_f = valid_counts.to(dtype)
                mean_losses = torch.where(
                    valid_counts > 0, valid_sums / valid_counts_f.clamp(min=1.0), zero
                )

                # collect this batch's results
                residuals_list.append(mean_losses)

        # concatenate all collected batch results
        if len(residuals_list) == 0:
            residuals = torch.tensor([], device=self.device)
        else:
            residuals = torch.cat(residuals_list, dim=0)  # (num_pairs,)

        return residuals

    def _compute_batched_loss(self, residuals, robustifier=False, delta=1.0):
        """Vectorized batched loss computation."""
        s_time = time.time()

        if residuals.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # Ensure even number of residuals (forward/backward pairs)
        assert residuals.shape[0] % 2 == 0, "Residual batch size must be even (pairs)."

        # Group pairs: reshape (num_pairs*2,) -> (num_pairs, 2)
        residuals_pairs = residuals.view(-1, 2)  # (num_pairs, 2)

        # Sum i->j and j->i directions which are now in the same row
        if robustifier:
            # Use Huber loss for robust cost
            pair_losses = F.huber_loss(
                residuals_pairs,
                torch.zeros_like(residuals_pairs),
                reduction="none",
                delta=delta,
            )
        pair_losses = residuals_pairs.sum(dim=1)  # (num_pairs,)

        # Mean over pairs
        loss = pair_losses.mean()

        self.timings["batched_loss_computation"] += time.time() - s_time
        return loss

    def _cache_projection_matrices(self):
        """Cache all P and K matrices to avoid recomputation"""
        self.P_cache = {}
        for image_name, image_data in self.images.items():
            self.P_cache[image_name] = image_data["P"].projection_matrix()

        self.K_cache = {}
        for cam_id, camera in self.intrinsics.items():
            self.K_cache[cam_id] = camera.intrinsic_matrix()

    def _cache_pad_and_dt_fields(self):
        """Cache all pad masks and distance fields for faster access"""
        self.pad_masks_cache = {}
        self.dt_fields_cache = {}

        for image_name, image_data in self.images.items():
            self.pad_masks_cache[image_name] = image_data["pad_mask"]
            self.dt_fields_cache[image_name] = image_data["dt_field"]

    def _load_and_preprocess_images(self):
        self.images = load_and_preprocess_images(
            self.image_path_list,
            self.images_path,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            device=self.device,
        )

    def _read_cameras_from_reconstruction(self):
        self.images, self.intrinsics = self.read_cameras_from_reconstruction(
            self.recon,
            self.images,
            single_camera_per_folder=self.single_camera_per_folder,
            load_with_pad=self.load_with_pad,
        )

    def _load_and_preprocess_depths(self):
        self.images = load_and_preprocess_depths(
            self.depths_path,
            self.images,
            target_size=self.images_size,
            max_workers=self.max_workers,
            load_with_pad=self.load_with_pad,
            device=self.device,
        )
        # check all depths have the same size
        depth_shapes = set()
        for image_name in self.images.keys():
            depth_shapes.add(self.images[image_name]["depth"].shape[-2:])
        if len(depth_shapes) > 1:
            # pad bottom right to make them equal
            max_h = max([shape[0] for shape in depth_shapes])
            max_w = max([shape[1] for shape in depth_shapes])
            for image_name in self.images.keys():
                depth = self.images[image_name]["depth"]
                h, w = depth.shape[-2:]
                if h < max_h or w < max_w:
                    pad_bottom = max_h - h
                    pad_right = max_w - w
                    pad = (0, pad_right, 0, pad_bottom)  # left, right, top, bottom
                    depth = F.pad(depth, pad, mode="constant", value=depth.max())
                    self.images[image_name]["depth"] = depth

        # add sampled depth at edges_padded locations
        for image_name in self.images.keys():
            edges_padded = self.images[image_name]["edges_padded"]  # (N, 2)
            depth = self.images[image_name]["depth"]  # (H, W)
            sampled_depth, _ = grid_sample_nan(edges_padded[None], depth[None])
            self.images[image_name]["sampled_depth"] = DepthMap(
                height=self.images[image_name]["hw"][0],
                width=self.images[image_name]["hw"][1],
                depth=sampled_depth.squeeze(),  # (N,)
                grad=self.grad_z,
            )

    def _load_and_preprocess_sky_masks(self, path):
        """Load and preprocess sky masks from the given path."""
        if path is None:
            return

        masks_path = glob.glob(os.path.join(path, "*.png")) + glob.glob(
            os.path.join(path, "*", "*.png")
        )

        # load sky masks
        sky_masks_dict = {}
        for mask_path in masks_path:
            mask = Image.open(mask_path).convert("L")
            rel_image_name = os.path.relpath(mask_path, path).replace(".png", ".jpg")
            h, w = self.images[rel_image_name]["hw"]

            mask_tensor = (
                torch.from_numpy(np.array(mask)).float().to(self.device) / 255.0
            )
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(
                mask_tensor, size=(h, w), mode="bilinear", align_corners=False
            )
            mask_tensor = mask_tensor.squeeze().bool()

            mask_tensor[:5, :] = True
            mask_tensor[-5:, :] = True
            mask_tensor[:, :5] = True
            mask_tensor[:, -5:] = True

            sky_masks_dict[rel_image_name] = mask_tensor

        # Update depth with sky masks — set to NaN
        for image_name in self.images.keys():
            if image_name in sky_masks_dict:
                self.images[image_name]["sky_mask"] = sky_masks_dict[image_name]
                depth = self.images[image_name]["depth"]
                sky_mask_tensor = sky_masks_dict[image_name]
                # Set sky regions to NaN so they're properly masked
                depth = depth.masked_fill(sky_mask_tensor, float("nan"))
                self.images[image_name]["depth_no_sky"] = depth

        # Remove old padded edges/masks before recalculating
        for image_name in self.images.keys():
            if "edges_padded" in self.images[image_name]:
                del self.images[image_name]["edges_padded"]
            if "pad_mask" in self.images[image_name]:
                del self.images[image_name]["pad_mask"]
            if "edges" in self.images[image_name]:
                del self.images[image_name]["edges"]
            if "sampled_depth" in self.images[image_name]:
                del self.images[image_name]["sampled_depth"]

        # Filter edges in sky regions
        for image_name in self.images.keys():
            if "sky_mask" in self.images[image_name]:
                sky_mask = self.images[image_name]["sky_mask"]
                edges_map = self.images[image_name]["edges_map"]

                if edges_map.numel() > 0:
                    # Remove edges that fall in sky regions
                    edges_map = edges_map.masked_fill(sky_mask, False)
                    self.images[image_name]["edges_map"] = edges_map
                    # Extract new edges from filtered map
                    self.images[image_name]["edges"] = (
                        edges_map.nonzero().flip(dims=(1, 0)).float()
                    )

        # Recalculate padding with new edge counts
        self._pad_edges()

        # Sample depth at new edge locations
        for image_name in self.images.keys():
            edges_padded = self.images[image_name]["edges_padded"]
            depth = self.images[image_name]["depth"]
            sampled_depth, _ = grid_sample_nan(edges_padded[None], depth[None])

            self.images[image_name]["sampled_depth"] = DepthMap(
                height=self.images[image_name]["hw"][0],
                width=self.images[image_name]["hw"][1],
                depth=sampled_depth.squeeze(),
                grad=self.grad_z,
            )

        # # Verify consistency
        # for image_name in self.images.keys():
        #     n_valid_edges = torch.sum(
        #         ~torch.isnan(self.images[image_name]["edges_padded"]).any(dim=1)
        #     )
        #     n_pad_mask = torch.sum(self.images[image_name]["pad_mask"])

        #     assert (
        #         n_valid_edges == n_pad_mask
        #     ), f"Mismatch for {image_name}: edges={n_valid_edges}, pad_mask={n_pad_mask}"

    def _extract_edges(self):
        for image_name in tqdm(self.images.keys(), desc=f"Extracting edges"):
            img_tensor = self.images[image_name]["image"].unsqueeze(0).to(self.device)
            edges_map = self.edge_extractor(img_tensor)
            edges = edges_map.squeeze().nonzero().flip(dims=(1, 0)).float()  # (N, 2)
            self.images[image_name].update(
                {"edges_map": edges_map.squeeze(), "edges": edges}
            )

        # pad to have same number of edges per image
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
        print(f"max edges per image: {self.max_edges:,}")

    @torch.no_grad()
    def _compute_distance_fields(self):
        dt_fields_shapes = []
        for image_name in tqdm(self.images.keys(), desc="Computing distance fields"):
            edges_map = self.images[image_name]["edges_map"]
            dt_field = compute_distance_field(
                edges_map,
                device=self.device,
            )
            self.images[image_name].update({"dt_field": dt_field})
            dt_fields_shapes.append(dt_field.shape)

        # if dt_fields_shapes is not equal, need to pad right bottom to make them equal
        if len(set(dt_fields_shapes)) > 1:
            max_h = max([shape[0] for shape in dt_fields_shapes])
            max_w = max([shape[1] for shape in dt_fields_shapes])
            for image_name in self.images.keys():
                dt_field = self.images[image_name]["dt_field"]
                h, w = dt_field.shape
                if h < max_h or w < max_w:
                    pad_bottom = max_h - h
                    pad_right = max_w - w
                    pad = (0, pad_right, 0, pad_bottom)  # left, right, top, bottom
                    dt_field = F.pad(
                        dt_field, pad, mode="constant", value=dt_field.max()
                    )
                    self.images[image_name]["dt_field"] = dt_field

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_viewgraph(self, type="exhaustive", window_size=10):
        """Compute viewgraph and filter by reprojection error."""
        if type == "exhaustive":
            # Estimate view graph from frustums
            viewgraph = build_view_graph_from_frustums(
                self.recon,
                z_near_default=0.1,
                z_far_default=5.0,
                max_view_angle_deg=30.0,
                distance_factor=2,
                verbose=False,
                images_with_depth=self.images,
            )

        elif type == "sequential":
            # Build sequential viewgraph based on sorted image names with window size 10
            image_names = sorted(list(self.images.keys()))
            viewgraph = []
            for i in range(len(image_names) - window_size):
                for j in range(1, window_size + 1):
                    viewgraph.append((image_names[i], image_names[i + j]))

        # Filter viewgraph by reprojection | This need to be runned as batch and speeded up
        viewgraph = filter_viewgraph_by_reprojection(
            viewgraph,
            self.images,
            self.intrinsics,
            th=0.025,  # roughly 25% of overlap
            sampling_factor=10,
            reprojection_error=3.0,
            P_cache=self.P_cache,
            K_cache=self.K_cache,
        )
        self.viewgraph = viewgraph

    def _load_viewgraph(self):
        """Load viewgraph from a text file formamted as [(img_i, img_j), ...]."""
        with open(self.viewgraph_path, "r") as f:
            lines = f.readlines()
        self.viewgraph = []
        for line in lines:
            i, j, _ = line.strip().split()
            self.viewgraph.append((i, j))
        print(
            f"Loaded GT viewgraph with {len(self.viewgraph):,} edges from {viewgraph_path}"
        )

    def _collect_parameters_to_optimize(self):
        params_to_optimize = {}

        # Collect parameters to optimize
        if self.grad_k:
            k_params = []
            for camera in self.intrinsics.values():
                k_params.extend(camera.parameters())
            params_to_optimize["k"] = k_params

        if self.grad_q:
            q_params = []
            for image_name, image_data in self.images.items():
                q_params.extend(
                    image_data["P"].parameters(q=True, t=False) if self.grad_q else []
                )
            params_to_optimize["q"] = q_params

        if self.grad_t:
            t_params = []
            for image_name, image_data in self.images.items():
                t_params.extend(
                    image_data["P"].parameters(q=False, t=True) if self.grad_t else []
                )
            params_to_optimize["t"] = t_params

        if self.grad_z:
            z_params = []
            for image_name, image_data in self.images.items():
                z_params.extend(
                    image_data["sampled_depth"].parameters() if self.grad_z else []
                )
            params_to_optimize["z"] = z_params

        return params_to_optimize

    def _print_params_summary(self, params_to_optimize):
        total_params = 0
        print("\nTotal parameters to optimize:")
        for key in ["k", "t", "q", "z"]:
            if key not in params_to_optimize:
                print(f"  {key}: {0:>16,} parameters")
                continue
            set_params = sum(p.numel() for p in params_to_optimize[key])
            print(f"  {key}: {set_params:>16,} parameters")
            total_params += set_params

        print(f"  {'Total':}: {total_params:>12,} parameters\n")

    def _load_optimizer(self, params):
        # Build parameter groups only for parameters that exist
        param_groups = []
        if "t" in params:
            param_groups.append(
                {"params": params["t"], "lr": self.lr * self.t_lr_scale}
            )
        if "k" in params:
            param_groups.append(
                {"params": params["k"], "lr": self.lr * self.k_lr_scale}
            )
        if "z" in params:
            param_groups.append(
                {"params": params["z"], "lr": self.lr * self.z_lr_scale}
            )
        if "q" in params:
            param_groups.append(
                {"params": params["q"], "lr": self.lr * self.q_lr_scale}
            )

        self.optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

    def _load_scheduler(self, name, optimizer, params=None):
        """
        Dynamically loads a PyTorch scheduler by name.

        Args:
            name (str): Name of the scheduler class (e.g., 'StepLR', 'CosineAnnealingLR').
            optimizer (torch.optim.Optimizer): The optimizer instance to attach the scheduler to.
            params (dict, optional): Keyword arguments for the scheduler (e.g., step_size, gamma).

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The initialized scheduler.
        """
        if not name:
            self.scheduler = None
            return

        params = {} if params is None else params

        # Retrieve all available schedulers in torch.optim.lr_scheduler
        schedulers = {
            cls_name.lower(): cls
            for cls_name, cls in torch.optim.lr_scheduler.__dict__.items()
            if isinstance(cls, type)
        }

        key = name.lower()
        if key not in schedulers:
            raise ValueError(
                f"Unknown scheduler '{name}'. Available: {list(schedulers.keys())}"
            )

        SchedulerClass = schedulers[key]
        self.scheduler = SchedulerClass(optimizer, **params)

        print(f"Using scheduler: {name} with params: {params}")

    def _scheduler_step(self, loss):
        if self.scheduler is not None:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

    def _unproject_edges_to_3D(self, batch_size=None):
        """Unproject 2D edges to 3D points for all images as a batch."""
        # Collect data
        image_names = list(self.images.keys())
        B = len(image_names)
        edges_list, depth_list, K_list, P_list = [], [], [], []

        for name in image_names:
            data = self.images[name]
            edges_2D = data["edges_padded"]  # (N, 2)
            depth_map = data["sampled_depth"]()  # (1, H, W)
            K = self.K_cache[data["cam_id"]]  # (3, 3)
            P = self.P_cache[name]  # (4, 4)

            edges_list.append(edges_2D)
            depth_list.append(depth_map)
            K_list.append(K)
            P_list.append(P)

        # Stack as batch
        # edges_2D may have varying N — pad or bucket if needed
        edges_batch = torch.stack(edges_list, dim=0).to(self.device)  # (B, N, 2)
        depth_batch = torch.stack(depth_list, dim=0).to(self.device)  # (B, 1, H, W)
        K_batch = torch.stack(K_list, dim=0).to(self.device)  # (B, 3, 3)
        P_batch = torch.stack(P_list, dim=0).to(self.device)  # (B, 4, 4)

        # Optionally chunk if batch too large for memory
        if batch_size is None:
            batch_size = B
        points_3D_list = []

        for i in range(0, B, batch_size):
            xy0 = edges_batch[i : i + batch_size]
            K0 = K_batch[i : i + batch_size]
            depth0 = depth_batch[i : i + batch_size]
            P0 = P_batch[i : i + batch_size]

            pts3d = unproject_2D_to_world(
                xy0=xy0, K0=K0, depth0=depth0, P0=P0
            )  # (bs, N, 3)
            points_3D_list.append(pts3d)

        points_3D = torch.cat(points_3D_list, dim=0)  # (B, N, 3)

        # Write back (differentiable tensors)
        for name, pts3d in zip(image_names, points_3D):
            self.images[name]["edges_3D"] = pts3d

    def __repr__(self):
        repr_str = f"Adjuster(\n"
        repr_str += f"  Reconstruction path: {self.reconstruction_path}\n"
        repr_str += f"  Images path: {self.images_path}\n"
        repr_str += f"  Depths path: {self.depths_path}\n"
        repr_str += f"  Number of images: {len(self.images)}\n"
        if hasattr(self, "viewgraph"):
            repr_str += f"  Number of viewgraph edges: {len(self.viewgraph):,}\n"

        total_params = 0
        params_to_optimize = self._collect_parameters_to_optimize()
        for key in ["k", "t", "q", "z"]:
            if key in params_to_optimize:
                set_params = sum(p.numel() for p in params_to_optimize[key])
                total_params += set_params

        repr_str += f"  Total parameters to optimize: {total_params:,}\n"
        converged_str = " (converged)" if getattr(self, "convergence", False) else ""
        repr_str += (
            f"  Number of optimization steps: {len(self.loss_list)}{converged_str}\n"
        )

        if len(self.loss_list) >= 2:
            initial_loss = self.loss_list[0]
            final_loss = self.loss_list[-1]
            delta = initial_loss - final_loss
            perc_improvement = (
                (delta / initial_loss) * 100 if initial_loss != 0 else 0.0
            )
            repr_str += f"  Loss improvement: {delta:.3f} ({perc_improvement:.2f}%)\n"

        repr_str += f")"
        return repr_str

    def _scaler_and_scheduler_steps(self, loss):
        # scaler step
        if hasattr(self, "scaler") and self.scaler is not None:
            # gradients computation
            s_time = time.time()
            self.scaler.scale(loss).backward()
            self.timings["gradient_computation"] += time.time() - s_time
            # parameter update
            s_time = time.time()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.timings["parameter_update"] += time.time() - s_time
        else:
            loss.backward()
            self.optimizer.step()

        # scheduler step
        if self.scheduler is not None:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(loss)
            else:
                self.scheduler.step()


if __name__ == "__main__":

    scene = "vienna_state_opera"
    reconstruction_path = (
        f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{scene}/sparse"
    )
    images_path = f"/home/mattia/Desktop/datasets/mydataset/data/{scene}/frames"
    depths_path = (
        f"/home/mattia/Desktop/Repos/vggt/wrapper_output/{scene}/sparse/depth_maps"
    )
    gt_path = f"/home/mattia/Desktop/datasets/mydataset/data/{scene}/colmap/sparse/0"

    adjuster = Adjuster(
        reconstruction_path=reconstruction_path,
        images_path=images_path,
        depths_path=depths_path,
        # viewgraph_path=viewgraph_path,  # if None, it uses the one in the reconstruction_path
        single_camera_per_folder=True,
        load_with_pad=False,
        lr=5e-3,
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=True,
        scheduler_name="ReduceLROnPlateau",
        scheduler_params={"factor": 0.5, "patience": 2, "min_lr": 1e-6},
        # gt_path=gt_path, # slows down a bit the optimization but useful for eval
        # viz=True,
        detector_params={
            "low_threshold": 0.20,
            "high_threshold": 0.25,
            "kernel_size": 7,
            "sigma": 2,
        },
    )

    adjuster.print_summary()

    adjuster(
        batch_size=128,  # saturate this value to fully utilize the GPU
        type="batched",  # "sequential" or "batched"
        max_steps=-1,
        quick_mode=False,  # if True, randomly samples of batch_size at each step
        gradient_tolerance=1e-6,  # stop if relative change in loss is below this value
    )

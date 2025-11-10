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
    unproject_to_world,
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
    Currently I am at an early stage of implementing this class. So i'll just describe what it should do.

    Inputs:
        - images: A batch of images (B, C, H, W)
        - depth_maps: Initial depth maps (B, 1, H, W)
        - pycolmap.Reconstruction: A pycolmap reconstruction object containing the initial poses and intrinsics
                                    (or directly the poses and intrinsics in torch)

    Outputs:
        optimized depth maps, updated pycolmap.Reconstruction
    """

    def __init__(
        self,
        reconstruction_path,
        images_path,
        depths_path,
        viewgraph_path=None,  # for testing with GT viewgraph
        lr=1e-3,
        single_camera_per_folder=True,
        load_with_pad=True,
        detector="canny",
        device="cuda",
        max_workers=-1,
        detector_params={},
        seed=0,
        optim="adamw",  # or "LM"
        scheduler_name=None,
        scheduler_params={},
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=False,
        viz=False,  # it true del non used stuff during computation
        gt_path=None,
    ):
        super().__init__()

        assert detector in ["canny"], f"Detector {detector} not supported."
        assert optim.lower() in [
            "adamw",
            "lm",
        ], f"Optimizer {optim} not supported."

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
        print(f"max edges per image: {self.max_edges:,}")

        ## Load depth maps
        s_time = time.time()
        self._load_and_preprocess_depths()
        self.timings["load_depth_maps"] = time.time() - s_time

        ## Compute Distance Fields
        s_time = time.time()
        self._compute_distance_fields()  # into self.images
        self.timings["compute_distance_fields"] = time.time() - s_time

        ## Viewgraph from frustums
        s_time = time.time()
        # compute and cache P and K matrices once
        self.P_cache, self.K_cache = None, None
        self._cache_projection_matrices()

        if self.viewgraph_path is None:
            self._compute_viewgraph()
        else:
            self._read_viewgraph()
        # vg is a lsit of names of image pairs, sort by the the first name
        self.viewgraph.sort(key=lambda x: (x[0], x[1]))
        self.timings["compute_viewgraph"] = time.time() - s_time

        # Create optimizer
        params_to_optimize = self._collect_parameters_to_optimize()
        self._print_params_summary(params_to_optimize)

        # Create optimizer with collected parameters
        self._load_optimizer(optim, params_to_optimize)
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
        gradient_tolerance=1e-6,
        quick_mode=True,  # "quick" (sample & backprop) or "full" (accumulate over all)
        batch_size=128,  # faster than 64 and 256
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
            )

        max_steps = max_steps if max_steps > 0 else 1_000
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

            loss = self._compute_batched_loss(residuals)

            # Backpropagate and update using GradScaler if available
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

            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            # Logging
            self.loss_list.append(loss.detach().item())
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.lr
            )
            self.lr_list.append(current_lr)

            # DEBUG: Evaluate AUC if GT available
            if self.gt_path is not None:
                opt = "/home/mattia/Desktop/Repos/batchsfm/optimized_reconstruction"
                self.build_reconstruction(opt)
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
                        f"change {rel_change:.6f} < {gradient_tolerance}",
                    )
                    self.convergence = True
                    break

            bar.set_postfix(
                loss=f"{self.loss_list[-1]:.6f}",
                auc5=(
                    f"{self.auc_list[-1][-1]:.4f}"
                    if self.gt_path is not None
                    else "N/A"
                ),
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

    @torch.no_grad()
    def build_reconstruction(
        self, output_path="optimized_reconstruction", save_points=False, verbose=False
    ):
        """
        Create a pycolmap.Reconstruction from images and intrinsics dictionaries.

        Args:
            images: dict with image_name -> {P: Pose, cam_id: int, scale: float, ...}
            intrinsics: dict with cam_id -> Camera
            output_path: path to save the reconstruction
            save_points: whether to save 3D points (empty here)
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

            # use xyzw -> wxyz
            # q = np.roll(q, shift=3)

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

        # 3. Points3D - empty for now
        if save_points:
            print("Saving with empty Points3D...")
            # TODO:similar to VGGT, unproject depth maps to create points3D
            pass

        # 4. Save reconstruction
        if verbose:
            print(f"Cameras: {len(reconstruction.cameras)}")
            print(f"Images: {len(reconstruction.images)}")
            print(f"Points3D: {len(reconstruction.points3D)}")

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

            pad_masks.append(self.images[i]["pad_mask"])
            pad_masks.append(self.images[j]["pad_mask"])

            dt_fields.append(self.images[j]["dt_field"])
            dt_fields.append(self.images[i]["dt_field"])

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

    def compute_batched_step(self, sampled_viewgraph, batch_size=1024):
        """Compute one optimization step over the sampled_viewgraph in a batched manner and return the loss."""

        # divide self.viewgraph in batches if len(self.viewgraph) > batch size
        sampled_viewgraphs = []
        if len(sampled_viewgraph) > batch_size:
            for i in range(0, len(sampled_viewgraph), batch_size):
                end = min(i + batch_size, len(sampled_viewgraph))
                sampled_viewgraphs.append(sampled_viewgraph[i:end])
        else:
            sampled_viewgraphs.append(sampled_viewgraph)

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
                )  # (B, N)

                # average residuals over valid points per image
                # TODO: try vectorized version
                for b in range(residuals.shape[0]):
                    pad_mask = pad_masks[b]
                    edge_loss = residuals[b]
                    valid = ~torch.isnan(edge_loss) & (pad_mask > 0)
                    valid_loss = edge_loss[valid]
                    mean_loss = (
                        valid_loss.mean()
                        if valid_loss.numel() > 0
                        else torch.tensor(0.0, device=self.device)
                    )
                    residuals_list.append(mean_loss)

        residuals = torch.stack(residuals_list)  # (num_pairs,)

        return residuals

    def _compute_batched_loss(self, residuals):
        """Vectorized batched loss computation."""
        s_time = time.time()

        if residuals.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # Ensure even number of residuals (forward/backward pairs)
        assert residuals.shape[0] % 2 == 0, "Residual batch size must be even (pairs)."

        # Group pairs: reshape (num_pairs*2,) -> (num_pairs, 2)
        residuals_pairs = residuals.view(-1, 2)  # (num_pairs, 2)

        # Sum i->j and j->i directions which are now in the same row
        pair_losses = residuals_pairs.sum(dim=1)  # (num_pairs,)

        # Mean over pairs
        loss = pair_losses.mean()

        self.timings["batched_loss_computation"] += time.time() - s_time
        return loss

    def _cache_projection_matrices(self):
        """Cache all P and K matrices to avoid recomputation"""
        self.P_cache = {}
        self.K_cache = {}

        for image_name, image_data in self.images.items():
            self.P_cache[image_name] = image_data["P"].projection_matrix()

        for cam_id, camera in self.intrinsics.items():
            self.K_cache[cam_id] = camera.intrinsic_matrix()

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

    def _extract_edges(self):
        for image_name in tqdm(self.images.keys(), desc=f"Extracting edges"):
            img_tensor = self.images[image_name]["image"].unsqueeze(0).to(self.device)
            edges_map = self.edge_extractor(img_tensor)
            edges = edges_map.squeeze().nonzero().flip(dims=(1, 0)).float()  # (N, 2)
            self.images[image_name].update(
                {"edges_map": edges_map.squeeze(), "edges": edges}
            )

        # pad with nans to have same number of edges per image
        max_edges = max(
            [self.images[img]["edges"].shape[0] for img in self.images.keys()]
        )
        for image_name in self.images.keys():
            edges = self.images[image_name]["edges"]
            n_edges = edges.shape[0]
            if n_edges < max_edges:
                pad_size = max_edges - n_edges
                # pad = torch.full((pad_size, 2), float("nan"), device=edges.device)
                pad = torch.zeros((pad_size, 2), device=edges.device)
                edges = torch.cat([edges, pad], dim=0)
                pad_mask = torch.zeros((max_edges,), device=edges.device)
                pad_mask[:n_edges] = 1.0
            self.images[image_name].update(
                {"edges_padded": edges, "pad_mask": pad_mask}
            )
        self.max_edges = max_edges

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
    def _compute_viewgraph(self):
        # Estimate view graph from frustums
        viewgraph = build_view_graph_from_frustums(
            self.recon,
            z_near_default=0.1,
            z_far_default=5.0,
            max_view_angle_deg=30.0,
            distance_factor=2,
            verbose=False,
        )
        # Filter viewgraph by reprojection | This need to be runned as batch and speeded up
        viewgraph = filter_viewgraph_by_reprojection(
            viewgraph,
            self.images,
            self.intrinsics,
            th=0.025,
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

    def _load_optimizer(self, optim_name, params):
        optim_name = optim_name.lower()
        self.use_pypose = False

        # Build parameter groups only for parameters that exist
        param_groups = []
        if "t" in params:
            param_groups.append({"params": params["t"], "lr": self.lr})
        if "k" in params:
            param_groups.append({"params": params["k"], "lr": self.lr * 0.5})
        if "z" in params:
            param_groups.append({"params": params["z"], "lr": self.lr * 0.5})
        if "q" in params:
            param_groups.append({"params": params["q"], "lr": self.lr * 0.1})

        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        else:
            print(f"Optimizer {optim_name} not recognized. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        self.optimizer = optimizer

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

            pts3d = unproject_to_world(
                xy0=xy0, K0=K0, depth0=depth0, P0=P0
            )  # (bs, N, 3)
            points_3D_list.append(pts3d)

        points_3D = torch.cat(points_3D_list, dim=0)  # (B, N, 3)

        # Write back (differentiable tensors)
        for name, pts3d in zip(image_names, points_3D):
            self.images[name]["edges_3D"] = pts3d


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
        viewgraph_path=None,  # if None, it uses the one in the reconstruction_path
        single_camera_per_folder=True,
        load_with_pad=False,
        lr=5e-3,
        grad_q=True,
        grad_t=True,
        grad_k=True,
        grad_z=True,
        scheduler_name="ReduceLROnPlateau",
        scheduler_params={"factor": 0.5, "patience": 3, "min_lr": 1e-6},
        gt_path=gt_path,  # slows down a bit the optimization
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
